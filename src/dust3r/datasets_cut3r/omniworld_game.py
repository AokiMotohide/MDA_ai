import argparse
import json
import os
import os.path as osp
import sys

import cv2
import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import pickle

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

from src.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


def load_split_info(scene_dir):
    with open(osp.join(scene_dir, "split_info.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_camera_poses(scene_dir, split_idx):
    split_info = load_split_info(scene_dir)
    idxs = split_info["split"][split_idx]
    frame_count = len(idxs)

    cam_file = osp.join(scene_dir, "camera", f"split_{split_idx}.json")
    with open(cam_file, "r", encoding="utf-8") as f:
        cam = json.load(f)

    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], frame_count, axis=0)
    intrinsics[:, 0, 0] = np.asarray(cam["focals"], dtype=np.float32)
    intrinsics[:, 1, 1] = np.asarray(cam["focals"], dtype=np.float32)
    intrinsics[:, 0, 2] = np.float32(cam["cx"])
    intrinsics[:, 1, 2] = np.float32(cam["cy"])

    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], frame_count, axis=0)
    quat_wxyz = np.asarray(cam["quats"], dtype=np.float32)
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)
    rotations = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    translations = np.asarray(cam["trans"], dtype=np.float32)

    extrinsics[:, :3, :3] = rotations
    extrinsics[:, :3, 3] = translations

    return intrinsics.astype(np.float32), extrinsics.astype(np.float32), idxs


def load_depth(depth_path):
    # depthmap = imageio.v2.imread(depth_path).astype(np.float32) / 65535.0
    depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
    if depthmap.ndim == 3:
        depthmap = depthmap[:, :, 0]

    near_mask = depthmap < 0.0015
    far_mask = depthmap > (65500.0 / 65535.0)

    near, far = 1.0, 1000.0
    depthmap = depthmap / (far - depthmap * (far - near)) / 0.004

    valid = ~(near_mask | far_mask)
    depthmap[~valid] = 0
    depthmap[far_mask] = -1
    return depthmap.astype(np.float32), valid, far_mask



class OmniWorldGame_Multi(BaseMultiViewDataset):
    def __init__(
        self,
        *args,
        ROOT,
        min_interval=5,
        max_interval=15,
        max_scenes=None,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_scenes = max_scenes
        super().__init__(*args, **kwargs)
        self._load_data()

    def _load_data(self):
        saved_path = osp.join(self.ROOT, f'pre-calculated-loaddata-{self.num_views}.pkl')
        if osp.exists(saved_path):
            with open(saved_path, 'rb') as f:
                sequences = pickle.load(f)
            self.sequences = sequences
            return

        scene_names = sorted(
            [d for d in os.listdir(self.ROOT) if osp.isdir(osp.join(self.ROOT, d))]
        )
        if self.max_scenes is not None:
            scene_names = scene_names[: self.max_scenes]

        required_len = 1 + (self.num_views - 1) * self.min_interval
        sequences = []

        for scene_name in tqdm(scene_names, desc="Loading OmniWorld_game"):
            scene_dir = osp.join(self.ROOT, scene_name)
            split_info_path = osp.join(scene_dir, "split_info.json")
            if not osp.isfile(split_info_path):
                continue

            try:
                split_info = load_split_info(scene_dir)
            except Exception:
                continue

            split_lists = split_info.get("split", [])
            if not split_lists:
                continue

            for split_idx, frame_indices in enumerate(split_lists):
                if len(frame_indices) < required_len:
                    continue

                cam_path = osp.join(scene_dir, "camera", f"split_{split_idx}.json")
                if not osp.isfile(cam_path):
                    continue

                try:
                    intrinsics, w2c, idxs = load_camera_poses(scene_dir, split_idx)
                except Exception:
                    continue

                if len(idxs) < required_len:
                    continue

                feasible_intervals = [
                    interval
                    for interval in range(self.min_interval, self.max_interval + 1)
                    if 1 + (self.num_views - 1) * interval <= len(idxs)
                ]
                if not feasible_intervals:
                    continue

                sequences.append(
                    dict(
                        scene_name=scene_name,
                        scene_dir=scene_dir,
                        split_idx=split_idx,
                        frame_indices=np.asarray(idxs, dtype=np.int32),
                        intrinsics=intrinsics.astype(np.float32),
                        w2c=w2c.astype(np.float32),
                        feasible_intervals=np.asarray(feasible_intervals, dtype=np.int32),
                    )
                )

        self.sequences = sequences
        with open(saved_path, 'wb') as f:
            pickle.dump(sequences, f)

    def __len__(self):
        return len(self.sequences)

    def get_stats(self):
        return f"{len(self.sequences)} sequences"

    def _sample_positions(self, seq_len, rng, num_views, feasible_intervals):
        interval = int(rng.choice(feasible_intervals))
        max_start = seq_len - 1 - interval * (num_views - 1)
        start_pos = int(rng.integers(0, max_start + 1))
        positions = [start_pos + interval * i for i in range(num_views)]
        return positions, interval

    def _get_views(self, idx, resolution, rng, num_views):
        seq = self.sequences[idx]
        seq_len = len(seq["frame_indices"])
        positions, sampled_interval = self._sample_positions(
            seq_len,
            rng,
            num_views,
            seq["feasible_intervals"],
        )

        views = []
        for view_order, pos in enumerate(positions):
            frame_idx = int(seq["frame_indices"][pos])
            basename = f"{frame_idx:06d}"

            rgb_path = osp.join(seq["scene_dir"], "color", basename + ".png")
            depth_path = osp.join(seq["scene_dir"], "depth", basename + ".png")

            rgb_image = imread_cv2(rgb_path)
            depthmap, _, far_mask = load_depth(depth_path)
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            depth_complete = depthmap.copy()

            intrinsics = seq["intrinsics"][pos].copy().astype(np.float32)
            w2c = seq["w2c"][pos].astype(np.float32)
            camera_pose = np.linalg.inv(w2c).astype(np.float32)
            assert (not np.isnan(depthmap).any()) and (not np.isinf(depthmap).any()), f"depthmap: {depthmap.shape}, {depthmap.min()}, {depthmap.max()}"
            assert (not np.isnan(intrinsics).any()) and (not np.isinf(intrinsics).any()), f"intrinsics: {intrinsics.shape}, {intrinsics.min()}, {intrinsics.max()}"
            assert (not np.isnan(camera_pose).any()) and (not np.isinf(camera_pose).any()), f"camera_pose: {camera_pose.shape}, {camera_pose.min()}, {camera_pose.max()}"

            rgb_image, depthmap, depth_complete, intrinsics = self._crop_resize_if_necessary2(
                rgb_image,
                depthmap,
                depth_complete,
                intrinsics,
                resolution,
                rng,
                info=f"{seq['scene_name']}/split_{seq['split_idx']}/{basename}",
            )

            # depthmap = self._filter_depth_by_quantile(depthmap, quantile=0.98)
            # depth_complete = self._filter_depth_by_quantile(depth_complete, quantile=0.98)

            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, view_order, rng, p=[0.9, 0.05, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    depth_complete=depth_complete.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="OmniWorldGame",
                    label=f"{seq['scene_name']}_split{seq['split_idx']}_{basename}",
                    instance=rgb_path,
                    is_metric=self.is_metric,
                    is_video=True,
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                    sky_mask=(depthmap < -0.1).astype(np.float32),
                )
            )
        
        all_depthmaps = np.stack([view['depthmap'] for view in views], axis=0)
        all_depth_completes = np.stack([view['depth_complete'] for view in views], axis=0)
        all_depthmaps = self._filter_depth_by_quantile(all_depthmaps, quantile=0.98)
        all_depth_completes = self._filter_depth_by_quantile(all_depth_completes, quantile=0.98)
        for view, depthmap, depth_complete in zip(views, all_depthmaps, all_depth_completes):
            view['depthmap'] = depthmap
            view['depth_complete'] = depth_complete
        assert (not np.isnan(depthmap).any()) and (not np.isinf(depthmap).any()), f"depthmap: {depthmap.shape}, {depthmap.min()}, {depthmap.max()}"
        assert (not np.isnan(depth_complete).any()) and (not np.isinf(depth_complete).any()), f"depth_complete: {depth_complete.shape}, {depth_complete.min()}, {depth_complete.max()}"
        assert len(views) == num_views
        return views, positions