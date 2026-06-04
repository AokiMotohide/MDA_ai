# --------------------------------------------------------
# Dataloader for preprocessed project-aria dataset
# --------------------------------------------------------
import os.path as osp
import os
import cv2
import numpy as np
import math
import sys  # noqa: E402

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

from src.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class Aria_Seq(BaseMultiViewDataset):

    def __init__(
        self,
        ROOT="data/cut3r_data/processed_ase_2k",
        scene_name=None,  # specify scene name(s) to load
        sample_freq=1,  # stride of the frmaes inside the sliding window
        start_freq=1,  # start frequency for the sliding window
        filter=False,  # filter out the windows with abnormally large stride
        depth_is_distance=True,  # True if depthmap stores radial distance
        rand_sel=False,  # randomly select views from a window
        winsize=0,  # window size to randomly select views
        sel_num=0,  # number of combinations to randomly select from a window
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.sample_freq = sample_freq
        self.start_freq = start_freq
        self.is_metric = True
        self.depth_is_distance = depth_is_distance

        self.rand_sel = rand_sel
        if rand_sel:
            assert winsize > 0 and sel_num > 0
            comb_num = math.comb(winsize - 1, self.num_views - 2)
            assert comb_num >= sel_num
            self.winsize = winsize
            self.sel_num = sel_num
        else:
            self.winsize = sample_freq * (self.num_views - 1)

        self.scene_names = os.listdir(self.ROOT)
        self.scene_names = [
            int(scene_name) for scene_name in self.scene_names if scene_name.isdigit()
        ]
        self.scene_names = sorted(self.scene_names)
        self.scene_names = [str(scene_name) for scene_name in self.scene_names]
        total_scene_num = len(self.scene_names)

        if self.split == "train":
            # choose 90% of the data as training set
            self.scene_names = self.scene_names[: int(total_scene_num * 0.9)]
        elif self.split == "test":
            self.scene_names = self.scene_names[int(total_scene_num * 0.9) :]
        if scene_name is not None:
            assert self.split is None
            if isinstance(scene_name, list):
                self.scene_names = scene_name
            else:
                if isinstance(scene_name, int):
                    scene_name = str(scene_name)
                assert isinstance(scene_name, str)
                self.scene_names = [scene_name]

        self._load_data(filter=filter)
        print(self)

    def filter_windows(self, sid, eid, image_names):
        return False

    def _load_data(self, filter=False):
        self.sceneids = []
        self.images = []
        self.intrinsics = []  # scene_num*(3,3)
        self.win_bid = []

        num_count = 0
        for id, scene_name in enumerate(self.scene_names):
            scene_dir = os.path.join(self.ROOT, scene_name)
            image_names = os.listdir(os.path.join(scene_dir, "color"))
            image_names = sorted(image_names)
            intrinsic = np.loadtxt(
                os.path.join(scene_dir, "intrinsic", "intrinsic_color.txt")
            )[:3, :3]
            image_num = len(image_names)
            # precompute the window indices
            for i in range(0, image_num, self.start_freq):
                last_id = i + self.winsize
                if last_id >= image_num:
                    break
                if filter and self.filter_windows(i, last_id, image_names):
                    continue
                self.win_bid.append((num_count + i, num_count + last_id))

            self.intrinsics.append(intrinsic)
            self.images += image_names
            self.sceneids += [
                id,
            ] * image_num
            num_count += image_num
        self.intrinsics = np.stack(self.intrinsics, axis=0)
        assert len(self.sceneids) == len(
            self.images
        ), f"{len(self.sceneids)}, {len(self.images)}"

    def __len__(self):
        if self.rand_sel:
            return self.sel_num * len(self.win_bid)
        return len(self.win_bid)

    def get_img_idxes(self, idx, rng, num_views):
        if self.rand_sel:
            sid, eid = self.win_bid[idx // self.sel_num]
            if idx % self.sel_num == 0:
                return np.linspace(sid, eid, num_views, endpoint=True, dtype=int)

            if self.num_views == 2:
                return [sid, eid]
            sel_ids = rng.choice(range(sid + 1, eid), num_views - 2, replace=False)
            sel_ids.sort()
            return [sid] + list(sel_ids) + [eid]
        else:
            sid, eid = self.win_bid[idx]
            return [sid + i * self.sample_freq for i in range(num_views)]

    @staticmethod
    def _distance_to_z_depth(depthmap, intrinsics):
        """Convert radial distance to z-depth in camera coordinates."""
        H, W = depthmap.shape
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x = (u - cx) / fx
        y = (v - cy) / fy
        denom = np.sqrt(x * x + y * y + 1.0)
        z = np.where(denom > 0, depthmap / denom, 0.0)
        return z.astype(np.float32)

    def _get_views(self, idx, resolution, rng, num_views):

        image_idxes = self.get_img_idxes(idx, rng, num_views)
        views = []
        for view_idx in image_idxes:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scene_names[scene_id])

            intrinsics = self.intrinsics[scene_id]
            basename = self.images[view_idx]
            if scene_dir == "data/cut3r_data/ase_processed/1917" and basename=="0000284.jpg":
                return self._get_views((idx + 999)%len(self), resolution, rng, num_views)
            
            camera_pose = np.loadtxt(
                osp.join(scene_dir, "pose", basename.replace(".jpg", ".txt"))
            )
            if not np.isfinite(camera_pose).all():
                camera_pose = np.eye(4)
            # Load RGB image
            rgb_image = imread_cv2(osp.join(scene_dir, "color", basename))
            # Load depthmap
            depthmap = imread_cv2(
                osp.join(scene_dir, "depth", basename.replace(".jpg", ".png")),
                cv2.IMREAD_UNCHANGED,
            )
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            depthmap = depthmap.astype(np.float32) / 1000
            if self.depth_is_distance:
                depthmap = self._distance_to_z_depth(depthmap, intrinsics)
            depthmap[depthmap > 20] = 0  # invalid

            depth_complete = depthmap.copy()
            rgb_image, depthmap, depth_complete, intrinsics = self._crop_resize_if_necessary2(
                rgb_image, depthmap, depth_complete, intrinsics, resolution, rng, info=view_idx
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    depth_complete=depth_complete.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="Aria",
                    label=self.scene_names[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    # other stuffs in cut3r
                    is_metric=self.is_metric,
                    is_video=False,
                    quantile=np.array(1.0, dtype=np.float32),
                    img_mask=True,
                    ray_mask=False,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                    sky_mask=np.zeros_like(depthmap).astype(np.float32),
                )
            )
        return views, 0
