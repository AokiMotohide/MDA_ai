import os.path as osp
from pdb import set_trace as st
import pickle
import os
import sys
import itertools

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np
from tqdm import tqdm

from src.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset
from src.dust3r.utils.image import imread_cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.ndimage import binary_dilation

def has_any_file(path):
    with os.scandir(path) as it:
        for _ in it:
            return True
    return False

def list_subdirs(path):
    """Efficient subdirectory listing using scandir."""
    try:
        return [e.name for e in os.scandir(path) if e.is_dir()]
    except FileNotFoundError:
        return []

def check_scene(scene_root, ROOT):
    """Check and return valid subscene paths under one scene."""
    valid_subscenes = []
    scene_path = osp.join(ROOT, scene_root)
    for f in list_subdirs(scene_path):
        sub_path = osp.join(scene_path, f)
        if has_any_file(sub_path):
            valid_subscenes.append(osp.join(scene_root, f))
    return valid_subscenes


def process_one_subscene(scene, ROOT, num_views, allow_repeat):
    """Return processed info for one subscene, or None if invalid."""
    scene_dir = osp.join(ROOT, scene, "dense")
    rgb_dir = osp.join(scene_dir, "rgb")

    if not osp.isdir(rgb_dir):
        return None

    try:
        rgb_paths = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    except FileNotFoundError:
        return None

    if not rgb_paths:
        return None

    # check all subfolders
    for sub_dir in ['cam', 'depth', 'outlier_mask', 'sky_mask']:
        sub_full = osp.join(scene_dir, sub_dir)
        try:
            if not osp.isdir(sub_full) or len(os.listdir(sub_full)) != len(rgb_paths):
                return None
        except FileNotFoundError:
            return None

    num_imgs = len(rgb_paths)
    cut_off = (num_views if not allow_repeat else max(num_views // 3, 3))
    if num_imgs < cut_off:
        return None

    return dict(scene=scene, num_imgs=num_imgs, rgb_paths=rgb_paths, cut_off=cut_off)


class DL3DV_Multi(BaseMultiViewDataset):

    def __init__(self, *args, split, ROOT, SUBSAMPLE=False, SAMPLENUM=1000, DEBUG=False, better_filter=True, 
                 non_sequential_readout=False, max_interval=8, max_interval_test=16, valid_depth=True, **kwargs):
        self.ROOT = ROOT
        self.SUBSAMPLE = SUBSAMPLE
        self.SAMPLENUM = SAMPLENUM
        self.DEBUG = DEBUG
        self.video = True
        # self.max_interval = 20
        self.max_interval = max_interval
        self.max_interval_test = max_interval_test
        self.is_metric = False
        self.valid_depth = valid_depth
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data()
        self.better_filter = better_filter
        self.non_sequential_readout = non_sequential_readout
        

    def _load_data(self):
        cache_path = osp.join(self.ROOT, f'pre-calculated-loaddata-{self.num_views}.pkl')

        if osp.exists(cache_path):
            with open(cache_path, 'rb') as f:
                pre = pickle.load(f)
            self.scenes = pre['scenes']
            self.sceneids = pre['sceneids']
            self.images = pre['images']
            self.start_img_ids = pre['start_img_ids']
            self.scene_img_list = pre['scene_img_list']
            
            if self.SUBSAMPLE and self.SAMPLENUM != 1:
                datalen = len(self.start_img_ids)
                interval = datalen // self.SAMPLENUM
                self.start_img_ids = self.start_img_ids[::interval]
                
            elif self.SUBSAMPLE and self.SAMPLENUM == 1:
                # print(np.unique(self.sceneids), np.where(np.array(self.sceneids) == 3), np.where(np.array(self.sceneids) == 3)[0])
                idxs_scene_id_2 = np.where(np.array(self.sceneids) == 2)[0]
                self.start_img_ids = [self.start_img_ids[i] for i in idxs_scene_id_2]
                # print('self.start_img_ids', len(self.start_img_ids))

            return

        # === 1. Efficient scene listing
        self.all_scenes = sorted([
            e.name for e in os.scandir(self.ROOT) if e.is_dir()
        ])

        # === 2. Parallel subscene scanning
        if osp.exists(osp.join(self.ROOT, f'subscenes-{self.num_views}.pkl')):
            with open(osp.join(self.ROOT, f'subscenes-{self.num_views}.pkl'), 'rb') as f:
                subscenes = pickle.load(f)
        
        else:
            subscenes = []
            with ThreadPoolExecutor(max_workers=8) as executor:  # adjust threads if needed
                futures = {
                    executor.submit(check_scene, scene, self.ROOT): scene
                    for scene in self.all_scenes
                }
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning scenes"):
                    subscenes.extend(fut.result())

            with open(osp.join(self.ROOT, f'subscenes-{self.num_views}.pkl'), 'wb') as f:
                pickle.dump(subscenes, f)
            
        # === 3. Sequentially process filtered subscenes
        results = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(process_one_subscene, scene, self.ROOT, self.num_views, self.allow_repeat)
                for scene in subscenes
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading valid subscenes"):
                res = fut.result()
                if res is not None:
                    results.append(res)

        # === 4. Merge sequentially to preserve indexing
        offset = 0
        scenes, sceneids, images, scene_img_list, start_img_ids = [], [], [], [], []
        j = 0

        for res in sorted(results, key=lambda x: x["scene"]):  # stable order
            scene = res["scene"]
            num_imgs = res["num_imgs"]
            rgb_paths = res["rgb_paths"]
            cut_off = res["cut_off"]

            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[:num_imgs - cut_off + 1]

            scenes.append(scene)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend(rgb_paths)
            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            j += 1
    
        # === 4. Save cache
        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

        with open(cache_path, 'wb') as f:
            pickle.dump(
                dict(
                    scenes=self.scenes,
                    sceneids=self.sceneids,
                    images=images,
                    start_img_ids=start_img_ids,
                    scene_img_list=scene_img_list,
                ),
                f,
            )

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        # print('idx', idx)
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            block_shuffle=25,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for i, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id], "dense")

            rgb_path = self.images[view_idx]
            basename = rgb_path[:-4]

            rgb_image = imread_cv2(osp.join(scene_dir, "rgb", rgb_path),
                                   cv2.IMREAD_COLOR)
            depthmap = np.load(osp.join(scene_dir, "depth",
                                        basename + ".npy")).astype(np.float32)
            depth_complete = depthmap.copy()
            # depth_complete[:] = -1
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np.load(osp.join(scene_dir, "cam", basename + ".npz"))
            sky_mask = (cv2.imread(osp.join(scene_dir, "sky_mask", rgb_path),
                                   cv2.IMREAD_UNCHANGED) >= 127)
            outlier_mask = (cv2.imread(
                osp.join(scene_dir, "outlier_mask", rgb_path),
                cv2.IMREAD_UNCHANGED) >= 127)
            if self.better_filter: 
                outlier_mask = binary_dilation(outlier_mask, iterations=2)
                
            depthmap[sky_mask] = -1.0
            depthmap[outlier_mask] = 0.0
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            threshold = (np.percentile(depthmap[depthmap > 0], 98)
                         if depthmap[depthmap > 0].size > 0 else 0)
            depthmap[depthmap > threshold] = 0.0
            
            if not self.valid_depth:
                depthmap[:] = 0.0

            intrinsics = cam_file["intrinsic"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            rgb_image, depthmap, depth_complete, intrinsics = self._crop_resize_if_necessary2(
                rgb_image,
                depthmap,
                depth_complete,
                intrinsics,
                resolution,
                rng=rng,
                info=view_idx)

            img_mask = True
            ray_mask = False

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    depth_complete=depth_complete.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="dl3dv",
                    label=self.scenes[scene_id] + "_" + rgb_path,
                    instance=osp.join(scene_dir, "rgb", rgb_path),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.9, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                ))
            
        return views, pos

