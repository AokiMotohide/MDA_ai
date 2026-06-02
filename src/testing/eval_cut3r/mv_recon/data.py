import os
import cv2
import json
import numpy as np
import os.path as osp
import glob
from collections import deque
import random
from PIL import Image
from src.testing.eval_cut3r.mv_recon.base import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import src.testing.eval_cut3r.mv_recon.dataset_utils.cropping as cropping


def shuffle_deque(dq, seed=None):
    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Convert deque to list, shuffle, and convert back
    shuffled_list = list(dq)
    random.shuffle(shuffled_list)
    return deque(shuffled_list)


class SevenScenes(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        if self.tuple_list is not None:
            # Use pre-defined simplerecon scene_ids
            self.scene_list = [
                "stairs/seq-06",
                "stairs/seq-02",
                "pumpkin/seq-06",
                "chess/seq-01",
                "heads/seq-02",
                "fire/seq-02",
                "office/seq-03",
                "pumpkin/seq-03",
                "redkitchen/seq-07",
                "chess/seq-02",
                "office/seq-01",
                "redkitchen/seq-01",
                "fire/seq-01",
            ]
            print(f"Found {len(self.scene_list)} sequences in split {self.split}")
            return

        scenes = os.listdir(base_dir)

        file_split = {"train": "TrainSplit.txt", "test": "TestSplit.txt"}[self.split]

        self.scene_list = []
        for scene in scenes:
            if self.test_id is not None and scene != self.test_id:
                continue
            # read file split
            with open(osp.join(base_dir, scene, file_split)) as f:
                seq_ids = f.read().splitlines()

                for seq_id in seq_ids:
                    # seq is string, take the int part and make it 01, 02, 03
                    # seq_id = 'seq-{:2d}'.format(int(seq_id))
                    num_part = "".join(filter(str.isdigit, seq_id))
                    seq_id = f"seq-{num_part.zfill(2)}"
                    if self.seq_id is not None and seq_id != self.seq_id:
                        continue
                    self.scene_list.append(f"{scene}/{seq_id}")

        print(f"Found {len(self.scene_list)} sequences in split {self.split}")

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]
            seq_id = idx % self.num_seq

            data_path = osp.join(self.ROOT, scene_id)
            num_files = len([name for name in os.listdir(data_path) if "color" in name])
            img_idxs = [f"{i:06d}" for i in range(num_files)]
            img_idxs = img_idxs[:: self.kf_every]

        # Intrinsics used in SimpleRecon
        fx, fy, cx, cy = 525, 525, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        views = []
        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()
            impath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.color.png")
            depthpath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.depth.proj.png")
            posepath = osp.join(self.ROOT, scene_id, f"frame-{im_idx}.pose.txt")

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            depthmap[depthmap == 65535] = 0
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            camera_pose = np.loadtxt(posepath).astype(np.float32)

            if resolution != (224, 224) or self.rebuttal:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="7scenes",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )
        return views


class DTU(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=49,
        num_frames=5,
        min_thresh=10,
        max_thresh=30,
        test_id=None,
        full_video=False,
        sample_pairs=False,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.sample_pairs = sample_pairs

        # load all scenes
        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):

        if self.test_id is None:
            self.scene_list = os.listdir(osp.join(base_dir))
            print(f"Found {len(self.scene_list)} scenes in split {self.split}")

        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]

            print(f"Test_id: {self.test_id}")

    def load_cam_mvsnet(self, file, interval_scale=1):
        """read camera txt file"""
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                cam[0][i][j] = words[extrinsic_index]

        # read intrinsic
        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                cam[1][i][j] = words[intrinsic_index]

        if len(words) == 29:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = 192
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 30:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
        elif len(words) == 31:
            cam[1][3][0] = words[27]
            cam[1][3][1] = float(words[28]) * interval_scale
            cam[1][3][2] = words[29]
            cam[1][3][3] = words[30]
        else:
            cam[1][3][0] = 0
            cam[1][3][1] = 0
            cam[1][3][2] = 0
            cam[1][3][3] = 0

        extrinsic = cam[0].astype(np.float32)
        intrinsic = cam[1].astype(np.float32)

        return intrinsic, extrinsic

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]
        seq_id = idx % self.num_seq

        image_path = osp.join(self.ROOT, scene_id, "images")
        depth_path = osp.join(self.ROOT, scene_id, "depths")
        mask_path = osp.join(self.ROOT, scene_id, "binary_masks")
        cam_path = osp.join(self.ROOT, scene_id, "cams")
        pairs_path = osp.join(self.ROOT, scene_id, "pair.txt")

        if not self.full_video:
            img_idxs = self.sample_pairs(pairs_path, seq_id)
        else:
            img_idxs = sorted(os.listdir(image_path))
            img_idxs = img_idxs[:: self.kf_every]

        views = []
        imgs_idxs = deque(img_idxs)

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.pop()
            impath = osp.join(image_path, im_idx)
            depthpath = osp.join(depth_path, im_idx.replace(".jpg", ".npy"))
            campath = osp.join(cam_path, im_idx.replace(".jpg", "_cam.txt"))
            maskpath = osp.join(mask_path, im_idx.replace(".jpg", ".png"))

            rgb_image = imread_cv2(impath)
            depthmap = np.load(depthpath)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)

            mask = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED) / 255.0
            mask = mask.astype(np.float32)

            mask[mask > 0.5] = 1.0
            mask[mask < 0.5] = 0.0

            mask = cv2.resize(
                mask,
                (depthmap.shape[1], depthmap.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            kernel = np.ones((10, 10), np.uint8)  # Define the erosion kernel
            mask = cv2.erode(mask, kernel, iterations=1)
            depthmap = depthmap * mask

            cur_intrinsics, camera_pose = self.load_cam_mvsnet(open(campath, "r"))
            intrinsics = cur_intrinsics[:3, :3]
            camera_pose = np.linalg.inv(camera_pose)

            if resolution != (224, 224):
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="dtu",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )

        return views


class NRGBD(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):

        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
            # with open(tuple_path) as f:
            #     self.tuple_list = f.read().splitlines()

        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):

        scenes = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]

        if self.test_id is not None:
            self.scene_list = [self.test_id]

        else:
            self.scene_list = scenes

        print(f"Found {len(self.scene_list)} sequences in split {self.split}")

    def load_poses(self, path):
        file = open(path, "r")
        lines = file.readlines()
        file.close()
        poses = []
        valid = []
        lines_per_matrix = 4
        for i in range(0, len(lines), lines_per_matrix):
            if "nan" in lines[i]:
                valid.append(False)
                poses.append(np.eye(4, 4, dtype=np.float32).tolist())
            else:
                valid.append(True)
                pose_floats = [
                    [float(x) for x in line.split()]
                    for line in lines[i : i + lines_per_matrix]
                ]
                poses.append(pose_floats)

        return np.array(poses, dtype=np.float32), valid

    def _get_views(self, idx, resolution, rng):

        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            img_idxs = line[1:]

        else:
            scene_id = self.scene_list[idx // self.num_seq]

            num_files = len(os.listdir(os.path.join(self.ROOT, scene_id, "images")))
            img_idxs = [f"{i}" for i in range(num_files)]
            img_idxs = img_idxs[:: min(self.kf_every, len(img_idxs) // 2)]

        fx, fy, cx, cy = 554.2562584220408, 554.2562584220408, 320, 240
        intrinsics_ = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        posepath = osp.join(self.ROOT, scene_id, f"poses.txt")
        camera_poses, valids = self.load_poses(posepath)

        imgs_idxs = deque(img_idxs)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs)
        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            impath = osp.join(self.ROOT, scene_id, "images", f"img{im_idx}.png")
            depthpath = osp.join(self.ROOT, scene_id, "depth", f"depth{im_idx}.png")

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0
            depthmap[depthmap > 10] = 0
            depthmap[depthmap < 1e-3] = 0

            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            camera_pose = camera_poses[int(im_idx)]
            # gl to cv
            camera_pose[:, 1:3] *= -1.0
            if resolution != (224, 224) or self.rebuttal:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="nrgbd",
                    label=osp.join(scene_id, im_idx),
                    instance=impath,
                )
            )

        return views


class ScanNetV2(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=10,
        test_id=None,
        full_video=False,
        kf_every=1,
        use_90=True,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.use_90 = use_90

        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        if self.test_id is not None:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
        else:
            self.scene_list = sorted(
                d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
            )
        print(f"Found {len(self.scene_list)} scenes in split {self.split}")

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]
        scene_dir = osp.join(self.ROOT, scene_id)

        # Load intrinsics (shared across all frames in a scene)
        intrinsic_path = osp.join(scene_dir, "intrinsic", "intrinsic_color.txt")
        K_full = np.loadtxt(intrinsic_path, dtype=np.float32)
        intrinsics_ = K_full[:3, :3]

        # Choose between pre-sampled 90-frame subset or full sequence
        if self.use_90 and osp.isdir(osp.join(scene_dir, "color_90")):
            color_dir = osp.join(scene_dir, "color_90")
            depth_dir = osp.join(scene_dir, "depth_90")
            pose_file = osp.join(scene_dir, "pose_90.txt")

            # pose_90.txt: one line per frame, 16 floats (4x4 matrix row-major)
            pose_lines = open(pose_file).read().strip().split("\n")
            frame_names = sorted(os.listdir(color_dir))
            frame_names = frame_names[:: self.kf_every]

            poses = {}
            for i, fname in enumerate(sorted(os.listdir(color_dir))):
                vals = list(map(float, pose_lines[i].split()))
                poses[fname] = np.array(vals, dtype=np.float32).reshape(4, 4)
            img_list = frame_names
        else:
            color_dir = osp.join(scene_dir, "color")
            depth_dir = osp.join(scene_dir, "depth")
            pose_dir = osp.join(scene_dir, "pose")

            all_frames = sorted(os.listdir(color_dir), key=lambda x: int(osp.splitext(x)[0]))
            img_list = all_frames[:: self.kf_every]

            poses = {}
            for fname in img_list:
                fid = osp.splitext(fname)[0]
                pose_path = osp.join(pose_dir, f"{fid}.txt")
                pose = np.loadtxt(pose_path, dtype=np.float32)
                poses[fname] = pose

        views = []
        imgs_idxs = deque(img_list)

        while len(imgs_idxs) > 0:
            fname = imgs_idxs.popleft()
            fid = osp.splitext(fname)[0]

            impath = osp.join(color_dir, fname)
            # depth files are PNG (same numeric id)
            depth_fname = f"{fid}.png"
            depthpath = osp.join(depth_dir, depth_fname)

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0  # mm -> m
            depthmap[depthmap > self.max_thresh] = 0
            depthmap[depthmap < 1e-3] = 0

            camera_pose = poses[fname]  # cam-to-world 4x4
            if "nan" in str(camera_pose) or np.any(np.isnan(camera_pose)):
                continue

            # Resize RGB to match depth resolution if needed
            if rgb_image.shape[:2] != depthmap.shape[:2]:
                rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))

            if resolution != (224, 224):
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="scannetv2",
                    label=osp.join(scene_id, fid),
                    instance=impath,
                )
            )

        return views


class ETH3D(BaseStereoViewDataset):
    def __init__(
        self,
        num_seq=1,
        num_frames=5,
        min_thresh=10,
        max_thresh=100,
        test_id=None,
        full_video=False,
        tuple_list=None,
        seq_id=None,
        rebuttal=False,
        shuffle_seed=-1,
        kf_every=1,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.num_frames = num_frames
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every
        self.seq_id = seq_id
        self.rebuttal = rebuttal
        self.shuffle_seed = shuffle_seed

        # load all scenes
        self.load_all_tuples(tuple_list)
        self.load_all_scenes(ROOT)

    def __len__(self):
        if self.tuple_list is not None:
            return len(self.tuple_list)
        return len(self.scene_list) * self.num_seq

    def load_all_tuples(self, tuple_list):
        if tuple_list is not None:
            self.tuple_list = tuple_list
        else:
            self.tuple_list = None

    def load_all_scenes(self, base_dir):
        self.scene_list = sorted(
            [d for d in os.listdir(base_dir) if os.path.isdir(osp.join(base_dir, d))]
        )
        if self.test_id is not None:
            if isinstance(self.test_id, list):
                self.scene_list = [s for s in self.scene_list if s in self.test_id]
            else:
                self.scene_list = [s for s in self.scene_list if s == self.test_id]
        print(f"Found {len(self.scene_list)} scenes in split {self.split}")

    def _parse_scene_meta(self, scene_dir):
        meta_path = osp.join(scene_dir, "scene_meta.json")
        with open(meta_path, "r") as f:
            scene_meta = json.load(f)
        frames_meta = {item["frame_name"]: item for item in scene_meta["frames"]}
        return frames_meta

    def _select_frame_names(self, all_names, seq_id):
        if len(all_names) == 0:
            return []
        if self.full_video:
            return all_names[:: max(1, self.kf_every)]

        step = max(1, self.kf_every)
        if len(all_names) <= self.num_frames:
            return all_names

        window = step * (self.num_frames - 1) + 1
        max_start = max(0, len(all_names) - window)
        if self.num_seq > 1 and max_start > 0:
            start = int(round((seq_id / max(1, self.num_seq - 1)) * max_start))
        else:
            start = max_start // 2

        frame_names = []
        for i in range(self.num_frames):
            cur = min(start + i * step, len(all_names) - 1)
            frame_names.append(all_names[cur])
        return frame_names

    def _get_views(self, idx, resolution, rng):
        if self.tuple_list is not None:
            line = self.tuple_list[idx].split(" ")
            scene_id = line[0]
            selected_names = [osp.splitext(name)[0] for name in line[1:]]
            seq_id = 0
        else:
            scene_id = self.scene_list[idx // self.num_seq]
            seq_id = idx % self.num_seq

            scene_image_dir = osp.join(self.ROOT, scene_id, "images")
            image_files = sorted(
                [
                    f
                    for f in os.listdir(scene_image_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
            )
            all_names = [osp.splitext(f)[0] for f in image_files]
            selected_names = self._select_frame_names(all_names, seq_id)

        scene_dir = osp.join(self.ROOT, scene_id)
        scene_image_dir = osp.join(scene_dir, "images")
        scene_depth_dir = osp.join(scene_dir, "depth")
        scene_meta = self._parse_scene_meta(scene_dir)

        # Map basename -> full filename for robust loading.
        image_files = sorted(
            [
                f
                for f in os.listdir(scene_image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        image_name_to_file = {osp.splitext(f)[0]: f for f in image_files}

        imgs_idxs = deque(selected_names)
        if self.shuffle_seed >= 0:
            imgs_idxs = shuffle_deque(imgs_idxs, seed=self.shuffle_seed)

        views = []
        while len(imgs_idxs) > 0:
            frame_name = imgs_idxs.popleft()
            if frame_name not in image_name_to_file:
                continue

            img_file = image_name_to_file[frame_name]
            impath = osp.join(scene_image_dir, img_file)
            depthpath = osp.join(scene_depth_dir, f"{frame_name}.exr")

            rgb_image = imread_cv2(impath)
            depthmap = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
            if depthmap is None:
                continue
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)
            depthmap[depthmap < 1e-4] = 0.0

            meta = scene_meta.get(frame_name, scene_meta.get(img_file))
            if meta is None:
                continue

            intrinsics_ = np.array(
                [
                    [meta["fl_x"], 0.0, meta["cx"]],
                    [0.0, meta["fl_y"], meta["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            camera_pose = np.array(meta["transform_matrix"], dtype=np.float32)

            if resolution != (224, 224) or self.rebuttal:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath, fixed_dir=True
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics_, (512, 384), rng=rng, info=impath
                )
                W, H = rgb_image.size
                cx = W // 2
                cy = H // 2
                l, t = cx - 112, cy - 112
                r, b = cx + 112, cy + 112
                crop_bbox = (l, t, r, b)
                rgb_image, depthmap, intrinsics = cropping.crop_image_depthmap(
                    rgb_image, depthmap, intrinsics, crop_bbox
                )
            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="eth3d",
                    label=osp.join(scene_id, frame_name),
                    instance=impath,
                )
            )
        return views


class HiRoom(BaseStereoViewDataset):
    def __init__(
        self,
        *args,
        ROOT,
        **kwargs,
    ):

        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = 1
        self.load_all_scenes(ROOT)

    def __len__(self):

        return len(self.scene_list) * self.num_seq

    def _resolve_scene_list_path(self, base_dir, data_root):
        # Prefer explicit scene list (same split used by raw DA3-BENCH eval).
        candidates = [
            osp.join(base_dir, "selected_scene_list_val.txt"),
            osp.join(osp.dirname(data_root), "selected_scene_list_val.txt"),
        ]
        for path in candidates:
            if osp.exists(path):
                return path
        return None

    def load_all_scenes(self, base_dir):
        scene_list_path = os.path.join(base_dir, "selected_scene_list_val.txt")

        with open(scene_list_path, "r") as f:
            scene_list = [line.strip() for line in f.readlines() if line.strip()]
        self.scene_list = [osp.join(base_dir, 'data', s) for s in scene_list]
        print(f"Found {len(self.scene_list)} scenes")

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]

        scene_dir = scene_id
        image_dir = osp.join(scene_dir, "image")
        depth_dir = osp.join(scene_dir, "depth")
        pose_dir = osp.join(scene_dir, "pose")
        mask_dir = osp.join(scene_dir, "aliasing_mask")
        intr_path = osp.join(scene_dir, "cam_K.npy")

        intrinsics_ = np.load(intr_path).astype(np.float32)

        image_files = sorted(
            [
                f
                for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        depth_files = {
            osp.splitext(f)[0]
            for f in os.listdir(depth_dir)
            if f.lower().endswith((".png", ".npy", ".exr"))
        }
        pose_files = {
            osp.splitext(f)[0]
            for f in os.listdir(pose_dir)
            if f.lower().endswith(".npy")
        }

        all_frame_ids = [
            osp.splitext(f)[0]
            for f in image_files
            if osp.splitext(f)[0] in depth_files and osp.splitext(f)[0] in pose_files
        ]
        imgs_idxs = deque(all_frame_ids)

        views = []
        while len(imgs_idxs) > 0:
            frame_id = imgs_idxs.popleft()
            img_file = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = f"{frame_id}{ext}"
                if osp.exists(osp.join(image_dir, cand)):
                    img_file = cand
                    break
            if img_file is None:
                continue

            impath = osp.join(image_dir, img_file)
            depthpath = osp.join(depth_dir, f"{frame_id}.png")
            posepath = osp.join(pose_dir, f"{frame_id}.npy")
            maskpath = osp.join(mask_dir, f"{frame_id}.png")

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            if depthmap is None:
                raise ValueError(f"Depthmap not found for {impath}")
            # Raw HiRoom stores depth in uint16 mapped to [0, 100] meters.
            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 65535.0 * 100.0
            valid_mask = depthmap > 0.0

            assert osp.exists(maskpath)
            mask = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED)
            
            if mask.ndim == 3:
                mask = mask[..., 0]
                
            aliasing_mask = (mask > 0).astype(np.uint8)
            valid_mask = valid_mask & (aliasing_mask < 0.5)

            depthmap = depthmap * valid_mask.astype(np.float32)

            camera_pose = np.load(posepath).astype(np.float32)
            camera_pose = np.linalg.inv(camera_pose)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics_, resolution, rng=rng, info=impath
            )
        
            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset="hiroom",
                    label=osp.join(os.path.basename(os.path.dirname(scene_id)) + '_' + os.path.basename(scene_id), frame_id),
                    instance=impath,
                )
            )
        return views



class LayeredDepth(BaseStereoViewDataset):
    def __init__(
        self,
        *args,
        ROOT,
        **kwargs,
    ):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = 1
        self.load_all_scenes(ROOT)

    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        self.scene_list = sorted(
            [d for d in os.listdir(base_dir) if os.path.isdir(osp.join(base_dir, d))]
        )
        print(f"Found {len(self.scene_list)} scenes in split {self.split}")

    def _get_views(self, idx, resolution, rng):
        scene_id = self.scene_list[idx // self.num_seq]
        scene_dir = osp.join(self.ROOT, scene_id)

        impath = osp.join(scene_dir, "image.png")
        depthpath = osp.join(scene_dir, "depth_1.png")

        rgb_image = imread_cv2(impath)
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        if depthmap is None:
            raise ValueError(f"Depthmap not found for {impath}")

        depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)
        depthmap[depthmap < 1e-6] = 0.0

        h, w = depthmap.shape[:2]
        f = float(max(w, h))
        intrinsics_ = np.array(
            [[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        camera_pose = np.eye(4, dtype=np.float32)
        
        extra_depthmap_list = []
        for d_idx in range(2, 9):
            d_file = osp.join(scene_dir, f"depth_{d_idx}.png")
            if osp.exists(d_file):
                d = np.array(Image.open(d_file)).astype(np.float32)
                d_complete = d.copy()
                d[~np.isfinite(d)] = 0
                extra_depthmap_list.append(d)

        if extra_depthmap_list:
            extra_depthmap = np.stack(extra_depthmap_list, axis=0)
            # Shape: N_layers, H, W, pick the largest valid depths per pixel
            extra_depthmap = np.maximum.reduce(extra_depthmap)
        else:
            extra_depthmap = np.zeros((0, depthmap.shape[0], depthmap.shape[1]), dtype=np.float32)


        rgb_image, depthmap, extra_depthmap, intrinsics = self._crop_resize_if_necessary2(
            rgb_image, depthmap, extra_depthmap, intrinsics_, resolution, rng=rng, info=idx
        )

        return [
            dict(
                img=rgb_image,
                depthmap=depthmap,
                extra_depthmap=extra_depthmap.astype(np.float32),
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset="layered_depth",
                label=scene_id,
                instance=impath,
            )
        ]