import itertools
import random
import types
import numpy as np
import PIL
import torch
import torchvision.transforms as tvf

import src.dust3r.datasets_cut3r.utils.cropping as cropping
from src.dust3r.datasets.base.easy_dataset import EasyDataset  # here we use fast's EasyDataset
from src.dust3r.datasets_cut3r.utils.transforms import SeqColorJitter
from src.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

ImgNorm = tvf.Compose(
    [tvf.ToTensor(), tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def normalize_camera_pose_w_points(
    views,
    first_view_camera_pose_inv=None,
    median_dist=None,
):
    if first_view_camera_pose_inv is None:
        first_view_camera_pose = views[0]["camera_pose"].copy()
        first_view_camera_pose_inv = np.linalg.inv(first_view_camera_pose)

    camera_poses = [first_view_camera_pose_inv @ view["camera_pose"] for view in views]
    camera_poses = np.stack(camera_poses, axis=0)

    # normalize
    if median_dist is None:
        for i, view in enumerate(views):
            view["camera_pose"] = camera_poses[i]

            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
            views[i]["pts3d"] = pts3d
            views[i]["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

        all_pts3d = np.concatenate([view["pts3d"] for view in views], axis=0)
        all_valid_mask = np.concatenate([view["valid_mask"] for view in views], axis=0)
        median_dist = np.mean(
            np.linalg.norm(all_pts3d[all_valid_mask].reshape(-1, 3), axis=-1)
        )
        if (not np.isfinite(median_dist)) or (median_dist <= 1e-5):
            median_dist = 1.0

    camera_poses[:, :3, 3] = camera_poses[:, :3, 3] / median_dist

    for i in range(len(views)):
        views[i]["camera_pose"] = camera_poses[i]
        views[i]["pts3d"] = views[i]["pts3d"] / median_dist

    return views, first_view_camera_pose_inv, median_dist


class BaseMultiViewDataset(EasyDataset):
    """Define all basic options.

    Usage:
        class MyDataset (BaseMultiViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(
        self,
        *,  # only keyword arguments
        num_views=None,
        split=None,
        resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
        transform=ImgNorm,
        aug_crop=False,
        n_corres=0,
        nneg=0,
        seed=None,
        allow_repeat=False,
        seq_aug_crop=False,
        rand_sampling=False,
        imgcrop=0,
        **kwargs,
    ):
        assert num_views is not None, "undefined num_views"
        self.num_views = num_views
        self.split = split
        self._set_resolutions(resolution)

        self.n_corres = n_corres
        self.nneg = nneg
        assert (
            self.n_corres == "all"
            or isinstance(self.n_corres, int)
            or (isinstance(self.n_corres, list) and len(self.n_corres) == self.num_views)
        ), f"Error, n_corres should either be 'all', a single integer or a list of length {self.num_views}"
        assert self.nneg == 0 or self.n_corres != "all", "nneg should be 0 if n_corres is all"

        if isinstance(transform, str):
            transform = eval(transform)

        if isinstance(transform, types.FunctionType) and transform.__name__ == "SeqColorJitter":
            transform = SeqColorJitter()
        self.transform = transform

        self.aug_crop = aug_crop
        self.seed = seed
        self.allow_repeat = allow_repeat
        self.seq_aug_crop = seq_aug_crop
        self.rand_sampling = rand_sampling
        self.imgcrop = imgcrop

    def __len__(self):
        return len(self.scenes)

    @staticmethod
    def blockwise_shuffle(x, rng, block_shuffle):
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """
        args:
            num_views: number of views to return
            id_ref: the reference id (first id)
            ids_all: all the ids
            rng: random number generator
            max_interval: maximum interval between two views
        returns:
            pos: list of positions of the views in ids_all, i.e., index for ids_all
            is_video: True if the views are consecutive
        """
        assert num_views >= 1, f"num_views should be >= 1, got {num_views}"
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        assert id_ref in ids_all, "??????"
        pos_ref = ids_all.index(id_ref)

        # Single-view sampling is a degenerate case: always return the reference frame.
        if num_views == 1:
            return [pos_ref], False

        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            intervals = [
                rng.choice(range(min_interval, max_interval + 1)) for _ in range(num_views - 1)
            ]

            # if video or collection
            if rng.random() < video_prob:
                # if fixed interval or random
                if rng.random() < fix_interval_prob:
                    # regular interval
                    fixed_interval = rng.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = pos + rng.choice(pos_candidates, num_views - len(pos), replace=False).tolist()

            pos = sorted(pos) if is_video else self.blockwise_shuffle(pos, rng, block_shuffle)
        else:
            # assert self.allow_repeat
            uniq_num = remaining_sum
            new_pos_ref = rng.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [
                rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
            ]

            revisit_random = rng.random()
            video_random = rng.random()

            if rng.random() < fix_interval_prob and video_random < video_prob:
                # regular interval
                fixed_interval = rng.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:  # revisit, video / collection
                is_video = video_random < video_prob
                pos = self.blockwise_shuffle(pos, rng, block_shuffle) if not is_video else pos
                num_full_repeat = num_views // uniq_num
                pos = pos * num_full_repeat + pos[: num_views - len(pos) * num_full_repeat]
            elif revisit_random < 0.9:  # random
                pos = rng.choice(pos, num_views, replace=True)
            else:  # ordered
                pos = sorted(rng.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        return pos, is_video

    def get_img_and_ray_masks(self, is_metric, v, rng, p=[0.8, 0.15, 0.05]):
        # generate img mask and raymap mask
        if v == 0 or (not is_metric):
            img_mask = True
            ray_mask = False
        else:
            img_mask = True
            ray_mask = False
        return img_mask, ray_mask

    def _filter_depth_by_quantile(self, depthmap, quantile=0.98):
        valid_mask = depthmap > 0
        if not np.any(valid_mask):
            return depthmap

        depth_thresh = np.quantile(depthmap[valid_mask], quantile).astype(np.float32)
        depthmap[depthmap > depth_thresh] = -1.0
        return depthmap

    def get_stats(self):
        return f"{len(self)} groups of views"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, resolution, rng, num_views):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if isinstance(idx, (tuple, list, np.ndarray)):
            # the idx is specifying the aspect-ratio
            if len(idx) == 3:
                idx, ar_idx, nview = idx
            else:
                idx, ar_idx = idx
                nview = self.num_views
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
            nview = self.num_views

        assert nview >= 1 and nview <= self.num_views
        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.randint(0, 2**32, (1,)).item()
            self._rng = np.random.default_rng(seed=seed)

        if self.aug_crop > 1 and self.seq_aug_crop:
            self.delta_target_resolution = self._rng.integers(0, self.aug_crop)

        resolution = self._resolutions[
            ar_idx
        ]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        try:
            views, pos = self._get_views(idx, resolution, self._rng, nview)
        except Exception as e:
            print(f"Error in _get_views for view {idx}: {e}")
            return self.__getitem__(((idx + 999) % len(self), ar_idx))

        assert len(views) == nview

        if "camera_pose" not in views[0]:
            views[0]["camera_pose"] = np.ones((4, 4), dtype=np.float32)

        views, first_view_camera_pose_inv, median_dist = normalize_camera_pose_w_points(
            views
        )

        imgs_tmp = [view["img"] for view in views]
        if (np.abs(np.array(imgs_tmp)) < 1e-6).all():
            print("All images are zero for view")
            for k, v in views[0].items():
                if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
                    print("dtype", k, v.shape, v.max(), v.min())
                else:
                    print("dtype", k, (v))

        for v, view in enumerate(views):
            view["idx"] = (idx, ar_idx, v)

            # encode the image
            width, height = view["img"].size

            view["true_shape"] = np.int32((height, width))
            view["img"] = ImgNorm(view["img"])
            blk_img_flag = view["img"].max() == -1  # some training imgs are purely black.
            if "sky_mask" not in view:
                view["sky_mask"] = np.zeros_like(view["depthmap"], dtype=np.float32)
                view["sky_mask_valid"] = np.zeros_like(view["depthmap"], dtype=np.float32)
            else:
                view["sky_mask_valid"] = np.ones_like(view["depthmap"], dtype=np.float32)

            assert "camera_intrinsics" in view
            if "camera_pose" not in view:
                view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(
                    view["camera_pose"]
                ).all(), f"NaN in camera pose for view {view_name(view)}"

            assert np.isfinite(
                view["depthmap"]
            ).all(), f"NaN in depthmap for view {view_name(view)}"

            view["depthmap"] = view["depthmap"] / median_dist
            view["depth_complete"] = view["depth_complete"] / median_dist
            if "extra_depthmap" in view:
                view["extra_depthmap"] = view["extra_depthmap"] / median_dist
            else:
                view["extra_depthmap"] = view["depthmap"].copy() * 0.0

            if "glass_mask" not in view:
                view["glass_mask"] = view["extra_depthmap"].copy() * 0.0

            if "pts3d" not in view:
                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
                view["pts3d"] = pts3d
                view["valid_mask"] = (
                    valid_mask & np.isfinite(pts3d).all(axis=-1) & (not blk_img_flag)
                )

            if blk_img_flag:
                print(f"{view['dataset']}-{view['instance']}-{view['label']} is purely black!")

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            view["is_metric_scale"] = view["is_metric"]  # for loss compat issue

        return views

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f"Bad type for {width=} {type(width)=}, should be int"
            assert isinstance(
                height, int
            ), f"Bad type for {height=} {type(height)=}, should be int"
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(
        self, image, depthmap, intrinsics, resolution, rng=None, info=None
    ):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        # transpose the resolution if necessary
        W, H = image.size  # new size

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        return image, depthmap, intrinsics2

    def _crop_resize_if_necessary2(
        self, image, depthmap, depthmap_complete, intrinsics, resolution, rng=None, info=None
    ):
        """Principal-point-centered safe crop, Lanczos downscale, then a final tight crop to `resolution`."""
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        if (
            not self.rand_sampling
            and self.imgcrop == 1
            and ((rng.random() if rng is not None else random.random()) < 0.5)
        ):
            # Randomly tighten the principal-point-centered crop (zoom-in augmentation).
            crop_scale = rng.uniform(0.2, 1.0) if rng is not None else random.uniform(0.2, 1.0)
            min_margin_x = max(1, int(min_margin_x * crop_scale))
            min_margin_y = max(1, int(min_margin_y * crop_scale))

        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        image, depthmap, depthmap_complete, intrinsics = cropping.crop_image_depthmap2(
            image, depthmap, depthmap_complete, intrinsics, (l, t, r, b)
        )

        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )

        rescale = (
            cropping.rescale_image_depthmap2_sampling
            if self.rand_sampling
            else cropping.rescale_image_depthmap2
        )
        image, depthmap, depthmap_complete, intrinsics = rescale(
            image, depthmap, depthmap_complete, intrinsics, target_resolution
        )

        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, depthmap_complete, intrinsics2 = cropping.crop_image_depthmap2(
            image, depthmap, depthmap_complete, intrinsics, crop_bbox
        )

        return image, depthmap, depthmap_complete, intrinsics2


def is_good_type(v):
    """returns (is_good, err_msg)"""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8, np.float64):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view["true_shape"]

    if width < height:
        # rectify portrait to landscape
        assert view["img"].shape == (3, height, width)
        view["img"] = view["img"].swapaxes(1, 2)

        assert view["valid_mask"].shape == (height, width)
        view["valid_mask"] = view["valid_mask"].swapaxes(0, 1)

        assert view["depthmap"].shape == (height, width)
        view["depthmap"] = view["depthmap"].swapaxes(0, 1)

        assert view["pts3d"].shape == (height, width, 3)
        view["pts3d"] = view["pts3d"].swapaxes(0, 1)

        # transpose x and y pixels
        view["camera_intrinsics"] = view["camera_intrinsics"][[1, 0, 2]]

        assert view["ray_map"].shape == (height, width, 6)
        view["ray_map"] = view["ray_map"].swapaxes(0, 1)

        assert view["sky_mask"].shape == (height, width)
        view["sky_mask"] = view["sky_mask"].swapaxes(0, 1)

        if "corres" in view:
            # transpose correspondences x and y
            view["corres"][0] = view["corres"][0][:, [1, 0]]
            view["corres"][1] = view["corres"][1][:, [1, 0]]
