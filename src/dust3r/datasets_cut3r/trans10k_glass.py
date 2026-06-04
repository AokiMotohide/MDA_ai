import os
import os.path as osp
import sys
from PIL import Image
import numpy as np
import torch

sys.path.append(osp.join(osp.dirname(__file__), "..", "..", ".."))
from src.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset


class Trans10KGlassDataset(BaseMultiViewDataset):
    """
    PyTorch Dataset for Trans10K images and glass segmentations.
    """
    def __init__(self, *args, ROOT, split='train', **kwargs):
        """
        Args:
            ROOT (str): Path to the dataset root (e.g., 'data/cut3r_data/glass_segmentations/Trans10K')
            split (str): 'train', 'validation', or 'test'
        """
        self.ROOT = osp.join(ROOT, split)
        self.split = split
        self.is_metric = False
        
        self.image_dir = os.path.join(self.ROOT, 'images')
        self.mask_dir = os.path.join(self.ROOT, 'masks')
        
        super().__init__(*args, **kwargs)

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Directory not found: {self.image_dir}")

        # Get all image filenames
        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        random_indices = rng.choice(len(self.images), num_views, replace=False)
        views = []
        for r_idx in random_indices:
            img_name = self.images[r_idx]
            img_path = os.path.join(self.image_dir, img_name)
            
            # Construct mask path: Trans10K masks are named as {img_name_without_ext}_mask.png
            mask_name = os.path.splitext(img_name)[0] + '_mask.png'
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            image = Image.open(img_path).convert("RGB")
            
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0
            else:
                # Fallback if mask is missing
                w, h = image.size
                mask = np.zeros((h, w), dtype=np.float32)
            
            # Camera Params (Boilerplate as in LayeredDepth_Multi)
            w, h = image.size
            f = max(w, h)
            intrinsics = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
            camera_pose = np.eye(4, dtype=np.float32)

            # Use 1.0 for depthmap to avoid normalization issues (since we don't have depth)
            depthmap = np.ones((h, w), dtype=np.float32)
            extra_depthmap = mask.copy() # Glass mask in extra_depthmap

            # Apply standard multi-view processing (crops/resizes)
            image, depthmap, extra_depthmap, intrinsics = self._crop_resize_if_necessary2(
                image, depthmap, extra_depthmap, intrinsics, resolution, rng=rng, info=r_idx
            )
 
            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, 0, rng, p=[1.0, 0.0, 0.0])

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap.astype(np.float32),
                    depth_complete=depthmap.astype(np.float32),
                    extra_depthmap=depthmap.astype(np.float32),
                    glass_mask=extra_depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="trans10k_glass",
                    label=img_name,
                    instance=f"{r_idx}",
                    is_metric=self.is_metric,
                    is_video=False,
                    quantile=np.array(1.0, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=True,
                    reset=False,
                )
            )
        assert len(views) == num_views
        return views, [0]
