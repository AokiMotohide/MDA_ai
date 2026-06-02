import os
import json
import torch
from PIL import Image
import numpy as np
import sys
import os.path as osp

sys.path.append(osp.join(osp.dirname(__file__), "..", "..", ".."))
from src.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset


class ADE20KGlassDataset(BaseMultiViewDataset):
    """
    PyTorch Dataset for ADE20K images and glass segmentations.
    Glass is identified by 'windowpane, window' or 'glass, plate glass' in the metadata.
    """
    def __init__(self, *args, ROOT, split='train', **kwargs):
        """
        Args:
            ROOT (str): Path to the extracted ADE20K dataset (e.g., 'data/cut3r_data/glass_segmentations/extracted_ADE20K').
            split (str): 'train' or 'validation'.
        """
        self.ROOT = os.path.join(ROOT, split)
        self.split = split
        self.images_dir = os.path.join(self.ROOT, 'images')
        self.meta_dir = os.path.join(self.ROOT, 'meta')
        self.instances_dir = os.path.join(self.ROOT, 'instances')
        self.is_metric = False # ADE20K doesn't have metric depth by default in this setup
        
        super().__init__(*args, **kwargs)

        # List all image files
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Directory not found: {self.images_dir}")
        self.image_filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.png'))])
        
    def __len__(self):
        return len(self.image_filenames)

    def _get_views(self, idx, resolution, rng, num_views):
        # ADE20K is usually single view, but we can support num_views by random sampling if needed
        # or just return the same image if num_views > 1. 
        # LayeredDepth_Multi samples random indices.
        random_indices = rng.choice(len(self.image_filenames), num_views, replace=False)
        views = []
        for r_idx in random_indices:
            img_name = self.image_filenames[r_idx]
            img_path = os.path.join(self.images_dir, img_name)
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Load metadata to find glass instances
            base_name = os.path.splitext(img_name)[0]
            meta_path = os.path.join(self.meta_dir, f"{base_name}.json")
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            glass_keywords = ['window', 'glass', 'mirror']
            
            glass_instance_ids = []
            for obj in meta.get('objects', []):
                if any(kw in obj['name'].lower() for kw in glass_keywords):
                    glass_instance_ids.append(obj['id'])
                elif any(kw in obj['attributes'].lower() for kw in glass_keywords):
                    glass_instance_ids.append(obj['id'])
            
            # Create a combined mask
            width, height = image.size
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Load each glass instance mask and merge
            instance_folder = os.path.join(self.instances_dir, base_name)
            if os.path.exists(instance_folder) and glass_instance_ids:
                for inst_id in glass_instance_ids:
                    inst_mask_path = os.path.join(instance_folder, f"inst_{inst_id:03d}.png")
                    if os.path.exists(inst_mask_path):
                        inst_mask = np.array(Image.open(inst_mask_path))
                        mask[inst_mask > 0] = 1
            
            # Camera Params (Boilerplate as in LayeredDepth_Multi)
            w, h = image.size
            f = max(w, h)
            intrinsics = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
            camera_pose = np.eye(4, dtype=np.float32)

            # For ADE20K glass, we don't have depth, so we use the mask as a "depth-like" representation 
            # or just zeros if we want to match the format. 
            # LayeredDepth_Multi uses depthmap and extra_depthmap.
            # Let's put the glass mask in extra_depthmap to match the "layered" concept if appropriate,
            # but usually depthmap is required.
            depthmap = np.ones((h, w), dtype=np.float32)
            extra_depthmap = mask

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
                    dataset="ade20k_glass",
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

if __name__ == "__main__":
    # Quick test
    root = "data/cut3r_data/glass_segmentations/extracted_ADE20K"
    dataset = ADE20KGlassDataset(ROOT=root, split='train', num_views=1, resolution=(512, 512))
    print(f"Dataset size: {len(dataset)}")
    
    # Create logs directory for visualization
    log_dir = "logs/glass_seg_examples"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Saving visualization examples to {log_dir}")

    # Visualize first 10 examples
    for i in range(min(30, len(dataset))):
        views = dataset[i]
        view = views[0]
        
        # Convert back to numpy for visualization
        # img is already normalized by ImgNorm in BaseMultiViewDataset.__getitem__
        # We need to unnormalize it or just use a simple visualization
        img_tensor = view["img"]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 0.225 + 0.45).clip(0, 1) # Rough unnorm
        img_np = (img_np * 255).astype(np.uint8)
        
        # mask is in extra_depthmap
        mask_tensor = view["extra_depthmap"]
        mask_np = (mask_tensor * 255).astype(np.uint8)
        
        # Create a 3-channel mask for concatenation
        mask_rgb = np.stack([mask_np, mask_np, mask_np], axis=-1)
        
        # Create overlay
        overlay = (0.5 * img_np + 0.5 * mask_rgb).astype(np.uint8)
        
        # Concatenate horizontally: (1) image, (2) overlay
        vis_img = np.concatenate([img_np, overlay], axis=1)
        
        # Save using PIL
        save_path = os.path.join(log_dir, f"example_{i:03d}.png")
        Image.fromarray(vis_img).save(save_path)
        
        if i == 0:
            print(f"Image shape: {img_tensor.shape}, Mask shape: {mask_tensor.shape}")
    
    print("Done.")

