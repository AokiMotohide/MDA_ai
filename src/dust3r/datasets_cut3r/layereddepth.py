import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
import sys
import matplotlib.pyplot as plt
import json
import random

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from src.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset


def random_sequence(max_frames, seq_len, max_interval=3):
    """
    Randomly select seq_len frame indices from [0, max_frames),
    with interval between consecutive frames in [1, max_interval].

    Returns:
        list of indices (sorted, increasing)
    """
    assert max_frames > 0
    assert seq_len > 0
    assert max_interval >= 1

    while True:
        start = random.randint(0, max_frames - 1)
        seq = [start]

        for _ in range(seq_len - 1):
            step = random.randint(1, max_interval)
            next_idx = seq[-1] + step
            if next_idx >= max_frames:
                break
            seq.append(next_idx)

        if len(seq) == seq_len:
            return seq


class LayeredDepth_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, split="train", only_validated_files=True, is_video=False, **kwargs):
        self.ROOT = osp.join(ROOT, split)  # Point to /train or /val
        self.split = split
        self.is_metric = True
        self.only_validated_files = only_validated_files
        self.is_video = is_video
        # kwargs["num_views"] = 1
        super().__init__(*args, **kwargs)

        self._load_data()

    def _load_data(self):
        # The folders are named 0, 1, 2...
        # We find all numeric directories and sort them
        if not osp.exists(self.ROOT):
            raise FileNotFoundError(f"Directory not found: {self.ROOT}")

        if self.only_validated_files:
            valid_json_file = "data/cut3r_data/layered_depth/trainset_valid_samples.json"
            with open(valid_json_file, "r") as f:
                self.valid_samples = json.load(f)
            self.sample_dirs = [sample["scene_name"] for sample in self.valid_samples]
        else:
            self.sample_dirs = sorted(
                [d for d in os.listdir(self.ROOT) if d.isdigit()], key=lambda x: int(x)
            )
        self.total_len = len(self.sample_dirs)
        print(f"Loaded LayeredDepth from {self.ROOT} with {self.total_len} samples.")

    def __len__(self):
        return self.total_len

    def _get_views(self, idx, resolution, rng, num_views):
        # Get the specific folder for this index
        random_indices = rng.choice(len(self.sample_dirs), num_views, replace=False)
        if self.is_video:
            random_indices = random_sequence(len(self.sample_dirs), num_views)
            if random.random() < 0.5:
                random_indices = random_indices[::-1]
            
            if random.random() < 0.5:
                # rand permute
                random_indices = random.sample(random_indices, len(random_indices))
                
        views = []
        for idx in random_indices:
            folder_path = osp.join(self.ROOT, self.sample_dirs[idx])

            # 1. Load RGB
            rgb_path = osp.join(folder_path, "image.png")
            rgb_image = Image.open(rgb_path).convert("RGB")

            # 2. Load Primary Depth (Assuming depth_1 is primary, or whichever you prefer)
            # Using depth_1.png as the main depth map
            depth_path = osp.join(folder_path, "depth_1.png")
            depthmap = np.array(Image.open(depth_path)).astype(np.float32)

            # 3. Load Extra Depths (depth_2 through depth_8)
            # Note: Camera params needed for cropping - compute temporarily
            w_temp, h_temp = rgb_image.size
            f_temp = max(w_temp, h_temp)
            intrinsics_temp = np.array(
                [[f_temp, 0, w_temp / 2], [0, f_temp, h_temp / 2], [0, 0, 1]], dtype=np.float32
            )

            extra_depthmap_list = []
            for d_idx in range(2, 9):
                d_file = osp.join(folder_path, f"depth_{d_idx}.png")
                if osp.exists(d_file):
                    d = np.array(Image.open(d_file)).astype(np.float32)
                    d_complete = d.copy()
                    d[~np.isfinite(d)] = 0
                    # Create a temporary PIL image for cropping
                    # d_pil = Image.fromarray(d.astype(np.uint8))
                    # _, d_cropped, d_complete_cropped, _ = self._crop_resize_if_necessary2(
                    #     d_pil, d, d_complete, intrinsics_temp, resolution, rng=rng, info=idx
                    # )
                    extra_depthmap_list.append(d)

            if extra_depthmap_list:
                extra_depthmap = np.stack(extra_depthmap_list, axis=0)
                # Shape: N_layers, H, W, pick the largest valid depths per pixel
                extra_depthmap = np.maximum.reduce(extra_depthmap)
            else:
                extra_depthmap = np.zeros((0, depthmap.shape[0], depthmap.shape[1]), dtype=np.float32)

            # do the cropping/resizing here for exrtra depths as well
            # extra_depthmap_pil = Image.fromarray(extra_depthmap.copy().astype(np.uint8))  # Use first for size
            # _, extra_depthmap, _, _ = self._crop_resize_if_necessary2(
            #     extra_depthmap_pil,
            #     extra_depthmap,
            #     extra_depthmap.copy(),
            #     intrinsics_temp,
            #     resolution,
            #     rng=rng,
            #     info=idx,
            # )
            # --- Camera Params (Boilerplate) ---
            w, h = rgb_image.size
            f = max(w, h)
            intrinsics = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float32)
            camera_pose = np.eye(4, dtype=np.float32)

            # depth_complete = depthmap.copy()
            depthmap[~np.isfinite(depthmap)] = 0

            # Apply standard multi-view processing (crops/resizes)
            rgb_image, depthmap, extra_depthmap, intrinsics = self._crop_resize_if_necessary2(
                rgb_image, depthmap, extra_depthmap, intrinsics, resolution, rng=rng, info=idx
            )

            extra_depthmap[ np.absolute(extra_depthmap - depthmap) < 1e-6] = 0

            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, 0, rng, p=[1.0, 0.0, 0.0])
            
            depthmap_max = np.max(depthmap)
            depthmap = depthmap / (depthmap_max + 1e-8)
            extra_depthmap = extra_depthmap / (depthmap_max + 1e-8)

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),  # Primary depth H,W
                    depth_complete=extra_depthmap.copy(),
                    extra_depthmap=extra_depthmap.astype(
                        np.float32
                    ),  # Extra depths N_layers,H,W, 0 is invalid
                    glass_mask=(extra_depthmap.astype(np.float32) > 0).astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="layered_depth",
                    label=self.sample_dirs[idx],
                    instance=f"{idx}",
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
    # Example usage and visualization
    # Set the ROOT path via environment variable or default to a local path
    root_path = "/nfs/turbo/coe-jungaocv-turbo2/shared_data/tmp/LayeredDepth/layered_depth"
    dataset = LayeredDepth_Multi(ROOT=root_path, split="train", resolution=(504, 378))
    print(f"Dataset length: {len(dataset)}")

    num_to_viz = min(5, len(dataset))
    if num_to_viz > 0:
        fig, axes = plt.subplots(num_to_viz, 3, figsize=(15, 5 * num_to_viz))

        for i in range(num_to_viz):
            idx = i * (len(dataset) // num_to_viz)  # Spread samples across dataset
            sample = dataset[idx]
            view = sample[0]

            row_axes = axes[i] if num_to_viz > 1 else axes

            # Visualize RGB
            rgb = view["img"].permute(1, 2, 0).cpu().numpy()
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            row_axes[0].imshow(rgb)
            row_axes[0].set_title(f"RGB {view['label']}")

            # Visualize Primary Depth
            depth = view["depthmap"]
            if torch.is_tensor(depth):
                depth = depth.cpu().numpy()
            row_axes[1].imshow(depth)
            row_axes[1].set_title(f"Primary Depth {idx}")

            # Visualize Extra Depths (Merged)
            extra_depth = view["extra_depthmap"]
            if torch.is_tensor(extra_depth):
                extra_depth = extra_depth.cpu().numpy()
            row_axes[2].imshow(extra_depth)
            row_axes[2].set_title("Extra Depth")

        plt.tight_layout()
        save_path = "layered_depth_samples_visualization.png"
        fig.savefig(save_path)
        print(f"Saved visualization of {num_to_viz} samples to {save_path}")
