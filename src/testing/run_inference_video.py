"""
Video / sequence inference: feed all images in a folder to the model in a
single forward pass (multi-view mode), then save per-frame results.

Unlike run_inference_folder.py which processes each image independently,
this script feeds the entire sequence at once so the model can leverage
cross-frame information.

Example:
    python src/testing/run_inference_video.py \
        --model_name mda_mog_sky_l2 \
        --img_path eval_results/teaser_imgs \
        --output_dir eval_results/video_inference/teaser
"""

import os
import sys
import torch
import argparse
import numpy as np
from glob import glob
from PIL import Image
from PIL.ImageOps import exif_transpose

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.testing.utils.model_choice import choose_model, CONFIGS
from src.dust3r.utils.image import ImgNorm
from src.testing.eval_cut3r.video_depth.utils import save_depth_maps
from depth_anything_3.model.utils.transform import (
    pose_encoding_to_extri_intri,
    unproject_depth_map_to_point_map,
)


def _resize_pil_image_local(img, long_edge_size):
    src_long_edge = max(img.size)
    if src_long_edge > long_edge_size:
        interp = Image.LANCZOS
    else:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / src_long_edge)) for x in img.size)
    return img.resize(new_size, interp)


def load_images_for_eval_safe(
    folder_or_list, size, square_ok=False, verbose=True, crop=True,
    patch_size=16, img_norm=None,
):
    if isinstance(folder_or_list, str):
        if verbose:
            print(f">> Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))
    elif isinstance(folder_or_list, list):
        if verbose:
            print(f">> Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list
    else:
        raise ValueError(f"bad {folder_or_list=} ({type(folder_or_list)})")

    supported_images_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    imgs = []
    norm = ImgNorm if img_norm is None else img_norm

    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(Image.open(os.path.join(root, path))).convert("RGB")

        w_src, h_src = img.size
        if size == 224:
            img = _resize_pil_image_local(img, round(size * max(w_src / h_src, h_src / w_src)))
        else:
            img = _resize_pil_image_local(img, size)

        w_resized, h_resized = img.size
        cx, cy = w_resized // 2, h_resized // 2

        if size == 224:
            half = min(cx, cy)
            if crop:
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:
                target_w = int(2 * half)
                target_h = int(2 * half)
                img = img.resize((target_w, target_h), Image.LANCZOS)
        else:
            halfw = ((2 * cx) // patch_size) * (patch_size // 2)
            halfh = ((2 * cy) // patch_size) * (patch_size // 2)
            if (not square_ok) and (w_resized == h_resized):
                halfh = int(round(3 * halfw / 4))

            if crop:
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
            else:
                target_w = int(2 * halfw)
                target_h = int(2 * halfh)
                img = img.resize((target_w, target_h), Image.LANCZOS)

        w_out, h_out = img.size
        if verbose:
            print(f" - adding {path} with resolution {w_src}x{h_src} --> {w_out}x{h_out}")

        imgs.append(
            dict(
                img=norm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
            )
        )

    assert imgs, "no images found at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")
    return imgs


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Run video/sequence inference with Depth-Anything-3."
    )
    parser.add_argument("--model_name", type=str, default="mda_mog_sky_l2",
                        help="name of the model (see src/testing/utils/model_choice.py)")
    parser.add_argument("--img_path", type=str, required=True,
                        help="Path to folder of sequential images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to save inference results")
    parser.add_argument("--size", type=int, default=512, help="Image size for inference")

    # Model configs
    parser.add_argument("--crop_center_112", type=int, default=0)
    parser.add_argument("--cam_inp", type=int, default=0)
    parser.add_argument("--gt_cam_output", type=int, default=0)
    parser.add_argument("--output_normalize", type=int, default=0)
    parser.add_argument("--output_double_layers", type=int, default=0)
    parser.add_argument("--output_sharp_boundary", type=int, default=0)
    parser.add_argument("--render_sideview", type=int, default=1,
                        help="Render sideview point cloud visualization")
    parser.add_argument("--max_chunk", type=int, default=64,
                        help="Max frames fed to the model per forward pass "
                             "(chunked when num_views > max_chunk). "
                             "Note: chunking breaks cross-chunk attention.")
    parser.add_argument("--pcd_depth_min", type=float, default=1e-3,
                        help="Minimum depth (inclusive) for sideview point cloud.")
    parser.add_argument("--pcd_depth_max", type=float, default=float("inf"),
                        help="Maximum depth (exclusive) for sideview point cloud.")
    return parser


def prepare_views(img_list, size, patch_size, img_norm):
    """Prepare views for ALL frames in the sequence (single inference call).

    The model attends across all views in a single forward pass and requires
    a uniform spatial size whose H and W are both multiples of ``patch_size``
    (14 or 16, returned by choose_model). To handle folders with mixed
    aspect ratios without distorting the geometry, every frame is first
    *center-cropped* to a common aspect ratio (locked by frame 0's loaded
    dims), then resized to a common (target_h, target_w) that's snapped down
    to a multiple of ``patch_size``.
    """
    images = load_images_for_eval_safe(
        img_list, size=size, crop=False, patch_size=patch_size, img_norm=img_norm,
        square_ok=True,
    )

    # Frame 0 fixes the aspect ratio and the final (target_h, target_w).
    # Snap down to multiples of patch_size required by the backbone.
    h0, w0 = images[0]["img"].shape[-2:]
    target_h = (h0 // patch_size) * patch_size
    target_w = (w0 // patch_size) * patch_size
    assert target_h > 0 and target_w > 0, (
        f"Target dims after patch-size snap are non-positive: "
        f"h0={h0}, w0={w0}, patch_size={patch_size}"
    )
    ar_target = target_h / target_w

    for img_data in images:
        img_t = img_data["img"]  # (1, 3, H, W)
        cur_h, cur_w = img_t.shape[-2:]

        # 1) Center-crop the largest region with aspect ratio == ar_target.
        if cur_h / cur_w > ar_target:
            crop_h = int(round(cur_w * ar_target))
            crop_w = cur_w
        else:
            crop_h = cur_h
            crop_w = int(round(cur_h / ar_target))
        top = (cur_h - crop_h) // 2
        left = (cur_w - crop_w) // 2
        img_t = img_t[..., top:top + crop_h, left:left + crop_w]

        # 2) Resize to (target_h, target_w) only if cropping didn't already
        #    land us there.
        if img_t.shape[-2:] != (target_h, target_w):
            img_t = torch.nn.functional.interpolate(
                img_t, size=(target_h, target_w),
                mode="bilinear", align_corners=False, antialias=True,
            )

        img_data["img"] = img_t
        img_data["true_shape"] = np.int32([[target_h, target_w]])

    views = []
    for i, img_data in enumerate(images):
        view = {
            "img": img_data["img"],
            "ray_map": torch.full(
                (img_data["img"].shape[0], 6,
                 img_data["img"].shape[-2], img_data["img"].shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(img_data["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(
                np.eye(4).astype(np.float32)
            ).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views


def _preds_to_cpu(obj):
    """Recursively move tensors in a predictions dict to CPU to free GPU memory."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _preds_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_preds_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_preds_to_cpu(v) for v in obj)
    return obj


def _concat_frame_dim(values):
    """Concatenate a list of per-chunk values along the frame axis.

    - Tensors with ndim >= 2: concat along dim=1 (batch=0, frame=1).
    - Tensors with ndim == 1: concat along dim=0 (assume frame dim is 0).
    - Lists: recurse element-wise when lengths match; otherwise flatten.
    - Other types (scalars, strings): keep first-chunk value.
    """
    if len(values) == 1:
        return values[0]
    head = values[0]
    if isinstance(head, torch.Tensor):
        if all(isinstance(v, torch.Tensor) for v in values):
            dim = 1 if head.ndim >= 2 else 0
            return torch.cat(values, dim=dim)
        return head
    if isinstance(head, list):
        if all(isinstance(v, list) and len(v) == len(head) for v in values):
            return [_concat_frame_dim([v[i] for v in values]) for i in range(len(head))]
        merged = []
        for v in values:
            merged.extend(v)
        return merged
    return head


def _merge_chunk_predictions(chunks):
    """Merge per-chunk predictions into a single predictions dict.

    Concatenates the fields this script reads downstream along the frame axis.
    `views` is kept as the first chunk's copy because its dict structure is
    complex; MoG visualization is invoked per-chunk separately.
    """
    if len(chunks) == 1:
        return chunks[0]

    merged = dict(chunks[0])
    for key in ("depth", "images", "pose_enc", "sky_mask"):
        if all(key in c for c in chunks):
            merged[key] = _concat_frame_dim([c[key] for c in chunks])

    if all("raw_preds" in c for c in chunks):
        raw_merged = dict(chunks[0]["raw_preds"])
        raw_keys = set()
        for c in chunks:
            raw_keys.update(c["raw_preds"].keys())
        for key in raw_keys:
            if not all(key in c["raw_preds"] for c in chunks):
                continue
            raw_merged[key] = _concat_frame_dim([c["raw_preds"][key] for c in chunks])
        merged["raw_preds"] = raw_merged

    return merged


def main():
    args = get_args_parser().parse_args()

    CONFIGS["crop_center_112"] = bool(args.crop_center_112)
    CONFIGS["cam_inp"] = bool(args.cam_inp)
    CONFIGS["gt_cam_output"] = bool(args.gt_cam_output)
    CONFIGS["output_normalize"] = bool(args.output_normalize)
    CONFIGS["output_double_layers"] = bool(args.output_double_layers)
    CONFIGS["output_sharp_boundary"] = bool(args.output_sharp_boundary)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading model: {args.model_name}")
    loaded = choose_model(args.model_name)
    model = loaded.model
    patch_size = loaded.patch_size
    img_norm = loaded.img_norm
    model.to(device)
    model.eval()

    inference_size = args.size
    name_lower = args.model_name.lower()
    is_mog = "mog" in name_lower
    is_ppd = name_lower.startswith("ppd")

    # Collect all images in the folder (sorted for temporal order).
    # Keep this list aligned with the allow-list in load_images_for_eval_safe.
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    filelist = []
    for ext in extensions:
        filelist.extend(glob(os.path.join(args.img_path, ext)))
        filelist.extend(glob(os.path.join(args.img_path, ext.upper())))
    filelist.sort()

    if not filelist:
        print(f"No images found in {args.img_path}")
        return

    print(f"Found {len(filelist)} images. Running sequence inference...")

    if args.output_sharp_boundary:
        output_dir = os.path.join(args.output_dir, args.model_name + "_sharp_boundary")
    else:
        output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare ALL views at once (the key difference from run_inference_folder.py)
    views = prepare_views(filelist, inference_size, patch_size, img_norm)

    num_views = len(filelist)
    max_chunk = max(1, args.max_chunk)

    # Chunked inference: the model sees at most max_chunk frames per forward pass.
    # Per-chunk predictions are moved to CPU to free GPU memory between chunks,
    # then concatenated along the frame dim.
    chunk_predictions = []
    for chunk_start in range(0, num_views, max_chunk):
        chunk_end = min(chunk_start + max_chunk, num_views)
        print(f">> Inference on frames [{chunk_start}:{chunk_end}] of {num_views}")
        chunk_views = views[chunk_start:chunk_end]
        with torch.no_grad():
            chunk_pred = model.inference(
                chunk_views, device, is_mog=is_mog, use_sky_mask=True, **CONFIGS
            )
        chunk_predictions.append(_preds_to_cpu(chunk_pred))
        del chunk_pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = _merge_chunk_predictions(chunk_predictions)

    # Save raw per-view npz files for downstream visualization (matches the
    # launch_final_score.py schema: depth_pred / depth_gt / valid_mask / rgb).
    # This is a folder-inference script — depth_gt / valid_mask are placeholders
    # (zeros / ones) since no ground truth is available.
    raw_save_path = os.path.join(output_dir, "raw")
    os.makedirs(raw_save_path, exist_ok=True)
    scene_id = os.path.basename(os.path.normpath(args.img_path)) or "sequence"

    depth_pred_all_np = (
        predictions["depth"].squeeze(0)[:num_views].cpu().numpy()
    )  # (L, H, W)
    imgs_hwc_np = (
        predictions["images"].squeeze(0)[:num_views]
        .permute(0, 2, 3, 1).cpu().numpy()
    )  # (L, H, W, 3)

    _mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _std_rgb = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for j in range(num_views):
        depth_pred_j = depth_pred_all_np[j]
        depth_gt_j = np.zeros_like(depth_pred_j)
        valid_mask_j = np.ones_like(depth_pred_j, dtype=bool)

        img_j = imgs_hwc_np[j]
        if is_ppd:
            rgb_image_j = (np.clip(img_j, 0, 1) * 255).astype(np.uint8)
        else:
            rgb_image_j = (
                np.clip(img_j * _std_rgb + _mean_rgb, 0, 1) * 255
            ).astype(np.uint8)

        np.savez_compressed(
            os.path.join(raw_save_path, f"{scene_id.replace('/', '_')}_view_{j}.npz"),
            depth_pred=depth_pred_j,
            depth_gt=depth_gt_j,
            valid_mask=valid_mask_j,
            rgb=rgb_image_j,
        )
    print(f"Saved {num_views} raw per-view npz files to {raw_save_path}")

    # Save per-frame depth maps
    if args.output_double_layers:
        depth_maps = predictions["depth"].squeeze(0)  # (L, H, W)
        raw_layer_depths = predictions["raw_preds"]["depth"]

        if args.output_sharp_boundary:
            assert isinstance(raw_layer_depths, list)
            sharp_depth = predictions["depth"]
            mog_weight_raw = predictions["raw_preds"]["mog_weight_raw"]
            transparent_pixels = mog_weight_raw.sum(dim=-1) > 1.5
            opaque_pixels = mog_weight_raw.sum(dim=-1) <= 1.5
            sharp_depth = sharp_depth * opaque_pixels + transparent_pixels * raw_layer_depths[0]

            layer_depths = [sharp_depth.detach().squeeze(0)]
            for layer in [raw_layer_depths[-1]]:
                layer_tensor = layer.detach()
                if layer_tensor.ndim == 4 and layer_tensor.shape[0] == 1:
                    layer_tensor = layer_tensor.squeeze(0)
                layer_depths.append(layer_tensor[:num_views])

        elif isinstance(raw_layer_depths, list):
            layer_depths = []
            for layer in [raw_layer_depths[0], raw_layer_depths[-1]]:
                layer_tensor = layer.detach()
                if layer_tensor.ndim == 4 and layer_tensor.shape[0] == 1:
                    layer_tensor = layer_tensor.squeeze(0)
                elif layer_tensor.ndim != 3:
                    continue
                layer_depths.append(layer_tensor[:num_views])
        else:
            raw_squeezed = (
                raw_layer_depths.squeeze(0)
                if raw_layer_depths.ndim == 4
                else raw_layer_depths
            )
            layer_depths = [raw_squeezed] * 2

        for j, file_path in enumerate(filelist):
            img_name = os.path.basename(file_path)
            name_no_ext = os.path.splitext(img_name)[0]
            out_subdir = os.path.join(output_dir, name_no_ext)
            os.makedirs(out_subdir, exist_ok=True)

            fused_depth = depth_maps[j : j + 1]
            save_depth_maps(None, out_subdir, conf_self=None, depth_maps=fused_depth.cpu())

            for layer_idx, layer_tensor in enumerate(layer_depths):
                layer_out = os.path.join(out_subdir, f"layer_{layer_idx:02d}")
                os.makedirs(layer_out, exist_ok=True)
                save_depth_maps(
                    None, layer_out, conf_self=None,
                    depth_maps=layer_tensor[j : j + 1].cpu(),
                )
    else:
        depth_maps = predictions["depth"].squeeze(0)  # (L, H, W)
        save_depth_maps(None, output_dir, conf_self=None, depth_maps=depth_maps[:num_views].cpu())

    # MoG visualization: run per-chunk because views/raw_preds must stay
    # frame-consistent and we don't merge `views` above.
    # Sideview point cloud rendering
    if args.render_sideview:
        try:
            with torch.no_grad():
                sv_extrinsic, sv_intrinsic = pose_encoding_to_extri_intri(
                    predictions["pose_enc"], predictions["images"].shape[-2:]
                )
                sv_depth = predictions["depth"].cpu().numpy().squeeze(0)[:num_views]
                sv_ext_w2c_np = sv_extrinsic.cpu().numpy().squeeze(0)[:num_views]
                sv_K_np = sv_intrinsic.cpu().numpy().squeeze(0)[:num_views]

                sv_pts3d = unproject_depth_map_to_point_map(
                    sv_depth, sv_ext_w2c_np, sv_K_np,
                )

                sv_imgs = (
                    predictions["images"]
                    .squeeze(0)[:num_views]
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                if is_ppd:
                    sv_colors = np.clip(sv_imgs, 0, 1)
                else:
                    _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    _std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    sv_colors = np.clip(sv_imgs * _std + _mean, 0, 1)

                # Build validity mask: depth range + sky filtering
                sv_masks = (
                    (sv_depth > args.pcd_depth_min) & (sv_depth < args.pcd_depth_max)
                ).astype(np.float32)

                # Extract sky mask from model predictions
                raw_preds = predictions["raw_preds"]
                n_real = predictions["depth"].shape[1]
                if "sky_mask" in predictions:
                    sky_mask = predictions["sky_mask"][0, :n_real].cpu().numpy() > 0.5
                elif "mog_weight_full" in raw_preds:
                    mwf = raw_preds["mog_weight_full"][0, :n_real]
                    sky_mask = (mwf.argmax(dim=-1) == mwf.shape[-1] - 1).cpu().numpy()
                elif "sky_mask" in raw_preds:
                    sky_mask = raw_preds["sky_mask"][0, :n_real].cpu().numpy() > 0.5
                else:
                    sky_mask = None

                if sky_mask is not None:
                    sv_masks[sky_mask[:num_views]] = 0.0
                    print(f"Sky mask applied: {sky_mask[:num_views].sum()} sky pixels removed")

                # Save combined 3D point cloud (all frames merged)
                import open3d as o3d
                all_pts = []
                all_colors = []
                for fid in range(num_views):
                    mask = sv_masks[fid] > 0.5  # (H, W)
                    pts = sv_pts3d[fid][mask]  # (N, 3)
                    cols = sv_colors[fid][mask]  # (N, 3)
                    all_pts.append(pts)
                    all_colors.append(cols)
                all_pts = np.concatenate(all_pts, axis=0)
                all_colors = np.concatenate(all_colors, axis=0)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(all_pts)
                pcd.colors = o3d.utility.Vector3dVector(all_colors)
                pcd_path = os.path.join(output_dir, "pointcloud.ply")
                o3d.io.write_point_cloud(pcd_path, pcd)
                print(f"Point cloud saved: {len(all_pts)} points -> {pcd_path}")

                # Persist per-frame extrinsics (world-to-camera, OpenCV) and
                # intrinsics alongside the point cloud so downstream renderers
                # can re-project back into camera space.
                n_sv = len(sv_depth)
                sv_ext_w2c_4x4 = np.zeros((n_sv, 4, 4), dtype=np.float64)
                sv_ext_w2c_4x4[:, :3, :] = sv_ext_w2c_np.astype(np.float64)
                sv_ext_w2c_4x4[:, 3, 3] = 1.0
                cameras_path = os.path.join(output_dir, "cameras.npz")
                np.savez_compressed(
                    cameras_path,
                    extrinsics_w2c=sv_ext_w2c_4x4,
                    intrinsics=sv_K_np.astype(np.float64),
                )
                print(f"Cameras saved: {n_sv} views -> {cameras_path}")

        except Exception as e:
            print(f"Warning: sideview rendering failed: {e}")

    print(f"Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
