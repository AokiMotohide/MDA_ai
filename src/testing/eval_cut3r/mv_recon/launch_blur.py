"""
Boundary-blur ablation study.

Simulate degraded inputs by downsampling each frame by factor s via area
averaging, then upsampling back to model resolution via bicubic interpolation.
Reports mean edge-aware Chamfer Distance on NRGBD, 7Scenes, and HiRoom
for s in {1, 2, 4, 8}.
"""

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import time
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
import cv2
import json
from torch.utils.data._utils.collate import default_collate
from depth_anything_3.utils.geometry import affine_inverse
from tqdm import tqdm
from collections import defaultdict
import random
from src.testing.eval_cut3r.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri, unproject_depth_map_to_point_map
from src.testing.utils.model_choice import choose_model, CONFIGS

from dust3r.utils.geometry import geotrf
from copy import deepcopy

PATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
SKIP_SCENE_IDS = {"828785_cam_sampled_12"}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(420000)


def apply_blur_degradation(img, blur_factor):
    """Downsample by blur_factor (area avg), then upsample back (bicubic).

    Args:
        img: (B, C, H, W) tensor
        blur_factor: integer downsampling factor s
    Returns:
        degraded image of the same shape
    """
    if blur_factor <= 1:
        return img
    _, _, H, W = img.shape
    small = F.interpolate(img, size=(H // blur_factor, W // blur_factor), mode='area')
    restored = F.interpolate(small, size=(H, W), mode='bicubic', align_corners=False)
    return restored


def get_depth_viz(depth, colormap=cv2.COLORMAP_INFERNO, depth_min=None, depth_max=None):
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    if depth_min is None:
        depth_min, depth_max = depth.min(), depth.max()
        depth_diff = depth_max - depth_min
        depth_mid = depth_min + depth_diff / 2.0
        depth_min = depth_mid - depth_diff * 1.5 / 2.0
        depth_max = depth_mid + depth_diff * 1.5 / 2.0
    if depth_max - depth_min > 1e-6:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = depth - depth_min
    depth_viz = (depth_norm * 255).astype(np.uint8)
    depth_viz = cv2.applyColorMap(depth_viz, colormap)
    return depth_viz, depth_min, depth_max


def maybe_visualize_depth_triplet(
    args,
    save_path,
    scene_id,
    data_idx,
    view_idx,
    depth_pred,
    depth_gt,
    original_image,
    is_ppd_model,
):
    pred_depth_min, pred_depth_max = depth_pred.min(), depth_pred.max()
    gt_depth_min, gt_depth_max = depth_gt.min(), depth_gt.max()
    depth_min = min(pred_depth_min, gt_depth_min)
    depth_max = max(pred_depth_max, gt_depth_max)
    pred_viz, _, _ = get_depth_viz(depth_pred, depth_min=depth_min, depth_max=depth_max)
    gt_viz, _, _ = get_depth_viz(depth_gt, depth_min=depth_min, depth_max=depth_max)
    error = np.abs(depth_pred - depth_gt)
    error_viz, _, _ = get_depth_viz(error, depth_min=0, depth_max=1)
    if is_ppd_model:
        orig_viz = np.clip(original_image, 0.0, 1.0)
    else:
        imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        orig_viz = np.clip(original_image * imagenet_std + imagenet_mean, 0.0, 1.0)
    orig_viz = (orig_viz * 255).astype(np.uint8)
    orig_viz = cv2.cvtColor(orig_viz, cv2.COLOR_RGB2BGR)
    combined_viz = np.hstack((orig_viz, gt_viz, pred_viz, error_viz))

    if args.save_vis:
        vis_save_path = osp.join(save_path, "vis")
        os.makedirs(vis_save_path, exist_ok=True)
        cv2.imwrite(
            osp.join(vis_save_path, f"{scene_id.replace('/', '_')}_sample{data_idx}_view{view_idx}.png"),
            combined_viz,
        )


def get_args_parser():
    parser = argparse.ArgumentParser("Boundary-blur ablation — edge-aware Chamfer Distance", add_help=False)
    parser.add_argument("--weights", type=str, default="", help="ckpt name")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--conf_thresh", type=float, default=0.0, help="confidence threshold")
    parser.add_argument("--output_dir", type=str, default="", help="value for outdir")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--crop_center_112", type=int, default=0)
    parser.add_argument("--cam_inp", type=int, default=0)
    parser.add_argument("--gt_cam_output", type=int, default=0)
    parser.add_argument("--output_normalize", type=int, default=0)
    parser.add_argument(
        "--blur_factors",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="list of blur downsampling factors s (default: 1 2 4 8)",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="save per-view depth visualizations (gt/pred/error)",
    )
    return parser


def run_one_blur_factor(args, blur_factor, model, patch_size, ImgNorm_used, mysize, criterion):
    """Run evaluation for a single blur factor and return per-dataset metrics."""
    global device

    from src.testing.eval_cut3r.mv_recon.data import SevenScenes, NRGBD, HiRoom
    from src.testing.eval_cut3r.mv_recon.utils import accuracy, completion

    args.size = mysize
    model_name_lower = args.model_name.lower()
    is_ppd_model = (
        "ppd" in model_name_lower
        or "ppdv" in model_name_lower
        or "ppvd" in model_name_lower
    )

    if args.size == 1024 and patch_size == 16:
        resolution = (1024, 768)
    elif args.size == 512 and patch_size == 16:
        resolution = (512, 384)
    elif args.size == 504 and patch_size == 14:
        resolution = (504, 378)
    elif args.size == 518 and patch_size == 14:
        resolution = (518, 378)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError

    datasets_all = {
        "7scenes": SevenScenes(
            split="test",
            ROOT="./data/cut3r_data/7scenes",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=200,
            transform=ImgNorm_used,
        ),
        "NRGBD_100": NRGBD(
            split="test",
            ROOT="./data/cut3r_data/neural_rgbd",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=100,
            transform=ImgNorm_used,
        ),
        "HiRoom": HiRoom(
            split="test",
            ROOT="./data/cut3r_data/DA3-BENCH/hiroom",
            resolution=(max(resolution), max(resolution)),
            transform=ImgNorm_used,
        ),
    }

    blur_tag = f"blur_s{blur_factor}"
    save_dir = osp.join(args.output_dir, blur_tag)
    os.makedirs(save_dir, exist_ok=True)

    dataset_metrics = {}  # dataset_name -> {acc, comp, chamfer, ...}

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(save_dir, name_data + "_marginVideo")
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, "logs_0.txt")
            if os.path.exists(log_file):
                os.remove(log_file)

            with open(osp.join(save_path, "config.json"), "w") as f:
                json.dump({**CONFIGS, "blur_factor": blur_factor}, f)

            acc_all = 0
            comp_all = 0
            nc1_all = 0
            nc2_all = 0
            acc_all_med = 0
            comp_all_med = 0
            nc1_all_med = 0
            nc2_all_med = 0
            n_scenes = 0
            fps_all = []
            time_all = []

            idxs = list(range(len(dataset)))

            for data_idx in tqdm(idxs, desc=f"{name_data} s={blur_factor}"):
                batch = default_collate([dataset[data_idx]])
                scene_id = batch[0]["label"][0].rsplit("/", 1)[0]
                if scene_id in SKIP_SCENE_IDS:
                    print(f"Skipping bad sequence: {scene_id}")
                    continue
                ignore_keys = set(["dataset", "label", "instance", "idx", "true_shape", "rng"])
                for view in batch:
                    for name in view.keys():
                        if name in ignore_keys:
                            continue
                        if isinstance(view[name], (tuple, list)):
                            view[name] = [x.to(device, non_blocking=True) for x in view[name]]
                        else:
                            view[name] = view[name].to(device, non_blocking=True)

                # --- Apply blur degradation to input images ---
                for view in batch:
                    view["img"] = apply_blur_degradation(view["img"], blur_factor)

                revisit = args.revisit
                update = not args.freeze
                if revisit > 1:
                    new_views = []
                    for r in range(revisit):
                        for i in range(len(batch)):
                            new_view = deepcopy(batch[i])
                            new_view["idx"] = [
                                (r * len(batch) + i) for _ in range(len(batch[i]["idx"]))
                            ]
                            new_view["instance"] = [
                                str(r * len(batch) + i) for _ in range(len(batch[i]["instance"]))
                            ]
                            if r > 0:
                                if not update:
                                    new_view["update"] = torch.zeros_like(batch[i]["update"]).bool()
                            new_views.append(new_view)
                    batch = new_views

                batch_cpu = [
                    {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
                    for sample in batch
                ]

                with torch.cuda.amp.autocast(enabled=False):
                    start = time.time()
                    predictions = model.inference(batch, device, **CONFIGS)
                    end = time.time()

                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        predictions["pose_enc"], predictions["images"].shape[-2:]
                    )
                    extrinsic = affine_inverse(extrinsic)
                    world_points_from_depth = unproject_depth_map_to_point_map(
                        predictions["depth"].cpu().numpy().squeeze(0),
                        extrinsic.cpu().numpy().squeeze(0),
                        intrinsic.cpu().numpy().squeeze(0),
                    )
                    world_points_from_depth = (
                        torch.from_numpy(world_points_from_depth).unsqueeze(0).to(device=device)
                    )

                    preds = world_points_from_depth
                    confs = predictions["depth_conf"]

                    all_preds = []
                    for idx in range(preds.shape[1]):
                        all_preds.append(
                            {'pts3d': preds[0][idx:idx+1].cpu(), 'conf': confs[0][idx:idx+1].cpu()}
                        )
                    preds = all_preds

                valid_length = len(preds) // revisit
                preds = preds[-valid_length:]
                batch = batch[-valid_length:]
                fps = len(batch) / (end - start)
                print(
                    f"Finished reconstruction for {name_data} {data_idx+1}/{len(dataset)}, FPS: {fps:.2f}"
                )
                fps_all.append(fps)
                time_all.append(end - start)

                # --- Criterion alignment ---
                gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                    criterion.get_all_pts3d_t(batch_cpu, preds)
                )
                pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
                    monitoring["pred_scale"],
                    monitoring["gt_scale"],
                    monitoring["pred_shift_z"],
                    monitoring["gt_shift_z"],
                )

                in_camera1 = None
                pts_all = []
                pts_gt_all = []
                masks_all = []
                masks_gt_all = []
                images_all = []
                # Extra accumulators so the scene .npy matches the schema
                # expected by launch_final_vis.py::render_scene.
                intrinsics_all = []
                cam_poses_all = []
                depth_pred_all = []
                depth_gt_all = []
                scene_id = batch[0]["label"][0].rsplit("/", 1)[0]

                for j, view in enumerate(batch):
                    if in_camera1 is None:
                        in_camera1 = view["camera_pose"][0].cpu()

                    image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                    mask = view["valid_mask"].cpu().numpy()[0]

                    pts = pred_pts[j].cpu().numpy()[0]
                    pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                    assert not CONFIGS["crop_center_112"]

                    # Align predicted 3D points to the ground truth
                    pts[..., -1] += gt_shift_z.cpu().numpy().item()
                    pts = geotrf(in_camera1, pts)
                    pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                    pts_gt = geotrf(in_camera1, pts_gt)

                    # Edge-aware margin mask (always applied)
                    depth_gt = view["depthmap"][0].cpu().numpy()
                    min_val = depth_gt.min()
                    max_val = depth_gt.max()
                    norm_depth = (depth_gt - min_val) / (max_val - min_val + 1e-5)
                    norm_depth = np.clip(norm_depth, 0.0, 1.0)
                    depth_uint8 = (norm_depth * 255).astype(np.uint8)
                    edge = cv2.Canny(depth_uint8, 100, 200)
                    # kernel = np.ones((3, 3), np.uint8)
                    # edge_v2 = cv2.dilate(edge, kernel, iterations=1)
                    edge_v2 = edge

                    mask_margin = (edge_v2 > 0.5)
                    mask_margin_gt = (edge_v2 > 0.5)
                    
                    if name_data == "HiRoom":
                        kernel = np.ones((2, 2), np.uint8)
                        mask_dialate = (cv2.dilate(1 - mask.astype(np.uint8), kernel, iterations=1) < 0.5)
                    else:
                        mask_dialate = mask

                    mask_gt = mask_dialate & mask_margin_gt
                    mask = mask_dialate & mask_margin

                    depth_pred = predictions["depth"][0, j].cpu().numpy()
                    depth_gt_vis = depth_gt

                    maybe_visualize_depth_triplet(
                        args=args,
                        save_path=save_path,
                        scene_id=scene_id,
                        data_idx=data_idx,
                        view_idx=j,
                        depth_pred=depth_pred,
                        depth_gt=depth_gt_vis,
                        original_image=image,
                        is_ppd_model=is_ppd_model,
                    )

                    images_all.append((image[None, ...] + 1.0) / 2.0)
                    pts_all.append(pts[None, ...])
                    pts_gt_all.append(pts_gt[None, ...])
                    masks_all.append(mask[None, ...])
                    masks_gt_all.append(mask_gt[None, ...])
                    depth_pred_all.append(depth_pred[None, ...])
                    depth_gt_all.append(depth_gt_vis[None, ...])
                    intrinsics_all.append(
                        view["camera_intrinsics"][0].cpu().numpy()
                    )
                    cam_poses_all.append(view["camera_pose"][0].cpu().numpy())

                    # Save raw per-view NPZ for downstream visualization
                    # (matches launch_final_score.py's raw/*.npz schema).
                    raw_save_path = osp.join(save_path, "raw")
                    os.makedirs(raw_save_path, exist_ok=True)
                    if is_ppd_model:
                        rgb_image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                    else:
                        _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        _std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        rgb_image = (
                            np.clip(image * _std + _mean, 0, 1) * 255
                        ).astype(np.uint8)
                    depth_pred_full = predictions["depth"][0, j].cpu().numpy()
                    depth_gt_full = view["depthmap"][0].cpu().numpy()
                    valid_mask_np = view["valid_mask"][0].cpu().numpy()
                    np.savez_compressed(
                        os.path.join(
                            raw_save_path,
                            f"{scene_id.replace('/', '_')}_view_{j}.npz",
                        ),
                        depth_pred=depth_pred_full,
                        depth_gt=depth_gt_full,
                        valid_mask=valid_mask_np,
                        rgb=rgb_image,
                    )

                if len(pts_all) == 0:
                    continue

                images_all = np.concatenate(images_all, axis=0)
                pts_all = np.concatenate(pts_all, axis=0)
                pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                masks_all = np.concatenate(masks_all, axis=0)
                masks_gt_all = np.concatenate(masks_gt_all, axis=0)

                # Save scene data consumed by launch_final_vis.py::render_scene.
                save_params = {
                    "images_all":    images_all,
                    "pts_all":       pts_all,
                    "pts_gt_all":    pts_gt_all,
                    "masks_all":     masks_all,
                    "masks_gt_all":  masks_gt_all,
                    "depth_pred_all": np.concatenate(depth_pred_all, axis=0),
                    "depth_gt_all":   np.concatenate(depth_gt_all, axis=0),
                    "intrinsics_all": np.stack(intrinsics_all, axis=0),
                    "cam_poses_all":  np.stack(cam_poses_all, axis=0),
                    "is_ppd":         is_ppd_model,
                }
                np.save(
                    os.path.join(
                        save_path, f"{scene_id.replace('/', '_')}.npy",
                    ),
                    save_params,
                )

                pts_all_masked = pts_all[masks_all > 0]
                pts_gt_all_masked = pts_gt_all[masks_gt_all > 0]
                images_all_masked = images_all[masks_all > 0]
                images_gt_all_masked = images_all[masks_gt_all > 0]

                if len(pts_all_masked) == 0 or len(pts_gt_all_masked) == 0:
                    continue

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_all_masked.reshape(-1, 3))
                pcd.colors = o3d.utility.Vector3dVector(images_all_masked.reshape(-1, 3))

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked.reshape(-1, 3))
                pcd_gt.colors = o3d.utility.Vector3dVector(images_gt_all_masked.reshape(-1, 3))

                threshold = 0.1
                trans_init = np.eye(4)
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd,
                    pcd_gt,
                    threshold,
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )
                pcd = pcd.transform(reg_p2p.transformation)
                pcd.estimate_normals()
                pcd_gt.estimate_normals()

                gt_normal = np.asarray(pcd_gt.normals)
                pred_normal = np.asarray(pcd.normals)
                acc, acc_med, nc1, nc1_med = accuracy(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                comp, comp_med, nc2, nc2_med = completion(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                chamfer = (acc + comp) / 2.0
                chamfer_med = (acc_med + comp_med) / 2.0

                print(
                    f"Idx: {scene_id}, Acc: {acc:.4f}, Comp: {comp:.4f}, "
                    f"Chamfer: {chamfer:.4f}, NC1: {nc1:.4f}, NC2: {nc2:.4f}"
                )
                with open(log_file, "a") as f_log:
                    print(
                        f"Idx: {scene_id}, Acc: {acc:.4f}, Comp: {comp:.4f}, "
                        f"Chamfer: {chamfer:.4f}, NC1: {nc1:.4f}, NC2: {nc2:.4f} - "
                        f"Acc_med: {acc_med:.4f}, Comp_med: {comp_med:.4f}, "
                        f"Chamfer_med: {chamfer_med:.4f}, NC1_med: {nc1_med:.4f}, NC2_med: {nc2_med:.4f}",
                        file=f_log,
                    )

                acc_all += acc
                comp_all += comp
                nc1_all += nc1
                nc2_all += nc2
                acc_all_med += acc_med
                comp_all_med += comp_med
                nc1_all_med += nc1_med
                nc2_all_med += nc2_med
                n_scenes += 1

                torch.cuda.empty_cache()

            # Per-dataset summary
            if n_scenes > 0:
                mean_acc = acc_all / n_scenes
                mean_comp = comp_all / n_scenes
                mean_chamfer = (mean_acc + mean_comp) / 2.0
                mean_nc1 = nc1_all / n_scenes
                mean_nc2 = nc2_all / n_scenes
                mean_acc_med = acc_all_med / n_scenes
                mean_comp_med = comp_all_med / n_scenes
                mean_chamfer_med = (mean_acc_med + mean_comp_med) / 2.0
                mean_fps = np.mean(fps_all) if fps_all else 0.0
                mean_time = np.mean(time_all) if time_all else 0.0

                summary = (
                    f"=== {name_data} | blur s={blur_factor} | {n_scenes} scenes ===\n"
                    f"  Acc:     {mean_acc:.4f}  (med {mean_acc_med:.4f})\n"
                    f"  Comp:    {mean_comp:.4f}  (med {mean_comp_med:.4f})\n"
                    f"  Chamfer: {mean_chamfer:.4f}  (med {mean_chamfer_med:.4f})\n"
                    f"  NC1:     {mean_nc1:.4f}  NC2: {mean_nc2:.4f}\n"
                    f"  FPS:     {mean_fps:.2f}  (mean time per scene: {mean_time:.3f}s)\n"
                )
                print(summary)
                with open(log_file, "a") as f:
                    f.write(summary)

                dataset_metrics[name_data] = {
                    "acc": mean_acc,
                    "comp": mean_comp,
                    "chamfer": mean_chamfer,
                    "nc1": mean_nc1,
                    "nc2": mean_nc2,
                    "acc_med": mean_acc_med,
                    "comp_med": mean_comp_med,
                    "chamfer_med": mean_chamfer_med,
                    "n_scenes": n_scenes,
                    "fps_mean": mean_fps,
                    "fps_all": fps_all,
                    "time_mean": mean_time,
                }

    return dataset_metrics


def main(args):
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = choose_model(args.model_name)
    model = loaded.model
    checkpoint_path = loaded.checkpoint_path
    patch_size = loaded.patch_size
    ImgNorm_used = loaded.img_norm
    mysize = loaded.model_size
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    values = [str(int(v)) for k, v in CONFIGS.items()]
    values = ''.join(values)
    save_dir = osp.join(args.output_dir, values)
    args.output_dir = save_dir
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}  # blur_factor -> dataset_metrics

    for s in args.blur_factors:
        print(f"\n{'='*60}")
        print(f"  Running blur factor s = {s}")
        print(f"{'='*60}\n")
        all_results[s] = run_one_blur_factor(
            args, s, model, patch_size, ImgNorm_used, mysize, criterion
        )

    # --- Final summary table ---
    summary_path = osp.join(args.output_dir, "blur_ablation_summary.txt")
    with open(summary_path, "w") as f:
        header = f"{'Dataset':<15}"
        for s in args.blur_factors:
            header += f"  s={s:<8}"
        f.write("Edge-aware Chamfer Distance (mean)\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        dataset_names = list(next(iter(all_results.values())).keys()) if all_results else []
        for dname in dataset_names:
            row = f"{dname:<15}"
            for s in args.blur_factors:
                val = all_results[s].get(dname, {}).get("chamfer", float('nan'))
                row += f"  {val:<8.4f}"
            f.write(row + "\n")

        # Overall mean across datasets
        row = f"{'Mean':<15}"
        for s in args.blur_factors:
            vals = [all_results[s][d]["chamfer"] for d in dataset_names if d in all_results[s]]
            mean_val = np.mean(vals) if vals else float('nan')
            row += f"  {mean_val:<8.4f}"
        f.write("-" * len(header) + "\n")
        f.write(row + "\n")

    with open(summary_path, "r") as f:
        print("\n" + f.read())

    # --- FPS summary table ---
    fps_path = osp.join(args.output_dir, "blur_ablation_fps.txt")
    with open(fps_path, "w") as f:
        header = f"{'Dataset':<15}"
        for s in args.blur_factors:
            header += f"  s={s:<8}"
        f.write("FPS (mean)\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        dataset_names = list(next(iter(all_results.values())).keys()) if all_results else []
        for dname in dataset_names:
            row = f"{dname:<15}"
            for s in args.blur_factors:
                val = all_results[s].get(dname, {}).get("fps_mean", float('nan'))
                row += f"  {val:<8.2f}"
            f.write(row + "\n")

        row = f"{'Mean':<15}"
        for s in args.blur_factors:
            vals = [all_results[s][d]["fps_mean"] for d in dataset_names if d in all_results[s]]
            mean_val = np.mean(vals) if vals else float('nan')
            row += f"  {mean_val:<8.2f}"
        f.write("-" * len(header) + "\n")
        f.write(row + "\n")

    with open(fps_path, "r") as f:
        print("\n" + f.read())

    # Also save as JSON for easy downstream parsing
    json_path = osp.join(args.output_dir, "blur_ablation_summary.json")
    json_results = {}
    for s in args.blur_factors:
        # Convert fps_all list for JSON serialization
        result_copy = {}
        for dname, metrics in all_results[s].items():
            m = dict(metrics)
            m["fps_all"] = [float(x) for x in m["fps_all"]]
            result_copy[dname] = m
        json_results[f"s={s}"] = result_copy
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Summary saved to {summary_path}")
    print(f"FPS saved to {fps_path}")
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    CONFIGS["crop_center_112"] = bool(args.crop_center_112)
    CONFIGS["cam_inp"] = bool(args.cam_inp)
    CONFIGS["gt_cam_output"] = bool(args.gt_cam_output)
    CONFIGS["output_normalize"] = bool(args.output_normalize)

    main(args)
