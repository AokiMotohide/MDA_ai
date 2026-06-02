import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import time
import torch
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(420000)

def get_depth_viz(depth, colormap=cv2.COLORMAP_INFERNO, depth_min=None, depth_max=None):
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    if depth_min is None:
        depth_min, depth_max = depth.min(), depth.max()
        depth_diff = depth_max - depth_min
        depth_mid = depth_min + depth_diff / 2.0
        depth_min = depth_mid - depth_diff * 1.2 / 2.0
        depth_max = depth_mid + depth_diff * 1.2 / 2.0
    if depth_max - depth_min > 1e-6:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = depth - depth_min
    depth_viz = (depth_norm * 255).astype(np.uint8)
    depth_viz = cv2.applyColorMap(depth_viz, colormap)
    return depth_viz, depth_min, depth_max

def get_args_parser():
    parser = argparse.ArgumentParser("3D Reconstruction evaluation — score only", add_help=False)
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--margin", action="store_true")
    parser.add_argument("--crop_center_112", type=int, default=0)
    parser.add_argument("--cam_inp", type=int, default=0)
    parser.add_argument("--gt_cam_output", type=int, default=0)
    parser.add_argument("--output_normalize", type=int, default=0)
    return parser


def main(args):
    global device
    global PATCH_SIZE

    from src.testing.eval_cut3r.mv_recon.data import SevenScenes, NRGBD, ETH3D, HiRoom
    from src.testing.eval_cut3r.mv_recon.utils import accuracy, completion, accuracy_raw, completion_raw

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name

    loaded = choose_model(model_name)
    model = loaded.model
    checkpoint_path = loaded.checkpoint_path
    patch_size = loaded.patch_size
    ImgNorm_used = loaded.img_norm
    mysize = loaded.model_size
    args.size = mysize

    if args.size == 1024 and patch_size==16:
        resolution = (1024, 768)
    elif args.size == 512 and patch_size==16:
        resolution = (512, 384)
    elif args.size == 504 and patch_size==14:
        resolution = (504, 378)
    elif args.size == 518 and patch_size==14:
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
        ),  # 20),
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

    values = [str(int(v)) for k, v in CONFIGS.items()]
    values = ''.join(values)
    save_dir = osp.join(args.output_dir, values)
    args.output_dir = save_dir

    os.makedirs(args.output_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            if args.margin:
                save_path = save_path + "_marginVideo"
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_0.txt")
            if os.path.exists(log_file):
                os.remove(log_file)

            with open(osp.join(save_path, f"config.json"), "w") as f:
                json.dump(CONFIGS, f)

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            fps_all = []
            time_all = []

            idxs = list(range(len(dataset)))
            if True:
                for data_idx in tqdm(idxs):
                    batch = default_collate([dataset[data_idx]])
                    ignore_keys = set(
                        [
                            "dataset",
                            "label",
                            "instance",
                            "idx",
                            "true_shape",
                            "rng",
                        ]
                    )
                    for view in batch:
                        for name in view.keys():
                            if name in ignore_keys:
                                continue
                            if isinstance(view[name], tuple) or isinstance(
                                view[name], list
                            ):
                                view[name] = [
                                    x.to(device, non_blocking=True) for x in view[name]
                                ]
                            else:
                                view[name] = view[name].to(device, non_blocking=True)

                    if True:
                        revisit = args.revisit
                        update = not args.freeze
                        if revisit > 1:
                            new_views = []
                            for r in range(revisit):
                                for i in range(len(batch)):
                                    new_view = deepcopy(batch[i])
                                    new_view["idx"] = [
                                        (r * len(batch) + i)
                                        for _ in range(len(batch[i]["idx"]))
                                    ]
                                    new_view["instance"] = [
                                        str(r * len(batch) + i)
                                        for _ in range(len(batch[i]["instance"]))
                                    ]
                                    if r > 0:
                                        if not update:
                                            new_view["update"] = torch.zeros_like(
                                                batch[i]["update"]
                                            ).bool()
                                    new_views.append(new_view)
                            batch = new_views

                        batch_cpu = [
                            {
                                k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in sample.items()
                            } for sample in batch
                        ]
                        with torch.cuda.amp.autocast(enabled=False):
                            start = time.time()
                            predictions = model.inference(batch, device, **CONFIGS)
                            end = time.time()

                            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], predictions["images"].shape[-2:])
                            extrinsic = affine_inverse(extrinsic)
                            world_points_from_depth = unproject_depth_map_to_point_map(
                                predictions["depth"].cpu().numpy().squeeze(0),
                                extrinsic.cpu().numpy().squeeze(0),
                                intrinsic.cpu().numpy().squeeze(0)
                            )
                            world_points_from_depth = torch.from_numpy(world_points_from_depth).unsqueeze(0).to(device=device)

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

                        # Evaluation
                        print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
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
                        pts_all_raw = []
                        pts_gt_all = []
                        images_all = []
                        masks_all = []
                        masks_gt_all = []
                        conf_all = []
                        intrinsics_all = []
                        cam_poses_all = []
                        depth_pred_all = []
                        depth_gt_all = []
                        is_ppd = 'ppd' in model_name or 'ppdv' in model_name or 'ppvd' in model_name

                        for j, view in enumerate(batch):
                            if in_camera1 is None:
                                in_camera1 = view["camera_pose"][0].cpu()

                            image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                            mask = view["valid_mask"].cpu().numpy()[0]

                            pts = pred_pts[j].cpu().numpy()[0]
                            conf = preds[j]["conf"].cpu().data.numpy()[0]

                            pts_gt = gt_pts[j].detach().cpu().numpy()[0]
                            pts_raw = world_points_from_depth[0, j].cpu().numpy()

                            if CONFIGS["crop_center_112"]:
                                resolution_0 = resolution[0] if isinstance(resolution, tuple) else resolution
                                crop_size = int(112 / 512 * resolution_0)
                                H, W = image.shape[:2]
                                cx = W // 2
                                cy = H // 2
                                l, t = cx - crop_size, cy - crop_size
                                r, b = cx + crop_size, cy + crop_size
                                image = image[t:b, l:r]
                                mask = mask[t:b, l:r]
                                pts = pts[t:b, l:r]
                                pts_gt = pts_gt[t:b, l:r]
                                pts_raw = pts_raw[t:b, l:r]

                            #### Align predicted 3D points to the ground truth
                            pts[..., -1] += gt_shift_z.cpu().numpy().item()
                            pts = geotrf(in_camera1, pts)

                            pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                            pts_gt = geotrf(in_camera1, pts_gt)

                            # Visualization
                            depth_pred = predictions["depth"][0, j].cpu().numpy()
                            depth_gt = view["depthmap"][0].cpu().numpy()
                            if args.margin:
                                depth_gt_clamp = np.clip(depth_gt, 0.1, 65)
                                min_val = depth_gt_clamp.min()
                                max_val = depth_gt_clamp.max()
                                norm_depth = (depth_gt - min_val) / (max_val - min_val + 1e-5)
                                norm_depth = torch.clamp(torch.from_numpy(norm_depth), 0.0, 1.0).numpy()
                                depth_uint8 = (norm_depth * 255).astype(np.uint8)
                                edge = cv2.Canny(depth_uint8, 100, 200)
                                edge_v2 = edge

                                mask_margin = (edge > 0.5)
                                mask_margin_gt = (edge_v2 > 0.5)
                                
                                kernel = np.ones((2, 2), np.uint8)
                                mask_dialate = (cv2.dilate(1 - mask.astype(np.uint8), kernel, iterations=1) < 0.5)

                                if CONFIGS["crop_center_112"]:
                                    mask_gt = mask_dialate & mask_margin_gt[t:b, l:r]
                                    mask = mask_dialate & mask_margin[t:b, l:r]
                                else:
                                    mask_gt = mask_dialate & mask_margin_gt
                                    mask = mask_dialate & mask_margin
                            else:
                                mask_gt = mask

                            images_all.append((image[None, ...] + 1.0) / 2.0)
                            pts_all.append(pts[None, ...])
                            pts_all_raw.append(pts_raw[None, ...])
                            pts_gt_all.append(pts_gt[None, ...])
                            masks_all.append(mask[None, ...])
                            conf_all.append(conf[None, ...])
                            masks_gt_all.append(mask_gt[None, ...])
                            # Crop depths to match image shape
                            dp = depth_pred
                            dg = depth_gt
                            if CONFIGS["crop_center_112"]:
                                dp = dp[t:b, l:r]
                                dg = dg[t:b, l:r]
                            depth_pred_all.append(dp[None, ...])
                            depth_gt_all.append(dg[None, ...])
                            intrinsics_all.append(view["camera_intrinsics"][0].cpu().numpy())
                            cam_poses_all.append(view["camera_pose"][0].cpu().numpy())

                            pred_depth_min, pred_depth_max = depth_pred.min(), depth_pred.max()
                            gt_depth_min, gt_depth_max = depth_gt.min(), depth_gt.max()
                            depth_min = min(pred_depth_min, gt_depth_min)
                            depth_max = max(pred_depth_max, gt_depth_max)
                            pred_viz, _, _ = get_depth_viz(depth_pred, depth_min=depth_min, depth_max=depth_max)
                            gt_viz, _, _ = get_depth_viz(depth_gt, depth_min=depth_min, depth_max=depth_max)
                            error = np.abs(depth_pred - depth_gt)
                            error_viz, _, _ = get_depth_viz(error, depth_min=0, depth_max=1)

                            # Concatenate images horizontally
                            combined_viz = np.hstack((gt_viz, pred_viz, error_viz))
                            vis_save_path = osp.join(save_path, "vis")
                            os.makedirs(vis_save_path, exist_ok=True)
                            scene_id = view["label"][0].rsplit("/", 1)[0]
                            cv2.imwrite(os.path.join(vis_save_path, f"{scene_id.replace('/', '_')}_view_{j}.png"), combined_viz)

                            # Save raw per-view data for downstream comparison
                            raw_save_path = osp.join(save_path, "raw")
                            os.makedirs(raw_save_path, exist_ok=True)
                            if is_ppd:
                                rgb_image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
                            else:
                                _mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                                _std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                                rgb_image = (np.clip(image * _std + _mean, 0, 1) * 255).astype(np.uint8)
                            valid_mask_np = view["valid_mask"][0].cpu().numpy()

                            # MoG-specific: slice the per-view sky component so
                            # downstream tools (e.g. the sky-filter renderer)
                            # can read raw sky data without needing the full
                            # per-scene .npy.
                            raw_save_kwargs = {}
                            raw_preds_dict_j = predictions.get("raw_preds")
                            if (isinstance(raw_preds_dict_j, dict)
                                    and "mog_weight_full" in raw_preds_dict_j):
                                mwf_j = raw_preds_dict_j["mog_weight_full"]
                                if hasattr(mwf_j, "detach"):
                                    mwf_j = mwf_j.detach().cpu().numpy()
                                if mwf_j.ndim >= 1 and mwf_j.shape[0] == 1:
                                    mwf_j = mwf_j[0]
                                # mwf_j now has shape (N, H, W, K); take view j.
                                if mwf_j.ndim >= 1 and mwf_j.shape[0] > j:
                                    raw_save_kwargs["mog_weight_full"] = mwf_j[j]

                            np.savez_compressed(
                                os.path.join(raw_save_path, f"{scene_id.replace('/', '_')}_view_{j}.npz"),
                                depth_pred=depth_pred,
                                depth_gt=depth_gt,
                                valid_mask=valid_mask_np,
                                rgb=rgb_image,
                                **raw_save_kwargs,
                            )

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_all_raw = np.concatenate(pts_all_raw, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)
                    masks_gt_all = np.concatenate(masks_gt_all, axis=0)

                    scene_id = view["label"][0].rsplit("/", 1)[0]

                    # Save scene data (used by launch_final_vis.py)
                    save_params = {}
                    save_params["images_all"] = images_all
                    save_params["pts_all"] = pts_all
                    save_params["pts_gt_all"] = pts_gt_all
                    save_params["masks_all"] = masks_all
                    save_params["masks_gt_all"] = masks_gt_all
                    save_params["depth_pred_all"] = np.concatenate(depth_pred_all, axis=0)
                    save_params["depth_gt_all"] = np.concatenate(depth_gt_all, axis=0)
                    save_params["intrinsics_all"] = np.stack(intrinsics_all, axis=0)
                    save_params["cam_poses_all"] = np.stack(cam_poses_all, axis=0)
                    save_params["is_ppd"] = is_ppd

                    # MoG-specific: save raw component weights if the model emits them.
                    raw_preds_dict = predictions.get("raw_preds")
                    if (isinstance(raw_preds_dict, dict)
                            and "mog_weight_full" in raw_preds_dict):
                        mwf = raw_preds_dict["mog_weight_full"]
                        if hasattr(mwf, "detach"):
                            mwf = mwf.detach().cpu().numpy()
                        # Drop leading batch dim (1, N, ...) -> (N, ...) to match
                        # the other per-scene arrays.
                        if mwf.ndim >= 1 and mwf.shape[0] == 1:
                            mwf = mwf[0]
                        save_params["mog_weight_full"] = mwf

                    np.save(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                        save_params,
                    )

                    if "DTU" in name_data:
                        threshold = 100
                    else:
                        threshold = 0.1

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_gt_all > 0]
                    images_all_masked = images_all[masks_all > 0]
                    images_gt_all_masked = images_all[masks_gt_all > 0]

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(
                        pts_all_masked.reshape(-1, 3)
                    )
                    pcd.colors = o3d.utility.Vector3dVector(
                        images_all_masked.reshape(-1, 3)
                    )

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(
                        pts_gt_all_masked.reshape(-1, 3)
                    )
                    pcd_gt.colors = o3d.utility.Vector3dVector(
                        images_gt_all_masked.reshape(-1, 3)
                    )

                    trans_init = np.eye(4)

                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        pcd,
                        pcd_gt,
                        threshold,
                        trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    )

                    transformation = reg_p2p.transformation

                    o3d.io.write_point_cloud(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                        pcd_gt,
                    )
                    o3d.io.write_point_cloud(
                        os.path.join(
                            save_path, f"{scene_id.replace('/', '_')}-mask.ply"
                        ),
                        pcd,
                    )

                    pcd = pcd.transform(transformation)
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

                    # Save per-point error data for visualization
                    acc_dists, acc_idx = accuracy_raw(np.asarray(pcd_gt.points), np.asarray(pcd.points))
                    comp_dists, comp_idx = completion_raw(np.asarray(pcd_gt.points), np.asarray(pcd.points))
                    np.savez_compressed(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}-error.npz"),
                        acc_distances=acc_dists,
                        acc_idx=acc_idx,
                        comp_distances=comp_dists,
                        comp_idx=comp_idx,
                        masks_all=masks_all,
                        masks_gt_all=masks_gt_all,
                        depth_pred_all=save_params["depth_pred_all"].astype(np.float32),
                        depth_gt_all=save_params["depth_gt_all"].astype(np.float32),
                    )

                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                        file=open(log_file, "a"),
                    )

                    acc_all += acc
                    comp_all += comp
                    nc1_all += nc1
                    nc2_all += nc2

                    acc_all_med += acc_med
                    comp_all_med += comp_med
                    nc1_all_med += nc1_med
                    nc2_all_med += nc2_med

                    # release cuda memory
                    torch.cuda.empty_cache()

            if True:
                to_write = ""
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            for key, value in data.items():
                                if key != "scene_id":
                                    metrics[key].append(float(value))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    c_name = "mean"
                    print_str = f"{c_name.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_num = np.mean(mean_metrics[m_name])
                        print_str = print_str + f"{m_name}: {print_num:.3f} | "
                    print_str = print_str + "\n"
                    f.write(to_write + print_str)


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    CONFIGS["crop_center_112"] = bool(args.crop_center_112)
    CONFIGS["cam_inp"] = bool(args.cam_inp)
    CONFIGS["gt_cam_output"] = bool(args.gt_cam_output)
    CONFIGS["output_normalize"] = bool(args.output_normalize)

    main(args)
