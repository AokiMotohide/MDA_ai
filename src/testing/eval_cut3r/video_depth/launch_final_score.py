import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))
import math
import cv2
import numpy as np
import torch
import argparse
import os.path as osp

from copy import deepcopy
from src.testing.eval_cut3r.video_depth.metadata import dataset_metadata
from src.testing.eval_cut3r.video_depth.utils import save_depth_maps
from src.testing.utils.model_choice import choose_model, CONFIGS
import json
from accelerate import PartialState
import time
from tqdm import tqdm
import random
from src.dust3r.utils.image import load_images_for_eval as load_images
from src.testing.eval_cut3r.video_depth.postprocess import estimate_focal_knowing_depth
from src.dust3r.utils.camera import pose_encoding_to_camera
from src.training.utils.debug_vis_utils import debug_vis_output_utils_separate_depth
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri, unproject_depth_map_to_point_map
import open3d as o3d

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _save_sequence_pointclouds(seq_dir, sv_depth, sv_ext_np, sv_K_np,
                               sv_imgs, is_ppd, mwf=None):
    """Write per-view (per-frame) PLY point clouds via o3d.io.write_point_cloud.

    For each frame fid, writes `{seq_dir}/pointclouds/frame{NN}.ply` with
    that frame's unprojected world-space points and colors. If `mwf` is
    provided, also writes `{seq_dir}/pointclouds_wsky/frame{NN}.ply` with
    sky pixels (argmax == K-1) dropped.
    """
    n_frames = sv_depth.shape[0]
    pad = 3 if n_frames >= 100 else 2

    pts_nhw3 = unproject_depth_map_to_point_map(sv_depth, sv_ext_np, sv_K_np)

    if is_ppd:
        colors_nhw3 = np.clip(sv_imgs, 0, 1).astype(np.float32)
    else:
        colors_nhw3 = np.clip(sv_imgs * _IMAGENET_STD + _IMAGENET_MEAN,
                              0, 1).astype(np.float32)

    base_dir = os.path.join(seq_dir, "pointclouds")
    os.makedirs(base_dir, exist_ok=True)

    sky_mask_nhw = None
    wsky_dir = None
    if mwf is not None:
        if mwf.ndim == 5 and mwf.shape[0] == 1:
            mwf = mwf[0]
        if mwf.ndim == 4 and mwf.shape[0] == n_frames:
            sky_mask_nhw = (mwf.argmax(axis=-1) == mwf.shape[-1] - 1)
            wsky_dir = os.path.join(seq_dir, "pointclouds_wsky")
            os.makedirs(wsky_dir, exist_ok=True)
        else:
            print(f"  pointcloud_wsky: mog_weight_full shape {mwf.shape} "
                  f"does not match n_frames={n_frames}; skipping sky-filtered PLYs.")

    for fid in range(n_frames):
        pts_i = pts_nhw3[fid].reshape(-1, 3).astype(np.float64)
        cols_i = colors_nhw3[fid].reshape(-1, 3).astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_i)
        pcd.colors = o3d.utility.Vector3dVector(cols_i)
        o3d.io.write_point_cloud(
            os.path.join(base_dir, f"frame{fid:0{pad}d}.ply"), pcd,
        )

        if wsky_dir is not None:
            keep = ~sky_mask_nhw[fid].reshape(-1)
            pcd_w = o3d.geometry.PointCloud()
            pcd_w.points = o3d.utility.Vector3dVector(pts_i[keep])
            pcd_w.colors = o3d.utility.Vector3dVector(cols_i[keep])
            o3d.io.write_point_cloud(
                os.path.join(wsky_dir, f"frame{fid:0{pad}d}.ply"), pcd_w,
            )

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(420000)
PATCH_SIZE = 16
IMG_NORM_USED = None
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        help="name of the model",
        default="ours",
    )

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default="224")

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )
    parser.add_argument("--crop_center_112", type=int, default=0)
    parser.add_argument("--cam_inp", type=int, default=0)
    parser.add_argument("--gt_cam_output", type=int, default=0)
    parser.add_argument("--output_normalize", type=int, default=0)
    return parser


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )


def eval_pose_estimation_dist(args, model, img_path, save_dir=None, mask_path=None):

    metadata = dataset_metadata.get(args.eval_dataset)
    anno_path = metadata.get("anno_path", None)

    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            if metadata.get("_single_image", False) and metadata.get("_recursive", False):
                import glob as _glob
                seq_list = sorted(
                    os.path.relpath(f, img_path)
                    for f in _glob.glob(os.path.join(img_path, "**", "*.png"), recursive=True)
                )
            elif metadata.get("_single_image", False):
                seq_list = sorted(
                    f for f in os.listdir(img_path)
                    if os.path.isfile(os.path.join(img_path, f))
                )
            else:
                seq_list = os.listdir(img_path)
                seq_list = [
                    seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
                ]
        seq_list = sorted(seq_list)

    values = [str(int(v)) for k, v in CONFIGS.items()]
    values = ''.join(values)
    save_dir = osp.join(args.output_dir, values)
    if save_dir is None:
        save_dir = args.output_dir

    os.makedirs(save_dir, exist_ok=True)
    with open(osp.join(save_dir, f"config.json"), "w") as f:
        json.dump(CONFIGS, f)

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    model_name = args.model_name
    is_ppd = 'ppd' in model_name or 'ppdv' in model_name or 'ppvd' in model_name

    with distributed_state.split_between_processes(seq_list) as seqs:
        load_img_size = args.size
        fps_all = []
        time_all = []
        seq_names = []
        for seq in tqdm(seqs):
            if True:
                dir_path = metadata["dir_path_func"](img_path, seq)

                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get(
                    "mask_path_seq_func", lambda mask_path, seq: None
                )
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                if metadata.get("_single_image", False):
                    filelist = [os.path.join(img_path, seq)]
                else:
                    filelist = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                    filelist.sort()
                    filelist = filelist[:: args.pose_eval_stride]

                views = prepare_input(
                    filelist,
                    [True for _ in filelist],
                    size=load_img_size,
                    crop=not args.no_crop,
                    patch_size=PATCH_SIZE,
                    img_norm=IMG_NORM_USED,
                )

                start = time.time()
                predictions = model.inference(views, device, **CONFIGS)
                end = time.time()

                n_frames = len(views)
                elapsed = end - start
                fps = n_frames / elapsed
                fps_all.append(fps)
                time_all.append(elapsed)
                seq_names.append(seq)
                print(f"Seq: {seq}, Frames: {n_frames}, FPS: {fps:.2f}, Time: {elapsed:.3f}s")

                seq_dir = f"{save_dir}/{seq}"
                os.makedirs(seq_dir, exist_ok=True)
                save_depth_maps(None, seq_dir, conf_self=None, depth_maps=predictions['depth'].squeeze().cpu())

                # Save extra data for visualization script
                with torch.no_grad():
                    sv_extrinsic, sv_intrinsic = pose_encoding_to_extri_intri(
                        predictions["pose_enc"], predictions["images"].shape[-2:]
                    )
                    sv_depth = predictions["depth"].cpu().numpy().squeeze(0)  # (N, H, W)
                    sv_ext_np = sv_extrinsic.cpu().numpy().squeeze(0)        # (N, 3, 4)
                    sv_K_np = sv_intrinsic.cpu().numpy().squeeze(0)          # (N, 3, 3)
                    sv_imgs = predictions["images"].squeeze(0).permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)

                # MoG-specific: pack raw component weights if emitted by the model.
                extra_save_kwargs = {}
                raw_preds_dict = predictions.get("raw_preds")
                if (isinstance(raw_preds_dict, dict)
                        and "mog_weight_full" in raw_preds_dict):
                    mwf = raw_preds_dict["mog_weight_full"]
                    if hasattr(mwf, "detach"):
                        mwf = mwf.detach().cpu().numpy()
                    if mwf.ndim >= 1 and mwf.shape[0] == 1:
                        mwf = mwf[0]
                    extra_save_kwargs["mog_weight_full"] = mwf

                np.savez_compressed(
                    os.path.join(seq_dir, "vis_data.npz"),
                    depth=sv_depth,
                    extrinsics=sv_ext_np,
                    intrinsics=sv_K_np,
                    images=sv_imgs,
                    is_ppd=is_ppd,
                    **extra_save_kwargs,
                )

                _save_sequence_pointclouds(
                    seq_dir, sv_depth, sv_ext_np, sv_K_np, sv_imgs, is_ppd,
                    mwf=extra_save_kwargs.get("mog_weight_full"),
                )

                is_mog = True if hasattr(model, 'net') and hasattr(model.net, 'head_mog') and (model.net.head_mog is not None) else False
                if is_mog:
                    with torch.no_grad():
                        views = predictions['views']
                        out_subdir_vis = os.path.join(seq_dir, 'vis')
                        bid = hash(views['instance'][0][0]) if isinstance(views['instance'][0], list) else views['instance'][0]
                        debug_vis_output_utils_separate_depth(
                            predictions['raw_preds'], views, bid, out_subdir_vis, complete=False)

        # Save FPS results
        if fps_all:
            mean_fps = np.mean(fps_all)
            mean_time = np.mean(time_all)
            total_time = np.sum(time_all)

            fps_log_path = osp.join(save_dir, f"fps_{distributed_state.process_index}.txt")
            with open(fps_log_path, "w") as f:
                f.write(f"Dataset: {args.eval_dataset}\n")
                f.write(f"Model: {args.model_name}\n")
                f.write(f"Sequences: {len(fps_all)}\n")
                f.write(f"Mean FPS: {mean_fps:.2f}\n")
                f.write(f"Mean time per seq: {mean_time:.3f}s\n")
                f.write(f"Total time: {total_time:.3f}s\n")
                f.write(f"\n{'Sequence':<50} {'FPS':>8} {'Time(s)':>10}\n")
                f.write("-" * 70 + "\n")
                for sname, sfps, stime in zip(seq_names, fps_all, time_all):
                    f.write(f"{sname:<50} {sfps:>8.2f} {stime:>10.3f}\n")

            print(f"\n=== FPS Summary ({args.eval_dataset}) ===")
            print(f"  Sequences: {len(fps_all)}")
            print(f"  Mean FPS:  {mean_fps:.2f}")
            print(f"  Mean time: {mean_time:.3f}s")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Saved to: {fps_log_path}")

            # Save as JSON for easy parsing
            fps_json_path = osp.join(save_dir, f"fps_{distributed_state.process_index}.json")
            fps_json = {
                "dataset": args.eval_dataset,
                "model": args.model_name,
                "n_sequences": len(fps_all),
                "mean_fps": float(mean_fps),
                "mean_time": float(mean_time),
                "total_time": float(total_time),
                "per_sequence": {
                    sname: {"fps": float(sfps), "time": float(stime)}
                    for sname, sfps, stime in zip(seq_names, fps_all, time_all)
                },
            }
            with open(fps_json_path, "w") as f:
                json.dump(fps_json, f, indent=2)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    CONFIGS["crop_center_112"] = bool(args.crop_center_112)
    CONFIGS["cam_inp"] = bool(args.cam_inp)
    CONFIGS["gt_cam_output"] = bool(args.gt_cam_output)
    CONFIGS["output_normalize"] = bool(args.output_normalize)

    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False
    args.no_crop = True

    def prepare_input(
        img_paths,
        img_mask,
        size,
        raymaps=None,
        raymap_mask=None,
        revisit=1,
        update=True,
        crop=True,
        patch_size=16,
        img_norm=None,
    ):
        images = load_images(
            img_paths,
            size=size,
            crop=crop,
            patch_size=patch_size,
            img_norm=img_norm,
        )
        views = []
        if raymaps is None and raymap_mask is None:
            num_views = len(images)

            for i in range(num_views):
                view = {
                    "img": images[i]["img"],
                    "ray_map": torch.full(
                        (
                            images[i]["img"].shape[0],
                            6,
                            images[i]["img"].shape[-2],
                            images[i]["img"].shape[-1],
                        ),
                        torch.nan,
                    ),
                    "true_shape": torch.from_numpy(images[i]["true_shape"]),
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
        else:

            num_views = len(images) + len(raymaps)
            assert len(img_mask) == len(raymap_mask) == num_views
            assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

            j = 0
            k = 0
            for i in range(num_views):
                view = {
                    "img": (
                        images[j]["img"]
                        if img_mask[i]
                        else torch.full_like(images[0]["img"], torch.nan)
                    ),
                    "ray_map": (
                        raymaps[k]
                        if raymap_mask[i]
                        else torch.full_like(raymaps[0], torch.nan)
                    ),
                    "true_shape": (
                        torch.from_numpy(images[j]["true_shape"])
                        if img_mask[i]
                        else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                    ),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.from_numpy(
                        np.eye(4).astype(np.float32)
                    ).unsqueeze(0),
                    "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                    "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                    "update": torch.tensor(img_mask[i]).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                if img_mask[i]:
                    j += 1
                if raymap_mask[i]:
                    k += 1
                views.append(view)
            assert j == len(images) and k == len(raymaps)

        if revisit > 1:
            new_views = []
            for r in range(revisit):
                for i in range(len(views)):
                    new_view = deepcopy(views[i])
                    new_view["idx"] = r * len(views) + i
                    new_view["instance"] = str(r * len(views) + i)
                    if r > 0:
                        if not update:
                            new_view["update"] = torch.tensor(False).unsqueeze(0)
                    new_views.append(new_view)
            return new_views
        return views

    model_name = args.model_name
    loaded = choose_model(model_name)
    model = loaded.model
    checkpoint_path = loaded.checkpoint_path
    patch_size = loaded.patch_size
    ImgNorm_used = loaded.img_norm
    mysize = loaded.model_size
    args.size = mysize
    PATCH_SIZE = patch_size
    IMG_NORM_USED = ImgNorm_used
    eval_pose_estimation(args, model, save_dir=args.output_dir)
