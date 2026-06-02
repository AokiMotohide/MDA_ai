
import math
import numpy as np
from scipy.spatial import cKDTree as KDTree
from math import isqrt
from typing import Literal, Optional
import torch
from einops import rearrange, repeat
from tqdm import tqdm
import moviepy.editor as mpy
import os

import sys
from add_ckpt_path import add_path_to_da3

from depth_anything_3.specs import Gaussians
from src.training.utils.gsplat_rendering_extrafeat import rasterization_extrafeat, rasterization_2dgs_extrafeat
from src.depth_anything_3.model.utils.gs_renderer import render_3dgs
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.layout_helpers import hcat, vcat
from depth_anything_3.utils.visualize import vis_depth_map_tensor
from depth_anything_3.utils.geometry import affine_inverse, get_world_rays, sample_image_grid, get_world_rays_corrected

VIDEO_QUALITY_MAP = {
    "low": {"crf": "28", "preset": "veryfast"},
    "medium": {"crf": "23", "preset": "medium"},
    "high": {"crf": "18", "preset": "slow"},
}



def export_to_gs_video(
    prediction: Prediction,
    export_dir: Optional[str] = None,
    extrinsics: Optional[torch.Tensor] = None,  # render views' world2cam, "b v 4 4"
    intrinsics: Optional[torch.Tensor] = None,  # render views' unnormed intrinsics, "b v 3 3"
    chunk_size: Optional[int] = 4,
    color_mode: Literal["RGB+D", "RGB+ED"] = "RGB+D",
    vis_depth: Optional[Literal["hcat", "vcat"]] = "hcat",
    enable_tqdm: Optional[bool] = True,
    output_name: Optional[str] = None,
    video_quality: Literal["low", "medium", "high"] = "low",
    use_2dgs: bool = False,
    render_func=None,
    gt_imgs: Optional[torch.Tensor] = None,
    gt_imgs_inp: Optional[torch.Tensor] = None,
) -> None:
    gs_world = prediction.gaussians

    # if render resolution is not provided, render the input ones
    H, W = prediction.depth.shape[-2:]
    
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] /= W
    intrinsics[..., 1, :] /= H

    color, depth, alphas, conf = chunk_rendering_test(
        gaussians=gs_world,
        tgt_extr=extrinsics,
        tgt_intr=intrinsics,
        image_shape=(H, W),
        chunk_size=chunk_size,
        use_sh=True,
        color_mode=color_mode,
        enable_tqdm=enable_tqdm,
        use_2dgs=use_2dgs,
        render_func=render_func
    )
    mask0 = (depth > 1e-3) & (alphas > 1e-2)
    
    if export_dir is None:
        gs_conf_percentile = torch.quantile(conf, 0.1, dim=1)
        depth_conf_mask = conf >= gs_conf_percentile[:, None]
        # print('depth_conf_mask', depth_conf_mask.shape, depth_conf_mask.min(), depth_conf_mask.max())
        # print('mask0', mask0.shape, mask0.min(), mask0.max())
        mask1 = mask0 & depth_conf_mask
        # print('mask1', mask1.shape, mask1.min(), mask1.max())
        rgb_masked = color * mask1.unsqueeze(2)
        color = color * mask0.float().unsqueeze(2)
        
        return color, mask0, rgb_masked, mask1

    # save as video
    ffmpeg_params = [
        "-crf",
        VIDEO_QUALITY_MAP[video_quality]["crf"],
        "-preset",
        VIDEO_QUALITY_MAP[video_quality]["preset"],
        "-pix_fmt",
        "yuv420p",
    ]  # best compatibility

    os.makedirs(os.path.join(export_dir, "gs_video"), exist_ok=True)
    for idx in range(color.shape[0]):
        video_i = color[idx]
        if vis_depth is not None:
            depth_i = vis_depth_map_tensor(depth[0])
            cat_fn = hcat if vis_depth == "hcat" else vcat
            video_i = torch.stack([cat_fn(c, d) for c, d in zip(video_i, depth_i)])
        frames = list(
            (video_i.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        )  # T x H x W x C, uint8, numpy()

        fps = 8
        clip = mpy.ImageSequenceClip(frames, fps=fps)
        output_name = f"{idx:04d}" if output_name is None else output_name
        save_path = os.path.join(export_dir, f"gs_video/{output_name}.mp4")
        # clip.write_videofile(save_path, codec="libx264", audio=False, bitrate="4000k")
        clip.write_videofile(
            save_path,
            codec="libx264",
            audio=False,
            fps=fps,
            ffmpeg_params=ffmpeg_params,
        )
        
    rgb_masked = color
    mask1 = mask0
    if conf is not None:
        gs_conf_percentile = torch.quantile(conf, 0.1, dim=1)
        depth_conf_mask = conf >= gs_conf_percentile[:, None]
        mask1 = mask1 & depth_conf_mask
        rgb_masked = color * depth_conf_mask.float().unsqueeze(2)
        for idx in range(color.shape[0]):
            video_i = rgb_masked[idx]
            frames = list(
                (video_i.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            )  # T x H x W x C, uint8, numpy()

            fps = 8
            clip = mpy.ImageSequenceClip(frames, fps=fps)
            output_name = f"{idx:04d}" if output_name is None else output_name
            save_path = os.path.join(export_dir, f"gs_video/{output_name}_masked.mp4")
            clip.write_videofile(
                save_path,
                codec="libx264",
                audio=False,
                fps=fps,
                ffmpeg_params=ffmpeg_params,
            )
    
    if gt_imgs is not None:
        for idx in range(color.shape[0]):
            video_i = gt_imgs[idx]
            frames = list(
                (video_i.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            )  # T x H x W x C, uint8, numpy()

            fps = 8
            clip = mpy.ImageSequenceClip(frames, fps=fps)
            output_name = f"{idx:04d}" if output_name is None else output_name
            save_path = os.path.join(export_dir, f"gs_video/{output_name}_gt.mp4")
            clip.write_videofile(
                save_path,
                codec="libx264",
                audio=False,
                fps=fps,
                ffmpeg_params=ffmpeg_params,
            )
    
    if gt_imgs_inp is not None:
        for idx in range(color.shape[0]):
            video_i = gt_imgs_inp[idx]
            frames = list(
                (video_i.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            )  # T x H x W x C, uint8, numpy()
            
            fps = 8
            clip = mpy.ImageSequenceClip(frames, fps=fps)
            output_name = f"{idx:04d}" if output_name is None else output_name
            save_path = os.path.join(export_dir, f"gs_video/{output_name}_gt_inp.mp4")
            clip.write_videofile(
                save_path,
                codec="libx264",
                audio=False,
                fps=fps,
                ffmpeg_params=ffmpeg_params,
            )
            
    return color, mask0, rgb_masked, mask1




def chunk_rendering_test(tgt_extr, tgt_intr, image_shape, gaussians, chunk_size=8, render_func=None, **kwargs):
    v = tgt_extr.shape[1]
    chunk_size = min(v, chunk_size)
    all_colors = []
    all_depths = []
    all_confs = []
    all_alphas = []
    for chunk_idx in range(math.ceil(v / chunk_size)):
        s = int(chunk_idx * chunk_size)
        e = int((chunk_idx + 1) * chunk_size)
        cur_n_view = tgt_extr[:, s:e].shape[1]
        conf = None
        if render_func is not None:
            color, depth, alphas, conf = render_func(
                extrinsics=rearrange(tgt_extr[:, s:e], "b v ... -> (b v) ..."),  # w2c
                intrinsics=rearrange(tgt_intr[:, s:e], "b v ... -> (b v) ..."),  # normed
                image_shape=image_shape,
                gaussian=gaussians,
                num_view=cur_n_view,
                **kwargs,
            )
        else:
            color, depth, alphas = render_3dgs(
                extrinsics=rearrange(tgt_extr[:, s:e], "b v ... -> (b v) ..."),  # w2c
                intrinsics=rearrange(tgt_intr[:, s:e], "b v ... -> (b v) ..."),  # normed
                image_shape=image_shape,
                gaussian=gaussians,
                num_view=cur_n_view,
                **kwargs,
            )
        
        all_colors.append(rearrange(color, "(b v) ... -> b v ...", v=cur_n_view))
        all_depths.append(rearrange(depth, "(b v) ... -> b v ...", v=cur_n_view))
        all_alphas.append(rearrange(alphas, "(b v) ... -> b v ...", v=cur_n_view))
        if conf is not None:
            all_confs.append(rearrange(conf, "(b v) ... -> b v ...", v=cur_n_view))
        
    all_colors = torch.cat(all_colors, dim=1)
    all_depths = torch.cat(all_depths, dim=1)
    all_alphas = torch.cat(all_alphas, dim=1)
    if len(all_confs) > 0:
        all_confs = torch.cat(all_confs, dim=1)
        return all_colors, all_depths, all_alphas, all_confs

    return all_colors, all_depths, all_alphas, None




def invalid_to_nans(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float("nan")
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr


def invalid_to_zeros(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0  # number of point per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz



def depth2points(intrinsics, depth, cam2worlds):
    device = intrinsics.device
    dtype = intrinsics.dtype
    H, W = depth.shape[-2:]
    b, v = cam2worlds.shape[:2]
    
    if cam2worlds.shape[-2] != 4:
        cam2worlds = torch.cat([cam2worlds, torch.zeros_like(cam2worlds[..., :1, :])], dim=-2)
        cam2worlds[..., 3, 3] = 1.0
    
    intr_normed = intrinsics.detach().clone()
    intr_normed[..., 0, :] /= W
    intr_normed[..., 1, :] /= H
    
    xy_ray, _ = sample_image_grid((H, W), device, disp=False)
    xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)  # b v h w xy
    origins, directions = get_world_rays_corrected(
        xy_ray,
        repeat(cam2worlds, "b v i j -> b v h w i j", h=H, w=W),
        repeat(intr_normed, "b v i j -> b v h w i j", h=H, w=W),
    )
    means_world = origins + directions * depth[..., None]
    # means_world = rearrange(means_world, "b v h w d -> b (v h w) d")
    return means_world


@torch.no_grad()
def get_joint_pointcloud_center_scale(pts, valid_masks=None, z_only=False, center=True):
    # set invalid points to NaN

    _pts = []
    for i in range(len(pts)):
        valid_mask = valid_masks[i] if valid_masks is not None else None
        _pt = invalid_to_nans(pts[i], valid_mask).reshape(len(pts[i]), -1, 3)
        _pts.append(_pt)

    _pts = torch.cat(_pts, dim=1)

    # compute median center
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median
