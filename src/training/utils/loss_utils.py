# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, floor
from src.depth_anything_3.utils.geometry import generate_raymaps, generate_raymaps_preds
from einops import rearrange
# from src.dust3r.utils.geometry import inv


def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max = 100):
    """
    Checks if 'loss_tensor' contains inf or nan. If it does, replace those 
    values with zero and print the name of the loss tensor.

    Args:
        loss_tensor (torch.Tensor): The loss tensor to check.
        loss_name (str): Name of the loss (for diagnostic prints).

    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
    """
        
    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        for _ in range(10):
            print(f"{loss_name} has inf or nan. Setting those values to 0.")
            assert False
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device),
            loss_tensor
        )

    loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor




def camera_loss(pred_pose_enc_list, gt_extrinsic, gt_intrinsic, image_size_hw, gt_pts3d_scale, pred_pts3d_scale, loss_type="l1", gamma=0.6, pose_encoding_type="relT_quaR_FoV", weight_T = 1.0, weight_R = 1.0, weight_fl = 0.5, frame_num = -100):
    num_predictions = len(pred_pose_enc_list)

    anchor_camera_inv = inv(gt_extrinsic[:, 0:1, :, :])
    # in dust3r frammework, dataset gt extrinsic is cam2world, but vggt predicts world2cam
    gt_extrinsic_aligned = inv(anchor_camera_inv @ gt_extrinsic)
    gt_pose_encoding = extri_intri_to_pose_encoding(gt_extrinsic_aligned, gt_intrinsic, image_size_hw, pose_encoding_type=pose_encoding_type, gt_pts3d_scale=gt_pts3d_scale)

    loss_T = loss_R = loss_fl = 0

    for i in range(num_predictions):
        i_weight = gamma ** (num_predictions - i - 1)

        cur_pred_pose_enc = pred_pose_enc_list[i] # B, S, 9
        cur_pred_pose_enc = torch.cat([
            cur_pred_pose_enc[:, :, :3] / pred_pts3d_scale.view(-1, 1, 1),
            cur_pred_pose_enc[:, :, 3:]
        ], dim=2)

        if frame_num>0:
            loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(cur_pred_pose_enc[:, :frame_num].clone(), gt_pose_encoding[:, :frame_num].clone(), loss_type=loss_type)
        else:
            loss_T_i, loss_R_i, loss_fl_i = camera_loss_single(cur_pred_pose_enc.clone(), gt_pose_encoding.clone(), loss_type=loss_type)

        loss_T += loss_T_i * i_weight
        loss_R += loss_R_i * i_weight
        loss_fl += loss_fl_i * i_weight

    loss_T = loss_T / num_predictions
    loss_R = loss_R / num_predictions
    loss_fl = loss_fl / num_predictions
    loss_camera = loss_T * weight_T + loss_R * weight_R + loss_fl * weight_fl

    loss_dict = {
        "loss_camera": loss_camera,
        "loss_T": loss_T,
        "loss_R": loss_R,
        "loss_fl": loss_fl
    }

    return loss_dict



def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="l1"):
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (cur_pred_pose_enc[..., 3:7] - gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] - gt_pose_encoding[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    loss_T = loss_T.clamp(max=100) # TODO: remove this
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl


def normalize_pointcloud(pts3d, valid_mask, normalize_using_first_view, eps=1e-3):
    """
    pts3d: B, S, H, W, 3
    valid_mask: B, S, H, W
    """
    if normalize_using_first_view:
        dist = pts3d[:, 0:1, ...].norm(dim=-1)
        valid_mask = valid_mask[:, 0:1, ...]
    else:
        dist = pts3d.norm(dim=-1)

    dist_sum = (dist * valid_mask).sum(dim=[1,2,3])
    valid_count = valid_mask.sum(dim=[1,2,3])

    avg_scale = (dist_sum / (valid_count + eps)).clamp(min=eps, max=1e3)

    pts3d = pts3d / (avg_scale.view(-1, 1, 1, 1, 1) + 1e-8)
    return pts3d, avg_scale


def depth_loss(depth, depth_conf, gt_depth, valid_mask, gamma=1.0, alpha=0.2, loss_type="conf", predict_disparity=False, affine_inv=False, gradient_loss= None, valid_range=-1, disable_conf=False, all_mean=False, normalize_gt=True, normalize_pred=False, normalize_using_first_view=False, normalize_with_metric_mask=False, is_metric_mask=None, gt_pts3d_scale=None, pred_pts3d_scale=None, **kwargs):
    gt_depth = gt_depth[..., None]

    if loss_type == "conf":
        conf_loss_dict = conf_loss(depth, depth_conf, gt_depth, valid_mask,
                               batch=None, normalize_pred=normalize_pred, normalize_gt=normalize_gt,
                               gamma=gamma, alpha=alpha, affine_inv=affine_inv, gradient_loss=gradient_loss, valid_range=valid_range, postfix="", disable_conf=disable_conf, all_mean=all_mean, normalize_using_first_view=normalize_using_first_view, normalize_with_metric_mask=normalize_with_metric_mask, is_metric_mask=is_metric_mask, gt_pts3d_scale=gt_pts3d_scale, pred_pts3d_scale=pred_pts3d_scale)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    return conf_loss_dict


def point_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask, normalize_pred=True, normalize_gt=True, gamma=1.0, alpha=0.2, affine_inv=False, gradient_loss=None, valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False, conf_loss_type="v1", gt_pts3d_scale=None, temporal_matching_loss=False, normalize_using_first_view=False, normalize_with_metric_mask=False, is_metric_mask=None, **kwargs):
    """
    pts3d: B, S, H, W, 3
    pts3d_conf: B, S, H, W
    gt_pts3d: B, S, H, W, 3
    valid_mask: B, S, H, W
    """
    if conf_loss_type == "v1":
        conf_loss_fn = conf_loss
    else:
        raise ValueError(f"Invalid conf loss type: {conf_loss_type}")

    conf_loss_dict = conf_loss_fn(pts3d, pts3d_conf, gt_pts3d, valid_mask, batch=None,
                                normalize_pred=normalize_pred, normalize_gt=normalize_gt, gamma=gamma, alpha=alpha, affine_inv=affine_inv,
                                gradient_loss=gradient_loss, valid_range=valid_range, camera_centric_reg=camera_centric_reg, disable_conf=disable_conf, all_mean=all_mean, gt_pts3d_scale=gt_pts3d_scale, temporal_matching_loss=temporal_matching_loss, normalize_using_first_view=normalize_using_first_view, normalize_with_metric_mask=normalize_with_metric_mask, is_metric_mask=is_metric_mask,)

    return conf_loss_dict


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100, return_mask=False):
    """
    Filters a loss tensor by keeping only values below a certain quantile threshold.
    Also clamps individual values to hard_max.

    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss

    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= 1000:
        # too small, just return
        if return_mask:
            return loss_tensor, torch.ones_like(loss_tensor, dtype=torch.bool)
        return loss_tensor

    # Randomly sample if tensor is too large
    if False:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values
    loss_tensor = loss_tensor.clamp(max=hard_max)

    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        if return_mask:
            return loss_tensor[quantile_mask], quantile_mask
        return loss_tensor[quantile_mask]
    if return_mask:
        return loss_tensor, torch.ones_like(loss_tensor, dtype=torch.bool)
    return loss_tensor


def conf_loss(pts3d, pts3d_conf, gt_pts3d, valid_mask, batch, 
              normalize_gt=True, normalize_pred=True, gamma=1.0, alpha=0.2, affine_inv=False, gradient_loss=None, 
              valid_range=-1, camera_centric_reg=-1, disable_conf=False, all_mean=False, postfix="", gt_pts3d_scale=None, 
              temporal_matching_loss=False, normalize_using_first_view=False, normalize_with_metric_mask=False, 
              is_metric_mask=None, pred_pts3d_scale=None):
    
    # normalize
    if gt_pts3d_scale is not None and pred_pts3d_scale is not None:
        gt_pts3d = gt_pts3d / (gt_pts3d_scale.view(-1, 1, 1, 1, 1) + 1e-8)
        pts3d = pts3d / (pred_pts3d_scale.view(-1, 1, 1, 1, 1) + 1e-8)
    elif normalize_with_metric_mask:
        assert is_metric_mask is not None
        non_metric_mask = ~is_metric_mask
        gt_pts3d_non_metric = gt_pts3d[non_metric_mask]
        valid_mask_non_metric = valid_mask[non_metric_mask]
        
        # Normalize non-metric points
        _, gt_pts3d_scale_non_metric = normalize_pointcloud(gt_pts3d_non_metric, valid_mask_non_metric, normalize_using_first_view)

        # for pred backpropagation, we have to normalize in place with divide operation, so we just get pred_scale here
        pred_pts3d_non_metric = pts3d[non_metric_mask].clone().detach()
        _, pred_pts3d_scale_non_metric = normalize_pointcloud(pred_pts3d_non_metric, valid_mask_non_metric, normalize_using_first_view)

        # Put normalized points back
        gt_pts3d_scale = torch.ones_like(is_metric_mask, dtype=gt_pts3d.dtype)
        gt_pts3d_scale[non_metric_mask] = gt_pts3d_scale_non_metric.to(gt_pts3d.dtype)
        gt_pts3d = gt_pts3d / (gt_pts3d_scale.view(-1, 1, 1, 1, 1) + 1e-8)

        pred_pts3d_scale = torch.ones_like(is_metric_mask, dtype=pts3d.dtype)
        pred_pts3d_scale[non_metric_mask] = pred_pts3d_scale_non_metric.to(pts3d.dtype)
        pts3d = pts3d / (pred_pts3d_scale.view(-1, 1, 1, 1, 1) + 1e-8)
    else:
        if normalize_gt:
            if gt_pts3d_scale is None:
                gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask, normalize_using_first_view)
            else:
                gt_pts3d = gt_pts3d / (gt_pts3d_scale.view(-1, 1, 1, 1, 1) + 1e-8)

        if normalize_pred:
            pts3d, pred_pts3d_scale = normalize_pointcloud(pts3d, valid_mask, normalize_using_first_view)

        if (not normalize_pred) and (not normalize_gt):
            gt_pts3d, gt_pts3d_scale = normalize_pointcloud(gt_pts3d, valid_mask, normalize_using_first_view)
            pts3d = pts3d / (gt_pts3d_scale.view(-1, 1, 1, 1, 1) + 1e-8)
            
    if affine_inv:
        raise NotImplementedError()
        # scale, shift = closed_form_scale_and_shift(pts3d, gt_pts3d, valid_mask)
        # pts3d = pts3d * scale + shift

    loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames, loss_temporal_matching = reg_loss(
        pts3d, gt_pts3d, valid_mask, gradient_loss=gradient_loss, temporal_matching_loss=temporal_matching_loss)

    if disable_conf:
        conf_loss_first_frame = gamma * loss_reg_first_frame
        conf_loss_other_frames = gamma * loss_reg_other_frames
    else:
        first_frame_conf = pts3d_conf[:, 0:1, ...]
        other_frames_conf = pts3d_conf[:, 1:, ...]
        first_frame_mask = valid_mask[:, 0:1, ...]
        other_frames_mask = valid_mask[:, 1:, ...]

        conf_loss_first_frame = gamma * loss_reg_first_frame * first_frame_conf[first_frame_mask] - alpha * torch.log(first_frame_conf[first_frame_mask])
        conf_loss_other_frames = gamma * loss_reg_other_frames * other_frames_conf[other_frames_mask] - alpha * torch.log(other_frames_conf[other_frames_mask])

    if valid_range>0:
        conf_loss_first_frame = filter_by_quantile(conf_loss_first_frame, valid_range)
        conf_loss_other_frames = filter_by_quantile(conf_loss_other_frames, valid_range)

    conf_loss_first_frame = check_and_fix_inf_nan(conf_loss_first_frame, f"conf_loss_first_frame{postfix}")
    conf_loss_other_frames = check_and_fix_inf_nan(conf_loss_other_frames, f"conf_loss_other_frames{postfix}")

    if all_mean and conf_loss_first_frame.numel() > 0 and conf_loss_other_frames.numel() > 0:
        all_conf_loss = torch.cat([conf_loss_first_frame, conf_loss_other_frames])
        conf_loss = all_conf_loss.mean() if all_conf_loss.numel() > 0 else 0

        # for logging only
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0
    else:
        conf_loss_first_frame = conf_loss_first_frame.mean() if conf_loss_first_frame.numel() > 0 else 0
        conf_loss_other_frames = conf_loss_other_frames.mean() if conf_loss_other_frames.numel() > 0 else 0

        conf_loss = conf_loss_first_frame + conf_loss_other_frames

    # Verified that the loss is the same

    loss_dict = {
        f"loss_conf{postfix}": conf_loss,
        f"loss_reg1{postfix}": loss_reg_first_frame.detach().mean() if loss_reg_first_frame.numel() > 0 else 0,
        f"loss_reg2{postfix}": loss_reg_other_frames.detach().mean() if loss_reg_other_frames.numel() > 0 else 0,
        f"loss_conf1{postfix}": conf_loss_first_frame,
        f"loss_conf2{postfix}": conf_loss_other_frames,
    }

    # loss_grad_first_frame and loss_grad_other_frames are already meaned
    loss_grad = loss_grad_first_frame + loss_grad_other_frames
    loss_dict[f"loss_grad1{postfix}"] = loss_grad_first_frame
    loss_dict[f"loss_grad2{postfix}"] = loss_grad_other_frames
    loss_dict[f"loss_grad{postfix}"] = loss_grad

    if temporal_matching_loss:
        loss_dict[f"loss_temporal_matching{postfix}"] = loss_temporal_matching
    else:
        loss_dict[f"loss_temporal_matching{postfix}"] = 0

    loss_dict[f"gt_pts3d_scale{postfix}"] = gt_pts3d_scale
    loss_dict[f"pred_pts3d_scale{postfix}"] = pred_pts3d_scale

    return loss_dict


def reg_loss(pts3d, gt_pts3d, valid_mask, gradient_loss=None, temporal_matching_loss=False):
    first_frame_pts3d = pts3d[:, 0:1, ...]
    first_frame_gt_pts3d = gt_pts3d[:, 0:1, ...]
    first_frame_mask = valid_mask[:, 0:1, ...]

    other_frames_pts3d = pts3d[:, 1:, ...]
    other_frames_gt_pts3d = gt_pts3d[:, 1:, ...]
    other_frames_mask = valid_mask[:, 1:, ...]

    loss_reg_first_frame = torch.norm(first_frame_gt_pts3d[first_frame_mask] - first_frame_pts3d[first_frame_mask], dim=-1)
    loss_reg_other_frames = torch.norm(other_frames_gt_pts3d[other_frames_mask] - other_frames_pts3d[other_frames_mask], dim=-1)

    if gradient_loss == "grad":
        bb, ss_f, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb*ss_f, hh, ww, nc), first_frame_gt_pts3d.reshape(bb*ss_f, hh, ww, nc), first_frame_mask.reshape(bb*ss_f, hh, ww))
        bb, ss_o, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb*ss_o, hh, ww, nc), other_frames_gt_pts3d.reshape(bb*ss_o, hh, ww, nc), other_frames_mask.reshape(bb*ss_o, hh, ww))

        # we all mean gradient loss
        loss_grad_other_frames *= (ss_o // ss_f)

    elif gradient_loss == "normal":
        bb, ss, hh, ww, nc = first_frame_pts3d.shape
        loss_grad_first_frame = gradient_loss_multi_scale(first_frame_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_gt_pts3d.reshape(bb*ss, hh, ww, nc), first_frame_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=normal_loss, scales=3)
        bb, ss, hh, ww, nc = other_frames_pts3d.shape
        loss_grad_other_frames = gradient_loss_multi_scale(other_frames_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_gt_pts3d.reshape(bb*ss, hh, ww, nc), other_frames_mask.reshape(bb*ss, hh, ww), gradient_loss_fn=normal_loss, scales=3)
    else:
        loss_grad_first_frame = 0
        loss_grad_other_frames = 0

    loss_reg_first_frame = check_and_fix_inf_nan(loss_reg_first_frame, "loss_reg_first_frame")
    loss_reg_other_frames = check_and_fix_inf_nan(loss_reg_other_frames, "loss_reg_other_frames")

    if temporal_matching_loss:
        # B, S, H, W, 3
        pred_diff = pts3d[:, 1:] - pts3d[:, :-1]
        gt_diff = gt_pts3d[:, 1:] - gt_pts3d[:, :-1]
        valid_mask = valid_mask[:, 1:] & valid_mask[:, :-1]

        loss_temporal_matching = F.l1_loss(pred_diff[valid_mask], gt_diff[valid_mask], reduction='none')
        loss_temporal_matching = check_and_fix_inf_nan(loss_temporal_matching, "loss_temporal_matching")
        valid_count = valid_mask.sum()
        loss_temporal_matching = (loss_temporal_matching.sum() / valid_count) if valid_count > 0 else 0
    else:
        loss_temporal_matching = 0

    return loss_reg_first_frame, loss_reg_other_frames, loss_grad_first_frame, loss_grad_other_frames, loss_temporal_matching


def normal_loss(prediction, target, gt_points_delta, mask, cos_eps=1e-8, conf=None, **kwargs):
    """
    Computes the normal-based loss by comparing the angle between
    predicted normals and ground-truth normals.

    prediction: (B, H, W, 3) - Predicted 3D coordinates/points
    target:     (B, H, W, 3) - Ground-truth 3D coordinates/points
    mask:       (B, H, W)    - Valid pixel mask (1 = valid, 0 = invalid)

    Returns: scalar (averaged over valid regions)
    """
    pred_normals, pred_valids, pred_weights = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids, gt_weights   = point_map_to_normal(target,     mask, eps=cos_eps)

    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)
    all_weights = pred_weights * gt_weights # shape: (4, B, H, W)
    all_weights = -F.max_pool2d(-all_weights, kernel_size=5, stride=1, padding=2)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # pred_normals = pred_normals[all_valid].clone()
    # gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    # pred_normals and gt_normals are (4, B, H, W, 3)
    # We want to compare corresponding normals where all_valid is True
    dot = torch.sum(pred_normals * gt_normals, dim=-1)  # shape: (4, B, H, W)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot  # shape: (4, B, H, W)
    
    loss = loss * all_weights * all_valid


    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            gamma = 1.0 # hard coded
            alpha = 0.2 # hard coded

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    point_map: (B, H, W, 3)  - 3D points laid out in a 2D grid
    mask:      (B, H, W)     - valid pixels (bool)

    Returns:
      normals: (4, B, H, W, 3)  - normal vectors for each of the 4 cross-product directions
      valids:  (4, B, H, W)     - corresponding valid masks
    """

    with torch.cuda.amp.autocast(enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Each pixel's neighbors
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Direction vectors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Four cross products (shape: B,H,W,3 each)
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity for each cross-product direction
        # We require that both directions' pixels are valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack them to shape (4,B,H,W,3), (4,B,H,W)
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize each direction's normal
        # shape is (4, B, H, W, 3), so dim=-1 is the vector dimension
        # clamp_min(eps) to avoid division by zero
        # lengths = torch.norm(normals, dim=-1, keepdim=True).clamp_min(eps)
        # normals = normals / lengths
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)
        
        weight = torch.stack(
            [torch.abs(up_dir[..., 2]), torch.abs(left_dir[..., 2]), torch.abs(down_dir[..., 2]), torch.abs(right_dir[..., 2])], dim=-1
        )
        weight = weight.max(dim=-1).values * 4
        weight = torch.exp( - weight**2).unsqueeze(0)
        # Zero out invalid entries so they don't pollute subsequent computations
        # normals = normals * valids.unsqueeze(-1)

    return normals, valids, weight.detach()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W

    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)


    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]
        gamma = 1.0
        alpha = 0.2

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)


    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    image_loss = check_and_fix_inf_nan(image_loss, "gradient_loss")

    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        image_loss = torch.sum(image_loss) / divisor

    return image_loss



def gradient_loss_weighted(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2, k=1e3, avg=False):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W

    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)
    
    gt_grad_x = torch.abs(target[:, :, 1:].detach() - target[:, :, :-1]) * k
    weight_x1 = torch.exp(
        (- gt_grad_x**2)
    )
    grad_x = grad_x * weight_x1
    
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)
    
    gt_grad_y = torch.abs(target[:, 1:, :].detach() - target[:, :-1, :]) * k
    weight_y1 = torch.exp(
        (- gt_grad_y**2)
    )
    grad_y = grad_y * weight_y1

    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = 1 / (1 / conf[:, :, 1:] + 1 / conf[:, :, :-1]).detach()
        conf_y = 1 / (1 / conf[:, 1:, :] + 1 / conf[:, :-1, :]).detach()
        
        # print('conf_x', conf_x.shape, conf_x.min(), conf_x.max(), conf_x.mean())
        # print('conf_y', conf_y.shape, conf_y.min(), conf_y.max(), conf_y.mean())
        grad_x = grad_x * conf_x.detach()
        grad_y = grad_y * conf_y.detach()


    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    image_loss = check_and_fix_inf_nan(image_loss, "gradient_loss")

    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        image_loss = torch.sum(image_loss) / divisor

    return image_loss


def gradient_img_weighted(target, mask, conf=None, gamma=1.0, alpha=0.2, k=1000, avg=False):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W

    mask = mask[..., None].expand(-1, -1, -1, target.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    gt_grad_x = torch.abs(target[:, :, 1:] - target[:, :, :-1]) * k
    # print('gt_grad_x', gt_grad_x.shape, gt_grad_x.mean())
    weight_x1 = torch.exp(
        (- gt_grad_x**2)
    )
    weight_x1 = mask_x * weight_x1
    
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    gt_grad_y = torch.abs(target[:, 1:, :] - target[:, :-1, :]) * k
    # print('gt_grad_y', gt_grad_y.shape, gt_grad_y.mean())
    weight_y1 = torch.exp(
        (- gt_grad_y**2)
    )
    weight_y1 = mask_y * weight_y1

    weight_x1 = F.pad(weight_x1, (0, 0, 0, 1), mode='constant', value=0)
    weight_y1 = F.pad(weight_y1, (0, 0, 0, 0, 0, 1), mode='constant', value=0)
    image_loss = torch.cat([weight_x1, weight_y1], dim=-1)
    return image_loss

    

def gradient_img_multi_scale(prediction, target, mask, scales=4, gradient_loss_fn = gradient_img_weighted, conf=None):
    """
    Compute gradient loss across multiple scales
    """

    total_gt = []
    total_pred = []
    for scale in range(scales):
        step = pow(2, scale)
        img_gt = gradient_loss_fn(
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            avg=False
        )
        img_pred = gradient_loss_fn(
            prediction[:, ::step, ::step],
            mask[:, ::step, ::step],
            avg=False
        )
        
        diff = torch.abs(target[:, ::step, ::step] - prediction[:, ::step, ::step])
        
        total_gt.append(img_gt * diff * 30)
        total_pred.append(img_pred * diff * 30)

    return total_gt, total_pred


def gradient_loss_multi_scale(prediction, target, mask, scales=4, gradient_loss_fn = gradient_loss, conf=None):
    """
    Compute gradient loss across multiple scales
    """

    total = 0
    for scale in range(scales):
        step = pow(2, scale)

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total



def gradient_loss_weighted_points(prediction, target, gt_points_delta, mask, conf=None, gamma=1.0, alpha=0.2, k=2, avg=False, loss_type='l1'):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W

    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)
    
    target_diff_z = torch.abs(target[:, :, 1:] - target[:, :, :-1])
    tangent = target_diff_z / torch.clamp(gt_points_delta[:, :, 1:, 0].unsqueeze(-1), min=1e-8)

    tangent = tangent * k
    weight_x1 = torch.exp(
        (- tangent**2)
    )
    # remove boundary-parts
    weight_x1 = -F.max_pool2d(-weight_x1, kernel_size=5, stride=1, padding=2)

    grad_x = grad_x * weight_x1
    
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)
    
    target_diff_z = torch.abs(target[:, 1:, :] - target[:, :-1, :])
    tangent = target_diff_z / torch.clamp(gt_points_delta[:, 1:, :, 0].unsqueeze(-1), min=1e-8)
    
    tangent = tangent * k
    weight_y1 = torch.exp(
        (- tangent**2)
    )
    weight_y1 = -F.max_pool2d(-weight_y1, kernel_size=5, stride=1, padding=2)
    grad_y = grad_y * weight_y1.detach()

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = 1 / (1 / conf[:, :, 1:] + 1 / conf[:, :, :-1]).detach()
        conf_y = 1 / (1 / conf[:, 1:, :] + 1 / conf[:, :-1, :]).detach()

        grad_x = grad_x * conf_x.detach()
        grad_y = grad_y * conf_y.detach()

    image_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    # image_loss = check_and_fix_inf_nan(image_loss, "gradient_loss")

    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        image_loss = torch.sum(image_loss) / divisor
    
    if 'l2' in loss_type:
        image_loss = image_loss * 0.1

    return image_loss


def gradient_loss_multi_scale_points(prediction, target, gt_points_delta, mask, scales=4, gradient_loss_fn=gradient_loss_weighted_points, conf=None, loss_type='l1'):
    """
    Compute gradient loss across multiple scales
    """

    total = 0
    for scale in range(scales):
        step = pow(2, scale)

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            gt_points_delta[:, ::step, ::step] * step,
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None,
            loss_type=loss_type,
        )

    total = total / scales
    return total



def gradient_img_weighted_points(target, gt_points_delta, mask, conf=None, gamma=1.0, alpha=0.2, k=2, avg=False):
    # prediction: B, H, W, C
    # target: B, H, W, C
    # mask: B, H, W

    mask = mask[..., None].expand(-1, -1, -1, target.shape[-1])

    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    target_diff_z = torch.abs(target[:, :, 1:] - target[:, :, :-1])
    tangent = target_diff_z / torch.clamp(gt_points_delta[:, :, 1:, 0].unsqueeze(-1), min=1e-8)
    tangent = tangent * k
    weight_x1 = torch.exp(
        (- tangent**2)
    )
    weight_x1 = mask_x * weight_x1
    
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    target_diff_z = torch.abs(target[:, 1:, :] - target[:, :-1, :])
    tangent = target_diff_z / torch.clamp(gt_points_delta[:, 1:, :, 1].unsqueeze(-1), min=1e-8)
    tangent = tangent * k
    weight_y1 = torch.exp(
        (- tangent**2)
    )
    weight_y1 = mask_y * weight_y1

    weight_x1 = F.pad(weight_x1, (0, 0, 0, 1), mode='constant', value=0)
    weight_y1 = F.pad(weight_y1, (0, 0, 0, 0, 0, 1), mode='constant', value=0)
    image_loss = torch.cat([weight_x1, weight_y1], dim=-1)
    return image_loss

    

def gradient_img_multi_scale_points(prediction, target, gt_points_delta, mask, scales=4, gradient_loss_fn = gradient_img_weighted_points, conf=None):
    """
    Compute gradient loss across multiple scales
    """

    total_gt = []
    total_pred = []
    for scale in range(scales):
        step = pow(2, scale)
        img_gt = gradient_loss_fn(
            target[:, ::step, ::step],
            gt_points_delta[:, ::step, ::step] * step,
            mask[:, ::step, ::step],
            avg=False
        )
        img_pred = gradient_loss_fn(
            prediction[:, ::step, ::step],
            gt_points_delta[:, ::step, ::step] * step,
            mask[:, ::step, ::step],
            avg=False
        )
        total_gt.append(img_gt)
        total_pred.append(img_pred)

    return total_gt, total_pred




def torch_quantile(
    input: torch.Tensor,
    q: float | torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Sanitization: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Sanitization: inteporlation
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Sanitization: out
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Logic
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Rectification: keepdim
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)


def conf_loss_func(loss_term, conf, mask, gamma, alpha, detach=False, avg=True):
    if detach:
        loss = gamma * loss_term.detach() * conf - alpha * torch.log(conf)
    else:
        loss = gamma * loss_term * conf - alpha * torch.log(conf)
    if mask is not None:
        loss = loss * mask
    
    # item1 = (loss_term * conf * mask).mean()
    # item2 = (alpha * torch.log(conf) * mask).mean()
    # item3 = (loss_term * mask).mean()
    # print('item1', item1.item(), item2.item(), item3.item())
    if not avg:
        return loss

    return loss.mean()


class LoadBalancingLoss(nn.Module):
    def __init__(self, num_experts: int, top_k: int = 1):
        """
        Args:
            num_experts: Total number of experts in the MoE layer.
            top_k: The number of experts selected per token (usually 1 or 2).
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, router_logits: torch.Tensor):
        """
        Computes the auxiliary load balancing loss.
        
        Args:
            router_logits: Tensor of shape (batch_size * seq_len, num_experts).
                           These are the raw output values from the router BEFORE Softmax.
        
        Returns:
            loss: Scalar tensor representing the auxiliary loss.
        """
        # 1. Calculate the 'Soft' Probabilities (P)
        # These are differentiable and allow gradients to flow to the router.
        # Shape: (batch_size, num_experts)
        probs = F.softmax(router_logits, dim=-1)
        
        # P_i: The average probability allocated to expert i across the batch.
        # Shape: (num_experts,)
        mean_probs = probs.mean(dim=0)

        # 2. Calculate the 'Hard' Dispatch Fractions (f)
        # This represents the actual discrete decisions made (non-differentiable).
        
        # Get the indices of the top-k experts
        # Shape: (batch_size, top_k)
        _, selected_experts = torch.topk(router_logits, k=self.top_k, dim=-1)
        
        # Create a mask of selected experts (One-hot or Multi-hot)
        # Shape: (batch_size, num_experts)
        expert_mask = torch.zeros_like(router_logits)
        expert_mask.scatter_(1, selected_experts, 1.0)
        
        # f_i: The fraction of tokens physically routed to expert i.
        # We detach this because we don't backprop through the discrete selection here.
        # Shape: (num_experts,)
        dispatch_fractions = expert_mask.mean(dim=0)

        # 3. Compute the Loss
        # Formula: N * sum(P_i * f_i)
        # Ideally, we want both vectors to be Uniform (value 1/N).
        # The dot product is minimized when both distributions are uniform.
        loss = self.num_experts * torch.sum(mean_probs * dispatch_fractions.detach())
        
        return loss


def load_balancing_loss(all_nll, max_indices):
    # all_nll: bb x ll x H x W x M
    # max_indices: bb x ll x H x W x 1
    # print('all_nll', all_nll.shape, all_nll.min(), all_nll.max())
    # print('max_indices', max_indices.shape, max_indices.min(), max_indices.max())
    bb, ll, hh, ww, m = all_nll.shape
    expert_mask = torch.zeros_like(all_nll)
    expert_mask.scatter_(-1, max_indices, 1.0)
    
    dispatch_fractions = expert_mask.reshape(bb*ll*hh*ww, m).mean(dim=0)
    probs = F.softmax(-all_nll, dim=-1)
    probs = probs.reshape(bb*ll*hh*ww, m)
    mean_probs = probs.mean(dim=0)
    loss = torch.sum(mean_probs * dispatch_fractions.detach())
    # print('load_balancing_loss', loss.item())
    return loss



def calculate_pose_scale_from_depths(pred_depth, gt_depth, valid_mask_depth, normalize_pred=True):
    pose_scales = []
    pose_scale_flatten = []
    bb = gt_depth.shape[0]
    valid_mask_depth_flatten = valid_mask_depth.reshape(bb, -1)
    gt_depth_flatten = gt_depth.reshape(bb, -1) * valid_mask_depth_flatten
    pred_depth_flatten = pred_depth.reshape(bb, -1) * valid_mask_depth_flatten
    
    for i in range(bb):
        valid_mask_depth_flatten_i = ((valid_mask_depth_flatten[i] > 0) & (gt_depth_flatten[i] > 0.01))
        gt_depth_flatten_valid = gt_depth_flatten[i][valid_mask_depth_flatten_i]
        pred_depth_flatten_valid = pred_depth_flatten[i][valid_mask_depth_flatten_i]
        if normalize_pred and (len(gt_depth_flatten_valid) > 100):
            pose_scales_flatten_valid = gt_depth_flatten_valid / torch.clamp(pred_depth_flatten_valid, min=1e-8)
            pose_scales_i = torch.median(pose_scales_flatten_valid, dim=0).values
            pose_scales.append(pose_scales_i)
            pose_scale_flatten.append(pred_depth_flatten_valid / torch.clamp(gt_depth_flatten_valid, min=1e-8))

        else:
            pose_scales.append(torch.ones_like(gt_depth_flatten[i][0]))

    pose_scales = torch.stack(pose_scales, dim=0).reshape(bb)
    if len(pose_scale_flatten) > 0:
        pose_scale_flatten = torch.cat(pose_scale_flatten, dim=0).reshape(-1)
    else:
        pose_scale_flatten = torch.zeros_like(pose_scales.reshape(-1))
    return pose_scales, pose_scale_flatten



def calculate_mean_pose_scale_from_depths(pred_depth, gt_depth, valid_mask_depth, normalize_pred=True):
    pose_scales = []
    pose_scale_flatten = []
    bb = gt_depth.shape[0]
    valid_mask_depth_flatten = valid_mask_depth.reshape(bb, -1)
    gt_depth_flatten = gt_depth.reshape(bb, -1) * valid_mask_depth_flatten
    pred_depth_flatten = pred_depth.reshape(bb, -1) * valid_mask_depth_flatten
    
    for i in range(bb):
        valid_mask_depth_flatten_i = ((valid_mask_depth_flatten[i] > 0) & (gt_depth_flatten[i] > 0.01))
        gt_depth_flatten_valid = gt_depth_flatten[i][valid_mask_depth_flatten_i]
        pred_depth_flatten_valid = pred_depth_flatten[i][valid_mask_depth_flatten_i]
        if normalize_pred and (len(gt_depth_flatten_valid) > 100):
            # pose_scales_flatten_valid = gt_depth_flatten_valid / torch.clamp(pred_depth_flatten_valid, min=1e-8)
            # pose_scales_i = torch.mean(pose_scales_flatten_valid)
            # gt_depth_flatten_valid_mean = torch.mean(gt_depth_flatten_valid)
            # pred_depth_flatten_valid_mean = torch.mean(pred_depth_flatten_valid)
            # pose_scales_i = gt_depth_flatten_valid_mean / torch.clamp(pred_depth_flatten_valid_mean, min=1e-8)
            # pose_scales_i_inv = pred_depth_flatten_valid / torch.clamp(gt_depth_flatten_valid, min=1e-8)
            # pose_scales_i_inv = torch.mean(pose_scales_i_inv)
            pose_scales_i = gt_depth_flatten_valid / torch.clamp(pred_depth_flatten_valid, min=1e-8)
            pose_scales_i = torch.mean(pose_scales_i)
            pose_scales.append(pose_scales_i)
            pose_scale_flatten.append(pred_depth_flatten_valid / torch.clamp(gt_depth_flatten_valid, min=1e-8))

        else:
            pose_scales.append(torch.ones_like(gt_depth_flatten[i][0]))

    pose_scales = torch.stack(pose_scales, dim=0).reshape(bb)
    if len(pose_scale_flatten) > 0:
        pose_scale_flatten = torch.cat(pose_scale_flatten, dim=0).reshape(-1)
    else:
        pose_scale_flatten = torch.zeros_like(pose_scales.reshape(-1))
    return pose_scales, pose_scale_flatten



def calculate_pose_scale_from_points(preds, gts, pred_depth_max):

    gt_origins_hw, gt_directions_hw = generate_raymaps(gts) # b, l, h, w, 3
    pred_origins_hw, pred_directions_hw = generate_raymaps_preds(preds) # b, l, h, w, 3
    valid_mask_depth = gts["valid_mask"]
    bb = valid_mask_depth.shape[0]
    gt_depth = gts["depthmap"]
    
    pred_pts3d = pred_origins_hw + pred_directions_hw * pred_depth_max
    gt_pts3d = gt_origins_hw + gt_directions_hw * gt_depth.unsqueeze(-1)
    valid_mask = valid_mask_depth.reshape(bb, -1)
    pred_pts3d = pred_pts3d.reshape(bb, -1, 3)
    gt_pts3d = gt_pts3d.reshape(bb, -1, 3)

    pose_scale_inv = []
    pose_scale_flatten = []
    for bid in range(bb):
        valid_mask_bid = valid_mask[bid]
        pred_pts3d_bid = pred_pts3d[bid][valid_mask_bid] # 3
        gt_pts3d_bid = gt_pts3d[bid][valid_mask_bid]
        pose_scale_inv_bid = pred_pts3d_bid.norm(dim=-1) / (gt_pts3d_bid.norm(dim=-1) + 1e-8)
        if len(pred_pts3d_bid) > 100:   
            pose_scale_flatten.append(pose_scale_inv_bid)
            
            pose_scale_inv_bid = torch.median(pose_scale_inv_bid, dim=0).values
            pose_scale_inv.append(pose_scale_inv_bid)
        else:
            pose_scale_inv.append(torch.ones_like(pose_scale_inv_bid))
    pose_scale_inv = torch.stack(pose_scale_inv, dim=0).detach()
    pose_scale_inv = torch.clamp(pose_scale_inv, min=0.2, max=5.0)
    if len(pose_scale_flatten) > 0:
        pose_scale_flatten = torch.cat(pose_scale_flatten, dim=0).reshape(-1)
    else:
        pose_scale_flatten = torch.zeros_like(pose_scale_inv.reshape(-1))
    
    return 1.0 / pose_scale_inv, pose_scale_flatten



def manual_gumbel_softmax(logits, logits_scale=1.0, tau=0.1, hard=False, dim=-1):
    # 1. Generate Uniform noise between 0 and 1
    # Adding a tiny epsilon (1e-20) to prevent log(0) errors
    logits = logits * logits_scale
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    
    # 2. Add Gumbel noise to logits and apply temperature-scaled softmax
    y_soft = F.softmax((logits + gumbel_noise) / tau, dim=dim)
    if hard:
        # 3. Straight-Through Estimator (if hard=True)
        # Get the actual argmax
        indices = y_soft.argmax(dim=dim, keepdim=True)
        
        # Create a true one-hot tensor
        y_hard = torch.zeros_like(logits).scatter_(dim, indices, 1.0)
        
        # The Straight-Through trick: 
        # y_hard - y_soft.detach() evaluates to exactly y_hard.
        # However, during backpropagation, PyTorch ignores the detached stuff 
        # and routes the gradients straight through `y_soft`.
        y_out = y_hard - y_soft.detach() + y_soft
        return y_out
        
    return y_soft


def normalize_twice():
    pose_scales_mean, _ = calculate_mean_pose_scale_from_depths(
        pred_depth_max, gt_depth, valid_mask_depth, self.normalize_pred
    )
    # pose_scales_mean = pose_scales_mean.clamp(0.01, 50.0)

    pred_depth_mean_aligned = pred_depth_max * rearrange(
        pose_scales_mean, "b -> b () () ()"
    )
    pose_scales_second, pose_scale_flatten = align_depths_by_median(
        pred_depth_mean_aligned,
        gt_depth,
        valid_mask_depth,
        type="weighted_median",
    )
    pose_scales = pose_scales_mean * pose_scales_second
    print("pose_scales_mean", pose_scales_mean)
    print("pose_scales_second", pose_scales_second)
    pose_scales = pose_scales.clamp(0.01, 50.0)

    reg_loss_mean = torch.abs(pose_scales_mean - 1).mean() * 0.01
    total_loss = total_loss + reg_loss_mean
    details[self_name + "_reg_loss_mean" + "/00"] = float(
        reg_loss_mean.detach()
    )



@torch.no_grad()
def align_depths_by_median(
    normalized_pred_depth,
    normalized_gt_depth,
    valid_mask_depth,
    type="median",
    min_valid_count=100,
    max_downsample_stride=8,
    target_pixels=16384,
):
    """
    type: 'median' or 'weighted_median'
    weighted_median: solves min sum_i (1/gt_depth_i) * |scale * pred_i - gt_i|
                     to give more importance to closer points.
    scale = median(gt_depth / pred_depth)
    return scale, scale_flatten
    """

    def weighted_median(values, weights):
        sorted_indices = torch.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumulative_weights = torch.cumsum(sorted_weights, dim=0)
        half_weight = cumulative_weights[-1] * 0.5
        median_idx = torch.searchsorted(cumulative_weights, half_weight, right=False)
        median_idx = torch.clamp(median_idx, min=0, max=sorted_values.numel() - 1)
        return sorted_values[median_idx]

    if type not in ["median", "weighted_median"]:
        raise ValueError(
            f"Unsupported type '{type}'. Expected 'median' or 'weighted_median'."
        )

    hh, ww = normalized_gt_depth.shape[-2:]
    adaptive_stride = int(round(((hh * ww) / float(target_pixels)) ** 0.5))
    adaptive_stride = max(1, min(max_downsample_stride, adaptive_stride))

    normalized_gt_depth = normalized_gt_depth[..., ::adaptive_stride, ::adaptive_stride]
    normalized_pred_depth = normalized_pred_depth[
        ..., ::adaptive_stride, ::adaptive_stride
    ]
    valid_mask_depth = valid_mask_depth[..., ::adaptive_stride, ::adaptive_stride]

    bb = normalized_gt_depth.shape[0]
    gt_depth_flatten = normalized_gt_depth.reshape(bb, -1)
    pred_depth_flatten = normalized_pred_depth.reshape(bb, -1)
    valid_mask_depth_flatten = valid_mask_depth.reshape(bb, -1) > 0

    scales = []
    scale_flatten = []
    for i in range(bb):
        valid_i = valid_mask_depth_flatten[i]
        valid_i = (
            valid_i
            & torch.isfinite(gt_depth_flatten[i])
            & torch.isfinite(pred_depth_flatten[i])
        )
        valid_i = (
            valid_i & (gt_depth_flatten[i] > 0.01) & (pred_depth_flatten[i] > 1e-8)
        )
        gt_valid = gt_depth_flatten[i][valid_i]
        pred_valid = pred_depth_flatten[i][valid_i]
        if gt_valid.numel() > min_valid_count:
            ratio_valid = gt_valid / torch.clamp(pred_valid, min=1e-8)
            scale_flatten.append(ratio_valid)
            if type == "median":
                scale_i = torch.median(ratio_valid, dim=0).values
            else:
                base_weights = 1.0 / torch.clamp(
                    gt_valid.clamp(min=0.1 * gt_valid.mean()), min=1e-8
                )
                weights = base_weights * pred_valid
                scale_i = weighted_median(ratio_valid, weights)
            scales.append(scale_i)
        else:
            scales.append(
                torch.tensor(
                    1.0,
                    device=normalized_gt_depth.device,
                    dtype=normalized_gt_depth.dtype,
                )
            )
    scales = torch.stack(scales, dim=0).reshape(bb)
    if len(scale_flatten) > 0:
        scale_flatten = torch.cat(scale_flatten, dim=0).reshape(-1)
    else:
        scale_flatten = torch.zeros_like(scales.reshape(-1))
    return scales, scale_flatten



def calculate_tangent_weight(gts):
    origins, directions = generate_raymaps(gts, use_cam2worlds=False)
    gt_depth_gradient = gts["depthmap"]
    valid_mask_depth = gts["valid_mask"]
    bb, ll, hh, ww = gt_depth_gradient.shape
    
    # gt_depth_gradient = F.interpolate(gt_depth_gradient, size=(hh//2, ww//2), mode='bilinear', align_corners=False)
    # gt_depth_gradient = F.interpolate(gt_depth_gradient, size=(hh, ww), mode='bilinear', align_corners=False)

    directions_xdisp = directions[:, :, :, 1: ] - directions[:, :, :, :-1]
    directions_ydisp = directions[:, :, 1:, :] - directions[:, :, :-1, :]

    gt_points_xdisp_delta = directions_xdisp[..., 0] * gt_depth_gradient[:, :, :, :-1]
    gt_points_ydisp_delta = directions_ydisp[..., 1] * gt_depth_gradient[:, :, :-1, :]
    
    gt_points_xdisp_delta = F.pad(gt_points_xdisp_delta, (1, 0), mode='constant', value=1)
    gt_points_ydisp_delta = F.pad(gt_points_ydisp_delta, (0, 0, 1, 0), mode='constant', value=1)
    
    gt_points_delta = torch.stack([gt_points_xdisp_delta, gt_points_ydisp_delta], dim=-1)
    gt_points_delta = torch.norm(gt_points_delta, dim=-1)
    gt_points_delta = torch.clamp(gt_points_delta, min=1e-6)
    
    grad_x = torch.abs(gt_depth_gradient[:, :, :, 1:] - gt_depth_gradient[:, :, :, :-1])
    grad_y = torch.abs(gt_depth_gradient[:, :, 1:, :] - gt_depth_gradient[:, :, :-1, :])
    
    grad_x = F.pad(grad_x, (1, 0), mode='constant', value=0)
    grad_y = F.pad(grad_y, (0, 0, 1, 0), mode='constant', value=0)
    
    assert grad_x.shape == gt_points_delta.shape, (grad_x.shape, gt_points_delta.shape)
    assert grad_y.shape == gt_points_delta.shape, (grad_y.shape, gt_points_delta.shape)
    
    valid_mask_x = valid_mask_depth[:, :, :, 1:] * valid_mask_depth[:, :, :, :-1]
    valid_mask_y = valid_mask_depth[:, :, 1:, :] * valid_mask_depth[:, :, :-1, :]
    valid_mask_x = F.pad(valid_mask_x, (1, 0), mode='constant', value=0)
    valid_mask_y = F.pad(valid_mask_y, (0, 0, 1, 0), mode='constant', value=0)
    
    grad_x = grad_x * valid_mask_x
    grad_y = grad_y * valid_mask_y
    
    tangent_x = grad_x / gt_points_delta
    tangent_y = grad_y / gt_points_delta
    
    tangent_weight = (torch.maximum(tangent_x, tangent_y) > 20).float()
    # print('tangent_weight raw', tangent_weight.max(), tangent_weight.min(), tangent_weight.mean())
    
    tangent_weight = F.max_pool2d(tangent_weight, kernel_size=7, stride=1, padding=3)
    # print('tangent_weight', tangent_weight.max(), tangent_weight.min(), tangent_weight.mean())
    
    return gt_points_delta, tangent_weight




def calculate_details_weight(pred_depth):
    # pred_depth: batch x L x H x W
    pred_depth = pred_depth.clone()
    batch_size, L, H, W = pred_depth.shape
    pred_depth = pred_depth.reshape(batch_size * L, 1, H, W)

    pred_depth_max = pred_depth.max()
    pred_depth[pred_depth < 1e-3] = pred_depth_max + 1

    max_pooled = F.max_pool2d(pred_depth, kernel_size=5, stride=1, padding=2)
    min_pooled = -F.max_pool2d(-max_pooled, kernel_size=5, stride=1, padding=2)

    result = min_pooled.reshape(batch_size, L, H, W)
    pred_depth = pred_depth.reshape(batch_size, L, H, W)
    depth_median = torch.median(pred_depth.reshape(batch_size, -1), dim=1).values
    min_pooled = min_pooled.reshape(batch_size, L, H, W)
    max_pooled = max_pooled.reshape(batch_size, L, H, W)
    mask = (
        (torch.abs(result - pred_depth) > 0.05 * depth_median.reshape(batch_size, 1, 1, 1))
        & (min_pooled < pred_depth_max + 0.1)
        & (max_pooled < pred_depth_max + 0.1)
        & (pred_depth < pred_depth_max + 0.1)
    )
    
    mask = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1)
    mask = (mask > 0.01)

    return mask

    