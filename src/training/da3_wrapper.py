from sympy.geometry import ellipse
from src.depth_anything_3.cfg import create_object, load_config
import json
import numpy as np
import torch
import yaml
# from src.depth_anything_3.model.da3 import DepthAnything3Net
import os
from torch import nn
from src.depth_anything_3.utils.utils_training import prepare_inputs
from src.depth_anything_3.model.mog_inference import find_gmm_mode_gpu_chunk, find_gmm_mode_gpu_multilayer
from einops import rearrange
from src.depth_anything_3.utils.geometry import affine_inverse
from src.depth_anything_3.model.utils.transform import extri_intri_to_pose_encoding
from src.testing.eval_cut3r.video_depth.tools import absolute_value_scaling2

class DA3Wrapper(nn.Module):
    def __init__(self, config_path, **kwargs):
        super().__init__()
        gs_names = ['gs_head', 'gs_adapter', 'controlnet', 'head_mog']
        
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)['config']
        
        else:
            assert config_path.endswith('.yaml'), 'config_path must be a yaml file'
            with open(config_path, 'r') as f:
                config = load_config(config_path)

        self.net = create_object(config)
        opt_cam_dec = config.get('opt_cam_dec', False)
        if opt_cam_dec:
            gs_names.append('cam_dec')
            
        if kwargs.get('full_train', False):
            for name, module in self.net.named_parameters():
                module.requires_grad = True
        elif len(kwargs.get('set_freeze', '')) > 0:
            self.set_freeze(kwargs['set_freeze'])
        else:
            for name, module in self.net.named_parameters():
                module.requires_grad = False
                if any(gs_name in name for gs_name in gs_names):
                    module.requires_grad = True
            
        if 'calculate_conf' in kwargs:
            self.calculate_conf = kwargs['calculate_conf']
        else:
            self.calculate_conf = False
        
        if 'loss_type' in kwargs:
            self.loss_type = kwargs['loss_type']
        else:
            self.loss_type = None
        
        if 'pretrain_as_possible' in kwargs:
            self.pretrain_as_possible = kwargs['pretrain_as_possible']
        else:
            self.pretrain_as_possible = False

    def set_freeze(self, freeze: str):
        if not freeze or freeze == "none":
            return

        print(f"Applying freeze strategy: {freeze}")

        # 1. Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        # 2. Parse modes (supports comma-separated list)
        modes = [m.strip() for m in freeze.split(",")]

        # Helper to get the actual backbone model
        backbone_model = None
        if hasattr(self.net, "backbone"):
            backbone_model = self.net.backbone
            # Handle the case where backbone wraps the actual model in .pretrained (e.g. DinoV2 wrapper)
            if hasattr(backbone_model, "pretrained"):
                backbone_model = backbone_model.pretrained

        alt_start = backbone_model.alt_start if hasattr(backbone_model, "alt_start") else None
        last_block_idx = (
            len(backbone_model.blocks)
            if backbone_model and hasattr(backbone_model, "blocks")
            else None
        )
        assert (
            alt_start is not None and last_block_idx is not None
        ), "Backbone must have alt_start and blocks for selective unfreezing"

        # --- Backbone Freeze Logic ---
        if "backbone" in modes:
            print("  Unfreezing full backbone")
            for param in backbone_model.parameters():
                param.requires_grad = True
        else:
            # selective backbone unfreezing
            if "global" in modes:
                print("  Unfreezing entire global backbone blocks")
                unfreeze_idx = [idx for idx in range(alt_start, last_block_idx) if idx % 2 == 1]
                if backbone_model and hasattr(backbone_model, "blocks"):
                    for idx in unfreeze_idx:
                        if idx < len(backbone_model.blocks):
                            for param in backbone_model.blocks[idx].parameters():
                                param.requires_grad = True

            if "local" in modes:
                print("  Unfreezing local backbone blocks")
                unfreeze_idx = [idx for idx in range(alt_start, last_block_idx) if idx % 2 == 0]
                if backbone_model and hasattr(backbone_model, "blocks"):
                    for idx in unfreeze_idx:
                        if idx < len(backbone_model.blocks):
                            for param in backbone_model.blocks[idx].parameters():
                                param.requires_grad = True

        # --- Camera Freeze Logic ---
        unfreeze_cam_dec = "cam_dec" in modes
        unfreeze_cam_enc = "cam_enc" in modes

        if unfreeze_cam_dec:
            print("  Unfreezing camera decoder")
            if hasattr(self.net, "cam_dec") and self.net.cam_dec is not None:
                for param in self.net.cam_dec.parameters():
                    param.requires_grad = True
            if hasattr(self.net, "aux_cam_decs"):
                for cam_dec in self.net.aux_cam_decs:
                    for param in cam_dec.parameters():
                        param.requires_grad = True

        if unfreeze_cam_enc:
            print("  Unfreezing camera encoder")
            if hasattr(self.net, "cam_enc") and self.net.cam_enc is not None:
                for param in self.net.cam_enc.parameters():
                    param.requires_grad = True

        if "head" in modes:
            print("  Unfreezing depth head")
            assert (
                hasattr(self.net, "head_mog") and self.net.head_mog is not None
            ), "Model must have head_mog to unfreeze head"
            if hasattr(self.net, "head_mog") and self.net.head_mog is not None:
                for param in self.net.head_mog.parameters():
                    param.requires_grad = True

        if 'lastlayer' in modes:
            print("  Unfreezing last layer")
            assert hasattr(self.net, "head_mog") and self.net.head_mog is not None
            for model_module in self.net.head_mog.scratch.output_conv2_list:
                for param in model_module.parameters():
                    param.requires_grad = True
                    
        # Log trainable parameters count
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        total_params_count = sum(p.numel() for p in trainable_params)
        print(f"Number of trainable parameters (tensors): {len(trainable_params)}")
        print(f"Total number of trainable parameters (elements): {total_params_count / 1e6:.2f} M")

        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        if len(trainable_names) > 0:
            print(f"Example trainable param: {trainable_names[0]}")
    
    def forward(self, *args, **kwargs):
        # dump_model_parameters(self.net, 'model_parameters.txt')
        return self.net(*args, **kwargs)
    
    
    def inference(self, views, device,
                  cam_inp=False,
                  gt_cam_output=False,
                  output_normalize=False,
                  loss_type=None,
                  is_multilayer_depth=False,
                  use_sky_mask=False,
                  **kwargs):
        
        if loss_type is None:
            loss_type = self.loss_type if self.loss_type is not None else 'l1'
        is_mog = True if hasattr(self.net, 'head_mog') and (self.net.head_mog is not None) else False
        is_moe = True if 'moe' in loss_type else False
        
        camera_pose0 = views[0]["camera_pose"]
        camera_pose0_inv = affine_inverse(camera_pose0)
        for view in views:
            view['camera_pose'] = torch.einsum("bij,bjk->bik", camera_pose0_inv, view['camera_pose'])
        
        len_views = len(views)

        views = prepare_inputs(views, device, new_gs_mask=False, rand_shuffle=False)
        if cam_inp:
            extrinsics = views["camera_extrinsics"]
            intrinsics = views["camera_intrinsics"]
            
            extrinsics0 = extrinsics[:, 0, :, :]
            extrinsics0_inv = affine_inverse(extrinsics0)
            extrinsics = torch.einsum("blij,bjk->blik", extrinsics, extrinsics0_inv)
            
        else:
            extrinsics = None
            intrinsics = None
        
        preds = self.net(views["img"], extrinsics=extrinsics, intrinsics=intrinsics, 
                               infer_gs=False)
        
        
        if is_mog and is_multilayer_depth:
            depth_inference, indices, extra_depth, transparent_mask = find_gmm_mode_gpu_multilayer(
                preds, loss_type=loss_type, saved_dir=None, chunk_size=16)
            preds['indices'] = indices
            preds['extra_depth'] = extra_depth
            preds['transparent_mask'] = transparent_mask
            pred_depth_conf = preds["depth_conf"][0]
            preds['depth_inference'] = depth_inference

        elif is_mog and (not is_multilayer_depth):
            depth_inference, indices = find_gmm_mode_gpu_chunk(preds, loss_type=loss_type,
                                                            saved_dir=None,
                                                            chunk_size=16)
            preds['indices'] = indices
            pred_depth_conf = preds["depth_conf"][0]
            preds['depth_inference'] = depth_inference
            extra_depth = None
            transparent_mask = None

        else:
            depth_inference = preds["depth"]
            pred_depth_conf = torch.ones_like(depth_inference) if len(preds["depth_conf"]) > 0 else preds["depth_conf"]
            extra_depth = None
            transparent_mask = None
            
        if len(preds["pose_enc_list"]) == 0:
            preds["pose_enc_list"] = torch.zeros_like(views["camera_pose"])
            
        if output_normalize:
            extrinsics_dist_mean = torch.norm(preds['extrinsics'][:, :, :3, 3], dim=-1).mean(dim=1)
            extrinsics_gt_dist_mean = torch.norm(extrinsics[:, :, :3, 3], dim=-1).mean(dim=1)
            pose_scales = extrinsics_gt_dist_mean / (extrinsics_dist_mean + 1e-8)

            depth_inference = depth_inference * rearrange(pose_scales, "b -> b () () ()")
            preds["pose_enc_list"][:, :, :3] = preds["pose_enc_list"][:, :, :3] * rearrange(pose_scales, "b -> b () () ()")

        if (gt_cam_output is False):
            depth_inference = depth_inference
        else:
            camera_pose = views["camera_pose"]
            pose_enc = extri_intri_to_pose_encoding(
                camera_pose,
                views["camera_intrinsics"],
                views["img"].shape[-2:],
            )
            preds["pose_enc_list"] = pose_enc

            gt_depth = views["depthmap"]
            valid_mask = views["valid_mask"]

            gt_depth_valid = gt_depth[valid_mask]
            pred_depth_valid = depth_inference[valid_mask]

            with torch.enable_grad():
                depth_aligned, scale, shift = depth_alignment_lad2(
                    pred_depth_valid.unsqueeze(0), gt_depth_valid.unsqueeze(0)
                )

            depth_inference = depth_inference * scale + shift

            preds["depth_inference"] = depth_inference
            preds["depth"] = [item * scale + shift for item in preds["depth"]]

            # Apply same alignment to the extra (through-glass) depth layer
            if extra_depth is not None:
                extra_depth = extra_depth * scale + shift
                preds["extra_depth"] = extra_depth

        # --- Sky processing: push sky-region depth far behind the scene ---
        sky_mask = None
        if use_sky_mask:
            if "sky_mask" in preds:
                sky_mask = preds["sky_mask"] > 0.5  # (B, N, H, W) bool
            elif "mog_weight_full" in preds:
                mwf = preds["mog_weight_full"]  # (B, N, H, W, L); last component is sky
                sky_mask = mwf.argmax(dim=-1) == (mwf.shape[-1] - 1)  # (B, N, H, W) bool

            if sky_mask is not None:
                sky_mask = sky_mask.to(depth_inference.device)
                # Set sky pixels to 2x the per-sequence max valid (non-sky) depth, falling
                # back to the per-sequence global max when the whole sequence is sky.
                bb = depth_inference.shape[0]
                depth_flat = depth_inference.reshape(bb, -1)
                mask_flat = sky_mask.reshape(bb, -1)
                max_valid = depth_flat.masked_fill(mask_flat, float("-inf")).max(dim=-1).values
                all_sky = torch.isinf(max_valid)
                max_valid = torch.where(all_sky, depth_flat.max(dim=-1).values, max_valid)
                sky_fill = (2.0 * max_valid)[:, None, None, None]  # (B, 1, 1, 1)
                depth_inference = torch.where(sky_mask, sky_fill, depth_inference)
                preds["depth_inference"] = depth_inference

        pose_enc_list = preds["pose_enc_list"][:, :len_views] if isinstance(preds["pose_enc_list"], torch.Tensor) else None
        predictions_new = {
            'depth': depth_inference[:, :len_views],
            'depth_conf': pred_depth_conf[:, :len_views],
            'pose_enc': pose_enc_list,
            'images': views["img"][:, :len_views],
            'raw_preds': preds,
            'views': views,
            # Multilayer outputs (None for single-layer models)
            'extra_depth': extra_depth[:, :len_views] if extra_depth is not None else None,
            'transparent_mask': transparent_mask[:, :len_views] if transparent_mask is not None else None,
            # Sky mask (None unless use_sky_mask=True and the model exposes sky info)
            'sky_mask': sky_mask[:, :len_views] if sky_mask is not None else None,
        }
        return predictions_new


class DA3WrapperNested(DA3Wrapper):
    def __init__(self, config_path, config_path_nested, **kwargs):
        super().__init__(
            config_path=config_path,
            **kwargs,
        )
        if config_path_nested.endswith('.json'):
            with open(config_path_nested, 'r') as f:
                config = json.load(f)['config']
        
        else:
            assert config_path_nested.endswith('.yaml'), 'config_path must be a yaml file'
            config = load_config(config_path_nested)
                
        self.net_nested = create_object(config)
        
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
    def inference(self, *args, **kwargs):
        out = super().inference(*args, **kwargs)
        out_nested = self.net_nested(
            out['views']['img'],
            extrinsics=out['views']['camera_extrinsics'],
            intrinsics=out['views']['camera_intrinsics'],
            infer_gs=False,
        )
        non_sky_mask = out_nested.sky < 0.3
        out['non_sky_mask'] = non_sky_mask
        out['sky_mask'] = (~non_sky_mask)

        return out


def dump_model_parameters(
    model: nn.Module,
    filepath: str = "model_parameters.txt",
    sort_by_name: bool = True,
):
    rows = []
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel

        rows.append((
            name,
            tuple(param.shape),
            param.requires_grad,
            numel,
        ))

    if sort_by_name:
        rows.sort(key=lambda x: x[0])

    with open(filepath, "w") as f:
        f.write("Model parameter summary\n")
        f.write("=" * 80 + "\n")
        f.write(
            f"{'Name':50s} {'Shape':20s} {'Grad':6s} {'#Params':>12s}\n"
        )
        f.write("-" * 80 + "\n")

        for name, shape, req_grad, numel in rows:
            f.write(
                f"{name:50s} {str(shape):20s} "
                f"{str(req_grad):6s} {numel:12d}\n"
            )

        f.write("-" * 80 + "\n")
        f.write(f"Total parameters     : {total_params:,}\n")
        f.write(f"Trainable parameters : {trainable_params:,}\n")
        f.write(f"Frozen parameters    : {total_params - trainable_params:,}\n")

    print(f"[✓] Parameter summary written to: {filepath}")


def depth_alignment_lstsq(
    predicted_depth_original,
    ground_truth_depth_original,
    align_with_lstsq=True,
    use_gpu=True
):
    """
    Evaluate the depth map using various metrics and return a depth error parity map, with an option for least squares alignment.

    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map.
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map.
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.

    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)

    # if the dimension is 3, flatten to 2d along the batch dimension
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    # put to device
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()

    predicted_depth = predicted_depth_original
    ground_truth_depth = ground_truth_depth_original
    
    # Convert to numpy for lstsq
    predicted_depth_np = predicted_depth_original.cpu().numpy().reshape(-1, 1)
    ground_truth_depth_np = ground_truth_depth_original.cpu().numpy().reshape(-1, 1)

    # Add a column of ones for the shift term
    A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])

    # Solve for scale (s) and shift (t) using least squares
    result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
    s, t = result[0][0], result[0][1]

    # convert to torch tensor
    s = torch.tensor(s, device=predicted_depth_original.device)
    t = torch.tensor(t, device=predicted_depth_original.device)

    # Apply scale and shift
    predicted_depth = s * predicted_depth + t
    return predicted_depth, s, t


def depth_alignment_lad2(
    predicted_depth_original,
    ground_truth_depth_original
):
    s_init = (
        torch.median(ground_truth_depth_original) / torch.median(predicted_depth_original)
    ).item()
    s, t = absolute_value_scaling2(
        predicted_depth_original,
        ground_truth_depth_original,
        s_init=s_init,
        lr=1e-4,
        max_iters=1000,
    )
    predicted_depth = s * predicted_depth_original + t
    return predicted_depth, s, t
    