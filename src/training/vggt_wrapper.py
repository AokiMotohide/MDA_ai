import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as F
import sys
from torch import nn

from src.stream3r.stream3r import STream3R
from src.stream3r.stream3r_mog import STream3RMog
from src.depth_anything_3.utils.utils_training import prepare_inputs
from src.depth_anything_3.model.mog_inference import find_gmm_mode_gpu_chunk
from einops import rearrange
from src.depth_anything_3.utils.geometry import affine_inverse
from src.training.da3_wrapper import depth_alignment_lstsq, depth_alignment_lad2
from src.depth_anything_3.model.utils.transform import extri_intri_to_pose_encoding


class VGGTWrapper(nn.Module):
    def __init__(self, config_path=None, is_mog=False, is_moe=False, **kwargs):
        super(VGGTWrapper, self).__init__()
        self.is_mog = is_mog
        self.pretrain_as_possible = kwargs.get('pretrain_as_possible', False)

        if is_mog:
            self.net = STream3RMog()
        else:
            self.net = STream3R()
        
        gs_names = ['head_mog']
        
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
        
        if 'loss_type' in kwargs:
            self.loss_type = kwargs['loss_type']
        else:
            self.loss_type = None
    
    
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
        if hasattr(self.net, "aggregator"):
            backbone_model = self.net.aggregator

        # --- Backbone Freeze Logic ---
        if "backbone" in modes:
            print("  Unfreezing full backbone")
            for param in backbone_model.parameters():
                param.requires_grad = True
        else:
            # selective backbone unfreezing
            if "global" in modes:
                for param in backbone_model.global_blocks.parameters():
                    param.requires_grad = True

            if "local" in modes:
                for param in backbone_model.frame_blocks.parameters():
                    param.requires_grad = True

        # --- Camera Freeze Logic ---
        unfreeze_cam_dec = "cam_dec" in modes

        if unfreeze_cam_dec:
            print("  Unfreezing camera decoder")
            if hasattr(self.net, "camera_head") and self.net.camera_head is not None:
                for param in self.net.camera_head.parameters():
                    param.requires_grad = True

        if "head" in modes:
            print("  Unfreezing depth head")
            if hasattr(self.net, "depth_head_mog") and self.net.depth_head_mog is not None:
                for param in self.net.depth_head_mog.parameters():
                    param.requires_grad = True

        assert hasattr(self.net, "camera_head") and self.net.camera_head is not None
        assert hasattr(self.net, "depth_head_mog") and self.net.depth_head_mog is not None
        
        if 'lastlayer' in modes:
            print("  Unfreezing last layer")
            assert hasattr(self.net, "depth_head_mog") and self.net.depth_head_mog is not None
            for model_module in self.net.depth_head_mog.scratch.output_conv2_list:
                for param in model_module.parameters():
                    param.requires_grad = True
        
        # Log trainable parameters count
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        total_params_count = sum(p.numel() for p in trainable_params)
        print(f"Number of trainable parameters (tensors): {len(trainable_params)}")
        print(f"Total number of trainable parameters (elements): {total_params_count / 1e6:.2f} M")

        trainable_names = [n for n, p in self.named_parameters() if p.requires_grad]
        if len(trainable_names) > 0:
            print(f"Example trainable param: {trainable_names}")
    
    def forward(self, x, **kwargs):
        # print('x', x.shape, x.min(), x.max())
        std0 = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(3, 1, 1)
        mean0 = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(3, 1, 1)
        x = x * std0 + mean0 # 0 - 1
        x = x * 2 - 1
        # print('x', x.shape, x.min(), x.max())

        predictions = self.net(x, mode="full")
        
        if isinstance(predictions['depth'], list):
            predictions['depth'] = [item.squeeze(-1) for item in predictions['depth']]
        else:
            predictions['depth'] = predictions['depth'].squeeze(-1)
        
        predictions_new = {
            'depth': predictions['depth'],
            'depth_conf': predictions['depth_conf'],
            'extrinsics': predictions['extrinsics'],
            'intrinsics': predictions['intrinsics'],
            'pose_enc_list': predictions['pose_enc_list'],
        }
        
        if self.is_mog:
            predictions_new['mog_weight'] = predictions['mog_weight']
        
        return predictions_new

    def inference(self, views, device,
                  cam_inp=False,
                  gt_cam_output=True,
                  output_normalize=False,
                  loss_type=None,
                  **kwargs):
        
        if loss_type is None:
            loss_type = self.loss_type if self.loss_type is not None else 'l1'
        is_mog = True if hasattr(self.net, 'depth_head_mog') and (self.net.depth_head_mog is not None) else False
        
        camera_pose0 = views[0]["camera_pose"]
        camera_pose0_inv = affine_inverse(camera_pose0)
        for view in views:
            view['camera_pose'] = torch.einsum("bij,bjk->bik", camera_pose0_inv, view['camera_pose'])
        
        len_views = len(views)
                
        views = prepare_inputs(views, device, new_gs_mask=False, rand_shuffle=False)
        
        x = views["img"]
        std0 = torch.tensor([0.229, 0.224, 0.225], device=x.device).reshape(3, 1, 1)
        mean0 = torch.tensor([0.485, 0.456, 0.406], device=x.device).reshape(3, 1, 1)
        x = x * std0 + mean0 # 0 - 1
        x = x * 2 - 1

        with torch.no_grad():
            preds = self.net(x, mode="full")
            if isinstance(preds['depth'], list):
                preds['depth'] = [item.squeeze(-1) for item in preds['depth']]
            else:
                preds['depth'] = preds['depth'].squeeze(-1)
            
        if is_mog:
            depth_inference, indices = find_gmm_mode_gpu_chunk(preds, loss_type=loss_type, 
                                                            saved_dir=None, 
                                                            chunk_size=16)
            preds['indices'] = indices
            pred_depth_conf = preds["depth_conf"][0] ################## TOD
            preds['depth_inference'] = depth_inference
            
        elif len(preds["depth_conf"]) > 0:
            depth_inference = preds["depth"]
            pred_depth_conf = preds["depth_conf"]
        
        else:
            depth_inference = preds["depth"]
            pred_depth_conf = torch.ones_like(depth_inference)
            
        if output_normalize:
            extrinsics = views["camera_extrinsics"]
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
                _, scale, shift = depth_alignment_lad2(
                    pred_depth_valid.unsqueeze(0), gt_depth_valid.unsqueeze(0)
                )

            depth_inference = depth_inference * scale + shift

            preds["depth_inference"] = depth_inference
            preds["depth"] = [item * scale + shift for item in preds["depth"]]

        predictions_new = {
            'depth': depth_inference[:, :len_views],
            'depth_conf': pred_depth_conf[:, :len_views],
            'pose_enc': preds["pose_enc_list"][:, :len_views],
            'images': views["img"][:, :len_views],
            'raw_preds': preds,
            'views': views,
        }
        return predictions_new