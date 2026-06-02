# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn
from addict import Dict
from omegaconf import DictConfig, OmegaConf

from depth_anything_3.cfg import create_object
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from depth_anything_3.utils.alignment import (
    apply_metric_scaling,
    compute_alignment_mask,
    compute_sky_mask,
    least_squares_scale_scalar,
    sample_tensor_for_quantile,
    set_sky_regions_to_max_depth,
)
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, map_pdf_to_opacity
from einops import rearrange, repeat

import torch.utils.checkpoint as checkpoint
import random

def _wrap_cfg(cfg_obj):
    return OmegaConf.create(cfg_obj)


class DepthAnything3NetCamMog(nn.Module):
    """
    Depth Anything 3 network for depth estimation and camera pose estimation.

    This network consists of:
    - Backbone: DinoV2 feature extractor
    - Head: DPT or DualDPT for depth prediction
    - Optional camera decoders for pose estimation
    - Optional GSDPT for 3DGS prediction

    Args:
        preset: Configuration preset containing network dimensions and settings

    Returns:
        Dictionary containing:
        - depth: Predicted depth map (B, H, W)
        - depth_conf: Depth confidence map (B, H, W)
        - extrinsics: Camera extrinsics (B, N, 4, 4)
        - intrinsics: Camera intrinsics (B, N, 3, 3)
        - gaussians: 3D Gaussian Splats (world space), type: model.gs_adapter.Gaussians
        - aux: Auxiliary features for specified layers
    """

    # Patch size for feature extraction
    PATCH_SIZE = 14

    def __init__(self, net, head, head_mog, cam_dec=None, cam_enc=None, gs_head=None, gs_adapter=None, filter_percentile_in_gs=None, opt_cam_dec=True, remove_cam=False, debug=False):
        """
        Initialize DepthAnything3Net with given yaml-initialized configuration.
        """
        super().__init__()
        self.backbone = net if isinstance(net, nn.Module) else create_object(_wrap_cfg(net))
        self.head = head if isinstance(head, nn.Module) else create_object(_wrap_cfg(head))
        self.head_mog = head_mog if isinstance(head_mog, nn.Module) else create_object(_wrap_cfg(head_mog))
        self.cam_dec, self.cam_enc = None, None
        if cam_dec is not None:
            self.cam_dec = (
                cam_dec if isinstance(cam_dec, nn.Module) else create_object(_wrap_cfg(cam_dec))
            )
            self.cam_enc = (
                cam_dec if isinstance(cam_enc, nn.Module) else create_object(_wrap_cfg(cam_enc))
            )
        self.gs_adapter, self.gs_head = None, None
        
        self.filter_percentile_in_gs = filter_percentile_in_gs
        self.infunc_rendering = False
        self.opt_cam_dec = opt_cam_dec
        self.remove_cam = remove_cam
        self.debug = debug

    def forward(
        self,
        x: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = [],
        infer_gs: bool = False,
        backbone_nograde: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input images (B, N, 3, H, W)
            extrinsics: Camera extrinsics (B, N, 4, 4) - unused
            intrinsics: Camera intrinsics (B, N, 3, 3) - unused
            feat_layers: List of layer indices to extract features from

        Returns:
            Dictionary containing predictions and auxiliary features
        """
        if self.remove_cam:
            extrinsics = None
            intrinsics = None
        elif self.training and random.random() < 0.8:
            extrinsics = None
            intrinsics = None
        elif self.training and random.random() < 0.2 and extrinsics is not None:
            extrinsics[:, :, :3, 3] = extrinsics[:, :, :3, 3] * random.uniform(0.01, 10)
            
        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None

        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            feats, _ = self.backbone(
                x, cam_token=cam_token, export_feat_layers=export_feat_layers
            )
            
        # feats = [[item for item in feat] for feat in feats]
        H, W = x.shape[-2], x.shape[-1]

        with torch.autocast(device_type=x.device.type, enabled=False):
            # Process features through depth head
            output = self._process_depth_head(feats, H, W)
            output = self._process_camera_estimation(feats, H, W, output)
            with torch.no_grad():
                if infer_gs:
                    output = self._process_gs_head(feats, H, W, output, x, extrinsics, intrinsics)
            
            # assert self.debug
            if self.debug:
                with torch.no_grad():
                    pretrained_output = self.head(feats, H, W, patch_start_idx=0, chunk_size=8, use_checkpoint=True)
                    output.pretrained_output = pretrained_output['depth']

        return output

    def _process_depth_head(
        self, feats: list[torch.Tensor], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Process features through the depth prediction head."""
        return self.head_mog(feats, H, W, patch_start_idx=0, chunk_size=4, use_checkpoint=True)

    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
        if self.cam_dec is not None:
            pose_enc = self.cam_dec(feats[-1][1])
            
            # Convert pose encoding to extrinsics and intrinsics
            output.pose_enc_list = pose_enc
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt

        return output

    def _process_gs_head(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        output: Dict[str, torch.Tensor],
        in_images: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Process 3DGS parameters estimation if 3DGS head is available."""
        if self.gs_head is None or self.gs_adapter is None:
            return output
        assert output.get("depth", None) is not None, "must provide MV depth for the GS head."

        # if GT camera poses are provided, use them
        # if extrinsics is not None and intrinsics is not None:
        if False:
            ctx_extr = extrinsics
            ctx_intr = intrinsics
        else:
            ctx_extr = output.get("extrinsics", None)
            ctx_intr = output.get("intrinsics", None)
            assert (
                ctx_extr is not None and ctx_intr is not None
            ), "must process camera info first if GT is not available"
        # print('intrinsics', intrinsics, ctx_intr)
        gt_extr = extrinsics
        # homo the extr if needed
        ctx_extr = as_homogeneous(ctx_extr)
        if gt_extr is not None:
            gt_extr = as_homogeneous(gt_extr)

        # forward through the gs_dpt head to get 'camera space' parameters
        gs_outs = self.gs_head(
            feats=feats,
            H=H,
            W=W,
            patch_start_idx=0,
            images=in_images,
        )
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf

        # convert to 'world space' 3DGS parameters; ready to export and render
        # gt_extr could be None, and will be used to align the pose scale if available
        # print('ctx_extr', ctx_extr)
        # print('ctx_intr', ctx_intr)
        gs_world = self.gs_adapter(
            extrinsics=ctx_extr,
            intrinsics=ctx_intr,
            depths=output.depth,
            opacities=map_pdf_to_opacity(densities),
            raw_gaussians=raw_gaussians,
            image_shape=(H, W),
            gt_extrinsics=gt_extr,
            in_images=in_images,
            depth_conf=output.depth_conf,
            filter_percentile_in_gs=self.filter_percentile_in_gs,
            gt_intrinsics=intrinsics
        )
        output.gaussians = gs_world
        output.raw_gaussians = raw_gaussians

        return output

    def _extract_auxiliary_features(
        self, feats: list[torch.Tensor], feat_layers: list[int], H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Extract auxiliary features from specified layers."""
        aux_features = Dict()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):
            # Reshape features to spatial dimensions
            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features

