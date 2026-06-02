from copy import copy, deepcopy
import numpy as np

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.training.utils.loss_utils import (
    filter_by_quantile,
    conf_loss_func,
    calculate_mean_pose_scale_from_depths,
    calculate_pose_scale_from_depths,
    gradient_loss_multi_scale_points,
)
from einops import rearrange
from src.depth_anything_3.utils.geometry import generate_raymaps, generate_raymaps_preds
from src.depth_anything_3.model.utils.transform import extri_intri_to_pose_encoding


class MultiLoss(nn.Module):
    """Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class MultiLossMultiSingleBase(MultiLoss):
    """Dataset-aware dispatch base.

    Splits a batch by ``gts['dataset'][i][0]`` into four groups and routes each
    group to a different ``compute_*_loss`` method, then length-weights the
    aggregate. Subclasses implement the four ``compute_*_loss`` slots; the
    default here is ``pass`` (which surfaces as a TypeError on unpack if a
    sample is ever routed to an unimplemented branch).
    """

    def __init__(self):
        super().__init__()
        self.single_dataset_names = ["urbansyn", "irs"]
        self.layered_dataset_names = ["layered_depth"]
        self.seg_dataset_names = ["ade20k_glass", "hsod_glass", "trans10k_glass"]
        self.monodepth = False  # baked; arm kept verbatim in `_route_group`

    def _route_group(self, dataset_name):
        """Group index (0=multi, 1=single, 2=layered, 3=seg) for a sample.

        Single source of truth shared by ``split_gts_and_preds`` and
        ``concat_dict``. The ``elif self.monodepth`` arm is dead with
        ``monodepth=False`` baked, but preserved verbatim for parity.
        """
        if dataset_name in self.single_dataset_names:
            return 1
        if dataset_name in self.layered_dataset_names:
            return 2
        if dataset_name in self.seg_dataset_names:
            return 3
        if self.monodepth:
            return 1
        return 0

    # ------------------------------------------------------------------
    # Dispatch helpers (ported from losses_mog.py:817-1000).
    # ------------------------------------------------------------------

    def combine_dict(self, list_of_dicts):
        if len(list_of_dicts) == 0:
            return {"depthmap": []}
        ks = list(list_of_dicts[0].keys())
        combined_dict = {k: [] for k in ks}
        for item in list_of_dicts:
            for k in ks:
                combined_dict[k].append(item[k])

        for k in ks:
            if k == "depth" or k == "depth_conf":
                num_elements = len(combined_dict[k][0])
                combined_dict[k] = [
                    torch.stack([item[i] for item in combined_dict[k]], dim=0)
                    for i in range(num_elements)
                ]
            elif isinstance(combined_dict[k][0], torch.Tensor):
                combined_dict[k] = torch.stack(combined_dict[k], dim=0)
            else:
                combined_dict[k] = combined_dict[k]

        return combined_dict

    def concat_dict(self, gts, preds, preds1, preds2, preds3, preds4):
        num_heads = len(preds["depth"])
        groups = (preds1, preds2, preds3, preds4)

        # For each sample (in original batch order): which group's preds dict
        # holds it, and at what index within that group.
        counters = [0, 0, 0, 0]
        routes = []
        for ds_entry in gts["dataset"]:
            g = self._route_group(ds_entry[0])
            routes.append((groups[g], counters[g]))
            counters[g] += 1

        # Per-head list-of-tensors keys.
        depth_lists = [[] for _ in range(num_heads)]
        conf_lists = [[] for _ in range(num_heads)]
        for current, ci in routes:
            if "depth" in current:
                for h in range(num_heads):
                    depth_lists[h].append(current["depth"][h][ci : ci + 1])
                    if "depth_conf" in current:
                        conf_lists[h].append(current["depth_conf"][h][ci : ci + 1])

        # Optional single-tensor keys.
        optional_keys = ("indices", "depth_inference", "tangent_weight", "depth_pretrained", "per_pixel_conf_loss")
        optional_acc = {k: [] for k in optional_keys}
        seg_acc = []
        for current, ci in routes:
            for k in optional_keys:
                if k in current:
                    optional_acc[k].append(current[k][ci : ci + 1])
            seg = current.get("seg_logits")
            if seg is not None and seg[0] is not None:
                seg_acc.append(seg[ci : ci + 1])

        new_dict = {"depth": [torch.cat(h, dim=0) for h in depth_lists]}
        new_dict["depth_conf"] = (
            [torch.cat(h, dim=0) for h in conf_lists] if conf_lists[0] else []
        )
        for k in optional_keys:
            if optional_acc[k]:
                new_dict[k] = torch.cat(optional_acc[k], dim=0)
        if seg_acc:
            new_dict["seg_logits"] = torch.cat(seg_acc, dim=0)
        return new_dict

    def split_dict(self, mydict, idx):
        new_dict = {}
        for k, v in mydict.items():
            if k == "depth" or k == "depth_conf":
                new_dict[k] = [item[idx] for item in v]
            elif isinstance(v, torch.Tensor) or isinstance(v, list):
                if k == "idx":
                    continue
                new_dict[k] = v[idx]
            else:
                new_dict[k] = v
        return new_dict

    def split_gts_and_preds(self, gts, preds):
        gt_groups = [[], [], [], []]
        pred_groups = [[], [], [], []]
        for i in range(gts["depthmap"].shape[0]):
            g = self._route_group(gts["dataset"][i][0])
            gt_groups[g].append(self.split_dict(gts, i))
            pred_groups[g].append(self.split_dict(preds, i))

        def safe_combine(lst):
            return self.combine_dict(lst) if lst else {"depthmap": []}

        return (
            safe_combine(gt_groups[0]),
            safe_combine(gt_groups[1]),
            safe_combine(gt_groups[2]),
            safe_combine(gt_groups[3]),
            safe_combine(pred_groups[0]),
            safe_combine(pred_groups[1]),
            safe_combine(pred_groups[2]),
            safe_combine(pred_groups[3]),
        )

    def compute_loss(self, gts, preds, **kw):
        gts1, gts2, gts3, gts4, preds1, preds2, preds3, preds4 = self.split_gts_and_preds(
            gts, preds
        )

        if len(gts1["depthmap"]) > 0:
            loss1, details1 = self.compute_multi_loss(gts1, preds1, **kw)
            length1 = len(gts1["depthmap"])
        else:
            loss1, details1 = 0.0, {}
            length1 = 0
        if len(gts2["depthmap"]) > 0:
            loss2, details2 = self.compute_singleview_loss(gts2, preds2, **kw)
            length2 = len(gts2["depthmap"])
        else:
            loss2, details2 = 0.0, {}
            length2 = 0
        if len(gts3["depthmap"]) > 0:
            loss3, details3 = self.compute_layereddepth_loss(gts3, preds3, **kw)
            length3 = len(gts3["depthmap"])
        else:
            loss3, details3 = 0.0, {}
            length3 = 0
        if len(gts4["depthmap"]) > 0:
            loss4, details4 = self.compute_segmentation_loss(gts4, preds4, **kw)
            length4 = len(gts4["depthmap"])
        else:
            loss4, details4 = 0.0, {}
            length4 = 0

        total_loss = (
            loss1 * length1 + loss2 * length2 + loss3 * length3 + loss4 * length4
        ) / (length1 + length2 + length3 + length4)

        details = {}
        all_keys = set(
            list(details1.keys())
            + list(details2.keys())
            + list(details3.keys())
            + list(details4.keys())
        )
        for key in all_keys:
            val1 = details1.get(key, 0.0)
            val2 = details2.get(key, 0.0)
            val3 = details3.get(key, 0.0)
            val4 = details4.get(key, 0.0)
            details[key] = (
                val1 * length1 + val2 * length2 + val3 * length3 + val4 * length4
            ) / (length1 + length2 + length3 + length4)

        self_name = self.self_name
        details[self_name + "_length1" + "/00"] = float(length1)
        details[self_name + "_length2" + "/00"] = float(length2)
        details[self_name + "_length3" + "/00"] = float(length3)
        details[self_name + "_length4" + "/00"] = float(length4)

        preds_new = self.concat_dict(gts, preds, preds1, preds2, preds3, preds4)
        preds["depth"] = preds_new["depth"]
        preds["indices"] = preds_new["indices"]
        for key in ("depth_inference", "tangent_weight", "seg_logits", "depth_pretrained", "per_pixel_conf_loss"):
            if key in preds_new:
                preds[key] = preds_new[key]

        # Strict access to match the original (`losses_mog.py:1055`). Fixtures
        # must supply `depth_complete`; the equivalence test does.
        gts["depthmap_sampling"] = gts["depth_complete"]
        return total_loss, details

    # ------------------------------------------------------------------
    # Four compute_*_loss slots. Subclasses override; default is `pass`.
    # ------------------------------------------------------------------

    def compute_multi_loss(self, gts, preds, **kw):
        pass

    def compute_singleview_loss(self, gts, preds, **kw):
        pass

    def compute_layereddepth_loss(self, gts, preds, **kw):
        pass

    def compute_segmentation_loss(self, gts, preds, **kw):
        pass


class MultiLossOnlyDepthMogMultiSingleCamMogSkymask(MultiLossMultiSingleBase):
    """Sky-aware MOG depth + camera + 3D-points loss.

    Flat re-implementation of the class defined at
    ``losses_cam_mog.py:502``. 24 of the original 30 init flags are baked at
    their defaults; the remaining 6 are kwargs here.
    """

    def __init__(
        self,
        is_metric: bool = False,
        gradient_loss: str = "grad",
        loss_type: str = "l1",
        use_grad_loss: bool = False,
        clamp_logp: bool = False,
        detach_depth_scale_in_ptsloss: bool = False,
        no_conf: bool = False,
        use_sky_loss: bool = True,
        median_scale: bool = True,
    ):
        super().__init__()
        self.gradient_loss = gradient_loss
        self.loss_type = loss_type
        self.use_grad_loss = use_grad_loss
        self.clamp_logp = clamp_logp
        self.detach_depth_scale_in_ptsloss = detach_depth_scale_in_ptsloss
        self.no_conf = no_conf
        self.use_sky_loss = use_sky_loss
        self.median_scale = median_scale
        
        self.alpha = 0.2
        self.self_name = (
            type(self)
            .__name__
            .replace("OnlyDepthMogMultiSingle", "")
            .replace("RayDepthMog", "")
        )

    def get_name(self):
        return "CausalLoss"

    # ------------------------------------------------------------------
    # MOG / depth helpers
    # ------------------------------------------------------------------

    def get_depth_std(self, gt_depth, valid_mask_depth):
        gt_depth_flatten = gt_depth.reshape(gt_depth.shape[0], -1)
        valid_mask_flatten = valid_mask_depth.reshape(valid_mask_depth.shape[0], -1)
        valid_mask_flatten_mean = valid_mask_flatten.float().mean(dim=1, keepdim=True)
        valid_mask_flatten_mean = torch.clamp(valid_mask_flatten_mean, min=1e-8)

        gt_depth_flatten = gt_depth_flatten * valid_mask_flatten
        gt_depth_flatten_std = (
            torch.abs(gt_depth_flatten).mean(dim=1, keepdim=True) / valid_mask_flatten_mean
        )
        return gt_depth_flatten_std

    def per_mog_depth_regression_loss_l1(
        self, gt_depth, pred_depth, gt_depth_flatten_std, pred_depth_conf, valid_mask_depth
    ):
        # `pi3_scale=False` baked → no rescaling; no-op reshape removed.
        gt_depth = torch.clamp(gt_depth, min=1e-8)
        depth_diff = torch.abs(gt_depth - pred_depth)
        bb, ss, hh, ww = gt_depth.shape

        conf_loss_0 = conf_loss_func(
            depth_diff.reshape(bb, ss, hh, ww, 1),
            pred_depth_conf.unsqueeze(-1),
            valid_mask_depth.unsqueeze(-1),
            gamma=1.0,
            alpha=self.alpha,
            avg=False,
        )
        conf_loss_0 = conf_loss_0 / self.alpha + np.log(2) + np.log(self.alpha)
        return conf_loss_0

    def per_mog_depth_regression_loss_l2(
        self, gt_depth, pred_depth, gt_depth_flatten_std, pred_depth_conf, valid_mask_depth
    ):
        depth_diff = torch.square(gt_depth - pred_depth)
        bb, ss, hh, ww = gt_depth.shape

        conf_loss_0 = conf_loss_func(
            depth_diff.reshape(bb, ss, hh, ww, 1),
            pred_depth_conf.unsqueeze(-1),
            valid_mask_depth.unsqueeze(-1),
            gamma=1.0,
            alpha=self.alpha,
            avg=False,
        )
        conf_loss_0 = conf_loss_0 / self.alpha + np.log(2 * np.pi) + np.log(self.alpha)
        conf_loss_0 = conf_loss_0 * 0.5
        return conf_loss_0

    def per_mog_point_regression_loss_l1(
        self, gt_pts, pred_pts, gt_pts_flatten_std, pred_pts_conf, valid_mask_pts
    ):
        # `if False:` branch removed; only the L1-distance branch remains.
        pts_diff = torch.norm(gt_pts - pred_pts, dim=-1)
        bb, ss, hh, ww, _ = gt_pts.shape
        conf_loss_0 = pts_diff.reshape(bb, ss, hh, ww, 1) * valid_mask_pts.unsqueeze(-1)
        return conf_loss_0

    def get_depth_inference(
        self,
        pred_depth_list,
        pred_depth_conf_list,
        mog_weight_list,
        gt_depth_flatten_std,
        valid_mask_depth,
        preds,
    ):
        bb, ss, hh, ww = pred_depth_list[0].shape
        all_nll = []
        for pred_depth_mean in pred_depth_list:
            pred_depth_conf_list_detach = [item.detach() for item in pred_depth_conf_list]
            pred_depth_list_detach = [item.detach() for item in pred_depth_list]
            nll, _ = self.get_nll(
                pred_depth_mean.clone().detach(),
                pred_depth_list_detach,
                gt_depth_flatten_std,
                pred_depth_conf_list_detach,
                torch.ones_like(valid_mask_depth),
                mog_weight_list,
            )
            all_nll.append(nll)

        all_nll = torch.stack(all_nll, dim=-1)
        max_indices = torch.argmin(all_nll, dim=-1, keepdim=True)
        pred_depth_list_stack = torch.stack(pred_depth_list, dim=-1)
        pred_depth_max = pred_depth_list_stack.gather(dim=-1, index=max_indices)
        pred_conf_list_stack = torch.stack(pred_depth_conf_list, dim=-1)
        pred_conf_max = pred_conf_list_stack.gather(dim=-1, index=max_indices)
        pred_conf_max = pred_conf_max.reshape(bb, ss, hh, ww)
        preds["depth_inference"] = pred_depth_max.reshape(bb, ss, hh, ww)
        preds["indices"] = max_indices.reshape(bb, ss, hh, ww)
        preds["pred_conf_max"] = pred_conf_max

        ll = torch.softmax(mog_weight_list, dim=-1)
        pred_depth_avg = (ll * pred_depth_list_stack).sum(dim=-1)
        pred_depth_avg = pred_depth_avg.reshape(bb, ss, hh, ww)
        preds["depth_inference_avg"] = pred_depth_avg
        return pred_depth_max, pred_conf_max, max_indices, all_nll

    def get_nll(
        self,
        gt_depth,
        pred_depth_list,
        gt_depth_flatten_std,
        pred_depth_conf_list,
        valid_mask_depth,
        mog_weight_list,
        clamp_logp=True,
    ):
        per_mog_depth_regression_loss = (
            self.per_mog_depth_regression_loss_l1
            if self.loss_type == "l1"
            else self.per_mog_depth_regression_loss_l2
        )

        if "log" in self.loss_type:
            gt_depth = torch.log(torch.clamp(gt_depth + 0.1, min=1e-6))
            pred_depth_list = [torch.log(item + 0.1) for item in pred_depth_list]

        conf_loss_list = []
        for i in range(len(pred_depth_list)):
            pred_depth = pred_depth_list[i]
            pred_depth_conf = pred_depth_conf_list[i]
            depth_loss_i = per_mog_depth_regression_loss(
                gt_depth,
                pred_depth,
                gt_depth_flatten_std,
                pred_depth_conf,
                valid_mask_depth,
            )
            conf_loss_list.append(depth_loss_i)

        conf_loss_list = torch.cat(conf_loss_list, dim=-1)
        reg_loss = 0.0
        if self.clamp_logp and clamp_logp:
            reg_loss = torch.square(mog_weight_list).mean() * 1e-3
            # `clamp_logp_value=4` baked.
            mog_weight_list = (
                mog_weight_list
                + torch.clamp(mog_weight_list.detach(), min=-4)
                - mog_weight_list.detach()
            )

        conf_loss = -torch.logsumexp(
            conf_loss_list * (-1) + mog_weight_list[..., : conf_loss_list.shape[-1]],
            dim=-1,
        )
        conf_loss = conf_loss + reg_loss
        return conf_loss, conf_loss_list

    def conf_loss_single(self, gts, preds, pose_scale):
        gt_depth = gts["depthmap"]
        pred_depth_list = preds["depth"]
        pred_depth_conf_list = preds["depth_conf"]
        mog_weight_list = preds["mog_weight"]
        valid_mask_depth = gts["valid_mask"]

        if len(pose_scale.shape) == 1:
            pose_scale = rearrange(pose_scale.detach(), "b -> b () () ()")
        elif len(pose_scale.shape) == 2:
            pose_scale = rearrange(pose_scale.detach(), "b s -> b s () ()")
        else:
            raise ValueError(f"Invalid pose scale shape: {pose_scale.shape}")
        pose_scale_inv = 1.0 / (pose_scale + 1e-8)
        gt_depth = gt_depth * pose_scale_inv

        conf_loss, conf_loss_list = self.get_nll(
            gt_depth,
            pred_depth_list,
            None,
            pred_depth_conf_list,
            valid_mask_depth,
            mog_weight_list,
        )
        
        max_mog_weight = torch.softmax(mog_weight_list, dim=-1).max(dim=-1).values
        # max_mog_weight = max_mog_weight[ gts["valid_mask_used"] ]
        assert max_mog_weight.shape == conf_loss.shape
        
        # if self.no_conf:
        #     # print('max_mog_weight', max_mog_weight.max(), max_mog_weight.min(), max_mog_weight.mean())
        #     conf_loss = conf_loss / (max_mog_weight.detach()**3)
        
        # if self.no_conf:
        #     if 'l1' in self.loss_type:
        #         conf_loss = conf_loss / preds['pred_conf_max'].detach()
        #     elif 'l2' in self.loss_type:
        #         conf_loss = conf_loss / torch.sqrt(preds['pred_conf_max'].detach() + 1e-5)
        preds['per_pixel_conf_loss'] = conf_loss.detach() * valid_mask_depth.detach() / (max_mog_weight.detach()**3)

        valid_bool = valid_mask_depth > 0
        conf_loss = conf_loss[valid_bool]
        conf_loss, quantile_mask = filter_by_quantile(conf_loss, 0.98, return_mask=True)

        valid_mask_used = torch.zeros_like(valid_mask_depth)
        valid_mask_used[valid_bool] = quantile_mask.to(valid_mask_depth.dtype)
        gts["valid_mask_used"] = (valid_mask_used > 0.5) & (valid_mask_depth > 0)
        
        max_mog_weight = max_mog_weight[ gts["valid_mask_used"] ]
        if self.no_conf:
            conf_loss = conf_loss / (max_mog_weight.detach()**3)

        conf_loss = conf_loss.mean() * self.alpha
        
        mask_no_label = (valid_mask_depth <= 0) & (gts["sky_mask"] <= 0)
        mog_weight_list_full = preds["mog_weight_full"] if "mog_weight_full" in preds else mog_weight_list
        if mask_no_label.sum() > 0:
            conf_loss = conf_loss - torch.softmax(mog_weight_list_full, dim=-1).max(dim=-1).values[mask_no_label].mean() * 1e-3 # max value --> 1 for no-label areas
        else:
            conf_loss = conf_loss

        if self.use_sky_loss:
            mog_weight_list_full = preds["mog_weight_full"]
            sky_mask = gts["sky_mask"]
            sky_mask_valid = gts["sky_mask_valid"]

            non_sky_mask = (1 - sky_mask) * sky_mask_valid + valid_mask_depth
            non_sky_mask = (non_sky_mask > 0.5).float()
            
            assert sky_mask.shape == mog_weight_list_full[..., -1].shape
            assert non_sky_mask.shape == mog_weight_list_full[..., -1].shape
            sky_term = -mog_weight_list_full[..., -1] * sky_mask
            nonsky_term = -torch.logsumexp(mog_weight_list_full[..., :-1], dim=-1) * non_sky_mask
            nll_sky = (sky_term + nonsky_term).sum() / (valid_mask_depth.sum() + 1e-3)
            return conf_loss, nll_sky * self.alpha
        else:
            return conf_loss, torch.zeros_like(conf_loss)

    def compute_camera_loss(self, gts, preds, pose_scales, **kw):
        details = {}
        self_name = type(self).__name__.replace("RayDepthMog", "")
        bb, ll, H, W = gts["depthmap"].shape
        gt_cam_encoding = extri_intri_to_pose_encoding(
            gts["camera_pose"], gts["camera_intrinsics"], (H, W)
        )
        pred_cam_encoding = preds["pose_enc_list"]

        pred_cam_encoding = torch.cat(
            [
                pred_cam_encoding[:, :, :3] * pose_scales.view(-1, 1, 1),
                pred_cam_encoding[:, :, 3:],
            ],
            dim=-1,
        )

        T_loss = torch.abs(gt_cam_encoding[..., :3] - pred_cam_encoding[..., :3]).mean()
        R_loss = torch.abs(gt_cam_encoding[..., 3:7] - pred_cam_encoding[..., 3:7]).mean()
        fl_loss = torch.abs(gt_cam_encoding[..., 7:] - pred_cam_encoding[..., 7:]).mean()
        cam_loss = T_loss + R_loss + fl_loss

        details[self_name + "_camera_loss" + "/00"] = float(cam_loss.detach())
        details[self_name + "_camera_loss_T" + "/00"] = float(T_loss.detach())
        details[self_name + "_camera_loss_R" + "/00"] = float(R_loss.detach())
        details[self_name + "_camera_loss_fl" + "/00"] = float(fl_loss.detach())
        return cam_loss, details

    def compute_pts3d_loss(self, gts, preds, pose_scales_depth=None):
        # `detach_depth_in_ptsloss=False` baked; `normalize_style='pi3'` baked
        # → the `if shape>100 and style!='vggt'` clause collapses to just
        # `if shape>100`.
        gt_origins_hw, gt_directions_hw = generate_raymaps(gts)
        pred_origins_hw, pred_directions_hw = generate_raymaps_preds(preds)

        if self.detach_depth_scale_in_ptsloss:
            pred_depth = preds["depth_inference"].detach()
        else:
            pred_depth = preds["depth_inference_avg"]

        gt_pts3d = gt_origins_hw + gt_directions_hw * gts["depthmap"].unsqueeze(-1)
        pred_pts3d = pred_origins_hw + pred_directions_hw * pred_depth.unsqueeze(-1)
        valid_mask_depth = gts["valid_mask"]

        valid_mask_depth_flatten = valid_mask_depth.reshape(valid_mask_depth.shape[0], -1)
        gt_pts3d_flatten = gt_pts3d.reshape(gt_pts3d.shape[0], -1, 3)
        pred_pts3d_flatten = pred_pts3d.reshape(pred_pts3d.shape[0], -1, 3)

        pose_scales = []
        pose_scale_flatten = []
        bb, ll, h, w, _ = gt_origins_hw.shape
        for i in range(bb):
            valid_mask_depth_flatten_i = (valid_mask_depth_flatten[i] > 0) & (
                gt_pts3d_flatten[i].norm(dim=-1) > 0.01
            )
            gt_pts3d_flatten_valid = gt_pts3d_flatten[i][valid_mask_depth_flatten_i]
            pred_pts3d_flatten_valid = pred_pts3d_flatten[i][valid_mask_depth_flatten_i]

            if gt_pts3d_flatten_valid.shape[0] > 100:
                pose_scales_flatten_valid = gt_pts3d_flatten_valid.norm(
                    dim=-1
                ) / torch.clamp(pred_pts3d_flatten_valid.norm(dim=-1), min=1e-8)
                pose_scales_i = torch.median(pose_scales_flatten_valid, dim=0).values
                pose_scales.append(pose_scales_i)
                pose_scale_flatten.append(
                    torch.mean(
                        gt_pts3d_flatten_valid.norm(dim=-1)
                        / torch.clamp(pred_pts3d_flatten_valid.norm(dim=-1), min=1e-8),
                        dim=0,
                        keepdim=True,
                    )
                )
            else:
                pose_scales.append(torch.ones_like(gt_pts3d_flatten[i][0, 0]))

        pose_scales = torch.stack(pose_scales, dim=0).reshape(bb)
        pose_scales_detach = pose_scales.detach()
        if len(pose_scale_flatten) > 0:
            pose_scale_flatten = torch.cat(pose_scale_flatten, dim=0)
        else:
            pose_scale_flatten = torch.ones_like(pose_scales)

        pred_pts3d_conf = preds["pred_conf_max"].detach()
        pred_pts3d = pred_pts3d * rearrange(pose_scales_detach, "b -> b () () () ()")

        conf_loss = self.per_mog_point_regression_loss_l1(
            gt_pts3d, pred_pts3d, None, pred_pts3d_conf, valid_mask_depth
        )
        conf_loss = conf_loss[valid_mask_depth > 0]
        conf_loss = filter_by_quantile(conf_loss, 0.98)

        reg_loss = torch.square(pose_scale_flatten - 1).mean() * 1e-3
        return conf_loss.mean() + reg_loss

    def compute_multi_loss(self, gts, preds, **kw):
        details = {}
        total_loss = 0.0
        self_name = self.self_name

        gt_depth = gts["depthmap"]
        valid_mask_depth = gts["valid_mask"]
        pred_depth_list = preds["depth"]
        pred_depth_conf_list = preds["depth_conf"]
        mog_weight_list = preds["mog_weight"]

        bb, ss, hh, ww = gt_depth.shape

        gt_depth_flatten_std = self.get_depth_std(gt_depth, valid_mask_depth)
        pred_depth_max, pred_conf_max, max_indices, all_nll = self.get_depth_inference(
            pred_depth_list,
            pred_depth_conf_list,
            mog_weight_list,
            gt_depth_flatten_std,
            valid_mask_depth,
            preds,
        )

        # `normalize_style='pi3'` baked → always emit the mean-scale regularizer.
        pose_scales, _ = calculate_pose_scale_from_depths(
            pred_depth_max, gt_depth, valid_mask_depth
        )
        # `in_door_vggt_normalize=False` baked → no per-sample remap.
        if not self.median_scale:
            pose_scales = torch.ones_like(pose_scales)
        pose_scales_detach = pose_scales.detach()
        
        # `larger_wi_weight=False` baked → single conf branch.
        conf_loss, nll_sky = self.conf_loss_single(gts, preds, pose_scales)
        total_loss = total_loss + conf_loss + nll_sky
        details[self_name + "_depth_loss" + "_conf" + "/00"] = float(conf_loss.detach())
        details[self_name + "_depth_loss" + "_nll_sky" + "/00"] = float(nll_sky.detach())

        camera_loss, details_camera = self.compute_camera_loss(gts, preds, pose_scales_detach)
        total_loss = total_loss + camera_loss
        details[self_name + "_camera_loss" + "/00"] = float(camera_loss.detach())
        details.update(details_camera)

        point_loss = self.compute_pts3d_loss(gts, preds, pose_scales)
        total_loss = total_loss + point_loss
        details[self_name + "_point_loss" + "/00"] = float(point_loss.detach())

        assert (not torch.isnan(point_loss)) and (not torch.isinf(point_loss))
        assert (not torch.isnan(conf_loss)) and (not torch.isinf(conf_loss))
        assert (not torch.isnan(camera_loss)) and (not torch.isinf(camera_loss))
        assert (not torch.isnan(pred_depth_max).any()) and (not torch.isinf(pred_depth_max).any())

        # `use_avg_grad_loss=False` baked → only `use_grad_loss` branch.
        if self.use_grad_loss:
            origins, directions = generate_raymaps(gts, use_cam2worlds=False)
            gt_depth_gradient = gts["depthmap"]
            pred_depth_max_gradient = pred_depth_max.reshape(bb, ss, hh, ww) * rearrange(
                pose_scales.detach(), "b -> b () () ()"
            )
            directions_xdisp = directions[:, :, :, 1:] - directions[:, :, :, :-1]
            directions_ydisp = directions[:, :, 1:, :] - directions[:, :, :-1, :]
            gt_points_xdisp_delta = directions_xdisp[..., 0] * gt_depth_gradient[:, :, :, :-1]
            gt_points_ydisp_delta = directions_ydisp[..., 1] * gt_depth_gradient[:, :, :-1, :]
            gt_points_xdisp_delta = F.pad(gt_points_xdisp_delta, (1, 0), mode="constant", value=1)
            gt_points_ydisp_delta = F.pad(
                gt_points_ydisp_delta, (0, 0, 1, 0), mode="constant", value=1
            )
            gt_points_delta = torch.stack(
                [gt_points_xdisp_delta, gt_points_ydisp_delta], dim=-1
            )
            gt_points_delta = torch.norm(gt_points_delta, dim=-1)
            gt_points_delta = torch.clamp(gt_points_delta, min=1e-6)

            loss_grad_frames = gradient_loss_multi_scale_points(
                pred_depth_max_gradient.reshape(bb * ss, hh, ww, 1),
                gt_depth_gradient.reshape(bb * ss, hh, ww, 1),
                gt_points_delta.reshape(bb * ss, hh, ww, 1),
                valid_mask_depth.reshape(bb * ss, hh, ww),
                conf=None,
                loss_type=self.loss_type,
            )
            grad_loss = loss_grad_frames
            total_loss = total_loss + grad_loss
            details[self_name + "_depth_loss" + "_grad" + "/00"] = float(grad_loss.detach())
        else:
            grad_loss = conf_loss * 0.0

        if self.median_scale:   
            pose_scales_mean, _ = calculate_mean_pose_scale_from_depths(
                pred_depth_max, gt_depth, gts["valid_mask_used"]
            )
            reg_loss = (
                torch.square(pose_scales_mean - 1).mean() + torch.square(1 / (pose_scales_mean + 1e-8) - 1).mean()
            )
            total_loss = total_loss + reg_loss * 0.1
            details[self_name + "_reg_loss" + "/00"] = float(reg_loss.detach())
            details[self_name + "_depth_loss" + "pose_scales_min" + "/00"] = pose_scales.min()
            details[self_name + "_depth_loss" + "pose_scales_max" + "/00"] = pose_scales.max()
            details[self_name + "_depth_loss" + "pose_scales_mean_min" + "/00"] = pose_scales_mean.min()
            details[self_name + "_depth_loss" + "pose_scales_mean_max" + "/00"] = pose_scales_mean.max()
        
        reg_loss = torch.square(torch.log(torch.stack(pred_depth_conf_list, dim=-1))).mean() * 1e-6
        total_loss = total_loss + reg_loss
        details[self_name + "_conf_reg_loss" + "/00"] = float(reg_loss.detach())

        preds["depth"] = [
            item * rearrange(pose_scales, "b -> b () () ()") for item in pred_depth_list
        ]
        preds["depth_inference"] = preds["depth_inference"] * rearrange(
            pose_scales, "b -> b () () ()"
        )

        return total_loss, details

