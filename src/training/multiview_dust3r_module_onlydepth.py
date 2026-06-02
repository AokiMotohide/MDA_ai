# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import re
import json
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.distributed import all_gather_object, barrier
from lightning import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics import MaxMetric, MeanMetric, MinMetric, SumMetric, Metric
from torchmetrics.aggregation import BaseAggregator
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import gc
import random
import math

import sys
sys.path.append(".")
sys.path.append("src")
# torch.autograd.set_detect_anomaly(True)
from src.testing.utils.test_utils import export_to_gs_video
from src.training.utils import pylogger
from src.training.utils.lora_utils import get_finetuning_model, save_trainable_parameters
from safetensors.torch import load_file
from src.training.utils.debug_vis_utils import debug_vis_output_utils_onlydepth, debug_vis_output_utils_separate_depth
from src.depth_anything_3.cfg import create_object, load_config
from src.depth_anything_3.utils.utils_training import prepare_inputs

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

class AccumulatedSum(BaseAggregator):
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            fn="sum",
            default_value=torch.tensor(0.0, dtype=torch.long),
            nan_strategy='warn',
            state_name="sum_value",
            **kwargs,
        )

    def update(self, value: int) -> None:
        self.sum_value += value

    def compute(self) -> torch.LongTensor:
        return self.sum_value

class MultiViewDUSt3RLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        train_criterion: torch.nn.Module,
        validation_criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        pretrained: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        eval_use_pts3d_from_local_head: bool = True,
        use_lora: bool = False,
        saved_path: str = "logs/",
        lr_layer_scale: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
        per_group_grad_clip: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net', 'train_criterion', 'validation_criterion'])

        self.use_lora = use_lora

        self.train_criterion = train_criterion
        self.validation_criterion = validation_criterion
        self.pretrained = pretrained
        self.net = net
        
        with torch.no_grad():
            self._load_pretrained_weights()
        
        self.net.train()
        self.validation_criterion = None
            
        self.resume_from_checkpoint = resume_from_checkpoint
        self.eval_use_pts3d_from_local_head = eval_use_pts3d_from_local_head

        # use register_buffer to save these with checkpoints
        # so that when we resume training, these bookkeeping variables are preserved
        self.register_buffer("epoch_fraction", torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.register_buffer("train_total_samples", torch.tensor(0, dtype=torch.long, device=self.device))
        self.register_buffer("train_total_images", torch.tensor(0, dtype=torch.long, device=self.device))

        self.train_total_samples_per_step = AccumulatedSum()  # these need to be reduced across GPUs, so use Metric
        self.train_total_images_per_step = AccumulatedSum()  # these need to be reduced across GPUs, so use Metric

        self.val_loss = MeanMetric()
        save_trainable_parameters(self.net)
        self.saved_path = saved_path
        self.lr_layer_scale = lr_layer_scale or {}
        self.per_group_grad_clip = per_group_grad_clip

    @classmethod
    def load_for_inference(cls, net):
        lit_module = cls(net=net, train_criterion=None, validation_criterion=None, optimizer=None, scheduler=None, compile=False)
        lit_module.eval()
        return lit_module

    def forward(self, views: Dict[str, torch.Tensor]) -> Any:
        extrinsics = views["camera_extrinsics"]
        intrinsics = views["camera_intrinsics"]
        
        alpha_img = None
            
        predictions = self.net(views["img"], extrinsics=extrinsics, intrinsics=intrinsics, 
                               infer_gs=False, backbone_nograde=True, views=views,
                               alpha_img=alpha_img)
        
        return predictions

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

        # the wandb logger lives in self.loggers
        # find the wandb logger and watch the model and gradients
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                self.wandb_logger = logger
                # log gradients, parameter histogram and model topology
                self.wandb_logger.watch(self.net, log="all", log_freq=128, log_graph=False)


    def on_before_optimizer_step(self, optimizer) -> None:
        """Per-parameter-group gradient clipping executed before each optimizer step."""
        if self.per_group_grad_clip is not None and self.per_group_grad_clip > 0:
            for group in optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], max_norm=self.per_group_grad_clip)

    def on_train_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        if hasattr(self.trainer.train_dataloader, "dataset") and hasattr(
            self.trainer.train_dataloader.dataset, "set_epoch"
        ):
            self.trainer.train_dataloader.dataset.set_epoch(self.current_epoch * self.trainer.world_size + self.global_rank)
        if hasattr(self.trainer.train_dataloader, "sampler") and hasattr(
            self.trainer.train_dataloader.sampler, "set_epoch"
        ):
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch * self.trainer.world_size + self.global_rank)
        # should have batch sampler
        if hasattr(self.trainer.train_dataloader, "batch_sampler") and hasattr(
            self.trainer.train_dataloader.batch_sampler, "set_epoch"
        ):
            print("Setting epoch for batched_sampler")
            self.trainer.train_dataloader.batch_sampler.set_epoch(self.current_epoch * self.trainer.world_size + self.global_rank)

    def on_save_checkpoint(self, checkpoint):
        # Force Python to release unreferenced memory before pickling starts
        gc.collect()
        print('on_save_checkpoint')
        
    def on_validation_epoch_start(self) -> None:
        # our custom dataset and sampler has to have epoch set by calling set_epoch
        for loader in self.trainer.val_dataloaders:
            if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(0)
            if hasattr(loader, "sampler") and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(0)

    def model_step(
        self, batch: List[Dict[str, torch.Tensor]], criterion: torch.nn.Module, batch_idx: int,
    ) -> Tuple[torch.Tensor, Dict]:
        device = self.device

        # Move data to device
        batch = prepare_inputs(batch, device, new_gs_mask=False, rand_shuffle=True)
        batch['use_2d_gaussian'] = True
        batch['global_steps'] = self.global_step
        preds = self.forward(batch)
        
        # Compute the loss in higher precision
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            loss, loss_details = criterion(batch, preds) if criterion is not None else None

        if batch_idx % 137 == 0:
            saved_path = self.saved_path.replace('checkpoints', 'vis')
            os.makedirs(saved_path, exist_ok=True)
            
            with torch.no_grad():
                if isinstance(preds['depth'], list):
                    if hasattr(self, "wandb_logger") and self.wandb_logger is not None:
                        vis_images = debug_vis_output_utils_separate_depth(
                            preds,
                            batch,
                            batch_idx,
                            saved_path=saved_path,
                            complete=False,
                            return_images=True,
                            max_images=4,
                            global_rank=self.global_rank
                        )
                        if len(vis_images) > 0:
                            self.wandb_logger.log_image(
                                key="train/debug_vis_output_utils_separate_depth",
                                images=[img for _, img in vis_images],
                                caption=[name for name, _ in vis_images],
                                step=self.global_step * self.trainer.world_size + self.global_rank,
                            )
                    else:
                        debug_vis_output_utils_separate_depth(preds, batch, batch_idx, saved_path, complete=False)
                        
                else:
                    debug_vis_output_utils_onlydepth(preds, batch, batch_idx, saved_path, complete=False)

        return batch, preds, loss, loss_details

    def training_step(
        self, batch: List[Dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:

        views, preds, loss, loss_details = self.model_step(batch, self.train_criterion, batch_idx)

        if not isinstance(loss, (torch.Tensor, dict, type(None))):  # this will cause a lightning.fabric.utilities.exceptions.MisconfigurationException
            # log loss and the batch information to help debugging
            # use print instead of log because the logger only logs on rank 0, but this could happen on any rank
            print(f"Loss is not a tensor or dict but {type(loss)}, value: {loss.item()}")
            print(f"Loss details: {loss_details}")
            print(f"Batch: {batch}")
            print(f"Batch index: {batch_idx}")
            print(f"Views: {views}")
            print(f"Preds: {preds}")
            loss = None  # set loss to None will still break the training loop in DDP, this is intended - we should fix the data to avoid nan loss in the first place
            return loss

        self.epoch_fraction = torch.tensor(self.trainer.current_epoch + batch_idx / self.trainer.num_training_batches, device=self.device)

        self.log("trainer/epoch", self.epoch_fraction, on_step=True, on_epoch=False, prog_bar=True)
        self.log("trainer/lr", self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0], on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True)

        # log the details of the loss
        if loss_details is not None:
            for key, value in loss_details.items():
                self.log(f"train_detail_{key}", value, on_step=True, on_epoch=False, prog_bar=False)
                match = re.search(r'/(\d{1,2})$', key)
                if match:
                    stripped_key = key[:match.start()]
                    self.log(f"train/{stripped_key}", value, on_step=True, on_epoch=False, prog_bar=False)

        # Log the total number of samples seen so far
        batch_size = views["img"].shape[0]
        self.train_total_samples_per_step(batch_size)  # aggregate across all GPUs
        self.train_total_samples += self.train_total_samples_per_step.compute()  # accumulate across all steps
        self.train_total_samples_per_step.reset()
        self.log("trainer/total_samples", self.train_total_samples, on_step=True, on_epoch=False, prog_bar=False)

        # Log the total number of images seen so far
        num_views = len(views["img"][0])
        n_image_cur_step = batch_size * num_views
        self.train_total_images_per_step(n_image_cur_step)  # aggregate across all GPUs
        self.train_total_images += self.train_total_images_per_step.compute()  # accumulate across all steps
        self.train_total_images_per_step.reset()
        self.log("trainer/total_images", self.train_total_images, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(
        self, batch: List[Dict[str, torch.Tensor]], batch_idx: int, dataloader_idx: int = 0,
    ) -> torch.Tensor:
        pass
        
    def on_validation_epoch_end(self) -> None:
        print("Exiting on_validation_epoch_end...") # <--- Add this
        self.log("val/loss", self.val_loss, prog_bar=True)

        # if we dont do these, wandb for some reason cannot display the validation loss with them as the x-axis
        self.log("trainer/epoch", self.epoch_fraction, sync_dist=True)
        self.log("trainer/total_samples", self.train_total_samples.cpu().item(), sync_dist=True)
        self.log("trainer/total_images", self.train_total_images.cpu().item(), sync_dist=True)

    def _build_param_groups(self, base_lr: float) -> List[Dict]:
        """Build parameter groups with per-layer LR scaling and optional weight decay override.

        ``self.lr_layer_scale`` maps a substring pattern to either:
          - a float: LR multiplier (``lr = base_lr * multiplier``)
          - a dict:  ``{scale: float, weight_decay: float}`` for both LR and weight decay control

        Parameters matching a pattern get the specified overrides.
        Parameters not matching any pattern use ``base_lr`` and the optimizer default weight decay.
        Patterns are checked in definition order; first match wins.
        """
        if not self.lr_layer_scale:
            return [{"params": list(self.trainer.model.parameters())}]

        groups: Dict[str, Dict] = {}  # pattern -> group dict
        default_params: List = []

        for name, param in self.trainer.model.named_parameters():
            if not param.requires_grad:
                continue
            matched = False
            for pattern, value in self.lr_layer_scale.items():
                if pattern in name:
                    key = pattern
                    if key not in groups:
                        if isinstance(value, (int, float)):
                            group = {"params": [], "lr": base_lr * value, "_name": key}
                        else:
                            group = {"params": [], "lr": base_lr * value["scale"], "_name": key}
                            if "weight_decay" in value:
                                group["weight_decay"] = value["weight_decay"]
                        groups[key] = group
                    groups[key]["params"].append(param)
                    matched = True
                    break
            if not matched:
                default_params.append(param)

        param_groups = []
        if default_params:
            param_groups.append({"params": default_params})
        for g in groups.values():
            tag = g.pop("_name")
            wd_str = f", wd={g['weight_decay']}" if "weight_decay" in g else ""
            log.info(f"  LR group '{tag}': lr={g['lr']:.2e}, #params={len(g['params'])}{wd_str}")
            param_groups.append(g)

        return param_groups

    def configure_optimizers(self) -> Dict[str, Any]:
        base_lr = self.hparams.optimizer.keywords.get("lr", 1e-4)
        param_groups = self._build_param_groups(base_lr)
        optimizer = self.hparams.optimizer(params=param_groups)

        if self.hparams.scheduler is not None:
            scheduler_config = self.hparams.scheduler

            # HACK: if the class is pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR,
            # both warmup_epochs and max_epochs should be scaled.
            # more specifically, max_epochs should be scaled to total number of steps that we will have during training,
            # and warmup_epochs should be scaled up proportionally.
            if scheduler_config.func is LinearWarmupCosineAnnealingLR:
                # Extract the keyword arguments from the partial object
                scheduler_kwargs = {k: v for k, v in scheduler_config.keywords.items()}
                original_warmup_epochs = scheduler_kwargs['warmup_epochs']
                original_max_epochs = scheduler_kwargs['max_epochs']

                total_steps = self.trainer.estimated_stepping_batches  # total number of total steps in all training epochs

                # Scale warmup_epochs and max_epochs
                scaled_warmup_epochs = int(original_warmup_epochs * total_steps / original_max_epochs)
                scaled_max_epochs = total_steps

                # Update the kwargs with scaled values
                scheduler_kwargs.update({
                    'warmup_epochs': scaled_warmup_epochs,
                    'max_epochs': scaled_max_epochs
                })

                # Re-initialize the scheduler with updated parameters
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer=optimizer,
                    **scheduler_kwargs
                )
            else:
                scheduler = scheduler_config(optimizer=optimizer)

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'name': 'train/lr',  # put lr inside train group in loggers
                    'scheduler': scheduler,
                    # 'interval': 'step' if scheduler_config.func is LinearWarmupCosineAnnealingLR else 'epoch',
                    'interval': 'step',
                    'frequency': 1,
                }
            }

        return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def _load_pretrained_weights(self) -> None:
        log.info(f"Loading pretrained weights from {self.pretrained}")

        if self.pretrained.endswith(".safetensors"):
            checkpoint = load_file(self.pretrained, device='cpu')
        else:
            checkpoint = torch.load(self.pretrained, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        skip_gs = self.pretrained == "checkpoints/DA3-GIANT/model.safetensors"
        stripped = {}
        for k, v in checkpoint.items():
            if skip_gs and ('gs_head' in k or 'gs_adapter' in k):
                continue
            for p in ('model.', 'net.net.', 'net.'):
                if k.startswith(p):
                    k = k[len(p):]
                    break
            stripped[k] = v

        model_state_dict = self.net.net.state_dict()
        checkpoint_new = {}
        for k, v in stripped.items():
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                checkpoint_new[k] = v

        if self.pretrained.endswith(("model.safetensors", "model.pt")):
            # Local generator so noise is identical across DDP ranks regardless of
            # the global torch RNG (which is typically seeded per-rank).
            gen = torch.Generator(device='cpu').manual_seed(0)

            def _noisy(x):
                noise = torch.randn(x.shape, generator=gen, dtype=x.dtype, device='cpu').to(x.device)
                return x + noise * torch.abs(x).mean() * 0.1

            base = 'head_mog.scratch.output_conv2.'
            variants = [base] + [f'head_mog.scratch.output_conv2_{i}.' for i in range(2, 9)]
            for k, v in stripped.items():
                if 'head.' not in k:
                    continue
                k_mog = k.replace('head.', 'head_mog.')
                if k_mog in model_state_dict and v.shape == model_state_dict[k_mog].shape:
                    checkpoint_new[k_mog] = v.clone()
                if base not in k_mog or not self.net.pretrain_as_possible:
                    continue
                for variant in variants:
                    k_var = k_mog.replace(base, variant)
                    if k_var not in model_state_dict:
                        continue
                    tmp_v = model_state_dict[k_var].detach().clone()
                    if tmp_v.shape == v.shape:
                        tmp_v = _noisy(v)
                    else:
                        tmp_v[:1] = _noisy(v[:1])
                        tmp_v[2:] = _noisy(v[1:])

                    checkpoint_new[k_var] = tmp_v.clone()
                    print('updated', k_var)

        missing_keys, unexpected_keys = self.net.net.load_state_dict(checkpoint_new, strict=False)
        log.info(f"Missing keys: {missing_keys}")
        log.info(f"Unexpected keys: {unexpected_keys}")

    @staticmethod
    def _update_ckpt_keys(ckpt, new_head_name='downstream_head', head_to_keep='downstream_head1', head_to_discard='downstream_head2'):
        """Helper function to use the weights of a model with multiple heads in a model with a single head.
        specifically, keep only the weights of the first head and delete the weights of the second head.
        """
        new_ckpt = {'model': {}}

        for key, value in ckpt['model'].items():
            if key.startswith(head_to_keep):
                new_key = key.replace(head_to_keep, new_head_name)
                new_ckpt['model'][new_key] = value
            elif key.startswith(head_to_discard):
                continue
            else:
                new_ckpt['model'][key] = value

        return new_ckpt

