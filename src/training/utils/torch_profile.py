import torch
import os
from lightning.pytorch import Callback

class CUDAMemoryCallback(Callback):
    def __init__(self, log_every_n_steps=50):
        self.log_every_n_steps = log_every_n_steps

    def on_fit_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not torch.cuda.is_available():
            return
        if (trainer.global_step + 1) % self.log_every_n_steps != 0:
            return

        # bytes -> MiB
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

        # appears in progress bar + logger (TensorBoard/W&B/etc.)
        pl_module.log_dict(
            {
                "cuda/alloc_mib": alloc,
                "cuda/reserved_mib": reserved,
                "cuda/peak_alloc_mib": peak_alloc,
                "cuda/peak_reserved_mib": peak_reserved,
            },
            on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
