from lightning.pytorch.plugins import PrecisionPlugin
from lightning.pytorch.plugins import IOPlugin
import torch

class NoInputCastBF16Plugin(PrecisionPlugin):
    def forward_context(self):
        # Standard autocast for forward
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def training_step(self, *args, **kwargs):
        # Temporarily disable autocast for data preprocessing
        with torch.autocast(device_type="cuda", enabled=False):
            return super().training_step(*args, **kwargs)


class SyncCheckpointIO(IOPlugin):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        torch.save(checkpoint, path)