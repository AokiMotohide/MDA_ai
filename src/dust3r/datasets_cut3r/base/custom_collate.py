from torch.utils.data._utils.collate import default_collate
import warnings

def collate_like_default(batch, drop_on_fail=True, min_size=1):
    """
    Default-collate with fault tolerance.

    Behavior:
    - Try default_collate on full batch
    - If it fails:
        - progressively drop samples and retry
        - if still fails, return None

    Args:
        batch: list of samples
        drop_on_fail: whether to drop samples on failure
        min_size: minimum batch size to allow after dropping

    Returns:
        Collated batch or None
    """
    if len(batch) == 0:
        return None

    # Fast path: identical to default behavior
    return default_collate(batch)
    # try:
    #     return default_collate(batch)
    # except Exception as e:
    #     if not drop_on_fail:
    #         raise

    #     warnings.warn(
    #         f"[collate] default_collate failed on batch of size {len(batch)}: {e}"
    #     )

    # Slow path: progressively drop samples
    valid = list(batch)

    while len(valid) >= min_size:
        try:
            return default_collate(valid)
        except Exception:
            # drop one sample and retry
            valid.pop()

    # Total failure
    warnings.warn("[collate] all samples in batch are invalid; skipping batch")
    return None
