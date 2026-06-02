import random
from torch.utils.data import DataLoader

class RatioDataLoader(DataLoader):
    """
    A DataLoader-like object that yields batches from two DataLoaders
    according to a Bernoulli ratio.

    - "Main" dataloader: this instance (super().__init__)
    - Secondary dataloader: self.dataloader2
    - Length: len(main dataloader)
    """
    def __init__(
        self,
        dataset, dataset2,
        sampler1=None, sampler2=None,
        num_workers=8, pin_memory=True, persistent_workers=False,
        multiprocessing_context=None,
        ratio_main=0.3,
        generator=None,          # optional torch.Generator for deterministic choice
        seed=None,               # optional python RNG seed for deterministic choice
    ):
        super().__init__(
            dataset,
            batch_sampler=sampler1,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )

        self.dataloader2 = DataLoader(
            dataset2,
            batch_sampler=sampler2,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
        )

        if not (0.0 <= ratio_main <= 1.0):
            raise ValueError(f"ratio_main must be in [0, 1], got {ratio_main}")
        self.ratio_main = float(ratio_main)

        # RNG for choosing which loader each step (kept in main process)
        self._py_rng = random.Random(seed) if seed is not None else random.Random()

        # Optional torch generator for choice (also main process); if provided, use it
        self._torch_gen = generator

    def __len__(self):
        # "Same length as dataloader a"
        return super().__len__()

    def _choose_main(self) -> bool:
        if self._torch_gen is not None:
            # torch-based deterministic choice if user passes a generator
            # sample uniform in [0,1)
            import torch
            return bool(torch.rand((), generator=self._torch_gen).item() < self.ratio_main)
        else:
            return self._py_rng.random() < self.ratio_main

    def __iter__(self):
        it1 = super().__iter__()          # iterator for main loader
        it2 = iter(self.dataloader2)      # iterator for secondary loader

        # Iterate exactly len(main loader) steps
        for _ in range(len(self)):
            use_main = self._choose_main()

            if use_main:
                try:
                    batch = next(it1)
                except StopIteration:
                    # if main ends early (e.g., sampler behavior), restart it
                    it1 = super().__iter__()
                    batch = next(it1)
            else:
                try:
                    batch = next(it2)
                except StopIteration:
                    it2 = iter(self.dataloader2)
                    batch = next(it2)

            yield batch
