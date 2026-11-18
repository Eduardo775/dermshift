"""PyTorch dataset on top of the array caches."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .cache import CacheArrays
from .transforms import eval_transform, seeded_apply, train_transform


class SegDataset(Dataset):
    """Images and masks of one cache.

    indices   rows of the cache to use (``None``: all rows)
    train     apply the training augmentation; the augmentation of a sample is a function of
              (seed, epoch, row), so results do not depend on the number of loader workers
    image_fn  optional function applied to the uint8 RGB image before normalisation (test-time transforms)
    """

    def __init__(self, cache_dir: str | Path, cfg: dict, indices=None, train: bool = False, seed: int = 0, image_fn=None):
        self.arr = CacheArrays(Path(cache_dir))
        self.idx = np.arange(len(self.arr)) if indices is None else np.asarray(indices, dtype=np.int64)
        self.train = train
        self.image_fn = image_fn
        self.tf = train_transform(cfg) if train else eval_transform(cfg)
        self.seed, self.epoch = int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k) -> dict:
        # k is a position, or (position, epoch) when it comes from engine.EpochSampler
        if isinstance(k, (tuple, list)):
            k, epoch = int(k[0]), int(k[1])
        else:
            epoch = self.epoch
        i = int(self.idx[k])
        img, msk = self.arr.image(i), self.arr.mask(i)
        if self.image_fn is not None:
            img = self.image_fn(img)
        if self.train:
            out = seeded_apply(self.tf, (self.seed * 1_000_003 + epoch * 100_003 + i) % (2**32 - 1), img, msk)
        else:
            out = self.tf(image=img, mask=msk)
        return {"image": out["image"].float(), "mask": out["mask"].float().unsqueeze(0), "index": i,
                "image_id": self.arr.meta.at[i, "image_id"]}
