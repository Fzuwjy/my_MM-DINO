"""Bound the released WHU full-image caches without changing sample values.

The public dataset allocates three independent LRU caches with capacity 100 in
every persistent DataLoader worker.  With four workers, five training epochs
populate enough full optical/SAR/label images to push the 90 GiB container over
``memory.high`` when the official test routine also materializes all full-size
predictions and labels.  Reducing only the cache capacity changes I/O reuse, not
filenames, random draws, crops, augmentations, tensors, or metric definitions.
"""

from __future__ import annotations

from functools import wraps


CACHE_CAPACITY = 2
PATCH_MARKER = "_mm_dino_whu_cache_compat"


def set_dataset_cache_capacity(dataset, capacity: int = CACHE_CAPACITY) -> None:
    if capacity < 0:
        raise ValueError("WHU cache capacity must be non-negative")
    dataset.cache_size = capacity
    for name in ("rgb_cache", "label_cache", "sar_cache"):
        cache = getattr(dataset, name)
        cache.capacity = capacity


def install_whu_cache_compat(capacity: int = CACHE_CAPACITY) -> None:
    from datasets.WHU_dataset import WHU_Dataset

    original_init = WHU_Dataset.__init__
    installed_capacity = getattr(original_init, PATCH_MARKER, None)
    if installed_capacity is not None:
        if installed_capacity != capacity:
            raise RuntimeError(
                f"WHU cache compatibility already installed with capacity "
                f"{installed_capacity}, requested {capacity}"
            )
        return

    @wraps(original_init)
    def init_with_bounded_caches(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        set_dataset_cache_capacity(self, capacity)

    setattr(init_with_bounded_caches, PATCH_MARKER, capacity)
    WHU_Dataset.__init__ = init_with_bounded_caches
