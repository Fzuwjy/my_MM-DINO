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
import os
from typing import Mapping


DEFAULT_CACHE_CAPACITY = 64
CACHE_CAPACITY_ENV = "MM_DINO_WHU_CACHE_CAPACITY"
PATCH_MARKER = "_mm_dino_whu_cache_compat"


def cache_capacity_from_environment(
    environ: Mapping[str, str] | None = None,
) -> int:
    """Resolve an infrastructure-only cache bound without changing samples."""

    values = os.environ if environ is None else environ
    raw_capacity = values.get(CACHE_CAPACITY_ENV)
    if raw_capacity is None:
        return DEFAULT_CACHE_CAPACITY
    try:
        capacity = int(raw_capacity)
    except ValueError as error:
        raise ValueError(
            f"{CACHE_CAPACITY_ENV} must be a non-negative integer, "
            f"got {raw_capacity!r}"
        ) from error
    if capacity < 0:
        raise ValueError(
            f"{CACHE_CAPACITY_ENV} must be non-negative, got {capacity}"
        )
    return capacity


CACHE_CAPACITY = cache_capacity_from_environment()


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
