"""Narrow compatibility shim for the released WHU training dataset.

The public ``WHU_Dataset`` returns training labels as NumPy ``int32`` arrays,
while its released soft cross-entropy loss uses ``torch.gather``, whose class
indices must be ``int64``.  Install this shim before constructing DataLoader
workers to change only the label storage dtype; label values are unchanged.
"""

from __future__ import annotations

from functools import wraps

import numpy as np


PATCH_MARKER = "_mm_dino_whu_int64_compat"


def label_to_int64(label: np.ndarray) -> np.ndarray:
    array = np.asarray(label)
    return array.astype(np.int64, copy=False)


def install_whu_label_dtype_compat() -> None:
    from datasets.WHU_dataset import WHU_Dataset

    original_getitem = WHU_Dataset.__getitem__
    if getattr(original_getitem, PATCH_MARKER, False):
        return

    @wraps(original_getitem)
    def getitem_with_int64_training_label(self, index):
        sample = original_getitem(self, index)
        if self.data_type != "train":
            return sample
        if len(sample) == 3:
            image, auxiliary, label = sample
            return image, auxiliary, label_to_int64(label)
        if len(sample) == 2:
            image, label = sample
            return image, label_to_int64(label)
        raise RuntimeError(f"Unexpected WHU sample structure: {len(sample)} values")

    setattr(getitem_with_int64_training_label, PATCH_MARKER, True)
    WHU_Dataset.__getitem__ = getitem_with_int64_training_label
