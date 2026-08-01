"""Unit contracts for the V4-A WHU padding experiment."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
import unittest

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from datasets.WHU_dataset import WHU_Dataset  # noqa: E402
from losses import DiceLoss, SoftCrossEntropyLoss  # noqa: E402
from utils.transform import crop  # noqa: E402


class V4PaddingProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.optical = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        self.label = np.asarray([[0, 1], [2, 3]], dtype=np.int32)
        self.aux = np.asarray([[11, 12], [13, 14]], dtype=np.uint8)

    def test_split_fill_uses_ignore_for_mask_and_zero_for_aux(self):
        optical, label, aux = crop(
            self.optical,
            self.label,
            self.aux,
            size=4,
            mask_fill=7,
            aux_fill=0,
        )

        np.testing.assert_array_equal(optical[:2, :2], self.optical)
        np.testing.assert_array_equal(label[:2, :2], self.label)
        np.testing.assert_array_equal(aux[:2, :2], self.aux)
        self.assertTrue(np.all(label[2:, :] == 7))
        self.assertTrue(np.all(label[:, 2:] == 7))
        self.assertTrue(np.all(aux[2:, :] == 0))
        self.assertTrue(np.all(aux[:, 2:] == 0))

    def test_default_call_preserves_released_shared_zero_fill(self):
        _, label, aux = crop(
            self.optical,
            self.label,
            self.aux,
            size=4,
        )

        self.assertTrue(np.all(label[2:, :] == 0))
        self.assertTrue(np.all(label[:, 2:] == 0))
        self.assertTrue(np.all(aux[2:, :] == 0))
        self.assertTrue(np.all(aux[:, 2:] == 0))

    def test_dataset_variant_is_explicit_and_defaults_off(self):
        parameter = inspect.signature(WHU_Dataset.__init__).parameters[
            "mask_padding_ignore"
        ]
        self.assertIs(parameter.default, False)

    def test_ignore_logits_do_not_change_ce_or_dice(self):
        generator = torch.Generator().manual_seed(11)
        logits = torch.randn(1, 7, 2, 2, generator=generator)
        changed = logits.clone()
        changed[:, :, 0, 1] = torch.linspace(-20.0, 20.0, 7)
        target = torch.tensor([[[0, 7], [1, 2]]], dtype=torch.long)
        losses = (
            SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=7),
            DiceLoss(smooth=0.05, ignore_index=7),
        )

        for loss_fn in losses:
            torch.testing.assert_close(
                loss_fn(logits, target),
                loss_fn(changed, target),
                rtol=0.0,
                atol=1e-7,
            )

    def test_soft_ce_ignore_is_still_normalized_over_all_pixels(self):
        logits = torch.zeros(1, 7, 1, 2)
        full_target = torch.tensor([[[0, 0]]], dtype=torch.long)
        ignored_target = torch.tensor([[[0, 7]]], dtype=torch.long)
        loss_fn = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=7)

        full_loss = loss_fn(logits, full_target)
        ignored_loss = loss_fn(logits, ignored_target)
        torch.testing.assert_close(
            ignored_loss,
            full_loss * 0.5,
            rtol=0.0,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
