"""CPU tests for the selected-crop Stage-B1 live execution primitive."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.phase_sparse_live_common import (
    forward_selected_phase_crops,
    normalized_phase_logits,
)


class _DummyMultimodalModel(torch.nn.Module):
    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        first = optical[:, :1] + sar[:, :1]
        second = optical[:, 1:2] - sar[:, :1]
        return torch.cat((first, second), dim=1)


class SparseLiveCommonTest(unittest.TestCase):
    def setUp(self):
        values = torch.arange(3 * 6 * 8, dtype=torch.float32).reshape(1, 3, 6, 8)
        self.optical = values / 10.0
        self.sar = torch.arange(6 * 8, dtype=torch.float32).reshape(1, 1, 6, 8) / 7.0
        self.windows = ((0, 6, 0, 5), (0, 6, 3, 8))
        self.model = _DummyMultimodalModel().eval()

    def test_full_execution_matches_an_independent_manual_slide(self):
        result = forward_selected_phase_crops(
            self.optical,
            self.sar,
            self.model,
            self.windows,
            (0, 1),
            n_output_channels=2,
            batch_size=2,
            require_full_coverage=True,
        )
        manual_sum = np.zeros((2, 6, 8), dtype=np.float32)
        manual_count = np.zeros((6, 8), dtype=np.int16)
        with torch.inference_mode():
            for crop_id, (y0, y1, x0, x1) in enumerate(self.windows):
                output = self.model(
                    self.optical[:, :, y0:y1, x0:x1],
                    self.sar[:, :, y0:y1, x0:x1],
                )[0].numpy()
                manual_sum[:, y0:y1, x0:x1] += output
                manual_count[y0:y1, x0:x1] += 1
        np.testing.assert_array_equal(result["sum_logits"], manual_sum)
        np.testing.assert_array_equal(result["count_mat"], manual_count)
        self.assertEqual(result["observed_crop_ids"], (0, 1))
        self.assertEqual(result["batch_calls"], 1)
        expected = manual_sum / manual_count[None]
        np.testing.assert_array_equal(normalized_phase_logits(result), expected)

    def test_sparse_execution_exposes_zero_count_without_leaking_extra_cache(self):
        result = forward_selected_phase_crops(
            self.optical,
            self.sar,
            self.model,
            self.windows,
            (1,),
            n_output_channels=2,
            batch_size=8,
        )
        self.assertEqual(result["observed_crop_ids"], (1,))
        self.assertTrue(np.all(result["count_mat"][:, :3] == 0))
        self.assertTrue(np.all(result["count_mat"][:, 3:] == 1))
        with self.assertRaisesRegex(AssertionError, "uncovered"):
            forward_selected_phase_crops(
                self.optical,
                self.sar,
                self.model,
                self.windows,
                (1,),
                n_output_channels=2,
                batch_size=8,
                require_full_coverage=True,
            )

    def test_invalid_or_noncanonical_crop_plan_fails_closed(self):
        for bad in ((1, 0), (1, 1), (2,), (True,)):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                forward_selected_phase_crops(
                    self.optical,
                    self.sar,
                    self.model,
                    self.windows,
                    bad,
                    n_output_channels=2,
                    batch_size=2,
                )


if __name__ == "__main__":
    unittest.main()
