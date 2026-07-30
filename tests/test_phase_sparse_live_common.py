"""CPU tests for the selected-crop Stage-B1 live execution primitive."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.phase_sparse_live_common import (
    forward_selected_phase_key_crops,
    forward_selected_phase_crops,
    normalized_phase_logits,
)


class _DummyMultimodalModel(torch.nn.Module):
    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        first = optical[:, :1] + sar[:, :1]
        second = optical[:, 1:2] - sar[:, :1]
        return torch.cat((first, second), dim=1)


class _BatchShapeRecordingModel(_DummyMultimodalModel):
    def __init__(self):
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(optical.shape[0]))
        return super().forward(optical, sar) + float(optical.shape[0])


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

    def test_global_phase_key_batches_pad_once_and_never_scatter_padding(self):
        model = _BatchShapeRecordingModel().eval()
        phase_tensors = {
            "x8": (self.optical, self.sar),
            "y8": (self.optical + 100.0, self.sar + 10.0),
            "xy8": (self.optical + 200.0, self.sar + 20.0),
        }
        result = forward_selected_phase_key_crops(
            phase_tensors,
            model,
            self.windows,
            {"x8": (0, 1), "y8": (0, 1), "xy8": (1,)},
            phase_order=("x8", "y8", "xy8"),
            n_output_channels=2,
            batch_size=4,
        )

        expected_real_keys = (
            ("x8", 0),
            ("x8", 1),
            ("y8", 0),
            ("y8", 1),
            ("xy8", 1),
        )
        expected_padding = (("xy8", 1),) * 3
        self.assertEqual(result["observed_phase_crop_keys"], expected_real_keys)
        self.assertEqual(result["padding_phase_crop_keys"], expected_padding)
        self.assertEqual(
            result["processed_phase_crop_keys"], expected_real_keys + expected_padding
        )
        self.assertEqual(result["selected_crop_samples"], 5)
        self.assertEqual(result["model_forward_crop_samples"], 8)
        self.assertEqual(result["padding_crop_samples"], 3)
        self.assertEqual(result["batch_calls"], 2)
        self.assertEqual(result["real_batch_sizes"], (4, 1))
        self.assertEqual(result["model_batch_sizes"], (4, 4))
        self.assertEqual(model.batch_sizes, [4, 4])

        for phase_name, selected in {
            "x8": (0, 1),
            "y8": (0, 1),
            "xy8": (1,),
        }.items():
            phase = result["phases"][phase_name]
            self.assertEqual(phase["observed_crop_ids"], selected)
            expected_count = np.zeros((6, 8), dtype=np.int16)
            for crop_id in selected:
                y0, y1, x0, x1 = self.windows[crop_id]
                expected_count[y0:y1, x0:x1] += 1
            np.testing.assert_array_equal(phase["count_mat"], expected_count)

            optical, sar = phase_tensors[phase_name]
            expected_sum = np.zeros((2, 6, 8), dtype=np.float32)
            for crop_id in selected:
                y0, y1, x0, x1 = self.windows[crop_id]
                first = optical[0, 0, y0:y1, x0:x1] + sar[0, 0, y0:y1, x0:x1]
                second = optical[0, 1, y0:y1, x0:x1] - sar[0, 0, y0:y1, x0:x1]
                expected_sum[0, y0:y1, x0:x1] += (first + 4.0).numpy()
                expected_sum[1, y0:y1, x0:x1] += (second + 4.0).numpy()
            np.testing.assert_array_equal(phase["sum_logits"], expected_sum)

        # The repeated xy8 crop is a model-shape filler only, not a second scatter.
        self.assertEqual(int(result["phases"]["xy8"]["count_mat"].max()), 1)

    def test_global_phase_key_plan_validation_fails_closed(self):
        active = {"x8": (self.optical, self.sar)}
        with self.assertRaisesRegex(ValueError, "phase_order"):
            forward_selected_phase_key_crops(
                active,
                self.model,
                self.windows,
                {"x8": (0,), "y8": ()},
                phase_order=("x8", "x8"),
                n_output_channels=2,
                batch_size=2,
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            forward_selected_phase_key_crops(
                active,
                self.model,
                self.windows,
                {"x8": (1, 0), "y8": ()},
                phase_order=("x8", "y8"),
                n_output_channels=2,
                batch_size=2,
            )
        with self.assertRaisesRegex(ValueError, "active phases"):
            forward_selected_phase_key_crops(
                {},
                self.model,
                self.windows,
                {"x8": (0,), "y8": ()},
                phase_order=("x8", "y8"),
                n_output_channels=2,
                batch_size=2,
            )


if __name__ == "__main__":
    unittest.main()
