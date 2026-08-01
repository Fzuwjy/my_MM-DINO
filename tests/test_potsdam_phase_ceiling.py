"""CPU checks for the Potsdam spatial-phase ceiling runner."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.evaluate_potsdam_phase_ceiling import (
    CLASS_NAMES,
    CONDITION_NAMES,
    load_partial_journal,
    partial_protocol_binding,
    ceiling_interpretation,
    common_translation_slices,
    four_phase_shifts,
    headline_miou,
    prediction_from_score_sum,
    translate_tensor,
    write_partial_journal,
)


class PotsdamPhaseCeilingTests(unittest.TestCase):
    def test_four_phase_square(self) -> None:
        self.assertEqual(
            four_phase_shifts(8), ((0, 0), (0, 8), (8, 0), (8, 8))
        )

    def test_translation_zero_fills_without_wraparound(self) -> None:
        value = torch.arange(12).reshape(1, 1, 3, 4)
        shifted = translate_tensor(value, 1, 2)
        expected = torch.zeros_like(value)
        expected[..., 1:, 2:] = value[..., :2, :2]
        torch.testing.assert_close(shifted, expected)

    def test_common_slices_match_whu_protocol_geometry(self) -> None:
        shifts = tuple(
            shift
            for offset in (8, 16)
            for shift in four_phase_shifts(offset)
            if shift != (0, 0)
        )
        original, shifted = common_translation_slices((6000, 6000), shifts, 512)
        self.assertEqual((original[0].start, original[0].stop), (512, 5472))
        self.assertEqual((original[1].start, original[1].stop), (512, 5472))
        self.assertEqual(
            (shifted[(16, 16)][0].start, shifted[(16, 16)][0].stop),
            (528, 5488),
        )

    def test_prediction_replaces_only_common_region(self) -> None:
        baseline = np.zeros((4, 5), dtype=np.int16)
        scores = torch.zeros((len(CLASS_NAMES), 2, 3))
        scores[2] = 1.0
        result = prediction_from_score_sum(
            baseline, scores, (slice(1, 3), slice(1, 4))
        )
        expected = baseline.copy()
        expected[1:3, 1:4] = 2
        np.testing.assert_array_equal(result, expected)

    def test_headline_miou_excludes_clutter(self) -> None:
        confusion = np.eye(len(CLASS_NAMES), dtype=np.int64) * 10
        confusion[-1, -1] = 0
        confusion[-1, 0] = 10
        self.assertEqual(headline_miou(confusion), 0.9)

    def test_interpretation_distinguishes_phase_specific_gain(self) -> None:
        result = ceiling_interpretation(0.08, 0.20, 0.10)
        self.assertEqual(result["outcome"], "POSITIVE_SUBPATCH_PHASE_CEILING")
        result = ceiling_interpretation(0.08, 0.12, 0.10)
        self.assertEqual(
            result["outcome"], "POSITIVE_ENSEMBLE_CEILING_NOT_PHASE_SPECIFIC"
        )

    def test_partial_journal_round_trip_and_binding_guard(self) -> None:
        binding = partial_protocol_binding(
            checkpoint_sha256="a" * 64,
            inference_batch_size=8,
            valid_margin=512,
            seed=42,
            sample_names=["tile_0", "tile_1"],
        )
        record = {
            name: np.eye(len(CLASS_NAMES), dtype=np.int64)
            for name in CONDITION_NAMES
        }
        record.update(
            {
                "image_index": 0,
                "sample_name": "tile_0",
                "shape_hw": [6000, 6000],
                "hashes": {
                    "label": "0" * 64,
                    **{name: str(index) * 64 for index, name in enumerate(CONDITION_NAMES, 1)},
                },
                "elapsed_seconds": 12.5,
                "bootstrap_replicates": 20,
                "bootstrap_seed": 7,
            }
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json.partial.json"
            write_partial_journal(path, binding, [record], complete=False)
            restored = load_partial_journal(path, binding, 20, 7)
            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0]["sample_name"], "tile_0")
            np.testing.assert_array_equal(restored[0]["k4_8"], record["k4_8"])
            changed = {**binding, "valid_margin": 0}
            with self.assertRaisesRegex(ValueError, "binding differs"):
                load_partial_journal(path, changed, 20, 7)


if __name__ == "__main__":
    unittest.main()
