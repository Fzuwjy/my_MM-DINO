"""CPU tests for the isolated H3 K1-to-K2 offline screen helpers."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from scripts.phase_closure_common import PHASE_NAMES
from scripts.phase_h3_screen_common import (
    physical_cost_summary,
    random_exact_cost_k2_levels,
    score_ranked_k2_levels,
    score_ranked_k2_levels_for_image,
)


def _geometry(
    x8_dependencies,
    *,
    image_ids=None,
    eligible=None,
    baseline_crop_counts=None,
):
    count = len(x8_dependencies)
    if image_ids is None:
        image_ids = [0] * count
    image_ids = np.asarray(image_ids, dtype=np.int64)
    image_count = int(image_ids.max()) + 1
    if eligible is None:
        eligible = [True] * count
    if baseline_crop_counts is None:
        baseline_crop_counts = [8] * image_count
    return {
        "phase_names": PHASE_NAMES,
        "sample_names": tuple(f"image-{index}" for index in range(image_count)),
        "baseline_crop_counts": np.asarray(
            baseline_crop_counts, dtype=np.int64
        ),
        "image_ids": image_ids,
        "eligible": np.asarray(eligible, dtype=bool),
        "cells_by_image": tuple(
            tuple(np.flatnonzero(image_ids == image_index).tolist())
            for image_index in range(image_count)
        ),
        "dependencies": {
            "x8": tuple(x8_dependencies),
            "y8": tuple(0 for _ in range(count)),
            "xy8": tuple(0 for _ in range(count)),
        },
    }


class ScoreRankedK2Test(unittest.TestCase):
    def test_lower_global_index_breaks_tie_and_ineligible_is_never_routed(self):
        geometry = _geometry(
            (0b001, 0b001, 0b010, 0b100),
            eligible=(True, True, True, False),
        )
        result = score_ranked_k2_levels([5.0, 5.0, 4.0, 100.0], 1 / 3, geometry)
        np.testing.assert_array_equal(result["initial_selected_indices"], [0])
        np.testing.assert_array_equal(result["zero_cost_selected_indices"], [1])
        np.testing.assert_array_equal(result["levels_by_cell"], [2, 2, 1, 1])
        self.assertEqual(result["closure"]["extra_crop_forwards"], 1)

    def test_floor_quota_is_per_image_and_single_image_helper_is_global(self):
        geometry = _geometry(
            (0b001, 0b010, 0b001, 0b010),
            image_ids=(0, 0, 1, 1),
        )
        result = score_ranked_k2_levels([1.0, 2.0, 4.0, 3.0], 0.5, geometry)
        np.testing.assert_array_equal(result["initial_selected_indices"], [1, 2])
        np.testing.assert_array_equal(result["levels_by_cell"], [1, 2, 2, 1])

        single = score_ranked_k2_levels_for_image(
            [1.0, 2.0, 4.0, 3.0], 0.5, geometry, 1
        )
        np.testing.assert_array_equal(single["levels_by_cell"], [1, 1, 2, 1])
        self.assertEqual(single["closure"]["per_image"][0]["extra_crop_forwards"], 0)

    def test_ineligible_none_is_allowed_but_eligible_nonfinite_fails_closed(self):
        geometry = _geometry((0b1, 0), eligible=(True, False))
        result = score_ranked_k2_levels(
            np.asarray([0.5, -np.inf]), 1.0, geometry
        )
        np.testing.assert_array_equal(result["levels_by_cell"], [2, 1])
        with self.assertRaisesRegex(ValueError, "eligible scores must be finite"):
            score_ranked_k2_levels([np.nan, None], 1.0, geometry)


class PhysicalCostTest(unittest.TestCase):
    def test_per_image_batch_padding_counts_physical_but_not_logical_cost(self):
        geometry = _geometry(
            (0b1, 0b1),
            image_ids=(0, 1),
            baseline_crop_counts=(8, 8),
        )
        summary = physical_cost_summary([2, 1], geometry)
        self.assertEqual(summary["selected_unique_x8_crop_samples"], 1)
        self.assertEqual(
            summary["processed_x8_crop_samples_including_padding"], 8
        )
        self.assertEqual(summary["padding_x8_crop_samples"], 7)
        self.assertAlmostEqual(summary["logical_unique_crop_cost_ratio"], 17 / 16)
        self.assertEqual(summary["physical_model_sample_cost_ratio"], 1.5)
        self.assertEqual(summary["per_image"][1]["padding_x8_crop_samples"], 0)


class RandomExactCostK2Test(unittest.TestCase):
    def test_signature_has_no_gt_or_confusion_input(self):
        parameters = inspect.signature(random_exact_cost_k2_levels).parameters
        self.assertNotIn("target", parameters)
        self.assertNotIn("confusion", parameters)
        self.assertNotIn("cells", parameters)

    def test_random_walk_is_repeatable_exact_and_exhausts_zero_cost_cells(self):
        geometry = _geometry((0b001, 0b001, 0b010))
        first = random_exact_cost_k2_levels(geometry, [1], 41, 3)
        second = random_exact_cost_k2_levels(geometry, [1], 41, 3)
        np.testing.assert_array_equal(first["levels_by_cell"], second["levels_by_cell"])
        np.testing.assert_array_equal(first["attempts_by_image"], second["attempts_by_image"])
        self.assertEqual(first["closure"]["extra_crop_forwards"], 1)
        self.assertEqual(np.count_nonzero(first["levels_by_cell"][:2] == 2), 2)
        np.testing.assert_array_equal(
            [item["extra_crop_forwards_by_phase"]["x8"]
             for item in first["closure"]["per_image"]],
            [1],
        )

    def test_random_walk_matches_every_image_target(self):
        geometry = _geometry(
            (0b001, 0b010, 0b001, 0b010),
            image_ids=(0, 0, 1, 1),
        )
        result = random_exact_cost_k2_levels(geometry, [1, 2], 9, 0)
        self.assertEqual(
            [item["extra_crop_forwards_by_phase"]["x8"]
             for item in result["closure"]["per_image"]],
            [1, 2],
        )

    def test_unreachable_exact_target_fails_closed(self):
        geometry = _geometry((0b11,), baseline_crop_counts=(8,))
        with self.assertRaisesRegex(RuntimeError, "could not exactly spend"):
            random_exact_cost_k2_levels(
                geometry, [1], 7, 0, max_attempts=3
            )


if __name__ == "__main__":
    unittest.main()
