"""Tests for the frozen D0/D1/D2 output-path decomposition."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.analyze_whu_phase_rescue_decomposition import (
    build_decomposition,
    projected_d1_levels,
)
from scripts.phase_closure_common import build_phase_closure_geometry


def _synthetic_inputs():
    image = {
        "loader_position": 0,
        "sample_name": "synthetic",
        "full_shape_hw": [512, 853],
        "common_bounds": {
            "y_start": 8,
            "y_stop": 504,
            "x_start": 8,
            "x_stop": 845,
        },
        "crop_grid": {
            "crop_count": 2,
            "rows": 1,
            "columns": 2,
            "row_starts": [0],
            "column_starts": [0, 341],
        },
        "confusion": {"k1": [[9, 2], [0, 9]]},
    }
    cell_confusions = (
        {
            "k1": [[5, 1], [0, 4]],
            "k2": [[6, 0], [0, 4]],
            "k4": [[6, 0], [0, 4]],
        },
        {
            "k1": [[4, 1], [0, 5]],
            "k2": [[5, 0], [0, 5]],
            "k4": [[5, 0], [0, 5]],
        },
    )
    cells = []
    for local_id, (window, ownership, confusion) in enumerate(
        zip(
            ((0, 512, 0, 512), (0, 512, 341, 853)),
            ((0, 512, 0, 426), (0, 512, 426, 853)),
            cell_confusions,
            strict=True,
        )
    ):
        cells.append(
            {
                "cell_index": local_id,
                "image_index": 0,
                "sample_name": "synthetic",
                "local_crop_id": local_id,
                "window_yxyx": list(window),
                "ownership_yxyx": list(ownership),
                "geometry_eligible": True,
                "confusion": confusion,
            }
        )
    stage_a = {
        "class_names": ["a", "b"],
        "images": [image],
        "cells": cells,
        "aggregate": {
            "endpoints": {
                "k1": {"full_image": {"confusion": [[9, 2], [0, 9]]}},
                "matched_k2": {
                    "full_image": {"confusion": [[11, 0], [0, 9]]}
                },
            }
        },
    }
    stage_b0 = {
        "exact_cost_a2_oracle": {
            "levels_by_cell": [1, 4],
            "full_image": {"confusion": [[10, 1], [0, 9]]},
        }
    }
    return stage_a, stage_b0


class RescueDecompositionTest(unittest.TestCase):
    def test_d1_is_a_fixed_projection_not_a_rerun_oracle(self):
        stage_a, _ = _synthetic_inputs()
        geometry = build_phase_closure_geometry(stage_a["images"], stage_a["cells"])
        np.testing.assert_array_equal(
            projected_d1_levels([1, 4], geometry), np.array([1, 2])
        )

    def test_decomposition_replays_d0_d1_d2_and_is_additive(self):
        stage_a, stage_b0 = _synthetic_inputs()
        output = build_decomposition(stage_b0, stage_a)
        conditions = output["conditions"]
        self.assertEqual(
            conditions["d1_frozen_k1_k2_projection"]["levels_by_cell"].tolist(),
            [1, 2],
        )
        self.assertEqual(
            conditions["d2_frozen_exact_cost_rescue"]["levels_by_cell"].tolist(),
            [1, 4],
        )
        decomposition = output["decomposition"]
        self.assertAlmostEqual(
            decomposition["d1_minus_d0_pp"]
            + decomposition["d2_minus_d1_pp"],
            decomposition["d2_minus_d0_pp"],
            places=12,
        )
        self.assertTrue(decomposition["d1_and_d2_x8_closure_identical"])

    def test_invalid_d2_confusion_fails_closed(self):
        stage_a, stage_b0 = _synthetic_inputs()
        stage_b0["exact_cost_a2_oracle"]["full_image"]["confusion"] = [
            [9, 2],
            [0, 9],
        ]
        with self.assertRaisesRegex(AssertionError, "D2"):
            build_decomposition(stage_b0, stage_a)


if __name__ == "__main__":
    unittest.main()
