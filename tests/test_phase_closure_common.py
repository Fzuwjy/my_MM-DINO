"""Synthetic checks for Stage-B0 exact shifted-phase crop closure."""

import inspect
import unittest

import numpy as np

from scripts.phase_closure_common import (
    PHASE_NAMES,
    build_phase_closure_geometry,
    closure_masks_from_levels,
    exact_cost_hierarchical_oracle,
    random_priority_control,
    random_priority_walk,
    summarize_closure,
)


class PhaseClosureGeometryTest(unittest.TestCase):
    @staticmethod
    def _two_window_manifest():
        images = [
            {
                "loader_position": 0,
                "sample_name": "synthetic",
                "full_shape_hw": [6, 8],
                "common_bounds": {
                    "y_start": 1,
                    "y_stop": 5,
                    "x_start": 1,
                    "x_stop": 6,
                },
                "crop_grid": {
                    "crop_count": 2,
                    "rows": 1,
                    "columns": 2,
                    "row_starts": [0],
                    "column_starts": [0, 3],
                },
            }
        ]
        cells = [
            {
                "cell_index": 0,
                "image_index": 0,
                "sample_name": "synthetic",
                "local_crop_id": 0,
                "window_yxyx": [0, 6, 0, 5],
                "ownership_yxyx": [0, 6, 0, 4],
                "geometry_eligible": True,
            },
            {
                "cell_index": 1,
                "image_index": 0,
                "sample_name": "synthetic",
                "local_crop_id": 1,
                "window_yxyx": [0, 6, 3, 8],
                "ownership_yxyx": [0, 6, 4, 8],
                "geometry_eligible": True,
            },
        ]
        shifts = {"x8": (0, 1), "y8": (1, 0), "xy8": (1, 1)}
        return images, cells, shifts

    @staticmethod
    def _build(images, cells, shifts):
        return build_phase_closure_geometry(
            images,
            cells,
            phase_shifts=shifts,
            crop_size=(6, 5),
            stride=(4, 3),
        )

    def test_positive_shift_half_open_intersection_and_deduplication(self):
        images, cells, shifts = self._two_window_manifest()
        geometry = self._build(images, cells, shifts)
        # Cell 1's routed x interval [4,6) becomes [5,7).  It must not
        # intersect window 0 ending at 5; using a negative shift would.
        self.assertEqual(geometry["dependencies"]["x8"][1], 0b10)
        self.assertEqual(geometry["dependencies"]["x8"][0], 0b11)

        closure = summarize_closure([2, 2], geometry)
        self.assertEqual(closure["extra_crop_forwards_by_phase"]["x8"], 2)
        self.assertEqual(closure["extra_crop_forwards"], 2)
        self.assertEqual(closure["forward_equivalent_cost"], 2.0)

    def test_k1_k2_k4_phase_nesting_and_order_independence(self):
        images, cells, shifts = self._two_window_manifest()
        geometry = self._build(images, cells, shifts)
        k1 = summarize_closure([1, 1], geometry)
        k2 = summarize_closure([2, 2], geometry)
        k4 = summarize_closure([4, 4], geometry)
        self.assertEqual(k1["extra_crop_forwards"], 0)
        self.assertEqual(k2["extra_crop_forwards_by_phase"], {"x8": 2, "y8": 0, "xy8": 0})
        self.assertEqual(k4["extra_crop_forwards_by_phase"], {"x8": 2, "y8": 2, "xy8": 2})
        first = closure_masks_from_levels([4, 2], geometry)
        second = closure_masks_from_levels(np.array([4, 2]), geometry)
        self.assertEqual(first, second)

    def test_common_support_is_applied_before_dependency_closure(self):
        images, cells, shifts = self._two_window_manifest()
        images[0]["common_bounds"] = {
            "y_start": 1,
            "y_stop": 5,
            "x_start": 1,
            "x_stop": 4,
        }
        cells[1]["geometry_eligible"] = False
        geometry = self._build(images, cells, shifts)
        self.assertEqual(geometry["dependencies"]["x8"][1], 0)
        with self.assertRaisesRegex(ValueError, "outside common-support"):
            summarize_closure([1, 2], geometry)

    def test_same_crop_id_is_distinct_across_phase_and_image(self):
        images, cells, shifts = self._two_window_manifest()
        second_image = {
            **images[0],
            "loader_position": 1,
            "sample_name": "second",
        }
        second_cells = []
        for source in cells:
            copied = dict(source)
            copied["cell_index"] = source["cell_index"] + 2
            copied["image_index"] = 1
            copied["sample_name"] = "second"
            second_cells.append(copied)
        geometry = self._build(images + [second_image], cells + second_cells, shifts)
        closure = summarize_closure([4, 1, 4, 1], geometry)
        self.assertEqual(closure["extra_crop_forwards_by_phase"]["x8"], 4)
        self.assertEqual(closure["extra_crop_forwards_by_phase"]["y8"], 4)
        self.assertEqual(closure["extra_crop_forwards_by_phase"]["xy8"], 4)

    def test_manifest_and_ownership_partition_fail_closed(self):
        images, cells, shifts = self._two_window_manifest()
        wrong_window = [dict(cell) for cell in cells]
        wrong_window[1]["window_yxyx"] = [0, 6, 2, 7]
        with self.assertRaisesRegex(
            ValueError, "source window|share one shape|row-major crop manifest"
        ):
            self._build(images, wrong_window, shifts)

        overlapping = [dict(cell) for cell in cells]
        overlapping[0]["ownership_yxyx"] = [0, 6, 0, 5]
        with self.assertRaisesRegex(ValueError, "midpoint rule|partition the image"):
            self._build(images, overlapping, shifts)


def _cell(index, k1, k2, k4):
    return {
        "cell_index": index,
        "confusion": {
            "k1": np.asarray(k1, dtype=np.int64),
            "k2": np.asarray(k2, dtype=np.int64),
            "k4": np.asarray(k4, dtype=np.int64),
        },
    }


def _manual_geometry(x_dependencies, *, baseline_crop_count=2):
    count = len(x_dependencies)
    return {
        "phase_names": PHASE_NAMES,
        "sample_names": ("synthetic",),
        "baseline_crop_counts": np.array([baseline_crop_count], dtype=np.int64),
        "image_ids": np.zeros(count, dtype=np.int64),
        "eligible": np.ones(count, dtype=bool),
        "cells_by_image": (tuple(range(count)),),
        "dependencies": {
            "x8": tuple(x_dependencies),
            "y8": tuple(1 for _ in range(count)),
            "xy8": tuple(1 for _ in range(count)),
        },
    }


class ExactCostGreedyTest(unittest.TestCase):
    def test_shared_closure_creates_audited_positive_zero_cost_action(self):
        wrong = [[0, 1], [0, 0]]
        correct = [[1, 0], [0, 0]]
        cells = [
            _cell(0, wrong, correct, correct),
            _cell(1, wrong, correct, correct),
            _cell(2, wrong, correct, correct),
        ]
        geometry = _manual_geometry((0b01, 0b01, 0b10))
        full_k1 = np.array([[10, 3], [0, 10]], dtype=np.int64)
        result = exact_cost_hierarchical_oracle(
            cells,
            geometry,
            full_k1,
            extra_crop_cap_by_image=np.array([2]),
            num_classes=2,
        )
        np.testing.assert_array_equal(result["levels_by_cell"], [2, 2, 2])
        self.assertEqual(result["closure"]["extra_crop_forwards"], 2)
        self.assertEqual(
            [action["incremental_extra_crop_forwards"] for action in result["actions"]],
            [1, 0, 1],
        )
        self.assertEqual(
            result["actions"][1]["ranking"],
            "positive_zero_cost_then_absolute_gain",
        )

    def test_nonpositive_action_is_not_taken_even_when_its_crop_is_free(self):
        wrong = [[0, 1], [0, 0]]
        correct = [[1, 0], [0, 0]]
        worse = [[0, 2], [0, 0]]
        partly_correct = [[1, 1], [0, 0]]
        cells = [
            _cell(0, wrong, correct, correct),
            _cell(1, partly_correct, worse, worse),
        ]
        geometry = _manual_geometry((0b01, 0b01))
        full_k1 = np.array([[11, 2], [0, 10]], dtype=np.int64)
        result = exact_cost_hierarchical_oracle(
            cells,
            geometry,
            full_k1,
            extra_crop_cap_by_image=np.array([1]),
            num_classes=2,
        )
        np.testing.assert_array_equal(result["levels_by_cell"], [2, 1])

    def test_nonpositive_k2_bridge_is_not_crossed_to_reach_better_k4(self):
        k1 = [[1, 1], [0, 0]]
        k2 = [[0, 2], [0, 0]]
        k4 = [[2, 0], [0, 0]]
        cells = [_cell(0, k1, k2, k4)]
        geometry = _manual_geometry((0b01,))
        full_k1 = np.array([[11, 1], [0, 10]], dtype=np.int64)
        result = exact_cost_hierarchical_oracle(
            cells,
            geometry,
            full_k1,
            extra_crop_cap_by_image=np.array([2]),
            num_classes=2,
        )
        np.testing.assert_array_equal(result["levels_by_cell"], [1])
        self.assertEqual(result["actions"], ())

    def test_per_image_cap_cannot_be_borrowed(self):
        wrong = [[0, 1], [0, 0]]
        correct = [[1, 0], [0, 0]]
        cells = [_cell(0, wrong, correct, correct), _cell(1, wrong, correct, correct)]
        geometry = {
            "phase_names": PHASE_NAMES,
            "sample_names": ("a", "b"),
            "baseline_crop_counts": np.array([1, 1]),
            "image_ids": np.array([0, 1]),
            "eligible": np.array([True, True]),
            "cells_by_image": ((0,), (1,)),
            "dependencies": {
                "x8": (1, 1),
                "y8": (1, 1),
                "xy8": (1, 1),
            },
        }
        result = exact_cost_hierarchical_oracle(
            cells,
            geometry,
            np.array([[10, 2], [0, 10]]),
            extra_crop_cap_by_image=np.array([0, 1]),
            num_classes=2,
        )
        np.testing.assert_array_equal(result["levels_by_cell"], [1, 2])


class ExactRandomControlTest(unittest.TestCase):
    def test_random_selector_has_no_gt_or_confusion_argument(self):
        parameters = inspect.signature(random_priority_walk).parameters
        self.assertNotIn("cells", parameters)
        self.assertNotIn("confusion", parameters)
        self.assertNotIn("target", parameters)

    def test_random_walk_is_repeatable_hierarchical_and_exact_cost(self):
        geometry = _manual_geometry((0b01, 0b01, 0b10))
        first = random_priority_walk(
            geometry,
            extra_crop_cap_by_image=np.array([1]),
            seed=42,
        )
        second = random_priority_walk(
            geometry,
            extra_crop_cap_by_image=np.array([1]),
            seed=42,
        )
        np.testing.assert_array_equal(first, second)
        closure = summarize_closure(first, geometry)
        self.assertEqual(closure["extra_crop_forwards"], 1)
        self.assertTrue(np.all(np.isin(first, (1, 2, 4))))

    def test_random_walk_fails_closed_when_exact_cost_is_unreachable(self):
        geometry = _manual_geometry((0b11,), baseline_crop_count=2)
        with self.assertRaisesRegex(RuntimeError, "could not exactly spend"):
            random_priority_walk(
                geometry,
                extra_crop_cap_by_image=np.array([1]),
                seed=7,
                max_attempts_per_image=2,
            )

    def test_random_walk_exhausts_zero_cost_actions_after_reaching_cap(self):
        geometry = _manual_geometry((0b01, 0b01))
        levels = random_priority_walk(
            geometry,
            extra_crop_cap_by_image=np.array([1]),
            seed=11,
        )
        # One paid K1->K2 action covers x8 for both cells; the remaining
        # promotion is then an explicitly selected zero-cost action.
        self.assertTrue(np.all(levels >= 2))
        self.assertEqual(summarize_closure(levels, geometry)["extra_crop_forwards"], 1)

    def test_random_control_matches_every_image_realized_cost(self):
        wrong = [[0, 1], [0, 0]]
        correct = [[1, 0], [0, 0]]
        cells = [_cell(0, wrong, correct, correct), _cell(1, wrong, correct, correct)]
        geometry = {
            "phase_names": PHASE_NAMES,
            "sample_names": ("a", "b"),
            "baseline_crop_counts": np.array([1, 1]),
            "image_ids": np.array([0, 1]),
            "eligible": np.array([True, True]),
            "cells_by_image": ((0,), (1,)),
            "dependencies": {
                "x8": (1, 1),
                "y8": (1, 1),
                "xy8": (1, 1),
            },
        }
        control = random_priority_control(
            cells,
            geometry,
            np.array([[10, 2], [0, 10]]),
            extra_crop_cap_by_image=np.array([1, 1]),
            replicates=3,
            seed=9,
        )
        np.testing.assert_array_equal(
            control["replicate_values"]["per_image_extra_crop_forwards"],
            np.ones((3, 2), dtype=np.int64),
        )


if __name__ == "__main__":
    unittest.main()
