"""Synthetic correctness checks for Stage-B1a sparse phase replay."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.phase_closure_common import build_phase_closure_geometry
from scripts.phase_sparse_replay_common import (
    accumulate_phase_crops,
    aligned_region,
    analytic_count_map,
    array_sha256,
    compose_policy_logits,
    endpoint_levels,
    expected_phase_crop_ids,
    expected_phase_crop_keys,
    frozen_middle_levels,
    policy_level_map,
    validate_observed_crop_keys,
    validate_live_phase_on_routed_pixels,
    validate_phase_replay_on_routed_pixels,
)


def _two_window_geometry():
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
    return build_phase_closure_geometry(
        images,
        cells,
        phase_shifts={"x8": (0, 1), "y8": (1, 0), "xy8": (1, 1)},
        crop_size=(6, 5),
        stride=(4, 3),
    )


def _crop_logits(crop_id: int) -> np.ndarray:
    y, x = np.indices((6, 5), dtype=np.float32)
    first = np.float32(crop_id * 100) + y * np.float32(10) + x
    return np.stack((first, -first - np.float32(1)), axis=0).astype(np.float32)


def _independent_dense(raw: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    score_sum = np.zeros((2, 6, 8), dtype=np.float32)
    count = np.zeros((6, 8), dtype=np.int16)
    for crop_id, (x0, x1) in enumerate(((0, 5), (3, 8))):
        score_sum[:, :, x0:x1] += raw[crop_id]
        count[:, x0:x1] += 1
    return {"sum_logits": score_sum, "count_mat": count}


class SparseAccumulatorTest(unittest.TestCase):
    def test_sparse_accumulator_matches_independent_dense_on_routed_pixels(self):
        geometry = _two_window_geometry()
        levels = np.array([1, 2], dtype=np.int64)
        self.assertEqual(expected_phase_crop_ids(levels, geometry, 0)["x8"], (1,))
        raw = {0: _crop_logits(0), 1: _crop_logits(1)}
        sparse = accumulate_phase_crops(
            raw, (1,), geometry, 0, num_classes=2
        )
        dense = _independent_dense(raw)
        level_map = policy_level_map(levels, geometry, 0)
        audit = validate_phase_replay_on_routed_pixels(
            sparse,
            dense,
            level_map,
            phase_name="x8",
            shift=(0, 1),
            common_bounds=(1, 5, 1, 6),
        )
        self.assertTrue(audit["count_mat_equal"])
        self.assertTrue(audit["sum_logits_equal"])
        self.assertTrue(audit["mean_logits_equal"])
        self.assertGreater(audit["routed_pixels"], 0)
        self.assertTrue(
            np.any(sparse["count_mat"] != dense["count_mat"]),
            "the sparse replay accidentally degenerated into a dense replay",
        )
        np.testing.assert_array_equal(
            sparse["count_mat"], analytic_count_map(geometry, 0, (1,))
        )

    def test_replay_fails_closed_on_bad_crop_inputs_and_bad_count(self):
        geometry = _two_window_geometry()
        raw = {0: _crop_logits(0), 1: _crop_logits(1)}
        with self.assertRaisesRegex(KeyError, "missing raw logits"):
            accumulate_phase_crops({0: raw[0]}, (1,), geometry, 0, num_classes=2)
        with self.assertRaisesRegex(ValueError, "unique and strictly increasing"):
            accumulate_phase_crops(raw, (1, 1), geometry, 0, num_classes=2)
        with self.assertRaisesRegex(TypeError, "float32"):
            accumulate_phase_crops(
                {1: raw[1].astype(np.float64)}, (1,), geometry, 0, num_classes=2
            )
        corrupted = raw[1].copy()
        corrupted[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            accumulate_phase_crops({1: corrupted}, (1,), geometry, 0, num_classes=2)

        maximum = np.full((2, 6, 5), np.finfo(np.float32).max, dtype=np.float32)
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            accumulate_phase_crops(
                {0: maximum, 1: maximum}, (0, 1), geometry, 0, num_classes=2
            )

        sparse = accumulate_phase_crops(raw, (1,), geometry, 0, num_classes=2)
        dense = _independent_dense(raw)
        sparse["count_mat"][2, 5] = 2
        with self.assertRaisesRegex(AssertionError, "count differs"):
            validate_phase_replay_on_routed_pixels(
                sparse,
                dense,
                policy_level_map([1, 2], geometry, 0),
                phase_name="x8",
                shift=(0, 1),
                common_bounds=(1, 5, 1, 6),
            )

    def test_live_validation_keeps_exact_count_but_allows_frozen_logit_tolerance(self):
        geometry = _two_window_geometry()
        raw = {0: _crop_logits(0), 1: _crop_logits(1)}
        dense = _independent_dense(raw)
        sparse = accumulate_phase_crops(raw, (1,), geometry, 0, num_classes=2)
        # Perturb only a routed value, within the frozen live tolerance.
        sparse["sum_logits"][0, 2, 5] += np.float32(1e-6)
        audit = validate_live_phase_on_routed_pixels(
            sparse,
            dense,
            policy_level_map([1, 2], geometry, 0),
            phase_name="x8",
            shift=(0, 1),
            common_bounds=(1, 5, 1, 6),
        )
        self.assertTrue(audit["phase_mean_logits_close"])
        sparse["sum_logits"][0, 2, 5] += np.float32(1e-2)
        with self.assertRaisesRegex(AssertionError, "tolerance"):
            validate_live_phase_on_routed_pixels(
                sparse,
                dense,
                policy_level_map([1, 2], geometry, 0),
                phase_name="x8",
                shift=(0, 1),
                common_bounds=(1, 5, 1, 6),
            )


class AlignmentAndMaskTest(unittest.TestCase):
    def test_inverse_alignment_uses_positive_asymmetric_shift_without_wrap(self):
        y, x = np.indices((7, 9), dtype=np.int64)
        encoded = y * 100 + x
        aligned = aligned_region(encoded, (1, 2), (1, 5, 1, 6))
        np.testing.assert_array_equal(aligned, encoded[2:6, 3:8])
        self.assertEqual(aligned[0, 0], 203)
        self.assertEqual(aligned[-1, -1], 507)
        with self.assertRaisesRegex(ValueError, "outside"):
            aligned_region(encoded, (3, 4), (1, 5, 1, 6))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            aligned_region(encoded, (-1, 0), (1, 5, 1, 6))

    @staticmethod
    def _constant_accumulation(value: float) -> dict[str, np.ndarray]:
        score_sum = np.zeros((2, 6, 8), dtype=np.float32)
        score_sum[0] = np.float32(value)
        score_sum[1] = np.float32(-value)
        return {
            "sum_logits": score_sum,
            "count_mat": np.ones((6, 8), dtype=np.int16),
        }

    def test_ownership_mask_blocks_closure_spill_and_clips_common_support(self):
        geometry = _two_window_geometry()
        normal = np.zeros((2, 6, 8), dtype=np.float32)
        phases = {
            "x8": self._constant_accumulation(1),
            "y8": self._constant_accumulation(10),
            "xy8": self._constant_accumulation(100),
        }
        k4 = compose_policy_logits(normal, phases, [1, 4], geometry, 0)
        logits = k4["logits"][0]
        np.testing.assert_array_equal(logits[1:5, 1:4], 0)  # K1 despite spill
        np.testing.assert_array_equal(logits[1:5, 4:6], 111)  # K4 ownership
        np.testing.assert_array_equal(logits[0], 0)  # outside common support
        self.assertEqual(logits[2, 3], 0)  # half-open midpoint left side
        self.assertEqual(logits[2, 4], 111)  # midpoint belongs to right cell

        k2 = compose_policy_logits(normal, phases, [1, 2], geometry, 0)
        np.testing.assert_array_equal(k2["logits"][0, 1:5, 4:6], 1)
        np.testing.assert_array_equal(k2["logits"][0, 1:5, 1:4], 0)

    def test_composition_can_skip_only_diagnostic_digests(self):
        geometry = _two_window_geometry()
        normal = np.zeros((2, 6, 8), dtype=np.float32)
        phases = {
            "x8": self._constant_accumulation(1),
            "y8": self._constant_accumulation(10),
            "xy8": self._constant_accumulation(100),
        }
        reference = compose_policy_logits(normal, phases, [1, 4], geometry, 0)
        unhashed = compose_policy_logits(
            normal,
            phases,
            [1, 4],
            geometry,
            0,
            include_digests=False,
        )

        self.assertNotIn("logits_sha256", unhashed)
        self.assertNotIn("prediction_sha256", unhashed)
        np.testing.assert_array_equal(unhashed["logits"], reference["logits"])
        np.testing.assert_array_equal(
            unhashed["prediction"], reference["prediction"]
        )
        np.testing.assert_array_equal(unhashed["level_map"], reference["level_map"])
        self.assertEqual(unhashed["phase_audit"], reference["phase_audit"])

    def test_composition_rejects_nonfinite_or_malformed_accumulations(self):
        geometry = _two_window_geometry()
        normal = np.zeros((2, 6, 8), dtype=np.float32)
        normal[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "normal_logits must be finite"):
            compose_policy_logits(normal, {}, [1, 1], geometry, 0)

        normal[0, 0, 0] = 0
        malformed = self._constant_accumulation(1)
        malformed["sum_logits"] = malformed["sum_logits"].astype(np.float64)
        with self.assertRaisesRegex(TypeError, "float32"):
            compose_policy_logits(normal, {"x8": malformed}, [1, 2], geometry, 0)

        nonfinite = self._constant_accumulation(1)
        nonfinite["sum_logits"][0, 2, 5] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            compose_policy_logits(normal, {"x8": nonfinite}, [1, 2], geometry, 0)

    def test_four_synthetic_policy_anchors_are_bit_exact(self):
        geometry = _two_window_geometry()
        y, x = np.indices((6, 8), dtype=np.float32)
        normal = np.stack((y + x, -y - x - np.float32(0.25))).astype(np.float32)
        phase_raw = {}
        for phase_index, phase_name in enumerate(("x8", "y8", "xy8"), start=1):
            adjustment = np.asarray(
                [phase_index * 3, -phase_index * 5], dtype=np.float32
            )[:, None, None]
            phase_raw[phase_name] = {
                crop_id: np.ascontiguousarray(_crop_logits(crop_id) + adjustment)
                for crop_id in (0, 1)
            }

        policies = {
            "k1": np.array([1, 1], dtype=np.int64),
            "k2": np.array([2, 2], dtype=np.int64),
            "k4": np.array([4, 4], dtype=np.int64),
            "middle": np.array([1, 4], dtype=np.int64),
        }
        common_bounds = (1, 5, 1, 6)
        y0, y1, x0, x1 = common_bounds
        for policy_name, levels in policies.items():
            with self.subTest(policy=policy_name):
                level_map = policy_level_map(levels, geometry, 0)
                sparse_accumulations = {}
                dense_accumulations = {}
                for phase_name in ("x8", "y8", "xy8"):
                    crop_ids = expected_phase_crop_ids(levels, geometry, 0)[phase_name]
                    if not crop_ids:
                        continue
                    sparse = accumulate_phase_crops(
                        phase_raw[phase_name], crop_ids, geometry, 0, num_classes=2
                    )
                    dense = _independent_dense(phase_raw[phase_name])
                    audit = validate_phase_replay_on_routed_pixels(
                        sparse,
                        dense,
                        level_map,
                        phase_name=phase_name,
                        shift=tuple(geometry["phase_shifts"][phase_name]),
                        common_bounds=common_bounds,
                    )
                    self.assertTrue(audit["mean_logits_equal"])
                    sparse_accumulations[phase_name] = sparse
                    dense_accumulations[phase_name] = dense

                replayed = compose_policy_logits(
                    normal, sparse_accumulations, levels, geometry, 0
                )
                expected = normal.copy()
                expected_common = expected[:, y0:y1, x0:x1]
                level_common = level_map[y0:y1, x0:x1]
                for phase_name, dense in dense_accumulations.items():
                    dy, dx = geometry["phase_shifts"][phase_name]
                    dense_sum = dense["sum_logits"][
                        :, y0 + dy : y1 + dy, x0 + dx : x1 + dx
                    ]
                    dense_count = dense["count_mat"][
                        y0 + dy : y1 + dy, x0 + dx : x1 + dx
                    ]
                    dense_mean = dense_sum / dense_count[None]
                    required = (
                        level_common >= 2
                        if phase_name == "x8"
                        else level_common == 4
                    )
                    expected_common[:, required] += dense_mean[:, required]
                self.assertEqual(replayed["logits"].dtype, np.float32)
                np.testing.assert_array_equal(replayed["logits"], expected)
                np.testing.assert_array_equal(
                    replayed["prediction"], expected.argmax(axis=0)
                )


class FrozenPolicyTest(unittest.TestCase):
    @staticmethod
    def _formal_first_image_geometry():
        row_starts = [0, 341, 682, 1023, 1364, 1705, 2046, 2387, 2728, 3069, 3192]
        column_starts = [
            0, 341, 682, 1023, 1364, 1705, 2046, 2387,
            2728, 3069, 3410, 3751, 4092, 4433, 4774, 5044,
        ]
        height, width, crop = 3704, 5556, 512

        def intervals(length, starts):
            boundaries = [0]
            boundaries.extend(
                (left + crop + right) // 2
                for left, right in zip(starts[:-1], starts[1:], strict=True)
            )
            boundaries.append(length)
            return list(zip(boundaries[:-1], boundaries[1:], strict=True))

        rows = intervals(height, row_starts)
        columns = intervals(width, column_starts)
        image = {
            "loader_position": 0,
            "sample_name": "NH49E001014",
            "full_shape_hw": [height, width],
            "common_bounds": {
                "y_start": 512, "y_stop": 3176,
                "x_start": 512, "x_stop": 5028,
            },
            "crop_grid": {
                "crop_count": 176,
                "rows": 11,
                "columns": 16,
                "row_starts": row_starts,
                "column_starts": column_starts,
            },
        }
        cells = []
        local_id = 0
        for row_index, y0 in enumerate(row_starts):
            for column_index, x0 in enumerate(column_starts):
                ownership = (
                    rows[row_index][0], rows[row_index][1],
                    columns[column_index][0], columns[column_index][1],
                )
                routed_y0 = max(ownership[0], 512)
                routed_y1 = min(ownership[1], 3176)
                routed_x0 = max(ownership[2], 512)
                routed_x1 = min(ownership[3], 5028)
                cells.append(
                    {
                        "cell_index": local_id,
                        "image_index": 0,
                        "sample_name": "NH49E001014",
                        "local_crop_id": local_id,
                        "window_yxyx": [y0, y0 + crop, x0, x0 + crop],
                        "ownership_yxyx": list(ownership),
                        "geometry_eligible": (
                            routed_y0 < routed_y1 and routed_x0 < routed_x1
                        ),
                    }
                )
                local_id += 1
        return build_phase_closure_geometry([image], cells)

    def test_frozen_middle_subset_has_exact_preregistered_crop_closure(self):
        geometry = self._formal_first_image_geometry()
        levels = frozen_middle_levels(geometry)
        self.assertEqual(
            {value: int(np.count_nonzero(levels == value)) for value in (1, 2, 4)},
            {1: 171, 2: 2, 4: 3},
        )
        ids = expected_phase_crop_ids(levels, geometry, 0)
        self.assertEqual(
            ids["x8"],
            (17, 18, 33, 34, 54, 55, 56, 57, 70, 71, 72, 73, 86, 87, 88, 89, 102, 103, 104, 105),
        )
        expected_y = (17, 18, 33, 34, 55, 56, 57, 70, 71, 72, 73, 86, 87, 88, 89, 102, 103, 104)
        self.assertEqual(ids["y8"], expected_y)
        self.assertEqual(ids["xy8"], expected_y)
        self.assertEqual(sum(len(value) for value in ids.values()), 56)
        self.assertEqual(
            array_sha256(np.asarray(ids["x8"], dtype="<i8")),
            "8de2bbe6d675922f4d33c9d12a1a843279db3f8dac67b0c8f845494029a31153",
        )
        self.assertEqual(
            array_sha256(np.asarray(ids["y8"], dtype="<i8")),
            "ef647420642775d70fe69afb3c43a7b260779b370bcc8bb95e777b0ed610a819",
        )

    def test_formal_first_image_endpoint_closures_are_frozen(self):
        geometry = self._formal_first_image_geometry()
        empty_sha = (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        common_sha = (
            "6d8dcf68a0b0ea6067aa6f408dafd4d0cbc5d5ae3d18de9b8d7bcdb4f903cb85"
        )
        for level, expected_counts in (
            (1, {"x8": 0, "y8": 0, "xy8": 0}),
            (2, {"x8": 126, "y8": 0, "xy8": 0}),
            (4, {"x8": 126, "y8": 126, "xy8": 126}),
        ):
            with self.subTest(level=level):
                ids = expected_phase_crop_ids(endpoint_levels(geometry, level), geometry, 0)
                self.assertEqual(
                    {name: len(values) for name, values in ids.items()}, expected_counts
                )
                for phase_name, values in ids.items():
                    expected_sha = common_sha if expected_counts[phase_name] else empty_sha
                    self.assertEqual(
                        array_sha256(np.asarray(values, dtype="<i8")), expected_sha
                    )

    def test_endpoint_and_observed_key_order_is_canonical(self):
        geometry = _two_window_geometry()
        k1 = endpoint_levels(geometry, 1)
        k2 = endpoint_levels(geometry, 2)
        k4 = endpoint_levels(geometry, 4)
        self.assertEqual(expected_phase_crop_keys(k1, geometry), ())
        self.assertEqual(
            expected_phase_crop_keys(k2, geometry),
            ((0, "x8", 0), (0, "x8", 1)),
        )
        self.assertEqual(
            expected_phase_crop_keys(k4, geometry),
            (
                (0, "x8", 0), (0, "x8", 1),
                (0, "y8", 0), (0, "y8", 1),
                (0, "xy8", 0), (0, "xy8", 1),
            ),
        )
        expected = expected_phase_crop_keys(k4, geometry)
        self.assertTrue(validate_observed_crop_keys(expected, expected)["equal"])
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            validate_observed_crop_keys(expected + expected[:1], expected)
        with self.assertRaisesRegex(AssertionError, "differs"):
            validate_observed_crop_keys(tuple(reversed(expected)), expected)
        with self.assertRaisesRegex(ValueError, "canonical order"):
            validate_observed_crop_keys(
                tuple(reversed(expected)), tuple(reversed(expected))
            )


if __name__ == "__main__":
    unittest.main()
