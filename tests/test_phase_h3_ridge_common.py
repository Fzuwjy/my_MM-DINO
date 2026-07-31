"""Tests for the sole preregistered H3 three-feature ridge gate."""

from __future__ import annotations

import inspect
import unittest

import numpy as np

import scripts.phase_h3_ridge_common as ridge_common
from scripts.phase_h3_ridge_common import (
    FEATURE_NAMES,
    fit_ridge_gate,
    predict_ridge_gate,
    singleton_k1_to_k2_targets,
)


def _miou(confusion):
    matrix = np.asarray(confusion, dtype=np.int64)
    diagonal = np.diag(matrix)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - diagonal
    supported = union > 0
    return float(np.mean(diagonal[supported] / union[supported]))


def _cell(index, image, features, k1, k2, *, eligible=True, extra_scores=None):
    scores = dict(zip(FEATURE_NAMES, features, strict=True))
    if extra_scores:
        scores.update(extra_scores)
    return {
        "cell_index": index,
        "image_index": image,
        "geometry_eligible": eligible,
        "scores": scores,
        "confusion": {"k1": k1, "k2": k2},
    }


def _dataset(*, constant_boundary=False):
    boundary = (0.25, 0.25, 0.25, 0.25) if constant_boundary else (0.1, 0.2, 0.3, 0.4)
    cells = [
        _cell(0, 0, (0.1, -0.9, boundary[0]), [[4, 1], [1, 4]], [[5, 0], [1, 4]]),
        _cell(1, 0, (0.4, -0.7, boundary[1]), [[3, 2], [1, 4]], [[3, 2], [0, 5]]),
        _cell(2, 1, (0.8, -0.4, boundary[2]), [[2, 3], [2, 3]], [[4, 1], [2, 3]]),
        _cell(3, 1, (1.2, -0.2, boundary[3]), [[3, 2], [2, 3]], [[3, 2], [1, 4]]),
    ]
    # Each reference deliberately contains additional full-image confusion
    # outside the two compact cells, so summing cells would be the wrong base.
    full_k1 = [
        np.asarray([[12, 4], [3, 11]], dtype=np.int64),
        np.asarray([[10, 7], [5, 10]], dtype=np.int64),
    ]
    return cells, full_k1


class SingletonTargetTest(unittest.TestCase):
    def test_uses_fit_pooled_full_image_confusion_and_exact_cell_delta(self):
        cells, full_k1 = _dataset()
        result = singleton_k1_to_k2_targets(
            cells, full_k1, [0], num_classes=2
        )
        np.testing.assert_array_equal(result["fit_cell_indices"], [0, 1])
        np.testing.assert_array_equal(
            result["pooled_full_k1_confusion"], full_k1[0]
        )
        expected = []
        for index in (0, 1):
            k1 = np.asarray(cells[index]["confusion"]["k1"])
            k2 = np.asarray(cells[index]["confusion"]["k2"])
            expected.append(_miou(full_k1[0] + k2 - k1) - _miou(full_k1[0]))
        np.testing.assert_allclose(result["targets"], expected, rtol=0, atol=1e-15)

    def test_heldout_full_confusion_is_neither_read_nor_used(self):
        cells, full_k1 = _dataset()
        # Object() would fail confusion validation if entry 1 were touched.
        first = singleton_k1_to_k2_targets(
            cells, [full_k1[0], object()], [0], num_classes=2
        )
        second = singleton_k1_to_k2_targets(
            cells,
            [full_k1[0], np.asarray([[900, 1], [1, 900]])],
            [0],
            num_classes=2,
        )
        np.testing.assert_array_equal(first["targets"], second["targets"])


class RidgeFitAndPredictTest(unittest.TestCase):
    def test_standardized_lambda_one_solution_is_deterministic(self):
        cells, full_k1 = _dataset()
        first = fit_ridge_gate(cells, full_k1, [0, 1], num_classes=2)
        second = fit_ridge_gate(cells, full_k1, [1, 0], num_classes=2)
        self.assertEqual(first, second)

        target = singleton_k1_to_k2_targets(
            cells, full_k1, [0, 1], num_classes=2
        )["targets"]
        x = np.asarray(
            [[cell["scores"][name] for name in FEATURE_NAMES] for cell in cells]
        )
        mean = x.mean(axis=0)
        scale = x.std(axis=0, ddof=0)
        z = (x - mean) / scale
        expected = np.linalg.solve(
            z.T @ z + np.eye(3), z.T @ (target - target.mean())
        )
        np.testing.assert_allclose(first["feature_mean"], mean, rtol=0, atol=1e-15)
        np.testing.assert_allclose(first["feature_scale"], scale, rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            first["standardized_coefficients"], expected, rtol=0, atol=1e-15
        )
        self.assertFalse(first["intercept_penalized"])
        self.assertTrue(first["target_centered"])
        self.assertEqual(first["ridge_lambda"], 1.0)

    def test_constant_feature_has_unit_scale_zero_weight_and_finite_prediction(self):
        cells, full_k1 = _dataset(constant_boundary=True)
        model = fit_ridge_gate(cells, full_k1, [0], num_classes=2)
        self.assertEqual(model["constant_feature_mask"], (False, False, True))
        self.assertEqual(model["feature_scale"][2], 1.0)
        self.assertEqual(model["standardized_coefficients"][2], 0.0)
        result = predict_ridge_gate(model, cells, [1])
        self.assertEqual(result["predicted_cell_indices"], (2, 3))
        self.assertTrue(
            all(np.isfinite(result["scores_by_cell"][index]) for index in (2, 3))
        )
        self.assertEqual(result["scores_by_cell"][:2], (None, None))

    def test_prediction_rejects_fit_overlap_and_does_not_read_heldout_confusion(self):
        cells, full_k1 = _dataset()
        model = fit_ridge_gate(cells, [full_k1[0], object()], [0], num_classes=2)
        for index in (2, 3):
            cells[index]["confusion"] = object()
        result = predict_ridge_gate(model, cells, [1])
        self.assertEqual(result["predicted_cell_indices"], (2, 3))
        with self.assertRaisesRegex(ValueError, "overlap"):
            predict_ridge_gate(model, cells, [0])

    def test_oracle_score_fields_cannot_change_fit_or_prediction(self):
        cells, full_k1 = _dataset()
        baseline = fit_ridge_gate(cells, full_k1, [0], num_classes=2)
        for cell in cells:
            cell["scores"]["oracle_singleton_global_miou_gain"] = 1e100
            cell["scores"]["oracle_singleton_per_image_miou_gain"] = -1e100
        changed = fit_ridge_gate(cells, full_k1, [0], num_classes=2)
        self.assertEqual(baseline, changed)
        self.assertEqual(
            predict_ridge_gate(baseline, cells, [1]),
            predict_ridge_gate(changed, cells, [1]),
        )

    def test_lambda_is_not_a_tunable_api(self):
        cells, full_k1 = _dataset()
        with self.assertRaisesRegex(ValueError, "fixed at 1.0"):
            fit_ridge_gate(
                cells, full_k1, [0], num_classes=2, ridge_lambda=0.1
            )


class IsolationContractTest(unittest.TestCase):
    def test_public_api_has_no_stage_b0_or_action_level_input(self):
        for function in (
            singleton_k1_to_k2_targets,
            fit_ridge_gate,
            predict_ridge_gate,
        ):
            names = set(inspect.signature(function).parameters)
            self.assertFalse(
                names.intersection(
                    {"stage_b0", "stage_b0_levels", "levels_by_cell", "action_map"}
                )
            )
        source = inspect.getsource(ridge_common)
        self.assertNotIn("phase_closure_common", source)
        self.assertNotIn("phase_utility_common", source)


if __name__ == "__main__":
    unittest.main()
