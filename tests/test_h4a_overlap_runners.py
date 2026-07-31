"""Unit checks for H4-A artifact binding and decision semantics."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_whu_h4a_overlap_screen import (
    SCORE_SPECS,
    _join_scores,
    _primary_decision,
    _spearman,
)
from scripts.evaluate_whu_h4a_overlap_statistics import (
    _confusion,
    _resolve_cache_path,
    _stage_cells_by_image,
)
from scripts.evaluate_whu_h4ab_response_statistics import (
    _compare_standalone_h4a,
    _validate_k2_endpoint,
)
from scripts.phase_overlap_common import array_sha256


def _stage_a() -> dict:
    return {
        "images": [{"ownership_cell_count": 2}],
        "cells": [
            {
                "cell_index": 0,
                "image_index": 0,
                "local_crop_id": 0,
                "geometry_eligible": True,
                "scores": {"k1_entropy": 0.1},
            },
            {
                "cell_index": 1,
                "image_index": 0,
                "local_crop_id": 1,
                "geometry_eligible": False,
                "scores": {"k1_entropy": None},
            },
        ],
    }


def _statistics() -> dict:
    keys = [key for _, key in SCORE_SPECS]
    return {
        "artifact_type": "whu_h4a_k1_overlap_disagreement",
        "schema_version": 1,
        "status": "PASS",
        "scope": "full-test",
        "execution_mode": "live-k1",
        "evaluated_images": 1,
        "full_test_length": 1,
        "source_stage_a": {"sha256": "a" * 64},
        "cells": [
            {
                "cell_index": 0,
                "image_index": 0,
                "local_crop_id": 0,
                "geometry_eligible": True,
                "scores": {key: 0.2 + index for index, key in enumerate(keys)},
            },
            {
                "cell_index": 1,
                "image_index": 0,
                "local_crop_id": 1,
                "geometry_eligible": False,
                "scores": {key: 0.3 + index for index, key in enumerate(keys)},
            },
        ],
    }


def _primary(
    *, strong: bool, candidate: float, random_delta_pp: float, selected_q: float
) -> dict:
    return {
        "strong_gate": {
            "formal_random_control": True,
            "known_checks_passed": strong,
            "observed": {
                "candidate_minus_random_p95_pp": random_delta_pp,
                "candidate_minus_matched_k2_pp": -0.01,
            },
        },
        "cross_fitted_policy": {
            "folds": [{"selected_q": selected_q}],
            "full_image": {"miou": candidate},
        },
    }


class H4AStatisticsHelpersTests(unittest.TestCase):
    def test_confusion_ignores_invalid_labels(self):
        prediction = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        label = np.asarray([[0, 1, 2], [3, 255, -1]], dtype=np.int64)
        confusion = _confusion(prediction, label)
        self.assertEqual(confusion.sum(), 4)
        for class_index in range(4):
            self.assertEqual(confusion[class_index, class_index], 1)

    def test_stage_cells_are_grouped_and_require_contiguous_indices(self):
        grouped = _stage_cells_by_image(_stage_a())
        self.assertEqual(len(grouped), 1)
        self.assertEqual([cell["cell_index"] for cell in grouped[0]], [0, 1])
        broken = _stage_a()
        broken["cells"][1]["cell_index"] = 9
        with self.assertRaisesRegex(ValueError, "global cell indices"):
            _stage_cells_by_image(broken)

    def test_cache_path_must_remain_below_manifest_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "crop.npy"
            child.write_bytes(b"x")
            self.assertEqual(
                _resolve_cache_path(root, "crop.npy", field="crop"),
                child.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                _resolve_cache_path(root, "../escape.npy", field="crop")


class H4AScreenBindingTests(unittest.TestCase):
    def test_join_hides_ineligible_response_values(self):
        joined = _join_scores(
            _stage_a(), _statistics(), stage_a_sha="a" * 64
        )
        for _, key in SCORE_SPECS:
            self.assertIsInstance(joined["cells"][0]["scores"][key], float)
            self.assertIsNone(joined["cells"][1]["scores"][key])

    def test_join_rejects_cell_identity_drift(self):
        statistics = _statistics()
        statistics["cells"][0]["local_crop_id"] = 3
        with self.assertRaisesRegex(ValueError, "identity differs"):
            _join_scores(_stage_a(), statistics, stage_a_sha="a" * 64)

    def test_report_only_spearman_handles_ties_deterministically(self):
        value = _spearman(
            np.asarray([1.0, 1.0, 2.0, 3.0]),
            np.asarray([0.0, 0.0, 1.0, 2.0]),
        )
        self.assertAlmostEqual(value, 1.0)


class H4APrimaryDecisionTests(unittest.TestCase):
    def test_strong_gate_authorizes_only_live_confirmation(self):
        decision = _primary_decision(
            _primary(strong=True, candidate=0.543, random_delta_pp=0.02, selected_q=0.5)
        )
        self.assertEqual(
            decision["outcome"],
            "PROVISIONAL_GO_H4A_ONE_LIVE_STRUCTURE_LATENCY_CONFIRMATION",
        )
        self.assertTrue(decision["live_h4a_structure_latency_authorized"])

    def test_mechanism_gate_can_authorize_h4b_without_claiming_h4a(self):
        decision = _primary_decision(
            _primary(
                strong=False,
                candidate=0.5425,
                random_delta_pp=0.01,
                selected_q=1.0 / 3.0,
            )
        )
        self.assertEqual(
            decision["outcome"],
            "KEEP_H4A_MECHANISM_SIGNAL_RUN_INDEPENDENT_H4B",
        )
        self.assertTrue(decision["h4b_implementation_authorized"])
        self.assertFalse(decision["h4a_confirmed"])

    def test_failed_primary_stops_k1_only_but_does_not_block_independent_h4b(self):
        decision = _primary_decision(
            _primary(
                strong=False,
                candidate=0.5420,
                random_delta_pp=-0.01,
                selected_q=0.5,
            )
        )
        self.assertEqual(
            decision["outcome"], "STOP_H4A_K1_OVERLAP_RESPONSE_NO_SIGNAL"
        )
        self.assertTrue(decision["h4b_implementation_authorized"])
        self.assertTrue(decision["h4b_independent_screen_pre_authorized"])


class H4ABCombinedRunnerTests(unittest.TestCase):
    def test_matched_k2_endpoint_uses_aligned_common_logits(self):
        k1_prediction = np.zeros((2, 3), dtype=np.int64)
        k1_logits = np.zeros((2, 2, 2), dtype=np.float32)
        k1_logits[0] = 1.0
        kx_logits = np.zeros_like(k1_logits)
        kx_logits[1, 0, 0] = 3.0
        expected = k1_prediction.copy()
        expected[:, 1:3] = (k1_logits + kx_logits).argmax(axis=0)
        label = expected.copy()
        reference = {
            "prediction_sha256": {"matched_k2": array_sha256(expected)},
            "confusion": {
                "matched_k2": [[5, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]],
            },
        }
        result = _validate_k2_endpoint(
            k1_prediction=k1_prediction,
            k1_logits=k1_logits,
            kx_logits=kx_logits,
            common_bounds=(0, 2, 1, 3),
            reference=reference,
            label=label,
        )
        self.assertTrue(result["prediction_equal"])
        wrong = kx_logits.copy()
        wrong[1, 0, 0] = 0.0
        with self.assertRaisesRegex(AssertionError, "SHA differs"):
            _validate_k2_endpoint(
                k1_prediction=k1_prediction,
                k1_logits=k1_logits,
                kx_logits=wrong,
                common_bounds=(0, 2, 1, 3),
                reference=reference,
                label=label,
            )

    def test_combined_smoke_requires_exact_standalone_h4a_scores(self):
        cells = [
            {
                "cell_index": 0,
                "scores": {"overlap_jsd_normalized": 0.25},
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "standalone.json"
            path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
            result = _compare_standalone_h4a(path, cells)
            self.assertTrue(result["bitwise_numeric_equality"])
            changed = [
                {
                    "cell_index": 0,
                    "scores": {"overlap_jsd_normalized": 0.25001},
                }
            ]
            with self.assertRaisesRegex(AssertionError, "differs"):
                _compare_standalone_h4a(path, changed)


if __name__ == "__main__":
    unittest.main()
