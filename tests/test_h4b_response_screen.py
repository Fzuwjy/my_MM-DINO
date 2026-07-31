"""Synthetic tests for frozen H4-B K2-first response arbitration."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_whu_h4b_k2_response_screen import (
    Q_VALUES,
    _decision,
    _join_response_scores,
    _validate_combined_collection_ledger,
    cross_fit_response_policy,
    matched_action_count_random,
    response_ranked_levels,
)


def _confusion(value: int = 1) -> list[list[int]]:
    return [[value, 0], [0, value]]


def _stage_and_geometry():
    cells = []
    images = []
    for image_index in range(2):
        image_cells = []
        for local_index in range(3):
            global_index = image_index * 3 + local_index
            eligible = local_index < 2
            cells.append(
                {
                    "cell_index": global_index,
                    "image_index": image_index,
                    "local_crop_id": local_index,
                    "geometry_eligible": eligible,
                    "confusion": {
                        "k1": _confusion(),
                        "k2": _confusion(),
                        "k4": _confusion(),
                    },
                }
            )
            image_cells.append(global_index)
        images.append(
            {
                "sample_name": f"image-{image_index}",
                "confusion": {"k1": _confusion(3)},
            }
        )
    geometry = {
        "eligible": np.asarray([True, True, False, True, True, False]),
        "cells_by_image": ((0, 1, 2), (3, 4, 5)),
    }
    stage = {
        "images": images,
        "cells": cells,
        "class_names": ["a", "b"],
        "aggregate": {
            "endpoints": {"k1": {"full_image": {"confusion": _confusion(6)}}}
        },
    }
    return stage, geometry


class ResponseRankingTests(unittest.TestCase):
    def test_quota_is_per_image_and_ties_use_lower_global_index(self):
        _, geometry = _stage_and_geometry()
        scores = np.asarray([5.0, 5.0, -np.inf, 1.0, 2.0, -np.inf])
        levels = response_ranked_levels(scores, 0.5, geometry)
        np.testing.assert_array_equal(levels, [2, 1, 1, 1, 2, 1])

    def test_crossfit_exact_tie_prefers_higher_q(self):
        stage, geometry = _stage_and_geometry()
        scores = np.asarray([2.0, 1.0, -np.inf, 2.0, 1.0, -np.inf])
        policy = cross_fit_response_policy(stage, geometry, scores)
        self.assertTrue(
            all(fold["selected_q"] == max(Q_VALUES) for fold in policy["folds"])
        )
        self.assertEqual(policy["folds"][0]["fit_images"], [1])
        self.assertEqual(policy["folds"][1]["fit_images"], [0])


class MatchedActionRandomTests(unittest.TestCase):
    def test_random_is_repeatable_and_matches_each_image_output_count(self):
        stage, geometry = _stage_and_geometry()
        target = np.asarray([2, 1, 1, 1, 2, 1], dtype=np.int64)
        first = matched_action_count_random(
            stage, geometry, target, replicates=4, seed=19
        )
        second = matched_action_count_random(
            stage, geometry, target, replicates=4, seed=19
        )
        self.assertEqual(first["target_k2_output_cells_by_image"], [1, 1])
        self.assertEqual(
            first["first_replicate_levels_by_cell"],
            second["first_replicate_levels_by_cell"],
        )
        np.testing.assert_array_equal(
            first["replicate_values"], second["replicate_values"]
        )
        levels = np.asarray(first["first_replicate_levels_by_cell"])
        self.assertEqual(np.count_nonzero(levels[:3] == 2), 1)
        self.assertEqual(np.count_nonzero(levels[3:] == 2), 1)


class H4BDecisionTests(unittest.TestCase):
    @staticmethod
    def _endpoints():
        return {
            "k1": {"miou": 0.50},
            "matched_k2": {"miou": 0.51},
            "k4": {"miou": 0.52},
        }

    @staticmethod
    def _cost():
        return {"physical_model_sample_cost_ratio": 1.727272727}

    def test_subthreshold_but_real_primary_signal_only_authorizes_h4c(self):
        decision = _decision(
            candidate_miou=0.5102,
            endpoints=self._endpoints(),
            physical_cost=self._cost(),
            random_control={"miou": {"p95": 0.5101}},
            formal=True,
        )
        self.assertEqual(decision["outcome"], "GO_H4C_RESPONSE_SIGNAL_ONLY")
        self.assertTrue(decision["h4c_implementation_authorized"])
        self.assertFalse(decision["live_structure_confirmation_authorized"])

    def test_known_strong_pass_requires_live_confirmation_before_h4c(self):
        decision = _decision(
            candidate_miou=0.514,
            endpoints=self._endpoints(),
            physical_cost=self._cost(),
            random_control={"miou": {"p95": 0.513}},
            formal=True,
        )
        self.assertEqual(
            decision["outcome"],
            "PROVISIONAL_GO_H4B_LIVE_STRUCTURE_CONFIRMATION",
        )
        self.assertTrue(decision["live_structure_confirmation_authorized"])
        self.assertFalse(decision["h4c_implementation_authorized"])
        self.assertTrue(
            decision["h4c_after_live_structure_confirmation_eligible"]
        )

    def test_no_primary_signal_stops_h4b_and_h4c(self):
        decision = _decision(
            candidate_miou=0.5099,
            endpoints=self._endpoints(),
            physical_cost=self._cost(),
            random_control={"miou": {"p95": 0.5098}},
            formal=True,
        )
        self.assertEqual(decision["outcome"], "STOP_H4B_SIMPLE_K2_RESPONSE_RULES")
        self.assertFalse(decision["h4c_implementation_authorized"])


class CombinedCollectionBindingTests(unittest.TestCase):
    @staticmethod
    def _statistics():
        return {
            "collection_cost": {
                "baseline_normal_real_and_model_samples": 3520,
                "x8_selected_real_crops": 2520,
                "x8_processed_model_samples_including_padding": 2560,
                "physical_model_sample_cost_ratio": 6080 / 3520,
            },
            "images": [
                {
                    "image_index": index,
                    "endpoint_validation": {
                        "prediction_equal": True,
                        "stage_a_confusion_equal": True,
                    },
                    "matched_k2_endpoint_validation": {
                        "prediction_equal": True,
                        "stage_a_confusion_equal": True,
                    },
                }
                for index in range(2)
            ],
        }

    def test_frozen_ledger_and_both_endpoints_are_required(self):
        statistics = self._statistics()
        _validate_combined_collection_ledger(statistics, image_count=2)
        statistics["images"][1]["matched_k2_endpoint_validation"][
            "prediction_equal"
        ] = False
        with self.assertRaisesRegex(ValueError, "image_1_endpoint_validation"):
            _validate_combined_collection_ledger(statistics, image_count=2)

    def test_physical_padding_cost_cannot_be_replaced_by_unique_crop_cost(self):
        statistics = self._statistics()
        statistics["collection_cost"][
            "physical_model_sample_cost_ratio"
        ] = (3520 + 2520) / 3520
        with self.assertRaisesRegex(ValueError, "physical_model_sample_cost_ratio"):
            _validate_combined_collection_ledger(statistics, image_count=2)

    def test_response_artifact_checkpoint_is_bound_to_stage_a(self):
        stage, _ = _stage_and_geometry()
        stage["baseline_checkpoint_sha256"] = "sealed-checkpoint"
        statistics = {
            "artifact_type": "whu_h4ab_k1_x8_response_statistics",
            "schema_version": 1,
            "status": "PASS",
            "scope": "full-test",
            "execution_mode": "live-k1-x8",
            "evaluated_images": 2,
            "full_test_length": 2,
            "source_stage_a": {"sha256": "stage-a-sha"},
            "source_execution": {
                "baseline_checkpoint": {"sha256": "sealed-checkpoint"}
            },
            "protocol": {"normal_execution_count": "exactly once per image"},
            "cells": [
                {
                    "cell_index": cell["cell_index"],
                    "image_index": cell["image_index"],
                    "local_crop_id": cell["local_crop_id"],
                    "geometry_eligible": cell["geometry_eligible"],
                    "k2_response_scores": {
                        name: 0.0
                        for name in (
                            "k1_minus_k2_entropy",
                            "k2_minus_k1_margin",
                            "k1_kx_jsd_normalized",
                            "k1_kx_argmax_flip_rate",
                            "k1_k2_argmax_flip_rate",
                        )
                    },
                }
                for cell in stage["cells"]
            ],
        }
        _join_response_scores(stage, statistics, stage_a_sha="stage-a-sha")
        statistics["source_execution"]["baseline_checkpoint"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            _join_response_scores(stage, statistics, stage_a_sha="stage-a-sha")


if __name__ == "__main__":
    unittest.main()
