"""Contract tests for the offline H3 K1-to-K2 screen runner."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_whu_phase_h3_screen import (
    _configuration_key,
    _score_arrays,
    next_step_for_decision,
    screen_gate,
)


class H3ScoreProvenanceTest(unittest.TestCase):
    def test_only_frozen_k1_scores_are_extracted_and_ineligible_stays_minus_inf(self):
        cells = [
            {
                "scores": {
                    "k1_entropy": 0.7,
                    "k1_negative_margin": -0.2,
                    "k1_predicted_boundary_density": 0.1,
                    "oracle_singleton_global_miou_gain": 99.0,
                }
            },
            {
                "scores": {
                    "k1_entropy": None,
                    "k1_negative_margin": None,
                    "k1_predicted_boundary_density": None,
                    "oracle_singleton_global_miou_gain": 100.0,
                }
            },
        ]
        result = _score_arrays(cells, np.asarray([True, False], dtype=bool))
        self.assertEqual(
            tuple(result),
            ("entropy", "negative_margin", "predicted_boundary_density"),
        )
        self.assertEqual(result["entropy"][0], 0.7)
        self.assertTrue(np.isneginf(result["entropy"][1]))

    def test_frozen_configuration_tie_order_is_miou_cost_score_then_q(self):
        better_miou = _configuration_key(
            train_miou=0.51, train_cost=1.9, score_order=2, q=0.5
        )
        lower_cost = _configuration_key(
            train_miou=0.50, train_cost=1.4, score_order=2, q=0.5
        )
        self.assertLess(better_miou, lower_cost)
        self.assertLess(
            _configuration_key(
                train_miou=0.5, train_cost=1.4, score_order=0, q=0.5
            ),
            _configuration_key(
                train_miou=0.5, train_cost=1.4, score_order=1, q=0.1
            ),
        )
        self.assertLess(
            _configuration_key(
                train_miou=0.5, train_cost=1.4, score_order=0, q=0.1
            ),
            _configuration_key(
                train_miou=0.5, train_cost=1.4, score_order=0, q=0.5
            ),
        )


class H3ScreenGateTest(unittest.TestCase):
    @staticmethod
    def _cost():
        return {
            "physical_model_sample_cost_ratio": 1.8,
            "per_image": [
                {"physical_model_sample_cost_ratio": 1.75},
                {"physical_model_sample_cost_ratio": 1.8},
            ],
        }

    def test_known_gate_pass_authorizes_only_one_live_confirmation(self):
        result = screen_gate(
            candidate_miou=0.515,
            k1_miou=0.50,
            matched_k2_miou=0.5145,
            k4_miou=0.52,
            cost=self._cost(),
            random_control={"miou": {"p95": 0.514}},
            formal_random_control=True,
        )
        self.assertEqual(
            result["outcome"],
            "PROVISIONAL_GO_ONE_LIVE_CONFIRMATION_KNOWN_GATES_PASS",
        )
        self.assertTrue(result["one_live_confirmation_authorized"])
        self.assertFalse(result["complete"])
        self.assertFalse(result["h3_confirmed"])
        self.assertIsNone(
            result["checks"]["small_error_not_worse_than_matched_k2"]
        )

    def test_efficacy_failure_stops_simple_scores(self):
        result = screen_gate(
            candidate_miou=0.509,
            k1_miou=0.50,
            matched_k2_miou=0.51,
            k4_miou=0.52,
            cost=self._cost(),
            random_control={"miou": {"p95": 0.508}},
            formal_random_control=True,
        )
        self.assertEqual(
            result["outcome"], "STOP_SIMPLE_K1_SCORE_SCREEN_KNOWN_GATE_FAILED"
        )
        self.assertFalse(result["one_live_confirmation_authorized"])

    def test_nonformal_random_never_makes_a_scientific_decision(self):
        result = screen_gate(
            candidate_miou=0.515,
            k1_miou=0.50,
            matched_k2_miou=0.5145,
            k4_miou=0.52,
            cost=self._cost(),
            random_control={"miou": {"p95": 0.514}},
            formal_random_control=False,
        )
        self.assertEqual(result["outcome"], "NOT_EVALUATED_NONFORMAL_RANDOM_CONTROL")
        self.assertIsNone(result["known_checks_passed"])
        self.assertFalse(result["one_live_confirmation_authorized"])
        self.assertIn("nonformal random control", result["interpretation"])
        self.assertIn("Smoke only", next_step_for_decision(result))


if __name__ == "__main__":
    unittest.main()
