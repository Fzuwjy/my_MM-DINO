"""Contract tests for the one allowed H3 ridge-gate runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_whu_phase_h3_ridge_gate import (
    _q_selection_key,
    _selection_scores,
    load_simple_h3_failure,
    next_step_for_ridge,
    ridge_gate_decision,
)


class RidgeRunnerInputTest(unittest.TestCase):
    @staticmethod
    def _simple_payload():
        return {
            "artifact_type": "whu_phase_h3_k1_to_k2_screen",
            "schema_version": 1,
            "status": "PASS",
            "source_artifacts": {
                "stage_a": {"sha256": "a"},
                "stage_b0": {"sha256": "b"},
                "full_stage_b_latency": {"sha256": "l"},
            },
            "protocol": {
                "formal_random_control": True,
                "random_replicates": 1000,
                "random_seed": 20260731,
            },
            "h3_screen_decision": {
                "scientific_decision_evaluated": True,
                "known_checks_passed": False,
                "one_live_confirmation_authorized": False,
                "outcome": "STOP_SIMPLE_K1_SCORE_SCREEN_KNOWN_GATE_FAILED",
            },
        }

    def test_loader_requires_the_formal_matching_simple_score_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simple.json"
            payload = self._simple_payload()
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_simple_h3_failure(
                path,
                stage_a_sha256="a",
                stage_b0_sha256="b",
                latency_sha256="l",
            )
            self.assertEqual(
                loaded["h3_screen_decision"]["outcome"],
                "STOP_SIMPLE_K1_SCORE_SCREEN_KNOWN_GATE_FAILED",
            )
            payload["h3_screen_decision"]["known_checks_passed"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not formally fail"):
                load_simple_h3_failure(
                    path,
                    stage_a_sha256="a",
                    stage_b0_sha256="b",
                    latency_sha256="l",
                )

    def test_selection_scores_populates_only_declared_prediction_indices(self):
        prediction = {
            "scores_by_cell": (None, 0.2, None, -0.1),
            "predicted_cell_indices": (1, 3),
        }
        self.assertEqual(
            _selection_scores(prediction, cell_count=4).tolist(),
            [0.0, 0.2, 0.0, -0.1],
        )
        prediction["scores_by_cell"] = (9.0, 0.2, None, -0.1)
        with self.assertRaisesRegex(ValueError, "unrequested cell"):
            _selection_scores(prediction, cell_count=4)

    def test_q_tie_is_miou_then_integer_processed_cost_then_lower_q(self):
        self.assertLess(
            _q_selection_key(pooled_miou=0.51, processed_x8_crops=20, q=0.5),
            _q_selection_key(pooled_miou=0.50, processed_x8_crops=10, q=0.1),
        )
        self.assertLess(
            _q_selection_key(pooled_miou=0.50, processed_x8_crops=10, q=0.5),
            _q_selection_key(pooled_miou=0.50, processed_x8_crops=11, q=0.1),
        )
        self.assertLess(
            _q_selection_key(pooled_miou=0.50, processed_x8_crops=10, q=0.1),
            _q_selection_key(pooled_miou=0.50, processed_x8_crops=10, q=0.5),
        )


class RidgeRunnerGateTest(unittest.TestCase):
    @staticmethod
    def _cost():
        return {
            "physical_model_sample_cost_ratio": 1.8,
            "per_image": [
                {"physical_model_sample_cost_ratio": 1.75},
                {"physical_model_sample_cost_ratio": 1.8},
            ],
        }

    def _decision(self, *, candidate: float, formal: bool):
        return ridge_gate_decision(
            candidate_miou=candidate,
            k1_miou=0.50,
            matched_k2_miou=0.5145,
            k4_miou=0.52,
            cost=self._cost(),
            random_control={"miou": {"p95": 0.514}},
            formal_random_control=formal,
        )

    def test_formal_pass_authorizes_only_one_live_confirmation(self):
        result = self._decision(candidate=0.515, formal=True)
        self.assertEqual(
            result["outcome"],
            "PROVISIONAL_GO_ONE_RIDGE_LIVE_CONFIRMATION_KNOWN_GATES_PASS",
        )
        self.assertTrue(result["one_live_confirmation_authorized"])
        self.assertFalse(result["h3_confirmed"])
        self.assertIn("one frozen live", next_step_for_ridge(result))

    def test_formal_failure_ends_h3(self):
        result = self._decision(candidate=0.509, formal=True)
        self.assertEqual(
            result["outcome"], "STOP_H3_RIDGE_GATE_KNOWN_GATE_FAILED"
        )
        self.assertFalse(result["one_live_confirmation_authorized"])
        self.assertIn("End H3", next_step_for_ridge(result))

    def test_nonformal_random_cannot_stop_or_authorize(self):
        result = self._decision(candidate=0.509, formal=False)
        self.assertEqual(
            result["outcome"], "NOT_EVALUATED_NONFORMAL_RIDGE_RANDOM_CONTROL"
        )
        self.assertIsNone(result["known_checks_passed"])
        self.assertFalse(result["one_live_confirmation_authorized"])
        self.assertIn("formal random", next_step_for_ridge(result))


if __name__ == "__main__":
    unittest.main()
