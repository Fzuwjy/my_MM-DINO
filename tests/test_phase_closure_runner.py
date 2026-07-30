"""Contract tests for the local Stage-B0 closure runner."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_whu_phase_closure import (
    atomic_write_json,
    load_stage_a,
    metric_summary,
    rescue_gate,
)


class StageB0RunnerTest(unittest.TestCase):
    @staticmethod
    def _minimal_stage_a():
        return {
            "artifact_type": "whu_phase_utility_stage_a",
            "schema_version": 3,
            "status": "PASS",
            "scope": "full-test",
            "evaluated_images": 1,
            "full_test_length": 1,
            "stage_a_decision": {
                "outcome": "GO_STAGE_B_A2_HIERARCHICAL_X",
                "passed": True,
                "stage_b_authorized": True,
                "selected_route": "a2_hierarchical_x",
            },
            "reference_validation": {
                "checked": True,
                "k1_prediction_sha256_equal": True,
                "k1_miou_within_tolerance": True,
            },
            "protocol": {
                "teacher_phases_dy_dx": [[0, 0], [0, 8], [8, 0], [8, 8]],
                "crop_size_hw": [512, 512],
                "stride_hw": [341, 341],
            },
            "images": [{}],
            "cells": [{}],
            "aggregate": {"total_cells": 1},
            "formal_stage_a_routes": {
                "a2_hierarchical_x": {
                    "per_image": {
                        "gate": {"passed": True},
                        "greedy_point": {"levels_by_cell": [1]},
                    }
                }
            },
        }

    def test_stage_a_loader_requires_frozen_authorized_route(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage_a.json"
            payload = self._minimal_stage_a()
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_stage_a(path)["schema_version"], 3)
            payload["schema_version"] = 4
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema version 3"):
                load_stage_a(path)
            payload["schema_version"] = 3
            payload["stage_a_decision"]["selected_route"] = "a1_binary"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not authorize"):
                load_stage_a(path)

    def test_atomic_json_is_strict_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_write_json(path, {"array": np.array([1, 2]), "value": 1.5})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"array": [1, 2], "value": 1.5},
            )
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                atomic_write_json(path, {"other": True})
            with self.assertRaises(ValueError):
                atomic_write_json(Path(directory) / "nan.json", {"value": np.nan})

    def test_metric_summary_uses_supported_classes(self):
        summary = metric_summary(
            np.array([[2, 0, 0], [1, 1, 0], [0, 0, 0]]),
            ("a", "b", "unused"),
        )
        self.assertAlmostEqual(summary["miou"], (2 / 3 + 1 / 2) / 2)
        self.assertIsNone(summary["class_iou_percent"]["unused"])

    @staticmethod
    def _gate_fixture():
        stage_a = {
            "aggregate": {
                "endpoints": {
                    "k1": {"full_image": {"miou": 0.50}},
                    "matched_k2": {
                        "full_image": {"miou": 0.51},
                        "common_support": {
                            "regions": {
                                "small": {"error_rate": 0.20},
                                "thin": {"error_rate": 0.30},
                            }
                        },
                    },
                    "k4": {"full_image": {"miou": 0.52}},
                }
            }
        }
        oracle = {
            "full_miou": 0.515,
            "regions": None,
            "closure": {
                "forward_equivalent_cost": 2.0,
                "per_image": [{"forward_equivalent_cost": 2.0}],
            },
        }
        random = {"miou": {"p95": 0.512}}
        return stage_a, oracle, random

    def test_missing_structure_yields_only_provisional_go(self):
        stage_a, oracle, random = self._gate_fixture()
        gate = rescue_gate(stage_a, oracle, random)
        self.assertEqual(
            gate["outcome"], "PROVISIONAL_GO_B1_KNOWN_B0_GATES_PASS"
        )
        self.assertTrue(gate["b1_implementation_and_smoke_authorized"])
        self.assertFalse(gate["complete"])
        self.assertIsNone(gate["checks"]["small_error_not_worse_than_matched_k2"])

    def test_any_known_failure_stops_before_b1(self):
        stage_a, oracle, random = self._gate_fixture()
        oracle["closure"]["per_image"][0]["forward_equivalent_cost"] = 2.01
        gate = rescue_gate(stage_a, oracle, random)
        self.assertEqual(gate["outcome"], "STOP_STAGE_B0_KNOWN_GATE_FAILED")
        self.assertFalse(gate["b1_implementation_and_smoke_authorized"])

    def test_available_structure_completes_b0_gate(self):
        stage_a, oracle, random = self._gate_fixture()
        oracle["regions"] = {
            "all": {"pixels": 10, "errors": 1, "error_rate": 0.10},
            "small": {"pixels": 10, "errors": 1, "error_rate": 0.10},
            "thin": {"pixels": 10, "errors": 2, "error_rate": 0.20},
        }
        gate = rescue_gate(stage_a, oracle, random)
        self.assertTrue(gate["complete"])
        self.assertEqual(gate["outcome"], "GO_B1_EXACT_SPARSE_EXECUTION")

    def test_structure_availability_does_not_relabel_a_cost_failure(self):
        stage_a, oracle, random = self._gate_fixture()
        oracle["closure"]["per_image"][0]["forward_equivalent_cost"] = 2.01
        oracle["regions"] = {
            "all": {"pixels": 10, "errors": 1, "error_rate": 0.10},
            "small": {"pixels": 10, "errors": 1, "error_rate": 0.10},
            "thin": {"pixels": 10, "errors": 2, "error_rate": 0.20},
        }
        gate = rescue_gate(stage_a, oracle, random)
        self.assertEqual(gate["outcome"], "STOP_STAGE_B0_KNOWN_GATE_FAILED")

    def test_only_structure_failure_is_labeled_as_structure_failure(self):
        stage_a, oracle, random = self._gate_fixture()
        oracle["regions"] = {
            "all": {"pixels": 10, "errors": 1, "error_rate": 0.10},
            "small": {"pixels": 10, "errors": 3, "error_rate": 0.30},
            "thin": {"pixels": 10, "errors": 2, "error_rate": 0.20},
        }
        gate = rescue_gate(stage_a, oracle, random)
        self.assertEqual(gate["outcome"], "STOP_STAGE_B0_STRUCTURE_GATE_FAILED")

    def test_nonformal_random_never_authorizes_b1(self):
        stage_a, oracle, random = self._gate_fixture()
        gate = rescue_gate(
            stage_a, oracle, random, formal_random_control=False
        )
        self.assertEqual(gate["outcome"], "NOT_EVALUATED_NONFORMAL_RANDOM")
        self.assertFalse(gate["scientific_decision_evaluated"])
        self.assertFalse(gate["b1_implementation_and_smoke_authorized"])


if __name__ == "__main__":
    unittest.main()
