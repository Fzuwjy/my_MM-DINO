import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_whu_phase_utility_audit import (
    DECISION_Q,
    Q_VALUES,
    _finite_or_none,
    _reference_checks,
    atomic_write_json,
    cell_means_from_common_map,
    finite_oracle_envelope,
    load_four_phase_reference,
    restrict_score_to_geometry,
    slide_window_manifest,
    stage_a_decision,
)
from scripts.phase_utility_common import (
    CellScoreVector,
    oracle_cell_miou_gain_scores,
)
from scripts.spatial_diagnostics_common import baseline_summary


class PhaseUtilityRunnerTests(unittest.TestCase):
    def test_nonfinite_empty_cell_score_serializes_as_null(self):
        self.assertIsNone(_finite_or_none(-np.inf))
        self.assertEqual(_finite_or_none(1.25), 1.25)

    def test_geometry_restriction_is_shared_by_oracle_and_deployment_scores(self):
        eligible = np.array([False, True, True], dtype=bool)
        oracle = restrict_score_to_geometry(
            CellScoreVector("oracle", np.array([9.0, 1.0, -1.0]), True),
            eligible,
        )
        deployment = restrict_score_to_geometry(
            CellScoreVector("entropy", np.array([3.0, 2.0, 1.0]), False),
            eligible,
        )
        self.assertTrue(np.isneginf(oracle.values[0]))
        self.assertTrue(np.isneginf(deployment.values[0]))
        self.assertTrue(oracle.uses_ground_truth)
        self.assertFalse(deployment.uses_ground_truth)

    def test_atomic_json_is_strict_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_write_json(path, {"value": 1.0})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1.0})
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                atomic_write_json(path, {"value": 2.0})
            invalid = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                atomic_write_json(invalid, {"value": -np.inf})
            self.assertFalse(invalid.exists())

    def test_slide_manifest_reproduces_end_backtracking(self):
        cases = {
            512: [0],
            853: [0, 341],
            854: [0, 341, 342],
            1100: [0, 341, 588],
        }
        for length, expected in cases.items():
            with self.subTest(length=length):
                manifest = slide_window_manifest((length, length))
                self.assertEqual(manifest["row_starts"], expected)
                self.assertEqual(manifest["column_starts"], expected)
                self.assertEqual(
                    len(manifest["windows"]), len(expected) * len(expected)
                )
                self.assertEqual(manifest["windows"][0].tolist(), [0, 512, 0, 512])
                self.assertEqual(
                    manifest["windows"][-1].tolist(),
                    [length - 512, length, length - 512, length],
                )

    def test_slide_manifest_rejects_small_sealed_image(self):
        with self.assertRaisesRegex(ValueError, "both image axes"):
            slide_window_manifest((511, 800))

    def test_cell_means_intersect_common_support_without_gt(self):
        cells = np.array(
            [
                [0, 4, 0, 4],
                [0, 4, 4, 8],
                [4, 8, 0, 4],
                [4, 8, 4, 8],
            ],
            dtype=np.int64,
        )
        common = np.arange(16, dtype=np.float32).reshape(4, 4)
        values = cell_means_from_common_map(cells, common, (2, 6, 2, 6))
        self.assertEqual(values[0], np.mean(common[:2, :2]))
        self.assertEqual(values[1], np.mean(common[:2, 2:]))
        self.assertEqual(values[2], np.mean(common[2:, :2]))
        self.assertEqual(values[3], np.mean(common[2:, 2:]))

    def test_cell_means_marks_nonintersecting_cell_negative_infinity(self):
        cells = np.array([[0, 2, 0, 2], [2, 4, 2, 4]], dtype=np.int64)
        values = cell_means_from_common_map(
            cells, np.ones((2, 2), dtype=np.float32), (2, 4, 2, 4)
        )
        self.assertTrue(np.isneginf(values[0]))
        self.assertEqual(values[1], 1.0)

    def test_singleton_oracle_uses_full_reference_confusion(self):
        full = np.zeros((7, 7), dtype=np.int64)
        full[0, 0] = 99
        full[0, 1] = 1
        full[1, 1] = 100
        cell_k1 = np.zeros((7, 7), dtype=np.int64)
        cell_k4 = np.zeros((7, 7), dtype=np.int64)
        cell_k1[0, 1] = 1
        cell_k4[0, 0] = 1
        stats = (
            {
                "cell_index": 0,
                "confusion": {"k1": cell_k1, "k2": cell_k1, "k4": cell_k4},
            },
        )
        score = oracle_cell_miou_gain_scores(
            stats, num_classes=7, reference_confusion=full
        )
        self.assertTrue(score.uses_ground_truth)
        self.assertGreater(score.values[0], 0.0)

    def test_finite_oracle_envelope_uses_higher_observed_miou(self):
        net = []
        singleton = []
        for index, q in enumerate(Q_VALUES):
            net.append({"requested_q": q, "full_image": {"miou": 0.5 + index / 100}})
            singleton.append(
                {"requested_q": q, "full_image": {"miou": 0.55 - index / 100}}
            )
        envelope = finite_oracle_envelope(
            {"net_correct": net, "singleton_global_miou": singleton}
        )
        self.assertEqual(envelope[0]["source_ranking"], "singleton_global_miou")
        self.assertEqual(envelope[-1]["source_ranking"], "net_correct")

    @staticmethod
    def _decision_fixture(oracle_miou=0.58, random_p95=0.57):
        endpoint = lambda miou, small, thin: {  # noqa: E731
            "full_image": {"miou": miou},
            "common_support": {
                "regions": {
                    "small": {"error_rate": small},
                    "thin": {"error_rate": thin},
                }
            },
        }
        endpoints = {
            "k1": endpoint(0.50, 0.25, 0.35),
            "matched_k2": endpoint(0.55, 0.20, 0.30),
            "k4": endpoint(0.60, 0.19, 0.29),
        }
        oracle = [
            {
                "requested_q": DECISION_Q,
                "source_ranking": "net_correct",
                "cost": {"forward_equivalent_cost": 2.0},
                "full_image": {"miou": oracle_miou},
                "common_support": {
                    "regions": {
                        "small": {"error_rate": 0.19},
                        "thin": {"error_rate": 0.29},
                    }
                },
            }
        ]
        random = [
            {"requested_q": DECISION_Q, "miou": {"p95": random_p95}}
        ]
        return endpoints, oracle, random

    def test_stage_a_decision_requires_all_fixed_gates(self):
        endpoints, oracle, random = self._decision_fixture()
        decision = stage_a_decision(endpoints, oracle, random, formal=True)
        self.assertEqual(decision["outcome"], "GO_STAGE_B")
        self.assertTrue(decision["checks"]["outperforms_equal_budget_random_p95"])

        endpoints, oracle, random = self._decision_fixture(random_p95=0.58)
        decision = stage_a_decision(endpoints, oracle, random, formal=True)
        self.assertEqual(decision["outcome"], "NO_GO_STOP_ROUTE")
        self.assertFalse(decision["passed"])

    def test_subset_never_makes_scientific_decision(self):
        decision = stage_a_decision({}, [], [], formal=False)
        self.assertEqual(decision["outcome"], "NOT_EVALUATED_SUBSET")
        self.assertIsNone(decision["passed"])

    def test_load_four_phase_reference_checks_sealed_protocol(self):
        payload = {
            "status": "PASS",
            "scope": "full-test",
            "evaluated_images": 20,
            "full_test_length": 20,
            "protocol": {
                "primary": [[0, 0], [0, 8], [8, 0], [8, 8]],
                "valid_margin": 512,
                "same_common_valid_region_for_primary_and_control": True,
            },
            "baseline_validation": {
                "checked": True,
                "prediction_sha256_equal": True,
                "label_sha256_equal": True,
                "confusion_equal": True,
                "miou_within_tolerance": True,
            },
            "two_phase_validation": {
                "checked": True,
                "candidate_prediction_sha256_equal": True,
                "confusion_equal": True,
                "miou_within_tolerance": True,
            },
            "aggregate": {
                "8": {
                    "candidate": {"miou": 0.5, "confusion": []},
                    "candidate_prediction_sha256": "abc",
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_four_phase_reference(path)["status"], "PASS")
            payload["protocol"]["valid_margin"] = 511
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid margin"):
                load_four_phase_reference(path)

    def test_full_reference_checks_cover_all_three_sealed_endpoints(self):
        labels = [f"class_{index}" for index in range(7)]
        k1 = np.zeros((7, 7), dtype=np.int64)
        k2 = np.zeros((7, 7), dtype=np.int64)
        k4 = np.zeros((7, 7), dtype=np.int64)
        k1[0, 0], k2[0, 0], k4[0, 0] = 10, 11, 12
        k1_metrics = baseline_summary(k1, labels)
        k2_metrics = baseline_summary(k2, labels)
        k4_metrics = baseline_summary(k4, labels)
        validation = _reference_checks(
            full_test=True,
            spatial_reference={
                "prediction_sha256": "k1",
                "label_sha256": "label",
                "aggregate": {"baseline": k1_metrics},
            },
            two_phase_reference={
                "aggregate": {
                    "8": {
                        "candidate_prediction_sha256": "k2",
                        "candidate": k2_metrics,
                    }
                }
            },
            four_phase_reference={
                "aggregate": {
                    "8": {
                        "candidate_prediction_sha256": "k4",
                        "candidate": k4_metrics,
                    }
                }
            },
            baseline_digest="k1",
            label_digest="label",
            legacy_k2_digest="k2",
            k4_digest="k4",
            k1_confusion=k1,
            legacy_k2_confusion=k2,
            k4_confusion=k4,
            class_names=labels,
            tolerance=0.0,
        )
        self.assertTrue(validation["checked"])
        self.assertTrue(all(value for key, value in validation.items() if key != "checked"))


if __name__ == "__main__":
    unittest.main()
