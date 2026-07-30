"""CPU tests for formal full B1 live-run accounting helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluate_whu_phase_sparse_full import (
    SMALL_REGION,
    THIN_REGION,
    _apply_registered_k2_gain_threshold,
    _load_b1b_smoke,
    _physical_cost_summary,
    _region_error_counts,
    _source_artifact_matches,
    _summarize_region_totals,
)


class PhaseSparseFullRunnerTest(unittest.TestCase):
    def test_physical_cost_counts_the_only_full_dataset_padding(self):
        records = [
            {
                "baseline_crop_samples": 176,
                "selected_extra_crop_samples": 176,
                "processed_extra_crop_samples": 176,
                "padding_crop_samples": 0,
            }
            for _ in range(19)
        ]
        records.append(
            {
                "baseline_crop_samples": 176,
                "selected_extra_crop_samples": 174,
                "processed_extra_crop_samples": 176,
                "padding_crop_samples": 2,
            }
        )
        result = _physical_cost_summary(records)
        self.assertEqual(result["baseline_crop_samples"], 3520)
        self.assertEqual(result["selected_extra_crop_samples"], 3518)
        self.assertEqual(
            result["processed_extra_crop_samples_including_padding"], 3520
        )
        self.assertEqual(result["padding_crop_samples"], 2)
        self.assertAlmostEqual(
            result["logical_unique_crop_cost_ratio"], 1.9994318181818183
        )
        self.assertEqual(result["physical_model_sample_cost_ratio"], 2.0)
        self.assertEqual(result["maximum_per_image_physical_cost_ratio"], 2.0)

    def test_region_counts_use_only_valid_common_support_pixels(self):
        target = np.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1],
                [0, 1, 7, 1, 1],
                [0, 1, 1, 1, 1],
            ],
            dtype=np.int64,
        )
        prediction = target.copy()
        prediction[1, 1] = 2
        prediction[1, 2] = 2
        prediction[2, 2] = 2  # Invalid label: must not enter any region count.
        small = np.zeros(target.shape, dtype=bool)
        small[1, 1] = True
        small[2, 2] = True
        thin = np.zeros(target.shape, dtype=bool)
        thin[1, 2] = True
        thin[3, 0] = True  # Outside common support.
        result = _region_error_counts(
            prediction,
            target,
            (1, 4, 1, 5),
            {SMALL_REGION: small, THIN_REGION: thin},
        )
        self.assertEqual(result["all"], {"pixels": 11, "errors": 2})
        self.assertEqual(result["small"], {"pixels": 1, "errors": 1})
        self.assertEqual(result["thin"], {"pixels": 1, "errors": 1})
        summary = _summarize_region_totals(result)
        self.assertAlmostEqual(summary["all"]["error_rate"], 2 / 11)
        self.assertEqual(summary["small"]["error_rate"], 1.0)

    def test_smoke_source_sha_comparison_fails_closed(self):
        smoke = {"source_artifacts": {"stage_a": {"sha256": "abc"}}}
        self.assertTrue(_source_artifact_matches(smoke, "stage_a", "abc"))
        self.assertFalse(_source_artifact_matches(smoke, "stage_a", "def"))
        self.assertFalse(_source_artifact_matches(smoke, "stage_b0", "abc"))

    def test_b1b_smoke_requires_passed_schema_v2_artifact(self):
        payload = {
            "status": "PASS",
            "artifact_type": "whu_phase_sparse_live_b1b_smoke",
            "schema_version": 2,
            "correctness_decision": {
                "outcome": "PASS_B1B_FIRST_IMAGE_CORRECTNESS_SMOKE",
                "passed": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_load_b1b_smoke(path), payload)
            payload["schema_version"] = 1
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not pass"):
                _load_b1b_smoke(path)

    def test_registered_k2_gain_threshold_is_inclusive(self):
        gate = {
            "checks": {"outperforms_uniform_k2": False, "other": True},
            "observed": {"miou_delta_over_k2": 0.0005},
            "thresholds": {},
        }
        result = _apply_registered_k2_gain_threshold(gate, 0.0005)
        self.assertTrue(result["passed"])
        self.assertNotIn("outperforms_uniform_k2", result["checks"])
        self.assertTrue(
            result["checks"]["gain_over_matched_k2_at_least_0_05pp"]
        )
        self.assertEqual(result["thresholds"]["miou_delta_over_k2_comparator"], ">=")


if __name__ == "__main__":
    unittest.main()
