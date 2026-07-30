"""CPU tests for the formal paired Stage-B latency protocol."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_whu_phase_sparse_latency import (
    EXPECTED_FULL_TEST_LENGTH,
    EXPECTED_LEDGER,
    PAIRED_REPEATS,
    POLICY_K4,
    POLICY_RESCUE,
    _add_ledger,
    _empty_ledger,
    _expected_policy_ledger,
    _load_full_b1,
    _pipeline_execution,
    _validate_exact_ledger,
    evaluate_primary_latency_gate,
    paired_policy_order,
)


class PairedOrderTest(unittest.TestCase):
    def test_checkerboard_order_is_balanced_in_every_repeat(self):
        aggregate = {POLICY_RESCUE: 0, POLICY_K4: 0}
        for repeat_index in range(PAIRED_REPEATS):
            first = {POLICY_RESCUE: 0, POLICY_K4: 0}
            for image_index in range(EXPECTED_FULL_TEST_LENGTH):
                order = paired_policy_order(repeat_index, image_index)
                self.assertEqual(set(order), {POLICY_RESCUE, POLICY_K4})
                first[order[0]] += 1
                aggregate[order[0]] += 1
            self.assertEqual(first, {POLICY_RESCUE: 10, POLICY_K4: 10})
        self.assertEqual(aggregate, {POLICY_RESCUE: 30, POLICY_K4: 30})

    def test_checkerboard_order_is_frozen_and_rejects_bad_indices(self):
        self.assertEqual(
            paired_policy_order(0, 0), (POLICY_RESCUE, POLICY_K4)
        )
        self.assertEqual(
            paired_policy_order(0, 1), (POLICY_K4, POLICY_RESCUE)
        )
        self.assertEqual(
            paired_policy_order(1, 0), (POLICY_K4, POLICY_RESCUE)
        )
        with self.assertRaisesRegex(ValueError, "repeat index"):
            paired_policy_order(-1, 0)
        with self.assertRaisesRegex(ValueError, "repeat index"):
            paired_policy_order(PAIRED_REPEATS, 0)
        with self.assertRaisesRegex(ValueError, "image index"):
            paired_policy_order(0, EXPECTED_FULL_TEST_LENGTH)


class PrimaryGateTest(unittest.TestCase):
    def test_gate_uses_median_of_paired_differences(self):
        # Independent medians would give 3 - 100 = -97.  The registered paired
        # deltas are [2, -97, 99], whose median is +2 and must fail.
        k4 = [0.0, 100.0, 101.0]
        rescue = [2.0, 3.0, 200.0]
        deltas = [right - left for left, right in zip(k4, rescue, strict=True)]
        result = evaluate_primary_latency_gate(deltas)
        self.assertEqual(result["median_paired_delta_ms"], 2.0)
        self.assertFalse(result["passed"])

    def test_gate_is_strictly_negative_without_tolerance(self):
        self.assertTrue(evaluate_primary_latency_gate([-3, -2, 4])["passed"])
        self.assertFalse(evaluate_primary_latency_gate([-1, 0, 2])["passed"])
        self.assertFalse(evaluate_primary_latency_gate([0, 0, 0])["passed"])

    def test_gate_rejects_wrong_count_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "requires 3"):
            evaluate_primary_latency_gate([-1, -2])
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_primary_latency_gate([-1, float("nan"), -2])


class LedgerTest(unittest.TestCase):
    def test_expected_ledger_counts_per_image_fixed_batch_padding(self):
        closure = {
            "baseline_k1_crop_forwards": 24,
            "extra_crop_forwards": 17,
            "per_image": [
                {"extra_crop_forwards": 8},
                {"extra_crop_forwards": 9},
            ],
        }
        self.assertEqual(
            _expected_policy_ledger(closure, batch_size=8),
            {
                "normal_selected_samples": 24,
                "normal_processed_samples": 24,
                "normal_batch_calls": 3,
                "shifted_selected_samples": 17,
                "shifted_processed_samples": 24,
                "shifted_padding_samples": 7,
                "shifted_batch_calls": 3,
                "total_processed_samples": 48,
                "total_batch_calls": 6,
            },
        )

    def test_ledger_accumulation_and_exact_validation_are_fail_closed(self):
        total = _empty_ledger()
        half = {key: value // 2 for key, value in EXPECTED_LEDGER[POLICY_RESCUE].items()}
        remainder = {
            key: EXPECTED_LEDGER[POLICY_RESCUE][key] - half[key]
            for key in half
        }
        _add_ledger(total, half)
        _add_ledger(total, remainder)
        _validate_exact_ledger(POLICY_RESCUE, total, EXPECTED_LEDGER[POLICY_RESCUE])
        total["shifted_padding_samples"] += 1
        with self.assertRaisesRegex(AssertionError, "frozen ledger"):
            _validate_exact_ledger(
                POLICY_RESCUE, total, EXPECTED_LEDGER[POLICY_RESCUE]
            )

    def test_frozen_ledgers_use_batch_eight_physical_counts(self):
        for ledger in EXPECTED_LEDGER.values():
            self.assertEqual(
                ledger["normal_processed_samples"],
                ledger["normal_batch_calls"] * 8,
            )
            self.assertEqual(
                ledger["shifted_processed_samples"],
                ledger["shifted_batch_calls"] * 8,
            )
            self.assertEqual(
                ledger["total_processed_samples"],
                ledger["normal_processed_samples"]
                + ledger["shifted_processed_samples"],
            )


class ArtifactAndBoundaryTest(unittest.TestCase):
    @staticmethod
    def _valid_full_b1() -> dict:
        return {
            "artifact_type": "whu_phase_sparse_live_b1_full",
            "schema_version": 1,
            "status": "PASS",
            "latency_decision_evaluated": False,
            "non_latency_decision": {
                "passed": True,
                "authorizes_same_primitive_latency_benchmark": True,
                "outcome": "PASS_B1_LIVE_OUTPUT_STRUCTURE_GATES_LATENCY_PENDING",
            },
        }

    def test_full_b1_loader_requires_actual_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "full.json"
            payload = self._valid_full_b1()
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(_load_full_b1(path), payload)
            payload["non_latency_decision"]["passed"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not authorize"):
                _load_full_b1(path)

    def test_primary_pipeline_has_no_empty_cache_or_digest_in_timed_compose(self):
        source = inspect.getsource(_pipeline_execution)
        self.assertNotIn("empty_cache", source)
        self.assertIn("include_digests=False", source)
        self.assertIn("forward_selected_phase_crops", source)
        self.assertIn("forward_selected_phase_key_crops", source)
        self.assertLess(source.index("wall_started_ns"), source.index("optical.to(device)"))
        self.assertLess(source.index("prediction ="), source.index("wall_finished_ns"))


if __name__ == "__main__":
    unittest.main()
