"""Pure unit tests for the paired V4-C E15->E30 restart comparator."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.compare_whu_v4_c_e30_continuation import (
    ARTIFACT_TYPE,
    CANDIDATE_VARIANT,
    CLEAN_VARIANT,
    CONTINUATION_MODE,
    compare_e30,
    decision,
    main,
    parse_args,
    validate_protocol_pair,
    validate_train_epoch_pair,
)


CLASS_NAMES = ("farmland", "city", "village", "water", "forest", "road", "other")


def _protocol(variant: str, *, scope: str = "formal-restart-screen") -> dict:
    protocol = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": 1,
        "continuation_mode": CONTINUATION_MODE,
        "source_epoch": 15,
        "first_continuation_epoch": 16,
        "target_epoch": 30,
        "stop_after_epoch": 30,
        "evaluation_epochs": [30],
        "not_equivalent_to_uninterrupted": True,
        "persistent_worker_state_restored": False,
        "variant": variant,
        "git_commit": "same-output-commit",
        "seed": 42,
        "scheduler_horizon_epochs": 50,
        "model_name": "DINOv3",
        "dataset_name": "WHU",
        "num_modalities": 2,
        "backbone_type": "dinov3_vits16",
        "use_lora": False,
        "train_batch_size_per_gpu": 8,
        "train_workers": 4,
        "inference_batch_size": 32,
        "max_train_batches": None,
        "max_test_images": None,
        "scope": scope,
        "train_dataset_length": 3200,
        "full_test_length": 20,
        "evaluated_test_length": 20,
        "mask_padding_ignore": True,
        "mask_fill": 7,
        "aux_fill": 0,
        "loss_change": "none",
        "restart_rng_fingerprints": {
            "python": "python-rng",
            "numpy": "numpy-rng",
            "torch_cpu": "torch-cpu-rng",
            "torch_cuda": ["torch-cuda-rng"],
            "loader_generator": "loader-rng",
        },
        "source_dir_resolved": "/sealed/clean",
        "source_protocol_sha256": "clean-protocol-sha",
        "source_git_commit": "clean-source-commit",
        "source_checkpoint_sha256": "clean-checkpoint-sha",
        "source_evaluation_sha256": "clean-evaluation-sha",
    }
    if variant == CANDIDATE_VARIANT:
        protocol["source_dir_resolved"] = "/sealed/candidate"
        protocol["candidate_clean_lineage"] = {
            "clean_reference_dir": "/sealed/clean",
            "clean_reference_protocol_sha256": "clean-protocol-sha",
            "clean_reference_git_commit": "clean-source-commit",
            "clean_reference_checkpoint_sha256": "clean-checkpoint-sha",
            "clean_reference_evaluation_sha256": "clean-evaluation-sha",
        }
    return protocol


def _trace() -> list[dict]:
    return [
        {
            "pair_sha256": f"pair-{index}",
            "optical_sha256": f"optical-{index}",
            "sar_sha256": f"sar-{index}",
            "raw_label_sha256": f"raw-label-{index}",
            "normalized_label_sha256": f"normalized-label-{index}",
        }
        for index in range(10)
    ]


def _mechanism(*, healthy: bool = True) -> dict:
    return {
        "optical_stem_mechanism": {
            "summary": {
                "observed_batches": 10,
                "all_readouts_finite": healthy,
                "stem_output_nonzero_batches": 10 if healthy else 0,
                "projection_gradient_nonzero_batches": 10 if healthy else 0,
                "upstream_gradient_nonzero_batches": 10 if healthy else 0,
            }
        }
    }


def _train(epoch: int, *, healthy: bool = True) -> dict:
    return {
        "epoch": epoch,
        "paired_data_sha256": f"paired-e{epoch}",
        "raw_label_sha256": f"raw-e{epoch}",
        "first_batch_trace": _trace(),
        **_mechanism(healthy=healthy),
    }


def _confusion(correct: int) -> list[list[int]]:
    matrix = np.zeros((7, 7), dtype=np.int64)
    for class_index in range(7):
        matrix[class_index, class_index] = correct
        matrix[class_index, (class_index + 1) % 7] = 10 - correct
    return matrix.tolist()


def _evaluation(
    correct: int,
    *,
    class_delta: float = 0.0,
    city_delta: float | None = None,
    road_delta: float | None = None,
) -> dict:
    class_iou = {
        name: 50.0 + index + class_delta for index, name in enumerate(CLASS_NAMES)
    }
    if city_delta is not None:
        class_iou["city"] = 51.0 + city_delta
    if road_delta is not None:
        class_iou["road"] = 55.0 + road_delta
    return {
        "label_sha256": "same-e30-label-stream",
        "per_image": [
            {"sample_name": "image-a", "confusion": _confusion(correct)},
            {"sample_name": "image-b", "confusion": _confusion(correct)},
            {"sample_name": "image-c", "confusion": _confusion(correct)},
        ],
        "aggregate": {"class_iou_percent": class_iou},
    }


def _gate_result(
    delta: float,
    *,
    city: float = 0.1,
    road: float = 0.1,
    healthy: bool = True,
) -> dict:
    return {
        "candidate_minus_clean_pp": delta,
        "class_iou_delta_candidate_minus_clean_pp": {
            **{name: 0.0 for name in CLASS_NAMES},
            "city": city,
            "road": road,
        },
        "candidate_train_e30_mechanism_health": {
            "healthy_for_formal_gate": healthy
        },
    }


class V4CE30ComparatorTest(unittest.TestCase):
    def test_protocol_locks_restart_identity_data_and_rng(self):
        clean = _protocol(CLEAN_VARIANT)
        candidate = _protocol(CANDIDATE_VARIANT)
        audit = validate_protocol_pair(clean, candidate)
        self.assertTrue(audit["restart_rng_fingerprints_equal"])

        corruptions = (
            ("artifact_type", "wrong"),
            ("continuation_mode", "uninterrupted"),
            ("source_epoch", 14),
            ("target_epoch", 31),
            ("stop_after_epoch", 29),
            ("not_equivalent_to_uninterrupted", False),
            ("persistent_worker_state_restored", True),
            ("git_commit", "different-commit"),
            ("seed", 43),
            ("train_workers", 2),
            ("restart_rng_fingerprints", {"python": "different"}),
            (
                "candidate_clean_lineage",
                {"clean_reference_checkpoint_sha256": "different"},
            ),
        )
        for field, value in corruptions:
            with self.subTest(field=field):
                bad = deepcopy(candidate)
                bad[field] = value
                with self.assertRaisesRegex(RuntimeError, "protocol|RNG|differs"):
                    validate_protocol_pair(clean, bad)

    def test_train_pair_requires_full_hashes_and_exact_first_ten_trace(self):
        clean = _train(16)
        candidate = deepcopy(clean)
        audit = validate_train_epoch_pair(
            clean, candidate, 16, require_ten_trace_records=True
        )
        self.assertTrue(audit["first_ten_trace_exactly_equal"])
        self.assertEqual(audit["trace_record_count"], 10)

        for field in ("paired_data_sha256", "raw_label_sha256"):
            with self.subTest(field=field):
                drift = deepcopy(candidate)
                drift[field] = "drift"
                with self.assertRaisesRegex(RuntimeError, "streams differ"):
                    validate_train_epoch_pair(
                        clean, drift, 16, require_ten_trace_records=True
                    )
        drift = deepcopy(candidate)
        drift["first_batch_trace"][4]["sar_sha256"] = "drift"
        with self.assertRaisesRegex(RuntimeError, "streams differ"):
            validate_train_epoch_pair(
                clean, drift, 16, require_ten_trace_records=True
            )

    def test_formal_trace_must_contain_ten_records(self):
        clean = _train(16)
        clean["first_batch_trace"] = clean["first_batch_trace"][:1]
        with self.assertRaisesRegex(RuntimeError, "expected 10"):
            validate_train_epoch_pair(
                clean, deepcopy(clean), 16, require_ten_trace_records=True
            )

    def test_compare_e30_reports_pooled_per_image_bootstrap_and_classes(self):
        clean = _evaluation(8)
        candidate = _evaluation(9, class_delta=0.4)
        result = compare_e30(
            clean,
            candidate,
            _train(30),
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )
        self.assertGreater(result["candidate_minus_clean_pp"], 0.1)
        self.assertEqual(len(result["per_image"]), 3)
        self.assertEqual(result["per_image_delta_summary_pp"]["positive_count"], 3)
        self.assertAlmostEqual(
            result["class_iou_delta_candidate_minus_clean_pp"]["city"], 0.4
        )
        bootstrap = result["paired_bootstrap_candidate_minus_clean"]
        self.assertGreater(bootstrap["low_pp"], 0.0)
        self.assertEqual(bootstrap["role"], "descriptive_only_not_a_gate")
        self.assertTrue(
            result["candidate_train_e30_mechanism_health"][
                "healthy_for_formal_gate"
            ]
        )

    def test_compare_e30_rejects_label_or_image_order_drift(self):
        clean = _evaluation(8)
        candidate = _evaluation(9)
        candidate["label_sha256"] = "drift"
        with self.assertRaisesRegex(RuntimeError, "label streams differ"):
            compare_e30(
                clean,
                candidate,
                _train(30),
                bootstrap_replicates=10,
                bootstrap_seed=7,
            )
        candidate = _evaluation(9)
        candidate["per_image"].reverse()
        with self.assertRaisesRegex(RuntimeError, "image order differs"):
            compare_e30(
                clean,
                candidate,
                _train(30),
                bootstrap_replicates=10,
                bootstrap_seed=7,
            )

    def test_e30_gate_requires_increment_safety_and_mechanism(self):
        passed = decision("formal-screen", _gate_result(0.10))
        self.assertEqual(
            passed["outcome"],
            "PASS_C_E30_INCREMENT_GATE_RUN_OFFICIAL_RESTART",
        )
        self.assertFalse(passed["bootstrap_used_for_gate"])

        cases = (
            ("small", _gate_result(0.099), "INCREMENT_BELOW_0.10_PP"),
            (
                "joint-sensitive-decline",
                _gate_result(0.20, city=-0.01, road=-0.02),
                "CITY_ROAD_JOINT_DECLINE",
            ),
            (
                "mechanism",
                _gate_result(0.20, healthy=False),
                "CANDIDATE_E30_MECHANISM_UNHEALTHY",
            ),
        )
        for label, result, reason in cases:
            with self.subTest(label=label):
                verdict = decision("formal-screen", result)
                self.assertEqual(verdict["outcome"], "STOP_C_E30_INCREMENT_GATE")
                self.assertIn(reason, verdict["failed_conditions"])

        one_sensitive_class = decision(
            "formal-screen", _gate_result(0.20, city=-0.01, road=0.02)
        )
        self.assertEqual(
            one_sensitive_class["outcome"],
            "PASS_C_E30_INCREMENT_GATE_RUN_OFFICIAL_RESTART",
        )

    def test_smoke_never_makes_a_scientific_decision(self):
        verdict = decision(
            "smoke", _gate_result(99.0, city=-99.0, road=-99.0, healthy=False)
        )
        self.assertEqual(verdict["scientific_decision"], "NONE")
        self.assertIn("SMOKE", verdict["outcome"])

    def test_cli_refuses_to_overwrite_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            output.write_text("sealed", encoding="utf-8")
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--clean-dir",
                        directory,
                        "--candidate-dir",
                        directory,
                        "--output-path",
                        str(output),
                    ]
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "sealed")

    def test_main_compares_all_e16_to_e30_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_dir = root / "clean"
            candidate_dir = root / "candidate"
            clean_dir.mkdir()
            candidate_dir.mkdir()
            (clean_dir / "protocol.json").write_text(
                json.dumps(_protocol(CLEAN_VARIANT)), encoding="utf-8"
            )
            (candidate_dir / "protocol.json").write_text(
                json.dumps(_protocol(CANDIDATE_VARIANT)), encoding="utf-8"
            )
            for epoch in range(16, 31):
                train = _train(epoch)
                (clean_dir / f"train_e{epoch}.json").write_text(
                    json.dumps(train), encoding="utf-8"
                )
                (candidate_dir / f"train_e{epoch}.json").write_text(
                    json.dumps(train), encoding="utf-8"
                )
            (clean_dir / "evaluation_e30.json").write_text(
                json.dumps(_evaluation(8)), encoding="utf-8"
            )
            (candidate_dir / "evaluation_e30.json").write_text(
                json.dumps(_evaluation(9, class_delta=0.2)), encoding="utf-8"
            )
            output = root / "comparison.json"
            main(
                [
                    "--clean-dir",
                    str(clean_dir),
                    "--candidate-dir",
                    str(candidate_dir),
                    "--output-path",
                    str(output),
                    "--bootstrap-replicates",
                    "20",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["train_e16_to_e30_pair_audits"]), 15)
            self.assertEqual(
                payload["outcome"],
                "PASS_C_E30_INCREMENT_GATE_RUN_OFFICIAL_RESTART",
            )
            self.assertTrue(payload["not_equivalent_to_uninterrupted"])


if __name__ == "__main__":
    unittest.main()
