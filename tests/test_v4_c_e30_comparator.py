"""Pure tests for the three-arm V4-C E15->E30 restart kill-test comparator."""

from __future__ import annotations

from copy import deepcopy
import hashlib
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
    FORMAL_EVALUATION_EPOCHS,
    OFFICIAL_VARIANT,
    compare_epoch,
    decision,
    describe_trajectories,
    main,
    parse_args,
    validate_completed_arm,
    validate_protocol_triplet,
    validate_paired_clean_bindings,
    validate_train_epoch_triplet,
)


CLASS_NAMES = ("farmland", "city", "village", "water", "forest", "road", "other")


def _protocol(variant: str, *, scope: str = "formal-restart-screen") -> dict:
    smoke = scope == "smoke"
    source_key = {
        OFFICIAL_VARIANT: "official",
        CLEAN_VARIANT: "clean",
        CANDIDATE_VARIANT: "candidate",
    }[variant]
    protocol = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": 1,
        "continuation_mode": CONTINUATION_MODE,
        "source_epoch": 15,
        "first_continuation_epoch": 16,
        "target_epoch": 30,
        "stop_after_epoch": 16 if smoke else 30,
        "evaluation_epochs": [16] if smoke else [20, 25, 30],
        "not_equivalent_to_uninterrupted": True,
        "persistent_worker_state_restored": False,
        "three_arm_policy": "official_clean_C_all_required",
        "official_arm_execution_policy": "unconditional",
        "variant": variant,
        "source_variant": variant,
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
        "persistent_workers": True,
        "inference_batch_size": 32,
        "max_train_batches": 1 if smoke else None,
        "max_test_images": 1 if smoke else None,
        "scope": scope,
        "train_dataset_length": 3200,
        "full_test_length": 20,
        "evaluated_test_length": 1 if smoke else 20,
        "aux_fill": 0,
        "loss_change": "none",
        "restart_rng_fingerprints": {
            "python": "python-rng",
            "numpy": "numpy-rng",
            "torch_cpu": "torch-cpu-rng",
            "torch_cuda": ["torch-cuda-rng"],
            "loader_generator": "loader-rng",
        },
        "source_dir_resolved": f"/sealed/{source_key}",
        "source_protocol_sha256": f"{source_key}-protocol-sha",
        "source_git_commit": (
            "v4-a-source-commit"
            if variant in (OFFICIAL_VARIANT, CLEAN_VARIANT)
            else "v4-c-source-commit"
        ),
        "source_checkpoint_sha256": f"{source_key}-checkpoint-sha",
        "source_evaluation_sha256": f"{source_key}-evaluation-sha",
        "use_optical_stem": variant == CANDIDATE_VARIANT,
        "mask_padding_ignore": variant != OFFICIAL_VARIANT,
        "mask_fill": 0 if variant == OFFICIAL_VARIANT else 7,
        "optical_stem_location": (
            "post-ACFM-L0-pre-FRN-single-injection"
            if variant == CANDIDATE_VARIANT
            else None
        ),
        "paired_clean_dir": None if variant == CLEAN_VARIANT else "/outputs/clean",
    }
    if variant == CANDIDATE_VARIANT:
        protocol["clean_baseline_variant"] = CLEAN_VARIANT
        protocol["candidate_clean_lineage"] = {
            "clean_reference_dir": "/sealed/clean",
            "clean_reference_protocol_sha256": "clean-protocol-sha",
            "clean_reference_git_commit": "v4-a-source-commit",
            "clean_reference_checkpoint_sha256": "clean-checkpoint-sha",
            "clean_reference_evaluation_sha256": "clean-evaluation-sha",
        }
    return protocol


def _trace(raw_label: str, *, count: int = 10) -> list[dict]:
    return [
        {
            "batch": index + 1,
            "pair_sha256": f"pair-{index}",
            "optical_sha256": f"optical-{index}",
            "sar_sha256": f"sar-{index}",
            "raw_label_sha256": f"{raw_label}-{index}",
            "normalized_label_sha256": f"normalized-{index}",
        }
        for index in range(count)
    ]


def _mechanism(*, healthy: bool = True, observed: int = 10) -> dict:
    return {
        "optical_stem_mechanism": {
            "summary": {
                "observed_batches": observed,
                "all_readouts_finite": healthy,
                "stem_output_nonzero_batches": observed if healthy else 0,
                "projection_gradient_nonzero_batches": observed if healthy else 0,
                "upstream_gradient_nonzero_batches": observed if healthy else 0,
            }
        }
    }


def _train(
    epoch: int,
    variant: str,
    *,
    healthy: bool = True,
    trace_count: int = 10,
) -> dict:
    raw = "official-raw" if variant == OFFICIAL_VARIANT else "ignore-raw"
    result = {
        "epoch": epoch,
        "paired_data_sha256": f"normalized-paired-e{epoch}",
        "raw_label_sha256": f"{raw}-e{epoch}",
        "first_batch_trace": _trace(raw, count=trace_count),
    }
    if variant == CANDIDATE_VARIANT:
        result.update(_mechanism(healthy=healthy, observed=trace_count))
    result["variant"] = variant
    result["continuation_mode"] = CONTINUATION_MODE
    return result


def _confusion(correct: int) -> list[list[int]]:
    matrix = np.zeros((7, 7), dtype=np.int64)
    for class_index in range(7):
        matrix[class_index, class_index] = correct
        matrix[class_index, (class_index + 1) % 7] = 10 - correct
    return matrix.tolist()


def _evaluation(
    correct: int,
    *,
    epoch: int = 20,
    variant: str = CLEAN_VARIANT,
    image_count: int = 3,
) -> dict:
    confusions = [np.asarray(_confusion(correct), dtype=np.int64) for _ in range(image_count)]
    pooled = np.sum(confusions, axis=0)
    denominator = pooled.sum(axis=1) + pooled.sum(axis=0) - np.diag(pooled)
    class_values = np.divide(
        np.diag(pooled),
        denominator,
        out=np.full(7, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    valid = denominator > 0
    miou = float(np.mean(class_values[valid]))
    return {
        "epoch": epoch,
        "variant": variant,
        "continuation_mode": CONTINUATION_MODE,
        "evaluated_images": image_count,
        "label_sha256": "same-test-label-stream",
        "per_image": [
            {
                "index": index,
                "sample_name": f"image-{index}",
                "confusion": confusion.tolist(),
            }
            for index, confusion in enumerate(confusions)
        ],
        "aggregate": {
            "confusion": pooled.tolist(),
            "pixels": int(pooled.sum()),
            "miou": miou,
            "miou_percent": miou * 100.0,
            "class_iou_percent": {
                name: (None if np.isnan(value) else float(value * 100.0))
                for name, value in zip(CLASS_NAMES, class_values, strict=True)
            },
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_summary(
    arm_dir: Path,
    protocol: dict,
    variant: str,
    *,
    evaluation_epochs: tuple[int, ...] = FORMAL_EVALUATION_EPOCHS,
) -> None:
    stop = int(protocol["stop_after_epoch"])
    training = []
    for epoch in range(16, stop + 1):
        path = arm_dir / f"train_e{epoch}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        training.append(
            {
                "epoch": epoch,
                "paired_data_sha256": payload["paired_data_sha256"],
                "train_path": path.name,
                "train_sha256": _sha256(path),
            }
        )
    evaluations = []
    for epoch in evaluation_epochs:
        evaluation_path = arm_dir / f"evaluation_e{epoch}.json"
        checkpoint_path = arm_dir / f"checkpoint_e{epoch}.pth"
        checkpoint_path.write_bytes(f"checkpoint-{variant}-e{epoch}".encode())
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluations.append(
            {
                "epoch": epoch,
                "miou_percent": evaluation["aggregate"]["miou_percent"],
                "evaluation_path": evaluation_path.name,
                "evaluation_sha256": _sha256(evaluation_path),
                "checkpoint_path": checkpoint_path.name,
                "checkpoint_sha256": _sha256(checkpoint_path),
            }
        )
    outcome = (
        "PASS_RESTART_SMOKE_CONTRACT"
        if protocol["scope"] == "smoke"
        else {
            OFFICIAL_VARIANT: "COMPLETES_UNCONDITIONAL_OFFICIAL_RESTART_ARM",
            CLEAN_VARIANT: "COMPLETES_REQUIRED_CLEAN_RESTART_ARM",
            CANDIDATE_VARIANT: "COMPLETES_REQUIRED_C_RESTART_ARM",
        }[variant]
    )
    summary = {
        "status": "PASS",
        "outcome": outcome,
        "scientific_decision": "NONE",
        "artifact_type": f"{ARTIFACT_TYPE}_summary",
        "continuation_mode": CONTINUATION_MODE,
        "variant": variant,
        "scope": protocol["scope"],
        "git_commit": protocol["git_commit"],
        "source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
        "source_epoch": 15,
        "stop_after_epoch": stop,
        "not_equivalent_to_uninterrupted": True,
        "training": training,
        "evaluations": evaluations,
    }
    (arm_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _decision_record(
    epoch: int,
    c_minus_clean: float,
    *,
    c_minus_official: float = 0.2,
    city: float = 0.1,
    road: float = 0.1,
    healthy: bool = True,
) -> dict:
    return {
        "epoch": epoch,
        "clean_minus_official_pp": c_minus_official - c_minus_clean,
        "candidate_minus_clean_pp": c_minus_clean,
        "candidate_minus_official_pp": c_minus_official,
        "class_iou_delta_candidate_minus_clean_pp": {
            **{name: 0.0 for name in CLASS_NAMES},
            "city": city,
            "road": road,
        },
        "candidate_mechanism_health": {"healthy_for_formal_gate": healthy},
    }


class V4CE30ThreeArmComparatorTest(unittest.TestCase):
    def test_protocol_locks_three_arms_sources_rng_and_candidate_lineage(self):
        official = _protocol(OFFICIAL_VARIANT)
        clean = _protocol(CLEAN_VARIANT)
        candidate = _protocol(CANDIDATE_VARIANT)
        audit = validate_protocol_triplet(official, clean, candidate)
        self.assertTrue(audit["restart_rng_fingerprints_equal"])
        self.assertTrue(audit["candidate_clean_lineage_equal"])
        self.assertEqual(audit["evaluation_epochs"], [20, 25, 30])

        corruptions = (
            ("official artifact", official, "artifact_type", "wrong"),
            ("clean mode", clean, "continuation_mode", "uninterrupted"),
            ("candidate source variant", candidate, "source_variant", "wrong"),
            ("official mask", official, "mask_padding_ignore", True),
            ("clean mask", clean, "mask_padding_ignore", False),
            ("candidate stem", candidate, "use_optical_stem", False),
            ("eval epochs", candidate, "evaluation_epochs", [30]),
            ("git", candidate, "git_commit", "different"),
            ("workers", official, "train_workers", 2),
            ("rng", clean, "restart_rng_fingerprints", {"python": "drift"}),
            ("source seal", official, "source_checkpoint_sha256", ""),
            (
                "lineage",
                candidate,
                "candidate_clean_lineage",
                {"clean_reference_checkpoint_sha256": "drift"},
            ),
        )
        for label, target, field, value in corruptions:
            with self.subTest(label=label):
                bad_official = deepcopy(official)
                bad_clean = deepcopy(clean)
                bad_candidate = deepcopy(candidate)
                selected = {
                    id(official): bad_official,
                    id(clean): bad_clean,
                    id(candidate): bad_candidate,
                }[id(target)]
                selected[field] = value
                with self.assertRaisesRegex(RuntimeError, "protocol|seal|lineage|RNG"):
                    validate_protocol_triplet(bad_official, bad_clean, bad_candidate)

        for arm, field in (
            ("official", "three_arm_policy"),
            ("official", "official_arm_execution_policy"),
            ("candidate", "clean_baseline_variant"),
        ):
            with self.subTest(missing=f"{arm}.{field}"):
                payloads = {
                    "official": deepcopy(official),
                    "clean": deepcopy(clean),
                    "candidate": deepcopy(candidate),
                }
                del payloads[arm][field]
                with self.assertRaisesRegex(RuntimeError, "protocol"):
                    validate_protocol_triplet(
                        payloads["official"],
                        payloads["clean"],
                        payloads["candidate"],
                    )

    def test_paired_clean_directories_are_bound_to_supplied_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dirs = {
                "official": root / "official",
                "clean": root / "clean",
                "candidate": root / "candidate",
            }
            for value in dirs.values():
                value.mkdir()
            protocols = {
                "official": _protocol(OFFICIAL_VARIANT),
                "clean": _protocol(CLEAN_VARIANT),
                "candidate": _protocol(CANDIDATE_VARIANT),
            }
            protocols["official"]["paired_clean_dir"] = str(dirs["clean"])
            protocols["candidate"]["paired_clean_dir"] = str(dirs["clean"])
            audit = validate_paired_clean_bindings(protocols, dirs)
            self.assertTrue(audit["official_bound_to_supplied_clean"])
            protocols["candidate"]["paired_clean_dir"] = str(dirs["official"])
            with self.assertRaisesRegex(RuntimeError, "supplied clean"):
                validate_paired_clean_bindings(protocols, dirs)

    def test_smoke_protocol_requires_only_e16(self):
        audit = validate_protocol_triplet(
            _protocol(OFFICIAL_VARIANT, scope="smoke"),
            _protocol(CLEAN_VARIANT, scope="smoke"),
            _protocol(CANDIDATE_VARIANT, scope="smoke"),
        )
        self.assertEqual(audit["evaluation_epochs"], [16])

    def test_training_triplet_allows_only_official_raw_label_difference(self):
        official = _train(16, OFFICIAL_VARIANT)
        clean = _train(16, CLEAN_VARIANT)
        candidate = _train(16, CANDIDATE_VARIANT)
        audit = validate_train_epoch_triplet(
            official, clean, candidate, 16, expected_trace_count=10
        )
        self.assertTrue(audit["paired_data_sha256_equal_three_arms"])
        self.assertTrue(audit["official_raw_label_differs_from_clean_by_design"])
        self.assertFalse(audit["official_raw_label_used_for_pairing_gate"])

        drift = deepcopy(official)
        drift["first_batch_trace"][4]["sar_sha256"] = "drift"
        with self.assertRaisesRegex(RuntimeError, "traces differ"):
            validate_train_epoch_triplet(
                drift, clean, candidate, 16, expected_trace_count=10
            )
        drift = deepcopy(candidate)
        drift["raw_label_sha256"] = "drift"
        with self.assertRaisesRegex(RuntimeError, "clean/C"):
            validate_train_epoch_triplet(
                official, clean, drift, 16, expected_trace_count=10
            )
        drift = deepcopy(official)
        drift["paired_data_sha256"] = "drift"
        with self.assertRaisesRegex(RuntimeError, "paired-data"):
            validate_train_epoch_triplet(
                drift, clean, candidate, 16, expected_trace_count=10
            )

    def test_formal_official_raw_labels_must_expose_design_difference(self):
        official = _train(16, OFFICIAL_VARIANT)
        clean = _train(16, CLEAN_VARIANT)
        candidate = _train(16, CANDIDATE_VARIANT)
        official["raw_label_sha256"] = clean["raw_label_sha256"]
        with self.assertRaisesRegex(RuntimeError, "unexpectedly equal"):
            validate_train_epoch_triplet(
                official, clean, candidate, 16, expected_trace_count=10
            )

    def test_trace_count_is_sealed(self):
        official = _train(16, OFFICIAL_VARIANT, trace_count=1)
        clean = _train(16, CLEAN_VARIANT, trace_count=1)
        candidate = _train(16, CANDIDATE_VARIANT, trace_count=1)
        with self.assertRaisesRegex(RuntimeError, "expected 10"):
            validate_train_epoch_triplet(
                official, clean, candidate, 16, expected_trace_count=10
            )
        audit = validate_train_epoch_triplet(
            official, clean, candidate, 16, expected_trace_count=1
        )
        self.assertEqual(audit["trace_record_count"], 1)

    def test_compare_epoch_reports_all_three_contrasts_classes_and_bootstraps(self):
        result = compare_epoch(
            _evaluation(7, variant=OFFICIAL_VARIANT),
            _evaluation(8, variant=CLEAN_VARIANT),
            _evaluation(9, variant=CANDIDATE_VARIANT),
            _train(20, CANDIDATE_VARIANT),
            epoch=20,
            expected_image_count=3,
            bootstrap_replicates=100,
            bootstrap_seed=7,
        )
        self.assertGreater(result["clean_minus_official_pp"], 0.0)
        self.assertGreater(result["candidate_minus_clean_pp"], 0.0)
        self.assertGreater(result["candidate_minus_official_pp"], 0.0)
        self.assertGreater(
            result["class_iou_deltas"]["clean_minus_official_pp"]["city"], 0.0
        )
        self.assertGreater(
            result["class_iou_deltas"]["candidate_minus_clean_pp"]["road"], 0.0
        )
        self.assertEqual(
            set(result["paired_image_bootstrap"]),
            {
                "clean_minus_official",
                "candidate_minus_clean",
                "candidate_minus_official",
            },
        )
        for bootstrap in result["paired_image_bootstrap"].values():
            self.assertEqual(bootstrap["role"], "descriptive_only_not_a_gate")
        self.assertTrue(
            result["candidate_mechanism_health"]["healthy_for_formal_gate"]
        )

    def test_compare_epoch_rejects_test_label_and_image_order_drift(self):
        official = _evaluation(7, variant=OFFICIAL_VARIANT)
        clean = _evaluation(8, variant=CLEAN_VARIANT)
        candidate = _evaluation(9, variant=CANDIDATE_VARIANT)
        drift = deepcopy(candidate)
        drift["label_sha256"] = "drift"
        with self.assertRaisesRegex(RuntimeError, "label streams differ"):
            compare_epoch(
                official,
                clean,
                drift,
                _train(20, CANDIDATE_VARIANT),
                epoch=20,
                expected_image_count=3,
                bootstrap_replicates=10,
                bootstrap_seed=7,
            )

    def test_evaluation_identity_count_and_aggregate_are_sealed(self):
        official = _evaluation(7, variant=OFFICIAL_VARIANT)
        clean = _evaluation(8, variant=CLEAN_VARIANT)
        candidate = _evaluation(9, variant=CANDIDATE_VARIANT)
        corruptions = []
        wrong_epoch = deepcopy(candidate)
        wrong_epoch["epoch"] = 25
        corruptions.append(("identity", wrong_epoch, "identity"))
        duplicate = deepcopy(candidate)
        duplicate["per_image"][1]["sample_name"] = duplicate["per_image"][0]["sample_name"]
        corruptions.append(("duplicate", duplicate, "duplicate"))
        short = deepcopy(candidate)
        short["per_image"] = short["per_image"][:2]
        short["evaluated_images"] = 2
        corruptions.append(("count", short, "identity|count"))
        stale_class = deepcopy(candidate)
        stale_class["aggregate"]["class_iou_percent"]["road"] += 1.0
        corruptions.append(("aggregate", stale_class, "differs from confusion"))
        for label, drift, pattern in corruptions:
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    compare_epoch(
                        official,
                        clean,
                        drift,
                        _train(20, CANDIDATE_VARIANT),
                        epoch=20,
                        expected_image_count=3,
                        bootstrap_replicates=10,
                        bootstrap_seed=7,
                    )
        drift = deepcopy(candidate)
        drift["per_image"][0]["sample_name"], drift["per_image"][1]["sample_name"] = (
            drift["per_image"][1]["sample_name"],
            drift["per_image"][0]["sample_name"],
        )
        with self.assertRaisesRegex(RuntimeError, "image order differs"):
            compare_epoch(
                official,
                clean,
                drift,
                _train(20, CANDIDATE_VARIANT),
                epoch=20,
                expected_image_count=3,
                bootstrap_replicates=10,
                bootstrap_seed=7,
            )

    def test_trajectory_shape_is_descriptive_and_separates_A(self):
        results = [
            {
                "epoch": epoch,
                "clean_minus_official_pp": a,
                "candidate_minus_clean_pp": c,
                "candidate_minus_official_pp": a + c,
            }
            for epoch, a, c in ((20, 0.8, 0.20), (25, 0.7, 0.15), (30, 0.6, 0.10))
        ]
        trajectories = describe_trajectories(results)
        self.assertEqual(trajectories["clean_minus_official"]["shape"], "nonincreasing")
        self.assertEqual(trajectories["candidate_minus_clean"]["shape"], "nonincreasing")
        self.assertFalse(trajectories["e20_e25_used_for_gate"])
        self.assertEqual(
            trajectories["clean_minus_official"]["role"],
            "descriptive_only_no_post_hoc_gate",
        )

    def test_e30_kill_gate_requires_increment_safety_mechanism_and_official(self):
        early = [
            _decision_record(20, -99.0, c_minus_official=-99.0),
            _decision_record(25, 99.0, c_minus_official=99.0),
        ]
        passed = decision("formal-restart-screen", early + [_decision_record(30, 0.10)])
        self.assertEqual(passed["outcome"], "SURVIVE_C_E30_KILL_TEST_NOT_CONFIRMED")
        self.assertTrue(passed["may_authorize_separate_confirmation"])
        self.assertFalse(passed["kill_test_pass_is_confirmation"])
        self.assertFalse(passed["e20_e25_used_for_gate"])
        self.assertFalse(passed["bootstrap_used_for_gate"])

        cases = (
            ("small", _decision_record(30, 0.099), "INCREMENT_BELOW_0.10_PP"),
            (
                "safety",
                _decision_record(30, 0.20, city=-0.01, road=-0.02),
                "CITY_ROAD_JOINT_DECLINE",
            ),
            (
                "mechanism",
                _decision_record(30, 0.20, healthy=False),
                "CANDIDATE_E30_MECHANISM_UNHEALTHY",
            ),
            (
                "official",
                _decision_record(30, 0.20, c_minus_official=0.0),
                "CANDIDATE_NOT_ABOVE_SAME_EPOCH_OFFICIAL",
            ),
        )
        for label, e30, reason in cases:
            with self.subTest(label=label):
                verdict = decision("formal-restart-screen", early + [e30])
                self.assertEqual(verdict["outcome"], "STOP_C_E30_KILL_TEST")
                self.assertFalse(verdict["may_authorize_separate_confirmation"])
                self.assertIn(reason, verdict["failed_conditions"])

    def test_smoke_never_makes_a_scientific_decision(self):
        record = _decision_record(16, 99.0)
        record["candidate_mechanism_health"]["healthy_for_smoke_contract"] = True
        verdict = decision("smoke", [record])
        self.assertEqual(verdict["scientific_decision"], "NONE")
        self.assertFalse(verdict["may_authorize_separate_confirmation"])
        self.assertIn("SMOKE", verdict["outcome"])
        record["candidate_mechanism_health"]["healthy_for_smoke_contract"] = False
        with self.assertRaisesRegex(RuntimeError, "mechanism is unhealthy"):
            decision("smoke", [record])

    def test_summary_status_and_all_artifact_hashes_are_required(self):
        def build_arm(root: Path) -> tuple[dict, dict]:
            protocol = _protocol(CANDIDATE_VARIANT, scope="smoke")
            protocol["evaluated_test_length"] = 1
            (root / "train_e16.json").write_text(
                json.dumps(_train(16, CANDIDATE_VARIANT, trace_count=1)),
                encoding="utf-8",
            )
            (root / "evaluation_e16.json").write_text(
                json.dumps(
                    _evaluation(
                        9,
                        epoch=16,
                        variant=CANDIDATE_VARIANT,
                        image_count=1,
                    )
                ),
                encoding="utf-8",
            )
            _write_summary(
                root,
                protocol,
                CANDIDATE_VARIANT,
                evaluation_epochs=(16,),
            )
            return protocol, json.loads(
                (root / "summary.json").read_text(encoding="utf-8")
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, summary = build_arm(root)
            audit = validate_completed_arm(
                root,
                protocol,
                summary,
                arm="candidate",
                variant=CANDIDATE_VARIANT,
            )
            self.assertTrue(audit["all_summary_artifact_sha256_verified"])
            failed = deepcopy(summary)
            failed["status"] = "INCOMPLETE"
            with self.assertRaisesRegex(RuntimeError, "completion contract"):
                validate_completed_arm(
                    root,
                    protocol,
                    failed,
                    arm="candidate",
                    variant=CANDIDATE_VARIANT,
                )
            train_path = root / "train_e16.json"
            original_train = train_path.read_bytes()
            train_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "train SHA256"):
                validate_completed_arm(
                    root,
                    protocol,
                    summary,
                    arm="candidate",
                    variant=CANDIDATE_VARIANT,
                )
            train_path.write_bytes(original_train)
            evaluation_path = root / "evaluation_e16.json"
            original_evaluation = evaluation_path.read_bytes()
            evaluation_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "evaluation SHA256"):
                validate_completed_arm(
                    root,
                    protocol,
                    summary,
                    arm="candidate",
                    variant=CANDIDATE_VARIANT,
                )
            evaluation_path.write_bytes(original_evaluation)
            checkpoint_path = root / "checkpoint_e16.pth"
            checkpoint_path.write_bytes(b"tampered checkpoint")
            with self.assertRaisesRegex(RuntimeError, "checkpoint SHA256"):
                validate_completed_arm(
                    root,
                    protocol,
                    summary,
                    arm="candidate",
                    variant=CANDIDATE_VARIANT,
                )

    def test_cli_requires_official_and_refuses_overwrite(self):
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--clean-dir",
                    "clean",
                    "--candidate-dir",
                    "candidate",
                    "--output-path",
                    "output.json",
                ]
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.json"
            output.write_text("sealed", encoding="utf-8")
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--official-dir",
                        directory,
                        "--clean-dir",
                        directory,
                        "--candidate-dir",
                        directory,
                        "--output-path",
                        str(output),
                    ]
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "sealed")

    def test_main_requires_all_epochs_and_emits_kill_test_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dirs = {
                OFFICIAL_VARIANT: root / "official",
                CLEAN_VARIANT: root / "clean",
                CANDIDATE_VARIANT: root / "candidate",
            }
            for variant, arm_dir in dirs.items():
                arm_dir.mkdir()
                protocol = _protocol(variant)
                protocol["evaluated_test_length"] = 3
                protocol["paired_clean_dir"] = (
                    None if variant == CLEAN_VARIANT else str(dirs[CLEAN_VARIANT])
                )
                (arm_dir / "protocol.json").write_text(
                    json.dumps(protocol), encoding="utf-8"
                )
                for epoch in range(16, 31):
                    (arm_dir / f"train_e{epoch}.json").write_text(
                        json.dumps(_train(epoch, variant)), encoding="utf-8"
                    )
            for epoch in FORMAL_EVALUATION_EPOCHS:
                values = {
                    OFFICIAL_VARIANT: _evaluation(
                        7, epoch=epoch, variant=OFFICIAL_VARIANT
                    ),
                    CLEAN_VARIANT: _evaluation(
                        8, epoch=epoch, variant=CLEAN_VARIANT
                    ),
                    CANDIDATE_VARIANT: _evaluation(
                        9, epoch=epoch, variant=CANDIDATE_VARIANT
                    ),
                }
                for variant, evaluation in values.items():
                    (dirs[variant] / f"evaluation_e{epoch}.json").write_text(
                        json.dumps(evaluation), encoding="utf-8"
                    )
            for variant, arm_dir in dirs.items():
                protocol = json.loads(
                    (arm_dir / "protocol.json").read_text(encoding="utf-8")
                )
                _write_summary(arm_dir, protocol, variant)
            output = root / "comparison.json"
            main(
                [
                    "--official-dir",
                    str(dirs[OFFICIAL_VARIANT]),
                    "--clean-dir",
                    str(dirs[CLEAN_VARIANT]),
                    "--candidate-dir",
                    str(dirs[CANDIDATE_VARIANT]),
                    "--output-path",
                    str(output),
                    "--bootstrap-replicates",
                    "20",
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["train_e16_to_stop_triplet_audits"]), 15)
            self.assertEqual(payload["evaluation_epochs"], [20, 25, 30])
            self.assertEqual(len(payload["epochs"]), 3)
            self.assertEqual(
                payload["outcome"], "SURVIVE_C_E30_KILL_TEST_NOT_CONFIRMED"
            )
            self.assertTrue(payload["kill_test_not_confirmation"])
            self.assertTrue(payload["not_equivalent_to_uninterrupted"])


if __name__ == "__main__":
    unittest.main()
