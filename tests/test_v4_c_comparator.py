"""Pure unit contracts for the V4-C three-way comparator.

These tests deliberately exercise the scientific gate independently from the
training runner.  In particular, a large aggregate gain must not hide the
pre-registered city/road safety failure, and the narrow gray-zone exception
must be backed by a still-growing stem rather than only a non-zero stem.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.compare_whu_v4_c_runs import (
    compare_epoch,
    decision,
    parse_args,
    validate_protocol_triplet,
)
from scripts.run_whu_v4_c_screen import CANDIDATE_VARIANT


CLASS_NAMES = ("farmland", "city", "village", "water", "forest", "road", "other")


def _mechanism(*, output_rms: float, healthy: bool = True) -> dict:
    """Shape returned by ``compare_epoch`` after inspecting a train artifact."""

    return {
        "healthy_for_formal_gate": healthy,
        "healthy_for_one_batch_smoke": healthy,
        "stem_output_mean_rms": output_rms,
        "summary": {
            "observed_batches": 10,
            "all_readouts_finite": healthy,
            "stem_output_nonzero_batches": 10 if healthy else 0,
            "projection_gradient_nonzero_batches": 10 if healthy else 0,
            "upstream_gradient_nonzero_batches": 10 if healthy else 0,
        },
    }


def _decision_record(
    epoch: int,
    delta: float,
    *,
    output_rms: float,
    city_delta: float = 0.01,
    road_delta: float = 0.01,
    healthy: bool = True,
) -> dict:
    class_delta = {name: 0.0 for name in CLASS_NAMES}
    class_delta.update(city=city_delta, road=road_delta)
    return {
        "epoch": epoch,
        "candidate_minus_clean_pp": delta,
        "candidate_minus_official_pp": delta + 0.9,
        "candidate_miou_percent": 52.0 + delta,
        "official_miou_percent": 51.0,
        "class_iou_delta_candidate_minus_clean_pp": class_delta,
        "mechanism_health": _mechanism(
            output_rms=output_rms,
            healthy=healthy,
        ),
    }


def _protocol_triplet() -> tuple[dict, dict, dict]:
    common = {
        "seed": 42,
        "scheduler_horizon_epochs": 50,
        "stop_after_epoch": 15,
        "evaluation_epochs": [5, 10, 15],
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
        "scope": "formal-screen",
        "train_dataset_length": 3200,
        "full_test_length": 20,
        "evaluated_test_length": 20,
    }
    official = {
        **common,
        "variant": "official",
        "initial_model_state_sha256": "sealed-initialization",
        "mask_padding_ignore": False,
        "mask_fill": 0,
        "aux_fill": 0,
    }
    clean = {
        **common,
        "variant": "mask-ignore",
        "initial_model_state_sha256": "sealed-initialization",
        "mask_padding_ignore": True,
        "mask_fill": 7,
        "aux_fill": 0,
    }
    candidate = {
        **common,
        "variant": CANDIDATE_VARIANT,
        "clean_baseline_variant": "mask-ignore",
        "clean_baseline_initial_state_sha256": "sealed-initialization",
        "mask_padding_ignore": True,
        "mask_fill": 7,
        "aux_fill": 0,
        "loss_change": "none",
        "use_optical_stem": True,
        "optical_stem_location": "post-ACFM-L0-pre-FRN-single-injection",
        "shared_initialization_audit": {
            "shared_parameters_bitwise_equal": True,
            "shared_parameter_records": {
                "decoder.weight": {
                    "bitwise_equal": True,
                    "clean_sha256": "same",
                    "candidate_sha256": "same",
                }
            },
        },
        "optimizer_membership_audit": {
            "stem_parameters_present_exactly_once": True,
        },
        "step0_prediction_probe": {
            "prediction_torch_equal": True,
            "prediction_sha256_equal": True,
            "clean_prediction_sha256": "same-prediction",
            "candidate_prediction_sha256": "same-prediction",
        },
    }
    return official, clean, candidate


def _confusion(correct: int) -> list[list[int]]:
    """Seven balanced classes with errors moved to the next class."""

    matrix = np.zeros((7, 7), dtype=np.int64)
    for class_index in range(7):
        matrix[class_index, class_index] = correct
        matrix[class_index, (class_index + 1) % 7] = 10 - correct
    return matrix.tolist()


def _train_payload(*, raw_label: str, correct_pair: str = "paired") -> dict:
    batches = [
        {
            "stem_output": {
                "rms": 0.1 + index * 0.01,
                "nonzero_fraction": 1.0,
                "finite": True,
            },
            "projection_gradient_l2": 1.0,
            "projection_gradient_finite": True,
            "upstream_gradient_l2": 1.0,
            "upstream_gradient_finite": True,
        }
        for index in range(10)
    ]
    return {
        "paired_data_sha256": correct_pair,
        "raw_label_sha256": raw_label,
        "valid_fraction": 0.86,
        "first_batch_trace": [
            {
                "pair_sha256": "pair-1",
                "optical_sha256": "optical-1",
                "sar_sha256": "sar-1",
                "raw_label_sha256": raw_label,
                "normalized_label_sha256": "normalized-1",
            }
        ],
        "optical_stem_mechanism": {
            "summary": {
                "observed_batches": 10,
                "all_readouts_finite": True,
                "stem_output_nonzero_batches": 10,
                "projection_gradient_nonzero_batches": 10,
                "upstream_gradient_nonzero_batches": 10,
            },
            "first_ten_batches": batches,
        },
    }


def _evaluation_payload(*, correct: int, class_offset: float = 0.0) -> dict:
    return {
        "label_sha256": "same-test-labels",
        "per_image": [
            {"sample_name": "image-a", "confusion": _confusion(correct)},
            {"sample_name": "image-b", "confusion": _confusion(correct)},
        ],
        "aggregate": {
            "class_iou_percent": {
                name: 40.0 + index + class_offset
                for index, name in enumerate(CLASS_NAMES)
            }
        },
    }


class V4CComparatorTest(unittest.TestCase):
    def test_strong_gain_cannot_override_city_road_safety_stop(self):
        records = [
            _decision_record(5, 0.40, output_rms=0.10),
            _decision_record(10, 0.30, output_rms=0.20),
            _decision_record(
                15,
                0.25,
                output_rms=0.30,
                city_delta=-0.01,
                road_delta=-0.02,
            ),
        ]
        verdict = decision("formal-screen", records)
        self.assertIn("SAFETY", verdict["outcome"])
        self.assertNotEqual(verdict["scientific_decision"], "EXTEND_ONCE_TO_E30")

    def test_only_one_sensitive_class_declining_does_not_trigger_joint_stop(self):
        records = [
            _decision_record(5, 0.40, output_rms=0.10),
            _decision_record(10, 0.30, output_rms=0.20),
            _decision_record(
                15,
                0.25,
                output_rms=0.30,
                city_delta=-0.01,
                road_delta=0.02,
            ),
        ]
        verdict = decision("formal-screen", records)
        self.assertEqual(verdict["outcome"], "PASS_C_E15_EXTEND_TO_E30")

    def test_gray_zone_requires_metric_and_stem_output_to_keep_rising(self):
        healthy_but_flat_stem = [
            _decision_record(5, 0.01, output_rms=0.20),
            _decision_record(10, 0.02, output_rms=0.20),
            _decision_record(15, 0.04, output_rms=0.19),
        ]
        verdict = decision("formal-screen", healthy_but_flat_stem)
        self.assertEqual(verdict["outcome"], "STOP_C_E15")

        rising_stem = [
            _decision_record(5, 0.01, output_rms=0.10),
            _decision_record(10, 0.02, output_rms=0.20),
            _decision_record(15, 0.04, output_rms=0.30),
        ]
        verdict = decision("formal-screen", rising_stem)
        self.assertEqual(verdict["outcome"], "PASS_C_E15_EXTEND_TO_E30")
        self.assertEqual(verdict["gate_band"], "GRAY_STRICTLY_RISING_HEALTHY")

    def test_gray_zone_rejects_inactive_upstream_gradient(self):
        records = [
            _decision_record(5, 0.01, output_rms=0.10),
            _decision_record(10, 0.02, output_rms=0.20),
            _decision_record(15, 0.04, output_rms=0.30, healthy=False),
        ]
        self.assertEqual(decision("formal-screen", records)["outcome"], "STOP_C_E15")

    def test_smoke_never_issues_a_scientific_promotion_decision(self):
        record = _decision_record(
            1,
            9.0,
            output_rms=0.0,
            city_delta=-9.0,
            road_delta=-9.0,
        )
        verdict = decision("smoke", [record])
        self.assertEqual(verdict["scientific_decision"], "NONE")

    def test_protocol_locks_clean_mask_and_single_injection(self):
        official, clean, candidate = _protocol_triplet()
        validate_protocol_triplet(official, clean, candidate)

        corruptions = (
            ("official mask", official, "mask_padding_ignore", True),
            ("clean mask", clean, "mask_padding_ignore", False),
            ("candidate mask", candidate, "mask_padding_ignore", False),
            ("candidate stem", candidate, "use_optical_stem", False),
            (
                "candidate injection",
                candidate,
                "optical_stem_location",
                "post-ACFM-L0-and-L1-double-injection",
            ),
            ("candidate loss", candidate, "loss_change", "new-loss"),
        )
        for label, target, field, value in corruptions:
            with self.subTest(label=label):
                bad_official = deepcopy(official)
                bad_clean = deepcopy(clean)
                bad_candidate = deepcopy(candidate)
                if target is official:
                    bad_official[field] = value
                elif target is clean:
                    bad_clean[field] = value
                else:
                    bad_candidate[field] = value
                with self.assertRaisesRegex(
                    RuntimeError, "mask|protocol|candidate|stem|loss|official|clean"
                ):
                    validate_protocol_triplet(
                        bad_official,
                        bad_clean,
                        bad_candidate,
                    )

    def test_compare_epoch_reports_primary_absolute_bootstrap_and_class_deltas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official_dir = root / "official"
            clean_dir = root / "clean"
            candidate_dir = root / "candidate"
            for path in (official_dir, clean_dir, candidate_dir):
                path.mkdir()

            (official_dir / "train_e15.json").write_text(
                json.dumps(_train_payload(raw_label="official-raw")),
                encoding="utf-8",
            )
            clean_train = _train_payload(raw_label="clean-raw")
            candidate_train = _train_payload(raw_label="clean-raw")
            (clean_dir / "train_e15.json").write_text(
                json.dumps(clean_train), encoding="utf-8"
            )
            (candidate_dir / "train_e15.json").write_text(
                json.dumps(candidate_train), encoding="utf-8"
            )
            (official_dir / "evaluation_e15.json").write_text(
                json.dumps(_evaluation_payload(correct=8, class_offset=0.0)),
                encoding="utf-8",
            )
            (clean_dir / "evaluation_e15.json").write_text(
                json.dumps(_evaluation_payload(correct=9, class_offset=1.0)),
                encoding="utf-8",
            )
            (candidate_dir / "evaluation_e15.json").write_text(
                json.dumps(_evaluation_payload(correct=10, class_offset=2.5)),
                encoding="utf-8",
            )

            result = compare_epoch(
                official_dir,
                clean_dir,
                candidate_dir,
                15,
                bootstrap_replicates=100,
                bootstrap_seed=7,
            )

        self.assertGreater(result["candidate_minus_clean_pp"], 0.0)
        self.assertGreater(
            result["candidate_minus_official_pp"],
            result["candidate_minus_clean_pp"],
        )
        self.assertEqual(
            result["class_iou_delta_candidate_minus_clean_pp"]["city"], 1.5
        )
        self.assertGreater(
            result["paired_bootstrap_candidate_minus_clean"]["low_pp"], 0.0
        )

    def test_compare_epoch_rejects_data_or_test_order_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / name for name in ("official", "clean", "candidate")}
            for path in paths.values():
                path.mkdir()
                (path / "evaluation_e15.json").write_text(
                    json.dumps(_evaluation_payload(correct=9)), encoding="utf-8"
                )
            (paths["official"] / "train_e15.json").write_text(
                json.dumps(_train_payload(raw_label="official")), encoding="utf-8"
            )
            (paths["clean"] / "train_e15.json").write_text(
                json.dumps(_train_payload(raw_label="clean")), encoding="utf-8"
            )
            (paths["candidate"] / "train_e15.json").write_text(
                json.dumps(_train_payload(raw_label="clean", correct_pair="drift")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "data streams differ"):
                compare_epoch(
                    paths["official"],
                    paths["clean"],
                    paths["candidate"],
                    15,
                    bootstrap_replicates=10,
                    bootstrap_seed=7,
                )

    def test_cli_refuses_to_overwrite_output(self):
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


if __name__ == "__main__":
    unittest.main()
