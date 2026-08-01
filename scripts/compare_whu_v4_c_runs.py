"""Compare one V4-C candidate with sealed V4-A clean and official artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_whu_v4_a_runs import (  # noqa: E402
    paired_bootstrap_ci,
    per_image_confusions,
    pooled_miou,
    read_json,
)
from scripts.run_whu_v4_a_screen import write_json_atomic  # noqa: E402
from scripts.run_whu_v4_c_screen import CANDIDATE_VARIANT  # noqa: E402


C_GRAY_MIN_PP = 0.03
C_STANDARD_MIN_PP = 0.05
C_STRONG_MIN_PP = 0.20
C_TRAJECTORY_TOLERANCE_PP = 0.03
C_E30_SECOND_SEED_MIN_PP = 0.10
FAITHFUL_HISTORICAL_BEST_MIOU_PERCENT = 54.145880
C_DECISION_RULE_VERSION = "v4_c_incremental_screen_v1"
OPTICAL_STEM_LOCATION = "post-ACFM-L0-pre-FRN-single-injection"


def _protocol_mismatches(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    fields: Sequence[str],
) -> dict[str, tuple[Any, Any]]:
    return {
        field: (reference.get(field), candidate.get(field))
        for field in fields
        if reference.get(field) != candidate.get(field)
    }


def validate_protocol_triplet(
    official: Mapping[str, Any],
    clean: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    if official.get("variant") != "official":
        raise RuntimeError("official artifact is not the V4-A official variant")
    if clean.get("variant") != "mask-ignore":
        raise RuntimeError("clean artifact is not the V4-A mask-ignore variant")
    if candidate.get("variant") != CANDIDATE_VARIANT:
        raise RuntimeError("candidate artifact is not mask-ignore+optical-stem")
    if candidate.get("clean_baseline_variant") != "mask-ignore":
        raise RuntimeError("candidate did not bind itself to the clean baseline")
    expected_variant_contracts = {
        "official": {
            "mask_padding_ignore": False,
            "mask_fill": 0,
            "aux_fill": 0,
        },
        "clean": {
            "mask_padding_ignore": True,
            "mask_fill": 7,
            "aux_fill": 0,
        },
        "candidate": {
            "mask_padding_ignore": True,
            "mask_fill": 7,
            "aux_fill": 0,
            "loss_change": "none",
            "use_optical_stem": True,
            "optical_stem_location": OPTICAL_STEM_LOCATION,
        },
    }
    variant_payloads = {
        "official": official,
        "clean": clean,
        "candidate": candidate,
    }
    contract_mismatches = {
        name: {
            field: (expected, variant_payloads[name].get(field))
            for field, expected in expected_fields.items()
            if variant_payloads[name].get(field) != expected
        }
        for name, expected_fields in expected_variant_contracts.items()
    }
    contract_mismatches = {
        name: values for name, values in contract_mismatches.items() if values
    }
    if contract_mismatches:
        raise RuntimeError(
            f"V4-C mask/stem protocol contract differs: {contract_mismatches}"
        )
    fields = (
        "seed",
        "scheduler_horizon_epochs",
        "stop_after_epoch",
        "evaluation_epochs",
        "model_name",
        "dataset_name",
        "num_modalities",
        "backbone_type",
        "use_lora",
        "train_batch_size_per_gpu",
        "train_workers",
        "inference_batch_size",
        "max_train_batches",
        "max_test_images",
        "scope",
        "train_dataset_length",
        "full_test_length",
        "evaluated_test_length",
    )
    mismatches = {
        "official_vs_clean": _protocol_mismatches(clean, official, fields),
        "candidate_vs_clean": _protocol_mismatches(clean, candidate, fields),
    }
    mismatches = {name: value for name, value in mismatches.items() if value}
    if mismatches:
        raise RuntimeError(f"V4-C protocol triplet differs: {mismatches}")

    clean_initial = clean.get("initial_model_state_sha256")
    if candidate.get("clean_baseline_initial_state_sha256") != clean_initial:
        raise RuntimeError(
            "candidate's in-process clean initialization does not reproduce the "
            "sealed V4-A clean initialization"
        )
    if official.get("initial_model_state_sha256") != clean_initial:
        raise RuntimeError("sealed V4-A official and clean initializations differ")

    shared = candidate.get("shared_initialization_audit", {})
    if not shared.get("shared_parameters_bitwise_equal"):
        raise RuntimeError("candidate lacks per-key shared-parameter equality")
    records = shared.get("shared_parameter_records")
    if not isinstance(records, dict) or not records:
        raise RuntimeError("candidate lacks per-key shared-parameter hashes")
    unequal = [
        name
        for name, record in records.items()
        if not record.get("bitwise_equal")
        or record.get("clean_sha256") != record.get("candidate_sha256")
    ]
    if unequal:
        raise RuntimeError(f"candidate shared parameters differ: {unequal}")

    optimizer = candidate.get("optimizer_membership_audit", {})
    if not optimizer.get("stem_parameters_present_exactly_once"):
        raise RuntimeError("candidate stem optimizer membership is not exactly once")
    step0 = candidate.get("step0_prediction_probe", {})
    if not step0.get("prediction_torch_equal") or not step0.get(
        "prediction_sha256_equal"
    ):
        raise RuntimeError("candidate failed the step-0 prediction equality probe")
    if step0.get("clean_prediction_sha256") != step0.get(
        "candidate_prediction_sha256"
    ):
        raise RuntimeError("candidate step-0 prediction SHA differs")


def _trace_signature(record: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            item.get("pair_sha256"),
            item.get("optical_sha256"),
            item.get("sar_sha256"),
            item.get("raw_label_sha256"),
            item.get("normalized_label_sha256"),
        )
        for item in record.get("first_batch_trace", [])
    ]


def mechanism_health(training: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = training.get("optical_stem_mechanism", {})
    summary = mechanism.get("summary", {})
    batches = mechanism.get("first_ten_batches", [])
    output_rms_values = [
        float(record["stem_output"]["rms"])
        for record in batches
        if isinstance(record, Mapping)
        and isinstance(record.get("stem_output"), Mapping)
        and record["stem_output"].get("rms") is not None
    ]
    output_rms = summary.get("mean_stem_output_rms_first_ten")
    if output_rms is None and output_rms_values:
        output_rms = float(np.mean(output_rms_values))
    output_rms_finite = output_rms is not None and bool(np.isfinite(output_rms))
    observed = int(summary.get("observed_batches", 0))
    healthy = bool(
        observed > 0
        and summary.get("all_readouts_finite")
        and output_rms_finite
        and int(summary.get("projection_gradient_nonzero_batches", 0)) > 0
        and int(summary.get("stem_output_nonzero_batches", 0)) > 0
        and int(summary.get("upstream_gradient_nonzero_batches", 0)) > 0
    )
    smoke_healthy = bool(
        observed == 1
        and summary.get("all_readouts_finite")
        and summary.get("first_batch_zero_output")
        and summary.get("first_batch_projection_gradient_nonzero")
        and summary.get("first_batch_upstream_gradient_delayed")
    )
    return {
        "healthy_for_formal_gate": healthy,
        "healthy_for_one_batch_smoke": smoke_healthy,
        "stem_output_mean_rms": output_rms,
        "summary": summary,
    }


def city_road_safety(record: Mapping[str, Any]) -> dict[str, Any]:
    class_delta = record.get("class_iou_delta_candidate_minus_clean_pp")
    if not isinstance(class_delta, Mapping):
        return {
            "evidence_present": False,
            "passes": False,
            "city_delta_pp": None,
            "road_delta_pp": None,
            "joint_decline": None,
        }
    city = class_delta.get("city")
    road = class_delta.get("road")
    evidence_present = city is not None and road is not None
    joint_decline = bool(
        evidence_present and float(city) < 0.0 and float(road) < 0.0
    )
    return {
        "evidence_present": evidence_present,
        "passes": bool(evidence_present and not joint_decline),
        "city_delta_pp": None if city is None else float(city),
        "road_delta_pp": None if road is None else float(road),
        "joint_decline": joint_decline if evidence_present else None,
    }


def formal_safety_stop(
    epoch: int, record: Mapping[str, Any]
) -> dict[str, Any] | None:
    safety = city_road_safety(record)
    if not safety["evidence_present"]:
        return {
            "outcome": f"FAIL_C_E{epoch}_SAFETY_EVIDENCE_MISSING",
            "scientific_decision": "FIX_EVIDENCE_BEFORE_PROMOTION",
            "decision_rule_version": C_DECISION_RULE_VERSION,
            "city_road_safety": safety,
        }
    if not safety["passes"]:
        return {
            "outcome": f"STOP_C_E{epoch}_CITY_ROAD_SAFETY",
            "scientific_decision": "STOP_C_ARM_SAFETY",
            "decision_rule_version": C_DECISION_RULE_VERSION,
            "city_road_safety": safety,
        }
    mechanism = record.get("mechanism_health", {})
    if not mechanism.get("healthy_for_formal_gate"):
        return {
            "outcome": f"STOP_C_E{epoch}",
            "scientific_decision": "STOP_C_ARM",
            "decision_rule_version": C_DECISION_RULE_VERSION,
            "stop_reason": "MECHANISM_INACTIVE",
            "city_road_safety": safety,
        }
    return None


def _class_delta(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float | None]:
    return {
        name: (
            None
            if reference[name] is None or candidate[name] is None
            else float(candidate[name] - reference[name])
        )
        for name in reference
    }


def compare_epoch(
    official_dir: Path,
    clean_dir: Path,
    candidate_dir: Path,
    epoch: int,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    official_train = read_json(official_dir / f"train_e{epoch}.json")
    clean_train = read_json(clean_dir / f"train_e{epoch}.json")
    candidate_train = read_json(candidate_dir / f"train_e{epoch}.json")
    if clean_train.get("paired_data_sha256") != candidate_train.get(
        "paired_data_sha256"
    ):
        raise RuntimeError(f"epoch {epoch} C and clean data streams differ")
    if clean_train.get("raw_label_sha256") != candidate_train.get(
        "raw_label_sha256"
    ):
        raise RuntimeError(f"epoch {epoch} C and clean raw labels differ")
    if _trace_signature(clean_train) != _trace_signature(candidate_train):
        raise RuntimeError(f"epoch {epoch} C and clean first-batch traces differ")
    if official_train.get("paired_data_sha256") != candidate_train.get(
        "paired_data_sha256"
    ):
        raise RuntimeError(f"epoch {epoch} official normalized data stream differs")

    official_eval = read_json(official_dir / f"evaluation_e{epoch}.json")
    clean_eval = read_json(clean_dir / f"evaluation_e{epoch}.json")
    candidate_eval = read_json(candidate_dir / f"evaluation_e{epoch}.json")
    label_hashes = {
        official_eval.get("label_sha256"),
        clean_eval.get("label_sha256"),
        candidate_eval.get("label_sha256"),
    }
    if len(label_hashes) != 1:
        raise RuntimeError(f"epoch {epoch} test label streams differ")
    image_orders = [
        [item.get("sample_name") for item in evaluation["per_image"]]
        for evaluation in (official_eval, clean_eval, candidate_eval)
    ]
    if image_orders[0] != image_orders[1] or image_orders[1] != image_orders[2]:
        raise RuntimeError(f"epoch {epoch} test image orders differ")

    official_confusions = per_image_confusions(official_eval)
    clean_confusions = per_image_confusions(clean_eval)
    candidate_confusions = per_image_confusions(candidate_eval)
    official_miou = pooled_miou(official_confusions)
    clean_miou = pooled_miou(clean_confusions)
    candidate_miou = pooled_miou(candidate_confusions)
    clean_class = clean_eval["aggregate"]["class_iou_percent"]
    official_class = official_eval["aggregate"]["class_iou_percent"]
    candidate_class = candidate_eval["aggregate"]["class_iou_percent"]
    return {
        "epoch": epoch,
        "paired_data_sha256": candidate_train["paired_data_sha256"],
        "raw_label_sha256": candidate_train["raw_label_sha256"],
        "candidate_and_clean_data_exactly_paired": True,
        "official_miou_percent": official_miou,
        "clean_miou_percent": clean_miou,
        "candidate_miou_percent": candidate_miou,
        "candidate_minus_clean_pp": candidate_miou - clean_miou,
        "candidate_minus_official_pp": candidate_miou - official_miou,
        "clean_minus_official_pp": clean_miou - official_miou,
        "class_iou_delta_candidate_minus_clean_pp": _class_delta(
            clean_class, candidate_class
        ),
        "class_iou_delta_candidate_minus_official_pp": _class_delta(
            official_class, candidate_class
        ),
        "paired_bootstrap_candidate_minus_clean": paired_bootstrap_ci(
            clean_confusions,
            candidate_confusions,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + epoch,
        ),
        "mechanism_health": mechanism_health(candidate_train),
    }


def decision(scope: str, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if scope == "smoke":
        mechanism = results[-1].get("mechanism_health", {}) if results else {}
        if not mechanism.get("healthy_for_one_batch_smoke"):
            return {
                "outcome": "FAIL_C_SMOKE_MECHANISM",
                "scientific_decision": "FIX_BEFORE_FORMAL_SCREEN",
            }
        return {
            "outcome": "PASS_C_SMOKE_PAIR_AND_MECHANISM_CONTRACT",
            "scientific_decision": "NONE",
        }

    by_epoch = {int(record["epoch"]): record for record in results}
    if 50 in by_epoch:
        record = by_epoch[50]
        safety_stop = formal_safety_stop(50, record)
        if safety_stop is not None:
            return safety_stop
        delta = float(record["candidate_minus_clean_pp"])
        absolute = float(record["candidate_miou_percent"])
        success = bool(
            delta >= C_STRONG_MIN_PP
            and absolute > float(record["official_miou_percent"])
            and absolute > FAITHFUL_HISTORICAL_BEST_MIOU_PERCENT
        )
        return {
            "outcome": "PASS_C_E50_METHOD_SUCCESS" if success else "STOP_C_E50",
            "scientific_decision": "METHOD_SUCCESS" if success else "WEAK_OR_FAILED",
            "decision_rule_version": C_DECISION_RULE_VERSION,
            "primary_delta_pp": delta,
        }
    if 30 in by_epoch:
        record = by_epoch[30]
        safety_stop = formal_safety_stop(30, record)
        if safety_stop is not None:
            return safety_stop
        delta = float(record["candidate_minus_clean_pp"])
        pass_gate = bool(
            delta >= C_E30_SECOND_SEED_MIN_PP
            and float(record["candidate_miou_percent"])
            > float(record["official_miou_percent"])
        )
        return {
            "outcome": (
                "PASS_C_E30_AUTHORIZE_SECOND_SEED"
                if pass_gate
                else "STOP_C_E30"
            ),
            "scientific_decision": (
                "RUN_SECOND_SEED" if pass_gate else "DO_NOT_RUN_SECOND_SEED"
            ),
            "decision_rule_version": C_DECISION_RULE_VERSION,
            "primary_delta_pp": delta,
        }
    if 15 not in by_epoch:
        return {"outcome": "PENDING_C_E15", "scientific_decision": "NONE"}

    safety_stop = formal_safety_stop(15, by_epoch[15])
    if safety_stop is not None:
        return safety_stop
    delta15 = float(by_epoch[15]["candidate_minus_clean_pp"])
    delta10 = (
        None if 10 not in by_epoch else float(by_epoch[10]["candidate_minus_clean_pp"])
    )
    delta5 = (
        None if 5 not in by_epoch else float(by_epoch[5]["candidate_minus_clean_pp"])
    )
    mechanism_healthy = bool(
        by_epoch[15].get("mechanism_health", {}).get("healthy_for_formal_gate")
    )
    stem_output_rms = {
        epoch: by_epoch[epoch].get("mechanism_health", {}).get(
            "stem_output_mean_rms"
        )
        for epoch in (5, 10, 15)
        if epoch in by_epoch
    }
    stem_output_strictly_rising = bool(
        all(epoch in stem_output_rms for epoch in (5, 10, 15))
        and all(stem_output_rms[epoch] is not None for epoch in (5, 10, 15))
        and float(stem_output_rms[5]) < float(stem_output_rms[10])
        < float(stem_output_rms[15])
    )
    strong = delta15 >= C_STRONG_MIN_PP
    standard = bool(
        C_STANDARD_MIN_PP <= delta15 < C_STRONG_MIN_PP
        and delta10 is not None
        and delta15 >= delta10 - C_TRAJECTORY_TOLERANCE_PP
    )
    gray = bool(
        C_GRAY_MIN_PP <= delta15 < C_STANDARD_MIN_PP
        and delta5 is not None
        and delta10 is not None
        and delta5 < delta10 < delta15
        and mechanism_healthy
        and stem_output_strictly_rising
    )
    extend = strong or standard or gray
    if strong:
        band = "STRONG_REGARDLESS_OF_TREND"
    elif standard:
        band = "STANDARD_NONDECLINING"
    elif gray:
        band = "GRAY_STRICTLY_RISING_HEALTHY"
    elif delta15 >= C_STANDARD_MIN_PP:
        band = "POSITIVE_BUT_DECLINING"
    elif delta15 >= C_GRAY_MIN_PP:
        band = "GRAY_NOT_QUALIFIED"
    else:
        band = "BELOW_PRACTICAL_GATE"
    return {
        "outcome": "PASS_C_E15_EXTEND_TO_E30" if extend else "STOP_C_E15",
        "scientific_decision": "EXTEND_ONCE_TO_E30" if extend else "STOP_C_ARM",
        "decision_rule_version": C_DECISION_RULE_VERSION,
        "primary_effect": "candidate_minus_clean_pp",
        "delta5_pp": delta5,
        "delta10_pp": delta10,
        "delta15_pp": delta15,
        "gate_band": band,
        "mechanism_healthy": mechanism_healthy,
        "stem_output_mean_rms_by_epoch": stem_output_rms,
        "stem_output_strictly_rising": stem_output_strictly_rising,
        "city_road_safety": city_road_safety(by_epoch[15]),
        "thresholds_pp": {
            "gray_min": C_GRAY_MIN_PP,
            "standard_min": C_STANDARD_MIN_PP,
            "strong_min": C_STRONG_MIN_PP,
            "trajectory_tolerance": C_TRAJECTORY_TOLERANCE_PP,
        },
        "interpretation": (
            "C is promoted only by C-minus-clean.  A's gain is separately visible "
            "as clean-minus-official and never counted as the C contribution."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    args = parser.parse_args(argv)
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.bootstrap_replicates <= 0:
        parser.error("bootstrap replicates must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    official_protocol = read_json(args.official_dir / "protocol.json")
    clean_protocol = read_json(args.clean_dir / "protocol.json")
    candidate_protocol = read_json(args.candidate_dir / "protocol.json")
    validate_protocol_triplet(official_protocol, clean_protocol, candidate_protocol)
    epochs = tuple(int(value) for value in candidate_protocol["evaluation_epochs"])
    results = [
        compare_epoch(
            args.official_dir,
            args.clean_dir,
            args.candidate_dir,
            epoch,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        for epoch in epochs
    ]
    verdict = decision(str(candidate_protocol["scope"]), results)
    payload = {
        "status": "PASS",
        "artifact_type": "whu_v4_c_sealed_baseline_comparison",
        "candidate_git_commit": candidate_protocol["git_commit"],
        "sealed_v4_a_git_commit": clean_protocol["git_commit"],
        "seed": candidate_protocol["seed"],
        "scope": candidate_protocol["scope"],
        "baseline_bindings": {
            "official_dir": str(args.official_dir),
            "clean_dir": str(args.clean_dir),
            "candidate_dir": str(args.candidate_dir),
        },
        "initialization_contract": {
            "clean_initial_state_sha256": clean_protocol[
                "initial_model_state_sha256"
            ],
            "candidate_in_process_clean_sha256": candidate_protocol[
                "clean_baseline_initial_state_sha256"
            ],
            "shared_parameter_count": candidate_protocol[
                "shared_initialization_audit"
            ]["shared_parameter_count"],
            "shared_parameters_per_key_bitwise_equal": True,
            "candidate_full_model_sha_not_compared": True,
            "step0_prediction_sha_equal": True,
            "stem_optimizer_membership_exactly_once": True,
        },
        "epochs": results,
        **verdict,
    }
    write_json_atomic(args.output_path, payload)
    print(
        f"PASS outcome={payload['outcome']} output={args.output_path}", flush=True
    )
    for record in results:
        print(
            f"epoch={record['epoch']} official={record['official_miou_percent']:.6f}% "
            f"clean={record['clean_miou_percent']:.6f}% "
            f"candidate={record['candidate_miou_percent']:.6f}% "
            f"C-minus-A={record['candidate_minus_clean_pp']:+.6f}pp "
            f"C-minus-official={record['candidate_minus_official_pp']:+.6f}pp",
            flush=True,
        )


if __name__ == "__main__":
    main()
