"""Compare the paired V4-C epoch-boundary restart arms at E30.

This comparator is intentionally pair-only: the clean ``mask-ignore`` arm and
the ``mask-ignore+optical-stem`` arm must have been restarted from their own
E15 checkpoints under one identical restart protocol.  The result is a cheap
screen and is explicitly not represented as equivalent to uninterrupted E30
training.  Bootstrap intervals are descriptive and never define the gate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ARTIFACT_TYPE = "whu_v4_c_epoch_boundary_restart"
COMPARISON_ARTIFACT_TYPE = "whu_v4_c_e30_restart_pair_comparison"
CONTINUATION_MODE = "epoch_boundary_paired_restart_screen"
CLEAN_VARIANT = "mask-ignore"
CANDIDATE_VARIANT = "mask-ignore+optical-stem"
SOURCE_EPOCH = 15
TARGET_EPOCH = 30
E30_INCREMENT_MIN_PP = 0.10
DECISION_RULE_VERSION = "v4_c_e30_epoch_boundary_restart_v1"
FORMAL_SCOPE = "formal-restart-screen"
CLASS_NAMES = ("farmland", "city", "village", "water", "forest", "road", "other")


# These fields determine the paired restart's data stream or evaluation scope.
# Variant-specific model fields and the two source checkpoint paths/hashes are
# deliberately absent: those must differ between clean and C.
PAIRED_DATA_PROTOCOL_FIELDS = (
    "scheduler_horizon_epochs",
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
    "mask_padding_ignore",
    "mask_fill",
    "aux_fill",
    "loss_change",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _required(payload: Mapping[str, Any], field: str, *, arm: str) -> Any:
    if field not in payload:
        raise RuntimeError(f"{arm} protocol lacks required field {field!r}")
    return payload[field]


def validate_protocol_pair(
    clean: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the sealed E15->E30 paired-restart protocol."""

    for arm, payload, variant in (
        ("clean", clean, CLEAN_VARIANT),
        ("candidate", candidate, CANDIDATE_VARIANT),
    ):
        expected = {
            "artifact_type": ARTIFACT_TYPE,
            "continuation_mode": CONTINUATION_MODE,
            "source_epoch": SOURCE_EPOCH,
            "first_continuation_epoch": SOURCE_EPOCH + 1,
            "target_epoch": TARGET_EPOCH,
            "stop_after_epoch": TARGET_EPOCH,
            "evaluation_epochs": [TARGET_EPOCH],
            "not_equivalent_to_uninterrupted": True,
            "persistent_worker_state_restored": False,
            "scope": FORMAL_SCOPE,
            "variant": variant,
        }
        mismatches = {
            field: (value, payload.get(field))
            for field, value in expected.items()
            if payload.get(field) != value
        }
        if mismatches:
            raise RuntimeError(f"{arm} continuation protocol differs: {mismatches}")

    paired_identity_fields = ("git_commit", "seed")
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field in (*paired_identity_fields, *PAIRED_DATA_PROTOCOL_FIELDS):
        clean_value = _required(clean, field, arm="clean")
        candidate_value = _required(candidate, field, arm="candidate")
        if clean_value != candidate_value:
            mismatches[field] = (clean_value, candidate_value)

    # A runner may additionally serialize its data options as one nested object.
    # If present it is part of the sealed pair and must be present and equal in
    # both arms, in addition to the explicit fields above.
    if "data_configuration" in clean or "data_configuration" in candidate:
        clean_data = _required(clean, "data_configuration", arm="clean")
        candidate_data = _required(candidate, "data_configuration", arm="candidate")
        if clean_data != candidate_data:
            mismatches["data_configuration"] = (clean_data, candidate_data)

    clean_rng = _required(clean, "restart_rng_fingerprints", arm="clean")
    candidate_rng = _required(
        candidate, "restart_rng_fingerprints", arm="candidate"
    )
    if not isinstance(clean_rng, Mapping) or not clean_rng:
        raise RuntimeError("restart RNG fingerprints must be a non-empty object")
    if clean_rng != candidate_rng:
        mismatches["restart_rng_fingerprints"] = (clean_rng, candidate_rng)
    candidate_lineage = _required(
        candidate, "candidate_clean_lineage", arm="candidate"
    )
    if not isinstance(candidate_lineage, Mapping):
        raise RuntimeError("candidate clean lineage is not an object")
    lineage_fields = {
        "clean_reference_dir": "source_dir_resolved",
        "clean_reference_protocol_sha256": "source_protocol_sha256",
        "clean_reference_git_commit": "source_git_commit",
        "clean_reference_checkpoint_sha256": "source_checkpoint_sha256",
        "clean_reference_evaluation_sha256": "source_evaluation_sha256",
    }
    lineage_mismatches = {
        candidate_field: (
            _required(clean, clean_field, arm="clean"),
            candidate_lineage.get(candidate_field),
        )
        for candidate_field, clean_field in lineage_fields.items()
        if _required(clean, clean_field, arm="clean")
        != candidate_lineage.get(candidate_field)
    }
    if lineage_mismatches:
        mismatches["candidate_clean_lineage"] = (
            "sealed clean continuation source",
            lineage_mismatches,
        )
    if mismatches:
        raise RuntimeError(f"paired continuation protocol differs: {mismatches}")

    return {
        "output_git_commit_equal": True,
        "seed_equal": True,
        "data_configuration_equal": True,
        "restart_rng_fingerprints_equal": True,
        "restart_rng_fingerprints": dict(clean_rng),
        "candidate_clean_lineage_equal": True,
    }


def validate_train_epoch_pair(
    clean: Mapping[str, Any],
    candidate: Mapping[str, Any],
    epoch: int,
    *,
    require_ten_trace_records: bool,
) -> dict[str, Any]:
    """Require full-epoch streams and the recorded first-ten trace to match."""

    if int(clean.get("epoch", epoch)) != epoch or int(
        candidate.get("epoch", epoch)
    ) != epoch:
        raise RuntimeError(f"train artifact epoch metadata differs at E{epoch}")
    comparisons = {
        "paired_data_sha256": (
            _required(clean, "paired_data_sha256", arm="clean train"),
            _required(candidate, "paired_data_sha256", arm="candidate train"),
        ),
        "raw_label_sha256": (
            _required(clean, "raw_label_sha256", arm="clean train"),
            _required(candidate, "raw_label_sha256", arm="candidate train"),
        ),
        "first_batch_trace": (
            _required(clean, "first_batch_trace", arm="clean train"),
            _required(candidate, "first_batch_trace", arm="candidate train"),
        ),
    }
    mismatches = {
        field: values for field, values in comparisons.items() if values[0] != values[1]
    }
    if mismatches:
        raise RuntimeError(f"E{epoch} paired restart training streams differ: {mismatches}")
    trace = comparisons["first_batch_trace"][0]
    if not isinstance(trace, list):
        raise RuntimeError(f"E{epoch} first-batch trace is not a list")
    if require_ten_trace_records and len(trace) != 10:
        raise RuntimeError(
            f"E{epoch} formal first-batch trace has {len(trace)} records, expected 10"
        )
    if not trace:
        raise RuntimeError(f"E{epoch} first-batch trace is empty")
    return {
        "epoch": epoch,
        "paired_data_sha256": comparisons["paired_data_sha256"][0],
        "raw_label_sha256": comparisons["raw_label_sha256"][0],
        "first_ten_trace_exactly_equal": True,
        "trace_record_count": len(trace),
    }


def per_image_confusions(evaluation: Mapping[str, Any]) -> np.ndarray:
    records = evaluation.get("per_image")
    if not isinstance(records, list) or not records:
        raise RuntimeError("evaluation lacks per-image confusion records")
    result = np.asarray([record["confusion"] for record in records], dtype=np.int64)
    if result.ndim != 3 or result.shape[1:] != (7, 7):
        raise RuntimeError("per-image confusion shape changed")
    return result


def pooled_miou(confusions: np.ndarray) -> float:
    matrix = np.asarray(confusions, dtype=np.int64)
    if matrix.ndim == 3:
        matrix = matrix.sum(axis=0)
    if matrix.shape != (7, 7):
        raise RuntimeError("confusion shape changed")
    denominator = matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix)
    valid = denominator > 0
    if not np.any(valid):
        raise RuntimeError("confusion has no valid class")
    iou = np.divide(
        np.diag(matrix),
        denominator,
        out=np.zeros(7, dtype=np.float64),
        where=valid,
    )
    return float(np.mean(iou[valid]) * 100.0)


def paired_bootstrap_ci(
    clean: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if clean.shape != candidate.shape:
        raise RuntimeError("paired per-image confusion arrays differ in shape")
    rng = np.random.default_rng(seed)
    image_count = clean.shape[0]
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = rng.integers(0, image_count, size=image_count)
        deltas[replicate] = pooled_miou(candidate[indices]) - pooled_miou(
            clean[indices]
        )
    low, high = np.percentile(deltas, (2.5, 97.5))
    return {
        "replicates": replicates,
        "seed": seed,
        "low_pp": float(low),
        "high_pp": float(high),
        "descriptive_positive_fraction": float(np.mean(deltas > 0.0)),
        "role": "descriptive_only_not_a_gate",
        "interpretation": (
            "exploratory paired-image stability under reused official test data"
        ),
    }


def _class_delta(
    clean: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, float | None]:
    if set(clean) != set(candidate):
        raise RuntimeError("clean and candidate class-IoU keys differ")
    return {
        name: (
            None
            if clean[name] is None or candidate[name] is None
            else float(candidate[name]) - float(clean[name])
        )
        for name in clean
    }


def mechanism_health(training: Mapping[str, Any]) -> dict[str, Any]:
    mechanism = training.get("optical_stem_mechanism", {})
    summary = mechanism.get("summary", {}) if isinstance(mechanism, Mapping) else {}
    healthy = bool(
        int(summary.get("observed_batches", 0)) > 0
        and summary.get("all_readouts_finite") is True
        and int(summary.get("stem_output_nonzero_batches", 0)) > 0
        and int(summary.get("projection_gradient_nonzero_batches", 0)) > 0
        and int(summary.get("upstream_gradient_nonzero_batches", 0)) > 0
    )
    return {
        "healthy_for_formal_gate": healthy,
        "summary": dict(summary) if isinstance(summary, Mapping) else {},
    }


def compare_e30(
    clean_evaluation: Mapping[str, Any],
    candidate_evaluation: Mapping[str, Any],
    candidate_train_e30: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if clean_evaluation.get("label_sha256") != candidate_evaluation.get(
        "label_sha256"
    ):
        raise RuntimeError("E30 evaluation label streams differ")
    clean_records = clean_evaluation.get("per_image")
    candidate_records = candidate_evaluation.get("per_image")
    if not isinstance(clean_records, list) or not isinstance(candidate_records, list):
        raise RuntimeError("E30 evaluation lacks per-image records")
    clean_names = [record.get("sample_name") for record in clean_records]
    candidate_names = [record.get("sample_name") for record in candidate_records]
    if clean_names != candidate_names:
        raise RuntimeError("E30 evaluation image order differs")

    clean_confusions = per_image_confusions(clean_evaluation)
    candidate_confusions = per_image_confusions(candidate_evaluation)
    if clean_confusions.shape != candidate_confusions.shape:
        raise RuntimeError("E30 paired per-image confusion count differs")
    clean_miou = pooled_miou(clean_confusions)
    candidate_miou = pooled_miou(candidate_confusions)
    per_image = []
    for name, clean_confusion, candidate_confusion in zip(
        clean_names, clean_confusions, candidate_confusions, strict=True
    ):
        clean_image_miou = pooled_miou(clean_confusion)
        candidate_image_miou = pooled_miou(candidate_confusion)
        per_image.append(
            {
                "sample_name": name,
                "clean_miou_percent": clean_image_miou,
                "candidate_miou_percent": candidate_image_miou,
                "candidate_minus_clean_pp": candidate_image_miou
                - clean_image_miou,
            }
        )
    deltas = np.asarray(
        [record["candidate_minus_clean_pp"] for record in per_image],
        dtype=np.float64,
    )
    clean_class = clean_evaluation.get("aggregate", {}).get(
        "class_iou_percent", {}
    )
    candidate_class = candidate_evaluation.get("aggregate", {}).get(
        "class_iou_percent", {}
    )
    class_delta = _class_delta(clean_class, candidate_class)
    mechanism = mechanism_health(candidate_train_e30)
    return {
        "epoch": TARGET_EPOCH,
        "evaluation_label_sha256": clean_evaluation["label_sha256"],
        "evaluation_image_order_exactly_equal": True,
        "clean_miou_percent": clean_miou,
        "candidate_miou_percent": candidate_miou,
        "candidate_minus_clean_pp": candidate_miou - clean_miou,
        "class_iou_delta_candidate_minus_clean_pp": class_delta,
        "per_image": per_image,
        "per_image_delta_summary_pp": {
            "count": int(deltas.size),
            "positive_count": int(np.count_nonzero(deltas > 0.0)),
            "mean": float(np.mean(deltas)),
            "median": float(np.median(deltas)),
            "min": float(np.min(deltas)),
            "max": float(np.max(deltas)),
        },
        "paired_bootstrap_candidate_minus_clean": paired_bootstrap_ci(
            clean_confusions,
            candidate_confusions,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "candidate_train_e30_mechanism_health": mechanism,
    }


def city_road_safety(result: Mapping[str, Any]) -> dict[str, Any]:
    deltas = result.get("class_iou_delta_candidate_minus_clean_pp", {})
    city = deltas.get("city") if isinstance(deltas, Mapping) else None
    road = deltas.get("road") if isinstance(deltas, Mapping) else None
    evidence_present = city is not None and road is not None
    joint_decline = bool(
        evidence_present and float(city) < 0.0 and float(road) < 0.0
    )
    return {
        "evidence_present": evidence_present,
        "city_delta_pp": None if city is None else float(city),
        "road_delta_pp": None if road is None else float(road),
        "joint_decline": joint_decline if evidence_present else None,
        "passes": bool(evidence_present and not joint_decline),
    }


def decision(scope: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered E30 incremental gate (never the bootstrap)."""

    if scope == "smoke":
        return {
            "outcome": "PASS_C_E30_RESTART_SMOKE_CONTRACT",
            "scientific_decision": "NONE",
            "bootstrap_used_for_gate": False,
        }
    delta = float(result["candidate_minus_clean_pp"])
    safety = city_road_safety(result)
    mechanism = result.get("candidate_train_e30_mechanism_health", {})
    mechanism_healthy = bool(mechanism.get("healthy_for_formal_gate"))
    pass_gate = bool(
        delta >= E30_INCREMENT_MIN_PP
        and safety["passes"]
        and mechanism_healthy
    )
    failed_conditions = []
    if delta < E30_INCREMENT_MIN_PP:
        failed_conditions.append("INCREMENT_BELOW_0.10_PP")
    if not safety["passes"]:
        failed_conditions.append(
            "CITY_ROAD_JOINT_DECLINE"
            if safety["joint_decline"]
            else "CITY_ROAD_EVIDENCE_MISSING"
        )
    if not mechanism_healthy:
        failed_conditions.append("CANDIDATE_E30_MECHANISM_UNHEALTHY")
    return {
        "outcome": (
            "PASS_C_E30_INCREMENT_GATE_RUN_OFFICIAL_RESTART"
            if pass_gate
            else "STOP_C_E30_INCREMENT_GATE"
        ),
        "scientific_decision": (
            "RUN_OFFICIAL_E15_TO_E30_RESTART"
            if pass_gate
            else "DO_NOT_RUN_OFFICIAL_RESTART"
        ),
        "decision_rule_version": DECISION_RULE_VERSION,
        "primary_effect": "candidate_minus_clean_pp",
        "primary_delta_pp": delta,
        "minimum_delta_pp": E30_INCREMENT_MIN_PP,
        "city_road_safety": safety,
        "candidate_train_e30_mechanism_healthy": mechanism_healthy,
        "failed_conditions": failed_conditions,
        "bootstrap_used_for_gate": False,
        "restart_scope": "cheap_paired_screen_not_uninterrupted_training",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    clean_protocol = read_json(args.clean_dir / "protocol.json")
    candidate_protocol = read_json(args.candidate_dir / "protocol.json")
    protocol_audit = validate_protocol_pair(clean_protocol, candidate_protocol)
    scope = str(clean_protocol["scope"])
    require_ten = scope != "smoke"
    epoch_pair_audits = []
    for epoch in range(SOURCE_EPOCH + 1, TARGET_EPOCH + 1):
        epoch_pair_audits.append(
            validate_train_epoch_pair(
                read_json(args.clean_dir / f"train_e{epoch}.json"),
                read_json(args.candidate_dir / f"train_e{epoch}.json"),
                epoch,
                require_ten_trace_records=require_ten,
            )
        )
    candidate_train_e30 = read_json(args.candidate_dir / "train_e30.json")
    result = compare_e30(
        read_json(args.clean_dir / "evaluation_e30.json"),
        read_json(args.candidate_dir / "evaluation_e30.json"),
        candidate_train_e30,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    verdict = decision(scope, result)
    payload = {
        "status": "PASS",
        "artifact_type": COMPARISON_ARTIFACT_TYPE,
        "continuation_mode": CONTINUATION_MODE,
        "source_epoch": SOURCE_EPOCH,
        "target_epoch": TARGET_EPOCH,
        "not_equivalent_to_uninterrupted": True,
        "git_commit": clean_protocol["git_commit"],
        "seed": clean_protocol["seed"],
        "scope": scope,
        "bindings": {
            "clean_dir": str(args.clean_dir),
            "candidate_dir": str(args.candidate_dir),
        },
        "protocol_pair_audit": protocol_audit,
        "train_e16_to_e30_pair_audits": epoch_pair_audits,
        "e30": result,
        **verdict,
    }
    write_json_atomic(args.output_path, payload)
    print(
        f"PASS outcome={payload['outcome']} "
        f"C-minus-clean={result['candidate_minus_clean_pp']:+.6f}pp "
        f"output={args.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
