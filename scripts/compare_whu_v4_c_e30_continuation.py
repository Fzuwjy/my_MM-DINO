"""Compare the official/clean/V4-C E15->E30 restart kill-test arms.

All three arms are mandatory.  Formal comparisons are observed at E20, E25,
and E30; E20/E25 describe trajectory shape but do not create a post-hoc gate.
The pre-registered E30 C-minus-clean gate remains primary.  Every result is an
epoch-boundary paired-restart screen, not an uninterrupted-training result and
not a confirmation test.  Paired-image bootstrap intervals are descriptive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ARTIFACT_TYPE = "whu_v4_three_arm_epoch_boundary_restart"
COMPARISON_ARTIFACT_TYPE = "whu_v4_c_e30_restart_three_arm_kill_test"
CONTINUATION_MODE = "epoch_boundary_paired_restart_kill_test"
OFFICIAL_VARIANT = "official"
CLEAN_VARIANT = "mask-ignore"
CANDIDATE_VARIANT = "mask-ignore+optical-stem"
SOURCE_EPOCH = 15
FIRST_CONTINUATION_EPOCH = 16
TARGET_EPOCH = 30
FORMAL_EVALUATION_EPOCHS = (20, 25, 30)
E30_INCREMENT_MIN_PP = 0.10
DECISION_RULE_VERSION = "v4_c_e30_three_arm_kill_test_v2"
FORMAL_SCOPE = "formal-restart-screen"
OPTICAL_STEM_LOCATION = "post-ACFM-L0-pre-FRN-single-injection"
CLASS_NAMES = ("farmland", "city", "village", "water", "forest", "road", "other")


# These fields determine the restart/evaluation pairing and must be identical.
# Mask-label fields are intentionally excluded because official keeps released
# zero padding while clean and C use ignore padding.
PAIRED_PROTOCOL_FIELDS = (
    "git_commit",
    "seed",
    "scheduler_horizon_epochs",
    "model_name",
    "dataset_name",
    "num_modalities",
    "backbone_type",
    "use_lora",
    "train_batch_size_per_gpu",
    "train_workers",
    "persistent_workers",
    "inference_batch_size",
    "max_train_batches",
    "max_test_images",
    "scope",
    "train_dataset_length",
    "full_test_length",
    "evaluated_test_length",
    "aux_fill",
    "loss_change",
)

SOURCE_SEAL_FIELDS = (
    "source_dir_resolved",
    "source_git_commit",
    "source_protocol_sha256",
    "source_checkpoint_sha256",
    "source_evaluation_sha256",
)

RNG_NON_CUDA_FIELDS = (
    "python_sha256",
    "numpy_sha256",
    "torch_cpu_sha256",
    "loader_generator_sha256",
)
RNG_FINGERPRINT_FIELDS = (*RNG_NON_CUDA_FIELDS, "torch_cuda_sha256", "combined_sha256")


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
        raise RuntimeError(f"{arm} lacks required field {field!r}")
    return payload[field]


def _nonempty_seals(payload: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    seals: dict[str, Any] = {}
    for field in SOURCE_SEAL_FIELDS:
        value = _required(payload, field, arm=f"{arm} protocol")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"{arm} protocol has an empty source seal {field!r}")
        seals[field] = value
    return seals


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_rng_fingerprints(value: Any, *, arm: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{arm} restart RNG fingerprints are not an object")
    if set(value) != set(RNG_FINGERPRINT_FIELDS):
        raise RuntimeError(f"{arm} restart RNG fingerprint fields differ")
    invalid = [field for field in RNG_NON_CUDA_FIELDS if not _is_sha256(value[field])]
    cuda = value["torch_cuda_sha256"]
    if not isinstance(cuda, list) or not cuda or any(not _is_sha256(item) for item in cuda):
        raise RuntimeError(f"{arm} CUDA RNG fingerprints must be a non-empty SHA256 list")
    if not _is_sha256(value["combined_sha256"]):
        invalid.append("combined_sha256")
    if invalid:
        raise RuntimeError(f"{arm} restart RNG fingerprints are invalid: {invalid}")
    components = {field: value[field] for field in RNG_FINGERPRINT_FIELDS if field != "combined_sha256"}
    expected_combined = hashlib.sha256(
        json.dumps(components, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if value["combined_sha256"] != expected_combined:
        raise RuntimeError(f"{arm} combined RNG fingerprint is internally inconsistent")
    return {
        **{field: value[field] for field in RNG_NON_CUDA_FIELDS},
        "torch_cuda_sha256": list(cuda),
        "combined_sha256": value["combined_sha256"],
    }


def _expected_epochs(scope: str) -> tuple[int, tuple[int, ...]]:
    if scope == "smoke":
        return FIRST_CONTINUATION_EPOCH, (FIRST_CONTINUATION_EPOCH,)
    if scope == FORMAL_SCOPE:
        return TARGET_EPOCH, FORMAL_EVALUATION_EPOCHS
    raise RuntimeError(f"unsupported continuation scope: {scope!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_at_exact_epochs(
    value: Any,
    expected_epochs: Sequence[int],
    *,
    arm: str,
    kind: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise RuntimeError(f"{arm} summary {kind} records are not a list of objects")
    actual_epochs = [int(item.get("epoch", -1)) for item in value]
    if actual_epochs != list(expected_epochs):
        raise RuntimeError(
            f"{arm} summary {kind} epochs differ: "
            f"{actual_epochs} != {list(expected_epochs)}"
        )
    return list(value)


def validate_completed_arm(
    directory: Path,
    protocol: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    arm: str,
    variant: str,
) -> dict[str, Any]:
    """Require a completed runner summary and re-hash every sealed artifact."""

    scope = str(protocol["scope"])
    stop_after_epoch, evaluation_epochs = _expected_epochs(scope)
    expected_outcome = (
        "PASS_RESTART_SMOKE_CONTRACT"
        if scope == "smoke"
        else {
            OFFICIAL_VARIANT: "COMPLETES_UNCONDITIONAL_OFFICIAL_RESTART_ARM",
            CLEAN_VARIANT: "COMPLETES_REQUIRED_CLEAN_RESTART_ARM",
            CANDIDATE_VARIANT: "COMPLETES_REQUIRED_C_RESTART_ARM",
        }[variant]
    )
    expected_summary = {
        "status": "PASS",
        "outcome": expected_outcome,
        "scientific_decision": "NONE",
        "artifact_type": f"{ARTIFACT_TYPE}_summary",
        "continuation_mode": CONTINUATION_MODE,
        "variant": variant,
        "scope": scope,
        "git_commit": protocol["git_commit"],
        "source_checkpoint_sha256": protocol["source_checkpoint_sha256"],
        "source_epoch": SOURCE_EPOCH,
        "stop_after_epoch": stop_after_epoch,
        "not_equivalent_to_uninterrupted": True,
    }
    mismatches = {
        field: (expected, summary.get(field))
        for field, expected in expected_summary.items()
        if summary.get(field) != expected
    }
    if mismatches:
        raise RuntimeError(f"{arm} summary completion contract differs: {mismatches}")

    train_epochs = list(range(FIRST_CONTINUATION_EPOCH, stop_after_epoch + 1))
    train_records = _records_at_exact_epochs(
        summary.get("training"), train_epochs, arm=arm, kind="training"
    )
    train_audits = []
    for epoch, record in zip(train_epochs, train_records, strict=True):
        filename = f"train_e{epoch}.json"
        if record.get("train_path") != filename:
            raise RuntimeError(f"{arm} summary E{epoch} train path changed")
        path = directory / filename
        actual_sha = file_sha256(path)
        if record.get("train_sha256") != actual_sha:
            raise RuntimeError(f"{arm} E{epoch} train SHA256 differs from summary")
        training = read_json(path)
        identity = {
            "epoch": epoch,
            "variant": variant,
            "continuation_mode": CONTINUATION_MODE,
        }
        identity_mismatches = {
            field: (expected, training.get(field))
            for field, expected in identity.items()
            if training.get(field) != expected
        }
        if identity_mismatches:
            raise RuntimeError(
                f"{arm} E{epoch} train identity differs: {identity_mismatches}"
            )
        if record.get("paired_data_sha256") != training.get("paired_data_sha256"):
            raise RuntimeError(f"{arm} E{epoch} summary paired-data SHA differs")
        train_audits.append({"epoch": epoch, "train_sha256": actual_sha})

    evaluation_records = _records_at_exact_epochs(
        summary.get("evaluations"), evaluation_epochs, arm=arm, kind="evaluation"
    )
    evaluation_audits = []
    for epoch, record in zip(evaluation_epochs, evaluation_records, strict=True):
        evaluation_filename = f"evaluation_e{epoch}.json"
        checkpoint_filename = f"checkpoint_e{epoch}.pth"
        if record.get("evaluation_path") != evaluation_filename:
            raise RuntimeError(f"{arm} summary E{epoch} evaluation path changed")
        if record.get("checkpoint_path") != checkpoint_filename:
            raise RuntimeError(f"{arm} summary E{epoch} checkpoint path changed")
        evaluation_path = directory / evaluation_filename
        checkpoint_path = directory / checkpoint_filename
        evaluation_sha = file_sha256(evaluation_path)
        checkpoint_sha = file_sha256(checkpoint_path)
        if record.get("evaluation_sha256") != evaluation_sha:
            raise RuntimeError(f"{arm} E{epoch} evaluation SHA256 differs from summary")
        if record.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError(f"{arm} E{epoch} checkpoint SHA256 differs from summary")
        if checkpoint_path.stat().st_size <= 0:
            raise RuntimeError(f"{arm} E{epoch} checkpoint is empty")
        evaluation = read_json(evaluation_path)
        identity = {
            "epoch": epoch,
            "variant": variant,
            "continuation_mode": CONTINUATION_MODE,
        }
        identity_mismatches = {
            field: (expected, evaluation.get(field))
            for field, expected in identity.items()
            if evaluation.get(field) != expected
        }
        if identity_mismatches:
            raise RuntimeError(
                f"{arm} E{epoch} evaluation identity differs: {identity_mismatches}"
            )
        evaluation_miou = evaluation.get("aggregate", {}).get("miou_percent")
        if (
            evaluation_miou is None
            or record.get("miou_percent") is None
            or not np.isclose(
                float(record["miou_percent"]),
                float(evaluation_miou),
                rtol=0.0,
                atol=1e-10,
            )
        ):
            raise RuntimeError(f"{arm} E{epoch} summary mIoU differs from evaluation")
        evaluation_audits.append(
            {
                "epoch": epoch,
                "evaluation_sha256": evaluation_sha,
                "checkpoint_sha256": checkpoint_sha,
            }
        )
    return {
        "summary_status": "PASS",
        "outcome": expected_outcome,
        "training": train_audits,
        "evaluations": evaluation_audits,
        "all_summary_artifact_sha256_verified": True,
    }


def validate_paired_clean_bindings(
    protocols: Mapping[str, Mapping[str, Any]],
    directories: Mapping[str, Path],
) -> dict[str, Any]:
    resolved = {arm: directory.resolve() for arm, directory in directories.items()}
    if len(set(resolved.values())) != 3:
        raise RuntimeError("official, clean, and candidate directories must be distinct")
    if protocols["clean"].get("paired_clean_dir") is not None:
        raise RuntimeError("clean protocol must not bind another paired clean directory")
    for arm in ("official", "candidate"):
        value = _required(protocols[arm], "paired_clean_dir", arm=f"{arm} protocol")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"{arm} protocol lacks a paired clean directory")
        if Path(value).resolve() != resolved["clean"]:
            raise RuntimeError(f"{arm} protocol is not bound to the supplied clean arm")
    return {
        "three_output_directories_distinct": True,
        "clean_has_no_parent_binding": True,
        "official_bound_to_supplied_clean": True,
        "candidate_bound_to_supplied_clean": True,
    }


def validate_protocol_triplet(
    official: Mapping[str, Any],
    clean: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the three sealed restart protocols and C->clean lineage."""

    payloads = {
        "official": official,
        "clean": clean,
        "candidate": candidate,
    }
    variants = {
        "official": OFFICIAL_VARIANT,
        "clean": CLEAN_VARIANT,
        "candidate": CANDIDATE_VARIANT,
    }
    scope = str(_required(official, "scope", arm="official protocol"))
    stop_after_epoch, evaluation_epochs = _expected_epochs(scope)
    common_expected: dict[str, Any] = {
        "artifact_type": ARTIFACT_TYPE,
        "continuation_mode": CONTINUATION_MODE,
        "source_epoch": SOURCE_EPOCH,
        "first_continuation_epoch": FIRST_CONTINUATION_EPOCH,
        "target_epoch": TARGET_EPOCH,
        "stop_after_epoch": stop_after_epoch,
        "evaluation_epochs": list(evaluation_epochs),
        "not_equivalent_to_uninterrupted": True,
        "persistent_worker_state_restored": False,
        "three_arm_policy": "official_clean_C_all_required",
        "official_arm_execution_policy": "unconditional",
        "scope": scope,
    }
    variant_expected = {
        "official": {
            "variant": OFFICIAL_VARIANT,
            "source_variant": OFFICIAL_VARIANT,
            "mask_padding_ignore": False,
            "mask_fill": 0,
            "aux_fill": 0,
            "use_optical_stem": False,
        },
        "clean": {
            "variant": CLEAN_VARIANT,
            "source_variant": CLEAN_VARIANT,
            "mask_padding_ignore": True,
            "mask_fill": 7,
            "aux_fill": 0,
            "use_optical_stem": False,
        },
        "candidate": {
            "variant": CANDIDATE_VARIANT,
            "source_variant": CANDIDATE_VARIANT,
            "mask_padding_ignore": True,
            "mask_fill": 7,
            "aux_fill": 0,
            "loss_change": "none",
            "use_optical_stem": True,
            "optical_stem_location": OPTICAL_STEM_LOCATION,
            "clean_baseline_variant": CLEAN_VARIANT,
        },
    }
    mismatches: dict[str, Any] = {}
    seals: dict[str, dict[str, Any]] = {}
    for arm, payload in payloads.items():
        expected = {**common_expected, **variant_expected[arm]}
        arm_mismatches = {
            field: (value, payload.get(field))
            for field, value in expected.items()
            if payload.get(field) != value
        }
        if arm_mismatches:
            mismatches[arm] = arm_mismatches
        seals[arm] = _nonempty_seals(payload, arm=arm)

    for field in PAIRED_PROTOCOL_FIELDS:
        values = {
            arm: _required(payload, field, arm=f"{arm} protocol")
            for arm, payload in payloads.items()
        }
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            mismatches[field] = values

    rng_values = {
        arm: _validated_rng_fingerprints(
            _required(payload, "restart_rng_fingerprints", arm=f"{arm} protocol"),
            arm=arm,
        )
        for arm, payload in payloads.items()
    }
    if rng_values["official"] != rng_values["clean"]:
        mismatches["official_clean_restart_rng_fingerprints"] = {
            "official": rng_values["official"],
            "clean": rng_values["clean"],
        }
    candidate_non_cuda_differences = {
        field: (rng_values["clean"][field], rng_values["candidate"][field])
        for field in RNG_NON_CUDA_FIELDS
        if rng_values["clean"][field] != rng_values["candidate"][field]
    }
    if candidate_non_cuda_differences:
        mismatches["candidate_clean_non_cuda_restart_rng"] = (
            candidate_non_cuda_differences
        )
    candidate_cuda_equal = (
        rng_values["candidate"]["torch_cuda_sha256"]
        == rng_values["clean"]["torch_cuda_sha256"]
    )
    candidate_combined_equal = (
        rng_values["candidate"]["combined_sha256"]
        == rng_values["clean"]["combined_sha256"]
    )

    lineage = _required(candidate, "candidate_clean_lineage", arm="candidate protocol")
    if not isinstance(lineage, Mapping):
        raise RuntimeError("candidate clean lineage is not an object")
    lineage_fields = {
        "clean_reference_dir": "source_dir_resolved",
        "clean_reference_protocol_sha256": "source_protocol_sha256",
        "clean_reference_git_commit": "source_git_commit",
        "clean_reference_checkpoint_sha256": "source_checkpoint_sha256",
        "clean_reference_evaluation_sha256": "source_evaluation_sha256",
    }
    lineage_mismatches = {
        candidate_field: (seals["clean"][clean_field], lineage.get(candidate_field))
        for candidate_field, clean_field in lineage_fields.items()
        if lineage.get(candidate_field) != seals["clean"][clean_field]
    }
    if lineage_mismatches:
        mismatches["candidate_clean_lineage"] = lineage_mismatches

    if mismatches:
        raise RuntimeError(f"three-arm continuation protocol differs: {mismatches}")

    return {
        "variants": variants,
        "scope": scope,
        "evaluation_epochs": list(evaluation_epochs),
        "output_git_commit_equal": True,
        "seed_equal": True,
        "paired_protocol_fields_equal": True,
        "official_clean_restart_rng_fingerprints_equal": True,
        "candidate_clean_non_cuda_restart_rng_equal": True,
        "candidate_clean_torch_cuda_rng_equal": candidate_cuda_equal,
        "candidate_clean_combined_rng_equal_descriptive_only": (
            candidate_combined_equal
        ),
        "combined_rng_hash_used_as_three_arm_gate": False,
        "candidate_variant_local_stochastic_trajectory": {
            "allowed_difference": "torch_cuda_sha256_only",
            "torch_cuda_equal_to_clean": candidate_cuda_equal,
            "observed": (
                "shared_cuda_rng_state"
                if candidate_cuda_equal
                else "variant_local_cuda_rng_state"
            ),
            "data_stream_pairing_still_required": True,
        },
        "source_seals_present": {arm: True for arm in payloads},
        "candidate_clean_lineage_equal": True,
        "intentional_protocol_difference": (
            "official trains released zero-padded raw labels; clean and C train "
            "ignore-padded raw labels"
        ),
    }


def _trace_projection(
    training: Mapping[str, Any],
    *,
    arm: str,
    expected_count: int,
    include_raw_label: bool,
) -> list[tuple[Any, ...]]:
    trace = _required(training, "first_batch_trace", arm=f"{arm} train")
    if not isinstance(trace, list):
        raise RuntimeError(f"{arm} first-batch trace is not a list")
    if len(trace) != expected_count:
        raise RuntimeError(
            f"{arm} first-batch trace has {len(trace)} records, expected {expected_count}"
        )
    projected: list[tuple[Any, ...]] = []
    fields = [
        "batch",
        "pair_sha256",
        "optical_sha256",
        "sar_sha256",
        "normalized_label_sha256",
    ]
    if include_raw_label:
        fields.append("raw_label_sha256")
    for index, item in enumerate(trace):
        if not isinstance(item, Mapping):
            raise RuntimeError(f"{arm} trace record {index} is not an object")
        values = tuple(_required(item, field, arm=f"{arm} trace record {index}") for field in fields)
        projected.append(values)
    return projected


def validate_train_epoch_triplet(
    official: Mapping[str, Any],
    clean: Mapping[str, Any],
    candidate: Mapping[str, Any],
    epoch: int,
    *,
    expected_trace_count: int,
) -> dict[str, Any]:
    """Validate only scientifically comparable training-stream fields."""

    payloads = {"official": official, "clean": clean, "candidate": candidate}
    for arm, payload in payloads.items():
        if int(payload.get("epoch", -1)) != epoch:
            raise RuntimeError(f"{arm} train artifact is not E{epoch}")

    paired_hashes = {
        arm: _required(payload, "paired_data_sha256", arm=f"{arm} train")
        for arm, payload in payloads.items()
    }
    if len(set(paired_hashes.values())) != 1:
        raise RuntimeError(f"E{epoch} normalized paired-data streams differ: {paired_hashes}")

    comparable_traces = {
        arm: _trace_projection(
            payload,
            arm=arm,
            expected_count=expected_trace_count,
            include_raw_label=False,
        )
        for arm, payload in payloads.items()
    }
    if not (
        comparable_traces["official"]
        == comparable_traces["clean"]
        == comparable_traces["candidate"]
    ):
        raise RuntimeError(f"E{epoch} optical/SAR/normalized-label traces differ")

    clean_full_trace = _trace_projection(
        clean,
        arm="clean",
        expected_count=expected_trace_count,
        include_raw_label=True,
    )
    candidate_full_trace = _trace_projection(
        candidate,
        arm="candidate",
        expected_count=expected_trace_count,
        include_raw_label=True,
    )
    if clean_full_trace != candidate_full_trace:
        raise RuntimeError(f"E{epoch} clean/C raw-label traces differ")
    clean_raw = _required(clean, "raw_label_sha256", arm="clean train")
    candidate_raw = _required(candidate, "raw_label_sha256", arm="candidate train")
    if clean_raw != candidate_raw:
        raise RuntimeError(f"E{epoch} clean/C full raw-label streams differ")
    official_raw = _required(official, "raw_label_sha256", arm="official train")
    if expected_trace_count == 10 and official_raw == clean_raw:
        raise RuntimeError(
            f"E{epoch} formal official/clean raw-label streams unexpectedly equal"
        )

    return {
        "epoch": epoch,
        "paired_data_sha256": paired_hashes["clean"],
        "paired_data_sha256_equal_three_arms": True,
        "optical_sar_normalized_label_trace_equal_three_arms": True,
        "trace_record_count": expected_trace_count,
        "clean_candidate_raw_label_sha256_equal": True,
        "official_raw_label_sha256": official_raw,
        "clean_raw_label_sha256": clean_raw,
        "candidate_raw_label_sha256": candidate_raw,
        "official_raw_label_differs_from_clean_by_design": official_raw != clean_raw,
        "official_raw_label_used_for_pairing_gate": False,
    }


def per_image_confusions(evaluation: Mapping[str, Any]) -> np.ndarray:
    records = evaluation.get("per_image")
    if not isinstance(records, list) or not records:
        raise RuntimeError("evaluation lacks per-image confusion records")
    result = np.asarray([record["confusion"] for record in records], dtype=np.int64)
    if result.ndim != 3 or result.shape[1:] != (7, 7):
        raise RuntimeError("per-image confusion shape changed")
    if np.any(result < 0) or np.any(result.sum(axis=(1, 2)) <= 0):
        raise RuntimeError("per-image confusion contains negative or empty counts")
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


def class_iou_percent(confusions: np.ndarray) -> dict[str, float | None]:
    matrix = np.asarray(confusions, dtype=np.int64)
    if matrix.ndim == 3:
        matrix = matrix.sum(axis=0)
    if matrix.shape != (7, 7) or np.any(matrix < 0):
        raise RuntimeError("confusion shape or counts changed")
    denominator = matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix)
    result: dict[str, float | None] = {}
    for index, name in enumerate(CLASS_NAMES):
        result[name] = (
            None
            if denominator[index] <= 0
            else float(matrix[index, index] / denominator[index] * 100.0)
        )
    return result


def _validate_aggregate_against_confusions(
    evaluation: Mapping[str, Any],
    confusions: np.ndarray,
    *,
    arm: str,
    epoch: int,
) -> dict[str, float | None]:
    aggregate = evaluation.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise RuntimeError(f"{arm} E{epoch} evaluation lacks aggregate metrics")
    pooled = confusions.sum(axis=0)
    serialized = np.asarray(aggregate.get("confusion"), dtype=np.int64)
    if serialized.shape != (7, 7) or not np.array_equal(serialized, pooled):
        raise RuntimeError(f"{arm} E{epoch} aggregate confusion differs from per-image sum")
    if aggregate.get("pixels") != int(pooled.sum()):
        raise RuntimeError(f"{arm} E{epoch} aggregate pixel count differs")
    expected_miou_percent = pooled_miou(pooled)
    for field, expected in (
        ("miou", expected_miou_percent / 100.0),
        ("miou_percent", expected_miou_percent),
    ):
        actual = aggregate.get(field)
        if actual is None or not np.isfinite(float(actual)) or not np.isclose(
            float(actual), expected, rtol=0.0, atol=1e-10
        ):
            raise RuntimeError(f"{arm} E{epoch} aggregate {field} differs from confusion")
    expected_class = class_iou_percent(pooled)
    actual_class = aggregate.get("class_iou_percent")
    if not isinstance(actual_class, Mapping) or set(actual_class) != set(CLASS_NAMES):
        raise RuntimeError(f"{arm} E{epoch} aggregate class-IoU keys differ")
    for name, expected in expected_class.items():
        actual = actual_class.get(name)
        if expected is None:
            if actual is not None:
                raise RuntimeError(f"{arm} E{epoch} {name} IoU should be null")
        elif actual is None or not np.isfinite(float(actual)) or not np.isclose(
            float(actual), expected, rtol=0.0, atol=1e-10
        ):
            raise RuntimeError(
                f"{arm} E{epoch} aggregate {name} IoU differs from confusion"
            )
    return expected_class


def paired_bootstrap_ci(
    reference: np.ndarray,
    comparison: np.ndarray,
    *,
    contrast: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if reference.shape != comparison.shape:
        raise RuntimeError(f"{contrast} paired confusion arrays differ in shape")
    rng = np.random.default_rng(seed)
    image_count = reference.shape[0]
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = rng.integers(0, image_count, size=image_count)
        deltas[replicate] = pooled_miou(comparison[indices]) - pooled_miou(
            reference[indices]
        )
    low, high = np.percentile(deltas, (2.5, 97.5))
    return {
        "contrast": contrast,
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
    reference: Mapping[str, Any], comparison: Mapping[str, Any]
) -> dict[str, float | None]:
    if set(reference) != set(comparison):
        raise RuntimeError("class-IoU keys differ between comparison arms")
    return {
        name: (
            None
            if reference[name] is None or comparison[name] is None
            else float(comparison[name]) - float(reference[name])
        )
        for name in reference
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
    smoke_healthy = bool(
        int(summary.get("observed_batches", 0)) == 1
        and summary.get("all_readouts_finite") is True
        and int(summary.get("stem_output_nonzero_batches", 0)) == 1
        and int(summary.get("projection_gradient_nonzero_batches", 0)) == 1
        and int(summary.get("upstream_gradient_nonzero_batches", 0)) == 1
    )
    return {
        "healthy_for_formal_gate": healthy,
        "healthy_for_smoke_contract": smoke_healthy,
        "summary": dict(summary) if isinstance(summary, Mapping) else {},
    }


def _validate_evaluation_triplet(
    official: Mapping[str, Any],
    clean: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    epoch: int,
    expected_image_count: int,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float | None],
    dict[str, float | None],
    dict[str, float | None],
]:
    evaluations = (official, clean, candidate)
    arms = ("official", "clean", "candidate")
    variants = (OFFICIAL_VARIANT, CLEAN_VARIANT, CANDIDATE_VARIANT)
    confusions = []
    class_ious = []
    image_orders: list[list[str]] = []
    for arm, variant, evaluation in zip(arms, variants, evaluations, strict=True):
        identity = {
            "epoch": epoch,
            "variant": variant,
            "continuation_mode": CONTINUATION_MODE,
            "evaluated_images": expected_image_count,
        }
        mismatches = {
            field: (expected, evaluation.get(field))
            for field, expected in identity.items()
            if evaluation.get(field) != expected
        }
        if mismatches:
            raise RuntimeError(f"{arm} E{epoch} evaluation identity differs: {mismatches}")
        records = evaluation.get("per_image")
        if not isinstance(records, list) or len(records) != expected_image_count:
            raise RuntimeError(
                f"{arm} E{epoch} evaluated image count differs from protocol"
            )
        names = [record.get("sample_name") for record in records]
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise RuntimeError(f"{arm} E{epoch} has an empty sample name")
        if len(set(names)) != len(names):
            raise RuntimeError(f"{arm} E{epoch} has duplicate sample names")
        indices = [record.get("index") for record in records]
        if indices != list(range(expected_image_count)):
            raise RuntimeError(f"{arm} E{epoch} per-image indices differ")
        arm_confusions = per_image_confusions(evaluation)
        arm_class = _validate_aggregate_against_confusions(
            evaluation, arm_confusions, arm=arm, epoch=epoch
        )
        image_orders.append(names)
        confusions.append(arm_confusions)
        class_ious.append(arm_class)

    labels = [evaluation.get("label_sha256") for evaluation in evaluations]
    if (
        any(not isinstance(value, str) or not value for value in labels)
        or len(set(labels)) != 1
    ):
        raise RuntimeError(f"E{epoch} evaluation label streams differ")
    if not (image_orders[0] == image_orders[1] == image_orders[2]):
        raise RuntimeError(f"E{epoch} evaluation image order differs")
    if not (confusions[0].shape == confusions[1].shape == confusions[2].shape):
        raise RuntimeError(f"E{epoch} paired per-image confusion counts differ")
    return (
        image_orders[0],
        confusions[0],
        confusions[1],
        confusions[2],
        class_ious[0],
        class_ious[1],
        class_ious[2],
    )


def compare_epoch(
    official_evaluation: Mapping[str, Any],
    clean_evaluation: Mapping[str, Any],
    candidate_evaluation: Mapping[str, Any],
    candidate_training: Mapping[str, Any],
    *,
    epoch: int,
    expected_image_count: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    (
        names,
        official_confusions,
        clean_confusions,
        candidate_confusions,
        official_class,
        clean_class,
        candidate_class,
    ) = (
        _validate_evaluation_triplet(
            official_evaluation,
            clean_evaluation,
            candidate_evaluation,
            epoch=epoch,
            expected_image_count=expected_image_count,
        )
    )
    official_miou = pooled_miou(official_confusions)
    clean_miou = pooled_miou(clean_confusions)
    candidate_miou = pooled_miou(candidate_confusions)
    class_deltas = {
        "clean_minus_official_pp": _class_delta(official_class, clean_class),
        "candidate_minus_clean_pp": _class_delta(clean_class, candidate_class),
        "candidate_minus_official_pp": _class_delta(official_class, candidate_class),
    }
    per_image = []
    for name, official_confusion, clean_confusion, candidate_confusion in zip(
        names,
        official_confusions,
        clean_confusions,
        candidate_confusions,
        strict=True,
    ):
        official_image = pooled_miou(official_confusion)
        clean_image = pooled_miou(clean_confusion)
        candidate_image = pooled_miou(candidate_confusion)
        per_image.append(
            {
                "sample_name": name,
                "official_miou_percent": official_image,
                "clean_miou_percent": clean_image,
                "candidate_miou_percent": candidate_image,
                "clean_minus_official_pp": clean_image - official_image,
                "candidate_minus_clean_pp": candidate_image - clean_image,
                "candidate_minus_official_pp": candidate_image - official_image,
            }
        )
    base_seed = bootstrap_seed + epoch * 10
    return {
        "epoch": epoch,
        "evaluation_label_sha256": official_evaluation["label_sha256"],
        "evaluation_image_order_exactly_equal": True,
        "official_miou_percent": official_miou,
        "clean_miou_percent": clean_miou,
        "candidate_miou_percent": candidate_miou,
        "clean_minus_official_pp": clean_miou - official_miou,
        "candidate_minus_clean_pp": candidate_miou - clean_miou,
        "candidate_minus_official_pp": candidate_miou - official_miou,
        "class_iou_deltas": class_deltas,
        # Compatibility alias used directly by the pre-registered safety gate.
        "class_iou_delta_candidate_minus_clean_pp": class_deltas[
            "candidate_minus_clean_pp"
        ],
        "per_image": per_image,
        "paired_image_bootstrap": {
            "clean_minus_official": paired_bootstrap_ci(
                official_confusions,
                clean_confusions,
                contrast="clean_minus_official",
                replicates=bootstrap_replicates,
                seed=base_seed + 1,
            ),
            "candidate_minus_clean": paired_bootstrap_ci(
                clean_confusions,
                candidate_confusions,
                contrast="candidate_minus_clean",
                replicates=bootstrap_replicates,
                seed=base_seed + 2,
            ),
            "candidate_minus_official": paired_bootstrap_ci(
                official_confusions,
                candidate_confusions,
                contrast="candidate_minus_official",
                replicates=bootstrap_replicates,
                seed=base_seed + 3,
            ),
        },
        "candidate_mechanism_health": mechanism_health(candidate_training),
    }


def _trajectory(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    epochs = np.asarray(FORMAL_EVALUATION_EPOCHS, dtype=np.float64)
    adjacent = np.diff(array)
    if np.all(adjacent <= 0.0):
        shape = "nonincreasing"
    elif np.all(adjacent >= 0.0):
        shape = "nondecreasing"
    else:
        shape = "mixed"
    slope = float(np.polyfit(epochs, array, 1)[0])
    return {
        "epochs": list(FORMAL_EVALUATION_EPOCHS),
        "values_pp": [float(value) for value in array],
        "adjacent_changes_pp": [float(value) for value in adjacent],
        "linear_slope_pp_per_epoch": slope,
        "shape": shape,
        "role": "descriptive_only_no_post_hoc_gate",
    }


def describe_trajectories(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_epoch = {int(record["epoch"]): record for record in results}
    missing = [epoch for epoch in FORMAL_EVALUATION_EPOCHS if epoch not in by_epoch]
    if missing:
        raise RuntimeError(f"formal kill test lacks trajectory epochs: {missing}")
    return {
        "clean_minus_official": _trajectory(
            [by_epoch[epoch]["clean_minus_official_pp"] for epoch in FORMAL_EVALUATION_EPOCHS]
        ),
        "candidate_minus_clean": _trajectory(
            [by_epoch[epoch]["candidate_minus_clean_pp"] for epoch in FORMAL_EVALUATION_EPOCHS]
        ),
        "candidate_minus_official": _trajectory(
            [by_epoch[epoch]["candidate_minus_official_pp"] for epoch in FORMAL_EVALUATION_EPOCHS]
        ),
        "e20_e25_used_for_gate": False,
        "interpretation": (
            "E20/E25 describe persistence/decay shape only; they do not create a "
            "new threshold.  The A clean-minus-official trajectory is reported "
            "without a post-hoc A gate."
        ),
    }


def city_road_safety(result: Mapping[str, Any]) -> dict[str, Any]:
    deltas = result.get("class_iou_delta_candidate_minus_clean_pp", {})
    city = deltas.get("city") if isinstance(deltas, Mapping) else None
    road = deltas.get("road") if isinstance(deltas, Mapping) else None
    evidence_present = city is not None and road is not None
    joint_decline = bool(evidence_present and float(city) < 0.0 and float(road) < 0.0)
    return {
        "evidence_present": evidence_present,
        "city_delta_pp": None if city is None else float(city),
        "road_delta_pp": None if road is None else float(road),
        "joint_decline": joint_decline if evidence_present else None,
        "passes": bool(evidence_present and not joint_decline),
    }


def decision(scope: str, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply only the frozen E30 kill-test gates; E20/E25 remain descriptive."""

    if scope == "smoke":
        if len(results) != 1 or int(results[0].get("epoch", -1)) != FIRST_CONTINUATION_EPOCH:
            raise RuntimeError("smoke comparison lacks its sole E16 observation")
        mechanism = results[0].get("candidate_mechanism_health", {})
        if not mechanism.get("healthy_for_smoke_contract"):
            raise RuntimeError("candidate smoke mechanism is unhealthy")
        return {
            "outcome": "PASS_C_E30_THREE_ARM_RESTART_SMOKE_CONTRACT",
            "scientific_decision": "NONE",
            "may_authorize_separate_confirmation": False,
            "bootstrap_used_for_gate": False,
            "e20_e25_used_for_gate": False,
            "kill_test_pass_is_confirmation": False,
            "not_equivalent_to_uninterrupted": True,
        }
    by_epoch = {int(record["epoch"]): record for record in results}
    if TARGET_EPOCH not in by_epoch:
        raise RuntimeError("formal kill test lacks E30")
    e30 = by_epoch[TARGET_EPOCH]
    delta = float(e30["candidate_minus_clean_pp"])
    above_official = float(e30["candidate_minus_official_pp"]) > 0.0
    safety = city_road_safety(e30)
    mechanism = e30.get("candidate_mechanism_health", {})
    mechanism_healthy = bool(mechanism.get("healthy_for_formal_gate"))
    survived = bool(
        delta >= E30_INCREMENT_MIN_PP
        and safety["passes"]
        and mechanism_healthy
        and above_official
    )
    failed_conditions: list[str] = []
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
    if not above_official:
        failed_conditions.append("CANDIDATE_NOT_ABOVE_SAME_EPOCH_OFFICIAL")
    return {
        "outcome": (
            "SURVIVE_C_E30_KILL_TEST_NOT_CONFIRMED"
            if survived
            else "STOP_C_E30_KILL_TEST"
        ),
        "scientific_decision": (
            "ELIGIBLE_FOR_SEPARATE_CONFIRMATION_NOT_CONFIRMED"
            if survived
            else "STOP_C_ROUTE_UNDER_PREREGISTERED_KILL_TEST"
        ),
        "may_authorize_separate_confirmation": survived,
        "decision_rule_version": DECISION_RULE_VERSION,
        "primary_effect": "candidate_minus_clean_pp_at_e30",
        "primary_delta_pp": delta,
        "minimum_delta_pp": E30_INCREMENT_MIN_PP,
        "candidate_minus_official_pp_at_e30": float(
            e30["candidate_minus_official_pp"]
        ),
        "candidate_above_same_epoch_official": above_official,
        "city_road_safety": safety,
        "candidate_e30_mechanism_healthy": mechanism_healthy,
        "failed_conditions": failed_conditions,
        "bootstrap_used_for_gate": False,
        "e20_e25_used_for_gate": False,
        "kill_test_pass_is_confirmation": False,
        "not_equivalent_to_uninterrupted": True,
        "restart_scope": "epoch_boundary_restart_kill_test_not_uninterrupted",
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
    directories = {
        "official": args.official_dir,
        "clean": args.clean_dir,
        "candidate": args.candidate_dir,
    }
    protocols = {
        arm: read_json(directory / "protocol.json")
        for arm, directory in directories.items()
    }
    protocol_audit = validate_protocol_triplet(
        protocols["official"], protocols["clean"], protocols["candidate"]
    )
    paired_clean_binding_audit = validate_paired_clean_bindings(
        protocols, directories
    )
    summaries = {
        arm: read_json(directory / "summary.json")
        for arm, directory in directories.items()
    }
    completion_audits = {
        arm: validate_completed_arm(
            directories[arm],
            protocols[arm],
            summaries[arm],
            arm=arm,
            variant={
                "official": OFFICIAL_VARIANT,
                "clean": CLEAN_VARIANT,
                "candidate": CANDIDATE_VARIANT,
            }[arm],
        )
        for arm in directories
    }
    scope = str(protocols["official"]["scope"])
    stop_after_epoch, evaluation_epochs = _expected_epochs(scope)
    trace_count = 1 if scope == "smoke" else 10
    train_audits = []
    for epoch in range(FIRST_CONTINUATION_EPOCH, stop_after_epoch + 1):
        train_audits.append(
            validate_train_epoch_triplet(
                read_json(args.official_dir / f"train_e{epoch}.json"),
                read_json(args.clean_dir / f"train_e{epoch}.json"),
                read_json(args.candidate_dir / f"train_e{epoch}.json"),
                epoch,
                expected_trace_count=trace_count,
            )
        )
    results = []
    for epoch in evaluation_epochs:
        results.append(
            compare_epoch(
                read_json(args.official_dir / f"evaluation_e{epoch}.json"),
                read_json(args.clean_dir / f"evaluation_e{epoch}.json"),
                read_json(args.candidate_dir / f"evaluation_e{epoch}.json"),
                read_json(args.candidate_dir / f"train_e{epoch}.json"),
                epoch=epoch,
                expected_image_count=int(protocols["official"]["evaluated_test_length"]),
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
    trajectories = None if scope == "smoke" else describe_trajectories(results)
    verdict = decision(scope, results)
    payload = {
        "status": "PASS",
        "artifact_type": COMPARISON_ARTIFACT_TYPE,
        "continuation_mode": CONTINUATION_MODE,
        "source_epoch": SOURCE_EPOCH,
        "target_epoch": TARGET_EPOCH,
        "evaluation_epochs": list(evaluation_epochs),
        "not_equivalent_to_uninterrupted": True,
        "kill_test_not_confirmation": True,
        "git_commit": protocols["official"]["git_commit"],
        "seed": protocols["official"]["seed"],
        "scope": scope,
        "bindings": {
            arm + "_dir": str(directory) for arm, directory in directories.items()
        },
        "protocol_triplet_audit": protocol_audit,
        "paired_clean_binding_audit": paired_clean_binding_audit,
        "completed_arm_audits": completion_audits,
        "train_e16_to_stop_triplet_audits": train_audits,
        "epochs": results,
        "trajectories": trajectories,
        **verdict,
    }
    write_json_atomic(args.output_path, payload)
    last = results[-1]
    print(
        f"PASS outcome={payload['outcome']} "
        f"E{last['epoch']} clean-official={last['clean_minus_official_pp']:+.6f}pp "
        f"C-clean={last['candidate_minus_clean_pp']:+.6f}pp "
        f"C-official={last['candidate_minus_official_pp']:+.6f}pp "
        f"output={args.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
