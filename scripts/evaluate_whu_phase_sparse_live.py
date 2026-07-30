"""Run the first-image Stage-B1b exact sparse-phase correctness smoke.

This foreground runner is correctness-only.  It executes the complete normal
K1 slide, builds one support-pruned K4 reference, then executes the frozen
middle subset and the first-image slice of the frozen B0 rescue policy.  It
does not evaluate efficacy, latency, deployment routing, or any scientific
Stage-B gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_whu_phase_rescue_decomposition import load_stage_b0  # noqa: E402
from scripts.diagnose_whu_spatial_errors import (  # noqa: E402
    build_test_loader,
    load_model,
)
from scripts.evaluate_whu_phase_closure import (  # noqa: E402
    _git_revision,
    atomic_write_json,
    file_sha256,
    load_stage_a,
    metric_summary,
)
from scripts.evaluate_whu_translation_consistency import (  # noqa: E402
    translate_tensor,
)
from scripts.phase_closure_common import (  # noqa: E402
    PHASE_NAMES,
    build_phase_closure_geometry,
    confusion_for_levels,
    summarize_closure,
    validate_levels,
)
from scripts.phase_sparse_live_common import (  # noqa: E402
    forward_selected_phase_key_crops,
    forward_selected_phase_crops,
    normalized_phase_logits,
)
from scripts.phase_sparse_replay_common import (  # noqa: E402
    MIDDLE_SAMPLE_NAME,
    array_sha256,
    compose_policy_logits,
    endpoint_levels,
    expected_phase_crop_ids,
    expected_phase_crop_keys,
    frozen_middle_levels,
    policy_level_map,
    validate_live_phase_on_routed_pixels,
    validate_observed_crop_keys,
)
from scripts.phase_utility_common import confusion_from_arrays  # noqa: E402
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)


ARTIFACT_TYPE = "whu_phase_sparse_live_b1b_smoke"
SCHEMA_VERSION = 2
NUM_CLASSES = 7
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-a-json", type=Path, required=True)
    parser.add_argument("--stage-b0-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (
        args.baseline_checkpoint,
        args.stage_a_json,
        args.stage_b0_json,
    ):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def _phase_key_subset(
    levels: np.ndarray, geometry: Mapping[str, Any], image_index: int
) -> tuple[tuple[int, str, int], ...]:
    return tuple(
        key
        for key in expected_phase_crop_keys(levels, geometry)
        if key[0] == image_index
    )


def _execution_summary(execution: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "crop_samples": int(execution["crop_samples"]),
        "batch_calls": int(execution["batch_calls"]),
        "batch_sizes": list(execution["batch_sizes"]),
        "mean_actual_batch_size": execution["mean_actual_batch_size"],
        "observed_crop_ids": list(execution["observed_crop_ids"]),
        "sum_logits_sha256": array_sha256(execution["sum_logits"]),
        "count_mat_sha256": array_sha256(execution["count_mat"]),
    }


def _global_phase_execution_summary(execution: Mapping[str, Any]) -> dict[str, Any]:
    phase_summaries: dict[str, Any] = {}
    for phase_name in PHASE_NAMES:
        phase = execution["phases"].get(phase_name)
        if phase is None:
            phase_summaries[phase_name] = {
                "crop_samples": 0,
                "observed_crop_ids": [],
            }
            continue
        phase_summaries[phase_name] = {
            "crop_samples": int(phase["crop_samples"]),
            "observed_crop_ids": list(phase["observed_crop_ids"]),
            "sum_logits_sha256": array_sha256(phase["sum_logits"]),
            "count_mat_sha256": array_sha256(phase["count_mat"]),
        }
    return {
        "packing_scope": "single-image global pool across phases",
        "canonical_order": "phase_order_then_ascending_local_crop_id",
        "selected_crop_samples": int(execution["selected_crop_samples"]),
        "model_forward_crop_samples": int(execution["model_forward_crop_samples"]),
        "padding_crop_samples": int(execution["padding_crop_samples"]),
        "batch_calls": int(execution["batch_calls"]),
        "real_batch_sizes": list(execution["real_batch_sizes"]),
        "model_batch_sizes": list(execution["model_batch_sizes"]),
        "mean_real_batch_size": execution["mean_real_batch_size"],
        "mean_model_batch_size": execution["mean_model_batch_size"],
        "padding_policy": execution["padding_policy"],
        "observed_phase_crop_keys": [
            [phase_name, int(crop_id)]
            for phase_name, crop_id in execution["observed_phase_crop_keys"]
        ],
        "processed_phase_crop_keys": [
            [phase_name, int(crop_id)]
            for phase_name, crop_id in execution["processed_phase_crop_keys"]
        ],
        "padding_phase_crop_keys": [
            [phase_name, int(crop_id)]
            for phase_name, crop_id in execution["padding_phase_crop_keys"]
        ],
        "phases": phase_summaries,
        "wall_seconds": float(execution["wall_seconds"]),
    }


def _add_model_sample_cost(
    execution_summary: dict[str, Any], baseline_crop_samples: int
) -> None:
    baseline = int(baseline_crop_samples)
    if baseline <= 0:
        raise ValueError("baseline crop sample count must be positive")
    selected = int(execution_summary["selected_crop_samples"])
    processed = int(execution_summary["model_forward_crop_samples"])
    execution_summary["cost_accounting"] = {
        "baseline_crop_samples": baseline,
        "selected_extra_crop_samples": selected,
        "processed_extra_crop_samples_including_padding": processed,
        "logical_unique_crop_cost_ratio": float((baseline + selected) / baseline),
        "physical_model_sample_cost_ratio": float((baseline + processed) / baseline),
        "padding_is_counted_as_model_compute": True,
    }


def _execute_shifted_phases(
    optical: torch.Tensor,
    sar: torch.Tensor,
    model: torch.nn.Module,
    levels: np.ndarray,
    geometry: Mapping[str, Any],
    image_index: int,
    *,
    device: torch.device,
    batch_size: int,
    label: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    crop_ids = expected_phase_crop_ids(levels, geometry, image_index)
    windows = geometry["windows_by_image"][image_index]
    phase_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for phase_name in PHASE_NAMES:
        selected = crop_ids[phase_name]
        if not selected:
            continue
        dy, dx = geometry["phase_shifts"][phase_name]
        phase_optical = translate_tensor(optical, dy, dx).to(device)
        phase_sar = translate_tensor(sar, dy, dx).to(device)
        phase_tensors[phase_name] = (phase_optical, phase_sar)
    execution = forward_selected_phase_key_crops(
        phase_tensors,
        model,
        windows,
        crop_ids,
        phase_order=PHASE_NAMES,
        n_output_channels=NUM_CLASSES,
        batch_size=batch_size,
        decoder_head_type="linear",
    )
    accumulations = execution["phases"]
    summary = _global_phase_execution_summary(execution)
    observed_keys = [
        (image_index, phase_name, int(crop_id))
        for phase_name, crop_id in execution["observed_phase_crop_keys"]
    ]
    expected_keys = _phase_key_subset(levels, geometry, image_index)
    key_validation = validate_observed_crop_keys(tuple(observed_keys), expected_keys)
    key_validation["expected_keys"] = [list(key) for key in expected_keys]
    key_validation["observed_keys"] = [list(key) for key in observed_keys]
    print(
        f"{label} selected={summary['selected_crop_samples']} "
        f"processed={summary['model_forward_crop_samples']} "
        f"padding={summary['padding_crop_samples']} "
        f"batches={summary['batch_calls']} PASS",
        flush=True,
    )
    del phase_tensors, execution
    torch.cuda.empty_cache()
    return accumulations, summary, key_validation


def _anchor_summary(
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    expected_image: Mapping[str, Any],
    class_names: Sequence[str],
) -> dict[str, Any]:
    values = np.ascontiguousarray(np.asarray(prediction, dtype=np.int64))
    expected_sha = str(expected_image["prediction_sha256"][name])
    actual_sha = array_sha256(values)
    actual_confusion = confusion_from_arrays(values, target, NUM_CLASSES)
    expected_confusion = np.asarray(expected_image["confusion"][name], dtype=np.int64)
    sha_equal = actual_sha == expected_sha
    confusion_equal = bool(np.array_equal(actual_confusion, expected_confusion))
    if not sha_equal or not confusion_equal:
        raise AssertionError(
            f"{name} does not reproduce the Stage-A first-image anchor: "
            f"prediction_sha_equal={sha_equal} confusion_equal={confusion_equal} "
            f"actual_sha={actual_sha} expected_sha={expected_sha}"
        )
    return {
        "name": name,
        "prediction_sha256": actual_sha,
        "expected_prediction_sha256": expected_sha,
        "prediction_sha256_equal": sha_equal,
        "confusion_equal": confusion_equal,
        "metrics": metric_summary(actual_confusion, class_names),
    }


def _expected_image_confusion(
    stage_a: Mapping[str, Any],
    geometry: Mapping[str, Any],
    levels: np.ndarray,
    image_index: int,
) -> np.ndarray:
    indices = tuple(int(value) for value in geometry["cells_by_image"][image_index])
    cells = [stage_a["cells"][index] for index in indices]
    local_levels = levels[np.asarray(indices, dtype=np.int64)]
    full_k1 = np.asarray(
        stage_a["images"][image_index]["confusion"]["k1"], dtype=np.int64
    )
    return confusion_for_levels(cells, local_levels, full_k1)


def _logit_tolerance_summary(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left = np.asarray(candidate)
    right = np.asarray(reference)
    if left.dtype != np.float32 or right.dtype != np.float32:
        raise TypeError("final logits must both be float32")
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("final logits must have matching CHW shapes")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("final logits must be finite")
    close = True
    maximum = 0.0
    for y0 in range(0, left.shape[1], 128):
        y1 = min(y0 + 128, left.shape[1])
        left_chunk = left[:, y0:y1]
        right_chunk = right[:, y0:y1]
        difference = np.abs(left_chunk - right_chunk)
        if difference.size:
            maximum = max(maximum, float(difference.max()))
        if not np.all(difference <= atol + rtol * np.abs(right_chunk)):
            close = False
            break
    if not close:
        raise AssertionError("sparse final logits exceed the frozen live tolerance")
    return {
        "close": True,
        "atol": float(atol),
        "rtol": float(rtol),
        "maximum_absolute_difference": maximum,
    }


def _policy_correctness(
    *,
    name: str,
    levels: np.ndarray,
    normal_logits: np.ndarray,
    dense_phases: Mapping[str, Mapping[str, Any]],
    sparse_phases: Mapping[str, Mapping[str, Any]],
    sparse_execution: Mapping[str, Any],
    geometry: Mapping[str, Any],
    stage_a: Mapping[str, Any],
    target: np.ndarray,
    image_index: int,
) -> dict[str, Any]:
    level_map = policy_level_map(levels, geometry, image_index)
    common_bounds = tuple(int(v) for v in geometry["common_bounds"][image_index])
    phase_checks: dict[str, Any] = {}
    selected_ids = expected_phase_crop_ids(levels, geometry, image_index)
    for phase_name in PHASE_NAMES:
        if not selected_ids[phase_name]:
            continue
        phase_checks[phase_name] = validate_live_phase_on_routed_pixels(
            sparse_phases[phase_name],
            dense_phases[phase_name],
            level_map,
            phase_name=phase_name,
            shift=tuple(geometry["phase_shifts"][phase_name]),
            common_bounds=common_bounds,
            atol=LOGIT_ATOL,
            rtol=LOGIT_RTOL,
        )

    dense_output = compose_policy_logits(
        normal_logits, dense_phases, levels, geometry, image_index
    )
    sparse_output = compose_policy_logits(
        normal_logits, sparse_phases, levels, geometry, image_index
    )
    final_logit_check = _logit_tolerance_summary(
        sparse_output["logits"],
        dense_output["logits"],
        atol=LOGIT_ATOL,
        rtol=LOGIT_RTOL,
    )
    prediction_equal = bool(
        np.array_equal(sparse_output["prediction"], dense_output["prediction"])
    )
    prediction_sha_equal = (
        sparse_output["prediction_sha256"] == dense_output["prediction_sha256"]
    )
    expected_confusion = _expected_image_confusion(
        stage_a, geometry, levels, image_index
    )
    dense_confusion = confusion_from_arrays(
        dense_output["prediction"], target, NUM_CLASSES
    )
    sparse_confusion = confusion_from_arrays(
        sparse_output["prediction"], target, NUM_CLASSES
    )
    dense_confusion_equal = bool(np.array_equal(dense_confusion, expected_confusion))
    sparse_confusion_equal = bool(np.array_equal(sparse_confusion, expected_confusion))
    if not all(
        (
            prediction_equal,
            prediction_sha_equal,
            dense_confusion_equal,
            sparse_confusion_equal,
        )
    ):
        raise AssertionError(f"{name} sparse/dense prediction or confusion differs")
    closure = summarize_closure(levels, geometry)["per_image"][image_index]
    return {
        "name": name,
        "role": "correctness-only frozen policy; not an efficacy result",
        "selected_counts": closure["selected_counts"],
        "expected_closure": closure,
        "execution": sparse_execution,
        "phase_checks": phase_checks,
        "final_logit_check": final_logit_check,
        "prediction_equal": prediction_equal,
        "prediction_sha256_equal": prediction_sha_equal,
        "prediction_sha256": sparse_output["prediction_sha256"],
        "dense_prediction_sha256": dense_output["prediction_sha256"],
        "dense_confusion_equal_expected": dense_confusion_equal,
        "sparse_confusion_equal_expected": sparse_confusion_equal,
        "metrics": metric_summary(sparse_confusion, stage_a["class_names"]),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the first-image B1b live smoke")

    stage_a = load_stage_a(args.stage_a_json)
    stage_b0 = load_stage_b0(args.stage_b0_json)
    stage_a_sha = file_sha256(args.stage_a_json)
    stage_b0_sha = file_sha256(args.stage_b0_json)
    checkpoint_sha = file_sha256(args.baseline_checkpoint)
    if stage_b0.get("source_stage_a", {}).get("sha256") != stage_a_sha:
        raise AssertionError("Stage-B0 source Stage-A SHA256 differs from input")
    if stage_a.get("baseline_checkpoint_sha256") != checkpoint_sha:
        raise AssertionError("checkpoint SHA256 differs from Stage-A")
    recorded_batch_size = stage_a.get("reproducibility", {}).get(
        "inference_batch_size"
    )
    if recorded_batch_size != args.inference_batch_size:
        raise ValueError(
            "live smoke batch size must equal the Stage-A anchor batch size "
            f"({recorded_batch_size})"
        )

    geometry = build_phase_closure_geometry(stage_a["images"], stage_a["cells"])
    image_index = 0
    expected_image = stage_a["images"][image_index]
    if expected_image["sample_name"] != MIDDLE_SAMPLE_NAME:
        raise AssertionError("first Stage-A image differs from the frozen middle sample")
    rescue_levels = validate_levels(
        stage_b0["exact_cost_a2_oracle"]["levels_by_cell"], geometry
    )
    middle_levels = frozen_middle_levels(geometry)
    k2_levels = endpoint_levels(geometry, 2)
    k4_levels = endpoint_levels(geometry, 4)

    b0_rescue_image = stage_b0["exact_cost_a2_oracle"]["closure"]["per_image"][0]
    computed_rescue_ids = expected_phase_crop_ids(rescue_levels, geometry, image_index)
    if any(
        list(computed_rescue_ids[name])
        != b0_rescue_image["phase_crop_ids"][name]
        for name in PHASE_NAMES
    ):
        raise AssertionError("first-image rescue closure differs from Stage-B0")
    b0_k4_image = stage_b0["minimal_common_support_endpoints"]["k4"][
        "closure"
    ]["per_image"][0]
    computed_k4_ids = expected_phase_crop_ids(k4_levels, geometry, image_index)
    if any(
        list(computed_k4_ids[name]) != b0_k4_image["phase_crop_ids"][name]
        for name in PHASE_NAMES
    ):
        raise AssertionError("support-pruned K4 closure differs from Stage-B0")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    model, cfg = load_model(args.baseline_checkpoint, args.seed)
    if tuple(cfg["window_size"]) != tuple(geometry["crop_size"]):
        raise AssertionError("model crop size differs from closure geometry")
    if len(cfg["labels"]) != NUM_CLASSES:
        raise AssertionError("model class count differs from Stage-A")
    loader, sample_names, full_test_length = build_test_loader(
        1, cfg["window_size"]
    )
    if full_test_length != stage_a["full_test_length"]:
        raise AssertionError("current WHU test length differs from Stage-A")
    if sample_names != [MIDDLE_SAMPLE_NAME]:
        raise AssertionError("current WHU first sample differs from Stage-A")
    (optical, sar, label_tensor) = next(iter(loader))
    target_batch = np.ascontiguousarray(
        label_tensor.numpy().astype(np.int64, copy=False)
    )
    if target_batch.ndim != 3 or target_batch.shape[0] != 1:
        raise AssertionError("WHU label must have shape [1,H,W]")
    target = target_batch[0]
    shape = tuple(int(value) for value in target.shape)
    if shape != tuple(geometry["image_shapes"][image_index]):
        raise AssertionError("current first-image shape differs from Stage-A")
    if tuple(optical.shape[-2:]) != shape or tuple(sar.shape[-2:]) != shape:
        raise AssertionError("RGB, SAR, and label shapes differ")

    model.to(device)
    model.eval()
    windows = geometry["windows_by_image"][image_index]
    all_crop_ids = tuple(range(len(windows)))
    print(
        f"sample={MIDDLE_SAMPLE_NAME} shape={shape} baseline_crops={len(windows)}",
        flush=True,
    )
    normal_execution = forward_selected_phase_crops(
        optical.to(device),
        sar.to(device),
        model,
        windows,
        all_crop_ids,
        n_output_channels=NUM_CLASSES,
        batch_size=args.inference_batch_size,
        decoder_head_type="linear",
        require_full_coverage=True,
    )
    if tuple(normal_execution["observed_crop_ids"]) != all_crop_ids:
        raise AssertionError("normal K1 did not execute the complete row-major slide")
    normal_logits = normalized_phase_logits(normal_execution)
    normal_execution_summary = _execution_summary(normal_execution)
    normal_prediction = np.ascontiguousarray(
        normal_logits.argmax(axis=0).astype(np.int64, copy=False)
    )
    k1_anchor = _anchor_summary(
        "k1", normal_prediction, target, expected_image, stage_a["class_names"]
    )
    del normal_execution
    torch.cuda.empty_cache()
    print("anchor=K1 PASS", flush=True)

    dense_phases, dense_phase_summaries, dense_key_validation = (
        _execute_shifted_phases(
            optical,
            sar,
            model,
            k4_levels,
            geometry,
            image_index,
            device=device,
            batch_size=args.inference_batch_size,
            label="support-pruned-k4-reference",
        )
    )
    _add_model_sample_cost(
        dense_phase_summaries, normal_execution_summary["crop_samples"]
    )
    k2_output = compose_policy_logits(
        normal_logits,
        {"x8": dense_phases["x8"]},
        k2_levels,
        geometry,
        image_index,
    )
    k2_anchor = _anchor_summary(
        "matched_k2",
        k2_output["prediction"],
        target,
        expected_image,
        stage_a["class_names"],
    )
    del k2_output
    k4_output = compose_policy_logits(
        normal_logits, dense_phases, k4_levels, geometry, image_index
    )
    k4_anchor = _anchor_summary(
        "k4",
        k4_output["prediction"],
        target,
        expected_image,
        stage_a["class_names"],
    )
    del k4_output
    print("anchors=K2,K4 PASS", flush=True)

    middle_phases, middle_summaries, middle_key_validation = (
        _execute_shifted_phases(
            optical,
            sar,
            model,
            middle_levels,
            geometry,
            image_index,
            device=device,
            batch_size=args.inference_batch_size,
            label="frozen-middle",
        )
    )
    _add_model_sample_cost(
        middle_summaries, normal_execution_summary["crop_samples"]
    )
    middle_result = _policy_correctness(
        name="frozen_middle",
        levels=middle_levels,
        normal_logits=normal_logits,
        dense_phases=dense_phases,
        sparse_phases=middle_phases,
        sparse_execution={
            "global_batching": middle_summaries,
            "key_validation": middle_key_validation,
        },
        geometry=geometry,
        stage_a=stage_a,
        target=target,
        image_index=image_index,
    )
    del middle_phases
    print("policy=frozen-middle PASS", flush=True)

    rescue_phases, rescue_summaries, rescue_key_validation = (
        _execute_shifted_phases(
            optical,
            sar,
            model,
            rescue_levels,
            geometry,
            image_index,
            device=device,
            batch_size=args.inference_batch_size,
            label="b0-rescue-first-image",
        )
    )
    _add_model_sample_cost(
        rescue_summaries, normal_execution_summary["crop_samples"]
    )
    rescue_result = _policy_correctness(
        name="b0_rescue_first_image",
        levels=rescue_levels,
        normal_logits=normal_logits,
        dense_phases=dense_phases,
        sparse_phases=rescue_phases,
        sparse_execution={
            "global_batching": rescue_summaries,
            "key_validation": rescue_key_validation,
        },
        geometry=geometry,
        stage_a=stage_a,
        target=target,
        image_index=image_index,
    )
    del rescue_phases
    expected_b0_miou = float(
        stage_b0["exact_cost_a2_oracle"]["stability"]["per_image"][0][
            "candidate_miou"
        ]
    )
    if abs(rescue_result["metrics"]["miou"] - expected_b0_miou) > 1e-12:
        raise AssertionError("first-image rescue mIoU differs from Stage-B0")
    rescue_result["stage_b0_candidate_miou_equal"] = True
    rescue_result["stage_b0_candidate_miou"] = expected_b0_miou
    print("policy=b0-rescue-first-image PASS", flush=True)

    payload = {
        "status": "PASS",
        "status_meaning": (
            "the first-image B1b implementation reproduced all frozen correctness "
            "anchors; no scientific or latency decision was evaluated"
        ),
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "first-test-image-correctness-only",
        "scientific_scope": (
            "B1b implementation smoke for exact sparse shifted-phase execution only"
        ),
        "scientific_decision_evaluated": False,
        "latency_decision_evaluated": False,
        "source_artifacts": {
            "stage_a": {
                "path": str(args.stage_a_json.resolve()),
                "sha256": stage_a_sha,
            },
            "stage_b0": {
                "path": str(args.stage_b0_json.resolve()),
                "sha256": stage_b0_sha,
            },
            "baseline_checkpoint": {
                "path": str(args.baseline_checkpoint.resolve()),
                "sha256": checkpoint_sha,
            },
        },
        "protocol": {
            "phase_order": list(PHASE_NAMES),
            "phase_shifts_dy_dx": {
                name: list(geometry["phase_shifts"][name]) for name in PHASE_NAMES
            },
            "crop_size_hw": list(geometry["crop_size"]),
            "stride_hw": list(geometry["stride"]),
            "logit_atol": LOGIT_ATOL,
            "logit_rtol": LOGIT_RTOL,
            "count_requirement": "integer exact on every routed pixel",
            "prediction_requirement": "array, SHA256, and confusion exact",
            "normal_k1": "complete sealed row-major slide",
            "dense_reference": "support-pruned K4 closure for the first image",
            "shifted_crop_batch_packing": (
                "one per-image global pool in phase order then local crop-ID order"
            ),
            "model_batch_shape": (
                "every non-empty shifted model call equals inference_batch_size"
            ),
            "final_batch_padding": (
                "cyclic duplicate from the final real batch; duplicate outputs are "
                "discarded and counted as physical model samples"
            ),
        },
        "sample": {
            "image_index": image_index,
            "sample_name": MIDDLE_SAMPLE_NAME,
            "shape_hw": list(shape),
        },
        "normal_k1_execution": normal_execution_summary,
        "support_pruned_k4_reference": {
            "global_batching": dense_phase_summaries,
            "key_validation": dense_key_validation,
        },
        "stage_a_first_image_anchors": {
            "k1": k1_anchor,
            "matched_k2": k2_anchor,
            "k4": k4_anchor,
        },
        "policies": {
            "frozen_middle": middle_result,
            "b0_rescue_first_image": rescue_result,
        },
        "correctness_decision": {
            "outcome": "PASS_B1B_FIRST_IMAGE_CORRECTNESS_SMOKE",
            "passed": True,
            "authorizes_full_b1": False,
            "meaning": (
                "implementation smoke only; a separate user-run full B1 protocol "
                "is required before any H2 conclusion"
            ),
        },
        "explicit_non_claims": [
            "no efficacy or method-selection result",
            "no latency, throughput, memory, or deployment result",
            "no small/thin structural gate",
            "no full-test B1 result",
            "no H2 or H3 confirmation",
        ],
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "live_common_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_sparse_live_common.py"
            ),
            "replay_common_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_sparse_replay_common.py"
            ),
            "seed": args.seed,
            "inference_batch_size": args.inference_batch_size,
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    atomic_write_json(args.output_path, payload)
    print(f"sparse_live_result={args.output_path.resolve()}", flush=True)
    print("sparse_live_status=PASS", flush=True)


if __name__ == "__main__":
    main()
