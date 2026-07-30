"""Run the formal 20-image Stage-B1 live output and structure evaluation.

This runner reuses the exact fixed-batch sparse operator that passed the
first-image B1b smoke.  It evaluates only the frozen Stage-B0 rescue levels:
normal K1 plus the exact per-image phase-crop closure.  The run closes live
confusion, logical and physical crop cost, per-image stability, and aggregate
small/thin errors.  Latency is deliberately left to a separate benchmark
using the same execution primitive and is never inferred from this pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
from scripts.evaluate_whu_phase_sparse_live import (  # noqa: E402
    _add_model_sample_cost,
    _anchor_summary,
    _execute_shifted_phases,
    _execution_summary,
    _expected_image_confusion,
)
from scripts.phase_closure_common import (  # noqa: E402
    PHASE_NAMES,
    build_phase_closure_geometry,
    summarize_closure,
    validate_levels,
)
from scripts.phase_sparse_live_common import (  # noqa: E402
    forward_selected_phase_crops,
    normalized_phase_logits,
)
from scripts.phase_sparse_replay_common import (  # noqa: E402
    analytic_count_map,
    array_sha256,
    compose_policy_logits,
    expected_phase_crop_ids,
)
from scripts.phase_utility_common import (  # noqa: E402
    PhaseUtilityGateThresholds,
    confusion_from_arrays,
    evaluate_phase_utility_gate,
)
from scripts.spatial_diagnostics_common import (  # noqa: E402
    component_geometry_masks,
)
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)


ARTIFACT_TYPE = "whu_phase_sparse_live_b1_full"
SCHEMA_VERSION = 1
NUM_CLASSES = 7
SMALL_REGION = "component_area_le_256px2"
THIN_REGION = "component_thickness_le_4px"
REGION_NAMES = ("all", "small", "thin")
METRIC_TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-a-json", type=Path, required=True)
    parser.add_argument("--stage-b0-json", type=Path, required=True)
    parser.add_argument("--b1b-smoke-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (
        args.baseline_checkpoint,
        args.stage_a_json,
        args.stage_b0_json,
        args.b1b_smoke_json,
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


def _load_b1b_smoke(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("B1b smoke artifact must contain a JSON object")
    if payload.get("artifact_type") != "whu_phase_sparse_live_b1b_smoke":
        raise ValueError("input is not a B1b sparse-live smoke artifact")
    decision = payload.get("correctness_decision", {})
    if (
        payload.get("status") != "PASS"
        or payload.get("schema_version") != 2
        or decision.get("outcome") != "PASS_B1B_FIRST_IMAGE_CORRECTNESS_SMOKE"
        or decision.get("passed") is not True
    ):
        raise ValueError("B1b correctness smoke did not pass")
    return payload


def _region_error_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    common_bounds: Sequence[int],
    region_masks: Mapping[str, np.ndarray],
) -> dict[str, dict[str, int]]:
    candidate = np.asarray(prediction)
    labels = np.asarray(target)
    if candidate.shape != labels.shape or candidate.ndim != 2:
        raise ValueError("prediction and target must be matching 2-D arrays")
    y0, y1, x0, x1 = (int(value) for value in common_bounds)
    if y0 < 0 or x0 < 0 or y0 >= y1 or x0 >= x1:
        raise ValueError("common support bounds are invalid")
    if y1 > labels.shape[0] or x1 > labels.shape[1]:
        raise ValueError("common support lies outside the image")
    valid = (labels >= 0) & (labels < NUM_CLASSES)
    common = np.zeros(labels.shape, dtype=bool)
    common[y0:y1, x0:x1] = True
    base = valid & common
    masks = {
        "all": base,
        "small": base & np.asarray(region_masks[SMALL_REGION], dtype=bool),
        "thin": base & np.asarray(region_masks[THIN_REGION], dtype=bool),
    }
    result: dict[str, dict[str, int]] = {}
    for name in REGION_NAMES:
        mask = masks[name]
        if mask.shape != labels.shape:
            raise ValueError(f"{name} region mask shape differs from target")
        result[name] = {
            "pixels": int(np.count_nonzero(mask)),
            "errors": int(np.count_nonzero((candidate != labels) & mask)),
        }
    return result


def _summarize_region_totals(
    totals: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name in REGION_NAMES:
        pixels = int(totals[name]["pixels"])
        errors = int(totals[name]["errors"])
        if pixels <= 0 or errors < 0 or errors > pixels:
            raise ValueError(f"invalid aggregate region counts for {name}")
        result[name] = {
            "pixels": pixels,
            "errors": errors,
            "error_rate": float(errors / pixels),
        }
    return result


def _validate_live_count_maps(
    phase_accumulations: Mapping[str, Mapping[str, Any]],
    selected_ids: Mapping[str, Sequence[int]],
    geometry: Mapping[str, Any],
    image_index: int,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for phase_name in PHASE_NAMES:
        selected = tuple(int(value) for value in selected_ids[phase_name])
        accumulation = phase_accumulations.get(phase_name)
        if not selected:
            if accumulation is not None:
                raise AssertionError(f"unexpected live accumulation for {phase_name}")
            checks[phase_name] = {
                "selected_crop_samples": 0,
                "count_mat_equal_analytic": True,
            }
            continue
        if accumulation is None:
            raise AssertionError(f"missing live accumulation for {phase_name}")
        expected = analytic_count_map(geometry, image_index, selected)
        actual = np.asarray(accumulation["count_mat"])
        equal = bool(np.array_equal(actual, expected))
        if not equal:
            raise AssertionError(f"{phase_name} live count differs from analytic coverage")
        checks[phase_name] = {
            "selected_crop_samples": len(selected),
            "count_mat_equal_analytic": equal,
            "count_mat_sha256": array_sha256(actual),
        }
    return checks


def _physical_cost_summary(
    per_image: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = sum(int(item["baseline_crop_samples"]) for item in per_image)
    selected = sum(int(item["selected_extra_crop_samples"]) for item in per_image)
    processed = sum(int(item["processed_extra_crop_samples"]) for item in per_image)
    padding = sum(int(item["padding_crop_samples"]) for item in per_image)
    if baseline <= 0 or processed != selected + padding:
        raise ValueError("invalid full-run crop accounting")
    logical_per_image = [
        float(
            (int(item["baseline_crop_samples"]) + int(item["selected_extra_crop_samples"]))
            / int(item["baseline_crop_samples"])
        )
        for item in per_image
    ]
    physical_per_image = [
        float(
            (int(item["baseline_crop_samples"])
             + int(item["processed_extra_crop_samples"]))
            / int(item["baseline_crop_samples"])
        )
        for item in per_image
    ]
    return {
        "baseline_crop_samples": baseline,
        "selected_extra_crop_samples": selected,
        "processed_extra_crop_samples_including_padding": processed,
        "padding_crop_samples": padding,
        "logical_unique_crop_cost_ratio": float((baseline + selected) / baseline),
        "physical_model_sample_cost_ratio": float((baseline + processed) / baseline),
        "per_image_logical_cost_ratios": logical_per_image,
        "per_image_physical_cost_ratios": physical_per_image,
        "maximum_per_image_logical_cost_ratio": max(logical_per_image),
        "maximum_per_image_physical_cost_ratio": max(physical_per_image),
        "padding_is_counted_as_model_compute": True,
    }


def _source_artifact_matches(
    smoke: Mapping[str, Any],
    name: str,
    expected_sha256: str,
) -> bool:
    source = smoke.get("source_artifacts", {}).get(name, {})
    return source.get("sha256") == expected_sha256


def _apply_registered_k2_gain_threshold(
    gate: dict[str, Any], minimum_delta: float
) -> dict[str, Any]:
    """Apply the preregistered inclusive K2 gain comparator to a gate result."""

    minimum = float(minimum_delta)
    delta = float(gate["observed"]["miou_delta_over_k2"])
    passed = bool(delta + METRIC_TOLERANCE >= minimum)
    gate["checks"].pop("outperforms_uniform_k2", None)
    gate["checks"]["gain_over_matched_k2_at_least_0_05pp"] = passed
    gate["thresholds"]["min_miou_delta_over_k2"] = minimum
    gate["thresholds"]["miou_delta_over_k2_comparator"] = ">="
    gate["passed"] = all(gate["checks"].values())
    gate["outcome"] = "GO" if gate["passed"] else "NO_GO"
    return gate


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal full B1 live run")

    stage_a = load_stage_a(args.stage_a_json)
    stage_b0 = load_stage_b0(args.stage_b0_json)
    b1b_smoke = _load_b1b_smoke(args.b1b_smoke_json)
    stage_a_sha = file_sha256(args.stage_a_json)
    stage_b0_sha = file_sha256(args.stage_b0_json)
    b1b_smoke_sha = file_sha256(args.b1b_smoke_json)
    checkpoint_sha = file_sha256(args.baseline_checkpoint)
    if stage_b0.get("source_stage_a", {}).get("sha256") != stage_a_sha:
        raise AssertionError("Stage-B0 source Stage-A SHA256 differs from input")
    if stage_a.get("baseline_checkpoint_sha256") != checkpoint_sha:
        raise AssertionError("checkpoint SHA256 differs from Stage-A")
    if not _source_artifact_matches(b1b_smoke, "stage_a", stage_a_sha):
        raise AssertionError("B1b smoke used a different Stage-A artifact")
    if not _source_artifact_matches(b1b_smoke, "stage_b0", stage_b0_sha):
        raise AssertionError("B1b smoke used a different Stage-B0 artifact")
    if not _source_artifact_matches(
        b1b_smoke, "baseline_checkpoint", checkpoint_sha
    ):
        raise AssertionError("B1b smoke used a different checkpoint")
    current_live_common_sha = file_sha256(
        REPO_ROOT / "scripts" / "phase_sparse_live_common.py"
    )
    if (
        b1b_smoke.get("reproducibility", {}).get("live_common_sha256")
        != current_live_common_sha
    ):
        raise AssertionError("B1b smoke used a different sparse execution primitive")
    smoke_batch_size = b1b_smoke.get("reproducibility", {}).get(
        "inference_batch_size"
    )
    if smoke_batch_size != args.inference_batch_size:
        raise ValueError("full B1 batch size differs from the passed B1b smoke")
    recorded_batch_size = stage_a.get("reproducibility", {}).get(
        "inference_batch_size"
    )
    if recorded_batch_size != args.inference_batch_size:
        raise ValueError("full B1 batch size differs from the Stage-A anchor")

    geometry = build_phase_closure_geometry(stage_a["images"], stage_a["cells"])
    rescue_levels = validate_levels(
        stage_b0["exact_cost_a2_oracle"]["levels_by_cell"], geometry
    )
    closure = summarize_closure(rescue_levels, geometry)
    b0_closure = stage_b0["exact_cost_a2_oracle"]["closure"]
    stability_records = stage_b0["exact_cost_a2_oracle"]["stability"]["per_image"]
    expected_full_test_length = int(stage_a["full_test_length"])
    expected_images = stage_a["images"]
    if len(expected_images) != expected_full_test_length:
        raise AssertionError("Stage-A image manifest length differs from full test")
    if len(b0_closure["per_image"]) != expected_full_test_length:
        raise AssertionError("Stage-B0 closure image count differs from full test")
    if len(stability_records) != expected_full_test_length:
        raise AssertionError("Stage-B0 stability image count differs from full test")
    if closure["baseline_k1_crop_forwards"] != b0_closure[
        "baseline_k1_crop_forwards"
    ]:
        raise AssertionError("reconstructed baseline crop total differs from Stage-B0")
    if closure["extra_crop_forwards"] != b0_closure["extra_crop_forwards"]:
        raise AssertionError("reconstructed extra crop total differs from Stage-B0")
    for phase_name in PHASE_NAMES:
        if closure["extra_crop_forwards_by_phase"][phase_name] != b0_closure[
            "extra_crop_forwards_by_phase"
        ][phase_name]:
            raise AssertionError(f"reconstructed {phase_name} crop total differs")
    for image_index, b0_image in enumerate(b0_closure["per_image"]):
        expected_name = str(expected_images[image_index]["sample_name"])
        stability = stability_records[image_index]
        if b0_image["image_index"] != image_index:
            raise AssertionError("Stage-B0 closure image index differs")
        if b0_image["sample_name"] != expected_name:
            raise AssertionError("Stage-B0 closure sample order differs")
        if stability["image_index"] != image_index:
            raise AssertionError("Stage-B0 stability image index differs")
        if stability["sample_name"] != expected_name:
            raise AssertionError("Stage-B0 stability sample order differs")
        selected = expected_phase_crop_ids(rescue_levels, geometry, image_index)
        if any(
            list(selected[name]) != b0_image["phase_crop_ids"][name]
            for name in PHASE_NAMES
        ):
            raise AssertionError(f"image {image_index} crop closure differs from Stage-B0")
        baseline_crops = int(b0_image["baseline_k1_crop_forwards"])
        if baseline_crops % args.inference_batch_size != 0:
            raise AssertionError("normal K1 would use an unsealed short model batch")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    model, cfg = load_model(args.baseline_checkpoint, args.seed)
    if tuple(cfg["window_size"]) != tuple(geometry["crop_size"]):
        raise AssertionError("model crop size differs from closure geometry")
    if len(cfg["labels"]) != NUM_CLASSES:
        raise AssertionError("model class count differs from Stage-A")
    loader, sample_names, full_test_length = build_test_loader(None, cfg["window_size"])
    expected_names = [str(image["sample_name"]) for image in stage_a["images"]]
    if full_test_length != int(stage_a["full_test_length"]):
        raise AssertionError("current WHU test length differs from Stage-A")
    if sample_names != expected_names or len(sample_names) != full_test_length:
        raise AssertionError("current WHU test sample order differs from Stage-A")
    if int(stage_a["evaluated_images"]) != full_test_length:
        raise AssertionError("Stage-A artifact is not a full-test artifact")

    expected_component_definition = stage_a["region_reporting"]["definitions"][
        "components"
    ]
    if stage_a["region_reporting"]["small_region"] != SMALL_REGION:
        raise AssertionError("Stage-A small-region definition changed")
    if stage_a["region_reporting"]["thin_region"] != THIN_REGION:
        raise AssertionError("Stage-A thin-region definition changed")
    k2_regions = stage_a["aggregate"]["endpoints"]["matched_k2"][
        "common_support"
    ]["regions"]

    model.to(device)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    aggregate_confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    region_totals = {
        name: {"pixels": 0, "errors": 0} for name in REGION_NAMES
    }
    phase_crop_totals = {name: 0 for name in PHASE_NAMES}
    cost_records: list[dict[str, int]] = []
    image_records: list[dict[str, Any]] = []
    label_digest = hashlib.sha256()
    k1_prediction_digest = hashlib.sha256()
    candidate_prediction_digest = hashlib.sha256()
    run_started = time.perf_counter()

    for image_index, (optical, sar, label_tensor) in enumerate(loader):
        image_started = time.perf_counter()
        sample_name = sample_names[image_index]
        target_batch = np.ascontiguousarray(
            label_tensor.numpy().astype(np.int64, copy=False)
        )
        if target_batch.ndim != 3 or target_batch.shape[0] != 1:
            raise AssertionError("WHU label must have shape [1,H,W]")
        target = target_batch[0]
        shape = tuple(int(value) for value in target.shape)
        if shape != tuple(geometry["image_shapes"][image_index]):
            raise AssertionError(f"image {image_index} shape differs from Stage-A")
        if tuple(optical.shape[-2:]) != shape or tuple(sar.shape[-2:]) != shape:
            raise AssertionError("RGB, SAR, and label shapes differ")
        label_digest.update(target_batch.tobytes())

        windows = geometry["windows_by_image"][image_index]
        all_crop_ids = tuple(range(len(windows)))
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
        if not np.array_equal(
            normal_execution["count_mat"],
            analytic_count_map(geometry, image_index),
        ):
            raise AssertionError("normal K1 count differs from analytic coverage")
        normal_logits = normalized_phase_logits(normal_execution)
        normal_summary = _execution_summary(normal_execution)
        normal_prediction = np.ascontiguousarray(
            normal_logits.argmax(axis=0).astype(np.int64, copy=False)
        )
        k1_anchor = _anchor_summary(
            "k1",
            normal_prediction,
            target,
            stage_a["images"][image_index],
            stage_a["class_names"],
        )
        k1_prediction_digest.update(normal_prediction.tobytes())
        del normal_execution, normal_prediction
        torch.cuda.empty_cache()

        phase_accumulations, phase_summary, key_validation = (
            _execute_shifted_phases(
                optical,
                sar,
                model,
                rescue_levels,
                geometry,
                image_index,
                device=device,
                batch_size=args.inference_batch_size,
                label=f"b1-full image={image_index + 1}/{full_test_length}",
            )
        )
        _add_model_sample_cost(phase_summary, normal_summary["crop_samples"])
        selected_ids = expected_phase_crop_ids(
            rescue_levels, geometry, image_index
        )
        count_checks = _validate_live_count_maps(
            phase_accumulations,
            selected_ids,
            geometry,
            image_index,
        )
        candidate = compose_policy_logits(
            normal_logits,
            phase_accumulations,
            rescue_levels,
            geometry,
            image_index,
        )
        prediction = candidate["prediction"]
        confusion = confusion_from_arrays(prediction, target, NUM_CLASSES)
        expected_confusion = _expected_image_confusion(
            stage_a, geometry, rescue_levels, image_index
        )
        confusion_equal = bool(np.array_equal(confusion, expected_confusion))
        if not confusion_equal:
            raise AssertionError(
                f"image {image_index} live rescue confusion differs from Stage-B0"
            )
        metrics = metric_summary(confusion, stage_a["class_names"])
        stability = stability_records[image_index]
        if stability["image_index"] != image_index:
            raise AssertionError("Stage-B0 stability image index differs")
        if stability["sample_name"] != sample_name:
            raise AssertionError("Stage-B0 stability sample order differs")
        if abs(float(stability["candidate_miou"]) - metrics["miou"]) > METRIC_TOLERANCE:
            raise AssertionError("live per-image mIoU differs from Stage-B0")
        if int(stability["candidate_errors"]) != int(metrics["errors"]):
            raise AssertionError("live per-image errors differ from Stage-B0")

        region_masks, component_metadata = component_geometry_masks(
            target,
            NUM_CLASSES,
            area_thresholds=(256,),
            thickness_thresholds=(4,),
        )
        if component_metadata != expected_component_definition:
            raise AssertionError("component-region definition differs from Stage-A")
        local_regions = _region_error_counts(
            prediction,
            target,
            geometry["common_bounds"][image_index],
            region_masks,
        )
        for region_name in REGION_NAMES:
            region_totals[region_name]["pixels"] += local_regions[region_name][
                "pixels"
            ]
            region_totals[region_name]["errors"] += local_regions[region_name][
                "errors"
            ]
        for phase_name in PHASE_NAMES:
            phase_crop_totals[phase_name] += len(selected_ids[phase_name])
        candidate_prediction_digest.update(prediction.tobytes())

        selected_extra = int(phase_summary["selected_crop_samples"])
        processed_extra = int(phase_summary["model_forward_crop_samples"])
        padding = int(phase_summary["padding_crop_samples"])
        expected_processed_extra = (
            (selected_extra + args.inference_batch_size - 1)
            // args.inference_batch_size
            * args.inference_batch_size
            if selected_extra
            else 0
        )
        if processed_extra != expected_processed_extra:
            raise AssertionError("shifted model samples differ from fixed-batch plan")
        if padding != expected_processed_extra - selected_extra:
            raise AssertionError("shifted padding count differs from fixed-batch plan")
        cost_records.append(
            {
                "baseline_crop_samples": int(normal_summary["crop_samples"]),
                "selected_extra_crop_samples": selected_extra,
                "processed_extra_crop_samples": processed_extra,
                "padding_crop_samples": padding,
            }
        )
        closure_image = closure["per_image"][image_index]
        if selected_extra != int(closure_image["extra_crop_forwards"]):
            raise AssertionError("live selected extra crops differ from closure")
        image_records.append(
            {
                "image_index": image_index,
                "sample_name": sample_name,
                "shape_hw": list(shape),
                "normal_k1": {
                    "anchor": k1_anchor,
                    "execution": normal_summary,
                },
                "rescue": {
                    "selected_counts": closure_image["selected_counts"],
                    "execution": phase_summary,
                    "key_validation": key_validation,
                    "analytic_count_checks": count_checks,
                    "prediction_sha256": array_sha256(prediction),
                    "confusion_equal_expected": confusion_equal,
                    "confusion": confusion.tolist(),
                    "metrics": metrics,
                    "stage_b0_candidate_miou_equal": True,
                    "regions_on_common_support": local_regions,
                    "component_region_definition": component_metadata,
                },
                "wall_seconds": float(time.perf_counter() - image_started),
            }
        )
        aggregate_confusion += confusion
        print(
            f"image={image_index + 1}/{full_test_length} name={sample_name} "
            f"selected={selected_extra} processed={processed_extra} "
            f"padding={padding} miou={metrics['miou_percent']:.6f}% PASS",
            flush=True,
        )
        del (
            phase_accumulations,
            candidate,
            prediction,
            normal_logits,
            region_masks,
            target_batch,
        )
        torch.cuda.empty_cache()

    wall_seconds = time.perf_counter() - run_started
    label_streaming_sha = label_digest.hexdigest()
    expected_label_streaming_sha = str(stage_a["prediction_sha256"]["label"])
    if label_streaming_sha != expected_label_streaming_sha:
        raise AssertionError("aggregate label streaming SHA differs from Stage-A")
    k1_streaming_sha = k1_prediction_digest.hexdigest()
    expected_k1_streaming_sha = str(stage_a["prediction_sha256"]["k1"])
    if k1_streaming_sha != expected_k1_streaming_sha:
        raise AssertionError("aggregate K1 streaming prediction SHA differs from Stage-A")
    candidate_streaming_sha = candidate_prediction_digest.hexdigest()
    expected_full_confusion = np.asarray(
        stage_b0["exact_cost_a2_oracle"]["full_image"]["confusion"],
        dtype=np.int64,
    )
    aggregate_confusion_equal = bool(
        np.array_equal(aggregate_confusion, expected_full_confusion)
    )
    if not aggregate_confusion_equal:
        raise AssertionError("aggregate live rescue confusion differs from Stage-B0")
    aggregate_metrics = metric_summary(
        aggregate_confusion, stage_a["class_names"]
    )
    expected_miou = float(
        stage_b0["exact_cost_a2_oracle"]["full_image"]["miou"]
    )
    if abs(aggregate_metrics["miou"] - expected_miou) > METRIC_TOLERANCE:
        raise AssertionError("aggregate live rescue mIoU differs from Stage-B0")

    regions = _summarize_region_totals(region_totals)
    for region_name in REGION_NAMES:
        if int(regions[region_name]["pixels"]) != int(
            k2_regions[region_name]["pixels"]
        ):
            raise AssertionError(
                f"aggregate {region_name} support differs from Stage-A"
            )
    physical_cost = _physical_cost_summary(cost_records)
    expected_processed_total = sum(
        (
            (int(item["extra_crop_forwards"]) + args.inference_batch_size - 1)
            // args.inference_batch_size
            * args.inference_batch_size
        )
        for item in closure["per_image"]
    )
    if physical_cost["processed_extra_crop_samples_including_padding"] != int(
        expected_processed_total
    ):
        raise AssertionError("aggregate processed crop total differs from batch plan")
    expected_padding_total = expected_processed_total - int(
        b0_closure["extra_crop_forwards"]
    )
    if physical_cost["padding_crop_samples"] != expected_padding_total:
        raise AssertionError("aggregate padding total differs from batch plan")
    if physical_cost["baseline_crop_samples"] != int(
        b0_closure["baseline_k1_crop_forwards"]
    ):
        raise AssertionError("live baseline crop total differs from Stage-B0")
    if physical_cost["selected_extra_crop_samples"] != int(
        b0_closure["extra_crop_forwards"]
    ):
        raise AssertionError("live selected crop total differs from Stage-B0")
    for phase_name in PHASE_NAMES:
        if phase_crop_totals[phase_name] != int(
            b0_closure["extra_crop_forwards_by_phase"][phase_name]
        ):
            raise AssertionError(f"live {phase_name} crop total differs from Stage-B0")
    normal_batch_calls = sum(
        int(record["normal_k1"]["execution"]["batch_calls"])
        for record in image_records
    )
    shifted_batch_calls = sum(
        int(record["rescue"]["execution"]["batch_calls"])
        for record in image_records
    )
    execution_totals = {
        "normal_model_samples": int(physical_cost["baseline_crop_samples"]),
        "shifted_selected_samples": int(
            physical_cost["selected_extra_crop_samples"]
        ),
        "shifted_model_samples_including_padding": int(
            physical_cost["processed_extra_crop_samples_including_padding"]
        ),
        "total_model_samples_including_padding": int(
            physical_cost["baseline_crop_samples"]
            + physical_cost["processed_extra_crop_samples_including_padding"]
        ),
        "normal_batch_calls": normal_batch_calls,
        "shifted_batch_calls": shifted_batch_calls,
        "total_batch_calls": normal_batch_calls + shifted_batch_calls,
        "model_batch_size": args.inference_batch_size,
    }
    if normal_batch_calls * args.inference_batch_size != int(
        physical_cost["baseline_crop_samples"]
    ):
        raise AssertionError("normal batch-call accounting is inconsistent")
    if shifted_batch_calls * args.inference_batch_size != int(
        physical_cost["processed_extra_crop_samples_including_padding"]
    ):
        raise AssertionError("shifted batch-call accounting is inconsistent")

    endpoints = stage_a["aggregate"]["endpoints"]
    random_p95 = float(stage_b0["exact_cost_random_control"]["miou"]["p95"])
    gate = evaluate_phase_utility_gate(
        k1_miou=float(endpoints["k1"]["full_image"]["miou"]),
        k2_miou=float(endpoints["matched_k2"]["full_image"]["miou"]),
        k4_miou=float(endpoints["k4"]["full_image"]["miou"]),
        mixed_miou=float(aggregate_metrics["miou"]),
        forward_equivalent_cost=float(
            physical_cost["physical_model_sample_cost_ratio"]
        ),
        k2_small_error_rate=float(k2_regions["small"]["error_rate"]),
        mixed_small_error_rate=float(regions["small"]["error_rate"]),
        k2_thin_error_rate=float(k2_regions["thin"]["error_rate"]),
        mixed_thin_error_rate=float(regions["thin"]["error_rate"]),
        random_p95_miou=random_p95,
        thresholds=PhaseUtilityGateThresholds(
            max_forward_equivalent_cost=2.0,
            min_k4_gain_retention=0.70,
            min_miou_delta_over_k2=0.0005,
            max_small_error_rate_delta=0.0,
            max_thin_error_rate_delta=0.0,
        ),
    )
    gate = _apply_registered_k2_gain_threshold(gate, 0.0005)
    k1_miou = float(endpoints["k1"]["full_image"]["miou"])
    k2_miou = float(endpoints["matched_k2"]["full_image"]["miou"])
    k4_miou = float(endpoints["k4"]["full_image"]["miou"])
    candidate_miou = float(aggregate_metrics["miou"])
    efficacy_summary = {
        "k1_miou": k1_miou,
        "matched_k2_miou": k2_miou,
        "k4_miou": k4_miou,
        "candidate_miou": candidate_miou,
        "candidate_minus_k1_pp": float((candidate_miou - k1_miou) * 100.0),
        "candidate_minus_matched_k2_pp": float(
            (candidate_miou - k2_miou) * 100.0
        ),
        "candidate_minus_k4_pp": float((candidate_miou - k4_miou) * 100.0),
        "k4_gain_retention": gate["observed"]["k4_gain_retention"],
        "random_p95_miou": random_p95,
        "candidate_minus_random_p95_pp": float(
            (candidate_miou - random_p95) * 100.0
        ),
    }
    per_image_physical_pass = bool(
        physical_cost["maximum_per_image_physical_cost_ratio"]
        <= 2.0 + METRIC_TOLERANCE
    )
    k2_deltas_pp = np.asarray(
        [float(record["candidate_minus_k2_pp"]) for record in stability_records],
        dtype=np.float64,
    )
    k1_deltas_pp = np.asarray(
        [float(record["candidate_minus_k1_pp"]) for record in stability_records],
        dtype=np.float64,
    )
    k4_deltas_pp = np.asarray(
        [float(record["candidate_minus_k4_pp"]) for record in stability_records],
        dtype=np.float64,
    )
    stability_descriptive = {
        "role": "descriptive audit; not an additional preregistered hard gate",
        "images_above_k1": int(np.count_nonzero(k1_deltas_pp > 0.0)),
        "images_above_matched_k2": int(np.count_nonzero(k2_deltas_pp > 0.0)),
        "images_at_least_0_05pp_above_matched_k2": int(
            np.count_nonzero(k2_deltas_pp >= 0.05)
        ),
        "images_above_k4": int(np.count_nonzero(k4_deltas_pp > 0.0)),
        "candidate_minus_k2_pp_minimum": float(k2_deltas_pp.min()),
        "candidate_minus_k2_pp_median": float(np.median(k2_deltas_pp)),
        "candidate_minus_k2_pp_maximum": float(k2_deltas_pp.max()),
    }
    non_latency_checks = {
        "aggregate_confusion_exact_stage_b0": aggregate_confusion_equal,
        "aggregate_miou_exact_stage_b0": True,
        "all_image_confusions_exact_stage_b0": True,
        "all_image_k1_anchors_exact_stage_a": True,
        "aggregate_label_streaming_sha_exact_stage_a": True,
        "aggregate_k1_streaming_sha_exact_stage_a": True,
        "all_executed_crop_keys_exact_closure": True,
        "all_count_maps_equal_analytic_coverage": True,
        "all_normal_model_batch_shapes_equal_frozen_batch_size": all(
            all(
                size == args.inference_batch_size
                for size in record["normal_k1"]["execution"]["batch_sizes"]
            )
            for record in image_records
        ),
        "all_shifted_model_batch_shapes_equal_frozen_batch_size": all(
            all(
                size == args.inference_batch_size
                for size in record["rescue"]["execution"]["model_batch_sizes"]
            )
            for record in image_records
        ),
        "overall_logical_cost_at_most_2x": bool(
            physical_cost["logical_unique_crop_cost_ratio"]
            <= 2.0 + METRIC_TOLERANCE
        ),
        "every_image_logical_cost_at_most_2x": bool(
            physical_cost["maximum_per_image_logical_cost_ratio"]
            <= 2.0 + METRIC_TOLERANCE
        ),
        "overall_physical_cost_at_most_2x": bool(
            physical_cost["physical_model_sample_cost_ratio"]
            <= 2.0 + METRIC_TOLERANCE
        ),
        "every_image_physical_cost_at_most_2x": per_image_physical_pass,
        "small_error_count_not_worse_than_matched_k2": bool(
            int(regions["small"]["errors"]) <= int(k2_regions["small"]["errors"])
        ),
        "thin_error_count_not_worse_than_matched_k2": bool(
            int(regions["thin"]["errors"]) <= int(k2_regions["thin"]["errors"])
        ),
        "utility_and_structure_gate_passed": bool(gate["passed"]),
    }
    non_latency_passed = all(non_latency_checks.values())
    if non_latency_passed:
        outcome = "PASS_B1_LIVE_OUTPUT_STRUCTURE_GATES_LATENCY_PENDING"
        meaning = (
            "the frozen rescue reproduced Stage-B0 on all 20 live images and "
            "passed physical-cost, efficacy, random, and small/thin gates; a "
            "same-primitive latency benchmark is still required for full Stage-B/H2"
        )
    else:
        outcome = "NO_GO_B1_LIVE_NON_LATENCY_GATE_FAILED"
        meaning = (
            "the live runner completed exactly, but at least one frozen cost, "
            "efficacy, stability, random, or structure gate failed"
        )

    payload = {
        "status": "PASS",
        "status_meaning": "formal full-test live execution completed without runner error",
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "full-official-test-live-output-and-structure",
        "scientific_scope": (
            "frozen Stage-B0 rescue evaluated on the 20-image official test set; "
            "the user explicitly allowed official test for current method selection"
        ),
        "scientific_decision_evaluated": True,
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
            "b1b_smoke": {
                "path": str(args.b1b_smoke_json.resolve()),
                "sha256": b1b_smoke_sha,
            },
            "baseline_checkpoint": {
                "path": str(args.baseline_checkpoint.resolve()),
                "sha256": checkpoint_sha,
            },
        },
        "protocol": {
            "evaluated_images": full_test_length,
            "phase_order": list(PHASE_NAMES),
            "phase_shifts_dy_dx": {
                name: list(geometry["phase_shifts"][name]) for name in PHASE_NAMES
            },
            "crop_size_hw": list(geometry["crop_size"]),
            "stride_hw": list(geometry["stride"]),
            "shifted_batch_packing": (
                "per-image global phase pool; fixed model batch shape; cyclic "
                "duplicate padding only in the final batch; padding not scattered"
            ),
            "cost_policy": (
                "both unique selected crop keys and all model samples including "
                "padding are reported; physical model samples drive the 2x gate"
            ),
            "small_region": SMALL_REGION,
            "thin_region": THIN_REGION,
            "latency": "not evaluated by this correctness-oriented run",
            "future_latency_primary_gate": (
                "one untimed first-image warm-up per policy, then three paired "
                "full-test repeats with alternating order; primary statistic is "
                "median(rescue_total_e2e_ms - k4_total_e2e_ms), required < 0"
            ),
        },
        "images": image_records,
        "aggregate": {
            "confusion": aggregate_confusion.tolist(),
            "confusion_equal_stage_b0": aggregate_confusion_equal,
            "label_streaming_sha256": label_streaming_sha,
            "expected_label_streaming_sha256": expected_label_streaming_sha,
            "k1_prediction_streaming_sha256": k1_streaming_sha,
            "expected_k1_prediction_streaming_sha256": expected_k1_streaming_sha,
            "candidate_prediction_streaming_sha256": candidate_streaming_sha,
            "metrics": aggregate_metrics,
            "efficacy": efficacy_summary,
            "regions_on_common_support": regions,
            "matched_k2_regions_on_common_support": k2_regions,
            "phase_crop_samples": phase_crop_totals,
            "cost": physical_cost,
            "execution": execution_totals,
            "stability_descriptive": stability_descriptive,
        },
        "utility_and_structure_gate": gate,
        "non_latency_decision": {
            "outcome": outcome,
            "passed": non_latency_passed,
            "checks": non_latency_checks,
            "authorizes_same_primitive_latency_benchmark": non_latency_passed,
            "full_stage_b_scientific_pass": False,
            "h2_confirmed": False,
            "meaning": meaning,
        },
        "explicit_non_claims": [
            "no latency, throughput, or deployment result",
            "no learned or deployable router",
            "no validation-set or cross-dataset generalization claim",
            "no full Stage-B or H2 confirmation until same-primitive latency passes",
        ],
        "runtime": {
            "wall_seconds": float(wall_seconds),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "role": "execution record only; not a latency benchmark",
        },
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "live_common_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_sparse_live_common.py"
            ),
            "passed_b1b_runner_sha256": b1b_smoke["reproducibility"][
                "runner_sha256"
            ],
            "current_first_image_runner_sha256": file_sha256(
                REPO_ROOT / "scripts" / "evaluate_whu_phase_sparse_live.py"
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
    print(f"b1_full_result={args.output_path.resolve()}", flush=True)
    print(f"b1_full_outcome={outcome}", flush=True)
    print("b1_full_status=PASS", flush=True)


if __name__ == "__main__":
    main()
