"""Confirm the final Stage-B latency gate with the frozen live sparse operator.

This benchmark is intentionally narrow.  It compares the complete frozen
Stage-B0 rescue pipeline against the support-pruned K4 endpoint after the
20-image B1 output/structure run has passed.  Both arms independently execute
normal K1, shifted-canvas construction, fixed-shape sparse crop forwards,
inverse alignment, policy composition, and argmax.  No logits or predictions
are shared across arms.

The preregistered primary is the median of three paired full-test differences
``rescue_total_e2e_ms - support_pruned_k4_total_e2e_ms`` and passes only when it
is strictly negative.  Hashing, confusion checks, logging, and JSON writing are
kept outside the timed region.  The GT-informed action map is already frozen;
this script does not train or pretend to provide a deployable router.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
)
from scripts.evaluate_whu_phase_sparse_full import _load_b1b_smoke  # noqa: E402
from scripts.evaluate_whu_translation_consistency import translate_tensor  # noqa: E402
from scripts.phase_closure_common import (  # noqa: E402
    PHASE_NAMES,
    build_phase_closure_geometry,
    summarize_closure,
    validate_levels,
)
from scripts.phase_sparse_live_common import (  # noqa: E402
    forward_selected_phase_crops,
    forward_selected_phase_key_crops,
    normalized_phase_logits,
)
from scripts.phase_sparse_replay_common import (  # noqa: E402
    analytic_count_map,
    array_sha256,
    compose_policy_logits,
    endpoint_levels,
    expected_phase_crop_ids,
)
from scripts.phase_utility_common import confusion_from_arrays  # noqa: E402
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)


ARTIFACT_TYPE = "whu_phase_sparse_latency_b1"
SCHEMA_VERSION = 1
FULL_B1_ARTIFACT_TYPE = "whu_phase_sparse_live_b1_full"
FULL_B1_SCHEMA_VERSION = 1
NUM_CLASSES = 7
PAIRED_REPEATS = 3
FROZEN_BATCH_SIZE = 8
POLICY_RESCUE = "rescue"
POLICY_K4 = "support_pruned_k4"
POLICIES = (POLICY_RESCUE, POLICY_K4)
WARMUP_ORDER = (POLICY_RESCUE, POLICY_K4)
PRIMARY_EXPRESSION = (
    "median(rescue_total_e2e_ms - support_pruned_k4_total_e2e_ms) < 0"
)
EXPECTED_FULL_TEST_LENGTH = 20
EXPECTED_LEDGER = {
    POLICY_RESCUE: {
        "normal_selected_samples": 3520,
        "normal_processed_samples": 3520,
        "normal_batch_calls": 440,
        "shifted_selected_samples": 3518,
        "shifted_processed_samples": 3520,
        "shifted_padding_samples": 2,
        "shifted_batch_calls": 440,
        "total_processed_samples": 7040,
        "total_batch_calls": 880,
    },
    POLICY_K4: {
        "normal_selected_samples": 3520,
        "normal_processed_samples": 3520,
        "normal_batch_calls": 440,
        "shifted_selected_samples": 7560,
        "shifted_processed_samples": 7680,
        "shifted_padding_samples": 120,
        "shifted_batch_calls": 960,
        "total_processed_samples": 11200,
        "total_batch_calls": 1400,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-a-json", type=Path, required=True)
    parser.add_argument("--stage-b0-json", type=Path, required=True)
    parser.add_argument("--b1b-smoke-json", type=Path, required=True)
    parser.add_argument("--b1-full-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--paired-repeats", type=int, default=PAIRED_REPEATS)
    parser.add_argument(
        "--inference-batch-size", type=int, default=FROZEN_BATCH_SIZE
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (
        args.baseline_checkpoint,
        args.stage_a_json,
        args.stage_b0_json,
        args.b1b_smoke_json,
        args.b1_full_json,
    ):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.paired_repeats != PAIRED_REPEATS:
        parser.error(f"formal latency protocol requires exactly {PAIRED_REPEATS} repeats")
    if args.inference_batch_size != FROZEN_BATCH_SIZE:
        parser.error(
            f"formal latency protocol requires batch size {FROZEN_BATCH_SIZE}"
        )
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    return args


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain a JSON object")
    return payload


def _load_full_b1(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path, label="full B1 artifact")
    decision = payload.get("non_latency_decision", {})
    if payload.get("artifact_type") != FULL_B1_ARTIFACT_TYPE:
        raise ValueError("input is not a formal full-B1 live artifact")
    if payload.get("schema_version") != FULL_B1_SCHEMA_VERSION:
        raise ValueError("unsupported full-B1 artifact schema")
    if payload.get("status") != "PASS":
        raise ValueError("full-B1 runner did not complete successfully")
    if payload.get("latency_decision_evaluated") is not False:
        raise ValueError("full-B1 source unexpectedly claims a latency decision")
    if (
        decision.get("passed") is not True
        or decision.get("authorizes_same_primitive_latency_benchmark") is not True
        or decision.get("outcome")
        != "PASS_B1_LIVE_OUTPUT_STRUCTURE_GATES_LATENCY_PENDING"
    ):
        raise ValueError("full-B1 non-latency gates did not authorize latency")
    return payload


def paired_policy_order(repeat_index: int, image_index: int) -> tuple[str, str]:
    """Return the preregistered balanced within-image arm order."""

    repeat_index = int(repeat_index)
    image_index = int(image_index)
    if repeat_index < 0 or repeat_index >= PAIRED_REPEATS:
        raise ValueError("repeat index is outside the formal protocol")
    if image_index < 0 or image_index >= EXPECTED_FULL_TEST_LENGTH:
        raise ValueError("image index is outside the formal protocol")
    if (repeat_index + image_index) % 2 == 0:
        return (POLICY_RESCUE, POLICY_K4)
    return (POLICY_K4, POLICY_RESCUE)


def evaluate_primary_latency_gate(
    paired_deltas_ms: Sequence[float],
) -> dict[str, Any]:
    """Apply the frozen three-pair median-difference gate."""

    values = np.asarray(tuple(paired_deltas_ms), dtype=np.float64)
    if values.shape != (PAIRED_REPEATS,):
        raise ValueError(f"primary latency gate requires {PAIRED_REPEATS} deltas")
    if not np.all(np.isfinite(values)):
        raise ValueError("paired latency deltas must be finite")
    median = float(np.median(values))
    passed = bool(median < 0.0)
    return {
        "evaluated": True,
        "primary_statistic": (
            "median of three paired full-test rescue-minus-K4 total-E2E deltas"
        ),
        "expression": PRIMARY_EXPRESSION,
        "paired_deltas_ms": [float(value) for value in values],
        "median_paired_delta_ms": median,
        "threshold_ms": 0.0,
        "comparator": "<",
        "passed": passed,
        "outcome": (
            "PASS_FULL_STAGE_B_H2_GT_ORACLE_EXECUTION_FEASIBILITY"
            if passed
            else "NO_GO_STAGE_B_LATENCY_NOT_FASTER_THAN_SUPPORT_PRUNED_K4"
        ),
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("timing distribution requires finite values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _empty_ledger() -> dict[str, int]:
    return {
        "normal_selected_samples": 0,
        "normal_processed_samples": 0,
        "normal_batch_calls": 0,
        "shifted_selected_samples": 0,
        "shifted_processed_samples": 0,
        "shifted_padding_samples": 0,
        "shifted_batch_calls": 0,
        "total_processed_samples": 0,
        "total_batch_calls": 0,
    }


def _add_ledger(total: dict[str, int], item: Mapping[str, int]) -> None:
    if set(total) != set(item):
        raise ValueError("latency ledger keys differ")
    for key in total:
        value = int(item[key])
        if value < 0:
            raise ValueError("latency ledger values must be non-negative")
        total[key] += value


def _validate_exact_ledger(
    policy: str, actual: Mapping[str, int], expected: Mapping[str, int]
) -> None:
    normalized = {key: int(value) for key, value in actual.items()}
    target = {key: int(value) for key, value in expected.items()}
    if normalized != target:
        raise AssertionError(
            f"{policy} physical workload differs from the frozen ledger: "
            f"actual={normalized} expected={target}"
        )


def _expected_policy_ledger(
    closure: Mapping[str, Any], *, batch_size: int
) -> dict[str, int]:
    normal = int(closure["baseline_k1_crop_forwards"])
    shifted_selected = int(closure["extra_crop_forwards"])
    shifted_processed = sum(
        math.ceil(int(image["extra_crop_forwards"]) / batch_size) * batch_size
        for image in closure["per_image"]
    )
    if normal % batch_size:
        raise AssertionError("normal K1 workload contains a short batch")
    return {
        "normal_selected_samples": normal,
        "normal_processed_samples": normal,
        "normal_batch_calls": normal // batch_size,
        "shifted_selected_samples": shifted_selected,
        "shifted_processed_samples": shifted_processed,
        "shifted_padding_samples": shifted_processed - shifted_selected,
        "shifted_batch_calls": shifted_processed // batch_size,
        "total_processed_samples": normal + shifted_processed,
        "total_batch_calls": (normal + shifted_processed) // batch_size,
    }


def _expected_image_ledger(
    closure_image: Mapping[str, Any], *, batch_size: int
) -> dict[str, int]:
    normal = int(closure_image["baseline_k1_crop_forwards"])
    shifted_selected = int(closure_image["extra_crop_forwards"])
    shifted_processed = math.ceil(shifted_selected / batch_size) * batch_size
    if normal <= 0 or normal % batch_size:
        raise AssertionError("per-image normal K1 workload contains a short batch")
    return {
        "normal_selected_samples": normal,
        "normal_processed_samples": normal,
        "normal_batch_calls": normal // batch_size,
        "shifted_selected_samples": shifted_selected,
        "shifted_processed_samples": shifted_processed,
        "shifted_padding_samples": shifted_processed - shifted_selected,
        "shifted_batch_calls": shifted_processed // batch_size,
        "total_processed_samples": normal + shifted_processed,
        "total_batch_calls": (normal + shifted_processed) // batch_size,
    }


def _validate_closure_against_source(
    *,
    policy: str,
    levels: np.ndarray,
    geometry: Mapping[str, Any],
    computed: Mapping[str, Any],
    source: Mapping[str, Any],
    batch_size: int,
) -> dict[str, int]:
    for key in (
        "baseline_k1_crop_forwards",
        "extra_crop_forwards",
        "extra_crop_forwards_by_phase",
    ):
        if computed[key] != source[key]:
            raise AssertionError(f"{policy} reconstructed closure differs for {key}")
    if len(computed["per_image"]) != EXPECTED_FULL_TEST_LENGTH:
        raise AssertionError(f"{policy} closure is not the formal 20-image set")
    if len(source["per_image"]) != EXPECTED_FULL_TEST_LENGTH:
        raise AssertionError(f"{policy} source closure is not the formal 20-image set")
    for image_index, (actual, reference) in enumerate(
        zip(computed["per_image"], source["per_image"], strict=True)
    ):
        if actual["image_index"] != image_index or reference["image_index"] != image_index:
            raise AssertionError(f"{policy} closure image index differs")
        if actual["sample_name"] != reference["sample_name"]:
            raise AssertionError(f"{policy} closure sample order differs")
        selected = expected_phase_crop_ids(levels, geometry, image_index)
        for phase_name in PHASE_NAMES:
            if list(selected[phase_name]) != reference["phase_crop_ids"][phase_name]:
                raise AssertionError(
                    f"{policy} {phase_name} crop keys differ on image {image_index}"
                )
        if int(actual["extra_crop_forwards"]) != int(
            reference["extra_crop_forwards"]
        ):
            raise AssertionError(f"{policy} per-image crop count differs")
    ledger = _expected_policy_ledger(computed, batch_size=batch_size)
    _validate_exact_ledger(policy, ledger, EXPECTED_LEDGER[policy])
    return ledger


def _validate_full_b1_sources(
    full_b1: Mapping[str, Any],
    *,
    stage_a_sha: str,
    stage_b0_sha: str,
    b1b_sha: str,
    checkpoint_sha: str,
    live_common_sha: str,
    batch_size: int,
) -> None:
    expected = {
        "stage_a": stage_a_sha,
        "stage_b0": stage_b0_sha,
        "b1b_smoke": b1b_sha,
        "baseline_checkpoint": checkpoint_sha,
    }
    sources = full_b1.get("source_artifacts", {})
    for name, sha256 in expected.items():
        if sources.get(name, {}).get("sha256") != sha256:
            raise AssertionError(f"full-B1 {name} source SHA256 differs")
    reproducibility = full_b1.get("reproducibility", {})
    if reproducibility.get("live_common_sha256") != live_common_sha:
        raise AssertionError("full-B1 used a different sparse execution primitive")
    if reproducibility.get("inference_batch_size") != batch_size:
        raise AssertionError("full-B1 used a different inference batch size")
    if full_b1.get("protocol", {}).get("evaluated_images") != EXPECTED_FULL_TEST_LENGTH:
        raise AssertionError("full-B1 artifact is not the formal 20-image run")
    if len(full_b1.get("images", [])) != EXPECTED_FULL_TEST_LENGTH:
        raise AssertionError("full-B1 image records are incomplete")


def _pipeline_execution(
    *,
    optical: torch.Tensor,
    sar: torch.Tensor,
    model: torch.nn.Module,
    levels: np.ndarray,
    selected_ids: Mapping[str, Sequence[int]],
    geometry: Mapping[str, Any],
    image_index: int,
    device: torch.device,
    batch_size: int,
    validate_count_maps: bool,
) -> dict[str, Any]:
    """Execute one complete policy and keep diagnostics outside its wall timer."""

    if optical.device.type != "cpu" or sar.device.type != "cpu":
        raise ValueError("paired latency inputs must begin on CPU")
    windows = geometry["windows_by_image"][image_index]
    all_crop_ids = tuple(range(len(windows)))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    wall_started_ns = time.perf_counter_ns()
    start_event.record()

    normal_started_ns = time.perf_counter_ns()
    normal_execution = forward_selected_phase_crops(
        optical.to(device),
        sar.to(device),
        model,
        windows,
        all_crop_ids,
        n_output_channels=NUM_CLASSES,
        batch_size=batch_size,
        decoder_head_type="linear",
        require_full_coverage=True,
    )
    normal_logits = normalized_phase_logits(normal_execution)
    normal_finished_ns = time.perf_counter_ns()

    shifted_started_ns = time.perf_counter_ns()
    phase_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for phase_name in PHASE_NAMES:
        if not selected_ids[phase_name]:
            continue
        dy, dx = geometry["phase_shifts"][phase_name]
        phase_tensors[phase_name] = (
            translate_tensor(optical, dy, dx).to(device),
            translate_tensor(sar, dy, dx).to(device),
        )
    shifted_execution = forward_selected_phase_key_crops(
        phase_tensors,
        model,
        windows,
        selected_ids,
        phase_order=PHASE_NAMES,
        n_output_channels=NUM_CLASSES,
        batch_size=batch_size,
        decoder_head_type="linear",
    )
    composed = compose_policy_logits(
        normal_logits,
        shifted_execution["phases"],
        levels,
        geometry,
        image_index,
        include_digests=False,
    )
    prediction = np.ascontiguousarray(composed["prediction"])
    phase_audit = composed["phase_audit"]
    shifted_finished_ns = time.perf_counter_ns()

    end_event.record()
    torch.cuda.synchronize(device)
    wall_finished_ns = time.perf_counter_ns()
    cuda_event_span_ms = float(start_event.elapsed_time(end_event))
    peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))

    # Everything below is diagnostic and deliberately outside the primary timer.
    normal_prediction = np.ascontiguousarray(
        normal_logits.argmax(axis=0).astype(np.int64)
    )
    if validate_count_maps:
        expected_normal_count = analytic_count_map(geometry, image_index)
        if not np.array_equal(normal_execution["count_mat"], expected_normal_count):
            raise AssertionError("timed normal K1 count differs from analytic coverage")
    observed_keys = tuple(shifted_execution["observed_phase_crop_keys"])
    expected_keys = tuple(
        (phase_name, int(crop_id))
        for phase_name in PHASE_NAMES
        for crop_id in selected_ids[phase_name]
    )
    if observed_keys != expected_keys:
        raise AssertionError("timed shifted crop keys differ from the frozen plan")
    for phase_name in PHASE_NAMES:
        selected = tuple(int(value) for value in selected_ids[phase_name])
        phase = shifted_execution["phases"].get(phase_name)
        if not selected:
            if phase is not None:
                raise AssertionError("inactive phase unexpectedly produced accumulation")
            continue
        if phase is None:
            raise AssertionError("active phase did not produce accumulation")
        if validate_count_maps:
            expected_count = analytic_count_map(geometry, image_index, selected)
            if not np.array_equal(phase["count_mat"], expected_count):
                raise AssertionError(
                    f"timed {phase_name} count differs from analytic coverage"
                )

    normal_samples = int(normal_execution["crop_samples"])
    normal_calls = int(normal_execution["batch_calls"])
    shifted_selected = int(shifted_execution["selected_crop_samples"])
    shifted_processed = int(shifted_execution["model_forward_crop_samples"])
    shifted_padding = int(shifted_execution["padding_crop_samples"])
    shifted_calls = int(shifted_execution["batch_calls"])
    if any(int(size) != batch_size for size in normal_execution["batch_sizes"]):
        raise AssertionError("timed normal execution used a non-frozen batch shape")
    if any(int(size) != batch_size for size in shifted_execution["model_batch_sizes"]):
        raise AssertionError("timed shifted execution used a non-frozen batch shape")
    if shifted_processed != shifted_selected + shifted_padding:
        raise AssertionError("timed shifted padding ledger is inconsistent")
    ledger = {
        "normal_selected_samples": normal_samples,
        "normal_processed_samples": normal_samples,
        "normal_batch_calls": normal_calls,
        "shifted_selected_samples": shifted_selected,
        "shifted_processed_samples": shifted_processed,
        "shifted_padding_samples": shifted_padding,
        "shifted_batch_calls": shifted_calls,
        "total_processed_samples": normal_samples + shifted_processed,
        "total_batch_calls": normal_calls + shifted_calls,
    }
    timing = {
        "pipeline_e2e_ms": float((wall_finished_ns - wall_started_ns) / 1e6),
        "normal_component_wall_ms": float(
            (normal_finished_ns - normal_started_ns) / 1e6
        ),
        "shifted_compose_component_wall_ms": float(
            (shifted_finished_ns - shifted_started_ns) / 1e6
        ),
        "cuda_event_wall_span_ms": cuda_event_span_ms,
        "primitive_core_wall_ms": float(
            1000.0
            * (
                float(normal_execution["wall_seconds"])
                + float(shifted_execution["wall_seconds"])
            )
        ),
        "peak_cuda_memory_bytes": peak_memory_bytes,
    }
    del (
        composed,
        normal_logits,
        normal_execution,
        shifted_execution,
        phase_tensors,
    )
    return {
        "prediction": prediction,
        "normal_prediction": normal_prediction,
        "phase_audit": phase_audit,
        "timing": timing,
        "ledger": ledger,
    }


def _validate_prediction(
    *,
    policy: str,
    result: Mapping[str, Any],
    target: np.ndarray,
    expected_policy_sha: str,
    expected_k1_sha: str,
    expected_policy_confusion: np.ndarray,
) -> dict[str, Any]:
    prediction = np.asarray(result["prediction"], dtype=np.int64)
    normal_prediction = np.asarray(result["normal_prediction"], dtype=np.int64)
    if prediction.shape != target.shape or normal_prediction.shape != target.shape:
        raise AssertionError("timed prediction shape differs from the target")
    actual_sha = array_sha256(np.ascontiguousarray(prediction))
    actual_k1_sha = array_sha256(np.ascontiguousarray(normal_prediction))
    confusion = confusion_from_arrays(prediction, target, NUM_CLASSES)
    if actual_sha != expected_policy_sha:
        raise AssertionError(f"{policy} timed prediction SHA differs")
    if actual_k1_sha != expected_k1_sha:
        raise AssertionError(f"{policy} timed K1 prediction SHA differs")
    if not np.array_equal(confusion, expected_policy_confusion):
        raise AssertionError(f"{policy} timed confusion differs")
    return {
        "prediction_sha256": actual_sha,
        "expected_prediction_sha256": expected_policy_sha,
        "prediction_sha256_equal": True,
        "normal_k1_prediction_sha256": actual_k1_sha,
        "expected_normal_k1_prediction_sha256": expected_k1_sha,
        "normal_k1_prediction_sha256_equal": True,
        "confusion_equal_expected": True,
        "confusion": confusion,
    }


def _policy_summary(repeats: Sequence[Mapping[str, Any]], policy: str) -> dict[str, Any]:
    totals = [float(repeat["policies"][policy]["total_e2e_ms"]) for repeat in repeats]
    pipeline_values = [
        float(image["policies"][policy]["pipeline_e2e_ms"])
        for repeat in repeats
        for image in repeat["images"]
    ]
    total_values = [
        float(image["policies"][policy]["total_e2e_ms_including_shared_input"])
        for repeat in repeats
        for image in repeat["images"]
    ]
    normal_values = [
        float(image["policies"][policy]["normal_component_wall_ms"])
        for repeat in repeats
        for image in repeat["images"]
    ]
    shifted_values = [
        float(image["policies"][policy]["shifted_compose_component_wall_ms"])
        for repeat in repeats
        for image in repeat["images"]
    ]
    cuda_event_values = [
        float(image["policies"][policy]["cuda_event_wall_span_ms"])
        for repeat in repeats
        for image in repeat["images"]
    ]
    core_values = [
        float(image["policies"][policy]["primitive_core_wall_ms"])
        for repeat in repeats
        for image in repeat["images"]
    ]
    peak_values = [
        int(image["policies"][policy]["peak_cuda_memory_bytes"])
        for repeat in repeats
        for image in repeat["images"]
    ]
    return {
        "full_test_total_e2e_ms": {
            "values": totals,
            "median": float(np.median(np.asarray(totals, dtype=np.float64))),
        },
        "per_image_total_e2e_ms": _distribution(total_values),
        "per_image_pipeline_e2e_ms": _distribution(pipeline_values),
        "per_image_normal_component_wall_ms": _distribution(normal_values),
        "per_image_shifted_compose_component_wall_ms": _distribution(
            shifted_values
        ),
        "per_image_cuda_event_wall_span_ms": _distribution(cuda_event_values),
        "per_image_primitive_core_wall_ms": _distribution(core_values),
        "maximum_peak_cuda_memory_bytes": max(peak_values),
        "throughput_physical_crop_samples_per_second_by_repeat": [
            float(
                repeat["policies"][policy]["ledger"]["total_processed_samples"]
                / (repeat["policies"][policy]["total_e2e_ms"] / 1000.0)
            )
            for repeat in repeats
        ],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal Stage-B latency run")
    torch.cuda.set_device(device)

    stage_a = load_stage_a(args.stage_a_json)
    stage_b0 = load_stage_b0(args.stage_b0_json)
    b1b_smoke = _load_b1b_smoke(args.b1b_smoke_json)
    full_b1 = _load_full_b1(args.b1_full_json)
    stage_a_sha = file_sha256(args.stage_a_json)
    stage_b0_sha = file_sha256(args.stage_b0_json)
    b1b_sha = file_sha256(args.b1b_smoke_json)
    full_b1_sha = file_sha256(args.b1_full_json)
    checkpoint_sha = file_sha256(args.baseline_checkpoint)
    live_common_path = REPO_ROOT / "scripts" / "phase_sparse_live_common.py"
    replay_common_path = REPO_ROOT / "scripts" / "phase_sparse_replay_common.py"
    live_common_sha = file_sha256(live_common_path)
    replay_common_sha = file_sha256(replay_common_path)
    _validate_full_b1_sources(
        full_b1,
        stage_a_sha=stage_a_sha,
        stage_b0_sha=stage_b0_sha,
        b1b_sha=b1b_sha,
        checkpoint_sha=checkpoint_sha,
        live_common_sha=live_common_sha,
        batch_size=args.inference_batch_size,
    )
    if stage_b0.get("source_stage_a", {}).get("sha256") != stage_a_sha:
        raise AssertionError("Stage-B0 source Stage-A SHA256 differs")
    if stage_a.get("baseline_checkpoint_sha256") != checkpoint_sha:
        raise AssertionError("checkpoint SHA256 differs from Stage-A")
    if b1b_smoke.get("reproducibility", {}).get(
        "live_common_sha256"
    ) != live_common_sha:
        raise AssertionError("B1b smoke used a different sparse execution primitive")

    geometry = build_phase_closure_geometry(stage_a["images"], stage_a["cells"])
    rescue_levels = validate_levels(
        stage_b0["exact_cost_a2_oracle"]["levels_by_cell"], geometry
    )
    k4_levels = endpoint_levels(geometry, 4)
    levels_by_policy = {POLICY_RESCUE: rescue_levels, POLICY_K4: k4_levels}
    closure_by_policy = {
        policy: summarize_closure(levels, geometry)
        for policy, levels in levels_by_policy.items()
    }
    source_closures = {
        POLICY_RESCUE: stage_b0["exact_cost_a2_oracle"]["closure"],
        POLICY_K4: stage_b0["minimal_common_support_endpoints"]["k4"]["closure"],
    }
    expected_ledgers = {
        policy: _validate_closure_against_source(
            policy=policy,
            levels=levels_by_policy[policy],
            geometry=geometry,
            computed=closure_by_policy[policy],
            source=source_closures[policy],
            batch_size=args.inference_batch_size,
        )
        for policy in POLICIES
    }
    expected_image_ledgers = {
        policy: [
            _expected_image_ledger(image, batch_size=args.inference_batch_size)
            for image in closure_by_policy[policy]["per_image"]
        ]
        for policy in POLICIES
    }
    expected_images = stage_a["images"]
    if len(expected_images) != EXPECTED_FULL_TEST_LENGTH:
        raise AssertionError("Stage-A artifact is not the formal 20-image set")
    for image_index, full_image in enumerate(full_b1["images"]):
        expected_name = str(expected_images[image_index]["sample_name"])
        if (
            full_image.get("image_index") != image_index
            or full_image.get("sample_name") != expected_name
        ):
            raise AssertionError("full-B1 image order differs from Stage-A")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    model, cfg = load_model(args.baseline_checkpoint, args.seed)
    if tuple(cfg["window_size"]) != tuple(geometry["crop_size"]):
        raise AssertionError("model crop size differs from closure geometry")
    if len(cfg["labels"]) != NUM_CLASSES:
        raise AssertionError("model class count differs from Stage-A")
    loader, sample_names, full_test_length = build_test_loader(
        None, cfg["window_size"]
    )
    expected_names = [str(image["sample_name"]) for image in expected_images]
    if (
        full_test_length != EXPECTED_FULL_TEST_LENGTH
        or sample_names != expected_names
        or len(sample_names) != EXPECTED_FULL_TEST_LENGTH
    ):
        raise AssertionError("current WHU test set differs from the sealed artifacts")
    model.to(device)
    model.eval()

    selected_by_policy = {
        policy: [
            expected_phase_crop_ids(levels, geometry, image_index)
            for image_index in range(EXPECTED_FULL_TEST_LENGTH)
        ]
        for policy, levels in levels_by_policy.items()
    }
    expected_prediction_shas = {
        POLICY_RESCUE: [
            str(image["rescue"]["prediction_sha256"])
            for image in full_b1["images"]
        ],
        POLICY_K4: [
            str(image["prediction_sha256"]["k4"]) for image in expected_images
        ],
    }
    expected_confusions = {
        POLICY_RESCUE: [
            np.asarray(image["rescue"]["confusion"], dtype=np.int64)
            for image in full_b1["images"]
        ],
        POLICY_K4: [
            np.asarray(image["confusion"]["k4"], dtype=np.int64)
            for image in expected_images
        ],
    }
    expected_aggregate_confusions = {
        POLICY_RESCUE: np.asarray(full_b1["aggregate"]["confusion"], dtype=np.int64),
        POLICY_K4: np.asarray(
            stage_a["aggregate"]["endpoints"]["k4"]["full_image"]["confusion"],
            dtype=np.int64,
        ),
    }
    expected_streaming_shas = {
        POLICY_RESCUE: str(
            full_b1["aggregate"]["candidate_prediction_streaming_sha256"]
        ),
        POLICY_K4: str(stage_a["prediction_sha256"]["k4"]),
    }
    expected_k1_streaming_sha = str(stage_a["prediction_sha256"]["k1"])
    expected_label_streaming_sha = str(stage_a["prediction_sha256"]["label"])

    # One untimed first-image warm-up per measured policy, including a full
    # independent K1 prefix.  Validation occurs before the long repeats.
    warm_optical, warm_sar, warm_label = next(iter(loader))
    warm_target = np.ascontiguousarray(
        warm_label.numpy().astype(np.int64, copy=False)[0]
    )
    warmups: list[dict[str, Any]] = []
    for policy in WARMUP_ORDER:
        warm_result = _pipeline_execution(
            optical=warm_optical,
            sar=warm_sar,
            model=model,
            levels=levels_by_policy[policy],
            selected_ids=selected_by_policy[policy][0],
            geometry=geometry,
            image_index=0,
            device=device,
            batch_size=args.inference_batch_size,
            validate_count_maps=True,
        )
        validation = _validate_prediction(
            policy=policy,
            result=warm_result,
            target=warm_target,
            expected_policy_sha=expected_prediction_shas[policy][0],
            expected_k1_sha=str(expected_images[0]["prediction_sha256"]["k1"]),
            expected_policy_confusion=expected_confusions[policy][0],
        )
        _validate_exact_ledger(
            policy, warm_result["ledger"], expected_image_ledgers[policy][0]
        )
        warmups.append(
            {
                "policy": policy,
                "image_index": 0,
                "sample_name": expected_names[0],
                "completed": True,
                "synchronized": True,
                "timing_discarded": True,
                "prediction_sha256_equal": validation["prediction_sha256_equal"],
                "normal_k1_prediction_sha256_equal": validation[
                    "normal_k1_prediction_sha256_equal"
                ],
                "ledger": warm_result["ledger"],
            }
        )
        print(f"warmup policy={policy} image={expected_names[0]} PASS", flush=True)
        del warm_result, validation
    del warm_optical, warm_sar, warm_label, warm_target

    repeat_records: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    for repeat_index in range(args.paired_repeats):
        loader_iterator = iter(loader)
        policy_digests = {policy: hashlib.sha256() for policy in POLICIES}
        k1_digests = {policy: hashlib.sha256() for policy in POLICIES}
        label_digest = hashlib.sha256()
        aggregate_confusions = {
            policy: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
            for policy in POLICIES
        }
        ledgers = {policy: _empty_ledger() for policy in POLICIES}
        first_counts = {policy: 0 for policy in POLICIES}
        image_records: list[dict[str, Any]] = []
        policy_total_ms = {policy: 0.0 for policy in POLICIES}
        shared_input_total_ms = 0.0

        for image_index in range(EXPECTED_FULL_TEST_LENGTH):
            input_started_ns = time.perf_counter_ns()
            optical, sar, label_tensor = next(loader_iterator)
            shared_input_ms = float(
                (time.perf_counter_ns() - input_started_ns) / 1e6
            )
            target_batch = np.ascontiguousarray(
                label_tensor.numpy().astype(np.int64, copy=False)
            )
            if target_batch.ndim != 3 or target_batch.shape[0] != 1:
                raise AssertionError("WHU label must have shape [1,H,W]")
            target = target_batch[0]
            shape = tuple(int(value) for value in target.shape)
            if shape != tuple(geometry["image_shapes"][image_index]):
                raise AssertionError("latency image shape differs from Stage-A")
            if tuple(optical.shape[-2:]) != shape or tuple(sar.shape[-2:]) != shape:
                raise AssertionError("latency RGB/SAR/label shapes differ")
            shared_input_total_ms += shared_input_ms
            label_digest.update(target_batch.tobytes())

            order = paired_policy_order(repeat_index, image_index)
            first_counts[order[0]] += 1
            arm_records: dict[str, Any] = {}
            raw_results: dict[str, dict[str, Any]] = {}
            for policy in order:
                result = _pipeline_execution(
                    optical=optical,
                    sar=sar,
                    model=model,
                    levels=levels_by_policy[policy],
                    selected_ids=selected_by_policy[policy][image_index],
                    geometry=geometry,
                    image_index=image_index,
                    device=device,
                    batch_size=args.inference_batch_size,
                    validate_count_maps=False,
                )
                _validate_exact_ledger(
                    policy,
                    result["ledger"],
                    expected_image_ledgers[policy][image_index],
                )
                _add_ledger(ledgers[policy], result["ledger"])
                total_e2e_ms = (
                    shared_input_ms + float(result["timing"]["pipeline_e2e_ms"])
                )
                policy_total_ms[policy] += total_e2e_ms
                arm_records[policy] = {
                    **result["timing"],
                    "total_e2e_ms_including_shared_input": total_e2e_ms,
                    "ledger": result["ledger"],
                    "phase_audit": result["phase_audit"],
                }
                raw_results[policy] = result

            # Finish both timed arms before any prediction/hash/confusion audit.
            # This prevents the first arm from giving the second a
            # policy-dependent untimed validation and cooling interval.
            for policy in POLICIES:
                result = raw_results[policy]
                validation = _validate_prediction(
                    policy=policy,
                    result=result,
                    target=target,
                    expected_policy_sha=expected_prediction_shas[policy][image_index],
                    expected_k1_sha=str(
                        expected_images[image_index]["prediction_sha256"]["k1"]
                    ),
                    expected_policy_confusion=expected_confusions[policy][image_index],
                )
                policy_digests[policy].update(
                    np.ascontiguousarray(result["prediction"]).tobytes()
                )
                k1_digests[policy].update(
                    np.ascontiguousarray(result["normal_prediction"]).tobytes()
                )
                aggregate_confusions[policy] += validation["confusion"]
                arm_records[policy].update(
                    {
                        "prediction_sha256": validation["prediction_sha256"],
                        "prediction_sha256_equal": True,
                        "normal_k1_prediction_sha256": validation[
                            "normal_k1_prediction_sha256"
                        ],
                        "normal_k1_prediction_sha256_equal": True,
                        "confusion_equal_expected": True,
                    }
                )
                del validation
            del result, raw_results

            delta_ms = float(
                arm_records[POLICY_RESCUE][
                    "total_e2e_ms_including_shared_input"
                ]
                - arm_records[POLICY_K4]["total_e2e_ms_including_shared_input"]
            )
            image_records.append(
                {
                    "image_index": image_index,
                    "sample_name": sample_names[image_index],
                    "shape_hw": list(shape),
                    "execution_order": list(order),
                    "shared_input_load_ms": shared_input_ms,
                    "policies": arm_records,
                    "paired_rescue_minus_k4_total_e2e_ms": delta_ms,
                }
            )
            print(
                f"repeat={repeat_index + 1}/{args.paired_repeats} "
                f"image={image_index + 1}/{EXPECTED_FULL_TEST_LENGTH} "
                f"name={sample_names[image_index]} order={order[0]}>{order[1]} "
                f"rescue={arm_records[POLICY_RESCUE]['total_e2e_ms_including_shared_input']:.3f}ms "
                f"k4={arm_records[POLICY_K4]['total_e2e_ms_including_shared_input']:.3f}ms "
                f"delta={delta_ms:.3f}ms PASS",
                flush=True,
            )
            del optical, sar, label_tensor, target_batch, target, arm_records

        try:
            next(loader_iterator)
        except StopIteration:
            pass
        else:
            raise AssertionError("latency loader contains more than 20 images")
        if first_counts != {POLICY_RESCUE: 10, POLICY_K4: 10}:
            raise AssertionError("paired execution order is not balanced per repeat")
        for policy in POLICIES:
            _validate_exact_ledger(policy, ledgers[policy], expected_ledgers[policy])
            if policy_digests[policy].hexdigest() != expected_streaming_shas[policy]:
                raise AssertionError(f"{policy} streaming prediction SHA differs")
            if k1_digests[policy].hexdigest() != expected_k1_streaming_sha:
                raise AssertionError(f"{policy} K1 streaming prediction SHA differs")
            if not np.array_equal(
                aggregate_confusions[policy], expected_aggregate_confusions[policy]
            ):
                raise AssertionError(f"{policy} aggregate confusion differs")
        if label_digest.hexdigest() != expected_label_streaming_sha:
            raise AssertionError("latency label streaming SHA differs")
        repeat_delta_ms = float(
            policy_total_ms[POLICY_RESCUE] - policy_total_ms[POLICY_K4]
        )
        repeat_records.append(
            {
                "repeat_index": repeat_index,
                "images": image_records,
                "shared_input_load_total_ms": shared_input_total_ms,
                "first_policy_counts": first_counts,
                "policies": {
                    policy: {
                        "total_e2e_ms": float(policy_total_ms[policy]),
                        "ledger": ledgers[policy],
                        "prediction_streaming_sha256": policy_digests[
                            policy
                        ].hexdigest(),
                        "prediction_streaming_sha256_equal": True,
                        "k1_prediction_streaming_sha256": k1_digests[
                            policy
                        ].hexdigest(),
                        "k1_prediction_streaming_sha256_equal": True,
                        "aggregate_confusion_equal_expected": True,
                    }
                    for policy in POLICIES
                },
                "rescue_minus_k4_total_e2e_ms": repeat_delta_ms,
                "rescue_over_k4_total_e2e_ratio": float(
                    policy_total_ms[POLICY_RESCUE] / policy_total_ms[POLICY_K4]
                ),
                "correctness_and_workload_validation": "PASS",
            }
        )
        print(
            f"repeat={repeat_index + 1}/{args.paired_repeats} "
            f"rescue_total={policy_total_ms[POLICY_RESCUE]:.3f}ms "
            f"k4_total={policy_total_ms[POLICY_K4]:.3f}ms "
            f"delta={repeat_delta_ms:.3f}ms PASS",
            flush=True,
        )

    benchmark_wall_seconds = float(time.perf_counter() - benchmark_started)
    paired_deltas = [
        float(record["rescue_minus_k4_total_e2e_ms"])
        for record in repeat_records
    ]
    latency_decision = evaluate_primary_latency_gate(paired_deltas)
    latency_passed = bool(latency_decision["passed"])
    total_first_counts = {
        policy: sum(
            int(record["first_policy_counts"][policy]) for record in repeat_records
        )
        for policy in POLICIES
    }
    if total_first_counts != {POLICY_RESCUE: 30, POLICY_K4: 30}:
        raise AssertionError("aggregate paired execution order is not balanced")
    all_image_deltas = [
        float(image["paired_rescue_minus_k4_total_e2e_ms"])
        for repeat in repeat_records
        for image in repeat["images"]
    ]
    summary = {
        "paired_full_test_deltas_ms": paired_deltas,
        "median_paired_full_test_delta_ms": latency_decision[
            "median_paired_delta_ms"
        ],
        "policies": {
            policy: _policy_summary(repeat_records, policy) for policy in POLICIES
        },
        "per_image_paired_rescue_minus_k4_total_e2e_ms": _distribution(
            all_image_deltas
        ),
        "images_with_rescue_faster": int(
            np.count_nonzero(np.asarray(all_image_deltas) < 0.0)
        ),
        "paired_image_observations": len(all_image_deltas),
        "first_policy_counts": total_first_counts,
        "model_only_gpu_time": {
            "reported": False,
            "reason": (
                "per-forward CUDA events would instrument K4 more often than rescue "
                "and bias the primary; CUDA-event wall-span and unchanged "
                "primitive-core wall times are reported descriptively instead"
            ),
            "hard_gate": False,
        },
    }

    payload = {
        "status": "PASS",
        "status_meaning": (
            "formal paired latency runner completed with all correctness and workload "
            "anchors intact; scientific outcome is stored separately"
        ),
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "formal-stage-b-paired-same-primitive-latency-confirmation",
        "scientific_scope": (
            "official 20-image test-set method-selection audit of the frozen "
            "GT-informed rescue oracle versus support-pruned K4"
        ),
        "scientific_decision_evaluated": True,
        "latency_decision_evaluated": True,
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
                "sha256": b1b_sha,
            },
            "b1_full": {
                "path": str(args.b1_full_json.resolve()),
                "sha256": full_b1_sha,
                "non_latency_outcome": full_b1["non_latency_decision"]["outcome"],
                "candidate_prediction_streaming_sha256": full_b1["aggregate"][
                    "candidate_prediction_streaming_sha256"
                ],
            },
            "baseline_checkpoint": {
                "path": str(args.baseline_checkpoint.resolve()),
                "sha256": checkpoint_sha,
            },
        },
        "protocol": {
            "measured_policies": list(POLICIES),
            "primary_reference": POLICY_K4,
            "evaluated_images_per_repeat": EXPECTED_FULL_TEST_LENGTH,
            "paired_repeats": args.paired_repeats,
            "warmup": {
                "count_per_policy": 1,
                "image_index": 0,
                "sample_name": expected_names[0],
                "order": list(WARMUP_ORDER),
                "timing_discarded": True,
            },
            "paired_order_rule": (
                "rescue first iff (repeat_index + image_index) is even; otherwise "
                "K4 first; exactly 10/10 first positions per repeat and 30/30 total"
            ),
            "primary_expression": PRIMARY_EXPRESSION,
            "primary_unit": (
                "one paired observation is the sum of all 20 per-image total-E2E "
                "times within a repeat"
            ),
            "total_e2e_boundary": (
                "the DataLoader yield is measured once per paired image and allocated "
                "equally to both arms; label-only audit conversion is excluded; each "
                "arm then independently "
                "runs RGB/SAR H2D, full normal K1, shifted-canvas construction/H2D, "
                "the unchanged fixed-batch sparse primitive, D2H, inverse alignment, "
                "policy composition, and argmax"
            ),
            "excluded_from_timing": [
                "checkpoint/model/artifact loading",
                "warm-up",
                "precomputed frozen crop-plan construction",
                "prediction/logit hashing",
                "confusion and count-map validation",
                "console logging and JSON serialization",
                "future learned router cost because no deployable router exists",
            ],
            "correctness_validation": (
                "first-image warm-ups recheck analytic normal/phase count maps; "
                "formal repeats check exact crop-key order, batch shape, per-image "
                "and aggregate physical ledgers, K1/policy prediction SHA, streaming "
                "SHA, and confusion. Full 20-image analytic count maps were already "
                "sealed by the authorizing full-B1 artifact"
            ),
            "cuda_synchronization": (
                "synchronize immediately before wall/event start and after output "
                "materialization; the unchanged common primitive retains its internal "
                "synchronization"
            ),
            "phase_order": list(PHASE_NAMES),
            "phase_shifts_dy_dx": {
                name: list(geometry["phase_shifts"][name]) for name in PHASE_NAMES
            },
            "batch_size": args.inference_batch_size,
            "precision": "FP32; autocast disabled",
            "padding": (
                "per-image global phase pool with cyclic final-real-batch padding; "
                "padding outputs discarded but all physical samples counted"
            ),
            "independent_k1_prefix_per_arm": True,
            "no_inter_arm_empty_cache": True,
            "unchanged_operator_audit_overhead": (
                "the common live primitive retains its per-batch finite-output check "
                "and synchronization; the primary is audited operational pipeline "
                "latency, not a lean kernel-only or model-only measurement"
            ),
            "standalone_k1_k2_descriptive_curve": {
                "evaluated": False,
                "reason": (
                    "not part of the frozen K4-vs-rescue hard gate and deferred under "
                    "the current rapid method-selection scope; K1 is nevertheless "
                    "independently executed inside every measured arm"
                ),
            },
        },
        "expected_workload_per_repeat": expected_ledgers,
        "warmups": warmups,
        "paired_repeats": repeat_records,
        "summary": summary,
        "latency_decision": {
            **latency_decision,
            "full_b1_non_latency_passed": True,
            "full_stage_b_scientific_pass": latency_passed,
            "h2_confirmed": latency_passed,
            "meaning": (
                "the frozen GT-informed rescue passes the preregistered same-primitive "
                "latency gate as well as the already sealed B1 output/structure gates"
                if latency_passed
                else "the runner and outputs are valid, but the frozen rescue does not "
                "beat support-pruned K4 on the preregistered paired total-E2E primary"
            ),
        },
        "explicit_non_claims": [
            "no learned or deployable router",
            (
                "no router-inclusive deployable end-to-end latency because the "
                "frozen crop plan is precomputed"
            ),
            (
                "no separately instrumented model-only GPU-forward latency or "
                "GPU-kernel speed attribution; CUDA-event wall-span and "
                "primitive-core wall are not model-only"
            ),
            "no unseen-validation or cross-dataset latency generalization",
            "no hardware-independent speed claim",
            "no statistical-significance claim from three paired repeats",
            "no standalone K1/K2 latency curve in this rapid method-selection run",
            "no minimum speedup magnitude beyond the preregistered strict-negative gate",
        ],
        "runtime": {
            "benchmark_wall_seconds_including_untimed_validation": benchmark_wall_seconds,
            "role": "operational record; the primary uses paired timed regions only",
        },
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "live_common_sha256": live_common_sha,
            "replay_common_sha256": replay_common_sha,
            "full_b1_runner_sha256": full_b1["reproducibility"]["runner_sha256"],
            "seed": args.seed,
            "inference_batch_size": args.inference_batch_size,
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
        },
    }
    atomic_write_json(args.output_path, payload)
    print(f"latency_result={args.output_path.resolve()}", flush=True)
    print(f"latency_outcome={latency_decision['outcome']}", flush=True)
    print(
        f"latency_median_rescue_minus_k4_ms="
        f"{latency_decision['median_paired_delta_ms']:.6f}",
        flush=True,
    )
    print("latency_status=PASS", flush=True)


if __name__ == "__main__":
    main()
