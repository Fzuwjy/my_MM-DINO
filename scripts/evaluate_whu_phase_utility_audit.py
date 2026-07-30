"""Run the preregistered WHU Stage-A phase-utility audit.

This is a zero-training, post-aggregation spatial audit.  It computes the
sealed normal view and the three non-zero 8 px phase views, reconstructs K1,
matched-K2, legacy-K2, and K4, and asks whether the K4 gain is concentrated in
a small number of mutually exclusive sliding-window ownership cells.

The reported ``1 + 3q`` cost is deliberately labelled a proxy.  This runner
does not claim to execute sparse phase crops; an exact routed-window runner is
only warranted if the fixed Stage-A gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scripts.cache_whu_phase_teacher import (  # noqa: E402
    CONTROL_OFFSET,
    CROP_SIZE,
    PHASE_OFFSET,
    STRIDE,
    VALID_MARGIN,
    aligned_phase_crop,
    bounds_from_slice,
    build_phase_protocol,
    phase_common_slices,
)
from scripts.diagnose_whu_spatial_errors import (  # noqa: E402
    build_test_loader,
    file_sha256,
    load_model,
)
from scripts.evaluate_whu_phase_ensemble_2d import (  # noqa: E402
    load_two_phase_reference,
    prediction_from_aligned_score_sum,
)
from scripts.evaluate_whu_translation_consistency import (  # noqa: E402
    load_spatial_reference,
    translate_tensor,
)
from scripts.phase_utility_common import (  # noqa: E402
    CellScoreVector,
    aggregate_cell_assignment,
    binary_phase_cost,
    build_cell_ownership_bounds,
    cell_phase_statistics,
    deployment_score_vector,
    deterministic_random_indices,
    deterministic_top_q_indices,
    evaluate_phase_utility_gate,
    oracle_cell_miou_gain_scores,
    oracle_net_correct_scores,
    validate_cell_partition,
)
from scripts.spatial_diagnostics_common import (  # noqa: E402
    baseline_summary,
    build_spatial_region_masks,
    common_translation_slices,
    confusion_from_arrays,
    mean_iou_from_confusion,
    semantic_boundary_mask,
)
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.inference import slide_inference  # noqa: E402


NUM_CLASSES = 7
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "whu_phase_utility_stage_a"
Q_VALUES = (0.0, 0.10, 0.20, 1.0 / 3.0, 0.50, 1.0)
DECISION_Q = 1.0 / 3.0
RANDOM_REPLICATES = 1000
RANDOM_SEED = 20260730
SMALL_REGION = "component_area_le_256px2"
THIN_REGION = "component_thickness_le_4px"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "WHU Stage-A GT-informed ownership-cell phase-utility audit"
        )
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--spatial-diagnostic-json", type=Path, required=True)
    parser.add_argument("--two-phase-reference-json", type=Path, required=True)
    parser.add_argument("--four-phase-reference-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--miou-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    for path in (
        args.baseline_checkpoint,
        args.spatial_diagnostic_json,
        args.two_phase_reference_json,
        args.four_phase_reference_json,
    ):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.miou_tolerance < 0:
        parser.error("--miou-tolerance cannot be negative")
    return args


def load_four_phase_reference(path: Path) -> dict[str, Any]:
    """Load and validate the immutable full-test K4 result artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("scope") != "full-test":
        raise ValueError("four-phase reference must be a full-test PASS")
    if payload.get("evaluated_images") != payload.get("full_test_length"):
        raise ValueError("four-phase reference does not cover the full test set")
    protocol = payload.get("protocol", {})
    expected_primary = [list(shift) for shift in ((0, 0), (0, 8), (8, 0), (8, 8))]
    if protocol.get("primary") != expected_primary:
        raise ValueError("four-phase reference does not use the sealed 8px phases")
    if protocol.get("valid_margin") != VALID_MARGIN:
        raise ValueError("four-phase reference valid margin differs from protocol")
    if protocol.get("same_common_valid_region_for_primary_and_control") is not True:
        raise ValueError("four-phase reference lacks the sealed common support")
    for name, required in (
        (
            "baseline_validation",
            (
                "prediction_sha256_equal",
                "label_sha256_equal",
                "confusion_equal",
                "miou_within_tolerance",
            ),
        ),
        (
            "two_phase_validation",
            (
                "candidate_prediction_sha256_equal",
                "confusion_equal",
                "miou_within_tolerance",
            ),
        ),
    ):
        validation = payload.get(name, {})
        if not validation.get("checked") or not all(
            validation.get(key) is True for key in required
        ):
            raise ValueError(f"four-phase reference lacks strict {name}")
    primary = payload.get("aggregate", {}).get(str(PHASE_OFFSET), {})
    if not isinstance(primary.get("candidate"), dict) or not primary.get(
        "candidate_prediction_sha256"
    ):
        raise ValueError("four-phase reference lacks the primary K4 result")
    return payload


def slide_window_manifest(
    image_shape: tuple[int, int],
    crop_size: tuple[int, int] = CROP_SIZE,
    stride: tuple[int, int] = STRIDE,
) -> dict[str, Any]:
    """Reproduce the exact row-major crop coordinates of ``slide_inference``."""

    height, width = (int(value) for value in image_shape)
    crop_height, crop_width = (int(value) for value in crop_size)
    stride_height, stride_width = (int(value) for value in stride)
    if min(height, width, crop_height, crop_width, stride_height, stride_width) <= 0:
        raise ValueError("image, crop, and stride sizes must be positive")
    if height < crop_height or width < crop_width:
        raise ValueError("sealed WHU Stage-A requires both image axes >= crop size")

    row_count = max(height - crop_height + stride_height - 1, 0) // stride_height + 1
    column_count = max(width - crop_width + stride_width - 1, 0) // stride_width + 1
    windows: list[tuple[int, int, int, int]] = []
    row_starts: list[int] = []
    column_starts: list[int] = []
    for row in range(row_count):
        y0 = row * stride_height
        y1 = min(y0 + crop_height, height)
        y0 = max(y1 - crop_height, 0)
        row_starts.append(y0)
    for column in range(column_count):
        x0 = column * stride_width
        x1 = min(x0 + crop_width, width)
        x0 = max(x1 - crop_width, 0)
        column_starts.append(x0)
    for y0 in row_starts:
        for x0 in column_starts:
            windows.append(
                (y0, y0 + crop_height, x0, x0 + crop_width)
            )
    array = np.asarray(windows, dtype=np.int64)
    if len(array) != row_count * column_count:
        raise AssertionError("crop manifest cardinality mismatch")
    return {
        "windows": array,
        "row_count": row_count,
        "column_count": column_count,
        "row_starts": row_starts,
        "column_starts": column_starts,
    }


def _slice_bounds(region: tuple[slice, slice], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    y0, y1, y_step = region[0].indices(shape[0])
    x0, x1, x_step = region[1].indices(shape[1])
    if y_step != 1 or x_step != 1:
        raise ValueError("common support must use unit-stride slices")
    return y0, y1, x0, x1


def cell_means_from_common_map(
    cell_bounds: np.ndarray,
    common_values: np.ndarray,
    common_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Average a GT-free common-support map inside each ownership rectangle."""

    bounds = np.asarray(cell_bounds, dtype=np.int64)
    values = np.asarray(common_values)
    y_start, y_stop, x_start, x_stop = (int(value) for value in common_bounds)
    if values.ndim != 2 or values.shape != (y_stop - y_start, x_stop - x_start):
        raise ValueError("common value map shape differs from common bounds")
    if np.any(~np.isfinite(values)):
        raise ValueError("deployment score map contains non-finite values")
    result = np.full(len(bounds), -np.inf, dtype=np.float64)
    for index, (cell_y0, cell_y1, cell_x0, cell_x1) in enumerate(bounds):
        y0, y1 = max(int(cell_y0), y_start), min(int(cell_y1), y_stop)
        x0, x1 = max(int(cell_x0), x_start), min(int(cell_x1), x_stop)
        if y0 < y1 and x0 < x1:
            region = values[y0 - y_start : y1 - y_start, x0 - x_start : x1 - x_start]
            result[index] = float(np.mean(region, dtype=np.float64))
    return result


def _endpoint_summary(
    full_confusion: np.ndarray,
    common_aggregate: Mapping[str, Any],
    class_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "full_image": baseline_summary(full_confusion, class_names),
        "common_support": {
            "metrics": baseline_summary(common_aggregate["confusion"], class_names),
            "regions": common_aggregate["regions"],
        },
    }


def evaluate_binary_point(
    cell_stats: Sequence[dict[str, Any]],
    selected_indices: np.ndarray,
    outside_common_confusion: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Evaluate one post-aggregation K1/K4 cell assignment."""

    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(cell_stats)):
        raise ValueError("selected cell indices are out of range")
    levels = np.ones(len(cell_stats), dtype=np.int64)
    levels[selected] = 4
    common = aggregate_cell_assignment(
        cell_stats, levels, num_classes=NUM_CLASSES
    )
    full_confusion = np.asarray(outside_common_confusion, dtype=np.int64) + np.asarray(
        common["confusion"], dtype=np.int64
    )
    metrics = baseline_summary(full_confusion, class_names)
    cost = binary_phase_cost(len(selected), len(cell_stats))
    return {
        "requested_q": None,
        "selected_indices": selected.tolist(),
        "cost": cost,
        "full_image": metrics,
        "common_support": {
            "metrics": baseline_summary(common["confusion"], class_names),
            "regions": common["regions"],
        },
    }


def evaluate_score_curve(
    cell_stats: Sequence[dict[str, Any]],
    score: CellScoreVector,
    outside_common_confusion: np.ndarray,
    class_names: Sequence[str],
    *,
    deployment: bool,
) -> list[dict[str, Any]]:
    points = []
    for q in Q_VALUES:
        selected = deterministic_top_q_indices(score, q, deployment=deployment)
        point = evaluate_binary_point(
            cell_stats, selected, outside_common_confusion, class_names
        )
        point["requested_q"] = q
        points.append(point)
    return points


def restrict_score_to_geometry(
    score: CellScoreVector,
    eligible_mask: np.ndarray | Sequence[bool],
) -> CellScoreVector:
    """Make public common-support geometry identical across all rankers."""

    eligible = np.asarray(eligible_mask)
    if eligible.shape != score.values.shape or eligible.dtype != np.bool_:
        raise TypeError("eligible_mask must be a bool vector matching the score")
    values = score.values.copy()
    values[~eligible] = -np.inf
    return CellScoreVector(score.name, values, score.uses_ground_truth)


def random_control_curve(
    cell_stats: Sequence[dict[str, Any]],
    full_k1_confusion: np.ndarray,
    *,
    eligible_mask: np.ndarray | Sequence[bool] | None,
) -> list[dict[str, Any]]:
    """Return fixed-seed equal-count random mIoU distributions."""

    k1 = np.asarray(full_k1_confusion, dtype=np.int64)
    deltas = np.stack(
        [
            np.asarray(record["confusion"]["k4"], dtype=np.int64)
            - np.asarray(record["confusion"]["k1"], dtype=np.int64)
            for record in cell_stats
        ]
    )
    result = []
    for q in Q_VALUES:
        values = np.empty(RANDOM_REPLICATES, dtype=np.float64)
        selected_count = None
        for replicate in range(RANDOM_REPLICATES):
            selected = deterministic_random_indices(
                len(cell_stats),
                q,
                seed=RANDOM_SEED,
                replicate=replicate,
                eligible_mask=eligible_mask,
            )
            selected_count = len(selected)
            confusion = k1 + (
                deltas[selected].sum(axis=0) if len(selected) else 0
            )
            values[replicate] = mean_iou_from_confusion(confusion)
        if selected_count is None:
            raise AssertionError("random control did not execute")
        quantiles = np.quantile(values, (0.05, 0.50, 0.95))
        result.append(
            {
                "requested_q": q,
                "cost": binary_phase_cost(selected_count, len(cell_stats)),
                "replicates": RANDOM_REPLICATES,
                "seed": RANDOM_SEED,
                "sampling_pool": (
                    "geometry-eligible cells first; overflow from outer cells"
                    if eligible_mask is not None
                    else "all real cells"
                ),
                "geometry_eligible_cells": (
                    int(np.count_nonzero(eligible_mask))
                    if eligible_mask is not None
                    else None
                ),
                "miou": {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "p05": float(quantiles[0]),
                    "p50": float(quantiles[1]),
                    "p95": float(quantiles[2]),
                    "min": float(values.min()),
                    "max": float(values.max()),
                },
            }
        )
    return result


def finite_oracle_envelope(
    curves: Mapping[str, Sequence[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Choose the higher observed mIoU of two preregistered oracle rankings."""

    expected = ("net_correct", "singleton_global_miou")
    if tuple(curves) != expected:
        raise ValueError(f"oracle curves must be ordered as {expected}")
    if any(len(curves[name]) != len(Q_VALUES) for name in expected):
        raise ValueError("oracle curve length differs from Q_VALUES")
    envelope = []
    for point_index, q in enumerate(Q_VALUES):
        candidates = [(name, curves[name][point_index]) for name in expected]
        name, point = max(
            candidates,
            key=lambda item: (
                item[1]["full_image"]["miou"],
                -expected.index(item[0]),
            ),
        )
        copied = dict(point)
        copied["requested_q"] = q
        copied["source_ranking"] = name
        envelope.append(copied)
    return envelope


def _point_at_q(points: Sequence[dict[str, Any]], q: float) -> dict[str, Any]:
    matches = [point for point in points if math.isclose(point["requested_q"], q)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one curve point at q={q}")
    return matches[0]


def stage_a_decision(
    endpoints: Mapping[str, Any],
    oracle_envelope: Sequence[dict[str, Any]],
    random_curve: Sequence[dict[str, Any]],
    *,
    formal: bool,
) -> dict[str, Any]:
    """Apply the fixed q=1/3, proxy-cost <=2x Stage-A gate."""

    if not formal:
        return {
            "outcome": "NOT_EVALUATED_SUBSET",
            "passed": None,
            "reason": (
                "Subset smoke validates execution and internal geometry only; "
                "it cannot make the preregistered scientific decision."
            ),
        }
    mixed = _point_at_q(oracle_envelope, DECISION_Q)
    random = _point_at_q(random_curve, DECISION_Q)
    gate = evaluate_phase_utility_gate(
        k1_miou=endpoints["k1"]["full_image"]["miou"],
        k2_miou=endpoints["matched_k2"]["full_image"]["miou"],
        k4_miou=endpoints["k4"]["full_image"]["miou"],
        mixed_miou=mixed["full_image"]["miou"],
        forward_equivalent_cost=mixed["cost"]["forward_equivalent_cost"],
        k2_small_error_rate=endpoints["matched_k2"]["common_support"]["regions"][
            "small"
        ]["error_rate"],
        mixed_small_error_rate=mixed["common_support"]["regions"]["small"][
            "error_rate"
        ],
        k2_thin_error_rate=endpoints["matched_k2"]["common_support"]["regions"][
            "thin"
        ]["error_rate"],
        mixed_thin_error_rate=mixed["common_support"]["regions"]["thin"][
            "error_rate"
        ],
        random_p95_miou=random["miou"]["p95"],
    )
    gate["observed"]["decision_requested_q"] = DECISION_Q
    gate["observed"]["decision_source_ranking"] = mixed["source_ranking"]
    gate["observed"]["oracle_miou"] = mixed["full_image"]["miou"]
    gate["observed"]["random_control"] = (
        "geometry-aware equal-count random; all-window cost denominator"
    )
    gate["outcome"] = "GO_STAGE_B" if gate["passed"] else "NO_GO_STOP_ROUTE"
    gate["interpretation"] = (
        "Proceed to an exact routed-window simulator; Stage A itself is not a "
        "deployable sparse method."
        if gate["passed"]
        else "Stop the current Phase-on-Demand resource route; do not train a gate."
    )
    return gate


def _reference_checks(
    *,
    full_test: bool,
    spatial_reference: Mapping[str, Any],
    two_phase_reference: Mapping[str, Any],
    four_phase_reference: Mapping[str, Any],
    baseline_digest: str,
    label_digest: str,
    legacy_k2_digest: str,
    k4_digest: str,
    k1_confusion: np.ndarray,
    legacy_k2_confusion: np.ndarray,
    k4_confusion: np.ndarray,
    class_names: Sequence[str],
    tolerance: float,
) -> dict[str, Any]:
    validation: dict[str, Any] = {"checked": full_test}
    if not full_test:
        validation["reason"] = "immutable references contain only full-test digests"
        return validation

    expected_k2 = two_phase_reference["aggregate"][str(PHASE_OFFSET)]
    expected_k4 = four_phase_reference["aggregate"][str(PHASE_OFFSET)]
    k1_metrics = baseline_summary(k1_confusion, class_names)
    k2_metrics = baseline_summary(legacy_k2_confusion, class_names)
    k4_metrics = baseline_summary(k4_confusion, class_names)
    checks = {
        "k1_prediction_sha256_equal": (
            baseline_digest == spatial_reference["prediction_sha256"]
        ),
        "label_sha256_equal": label_digest == spatial_reference["label_sha256"],
        "k1_confusion_equal": (
            k1_confusion.tolist()
            == spatial_reference["aggregate"]["baseline"]["confusion"]
        ),
        "k1_miou_within_tolerance": abs(
            k1_metrics["miou"] - spatial_reference["aggregate"]["baseline"]["miou"]
        )
        <= tolerance,
        "legacy_k2_prediction_sha256_equal": (
            legacy_k2_digest == expected_k2["candidate_prediction_sha256"]
        ),
        "legacy_k2_confusion_equal": (
            legacy_k2_confusion.tolist() == expected_k2["candidate"]["confusion"]
        ),
        "legacy_k2_miou_within_tolerance": abs(
            k2_metrics["miou"] - expected_k2["candidate"]["miou"]
        )
        <= tolerance,
        "k4_prediction_sha256_equal": (
            k4_digest == expected_k4["candidate_prediction_sha256"]
        ),
        "k4_confusion_equal": (
            k4_confusion.tolist() == expected_k4["candidate"]["confusion"]
        ),
        "k4_miou_within_tolerance": abs(
            k4_metrics["miou"] - expected_k4["candidate"]["miou"]
        )
        <= tolerance,
    }
    validation.update(checks)
    if not all(checks.values()):
        raise AssertionError(f"Stage-A sealed reference reproduction failed: {checks}")
    return validation


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _finite_or_none(value: float | np.floating) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create one strict-JSON result without replacing an artifact."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WHU phase-utility evaluation")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    spatial_reference = load_spatial_reference(args.spatial_diagnostic_json)
    two_phase_reference = load_two_phase_reference(args.two_phase_reference_json)
    four_phase_reference = load_four_phase_reference(args.four_phase_reference_json)
    checkpoint_sha = file_sha256(args.baseline_checkpoint)
    for name, reference in (
        ("spatial", spatial_reference),
        ("two-phase", two_phase_reference),
        ("four-phase", four_phase_reference),
    ):
        if checkpoint_sha != reference["baseline_checkpoint_sha256"]:
            raise AssertionError(f"checkpoint differs from {name} reference")

    model, cfg = load_model(args.baseline_checkpoint, args.seed)
    protocol = build_phase_protocol()
    if tuple(cfg["window_size"]) != tuple(CROP_SIZE):
        raise AssertionError("model config crop size differs from sealed protocol")
    stride = tuple(int(size * 2 / 3) for size in cfg["window_size"])
    if stride != tuple(STRIDE):
        raise AssertionError("derived slide stride differs from sealed protocol")
    if len(cfg["labels"]) != NUM_CLASSES:
        raise AssertionError("class-name count differs from sealed protocol")
    loader, sample_names, full_test_length = build_test_loader(
        args.max_images, cfg["window_size"]
    )
    if any(
        reference["full_test_length"] != full_test_length
        for reference in (spatial_reference, two_phase_reference, four_phase_reference)
    ):
        raise AssertionError("current WHU test length differs from references")
    manifest = spatial_reference["images"][: len(sample_names)]
    if [record["sample_name"] for record in manifest] != sample_names:
        raise AssertionError("current WHU test order differs from spatial reference")

    phases = tuple(
        tuple(int(value) for value in shift)
        for shift in protocol["teacher_phases_dy_dx"]
    )
    expected_phases = ((0, 0), (0, PHASE_OFFSET), (PHASE_OFFSET, 0), (PHASE_OFFSET, PHASE_OFFSET))
    if phases != expected_phases:
        raise AssertionError("teacher phase protocol changed")

    device = torch.device(args.device)
    model.to(device)
    model.eval()
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"baseline_checkpoint_sha256={checkpoint_sha}")
    print(f"phases={phases}")
    print(f"evaluated_images={len(loader.dataset)}/{full_test_length}")
    print("scientific_scope=Stage-A post-aggregation optimistic cell audit")

    digests = {
        name: hashlib.sha256()
        for name in ("label", "k1", "legacy_k2", "matched_k2", "k4")
    }
    full_confusions = {
        name: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for name in ("k1", "legacy_k2", "matched_k2", "k4")
    }
    all_cell_stats: list[dict[str, Any]] = []
    score_values: dict[str, list[float]] = {
        "entropy": [],
        "negative_margin": [],
        "predicted_boundary_density": [],
    }
    image_records: list[dict[str, Any]] = []
    region_definitions: dict[str, Any] | None = None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for loader_position, ((optical, sar, label_tensor), expected_image) in enumerate(
            zip(loader, manifest, strict=True)
        ):
            label_full = np.ascontiguousarray(
                label_tensor.numpy().astype(np.int64, copy=False)
            )
            if label_full.ndim != 3 or label_full.shape[0] != 1:
                raise AssertionError("WHU label batch must have shape [1,H,W]")
            shape = tuple(int(value) for value in label_full.shape[-2:])
            if tuple(optical.shape[-2:]) != shape or tuple(sar.shape[-2:]) != shape:
                raise AssertionError("RGB, SAR, and label full-image shapes differ")
            crop_manifest = slide_window_manifest(shape)
            windows = crop_manifest["windows"]
            cell_bounds = build_cell_ownership_bounds(shape, windows)
            cell_areas = validate_cell_partition(cell_bounds, windows, shape)

            baseline_scores = slide_inference(
                optical.to(device),
                model,
                dsm=sar.to(device),
                n_output_channels=NUM_CLASSES,
                crop_size=cfg["window_size"],
                stride=stride,
                batch_size=args.inference_batch_size,
            )
            if tuple(baseline_scores.shape) != (1, NUM_CLASSES, *shape):
                raise AssertionError("normal slide score shape differs from protocol")
            if not torch.isfinite(baseline_scores).all():
                raise FloatingPointError("normal slide scores contain non-finite values")
            k1_prediction = np.ascontiguousarray(
                baseline_scores.argmax(dim=1).numpy().astype(np.int64, copy=False)
            )

            original_slice, shifted_slices = phase_common_slices(shape)
            common_bounds = _slice_bounds(original_slice, shape)
            x_original_slice, x_shifted_slices = common_translation_slices(
                shape,
                ((0, PHASE_OFFSET), (0, CONTROL_OFFSET)),
                VALID_MARGIN,
            )
            k4_sum = aligned_phase_crop(
                baseline_scores, (0, 0), original_slice, shifted_slices
            ).clone()
            matched_k2_sum = None
            legacy_k2_sum = None
            for shift in phases[1:]:
                dy, dx = shift
                shifted_scores = slide_inference(
                    translate_tensor(optical, dy, dx).to(device),
                    model,
                    dsm=translate_tensor(sar, dy, dx).to(device),
                    n_output_channels=NUM_CLASSES,
                    crop_size=cfg["window_size"],
                    stride=stride,
                    batch_size=args.inference_batch_size,
                )
                if tuple(shifted_scores.shape) != (1, NUM_CLASSES, *shape):
                    raise AssertionError(f"phase {shift} score shape differs from protocol")
                if not torch.isfinite(shifted_scores).all():
                    raise FloatingPointError(f"phase {shift} scores contain non-finite values")
                aligned = aligned_phase_crop(
                    shifted_scores, shift, original_slice, shifted_slices
                )
                k4_sum.add_(aligned)
                if shift == (0, PHASE_OFFSET):
                    matched_k2_sum = (
                        aligned_phase_crop(
                            baseline_scores, (0, 0), original_slice, shifted_slices
                        ).clone()
                        + aligned
                    )
                    legacy_k2_sum = (
                        baseline_scores[
                            0, :, x_original_slice[0], x_original_slice[1]
                        ].clone()
                        + shifted_scores[
                            0,
                            :,
                            x_shifted_slices[shift][0],
                            x_shifted_slices[shift][1],
                        ]
                    )
                del shifted_scores, aligned
            if matched_k2_sum is None or legacy_k2_sum is None:
                raise AssertionError("failed to construct the fixed x-phase K2")

            matched_k2_prediction = prediction_from_aligned_score_sum(
                k1_prediction, matched_k2_sum, original_slice
            )
            legacy_k2_prediction = prediction_from_aligned_score_sum(
                k1_prediction, legacy_k2_sum, x_original_slice
            )
            k4_prediction = prediction_from_aligned_score_sum(
                k1_prediction, k4_sum, original_slice
            )
            if not np.array_equal(
                matched_k2_prediction[0][original_slice],
                legacy_k2_prediction[0][original_slice],
            ):
                raise AssertionError("matched K2 and legacy K2 differ on K4 common support")

            predictions = {
                "k1": k1_prediction,
                "legacy_k2": legacy_k2_prediction,
                "matched_k2": matched_k2_prediction,
                "k4": k4_prediction,
            }
            image_confusions = {}
            for name, prediction in predictions.items():
                confusion = confusion_from_arrays(
                    prediction[0], label_full[0], NUM_CLASSES
                )
                full_confusions[name] += confusion
                image_confusions[name] = confusion
                digests[name].update(prediction.tobytes())
            digests["label"].update(label_full.tobytes())

            region_masks, current_region_definitions = build_spatial_region_masks(
                label_full[0],
                NUM_CLASSES,
                boundary_radii=(0, 1, 2, 4, 8),
                component_area_thresholds=(256, 1024, 4096),
                component_thickness_thresholds=(4, 8, 16),
                patch_size=16,
                union_boundary_radius=0,
                union_component_area=256,
                union_component_thickness=4,
            )
            if region_definitions is None:
                region_definitions = current_region_definitions
            elif region_definitions != current_region_definitions:
                raise AssertionError("spatial region definitions changed between images")
            common_mask = np.zeros(shape, dtype=bool)
            common_mask[original_slice] = True
            local_stats = cell_phase_statistics(
                cell_bounds,
                k1_prediction[0],
                matched_k2_prediction[0],
                k4_prediction[0],
                label_full[0],
                NUM_CLASSES,
                valid_mask=common_mask,
                small_mask=region_masks[SMALL_REGION],
                thin_mask=region_masks[THIN_REGION],
            )
            local_k1_levels = np.ones(len(local_stats), dtype=np.int64)
            local_k2_levels = np.full(len(local_stats), 2, dtype=np.int64)
            local_k4_levels = np.full(len(local_stats), 4, dtype=np.int64)
            for level_name, levels, prediction in (
                ("k1", local_k1_levels, k1_prediction),
                ("k2", local_k2_levels, matched_k2_prediction),
                ("k4", local_k4_levels, k4_prediction),
            ):
                aggregate = aggregate_cell_assignment(
                    local_stats, levels, num_classes=NUM_CLASSES
                )
                direct = confusion_from_arrays(
                    prediction[0], label_full[0], NUM_CLASSES, mask=common_mask
                )
                if not np.array_equal(aggregate["confusion"], direct):
                    raise AssertionError(
                        f"ownership cells do not reproduce {level_name} common confusion"
                    )

            common_logits = baseline_scores[
                0, :, original_slice[0], original_slice[1]
            ].float()
            probabilities = torch.softmax(common_logits, dim=0)
            entropy_map = (
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=0).numpy()
            )
            top_two = probabilities.topk(2, dim=0).values
            negative_margin_map = (-(top_two[0] - top_two[1])).numpy()
            boundary_map = semantic_boundary_mask(
                k1_prediction[0], NUM_CLASSES
            )[original_slice].astype(np.float32, copy=False)
            local_scores = {
                "entropy": cell_means_from_common_map(
                    cell_bounds, entropy_map, common_bounds
                ),
                "negative_margin": cell_means_from_common_map(
                    cell_bounds, negative_margin_map, common_bounds
                ),
                "predicted_boundary_density": cell_means_from_common_map(
                    cell_bounds, boundary_map, common_bounds
                ),
            }

            base_cell_index = len(all_cell_stats)
            for local_index, record in enumerate(local_stats):
                copied = dict(record)
                copied["cell_index"] = base_cell_index + local_index
                copied["image_index"] = loader_position
                copied["dataset_index"] = loader_position
                copied["sample_name"] = expected_image["sample_name"]
                copied["local_crop_id"] = local_index
                copied["window_yxyx"] = windows[local_index].tolist()
                copied["ownership_yxyx"] = cell_bounds[local_index].tolist()
                all_cell_stats.append(copied)
                for score_name in score_values:
                    score_values[score_name].append(
                        float(local_scores[score_name][local_index])
                    )

            image_records.append(
                {
                    "loader_position": loader_position,
                    "dataset_index": loader_position,
                    "sample_name": expected_image["sample_name"],
                    "full_shape_hw": list(shape),
                    "common_bounds": bounds_from_slice(shape, original_slice),
                    "crop_grid": {
                        "rows": crop_manifest["row_count"],
                        "columns": crop_manifest["column_count"],
                        "crop_count": len(windows),
                        "row_starts": crop_manifest["row_starts"],
                        "column_starts": crop_manifest["column_starts"],
                        "cell_geometric_area_sum": int(cell_areas.sum()),
                    },
                    "valid_pixels_full": int(
                        np.count_nonzero(
                            (label_full[0] >= 0) & (label_full[0] < NUM_CLASSES)
                        )
                    ),
                    "valid_pixels_common": int(
                        np.count_nonzero(
                            common_mask
                            & (label_full[0] >= 0)
                            & (label_full[0] < NUM_CLASSES)
                        )
                    ),
                    "prediction_sha256": {
                        name: hashlib.sha256(prediction.tobytes()).hexdigest()
                        for name, prediction in predictions.items()
                    },
                    "confusion": {
                        name: confusion.tolist()
                        for name, confusion in image_confusions.items()
                    },
                    "ownership_cell_count": len(local_stats),
                }
            )
            baseline_image_miou = baseline_summary(
                image_confusions["k1"], cfg["labels"]
            )["miou_percent"]
            k4_image_miou = baseline_summary(
                image_confusions["k4"], cfg["labels"]
            )["miou_percent"]
            print(
                f"image={loader_position + 1}/{len(loader.dataset)} "
                f"name={expected_image['sample_name']} cells={len(local_stats)} "
                f"k4_minus_k1={k4_image_miou - baseline_image_miou:+.4f}pp",
                flush=True,
            )
            del (
                optical,
                sar,
                label_tensor,
                label_full,
                baseline_scores,
                k1_prediction,
                legacy_k2_prediction,
                matched_k2_prediction,
                k4_prediction,
                matched_k2_sum,
                legacy_k2_sum,
                k4_sum,
                region_masks,
                common_mask,
                probabilities,
                top_two,
                entropy_map,
                negative_margin_map,
                boundary_map,
            )

    if not all_cell_stats:
        raise AssertionError("Stage-A produced no ownership cells")
    if region_definitions is None:
        raise AssertionError("Stage-A produced no spatial-region definitions")
    torch.cuda.synchronize(device)
    inference_elapsed = time.perf_counter() - started

    all_k1 = np.ones(len(all_cell_stats), dtype=np.int64)
    all_k2 = np.full(len(all_cell_stats), 2, dtype=np.int64)
    all_k4 = np.full(len(all_cell_stats), 4, dtype=np.int64)
    common_endpoints = {
        "k1": aggregate_cell_assignment(
            all_cell_stats, all_k1, num_classes=NUM_CLASSES
        ),
        "matched_k2": aggregate_cell_assignment(
            all_cell_stats, all_k2, num_classes=NUM_CLASSES
        ),
        "k4": aggregate_cell_assignment(
            all_cell_stats, all_k4, num_classes=NUM_CLASSES
        ),
    }
    outside_common_confusion = (
        full_confusions["k1"] - common_endpoints["k1"]["confusion"]
    )
    if np.any(outside_common_confusion < 0):
        raise AssertionError("common K1 confusion is not a subset of full K1")
    for name in ("k1", "matched_k2", "k4"):
        reconstructed = outside_common_confusion + common_endpoints[name]["confusion"]
        if not np.array_equal(reconstructed, full_confusions[name]):
            raise AssertionError(f"cell endpoint does not reproduce dense {name}")

    endpoints = {
        name: _endpoint_summary(
            full_confusions[name], common_endpoints[name], cfg["labels"]
        )
        for name in ("k1", "matched_k2", "k4")
    }
    endpoints["legacy_k2"] = {
        "full_image": baseline_summary(full_confusions["legacy_k2"], cfg["labels"]),
        "role": "sealed-reference reproduction only; not used by the Stage-A gate",
    }

    reference_validation = _reference_checks(
        full_test=args.max_images is None,
        spatial_reference=spatial_reference,
        two_phase_reference=two_phase_reference,
        four_phase_reference=four_phase_reference,
        baseline_digest=digests["k1"].hexdigest(),
        label_digest=digests["label"].hexdigest(),
        legacy_k2_digest=digests["legacy_k2"].hexdigest(),
        k4_digest=digests["k4"].hexdigest(),
        k1_confusion=full_confusions["k1"],
        legacy_k2_confusion=full_confusions["legacy_k2"],
        k4_confusion=full_confusions["k4"],
        class_names=cfg["labels"],
        tolerance=args.miou_tolerance,
    )

    geometry_eligible = np.isfinite(
        np.asarray(score_values["entropy"], dtype=np.float64)
    )
    for name in ("negative_margin", "predicted_boundary_density"):
        if not np.array_equal(
            geometry_eligible,
            np.isfinite(np.asarray(score_values[name], dtype=np.float64)),
        ):
            raise AssertionError("K1 score maps disagree on geometry eligibility")
    raw_scores = {
        "net_correct": oracle_net_correct_scores(all_cell_stats),
        "singleton_global_miou": oracle_cell_miou_gain_scores(
            all_cell_stats,
            num_classes=NUM_CLASSES,
            reference_confusion=full_confusions["k1"],
        ),
        "entropy": deployment_score_vector("k1_mean_entropy", score_values["entropy"]),
        "negative_margin": deployment_score_vector(
            "k1_negative_mean_top1_top2_margin", score_values["negative_margin"]
        ),
        "predicted_boundary_density": deployment_score_vector(
            "k1_predicted_boundary_density",
            score_values["predicted_boundary_density"],
        ),
    }
    scores = {
        name: restrict_score_to_geometry(score, geometry_eligible)
        for name, score in raw_scores.items()
    }
    oracle_curves = {
        name: evaluate_score_curve(
            all_cell_stats,
            scores[name],
            outside_common_confusion,
            cfg["labels"],
            deployment=False,
        )
        for name in ("net_correct", "singleton_global_miou")
    }
    oracle_envelope = finite_oracle_envelope(oracle_curves)
    simple_curves = {
        name: evaluate_score_curve(
            all_cell_stats,
            scores[name],
            outside_common_confusion,
            cfg["labels"],
            deployment=True,
        )
        for name in ("entropy", "negative_margin", "predicted_boundary_density")
    }
    geometry_random_curve = random_control_curve(
        all_cell_stats,
        full_confusions["k1"],
        eligible_mask=geometry_eligible,
    )
    naive_random_curve = random_control_curve(
        all_cell_stats,
        full_confusions["k1"],
        eligible_mask=None,
    )
    decision = stage_a_decision(
        endpoints,
        oracle_envelope,
        geometry_random_curve,
        formal=args.max_images is None,
    )
    total_elapsed = time.perf_counter() - started

    compact_cells = []
    singleton_values = scores["singleton_global_miou"].values
    for index, record in enumerate(all_cell_stats):
        transition = record["transitions"]["k1_to_k4"]
        compact_cells.append(
            {
                "cell_index": index,
                "image_index": record["image_index"],
                "dataset_index": record["dataset_index"],
                "sample_name": record["sample_name"],
                "local_crop_id": record["local_crop_id"],
                "window_yxyx": record["window_yxyx"],
                "ownership_yxyx": record["ownership_yxyx"],
                "geometry_eligible": bool(geometry_eligible[index]),
                "valid_pixels_common": record["regions"]["all"]["pixels"],
                "confusion": record["confusion"],
                "k1_to_k4": transition,
                "scores": {
                    "oracle_singleton_global_miou_gain": _finite_or_none(
                        singleton_values[index]
                    ),
                    "k1_entropy": _finite_or_none(score_values["entropy"][index]),
                    "k1_negative_margin": _finite_or_none(
                        score_values["negative_margin"][index]
                    ),
                    "k1_predicted_boundary_density": _finite_or_none(
                        score_values["predicted_boundary_density"][index]
                    ),
                },
            }
        )

    output = {
        "status": "PASS",
        "status_meaning": (
            "Execution, geometry, and endpoint checks passed. Scientific GO/NO-GO "
            "is reported separately and is absent for subset smoke."
        ),
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "full-test" if args.max_images is None else "subset-smoke",
        "scientific_scope": (
            "GT-informed post-aggregation ownership-cell upper-bound audit. It "
            "tests spatial concentration (H1), not exact sparse executability (H2)."
        ),
        "full_test_length": full_test_length,
        "evaluated_images": len(loader.dataset),
        "selection_mode": "test-prefix",
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": checkpoint_sha,
        "references": {
            "spatial_diagnostic": {
                "path": str(args.spatial_diagnostic_json.resolve()),
                "sha256": file_sha256(args.spatial_diagnostic_json),
            },
            "two_phase": {
                "path": str(args.two_phase_reference_json.resolve()),
                "sha256": file_sha256(args.two_phase_reference_json),
            },
            "four_phase": {
                "path": str(args.four_phase_reference_json.resolve()),
                "sha256": file_sha256(args.four_phase_reference_json),
            },
        },
        "reference_validation": reference_validation,
        "protocol": {
            **protocol,
            "stage_a_cell_ownership": (
                "one midpoint-owned rectangle per real row-major 512/341 window"
            ),
            "stage_a_action_space": "binary K1/K4",
            "proxy_cost": "1 + 3 * selected_cell_fraction",
            "proxy_cost_warning": (
                "post-aggregation optimistic proxy, not executed crop forwards or latency"
            ),
            "q_values": list(Q_VALUES),
            "decision_q": DECISION_Q,
            "random_replicates": RANDOM_REPLICATES,
            "random_seed": RANDOM_SEED,
            "geometry_eligible_cells": int(np.count_nonzero(geometry_eligible)),
            "random_gate_control": (
                "sample geometry-eligible cells first; selected count and cost "
                "denominator retain all real windows"
            ),
            "naive_all_cell_random_role": "descriptive only; never used by the gate",
            "matched_k2": (
                "normal+x8 logits on exactly the sealed K4 common support"
            ),
        },
        "region_reporting": {
            "small_region": SMALL_REGION,
            "thin_region": THIN_REGION,
            "support": "valid labels inside the sealed K4 common support",
            "masks_may_overlap": True,
            "definitions": region_definitions,
        },
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "common_module_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_utility_common.py"
            ),
            "seed": args.seed,
            "inference_batch_size": args.inference_batch_size,
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
        "class_names": list(cfg["labels"]),
        "prediction_sha256": {
            name: digest.hexdigest() for name, digest in digests.items()
        },
        "images": image_records,
        "aggregate": {
            "total_cells": len(all_cell_stats),
            "endpoints": endpoints,
            "oracle_rankings": oracle_curves,
            "finite_oracle_envelope": oracle_envelope,
            "simple_k1_scores": simple_curves,
            "geometry_aware_equal_count_random": geometry_random_curve,
            "naive_all_cell_random_descriptive": naive_random_curve,
        },
        "stage_a_decision": decision,
        "cells": compact_cells,
        "runtime": {
            "inference_and_cell_statistics_seconds": inference_elapsed,
            "total_seconds": total_elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        },
    }
    atomic_write_json(args.output_path, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "scope": output["scope"],
                "evaluated_images": output["evaluated_images"],
                "reference_validation": reference_validation,
                "k1_miou_percent": endpoints["k1"]["full_image"]["miou_percent"],
                "matched_k2_miou_percent": endpoints["matched_k2"]["full_image"][
                    "miou_percent"
                ],
                "k4_miou_percent": endpoints["k4"]["full_image"]["miou_percent"],
                "stage_a_decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"phase_utility_result={args.output_path.resolve()}")
    print("phase_utility_status=PASS")


if __name__ == "__main__":
    main()
