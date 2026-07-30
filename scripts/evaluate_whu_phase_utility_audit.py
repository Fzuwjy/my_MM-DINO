"""Run the preregistered WHU Stage-A phase-utility audit.

This is a zero-training, post-aggregation spatial audit.  It computes the
sealed normal view and the three non-zero 8 px phase views, reconstructs K1,
fixed x-K2, descriptive y-K2, legacy-K2, and K4, and asks whether the dense
phase gain is concentrated in mutually exclusive sliding-window ownership
cells.  Two fixed hypotheses are tested: binary K1/K4 routing (A1), and a
nested K1 -> x-K2 -> K4 route (A2).  Both must survive global-pooled and
per-image-capped proxy budgets before Stage B is authorized.

All costs in this file are deliberately labelled post-aggregation proxies.
This runner does not claim to execute sparse phase crops; an exact
routed-window simulator is only warranted if the fixed Stage-A gate passes.
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
    binary_greedy_oracle,
    binary_group_budget_oracle,
    binary_phase_cost,
    build_cell_ownership_bounds,
    cell_phase_statistics,
    deployment_score_vector,
    deterministic_random_indices,
    deterministic_top_q_indices,
    evaluate_phase_utility_gate,
    hierarchical_greedy_oracle,
    hierarchical_group_budget_oracle,
    hierarchical_phase_cost,
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
SCHEMA_VERSION = 2
ARTIFACT_TYPE = "whu_phase_utility_stage_a"
Q_VALUES = (0.0, 0.10, 0.20, 1.0 / 3.0, 0.50, 1.0)
# Retained only for the legacy fixed-ranking regression helper below.
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


def grouped_top_q_indices(
    score: CellScoreVector,
    q: float,
    group_ids: np.ndarray | Sequence[int],
    *,
    deployment: bool,
) -> np.ndarray:
    """Apply the same fractional budget independently inside every image."""

    groups = np.asarray(group_ids)
    if groups.shape != score.values.shape or not np.issubdtype(
        groups.dtype, np.integer
    ):
        raise TypeError("group_ids must be an integer vector matching the score")
    selected_parts = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group).astype(np.int64, copy=False)
        local_score = CellScoreVector(
            score.name,
            score.values[indices],
            score.uses_ground_truth,
        )
        local_selected = deterministic_top_q_indices(
            local_score, q, deployment=deployment
        )
        selected_parts.append(indices[local_selected])
    if not selected_parts:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(selected_parts).astype(np.int64, copy=False))


def grouped_singleton_miou_scores(
    cell_stats: Sequence[dict[str, Any]],
    group_ids: np.ndarray | Sequence[int],
    reference_confusion_by_group: Mapping[int, np.ndarray],
) -> CellScoreVector:
    """Compute singleton mIoU scores against each image's full K1 confusion."""

    groups = np.asarray(group_ids)
    if groups.shape != (len(cell_stats),) or not np.issubdtype(
        groups.dtype, np.integer
    ):
        raise TypeError("group_ids must be an integer vector matching cell_stats")
    values = np.empty(len(cell_stats), dtype=np.float64)
    for group in np.unique(groups):
        group_value = int(group)
        if group_value not in reference_confusion_by_group:
            raise ValueError(f"missing reference confusion for group {group_value}")
        indices = np.flatnonzero(groups == group).astype(np.int64, copy=False)
        local_records = []
        for local_index, global_index in enumerate(indices):
            record = dict(cell_stats[int(global_index)])
            record["cell_index"] = local_index
            local_records.append(record)
        local_score = oracle_cell_miou_gain_scores(
            local_records,
            num_classes=NUM_CLASSES,
            reference_confusion=reference_confusion_by_group[group_value],
        )
        values[indices] = local_score.values
    return CellScoreVector(
        "oracle_k1_to_k4_singleton_per_image_miou_gain",
        values,
        True,
    )


def evaluate_grouped_score_curve(
    cell_stats: Sequence[dict[str, Any]],
    score: CellScoreVector,
    group_ids: np.ndarray | Sequence[int],
    outside_common_confusion: np.ndarray,
    class_names: Sequence[str],
    *,
    deployment: bool,
) -> list[dict[str, Any]]:
    points = []
    for q in Q_VALUES:
        selected = grouped_top_q_indices(
            score, q, group_ids, deployment=deployment
        )
        point = evaluate_binary_point(
            cell_stats, selected, outside_common_confusion, class_names
        )
        point["requested_q"] = q
        point["budget_scope"] = "per-image"
        points.append(point)
    return points


def random_control_curve(
    cell_stats: Sequence[dict[str, Any]],
    full_k1_confusion: np.ndarray,
    *,
    eligible_mask: np.ndarray | Sequence[bool] | None,
    group_ids: np.ndarray | Sequence[int] | None = None,
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
    groups = None if group_ids is None else np.asarray(group_ids)
    if groups is not None and (
        groups.shape != (len(cell_stats),)
        or not np.issubdtype(groups.dtype, np.integer)
    ):
        raise TypeError("group_ids must be an integer vector matching cell_stats")
    eligible_array = (
        None if eligible_mask is None else np.asarray(eligible_mask, dtype=bool)
    )
    for q in Q_VALUES:
        values = np.empty(RANDOM_REPLICATES, dtype=np.float64)
        selected_count = None
        for replicate in range(RANDOM_REPLICATES):
            if groups is None:
                selected = deterministic_random_indices(
                    len(cell_stats),
                    q,
                    seed=RANDOM_SEED,
                    replicate=replicate,
                    eligible_mask=eligible_array,
                )
            else:
                selected_parts = []
                unique_groups = np.unique(groups)
                for group_position, group in enumerate(unique_groups):
                    indices = np.flatnonzero(groups == group).astype(
                        np.int64, copy=False
                    )
                    local_eligible = (
                        None
                        if eligible_array is None
                        else eligible_array[indices]
                    )
                    local = deterministic_random_indices(
                        len(indices),
                        q,
                        seed=RANDOM_SEED,
                        replicate=replicate * len(unique_groups) + group_position,
                        eligible_mask=local_eligible,
                    )
                    selected_parts.append(indices[local])
                selected = np.sort(
                    np.concatenate(selected_parts).astype(np.int64, copy=False)
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
                "budget_scope": "global" if groups is None else "per-image",
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


def _exact_count_random_indices(
    total_cells: int,
    selected_count: int,
    *,
    seed: int,
    replicate: int,
    eligible_mask: np.ndarray | None,
) -> np.ndarray:
    if total_cells == 0 and selected_count == 0:
        return np.empty(0, dtype=np.int64)
    if selected_count < 0 or selected_count > total_cells:
        raise ValueError("selected_count must lie inside the random pool")
    q = 1.0 if selected_count == total_cells else selected_count / total_cells
    selected = deterministic_random_indices(
        total_cells,
        q,
        seed=seed,
        replicate=replicate,
        eligible_mask=eligible_mask,
    )
    if len(selected) != selected_count:
        raise AssertionError("exact-count random selection changed cardinality")
    return selected


def matched_action_random_control(
    cell_stats: Sequence[dict[str, Any]],
    full_k1_confusion: np.ndarray,
    target_levels: np.ndarray | Sequence[int],
    eligible_mask: np.ndarray | Sequence[bool],
    group_ids: np.ndarray | Sequence[int],
    *,
    route_kind: str,
) -> dict[str, Any]:
    """Match K2-only/K4 counts globally or per image for one oracle policy."""

    levels = np.asarray(target_levels, dtype=np.int64)
    eligible = np.asarray(eligible_mask)
    groups = np.asarray(group_ids)
    if levels.shape != (len(cell_stats),) or np.any(~np.isin(levels, (1, 2, 4))):
        raise ValueError("target_levels must contain one K1/K2/K4 level per cell")
    if eligible.shape != levels.shape or eligible.dtype != np.bool_:
        raise TypeError("eligible_mask must be a bool vector matching target_levels")
    if groups.shape != levels.shape or not np.issubdtype(groups.dtype, np.integer):
        raise TypeError("group_ids must be an integer vector matching target_levels")
    if route_kind not in ("a1_binary", "a2_hierarchical_x"):
        raise ValueError("route_kind must identify the fixed A1 or A2 action space")
    if route_kind == "a1_binary" and np.any(levels == 2):
        raise ValueError("A1 random control cannot receive K2 target levels")

    unique_groups = np.unique(groups)
    target_counts = {}
    for group in unique_groups:
        group_levels = levels[groups == group]
        target_counts[int(group)] = {
            "k2_only": int(np.count_nonzero(group_levels == 2)),
            "k4": int(np.count_nonzero(group_levels == 4)),
        }
    common_k1 = aggregate_cell_assignment(
        cell_stats,
        np.ones(len(cell_stats), dtype=np.int64),
        num_classes=NUM_CLASSES,
    )["confusion"]
    full_k1 = np.asarray(full_k1_confusion, dtype=np.int64)
    outside = full_k1 - common_k1
    if np.any(outside < 0):
        raise ValueError("cell K1 confusion is not contained in full K1 confusion")
    confusion_by_level = {
        level: np.stack(
            [
                np.asarray(record["confusion"][f"k{level}"], dtype=np.int64)
                for record in cell_stats
            ],
            axis=0,
        )
        for level in (1, 2, 4)
    }
    delta_k1_to_k2 = confusion_by_level[2] - confusion_by_level[1]
    delta_k2_to_k4 = confusion_by_level[4] - confusion_by_level[2]
    values = np.empty(RANDOM_REPLICATES, dtype=np.float64)
    for replicate in range(RANDOM_REPLICATES):
        promoted_parts = []
        k4_parts = []
        for group_position, group in enumerate(unique_groups):
            indices = np.flatnonzero(groups == group).astype(np.int64, copy=False)
            counts = target_counts[int(group)]
            promoted_count = counts["k2_only"] + counts["k4"]
            eligible_count = int(np.count_nonzero(eligible[indices]))
            if promoted_count > eligible_count:
                raise ValueError(
                    "matched formal random cannot reproduce target action counts "
                    "inside the public geometry-eligible pool"
                )
            promoted_local = _exact_count_random_indices(
                len(indices),
                promoted_count,
                seed=RANDOM_SEED,
                replicate=replicate * len(unique_groups) * 2 + group_position * 2,
                eligible_mask=eligible[indices],
            )
            promoted_global = indices[promoted_local]
            promoted_parts.append(promoted_global)
            k4_within_promoted = _exact_count_random_indices(
                promoted_count,
                counts["k4"],
                seed=RANDOM_SEED + 1,
                replicate=(
                    replicate * len(unique_groups) * 2 + group_position * 2 + 1
                ),
                eligible_mask=None,
            )
            k4_parts.append(promoted_global[k4_within_promoted])
        promoted = np.concatenate(promoted_parts).astype(np.int64, copy=False)
        k4 = np.concatenate(k4_parts).astype(np.int64, copy=False)
        confusion = full_k1.copy()
        if len(promoted):
            confusion += delta_k1_to_k2[promoted].sum(axis=0, dtype=np.int64)
        if len(k4):
            confusion += delta_k2_to_k4[k4].sum(axis=0, dtype=np.int64)
        values[replicate] = mean_iou_from_confusion(confusion)
    quantiles = np.quantile(values, (0.05, 0.50, 0.95))
    k2_or_k4 = int(np.count_nonzero(levels != 1))
    k4_count = int(np.count_nonzero(levels == 4))
    binary_target = route_kind == "a1_binary"
    cost = (
        binary_phase_cost(k4_count, len(cell_stats))
        if binary_target
        else hierarchical_phase_cost(k2_or_k4, k4_count, len(cell_stats))
    )
    return {
        "replicates": RANDOM_REPLICATES,
        "seed": RANDOM_SEED,
        "route_kind": route_kind,
        "matching": (
            "per-group K4 counts"
            if binary_target
            else "per-group K2-only/K4 counts with K4 nested inside K2"
        ),
        "target_counts_by_group": target_counts,
        "cost": cost,
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
    """Evaluate the legacy fixed-q A1 fixture used by regression tests.

    Schema-v2 ``main`` never calls this helper: formal decisions use dynamic
    A1/A2 greedy points and :func:`arbitrate_stage_a_routes`.
    """

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
    return route_decision_from_point(
        endpoints,
        mixed,
        random,
        formal=True,
        route_name="A1_binary_k1_k4",
        budget_scope=mixed.get("budget_scope", "global"),
    )


def route_decision_from_point(
    endpoints: Mapping[str, Any],
    mixed: Mapping[str, Any],
    random: Mapping[str, Any],
    *,
    formal: bool,
    route_name: str,
    budget_scope: str,
) -> dict[str, Any]:
    """Apply the common scientific-feasibility gate to one fixed policy."""

    if not formal:
        return {
            "outcome": "NOT_EVALUATED_SUBSET",
            "passed": None,
            "route_name": route_name,
            "budget_scope": budget_scope,
            "reason": (
                "Subset smoke validates execution and internal geometry only; "
                "it cannot make the preregistered scientific decision."
            ),
        }
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
    gate["route_name"] = route_name
    gate["budget_scope"] = budget_scope
    gate["observed"]["decision_requested_q"] = mixed.get("requested_q")
    gate["observed"]["decision_requested_cost"] = mixed.get("requested_cost")
    gate["observed"]["decision_source_ranking"] = mixed.get(
        "source_ranking", "constructive_greedy"
    )
    gate["observed"]["oracle_miou"] = mixed["full_image"]["miou"]
    gate["observed"]["random_control"] = (
        "geometry-aware random matched to the policy's action counts, hierarchy "
        "when applicable, and budget scope; all-window cost denominator"
    )
    gate["outcome"] = "PASS_FEASIBILITY" if gate["passed"] else "FAIL_FEASIBILITY"
    gate["interpretation"] = (
        "This policy passes one Stage-A feasibility gate; overall A1/A2 and "
        "global/per-image arbitration still applies."
        if gate["passed"]
        else "This policy fails its Stage-A feasibility gate."
    )
    return gate


def greedy_snapshot_point(
    snapshot: Mapping[str, Any],
    class_names: Sequence[str],
    *,
    route_name: str,
    budget_scope: str,
    source_ranking: str,
) -> dict[str, Any]:
    """Convert one common-module greedy snapshot into the runner schema."""

    aggregate = snapshot["aggregate"]
    cost = snapshot["cost"]
    total_cells = int(cost["total_cells"])
    if total_cells <= 0:
        raise ValueError("greedy snapshot must cover at least one cell")
    requested_extra = int(snapshot["requested_extra_proxy_cost"])
    actual_extra = int(snapshot["actual_extra_proxy_cost"])
    requested_cost = 1.0 + requested_extra / total_cells
    if actual_extra > requested_extra:
        raise AssertionError("greedy snapshot exceeded its requested proxy budget")
    return {
        "route_name": route_name,
        "budget_scope": budget_scope,
        "source_ranking": source_ranking,
        "requested_cost": float(requested_cost),
        "requested_extra_proxy_cost": requested_extra,
        "actual_extra_proxy_cost": actual_extra,
        "chosen_action_count": int(snapshot["chosen_action_count"]),
        "selected_counts": dict(snapshot["selected_counts"]),
        "levels_by_cell": np.asarray(
            snapshot["levels_by_cell"], dtype=np.int64
        ),
        "cost": dict(cost),
        "full_image": baseline_summary(
            np.asarray(aggregate["full_confusion"], dtype=np.int64), class_names
        ),
        "common_support": {
            "metrics": baseline_summary(
                np.asarray(aggregate["common_support"]["confusion"], dtype=np.int64),
                class_names,
            ),
            "regions": aggregate["common_support"]["regions"],
        },
    }


def arbitrate_stage_a_routes(
    route_decisions: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    formal: bool,
) -> dict[str, Any]:
    """Apply the frozen A2-first, both-budget Stage-A arbitration rule."""

    route_order = ("a2_hierarchical_x", "a1_binary")
    for route in route_order:
        if route not in route_decisions:
            raise ValueError(f"missing Stage-A route decision: {route}")
        if set(route_decisions[route]) != {"global", "per_image"}:
            raise ValueError(f"route {route} must contain global and per_image gates")
    pass_matrix = {
        route: {
            scope: route_decisions[route][scope].get("passed")
            for scope in ("global", "per_image")
        }
        for route in route_order
    }
    if not formal:
        return {
            "outcome": "NOT_EVALUATED_SUBSET",
            "passed": None,
            "stage_b_authorized": False,
            "selected_route": None,
            "route_pass_matrix": pass_matrix,
            "reason": (
                "Subset smoke validates execution only; A1/A2 scientific "
                "arbitration requires the sealed full test set."
            ),
        }

    for route in route_order:
        if all(pass_matrix[route][scope] is True for scope in ("global", "per_image")):
            return {
                "outcome": (
                    "GO_STAGE_B_A2_HIERARCHICAL_X"
                    if route == "a2_hierarchical_x"
                    else "GO_STAGE_B_A1_BINARY"
                ),
                "passed": True,
                "stage_b_authorized": True,
                "selected_route": route,
                "route_pass_matrix": pass_matrix,
                "reason": (
                    "The selected fixed action space passes both the optimistic "
                    "global-pooled and deployability-oriented per-image-capped "
                    "scientific-feasibility gates."
                ),
            }

    pooled_only = [
        route
        for route in route_order
        if pass_matrix[route]["global"] is True
        and pass_matrix[route]["per_image"] is not True
    ]
    if pooled_only:
        return {
            "outcome": "POOLED_ONLY_NO_WINDOW_STAGE_B",
            "passed": False,
            "stage_b_authorized": False,
            "selected_route": None,
            "pooled_only_routes": pooled_only,
            "route_pass_matrix": pass_matrix,
            "reason": (
                "At least one route works only when proxy compute may move across "
                "images. That is evidence for dataset/image-level heterogeneity, "
                "not for the current per-image window-routing constraint."
            ),
        }
    return {
        "outcome": "NO_GO_STOP_CURRENT_PHASE_ON_DEMAND_ROUTE",
        "passed": False,
        "stage_b_authorized": False,
        "selected_route": None,
        "route_pass_matrix": pass_matrix,
        "reason": (
            "Neither preregistered action space passes both Stage-A budget "
            "scopes. Stop this ownership-cell/fixed-phase/2x route; this does "
            "not prove every dynamic phase method impossible."
        ),
    }


def _reindexed_cell_subset(
    cell_stats: Sequence[dict[str, Any]], indices: np.ndarray
) -> list[dict[str, Any]]:
    result = []
    for local_index, global_index in enumerate(indices):
        record = dict(cell_stats[int(global_index)])
        record["cell_index"] = local_index
        result.append(record)
    return result


def _error_count(confusion: np.ndarray) -> int:
    matrix = np.asarray(confusion, dtype=np.int64)
    return int(matrix.sum() - np.trace(matrix))


def _region_rate_from_totals(totals: Mapping[str, int]) -> float | None:
    pixels = int(totals["pixels"])
    errors = int(totals["errors"])
    return float(errors / pixels) if pixels else None


def fixed_policy_stability(
    cell_stats: Sequence[dict[str, Any]],
    levels_by_cell: np.ndarray | Sequence[int],
    group_ids: np.ndarray | Sequence[int],
    full_confusions_by_group: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Describe per-image and fixed-policy LOO behavior without adding a gate."""

    levels = np.asarray(levels_by_cell, dtype=np.int64)
    groups = np.asarray(group_ids)
    if levels.shape != (len(cell_stats),) or np.any(~np.isin(levels, (1, 2, 4))):
        raise ValueError("levels_by_cell must contain one K1/K2/K4 level per cell")
    if groups.shape != levels.shape or not np.issubdtype(groups.dtype, np.integer):
        raise TypeError("group_ids must be an integer vector matching levels")

    group_order = [int(value) for value in dict.fromkeys(groups.tolist())]
    per_image = []
    internal = []
    for group in group_order:
        if group not in full_confusions_by_group:
            raise ValueError(f"missing full confusions for image {group}")
        indices = np.flatnonzero(groups == group).astype(np.int64, copy=False)
        local_records = _reindexed_cell_subset(cell_stats, indices)
        local_levels = levels[indices]
        assignments = {
            "k1": np.ones(len(indices), dtype=np.int64),
            "k2": np.full(len(indices), 2, dtype=np.int64),
            "k4": np.full(len(indices), 4, dtype=np.int64),
            "route": local_levels,
        }
        common = {
            name: aggregate_cell_assignment(
                local_records, assignment, num_classes=NUM_CLASSES
            )
            for name, assignment in assignments.items()
        }
        references = {
            name: np.asarray(full_confusions_by_group[group][name], dtype=np.int64)
            for name in ("k1", "k2", "k4")
        }
        route_confusion = (
            references["k1"]
            + np.asarray(common["route"]["confusion"], dtype=np.int64)
            - np.asarray(common["k1"]["confusion"], dtype=np.int64)
        )
        if np.any(route_confusion < 0):
            raise AssertionError("fixed route produced negative per-image confusion")
        route_miou = mean_iou_from_confusion(route_confusion)
        endpoint_mious = {
            name: mean_iou_from_confusion(confusion)
            for name, confusion in references.items()
        }
        selected_counts = {
            "k1": int(np.count_nonzero(local_levels == 1)),
            "k2": int(np.count_nonzero(local_levels == 2)),
            "k4": int(np.count_nonzero(local_levels == 4)),
        }
        extra_cost = selected_counts["k2"] + 3 * selected_counts["k4"]
        small_delta = None
        thin_delta = None
        for region_name in ("small", "thin"):
            route_rate = common["route"]["regions"][region_name]["error_rate"]
            k2_rate = common["k2"]["regions"][region_name]["error_rate"]
            delta = (
                None
                if route_rate is None or k2_rate is None
                else float(route_rate - k2_rate)
            )
            if region_name == "small":
                small_delta = delta
            else:
                thin_delta = delta
        per_image.append(
            {
                "image_index": group,
                "cell_count": len(indices),
                "selected_counts": selected_counts,
                "actual_extra_proxy_cost": extra_cost,
                "forward_equivalent_cost": float(1.0 + extra_cost / len(indices)),
                "route_miou_percent": float(route_miou * 100.0),
                "delta_over_k1_pp": float(
                    (route_miou - endpoint_mious["k1"]) * 100.0
                ),
                "delta_over_k2x_pp": float(
                    (route_miou - endpoint_mious["k2"]) * 100.0
                ),
                "delta_below_k4_pp": float(
                    (route_miou - endpoint_mious["k4"]) * 100.0
                ),
                "error_pixels_delta_over_k2x": (
                    _error_count(route_confusion) - _error_count(references["k2"])
                ),
                "small_error_rate_delta_over_k2x": small_delta,
                "thin_error_rate_delta_over_k2x": thin_delta,
            }
        )
        internal.append(
            {
                "group": group,
                "cell_count": len(indices),
                "extra_cost": extra_cost,
                "route_confusion": route_confusion,
                "references": references,
                "route_regions": common["route"]["regions"],
                "k2_regions": common["k2"]["regions"],
            }
        )

    deltas_over_k2 = np.asarray(
        [record["delta_over_k2x_pp"] for record in per_image], dtype=np.float64
    )
    deltas_over_k1 = np.asarray(
        [record["delta_over_k1_pp"] for record in per_image], dtype=np.float64
    )
    absolute_error_changes = np.sort(
        np.abs(
            np.asarray(
                [record["error_pixels_delta_over_k2x"] for record in per_image],
                dtype=np.float64,
            )
        )
    )[::-1]
    absolute_total = float(absolute_error_changes.sum())
    top_three_share = (
        float(absolute_error_changes[:3].sum() / absolute_total)
        if absolute_total
        else 0.0
    )

    loo = []
    if len(internal) > 1:
        total_confusions = {
            name: sum(
                (
                    item["route_confusion"]
                    if name == "route"
                    else item["references"][name]
                    for item in internal
                ),
                start=np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64),
            )
            for name in ("route", "k1", "k2", "k4")
        }
        total_regions = {
            source: {
                region: {
                    "pixels": sum(
                        int(item[f"{source}_regions"][region]["pixels"])
                        for item in internal
                    ),
                    "errors": sum(
                        int(item[f"{source}_regions"][region]["errors"])
                        for item in internal
                    ),
                }
                for region in ("small", "thin")
            }
            for source in ("route", "k2")
        }
        total_cells = sum(item["cell_count"] for item in internal)
        total_extra = sum(item["extra_cost"] for item in internal)
        for removed in internal:
            remaining_confusions = {
                name: total_confusions[name]
                - (
                    removed["route_confusion"]
                    if name == "route"
                    else removed["references"][name]
                )
                for name in total_confusions
            }
            remaining_cells = total_cells - removed["cell_count"]
            remaining_extra = total_extra - removed["extra_cost"]
            mious = {
                name: mean_iou_from_confusion(confusion)
                for name, confusion in remaining_confusions.items()
            }
            positive_k4 = mious["k4"] > mious["k1"] + 1e-12
            retention = (
                float(
                    (mious["route"] - mious["k1"])
                    / (mious["k4"] - mious["k1"])
                )
                if positive_k4
                else None
            )
            region_deltas = {}
            for region in ("small", "thin"):
                remaining_region = {}
                for source in ("route", "k2"):
                    totals = {
                        key: total_regions[source][region][key]
                        - int(removed[f"{source}_regions"][region][key])
                        for key in ("pixels", "errors")
                    }
                    remaining_region[source] = _region_rate_from_totals(totals)
                region_deltas[region] = (
                    None
                    if any(value is None for value in remaining_region.values())
                    else float(remaining_region["route"] - remaining_region["k2"])
                )
            cost = float(1.0 + remaining_extra / remaining_cells)
            checks = {
                "positive_full_k4_gain": positive_k4,
                "cost_within_budget": cost <= 2.0 + 1e-12,
                "retains_k4_gain": retention is not None and retention >= 0.70 - 1e-12,
                "outperforms_uniform_k2x": mious["route"] > mious["k2"] + 1e-12,
                "small_not_worse_than_k2x": (
                    region_deltas["small"] is not None
                    and region_deltas["small"] <= 1e-12
                ),
                "thin_not_worse_than_k2x": (
                    region_deltas["thin"] is not None
                    and region_deltas["thin"] <= 1e-12
                ),
            }
            loo.append(
                {
                    "removed_image_index": removed["group"],
                    "fixed_policy_not_reoptimized": True,
                    "route_miou_percent": float(mious["route"] * 100.0),
                    "delta_over_k2x_pp": float(
                        (mious["route"] - mious["k2"]) * 100.0
                    ),
                    "k4_gain_retention": retention,
                    "forward_equivalent_cost": cost,
                    "small_error_rate_delta_over_k2x": region_deltas["small"],
                    "thin_error_rate_delta_over_k2x": region_deltas["thin"],
                    "descriptive_partial_checks_without_random": checks,
                    "descriptive_partial_checks_passed": all(checks.values()),
                }
            )

    return {
        "role": (
            "descriptive stability only; it does not add a hard Stage-A gate "
            "and does not estimate deployment generalization"
        ),
        "per_image": per_image,
        "summary": {
            "image_count": len(per_image),
            "positive_image_count_vs_k1": int(np.count_nonzero(deltas_over_k1 > 0.0)),
            "positive_image_count_vs_k2x": int(np.count_nonzero(deltas_over_k2 > 0.0)),
            "median_delta_over_k1_pp": float(np.median(deltas_over_k1)),
            "median_delta_over_k2x_pp": float(np.median(deltas_over_k2)),
            "top_three_share_of_absolute_error_pixel_change_vs_k2x": top_three_share,
        },
        "leave_one_image_out_fixed_policy": loo,
        "loo_limitation": (
            "The random p95 control is not recomputed after removal, so these "
            "are deterministic partial checks rather than repeated formal gates."
            if loo
            else "LOO is undefined for a one-image subset smoke."
        ),
    }


def k2_axis_diagnostic(
    image_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report fixed y-K2 versus preregistered x-K2 without selecting an axis."""

    rows = []
    for image in image_records:
        x_confusion = np.asarray(image["confusion"]["matched_k2"], dtype=np.int64)
        y_confusion = np.asarray(
            image["confusion"]["matched_k2_y"], dtype=np.int64
        )
        x_miou = mean_iou_from_confusion(x_confusion)
        y_miou = mean_iou_from_confusion(y_confusion)
        rows.append(
            {
                "image_index": int(image["loader_position"]),
                "sample_name": image["sample_name"],
                "k2x_miou_percent": float(x_miou * 100.0),
                "k2y_miou_percent": float(y_miou * 100.0),
                "k2y_minus_k2x_pp": float((y_miou - x_miou) * 100.0),
            }
        )
    differences = np.asarray(
        [row["k2y_minus_k2x_pp"] for row in rows], dtype=np.float64
    )
    return {
        "role": (
            "descriptive only; y-K2 never enters A2 actions, greedy selection, "
            "random controls, gates, or per-cell best-axis selection"
        ),
        "per_image": rows,
        "summary": {
            "image_count": len(rows),
            "k2y_better_image_count": int(np.count_nonzero(differences > 0.0)),
            "k2x_better_image_count": int(np.count_nonzero(differences < 0.0)),
            "median_k2y_minus_k2x_pp": float(np.median(differences)),
        },
    }


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
        for name in (
            "label",
            "k1",
            "legacy_k2",
            "matched_k2",
            "matched_k2_y",
            "k4",
        )
    }
    full_confusions = {
        name: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for name in ("k1", "legacy_k2", "matched_k2", "matched_k2_y", "k4")
    }
    matched_k2_y_common = {
        "confusion": np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64),
        "regions": {
            name: {"pixels": 0, "errors": 0, "error_rate": None}
            for name in ("all", "small", "thin")
        },
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
            matched_k2_y_sum = None
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
                elif shift == (PHASE_OFFSET, 0):
                    matched_k2_y_sum = (
                        aligned_phase_crop(
                            baseline_scores, (0, 0), original_slice, shifted_slices
                        ).clone()
                        + aligned
                    )
                del shifted_scores, aligned
            if (
                matched_k2_sum is None
                or matched_k2_y_sum is None
                or legacy_k2_sum is None
            ):
                raise AssertionError("failed to construct the fixed x/y K2 views")

            matched_k2_prediction = prediction_from_aligned_score_sum(
                k1_prediction, matched_k2_sum, original_slice
            )
            legacy_k2_prediction = prediction_from_aligned_score_sum(
                k1_prediction, legacy_k2_sum, x_original_slice
            )
            matched_k2_y_prediction = prediction_from_aligned_score_sum(
                k1_prediction, matched_k2_y_sum, original_slice
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
                "matched_k2_y": matched_k2_y_prediction,
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
            local_y_stats = cell_phase_statistics(
                cell_bounds,
                k1_prediction[0],
                matched_k2_y_prediction[0],
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
            local_y_aggregate = aggregate_cell_assignment(
                local_y_stats, local_k2_levels, num_classes=NUM_CLASSES
            )
            direct_y = confusion_from_arrays(
                matched_k2_y_prediction[0],
                label_full[0],
                NUM_CLASSES,
                mask=common_mask,
            )
            if not np.array_equal(local_y_aggregate["confusion"], direct_y):
                raise AssertionError(
                    "ownership cells do not reproduce matched y-K2 common confusion"
                )
            matched_k2_y_common["confusion"] += local_y_aggregate["confusion"]
            for region_name, totals in matched_k2_y_common["regions"].items():
                local_region = local_y_aggregate["regions"][region_name]
                totals["pixels"] += int(local_region["pixels"])
                totals["errors"] += int(local_region["errors"])

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
                matched_k2_y_prediction,
                k4_prediction,
                matched_k2_sum,
                matched_k2_y_sum,
                legacy_k2_sum,
                k4_sum,
                region_masks,
                local_y_stats,
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
    for totals in matched_k2_y_common["regions"].values():
        pixels = int(totals["pixels"])
        errors = int(totals["errors"])
        totals["error_rate"] = float(errors / pixels) if pixels else None
    outside_common_confusion = (
        full_confusions["k1"] - common_endpoints["k1"]["confusion"]
    )
    if np.any(outside_common_confusion < 0):
        raise AssertionError("common K1 confusion is not a subset of full K1")
    for name in ("k1", "matched_k2", "k4"):
        reconstructed = outside_common_confusion + common_endpoints[name]["confusion"]
        if not np.array_equal(reconstructed, full_confusions[name]):
            raise AssertionError(f"cell endpoint does not reproduce dense {name}")
    reconstructed_y = outside_common_confusion + matched_k2_y_common["confusion"]
    if not np.array_equal(reconstructed_y, full_confusions["matched_k2_y"]):
        raise AssertionError("cell endpoint does not reproduce dense matched y-K2")

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
    endpoints["matched_k2_y_descriptive"] = {
        **_endpoint_summary(
            full_confusions["matched_k2_y"],
            matched_k2_y_common,
            cfg["labels"],
        ),
        "role": (
            "descriptive axis diagnostic only; the preregistered hierarchy uses x-K2"
        ),
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
    group_ids = np.asarray(
        [int(record["image_index"]) for record in all_cell_stats], dtype=np.int64
    )
    group_order = [int(value) for value in dict.fromkeys(group_ids.tolist())]
    cell_count_by_group = {
        group: int(np.count_nonzero(group_ids == group)) for group in group_order
    }
    reference_k1_by_group = {
        int(image["loader_position"]): np.asarray(
            image["confusion"]["k1"], dtype=np.int64
        )
        for image in image_records
    }
    full_confusions_by_group = {
        int(image["loader_position"]): {
            "k1": np.asarray(image["confusion"]["k1"], dtype=np.int64),
            "k2": np.asarray(
                image["confusion"]["matched_k2"], dtype=np.int64
            ),
            "k4": np.asarray(image["confusion"]["k4"], dtype=np.int64),
        }
        for image in image_records
    }
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
    scores["singleton_per_image_miou"] = restrict_score_to_geometry(
        grouped_singleton_miou_scores(
            all_cell_stats, group_ids, reference_k1_by_group
        ),
        geometry_eligible,
    )
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
    per_image_binary_oracle_curves = {
        "net_correct": evaluate_grouped_score_curve(
            all_cell_stats,
            scores["net_correct"],
            group_ids,
            outside_common_confusion,
            cfg["labels"],
            deployment=False,
        ),
        "singleton_per_image_miou": evaluate_grouped_score_curve(
            all_cell_stats,
            scores["singleton_per_image_miou"],
            group_ids,
            outside_common_confusion,
            cfg["labels"],
            deployment=False,
        ),
    }
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

    total_cells = len(all_cell_stats)
    global_extra_budget = total_cells
    per_image_extra_budgets = dict(cell_count_by_group)
    binary_global_run = binary_greedy_oracle(
        all_cell_stats,
        (global_extra_budget,),
        num_classes=NUM_CLASSES,
        reference_confusion=full_confusions["k1"],
    )
    hierarchical_global_run = hierarchical_greedy_oracle(
        all_cell_stats,
        (global_extra_budget,),
        num_classes=NUM_CLASSES,
        reference_confusion=full_confusions["k1"],
    )
    binary_per_image_run = binary_group_budget_oracle(
        all_cell_stats,
        group_ids.tolist(),
        per_image_extra_budgets,
        num_classes=NUM_CLASSES,
        reference_confusion_by_group=reference_k1_by_group,
    )
    hierarchical_per_image_run = hierarchical_group_budget_oracle(
        all_cell_stats,
        group_ids.tolist(),
        per_image_extra_budgets,
        num_classes=NUM_CLASSES,
        reference_confusion_by_group=reference_k1_by_group,
    )

    route_points = {
        "a1_binary": {
            "global": greedy_snapshot_point(
                binary_global_run["snapshots"][0],
                cfg["labels"],
                route_name="A1_binary_k1_k4",
                budget_scope="global-pooled",
                source_ranking="gt_informed_dynamic_global_miou_marginal",
            ),
            "per_image": greedy_snapshot_point(
                binary_per_image_run["snapshot"],
                cfg["labels"],
                route_name="A1_binary_k1_k4",
                budget_scope="per-image-capped",
                source_ranking=(
                    "gt_informed_dynamic_global_miou_marginal_under_per_image_caps"
                ),
            ),
        },
        "a2_hierarchical_x": {
            "global": greedy_snapshot_point(
                hierarchical_global_run["snapshots"][0],
                cfg["labels"],
                route_name="A2_hierarchical_k1_k2x_k4",
                budget_scope="global-pooled",
                source_ranking=(
                    "gt_informed_dynamic_global_miou_marginal_per_proxy_cost"
                ),
            ),
            "per_image": greedy_snapshot_point(
                hierarchical_per_image_run["snapshot"],
                cfg["labels"],
                route_name="A2_hierarchical_k1_k2x_k4",
                budget_scope="per-image-capped",
                source_ranking=(
                    "gt_informed_dynamic_global_miou_marginal_per_proxy_cost_"
                    "under_per_image_caps"
                ),
            ),
        },
    }
    for route_scopes in route_points.values():
        for point in route_scopes.values():
            levels = np.asarray(point["levels_by_cell"], dtype=np.int64)
            if np.any((levels != 1) & (~geometry_eligible)):
                raise AssertionError(
                    "greedy selected a cell outside public common-support geometry"
                )

    global_random_groups = np.zeros(total_cells, dtype=np.int64)
    route_random_controls = {}
    for route, scopes in route_points.items():
        route_random_controls[route] = {}
        for scope, point in scopes.items():
            random_groups = global_random_groups if scope == "global" else group_ids
            control = matched_action_random_control(
                all_cell_stats,
                full_confusions["k1"],
                point["levels_by_cell"],
                geometry_eligible,
                random_groups,
                route_kind=route,
            )
            control["route_name"] = point["route_name"]
            control["budget_scope"] = point["budget_scope"]
            if not math.isclose(
                control["cost"]["forward_equivalent_cost"],
                point["cost"]["forward_equivalent_cost"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AssertionError(
                    "matched random control changed the target policy proxy cost"
                )
            route_random_controls[route][scope] = control

    formal_decision = args.max_images is None
    route_decisions = {
        route: {
            scope: route_decision_from_point(
                endpoints,
                route_points[route][scope],
                route_random_controls[route][scope],
                formal=formal_decision,
                route_name=route_points[route][scope]["route_name"],
                budget_scope=route_points[route][scope]["budget_scope"],
            )
            for scope in ("global", "per_image")
        }
        for route in ("a1_binary", "a2_hierarchical_x")
    }
    decision = arbitrate_stage_a_routes(route_decisions, formal=formal_decision)
    stability = {
        route: {
            scope: fixed_policy_stability(
                all_cell_stats,
                route_points[route][scope]["levels_by_cell"],
                group_ids,
                full_confusions_by_group,
            )
            for scope in ("global", "per_image")
        }
        for route in ("a1_binary", "a2_hierarchical_x")
    }
    axis_diagnostic = k2_axis_diagnostic(image_records)
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
                    "oracle_singleton_per_image_miou_gain": _finite_or_none(
                        scores["singleton_per_image_miou"].values[index]
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

    stage_a_routes = {
        "a1_binary": {
            "action_space": "K1 -> K4 with incremental proxy cost +3",
            "selection_rule": (
                "GT-informed constructive greedy; recompute current merged full-"
                "dataset mIoU marginal after every action; stop at no positive gain"
            ),
            "global": {
                "greedy_point": route_points["a1_binary"]["global"],
                "action_trace": binary_global_run["actions"][:
                    route_points["a1_binary"]["global"]["chosen_action_count"]
                ],
                "random_control": route_random_controls["a1_binary"]["global"],
                "gate": route_decisions["a1_binary"]["global"],
                "stability": stability["a1_binary"]["global"],
            },
            "per_image": {
                "greedy_point": route_points["a1_binary"]["per_image"],
                "action_trace": binary_per_image_run["actions"],
                "group_caps_and_usage": binary_per_image_run["groups"],
                "random_control": route_random_controls["a1_binary"]["per_image"],
                "gate": route_decisions["a1_binary"]["per_image"],
                "stability": stability["a1_binary"]["per_image"],
            },
        },
        "a2_hierarchical_x": {
            "action_space": (
                "K1 -> fixed x-K2 (+1) -> K4 (+2); K4 requires prior x-K2"
            ),
            "selection_rule": (
                "GT-informed constructive greedy; recompute current merged full-"
                "dataset mIoU marginal per incremental proxy cost after every "
                "action; never cross a non-positive bridge"
            ),
            "global": {
                "greedy_point": route_points["a2_hierarchical_x"]["global"],
                "action_trace": hierarchical_global_run["actions"],
                "random_control": route_random_controls[
                    "a2_hierarchical_x"
                ]["global"],
                "gate": route_decisions["a2_hierarchical_x"]["global"],
                "stability": stability["a2_hierarchical_x"]["global"],
            },
            "per_image": {
                "greedy_point": route_points["a2_hierarchical_x"]["per_image"],
                "action_trace": hierarchical_per_image_run["actions"],
                "group_caps_and_usage": hierarchical_per_image_run["groups"],
                "random_control": route_random_controls[
                    "a2_hierarchical_x"
                ]["per_image"],
                "gate": route_decisions["a2_hierarchical_x"]["per_image"],
                "stability": stability["a2_hierarchical_x"]["per_image"],
            },
        },
    }

    output = {
        "status": "PASS",
        "status_meaning": (
            "Execution, geometry, endpoint, A1/A2 construction, and reporting "
            "checks passed. Scientific GO/NO-GO is separate and absent for smoke."
        ),
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "full-test" if args.max_images is None else "subset-smoke",
        "scientific_scope": (
            "GT-informed post-aggregation ownership-cell feasibility audit for "
            "fixed A1/A2 action spaces. It tests spatial concentration (H1), "
            "not exact sparse executability (H2) or router generalization (H3)."
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
            "stage_a_action_spaces": {
                "a1_binary": {
                    "levels": [1, 4],
                    "proxy_cost": "1 + 3*q4",
                },
                "a2_hierarchical_x": {
                    "levels": [1, 2, 4],
                    "fixed_k2": "normal + x8 shifted-canvas phase",
                    "precedence": "K1 -> K2x -> K4",
                    "proxy_cost": "1 + q(K2-or-K4) + 2*q4",
                },
            },
            "proxy_cost_warning": (
                "post-aggregation optimistic proxy, not executed crop forwards or latency"
            ),
            "q_values": list(Q_VALUES),
            "q_values_role": (
                "descriptive legacy fixed-ranking A1 curves only; never the formal gate"
            ),
            "formal_proxy_budget": {
                "requested_forward_equivalent_cost": 2.0,
                "global_pooled_extra_budget": global_extra_budget,
                "per_image_extra_budget": per_image_extra_budgets,
                "per_image_objective": (
                    "one merged full-dataset mIoU objective under independent image caps"
                ),
                "promotion_requirement": (
                    "the same candidate action space must pass both scopes"
                ),
            },
            "constructive_greedy_role": (
                "GT-informed optimistic search heuristic, not a mathematical optimum "
                "and not a deployable router"
            ),
            "random_replicates": RANDOM_REPLICATES,
            "random_seed": RANDOM_SEED,
            "geometry_eligible_cells": int(np.count_nonzero(geometry_eligible)),
            "random_gate_control": (
                "match K2-only/K4 action counts globally or per image, preserve "
                "K4 nesting, sample public geometry first, and retain every real "
                "window in the proxy-cost denominator"
            ),
            "naive_all_cell_random_role": "descriptive only; never used by the gate",
            "matched_k2x": (
                "normal+x8 logits on exactly the sealed K4 common support; the "
                "only K2 admitted to A2 and all gates"
            ),
            "matched_k2y": (
                "normal+y8 free descriptive endpoint only; never used to select "
                "an axis, rescue a gate, or alter the registered hierarchy"
            ),
            "candidate_priority": (
                "A2 is selected if both scopes pass; otherwise A1 may fall back if "
                "both scopes pass; otherwise no Stage B"
            ),
            "stage_a_gate_meaning": (
                "scientific feasibility only: cost<=2x, >=70% K4-gain retention, "
                "strictly above fixed K2x, small/thin safe, and above matched random p95"
            ),
            "stage_b_frozen_requirements": {
                "implementation_trigger": "only after Stage-A authorization",
                "exact_cost": (
                    "deduplicated dependency closure over (image, phase, crop_id)"
                ),
                "random": "match exact incremental crop-forward cost",
                "correctness": (
                    "K1/K2x/K4 endpoints plus one fixed nontrivial middle-subset "
                    "prediction/confusion/dependency-closure equality test"
                ),
                "practical_gate": (
                    "exact overall/per-image cost<=2x, >=70% retention, >=+0.05pp "
                    "over K2x, small/thin safe, above exact random p95, and measured "
                    "latency below dense K4"
                ),
            },
            "e0_phase_context": (
                "341 mod 16 = 5, so ordinary overlapping E0 crops already rotate "
                "local crop-origin patch phases; K2/K4 add shifted-canvas phase "
                "passes beyond that implicit coverage. K1/K2/K4 count these "
                "shifted-canvas passes, not unique ViT patch-residue classes"
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
            "descriptive_global_binary_rankings": oracle_curves,
            "descriptive_global_finite_oracle_envelope": oracle_envelope,
            "descriptive_per_image_binary_rankings": per_image_binary_oracle_curves,
            "simple_k1_scores": simple_curves,
            "descriptive_geometry_aware_equal_count_random": geometry_random_curve,
            "naive_all_cell_random_descriptive": naive_random_curve,
        },
        "formal_stage_a_routes": stage_a_routes,
        "descriptive_diagnostics": {
            "fixed_k2_axis": axis_diagnostic,
            "stability_note": (
                "Per-image and fixed-policy LOO results are embedded under each "
                "route/scope and never add an unregistered hard gate."
            ),
        },
        "stage_a_route_decisions": route_decisions,
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
                "matched_k2_y_miou_percent": endpoints[
                    "matched_k2_y_descriptive"
                ]["full_image"]["miou_percent"],
                "k4_miou_percent": endpoints["k4"]["full_image"]["miou_percent"],
                "stage_a_decision": decision,
                "route_gate_pass": {
                    route: {
                        scope: route_decisions[route][scope]["passed"]
                        for scope in ("global", "per_image")
                    }
                    for route in ("a1_binary", "a2_hierarchical_x")
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"phase_utility_result={args.output_path.resolve()}")
    print("phase_utility_status=PASS")


if __name__ == "__main__":
    main()
