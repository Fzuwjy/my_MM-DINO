"""Pure NumPy helpers for the zero-training phase-utility audit.

The audit compares predictions obtained with one, two, and four spatial
phases (``K1``, ``K2``, and ``K4``).  Sliding windows overlap, so this module
first derives one mutually exclusive rectangular ownership cell per crop.  A
mixed-compute prediction can then choose one phase level per cell without
double counting overlap pixels.  Ownership is represented by ``[N, 4]``
bounds rather than an ``H x W`` map, which is essential for WHU-scale images.

Ground truth is intentionally confined to metric and oracle-score helpers.
Reconstruction, deployment-score construction, deterministic selection, and
cost accounting do not accept a target array.  ``CellScoreVector`` carries an
explicit ``uses_ground_truth`` flag, and deployment selection rejects an
oracle score vector at runtime.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


PHASE_LEVELS = (1, 2, 4)
REGION_NAMES = ("all", "small", "thin")


def _require_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validated_bool_mask(
    mask: np.ndarray | None,
    shape: tuple[int, int],
    *,
    name: str,
    default: bool,
) -> np.ndarray:
    if mask is None:
        return np.full(shape, default, dtype=bool)
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(f"{name} shape differs from image: {mask.shape} != {shape}")
    if mask.dtype != np.bool_:
        raise TypeError(f"{name} must have bool dtype")
    return np.ascontiguousarray(mask)


def _validated_windows(
    windows: np.ndarray | Sequence[Sequence[int]],
    image_shape: tuple[int, int],
) -> np.ndarray:
    windows = np.asarray(windows)
    if windows.ndim != 2 or windows.shape[1:] != (4,):
        raise ValueError("windows must have shape [N, 4] in (y0, y1, x0, x1) order")
    if windows.shape[0] == 0:
        raise ValueError("windows must not be empty")
    if not np.issubdtype(windows.dtype, np.integer) or windows.dtype == np.bool_:
        raise TypeError("windows must contain integer coordinates")
    windows = np.ascontiguousarray(windows, dtype=np.int64)
    height, width = image_shape
    y0, y1, x0, x1 = windows.T
    if np.any(y0 < 0) or np.any(x0 < 0):
        raise ValueError("window starts must be non-negative")
    if np.any(y1 > height) or np.any(x1 > width):
        raise ValueError("windows must lie inside the image")
    if np.any(y0 >= y1) or np.any(x0 >= x1):
        raise ValueError("every window must have positive height and width")
    return windows


def _validated_image_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
        or any(isinstance(value, (bool, np.bool_)) for value in image_shape)
    ):
        raise TypeError("image_shape must be a (height, width) integer tuple")
    return (
        _require_positive_int(image_shape[0], name="image height"),
        _require_positive_int(image_shape[1], name="image width"),
    )


def _axis_ownership_intervals(
    axis_length: int,
    starts: np.ndarray,
    crop_size: int,
) -> np.ndarray:
    """Return midpoint ownership intervals for one sliding-window axis."""

    starts = np.asarray(starts, dtype=np.int64)
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("axis starts must be a non-empty 1-D array")
    if np.any(starts[1:] <= starts[:-1]):
        raise ValueError("axis starts must be strictly increasing")
    if starts[0] != 0 or starts[-1] + crop_size != axis_length:
        raise ValueError("axis starts must include the initial and end-aligned crops")
    if np.any(starts[1:] > starts[:-1] + crop_size):
        raise ValueError("sliding crops must cover the complete axis")

    boundaries = np.empty(starts.size + 1, dtype=np.int64)
    boundaries[0] = 0
    boundaries[-1] = axis_length
    if starts.size > 1:
        boundaries[1:-1] = (
            starts[:-1] + crop_size + starts[1:]
        ) // 2
    if np.any(boundaries[1:] <= boundaries[:-1]):
        raise ValueError("midpoint ownership produced an empty axis interval")
    intervals = np.column_stack((boundaries[:-1], boundaries[1:]))
    if np.any(intervals[:, 0] < starts) or np.any(
        intervals[:, 1] > starts + crop_size
    ):
        raise ValueError("an ownership interval lies outside its source crop")
    return intervals


def _validated_cell_bounds(
    cell_bounds: np.ndarray | Sequence[Sequence[int]],
    image_shape: tuple[int, int],
    *,
    require_partition: bool = True,
) -> np.ndarray:
    shape = _validated_image_shape(image_shape)
    bounds = np.asarray(cell_bounds)
    if bounds.ndim != 2 or bounds.shape[1:] != (4,) or bounds.shape[0] == 0:
        raise ValueError("cell_bounds must have non-empty shape [N, 4]")
    if not np.issubdtype(bounds.dtype, np.integer) or bounds.dtype == np.bool_:
        raise TypeError("cell_bounds must contain integer coordinates")
    bounds = np.ascontiguousarray(bounds, dtype=np.int64)
    height, width = shape
    y0, y1, x0, x1 = bounds.T
    if (
        np.any(y0 < 0)
        or np.any(x0 < 0)
        or np.any(y1 > height)
        or np.any(x1 > width)
        or np.any(y0 >= y1)
        or np.any(x0 >= x1)
    ):
        raise ValueError("cell bounds must be non-empty rectangles inside the image")

    if require_partition:
        # Validate exact once-only coverage on the coordinate-compressed grid,
        # never on an H x W raster.
        y_edges = np.unique(np.concatenate(([0, height], y0, y1)))
        x_edges = np.unique(np.concatenate(([0, width], x0, x1)))
        difference = np.zeros((len(y_edges) + 1, len(x_edges) + 1), dtype=np.int32)
        y_lookup = {int(value): index for index, value in enumerate(y_edges)}
        x_lookup = {int(value): index for index, value in enumerate(x_edges)}
        for top, bottom, left, right in bounds:
            iy0, iy1 = y_lookup[int(top)], y_lookup[int(bottom)]
            ix0, ix1 = x_lookup[int(left)], x_lookup[int(right)]
            difference[iy0, ix0] += 1
            difference[iy1, ix0] -= 1
            difference[iy0, ix1] -= 1
            difference[iy1, ix1] += 1
        coverage = difference.cumsum(axis=0).cumsum(axis=1)[
            : len(y_edges) - 1, : len(x_edges) - 1
        ]
        if np.any(coverage != 1):
            raise ValueError("cell bounds must cover every image pixel exactly once")
    return bounds


def build_cell_ownership(
    image_shape: tuple[int, int],
    windows: np.ndarray | Sequence[Sequence[int]],
) -> np.ndarray:
    """Return one midpoint-owned rectangular cell per sliding crop.

    Let sorted crop starts on an axis be ``s`` and the common crop length be
    ``C``.  The interior ownership boundary is preregistered as
    ``floor((s[i-1] + C + s[i]) / 2)``; the outer boundaries are zero and the
    full axis length.  The 2-D cells are Cartesian products of these axis
    intervals and are returned in the same order as ``windows``.

    No H x W ownership or distance array is allocated, and ground truth is
    never observed.
    """

    shape = _validated_image_shape(image_shape)
    windows_array = _validated_windows(windows, shape)
    heights = windows_array[:, 1] - windows_array[:, 0]
    widths = windows_array[:, 3] - windows_array[:, 2]
    if np.any(heights != heights[0]) or np.any(widths != widths[0]):
        raise ValueError("all sliding windows must share one crop shape")
    crop_height, crop_width = int(heights[0]), int(widths[0])
    row_starts = np.unique(windows_array[:, 0])
    column_starts = np.unique(windows_array[:, 2])
    expected_count = len(row_starts) * len(column_starts)
    if len(windows_array) != expected_count:
        raise ValueError("windows must form one complete Cartesian sliding grid")
    pairs = {(int(row[0]), int(row[2])) for row in windows_array}
    expected_pairs = {
        (int(row_start), int(column_start))
        for row_start in row_starts
        for column_start in column_starts
    }
    if pairs != expected_pairs:
        raise ValueError("windows contain duplicate or missing Cartesian grid positions")

    row_intervals = _axis_ownership_intervals(
        shape[0], row_starts, crop_height
    )
    column_intervals = _axis_ownership_intervals(
        shape[1], column_starts, crop_width
    )
    row_lookup = {int(value): index for index, value in enumerate(row_starts)}
    column_lookup = {
        int(value): index for index, value in enumerate(column_starts)
    }
    bounds = np.empty_like(windows_array)
    for index, window in enumerate(windows_array):
        row = row_intervals[row_lookup[int(window[0])]]
        column = column_intervals[column_lookup[int(window[2])]]
        bounds[index] = (row[0], row[1], column[0], column[1])
    validate_cell_partition(bounds, windows_array, shape)
    return bounds


def build_cell_ownership_bounds(
    image_shape: tuple[int, int],
    windows: np.ndarray | Sequence[Sequence[int]],
) -> np.ndarray:
    """Explicitly named alias for :func:`build_cell_ownership`."""

    return build_cell_ownership(image_shape, windows)


def validate_cell_partition(
    cell_bounds: np.ndarray | Sequence[Sequence[int]],
    windows: np.ndarray | Sequence[Sequence[int]],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Validate exact coverage and cell-within-source-window containment.

    The returned vector contains each ownership rectangle's geometric area.
    Evaluation masks are applied later, so a real outer cell may legitimately
    contribute zero evaluated pixels while remaining in compute accounting.
    """

    shape = _validated_image_shape(image_shape)
    bounds = _validated_cell_bounds(cell_bounds, shape)
    windows_array = _validated_windows(windows, shape)
    if bounds.shape != windows_array.shape:
        raise ValueError("cell_bounds and windows must have matching shapes")
    if np.any(bounds[:, 0] < windows_array[:, 0]) or np.any(
        bounds[:, 1] > windows_array[:, 1]
    ) or np.any(bounds[:, 2] < windows_array[:, 2]) or np.any(
        bounds[:, 3] > windows_array[:, 3]
    ):
        raise ValueError("every ownership cell must lie inside its source window")
    return (
        (bounds[:, 1] - bounds[:, 0]) * (bounds[:, 3] - bounds[:, 2])
    ).astype(np.int64, copy=False)


def _validate_semantic_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_classes = _require_positive_int(num_classes, name="num_classes")
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("prediction and target must both be 2-D")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    if not np.issubdtype(prediction.dtype, np.integer) or prediction.dtype == np.bool_:
        raise TypeError("prediction must contain integer class indices")
    if not np.issubdtype(target.dtype, np.integer) or target.dtype == np.bool_:
        raise TypeError("target must contain integer class indices")
    prediction = np.ascontiguousarray(prediction)
    target = np.ascontiguousarray(target)
    valid = (target >= 0) & (target < num_classes)
    invalid_prediction = valid & (
        (prediction < 0) | (prediction >= num_classes)
    )
    if np.any(invalid_prediction):
        values = np.unique(prediction[invalid_prediction])
        raise ValueError(
            "prediction contains out-of-range values on valid target pixels: "
            f"{values[:10].tolist()}"
        )
    return prediction, target, valid


def confusion_from_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a target-row/prediction-column confusion matrix."""

    prediction, target, valid = _validate_semantic_arrays(
        prediction, target, num_classes
    )
    if mask is not None:
        valid &= _validated_bool_mask(
            mask, target.shape, name="mask", default=True
        )
    encoded = (
        target[valid].astype(np.int64, copy=False) * int(num_classes)
        + prediction[valid].astype(np.int64, copy=False)
    )
    return np.bincount(encoded, minlength=int(num_classes) ** 2).reshape(
        int(num_classes), int(num_classes)
    )


def class_iou_from_confusion(confusion: np.ndarray) -> np.ndarray:
    """Return per-class IoU, with ``NaN`` for classes whose union is zero."""

    confusion = np.asarray(confusion)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")
    if not np.issubdtype(confusion.dtype, np.number):
        raise TypeError("confusion must be numeric")
    values = confusion.astype(np.float64, copy=False)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("confusion must contain finite, non-negative values")
    diagonal = np.diag(values)
    union = values.sum(axis=1) + values.sum(axis=0) - diagonal
    iou = np.full(diagonal.shape, np.nan, dtype=np.float64)
    np.divide(diagonal, union, out=iou, where=union > 0)
    return iou


def class_ious_from_confusion(confusion: np.ndarray) -> np.ndarray:
    """Plural alias matching the existing spatial-diagnostics helper."""

    return class_iou_from_confusion(confusion)


def mean_iou_from_confusion(confusion: np.ndarray) -> float:
    """Average IoU over exactly the classes whose union is greater than zero."""

    ious = class_iou_from_confusion(confusion)
    supported = np.isfinite(ious)
    return float(ious[supported].mean()) if np.any(supported) else float("nan")


def phase_transition_counts(
    reference: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    mask: np.ndarray | None = None,
) -> dict[str, int]:
    """Count fixes and breaks from ``reference`` to ``candidate``.

    A fix is reference-wrong/candidate-correct; a break is
    reference-correct/candidate-wrong.  Both predictions being wrong is never
    counted as a fix, even if their class labels differ.
    """

    reference, target, valid = _validate_semantic_arrays(
        reference, target, num_classes
    )
    candidate, candidate_target, _ = _validate_semantic_arrays(
        candidate, target, num_classes
    )
    if not np.array_equal(candidate_target, target):  # defensive; same object normally
        raise ValueError("candidate target differs from reference target")
    if mask is not None:
        valid &= _validated_bool_mask(
            mask, target.shape, name="mask", default=True
        )
    reference_correct = (reference == target) & valid
    candidate_correct = (candidate == target) & valid
    fixed = (~reference_correct) & candidate_correct & valid
    broken = reference_correct & (~candidate_correct) & valid
    changed = (reference != candidate) & valid
    fixed_count = int(np.count_nonzero(fixed))
    broken_count = int(np.count_nonzero(broken))
    return {
        "pixels": int(np.count_nonzero(valid)),
        "reference_correct": int(np.count_nonzero(reference_correct)),
        "candidate_correct": int(np.count_nonzero(candidate_correct)),
        "fixed": fixed_count,
        "broken": broken_count,
        "net_correct": fixed_count - broken_count,
        "changed": int(np.count_nonzero(changed)),
    }


def _validate_phase_predictions(
    k1: np.ndarray,
    k2: np.ndarray,
    k4: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k1, target, valid = _validate_semantic_arrays(k1, target, num_classes)
    validated_candidates = []
    for name, candidate in (("k2", k2), ("k4", k4)):
        candidate = np.asarray(candidate)
        if candidate.ndim != 2 or candidate.shape != target.shape:
            raise ValueError(f"{name} prediction shape differs from target")
        if (
            not np.issubdtype(candidate.dtype, np.integer)
            or candidate.dtype == np.bool_
        ):
            raise TypeError(f"{name} prediction must contain integer class indices")
        candidate = np.ascontiguousarray(candidate)
        invalid_prediction = valid & (
            (candidate < 0) | (candidate >= int(num_classes))
        )
        if np.any(invalid_prediction):
            values = np.unique(candidate[invalid_prediction])
            raise ValueError(
                f"{name} prediction contains out-of-range values on valid target "
                f"pixels: {values[:10].tolist()}"
            )
        validated_candidates.append(candidate)
    k2, k4 = validated_candidates
    return k1, k2, k4, target, valid


def _confusion_on_mask(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    encoded = (
        target[mask].astype(np.int64, copy=False) * num_classes
        + prediction[mask].astype(np.int64, copy=False)
    )
    return np.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def cell_phase_statistics(
    cell_bounds: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    k4: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    *,
    num_cells: int | None = None,
    valid_mask: np.ndarray | None = None,
    small_mask: np.ndarray | None = None,
    thin_mask: np.ndarray | None = None,
) -> tuple[dict[str, Any], ...]:
    """Compute per-cell K1/K2/K4 confusions and fix/break statistics.

    Statistics are streamed one ownership rectangle at a time.  No H x W
    owner map and no N x H x W mask stack is constructed.  ``valid_mask`` can
    restrict the audit to common phase support; real outer cells that then
    contain zero evaluated pixels are still emitted and remain in the cost
    denominator.
    """

    k1, k2, k4, target, label_valid = _validate_phase_predictions(
        k1, k2, k4, target, num_classes
    )
    bounds = _validated_cell_bounds(cell_bounds, target.shape)
    if num_cells is not None and _require_positive_int(
        num_cells, name="num_cells"
    ) != len(bounds):
        raise ValueError("num_cells must equal the number of ownership bounds")
    num_cells = len(bounds)

    if valid_mask is not None:
        label_valid &= _validated_bool_mask(
            valid_mask, target.shape, name="valid_mask", default=True
        )
    small = (
        _validated_bool_mask(
            small_mask, target.shape, name="small_mask", default=False
        )
        if small_mask is not None
        else None
    )
    thin = (
        _validated_bool_mask(
            thin_mask, target.shape, name="thin_mask", default=False
        )
        if thin_mask is not None
        else None
    )
    predictions = {"k1": k1, "k2": k2, "k4": k4}
    transition_pairs = {
        "k1_to_k2": (k1, k2),
        "k1_to_k4": (k1, k4),
        "k2_to_k4": (k2, k4),
    }

    records: list[dict[str, Any]] = []
    for cell_index, (y0, y1, x0, x1) in enumerate(bounds):
        cell_slice = np.s_[y0:y1, x0:x1]
        cell_target = target[cell_slice]
        cell_valid = label_valid[cell_slice]
        cell_small = (
            small[cell_slice]
            if small is not None
            else np.zeros(cell_valid.shape, dtype=bool)
        )
        cell_thin = (
            thin[cell_slice]
            if thin is not None
            else np.zeros(cell_valid.shape, dtype=bool)
        )
        region_masks = {
            "all": cell_valid,
            "small": cell_valid & cell_small,
            "thin": cell_valid & cell_thin,
        }
        cell_predictions = {
            level: prediction[cell_slice]
            for level, prediction in predictions.items()
        }
        confusions = {
            level: _confusion_on_mask(
                prediction,
                cell_target,
                cell_valid,
                int(num_classes),
            )
            for level, prediction in cell_predictions.items()
        }

        record_regions = {}
        for region_name, region_mask in region_masks.items():
            pixels = int(np.count_nonzero(region_mask))
            correct = {
                level: int(
                    np.count_nonzero(region_mask & (prediction == cell_target))
                )
                for level, prediction in cell_predictions.items()
            }
            record_regions[region_name] = {
                "pixels": pixels,
                "correct": correct,
                "errors": {level: pixels - value for level, value in correct.items()},
            }

        record_transitions = {}
        for transition_name, (reference, candidate) in transition_pairs.items():
            cell_reference = reference[cell_slice]
            cell_candidate = candidate[cell_slice]
            reference_correct = cell_reference == cell_target
            candidate_correct = cell_candidate == cell_target
            transition_regions = {}
            for region_name, region_mask in region_masks.items():
                fixed = int(
                    np.count_nonzero(
                        region_mask & (~reference_correct) & candidate_correct
                    )
                )
                broken = int(
                    np.count_nonzero(
                        region_mask & reference_correct & (~candidate_correct)
                    )
                )
                transition_regions[region_name] = {
                    "pixels": record_regions[region_name]["pixels"],
                    "reference_correct": int(
                        np.count_nonzero(region_mask & reference_correct)
                    ),
                    "candidate_correct": int(
                        np.count_nonzero(region_mask & candidate_correct)
                    ),
                    "fixed": fixed,
                    "broken": broken,
                    "net_correct": fixed - broken,
                    "changed": int(
                        np.count_nonzero(
                            region_mask & (cell_reference != cell_candidate)
                        )
                    ),
                }
            record_transitions[transition_name] = transition_regions
        records.append(
            {
                "cell_index": cell_index,
                "bounds": (int(y0), int(y1), int(x0), int(x1)),
                "pixels": record_regions["all"]["pixels"],
                "confusion": confusions,
                "regions": record_regions,
                "transitions": record_transitions,
            }
        )
    return tuple(records)


def _validated_cell_records(
    cell_stats: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    records = tuple(cell_stats)
    if not records:
        raise ValueError("cell_stats must not be empty")
    indices = []
    for record in records:
        if not isinstance(record, dict) or "cell_index" not in record:
            raise TypeError("each cell_stats entry must be a cell-statistics dictionary")
        indices.append(record["cell_index"])
    if indices != list(range(len(records))):
        raise ValueError("cell_stats must be ordered with contiguous cell indices")
    return records


def _validated_levels(levels_by_cell: np.ndarray | Sequence[int], num_cells: int) -> np.ndarray:
    levels = np.asarray(levels_by_cell)
    if levels.ndim != 1 or len(levels) != num_cells:
        raise ValueError(f"levels_by_cell must have shape ({num_cells},)")
    if not np.issubdtype(levels.dtype, np.integer) or levels.dtype == np.bool_:
        raise TypeError("levels_by_cell must contain integer phase levels")
    levels = np.ascontiguousarray(levels, dtype=np.int64)
    if np.any(~np.isin(levels, PHASE_LEVELS)):
        raise ValueError("levels_by_cell values must be one of 1, 2, or 4")
    return levels


def aggregate_cell_assignment(
    cell_stats: Sequence[dict[str, Any]],
    levels_by_cell: np.ndarray | Sequence[int],
    *,
    num_classes: int,
) -> dict[str, Any]:
    """Aggregate one per-cell phase assignment without rebuilding an image."""

    records = _validated_cell_records(cell_stats)
    num_classes = _require_positive_int(num_classes, name="num_classes")
    levels = _validated_levels(levels_by_cell, len(records))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    region_totals = {
        name: {"pixels": 0, "errors": 0} for name in REGION_NAMES
    }
    for record, level in zip(records, levels, strict=True):
        level_name = f"k{int(level)}"
        cell_confusion = np.asarray(record["confusion"][level_name])
        if cell_confusion.shape != confusion.shape:
            raise ValueError("cell confusion shape does not match num_classes")
        if not np.issubdtype(cell_confusion.dtype, np.integer):
            raise TypeError("cell confusions must contain integer counts")
        if np.any(cell_confusion < 0):
            raise ValueError("cell confusions must be non-negative")
        confusion += cell_confusion.astype(np.int64, copy=False)
        for region_name in REGION_NAMES:
            region = record["regions"][region_name]
            region_totals[region_name]["pixels"] += int(region["pixels"])
            region_totals[region_name]["errors"] += int(
                region["errors"][level_name]
            )

    regions = {}
    for region_name, totals in region_totals.items():
        pixels = totals["pixels"]
        errors = totals["errors"]
        if pixels < 0 or errors < 0 or errors > pixels:
            raise ValueError("cell region counts are inconsistent")
        regions[region_name] = {
            "pixels": pixels,
            "errors": errors,
            "error_rate": float(errors / pixels) if pixels else None,
        }
    return {
        "confusion": confusion,
        "miou": mean_iou_from_confusion(confusion),
        "regions": regions,
    }


def reconstruct_mixed_prediction(
    cell_bounds: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    k4: np.ndarray,
    levels_by_cell: np.ndarray | Sequence[int],
) -> np.ndarray:
    """Build a mixed prediction by filling each ownership rectangle once.

    This deployment-safe operation has no target or ground-truth argument.
    """

    predictions = [np.asarray(value) for value in (k1, k2, k4)]
    if any(value.ndim != 2 for value in predictions):
        raise ValueError("phase predictions must all be 2-D")
    if any(value.shape != predictions[0].shape for value in predictions[1:]):
        raise ValueError("phase prediction shapes must match")
    if any(
        not np.issubdtype(value.dtype, np.integer) or value.dtype == np.bool_
        for value in predictions
    ):
        raise TypeError("phase predictions must contain integer class indices")
    bounds = _validated_cell_bounds(cell_bounds, predictions[0].shape)
    levels = _validated_levels(levels_by_cell, len(bounds))

    mixed = predictions[0].copy()
    source_by_level = {1: predictions[0], 2: predictions[1], 4: predictions[2]}
    for (y0, y1, x0, x1), level in zip(bounds, levels, strict=True):
        if level == 1:
            continue
        cell_slice = np.s_[y0:y1, x0:x1]
        mixed[cell_slice] = source_by_level[int(level)][cell_slice]
    return mixed


@dataclass(frozen=True)
class CellScoreVector:
    """Cell scores with explicit ground-truth provenance."""

    name: str
    values: np.ndarray
    uses_ground_truth: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("score name must be a non-empty string")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("score values must be a non-empty 1-D array")
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise ValueError("score values may be finite or -inf, but not NaN/+inf")
        if not isinstance(self.uses_ground_truth, (bool, np.bool_)):
            raise TypeError("uses_ground_truth must be bool")
        values = np.ascontiguousarray(values.copy())
        values.flags.writeable = False
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "uses_ground_truth", bool(self.uses_ground_truth))


def deployment_score_vector(
    name: str, values: np.ndarray | Sequence[float]
) -> CellScoreVector:
    """Declare a score vector whose construction did not use ground truth."""

    return CellScoreVector(
        name=name, values=np.asarray(values, dtype=np.float64), uses_ground_truth=False
    )


def oracle_net_correct_scores(
    cell_stats: Sequence[dict[str, Any]],
    *,
    transition: str = "k1_to_k4",
    region: str = "all",
) -> CellScoreVector:
    """Return GT-dependent per-cell fixed-minus-broken pixel scores."""

    records = _validated_cell_records(cell_stats)
    if transition not in ("k1_to_k2", "k1_to_k4", "k2_to_k4"):
        raise ValueError("unsupported transition")
    if region not in REGION_NAMES:
        raise ValueError("region must be all, small, or thin")
    values = np.array(
        [record["transitions"][transition][region]["net_correct"] for record in records],
        dtype=np.float64,
    )
    return CellScoreVector(
        name=f"oracle_{transition}_{region}_net_correct",
        values=values,
        uses_ground_truth=True,
    )


def oracle_cell_miou_gain_scores(
    cell_stats: Sequence[dict[str, Any]],
    *,
    num_classes: int,
    reference_level: int = 1,
    candidate_level: int = 4,
    reference_confusion: np.ndarray | None = None,
) -> CellScoreVector:
    """Return GT-dependent global-mIoU gain from switching one cell at a time.

    ``reference_confusion`` should be the full-image reference confusion.  The
    cell confusions may cover only common phase support; each singleton switch
    then applies ``full_reference + cell_candidate - cell_reference``.  If the
    full confusion is omitted, the reference is limited to the union of cells,
    which is useful for unit tests but not the registered full-image audit.
    """

    records = _validated_cell_records(cell_stats)
    num_classes = _require_positive_int(num_classes, name="num_classes")
    if reference_level not in PHASE_LEVELS or candidate_level not in PHASE_LEVELS:
        raise ValueError("phase levels must be one of 1, 2, or 4")
    if reference_level == candidate_level:
        raise ValueError("reference and candidate levels must differ")
    reference_name = f"k{reference_level}"
    candidate_name = f"k{candidate_level}"
    cell_reference = np.zeros((num_classes, num_classes), dtype=np.int64)
    for record in records:
        cell_reference += np.asarray(
            record["confusion"][reference_name], dtype=np.int64
        )
    if reference_confusion is None:
        baseline = cell_reference
    else:
        raw_reference = np.asarray(reference_confusion)
        if raw_reference.shape != (num_classes, num_classes):
            raise ValueError("reference_confusion shape does not match num_classes")
        if not np.issubdtype(raw_reference.dtype, np.integer):
            raise TypeError("reference_confusion must contain integer counts")
        if np.any(raw_reference < 0):
            raise ValueError("reference_confusion must be non-negative")
        baseline = raw_reference.astype(np.int64, copy=True)
        if np.any(baseline < cell_reference):
            raise ValueError(
                "reference_confusion must contain all cell reference confusion"
            )
    baseline_miou = mean_iou_from_confusion(baseline)
    if not np.isfinite(baseline_miou):
        raise ValueError("oracle mIoU scores require at least one supported class")
    values = []
    for record in records:
        switched = (
            baseline
            - np.asarray(record["confusion"][reference_name], dtype=np.int64)
            + np.asarray(record["confusion"][candidate_name], dtype=np.int64)
        )
        values.append(mean_iou_from_confusion(switched) - baseline_miou)
    return CellScoreVector(
        name=f"oracle_k{reference_level}_to_k{candidate_level}_miou_gain",
        values=np.asarray(values, dtype=np.float64),
        uses_ground_truth=True,
    )


def _selection_count(total_cells: int, q: float) -> int:
    total_cells = _require_positive_int(total_cells, name="total_cells")
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, (int, float, np.number)):
        raise TypeError("q must be numeric")
    q = float(q)
    if not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError("q must be finite and lie in [0, 1]")
    if q == 1.0:
        return total_cells
    # Floor keeps the realized selection within the requested compute budget.
    return int(math.floor(q * total_cells + 1e-12))


def deterministic_top_q_indices(
    scores: CellScoreVector,
    q: float,
    *,
    deployment: bool = False,
) -> np.ndarray:
    """Select exactly ``floor(q*N)`` cells, breaking ties by lower index."""

    if not isinstance(scores, CellScoreVector):
        raise TypeError("scores must be a CellScoreVector with explicit provenance")
    if not isinstance(deployment, (bool, np.bool_)):
        raise TypeError("deployment must be bool")
    if deployment and scores.uses_ground_truth:
        raise ValueError("deployment selection must not use ground-truth-derived scores")
    count = _selection_count(len(scores.values), q)
    indices = np.arange(len(scores.values), dtype=np.int64)
    order = np.lexsort((indices, -scores.values))
    return order[:count].astype(np.int64, copy=True)


def deterministic_random_indices(
    total_cells: int,
    q: float,
    *,
    seed: int,
    replicate: int = 0,
    eligible_mask: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Return a reproducible without-replacement random-control selection.

    When ``eligible_mask`` is supplied, geometry-eligible cells are sampled
    first.  The selected count and cost denominator still use every real
    window.  If the requested count exceeds the eligible pool, all eligible
    cells are retained and the remainder is sampled from ineligible cells.
    This lets random controls share public support geometry with a router
    without using ground truth or future phase outputs.
    """

    total_cells = _require_positive_int(total_cells, name="total_cells")
    seed = _require_nonnegative_int(seed, name="seed")
    replicate = _require_nonnegative_int(replicate, name="replicate")
    count = _selection_count(total_cells, q)
    seed_sequence = np.random.SeedSequence([seed, replicate])
    generator = np.random.default_rng(seed_sequence)
    if eligible_mask is None:
        selected = generator.choice(total_cells, size=count, replace=False)
        return np.sort(selected.astype(np.int64, copy=False))

    raw_eligible = np.asarray(eligible_mask)
    if raw_eligible.shape != (total_cells,) or raw_eligible.dtype != np.bool_:
        raise TypeError(
            f"eligible_mask must be a bool vector with shape ({total_cells},)"
        )
    eligible = np.flatnonzero(raw_eligible).astype(np.int64, copy=False)
    ineligible = np.flatnonzero(~raw_eligible).astype(np.int64, copy=False)
    eligible_count = min(count, len(eligible))
    selected_parts = [
        generator.choice(eligible, size=eligible_count, replace=False)
        if eligible_count
        else np.empty(0, dtype=np.int64)
    ]
    remainder = count - eligible_count
    if remainder:
        selected_parts.append(
            generator.choice(ineligible, size=remainder, replace=False)
        )
    selected = np.concatenate(selected_parts)
    return np.sort(selected.astype(np.int64, copy=False))


def binary_phase_cost(selected_count: int, total_cells: int) -> dict[str, Any]:
    """Return binary K1/K4 compute: ``1 + 3 * selected_fraction``."""

    selected_count = _require_nonnegative_int(
        selected_count, name="selected_count"
    )
    total_cells = _require_positive_int(total_cells, name="total_cells")
    if selected_count > total_cells:
        raise ValueError("selected_count must not exceed total_cells")
    q4 = selected_count / total_cells
    return {
        "scheme": "binary_k1_k4",
        "selected_count": selected_count,
        "total_cells": total_cells,
        "q4": float(q4),
        "forward_equivalent_cost": float(1.0 + 3.0 * q4),
    }


def hierarchical_phase_cost(
    k2_or_k4_count: int,
    k4_count: int,
    total_cells: int,
) -> dict[str, Any]:
    """Return nested K1/K2/K4 compute: ``1 + q2 + 2*q4``.

    ``k2_or_k4_count`` counts all cells receiving at least K2 compute, and
    ``k4_count`` is the nested subset promoted from K2 to K4.
    """

    k2_or_k4_count = _require_nonnegative_int(
        k2_or_k4_count, name="k2_or_k4_count"
    )
    k4_count = _require_nonnegative_int(k4_count, name="k4_count")
    total_cells = _require_positive_int(total_cells, name="total_cells")
    if k2_or_k4_count > total_cells:
        raise ValueError("k2_or_k4_count must not exceed total_cells")
    if k4_count > k2_or_k4_count:
        raise ValueError("K4 cells must be a subset of K2-or-K4 cells")
    q2 = k2_or_k4_count / total_cells
    q4 = k4_count / total_cells
    return {
        "scheme": "hierarchical_k1_k2_k4",
        "k2_or_k4_count": k2_or_k4_count,
        "k4_count": k4_count,
        "total_cells": total_cells,
        "q2": float(q2),
        "q4": float(q4),
        "forward_equivalent_cost": float(1.0 + q2 + 2.0 * q4),
    }


def gain_retention(k1_miou: float, k4_miou: float, mixed_miou: float) -> float:
    """Return the fraction of positive full-K4 gain retained by a mixture."""

    values = tuple(float(value) for value in (k1_miou, k4_miou, mixed_miou))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("mIoU values must be finite")
    denominator = values[1] - values[0]
    if denominator <= 0.0:
        raise ValueError("gain retention requires K4 mIoU to exceed K1 mIoU")
    return float((values[2] - values[0]) / denominator)


@dataclass(frozen=True)
class PhaseUtilityGateThresholds:
    """Preregistered thresholds for promoting a selective phase policy."""

    max_forward_equivalent_cost: float = 2.0
    min_k4_gain_retention: float = 0.70
    min_miou_delta_over_k2: float = 0.0
    max_small_error_rate_delta: float = 0.0
    max_thin_error_rate_delta: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.max_forward_equivalent_cost,
            self.min_k4_gain_retention,
            self.min_miou_delta_over_k2,
            self.max_small_error_rate_delta,
            self.max_thin_error_rate_delta,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("gate thresholds must be finite")
        if self.max_forward_equivalent_cost < 1.0:
            raise ValueError("maximum forward-equivalent cost must be at least one")
        if not 0.0 <= self.min_k4_gain_retention <= 1.0:
            raise ValueError("minimum K4-gain retention must lie in [0, 1]")
        if self.max_small_error_rate_delta < 0.0:
            raise ValueError("small-region error tolerance must be non-negative")
        if self.max_thin_error_rate_delta < 0.0:
            raise ValueError("thin-region error tolerance must be non-negative")


def _finite_float(value: float | None, *, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def evaluate_phase_utility_gate(
    *,
    k1_miou: float,
    k2_miou: float,
    k4_miou: float,
    mixed_miou: float,
    forward_equivalent_cost: float,
    k2_small_error_rate: float | None,
    mixed_small_error_rate: float | None,
    k2_thin_error_rate: float | None,
    mixed_thin_error_rate: float | None,
    random_p95_miou: float,
    thresholds: PhaseUtilityGateThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate the fixed cost/retention/K2/structure/random promotion gate.

    All mIoUs must use one common unit (fractions or percentages); all error
    rates likewise use one common unit.  Missing small/thin support fails
    closed because non-degradation cannot then be established.
    ``random_p95_miou`` must come from an equal-count, equal-proxy-cost random
    routing control under the runner's fixed seed and replicate protocol.
    """

    thresholds = thresholds or PhaseUtilityGateThresholds()
    if not isinstance(thresholds, PhaseUtilityGateThresholds):
        raise TypeError("thresholds must be PhaseUtilityGateThresholds")
    k1 = _finite_float(k1_miou, name="k1_miou")
    k2 = _finite_float(k2_miou, name="k2_miou")
    k4 = _finite_float(k4_miou, name="k4_miou")
    mixed = _finite_float(mixed_miou, name="mixed_miou")
    cost = _finite_float(
        forward_equivalent_cost, name="forward_equivalent_cost"
    )
    random_p95 = _finite_float(random_p95_miou, name="random_p95_miou")
    tolerance = 1e-12

    positive_k4_gain = k4 > k1 + tolerance
    retention = (
        gain_retention(k1, k4, mixed) if positive_k4_gain else None
    )
    k2_delta = mixed - k2

    try:
        k2_small = _finite_float(
            k2_small_error_rate, name="k2_small_error_rate"
        )
        mixed_small = _finite_float(
            mixed_small_error_rate, name="mixed_small_error_rate"
        )
        small_delta: float | None = mixed_small - k2_small
    except (TypeError, ValueError):
        small_delta = None
    try:
        k2_thin = _finite_float(k2_thin_error_rate, name="k2_thin_error_rate")
        mixed_thin = _finite_float(
            mixed_thin_error_rate, name="mixed_thin_error_rate"
        )
        thin_delta: float | None = mixed_thin - k2_thin
    except (TypeError, ValueError):
        thin_delta = None

    checks = {
        "positive_full_k4_gain": positive_k4_gain,
        "cost_within_budget": (
            cost <= thresholds.max_forward_equivalent_cost + tolerance
        ),
        "retains_k4_gain": (
            retention is not None
            and retention + tolerance >= thresholds.min_k4_gain_retention
        ),
        "outperforms_uniform_k2": (
            k2_delta > thresholds.min_miou_delta_over_k2 + tolerance
        ),
        "small_not_worse_than_k2": (
            small_delta is not None
            and small_delta
            <= thresholds.max_small_error_rate_delta + tolerance
        ),
        "thin_not_worse_than_k2": (
            thin_delta is not None
            and thin_delta <= thresholds.max_thin_error_rate_delta + tolerance
        ),
        "outperforms_equal_budget_random_p95": mixed > random_p95 + tolerance,
    }
    passed = all(checks.values())
    return {
        "outcome": "GO" if passed else "NO_GO",
        "passed": passed,
        "checks": checks,
        "observed": {
            "forward_equivalent_cost": cost,
            "k4_gain_retention": retention,
            "miou_delta_over_k2": float(k2_delta),
            "small_error_rate_delta": small_delta,
            "thin_error_rate_delta": thin_delta,
            "random_p95_miou": random_p95,
            "miou_delta_over_random_p95": float(mixed - random_p95),
        },
        "thresholds": {
            "max_forward_equivalent_cost": thresholds.max_forward_equivalent_cost,
            "min_k4_gain_retention": thresholds.min_k4_gain_retention,
            "min_miou_delta_over_k2": thresholds.min_miou_delta_over_k2,
            "max_small_error_rate_delta": thresholds.max_small_error_rate_delta,
            "max_thin_error_rate_delta": thresholds.max_thin_error_rate_delta,
        },
    }


def binary_selection_curve(
    cell_stats: Sequence[dict[str, Any]],
    scores: CellScoreVector,
    q_values: Iterable[float],
    *,
    num_classes: int,
    deployment: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Evaluate a deterministic K1/K4 curve from cell statistics only."""

    records = _validated_cell_records(cell_stats)
    if len(scores.values) != len(records):
        raise ValueError("score length differs from cell_stats")
    curve = []
    for q in q_values:
        selected = deterministic_top_q_indices(
            scores, q, deployment=deployment
        )
        levels = np.ones(len(records), dtype=np.int64)
        levels[selected] = 4
        aggregate = aggregate_cell_assignment(
            records, levels, num_classes=num_classes
        )
        cost = binary_phase_cost(len(selected), len(records))
        curve.append(
            {
                "requested_q": float(q),
                "selected_indices": selected,
                "levels_by_cell": levels,
                "cost": cost,
                "aggregate": aggregate,
            }
        )
    return tuple(curve)
