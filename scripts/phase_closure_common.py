"""Exact shifted-phase crop closure for Stage-B routing audits.

Stage A assigns one K1/K2/K4 level to each midpoint-owned sliding-window
cell.  Those cell costs are only a post-aggregation proxy: a routed cell
depends on every shifted-canvas crop whose output contributes to any pixel of
the routed part of that cell.  This module computes that dependency closure
without running the model and provides the exact-cost A2 greedy and
random-priority controls frozen before the formal B0 artifact is generated.

The geometry is deliberately expressed in half-open rectangles.  For each
cell and phase, the routed region is ``ownership intersect common support``;
the whole region is translated by the phase shift, then every same-phase
window with positive-area intersection is included.  Invalid-label holes do
not reduce the dependency closure.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PHASE_NAMES = ("x8", "y8", "xy8")
PHASE_SHIFTS = {
    "x8": (0, 8),
    "y8": (8, 0),
    "xy8": (8, 8),
}
PHASE_LEVELS = (1, 2, 4)
SEALED_CROP_SIZE = (512, 512)
SEALED_STRIDE = (341, 341)


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _rectangle(value: Any, *, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a four-element sequence")
    if len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    y0, y1, x0, x1 = (
        _integer(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if y0 < 0 or x0 < 0 or y0 >= y1 or x0 >= x1:
        raise ValueError(f"{name} must be a positive half-open rectangle")
    return y0, y1, x0, x1


def _rectangles_intersect(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    return (
        max(first[0], second[0]) < min(first[1], second[1])
        and max(first[2], second[2]) < min(first[3], second[3])
    )


def _clip_rectangle(
    rectangle: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    clipped = (
        max(rectangle[0], bounds[0]),
        min(rectangle[1], bounds[1]),
        max(rectangle[2], bounds[2]),
        min(rectangle[3], bounds[3]),
    )
    if clipped[0] >= clipped[1] or clipped[2] >= clipped[3]:
        return None
    return clipped


def _common_bounds(image: Mapping[str, Any]) -> tuple[int, int, int, int]:
    raw = image.get("common_bounds")
    if not isinstance(raw, Mapping):
        raise TypeError("image common_bounds must be a mapping")
    return _rectangle(
        (raw.get("y_start"), raw.get("y_stop"), raw.get("x_start"), raw.get("x_stop")),
        name="common_bounds",
    )


def _bit_indices(mask: int) -> list[int]:
    result = []
    remaining = int(mask)
    while remaining:
        lowest = remaining & -remaining
        result.append(lowest.bit_length() - 1)
        remaining ^= lowest
    return result


def _mask_digest(mask: int) -> str:
    crop_ids = np.asarray(_bit_indices(mask), dtype="<i8")
    return hashlib.sha256(crop_ids.tobytes()).hexdigest()


def _slide_axis_starts(length: int, crop: int, stride: int) -> tuple[int, ...]:
    if length < crop:
        raise ValueError("image axis is smaller than the sealed crop")
    count = max(length - crop + stride - 1, 0) // stride + 1
    starts = []
    for index in range(count):
        start = index * stride
        stop = min(start + crop, length)
        starts.append(max(stop - crop, 0))
    return tuple(starts)


def _midpoint_intervals(
    length: int, starts: Sequence[int], crop: int
) -> tuple[tuple[int, int], ...]:
    values = tuple(int(value) for value in starts)
    if not values or values[0] != 0 or values[-1] + crop != length:
        raise ValueError("crop starts do not include both sealed slide endpoints")
    boundaries = [0]
    boundaries.extend(
        (left + crop + right) // 2
        for left, right in zip(values[:-1], values[1:], strict=True)
    )
    boundaries.append(length)
    intervals = tuple(zip(boundaries[:-1], boundaries[1:], strict=True))
    if any(start >= stop for start, stop in intervals):
        raise ValueError("midpoint ownership contains an empty interval")
    return intervals


def build_phase_closure_geometry(
    images: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    *,
    phase_shifts: Mapping[str, tuple[int, int]] = PHASE_SHIFTS,
    crop_size: tuple[int, int] = SEALED_CROP_SIZE,
    stride: tuple[int, int] = SEALED_STRIDE,
) -> dict[str, Any]:
    """Validate a Stage-A manifest and build per-cell crop-dependency masks."""

    image_records = tuple(images)
    cell_records = tuple(cells)
    if not image_records or not cell_records:
        raise ValueError("images and cells must not be empty")
    if tuple(phase_shifts) != PHASE_NAMES:
        raise ValueError(f"phase_shifts must preserve order {PHASE_NAMES}")
    shifts = {
        name: (
            _integer(shift[0], name=f"{name}.dy", minimum=0),
            _integer(shift[1], name=f"{name}.dx", minimum=0),
        )
        for name, shift in phase_shifts.items()
    }
    if not isinstance(crop_size, tuple) or len(crop_size) != 2:
        raise TypeError("crop_size must be a two-integer tuple")
    if not isinstance(stride, tuple) or len(stride) != 2:
        raise TypeError("stride must be a two-integer tuple")
    crop_height = _integer(crop_size[0], name="crop height", minimum=1)
    crop_width = _integer(crop_size[1], name="crop width", minimum=1)
    stride_height = _integer(stride[0], name="stride height", minimum=1)
    stride_width = _integer(stride[1], name="stride width", minimum=1)

    sample_names: list[str] = []
    image_shapes: list[tuple[int, int]] = []
    common_bounds: list[tuple[int, int, int, int]] = []
    baseline_crop_counts: list[int] = []
    row_starts_by_image: list[tuple[int, ...]] = []
    column_starts_by_image: list[tuple[int, ...]] = []
    cells_by_image: list[list[int]] = [[] for _ in image_records]
    for expected_index, image in enumerate(image_records):
        image_index = _integer(
            image.get("loader_position"), name="image.loader_position", minimum=0
        )
        if image_index != expected_index:
            raise ValueError("images must be ordered by contiguous loader_position")
        sample_name = str(image.get("sample_name", ""))
        if not sample_name:
            raise ValueError("every image must have a sample_name")
        shape = image.get("full_shape_hw")
        if not isinstance(shape, Sequence) or len(shape) != 2:
            raise ValueError("image full_shape_hw must contain height and width")
        height = _integer(shape[0], name="image height", minimum=1)
        width = _integer(shape[1], name="image width", minimum=1)
        bounds = _common_bounds(image)
        if bounds[1] > height or bounds[3] > width:
            raise ValueError("common support lies outside the full image")
        grid = image.get("crop_grid")
        if not isinstance(grid, Mapping):
            raise TypeError("image crop_grid must be a mapping")
        crop_count = _integer(
            grid.get("crop_count"), name="crop_grid.crop_count", minimum=1
        )
        row_count = _integer(grid.get("rows"), name="crop_grid.rows", minimum=1)
        column_count = _integer(
            grid.get("columns"), name="crop_grid.columns", minimum=1
        )
        raw_row_starts = grid.get("row_starts")
        raw_column_starts = grid.get("column_starts")
        if not isinstance(raw_row_starts, Sequence) or isinstance(
            raw_row_starts, (str, bytes)
        ):
            raise TypeError("crop_grid.row_starts must be a sequence")
        if not isinstance(raw_column_starts, Sequence) or isinstance(
            raw_column_starts, (str, bytes)
        ):
            raise TypeError("crop_grid.column_starts must be a sequence")
        row_starts = tuple(
            _integer(value, name="crop_grid.row_start", minimum=0)
            for value in raw_row_starts
        )
        column_starts = tuple(
            _integer(value, name="crop_grid.column_start", minimum=0)
            for value in raw_column_starts
        )
        if len(row_starts) != row_count or len(column_starts) != column_count:
            raise ValueError("crop-grid dimensions differ from their start arrays")
        if crop_count != row_count * column_count:
            raise ValueError("crop count differs from row/column Cartesian product")
        if tuple(sorted(set(row_starts))) != row_starts or tuple(
            sorted(set(column_starts))
        ) != column_starts:
            raise ValueError("crop starts must be strictly increasing")
        expected_row_starts = _slide_axis_starts(
            height, crop_height, stride_height
        )
        expected_column_starts = _slide_axis_starts(
            width, crop_width, stride_width
        )
        if row_starts != expected_row_starts or column_starts != expected_column_starts:
            raise ValueError("crop starts differ from sealed slide end-backtracking")
        sample_names.append(sample_name)
        image_shapes.append((height, width))
        common_bounds.append(bounds)
        baseline_crop_counts.append(crop_count)
        row_starts_by_image.append(row_starts)
        column_starts_by_image.append(column_starts)

    if [cell.get("cell_index") for cell in cell_records] != list(
        range(len(cell_records))
    ):
        raise ValueError("cells must be ordered by contiguous global cell_index")

    image_ids = np.empty(len(cell_records), dtype=np.int64)
    local_ids = np.empty(len(cell_records), dtype=np.int64)
    eligible = np.empty(len(cell_records), dtype=bool)
    ownerships: list[tuple[int, int, int, int]] = []
    windows_by_image: list[list[tuple[int, int, int, int] | None]] = [
        [None] * count for count in baseline_crop_counts
    ]
    for index, cell in enumerate(cell_records):
        image_index = _integer(
            cell.get("image_index"), name="cell.image_index", minimum=0
        )
        if image_index >= len(image_records):
            raise ValueError("cell image_index is out of range")
        if str(cell.get("sample_name", "")) != sample_names[image_index]:
            raise ValueError("cell sample_name differs from its image")
        local_id = _integer(
            cell.get("local_crop_id"), name="cell.local_crop_id", minimum=0
        )
        if local_id >= baseline_crop_counts[image_index]:
            raise ValueError("cell local_crop_id is out of range")
        if windows_by_image[image_index][local_id] is not None:
            raise ValueError("duplicate local crop id inside one image")
        window = _rectangle(cell.get("window_yxyx"), name="cell.window_yxyx")
        ownership = _rectangle(
            cell.get("ownership_yxyx"), name="cell.ownership_yxyx"
        )
        height, width = image_shapes[image_index]
        if window[1] > height or window[3] > width:
            raise ValueError("cell source window lies outside its image")
        if (
            ownership[0] < window[0]
            or ownership[1] > window[1]
            or ownership[2] < window[2]
            or ownership[3] > window[3]
        ):
            raise ValueError("cell ownership lies outside its source window")
        routed = _clip_rectangle(ownership, common_bounds[image_index])
        raw_eligible = cell.get("geometry_eligible")
        if not isinstance(raw_eligible, (bool, np.bool_)):
            raise TypeError("cell geometry_eligible must be boolean")
        if bool(raw_eligible) != (routed is not None):
            raise ValueError("geometry eligibility differs from common-support overlap")
        image_ids[index] = image_index
        local_ids[index] = local_id
        eligible[index] = bool(raw_eligible)
        ownerships.append(ownership)
        windows_by_image[image_index][local_id] = window
        cells_by_image[image_index].append(index)

    for image_index, windows in enumerate(windows_by_image):
        if any(window is None for window in windows):
            raise ValueError(f"image {image_index} does not contain every crop id")
        if len(cells_by_image[image_index]) != baseline_crop_counts[image_index]:
            raise ValueError("one ownership cell per baseline crop is required")
        concrete_windows = tuple(window for window in windows if window is not None)
        expected_windows = tuple(
            (y0, y0 + crop_height, x0, x0 + crop_width)
            for y0 in row_starts_by_image[image_index]
            for x0 in column_starts_by_image[image_index]
        )
        if concrete_windows != expected_windows:
            raise ValueError(
                "local crop ids/windows differ from the sealed row-major crop manifest"
            )

        row_intervals = _midpoint_intervals(
            image_shapes[image_index][0], row_starts_by_image[image_index], crop_height
        )
        column_intervals = _midpoint_intervals(
            image_shapes[image_index][1],
            column_starts_by_image[image_index],
            crop_width,
        )
        expected_ownerships = tuple(
            (row[0], row[1], column[0], column[1])
            for row in row_intervals
            for column in column_intervals
        )
        actual_ownerships = tuple(
            ownerships[index] for index in cells_by_image[image_index]
        )
        if actual_ownerships != expected_ownerships:
            raise ValueError("ownership rectangles differ from the midpoint rule")

        height, width = image_shapes[image_index]
        group_ownerships = list(actual_ownerships)
        y_edges = sorted(
            {0, height}
            | {rectangle[0] for rectangle in group_ownerships}
            | {rectangle[1] for rectangle in group_ownerships}
        )
        x_edges = sorted(
            {0, width}
            | {rectangle[2] for rectangle in group_ownerships}
            | {rectangle[3] for rectangle in group_ownerships}
        )
        y_codes = {value: index for index, value in enumerate(y_edges)}
        x_codes = {value: index for index, value in enumerate(x_edges)}
        coverage = np.zeros((len(y_edges) - 1, len(x_edges) - 1), dtype=np.int16)
        for y0, y1, x0, x1 in group_ownerships:
            coverage[y_codes[y0] : y_codes[y1], x_codes[x0] : x_codes[x1]] += 1
        if not np.all(coverage == 1):
            raise ValueError("ownership rectangles do not partition the image exactly")

    dependencies = {name: [0] * len(cell_records) for name in PHASE_NAMES}
    routed_rectangles: list[tuple[int, int, int, int] | None] = []
    for index, ownership in enumerate(ownerships):
        image_index = int(image_ids[index])
        routed = _clip_rectangle(ownership, common_bounds[image_index])
        routed_rectangles.append(routed)
        if routed is None:
            continue
        height, width = image_shapes[image_index]
        for phase_name in PHASE_NAMES:
            dy, dx = shifts[phase_name]
            shifted = (
                routed[0] + dy,
                routed[1] + dy,
                routed[2] + dx,
                routed[3] + dx,
            )
            if shifted[1] > height or shifted[3] > width:
                raise ValueError("shifted routed region lies outside its phase canvas")
            mask = 0
            for crop_id, window in enumerate(windows_by_image[image_index]):
                assert window is not None
                if _rectangles_intersect(shifted, window):
                    mask |= 1 << crop_id
            if mask == 0:
                raise AssertionError("eligible cell has no phase-crop dependency")
            dependencies[phase_name][index] = mask

    return {
        "phase_names": PHASE_NAMES,
        "phase_shifts": shifts,
        "crop_size": (crop_height, crop_width),
        "stride": (stride_height, stride_width),
        "sample_names": tuple(sample_names),
        "image_shapes": tuple(image_shapes),
        "common_bounds": tuple(common_bounds),
        "baseline_crop_counts": np.asarray(baseline_crop_counts, dtype=np.int64),
        "image_ids": image_ids,
        "local_crop_ids": local_ids,
        "eligible": eligible,
        "cells_by_image": tuple(tuple(values) for values in cells_by_image),
        "routed_rectangles": tuple(routed_rectangles),
        "windows_by_image": tuple(tuple(values) for values in windows_by_image),
        "dependencies": {
            name: tuple(values) for name, values in dependencies.items()
        },
    }


def validate_levels(
    levels_by_cell: Sequence[int] | np.ndarray,
    geometry: Mapping[str, Any],
) -> np.ndarray:
    levels = np.asarray(levels_by_cell)
    cell_count = len(geometry["image_ids"])
    if levels.ndim != 1 or len(levels) != cell_count:
        raise ValueError(f"levels_by_cell must have shape ({cell_count},)")
    if not np.issubdtype(levels.dtype, np.integer) or levels.dtype == np.bool_:
        raise TypeError("levels_by_cell must contain integers")
    levels = np.ascontiguousarray(levels, dtype=np.int64)
    if np.any(~np.isin(levels, PHASE_LEVELS)):
        raise ValueError("levels must be K1, K2, or K4")
    if np.any((levels != 1) & (~np.asarray(geometry["eligible"], dtype=bool))):
        raise ValueError("a routed level was assigned outside common-support geometry")
    return levels


def closure_masks_from_levels(
    levels_by_cell: Sequence[int] | np.ndarray,
    geometry: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    levels = validate_levels(levels_by_cell, geometry)
    image_count = len(geometry["baseline_crop_counts"])
    masks = {name: [0] * image_count for name in PHASE_NAMES}
    image_ids = geometry["image_ids"]
    dependencies = geometry["dependencies"]
    for cell_index, level in enumerate(levels):
        image_index = int(image_ids[cell_index])
        if level >= 2:
            masks["x8"][image_index] |= dependencies["x8"][cell_index]
        if level == 4:
            masks["y8"][image_index] |= dependencies["y8"][cell_index]
            masks["xy8"][image_index] |= dependencies["xy8"][cell_index]
    return {name: tuple(values) for name, values in masks.items()}


def summarize_closure(
    levels_by_cell: Sequence[int] | np.ndarray,
    geometry: Mapping[str, Any],
    *,
    include_crop_ids: bool = True,
) -> dict[str, Any]:
    """Summarize deduplicated phase-crop forwards for one fixed action map."""

    levels = validate_levels(levels_by_cell, geometry)
    masks = closure_masks_from_levels(levels, geometry)
    baseline_counts = np.asarray(geometry["baseline_crop_counts"], dtype=np.int64)
    image_ids = np.asarray(geometry["image_ids"], dtype=np.int64)
    per_image = []
    for image_index, baseline_count in enumerate(baseline_counts):
        phase_counts = {
            name: int(masks[name][image_index].bit_count()) for name in PHASE_NAMES
        }
        extra = int(sum(phase_counts.values()))
        group_levels = levels[image_ids == image_index]
        record = {
            "image_index": image_index,
            "sample_name": geometry["sample_names"][image_index],
            "baseline_k1_crop_forwards": int(baseline_count),
            "selected_counts": {
                "k1": int(np.count_nonzero(group_levels == 1)),
                "k2": int(np.count_nonzero(group_levels == 2)),
                "k4": int(np.count_nonzero(group_levels == 4)),
            },
            "extra_crop_forwards_by_phase": phase_counts,
            "extra_crop_forwards": extra,
            "forward_equivalent_cost": float(1.0 + extra / baseline_count),
            "phase_crop_id_sha256": {
                name: _mask_digest(masks[name][image_index]) for name in PHASE_NAMES
            },
        }
        if include_crop_ids:
            record["phase_crop_ids"] = {
                name: _bit_indices(masks[name][image_index]) for name in PHASE_NAMES
            }
        per_image.append(record)

    baseline_total = int(baseline_counts.sum())
    phase_totals = {
        name: int(sum(mask.bit_count() for mask in masks[name]))
        for name in PHASE_NAMES
    }
    extra_total = int(sum(phase_totals.values()))
    return {
        "cost_definition": (
            "1 + deduplicated extra (image, phase, crop_id) forwards / all "
            "normal-phase crop forwards"
        ),
        "baseline_k1_crop_forwards": baseline_total,
        "extra_crop_forwards_by_phase": phase_totals,
        "extra_crop_forwards": extra_total,
        "forward_equivalent_cost": float(1.0 + extra_total / baseline_total),
        "selected_counts": {
            "k1": int(np.count_nonzero(levels == 1)),
            "k2": int(np.count_nonzero(levels == 2)),
            "k4": int(np.count_nonzero(levels == 4)),
        },
        "per_image": per_image,
    }


def mean_iou_from_confusion(confusion: np.ndarray) -> float:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion must be a square matrix")
    if np.any(matrix < 0):
        raise ValueError("confusion must be non-negative")
    diagonal = np.diag(matrix)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - diagonal
    supported = union > 0
    if not np.any(supported):
        raise ValueError("confusion has no supported class")
    return float(np.mean(diagonal[supported] / union[supported]))


def _batch_miou(confusions: np.ndarray) -> np.ndarray:
    matrices = np.asarray(confusions, dtype=np.int64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("confusions must have shape [N,C,C]")
    if np.any(matrices < 0):
        raise ValueError("confusions must be non-negative")
    diagonal = np.diagonal(matrices, axis1=1, axis2=2)
    union = matrices.sum(axis=2) + matrices.sum(axis=1) - diagonal
    supported = union > 0
    counts = supported.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("a confusion has no supported class")
    iou = np.zeros(diagonal.shape, dtype=np.float64)
    np.divide(diagonal, union, out=iou, where=supported)
    return iou.sum(axis=1) / counts


def confusion_for_levels(
    cells: Sequence[Mapping[str, Any]],
    levels_by_cell: Sequence[int] | np.ndarray,
    full_k1_confusion: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    levels = np.asarray(levels_by_cell, dtype=np.int64)
    records = tuple(cells)
    if levels.shape != (len(records),):
        raise ValueError("levels length differs from cells")
    result = np.asarray(full_k1_confusion, dtype=np.int64).copy()
    for record, level in zip(records, levels, strict=True):
        confusions = record.get("confusion")
        if not isinstance(confusions, Mapping):
            raise TypeError("cell confusion must be a mapping")
        k1 = np.asarray(confusions["k1"], dtype=np.int64)
        chosen = np.asarray(confusions[f"k{int(level)}"], dtype=np.int64)
        if k1.shape != result.shape or chosen.shape != result.shape:
            raise ValueError("cell confusion shape differs from full confusion")
        result += chosen - k1
    if np.any(result < 0):
        raise AssertionError("mixed assignment produced negative confusion")
    return result


def region_summary_for_levels(
    cells: Sequence[Mapping[str, Any]],
    levels_by_cell: Sequence[int] | np.ndarray,
) -> dict[str, dict[str, Any]] | None:
    """Aggregate optional per-cell region counts; return ``None`` if absent."""

    records = tuple(cells)
    levels = np.asarray(levels_by_cell, dtype=np.int64)
    if levels.shape != (len(records),):
        raise ValueError("levels length differs from cells")
    if not all(isinstance(record.get("regions"), Mapping) for record in records):
        return None
    totals = {
        name: {"pixels": 0, "errors": 0} for name in ("all", "small", "thin")
    }
    for record, level in zip(records, levels, strict=True):
        for name in totals:
            region = record["regions"][name]
            pixels = _integer(region["pixels"], name=f"{name}.pixels", minimum=0)
            errors = _integer(
                region["errors"][f"k{int(level)}"],
                name=f"{name}.errors.k{int(level)}",
                minimum=0,
            )
            if errors > pixels:
                raise ValueError("region errors exceed pixels")
            totals[name]["pixels"] += pixels
            totals[name]["errors"] += errors
    return {
        name: {
            **values,
            "error_rate": (
                float(values["errors"] / values["pixels"])
                if values["pixels"]
                else None
            ),
        }
        for name, values in totals.items()
    }


def _confusion_stacks(
    cells: Sequence[Mapping[str, Any]],
    num_classes: int,
) -> dict[int, np.ndarray]:
    stacks = {}
    for level in PHASE_LEVELS:
        values = []
        for record in cells:
            matrix = np.asarray(record["confusion"][f"k{level}"], dtype=np.int64)
            if matrix.shape != (num_classes, num_classes) or np.any(matrix < 0):
                raise ValueError("cell confusion is invalid")
            values.append(matrix)
        stacks[level] = np.stack(values, axis=0)
    target_counts = stacks[1].sum(axis=2)
    if not np.array_equal(stacks[2].sum(axis=2), target_counts) or not np.array_equal(
        stacks[4].sum(axis=2), target_counts
    ):
        raise ValueError("K1/K2/K4 cell target counts differ")
    return stacks


def exact_cost_hierarchical_oracle(
    cells: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    full_k1_confusion: Sequence[Sequence[int]] | np.ndarray,
    *,
    extra_crop_cap_by_image: Sequence[int] | np.ndarray,
    num_classes: int,
) -> dict[str, Any]:
    """Run the frozen per-image-capped exact-cost A2 constructive oracle.

    Positive zero-cost actions are ranked first by absolute mIoU gain.  Other
    actions are ranked by gain per newly required crop.  Exact ties go to the
    lower global cell index.  K1->K2 bridges with non-positive current gain are
    never crossed.
    """

    records = tuple(cells)
    stacks = _confusion_stacks(records, num_classes)
    image_ids = np.asarray(geometry["image_ids"], dtype=np.int64)
    eligible_mask = np.asarray(geometry["eligible"], dtype=bool)
    image_count = len(geometry["baseline_crop_counts"])
    caps = np.asarray(extra_crop_cap_by_image)
    if caps.shape != (image_count,) or not np.issubdtype(caps.dtype, np.integer):
        raise ValueError("extra_crop_cap_by_image has the wrong shape or dtype")
    caps = np.ascontiguousarray(caps, dtype=np.int64)
    if np.any(caps < 0):
        raise ValueError("exact crop caps must be non-negative")

    levels = np.ones(len(records), dtype=np.int64)
    used_masks = {name: [0] * image_count for name in PHASE_NAMES}
    used_cost = np.zeros(image_count, dtype=np.int64)
    confusion = np.asarray(full_k1_confusion, dtype=np.int64).copy()
    if confusion.shape != (num_classes, num_classes) or np.any(confusion < 0):
        raise ValueError("full_k1_confusion is invalid")
    current_miou = mean_iou_from_confusion(confusion)
    dependencies = geometry["dependencies"]
    actions: list[dict[str, Any]] = []

    while True:
        candidate_indices = np.flatnonzero(eligible_mask & (levels < 4))
        if candidate_indices.size == 0:
            break
        from_levels = levels[candidate_indices]
        to_levels = np.where(from_levels == 1, 2, 4)
        costs = np.empty(len(candidate_indices), dtype=np.int64)
        phase_additions: list[tuple[int, int, int]] = []
        for cell_index, from_level in zip(
            candidate_indices, from_levels, strict=True
        ):
            image_index = int(image_ids[cell_index])
            if from_level == 1:
                add_x = dependencies["x8"][cell_index] & ~used_masks["x8"][
                    image_index
                ]
                additions = (add_x, 0, 0)
            else:
                if dependencies["x8"][cell_index] & ~used_masks["x8"][image_index]:
                    raise AssertionError("K2 cell is missing its x-phase closure")
                add_y = dependencies["y8"][cell_index] & ~used_masks["y8"][
                    image_index
                ]
                add_xy = dependencies["xy8"][cell_index] & ~used_masks["xy8"][
                    image_index
                ]
                additions = (0, add_y, add_xy)
            phase_additions.append(additions)
        for index, additions in enumerate(phase_additions):
            costs[index] = sum(mask.bit_count() for mask in additions)

        fits = used_cost[image_ids[candidate_indices]] + costs <= caps[
            image_ids[candidate_indices]
        ]
        if not np.any(fits):
            break
        candidate_indices = candidate_indices[fits]
        from_levels = from_levels[fits]
        to_levels = to_levels[fits]
        costs = costs[fits]
        phase_additions = [
            additions
            for additions, keep in zip(phase_additions, fits, strict=True)
            if keep
        ]

        deltas = np.empty(
            (len(candidate_indices), num_classes, num_classes), dtype=np.int64
        )
        from_k1 = from_levels == 1
        deltas[from_k1] = (
            stacks[2][candidate_indices[from_k1]]
            - stacks[1][candidate_indices[from_k1]]
        )
        deltas[~from_k1] = (
            stacks[4][candidate_indices[~from_k1]]
            - stacks[2][candidate_indices[~from_k1]]
        )
        candidate_mious = _batch_miou(confusion[None, :, :] + deltas)
        gains = candidate_mious - current_miou
        positive = gains > 0.0
        if not np.any(positive):
            break
        candidate_indices = candidate_indices[positive]
        from_levels = from_levels[positive]
        to_levels = to_levels[positive]
        costs = costs[positive]
        deltas = deltas[positive]
        candidate_mious = candidate_mious[positive]
        gains = gains[positive]
        phase_additions = [
            additions
            for additions, keep in zip(phase_additions, positive, strict=True)
            if keep
        ]

        zero_cost = costs == 0
        if np.any(zero_cost):
            positions = np.flatnonzero(zero_cost)
            order = np.lexsort(
                (candidate_indices[positions], -gains[positions])
            )
            chosen_position = int(positions[order[0]])
            ranking = "positive_zero_cost_then_absolute_gain"
        else:
            ratios = gains / costs
            order = np.lexsort((candidate_indices, -ratios))
            chosen_position = int(order[0])
            ranking = "positive_gain_per_new_crop"

        cell_index = int(candidate_indices[chosen_position])
        image_index = int(image_ids[cell_index])
        from_level = int(from_levels[chosen_position])
        to_level = int(to_levels[chosen_position])
        additions = phase_additions[chosen_position]
        phase_counts = {
            name: int(mask.bit_count())
            for name, mask in zip(PHASE_NAMES, additions, strict=True)
        }
        incremental_cost = int(sum(phase_counts.values()))
        for name, mask in zip(PHASE_NAMES, additions, strict=True):
            used_masks[name][image_index] |= mask
        used_cost[image_index] += incremental_cost
        levels[cell_index] = to_level
        confusion += deltas[chosen_position]
        next_miou = float(candidate_mious[chosen_position])
        actions.append(
            {
                "step": len(actions) + 1,
                "cell_index": cell_index,
                "image_index": image_index,
                "from_level": from_level,
                "to_level": to_level,
                "ranking": ranking,
                "incremental_extra_crop_forwards_by_phase": phase_counts,
                "incremental_extra_crop_forwards": incremental_cost,
                "image_cumulative_extra_crop_forwards": int(used_cost[image_index]),
                "full_miou_before": float(current_miou),
                "full_miou_after": next_miou,
                "full_miou_gain": float(next_miou - current_miou),
                "full_miou_gain_per_new_crop": (
                    None
                    if incremental_cost == 0
                    else float((next_miou - current_miou) / incremental_cost)
                ),
            }
        )
        current_miou = next_miou

    closure = summarize_closure(levels, geometry)
    if closure["extra_crop_forwards"] != int(used_cost.sum()):
        raise AssertionError("oracle action costs disagree with final closure")
    reconstructed = confusion_for_levels(records, levels, full_k1_confusion)
    if not np.array_equal(reconstructed, confusion):
        raise AssertionError("oracle action trace disagrees with final confusion")
    return {
        "selection_rule": (
            "GT-informed constructive A2 greedy with exact marginal unique-crop "
            "cost, pooled mIoU utility, independent per-image caps, no nonpositive "
            "bridge crossing, and lower-cell-index ties"
        ),
        "levels_by_cell": levels,
        "actions": tuple(actions),
        "used_extra_crop_forwards_by_image": used_cost,
        "caps_by_image": caps,
        "closure": closure,
        "full_confusion": confusion,
        "full_miou": float(current_miou),
        "regions": region_summary_for_levels(records, levels),
    }


def random_priority_walk(
    geometry: Mapping[str, Any],
    *,
    extra_crop_cap_by_image: Sequence[int] | np.ndarray,
    seed: int,
    max_attempts_per_image: int = 128,
    return_metadata: bool = False,
) -> np.ndarray | dict[str, Any]:
    """Return one GT-free A2 policy from fixed random action priorities.

    Each image is walked independently to the same realized exact crop cost as
    the candidate.  Both transition priorities are sampled once per attempt.
    At every step, the lowest-priority currently eligible action that fits the
    remaining cap is taken.  A dead-end that underspends is deterministically
    retried and failure to obtain exact equality is fatal.  Zero-cost actions
    are ordinary fitting actions and are therefore auditable in the final
    action-count distribution.
    """

    baseline_counts = np.asarray(geometry["baseline_crop_counts"], dtype=np.int64)
    caps = np.asarray(extra_crop_cap_by_image)
    if caps.shape != baseline_counts.shape or not np.issubdtype(caps.dtype, np.integer):
        raise ValueError("random exact caps have the wrong shape or dtype")
    caps = np.ascontiguousarray(caps, dtype=np.int64)
    if np.any(caps < 0):
        raise ValueError("random exact caps must be non-negative")
    rng = np.random.default_rng(_integer(seed, name="seed", minimum=0))
    maximum_attempts = _integer(
        max_attempts_per_image, name="max_attempts_per_image", minimum=1
    )
    levels = np.ones(len(geometry["image_ids"]), dtype=np.int64)
    attempts_by_image = np.zeros(len(baseline_counts), dtype=np.int64)
    dependencies = geometry["dependencies"]
    eligible = np.asarray(geometry["eligible"], dtype=bool)

    for image_index, group in enumerate(geometry["cells_by_image"]):
        group_indices = np.asarray(group, dtype=np.int64)
        group_indices = group_indices[eligible[group_indices]]
        local_position = {
            int(cell_index): position
            for position, cell_index in enumerate(group_indices)
        }
        target_cost = int(caps[image_index])
        accepted_levels = None
        for attempt in range(1, maximum_attempts + 1):
            group_levels = np.ones(len(group_indices), dtype=np.int64)
            priorities = {
                1: rng.random(len(group_indices)),
                2: rng.random(len(group_indices)),
            }
            used = {name: 0 for name in PHASE_NAMES}
            cost = 0
            while True:
                candidates: list[
                    tuple[float, int, int, int, tuple[int, int, int]]
                ] = []
                for group_position, cell_index in enumerate(group_indices):
                    level = int(group_levels[group_position])
                    if level == 4:
                        continue
                    if level == 1:
                        additions = (
                            dependencies["x8"][cell_index] & ~used["x8"],
                            0,
                            0,
                        )
                    else:
                        additions = (
                            0,
                            dependencies["y8"][cell_index] & ~used["y8"],
                            dependencies["xy8"][cell_index] & ~used["xy8"],
                        )
                    increment = sum(mask.bit_count() for mask in additions)
                    if cost + increment <= target_cost:
                        candidates.append(
                            (
                                float(
                                    priorities[level][
                                        local_position[int(cell_index)]
                                    ]
                                ),
                                int(cell_index),
                                group_position,
                                increment,
                                additions,
                            )
                        )
                if not candidates:
                    break
                _, _, group_position, increment, additions = min(
                    candidates, key=lambda item: (item[0], item[1])
                )
                for name, mask in zip(PHASE_NAMES, additions, strict=True):
                    used[name] |= mask
                cost += increment
                group_levels[group_position] = (
                    2 if group_levels[group_position] == 1 else 4
                )
            if cost == target_cost:
                accepted_levels = group_levels
                attempts_by_image[image_index] = attempt
                break
        if accepted_levels is None:
            raise RuntimeError(
                f"random priority walk could not exactly spend image {image_index} "
                f"target {target_cost} after {maximum_attempts} attempts"
            )
        levels[group_indices] = accepted_levels
    if return_metadata:
        return {
            "levels_by_cell": levels,
            "attempts_by_image": attempts_by_image,
            "successful_images": len(baseline_counts),
        }
    return levels


def random_priority_control(
    cells: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    full_k1_confusion: Sequence[Sequence[int]] | np.ndarray,
    *,
    extra_crop_cap_by_image: Sequence[int] | np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate deterministic exact-budget random-priority A2 controls."""

    count = _integer(replicates, name="replicates", minimum=1)
    base_seed = _integer(seed, name="seed", minimum=0)
    values = np.empty(count, dtype=np.float64)
    costs = np.empty(count, dtype=np.int64)
    action_counts = np.empty(count, dtype=np.int64)
    per_image_costs = np.empty(
        (count, len(geometry["baseline_crop_counts"])), dtype=np.int64
    )
    attempts = np.empty_like(per_image_costs)
    candidate_seeds = np.random.SeedSequence(base_seed).generate_state(
        count, dtype=np.uint64
    )
    for replicate, replicate_seed in enumerate(candidate_seeds):
        walk = random_priority_walk(
            geometry,
            extra_crop_cap_by_image=extra_crop_cap_by_image,
            seed=int(replicate_seed),
            return_metadata=True,
        )
        levels = walk["levels_by_cell"]
        attempts[replicate] = walk["attempts_by_image"]
        confusion = confusion_for_levels(cells, levels, full_k1_confusion)
        values[replicate] = mean_iou_from_confusion(confusion)
        closure = summarize_closure(levels, geometry, include_crop_ids=False)
        costs[replicate] = closure["extra_crop_forwards"]
        action_counts[replicate] = int(
            np.count_nonzero(levels >= 2) + np.count_nonzero(levels == 4)
        )
        per_image_costs[replicate] = np.asarray(
            [item["extra_crop_forwards"] for item in closure["per_image"]],
            dtype=np.int64,
        )
    caps = np.asarray(extra_crop_cap_by_image, dtype=np.int64)
    expected_costs = np.broadcast_to(caps[None, :], per_image_costs.shape)
    if not np.array_equal(per_image_costs, expected_costs):
        raise AssertionError(
            "a random control did not exactly match every realized per-image cost"
        )
    return {
        "role": (
            "GT-free random-priority hierarchical walks on the same eligible cells "
            "and per-image exact-crop caps; p95 is a mechanism control, not a "
            "confidence interval"
        ),
        "replicates": count,
        "seed": base_seed,
        "miou": {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5, method="linear")),
            "p95": float(np.percentile(values, 95, method="linear")),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        },
        "quantile_method": "numpy.percentile(method='linear')",
        "extra_crop_forwards": {
            "mean": float(costs.mean()),
            "minimum": int(costs.min()),
            "maximum": int(costs.max()),
            "candidate_caps_total": int(caps.sum()),
        },
        "action_count": {
            "mean": float(action_counts.mean()),
            "minimum": int(action_counts.min()),
            "maximum": int(action_counts.max()),
        },
        "construction_attempts_per_image": {
            "mean": float(attempts.mean()),
            "median": float(np.median(attempts)),
            "maximum": int(attempts.max()),
            "successful_image_walks": int(attempts.size),
            "failed_image_walks": 0,
            "maximum_attempts_allowed": 128,
        },
        "per_image_extra_crop_forwards": {
            "minimum": per_image_costs.min(axis=0),
            "median": np.median(per_image_costs, axis=0),
            "maximum": per_image_costs.max(axis=0),
            "caps": caps,
        },
        "replicate_values": {
            "miou": values,
            "extra_crop_forwards": costs,
            "action_count": action_counts,
            "per_image_extra_crop_forwards": per_image_costs,
            "attempts_by_image": attempts,
        },
    }
