"""Pure NumPy building blocks for Stage-B1 cached sparse-phase replay.

The module is deliberately model- and dataset-I/O-free.  It replays raw
per-crop float32 logits in the sealed row-major crop order, checks sparse
sum/count maps against a dense replay on exactly the routed physical pixels,
inverse-aligns each phase, and applies the frozen ownership action mask.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from scripts.phase_closure_common import (
    PHASE_NAMES,
    closure_masks_from_levels,
    validate_levels,
)


MIDDLE_SAMPLE_NAME = "NH49E001014"
MIDDLE_LOCAL_LEVELS = {17: 4, 71: 2, 72: 4, 87: 4, 88: 2}


def _validated_image_index(
    geometry: Mapping[str, Any], image_index: int
) -> int:
    if isinstance(image_index, (bool, np.bool_)) or not isinstance(
        image_index, (int, np.integer)
    ):
        raise TypeError("image_index must be an integer")
    result = int(image_index)
    image_count = len(geometry["baseline_crop_counts"])
    if result < 0 or result >= image_count:
        raise IndexError("image_index is out of range")
    return result


def array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(values.tobytes()).hexdigest()


def bit_mask_to_ids(mask: int) -> tuple[int, ...]:
    if not isinstance(mask, (int, np.integer)) or isinstance(mask, (bool, np.bool_)):
        raise TypeError("crop dependency mask must be an integer")
    value = int(mask)
    if value < 0:
        raise ValueError("crop dependency mask must be non-negative")
    result: list[int] = []
    while value:
        least = value & -value
        result.append(least.bit_length() - 1)
        value ^= least
    return tuple(result)


def expected_phase_crop_ids(
    levels_by_cell: Sequence[int] | np.ndarray,
    geometry: Mapping[str, Any],
    image_index: int,
) -> dict[str, tuple[int, ...]]:
    levels = validate_levels(levels_by_cell, geometry)
    image_index = _validated_image_index(geometry, image_index)
    masks = closure_masks_from_levels(levels, geometry)
    return {
        phase_name: bit_mask_to_ids(masks[phase_name][image_index])
        for phase_name in PHASE_NAMES
    }


def expected_phase_crop_keys(
    levels_by_cell: Sequence[int] | np.ndarray,
    geometry: Mapping[str, Any],
) -> tuple[tuple[int, str, int], ...]:
    """Return canonical (image, phase, crop_id) execution keys."""

    levels = validate_levels(levels_by_cell, geometry)
    masks = closure_masks_from_levels(levels, geometry)
    return tuple(
        (image_index, phase_name, crop_id)
        for image_index in range(len(geometry["baseline_crop_counts"]))
        for phase_name in PHASE_NAMES
        for crop_id in bit_mask_to_ids(masks[phase_name][image_index])
    )


def crop_key_sha256(keys: Sequence[tuple[int, str, int]]) -> str:
    canonical = "\n".join(
        f"{int(image_index)}\t{phase_name}\t{int(crop_id)}"
        for image_index, phase_name, crop_id in keys
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_observed_crop_keys(
    observed: Sequence[tuple[int, str, int]],
    expected: Sequence[tuple[int, str, int]],
) -> dict[str, Any]:
    """Require the actual execution trace to equal the canonical plan exactly."""

    def normalize(
        values: Sequence[tuple[int, str, int]], *, name: str
    ) -> tuple[tuple[int, str, int], ...]:
        result: list[tuple[int, str, int]] = []
        for key in values:
            if not isinstance(key, tuple) or len(key) != 3:
                raise TypeError(f"{name} crop key must be a three-tuple")
            image_index, phase_name, crop_id = key
            if isinstance(image_index, (bool, np.bool_)) or not isinstance(
                image_index, (int, np.integer)
            ):
                raise TypeError(f"{name} image index must be an integer")
            if phase_name not in PHASE_NAMES:
                raise ValueError(f"{name} phase name is invalid: {phase_name}")
            if isinstance(crop_id, (bool, np.bool_)) or not isinstance(
                crop_id, (int, np.integer)
            ):
                raise TypeError(f"{name} crop id must be an integer")
            if int(image_index) < 0 or int(crop_id) < 0:
                raise ValueError(f"{name} crop key values must be non-negative")
            result.append((int(image_index), str(phase_name), int(crop_id)))
        return tuple(result)

    actual = normalize(observed, name="observed")
    planned = normalize(expected, name="expected")
    if len(set(planned)) != len(planned):
        raise ValueError("expected crop closure contains a duplicate key")
    phase_order = {name: index for index, name in enumerate(PHASE_NAMES)}
    canonical_plan = tuple(
        sorted(
            planned,
            key=lambda key: (key[0], phase_order[key[1]], key[2]),
        )
    )
    if planned != canonical_plan:
        raise ValueError("expected crop closure is not in canonical order")
    if len(set(actual)) != len(actual):
        raise AssertionError("observed crop execution contains a duplicate key")
    if actual != planned:
        raise AssertionError("observed crop execution differs from the frozen closure")
    return {
        "equal": True,
        "count": len(actual),
        "sha256": crop_key_sha256(actual),
    }


def _validated_crop_ids(
    crop_ids: Sequence[int], *, crop_count: int
) -> tuple[int, ...]:
    if isinstance(crop_ids, (str, bytes)):
        raise TypeError("crop_ids must be an integer sequence")
    raw_values = tuple(crop_ids)
    if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise TypeError("crop_ids must not contain booleans")
    if any(not isinstance(value, (int, np.integer)) for value in raw_values):
        raise TypeError("crop_ids must contain integers")
    values = tuple(int(value) for value in raw_values)
    if any(value < 0 or value >= crop_count for value in values):
        raise ValueError("crop_id is out of range")
    if values != tuple(sorted(set(values))):
        raise ValueError("crop_ids must be unique and strictly increasing")
    return values


def analytic_count_map(
    geometry: Mapping[str, Any],
    image_index: int,
    crop_ids: Sequence[int] | None = None,
) -> np.ndarray:
    image_index = _validated_image_index(geometry, image_index)
    height, width = geometry["image_shapes"][image_index]
    windows = geometry["windows_by_image"][image_index]
    selected = _validated_crop_ids(
        range(len(windows)) if crop_ids is None else crop_ids,
        crop_count=len(windows),
    )
    count = np.zeros((height, width), dtype=np.int16)
    for crop_id in selected:
        y0, y1, x0, x1 = windows[crop_id]
        count[y0:y1, x0:x1] += 1
    return count


def accumulate_phase_crops(
    crop_logits: Mapping[int, np.ndarray],
    crop_ids: Sequence[int],
    geometry: Mapping[str, Any],
    image_index: int,
    *,
    num_classes: int,
) -> dict[str, Any]:
    """Replay raw crops in canonical local-crop order into float32 sum/count."""

    image_index = _validated_image_index(geometry, image_index)
    height, width = geometry["image_shapes"][image_index]
    windows = geometry["windows_by_image"][image_index]
    selected = _validated_crop_ids(crop_ids, crop_count=len(windows))
    if not isinstance(crop_logits, Mapping):
        raise TypeError("crop_logits must map local crop ids to arrays")
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    score_sum = np.zeros((num_classes, height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.int16)
    crop_hashes: dict[int, str] = {}
    for crop_id in selected:
        if crop_id not in crop_logits:
            raise KeyError(f"missing raw logits for crop_id={crop_id}")
        values = np.asarray(crop_logits[crop_id])
        if values.dtype != np.float32:
            raise TypeError("raw crop logits must be float32 for exact replay")
        if values.ndim != 3 or values.shape[0] != num_classes:
            raise ValueError("raw crop logits must have shape [class,height,width]")
        if not np.all(np.isfinite(values)):
            raise ValueError("raw crop logits must be finite")
        y0, y1, x0, x1 = windows[crop_id]
        expected_shape = (num_classes, y1 - y0, x1 - x0)
        if values.shape != expected_shape:
            raise ValueError(
                f"crop {crop_id} shape {values.shape} differs from {expected_shape}"
            )
        contiguous = np.ascontiguousarray(values)
        with np.errstate(over="ignore", invalid="ignore"):
            score_sum[:, y0:y1, x0:x1] += contiguous
        count[y0:y1, x0:x1] += 1
        crop_hashes[crop_id] = array_sha256(contiguous)
    if not np.all(np.isfinite(score_sum)):
        raise FloatingPointError("replayed sum logits contain non-finite values")
    expected_count = analytic_count_map(geometry, image_index, selected)
    if not np.array_equal(count, expected_count):
        raise AssertionError("replayed count map differs from analytic crop coverage")
    return {
        "sum_logits": score_sum,
        "count_mat": count,
        "crop_ids": selected,
        "crop_logits_sha256": crop_hashes,
        "sum_logits_sha256": array_sha256(score_sum),
        "count_mat_sha256": array_sha256(count),
    }


def policy_level_map(
    levels_by_cell: Sequence[int] | np.ndarray,
    geometry: Mapping[str, Any],
    image_index: int,
) -> np.ndarray:
    """Rasterize action levels; closure spill never changes this map."""

    levels = validate_levels(levels_by_cell, geometry)
    image_index = _validated_image_index(geometry, image_index)
    height, width = geometry["image_shapes"][image_index]
    result = np.ones((height, width), dtype=np.uint8)
    for cell_index in geometry["cells_by_image"][image_index]:
        routed = geometry["routed_rectangles"][cell_index]
        if routed is None:
            if levels[cell_index] != 1:
                raise AssertionError("ineligible cell was routed")
            continue
        y0, y1, x0, x1 = routed
        result[y0:y1, x0:x1] = np.uint8(levels[cell_index])
    return result


def phase_required_mask(level_map: np.ndarray, phase_name: str) -> np.ndarray:
    levels = np.asarray(level_map)
    if levels.ndim != 2 or np.any(~np.isin(levels, (1, 2, 4))):
        raise ValueError("level_map must be a 2-D K1/K2/K4 map")
    if phase_name == "x8":
        return levels >= 2
    if phase_name in ("y8", "xy8"):
        return levels == 4
    raise ValueError(f"unknown phase_name: {phase_name}")


def aligned_region(
    array: np.ndarray,
    shift: tuple[int, int],
    original_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Sample a shifted-canvas array at original coordinates plus (dy,dx)."""

    values = np.asarray(array)
    if values.ndim not in (2, 3):
        raise ValueError("aligned array must be HW or CHW")
    dy, dx = (int(value) for value in shift)
    if dy < 0 or dx < 0:
        raise ValueError("sealed phase shifts must be non-negative")
    y0, y1, x0, x1 = (int(value) for value in original_bounds)
    height, width = values.shape[-2:]
    if y0 < 0 or x0 < 0 or y0 >= y1 or x0 >= x1:
        raise ValueError("original bounds are invalid")
    if y1 + dy > height or x1 + dx > width:
        raise ValueError("shifted original bounds lie outside the canvas")
    return values[..., y0 + dy : y1 + dy, x0 + dx : x1 + dx]


def _validated_phase_accumulation(
    accumulation: Mapping[str, Any],
    *,
    name: str,
    expected_hw: tuple[int, int] | None = None,
    expected_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(accumulation, Mapping):
        raise TypeError(f"{name} phase accumulation must be a mapping")
    if "sum_logits" not in accumulation or "count_mat" not in accumulation:
        raise KeyError(f"{name} phase accumulation lacks sum_logits/count_mat")
    score_sum = np.asarray(accumulation["sum_logits"])
    count = np.asarray(accumulation["count_mat"])
    if score_sum.dtype != np.float32 or score_sum.ndim != 3:
        raise TypeError(f"{name} sum_logits must be a float32 CHW array")
    if (
        count.ndim != 2
        or not np.issubdtype(count.dtype, np.integer)
        or count.dtype == np.bool_
    ):
        raise TypeError(f"{name} count_mat must be a 2-D integer array")
    if score_sum.shape[1:] != count.shape:
        raise ValueError(f"{name} sum_logits/count_mat shapes differ")
    if expected_hw is not None and count.shape != tuple(expected_hw):
        raise ValueError(f"{name} phase canvas shape differs from the image")
    if expected_classes is not None and score_sum.shape[0] != int(expected_classes):
        raise ValueError(f"{name} phase class count differs from normal logits")
    if not np.all(np.isfinite(score_sum)):
        raise ValueError(f"{name} sum_logits must be finite")
    if np.any(count < 0):
        raise ValueError(f"{name} count_mat must be non-negative")
    uncovered = count == 0
    if np.any(score_sum[:, uncovered] != 0):
        raise ValueError(f"{name} has nonzero logits where count_mat is zero")
    return score_sum, count


def validate_phase_replay_on_routed_pixels(
    sparse: Mapping[str, Any],
    dense: Mapping[str, Any],
    level_map: np.ndarray,
    *,
    phase_name: str,
    shift: tuple[int, int],
    common_bounds: tuple[int, int, int, int],
    compare_sum_logits: bool = True,
) -> dict[str, Any]:
    """Require exact count/sum equality wherever this phase is authorized."""

    required_full = phase_required_mask(level_map, phase_name)
    sparse_sum_full, sparse_count_full = _validated_phase_accumulation(
        sparse,
        name=f"{phase_name} sparse",
        expected_hw=required_full.shape,
    )
    dense_sum_full, dense_count_full = _validated_phase_accumulation(
        dense,
        name=f"{phase_name} dense",
        expected_hw=required_full.shape,
        expected_classes=sparse_sum_full.shape[0],
    )
    y0, y1, x0, x1 = common_bounds
    required = required_full[y0:y1, x0:x1]
    sparse_count = aligned_region(sparse_count_full, shift, common_bounds)
    dense_count = aligned_region(dense_count_full, shift, common_bounds)
    if sparse_count.shape != required.shape or dense_count.shape != required.shape:
        raise AssertionError("aligned count map shape differs from routed mask")
    routed_pixels = int(np.count_nonzero(required))
    count_equal = bool(np.array_equal(sparse_count[required], dense_count[required]))
    positive = bool(np.all(sparse_count[required] > 0)) if routed_pixels else True
    if not count_equal or not positive:
        raise AssertionError(
            f"{phase_name} sparse count differs from dense on routed pixels"
        )

    sum_equal: bool | None = None
    mean_equal: bool | None = None
    if compare_sum_logits:
        sparse_sum = aligned_region(sparse_sum_full, shift, common_bounds)
        dense_sum = aligned_region(dense_sum_full, shift, common_bounds)
        if sparse_sum.shape != dense_sum.shape:
            raise AssertionError("aligned sparse/dense sum-logit shapes differ")
        sum_equal = bool(np.array_equal(sparse_sum[:, required], dense_sum[:, required]))
        if not sum_equal:
            raise AssertionError(
                f"{phase_name} sparse sum logits differ from dense on routed pixels"
            )
        if routed_pixels:
            sparse_mean = sparse_sum[:, required] / sparse_count[required][None]
            dense_mean = dense_sum[:, required] / dense_count[required][None]
            mean_equal = bool(np.array_equal(sparse_mean, dense_mean))
        else:
            mean_equal = True
        if not mean_equal:
            raise AssertionError(
                f"{phase_name} sparse mean logits differ from dense on routed pixels"
            )
    return {
        "phase_name": phase_name,
        "routed_pixels": routed_pixels,
        "count_mat_equal": count_equal,
        "count_mat_positive": positive,
        "sum_logits_equal": sum_equal,
        "mean_logits_equal": mean_equal,
    }


def validate_live_phase_on_routed_pixels(
    sparse: Mapping[str, Any],
    dense: Mapping[str, Any],
    level_map: np.ndarray,
    *,
    phase_name: str,
    shift: tuple[int, int],
    common_bounds: tuple[int, int, int, int],
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Live analogue: count is exact while normalized logits use tolerance."""

    atol, rtol = float(atol), float(rtol)
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("atol and rtol must be finite and non-negative")
    required_full = phase_required_mask(level_map, phase_name)
    sparse_sum_full, sparse_count_full = _validated_phase_accumulation(
        sparse,
        name=f"{phase_name} live sparse",
        expected_hw=required_full.shape,
    )
    dense_sum_full, dense_count_full = _validated_phase_accumulation(
        dense,
        name=f"{phase_name} live dense",
        expected_hw=required_full.shape,
        expected_classes=sparse_sum_full.shape[0],
    )
    y0, y1, x0, x1 = common_bounds
    required = required_full[y0:y1, x0:x1]
    sparse_count = aligned_region(sparse_count_full, shift, common_bounds)
    dense_count = aligned_region(dense_count_full, shift, common_bounds)
    routed_pixels = int(np.count_nonzero(required))
    count_equal = bool(np.array_equal(sparse_count[required], dense_count[required]))
    positive = bool(np.all(sparse_count[required] > 0)) if routed_pixels else True
    if not count_equal or not positive:
        raise AssertionError(
            f"{phase_name} live sparse count differs from dense on routed pixels"
        )
    sparse_mean, _ = _phase_mean_on_common(
        sparse,
        shift,
        common_bounds,
        phase_name=f"{phase_name} live sparse",
        expected_hw=required_full.shape,
        expected_classes=sparse_sum_full.shape[0],
    )
    dense_mean, _ = _phase_mean_on_common(
        dense,
        shift,
        common_bounds,
        phase_name=f"{phase_name} live dense",
        expected_hw=required_full.shape,
        expected_classes=sparse_sum_full.shape[0],
    )
    sparse_values = sparse_mean[:, required]
    dense_values = dense_mean[:, required]
    difference = np.abs(sparse_values - dense_values)
    logits_close = bool(
        np.allclose(sparse_values, dense_values, atol=atol, rtol=rtol)
    )
    if not logits_close:
        raise AssertionError(
            f"{phase_name} live sparse logits exceed the frozen tolerance"
        )
    return {
        "phase_name": phase_name,
        "routed_pixels": routed_pixels,
        "count_mat_equal": count_equal,
        "count_mat_positive": positive,
        "phase_mean_logits_close": logits_close,
        "atol": atol,
        "rtol": rtol,
        "maximum_absolute_difference": (
            float(difference.max()) if difference.size else 0.0
        ),
    }


def _phase_mean_on_common(
    accumulation: Mapping[str, Any],
    shift: tuple[int, int],
    common_bounds: tuple[int, int, int, int],
    *,
    phase_name: str,
    expected_hw: tuple[int, int],
    expected_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    score_sum_full, count_full = _validated_phase_accumulation(
        accumulation,
        name=phase_name,
        expected_hw=expected_hw,
        expected_classes=expected_classes,
    )
    score_sum = aligned_region(score_sum_full, shift, common_bounds)
    count = aligned_region(count_full, shift, common_bounds)
    mean = np.zeros(score_sum.shape, dtype=np.float32)
    np.divide(
        score_sum,
        count[None],
        out=mean,
        where=count[None] > 0,
        casting="unsafe",
    )
    return mean, count


def compose_policy_logits(
    normal_logits: np.ndarray,
    phase_accumulations: Mapping[str, Mapping[str, Any]],
    levels_by_cell: Sequence[int] | np.ndarray,
    geometry: Mapping[str, Any],
    image_index: int,
    *,
    include_digests: bool = True,
) -> dict[str, Any]:
    """Inverse-align phase means and apply the frozen ownership action mask.

    ``include_digests=False`` skips only the diagnostic SHA256 passes over the
    already materialized arrays.  The logits, prediction, ownership map, and
    phase audit are otherwise produced by the identical code path.  This is
    useful for latency measurement, where hashing must remain outside the
    timed inference boundary.
    """

    image_index = _validated_image_index(geometry, image_index)
    levels = validate_levels(levels_by_cell, geometry)
    normal = np.asarray(normal_logits)
    height, width = geometry["image_shapes"][image_index]
    if normal.dtype != np.float32 or normal.ndim != 3:
        raise TypeError("normal_logits must be a float32 CHW array")
    if normal.shape[1:] != (height, width):
        raise ValueError("normal_logits shape differs from the image")
    if normal.shape[0] <= 0:
        raise ValueError("normal_logits must contain at least one class")
    if not np.all(np.isfinite(normal)):
        raise ValueError("normal_logits must be finite")
    if not isinstance(phase_accumulations, Mapping):
        raise TypeError("phase_accumulations must be a mapping")
    unknown_phases = set(phase_accumulations) - set(PHASE_NAMES)
    if unknown_phases:
        raise ValueError(f"unknown phase accumulations: {sorted(unknown_phases)}")
    output = np.ascontiguousarray(normal.copy())
    level_map = policy_level_map(levels, geometry, image_index)
    common_bounds = tuple(int(value) for value in geometry["common_bounds"][image_index])
    y0, y1, x0, x1 = common_bounds
    output_common = output[:, y0:y1, x0:x1]
    level_common = level_map[y0:y1, x0:x1]
    phase_audit: dict[str, Any] = {}
    for phase_name in PHASE_NAMES:
        required = phase_required_mask(level_common, phase_name)
        accumulation = phase_accumulations.get(phase_name)
        if np.any(required) and accumulation is None:
            raise KeyError(f"missing accumulation for required phase {phase_name}")
        if not np.any(required):
            phase_audit[phase_name] = {"routed_pixels": 0, "added": False}
            continue
        phase_mean, phase_count = _phase_mean_on_common(
            accumulation,
            tuple(geometry["phase_shifts"][phase_name]),
            common_bounds,
            phase_name=phase_name,
            expected_hw=(height, width),
            expected_classes=normal.shape[0],
        )
        if phase_mean.shape != output_common.shape:
            raise AssertionError("aligned phase logits differ from common output shape")
        if np.any(phase_count[required] <= 0):
            raise AssertionError(f"{phase_name} has an uncovered routed pixel")
        with np.errstate(over="ignore", invalid="ignore"):
            output_common[:, required] += phase_mean[:, required]
        if not np.all(np.isfinite(output_common[:, required])):
            raise FloatingPointError(
                f"policy logits became non-finite after adding {phase_name}"
            )
        phase_audit[phase_name] = {
            "routed_pixels": int(np.count_nonzero(required)),
            "added": bool(np.any(required)),
        }
    prediction = np.ascontiguousarray(output.argmax(axis=0).astype(np.int64))
    result = {
        "logits": output,
        "prediction": prediction,
        "level_map": level_map,
        "phase_audit": phase_audit,
    }
    if include_digests:
        result["logits_sha256"] = array_sha256(output)
        result["prediction_sha256"] = array_sha256(prediction)
    return result


def frozen_middle_levels(
    geometry: Mapping[str, Any],
    *,
    sample_name: str = MIDDLE_SAMPLE_NAME,
) -> np.ndarray:
    """Return the preregistered first-image correctness-only middle subset."""

    matches = [
        index
        for index, name in enumerate(geometry["sample_names"])
        if str(name) == str(sample_name)
    ]
    if len(matches) != 1:
        raise ValueError("frozen middle sample must occur exactly once")
    image_index = matches[0]
    levels = np.ones(len(geometry["image_ids"]), dtype=np.int64)
    local_to_global = {
        int(geometry["local_crop_ids"][cell_index]): int(cell_index)
        for cell_index in geometry["cells_by_image"][image_index]
    }
    for local_id, level in MIDDLE_LOCAL_LEVELS.items():
        if local_id not in local_to_global:
            raise ValueError(f"middle local crop id {local_id} is absent")
        levels[local_to_global[local_id]] = level
    return validate_levels(levels, geometry)


def endpoint_levels(
    geometry: Mapping[str, Any], level: int
) -> np.ndarray:
    if int(level) not in (1, 2, 4):
        raise ValueError("endpoint level must be K1, K2, or K4")
    result = np.ones(len(geometry["eligible"]), dtype=np.int64)
    if int(level) != 1:
        result[np.asarray(geometry["eligible"], dtype=bool)] = int(level)
    return result
