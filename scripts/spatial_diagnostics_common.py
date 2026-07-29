"""Pure NumPy/SciPy helpers for spatial segmentation-error diagnostics.

The helpers in this module operate only on a semantic prediction and its
ground-truth label map.  They deliberately do not infer a causal explanation
from an error pattern: boundary, component, and patch statistics describe
where the released model is wrong, while oracle scores measure only an upper
bound obtained by replacing selected predictions with ground truth.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy import ndimage


EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def _validated_thresholds(
    values: Iterable[int], *, name: str, allow_zero: bool = False
) -> tuple[int, ...]:
    thresholds = tuple(sorted(set(int(value) for value in values)))
    lower_bound = 0 if allow_zero else 1
    if not thresholds or any(value < lower_bound for value in thresholds):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain {comparator} integers")
    return thresholds


def validate_semantic_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return contiguous 2-D prediction, target, and valid mask."""

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("prediction and target must both be 2-D")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} != {target.shape}"
        )
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not np.issubdtype(prediction.dtype, np.integer):
        raise TypeError("prediction must contain integer class indices")
    if not np.issubdtype(target.dtype, np.integer):
        raise TypeError("target must contain integer class indices")

    prediction = np.ascontiguousarray(prediction)
    target = np.ascontiguousarray(target)
    valid = (target >= 0) & (target < num_classes)
    invalid_predictions = valid & (
        (prediction < 0) | (prediction >= num_classes)
    )
    if np.any(invalid_predictions):
        values = np.unique(prediction[invalid_predictions])
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

    prediction, target, valid = validate_semantic_arrays(
        prediction, target, num_classes
    )
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != target.shape:
            raise ValueError(f"mask shape differs from target: {mask.shape}")
        valid &= mask
    encoded = (
        target[valid].astype(np.int64, copy=False) * num_classes
        + prediction[valid].astype(np.int64, copy=False)
    )
    return np.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def class_ious_from_confusion(confusion: np.ndarray) -> np.ndarray:
    confusion = np.asarray(confusion, dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")
    diagonal = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - diagonal
    result = np.full(diagonal.shape, np.nan, dtype=np.float64)
    np.divide(diagonal, union, out=result, where=union > 0)
    return result


def mean_iou_from_confusion(confusion: np.ndarray) -> float:
    values = class_ious_from_confusion(confusion)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def oracle_confusion(
    baseline_confusion: np.ndarray, region_confusion: np.ndarray
) -> np.ndarray:
    """Return confusion after predictions inside one region are made perfect."""

    baseline = np.asarray(baseline_confusion, dtype=np.int64)
    region = np.asarray(region_confusion, dtype=np.int64)
    if baseline.shape != region.shape or baseline.ndim != 2:
        raise ValueError("baseline and region confusions must have matching shapes")
    if np.any(region < 0) or np.any(region > baseline):
        raise ValueError("region confusion is not a subset of baseline confusion")
    result = baseline - region
    indices = np.diag_indices_from(result)
    result[indices] += region.sum(axis=1)
    return result


def semantic_boundary_mask(target: np.ndarray, num_classes: int) -> np.ndarray:
    """Mark both valid pixels adjacent to a horizontal or vertical class change."""

    target = np.asarray(target)
    if target.ndim != 2:
        raise ValueError("target must be 2-D")
    valid = (target >= 0) & (target < num_classes)
    boundary = np.zeros(target.shape, dtype=bool)

    horizontal_pair = valid[:, :-1] & valid[:, 1:]
    horizontal_change = horizontal_pair & (target[:, :-1] != target[:, 1:])
    boundary[:, :-1] |= horizontal_change
    boundary[:, 1:] |= horizontal_change

    vertical_pair = valid[:-1, :] & valid[1:, :]
    vertical_change = vertical_pair & (target[:-1, :] != target[1:, :])
    boundary[:-1, :] |= vertical_change
    boundary[1:, :] |= vertical_change
    return boundary


def boundary_region_masks(
    target: np.ndarray,
    num_classes: int,
    radii: Iterable[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build valid masks at Euclidean distances from semantic transitions."""

    radii = _validated_thresholds(radii, name="boundary radii", allow_zero=True)
    target = np.asarray(target)
    valid = (target >= 0) & (target < num_classes)
    anchors = semantic_boundary_mask(target, num_classes)
    if np.any(anchors):
        distances = ndimage.distance_transform_edt(~anchors)
    else:
        distances = np.full(target.shape, np.inf, dtype=np.float32)

    masks = {
        f"boundary_le_{radius}px": valid & (distances <= radius)
        for radius in radii
    }
    metadata = {
        "anchor_pixels": int(anchors.sum()),
        "definition": (
            "Both valid pixel centers adjacent to a 4-neighbour class transition "
            "are distance-zero anchors; Euclidean distance transform expands them."
        ),
    }
    return masks, metadata


def image_origin_mixed_patch_mask(
    target: np.ndarray,
    num_classes: int,
    patch_size: int = 16,
) -> np.ndarray:
    """Mark valid pixels in image-origin-aligned patches containing >=2 classes."""

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    target = np.asarray(target)
    if target.ndim != 2:
        raise ValueError("target must be 2-D")
    height, width = target.shape
    block_rows = (height + patch_size - 1) // patch_size
    block_cols = (width + patch_size - 1) // patch_size
    padded_height = block_rows * patch_size
    padded_width = block_cols * patch_size
    padded = np.full(
        (padded_height, padded_width), num_classes, dtype=np.int16
    )
    valid = (target >= 0) & (target < num_classes)
    padded[:height, :width] = np.where(valid, target, num_classes).astype(
        np.int16, copy=False
    )
    blocks = padded.reshape(
        block_rows, patch_size, block_cols, patch_size
    ).transpose(0, 2, 1, 3)
    class_count = np.zeros((block_rows, block_cols), dtype=np.uint8)
    for class_index in range(num_classes):
        class_count += np.any(blocks == class_index, axis=(2, 3))
    mixed_blocks = class_count >= 2
    expanded = np.repeat(
        np.repeat(mixed_blocks, patch_size, axis=0), patch_size, axis=1
    )[:height, :width]
    return valid & expanded


def component_geometry_masks(
    target: np.ndarray,
    num_classes: int,
    area_thresholds: Iterable[int],
    thickness_thresholds: Iterable[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build semantic-component area and thickness-proxy masks.

    Components are class-wise 8-connected regions, not annotated object
    instances.  The thickness proxy is ``2 * area / perimeter_pixel_count``;
    it is useful for repeatable morphology bins but is not an exact width.
    """

    area_thresholds = _validated_thresholds(
        area_thresholds, name="component area thresholds"
    )
    thickness_thresholds = _validated_thresholds(
        thickness_thresholds, name="component thickness thresholds"
    )
    target = np.asarray(target)
    if target.ndim != 2:
        raise ValueError("target must be 2-D")

    masks = {
        **{
            f"component_area_le_{threshold}px2": np.zeros(target.shape, dtype=bool)
            for threshold in area_thresholds
        },
        **{
            f"component_thickness_le_{threshold}px": np.zeros(
                target.shape, dtype=bool
            )
            for threshold in thickness_thresholds
        },
    }
    for class_index in range(num_classes):
        class_mask = target == class_index
        components, component_count = ndimage.label(
            class_mask, structure=EIGHT_CONNECTED
        )
        areas = np.bincount(components.reshape(-1), minlength=component_count + 1)
        eroded = ndimage.binary_erosion(
            class_mask, structure=EIGHT_CONNECTED, border_value=0
        )
        perimeter_mask = class_mask & ~eroded
        perimeter_counts = np.bincount(
            components[perimeter_mask], minlength=component_count + 1
        )
        thickness = np.full(component_count + 1, np.inf, dtype=np.float64)
        np.divide(
            2.0 * areas,
            perimeter_counts,
            out=thickness,
            where=perimeter_counts > 0,
        )
        areas[0] = 0
        thickness[0] = np.inf

        for threshold in area_thresholds:
            selected = areas <= threshold
            selected[0] = False
            masks[f"component_area_le_{threshold}px2"] |= selected[components]

        for threshold in thickness_thresholds:
            selected = thickness <= threshold
            selected[0] = False
            masks[f"component_thickness_le_{threshold}px"] |= selected[
                components
            ]

    metadata = {
        "connectivity": 8,
        "component_semantics": "class-wise semantic regions, not object instances",
        "thickness_proxy": "2 * component_area / 8-neighbour perimeter_pixel_count",
    }
    return masks, metadata


def build_spatial_region_masks(
    target: np.ndarray,
    num_classes: int,
    *,
    boundary_radii: Iterable[int] = (0, 1, 2, 4, 8),
    component_area_thresholds: Iterable[int] = (256, 1024, 4096),
    component_thickness_thresholds: Iterable[int] = (4, 8, 16),
    patch_size: int = 16,
    union_boundary_radius: int = 0,
    union_component_area: int = 256,
    union_component_thickness: int = 4,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build all first-gate region masks and their exact definitions."""

    boundary_radii = _validated_thresholds(
        boundary_radii, name="boundary radii", allow_zero=True
    )
    component_area_thresholds = _validated_thresholds(
        component_area_thresholds, name="component area thresholds"
    )
    component_thickness_thresholds = _validated_thresholds(
        component_thickness_thresholds,
        name="component thickness thresholds",
    )
    if union_boundary_radius not in boundary_radii:
        raise ValueError("union boundary radius must be one of boundary_radii")
    if union_component_area not in component_area_thresholds:
        raise ValueError(
            "union component area must be one of component_area_thresholds"
        )
    if union_component_thickness not in component_thickness_thresholds:
        raise ValueError(
            "union component thickness must be one of component_thickness_thresholds"
        )

    boundary_masks, boundary_metadata = boundary_region_masks(
        target, num_classes, boundary_radii
    )
    component_masks, component_metadata = component_geometry_masks(
        target,
        num_classes,
        component_area_thresholds,
        component_thickness_thresholds,
    )
    mixed_patch_name = f"image_origin_mixed_patch_{patch_size}px"
    masks = {
        **boundary_masks,
        **component_masks,
        mixed_patch_name: image_origin_mixed_patch_mask(
            target, num_classes, patch_size=patch_size
        ),
    }
    union_sources = (
        f"boundary_le_{union_boundary_radius}px",
        f"component_area_le_{union_component_area}px2",
        f"component_thickness_le_{union_component_thickness}px",
    )
    union_mask = np.zeros(np.asarray(target).shape, dtype=bool)
    for source in union_sources:
        union_mask |= masks[source]
    masks["actionable_union"] = union_mask

    metadata = {
        "boundary": boundary_metadata,
        "components": component_metadata,
        "mixed_patch": {
            "patch_size": patch_size,
            "alignment": "full-image top-left origin",
            "causal_limit": (
                "Descriptive GT mixedness only. Overlapping 512/341 sliding "
                "windows expose pixels at multiple model patch phases."
            ),
        },
        "actionable_union_sources": list(union_sources),
    }
    return masks, metadata


def _optional_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return None
    return float(numerator / denominator)


def baseline_summary(
    confusion: np.ndarray, class_names: Sequence[str]
) -> dict[str, Any]:
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (len(class_names), len(class_names)):
        raise ValueError("class names do not match confusion dimensions")
    class_ious = class_ious_from_confusion(confusion)
    pixels = int(confusion.sum())
    errors = int(pixels - np.trace(confusion))
    return {
        "confusion": confusion.tolist(),
        "valid_pixels": pixels,
        "errors": errors,
        "error_rate": float(errors / pixels) if pixels else None,
        "miou": mean_iou_from_confusion(confusion),
        "miou_percent": mean_iou_from_confusion(confusion) * 100.0,
        "class_iou_percent": {
            name: float(value * 100.0) if np.isfinite(value) else None
            for name, value in zip(class_names, class_ious, strict=True)
        },
    }


def region_summary(
    baseline_confusion: np.ndarray,
    region_confusion: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    baseline = np.asarray(baseline_confusion, dtype=np.int64)
    region = np.asarray(region_confusion, dtype=np.int64)
    if baseline.shape != (len(class_names), len(class_names)):
        raise ValueError("class names do not match confusion dimensions")
    oracle = oracle_confusion(baseline, region)
    baseline_iou = class_ious_from_confusion(baseline)
    oracle_iou = class_ious_from_confusion(oracle)

    total_pixels = int(baseline.sum())
    region_pixels = int(region.sum())
    total_errors = int(total_pixels - np.trace(baseline))
    region_errors = int(region_pixels - np.trace(region))
    complement_pixels = total_pixels - region_pixels
    complement_errors = total_errors - region_errors
    error_rate = region_errors / region_pixels if region_pixels else float("nan")
    global_error_rate = total_errors / total_pixels if total_pixels else float("nan")
    complement_error_rate = (
        complement_errors / complement_pixels if complement_pixels else float("nan")
    )
    baseline_miou = mean_iou_from_confusion(baseline)
    oracle_miou = mean_iou_from_confusion(oracle)

    per_class: dict[str, dict[str, Any]] = {}
    for index, class_name in enumerate(class_names):
        gt_pixels = int(region[index].sum())
        errors = int(gt_pixels - region[index, index])
        baseline_gt_pixels = int(baseline[index].sum())
        baseline_errors = int(baseline_gt_pixels - baseline[index, index])
        complement_gt_pixels = baseline_gt_pixels - gt_pixels
        complement_errors = baseline_errors - errors
        class_error_rate = errors / gt_pixels if gt_pixels else float("nan")
        class_complement_error_rate = (
            complement_errors / complement_gt_pixels
            if complement_gt_pixels
            else float("nan")
        )
        per_class[class_name] = {
            "gt_pixels": gt_pixels,
            "errors": errors,
            "error_rate": (
                float(class_error_rate) if np.isfinite(class_error_rate) else None
            ),
            "complement_error_rate": (
                float(class_complement_error_rate)
                if np.isfinite(class_complement_error_rate)
                else None
            ),
            "relative_error_risk": _optional_ratio(
                class_error_rate, class_complement_error_rate
            ),
            "oracle_iou_gain_pp": (
                float((oracle_iou[index] - baseline_iou[index]) * 100.0)
                if np.isfinite(oracle_iou[index]) and np.isfinite(baseline_iou[index])
                else None
            ),
        }

    return {
        "region_confusion": region.tolist(),
        "pixels": region_pixels,
        "coverage": float(region_pixels / total_pixels) if total_pixels else None,
        "errors": region_errors,
        "error_share": float(region_errors / total_errors) if total_errors else None,
        "error_rate": float(error_rate) if np.isfinite(error_rate) else None,
        "error_enrichment_over_global": _optional_ratio(
            error_rate, global_error_rate
        ),
        "complement_error_rate": (
            float(complement_error_rate)
            if np.isfinite(complement_error_rate)
            else None
        ),
        "relative_error_risk": _optional_ratio(error_rate, complement_error_rate),
        "oracle_confusion": oracle.tolist(),
        "oracle_miou_percent": float(oracle_miou * 100.0),
        "oracle_gain_pp": float((oracle_miou - baseline_miou) * 100.0),
        "per_class": per_class,
    }


def diagnose_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    class_names: Sequence[str],
    **region_kwargs: Any,
) -> dict[str, Any]:
    """Calculate baseline and spatial-region confusions for one image."""

    num_classes = len(class_names)
    prediction, target, _ = validate_semantic_arrays(
        prediction, target, num_classes
    )
    masks, definitions = build_spatial_region_masks(
        target, num_classes, **region_kwargs
    )
    baseline = confusion_from_arrays(prediction, target, num_classes)
    region_confusions = {
        name: confusion_from_arrays(
            prediction, target, num_classes, mask=mask
        )
        for name, mask in masks.items()
    }
    return {
        "baseline_confusion": baseline,
        "region_confusions": region_confusions,
        "definitions": definitions,
    }


def bootstrap_oracle_gain(
    baseline_confusions: np.ndarray,
    region_confusions: np.ndarray,
    sample_indices: np.ndarray,
) -> dict[str, Any] | None:
    """Image-cluster bootstrap interval for paired oracle mIoU gain."""

    baselines = np.asarray(baseline_confusions, dtype=np.int64)
    regions = np.asarray(region_confusions, dtype=np.int64)
    samples = np.asarray(sample_indices, dtype=np.int64)
    if baselines.ndim != 3 or baselines.shape != regions.shape:
        raise ValueError("confusion stacks must have matching [image, class, class] shape")
    if samples.ndim != 2 or samples.shape[1] != baselines.shape[0]:
        raise ValueError("sample_indices must be [replicate, image]")
    if baselines.shape[0] < 2 or samples.shape[0] == 0:
        return None

    sampled_baselines = baselines[samples].sum(axis=1)
    sampled_regions = regions[samples].sum(axis=1)
    sampled_oracles = sampled_baselines - sampled_regions
    diagonal_indices = np.arange(sampled_oracles.shape[1])
    sampled_oracles[:, diagonal_indices, diagonal_indices] += sampled_regions.sum(
        axis=2
    )

    def batch_miou(confusions: np.ndarray) -> np.ndarray:
        diagonal = np.diagonal(confusions, axis1=1, axis2=2)
        union = confusions.sum(axis=2) + confusions.sum(axis=1) - diagonal
        ious = np.full(union.shape, np.nan, dtype=np.float64)
        np.divide(diagonal, union, out=ious, where=union > 0)
        return np.nanmean(ious, axis=1)

    gains = (batch_miou(sampled_oracles) - batch_miou(sampled_baselines)) * 100.0
    lower, median, upper = np.percentile(gains, [2.5, 50.0, 97.5])
    return {
        "unit": "test image",
        "replicates": int(samples.shape[0]),
        "median_pp": float(median),
        "ci95_pp": [float(lower), float(upper)],
    }
