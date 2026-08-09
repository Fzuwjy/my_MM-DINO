"""Pure helpers for the EarthMiss scale-transition diagnostic.

This module deliberately has no dataset, DINOv3, or CUDA dependency.  It
turns paired feature tensors into bounded scalar statistics, accumulates those
statistics by tile/city, and writes deterministic JSON.  The runner lives in
``diagnose_earthmiss_scale_transition.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


NUM_CLASSES = 8
IGNORE_INDEX = 8
PAIR_REGIONS = ("valid", "boundary", "interior")
HAAR_BANDS = ("ll", "lh", "hl", "hh")
CROSS_SCALE_STAGE_ORDER = (
    "prn.cross_scale.P5_to_P4",
    "prn.cross_scale.P4_to_P3",
    "prn.cross_scale.P3_to_P2",
)

STAGE_ORDER = (
    "adapter.pre_resize.P2",
    "adapter.fused.P2",
    "adapter.pre_resize.P3",
    "adapter.fused.P3",
    "adapter.pre_resize.P4",
    "adapter.fused.P4",
    "adapter.pre_resize.P5",
    "adapter.fused.P5",
    "frm.P2",
    "frm.P3",
    "frm.P4",
    "frm.P5",
    "decoder.se_fused.P2",
    "decoder.se_fused.P3",
    "decoder.se_fused.P4",
    "decoder.se_fused.P5",
    "prn.resized.P5_to_P4",
    "prn.concat.P4",
    "prn.td.P4",
    "prn.resized.P4_to_P3",
    "prn.concat.P3",
    "prn.td.P3",
    "prn.resized.P3_to_P2",
    "prn.concat.P2",
    "prn.td.P2",
    "prn.out.P2",
    "prn.out.P3",
    "prn.out.P4",
    "logits.decoder",
    "logits.final",
)

STAGE_DESCRIPTIONS = {
    "adapter.pre_resize.P2": "DINO layer-2 projection, fused before x4 transpose-conv",
    "adapter.fused.P2": "adapter P2 after x4 transpose-conv and modality fusion",
    "adapter.pre_resize.P3": "DINO layer-5 projection, fused before x2 transpose-conv",
    "adapter.fused.P3": "adapter P3 after x2 transpose-conv and modality fusion",
    "adapter.pre_resize.P4": "DINO layer-8 projection, fused before identity resize",
    "adapter.fused.P4": "adapter P4 after identity resize and modality fusion",
    "adapter.pre_resize.P5": "DINO layer-11 projection, fused before stride-2 conv",
    "adapter.fused.P5": "adapter P5 after stride-2 conv and modality fusion",
    "frm.P2": "FRM output P2; duplicate decoder slots are equality-checked",
    "frm.P3": "FRM output P3; duplicate decoder slots are equality-checked",
    "frm.P4": "FRM output P4; duplicate decoder slots are equality-checked",
    "frm.P5": "FRM output P5; duplicate decoder slots are equality-checked",
    "decoder.se_fused.P2": "SEFusion output P2",
    "decoder.se_fused.P3": "SEFusion output P3",
    "decoder.se_fused.P4": "SEFusion output P4",
    "decoder.se_fused.P5": "SEFusion output P5",
    "prn.resized.P5_to_P4": "PRN top-down P5 after nearest resize to P4, before concat",
    "prn.concat.P4": "PRN P4 lateral/top-down concat, before td convolution",
    "prn.td.P4": "PRN P5-to-P4 nearest-resize, concat, and convolution output",
    "prn.resized.P4_to_P3": "PRN top-down P4 after nearest resize to P3, before concat",
    "prn.concat.P3": "PRN P3 lateral/top-down concat, before td convolution",
    "prn.td.P3": "PRN P4-to-P3 nearest-resize, concat, and convolution output",
    "prn.resized.P3_to_P2": "PRN top-down P3 after nearest resize to P2, before concat",
    "prn.concat.P2": "PRN P2 lateral/top-down concat, before td convolution",
    "prn.td.P2": "PRN P3-to-P2 nearest-resize, concat, and convolution output",
    "prn.out.P2": "PRN final P2 output",
    "prn.out.P3": "PRN final P3 output",
    "prn.out.P4": "PRN final P4 output",
    "logits.decoder": "decoder-head logits at P2 resolution",
    "logits.final": "model logits resized to the input crop",
}

CROSS_SCALE_STAGE_DESCRIPTIONS = {
    "prn.cross_scale.P5_to_P4": (
        "PRN resized P5 top-down feature versus same-grid P4 lateral feature"
    ),
    "prn.cross_scale.P4_to_P3": (
        "PRN resized P4 top-down feature versus same-grid P3 lateral feature"
    ),
    "prn.cross_scale.P3_to_P2": (
        "PRN resized P3 top-down feature versus same-grid P2 lateral feature"
    ),
}

AMPLIFICATION_EDGES = (
    ("adapter_resize_P2", "adapter.pre_resize.P2", "adapter.fused.P2", "exploratory"),
    ("adapter_resize_P3", "adapter.pre_resize.P3", "adapter.fused.P3", "exploratory"),
    (
        "adapter_identity_P4_control",
        "adapter.pre_resize.P4",
        "adapter.fused.P4",
        "negative_control",
    ),
    ("adapter_stride2_P5", "adapter.pre_resize.P5", "adapter.fused.P5", "primary"),
    ("frm_P2", "adapter.fused.P2", "frm.P2", "exploratory"),
    ("frm_P3", "adapter.fused.P3", "frm.P3", "exploratory"),
    ("frm_P4", "adapter.fused.P4", "frm.P4", "exploratory"),
    ("frm_P5", "adapter.fused.P5", "frm.P5", "exploratory"),
    ("se_fusion_P2", "frm.P2", "decoder.se_fused.P2", "exploratory"),
    ("se_fusion_P3", "frm.P3", "decoder.se_fused.P3", "exploratory"),
    ("se_fusion_P4", "frm.P4", "decoder.se_fused.P4", "exploratory"),
    ("se_fusion_P5", "frm.P5", "decoder.se_fused.P5", "exploratory"),
    (
        "prn_nearest_P5_to_P4",
        "decoder.se_fused.P5",
        "prn.resized.P5_to_P4",
        "descriptive_control",
    ),
    (
        "prn_nearest_P4_to_P3",
        "prn.td.P4",
        "prn.resized.P4_to_P3",
        "exploratory",
    ),
    (
        "prn_nearest_P3_to_P2",
        "prn.td.P3",
        "prn.resized.P3_to_P2",
        "exploratory",
    ),
    ("decoder_to_final_logits", "logits.decoder", "logits.final", "exploratory"),
)


def _as_bchw(tensor: torch.Tensor) -> torch.Tensor:
    """Return a detached float32 BCHW tensor, inferring square token grids."""

    tensor = torch.as_tensor(tensor).detach()
    if tensor.ndim == 4:
        result = tensor
    elif tensor.ndim == 3:
        batch, tokens, channels = tensor.shape
        side = math.isqrt(tokens)
        if side * side != tokens:
            raise ValueError(
                f"token count must form a square grid, got {tokens}"
            )
        result = tensor.transpose(1, 2).reshape(batch, channels, side, side)
    else:
        raise ValueError(f"feature must be BNC or BCHW, got {tuple(tensor.shape)}")
    result = result.to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(result).all():
        raise ValueError("feature contains non-finite values")
    return result.contiguous()


def semantic_boundary_mask(
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Mark both valid pixels adjacent to a 4-neighbour class transition."""

    target = torch.as_tensor(target)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.ndim != 3:
        raise ValueError("target must have shape HW or BHW")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    target = target.to(dtype=torch.int64, device="cpu")
    valid = (target >= 0) & (target < num_classes)
    boundary = torch.zeros_like(valid)

    horizontal_pair = valid[:, :, :-1] & valid[:, :, 1:]
    horizontal_change = horizontal_pair & (
        target[:, :, :-1] != target[:, :, 1:]
    )
    boundary[:, :, :-1] |= horizontal_change
    boundary[:, :, 1:] |= horizontal_change

    vertical_pair = valid[:, :-1, :] & valid[:, 1:, :]
    vertical_change = vertical_pair & (
        target[:, :-1, :] != target[:, 1:, :]
    )
    boundary[:, :-1, :] |= vertical_change
    boundary[:, 1:, :] |= vertical_change
    return boundary


def _native_area_sum(
    mask: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Sum a native-resolution mask inside each non-overlapping output cell.

    Requiring exact divisibility makes every native pixel belong to exactly one
    feature cell.  This avoids the overlapping bins used by adaptive pooling and
    lets region statistics at every stage retain the same native-pixel support.
    """

    mask = torch.as_tensor(mask, device="cpu")
    if mask.ndim != 3:
        raise ValueError("native mask must have shape BHW")
    if len(output_size) != 2 or any(int(value) <= 0 for value in output_size):
        raise ValueError("output_size must contain two positive integers")
    output_height, output_width = (int(value) for value in output_size)
    _, native_height, native_width = mask.shape
    if native_height % output_height or native_width % output_width:
        raise ValueError(
            "feature grid must exactly divide the native target, got "
            f"target={(native_height, native_width)} and output="
            f"{(output_height, output_width)}"
        )
    block_height = native_height // output_height
    block_width = native_width // output_width
    return (
        mask.to(torch.float64)
        .reshape(
            mask.shape[0],
            output_height,
            block_height,
            output_width,
            block_width,
        )
        .sum(dim=(2, 4))
        .contiguous()
    )


def native_semantic_region_area_weights(
    target: torch.Tensor,
    output_size: tuple[int, int],
    num_classes: int = NUM_CLASSES,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Map fixed native semantic support to a divisible feature grid.

    Values are native-pixel counts, not binary feature-cell masks.  Therefore
    the sum of each region's weights is invariant to feature-grid resolution.
    The second return value contains per-class *interior* native-pixel counts
    for prototype accumulation, with shape ``B,K,Hf,Wf``.
    """

    target = torch.as_tensor(target, dtype=torch.int64, device="cpu")
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.ndim != 3:
        raise ValueError("target must have shape HW or BHW")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    valid = (target >= 0) & (target < num_classes)
    boundary = semantic_boundary_mask(target, num_classes) & valid
    interior = valid & ~boundary
    region_weights = {
        "valid": _native_area_sum(valid, output_size),
        "boundary": _native_area_sum(boundary, output_size),
        "interior": _native_area_sum(interior, output_size),
    }
    class_weights = torch.stack(
        [
            _native_area_sum(interior & (target == class_id), output_size)
            for class_id in range(num_classes)
        ],
        dim=1,
    )
    return region_weights, class_weights


def downsample_semantic_regions(
    target: torch.Tensor,
    output_size: tuple[int, int],
    num_classes: int = NUM_CLASSES,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Map labels and conservative valid/boundary masks to a feature grid.

    A feature cell is valid only when every contributing input pixel is valid.
    Native-resolution boundaries are max-pooled so a one-pixel transition does
    not disappear during downsampling.
    """

    target = torch.as_tensor(target)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.ndim != 3:
        raise ValueError("target must have shape HW or BHW")
    if len(output_size) != 2 or any(int(value) <= 0 for value in output_size):
        raise ValueError("output_size must contain two positive integers")
    target = target.to(dtype=torch.int64, device="cpu")
    native_valid = (target >= 0) & (target < num_classes)
    native_boundary = semantic_boundary_mask(target, num_classes)
    size = tuple(int(value) for value in output_size)

    invalid_down = _native_area_sum(~native_valid, size) > 0
    boundary_down = _native_area_sum(native_boundary, size) > 0
    labels = F.interpolate(
        target.to(torch.float32).unsqueeze(1),
        size=size,
        mode="nearest",
    ).squeeze(1).to(torch.int64)
    valid = (~invalid_down) & (labels >= 0) & (labels < num_classes)
    boundary_down &= valid
    interior = valid & ~boundary_down
    return labels, {
        "valid": valid,
        "boundary": boundary_down,
        "interior": interior,
    }


def _evenly_spaced_indices(length: int, maximum: int) -> torch.Tensor:
    if length <= 0 or maximum <= 0:
        raise ValueError("length and maximum must be positive")
    if length <= maximum:
        return torch.arange(length, dtype=torch.int64)
    return torch.linspace(0, length - 1, maximum).round().to(torch.int64)


def centered_linear_cka(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    max_points: int = 128,
    max_channels: int = 128,
    sample_weights: torch.Tensor | None = None,
) -> tuple[float | None, int, int]:
    """Return deterministic, bounded centered linear CKA.

    Points and channels are evenly subsampled before the exact CKA formula is
    evaluated.  The report therefore calls this quantity ``subsampled_cka``.
    """

    left = torch.as_tensor(left, dtype=torch.float32, device="cpu")
    right = torch.as_tensor(right, dtype=torch.float32, device="cpu")
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("CKA inputs must have matching NC shapes")
    if max_points <= 0 or max_channels <= 0:
        raise ValueError("CKA limits must be positive")
    if sample_weights is not None:
        sample_weights = torch.as_tensor(
            sample_weights, dtype=torch.float64, device="cpu"
        ).reshape(-1)
        if sample_weights.shape[0] != left.shape[0]:
            raise ValueError("CKA sample weights must match the point count")
        if not torch.isfinite(sample_weights).all() or (sample_weights < 0).any():
            raise ValueError("CKA sample weights must be finite and non-negative")
    if left.shape[0] < 2 or left.shape[1] == 0:
        return None, int(left.shape[0]), int(left.shape[1])

    point_indices = _evenly_spaced_indices(left.shape[0], max_points)
    channel_indices = _evenly_spaced_indices(left.shape[1], max_channels)
    left = left.index_select(0, point_indices).index_select(1, channel_indices)
    right = right.index_select(0, point_indices).index_select(1, channel_indices)
    if sample_weights is None:
        weights = torch.ones(left.shape[0], dtype=torch.float64)
    else:
        weights = sample_weights.index_select(0, point_indices)
    positive = weights > 0
    left = left[positive]
    right = right[positive]
    weights = weights[positive]
    if left.shape[0] < 2 or float(weights.sum()) <= 0:
        return None, int(left.shape[0]), int(left.shape[1])
    left = left.to(torch.float64)
    right = right.to(torch.float64)
    normalized_weights = weights / weights.sum()
    left = left - (normalized_weights[:, None] * left).sum(dim=0, keepdim=True)
    right = right - (normalized_weights[:, None] * right).sum(dim=0, keepdim=True)
    weighted_left = left * torch.sqrt(weights)[:, None]
    weighted_right = right * torch.sqrt(weights)[:, None]
    cross = weighted_left.transpose(0, 1) @ weighted_right
    left_self = weighted_left.transpose(0, 1) @ weighted_left
    right_self = weighted_right.transpose(0, 1) @ weighted_right
    numerator = cross.square().sum()
    denominator = torch.sqrt(
        left_self.square().sum() * right_self.square().sum()
    )
    if not torch.isfinite(denominator) or float(denominator) <= 0.0:
        return None, int(left.shape[0]), int(left.shape[1])
    value = float((numerator / denominator).clamp(0.0, 1.0))
    return value, int(left.shape[0]), int(left.shape[1])


def orthogonal_haar_energies(feature: torch.Tensor) -> dict[str, float]:
    """Return energy in the four orthonormal 2-D Haar subbands."""

    feature = _as_bchw(feature)
    height, width = feature.shape[-2:]
    if height % 2 or width % 2:
        raise ValueError(
            f"Haar input height/width must be even, got {(height, width)}"
        )
    a = feature[:, :, 0::2, 0::2]
    b = feature[:, :, 0::2, 1::2]
    c = feature[:, :, 1::2, 0::2]
    d = feature[:, :, 1::2, 1::2]
    bands = {
        "ll": (a + b + c + d) * 0.5,
        "lh": (-a - b + c + d) * 0.5,
        "hl": (-a + b - c + d) * 0.5,
        "hh": (a - b - c + d) * 0.5,
    }
    return {
        name: float(values.to(torch.float64).square().sum())
        for name, values in bands.items()
    }


@dataclass
class BatchFeaturePairStatistics:
    channels: int
    spatial_shape: tuple[int, int]
    observations: int
    regions: dict[str, dict[str, float | int | None]]
    haar: dict[str, dict[str, float]]
    class_counts: torch.Tensor
    left_class_sums: torch.Tensor
    right_class_sums: torch.Tensor


def _region_statistics(
    left: torch.Tensor,
    right: torch.Tensor,
    area_weights: torch.Tensor,
    *,
    cka_max_points: int,
    cka_max_channels: int,
) -> dict[str, float | int | None]:
    area_weights = torch.as_tensor(
        area_weights, dtype=torch.float64, device="cpu"
    ).reshape(-1)
    selected = area_weights > 0
    area_weights = area_weights[selected]
    left = left.reshape(-1, left.shape[-1])[selected]
    right = right.reshape(-1, right.shape[-1])[selected]
    count = int(left.shape[0])
    channels = int(left.shape[1]) if left.ndim == 2 else 0
    native_pixel_weight = float(area_weights.sum())
    if count == 0:
        return {
            "position_count": 0,
            "native_pixel_weight": 0.0,
            "element_weight": 0.0,
            "cosine_distance_sum": 0.0,
            "diff_squared_sum": 0.0,
            "left_squared_sum": 0.0,
            "subsampled_cka": None,
            "cka_points": 0,
            "cka_channels": 0,
        }

    identical = torch.equal(left, right)
    left64 = left.to(torch.float64)
    right64 = right.to(torch.float64)
    dot = (left64 * right64).sum(dim=1)
    left_norm = torch.linalg.vector_norm(left64, dim=1)
    right_norm = torch.linalg.vector_norm(right64, dim=1)
    product = left_norm * right_norm
    similarity = torch.zeros_like(dot)
    nonzero = product > 0
    similarity[nonzero] = dot[nonzero] / product[nonzero]
    both_zero = (left_norm == 0) & (right_norm == 0)
    similarity[both_zero] = 1.0
    similarity = similarity.clamp(-1.0, 1.0)
    cka, cka_points, cka_channels = centered_linear_cka(
        left,
        right,
        max_points=cka_max_points,
        max_channels=cka_max_channels,
        sample_weights=area_weights,
    )
    difference = left64 - right64
    return {
        "position_count": count,
        "native_pixel_weight": native_pixel_weight,
        "element_weight": native_pixel_weight * channels,
        "cosine_distance_sum": (
            0.0
            if identical
            else float(((1.0 - similarity) * area_weights).sum())
        ),
        "diff_squared_sum": float(
            0.0
            if identical
            else (difference.square() * area_weights[:, None]).sum()
        ),
        "left_squared_sum": float(
            (left64.square() * area_weights[:, None]).sum()
        ),
        "subsampled_cka": cka,
        "cka_points": cka_points,
        "cka_channels": cka_channels,
    }


def feature_pair_batch_statistics(
    left_feature: torch.Tensor,
    right_feature: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int = NUM_CLASSES,
    cka_max_points: int = 128,
    cka_max_channels: int = 128,
) -> BatchFeaturePairStatistics:
    """Measure one paired window without retaining its raw activations.

    CKA is centered within one window and later averaged over windows.  A batch
    larger than one would change that estimand, so it is rejected explicitly.
    """

    left = _as_bchw(left_feature)
    right = _as_bchw(right_feature)
    if left.shape != right.shape:
        raise ValueError(
            f"paired feature shapes differ: {tuple(left.shape)} != {tuple(right.shape)}"
        )
    if left.shape[0] != 1:
        raise ValueError(
            "feature statistics require exactly one crop window; "
            "set feature_batch_size=1"
        )
    target = torch.as_tensor(target, dtype=torch.int64, device="cpu")
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.ndim != 3 or target.shape[0] != left.shape[0]:
        raise ValueError("target batch must match paired features")

    region_weights, class_area_weights = native_semantic_region_area_weights(
        target,
        left.shape[-2:],
        num_classes,
    )
    left_points = left.permute(0, 2, 3, 1).contiguous()
    right_points = right.permute(0, 2, 3, 1).contiguous()
    regions = {
        name: _region_statistics(
            left_points,
            right_points,
            region_weights[name],
            cka_max_points=cka_max_points,
            cka_max_channels=cka_max_channels,
        )
        for name in PAIR_REGIONS
    }

    channels = int(left.shape[1])
    class_counts = torch.zeros(num_classes, dtype=torch.int64)
    left_class_sums = torch.zeros(num_classes, channels, dtype=torch.float64)
    right_class_sums = torch.zeros_like(left_class_sums)
    flat_left = left_points.reshape(-1, channels).to(torch.float64)
    flat_right = right_points.reshape(-1, channels).to(torch.float64)
    for class_id in range(num_classes):
        weights = class_area_weights[:, class_id].reshape(-1)
        count = int(weights.sum())
        class_counts[class_id] = count
        if count:
            left_class_sums[class_id] = (flat_left * weights[:, None]).sum(dim=0)
            right_class_sums[class_id] = (flat_right * weights[:, None]).sum(dim=0)

    return BatchFeaturePairStatistics(
        channels=channels,
        spatial_shape=tuple(int(value) for value in left.shape[-2:]),
        observations=int(left.shape[0]),
        regions=regions,
        haar={
            "left": orthogonal_haar_energies(left),
            "right": orthogonal_haar_energies(right),
        },
        class_counts=class_counts,
        left_class_sums=left_class_sums,
        right_class_sums=right_class_sums,
    )


class _RegionAccumulator:
    def __init__(self) -> None:
        self.position_count = 0
        self.native_pixel_weight = 0.0
        self.element_weight = 0.0
        self.cosine_distance_sum = 0.0
        self.diff_squared_sum = 0.0
        self.left_squared_sum = 0.0
        self.cka_window_sum = 0.0
        self.cka_windows = 0
        self.cka_sampled_points = 0
        self.cka_channels: set[int] = set()

    def update(self, values: Mapping[str, Any]) -> None:
        self.position_count += int(values["position_count"])
        self.native_pixel_weight += float(values["native_pixel_weight"])
        self.element_weight += float(values["element_weight"])
        self.cosine_distance_sum += float(values["cosine_distance_sum"])
        self.diff_squared_sum += float(values["diff_squared_sum"])
        self.left_squared_sum += float(values["left_squared_sum"])
        cka = values.get("subsampled_cka")
        if cka is not None:
            self.cka_window_sum += float(cka)
            self.cka_windows += 1
            self.cka_sampled_points += int(values["cka_points"])
            self.cka_channels.add(int(values["cka_channels"]))

    def summary(self) -> dict[str, Any]:
        cosine_distance = (
            self.cosine_distance_sum / self.native_pixel_weight
            if self.native_pixel_weight
            else None
        )
        if self.element_weight:
            rms_delta = math.sqrt(self.diff_squared_sum / self.element_weight)
            left_rms = math.sqrt(self.left_squared_sum / self.element_weight)
            if left_rms > 0:
                relative_rms = rms_delta / left_rms
            elif rms_delta == 0:
                relative_rms = 0.0
            else:
                relative_rms = None
        else:
            rms_delta = None
            left_rms = None
            relative_rms = None
        return {
            "position_count": self.position_count,
            "native_pixel_weight": self.native_pixel_weight,
            "element_weight": self.element_weight,
            "cosine_distance": cosine_distance,
            "rms_delta": rms_delta,
            "left_rms": left_rms,
            "relative_rms_to_left": relative_rms,
            "mean_per_window_subsampled_linear_cka": (
                self.cka_window_sum / self.cka_windows
                if self.cka_windows
                else None
            ),
            "cka_windows": self.cka_windows,
            "cka_sampled_points_total": self.cka_sampled_points,
            "cka_channel_counts": sorted(self.cka_channels),
        }


class StagePairAccumulator:
    def __init__(self, num_classes: int, *, include_prototypes: bool) -> None:
        self.num_classes = num_classes
        self.include_prototypes = include_prototypes
        self.channels: int | None = None
        self.spatial_shapes: set[tuple[int, int]] = set()
        self.observations = 0
        self.regions = {name: _RegionAccumulator() for name in PAIR_REGIONS}
        self.haar = {
            side: {band: 0.0 for band in HAAR_BANDS}
            for side in ("left", "right")
        }
        self.class_counts: torch.Tensor | None = None
        self.left_class_sums: torch.Tensor | None = None
        self.right_class_sums: torch.Tensor | None = None

    def update(self, values: BatchFeaturePairStatistics) -> None:
        if self.channels is None:
            self.channels = values.channels
            if self.include_prototypes:
                self.class_counts = torch.zeros(
                    self.num_classes, dtype=torch.int64
                )
                self.left_class_sums = torch.zeros(
                    self.num_classes, self.channels, dtype=torch.float64
                )
                self.right_class_sums = torch.zeros_like(self.left_class_sums)
        elif self.channels != values.channels:
            raise ValueError(
                f"stage channel count changed: {self.channels} != {values.channels}"
            )
        self.spatial_shapes.add(values.spatial_shape)
        self.observations += values.observations
        for name in PAIR_REGIONS:
            self.regions[name].update(values.regions[name])
        for side in ("left", "right"):
            for band in HAAR_BANDS:
                self.haar[side][band] += float(values.haar[side][band])
        if self.include_prototypes:
            assert self.class_counts is not None
            assert self.left_class_sums is not None
            assert self.right_class_sums is not None
            self.class_counts += values.class_counts
            self.left_class_sums += values.left_class_sums
            self.right_class_sums += values.right_class_sums

    @staticmethod
    def _frequency_summary(energies: Mapping[str, float]) -> dict[str, Any]:
        total = float(sum(energies.values()))
        fractions = {
            band: (float(energies[band]) / total if total > 0 else None)
            for band in HAAR_BANDS
        }
        high = sum(float(energies[band]) for band in HAAR_BANDS if band != "ll")
        return {
            "representation_grid_energy": {
                band: float(energies[band]) for band in HAAR_BANDS
            },
            "representation_grid_energy_fraction": fractions,
            "representation_grid_high_frequency_fraction": (
                high / total if total > 0 else None
            ),
            "total_energy": total,
        }

    @staticmethod
    def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> float | None:
        denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        if float(denominator) > 0:
            similarity = float((left @ right) / denominator)
            return 1.0 - max(-1.0, min(1.0, similarity))
        if torch.equal(left, right):
            return 0.0
        return None

    @classmethod
    def _prototype_separation(
        cls,
        prototypes: list[torch.Tensor | None],
    ) -> dict[str, Any]:
        per_class: list[float | None] = [None] * len(prototypes)
        pairwise = []
        for left_id, left in enumerate(prototypes):
            if left is None:
                continue
            distances = []
            for right_id in range(left_id + 1, len(prototypes)):
                right = prototypes[right_id]
                if right is None:
                    continue
                distance = cls._cosine_distance(left, right)
                if distance is not None:
                    pairwise.append(distance)
                    distances.append(distance)
            for right_id in range(left_id):
                right = prototypes[right_id]
                if right is None:
                    continue
                distance = cls._cosine_distance(left, right)
                if distance is not None:
                    distances.append(distance)
            if distances:
                per_class[left_id] = min(distances)
        return {
            "classes_with_prototypes": sum(item is not None for item in prototypes),
            "pair_count": len(pairwise),
            "mean_pairwise_cosine_distance": (
                sum(pairwise) / len(pairwise) if pairwise else None
            ),
            "minimum_pairwise_cosine_distance": min(pairwise) if pairwise else None,
            "nearest_other_class_cosine_distance": per_class,
        }

    def summary(self, left_label: str, right_label: str) -> dict[str, Any]:
        result = {
            "channels": self.channels,
            "spatial_shapes": [list(shape) for shape in sorted(self.spatial_shapes)],
            "observations": self.observations,
            "regions": {
                name: self.regions[name].summary() for name in PAIR_REGIONS
            },
            "frequency": {
                "definition": "one-level orthonormal Haar on each representation grid",
                "weighting": "activation-energy-weighted over channels and crop windows",
                "fixed_physical_frequency_across_stages": False,
                left_label: self._frequency_summary(self.haar["left"]),
                right_label: self._frequency_summary(self.haar["right"]),
            },
        }
        if not self.include_prototypes:
            return result

        assert self.class_counts is not None
        assert self.left_class_sums is not None
        assert self.right_class_sums is not None
        rows = []
        left_prototypes: list[torch.Tensor | None] = []
        right_prototypes: list[torch.Tensor | None] = []
        for class_id in range(self.num_classes):
            count = int(self.class_counts[class_id])
            distance = None
            left = None
            right = None
            if count:
                left = self.left_class_sums[class_id] / count
                right = self.right_class_sums[class_id] / count
                distance = self._cosine_distance(left, right)
            left_prototypes.append(left)
            right_prototypes.append(right)
            rows.append(
                {
                    "class_id": class_id,
                    "interior_native_pixels": count,
                    "prototype_cosine_distance": distance,
                }
            )
        result["by_class"] = rows
        result["prototype_separation"] = {
            left_label: self._prototype_separation(left_prototypes),
            right_label: self._prototype_separation(right_prototypes),
        }
        return result


def _metric_at(mapping: Mapping[str, Any], path: Iterable[str]) -> float | None:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not values:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    tensor = torch.tensor(values, dtype=torch.float64)
    mean = float(tensor.mean())
    if tensor.numel() < 2:
        return {"n": int(tensor.numel()), "mean": mean, "ci95": [None, None]}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        0,
        tensor.numel(),
        (resamples, tensor.numel()),
        generator=generator,
    )
    samples = tensor[indices].mean(dim=1)
    lower, upper = torch.quantile(
        samples, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return {
        "n": int(tensor.numel()),
        "mean": mean,
        "ci95": [float(lower), float(upper)],
    }


def bootstrap_city_cluster_mean_ci(
    values_by_city: Mapping[str, Iterable[float]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Percentile CI for a mean after resampling whole cities."""

    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    clusters = []
    for city in sorted(values_by_city):
        values = [
            float(value)
            for value in values_by_city[city]
            if math.isfinite(float(value))
        ]
        if values:
            tensor = torch.tensor(values, dtype=torch.float64)
            clusters.append((float(tensor.sum()), int(tensor.numel())))
    total_count = sum(count for _, count in clusters)
    total_sum = sum(value_sum for value_sum, _ in clusters)
    mean = total_sum / total_count if total_count else None
    if len(clusters) < 2:
        return {
            "n": total_count,
            "cities": len(clusters),
            "mean": mean,
            "ci95": [None, None],
        }
    statistics = torch.tensor(clusters, dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        0,
        len(clusters),
        (resamples, len(clusters)),
        generator=generator,
    )
    sampled = statistics[indices].sum(dim=1)
    means = sampled[:, 0] / sampled[:, 1]
    lower, upper = torch.quantile(
        means, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return {
        "n": total_count,
        "cities": len(clusters),
        "mean": mean,
        "ci95": [float(lower), float(upper)],
    }


def _stable_seed(base: int, text: str) -> int:
    value = int(base)
    for index, character in enumerate(text, start=1):
        value = (value + index * ord(character)) % (2**31 - 1)
    return value


def _city_from_tile_key(tile_key: str) -> str:
    if "/" not in tile_key:
        raise ValueError(f"tile key must have city/tile form, got {tile_key!r}")
    return tile_key.split("/", 1)[0]


def _tile_and_city_bootstrap(
    values_by_tile: Mapping[str, float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    tile_result = bootstrap_mean_ci(
        values_by_tile.values(), resamples=resamples, seed=seed
    )
    values_by_city: dict[str, list[float]] = {}
    for tile_key, value in values_by_tile.items():
        values_by_city.setdefault(_city_from_tile_key(tile_key), []).append(value)
    city_result = bootstrap_city_cluster_mean_ci(
        values_by_city,
        resamples=resamples,
        seed=_stable_seed(seed, "city_cluster"),
    )
    return {
        **tile_result,
        "tile_bootstrap": tile_result,
        "city_cluster_bootstrap": city_result,
    }


class ScaleTransitionAccumulator:
    """Accumulate paired stage statistics without retaining activations."""

    def __init__(
        self,
        *,
        stage_order: tuple[str, ...] = STAGE_ORDER,
        num_classes: int = NUM_CLASSES,
        left_label: str = "full",
        right_label: str = "sar",
    ) -> None:
        if not left_label or not right_label or left_label == right_label:
            raise ValueError("pair labels must be distinct non-empty strings")
        self.stage_order = tuple(stage_order)
        self.num_classes = num_classes
        self.left_label = left_label
        self.right_label = right_label
        self.pooled: dict[str, StagePairAccumulator] = {}
        self.by_city: dict[str, dict[str, StagePairAccumulator]] = {}
        self.by_tile: dict[str, dict[str, StagePairAccumulator]] = {}

    def update(
        self,
        stage: str,
        values: BatchFeaturePairStatistics,
        *,
        city: str,
        tile_id: str,
    ) -> None:
        if stage not in self.stage_order:
            raise KeyError(f"unknown diagnostic stage: {stage}")
        tile_key = f"{city}/{tile_id}"
        self.pooled.setdefault(
            stage,
            StagePairAccumulator(self.num_classes, include_prototypes=True),
        ).update(values)
        self.by_city.setdefault(city, {}).setdefault(
            stage,
            StagePairAccumulator(self.num_classes, include_prototypes=False),
        ).update(values)
        self.by_tile.setdefault(tile_key, {}).setdefault(
            stage,
            StagePairAccumulator(self.num_classes, include_prototypes=False),
        ).update(values)

    def _scope_summary(
        self, scope: Mapping[str, StagePairAccumulator]
    ) -> dict[str, Any]:
        return {
            stage: scope[stage].summary(self.left_label, self.right_label)
            for stage in self.stage_order
            if stage in scope
        }

    def summary(
        self,
        *,
        bootstrap_resamples: int,
        bootstrap_seed: int,
    ) -> dict[str, Any]:
        pooled = self._scope_summary(self.pooled)
        by_city = {
            city: self._scope_summary(stages)
            for city, stages in sorted(self.by_city.items())
        }
        by_tile = {
            tile: self._scope_summary(stages)
            for tile, stages in sorted(self.by_tile.items())
        }
        bootstrap: dict[str, Any] = {}
        city_cluster_bootstrap: dict[str, Any] = {}
        metric_paths = {
            "valid_cosine_distance": ("regions", "valid", "cosine_distance"),
            "boundary_cosine_distance": (
                "regions",
                "boundary",
                "cosine_distance",
            ),
            "interior_cosine_distance": (
                "regions",
                "interior",
                "cosine_distance",
            ),
            "valid_relative_rms": (
                "regions",
                "valid",
                "relative_rms_to_left",
            ),
            "boundary_relative_rms": (
                "regions",
                "boundary",
                "relative_rms_to_left",
            ),
            "valid_mean_per_window_subsampled_linear_cka": (
                "regions",
                "valid",
                "mean_per_window_subsampled_linear_cka",
            ),
        }
        for stage in self.stage_order:
            if stage not in pooled:
                continue
            stage_result = {}
            stage_city_result = {}
            for metric, path in metric_paths.items():
                values_by_tile = {
                    tile_key: value
                    for tile_key, tile in by_tile.items()
                    if stage in tile
                    for value in [_metric_at(tile[stage], path)]
                    if value is not None
                }
                stage_result[metric] = bootstrap_mean_ci(
                    values_by_tile.values(),
                    resamples=bootstrap_resamples,
                    seed=_stable_seed(bootstrap_seed, f"{stage}:{metric}"),
                )
                values_by_city: dict[str, list[float]] = {}
                for tile_key, value in values_by_tile.items():
                    values_by_city.setdefault(_city_from_tile_key(tile_key), []).append(
                        value
                    )
                stage_city_result[metric] = bootstrap_city_cluster_mean_ci(
                    values_by_city,
                    resamples=bootstrap_resamples,
                    seed=_stable_seed(
                        bootstrap_seed, f"{stage}:{metric}:city_cluster"
                    ),
                )
            bootstrap[stage] = stage_result
            city_cluster_bootstrap[stage] = stage_city_result
        return {
            "pair_labels": {
                "left": self.left_label,
                "right": self.right_label,
            },
            "stage_order": list(self.stage_order),
            "stage_descriptions": {
                stage: STAGE_DESCRIPTIONS.get(
                    stage, CROSS_SCALE_STAGE_DESCRIPTIONS.get(stage, stage)
                )
                for stage in self.stage_order
            },
            "pooled": pooled,
            "by_city": by_city,
            "by_tile": by_tile,
            "tile_bootstrap": bootstrap,
            "city_cluster_bootstrap": city_cluster_bootstrap,
            "inference_policy": {
                "stagewise_statistics": "exploratory",
                "tile_bootstrap_unit": "tile",
                "city_cluster_bootstrap_unit": "city",
            },
        }


def summarize_amplification(
    by_tile: Mapping[str, Mapping[str, Any]],
    *,
    left_label: str,
    right_label: str,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Summarize tile-level after-minus-before gap changes for fixed edges."""

    metric_paths = {
        "valid_cosine_distance_delta": (
            "regions",
            "valid",
            "cosine_distance",
        ),
        "boundary_cosine_distance_delta": (
            "regions",
            "boundary",
            "cosine_distance",
        ),
        "interior_cosine_distance_delta": (
            "regions",
            "interior",
            "cosine_distance",
        ),
        "valid_relative_rms_delta": (
            "regions",
            "valid",
            "relative_rms_to_left",
        ),
        "boundary_relative_rms_delta": (
            "regions",
            "boundary",
            "relative_rms_to_left",
        ),
    }
    result = {}
    for edge_name, before_stage, after_stage, analysis_role in AMPLIFICATION_EDGES:
        edge = {
            "before": before_stage,
            "after": after_stage,
            "analysis_role": analysis_role,
            "metrics": {},
        }
        for metric, path in metric_paths.items():
            values_by_tile = {}
            for tile_key, tile in by_tile.items():
                if before_stage not in tile or after_stage not in tile:
                    continue
                before = _metric_at(tile[before_stage], path)
                after = _metric_at(tile[after_stage], path)
                if before is not None and after is not None:
                    values_by_tile[tile_key] = after - before
            edge["metrics"][metric] = _tile_and_city_bootstrap(
                values_by_tile,
                resamples=resamples,
                seed=_stable_seed(seed, f"{edge_name}:{metric}"),
            )

        frequency_path = "representation_grid_high_frequency_fraction"
        frequency_values_by_tile = {}
        for tile_key, tile in by_tile.items():
            if before_stage not in tile or after_stage not in tile:
                continue
            left_before = _metric_at(
                tile[before_stage], ("frequency", left_label, frequency_path)
            )
            right_before = _metric_at(
                tile[before_stage], ("frequency", right_label, frequency_path)
            )
            left_after = _metric_at(
                tile[after_stage], ("frequency", left_label, frequency_path)
            )
            right_after = _metric_at(
                tile[after_stage], ("frequency", right_label, frequency_path)
            )
            if None not in (left_before, right_before, left_after, right_after):
                frequency_values_by_tile[tile_key] = (
                    (right_after - left_after) - (right_before - left_before)
                )
        frequency_metric = (
            f"{right_label}_minus_{left_label}_representation_grid_hf_"
            "fraction_difference_in_differences"
        )
        edge["metrics"][frequency_metric] = _tile_and_city_bootstrap(
            frequency_values_by_tile,
            resamples=resamples,
            seed=_stable_seed(seed, f"{edge_name}:hf_difference_in_differences"),
        )
        edge["frequency_interpretation"] = {
            "fixed_physical_frequency": False,
            "estimand": (
                f"({right_label}-{left_label})_after - "
                f"({right_label}-{left_label})_before"
            ),
            "warning": (
                "one-level Haar bands are relative to each representation grid; "
                "this is not a fixed-physical-frequency retention estimate"
            ),
        }
        result[edge_name] = edge
    return {
        "inference_policy": {
            "primary_edges": ["adapter_stride2_P5"],
            "negative_control_edges": ["adapter_identity_P4_control"],
            "descriptive_control_edges": ["prn_nearest_P5_to_P4"],
            "fam_primary_test": "cross_scale_alignment.prn.cross_scale.P5_to_P4",
            "all_other_edges": "exploratory",
        },
        "edges": result,
        # Direct aliases keep existing report readers usable while the schema
        # migrates to the explicit ``edges`` namespace.
        **result,
    }


def summarize_cross_scale_alignment_degradation(
    full_by_tile: Mapping[str, Mapping[str, Any]],
    sar_by_tile: Mapping[str, Mapping[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare within-endpoint coarse/lateral alignment for SAR versus Full.

    Each input is the ``by_tile`` summary of an endpoint-specific accumulator
    whose pair is ``(resized coarse, same-grid lateral)``.  Positive cosine/RMS
    deltas mean worse SAR alignment; negative CKA deltas mean worse SAR
    alignment because CKA is a similarity.
    """

    metric_paths = {
        "valid_cosine_distance_sar_minus_full": (
            "regions",
            "valid",
            "cosine_distance",
        ),
        "boundary_cosine_distance_sar_minus_full": (
            "regions",
            "boundary",
            "cosine_distance",
        ),
        "interior_cosine_distance_sar_minus_full": (
            "regions",
            "interior",
            "cosine_distance",
        ),
        "valid_relative_rms_sar_minus_full": (
            "regions",
            "valid",
            "relative_rms_to_left",
        ),
        "boundary_relative_rms_sar_minus_full": (
            "regions",
            "boundary",
            "relative_rms_to_left",
        ),
        "valid_mean_per_window_subsampled_linear_cka_sar_minus_full": (
            "regions",
            "valid",
            "mean_per_window_subsampled_linear_cka",
        ),
    }
    stage_results = {}
    for stage in CROSS_SCALE_STAGE_ORDER:
        role = "primary" if stage == CROSS_SCALE_STAGE_ORDER[0] else "exploratory"
        stage_result = {
            "description": CROSS_SCALE_STAGE_DESCRIPTIONS[stage],
            "analysis_role": role,
            "pair_within_each_endpoint": {
                "left": "resized_coarse",
                "right": "same_grid_lateral",
            },
            "metrics": {},
        }
        for metric, path in metric_paths.items():
            values_by_tile = {}
            for tile_key in sorted(set(full_by_tile) & set(sar_by_tile)):
                full_stages = full_by_tile[tile_key]
                sar_stages = sar_by_tile[tile_key]
                if stage not in full_stages or stage not in sar_stages:
                    continue
                full_value = _metric_at(full_stages[stage], path)
                sar_value = _metric_at(sar_stages[stage], path)
                if full_value is not None and sar_value is not None:
                    values_by_tile[tile_key] = sar_value - full_value
            stage_result["metrics"][metric] = _tile_and_city_bootstrap(
                values_by_tile,
                resamples=resamples,
                seed=_stable_seed(seed, f"cross_scale:{stage}:{metric}"),
            )
        stage_results[stage] = stage_result
    return {
        "inference_policy": {
            "primary_stage": "prn.cross_scale.P5_to_P4",
            "other_stages": "exploratory",
            "estimand": "SAR within-endpoint alignment metric minus Full metric",
            "distance_delta_direction": "positive means worse SAR alignment",
            "cka_delta_direction": "negative means worse SAR alignment",
        },
        "stages": stage_results,
        **stage_results,
    }


def segmentation_region_statistics(
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int = NUM_CLASSES,
    left_label: str = "full",
    right_label: str = "sar",
) -> dict[str, Any]:
    """Return endpoint error/disagreement rates on boundary and interior pixels."""

    left_logits = torch.as_tensor(left_logits, dtype=torch.float32, device="cpu")
    right_logits = torch.as_tensor(right_logits, dtype=torch.float32, device="cpu")
    target = torch.as_tensor(target, dtype=torch.int64, device="cpu")
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if (
        left_logits.ndim != 4
        or right_logits.shape != left_logits.shape
        or target.shape != left_logits.shape[:1] + left_logits.shape[-2:]
    ):
        raise ValueError("logits and target shapes do not match")
    if left_logits.shape[1] != num_classes:
        raise ValueError("logit channel count does not match num_classes")
    if not torch.isfinite(left_logits).all() or not torch.isfinite(right_logits).all():
        raise ValueError("logits contain non-finite values")

    left_prediction = left_logits.argmax(dim=1)
    right_prediction = right_logits.argmax(dim=1)
    valid = (target >= 0) & (target < num_classes)
    boundary = semantic_boundary_mask(target, num_classes) & valid
    regions = {
        "valid": valid,
        "boundary": boundary,
        "interior": valid & ~boundary,
    }
    result: dict[str, Any] = {"regions": {}, "by_class": []}
    result["pair_labels"] = {"left": left_label, "right": right_label}

    def paired_outcome(mask: torch.Tensor) -> dict[str, Any]:
        count = int(mask.sum())
        left_error_mask = (left_prediction != target) & mask
        right_error_mask = (right_prediction != target) & mask
        left_errors = int(left_error_mask.sum())
        right_errors = int(right_error_mask.sum())
        left_correct_right_wrong = int((~left_error_mask & right_error_mask & mask).sum())
        right_correct_left_wrong = int((~right_error_mask & left_error_mask & mask).sum())
        disagreement = int(((left_prediction != right_prediction) & mask).sum())
        return {
            "pixels": count,
            left_label: {
                "error_pixels": left_errors,
                "error_rate": left_errors / count if count else None,
            },
            right_label: {
                "error_pixels": right_errors,
                "error_rate": right_errors / count if count else None,
            },
            "endpoint_disagreement_pixels": disagreement,
            "endpoint_disagreement_rate": disagreement / count if count else None,
            "relative_degradation": {
                "right_minus_left_error_rate": (
                    (right_errors - left_errors) / count if count else None
                ),
                "left_correct_right_wrong_pixels": left_correct_right_wrong,
                "left_correct_right_wrong_rate": (
                    left_correct_right_wrong / count if count else None
                ),
                "right_correct_left_wrong_pixels": right_correct_left_wrong,
                "right_correct_left_wrong_rate": (
                    right_correct_left_wrong / count if count else None
                ),
            },
        }

    for name, mask in regions.items():
        result["regions"][name] = paired_outcome(mask)
    for class_id in range(num_classes):
        mask = valid & (target == class_id)
        row = paired_outcome(mask)
        row["class_id"] = class_id
        result["by_class"].append(row)
    return result


def pearson_correlation(
    x_values: Iterable[float],
    y_values: Iterable[float],
) -> dict[str, Any]:
    x_values = list(x_values)
    y_values = list(y_values)
    if len(x_values) != len(y_values):
        raise ValueError("Pearson inputs must have equal lengths")
    pairs = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values, strict=True)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return {"n": len(pairs), "r": None}
    x = torch.tensor([pair[0] for pair in pairs], dtype=torch.float64)
    y = torch.tensor([pair[1] for pair in pairs], dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) == 0.0:
        return {"n": len(pairs), "r": None}
    value = float((x @ y) / denominator)
    return {"n": len(pairs), "r": max(-1.0, min(1.0, value))}


def _city_cluster_correlation(
    rows: list[tuple[str, float, float]],
    *,
    resamples: int,
    seed: int,
    city_demeaned: bool,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    finite_rows = [
        (str(city), float(x), float(y))
        for city, x, y in rows
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    by_city: dict[str, list[tuple[float, float]]] = {}
    for city, x, y in finite_rows:
        by_city.setdefault(city, []).append((x, y))
    if city_demeaned:
        transformed = []
        for city in sorted(by_city):
            values = by_city[city]
            mean_x = sum(x for x, _ in values) / len(values)
            mean_y = sum(y for _, y in values) / len(values)
            transformed.extend(
                (city, x - mean_x, y - mean_y) for x, y in values
            )
    else:
        transformed = finite_rows

    point = pearson_correlation(
        [x for _, x, _ in transformed],
        [y for _, _, y in transformed],
    )
    cluster_statistics = []
    for city in sorted({city for city, _, _ in transformed}):
        values = [(x, y) for row_city, x, y in transformed if row_city == city]
        x = torch.tensor([item[0] for item in values], dtype=torch.float64)
        y = torch.tensor([item[1] for item in values], dtype=torch.float64)
        cluster_statistics.append(
            [
                float(x.numel()),
                float(x.sum()),
                float(y.sum()),
                float((x * x).sum()),
                float((y * y).sum()),
                float((x * y).sum()),
            ]
        )
    result = {
        "n": point["n"],
        "cities": len(cluster_statistics),
        "r": point["r"],
        "city_cluster_ci95": [None, None],
        "valid_bootstrap_resamples": 0,
    }
    if point["n"] < 3 or len(cluster_statistics) < 2:
        return result
    statistics = torch.tensor(cluster_statistics, dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randint(
        0,
        len(cluster_statistics),
        (resamples, len(cluster_statistics)),
        generator=generator,
    )
    sampled = statistics[indices].sum(dim=1)
    n, sum_x, sum_y, sum_xx, sum_yy, sum_xy = sampled.unbind(dim=1)
    covariance = sum_xy - sum_x * sum_y / n
    variance_x = (sum_xx - sum_x.square() / n).clamp_min(0.0)
    variance_y = (sum_yy - sum_y.square() / n).clamp_min(0.0)
    denominator = torch.sqrt(variance_x * variance_y)
    valid = denominator > 0
    correlations = (covariance[valid] / denominator[valid]).clamp(-1.0, 1.0)
    result["valid_bootstrap_resamples"] = int(correlations.numel())
    if correlations.numel():
        lower, upper = torch.quantile(
            correlations, torch.tensor([0.025, 0.975], dtype=torch.float64)
        )
        result["city_cluster_ci95"] = [float(lower), float(upper)]
    return result


def summarize_error_correlations(
    by_tile: Mapping[str, Mapping[str, Any]],
    tile_outcomes: Mapping[str, Mapping[str, Any]],
    *,
    left_label: str = "full",
    right_label: str = "sar",
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Relate feature gaps to paired deployment degradation by tile/city."""

    result: dict[str, Any] = {
        "inference_policy": {
            "analysis_role": "exploratory",
            "primary_outcome": f"{right_label}_minus_{left_label}_error_rate",
            "confidence_interval": "city-cluster percentile bootstrap",
        }
    }
    for stage in STAGE_ORDER:
        rows = []
        for tile_key, stages in by_tile.items():
            if stage not in stages or tile_key not in tile_outcomes:
                continue
            gap_valid = _metric_at(
                stages[stage], ("regions", "valid", "cosine_distance")
            )
            gap_boundary = _metric_at(
                stages[stage], ("regions", "boundary", "cosine_distance")
            )
            hf_left = _metric_at(
                stages[stage],
                (
                    "frequency",
                    left_label,
                    "representation_grid_high_frequency_fraction",
                ),
            )
            hf_right = _metric_at(
                stages[stage],
                (
                    "frequency",
                    right_label,
                    "representation_grid_high_frequency_fraction",
                ),
            )
            degradation_valid = _metric_at(
                tile_outcomes[tile_key],
                (
                    "regions",
                    "valid",
                    "relative_degradation",
                    "right_minus_left_error_rate",
                ),
            )
            degradation_boundary = _metric_at(
                tile_outcomes[tile_key],
                (
                    "regions",
                    "boundary",
                    "relative_degradation",
                    "right_minus_left_error_rate",
                ),
            )
            harmful_boundary = _metric_at(
                tile_outcomes[tile_key],
                (
                    "regions",
                    "boundary",
                    "relative_degradation",
                    "left_correct_right_wrong_rate",
                ),
            )
            rows.append(
                {
                    "city": _city_from_tile_key(tile_key),
                    "gap_valid": gap_valid,
                    "gap_boundary": gap_boundary,
                    "hf_difference": (
                        hf_right - hf_left
                        if hf_left is not None and hf_right is not None
                        else None
                    ),
                    "degradation_valid": degradation_valid,
                    "degradation_boundary": degradation_boundary,
                    "harmful_boundary": harmful_boundary,
                }
            )

        def correlation(x_name: str, y_name: str) -> dict[str, Any]:
            pairs = [
                (row["city"], row[x_name], row[y_name])
                for row in rows
                if row[x_name] is not None and row[y_name] is not None
            ]
            return {
                "raw": _city_cluster_correlation(
                    pairs,
                    resamples=resamples,
                    seed=_stable_seed(seed, f"{stage}:{x_name}:{y_name}:raw"),
                    city_demeaned=False,
                ),
                "city_demeaned": _city_cluster_correlation(
                    pairs,
                    resamples=resamples,
                    seed=_stable_seed(
                        seed, f"{stage}:{x_name}:{y_name}:city_demeaned"
                    ),
                    city_demeaned=True,
                ),
            }

        result[stage] = {
            "valid_gap_vs_relative_error_degradation": correlation(
                "gap_valid", "degradation_valid"
            ),
            "boundary_gap_vs_relative_error_degradation": correlation(
                "gap_boundary", "degradation_boundary"
            ),
            "boundary_gap_vs_left_correct_right_wrong_rate": correlation(
                "gap_boundary", "harmful_boundary"
            ),
            "right_minus_left_representation_grid_hf_vs_boundary_degradation": correlation(
                "hf_difference", "degradation_boundary"
            ),
        }
    return result


def strict_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a deterministic report without overwriting evidence."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic report: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(strict_json_text(payload))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-filesystem hard link publishes the fully fsynced inode and
            # fails atomically if another process already created ``path``.
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite diagnostic report created concurrently: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
