"""Reusable helpers for read-only auxiliary-modality diagnostics.

The helpers in this module intentionally live outside ``tasks/segmentation``.
They instrument the released model and datasets without changing the faithful
MM-DINO implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn


DEFAULT_MAX_SAMPLES = 1_000_000


def _sample_flat(values: np.ndarray, max_samples: int) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if flat.size <= max_samples:
        return flat
    indices = np.linspace(0, flat.size - 1, num=max_samples, dtype=np.int64)
    return flat[indices]


def array_summary(
    values: np.ndarray,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> dict[str, Any]:
    """Return JSON-serializable distribution statistics for an array."""

    array = np.asarray(values)
    sampled = _sample_flat(array, max_samples)
    finite_mask = np.isfinite(sampled)
    finite = sampled[finite_mask]
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "sampled_elements": int(sampled.size),
        "finite_ratio": float(finite_mask.mean()) if sampled.size else 1.0,
    }
    if finite.size == 0:
        result.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "nonzero_ratio": None,
                "quantiles": {},
                "unique_count_sampled": 0,
            }
        )
        return result

    finite64 = finite.astype(np.float64, copy=False)
    quantile_values = np.quantile(
        finite64,
        [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0],
    )
    result.update(
        {
            "min": float(quantile_values[0]),
            "max": float(quantile_values[-1]),
            "mean": float(finite64.mean()),
            "std": float(finite64.std()),
            "nonzero_ratio": float(np.count_nonzero(finite) / finite.size),
            "quantiles": {
                name: float(value)
                for name, value in zip(
                    ("p00", "p01", "p05", "p50", "p95", "p99", "p100"),
                    quantile_values,
                    strict=True,
                )
            },
            "unique_count_sampled": int(np.unique(finite).size),
        }
    )
    return result


def tensor_summary(
    tensor: torch.Tensor,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> dict[str, Any]:
    """Return distribution statistics without retaining a tensor reference."""

    return array_summary(
        tensor.detach().cpu().numpy(),
        max_samples=max_samples,
    )


def to_grayscale(values: np.ndarray) -> np.ndarray:
    """Convert common HWC/CHW arrays to a float32 grayscale image."""

    array = np.asarray(values)
    if array.ndim == 2:
        return array.astype(np.float32, copy=False)
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got {array.shape}")

    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        return array[..., 0].astype(np.float32, copy=False)
    if array.shape[-1] < 3:
        raise ValueError(f"Cannot infer channels for image with shape {array.shape}")

    rgb = array[..., :3].astype(np.float32, copy=False)
    return (
        0.2989 * rgb[..., 0]
        + 0.5870 * rgb[..., 1]
        + 0.1140 * rgb[..., 2]
    )


def gradient_magnitude(values: np.ndarray) -> np.ndarray:
    gray = to_grayscale(values)
    grad_y, grad_x = np.gradient(gray.astype(np.float32, copy=False))
    return np.hypot(grad_x, grad_y)


def _center_crop_image(values: np.ndarray, max_side: int) -> np.ndarray:
    array = np.asarray(values)
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    if array.ndim == 2:
        height, width = array.shape
        top = max((height - max_side) // 2, 0)
        left = max((width - max_side) // 2, 0)
        return array[top : top + min(height, max_side), left : left + min(width, max_side)]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D image, got {array.shape}")
    channel_first = array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4)
    height, width = (array.shape[1], array.shape[2]) if channel_first else array.shape[:2]
    top = max((height - max_side) // 2, 0)
    left = max((width - max_side) // 2, 0)
    if channel_first:
        return array[:, top : top + min(height, max_side), left : left + min(width, max_side)]
    return array[top : top + min(height, max_side), left : left + min(width, max_side), :]


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.0:
        return None
    return float(np.dot(x, y) / denominator)


def _shifted_overlap(
    reference: np.ndarray,
    moving: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != moving.shape:
        raise ValueError(
            f"Alignment inputs must have equal shape, got {reference.shape} and "
            f"{moving.shape}"
        )

    height, width = reference.shape
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError(f"Shift {(dy, dx)} is too large for shape {reference.shape}")

    ref_y = slice(max(dy, 0), min(height + dy, height))
    mov_y = slice(max(-dy, 0), min(height - dy, height))
    ref_x = slice(max(dx, 0), min(width + dx, width))
    mov_x = slice(max(-dx, 0), min(width - dx, width))
    return reference[ref_y, ref_x], moving[mov_y, mov_x]


def edge_alignment_summary(
    rgb: np.ndarray,
    auxiliary: np.ndarray,
    *,
    offsets: Sequence[int] = (-4, -2, -1, 0, 1, 2, 4),
    max_side: int = 1024,
) -> dict[str, Any]:
    """Report zero-shift and best-shift edge correlations.

    This is a screening statistic, not a registration estimator. Cross-sensor
    edges may have low correlation even when the files are correctly aligned.
    """

    rgb_crop = _center_crop_image(rgb, max_side)
    aux_crop = _center_crop_image(auxiliary, max_side)
    rgb_edges = gradient_magnitude(rgb_crop)
    aux_edges = gradient_magnitude(aux_crop)
    if rgb_edges.shape != aux_edges.shape:
        raise ValueError(
            f"RGB/Aux spatial shapes differ: {rgb_edges.shape} vs {aux_edges.shape}"
        )

    correlations: list[dict[str, Any]] = []
    for dy in offsets:
        for dx in offsets:
            ref, moving = _shifted_overlap(rgb_edges, aux_edges, int(dy), int(dx))
            correlation = _pearson(ref, moving)
            correlations.append(
                {"dy": int(dy), "dx": int(dx), "correlation": correlation}
            )

    valid = [item for item in correlations if item["correlation"] is not None]
    best = max(valid, key=lambda item: item["correlation"]) if valid else None
    zero = next(
        item for item in correlations if item["dy"] == 0 and item["dx"] == 0
    )
    return {
        "zero_shift_correlation": zero["correlation"],
        "best": best,
        "offsets": [int(value) for value in offsets],
        "analysis_shape": list(rgb_edges.shape),
    }


def fixed_derangement(length: int, seed: int) -> tuple[list[int], int]:
    """Return a deterministic cyclic derangement and its offset."""

    if length < 2:
        raise ValueError("A shuffled auxiliary condition requires at least 2 items")
    generator = np.random.default_rng(seed)
    offset = int(generator.integers(1, length))
    permutation = [int((index + offset) % length) for index in range(length)]
    return permutation, offset


class AuxiliaryConditionDataset(torch.utils.data.Dataset):
    """Apply distribution-preserving auxiliary interventions to a dataset."""

    MODES = {"normal", "aux-mean", "aux-shuffle"}

    def __init__(self, dataset: Any, mode: str, *, seed: int = 42):
        if mode not in self.MODES:
            raise ValueError(f"Unsupported auxiliary dataset mode: {mode}")
        self.dataset = dataset
        self.mode = mode
        self.permutation: list[int] | None = None
        self.permutation_offset: int | None = None
        if mode == "aux-shuffle":
            self.permutation, self.permutation_offset = fixed_derangement(
                len(dataset), seed
            )

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _unpack(item: Any) -> tuple[torch.Tensor, torch.Tensor, Any]:
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise ValueError(
                "Expected a multimodal dataset item (rgb, auxiliary, label)"
            )
        rgb, auxiliary, label = item
        if not isinstance(auxiliary, torch.Tensor):
            raise TypeError("Expected the auxiliary input to be a torch.Tensor")
        return rgb, auxiliary, label

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor, Any]:
        rgb, auxiliary, label = self._unpack(self.dataset[index])
        if self.mode == "aux-mean":
            auxiliary = torch.ones_like(auxiliary) * auxiliary.mean()
        elif self.mode == "aux-shuffle":
            assert self.permutation is not None
            _, auxiliary, _ = self._unpack(self.dataset[self.permutation[index]])
        if tuple(rgb.shape[-2:]) != tuple(auxiliary.shape[-2:]):
            raise ValueError(
                f"RGB/Aux shapes differ after intervention: {rgb.shape} vs "
                f"{auxiliary.shape}"
            )
        return rgb, auxiliary, label


def modality_weight_summary(
    adapter: nn.Module,
    *,
    auxiliary_scale: float = 1.0,
) -> dict[str, Any]:
    """Read the released adapter's raw and effective global modality weights."""

    if auxiliary_scale < 0.0:
        raise ValueError("auxiliary_scale must be non-negative")
    num_modalities = int(getattr(adapter, "num_modalities"))
    if num_modalities < 2:
        raise ValueError("Expected a multimodal SampleAdapter")

    raw = []
    sigmoid = []
    scaled = []
    for index in range(num_modalities):
        parameter = adapter.modality_weights[f"weight_modality_{index}"]
        raw_value = float(parameter.detach().cpu().item())
        sigmoid_value = float(torch.sigmoid(parameter.detach()).cpu().item())
        scale = auxiliary_scale if index == 1 else 1.0
        raw.append(raw_value)
        sigmoid.append(sigmoid_value)
        scaled.append(sigmoid_value * scale)
    denominator = sum(scaled)
    if denominator <= 0.0:
        raise ValueError("Scaled modality weights sum to zero")
    return {
        "raw": raw,
        "sigmoid": sigmoid,
        "auxiliary_scale": float(auxiliary_scale),
        "effective_normalized": [float(value / denominator) for value in scaled],
    }


class ScaledSampleAdapter(nn.Module):
    """Diagnostic wrapper that rescales Aux before released weight normalization.

    With ``auxiliary_scale=1`` this wrapper must be numerically identical to the
    released multimodal SampleAdapter.  A scale of zero removes Aux from the
    weighted sum and renormalizes the remaining modality weights.
    """

    def __init__(self, delegate: nn.Module, *, auxiliary_scale: float):
        super().__init__()
        if auxiliary_scale < 0.0:
            raise ValueError("auxiliary_scale must be non-negative")
        if int(getattr(delegate, "num_modalities")) < 2:
            raise ValueError("Expected a multimodal SampleAdapter")
        self.delegate = delegate
        self.auxiliary_scale = float(auxiliary_scale)

    def forward(
        self,
        *features_list: Sequence[torch.Tensor],
        patch_h: int | None = None,
        patch_w: int | None = None,
    ) -> Any:
        if len(features_list) <= 1:
            return self.delegate(
                *features_list,
                patch_h=patch_h,
                patch_w=patch_w,
            )
        if len(features_list) != self.delegate.num_modalities:
            raise ValueError(
                f"Number of modalities ({len(features_list)}) does not match "
                f"the adapter ({self.delegate.num_modalities})"
            )
        if patch_h is None or patch_w is None:
            raise ValueError("patch_h and patch_w are required")

        outputs: list[list[torch.Tensor]] = [
            [] for _ in range(len(features_list))
        ]
        for layer_index, modality_features in enumerate(zip(*features_list)):
            processed = []
            for feature in modality_features:
                feature = feature.permute(0, 2, 1).reshape(
                    feature.shape[0], feature.shape[-1], patch_h, patch_w
                )
                feature = self.delegate.projects[layer_index](feature)
                feature = self.delegate.resize_layers[layer_index](feature)
                processed.append(feature)

            weights = []
            for modality_index in range(len(processed)):
                parameter = self.delegate.modality_weights[
                    f"weight_modality_{modality_index}"
                ]
                weight = torch.sigmoid(parameter)
                if modality_index == 1:
                    weight = weight * self.auxiliary_scale
                weights.append(weight)
            denominator = sum(weights)
            if bool((denominator <= 0).item()):
                raise ValueError("Scaled modality weights sum to zero")
            normalized = [weight / denominator for weight in weights]
            fused = sum(
                weight * feature
                for weight, feature in zip(normalized, processed, strict=True)
            )
            for output in outputs:
                output.append(fused)
        return outputs
