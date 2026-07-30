"""Torch execution primitive for selected Stage-B1 sliding-window crops.

Unlike the public ``slide_inference`` baseline, this helper intentionally
allows uncovered pixels and exposes raw sum/count maps plus the actual crop
execution trace.  It does not choose a policy or alter the official baseline.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _canonical_crop_ids(crop_ids: Sequence[int], crop_count: int) -> tuple[int, ...]:
    raw = tuple(crop_ids)
    if any(isinstance(value, (bool, np.bool_)) for value in raw):
        raise TypeError("crop ids must not contain booleans")
    if any(not isinstance(value, (int, np.integer)) for value in raw):
        raise TypeError("crop ids must contain integers")
    values = tuple(int(value) for value in raw)
    if values != tuple(sorted(set(values))):
        raise ValueError("crop ids must be unique and strictly increasing")
    if any(value < 0 or value >= crop_count for value in values):
        raise ValueError("crop id is out of range")
    return values


def _decode_model_output(output: Any, decoder_head_type: str) -> torch.Tensor:
    if decoder_head_type == "linear":
        if not isinstance(output, torch.Tensor):
            raise TypeError("linear decoder output must be a tensor")
        return output
    if decoder_head_type == "m2f":
        if not isinstance(output, dict):
            raise TypeError("m2f decoder output must be a mapping")
        mask_pred, mask_cls = output["pred_masks"], output["pred_logits"]
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        return torch.einsum(
            "bqc,bqhw->bchw", mask_cls.to(torch.float), mask_pred.to(torch.float)
        )
    raise ValueError("decoder_head_type must be 'linear' or 'm2f'")


def forward_selected_phase_crops(
    inputs: torch.Tensor,
    dsm: torch.Tensor,
    model: torch.nn.Module,
    windows: Sequence[tuple[int, int, int, int]],
    crop_ids: Sequence[int],
    *,
    n_output_channels: int,
    batch_size: int,
    decoder_head_type: str = "linear",
    require_full_coverage: bool = False,
) -> dict[str, Any]:
    """Execute selected crops in ascending local-ID order and expose sum/count."""

    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 4 or inputs.shape[0] != 1:
        raise ValueError("inputs must be a BCHW tensor with batch size one")
    if not isinstance(dsm, torch.Tensor) or dsm.ndim != 4 or dsm.shape[0] != 1:
        raise ValueError("dsm must be a BCHW tensor with batch size one")
    if inputs.device != dsm.device or inputs.shape[-2:] != dsm.shape[-2:]:
        raise ValueError("RGB and SAR must share device and spatial shape")
    if not inputs.is_floating_point() or not dsm.is_floating_point():
        raise TypeError("RGB and SAR tensors must be floating point")
    n_output_channels = int(n_output_channels)
    batch_size = int(batch_size)
    if n_output_channels <= 0 or batch_size <= 0:
        raise ValueError("n_output_channels and batch_size must be positive")
    concrete_windows = tuple(tuple(int(value) for value in window) for window in windows)
    selected = _canonical_crop_ids(crop_ids, len(concrete_windows))
    height, width = (int(value) for value in inputs.shape[-2:])
    for y0, y1, x0, x1 in concrete_windows:
        if y0 < 0 or x0 < 0 or y0 >= y1 or x0 >= x1:
            raise ValueError("crop window is invalid")
        if y1 > height or x1 > width:
            raise ValueError("crop window lies outside the input canvas")

    score_sum = inputs.new_zeros((1, n_output_channels, height, width))
    count_mat = torch.zeros(
        (1, 1, height, width), dtype=torch.int16, device=inputs.device
    )
    observed: list[int] = []
    batch_sizes: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(selected), batch_size):
            batch_ids = selected[offset : offset + batch_size]
            optical_crops = []
            sar_crops = []
            for crop_id in batch_ids:
                y0, y1, x0, x1 = concrete_windows[crop_id]
                optical_crops.append(inputs[:, :, y0:y1, x0:x1])
                sar_crops.append(dsm[:, :, y0:y1, x0:x1])
            if not batch_ids:
                continue
            optical_batch = torch.cat(optical_crops, dim=0)
            sar_batch = torch.cat(sar_crops, dim=0)
            decoded = _decode_model_output(
                model(optical_batch, sar_batch), decoder_head_type
            )
            if decoded.ndim != 4 or decoded.shape[0] != len(batch_ids):
                raise ValueError("model crop output has an invalid batch shape")
            if decoded.shape[1] != n_output_channels:
                raise ValueError("model crop output has an unexpected channel count")
            if not torch.isfinite(decoded).all():
                raise ValueError("model crop output contains non-finite values")
            decoded = decoded.to(device=inputs.device, dtype=score_sum.dtype)
            for batch_index, crop_id in enumerate(batch_ids):
                y0, y1, x0, x1 = concrete_windows[crop_id]
                expected_shape = (y1 - y0, x1 - x0)
                if tuple(decoded.shape[-2:]) != expected_shape:
                    raise ValueError("model crop output differs from its window shape")
                score_sum[:, :, y0:y1, x0:x1] += decoded[batch_index]
                count_mat[:, :, y0:y1, x0:x1] += 1
                observed.append(crop_id)
            batch_sizes.append(len(batch_ids))
    if inputs.is_cuda:
        torch.cuda.synchronize(inputs.device)
    elapsed = time.perf_counter() - started
    if tuple(observed) != selected:
        raise AssertionError("actual crop execution order differs from the plan")
    if require_full_coverage and torch.any(count_mat == 0):
        raise AssertionError("full slide left an uncovered pixel")
    return {
        "sum_logits": np.ascontiguousarray(score_sum[0].cpu().numpy()),
        "count_mat": np.ascontiguousarray(count_mat[0, 0].cpu().numpy()),
        "observed_crop_ids": tuple(observed),
        "crop_samples": len(observed),
        "batch_calls": len(batch_sizes),
        "batch_sizes": tuple(batch_sizes),
        "mean_actual_batch_size": (
            float(np.mean(batch_sizes)) if batch_sizes else None
        ),
        "wall_seconds": float(elapsed),
    }


def forward_selected_phase_key_crops(
    phase_tensors: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    windows: Sequence[tuple[int, int, int, int]],
    crop_ids_by_phase: Mapping[str, Sequence[int]],
    *,
    phase_order: Sequence[str],
    n_output_channels: int,
    batch_size: int,
    decoder_head_type: str = "linear",
) -> dict[str, Any]:
    """Execute one image's phase-crop keys with a fixed model batch shape.

    Real keys are flattened in ``phase_order`` and then ascending local crop-ID
    order.  If the final batch is short, crops from that same final real batch
    are repeated cyclically until the frozen batch size is reached.  Repeated
    outputs are deliberately discarded: they never affect a phase sum/count or
    the observed real-key trace.

    Keeping the physical model batch shape fixed is an execution invariant,
    not an optional optimization.  The returned accounting therefore separates
    selected (logical) crop samples from all samples actually sent to the model.
    """

    phases = tuple(phase_order)
    if not phases or any(not isinstance(name, str) or not name for name in phases):
        raise ValueError("phase_order must contain non-empty phase names")
    if len(set(phases)) != len(phases):
        raise ValueError("phase_order must not contain duplicate phase names")
    if set(crop_ids_by_phase) != set(phases):
        raise ValueError("crop_ids_by_phase must contain exactly phase_order")
    n_output_channels = int(n_output_channels)
    batch_size = int(batch_size)
    if n_output_channels <= 0 or batch_size <= 0:
        raise ValueError("n_output_channels and batch_size must be positive")

    concrete_windows = tuple(tuple(int(value) for value in window) for window in windows)
    selected_by_phase = {
        phase_name: _canonical_crop_ids(
            crop_ids_by_phase[phase_name], len(concrete_windows)
        )
        for phase_name in phases
    }
    active_phases = tuple(
        phase_name for phase_name in phases if selected_by_phase[phase_name]
    )
    if set(phase_tensors) != set(active_phases):
        raise ValueError("phase_tensors must contain exactly the active phases")
    if not active_phases:
        return {
            "phases": {},
            "observed_phase_crop_keys": (),
            "processed_phase_crop_keys": (),
            "padding_phase_crop_keys": (),
            "selected_crop_samples": 0,
            "model_forward_crop_samples": 0,
            "padding_crop_samples": 0,
            "batch_calls": 0,
            "real_batch_sizes": (),
            "model_batch_sizes": (),
            "mean_real_batch_size": None,
            "mean_model_batch_size": None,
            "wall_seconds": 0.0,
            "padding_policy": "none-empty-plan",
        }

    reference_inputs: torch.Tensor | None = None
    reference_dsm: torch.Tensor | None = None
    height = width = 0
    for phase_name in active_phases:
        pair = phase_tensors[phase_name]
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("each phase_tensors value must be an RGB/SAR tensor tuple")
        inputs, dsm = pair
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim != 4
            or inputs.shape[0] != 1
        ):
            raise ValueError("phase RGB inputs must be BCHW tensors with batch size one")
        if not isinstance(dsm, torch.Tensor) or dsm.ndim != 4 or dsm.shape[0] != 1:
            raise ValueError("phase SAR inputs must be BCHW tensors with batch size one")
        if inputs.device != dsm.device or inputs.shape[-2:] != dsm.shape[-2:]:
            raise ValueError("phase RGB and SAR must share device and spatial shape")
        if not inputs.is_floating_point() or not dsm.is_floating_point():
            raise TypeError("phase RGB and SAR tensors must be floating point")
        if reference_inputs is None:
            reference_inputs, reference_dsm = inputs, dsm
            height, width = (int(value) for value in inputs.shape[-2:])
        elif (
            inputs.device != reference_inputs.device
            or inputs.dtype != reference_inputs.dtype
            or dsm.dtype != reference_dsm.dtype
            or tuple(inputs.shape) != tuple(reference_inputs.shape)
            or tuple(dsm.shape) != tuple(reference_dsm.shape)
        ):
            raise ValueError("all active phase tensors must share shape, dtype, and device")

    for y0, y1, x0, x1 in concrete_windows:
        if y0 < 0 or x0 < 0 or y0 >= y1 or x0 >= x1:
            raise ValueError("crop window is invalid")
        if y1 > height or x1 > width:
            raise ValueError("crop window lies outside the phase canvas")

    selected_keys = tuple(
        (phase_name, crop_id)
        for phase_name in phases
        for crop_id in selected_by_phase[phase_name]
    )
    remainder = len(selected_keys) % batch_size
    padding_count = (batch_size - remainder) % batch_size
    if padding_count:
        final_real_keys = selected_keys[-remainder:]
        padding_keys = tuple(
            final_real_keys[index % len(final_real_keys)]
            for index in range(padding_count)
        )
    else:
        padding_keys = ()
    processed_keys = selected_keys + padding_keys

    score_sums = {
        phase_name: reference_inputs.new_zeros(
            (1, n_output_channels, height, width)
        )
        for phase_name in active_phases
    }
    count_mats = {
        phase_name: torch.zeros(
            (1, 1, height, width),
            dtype=torch.int16,
            device=reference_inputs.device,
        )
        for phase_name in active_phases
    }
    observed_by_phase = {phase_name: [] for phase_name in active_phases}
    observed_keys: list[tuple[str, int]] = []
    real_batch_sizes: list[int] = []
    model_batch_sizes: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(processed_keys), batch_size):
            model_keys = processed_keys[offset : offset + batch_size]
            real_count = min(batch_size, len(selected_keys) - offset)
            if len(model_keys) != batch_size or real_count <= 0:
                raise AssertionError("fixed-shape phase batch plan is invalid")
            optical_crops = []
            sar_crops = []
            for phase_name, crop_id in model_keys:
                inputs, dsm = phase_tensors[phase_name]
                y0, y1, x0, x1 = concrete_windows[crop_id]
                optical_crops.append(inputs[:, :, y0:y1, x0:x1])
                sar_crops.append(dsm[:, :, y0:y1, x0:x1])
            optical_batch = torch.cat(optical_crops, dim=0)
            sar_batch = torch.cat(sar_crops, dim=0)
            decoded = _decode_model_output(
                model(optical_batch, sar_batch), decoder_head_type
            )
            if decoded.ndim != 4 or decoded.shape[0] != batch_size:
                raise ValueError("model phase-crop output has an invalid batch shape")
            if decoded.shape[1] != n_output_channels:
                raise ValueError("model phase-crop output has an unexpected channel count")
            if not torch.isfinite(decoded).all():
                raise ValueError("model phase-crop output contains non-finite values")
            decoded = decoded.to(device=reference_inputs.device, dtype=reference_inputs.dtype)
            for batch_index, (phase_name, crop_id) in enumerate(
                model_keys[:real_count]
            ):
                y0, y1, x0, x1 = concrete_windows[crop_id]
                if tuple(decoded.shape[-2:]) != (y1 - y0, x1 - x0):
                    raise ValueError("model phase-crop output differs from its window shape")
                score_sums[phase_name][:, :, y0:y1, x0:x1] += decoded[batch_index]
                count_mats[phase_name][:, :, y0:y1, x0:x1] += 1
                observed_by_phase[phase_name].append(crop_id)
                observed_keys.append((phase_name, crop_id))
            real_batch_sizes.append(real_count)
            model_batch_sizes.append(len(model_keys))
    if reference_inputs.is_cuda:
        torch.cuda.synchronize(reference_inputs.device)
    elapsed = time.perf_counter() - started
    if tuple(observed_keys) != selected_keys:
        raise AssertionError("actual real phase-crop order differs from the plan")
    if any(size != batch_size for size in model_batch_sizes):
        raise AssertionError("a model call did not preserve the frozen batch shape")

    phase_results: dict[str, dict[str, Any]] = {}
    for phase_name in active_phases:
        observed_ids = tuple(observed_by_phase[phase_name])
        if observed_ids != selected_by_phase[phase_name]:
            raise AssertionError("actual per-phase crop order differs from the plan")
        phase_results[phase_name] = {
            "sum_logits": np.ascontiguousarray(
                score_sums[phase_name][0].cpu().numpy()
            ),
            "count_mat": np.ascontiguousarray(
                count_mats[phase_name][0, 0].cpu().numpy()
            ),
            "observed_crop_ids": observed_ids,
            "crop_samples": len(observed_ids),
        }

    return {
        "phases": phase_results,
        "observed_phase_crop_keys": tuple(observed_keys),
        "processed_phase_crop_keys": processed_keys,
        "padding_phase_crop_keys": padding_keys,
        "selected_crop_samples": len(selected_keys),
        "model_forward_crop_samples": len(processed_keys),
        "padding_crop_samples": len(padding_keys),
        "batch_calls": len(model_batch_sizes),
        "real_batch_sizes": tuple(real_batch_sizes),
        "model_batch_sizes": tuple(model_batch_sizes),
        "mean_real_batch_size": float(np.mean(real_batch_sizes)),
        "mean_model_batch_size": float(np.mean(model_batch_sizes)),
        "wall_seconds": float(elapsed),
        "padding_policy": (
            "cyclic-duplicate-from-final-real-batch; padding outputs discarded"
        ),
    }


def normalized_phase_logits(execution: dict[str, Any]) -> np.ndarray:
    score_sum = np.asarray(execution["sum_logits"])
    count = np.asarray(execution["count_mat"])
    if score_sum.dtype != np.float32 or score_sum.ndim != 3:
        raise TypeError("sum_logits must be float32 CHW")
    if count.ndim != 2 or count.shape != score_sum.shape[1:]:
        raise ValueError("count_mat shape differs from sum_logits")
    result = np.zeros(score_sum.shape, dtype=np.float32)
    np.divide(
        score_sum,
        count[None],
        out=result,
        where=count[None] > 0,
        casting="unsafe",
    )
    return result
