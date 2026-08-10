"""Flow-alignment primitives for opt-in MM-DINO experiments.

The module is an independent implementation of the one-way coarse-to-fine
Flow Alignment Module described by Li et al., "Semantic Flow for Fast and
Accurate Scene Parsing" (ECCV 2020).  It deliberately keeps the released
MM-DINO nearest-neighbour top-down path as the baseline and adds only the
learned displacement residual relative to a zero-flow sampling reference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def target_pixel_flow_grid(flow: torch.Tensor) -> torch.Tensor:
    """Convert target-grid pixel offsets to an ``align_corners=False`` grid.

    ``flow[:, 0]`` is the source-sampling x offset and ``flow[:, 1]`` is the
    source-sampling y offset, both measured in pixels of the target grid.  A
    positive x offset therefore samples farther right and moves visible
    content to the left in the output; the impulse tests freeze this contract.
    """

    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [batch, 2, height, width]")
    batch, _, height, width = flow.shape
    if batch <= 0 or height <= 0 or width <= 0:
        raise ValueError("flow dimensions must be positive")
    if not flow.is_floating_point():
        raise TypeError("flow must be floating point")

    theta = flow.new_zeros((batch, 2, 3))
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    base_grid = F.affine_grid(
        theta,
        size=(batch, 1, height, width),
        align_corners=False,
    )
    pixel_to_normalized = flow.new_tensor(
        (2.0 / float(width), 2.0 / float(height))
    ).view(1, 1, 1, 2)
    return base_grid + flow.permute(0, 2, 3, 1) * pixel_to_normalized


def flow_warp(
    source: torch.Tensor,
    flow: torch.Tensor,
    *,
    padding_mode: str = "border",
) -> torch.Tensor:
    """Warp ``source`` to the target grid represented by ``flow``."""

    if source.ndim != 4:
        raise ValueError("source must have shape [batch, channels, height, width]")
    if source.shape[0] != flow.shape[0]:
        raise ValueError("source and flow batch sizes differ")
    if source.device != flow.device:
        raise ValueError("source and flow must be on the same device")
    if source.dtype != flow.dtype:
        raise ValueError("source and flow must have the same dtype")
    if padding_mode not in {"zeros", "border", "reflection"}:
        raise ValueError(f"unsupported padding mode: {padding_mode!r}")

    return F.grid_sample(
        source,
        target_pixel_flow_grid(flow),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )


class ResidualFlowAlignment(nn.Module):
    """One-way coarse-to-fine FAM with an exact nearest-path initialization.

    The learned branch returns

    ``nearest(high) + warp(high, predicted_flow) - warp(high, zero_flow)``.

    With a zero-initialized flow predictor, the two warp terms are identical,
    so enabling the module does not silently replace MM-DINO's released
    nearest interpolation with bilinear interpolation.  Unlike a zero-valued
    scalar gate, the flow predictor receives a gradient on the first backward.
    """

    def __init__(
        self,
        channels: int,
        flow_channels: Optional[int] = None,
        *,
        kernel_size: int = 3,
        padding_mode: str = "border",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if flow_channels is None:
            flow_channels = max(channels // 2, 1)
        if flow_channels <= 0:
            raise ValueError("flow_channels must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError(f"unsupported padding mode: {padding_mode!r}")

        self.channels = int(channels)
        self.flow_channels = int(flow_channels)
        self.kernel_size = int(kernel_size)
        self.padding_mode = padding_mode

        self.high_projection = nn.Conv2d(
            self.channels, self.flow_channels, kernel_size=1, bias=False
        )
        self.low_projection = nn.Conv2d(
            self.channels, self.flow_channels, kernel_size=1, bias=False
        )
        self.flow_predictor = nn.Conv2d(
            self.flow_channels * 2,
            2,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            bias=False,
        )
        nn.init.zeros_(self.flow_predictor.weight)

    def _validate_inputs(
        self,
        high_feature: torch.Tensor,
        low_feature: torch.Tensor,
        baseline_high: Optional[torch.Tensor],
    ) -> None:
        if high_feature.ndim != 4 or low_feature.ndim != 4:
            raise ValueError("alignment inputs must be four-dimensional")
        if high_feature.shape[0] != low_feature.shape[0]:
            raise ValueError("alignment input batch sizes differ")
        if high_feature.shape[1] != self.channels:
            raise ValueError("unexpected high-feature channel count")
        if low_feature.shape[1] != self.channels:
            raise ValueError("unexpected low-feature channel count")
        if high_feature.device != low_feature.device:
            raise ValueError("alignment inputs must be on the same device")
        if high_feature.dtype != low_feature.dtype:
            raise ValueError("alignment inputs must have the same dtype")
        if baseline_high is not None:
            expected = (
                high_feature.shape[0],
                self.channels,
                low_feature.shape[-2],
                low_feature.shape[-1],
            )
            if tuple(baseline_high.shape) != expected:
                raise ValueError(
                    "baseline high feature has the wrong shape: "
                    f"{tuple(baseline_high.shape)} != {expected}"
                )
            if (
                baseline_high.device != high_feature.device
                or baseline_high.dtype != high_feature.dtype
            ):
                raise ValueError(
                    "baseline high feature must match the input device and dtype"
                )

    def zero_flow_reference(
        self, high_feature: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        zero_flow = high_feature.new_zeros(
            (high_feature.shape[0], 2, output_size[0], output_size[1])
        )
        return flow_warp(
            high_feature,
            zero_flow,
            padding_mode=self.padding_mode,
        )

    def predict_flow(
        self, reference_high: torch.Tensor, low_feature: torch.Tensor
    ) -> torch.Tensor:
        if reference_high.shape[-2:] != low_feature.shape[-2:]:
            raise ValueError("flow-prediction inputs must share a spatial grid")
        return self.flow_predictor(
            torch.cat(
                [
                    self.high_projection(reference_high),
                    self.low_projection(low_feature),
                ],
                dim=1,
            )
        )

    def forward(
        self,
        high_feature: torch.Tensor,
        low_feature: torch.Tensor,
        *,
        baseline_high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_inputs(high_feature, low_feature, baseline_high)
        output_size = tuple(low_feature.shape[-2:])
        if baseline_high is None:
            baseline_high = F.interpolate(
                high_feature,
                size=output_size,
                mode="nearest",
            )

        reference_high = self.zero_flow_reference(high_feature, output_size)
        flow = self.predict_flow(reference_high, low_feature)
        warped_high = flow_warp(
            high_feature,
            flow,
            padding_mode=self.padding_mode,
        )
        return baseline_high + (warped_high - reference_high)
