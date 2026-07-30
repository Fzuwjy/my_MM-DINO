"""Common building blocks for the WHU spatial-phase distillation experiment.

This module deliberately does not alter MM-DINO's released forward path.  A
forward hook reads the final P2 feature produced by ``decoder.neck`` while the
entire E0 model remains frozen and in evaluation mode.  A small residual branch
can then learn a logit correction without changing the single-phase inference
contract.

The helpers here are independent of the training runner and dataset.  In
particular, phase-view geometry and the on-disk structural-mask cache belong to
their respective experiment modules.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


TensorSnapshot = dict[str, torch.Tensor]


def _group_count(channels: int, preferred_groups: int = 8) -> int:
    """Return the largest valid GroupNorm group count up to ``preferred_groups``."""

    if channels <= 0:
        raise ValueError("channels must be positive")
    for groups in range(min(channels, preferred_groups), 0, -1):
        if channels % groups == 0:
            return groups
    raise AssertionError("every positive channel count is divisible by one")


class FrozenE0P2Extractor(nn.Module):
    """Run a frozen E0 model and capture its final P2 decoder feature.

    The wrapped model must expose ``model.decoder.neck``.  The neck's first
    output is interpreted as final P2.  The base model is permanently frozen
    and forced back to ``eval`` both when :meth:`train` is called and immediately
    before every forward.  The forward uses ``torch.no_grad`` rather than
    ``torch.inference_mode`` so that the detached P2 tensor remains usable as an
    input whose consumer weights require gradients.

    A persistent hook avoids any change to the released decoder implementation.
    Call :meth:`remove_hook` when the extractor is no longer needed.
    """

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        if not isinstance(base_model, nn.Module):
            raise TypeError("base_model must be a torch.nn.Module")
        decoder = getattr(base_model, "decoder", None)
        neck = getattr(decoder, "neck", None)
        if not isinstance(neck, nn.Module):
            raise TypeError("base_model must expose decoder.neck as an nn.Module")

        self.base_model = base_model
        self.base_model.requires_grad_(False)
        self.base_model.eval()
        self._captured_p2: torch.Tensor | None = None
        self._hook_handle: Any | None = neck.register_forward_hook(self._capture_p2)

    def _capture_p2(
        self,
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        if self._captured_p2 is not None:
            raise RuntimeError("decoder.neck ran more than once in one E0 forward")
        if isinstance(output, torch.Tensor):
            p2 = output
        elif isinstance(output, (tuple, list)) and output:
            p2 = output[0]
        else:
            raise TypeError("decoder.neck must return a tensor or a non-empty sequence")
        if not isinstance(p2, torch.Tensor) or p2.ndim != 4:
            raise TypeError("decoder.neck P2 output must be a BCHW tensor")
        self._captured_p2 = p2.detach()

    @property
    def hook_is_active(self) -> bool:
        """Whether the P2 hook is still installed."""

        return self._hook_handle is not None

    def remove_hook(self) -> None:
        """Remove the persistent neck hook; safe to call more than once."""

        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._captured_p2 = None

    def train(self, mode: bool = True) -> "FrozenE0P2Extractor":
        """Set only this wrapper's flag while keeping the E0 model in eval mode."""

        self.training = bool(mode)
        self.base_model.eval()
        return self

    def forward(self, *modalities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(e0_logits, detached_p2)`` for one or more input modalities."""

        if not modalities:
            raise ValueError("at least one input modality is required")
        if not self.hook_is_active:
            raise RuntimeError("the P2 hook has been removed")

        self.base_model.eval()
        self._captured_p2 = None
        try:
            with torch.no_grad():
                logits = self.base_model(*modalities)
            p2 = self._captured_p2
        finally:
            self._captured_p2 = None

        if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
            raise TypeError("the E0 model must return BCHW logits")
        if p2 is None:
            raise RuntimeError("decoder.neck did not produce a P2 capture")
        if p2.shape[0] != logits.shape[0]:
            raise RuntimeError("captured P2 and E0 logits have different batch sizes")
        return logits.detach(), p2


class _DepthwiseSeparableResidualBlock(nn.Module):
    """One local residual block with depthwise and pointwise convolutions."""

    def __init__(self, channels: int, groups: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.depthwise_norm = nn.GroupNorm(groups, channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_norm = nn.GroupNorm(groups, channels)
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        values = self.activation(self.depthwise_norm(self.depthwise(inputs)))
        values = self.pointwise_norm(self.pointwise(values))
        return self.activation(residual + values)


class PhaseCorrectionBranch(nn.Module):
    """Low-capacity P2-to-logit residual branch with an exact-zero output head.

    The default architecture is the fixed first-screen design: a 256-to-64
    1x1 projection, two depthwise-separable 3x3 residual blocks with GroupNorm
    and GELU, followed by a 64-to-7 1x1 output projection.  The final weights
    and bias are explicitly zeroed, so every finite P2 input initially produces
    an exact all-zero correction.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        num_classes: int = 7,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or hidden_channels <= 0 or num_classes <= 0:
            raise ValueError("channel and class counts must be positive")
        groups = _group_count(hidden_channels)
        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_classes = int(num_classes)

        self.input_projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.residual_blocks = nn.Sequential(
            _DepthwiseSeparableResidualBlock(hidden_channels, groups),
            _DepthwiseSeparableResidualBlock(hidden_channels, groups),
        )
        self.output_projection = nn.Conv2d(
            hidden_channels, num_classes, kernel_size=1, bias=True
        )
        self.reset_output_projection()

    def reset_output_projection(self) -> None:
        """Restore the strict zero-residual initialization."""

        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        if p2.ndim != 4:
            raise ValueError("P2 must be a BCHW tensor")
        if p2.shape[1] != self.in_channels:
            raise ValueError(
                f"P2 has {p2.shape[1]} channels; expected {self.in_channels}"
            )
        values = self.input_projection(p2)
        values = self.residual_blocks(values)
        return self.output_projection(values)


class SinglePhaseCorrectedModel(nn.Module):
    """Single-pass E0 plus a learned P2 logit correction.

    ``base_model_or_extractor`` may be either the raw E0 model or an existing
    :class:`FrozenE0P2Extractor`.  The latter form allows several independently
    trained correction branches to share one frozen feature extractor.
    """

    def __init__(
        self,
        base_model_or_extractor: nn.Module,
        correction_branch: PhaseCorrectionBranch | None = None,
    ) -> None:
        super().__init__()
        if isinstance(base_model_or_extractor, FrozenE0P2Extractor):
            self.extractor = base_model_or_extractor
        else:
            self.extractor = FrozenE0P2Extractor(base_model_or_extractor)
        self.correction_branch = (
            correction_branch
            if correction_branch is not None
            else PhaseCorrectionBranch()
        )

    @property
    def base_model(self) -> nn.Module:
        """The immutable E0 model used by this corrected model."""

        return self.extractor.base_model

    def train(self, mode: bool = True) -> "SinglePhaseCorrectedModel":
        """Train/evaluate the branch without ever enabling E0 training mode."""

        super().train(mode)
        self.extractor.base_model.eval()
        return self

    def remove_hook(self) -> None:
        """Remove the underlying P2 hook."""

        self.extractor.remove_hook()

    def forward(self, *modalities: torch.Tensor) -> torch.Tensor:
        e0_logits, p2 = self.extractor(*modalities)
        correction = self.correction_branch(p2)
        if correction.shape[0] != e0_logits.shape[0]:
            raise RuntimeError("correction and E0 logits have different batch sizes")
        if correction.shape[1] != e0_logits.shape[1]:
            raise RuntimeError("correction and E0 logits have different class counts")
        if correction.shape[-2:] != e0_logits.shape[-2:]:
            correction = F.interpolate(
                correction,
                size=e0_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        correction = correction.to(dtype=e0_logits.dtype)
        return e0_logits + correction


def teacher_gain_mask(
    teacher_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float = 0.0,
    delta: float | None = None,
    valid_mask: torch.Tensor | None = None,
    ignore_index: int = 7,
) -> torch.Tensor:
    """Select pixels where the teacher better supports the ground-truth class.

    A valid pixel is selected when

    ``CE(teacher, label) + margin < CE(baseline, label)``.

    ``delta`` is a compatibility alias for ``margin``; specifying both with
    different values is rejected.  Labels equal to ``ignore_index`` are always
    excluded, and an optional ``valid_mask`` can impose an additional spatial
    constraint.  Any other label outside ``[0, num_classes)`` is treated as a
    data error.  The returned spatial mask has shape ``[B, H, W]`` and never
    participates in autograd.
    """

    if teacher_logits.ndim != 4 or baseline_logits.ndim != 4:
        raise ValueError("teacher and baseline logits must be BCHW tensors")
    if teacher_logits.shape != baseline_logits.shape:
        raise ValueError("teacher and baseline logits must have identical shapes")
    if labels.ndim != 3:
        raise ValueError("labels must have shape [B, H, W]")
    if labels.shape != (
        teacher_logits.shape[0],
        teacher_logits.shape[2],
        teacher_logits.shape[3],
    ):
        raise ValueError("labels do not match the logits batch/spatial shape")
    if teacher_logits.device != baseline_logits.device or labels.device != teacher_logits.device:
        raise ValueError("teacher logits, baseline logits, and labels must share a device")
    if delta is not None:
        if margin != 0.0 and float(delta) != float(margin):
            raise ValueError("margin and delta specify different gain thresholds")
        margin = float(delta)
    if margin < 0:
        raise ValueError("teacher-gain margin must be non-negative")
    if valid_mask is not None:
        if valid_mask.shape != labels.shape:
            raise ValueError("valid_mask must have the same shape as labels")
        if valid_mask.device != labels.device:
            raise ValueError("valid_mask and labels must share a device")

    num_classes = teacher_logits.shape[1]
    labels_long = labels.to(dtype=torch.long)
    valid = labels_long != int(ignore_index)
    invalid = valid & ((labels_long < 0) | (labels_long >= num_classes))
    if bool(invalid.any().item()):
        bad_values = torch.unique(labels_long[invalid]).detach().cpu().tolist()
        raise ValueError(f"labels contain out-of-range non-ignore values: {bad_values}")
    safe_labels = labels_long.masked_fill(~valid, 0).unsqueeze(1)

    with torch.no_grad():
        teacher_nll = -F.log_softmax(teacher_logits.detach(), dim=1).gather(
            1, safe_labels
        )
        baseline_nll = -F.log_softmax(baseline_logits.detach(), dim=1).gather(
            1, safe_labels
        )
        gain = teacher_nll.squeeze(1) + float(margin) < baseline_nll.squeeze(1)
        result = valid & gain
        if valid_mask is not None:
            result &= valid_mask.to(dtype=torch.bool)
        return result


def masked_teacher_student_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return masked full-resolution ``KL(teacher || student)``.

    The loss is summed over classes, averaged over selected pixels, and scaled
    by ``temperature**2``.  The teacher is detached.  An empty mask returns
    ``student_logits.sum() * 0`` so the result remains connected to the student
    graph and backward is valid.
    """

    if student_logits.ndim != 4 or teacher_logits.ndim != 4:
        raise ValueError("student and teacher logits must be BCHW tensors")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if mask.ndim != 3 or mask.shape != (
        student_logits.shape[0],
        student_logits.shape[2],
        student_logits.shape[3],
    ):
        raise ValueError("mask must have shape [B, H, W] matching the logits")
    if student_logits.device != teacher_logits.device or mask.device != student_logits.device:
        raise ValueError("student logits, teacher logits, and mask must share a device")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    selected = mask.to(dtype=torch.bool)
    selected_count = selected.sum()
    if int(selected_count.detach().item()) == 0:
        return student_logits.sum() * 0.0

    scaled_student = student_logits / float(temperature)
    scaled_teacher = teacher_logits.detach() / float(temperature)
    teacher_log_prob = F.log_softmax(scaled_teacher, dim=1)
    teacher_prob = teacher_log_prob.exp()
    student_log_prob = F.log_softmax(scaled_student, dim=1)
    per_pixel = (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=1)
    return (
        per_pixel.masked_select(selected).sum()
        / selected_count.to(dtype=per_pixel.dtype)
        * float(temperature) ** 2
    )


def masked_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compatibility name for :func:`masked_teacher_student_kl`."""

    return masked_teacher_student_kl(
        student_logits,
        teacher_logits,
        mask,
        temperature=temperature,
    )


def _snapshot(items: Any) -> TensorSnapshot:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in items
        if isinstance(tensor, torch.Tensor)
    }


def snapshot_named_parameters(module: nn.Module) -> TensorSnapshot:
    """Clone all named parameters to CPU for later mutation checks."""

    return _snapshot(module.named_parameters())


def snapshot_module_state(module: nn.Module) -> TensorSnapshot:
    """Clone the complete tensor state_dict to CPU."""

    return _snapshot(module.state_dict().items())


def snapshot_batchnorm_buffers(module: nn.Module) -> TensorSnapshot:
    """Clone running statistics and counters from every BatchNorm submodule."""

    values: list[tuple[str, torch.Tensor]] = []
    for module_name, child in module.named_modules():
        if not isinstance(child, nn.modules.batchnorm._BatchNorm):
            continue
        prefix = f"{module_name}." if module_name else ""
        for buffer_name in ("running_mean", "running_var", "num_batches_tracked"):
            buffer = getattr(child, buffer_name, None)
            if isinstance(buffer, torch.Tensor):
                values.append((prefix + buffer_name, buffer))
    return _snapshot(values)


def tensor_snapshot_sha256(snapshot: Mapping[str, torch.Tensor]) -> str:
    """Hash a named tensor snapshot deterministically, including metadata."""

    digest = hashlib.sha256()
    for name in sorted(snapshot):
        tensor = snapshot[name].detach().cpu().contiguous()
        metadata = f"{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode("utf-8")
        digest.update(len(metadata).to_bytes(8, byteorder="little", signed=False))
        digest.update(metadata)
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, byteorder="little", signed=False))
        digest.update(raw)
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    """SHA256 of all parameters and persistent buffers in ``module``."""

    return tensor_snapshot_sha256(snapshot_module_state(module))


def parameter_sha256(module: nn.Module) -> str:
    """SHA256 of named model parameters only."""

    return tensor_snapshot_sha256(snapshot_named_parameters(module))


def batchnorm_buffer_sha256(module: nn.Module) -> str:
    """SHA256 of BatchNorm running means, variances, and counters only."""

    return tensor_snapshot_sha256(snapshot_batchnorm_buffers(module))


def tensor_snapshots_equal(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> bool:
    """Return exact equality for two named tensor snapshots."""

    if set(first) != set(second):
        return False
    return all(torch.equal(first[name], second[name]) for name in first)
