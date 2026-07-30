"""Mathematical helpers for the matched WHU phase-residual probe.

The functions in this module are deliberately independent of the released
MM-DINO model and of the V1 distillation runner.  They implement only the
sealed tensor geometry and the deterministic fixed-batch decision metrics for
the residual probe:

* both teacher and E0 logits must come from the same full-image sliding
  operator;
* logits and predicted corrections are centered per pixel across classes;
* the oracle fix mask is the teacher-correct/E0-wrong subset of valid labels;
* every other valid pixel belongs to the keep mask;
* one global fix-target RMS fixes the SmoothL1 unit without a hyperparameter
  sweep; and
* behavior is measured after matched full-slide aggregation.

Nothing here performs training, caching, or model mutation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


NUM_CLASSES = 7


def _require_bchw(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4:
        raise ValueError(f"{name} must have BCHW shape")
    if not value.is_floating_point():
        raise TypeError(f"{name} must contain floating-point logits")


def _require_matching_logits(
    first_name: str,
    first: torch.Tensor,
    second_name: str,
    second: torch.Tensor,
) -> None:
    _require_bchw(first_name, first)
    _require_bchw(second_name, second)
    if first.shape != second.shape:
        raise ValueError(
            f"{first_name} and {second_name} shapes differ: "
            f"{tuple(first.shape)} != {tuple(second.shape)}"
        )
    if first.device != second.device:
        raise ValueError(f"{first_name} and {second_name} must be on the same device")


def _require_mask(name: str, mask: torch.Tensor, logits: torch.Tensor) -> None:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have bool dtype")
    expected = (logits.shape[0], logits.shape[2], logits.shape[3])
    if tuple(mask.shape) != expected:
        raise ValueError(f"{name} shape differs from logits: {tuple(mask.shape)} != {expected}")
    if mask.device != logits.device:
        raise ValueError(f"{name} and logits must be on the same device")


def center_class_logits(logits: torch.Tensor) -> torch.Tensor:
    """Remove the per-pixel common-logit degree of freedom.

    ``logits`` must be BCHW.  The returned tensor has zero class mean at every
    BHW position.  Adding an arbitrary B1HW shift to the input therefore leaves
    the result unchanged, up to floating-point roundoff.
    """

    _require_bchw("logits", logits)
    return logits - logits.mean(dim=1, keepdim=True)


def full_resolution_centered_delta(
    low_resolution_delta: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly resize a residual to ``reference_logits`` and center it.

    Centering is performed after interpolation.  Both operations are linear,
    so this is mathematically equivalent to centering before interpolation,
    while making the returned full-resolution invariant explicit.
    """

    _require_bchw("low_resolution_delta", low_resolution_delta)
    _require_bchw("reference_logits", reference_logits)
    if low_resolution_delta.shape[:2] != reference_logits.shape[:2]:
        raise ValueError("delta and reference batch/class dimensions differ")
    delta = low_resolution_delta
    if delta.shape[-2:] != reference_logits.shape[-2:]:
        delta = F.interpolate(
            delta,
            size=reference_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return center_class_logits(delta)


def matched_phase_residual_target(
    teacher_slide_logits: torch.Tensor,
    e0_slide_logits: torch.Tensor,
) -> torch.Tensor:
    """Return the centered four-phase residual under matched slide geometry.

    Exact addition of this target to ``e0_slide_logits`` reproduces the teacher
    softmax and argmax, because the two results differ only by a per-pixel
    common shift.  Inputs are detached and converted to float32 so a cached
    teacher can never accidentally become part of an autograd graph.
    """

    _require_matching_logits(
        "teacher_slide_logits",
        teacher_slide_logits,
        "e0_slide_logits",
        e0_slide_logits,
    )
    teacher = teacher_slide_logits.detach().to(dtype=torch.float32)
    e0 = e0_slide_logits.detach().to(dtype=torch.float32)
    if not torch.isfinite(teacher).all() or not torch.isfinite(e0).all():
        raise ValueError("matched slide logits must be finite")
    return center_class_logits(teacher - e0).detach()


def matched_residual_masks(
    teacher_slide_logits: torch.Tensor,
    e0_slide_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return disjoint ``(M_fix, M_keep)`` masks for matched slide logits.

    ``M_fix`` contains valid pixels where the four-phase teacher is correct and
    normal-phase E0 is wrong.  ``M_keep`` is *every other valid pixel*, not only
    E0-correct pixels.  Thus the two masks partition the valid region and leave
    no teacher/E0-both-wrong blind spot.
    """

    _require_matching_logits(
        "teacher_slide_logits",
        teacher_slide_logits,
        "e0_slide_logits",
        e0_slide_logits,
    )
    expected_labels = (
        teacher_slide_logits.shape[0],
        teacher_slide_logits.shape[2],
        teacher_slide_logits.shape[3],
    )
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if labels.is_floating_point() or labels.dtype == torch.bool:
        raise TypeError("labels must contain integer class indices")
    if tuple(labels.shape) != expected_labels:
        raise ValueError(
            f"labels shape differs from logits: {tuple(labels.shape)} != {expected_labels}"
        )
    if labels.device != teacher_slide_logits.device:
        raise ValueError("labels and logits must be on the same device")
    num_classes = teacher_slide_logits.shape[1]
    valid = labels.ge(0) & labels.lt(num_classes)
    if valid_mask is not None:
        _require_mask("valid_mask", valid_mask, teacher_slide_logits)
        valid &= valid_mask
    teacher_correct = teacher_slide_logits.argmax(dim=1).eq(labels) & valid
    e0_correct = e0_slide_logits.argmax(dim=1).eq(labels) & valid
    fix = teacher_correct & ~e0_correct
    keep = valid & ~fix
    return fix.detach(), keep.detach()


def _selected_class_values(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    _require_bchw("values", values)
    _require_mask("mask", mask, values)
    if not bool(mask.any()):
        raise ValueError("masked per-pixel normalization requires a non-empty mask")
    return values.permute(0, 2, 3, 1)[mask]


def fix_mask_rms_scale(
    centered_target: torch.Tensor,
    fix_mask: torch.Tensor,
) -> float:
    """Return one deterministic global RMS over ``M_fix x classes``.

    The target is centered again defensively.  Accumulation uses float64, while
    the returned Python float is suitable for sealing in a JSON protocol.  A
    zero or non-finite scale is an invalid probe rather than something to clamp
    silently.
    """

    target = center_class_logits(centered_target.detach().to(dtype=torch.float32))
    selected = _selected_class_values(target, fix_mask).to(dtype=torch.float64)
    scale = torch.sqrt(selected.square().mean())
    value = float(scale.item())
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"fix-target RMS must be finite and positive, got {value}")
    return value


def masked_scaled_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale: float,
    beta: float = 1.0,
) -> torch.Tensor:
    """SmoothL1 mean over selected pixels/classes in deterministic RMS units."""

    _require_matching_logits("prediction", prediction, "target", target)
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("scale must be finite and positive")
    if not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be finite and positive")
    predicted_values = _selected_class_values(prediction, mask)
    target_values = _selected_class_values(target.detach(), mask)
    return F.smooth_l1_loss(
        predicted_values / float(scale),
        target_values / float(scale),
        beta=float(beta),
        reduction="mean",
    )


def scaled_phase_residual_losses(
    predicted_delta: torch.Tensor,
    matched_target: torch.Tensor,
    fix_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    scale: float,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return independently normalized fix, keep, and summed residual losses."""

    _require_matching_logits(
        "predicted_delta", predicted_delta, "matched_target", matched_target
    )
    _require_mask("fix_mask", fix_mask, predicted_delta)
    _require_mask("keep_mask", keep_mask, predicted_delta)
    if bool((fix_mask & keep_mask).any()):
        raise ValueError("fix_mask and keep_mask must be disjoint")
    prediction = center_class_logits(predicted_delta)
    target = center_class_logits(matched_target.detach())
    fix = masked_scaled_smooth_l1(
        prediction,
        target,
        fix_mask,
        scale=scale,
        beta=beta,
    )
    keep = masked_scaled_smooth_l1(
        prediction,
        torch.zeros_like(prediction),
        keep_mask,
        scale=scale,
        beta=beta,
    )
    return {"fix": fix, "keep": keep, "total": fix + keep}


def semantic_confusion_matrix(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int = NUM_CLASSES,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a target-row/prediction-column integer confusion matrix."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not isinstance(prediction, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("prediction and labels must be torch.Tensor objects")
    if prediction.shape != labels.shape:
        raise ValueError("prediction and labels shapes differ")
    if prediction.ndim != 3:
        raise ValueError("prediction and labels must have BHW shape")
    if prediction.device != labels.device:
        raise ValueError("prediction and labels must be on the same device")
    if prediction.is_floating_point() or prediction.dtype == torch.bool:
        raise TypeError("prediction must contain integer class indices")
    if labels.is_floating_point() or labels.dtype == torch.bool:
        raise TypeError("labels must contain integer class indices")
    valid = labels.ge(0) & labels.lt(num_classes)
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or valid_mask.shape != labels.shape:
            raise ValueError("valid_mask must be a bool tensor matching labels")
        if valid_mask.device != labels.device:
            raise ValueError("valid_mask and labels must be on the same device")
        valid &= valid_mask
    invalid_predictions = valid & (prediction.lt(0) | prediction.ge(num_classes))
    if bool(invalid_predictions.any()):
        raise ValueError("prediction contains an out-of-range value on a valid pixel")
    encoded = (
        labels[valid].to(torch.int64) * num_classes
        + prediction[valid].to(torch.int64)
    )
    return torch.bincount(encoded, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def class_iou_from_confusion(confusion: torch.Tensor) -> torch.Tensor:
    """Return per-class IoU, using NaN only for zero-union classes."""

    if not isinstance(confusion, torch.Tensor) or confusion.ndim != 2:
        raise ValueError("confusion must be a square tensor")
    if confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square tensor")
    values = confusion.to(torch.float64)
    diagonal = values.diagonal()
    union = values.sum(dim=1) + values.sum(dim=0) - diagonal
    result = torch.full_like(diagonal, float("nan"))
    active = union.gt(0)
    result[active] = diagonal[active] / union[active]
    return result


def mean_iou_from_confusion(confusion: torch.Tensor) -> float:
    """Return the repository-standard mIoU over classes with non-zero union."""

    iou = class_iou_from_confusion(confusion)
    union_present = torch.isfinite(iou)
    if not bool(union_present.any()):
        raise ValueError("mIoU requires at least one valid ground-truth class")
    return float(iou[union_present].mean().item())


def _optional_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _miou_payload(confusion: torch.Tensor) -> dict[str, Any]:
    per_class = class_iou_from_confusion(confusion)
    target_present = confusion.sum(dim=1).gt(0)
    union_present = torch.isfinite(per_class)
    return {
        "confusion": confusion.detach().cpu().tolist(),
        "per_class_iou": [
            float(value) if bool(torch.isfinite(value)) else None
            for value in per_class.detach().cpu()
        ],
        "target_present": target_present.detach().cpu().tolist(),
        "target_present_class_count": int(target_present.sum().item()),
        "union_present": union_present.detach().cpu().tolist(),
        "union_present_class_count": int(union_present.sum().item()),
        "miou": mean_iou_from_confusion(confusion),
        "miou_percent": mean_iou_from_confusion(confusion) * 100.0,
        "class_inclusion_policy": "target-or-prediction union > 0",
    }


def phase_residual_behavior_statistics(
    e0_slide_logits: torch.Tensor,
    teacher_slide_logits: torch.Tensor,
    corrected_slide_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Measure matched full-slide repair, damage, net pixels, and 7-class mIoU."""

    _require_matching_logits(
        "e0_slide_logits", e0_slide_logits, "teacher_slide_logits", teacher_slide_logits
    )
    _require_matching_logits(
        "e0_slide_logits", e0_slide_logits, "corrected_slide_logits", corrected_slide_logits
    )
    if e0_slide_logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"the WHU probe requires exactly {NUM_CLASSES} classes")
    if not all(
        bool(torch.isfinite(logits).all())
        for logits in (e0_slide_logits, teacher_slide_logits, corrected_slide_logits)
    ):
        raise ValueError("behavior logits must be finite")
    fix_mask, keep_mask = matched_residual_masks(
        teacher_slide_logits,
        e0_slide_logits,
        labels,
        valid_mask=valid_mask,
    )
    valid = fix_mask | keep_mask
    e0_prediction = e0_slide_logits.argmax(dim=1)
    teacher_prediction = teacher_slide_logits.argmax(dim=1)
    corrected_prediction = corrected_slide_logits.argmax(dim=1)
    e0_correct = valid & e0_prediction.eq(labels)
    corrected_correct = valid & corrected_prediction.eq(labels)
    routed_fixed_mask = fix_mask & corrected_correct
    global_fixed_mask = valid & ~e0_correct & corrected_correct
    broken_mask = e0_correct & ~corrected_correct
    wrong_to_wrong_changed = (
        valid
        & ~e0_correct
        & ~corrected_correct
        & corrected_prediction.ne(e0_prediction)
    )

    fix_pixels = int(fix_mask.sum().item())
    keep_pixels = int(keep_mask.sum().item())
    routed_fixed = int(routed_fixed_mask.sum().item())
    global_fixed = int(global_fixed_mask.sum().item())
    broken = int(broken_mask.sum().item())
    routed_net = routed_fixed - broken
    global_net = global_fixed - broken
    correct_count_delta = int(corrected_correct.sum().item() - e0_correct.sum().item())
    if global_net != correct_count_delta:
        raise AssertionError("global fix/break accounting does not match correct-count delta")

    e0_confusion = semantic_confusion_matrix(
        e0_prediction, labels, num_classes=NUM_CLASSES, valid_mask=valid
    )
    teacher_confusion = semantic_confusion_matrix(
        teacher_prediction, labels, num_classes=NUM_CLASSES, valid_mask=valid
    )
    corrected_confusion = semantic_confusion_matrix(
        corrected_prediction, labels, num_classes=NUM_CLASSES, valid_mask=valid
    )
    e0_metrics = _miou_payload(e0_confusion)
    teacher_metrics = _miou_payload(teacher_confusion)
    corrected_metrics = _miou_payload(corrected_confusion)

    per_class: list[dict[str, Any]] = []
    for class_index in range(NUM_CLASSES):
        class_pixels = valid & labels.eq(class_index)
        class_fix = fix_mask & class_pixels
        class_e0_correct = e0_correct & class_pixels
        class_routed_fixed = routed_fixed_mask & class_pixels
        class_broken = broken_mask & class_pixels
        class_fix_count = int(class_fix.sum().item())
        class_e0_correct_count = int(class_e0_correct.sum().item())
        e0_iou = e0_metrics["per_class_iou"][class_index]
        corrected_iou = corrected_metrics["per_class_iou"][class_index]
        per_class.append(
            {
                "class_index": class_index,
                "valid_pixels": int(class_pixels.sum().item()),
                "fix_pixels": class_fix_count,
                "routed_fixed_pixels": int(class_routed_fixed.sum().item()),
                "routed_fix_rate": _optional_ratio(
                    int(class_routed_fixed.sum().item()), class_fix_count
                ),
                "e0_correct_pixels": class_e0_correct_count,
                "broken_pixels": int(class_broken.sum().item()),
                "broken_rate_within_e0_correct": _optional_ratio(
                    int(class_broken.sum().item()), class_e0_correct_count
                ),
                "e0_iou": e0_iou,
                "corrected_iou": corrected_iou,
                "corrected_minus_e0_iou_pp": (
                    None
                    if e0_iou is None or corrected_iou is None
                    else float((corrected_iou - e0_iou) * 100.0)
                ),
            }
        )

    return {
        "num_classes": NUM_CLASSES,
        "valid_pixels": int(valid.sum().item()),
        "fix_pixels": fix_pixels,
        "keep_pixels": keep_pixels,
        "routed_fixed_pixels": routed_fixed,
        "routed_fix_rate": _optional_ratio(routed_fixed, fix_pixels),
        "global_fixed_pixels": global_fixed,
        "broken_pixels": broken,
        "broken_rate_within_e0_correct": _optional_ratio(
            broken, int(e0_correct.sum().item())
        ),
        "broken_per_routed_fixed": _optional_ratio(broken, routed_fixed),
        "routed_net_pixels": routed_net,
        "global_net_correct_pixels": global_net,
        "correct_pixel_count_delta": correct_count_delta,
        "prediction_changed_on_keep_pixels": int(
            (keep_mask & corrected_prediction.ne(e0_prediction)).sum().item()
        ),
        "wrong_to_wrong_changed_pixels": int(wrong_to_wrong_changed.sum().item()),
        "e0": e0_metrics,
        "teacher": teacher_metrics,
        "corrected": corrected_metrics,
        "corrected_minus_e0_miou_pp": float(
            corrected_metrics["miou_percent"] - e0_metrics["miou_percent"]
        ),
        "per_class": per_class,
    }


@dataclass(frozen=True)
class PhaseResidualGateThresholds:
    """Pre-registered same-image full-slide gate for the residual probe."""

    minimum_routed_fix_rate: float = 0.50
    maximum_broken_per_routed_fixed: float = 0.25
    minimum_miou_delta_pp: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_routed_fix_rate <= 1.0:
            raise ValueError("minimum_routed_fix_rate must be within [0, 1]")
        if self.maximum_broken_per_routed_fixed < 0.0 or not math.isfinite(
            self.maximum_broken_per_routed_fixed
        ):
            raise ValueError("maximum_broken_per_routed_fixed must be finite and non-negative")
        if not math.isfinite(self.minimum_miou_delta_pp):
            raise ValueError("minimum_miou_delta_pp must be finite")


def phase_residual_probe_gate(
    statistics: dict[str, Any],
    thresholds: PhaseResidualGateThresholds = PhaseResidualGateThresholds(),
) -> dict[str, Any]:
    """Apply the pre-registered count-based repair/damage/mIoU resource gate."""

    required = {
        "fix_pixels",
        "routed_fixed_pixels",
        "broken_pixels",
        "routed_net_pixels",
        "corrected_minus_e0_miou_pp",
    }
    missing = sorted(required - set(statistics))
    if missing:
        raise KeyError(f"behavior statistics lack required keys: {missing}")
    fix_pixels = int(statistics["fix_pixels"])
    routed_fixed = int(statistics["routed_fixed_pixels"])
    broken = int(statistics["broken_pixels"])
    routed_net = int(statistics["routed_net_pixels"])
    miou_delta = float(statistics["corrected_minus_e0_miou_pp"])
    if min(fix_pixels, routed_fixed, broken) < 0:
        raise ValueError("gate pixel counts must be non-negative")
    if routed_fixed > fix_pixels:
        raise ValueError("routed_fixed_pixels cannot exceed fix_pixels")
    if routed_net != routed_fixed - broken:
        raise ValueError("routed_net_pixels is inconsistent with fix/break counts")
    if not math.isfinite(miou_delta):
        raise ValueError("mIoU delta must be finite")

    support_pass = fix_pixels > 0
    fix_rate_pass = (
        support_pass
        and routed_fixed >= thresholds.minimum_routed_fix_rate * fix_pixels
    )
    damage_pass = (
        routed_fixed > 0
        and broken
        <= thresholds.maximum_broken_per_routed_fixed * routed_fixed
    )
    # Strictly greater preserves the wording "higher than E0" when the default
    # threshold is zero.
    miou_pass = miou_delta > thresholds.minimum_miou_delta_pp
    checks = {
        "nonempty_fix_support": support_pass,
        "minimum_routed_fix_rate": fix_rate_pass,
        "maximum_broken_per_routed_fixed": damage_pass,
        "miou_strictly_above_threshold": miou_pass,
    }
    passed = all(checks.values())
    return {
        "outcome": "PASS" if passed else "FAIL",
        "passes": passed,
        "checks": checks,
        "thresholds": asdict(thresholds),
        "actual": {
            "fix_pixels": fix_pixels,
            "routed_fixed_pixels": routed_fixed,
            "routed_fix_rate": _optional_ratio(routed_fixed, fix_pixels),
            "broken_pixels": broken,
            "broken_per_routed_fixed": _optional_ratio(broken, routed_fixed),
            "routed_net_pixels": routed_net,
            "corrected_minus_e0_miou_pp": miou_delta,
        },
        "scope": (
            "same-one-image exact matched-full-slide resource screen only; "
            "not a generalization or paper-level result"
        ),
    }


__all__ = [
    "NUM_CLASSES",
    "PhaseResidualGateThresholds",
    "center_class_logits",
    "class_iou_from_confusion",
    "fix_mask_rms_scale",
    "full_resolution_centered_delta",
    "masked_scaled_smooth_l1",
    "matched_phase_residual_target",
    "matched_residual_masks",
    "mean_iou_from_confusion",
    "phase_residual_behavior_statistics",
    "phase_residual_probe_gate",
    "scaled_phase_residual_losses",
    "semantic_confusion_matrix",
]
