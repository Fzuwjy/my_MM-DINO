"""Pure helpers for the EarthMiss FRM-P5 prototype-transfer experiment."""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class PrototypeBatchStatistics:
    support_mask: torch.Tensor
    raw_support: torch.Tensor
    purity_mass: torch.Tensor
    effective_sample_size: torch.Tensor
    positive_cosine: torch.Tensor
    off_diagonal_cosine_mean: torch.Tensor


@dataclass(frozen=True)
class BatchClassPrototypes:
    prototypes: torch.Tensor
    support_mask: torch.Tensor
    raw_support: torch.Tensor
    purity_mass: torch.Tensor
    effective_sample_size: torch.Tensor


def exact_area_occupancy(
    target: torch.Tensor,
    output_size: tuple[int, int],
    *,
    num_classes: int = 8,
    ignore_index: int = 8,
) -> torch.Tensor:
    """Map native GT to exact per-class cell occupancy.

    The denominator is the geometric native-pixel area of each output cell.
    Ignore pixels contribute zero to every class and are not renormalized away.
    Exact integer divisibility is required so no interpolation convention can
    silently change the class support definition.
    """

    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError("target must have shape [B,H,W] or [B,1,H,W]")
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    if len(output_size) != 2 or min(output_size) <= 0:
        raise ValueError("output_size must contain two positive integers")

    height, width = target.shape[-2:]
    out_h, out_w = (int(output_size[0]), int(output_size[1]))
    if height % out_h or width % out_w:
        raise ValueError(
            "native target shape must be exactly divisible by the feature grid"
        )
    invalid = (target < 0) | (target >= num_classes)
    invalid &= target != ignore_index
    if invalid.any():
        bad = torch.unique(target[invalid]).detach().cpu().tolist()
        raise ValueError(f"target contains invalid class ids: {bad}")

    valid = (target >= 0) & (target < num_classes)
    safe_target = target.clamp(min=0, max=num_classes - 1)
    one_hot = F.one_hot(safe_target.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).to(dtype=torch.float32)
    one_hot = one_hot * valid.unsqueeze(1).to(dtype=one_hot.dtype)
    kernel_h, kernel_w = height // out_h, width // out_w
    return F.avg_pool2d(
        one_hot,
        kernel_size=(kernel_h, kernel_w),
        stride=(kernel_h, kernel_w),
    )


def _pool_class_prototypes(
    feature: torch.Tensor,
    weights: torch.Tensor,
    purity_mass: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    pooled = torch.einsum("bkhw,bchw->kc", weights, feature.float())
    pooled = pooled / purity_mass.clamp_min(eps).unsqueeze(1)
    return F.normalize(pooled, dim=1, eps=eps)


def batch_class_prototypes(
    feature: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int = 8,
    ignore_index: int = 8,
    minimum_raw_support: float = 2.0,
    eps: float = 1e-6,
) -> BatchClassPrototypes:
    """Construct the frozen batch-level soft-purity class prototypes."""

    if feature.ndim != 4:
        raise ValueError("feature must have shape [B,C,H,W]")
    if feature.shape[0] != target.shape[0]:
        raise ValueError("feature and target batch sizes must match")
    if minimum_raw_support <= 0:
        raise ValueError("minimum_raw_support must be positive")
    occupancy = exact_area_occupancy(
        target,
        feature.shape[-2:],
        num_classes=num_classes,
        ignore_index=ignore_index,
    ).to(device=feature.device)
    weights = occupancy.square()
    raw_support = occupancy.sum(dim=(0, 2, 3))
    purity_mass = weights.sum(dim=(0, 2, 3))
    weight_square_mass = weights.square().sum(dim=(0, 2, 3))
    effective_sample_size = purity_mass.square() / weight_square_mass.clamp_min(eps)
    support_mask = (raw_support >= minimum_raw_support) & (purity_mass > eps)
    prototypes = _pool_class_prototypes(
        feature,
        weights,
        purity_mass,
        eps=eps,
    )
    return BatchClassPrototypes(
        prototypes=prototypes,
        support_mask=support_mask,
        raw_support=raw_support,
        purity_mass=purity_mass,
        effective_sample_size=effective_sample_size,
    )


def supported_prototype_separation(batch: BatchClassPrototypes) -> torch.Tensor:
    """Mean off-diagonal cosine distance for one batch's supported classes."""

    supported = batch.prototypes[batch.support_mask]
    if supported.shape[0] < 2:
        return supported.new_tensor(float("nan"))
    cosine = supported @ supported.transpose(0, 1)
    off_diagonal = ~torch.eye(
        cosine.shape[0],
        dtype=torch.bool,
        device=cosine.device,
    )
    return (1.0 - cosine[off_diagonal]).mean()


def prototype_transfer_infonce(
    query_feature: torch.Tensor,
    anchor_feature: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int = 8,
    ignore_index: int = 8,
    minimum_raw_support: float = 2.0,
    temperature: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, PrototypeBatchStatistics]:
    """Class-balanced Full-state/SAR-state batch Prototype InfoNCE.

    Occupancy ``a`` is pooled over the complete batch.  Prototypes use the
    frozen soft-purity rule ``w=a**2``, while the two-cell eligibility rule uses
    unsquared native area ``sum(a)`` so thin classes are not excluded twice.
    The anchor is always detached inside this function.
    """

    if query_feature.ndim != 4 or anchor_feature.ndim != 4:
        raise ValueError("query and anchor features must have shape [B,C,H,W]")
    if query_feature.shape != anchor_feature.shape:
        raise ValueError("query and anchor feature shapes must match")
    if query_feature.shape[0] != target.shape[0]:
        raise ValueError("feature and target batch sizes must match")
    if minimum_raw_support <= 0:
        raise ValueError("minimum_raw_support must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    query_batch = batch_class_prototypes(
        query_feature,
        target,
        num_classes=num_classes,
        ignore_index=ignore_index,
        minimum_raw_support=minimum_raw_support,
        eps=eps,
    )
    anchor_batch = batch_class_prototypes(
        anchor_feature.detach(),
        target,
        num_classes=num_classes,
        ignore_index=ignore_index,
        minimum_raw_support=minimum_raw_support,
        eps=eps,
    )
    support_mask = query_batch.support_mask & anchor_batch.support_mask
    query_prototypes = query_batch.prototypes
    anchor_prototypes = anchor_batch.prototypes
    supported_query = query_prototypes[support_mask]
    supported_anchor = anchor_prototypes[support_mask]

    if supported_query.shape[0] < 2:
        loss = query_feature.sum() * 0.0
        positive_cosine = query_feature.new_empty((0,), dtype=torch.float32)
        off_diagonal_mean = query_feature.new_tensor(float("nan"), dtype=torch.float32)
    else:
        cosine = supported_query @ supported_anchor.transpose(0, 1)
        labels = torch.arange(cosine.shape[0], device=cosine.device)
        loss = F.cross_entropy(cosine / temperature, labels)
        positive_cosine = cosine.diagonal()
        off_diagonal = ~torch.eye(
            cosine.shape[0],
            dtype=torch.bool,
            device=cosine.device,
        )
        off_diagonal_mean = cosine[off_diagonal].mean()

    statistics = PrototypeBatchStatistics(
        support_mask=support_mask.detach(),
        raw_support=query_batch.raw_support.detach(),
        purity_mass=query_batch.purity_mass.detach(),
        effective_sample_size=query_batch.effective_sample_size.detach(),
        positive_cosine=positive_cosine.detach(),
        off_diagonal_cosine_mean=off_diagonal_mean.detach(),
    )
    return loss, statistics


def prototype_weight_multiplier(epoch: int) -> float:
    """Frozen E1-5 zero, E6-10 linear ramp, E11+ full schedule."""

    if epoch <= 0:
        raise ValueError("epoch must be positive")
    if epoch <= 5:
        return 0.0
    if epoch <= 10:
        return (epoch - 5) / 5.0
    return 1.0


def snapshot_batchnorm_buffers(module: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "running_mean": child.running_mean.detach().clone(),
            "running_var": child.running_var.detach().clone(),
            "num_batches_tracked": child.num_batches_tracked.detach().clone(),
        }
        for name, child in module.named_modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    }


def assert_batchnorm_buffers_equal(
    expected: dict[str, dict[str, torch.Tensor]],
    module: nn.Module,
) -> None:
    actual = snapshot_batchnorm_buffers(module)
    if actual.keys() != expected.keys():
        raise RuntimeError("BatchNorm module set changed")
    for name in expected:
        for field in expected[name]:
            if not torch.equal(expected[name][field], actual[name][field]):
                raise RuntimeError(f"BatchNorm buffer changed: {name}.{field}")


def restore_batchnorm_buffers(
    state: dict[str, dict[str, torch.Tensor]],
    module: nn.Module,
) -> None:
    modules = dict(module.named_modules())
    if state.keys() - modules.keys():
        raise RuntimeError("BatchNorm state contains unknown module names")
    with torch.no_grad():
        for name, fields in state.items():
            child = modules[name]
            if not isinstance(child, nn.modules.batchnorm._BatchNorm):
                raise RuntimeError(f"saved BatchNorm module changed type: {name}")
            for field, value in fields.items():
                getattr(child, field).copy_(value)


@contextlib.contextmanager
def temporary_batchnorm_eval(module: nn.Module):
    """Use persistent BN buffers without disabling gradients or other modules."""

    batchnorms = [
        child
        for child in module.modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    ]
    previous = [child.training for child in batchnorms]
    try:
        for child in batchnorms:
            child.eval()
        yield
    finally:
        for child, was_training in zip(batchnorms, previous):
            child.train(was_training)


def snapshot_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextlib.contextmanager
def preserve_rng_state():
    state = snapshot_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)
