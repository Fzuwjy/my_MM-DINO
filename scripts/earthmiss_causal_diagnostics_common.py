"""Pure contracts for failure-directed EarthMiss causal diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
import torch.nn.functional as F


NUM_CLASSES = 8
IGNORE_INDEX = 8
SCALE_NAMES = ("P2", "P3", "P4", "P5")
ORACLE_STAGE_ORDER = (
    "adapter.all",
    "adapter.P2",
    "adapter.P3",
    "adapter.P4",
    "adapter.P5",
    "frm.P2",
    "frm.P3",
    "frm.P4",
    "frm.P5",
    "frm.all",
    "se.P2",
    "se.P3",
    "se.P4",
    "se.P5",
    "se.all",
    "prn.all",
    "head.logits",
)
ORACLE_SCREEN_STAGES = (
    "adapter.P2",
    "adapter.P3",
    "adapter.P4",
    "adapter.P5",
    "frm.P2",
    "frm.P3",
    "frm.P4",
    "frm.P5",
    "se.P2",
    "se.P3",
    "se.P4",
    "se.P5",
)


def stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF


def _clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


def _same_container(reference: Any, values: Sequence[Any]) -> Any:
    if isinstance(reference, tuple):
        return tuple(values)
    if isinstance(reference, list):
        return list(values)
    raise TypeError(f"expected list/tuple tensor container, got {type(reference)!r}")


def _blend(current: torch.Tensor, source: torch.Tensor, alpha: float) -> torch.Tensor:
    if current.shape != source.shape:
        raise RuntimeError(
            f"oracle stage shape mismatch: {tuple(current.shape)} != "
            f"{tuple(source.shape)}"
        )
    if current.device != source.device:
        raise RuntimeError("oracle source and current tensors are on different devices")
    if alpha == 0.0:
        return current
    if alpha == 1.0:
        return source
    return torch.lerp(current, source, alpha)


class OracleStageIntervention(AbstractContextManager):
    """Capture Full-state stages and replace one SAR-state stage read-only.

    The released canonical SampleAdapter duplicates one fused pyramid into two
    decoder slots. FRM therefore runs twice with equal inputs. The hook contract
    checks both facts instead of assigning modality meaning to those slots.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.adapter = model.adapter
        self.decoder = model.decoder
        required = (
            "frm",
            "fusion1",
            "fusion2",
            "fusion3",
            "fusion4",
            "neck",
            "out_conv",
        )
        if self.adapter is None or any(
            not hasattr(self.decoder, name) for name in required
        ):
            raise TypeError("oracle intervention requires released Run C modules")
        self.handles: list[Any] = []
        self.mode: str | None = None
        self.stage: str | None = None
        self.alpha = 0.0
        self.sources: dict[str, Any] = {}
        self.counts: dict[str, int] = {}
        self.hits = 0

    def __enter__(self) -> "OracleStageIntervention":
        self.handles.append(self.adapter.register_forward_hook(self._adapter_hook))
        self.handles.append(self.decoder.frm.register_forward_hook(self._frm_hook))
        for index in range(4):
            self.handles.append(
                getattr(self.decoder, f"fusion{index + 1}").register_forward_hook(
                    self._fusion_hook(index)
                )
            )
        self.handles.append(self.decoder.neck.register_forward_hook(self._neck_hook))
        self.handles.append(
            self.decoder.out_conv.register_forward_hook(self._head_hook)
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.mode = None
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _increment(self, name: str) -> int:
        ordinal = self.counts.get(name, 0)
        self.counts[name] = ordinal + 1
        return ordinal

    def _adapter_hook(self, module, inputs, output):
        if self.mode is None:
            return None
        self._increment("adapter")
        if not isinstance(output, (list, tuple)) or len(output) != 2:
            raise RuntimeError("canonical adapter must return two slots")
        if any(not isinstance(slot, (list, tuple)) or len(slot) != 4 for slot in output):
            raise RuntimeError("canonical adapter slots must contain P2-P5")
        for index in range(4):
            if not torch.equal(output[0][index], output[1][index]):
                raise RuntimeError("canonical adapter slots are no longer equal")
        if self.mode == "capture":
            values = tuple(_clone_tensor(value) for value in output[0])
            self.sources["adapter.all"] = values
            for scale, value in zip(SCALE_NAMES, values, strict=True):
                self.sources[f"adapter.{scale}"] = value
            return None
        if self.stage == "adapter.all":
            source = self.sources["adapter.all"]
            slots = []
            for slot in output:
                slots.append(
                    _same_container(
                        slot,
                        [_blend(value, source[i], self.alpha) for i, value in enumerate(slot)],
                    )
                )
            self.hits += 1
            return _same_container(output, slots)
        if self.stage is None or not self.stage.startswith("adapter."):
            return None
        scale = self.stage.split(".", 1)[1]
        index = SCALE_NAMES.index(scale)
        slots = []
        for slot in output:
            values = list(slot)
            values[index] = _blend(
                values[index], self.sources[self.stage], self.alpha
            )
            slots.append(
                _same_container(slot, values)
            )
        self.hits += 1
        return _same_container(output, slots)

    def _frm_hook(self, module, inputs, output):
        if self.mode is None:
            return None
        ordinal = self._increment("frm")
        if not isinstance(output, (list, tuple)) or len(output) != 4:
            raise RuntimeError("FRM must return P2-P5")
        if ordinal > 1:
            raise RuntimeError("canonical FRM must run exactly twice")
        if self.mode == "capture":
            values = tuple(_clone_tensor(value) for value in output)
            if ordinal == 0:
                self.sources["frm.all"] = values
                for scale, value in zip(SCALE_NAMES, values, strict=True):
                    self.sources[f"frm.{scale}"] = value
            else:
                reference = self.sources["frm.all"]
                if any(
                    not torch.equal(left, right)
                    for left, right in zip(reference, values, strict=True)
                ):
                    raise RuntimeError("duplicate canonical FRM slots differ")
            return None
        if self.stage == "frm.all":
            source = self.sources["frm.all"]
            self.hits += 1
            return _same_container(
                output,
                [_blend(value, source[i], self.alpha) for i, value in enumerate(output)],
            )
        if self.stage is not None and self.stage.startswith("frm."):
            scale = self.stage.split(".", 1)[1]
            index = SCALE_NAMES.index(scale)
            values = list(output)
            values[index] = _blend(values[index], self.sources[self.stage], self.alpha)
            self.hits += 1
            return _same_container(output, values)
        return None

    def _fusion_hook(self, index: int):
        scale = SCALE_NAMES[index]

        def hook(module, inputs, output):
            if self.mode is None:
                return None
            self._increment(f"fusion.{scale}")
            key = f"se.{scale}"
            if self.mode == "capture":
                self.sources[key] = _clone_tensor(output)
                if index == 3:
                    self.sources["se.all"] = tuple(
                        self.sources[f"se.{name}"] for name in SCALE_NAMES
                    )
                return None
            if self.stage == "se.all":
                self.hits += 1
                return _blend(output, self.sources["se.all"][index], self.alpha)
            if self.stage == key:
                self.hits += 1
                return _blend(output, self.sources[key], self.alpha)
            return None

        return hook

    def _neck_hook(self, module, inputs, output):
        if self.mode is None:
            return None
        self._increment("neck")
        if not isinstance(output, (list, tuple)) or len(output) != 3:
            raise RuntimeError("PRN must return P2/P3/P4")
        if self.mode == "capture":
            self.sources["prn.all"] = tuple(_clone_tensor(value) for value in output)
            return None
        if self.stage != "prn.all":
            return None
        source = self.sources["prn.all"]
        self.hits += 1
        return _same_container(
            output,
            [_blend(value, source[i], self.alpha) for i, value in enumerate(output)],
        )

    def _head_hook(self, module, inputs, output):
        if self.mode is None:
            return None
        self._increment("head")
        if self.mode == "capture":
            self.sources["head.logits"] = _clone_tensor(output)
            return None
        if self.stage == "head.logits":
            self.hits += 1
            return _blend(output, self.sources["head.logits"], self.alpha)
        return None

    @staticmethod
    def _expected_counts() -> dict[str, int]:
        expected = {"adapter": 1, "frm": 2, "neck": 1, "head": 1}
        expected.update({f"fusion.{scale}": 1 for scale in SCALE_NAMES})
        return expected

    def _validate_counts(self) -> None:
        expected = self._expected_counts()
        if self.counts != expected:
            raise RuntimeError(
                f"oracle hook call contract changed: expected {expected}, got {self.counts}"
            )

    def capture_full(self, forward) -> torch.Tensor:
        self.mode = "capture"
        self.stage = None
        self.sources = {}
        self.counts = {}
        self.hits = 0
        try:
            logits = forward()
        finally:
            self.mode = None
        self._validate_counts()
        missing = [stage for stage in ORACLE_STAGE_ORDER if stage not in self.sources]
        if missing:
            raise RuntimeError(f"missing captured oracle stages: {missing}")
        return logits

    def intervene(self, stage: str, alpha: float, forward) -> torch.Tensor:
        if stage not in ORACLE_STAGE_ORDER:
            raise ValueError(f"unsupported oracle stage: {stage}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("oracle alpha must be in [0,1]")
        if not self.sources:
            raise RuntimeError("capture_full must run before intervention")
        self.mode = "intervene"
        self.stage = stage
        self.alpha = float(alpha)
        self.counts = {}
        self.hits = 0
        try:
            logits = forward()
        finally:
            self.mode = None
        self._validate_counts()
        expected_hits = {
            "adapter.all": 1,
            "adapter.P2": 1,
            "adapter.P3": 1,
            "adapter.P4": 1,
            "adapter.P5": 1,
            "frm.all": 2,
            "frm.P2": 2,
            "frm.P3": 2,
            "frm.P4": 2,
            "frm.P5": 2,
            "se.all": 4,
            "se.P2": 1,
            "se.P3": 1,
            "se.P4": 1,
            "se.P5": 1,
            "prn.all": 1,
            "head.logits": 1,
        }[stage]
        if self.hits != expected_hits:
            raise RuntimeError(
                f"oracle stage {stage} expected {expected_hits} hits, got {self.hits}"
            )
        return logits


class VariantLogitStitcher:
    """Overlap-average an arbitrary fixed manifest of crop-logit variants."""

    def __init__(
        self,
        height: int,
        width: int,
        num_classes: int,
        variants: Sequence[str],
    ) -> None:
        if len(variants) != len(set(variants)) or not variants:
            raise ValueError("variant names must be non-empty and unique")
        self.variants = tuple(variants)
        self.sums = {
            name: torch.zeros(1, num_classes, height, width, dtype=torch.float32)
            for name in self.variants
        }
        self.count = torch.zeros(1, 1, height, width, dtype=torch.int16)

    def add(
        self,
        coordinates: Sequence[tuple[int, int, int, int]],
        logits: Mapping[str, torch.Tensor],
    ) -> None:
        if set(logits) != set(self.variants):
            raise ValueError("crop logit variant manifest changed")
        for name in self.variants:
            values = logits[name].detach().to("cpu", torch.float32)
            if values.shape[0] != len(coordinates):
                raise ValueError("crop logit batch does not match coordinates")
            for index, (y1, y2, x1, x2) in enumerate(coordinates):
                if tuple(values.shape[-2:]) != (y2 - y1, x2 - x1):
                    raise ValueError("crop logits do not match coordinate extent")
                self.sums[name][:, :, y1:y2, x1:x2] += values[index]
        for y1, y2, x1, x2 in coordinates:
            self.count[:, :, y1:y2, x1:x2] += 1

    def finalize(self) -> dict[str, torch.Tensor]:
        if bool((self.count == 0).any()):
            raise RuntimeError("sliding-window reconstruction left uncovered pixels")
        denominator = self.count.to(torch.float32)
        return {name: values / denominator for name, values in self.sums.items()}


@dataclass(frozen=True)
class RecoverabilityExamples:
    features: torch.Tensor
    targets: torch.Tensor
    class_ids: torch.Tensor
    pure_cells: int
    sar_wrong_cells: int


def exact_class_occupancy(
    target: torch.Tensor,
    output_size: tuple[int, int],
    *,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError("target must have shape [B,H,W] or [B,1,H,W]")
    height, width = target.shape[-2:]
    out_h, out_w = map(int, output_size)
    if height % out_h or width % out_w:
        raise ValueError("target shape must be exactly divisible by feature grid")
    invalid = ((target < 0) | (target > ignore_index))
    if invalid.any():
        raise ValueError("target contains invalid EarthMiss labels")
    valid = (target >= 0) & (target < num_classes)
    safe = target.clamp(0, num_classes - 1)
    one_hot = F.one_hot(safe.long(), num_classes).permute(0, 3, 1, 2).float()
    one_hot *= valid.unsqueeze(1)
    return F.avg_pool2d(
        one_hot,
        kernel_size=(height // out_h, width // out_w),
        stride=(height // out_h, width // out_w),
    )


def recoverability_examples(
    sar_feature: torch.Tensor,
    full_logits: torch.Tensor,
    sar_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    purity_threshold: float = 0.75,
) -> RecoverabilityExamples:
    """Build pure P5-cell examples conditional on canonical SAR being wrong."""

    if sar_feature.ndim != 4:
        raise ValueError("SAR feature must be BCHW")
    if full_logits.shape != sar_logits.shape or full_logits.ndim != 4:
        raise ValueError("Full/SAR logits must have equal BCHW shapes")
    if not 0.5 < purity_threshold <= 1.0:
        raise ValueError("purity threshold must be in (0.5,1]")
    output_size = tuple(sar_feature.shape[-2:])
    occupancy = exact_class_occupancy(target, output_size)
    purity, class_ids = occupancy.max(dim=1)
    pure = purity >= purity_threshold
    pooled_full = F.adaptive_avg_pool2d(full_logits.float(), output_size).argmax(1)
    pooled_sar = F.adaptive_avg_pool2d(sar_logits.float(), output_size).argmax(1)
    sar_wrong = pure & (pooled_sar != class_ids)
    recoverable = pooled_full == class_ids
    features = sar_feature.permute(0, 2, 3, 1)[sar_wrong].float()
    return RecoverabilityExamples(
        features=features,
        targets=recoverable[sar_wrong].to(torch.int64),
        class_ids=class_ids[sar_wrong].to(torch.int64),
        pure_cells=int(pure.sum()),
        sar_wrong_cells=int(sar_wrong.sum()),
    )


def parameter_group_manifest(model: torch.nn.Module) -> dict[str, tuple[str, ...]]:
    """Return overlapping scientific parameter groups for gradient attribution."""

    names = [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]

    def select(*prefixes: str, contains: str | None = None) -> tuple[str, ...]:
        selected = [
            name
            for name in names
            if (prefixes and name.startswith(prefixes))
            or (contains is not None and contains in name)
        ]
        return tuple(selected)

    groups: dict[str, tuple[str, ...]] = {
        "adapter.all": select("adapter."),
        "adapter.modality_weights": select("adapter.modality_weights."),
        "frm.all": select("decoder.frm."),
        "sefusion.all": select(
            "decoder.fusion1.",
            "decoder.fusion2.",
            "decoder.fusion3.",
            "decoder.fusion4.",
        ),
        "prn.all": select("decoder.neck."),
        "head.all": select("decoder.out_conv."),
        "decoder.all": select("decoder."),
    }
    for index, scale in enumerate(SCALE_NAMES):
        groups[f"adapter.{scale}"] = tuple(
            name
            for name in names
            if name.startswith(f"adapter.projects.{index}.")
            or name.startswith(f"adapter.resize_layers.{index}.")
        )
        target = index + 2
        groups[f"frm.target.{scale}"] = tuple(
            name
            for name in names
            if name.startswith(
                f"decoder.frm.conv_scales.conv_scale{target}_"
            )
            or name.startswith(f"decoder.frm.conv_aggregation_s{target}.")
        )
        groups[f"sefusion.{scale}"] = select(f"decoder.fusion{index + 1}.")
    empty = [name for name, values in groups.items() if not values]
    if empty:
        raise RuntimeError(f"empty causal gradient parameter groups: {empty}")
    return groups


def gradient_tensor_primitives(
    full_gradients: Mapping[str, torch.Tensor | None],
    sar_gradients: Mapping[str, torch.Tensor | None],
    names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Reduce each parameter tensor once before overlapping group summaries."""

    result: dict[str, dict[str, Any]] = {}
    for name in names:
        full = full_gradients.get(name)
        sar = sar_gradients.get(name)
        if full is None and sar is None:
            result[name] = {
                "state": "inactive",
                "dot": 0.0,
                "full_sq": 0.0,
                "sar_sq": 0.0,
            }
            continue
        if full is None:
            result[name] = {
                "state": "sar_only",
                "dot": 0.0,
                "full_sq": 0.0,
                "sar_sq": float(sar.square().sum(dtype=torch.float64)),
            }
            continue
        if sar is None:
            result[name] = {
                "state": "full_only",
                "dot": 0.0,
                "full_sq": float(full.square().sum(dtype=torch.float64)),
                "sar_sq": 0.0,
            }
            continue
        tensor_dot = float((full * sar).sum(dtype=torch.float64))
        result[name] = {
            "state": "both",
            "dot": tensor_dot,
            "full_sq": float(full.square().sum(dtype=torch.float64)),
            "sar_sq": float(sar.square().sum(dtype=torch.float64)),
        }
    return result


def gradient_pair_statistics_from_primitives(
    primitives: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
    *,
    eps: float = 1e-30,
) -> dict[str, Any]:
    dot = sum(float(primitives[name]["dot"]) for name in names)
    full_sq = sum(float(primitives[name]["full_sq"]) for name in names)
    sar_sq = sum(float(primitives[name]["sar_sq"]) for name in names)
    states = [str(primitives[name]["state"]) for name in names]
    both = states.count("both")
    full_only = states.count("full_only")
    sar_only = states.count("sar_only")
    negative_tensor_dots = sum(
        primitives[name]["state"] == "both" and primitives[name]["dot"] < 0.0
        for name in names
    )
    full_norm = math.sqrt(full_sq)
    sar_norm = math.sqrt(sar_sq)
    denominator = full_norm * sar_norm
    cosine = dot / denominator if denominator > eps else None
    return {
        "parameter_tensors": len(names),
        "active_both": both,
        "full_only": full_only,
        "sar_only": sar_only,
        "full_norm": full_norm,
        "sar_norm": sar_norm,
        "sar_to_full_norm_ratio": sar_norm / full_norm if full_norm > eps else None,
        "dot": dot,
        "cosine": max(-1.0, min(1.0, cosine)) if cosine is not None else None,
        "conflict": cosine < 0.0 if cosine is not None else None,
        "negative_tensor_dot_fraction": (
            negative_tensor_dots / both if both else None
        ),
    }


def gradient_pair_statistics(
    full_gradients: Mapping[str, torch.Tensor | None],
    sar_gradients: Mapping[str, torch.Tensor | None],
    names: Sequence[str],
    *,
    eps: float = 1e-30,
) -> dict[str, Any]:
    """Compute one group-level Full-vs-SAR gradient geometry record."""

    primitives = gradient_tensor_primitives(full_gradients, sar_gradients, names)
    return gradient_pair_statistics_from_primitives(primitives, names, eps=eps)


def summarize_gradient_records(
    records: Sequence[Mapping[str, Mapping[str, Any]]],
    group_order: Sequence[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for group in group_order:
        rows = [record[group] for record in records]
        cosines = [row["cosine"] for row in rows if row["cosine"] is not None]
        ratios = [
            row["sar_to_full_norm_ratio"]
            for row in rows
            if row["sar_to_full_norm_ratio"] is not None
        ]

        def describe(values: Sequence[float]) -> dict[str, Any]:
            if not values:
                return {"n": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
            tensor = torch.tensor(values, dtype=torch.float64)
            return {
                "n": len(values),
                "mean": float(tensor.mean()),
                "median": float(torch.quantile(tensor, 0.5)),
                "minimum": float(tensor.min()),
                "maximum": float(tensor.max()),
            }

        summary[group] = {
            "cosine": describe(cosines),
            "sar_to_full_norm_ratio": describe(ratios),
            "negative_cosine_fraction": (
                sum(value < 0.0 for value in cosines) / len(cosines)
                if cosines
                else None
            ),
        }
    return summary
