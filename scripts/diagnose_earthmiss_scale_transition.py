"""Diagnose where Run C amplifies the Full-to-SAR representation gap.

The script is evaluation-only.  It loads one frozen Run C checkpoint, extracts
the RGB/SAR DINO outputs once per deterministic crop batch, and decodes the
same cached tensors in canonical Full and canonical SAR states.  Forward hooks
observe the released SampleAdapter/FRM/SEFusion/PRN path without modifying it.

No activation, image, logit cache, parameter update, or training artifact is
written.  The only output is one bounded JSON report containing aggregate
feature, frequency, boundary, class-prototype, and endpoint statistics.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Sequence
from itertools import islice
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import build_dataset  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from utils.earthmiss_metrics import EarthMissMetrics  # noqa: E402

from scripts.diagnose_earthmiss_missing_v3 import (  # noqa: E402
    load_frozen_model,
)
from scripts.earthmiss_scale_transition_common import (  # noqa: E402
    CROSS_SCALE_STAGE_ORDER,
    NUM_CLASSES,
    STAGE_ORDER,
    ScaleTransitionAccumulator,
    compact_by_tile_stage_summaries,
    feature_pair_batch_statistics,
    segmentation_region_statistics,
    summarize_amplification,
    summarize_cross_scale_alignment_degradation,
    summarize_error_correlations,
    write_json_exclusive,
)
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
)


SCHEMA = "earthmiss_scale_transition_diagnostic_v1"
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-scale-transition/"
    "run-c-val-diagnostic.json"
)
DEFAULT_WINDOW_SIZE = 512
DEFAULT_STRIDE = 341
DEFAULT_BOOTSTRAP_SEED = 20260810
EXPECTED_VAL_TILES = 277
EXPECTED_VAL_SELECTION_CLASS_IDS = list(range(7))
EXPECTED_RUN_C_CHECKPOINT_SHA256 = (
    "b038dfcfe771ca5c67da500acc2b88e74f066496cac332fc3b76d4377b73dff9"
)
EXPECTED_RUN_C_EPOCH = 15
EXPECTED_RUN_C_SEED = 42
SCALE_NAMES = ("P2", "P3", "P4", "P5")
DINO_STAGE_ORDER = (
    "dino.rgb_vs_sar.L02",
    "dino.rgb_vs_sar.L05",
    "dino.rgb_vs_sar.L08",
    "dino.rgb_vs_sar.L11",
)
CROSS_SCALE_FEATURE_PAIRS = {
    "prn.cross_scale.P5_to_P4": (
        "prn.resized.P5_to_P4",
        "decoder.se_fused.P4",
    ),
    "prn.cross_scale.P4_to_P3": (
        "prn.resized.P4_to_P3",
        "decoder.se_fused.P3",
    ),
    "prn.cross_scale.P3_to_P2": (
        "prn.resized.P3_to_P2",
        "decoder.se_fused.P2",
    ),
}
if tuple(CROSS_SCALE_FEATURE_PAIRS) != CROSS_SCALE_STAGE_ORDER:
    raise RuntimeError("runner/common cross-scale stage contracts differ")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--feature-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--smoke-tiles",
        type=int,
        default=0,
        help="Non-formal deterministic prefix length; zero means complete Val.",
    )
    parser.add_argument("--cka-max-points", type=int, default=128)
    parser.add_argument("--cka-max-channels", type=int, default=128)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--skip-cached-equivalence-check",
        action="store_true",
        help="Skip the first-batch exact cached-vs-direct forward assertion.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.window_size <= 0 or args.window_size % 64:
        raise ValueError("--window-size must be a positive multiple of 64")
    if args.stride <= 0 or args.stride > args.window_size:
        raise ValueError("--stride must be in [1, window-size]")
    if args.feature_batch_size != 1:
        raise ValueError(
            "--feature-batch-size must be exactly 1 to bound diagnostic memory"
        )
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.smoke_tiles < 0:
        raise ValueError("--smoke-tiles must be non-negative")
    if args.cka_max_points <= 1:
        raise ValueError("--cka-max-points must exceed one")
    if args.cka_max_channels <= 0:
        raise ValueError("--cka-max-channels must be positive")
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")


def sliding_window_coordinates(
    image_height: int,
    image_width: int,
    *,
    window_size: int,
    stride: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return the exact coordinate order used by ``slide_inference``."""

    if image_height < window_size or image_width < window_size:
        raise ValueError(
            "scale-transition diagnosis requires tiles at least as large as "
            f"the {window_size}x{window_size} crop; got "
            f"{image_height}x{image_width}"
        )
    if window_size <= 0 or stride <= 0 or stride > window_size:
        raise ValueError("invalid window/stride")
    h_grids = max(image_height - window_size + stride - 1, 0) // stride + 1
    w_grids = max(image_width - window_size + stride - 1, 0) // stride + 1
    coordinates = []
    for h_index in range(h_grids):
        for w_index in range(w_grids):
            y1 = h_index * stride
            x1 = w_index * stride
            y2 = min(y1 + window_size, image_height)
            x2 = min(x1 + window_size, image_width)
            y1 = max(y2 - window_size, 0)
            x1 = max(x2 - window_size, 0)
            coordinates.append((y1, y2, x1, x2))
    if len(set(coordinates)) != len(coordinates):
        raise RuntimeError("sliding-window plan contains duplicate coordinates")
    return tuple(coordinates)


def iter_crop_batches(
    rgb: torch.Tensor,
    sar: torch.Tensor,
    target: torch.Tensor,
    coordinates: Sequence[tuple[int, int, int, int]],
    *,
    batch_size: int,
) -> Iterator[
    tuple[
        tuple[tuple[int, int, int, int], ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
]:
    if rgb.ndim != 4 or sar.ndim != 4 or target.ndim != 3:
        raise ValueError("tile tensors must have shapes BCHW, BCHW, and BHW")
    if rgb.shape[0] != 1 or sar.shape[0] != 1 or target.shape[0] != 1:
        raise ValueError("diagnostic tile loader must use batch_size=1")
    if rgb.shape[-2:] != sar.shape[-2:] or rgb.shape[-2:] != target.shape[-2:]:
        raise ValueError("RGB, SAR, and target tile shapes differ")
    if batch_size <= 0:
        raise ValueError("crop batch_size must be positive")

    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = tuple(coordinates[start : start + batch_size])
        rgb_crops = []
        sar_crops = []
        target_crops = []
        for y1, y2, x1, x2 in batch_coordinates:
            rgb_crops.append(rgb[:, :, y1:y2, x1:x2])
            sar_crops.append(sar[:, :, y1:y2, x1:x2])
            target_crops.append(target[:, y1:y2, x1:x2])
        yield (
            batch_coordinates,
            torch.cat(rgb_crops, dim=0),
            torch.cat(sar_crops, dim=0),
            torch.cat(target_crops, dim=0),
        )


class LogitStitcher:
    """CPU reconstruction of the released overlap-averaged slide logits."""

    def __init__(self, height: int, width: int, num_classes: int) -> None:
        self.sums = {
            "full": torch.zeros(1, num_classes, height, width, dtype=torch.float32),
            "sar": torch.zeros(1, num_classes, height, width, dtype=torch.float32),
        }
        self.count = torch.zeros(1, 1, height, width, dtype=torch.int16)

    def add(
        self,
        coordinates: Sequence[tuple[int, int, int, int]],
        full_logits: torch.Tensor,
        sar_logits: torch.Tensor,
    ) -> None:
        full_logits = full_logits.detach().to("cpu", torch.float32)
        sar_logits = sar_logits.detach().to("cpu", torch.float32)
        if full_logits.shape != sar_logits.shape:
            raise ValueError("Full and SAR crop logits differ in shape")
        if full_logits.shape[0] != len(coordinates):
            raise ValueError("crop logit batch does not match coordinates")
        for index, (y1, y2, x1, x2) in enumerate(coordinates):
            expected = (y2 - y1, x2 - x1)
            if tuple(full_logits.shape[-2:]) != expected:
                raise ValueError("crop logits do not match their coordinate extent")
            self.sums["full"][:, :, y1:y2, x1:x2] += full_logits[index]
            self.sums["sar"][:, :, y1:y2, x1:x2] += sar_logits[index]
            self.count[:, :, y1:y2, x1:x2] += 1

    def finalize(self) -> dict[str, torch.Tensor]:
        if bool((self.count == 0).any()):
            raise RuntimeError("sliding-window reconstruction left uncovered pixels")
        denominator = self.count.to(torch.float32)
        return {
            endpoint: values / denominator
            for endpoint, values in self.sums.items()
        }


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu", torch.float32).clone()


class ActivationRecorder:
    """Read-only hooks for one canonical endpoint forward at a time."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.adapter = model.adapter
        self.decoder = model.decoder
        required = (
            "projects",
            "resize_layers",
            "modality_weights",
        )
        if self.adapter is None or any(
            not hasattr(self.adapter, name) for name in required
        ):
            raise TypeError("diagnostic requires the released SampleAdapter")
        decoder_required = (
            "frm",
            "fusion1",
            "fusion2",
            "fusion3",
            "fusion4",
            "neck",
            "out_conv",
        )
        if any(not hasattr(self.decoder, name) for name in decoder_required):
            raise TypeError("diagnostic requires the released base Decoder")
        self.handles: list[Any] = []
        self.active = False
        self.endpoint: str | None = None
        self.active_indices: tuple[int, ...] = ()
        self.counts: dict[str, int] = {}
        self.projected: dict[str, dict[int, torch.Tensor]] = {}
        self.stages: dict[str, torch.Tensor] = {}

    def __enter__(self) -> "ActivationRecorder":
        for scale_index, module in enumerate(self.adapter.projects):
            self.handles.append(
                module.register_forward_hook(self._project_hook(scale_index))
            )
        for scale_index, module in enumerate(self.adapter.resize_layers):
            self.handles.append(
                module.register_forward_hook(self._resize_hook(scale_index))
            )
        self.handles.append(self.adapter.register_forward_hook(self._adapter_hook))
        self.handles.append(self.decoder.frm.register_forward_hook(self._frm_hook))
        for scale_index in range(4):
            module = getattr(self.decoder, f"fusion{scale_index + 1}")
            self.handles.append(
                module.register_forward_hook(self._fusion_hook(scale_index))
            )
        td_mapping = ((2, "P4"), (1, "P3"), (0, "P2"))
        for module_index, output_scale in td_mapping:
            self.handles.append(
                self.decoder.neck.td_convs[module_index].register_forward_hook(
                    self._td_hook(output_scale)
                )
            )
        self.handles.append(self.decoder.neck.register_forward_hook(self._neck_hook))
        self.handles.append(self.decoder.out_conv.register_forward_hook(self._head_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.active = False
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _increment(self, key: str) -> int:
        ordinal = self.counts.get(key, 0)
        self.counts[key] = ordinal + 1
        return ordinal

    def _store_unique(self, key: str, tensor: torch.Tensor) -> None:
        if key in self.stages:
            raise RuntimeError(f"diagnostic stage recorded twice: {key}")
        self.stages[key] = _cpu_clone(tensor)

    def _project_hook(self, scale_index: int) -> Callable:
        scale = SCALE_NAMES[scale_index]

        def hook(module, inputs, output):
            if not self.active:
                return None
            ordinal = self._increment(f"project.{scale}")
            if ordinal >= len(self.active_indices):
                raise RuntimeError(f"unexpected extra project call for {scale}")
            modality_index = self.active_indices[ordinal]
            by_modality = self.projected.setdefault(scale, {})
            if modality_index in by_modality:
                raise RuntimeError(
                    f"projected modality {modality_index} recorded twice at {scale}"
                )
            by_modality[modality_index] = _cpu_clone(output)
            return None

        return hook

    def _resize_hook(self, scale_index: int) -> Callable:
        scale = SCALE_NAMES[scale_index]

        def hook(module, inputs, output):
            if self.active:
                self._increment(f"resize.{scale}")
            return None

        return hook

    def _adapter_hook(self, module, inputs, output):
        if not self.active:
            return None
        self._increment("adapter")
        if not isinstance(output, (list, tuple)) or len(output) != 2:
            raise RuntimeError("canonical adapter must return two decoder slots")
        if any(len(slot) != 4 for slot in output):
            raise RuntimeError("canonical adapter slots must contain four scales")
        for scale_index, scale in enumerate(SCALE_NAMES):
            if not torch.equal(output[0][scale_index], output[1][scale_index]):
                raise RuntimeError(
                    f"canonical adapter slots differ at {scale}; released behavior changed"
                )
            self._store_unique(f"adapter.fused.{scale}", output[0][scale_index])
        return None

    def _frm_hook(self, module, inputs, output):
        if not self.active:
            return None
        ordinal = self._increment("frm")
        if not isinstance(output, (list, tuple)) or len(output) != 4:
            raise RuntimeError("FRM must return four scales")
        if ordinal == 0:
            for scale, tensor in zip(SCALE_NAMES, output, strict=True):
                self._store_unique(f"frm.{scale}", tensor)
        elif ordinal == 1:
            for scale, tensor in zip(SCALE_NAMES, output, strict=True):
                if not torch.equal(self.stages[f"frm.{scale}"], _cpu_clone(tensor)):
                    raise RuntimeError(
                        f"duplicate canonical FRM slots differ at {scale}"
                    )
        else:
            raise RuntimeError("FRM was called more than twice")
        return None

    def _fusion_hook(self, scale_index: int) -> Callable:
        scale = SCALE_NAMES[scale_index]

        def hook(module, inputs, output):
            if self.active:
                self._increment(f"fusion.{scale}")
                self._store_unique(f"decoder.se_fused.{scale}", output)
            return None

        return hook

    def _td_hook(self, output_scale: str) -> Callable:
        def hook(module, inputs, output):
            if self.active:
                self._increment(f"td.{output_scale}")
                if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
                    raise RuntimeError(
                        f"PRN td_conv {output_scale} must receive one tensor"
                    )
                concatenated = inputs[0]
                if concatenated.ndim != 4:
                    raise RuntimeError(
                        f"PRN td_conv {output_scale} input must be BCHW"
                    )
                channels = concatenated.shape[1]
                if channels <= 0 or channels % 2:
                    raise RuntimeError(
                        f"PRN td_conv {output_scale} concat channels must be "
                        f"positive and even, got {channels}"
                    )
                coarse_input_scale = {
                    "P4": "P5",
                    "P3": "P4",
                    "P2": "P3",
                }[output_scale]
                self._store_unique(
                    f"prn.resized.{coarse_input_scale}_to_{output_scale}",
                    concatenated[:, : channels // 2],
                )
                self._store_unique(f"prn.concat.{output_scale}", concatenated)
                self._store_unique(f"prn.td.{output_scale}", output)
            return None

        return hook

    def _neck_hook(self, module, inputs, output):
        if not self.active:
            return None
        self._increment("neck")
        if not isinstance(output, (list, tuple)) or len(output) != 3:
            raise RuntimeError("PRN must return P2/P3/P4")
        for scale, tensor in zip(("P2", "P3", "P4"), output, strict=True):
            self._store_unique(f"prn.out.{scale}", tensor)
        return None

    def _head_hook(self, module, inputs, output):
        if self.active:
            self._increment("head")
            self._store_unique("logits.decoder", output)
        return None

    def _reset(self, endpoint: str, active_indices: tuple[int, ...]) -> None:
        if self.active:
            raise RuntimeError("activation recorder is already active")
        if endpoint not in {"full", "sar"}:
            raise ValueError(f"unsupported diagnostic endpoint: {endpoint}")
        expected = (0, 1) if endpoint == "full" else (1,)
        if active_indices != expected:
            raise ValueError(
                f"endpoint {endpoint} requires active indices {expected}, got "
                f"{active_indices}"
            )
        self.endpoint = endpoint
        self.active_indices = active_indices
        self.counts = {}
        self.projected = {}
        self.stages = {}

    def _expected_counts(self) -> dict[str, int]:
        modality_calls = len(self.active_indices)
        expected = {"adapter": 1, "frm": 2, "neck": 1, "head": 1}
        for scale in SCALE_NAMES:
            expected[f"project.{scale}"] = modality_calls
            expected[f"resize.{scale}"] = modality_calls
            expected[f"fusion.{scale}"] = 1
        for scale in ("P4", "P3", "P2"):
            expected[f"td.{scale}"] = 1
        return expected

    def _build_pre_resize_stages(self) -> None:
        weights = []
        for modality_index in self.active_indices:
            parameter = self.adapter.modality_weights[
                f"weight_modality_{modality_index}"
            ]
            weights.append(float(torch.sigmoid(parameter.detach()).cpu()))
        denominator = sum(weights)
        if denominator <= 0:
            raise RuntimeError("adapter modality weight denominator is not positive")
        normalized = [weight / denominator for weight in weights]
        for scale in SCALE_NAMES:
            if set(self.projected.get(scale, {})) != set(self.active_indices):
                raise RuntimeError(f"missing projected modalities at {scale}")
            fused = sum(
                weight * self.projected[scale][modality_index]
                for weight, modality_index in zip(
                    normalized, self.active_indices, strict=True
                )
            )
            self._store_unique(f"adapter.pre_resize.{scale}", fused)

    def run(
        self,
        endpoint: str,
        active_indices: tuple[int, ...],
        forward: Callable[[], torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        self._reset(endpoint, active_indices)
        try:
            self.active = True
            logits = forward()
        finally:
            self.active = False
        self._build_pre_resize_stages()
        self._store_unique("logits.final", logits)
        expected_counts = self._expected_counts()
        if self.counts != expected_counts:
            raise RuntimeError(
                f"{endpoint} hook call contract changed: expected "
                f"{expected_counts}, got {self.counts}"
            )
        missing = [stage for stage in STAGE_ORDER if stage not in self.stages]
        extra = sorted(set(self.stages) - set(STAGE_ORDER))
        if missing or extra:
            raise RuntimeError(
                f"diagnostic stage contract changed: missing={missing}, extra={extra}"
            )
        audit = {
            "active_indices": list(active_indices),
            "call_counts": dict(sorted(self.counts.items())),
            "canonical_adapter_slots_equal": True,
            "canonical_frm_slots_equal": True,
        }
        return logits, dict(self.stages), audit


def snapshot_batchnorm_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    suffixes = ("running_mean", "running_var", "num_batches_tracked")
    return {
        name: buffer.detach().to("cpu").clone()
        for name, buffer in model.named_buffers()
        if name.endswith(suffixes)
    }


def assert_buffers_unchanged(
    before: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> dict[str, Any]:
    after = snapshot_batchnorm_buffers(model)
    if set(before) != set(after):
        raise RuntimeError("BatchNorm buffer manifest changed during diagnosis")
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    if changed:
        raise RuntimeError(f"BatchNorm buffers changed during diagnosis: {changed}")
    return {"buffer_count": len(before), "unchanged": True}


@torch.inference_mode()
def verify_cached_forward_equivalence(
    model: torch.nn.Module,
    rgb: torch.Tensor,
    sar: torch.Tensor,
    backbone_outputs: tuple,
) -> dict[str, Any]:
    result = {}
    for endpoint in ("full", "sar"):
        availability = canonical_availability(
            endpoint,
            batch_size=rgb.shape[0],
            device=rgb.device,
        )
        cached = model.forward_from_backbone_outputs(
            rgb,
            sar,
            backbone_outputs=backbone_outputs,
            availability=availability,
        )
        direct = model(rgb, sar, availability=availability)
        maximum = float((cached - direct).abs().max())
        if not torch.equal(cached, direct):
            raise RuntimeError(
                f"cached {endpoint} forward differs from direct forward; "
                f"max_abs_delta={maximum}"
            )
        result[endpoint] = {
            "exact_equal": True,
            "max_abs_delta": maximum,
        }
    return result


def build_loader(args: argparse.Namespace):
    dataset = build_dataset(
        "EarthMiss",
        args.split,
        dataset_root=args.dataset_root,
        window_size=(args.window_size, args.window_size),
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
        cache_size=0,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _metrics_summary(
    pooled: dict[str, EarthMissMetrics],
    by_city: dict[str, dict[str, EarthMissMetrics]],
) -> dict[str, Any]:
    return {
        "pooled": {
            endpoint: evaluator.compute()
            for endpoint, evaluator in pooled.items()
        },
        "by_city": {
            city: {
                endpoint: evaluator.compute()
                for endpoint, evaluator in endpoints.items()
            }
            for city, endpoints in sorted(by_city.items())
        },
    }


@torch.inference_mode()
def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite diagnostic report: {output_path}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("EarthMiss scale-transition diagnosis requires CUDA")
    checkpoint_path = Path(args.checkpoint)
    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    device = torch.device("cuda")
    model, checkpoint_record = load_frozen_model(
        checkpoint_path,
        "c",
        weights_path,
        device,
    )
    checkpoint_binding = {
        "sha256": EXPECTED_RUN_C_CHECKPOINT_SHA256,
        "epoch": EXPECTED_RUN_C_EPOCH,
        "seed": EXPECTED_RUN_C_SEED,
    }
    observed_binding = {
        "sha256": checkpoint_record.get("sha256"),
        "epoch": checkpoint_record.get("epoch"),
        "seed": checkpoint_record.get("seed"),
    }
    if observed_binding != checkpoint_binding:
        raise ValueError(
            "scale-transition diagnosis is bound to the preserved Run C E15 "
            f"checkpoint: expected {checkpoint_binding}, got {observed_binding}"
        )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("diagnostic model must be frozen and in eval mode")
    dataset, loader = build_loader(args)
    expected_tiles = len(dataset)
    if expected_tiles != EXPECTED_VAL_TILES:
        raise RuntimeError(
            "EarthMiss Val manifest changed: expected "
            f"{EXPECTED_VAL_TILES} tiles, got {expected_tiles}"
        )
    processed_limit = (
        min(args.smoke_tiles, expected_tiles) if args.smoke_tiles else expected_tiles
    )
    if processed_limit <= 0:
        raise RuntimeError("diagnostic split contains no tiles")

    feature_accumulator = ScaleTransitionAccumulator()
    backbone_accumulator = ScaleTransitionAccumulator(
        stage_order=DINO_STAGE_ORDER,
        left_label="rgb",
        right_label="sar",
    )
    cross_scale_accumulators = {
        endpoint: ScaleTransitionAccumulator(
            stage_order=CROSS_SCALE_STAGE_ORDER,
            left_label="resized_coarse",
            right_label="lateral",
        )
        for endpoint in ("full", "sar")
    }
    endpoint_metrics = {
        endpoint: EarthMissMetrics() for endpoint in ("full", "sar")
    }
    city_metrics: dict[str, dict[str, EarthMissMetrics]] = {}
    tile_outcomes: dict[str, dict[str, Any]] = {}
    batchnorm_before = snapshot_batchnorm_buffers(model)
    cached_equivalence = None
    hook_contract: dict[str, Any] = {}
    crop_batches = 0
    crop_windows = 0

    with ActivationRecorder(model) as recorder:
        progress = tqdm(
            islice(loader, processed_limit),
            total=processed_limit,
            desc="scale-transition",
        )
        for tile_index, (rgb, sar, target) in enumerate(progress):
            sample = dataset.samples[tile_index]
            city = str(sample.city)
            tile_id = str(sample.tile_id)
            tile_key = f"{city}/{tile_id}"
            if tile_key in tile_outcomes:
                raise RuntimeError(f"duplicate EarthMiss tile: {tile_key}")
            height, width = target.shape[-2:]
            coordinates = sliding_window_coordinates(
                height,
                width,
                window_size=args.window_size,
                stride=args.stride,
            )
            stitcher = LogitStitcher(height, width, NUM_CLASSES)

            for (
                batch_coordinates,
                rgb_crops_cpu,
                sar_crops_cpu,
                target_crops,
            ) in iter_crop_batches(
                rgb,
                sar,
                target,
                coordinates,
                batch_size=args.feature_batch_size,
            ):
                crop_batches += 1
                crop_windows += len(batch_coordinates)
                rgb_crops = rgb_crops_cpu.to(device, non_blocking=True)
                sar_crops = sar_crops_cpu.to(device, non_blocking=True)
                backbone_outputs = model.extract_frozen_backbone_outputs(
                    rgb_crops,
                    sar_crops,
                )
                if cached_equivalence is None and not args.skip_cached_equivalence_check:
                    cached_equivalence = verify_cached_forward_equivalence(
                        model,
                        rgb_crops,
                        sar_crops,
                        backbone_outputs,
                    )

                for stage, rgb_tokens, sar_tokens in zip(
                    DINO_STAGE_ORDER,
                    backbone_outputs[0],
                    backbone_outputs[1],
                    strict=True,
                ):
                    statistics = feature_pair_batch_statistics(
                        rgb_tokens,
                        sar_tokens,
                        target_crops,
                        cka_max_points=args.cka_max_points,
                        cka_max_channels=args.cka_max_channels,
                    )
                    backbone_accumulator.update(
                        stage,
                        statistics,
                        city=city,
                        tile_id=tile_id,
                    )

                full_availability = canonical_availability(
                    "full",
                    batch_size=rgb_crops.shape[0],
                    device=device,
                )
                sar_availability = canonical_availability(
                    "sar",
                    batch_size=rgb_crops.shape[0],
                    device=device,
                )
                full_logits, full_stages, full_audit = recorder.run(
                    "full",
                    (0, 1),
                    lambda: model.forward_from_backbone_outputs(
                        rgb_crops,
                        sar_crops,
                        backbone_outputs=backbone_outputs,
                        availability=full_availability,
                    ),
                )
                sar_logits, sar_stages, sar_audit = recorder.run(
                    "sar",
                    (1,),
                    lambda: model.forward_from_backbone_outputs(
                        rgb_crops,
                        sar_crops,
                        backbone_outputs=backbone_outputs,
                        availability=sar_availability,
                    ),
                )
                for endpoint, audit in (
                    ("full", full_audit),
                    ("sar", sar_audit),
                ):
                    if endpoint not in hook_contract:
                        hook_contract[endpoint] = audit
                    elif hook_contract[endpoint] != audit:
                        raise RuntimeError(
                            f"{endpoint} hook contract changed between crop batches"
                        )

                for stage in STAGE_ORDER:
                    statistics = feature_pair_batch_statistics(
                        full_stages[stage],
                        sar_stages[stage],
                        target_crops,
                        cka_max_points=args.cka_max_points,
                        cka_max_channels=args.cka_max_channels,
                    )
                    feature_accumulator.update(
                        stage,
                        statistics,
                        city=city,
                        tile_id=tile_id,
                    )
                for endpoint, endpoint_stages in (
                    ("full", full_stages),
                    ("sar", sar_stages),
                ):
                    for stage, (
                        resized_coarse_stage,
                        lateral_stage,
                    ) in CROSS_SCALE_FEATURE_PAIRS.items():
                        statistics = feature_pair_batch_statistics(
                            endpoint_stages[resized_coarse_stage],
                            endpoint_stages[lateral_stage],
                            target_crops,
                            cka_max_points=args.cka_max_points,
                            cka_max_channels=args.cka_max_channels,
                        )
                        cross_scale_accumulators[endpoint].update(
                            stage,
                            statistics,
                            city=city,
                            tile_id=tile_id,
                        )
                stitcher.add(batch_coordinates, full_logits, sar_logits)

                del (
                    rgb_crops,
                    sar_crops,
                    backbone_outputs,
                    full_logits,
                    sar_logits,
                    full_stages,
                    sar_stages,
                )

            logits = stitcher.finalize()
            full_prediction = logits["full"].argmax(dim=1)
            sar_prediction = logits["sar"].argmax(dim=1)
            endpoint_metrics["full"].update(full_prediction, target)
            endpoint_metrics["sar"].update(sar_prediction, target)
            if city not in city_metrics:
                city_metrics[city] = {
                    endpoint: EarthMissMetrics() for endpoint in ("full", "sar")
                }
            city_metrics[city]["full"].update(full_prediction, target)
            city_metrics[city]["sar"].update(sar_prediction, target)
            tile_outcomes[tile_key] = segmentation_region_statistics(
                logits["full"],
                logits["sar"],
                target,
            )
            progress.set_postfix(tile=tile_key, crops=len(coordinates))

    buffer_audit = assert_buffers_unchanged(batchnorm_before, model)
    feature_summary = feature_accumulator.summary(
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    backbone_summary = backbone_accumulator.summary(
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    cross_scale_summaries = {
        endpoint: accumulator.summary(
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for endpoint, accumulator in cross_scale_accumulators.items()
    }
    cross_scale_alignment_degradation = (
        summarize_cross_scale_alignment_degradation(
            cross_scale_summaries["full"]["by_tile"],
            cross_scale_summaries["sar"]["by_tile"],
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
    )
    amplification = summarize_amplification(
        feature_summary["by_tile"],
        left_label="full",
        right_label="sar",
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    correlations = summarize_error_correlations(
        feature_summary["by_tile"],
        tile_outcomes,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    for summary in (
        feature_summary,
        backbone_summary,
        *cross_scale_summaries.values(),
    ):
        pair_labels = summary["pair_labels"]
        summary["by_tile"] = compact_by_tile_stage_summaries(
            summary["by_tile"],
            left_label=pair_labels["left"],
            right_label=pair_labels["right"],
        )
        summary["by_tile_format"] = "compact_scalar_evidence_v1"
    complete = processed_limit == expected_tiles and args.smoke_tiles == 0
    endpoint_metrics_summary = _metrics_summary(endpoint_metrics, city_metrics)
    if complete:
        for endpoint in ("full", "sar"):
            selection_class_ids = endpoint_metrics_summary["pooled"][endpoint][
                "selection_class_ids"
            ]
            if selection_class_ids != EXPECTED_VAL_SELECTION_CLASS_IDS:
                raise RuntimeError(
                    "EarthMiss Val class support changed for "
                    f"{endpoint}: expected {EXPECTED_VAL_SELECTION_CLASS_IDS}, "
                    f"got {selection_class_ids}"
                )
    report = {
        "schema": SCHEMA,
        "formal": complete,
        "training_was_performed": False,
        "scientific_logic_was_modified": False,
        "split": args.split,
        "repository_head": _git_head(),
        "checkpoint": checkpoint_record,
        "checkpoint_binding": {
            "expected": checkpoint_binding,
            "observed": observed_binding,
            "matched": True,
        },
        "dataset": {
            "expected_tiles": EXPECTED_VAL_TILES,
            "observed_manifest_tiles": expected_tiles,
            "processed_tiles": processed_limit,
            "processed_tile_keys": sorted(tile_outcomes),
            "expected_pooled_selection_class_ids": (
                EXPECTED_VAL_SELECTION_CLASS_IDS
            ),
            "rgb_normalization": {
                "policy": dataset.rgb_normalization,
                "mean": list(dataset.imagenet_mean),
                "std": list(dataset.imagenet_std),
            },
            "sar_normalization": {
                "policy": "earthmiss_metars_dataset_stats",
                "mean": list(dataset.sar_mean),
                "std": list(dataset.sar_std),
            },
        },
        "protocol": {
            "endpoint_order": ["full", "sar"],
            "availability": {
                "full": [True, True],
                "sar": [False, True],
            },
            "backbone_reuse": (
                "extract_frozen_backbone_outputs once, then paired "
                "forward_from_backbone_outputs"
            ),
            "cross_scale_alignment": (
                "within each endpoint, compare nearest-resized coarse PRN "
                "features with the same-grid lateral feature; the paired "
                "degradation estimand is SAR alignment minus Full alignment"
            ),
            "internal_feature_weighting": "crop_window_weighted_with_overlap",
            "by_tile_feature_storage": (
                "compact scalar evidence only; pooled/city scopes retain full "
                "aggregate summaries; no activation or prototype vectors stored"
            ),
            "whole_tile_logits": "released overlap-average reconstruction",
            "window_size": args.window_size,
            "stride": args.stride,
            "feature_batch_size": args.feature_batch_size,
            "crop_batches": crop_batches,
            "crop_windows": crop_windows,
            "cka": {
                "definition": "centered linear CKA after deterministic even subsampling",
                "max_points": args.cka_max_points,
                "max_channels": args.cka_max_channels,
            },
            "boundary": (
                "both valid native pixels adjacent to a 4-neighbour GT class "
                "change; feature-grid statistics use fixed native-pixel area "
                "weights whose regional totals are invariant to grid resolution"
            ),
            "frequency": (
                "descriptive orthonormal 2-D Haar energy on each representation "
                "grid; it is not evidence of cross-scale frequency preservation"
            ),
            "batchnorm": "raw online checkpoint buffers, model.eval(), no recalibration",
        },
        "runtime_audits": {
            "cached_vs_direct_first_batch": cached_equivalence,
            "cached_equivalence_check_skipped": args.skip_cached_equivalence_check,
            "hook_call_contract": hook_contract,
            "batchnorm_buffers": buffer_audit,
        },
        "endpoint_metrics": endpoint_metrics_summary,
        "tile_outcomes": dict(sorted(tile_outcomes.items())),
        "backbone_rgb_vs_sar": backbone_summary,
        "full_vs_sar_stages": feature_summary,
        "within_endpoint_cross_scale_alignment": cross_scale_summaries,
        "cross_scale_alignment_degradation": cross_scale_alignment_degradation,
        "gap_amplification_edges": amplification,
        "tile_level_error_correlations": correlations,
        "decision_support": {
            "status": "pending_manual_review",
            "hypothesis_test_scope": {
                "primary_tests": [
                    "adapter_stride2_P5",
                    "prn_cross_scale_P5_to_P4_alignment_degradation",
                ],
                "negative_controls": ["adapter_identity_P4_control"],
                "implementation_controls": ["prn_nearest_P5_to_P4"],
                "all_other_edges": "exploratory",
                "haar_interpretation": (
                    "representation-grid descriptive only; no cross-scale "
                    "frequency-preservation claim"
                ),
            },
            "primary_rejection_rule": (
                "Reject a primary mechanism unless its paired relative-"
                "degradation bootstrap supports the preregistered direction: "
                "positive after-minus-before Full-to-SAR boundary-gap change for "
                "adapter_stride2_P5, or worse SAR-minus-Full within-endpoint "
                "coarse/lateral alignment at prn.cross_scale.P5_to_P4 (positive "
                "distance/RMS or negative CKA). Absolute SAR error and Haar "
                "summaries are not primary tests."
            ),
            "adapter_rule": (
                "Only adapter_stride2_P5 can motivate FSD-Down-lite; P4 identity "
                "is the negative control. P4 and P5 are different DINO layers, "
                "so their raw features are never treated as a sequential downsample."
            ),
            "prn_rule": (
                "Only reproducibly worse SAR-versus-Full within-endpoint alignment "
                "between resized P5 coarse and same-grid P4 lateral features can "
                "motivate a single P5-to-P4 FAM-lite; nearest-resize gap "
                "preservation is an implementation control only."
            ),
            "no_module_rule": (
                "If neither condition holds, stop this structural line and retain "
                "Run C for the directed Full-to-SAR prototype-transfer line."
            ),
        },
    }
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if Path(args.output).exists():
        raise FileExistsError(
            f"refusing to overwrite diagnostic report: {args.output}"
        )
    report = diagnose(args)
    write_json_exclusive(args.output, report)
    full = report["endpoint_metrics"]["pooled"]["full"]
    sar = report["endpoint_metrics"]["pooled"]["sar"]
    print(
        "Full/SAR Val mIoU: "
        f"{full['mIoU'] * 100.0:.4f}/{sar['mIoU'] * 100.0:.4f}"
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
