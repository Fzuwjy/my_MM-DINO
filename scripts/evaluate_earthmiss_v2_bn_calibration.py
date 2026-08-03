"""Calibrate Decoder BatchNorm buffers for fixed EarthMiss checkpoints.

This is a post-hoc, Train-only diagnostic/deployment tool.  It never updates a
learned parameter and never selects a checkpoint from validation results.  A
fresh, identically seeded Train loader is used for every requested buffer bank,
after which the fixed checkpoint is evaluated on the three official Val cities
under original, wrong-state, and matched-state Decoder BN buffers.  The first
V2 matrix defaults to SAR/Full; RGB remains available as an explicit option.

The command is intentionally foreground-only.  It writes a small BN-buffer
bank next to a JSON report and refuses to overwrite either the source
checkpoint or an existing result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import EARTHMISS_CITIES, build_dataset  # noqa: E402
from models.MMDINO.availability import (  # noqa: E402
    AVAILABILITY_STATES,
    canonical_availability,
)
from models.MMDINO.dino_segment import build_model  # noqa: E402
from utils.earthmiss_metrics import EarthMissMetrics  # noqa: E402
from utils.inference import slide_inference  # noqa: E402

from scripts.evaluate_earthmiss_missing_v1 import (  # noqa: E402
    _checkpoint_uses_raw_logits,
)
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
    VAL_SELECTION_CLASS_IDS,
)


SCHEMA = "earthmiss_v2_decoder_bn_calibration_v1"
CALIBRATION_STATES = ("sar", "rgb", "full")
DEFAULT_CALIBRATION_STATES = ("sar", "full")
DEFAULT_ENDPOINTS = ("sar", "full")
OFFICIAL_VAL_CITIES = tuple(EARTHMISS_CITIES["val"])
CALIBRATION_BATCH_SIZE = 8
EXPECTED_TRAIN_TILES = 2641
EXPECTED_VAL_TILES = 277
EXPECTED_CALIBRATION_BATCHES = 330
EXPECTED_CALIBRATION_TILES = 2640
EXPECTED_DECODER_BN_COUNT = 40
CANONICAL_DECODER_SLOTS = 2
EXPECTED_BN_UPDATES_PER_MODULE = (
    EXPECTED_CALIBRATION_BATCHES * CANONICAL_DECODER_SLOTS
)

# The wrong-state cell is fixed before seeing results.  RGB uses the maximally
# disjoint SAR-only bank; the V2 primary matrix currently uses SAR and Full.
WRONG_BANK_BY_ENDPOINT = {
    "sar": "full",
    "full": "sar",
    "rgb": "sar",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output")
    parser.add_argument("--bank-output")
    parser.add_argument(
        "--calibration-states",
        nargs="+",
        choices=CALIBRATION_STATES,
        default=list(DEFAULT_CALIBRATION_STATES),
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        choices=CALIBRATION_STATES,
        default=list(DEFAULT_ENDPOINTS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--calibration-num-workers", type=int, default=4)
    parser.add_argument("--val-num-workers", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    return parser.parse_args()


def _hash_field(hasher: Any, value: str | bytes) -> None:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    hasher.update(len(payload).to_bytes(8, byteorder="little", signed=False))
    hasher.update(payload)


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def learned_parameter_sha256(model: nn.Module) -> str:
    """Hash parameter names, metadata, and values while excluding all buffers."""
    hasher = hashlib.sha256()
    _hash_field(hasher, "earthmiss-v2-learned-parameters-v1")
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        tensor = parameter.detach().to("cpu").contiguous()
        _hash_field(hasher, name)
        _hash_field(hasher, str(tensor.dtype))
        _hash_field(hasher, json.dumps(list(tensor.shape), separators=(",", ":")))
        _hash_field(hasher, "requires_grad=1" if parameter.requires_grad else "requires_grad=0")
        # Viewing as bytes also supports dtypes (for example bfloat16) that
        # NumPy cannot represent directly.
        _hash_field(
            hasher,
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
        )
    return hasher.hexdigest()


def bn_buffer_bank_sha256(
    buffers: Mapping[str, Mapping[str, torch.Tensor]],
) -> str:
    """Hash one named Decoder-BN buffer bank without serializing a model."""
    hasher = hashlib.sha256()
    _hash_field(hasher, "earthmiss-v2-decoder-bn-buffer-bank-v1")
    for module_name in sorted(buffers):
        _hash_field(hasher, module_name)
        for field in sorted(buffers[module_name]):
            tensor = torch.as_tensor(buffers[module_name][field]).detach().cpu().contiguous()
            _hash_field(hasher, field)
            _hash_field(hasher, str(tensor.dtype))
            _hash_field(
                hasher,
                json.dumps(list(tensor.shape), separators=(",", ":")),
            )
            _hash_field(
                hasher,
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
            )
    return hasher.hexdigest()


def build_deployment_composition(
    *,
    source_checkpoint_sha256: str,
    bn_bank_file_sha256: str,
    matched_sar_bank_sha256: str,
) -> dict[str, Any]:
    """Identify the deployable SAR artifact as weights plus matched BN buffers."""
    components = (
        source_checkpoint_sha256,
        bn_bank_file_sha256,
        matched_sar_bank_sha256,
        "sar",
    )
    if any(not isinstance(value, str) or not value for value in components):
        raise ValueError("Deployment composition hashes and bank entry must be non-empty")
    hasher = hashlib.sha256()
    _hash_field(hasher, "earthmiss-v2-sar-deployment-composition-v1")
    for value in components:
        _hash_field(hasher, value)
    return {
        "role": "primary_deployable_composite",
        "endpoint": "sar",
        "bn_condition": "matched_state",
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "bn_bank_file_sha256": bn_bank_file_sha256,
        "bn_bank_entry": "sar",
        "bn_bank_entry_sha256": matched_sar_bank_sha256,
        "composition_sha256": hasher.hexdigest(),
        "contains_full_model_copy": False,
        "requires_both_components": True,
    }


def decoder_batchnorm_modules(
    model: nn.Module,
    *,
    expected_count: int | None = None,
) -> OrderedDict[str, nn.BatchNorm2d]:
    """Return only standard ``BatchNorm2d`` modules inside ``model.decoder``."""
    decoder = getattr(model, "decoder", None)
    if not isinstance(decoder, nn.Module):
        raise ValueError("Model has no nn.Module decoder")

    modules: OrderedDict[str, nn.BatchNorm2d] = OrderedDict()
    for local_name, module in decoder.named_modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        qualified_name = f"decoder.{local_name}" if local_name else "decoder"
        if type(module) is not nn.BatchNorm2d:
            raise TypeError(
                "V2 calibration accepts only standard nn.BatchNorm2d inside "
                f"Decoder, got {type(module).__name__} at {qualified_name}"
            )
        if (
            not module.track_running_stats
            or module.running_mean is None
            or module.running_var is None
            or module.num_batches_tracked is None
        ):
            raise ValueError(f"Decoder BN does not track running stats: {qualified_name}")
        modules[qualified_name] = module

    if not modules:
        raise ValueError("Decoder contains no standard BatchNorm2d modules")
    if expected_count is not None and len(modules) != expected_count:
        raise ValueError(
            f"Expected {expected_count} Decoder BatchNorm2d modules, found {len(modules)}"
        )
    return modules


@torch.no_grad()
def snapshot_decoder_bn_buffers(
    model: nn.Module,
    *,
    expected_count: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for name, module in decoder_batchnorm_modules(
        model, expected_count=expected_count
    ).items():
        result[name] = {
            "running_mean": module.running_mean.detach().cpu().clone(),
            "running_var": module.running_var.detach().cpu().clone(),
            "num_batches_tracked": module.num_batches_tracked.detach()
            .cpu()
            .clone(),
        }
    return result


@torch.no_grad()
def load_decoder_bn_buffers(
    model: nn.Module,
    buffers: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    expected_parameter_sha256: str | None = None,
    expected_count: int | None = None,
) -> None:
    if expected_parameter_sha256 is not None:
        actual_hash = learned_parameter_sha256(model)
        if actual_hash != expected_parameter_sha256:
            raise ValueError("Learned-parameter hash differs from the BN bank")

    modules = decoder_batchnorm_modules(model, expected_count=expected_count)
    if set(buffers) != set(modules):
        missing = sorted(set(modules) - set(buffers))
        extra = sorted(set(buffers) - set(modules))
        raise ValueError(f"BN bank module mismatch: missing={missing}, extra={extra}")

    expected_fields = {"running_mean", "running_var", "num_batches_tracked"}
    for name, module in modules.items():
        values = buffers[name]
        if set(values) != expected_fields:
            raise ValueError(f"BN bank fields changed at {name}: {sorted(values)}")
        for field in sorted(expected_fields):
            destination = getattr(module, field)
            source = torch.as_tensor(values[field])
            if source.shape != destination.shape:
                raise ValueError(
                    f"BN bank shape changed at {name}.{field}: "
                    f"{tuple(source.shape)} vs {tuple(destination.shape)}"
                )
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


def seed_calibration_stream(seed: int) -> torch.Generator:
    """Reset the crop RNGs and return a fresh deterministic shuffle generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator().manual_seed(seed)


def build_calibration_loader(
    dataset: torch.utils.data.Dataset,
    *,
    seed: int,
    num_workers: int,
    batch_size: int = CALIBRATION_BATCH_SIZE,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("Calibration batch size must be positive and workers non-negative")
    generator = seed_calibration_stream(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # A state always receives a newly constructed loader and worker pool.
        persistent_workers=False,
        generator=generator,
    )


def _unpack_rgb_sar(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("EarthMiss calibration batches must contain RGB and SAR")
    rgb, sar = batch[0], batch[1]
    if not isinstance(rgb, torch.Tensor) or not isinstance(sar, torch.Tensor):
        raise TypeError("EarthMiss RGB and SAR batches must be tensors")
    if rgb.shape[0] != sar.shape[0]:
        raise ValueError("EarthMiss RGB and SAR batch sizes differ")
    return rgb, sar


@torch.no_grad()
def calibrate_decoder_bn(
    model: nn.Module,
    loader: Iterable[Any],
    *,
    state: str,
    device: torch.device | str,
    expected_batches: int | None = None,
    expected_tiles: int | None = None,
    expected_bn_count: int | None = None,
    expected_updates_per_bn: int | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Reset and cumulatively estimate Decoder BN buffers for one state."""
    if state not in CALIBRATION_STATES:
        raise ValueError(f"Unsupported calibration state: {state!r}")
    if expected_batches is not None and hasattr(loader, "__len__"):
        if len(loader) != expected_batches:  # type: ignore[arg-type]
            raise ValueError(
                f"Calibration loader has {len(loader)} batches, expected {expected_batches}"  # type: ignore[arg-type]
            )

    parameter_hash_before = learned_parameter_sha256(model)
    modules = decoder_batchnorm_modules(model, expected_count=expected_bn_count)
    original_momenta = {name: module.momentum for name, module in modules.items()}

    model.eval()
    for module in modules.values():
        module.reset_running_stats()
        module.momentum = None
        module.train()

    batches = 0
    tiles = 0
    iterator = tqdm(loader, desc=f"Calibrate BN {state}", leave=False) if show_progress else loader
    try:
        for batch in iterator:
            rgb, sar = _unpack_rgb_sar(batch)
            rgb = rgb.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            availability = canonical_availability(
                state, batch_size=rgb.shape[0], device=device
            )
            model(rgb, sar, availability=availability)
            batches += 1
            tiles += int(rgb.shape[0])
    finally:
        model.eval()
        for name, module in modules.items():
            module.momentum = original_momenta[name]

    if expected_batches is not None and batches != expected_batches:
        raise ValueError(f"Calibrated on {batches} batches, expected {expected_batches}")
    if expected_tiles is not None and tiles != expected_tiles:
        raise ValueError(f"Calibrated on {tiles} tiles, expected {expected_tiles}")

    parameter_hash_after = learned_parameter_sha256(model)
    if parameter_hash_after != parameter_hash_before:
        raise RuntimeError("A learned parameter changed during BN calibration")

    buffers = snapshot_decoder_bn_buffers(model, expected_count=expected_bn_count)
    update_counts = {
        name: int(values["num_batches_tracked"].item())
        for name, values in buffers.items()
    }
    unique_update_counts = set(update_counts.values())
    if expected_updates_per_bn is not None and unique_update_counts != {
        expected_updates_per_bn
    }:
        raise ValueError(
            "Decoder BN update counts differ from the canonical two-slot "
            f"contract: expected {expected_updates_per_bn}, got {update_counts}"
        )
    uniform_update_count = (
        next(iter(unique_update_counts)) if len(unique_update_counts) == 1 else None
    )

    return {
        "state": state,
        "logical_batches": batches,
        "tiles": tiles,
        "batch_size": getattr(loader, "batch_size", None),
        "drop_last": getattr(loader, "drop_last", None),
        "canonical_decoder_slots_per_logical_batch": CANONICAL_DECODER_SLOTS,
        "bn_updates_per_module": uniform_update_count,
        "learned_parameter_sha256": parameter_hash_after,
        "buffers": buffers,
    }


def build_evaluation_plan(
    endpoints: Sequence[str],
    available_banks: Iterable[str],
    *,
    checkpoint_epoch: int | None = None,
) -> list[dict[str, str]]:
    """Freeze original/wrong/matched cells and label the E15 primary cells."""
    bank_names = set(available_banks)
    if "original" not in bank_names:
        raise ValueError("Evaluation requires the original BN bank")

    rows = []
    for endpoint in endpoints:
        if endpoint not in CALIBRATION_STATES:
            raise ValueError(f"Unsupported endpoint: {endpoint!r}")
        wrong_bank = WRONG_BANK_BY_ENDPOINT[endpoint]
        for condition, bank in (
            ("original", "original"),
            ("wrong_state", wrong_bank),
            ("matched_state", endpoint),
        ):
            if bank not in bank_names:
                raise ValueError(
                    f"Endpoint {endpoint}/{condition} requires missing BN bank {bank!r}"
                )
            cell_role = "background"
            if checkpoint_epoch == 15 and condition == "matched_state":
                if endpoint == "sar":
                    cell_role = "primary"
                elif endpoint == "full":
                    cell_role = "paired_diagnostic"
            rows.append(
                {
                    "endpoint": endpoint,
                    "condition": condition,
                    "bn_bank": bank,
                    "cell_role": cell_role,
                }
            )
    return rows


def source_checkpoint_selection_metadata(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve source selection facts without inferring how it was produced."""
    return {
        "source_checkpoint_role": checkpoint.get("checkpoint_role"),
        "source_selection_state": checkpoint.get("selection_state"),
        "source_selection_metric": checkpoint.get("selection_metric"),
        "source_selection_score": checkpoint.get("selection_score"),
        "source_fixed_epoch_no_selection": checkpoint.get(
            "fixed_epoch_no_selection"
        ),
    }


def _slide_endpoint_logits(
    model: nn.Module,
    rgb: torch.Tensor,
    sar: torch.Tensor,
    state: str,
    *,
    device: torch.device | str,
    window_size: int,
    inference_batch_size: int,
) -> torch.Tensor:
    availability = canonical_availability(state, batch_size=1, device=device)
    stride = int(window_size * 2 / 3)
    return slide_inference(
        rgb,
        model,
        n_output_channels=8,
        crop_size=(window_size, window_size),
        stride=(stride, stride),
        dsm=sar,
        availability=availability,
        batch_size=inference_batch_size,
    )


@torch.no_grad()
def evaluate_val_by_city(
    model: nn.Module,
    loader: Iterable[Any],
    city_by_index: Sequence[str],
    *,
    state: str,
    device: torch.device | str,
    window_size: int,
    inference_batch_size: int,
    expected_cities: Sequence[str] = OFFICIAL_VAL_CITIES,
    expected_pooled_class_ids: Sequence[int] | None = VAL_SELECTION_CLASS_IDS,
    infer_fn: Callable[..., torch.Tensor] = _slide_endpoint_logits,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Return pooled and official-city confusion/mIoU for one fixed endpoint."""
    if state not in CALIBRATION_STATES:
        raise ValueError(f"Unsupported endpoint state: {state!r}")
    if getattr(loader, "batch_size", 1) != 1:
        raise ValueError("Val city accounting requires batch_size=1 and shuffle=False")
    if isinstance(loader, torch.utils.data.DataLoader) and not isinstance(
        loader.sampler, torch.utils.data.SequentialSampler
    ):
        raise ValueError("Val city accounting requires a SequentialSampler")

    observed_cities = tuple(OrderedDict.fromkeys(city_by_index))
    if observed_cities != tuple(expected_cities):
        raise ValueError(
            f"EarthMiss Val cities changed: expected {tuple(expected_cities)}, "
            f"got {observed_cities}"
        )

    model.eval()
    pooled = EarthMissMetrics()
    city_metrics = {city: EarthMissMetrics() for city in expected_cities}
    city_tiles = {city: 0 for city in expected_cities}
    iterator = tqdm(loader, desc=f"Val {state}", leave=False) if show_progress else loader
    sample_count = 0
    for sample_index, batch in enumerate(iterator):
        if sample_index >= len(city_by_index):
            raise ValueError("Val loader yielded more samples than its city manifest")
        if not isinstance(batch, (tuple, list)) or len(batch) < 3:
            raise ValueError("EarthMiss Val batches must contain RGB, SAR, and label")
        rgb, sar, label = batch[0], batch[1], batch[2]
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        logits = infer_fn(
            model,
            rgb,
            sar,
            state,
            device=device,
            window_size=window_size,
            inference_batch_size=inference_batch_size,
        )
        prediction = logits.argmax(dim=1)
        city = city_by_index[sample_index]
        pooled.update(prediction, label)
        city_metrics[city].update(prediction, label)
        city_tiles[city] += 1
        sample_count += 1

    if sample_count != len(city_by_index):
        raise ValueError(
            f"Val loader yielded {sample_count} samples for {len(city_by_index)} manifest rows"
        )

    pooled_result = pooled.compute()
    if (
        expected_pooled_class_ids is not None
        and pooled_result["selection_class_ids"]
        != list(expected_pooled_class_ids)
    ):
        raise ValueError(
            "EarthMiss pooled Val class support changed: expected "
            f"{list(expected_pooled_class_ids)}, got "
            f"{pooled_result['selection_class_ids']}"
        )

    return {
        "tiles": sample_count,
        "pooled": pooled_result,
        "cities": {
            city: {
                "tiles": city_tiles[city],
                **city_metrics[city].compute(),
            }
            for city in expected_cities
        },
    }


def _default_paths(checkpoint_path: Path) -> tuple[Path, Path]:
    stem = f"{checkpoint_path.stem}.earthmiss-v2-bn-calibration"
    return (
        checkpoint_path.with_name(f"{stem}.json"),
        checkpoint_path.with_name(f"{stem}.pt"),
    )


def validate_output_paths(
    checkpoint_path: Path,
    output_path: Path,
    bank_path: Path,
) -> None:
    resolved = {
        "checkpoint": checkpoint_path.resolve(),
        "output": output_path.resolve(),
        "bank": bank_path.resolve(),
    }
    if len(set(resolved.values())) != 3:
        raise ValueError("Checkpoint, JSON output, and BN bank must be different files")
    for path in (output_path, bank_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing V2 artifact: {path}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _build_datasets(args: Any):
    common = {
        "dataset_root": args.dataset_root,
        "window_size": (args.window_size, args.window_size),
        "model_name": "DINOv3",
        "modality": "multi",
        "backbone_type": "dinov3_vits16",
    }
    return (
        build_dataset("EarthMiss", "train", **common),
        build_dataset("EarthMiss", "val", **common),
    )


def main():
    args = parse_args()
    if args.window_size <= 0 or args.window_size % 16:
        raise ValueError("--window-size must be a positive multiple of 16")
    if (
        args.calibration_num_workers < 0
        or args.val_num_workers < 0
        or args.inference_batch_size <= 0
    ):
        raise ValueError("Worker counts must be non-negative and batch size positive")
    if len(set(args.calibration_states)) != len(args.calibration_states):
        raise ValueError("--calibration-states contains duplicates")
    if len(set(args.endpoints)) != len(args.endpoints):
        raise ValueError("--endpoints contains duplicates")
    # Fail before loading the model or scanning data if a requested endpoint
    # lacks its pre-registered matched/wrong-state banks.
    build_evaluation_plan(
        args.endpoints,
        ("original", *args.calibration_states),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("EarthMiss V2 BN calibration requires a CUDA GPU")

    checkpoint_path = Path(args.checkpoint)
    weights_path = Path(args.backbone_weights)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
    default_output, default_bank = _default_paths(checkpoint_path)
    output_path = Path(args.output) if args.output else default_output
    bank_path = Path(args.bank_output) if args.bank_output else default_bank
    validate_output_paths(checkpoint_path, output_path, bank_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint:
        raise ValueError("Checkpoint lacks a model state_dict")
    checkpoint_epoch = checkpoint.get("epoch")
    if checkpoint_epoch is not None:
        checkpoint_epoch = int(checkpoint_epoch)

    device = torch.device("cuda")
    model = build_model(
        model_name="DINOv3",
        backbone_weights=str(weights_path),
        backbone_type="dinov3_vits16",
        freeze_backbone=True,
        n_classes=8,
        use_lora=False,
        r=3,
        num_modalities=2,
        raw_logits=_checkpoint_uses_raw_logits(checkpoint),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    decoder_modules = decoder_batchnorm_modules(
        model, expected_count=EXPECTED_DECODER_BN_COUNT
    )
    parameter_hash = learned_parameter_sha256(model)

    train_dataset, val_dataset = _build_datasets(args)
    if len(train_dataset) != EXPECTED_TRAIN_TILES:
        raise ValueError(
            f"EarthMiss Train manifest changed: {len(train_dataset)} vs {EXPECTED_TRAIN_TILES}"
        )
    if len(val_dataset) != EXPECTED_VAL_TILES:
        raise ValueError(
            f"EarthMiss Val manifest changed: {len(val_dataset)} vs {EXPECTED_VAL_TILES}"
        )
    val_cities = [sample.city for sample in val_dataset.samples]
    if tuple(OrderedDict.fromkeys(val_cities)) != OFFICIAL_VAL_CITIES:
        raise ValueError("EarthMiss Val city order changed")

    original_buffers = snapshot_decoder_bn_buffers(
        model, expected_count=EXPECTED_DECODER_BN_COUNT
    )
    banks: dict[str, dict[str, dict[str, torch.Tensor]]] = {
        "original": original_buffers
    }
    calibration_records: dict[str, dict[str, Any]] = {}
    for state in args.calibration_states:
        load_decoder_bn_buffers(
            model,
            original_buffers,
            expected_parameter_sha256=parameter_hash,
            expected_count=EXPECTED_DECODER_BN_COUNT,
        )
        calibration_loader = build_calibration_loader(
            train_dataset,
            seed=args.seed,
            num_workers=args.calibration_num_workers,
            pin_memory=True,
        )
        calibrated = calibrate_decoder_bn(
            model,
            calibration_loader,
            state=state,
            device=device,
            expected_batches=EXPECTED_CALIBRATION_BATCHES,
            expected_tiles=EXPECTED_CALIBRATION_TILES,
            expected_bn_count=EXPECTED_DECODER_BN_COUNT,
            expected_updates_per_bn=EXPECTED_BN_UPDATES_PER_MODULE,
            show_progress=True,
        )
        banks[state] = calibrated.pop("buffers")
        calibration_records[state] = calibrated

    if learned_parameter_sha256(model) != parameter_hash:
        raise RuntimeError("Learned parameters changed while constructing BN banks")

    evaluation_plan = build_evaluation_plan(
        args.endpoints,
        banks,
        checkpoint_epoch=checkpoint_epoch,
    )
    checkpoint_sha256 = file_sha256(checkpoint_path)
    bank_entry_sha256 = {
        name: bn_buffer_bank_sha256(buffers) for name, buffers in banks.items()
    }
    primary_sar_composite_expected = any(
        row["endpoint"] == "sar"
        and row["condition"] == "matched_state"
        and row["cell_role"] == "primary"
        for row in evaluation_plan
    )
    bank_bundle = {
        "schema": SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "learned_parameter_sha256": parameter_hash,
        "decoder_bn_names": list(decoder_modules),
        "bank_entry_sha256": bank_entry_sha256,
        "artifact_contract": {
            "role": "bn_buffer_bank_component_requires_source_weights",
            "source_checkpoint_role": checkpoint.get("checkpoint_role"),
            "source_checkpoint_sha256": checkpoint_sha256,
            "matched_sar_bank_entry": "sar",
            "matched_sar_bank_entry_sha256": bank_entry_sha256["sar"],
            "primary_sar_composite_expected": primary_sar_composite_expected,
            "contains_full_model_copy": False,
            "contains_learned_parameters": False,
        },
        "banks": banks,
        "calibration": calibration_records,
    }
    torch.save(bank_bundle, bank_path)
    bank_file_sha256 = file_sha256(bank_path)
    deployment_composition = (
        build_deployment_composition(
            source_checkpoint_sha256=checkpoint_sha256,
            bn_bank_file_sha256=bank_file_sha256,
            matched_sar_bank_sha256=bank_entry_sha256["sar"],
        )
        if primary_sar_composite_expected
        else None
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.val_num_workers,
        pin_memory=True,
        persistent_workers=args.val_num_workers > 0,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": bank_bundle["checkpoint_sha256"],
            "run": checkpoint.get("run"),
            "seed": checkpoint.get("seed"),
            "epoch": checkpoint_epoch,
            "input_checkpoint_is_fixed": True,
            "posthoc_reselection": False,
            **source_checkpoint_selection_metadata(checkpoint),
        },
        "bn_bank": {
            "path": str(bank_path),
            "sha256": bank_file_sha256,
            "entry_sha256": bank_entry_sha256,
            "learned_parameter_sha256": parameter_hash,
            "decoder_scope_only": True,
            "decoder_batchnorm2d_count": len(decoder_modules),
            "contains_full_model_copy": False,
        },
        "deployment_composition": deployment_composition,
        "protocol": {
            "checkpoint_policy": "fixed_input_checkpoint_no_posthoc_reselection",
            "calibration_split": "official_train_only",
            "supported_calibration_states": list(CALIBRATION_STATES),
            "requested_calibration_states": list(args.calibration_states),
            "default_v2_matrix_states": list(DEFAULT_CALIBRATION_STATES),
            "rgb_bank_is_explicit_nondefault_background": True,
            "calibration_batch_size": CALIBRATION_BATCH_SIZE,
            "calibration_shuffle": True,
            "calibration_drop_last": True,
            "calibration_batches": EXPECTED_CALIBRATION_BATCHES,
            "calibration_tiles": EXPECTED_CALIBRATION_TILES,
            "canonical_decoder_slots_per_logical_batch": CANONICAL_DECODER_SLOTS,
            "decoder_bn_updates_per_module": EXPECTED_BN_UPDATES_PER_MODULE,
            "decoder_bn_update_note": (
                "Each canonical state supplies two equal adapter output slots to "
                "the shared FRM, so every FRM BN is called twice per logical batch"
            ),
            "calibration_seed_rebuilt_per_state": args.seed,
            "model_mode": "eval_except_target_decoder_bn",
            "bn_reset_running_stats": True,
            "bn_momentum": None,
            "optimizer": None,
            "val_cities": list(OFFICIAL_VAL_CITIES),
            "expected_val_tiles": EXPECTED_VAL_TILES,
            "selection_support": "pooled_gt_present",
            "expected_pooled_selection_class_ids": VAL_SELECTION_CLASS_IDS,
            "primary_cell": "E15 SAR endpoint + Train-SAR matched BN",
            "paired_diagnostic_cell": "E15 Full endpoint + Train-Full matched BN",
            "other_cells": "background",
        },
        "calibration": calibration_records,
        "evaluation_plan": evaluation_plan,
        "evaluations": {},
    }
    _write_json(output_path, result)

    for row in evaluation_plan:
        load_decoder_bn_buffers(
            model,
            banks[row["bn_bank"]],
            expected_parameter_sha256=parameter_hash,
            expected_count=EXPECTED_DECODER_BN_COUNT,
        )
        metrics = evaluate_val_by_city(
            model,
            val_loader,
            val_cities,
            state=row["endpoint"],
            device=device,
            window_size=args.window_size,
            inference_batch_size=args.inference_batch_size,
            show_progress=True,
        )
        endpoint_results = result["evaluations"].setdefault(row["endpoint"], {})
        cell_result = {
            "bn_bank": row["bn_bank"],
            "cell_role": row["cell_role"],
            **metrics,
        }
        if (
            row["endpoint"] == "sar"
            and row["condition"] == "matched_state"
            and row["cell_role"] == "primary"
        ):
            cell_result["deployment_composition"] = deployment_composition
        endpoint_results[row["condition"]] = cell_result
        _write_json(output_path, result)
        print(
            f"{row['endpoint']}/{row['condition']} ({row['bn_bank']}): "
            f"mIoU={metrics['pooled']['mIoU']:.6f}"
        )

    load_decoder_bn_buffers(
        model,
        original_buffers,
        expected_parameter_sha256=parameter_hash,
        expected_count=EXPECTED_DECODER_BN_COUNT,
    )
    print(f"saved report: {output_path}")
    print(f"saved BN bank: {bank_path}")


if __name__ == "__main__":
    main()
