"""Test whether the useful Full-to-SAR FRM-P2 correction is SAR-predictable.

The Run C checkpoint is frozen.  A fixed affine ridge probe maps canonical SAR
post-FRM P2 cells to the paired Full-minus-SAR P2 correction.  Seven Train
cities are evaluated by leave-one-city-out: six cities fit the probe and the
held-out city is decoded after injecting the predicted correction.  A global
mean correction and an equal-capacity ridge fit to within-class shuffled
targets distinguish sample-predictable signal from marginal or class-level
regularities.  MM-DINO is never updated and Val/Test are never accessed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import random
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

from datasets import EARTHMISS_CITIES, build_dataset  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from utils.earthmiss_metrics import EarthMissMetrics  # noqa: E402

from scripts.diagnose_earthmiss_missing_v3 import load_frozen_model  # noqa: E402
from scripts.diagnose_earthmiss_scale_transition import (  # noqa: E402
    EXPECTED_RUN_C_CHECKPOINT_SHA256,
    EXPECTED_RUN_C_EPOCH,
    EXPECTED_RUN_C_SEED,
    assert_buffers_unchanged,
    iter_crop_batches,
    sliding_window_coordinates,
    snapshot_batchnorm_buffers,
    verify_cached_forward_equivalence,
)
from scripts.earthmiss_causal_diagnostics_common import (  # noqa: E402
    NUM_CLASSES,
    VariantLogitStitcher,
    exact_class_occupancy,
    stable_seed,
)
from scripts.earthmiss_scale_transition_common import write_json_exclusive  # noqa: E402
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
)


SCHEMA = "earthmiss_frmp2_correction_recoverability_v1"
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/"
    "run-c-e15-train-frmp2-correction-recoverability.json"
)
EXPECTED_TRAIN_TILES = 2641
WINDOW_SIZE = 512
PROBE_STRIDE = 512
TRAIN_TILES_PER_CITY = 16
PURITY_THRESHOLD = 0.75
CELLS_PER_CLASS_PER_CROP = 64
MAX_CELLS_PER_CLASS_PER_CITY = 2048
RIDGE_ALPHA = 1e-2
CORRECTION_ALPHA = 0.25
SEED = 20260817
VARIANTS = (
    "full",
    "sar",
    "oracle_alpha_0.25",
    "ridge_alpha_0.25",
    "shuffled_ridge_alpha_0.25",
    "mean_alpha_0.25",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--tiles-per-city", type=int, default=TRAIN_TILES_PER_CITY)
    parser.add_argument(
        "--cells-per-class-crop",
        type=int,
        default=CELLS_PER_CLASS_PER_CROP,
    )
    parser.add_argument(
        "--max-cells-per-class-city",
        type=int,
        default=MAX_CELLS_PER_CLASS_PER_CITY,
    )
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--correction-alpha", type=float, default=CORRECTION_ALPHA)
    parser.add_argument("--purity-threshold", type=float, default=PURITY_THRESHOLD)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-cached-equivalence-check", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.tiles_per_city <= 0:
        raise ValueError("--tiles-per-city must be positive")
    if args.cells_per_class_crop <= 0:
        raise ValueError("--cells-per-class-crop must be positive")
    if args.max_cells_per_class_city <= 0:
        raise ValueError("--max-cells-per-class-city must be positive")
    if args.ridge_alpha <= 0.0:
        raise ValueError("--ridge-alpha must be positive")
    if not 0.0 < args.correction_alpha <= 1.0:
        raise ValueError("--correction-alpha must be in (0,1]")
    if not 0.5 < args.purity_threshold <= 1.0:
        raise ValueError("--purity-threshold must be in (0.5,1]")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    locked = {
        "ridge_alpha": RIDGE_ALPHA,
        "correction_alpha": CORRECTION_ALPHA,
        "purity_threshold": PURITY_THRESHOLD,
        "cells_per_class_crop": CELLS_PER_CLASS_PER_CROP,
        "max_cells_per_class_city": MAX_CELLS_PER_CLASS_PER_CITY,
        "seed": SEED,
    }
    observed = {name: getattr(args, name) for name in locked}
    if observed != locked:
        raise ValueError(
            "FRM P2 scientific parameters are frozen; only --tiles-per-city, "
            "--num-workers, paths, and the equivalence smoke flag may change: "
            f"{observed} != {locked}"
        )


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


def _assert_git_head_unchanged(start_head: str | None) -> str:
    end_head = _git_head()
    if start_head is None or end_head is None:
        raise RuntimeError("FRM-P2 diagnostic requires a readable Git HEAD")
    if end_head != start_head:
        raise RuntimeError(
            "repository HEAD changed during FRM-P2 diagnosis: "
            f"{start_head} -> {end_head}; refusing to write a report"
        )
    return start_head


class _IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        rgb, sar, target = self.dataset[source_index]
        return source_index, rgb, sar, target


class FRMP2Capture:
    """Capture the first of two equal canonical post-FRM P2 tensors."""

    def __init__(self, model) -> None:
        self.model = model
        self.handle = None
        self.active = False
        self.calls = 0
        self.value: torch.Tensor | None = None

    def __enter__(self):
        self.handle = self.model.decoder.frm.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, module, inputs, output):
        if not self.active:
            return None
        self.calls += 1
        if not isinstance(output, (list, tuple)) or len(output) != 4:
            raise RuntimeError("FRM must return P2-P5")
        value = output[0].detach().clone()
        if self.calls == 1:
            self.value = value
        elif self.calls == 2:
            if not torch.equal(self.value, value):
                raise RuntimeError("duplicate canonical FRM P2 slots differ")
        else:
            raise RuntimeError("canonical FRM called more than twice")
        return None

    def run(self, forward):
        self.calls = 0
        self.value = None
        self.active = True
        try:
            logits = forward()
        finally:
            self.active = False
        if self.calls != 2 or self.value is None:
            raise RuntimeError("FRM P2 capture call contract changed")
        return logits, self.value


class FRMP2AdditiveIntervention:
    """Add one precomputed correction to both equal canonical FRM-P2 slots."""

    def __init__(self, model) -> None:
        self.model = model
        self.handle = None
        self.active = False
        self.calls = 0
        self.correction: torch.Tensor | None = None

    def __enter__(self):
        self.handle = self.model.decoder.frm.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        self.correction = None
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, module, inputs, output):
        if not self.active:
            return None
        self.calls += 1
        if not isinstance(output, (list, tuple)) or len(output) != 4:
            raise RuntimeError("FRM must return P2-P5")
        if self.correction is None or output[0].shape != self.correction.shape:
            raise RuntimeError("FRM P2 correction shape mismatch")
        values = list(output)
        values[0] = values[0] + self.correction
        return tuple(values) if isinstance(output, tuple) else values

    def run(self, forward, correction: torch.Tensor):
        if not torch.isfinite(correction).all():
            raise RuntimeError("FRM P2 correction contains non-finite values")
        self.calls = 0
        self.correction = correction.detach()
        self.active = True
        try:
            logits = forward()
        finally:
            self.active = False
            self.correction = None
        if self.calls != 2:
            raise RuntimeError("FRM P2 intervention call contract changed")
        return logits


@dataclass(frozen=True)
class CitySamples:
    sar: torch.Tensor
    delta: torch.Tensor
    class_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.sar.ndim != 2 or self.delta.shape != self.sar.shape:
            raise ValueError("city samples must contain equal [N,C] tensors")
        if self.class_ids.shape != (self.sar.shape[0],):
            raise ValueError("city class_ids must have shape [N]")


@dataclass(frozen=True)
class CityMoments:
    n: int
    sum_x: torch.Tensor
    sum_y: torch.Tensor
    xtx: torch.Tensor
    xty: torch.Tensor
    xty_shuffled: torch.Tensor


@dataclass(frozen=True)
class AffineRidgeMap:
    mean_x: torch.Tensor
    scale_x: torch.Tensor
    weight: torch.Tensor
    mean_y: torch.Tensor

    def predict_rows(self, rows: torch.Tensor) -> torch.Tensor:
        dtype = rows.dtype
        device = rows.device
        mean_x = self.mean_x.to(device=device, dtype=dtype)
        scale_x = self.scale_x.to(device=device, dtype=dtype)
        weight = self.weight.to(device=device, dtype=dtype)
        mean_y = self.mean_y.to(device=device, dtype=dtype)
        return ((rows - mean_x) / scale_x) @ weight + mean_y

    def predict_bchw(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError("FRM P2 feature must be BCHW")
        rows = feature.permute(0, 2, 3, 1)
        prediction = self.predict_rows(rows)
        return prediction.permute(0, 3, 1, 2).contiguous()


def _build_dataset(args: argparse.Namespace):
    return build_dataset(
        "EarthMiss",
        "train",
        dataset_root=args.dataset_root,
        window_size=(WINDOW_SIZE, WINDOW_SIZE),
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
        apply_train_transform=False,
        cache_size=0,
    )


def _selected_indices(dataset, per_city: int, seed: int) -> dict[str, tuple[int, ...]]:
    by_city: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        by_city[str(sample.city)].append(index)
    expected = tuple(EARTHMISS_CITIES["train"])
    if set(by_city) != set(expected):
        raise RuntimeError("EarthMiss Train city manifest changed")
    result = {}
    for city in expected:
        indices = list(by_city[city])
        random.Random(stable_seed(seed, city)).shuffle(indices)
        if len(indices) < per_city:
            raise RuntimeError(f"city {city} has fewer than {per_city} tiles")
        result[city] = tuple(indices[:per_city])
    return result


def sample_pure_p2_cells(
    sar_p2: torch.Tensor,
    full_p2: torch.Tensor,
    target: torch.Tensor,
    *,
    purity_threshold: float,
    maximum_per_class: int,
    seed: int,
) -> CitySamples:
    if sar_p2.ndim != 4 or full_p2.shape != sar_p2.shape:
        raise ValueError("Full/SAR P2 tensors must have equal BCHW shapes")
    occupancy = exact_class_occupancy(target, tuple(sar_p2.shape[-2:]))
    purity, class_ids = occupancy.max(dim=1)
    valid = purity >= purity_threshold
    sar_rows = sar_p2.permute(0, 2, 3, 1).reshape(-1, sar_p2.shape[1])
    delta_rows = (full_p2 - sar_p2).permute(0, 2, 3, 1).reshape(
        -1, sar_p2.shape[1]
    )
    flat_classes = class_ids.reshape(-1)
    flat_valid = valid.reshape(-1)
    selected = []
    for class_id in range(NUM_CLASSES):
        indices = torch.nonzero(
            flat_valid & (flat_classes == class_id), as_tuple=False
        ).flatten().cpu()
        if indices.numel() > maximum_per_class:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(stable_seed(seed, f"class-{class_id}"))
            indices = indices[
                torch.randperm(indices.numel(), generator=generator)[:maximum_per_class]
            ]
        if indices.numel():
            selected.append(indices)
    if not selected:
        channels = sar_p2.shape[1]
        return CitySamples(
            sar=torch.empty(0, channels, dtype=torch.float32),
            delta=torch.empty(0, channels, dtype=torch.float32),
            class_ids=torch.empty(0, dtype=torch.int64),
        )
    indices = torch.cat(selected)
    device_indices = indices.to(sar_rows.device)
    return CitySamples(
        sar=sar_rows.index_select(0, device_indices).detach().cpu().float(),
        delta=delta_rows.index_select(0, device_indices).detach().cpu().float(),
        class_ids=flat_classes.index_select(0, device_indices).detach().cpu().long(),
    )


def _cap_city_samples(
    chunks: Sequence[CitySamples], maximum_per_class: int, seed: int
) -> CitySamples:
    if not chunks:
        raise RuntimeError("city produced no FRM P2 samples")
    sar = torch.cat([chunk.sar for chunk in chunks])
    delta = torch.cat([chunk.delta for chunk in chunks])
    class_ids = torch.cat([chunk.class_ids for chunk in chunks])
    selected = []
    for class_id in range(NUM_CLASSES):
        indices = torch.nonzero(class_ids == class_id, as_tuple=False).flatten()
        if indices.numel() > maximum_per_class:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(stable_seed(seed, f"cap-class-{class_id}"))
            indices = indices[
                torch.randperm(indices.numel(), generator=generator)[:maximum_per_class]
            ]
        if indices.numel():
            selected.append(indices)
    if not selected:
        raise RuntimeError("city cap removed all FRM P2 samples")
    indices = torch.cat(selected)
    return CitySamples(sar[indices], delta[indices], class_ids[indices])


def within_class_permutation(class_ids: torch.Tensor, seed: int) -> torch.Tensor:
    class_ids = class_ids.cpu().long()
    permutation = torch.arange(class_ids.numel())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for class_id in range(NUM_CLASSES):
        indices = torch.nonzero(class_ids == class_id, as_tuple=False).flatten()
        if indices.numel() > 1:
            permutation[indices] = indices[
                torch.randperm(indices.numel(), generator=generator)
            ]
    return permutation


@torch.inference_mode()
def city_moments(
    samples: CitySamples,
    device: torch.device,
    seed: int,
) -> CityMoments:
    x = samples.sar.float().to(device)
    y = samples.delta.float().to(device)
    permutation = within_class_permutation(samples.class_ids, seed).to(device)
    shuffled_y = y.index_select(0, permutation)
    return CityMoments(
        n=x.shape[0],
        sum_x=x.sum(0, dtype=torch.float64).cpu(),
        sum_y=y.sum(0, dtype=torch.float64).cpu(),
        xtx=(x.T @ x).double().cpu(),
        xty=(x.T @ y).double().cpu(),
        xty_shuffled=(x.T @ shuffled_y).double().cpu(),
    )


def combine_moments(
    moments: Mapping[str, CityMoments], excluded_city: str
) -> CityMoments:
    selected = [value for city, value in moments.items() if city != excluded_city]
    if not selected:
        raise ValueError("leave-one-city-out requires training cities")
    return CityMoments(
        n=sum(value.n for value in selected),
        sum_x=sum((value.sum_x for value in selected), torch.zeros_like(selected[0].sum_x)),
        sum_y=sum((value.sum_y for value in selected), torch.zeros_like(selected[0].sum_y)),
        xtx=sum((value.xtx for value in selected), torch.zeros_like(selected[0].xtx)),
        xty=sum((value.xty for value in selected), torch.zeros_like(selected[0].xty)),
        xty_shuffled=sum(
            (value.xty_shuffled for value in selected),
            torch.zeros_like(selected[0].xty_shuffled),
        ),
    )


def fit_affine_ridge(
    moments: CityMoments,
    ridge_alpha: float,
    *,
    shuffled: bool = False,
    mean_only: bool = False,
) -> AffineRidgeMap:
    if moments.n <= 1 or ridge_alpha <= 0.0:
        raise ValueError("ridge fit requires multiple samples and positive alpha")
    n = float(moments.n)
    mean_x = moments.sum_x / n
    mean_y = moments.sum_y / n
    covariance_xx = moments.xtx / n - torch.outer(mean_x, mean_x)
    cross = moments.xty_shuffled if shuffled else moments.xty
    covariance_xy = cross / n - torch.outer(mean_x, mean_y)
    variance = covariance_xx.diag().clamp_min(1e-8)
    scale_x = variance.sqrt()
    if mean_only:
        weight = torch.zeros(
            covariance_xy.shape, dtype=torch.float64, device=covariance_xy.device
        )
    else:
        normalized_xx = covariance_xx / torch.outer(scale_x, scale_x)
        normalized_xx = 0.5 * (normalized_xx + normalized_xx.T)
        normalized_xy = covariance_xy / scale_x[:, None]
        identity = torch.eye(normalized_xx.shape[0], dtype=torch.float64)
        weight = torch.linalg.solve(
            normalized_xx + ridge_alpha * identity,
            normalized_xy,
        )
    return AffineRidgeMap(
        mean_x=mean_x.float(),
        scale_x=scale_x.float(),
        weight=weight.float(),
        mean_y=mean_y.float(),
    )


def correction_prediction_metrics(
    probe: AffineRidgeMap, samples: CitySamples
) -> dict[str, Any]:
    x = samples.sar.float()
    target = samples.delta.float()
    prediction = probe.predict_rows(x)
    residual_energy = float((prediction - target).square().sum(dtype=torch.float64))
    target_energy = float(target.square().sum(dtype=torch.float64))
    prediction_energy = float(prediction.square().sum(dtype=torch.float64))
    dot = (prediction * target).sum(dim=1, dtype=torch.float64)
    denominator = prediction.norm(dim=1).double() * target.norm(dim=1).double()
    valid = denominator > 1e-12
    per_cell_cosine = dot[valid] / denominator[valid]
    pooled_denominator = (prediction_energy * target_energy) ** 0.5
    return {
        "n": int(x.shape[0]),
        "relative_mse_to_zero": (
            residual_energy / target_energy if target_energy > 0.0 else None
        ),
        "pooled_cosine": (
            float((prediction * target).sum(dtype=torch.float64)) / pooled_denominator
            if pooled_denominator > 0.0
            else None
        ),
        "mean_cell_cosine": (
            float(per_cell_cosine.mean()) if per_cell_cosine.numel() else None
        ),
        "predicted_to_target_rms_ratio": (
            (prediction_energy / target_energy) ** 0.5 if target_energy > 0.0 else None
        ),
    }


@torch.inference_mode()
def _extract_city_samples(
    model,
    capture: FRMP2Capture,
    dataset,
    selected: Mapping[str, Sequence[int]],
    args,
    device,
    equivalence_holder,
):
    full_availability = canonical_availability("full", batch_size=1, device=device)
    sar_availability = canonical_availability("sar", batch_size=1, device=device)
    samples_by_city = {}
    metadata = {}
    for city in EARTHMISS_CITIES["train"]:
        subset = _IndexedSubset(dataset, selected[city])
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        chunks = []
        crop_windows = 0
        tile_keys = []
        for source_index, rgb, sar, target in tqdm(
            loader, desc=f"frmp2-extract-{city}"
        ):
            index = int(source_index.item())
            sample = dataset.samples[index]
            tile_keys.append(f"{sample.city}/{sample.tile_id}")
            height, width = target.shape[-2:]
            if height % WINDOW_SIZE or width % WINDOW_SIZE:
                raise RuntimeError("FRM P2 probe requires 512-divisible native tiles")
            coordinates = sliding_window_coordinates(
                height, width, window_size=WINDOW_SIZE, stride=PROBE_STRIDE
            )
            for batch_coordinates, rgb_cpu, sar_cpu, target_cpu in iter_crop_batches(
                rgb, sar, target, coordinates, batch_size=1
            ):
                crop_windows += 1
                rgb_crop = rgb_cpu.to(device, non_blocking=True)
                sar_crop = sar_cpu.to(device, non_blocking=True)
                target_crop = target_cpu.to(device, non_blocking=True)
                backbone_outputs = model.extract_frozen_backbone_outputs(
                    rgb_crop, sar_crop
                )
                if (
                    equivalence_holder[0] is None
                    and not args.skip_cached_equivalence_check
                ):
                    equivalence_holder[0] = verify_cached_forward_equivalence(
                        model, rgb_crop, sar_crop, backbone_outputs
                    )

                def decode(availability):
                    return model.forward_from_backbone_outputs(
                        rgb_crop,
                        sar_crop,
                        backbone_outputs=backbone_outputs,
                        availability=availability,
                    )

                _, full_p2 = capture.run(lambda: decode(full_availability))
                _, sar_p2 = capture.run(lambda: decode(sar_availability))
                coordinate = batch_coordinates[0]
                chunks.append(
                    sample_pure_p2_cells(
                        sar_p2,
                        full_p2,
                        target_crop,
                        purity_threshold=args.purity_threshold,
                        maximum_per_class=args.cells_per_class_crop,
                        seed=stable_seed(
                            args.seed,
                            f"{sample.city}/{sample.tile_id}/{coordinate}",
                        ),
                    )
                )
        city_samples = _cap_city_samples(
            chunks,
            args.max_cells_per_class_city,
            stable_seed(args.seed, f"city-cap/{city}"),
        )
        samples_by_city[city] = city_samples
        metadata[city] = {
            "tiles": len(selected[city]),
            "crop_windows": crop_windows,
            "retained_cells": int(city_samples.sar.shape[0]),
            "class_cell_counts": {
                str(class_id): int((city_samples.class_ids == class_id).sum())
                for class_id in range(NUM_CLASSES)
            },
            "tile_keys": tile_keys,
        }
    return samples_by_city, metadata


def _metric_summary(pooled, by_city):
    return {
        "pooled": {name: metric.compute() for name, metric in pooled.items()},
        "by_city": {
            city: {name: metric.compute() for name, metric in variants.items()}
            for city, variants in sorted(by_city.items())
        },
    }


@torch.inference_mode()
def _evaluate_leave_one_city_out(
    model,
    capture: FRMP2Capture,
    intervention: FRMP2AdditiveIntervention,
    dataset,
    selected,
    probes,
    args,
    device,
):
    full_availability = canonical_availability("full", batch_size=1, device=device)
    sar_availability = canonical_availability("sar", batch_size=1, device=device)
    pooled = {name: EarthMissMetrics() for name in VARIANTS}
    by_city = {}
    crop_windows = 0
    for city in EARTHMISS_CITIES["train"]:
        city_metrics = {name: EarthMissMetrics() for name in VARIANTS}
        by_city[city] = city_metrics
        subset = _IndexedSubset(dataset, selected[city])
        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        city_probes = probes[city]
        for _, rgb, sar, target in tqdm(loader, desc=f"frmp2-eval-{city}"):
            height, width = target.shape[-2:]
            coordinates = sliding_window_coordinates(
                height, width, window_size=WINDOW_SIZE, stride=PROBE_STRIDE
            )
            stitcher = VariantLogitStitcher(
                height, width, NUM_CLASSES, VARIANTS
            )
            for batch_coordinates, rgb_cpu, sar_cpu, _ in iter_crop_batches(
                rgb, sar, target, coordinates, batch_size=1
            ):
                crop_windows += 1
                rgb_crop = rgb_cpu.to(device, non_blocking=True)
                sar_crop = sar_cpu.to(device, non_blocking=True)
                backbone_outputs = model.extract_frozen_backbone_outputs(
                    rgb_crop, sar_crop
                )

                def decode(availability):
                    return model.forward_from_backbone_outputs(
                        rgb_crop,
                        sar_crop,
                        backbone_outputs=backbone_outputs,
                        availability=availability,
                    )

                full_logits, full_p2 = capture.run(lambda: decode(full_availability))
                sar_logits, sar_p2 = capture.run(lambda: decode(sar_availability))
                exact = args.correction_alpha * (full_p2 - sar_p2)
                predicted = {
                    "oracle_alpha_0.25": exact,
                    "ridge_alpha_0.25": args.correction_alpha
                    * city_probes["ridge"].predict_bchw(sar_p2),
                    "shuffled_ridge_alpha_0.25": args.correction_alpha
                    * city_probes["shuffled_ridge"].predict_bchw(sar_p2),
                    "mean_alpha_0.25": args.correction_alpha
                    * city_probes["mean"].predict_bchw(sar_p2),
                }
                logits = {"full": full_logits, "sar": sar_logits}
                for name, correction in predicted.items():
                    logits[name] = intervention.run(
                        lambda: decode(sar_availability), correction
                    )
                stitcher.add(batch_coordinates, logits)
            tile_logits = stitcher.finalize()
            target_cpu = target.cpu()
            for name, logits in tile_logits.items():
                prediction = logits.argmax(dim=1)
                pooled[name].update(prediction, target_cpu)
                city_metrics[name].update(prediction, target_cpu)
    return _metric_summary(pooled, by_city), crop_windows


def _performance_decision(metrics) -> dict[str, Any]:
    pooled = metrics["pooled"]
    by_city = metrics["by_city"]
    comparisons = {
        "ridge_minus_sar_pp": 100.0
        * (pooled["ridge_alpha_0.25"]["mIoU"] - pooled["sar"]["mIoU"]),
        "ridge_minus_shuffled_pp": 100.0
        * (
            pooled["ridge_alpha_0.25"]["mIoU"]
            - pooled["shuffled_ridge_alpha_0.25"]["mIoU"]
        ),
        "ridge_minus_mean_pp": 100.0
        * (pooled["ridge_alpha_0.25"]["mIoU"] - pooled["mean_alpha_0.25"]["mIoU"]),
        "oracle_minus_sar_pp": 100.0
        * (pooled["oracle_alpha_0.25"]["mIoU"] - pooled["sar"]["mIoU"]),
    }
    city_rows = {}
    for city, values in by_city.items():
        ridge = values["ridge_alpha_0.25"]["mIoU"]
        city_rows[city] = {
            "ridge_minus_sar_pp": 100.0 * (ridge - values["sar"]["mIoU"]),
            "ridge_minus_shuffled_pp": 100.0
            * (ridge - values["shuffled_ridge_alpha_0.25"]["mIoU"]),
            "ridge_minus_mean_pp": 100.0
            * (ridge - values["mean_alpha_0.25"]["mIoU"]),
            "oracle_minus_sar_pp": 100.0
            * (values["oracle_alpha_0.25"]["mIoU"] - values["sar"]["mIoU"]),
        }
    counts = {
        key: sum(row[key] >= 0.0 for row in city_rows.values())
        for key in (
            "ridge_minus_sar_pp",
            "ridge_minus_shuffled_pp",
            "ridge_minus_mean_pp",
            "oracle_minus_sar_pp",
        )
    }
    supported = (
        comparisons["oracle_minus_sar_pp"] >= 0.25
        and counts["oracle_minus_sar_pp"] >= 5
        and comparisons["ridge_minus_sar_pp"] >= 0.25
        and comparisons["ridge_minus_shuffled_pp"] >= 0.25
        and comparisons["ridge_minus_mean_pp"] >= 0.25
        and counts["ridge_minus_sar_pp"] >= 5
        and counts["ridge_minus_shuffled_pp"] >= 5
        and counts["ridge_minus_mean_pp"] >= 5
    )
    return {
        "pooled": comparisons,
        "by_city": city_rows,
        "nonnegative_city_counts": counts,
        "recoverability_supported": supported,
        "decision": (
            "frmp2_linear_correction_recoverability_supported"
            if supported
            else "frmp2_linear_correction_recoverability_not_supported"
        ),
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite FRM P2 report: {output_path}")
    git_head_start = _assert_git_head_unchanged(_git_head())
    if not torch.cuda.is_available():
        raise RuntimeError("FRM P2 recoverability diagnosis requires CUDA")
    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
    device = torch.device("cuda")
    model, checkpoint = load_frozen_model(args.checkpoint, "c", weights_path, device)
    expected = {
        "sha256": EXPECTED_RUN_C_CHECKPOINT_SHA256,
        "epoch": EXPECTED_RUN_C_EPOCH,
        "seed": EXPECTED_RUN_C_SEED,
    }
    observed = {key: checkpoint.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"FRM P2 checkpoint mismatch: {observed} != {expected}")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("FRM P2 diagnostic model must be frozen and eval-mode")
    dataset = _build_dataset(args)
    if len(dataset) != EXPECTED_TRAIN_TILES:
        raise RuntimeError(
            f"EarthMiss Train manifest changed: {len(dataset)} != {EXPECTED_TRAIN_TILES}"
        )
    selected = _selected_indices(dataset, args.tiles_per_city, args.seed)
    buffers_before = snapshot_batchnorm_buffers(model)
    equivalence_holder = [None]
    with FRMP2Capture(model) as capture:
        samples, extraction = _extract_city_samples(
            model,
            capture,
            dataset,
            selected,
            args,
            device,
            equivalence_holder,
        )
        moments = {
            city: city_moments(
                values, device, stable_seed(args.seed, f"shuffle/{city}")
            )
            for city, values in samples.items()
        }
        probes = {}
        feature_metrics = {}
        for heldout_city in EARTHMISS_CITIES["train"]:
            training_moments = combine_moments(moments, heldout_city)
            probes[heldout_city] = {
                "ridge": fit_affine_ridge(training_moments, args.ridge_alpha),
                "shuffled_ridge": fit_affine_ridge(
                    training_moments, args.ridge_alpha, shuffled=True
                ),
                "mean": fit_affine_ridge(
                    training_moments, args.ridge_alpha, mean_only=True
                ),
            }
            feature_metrics[heldout_city] = {
                name: correction_prediction_metrics(probe, samples[heldout_city])
                for name, probe in probes[heldout_city].items()
            }
        with FRMP2AdditiveIntervention(model) as intervention:
            metrics, evaluation_crops = _evaluate_leave_one_city_out(
                model,
                capture,
                intervention,
                dataset,
                selected,
                probes,
                args,
                device,
            )
    decision = _performance_decision(metrics)
    parameter_grad_fields_all_none = all(
        parameter.grad is None for parameter in model.parameters()
    )
    if not parameter_grad_fields_all_none:
        raise RuntimeError("FRM P2 diagnostic populated model parameter gradients")
    formal = (
        args.tiles_per_city == TRAIN_TILES_PER_CITY
        and args.cells_per_class_crop == CELLS_PER_CLASS_PER_CROP
        and args.max_cells_per_class_city == MAX_CELLS_PER_CLASS_PER_CITY
        and args.ridge_alpha == RIDGE_ALPHA
        and args.correction_alpha == CORRECTION_ALPHA
        and args.purity_threshold == PURITY_THRESHOLD
        and args.seed == SEED
        and not args.skip_cached_equivalence_check
    )
    report = {
        "schema": SCHEMA,
        "formal": formal,
        "training_was_performed": False,
        "mm_dino_training_was_performed": False,
        "diagnostic_probe_was_fit": True,
        "optimizer_was_constructed": False,
        "parameter_grad_fields_all_none": parameter_grad_fields_all_none,
        "split": "Train only with leave-one-city-out; Val/Test not accessed",
        "git_head": git_head_start,
        "checkpoint": checkpoint,
        "protocol": {
            "cities": list(EARTHMISS_CITIES["train"]),
            "tiles_per_city": args.tiles_per_city,
            "window_size": WINDOW_SIZE,
            "stride": PROBE_STRIDE,
            "apply_train_transform": False,
            "purity_threshold": args.purity_threshold,
            "cells_per_class_crop": args.cells_per_class_crop,
            "max_cells_per_class_city": args.max_cells_per_class_city,
            "sample_storage_dtype": "float32",
            "ridge_alpha": args.ridge_alpha,
            "correction_alpha": args.correction_alpha,
            "input": "canonical SAR post-FRM P2",
            "target": "paired Full-minus-SAR post-FRM P2",
            "fit": "six-city affine 1x1 ridge; one held-out city; seven folds",
            "controls": [
                "zero correction (canonical SAR)",
                "training-city global mean correction",
                "equal-capacity ridge with target shuffled within city and GT class",
            ],
            "primary_gate": (
                "oracle>=+0.25pp and >=5/7 cities; ridge>=+0.25pp vs SAR, "
                "shuffled, and mean with >=5/7 nonnegative cities for each"
            ),
        },
        "extraction": extraction,
        "feature_prediction": feature_metrics,
        "metrics": metrics,
        "decision_evidence": decision,
        "processed": {
            "selected_tiles": sum(len(values) for values in selected.values()),
            "extraction_crop_windows": sum(
                values["crop_windows"] for values in extraction.values()
            ),
            "evaluation_crop_windows": evaluation_crops,
        },
        "cached_forward_equivalence": equivalence_holder[0],
        "batchnorm_audit": assert_buffers_unchanged(buffers_before, model),
        "interpretation": (
            "Success supports only a fixed linear SAR-P2 predictor on selected "
            "held-out Train cities. Failure rejects this linear correction unit; "
            "neither outcome estimates EarthMiss Test performance."
        ),
    }
    _assert_git_head_unchanged(git_head_start)
    write_json_exclusive(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = diagnose(args)
    print(
        {
            "output": args.output,
            "formal": report["formal"],
            "decision": report["decision_evidence"]["decision"],
            "pooled": report["decision_evidence"]["pooled"],
        }
    )


if __name__ == "__main__":
    main()
