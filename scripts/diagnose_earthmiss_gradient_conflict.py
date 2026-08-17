"""Measure Full/SAR segmentation-gradient conflict in frozen Run C.

This is a zero-update Train diagnostic.  Same-city batches are decoded twice
from the same cached RGB/SAR DINO tensors.  Full and canonical SAR segmentation
gradients are compared over Adapter, target-scale FRM, SEFusion, PRN, and head
parameter groups under both train-BN and eval-BN semantics.  Persistent buffers
and RNG are restored after every endpoint forward; no optimizer is constructed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import EARTHMISS_CITIES, build_dataset  # noqa: E402
from losses import DiceLoss, JointLoss, SoftCrossEntropyLoss  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402

from scripts.diagnose_earthmiss_missing_v3 import load_frozen_model  # noqa: E402
from scripts.diagnose_earthmiss_scale_transition import (  # noqa: E402
    EXPECTED_RUN_C_CHECKPOINT_SHA256,
    EXPECTED_RUN_C_EPOCH,
    EXPECTED_RUN_C_SEED,
    assert_buffers_unchanged,
    snapshot_batchnorm_buffers,
)
from scripts.earthmiss_causal_diagnostics_common import (  # noqa: E402
    gradient_pair_statistics_from_primitives,
    gradient_tensor_primitives,
    parameter_group_manifest,
    stable_seed,
    summarize_gradient_records,
)
from scripts.earthmiss_scale_transition_common import write_json_exclusive  # noqa: E402
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
)


SCHEMA = "earthmiss_missing_gradient_conflict_v1"
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/"
    "run-c-e15-train-gradient-conflict.json"
)
EXPECTED_TRAIN_TILES = 2641
WINDOW_SIZE = 512
BATCH_SIZE = 8
DEFAULT_BATCHES_PER_CITY = 8
SEED = 20260812


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batches-per-city", type=int, default=DEFAULT_BATCHES_PER_CITY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--bn-modes",
        nargs="+",
        choices=("train", "eval"),
        default=["train", "eval"],
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.batches_per_city <= 0:
        raise ValueError("--batches-per-city must be positive")
    if len(args.bn_modes) != len(set(args.bn_modes)):
        raise ValueError("--bn-modes must be unique")


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
        raise RuntimeError("gradient diagnostic requires a readable Git HEAD")
    if end_head != start_head:
        raise RuntimeError(
            "repository HEAD changed during gradient diagnosis: "
            f"{start_head} -> {end_head}; refusing to write a formal report"
        )
    return start_head


def _restore_buffers(snapshot: dict[str, torch.Tensor], model: torch.nn.Module) -> None:
    current = dict(model.named_buffers())
    if set(snapshot) != {
        name for name in current if name in snapshot
    }:
        missing = sorted(set(snapshot) - set(current))
        raise RuntimeError(f"BatchNorm buffers disappeared: {missing}")
    with torch.no_grad():
        for name, value in snapshot.items():
            current[name].copy_(value.to(device=current[name].device))


def _rng_snapshot() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state().clone(),
        "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()],
    }


def _rng_restore(snapshot: dict[str, Any]) -> None:
    torch.set_rng_state(snapshot["torch"])
    torch.cuda.set_rng_state_all(snapshot["cuda"])


def _same_city_batches(dataset, batches_per_city: int, seed: int):
    indices_by_city: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        indices_by_city[str(sample.city)].append(index)
    expected_cities = tuple(EARTHMISS_CITIES["train"])
    if set(indices_by_city) != set(expected_cities):
        raise RuntimeError("EarthMiss Train city manifest changed")
    schedule = []
    for city in expected_cities:
        indices = list(indices_by_city[city])
        if len(indices) < BATCH_SIZE:
            raise RuntimeError(
                f"city {city} lacks {BATCH_SIZE} tiles required for one "
                "within-batch-unique diagnostic batch"
            )
        generator = random.Random(stable_seed(seed, city))
        remaining: list[int] = []
        for _ in range(batches_per_city):
            batch: list[int] = []
            while len(batch) < BATCH_SIZE:
                if not remaining:
                    remaining = list(indices)
                    generator.shuffle(remaining)
                for index in tuple(remaining):
                    if index in batch:
                        continue
                    batch.append(index)
                    remaining.remove(index)
                    if len(batch) == BATCH_SIZE:
                        break
            if len(set(batch)) != BATCH_SIZE:
                raise RuntimeError(f"city {city} diagnostic batch is not unique")
            schedule.append((city, tuple(batch)))
    return tuple(schedule)


def _schedule_audit(dataset, schedule) -> dict[str, dict[str, int]]:
    available = defaultdict(int)
    for sample in dataset.samples:
        available[str(sample.city)] += 1
    scheduled: dict[str, list[int]] = defaultdict(list)
    for city, indices in schedule:
        scheduled[city].extend(indices)
    return {
        city: {
            "available_tiles": int(available[city]),
            "scheduled_examples": len(scheduled[city]),
            "scheduled_unique_tiles": len(set(scheduled[city])),
            "cross_batch_reuses": len(scheduled[city]) - len(set(scheduled[city])),
        }
        for city in EARTHMISS_CITIES["train"]
    }


def _load_batch(dataset, indices: Sequence[int]):
    rows = [dataset[index] for index in indices]
    if any(len(row) != 3 for row in rows):
        raise RuntimeError("gradient diagnostic requires RGB/SAR/GT rows")
    return tuple(torch.stack([row[column] for row in rows]) for column in range(3))


def _endpoint_gradients(
    model,
    criterion,
    rgb,
    sar,
    target,
    backbone_outputs,
    endpoint: str,
    named_parameters: dict[str, torch.nn.Parameter],
    mode: str,
    buffers: dict[str, torch.Tensor],
    rng: dict[str, Any],
):
    _restore_buffers(buffers, model)
    _rng_restore(rng)
    model.train(mode == "train")
    availability = canonical_availability(
        endpoint,
        batch_size=rgb.shape[0],
        device=rgb.device,
    )
    logits = model.forward_from_backbone_outputs(
        rgb,
        sar,
        backbone_outputs=backbone_outputs,
        availability=availability,
    )
    loss = criterion(logits, target)
    gradients = torch.autograd.grad(
        loss,
        tuple(named_parameters.values()),
        allow_unused=True,
        materialize_grads=False,
    )
    result = dict(zip(named_parameters, gradients, strict=True))
    scalar_loss = float(loss.detach())
    del logits, loss
    _restore_buffers(buffers, model)
    _rng_restore(rng)
    return scalar_loss, result


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite gradient report: {output_path}")
    git_head_start = _assert_git_head_unchanged(_git_head())
    if not torch.cuda.is_available():
        raise RuntimeError("gradient conflict diagnosis requires CUDA")
    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
    device = torch.device("cuda")
    model, checkpoint = load_frozen_model(
        args.checkpoint,
        "c",
        weights_path,
        device,
    )
    expected = {
        "sha256": EXPECTED_RUN_C_CHECKPOINT_SHA256,
        "epoch": EXPECTED_RUN_C_EPOCH,
        "seed": EXPECTED_RUN_C_SEED,
    }
    observed = {key: checkpoint.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"gradient audit checkpoint mismatch: {observed} != {expected}")

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("backbone."))
    named_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    groups = parameter_group_manifest(model)
    group_order = tuple(groups)
    criterion = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=8),
        DiceLoss(smooth=0.05, ignore_index=8),
        1.0,
        1.0,
    )
    dataset = build_dataset(
        "EarthMiss",
        "train",
        dataset_root=args.dataset_root,
        window_size=(WINDOW_SIZE, WINDOW_SIZE),
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
        apply_train_transform=True,
        cache_size=0,
    )
    if len(dataset) != EXPECTED_TRAIN_TILES:
        raise RuntimeError(
            f"EarthMiss Train manifest changed: {len(dataset)} != {EXPECTED_TRAIN_TILES}"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    schedule = _same_city_batches(dataset, args.batches_per_city, args.seed)
    schedule_audit = _schedule_audit(dataset, schedule)
    buffers_before = snapshot_batchnorm_buffers(model)
    records: dict[str, list[dict[str, Any]]] = {mode: [] for mode in args.bn_modes}

    for batch_index, (city, indices) in enumerate(
        tqdm(schedule, desc="gradient-conflict")
    ):
        rgb_cpu, sar_cpu, target_cpu = _load_batch(dataset, indices)
        rgb = rgb_cpu.to(device)
        sar = sar_cpu.to(device)
        target = target_cpu.to(device)
        model.eval()
        backbone_outputs = model.extract_frozen_backbone_outputs(rgb, sar)
        for mode in args.bn_modes:
            baseline_rng = _rng_snapshot()
            full_loss, full_gradients = _endpoint_gradients(
                model,
                criterion,
                rgb,
                sar,
                target,
                backbone_outputs,
                "full",
                named_parameters,
                mode,
                buffers_before,
                baseline_rng,
            )
            sar_loss, sar_gradients = _endpoint_gradients(
                model,
                criterion,
                rgb,
                sar,
                target,
                backbone_outputs,
                "sar",
                named_parameters,
                mode,
                buffers_before,
                baseline_rng,
            )
            primitives = gradient_tensor_primitives(
                full_gradients,
                sar_gradients,
                tuple(named_parameters),
            )
            group_rows = {
                group: gradient_pair_statistics_from_primitives(
                    primitives,
                    names,
                )
                for group, names in groups.items()
            }
            records[mode].append(
                {
                    "batch_index": batch_index,
                    "city": city,
                    "tile_indices": list(indices),
                    "supported_class_ids": torch.unique(
                        target[(target >= 0) & (target < 8)]
                    ).detach().cpu().tolist(),
                    "full_loss": full_loss,
                    "sar_loss": sar_loss,
                    "groups": group_rows,
                }
            )
            del full_gradients, sar_gradients
        del backbone_outputs, rgb, sar, target

    _restore_buffers(buffers_before, model)
    model.eval()
    parameter_grad_fields_all_none = all(
        parameter.grad is None for parameter in model.parameters()
    )
    if not parameter_grad_fields_all_none:
        raise RuntimeError("autograd.grad unexpectedly populated parameter .grad fields")
    summary = {}
    for mode, mode_records in records.items():
        overall_rows = [row["groups"] for row in mode_records]
        by_city = {}
        for city in EARTHMISS_CITIES["train"]:
            city_rows = [
                row["groups"] for row in mode_records if row["city"] == city
            ]
            by_city[city] = summarize_gradient_records(city_rows, group_order)
        summary[mode] = {
            "overall": summarize_gradient_records(overall_rows, group_order),
            "by_city": by_city,
        }

    report = {
        "schema": SCHEMA,
        "formal": (
            args.batches_per_city == DEFAULT_BATCHES_PER_CITY
            and set(args.bn_modes) == {"train", "eval"}
            and args.seed == SEED
        ),
        "training_was_performed": False,
        "optimizer_was_constructed": False,
        "split": "Train only; Val/Test not accessed",
        "git_head": git_head_start,
        "checkpoint": checkpoint,
        "protocol": {
            "seed": args.seed,
            "batch_size": BATCH_SIZE,
            "batches_per_city": args.batches_per_city,
            "sampling": (
                "deterministic shuffled without-replacement cycles; every batch "
                "contains unique tiles; cross-batch reuse is allowed and audited "
                "when a city has fewer tiles than requested sample positions"
            ),
            "schedule_audit": schedule_audit,
            "cities": list(EARTHMISS_CITIES["train"]),
            "bn_modes": list(args.bn_modes),
            "same_cached_backbone_per_endpoint": True,
            "same_endpoint_rng": True,
            "persistent_bn_restored_after_each_endpoint": True,
            "loss": "released SoftCE(0.05)+Dice(0.05), ignore=8",
            "gradient_order": "Full segmentation gradient vs SAR segmentation gradient",
        },
        "parameter_groups": {name: list(values) for name, values in groups.items()},
        "processed_batches": len(schedule),
        "parameter_grad_fields_all_none": parameter_grad_fields_all_none,
        "summary": summary,
        "records": records,
        "batchnorm_audit": assert_buffers_unchanged(buffers_before, model),
        "interpretation": (
            "Negative cosine is direct shared-parameter optimization conflict. "
            "Non-negative cosine does not prove privileged information is SAR-predictable."
        ),
    }
    _assert_git_head_unchanged(git_head_start)
    write_json_exclusive(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = diagnose(args)
    compact = {
        mode: {
            group: values["cosine"]
            for group, values in mode_summary["overall"].items()
        }
        for mode, mode_summary in report["summary"].items()
    }
    print({"output": args.output, "cosines": compact})


if __name__ == "__main__":
    main()
