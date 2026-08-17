"""Zero-update pairwise gradient audit for Full/Optical/SAR WHU endpoints."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import build_dataset  # noqa: E402
from losses import DiceLoss, JointLoss, SoftCrossEntropyLoss  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402

from scripts.diagnose_earthmiss_scale_transition import (  # noqa: E402
    assert_buffers_unchanged,
    snapshot_batchnorm_buffers,
)
from scripts.earthmiss_causal_diagnostics_common import (  # noqa: E402
    gradient_pair_statistics_from_primitives,
    gradient_tensor_primitives,
    parameter_group_manifest,
    summarize_gradient_records,
)
from scripts.earthmiss_scale_transition_common import write_json_exclusive  # noqa: E402
from scripts.whu_multideployment_diagnostics_common import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_RUN_C_CHECKPOINT,
    DEFAULT_WEIGHTS,
    ENDPOINTS,
    IGNORE_INDEX,
    NUM_CLASSES,
    load_historical_run_c,
    split_names,
)


SCHEMA = "whu_multideployment_gradient_conflict_v1"
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/mm-dino/outputs/whu-multideployment-causal-audit/"
    "historical-run-c-e50-gradients.json"
)
DEFAULT_TRAIN_SPLIT = str(REPO_ROOT / "splits" / "whu" / "official_train.txt")
WINDOW_SIZE = 512
BATCH_SIZE = 8
DEFAULT_BATCHES = 32
SEED = 20260817
PAIRS = (("full", "rgb"), ("full", "sar"), ("rgb", "sar"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_RUN_C_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--batches", type=int, default=DEFAULT_BATCHES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--bn-modes", nargs="+", choices=("train", "eval"),
        default=["train", "eval"],
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.batches <= 0:
        raise ValueError("--batches must be positive")
    if len(args.bn_modes) != len(set(args.bn_modes)):
        raise ValueError("--bn-modes must be unique")


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _restore_buffers(snapshot: dict[str, torch.Tensor], model: torch.nn.Module) -> None:
    current = dict(model.named_buffers())
    missing = sorted(set(snapshot) - set(current))
    if missing:
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


def _endpoint_gradients(
    model,
    criterion,
    optical,
    sar,
    target,
    backbone_outputs,
    endpoint: str,
    named_parameters,
    mode: str,
    buffers,
    rng,
):
    _restore_buffers(buffers, model)
    _rng_restore(rng)
    model.train(mode == "train")
    availability = canonical_availability(
        endpoint, batch_size=optical.shape[0], device=optical.device,
    )
    logits = model.forward_from_backbone_outputs(
        optical, sar, backbone_outputs=backbone_outputs,
        availability=availability,
    )
    loss = criterion(logits, target)
    gradients = torch.autograd.grad(
        loss, tuple(named_parameters.values()), allow_unused=True,
        materialize_grads=False,
    )
    result = dict(zip(named_parameters, gradients, strict=True))
    scalar = float(loss.detach())
    _restore_buffers(buffers, model)
    _rng_restore(rng)
    return scalar, result


def _rename_pair_statistics(values: dict[str, Any]) -> dict[str, Any]:
    result = dict(values)
    result["left_norm"] = result.pop("full_norm")
    result["right_norm"] = result.pop("sar_norm")
    result["right_to_left_norm_ratio"] = result.pop("sar_to_full_norm_ratio")
    result["left_only"] = result.pop("full_only")
    result["right_only"] = result.pop("sar_only")
    return result


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite gradient report: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("WHU gradient audit requires CUDA")
    if split_names(args.train_split) != split_names(DEFAULT_TRAIN_SPLIT):
        raise ValueError("historical gradient audit is bound to official_train.txt")
    device = torch.device("cuda")
    model, checkpoint = load_historical_run_c(
        args.checkpoint, args.backbone_weights, device, freeze_model=False,
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("backbone."))
    named_parameters = {
        name: parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    groups = parameter_group_manifest(model)
    group_order = tuple(groups)
    criterion = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=IGNORE_INDEX),
        DiceLoss(smooth=0.05, ignore_index=IGNORE_INDEX), 1.0, 1.0,
    )
    dataset = build_dataset(
        "WHU", "train", dataset_root=args.dataset_root,
        split_file=args.train_split, window_size=(WINDOW_SIZE, WINDOW_SIZE),
        model_name="DINOv3", modality="multi", backbone_type="dinov3_vits16",
        optical_bands="nir-r-g", cache_size=0, mask_padding_ignore=True,
    )
    if len(dataset.rgb_files) != 80:
        raise RuntimeError("WHU official Train manifest changed")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=True,
    )
    buffers_before = snapshot_batchnorm_buffers(model)
    records = {mode: [] for mode in args.bn_modes}
    for batch_index, (optical_cpu, sar_cpu, target_cpu) in enumerate(
        tqdm(loader, total=args.batches, desc="whu-gradient-conflict")
    ):
        if batch_index >= args.batches:
            break
        optical = optical_cpu.to(device)
        sar = sar_cpu.to(device)
        target = target_cpu.to(device, dtype=torch.long)
        model.eval()
        backbone_outputs = model.extract_frozen_backbone_outputs(optical, sar)
        for mode in args.bn_modes:
            baseline_rng = _rng_snapshot()
            endpoint_losses = {}
            endpoint_gradients = {}
            for endpoint in ENDPOINTS:
                loss, gradients = _endpoint_gradients(
                    model, criterion, optical, sar, target, backbone_outputs,
                    endpoint, named_parameters, mode, buffers_before, baseline_rng,
                )
                endpoint_losses[endpoint] = loss
                endpoint_gradients[endpoint] = gradients
            pair_rows = {}
            for left, right in PAIRS:
                primitives = gradient_tensor_primitives(
                    endpoint_gradients[left], endpoint_gradients[right],
                    tuple(named_parameters),
                )
                pair_rows[f"{left}_vs_{right}"] = {
                    group: _rename_pair_statistics(
                        gradient_pair_statistics_from_primitives(primitives, names)
                    )
                    for group, names in groups.items()
                }
            records[mode].append({
                "batch_index": batch_index,
                "supported_class_ids": torch.unique(
                    target[(target >= 0) & (target < NUM_CLASSES)]
                ).detach().cpu().tolist(),
                "endpoint_losses": endpoint_losses,
                "pairs": pair_rows,
            })
            del endpoint_gradients
        del backbone_outputs, optical, sar, target

    _restore_buffers(buffers_before, model)
    model.eval()
    if not all(parameter.grad is None for parameter in model.parameters()):
        raise RuntimeError("autograd.grad populated parameter .grad fields")
    summary = {}
    for mode, rows in records.items():
        summary[mode] = {}
        for left, right in PAIRS:
            pair = f"{left}_vs_{right}"
            # summarize_gradient_records only consumes cosine and the historical
            # ratio key, so provide a compatibility projection and rename it.
            compatibility_rows = []
            for row in rows:
                compatibility_rows.append({
                    group: {
                        **values,
                        "sar_to_full_norm_ratio": values["right_to_left_norm_ratio"],
                    }
                    for group, values in row["pairs"][pair].items()
                })
            values = summarize_gradient_records(compatibility_rows, group_order)
            for group in values.values():
                group["right_to_left_norm_ratio"] = group.pop(
                    "sar_to_full_norm_ratio"
                )
            summary[mode][pair] = values

    report = {
        "schema": SCHEMA,
        "formal": (
            args.batches == DEFAULT_BATCHES
            and args.seed == SEED
            and set(args.bn_modes) == {"train", "eval"}
        ),
        "training_was_performed": False,
        "optimizer_was_constructed": False,
        "split": "official Train only; Test not accessed",
        "git_head": _git_head(),
        "checkpoint": checkpoint,
        "protocol": {
            "deployment_endpoints": list(ENDPOINTS),
            "pairs": [f"{left}_vs_{right}" for left, right in PAIRS],
            "seed": args.seed,
            "batch_size": BATCH_SIZE,
            "batches": args.batches,
            "bn_modes": list(args.bn_modes),
            "same_cached_backbone_per_endpoint": True,
            "same_endpoint_rng": True,
            "persistent_bn_restored_after_each_endpoint": True,
            "loss": "released SoftCE(0.05)+Dice(0.05), ignore=7",
            "historical_limitation": "Run C did not train Optical-only batches",
        },
        "parameter_groups": {name: list(values) for name, values in groups.items()},
        "processed_batches": len(next(iter(records.values()))),
        "parameter_grad_fields_all_none": True,
        "summary": summary,
        "records": records,
        "batchnorm_audit": assert_buffers_unchanged(buffers_before, model),
        "interpretation": (
            "Negative pairwise cosine is direct shared-parameter conflict. "
            "Non-negative cosine does not prove one endpoint can reconstruct "
            "information available only in another endpoint."
        ),
    }
    write_json_exclusive(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = diagnose(args)
    compact = {
        mode: {
            pair: {
                group: values["cosine"]
                for group, values in groups.items()
            }
            for pair, groups in pairs.items()
        }
        for mode, pairs in report["summary"].items()
    }
    print({"output": args.output, "cosines": compact})


if __name__ == "__main__":
    main()
