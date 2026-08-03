"""Train the fixed EarthMiss V2 three-state Run D screen.

Every paired crop supervises canonical SAR, RGB, and Full states in that fixed
order. Frozen RGB/SAR DINO outputs are each extracted once, while the shared
Adapter/Decoder is executed independently for all three states. The run always
ends at E20; raw SAR/Full Val is recorded at E10/E15/E20, and fixed E15 is the
only primary checkpoint. This launcher is single-GPU and foreground-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from losses import DiceLoss, JointLoss, SoftCrossEntropyLoss  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from models.MMDINO.dino_segment import build_model  # noqa: E402
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
    VAL_SELECTION_CLASS_IDS,
    audit_datasets,
    build_loaders,
    evaluate,
    seed_everything,
)


SEED = 42
EPOCHS = 20
SCHEDULER_T_MAX = 50
VALIDATION_EPOCHS = (10, 15, 20)
PRIMARY_EPOCH = 15
TRAIN_STATE_ORDER = ("sar", "rgb", "full")
RAW_VAL_STATES = ("sar", "full")
PROTOCOL_REVISION = "earthmiss_missing_v2_all_state_fixed_e15_v1"
DEFAULT_OUTPUT_ROOT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v2-all-state"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--seed-role",
        choices=("auto", "exploratory_screen", "confirmatory"),
        default="auto",
        help=(
            "Scientific role of this seed. auto maps seed 42 to the exploratory "
            "screen and seeds 43/44 to confirmation."
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit Train/Val without loading the model or requiring a GPU.",
    )
    return parser.parse_args(argv)


def resolve_seed_role(seed, requested_role="auto"):
    reserved_roles = {
        42: "exploratory_screen",
        43: "confirmatory",
        44: "confirmatory",
    }
    reserved_role = reserved_roles.get(seed)
    if reserved_role is not None:
        if requested_role not in ("auto", reserved_role):
            raise ValueError(
                f"Seed {seed} is reserved for role {reserved_role}, not "
                f"{requested_role}"
            )
        return reserved_role
    if requested_role == "auto":
        raise ValueError(
            "--seed-role auto only recognizes exploratory seed 42 and "
            "confirmatory seeds 43/44; assign an explicit role for any other seed"
        )
    return requested_role


def fixed_snapshot_name(epoch):
    if epoch not in VALIDATION_EPOCHS:
        raise ValueError(f"E{epoch} is not a frozen V2 snapshot epoch")
    return f"fixed_e{epoch:02d}.pth"


def fixed_snapshot_role(epoch):
    if epoch not in VALIDATION_EPOCHS:
        raise ValueError(f"E{epoch} is not a frozen V2 snapshot epoch")
    if epoch == PRIMARY_EPOCH:
        return "primary_weights_requires_bn_bank"
    return "diagnostic_only"


def run_artifact_paths(output_dir):
    output_dir = Path(output_dir)
    return (
        output_dir / "run.json",
        output_dir / "metrics.jsonl",
        *(output_dir / fixed_snapshot_name(epoch) for epoch in VALIDATION_EPOCHS),
    )


def build_run_metadata(args, train_dataset, val_dataset, train_loader):
    steps_per_epoch = len(train_loader)
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "run": "D",
        "seed": args.seed,
        "seed_role": resolve_seed_role(args.seed, args.seed_role),
        "resume_supported": False,
        "interrupted_run_policy": "restart_from_scratch",
        "primary_cell": "E15/SAR/Train-SAR",
        "paired_diagnostic": "E15/Full/Train-Full",
        "train_tiles": len(train_dataset),
        "val_tiles": len(val_dataset),
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "epochs": EPOCHS,
        "train_policy": "paired all-state SAR/RGB/Full",
        "model": {
            "segmentation_head": "raw_conv1x1",
            "released_segmentation_head": "conv_bn_relu",
            "backbone": "dinov3_vits16",
            "backbone_frozen": True,
            "num_modalities": 2,
        },
        "augmentation": {
            "random_crop": [args.window_size, args.window_size],
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.5,
            "random_rotate90_p": 0.5,
        },
        "normalization": {
            "rgb": {
                "policy": train_dataset.rgb_normalization,
                "mean": list(train_dataset.imagenet_mean),
                "std": list(train_dataset.imagenet_std),
            },
            "sar": {
                "policy": "earthmiss_metars_dataset_stats",
                "mean": list(train_dataset.sar_mean),
                "std": list(train_dataset.sar_std),
            },
        },
        "loss": {
            "per_state": "SoftCrossEntropy(smoothing=0.05)+Dice(smoothing=0.05)",
            "ignore_index": 8,
            "state_weights": {state: 1.0 / 3.0 for state in TRAIN_STATE_ORDER},
            "backward_policy": "three_sequential_scaled_backward_calls",
        },
        "optimizer": {
            "type": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.01,
        },
        "scheduler": {
            "type": "CosineAnnealingLR",
            "T_max": SCHEDULER_T_MAX,
            "eta_min": 1e-7,
        },
        "data_loader": {
            "cached_source_arrays_immutable": True,
            "persistent_workers": args.num_workers > 0,
        },
        "budget": {
            "sampling": "one_crop_per_tile_per_epoch",
            "steps_per_epoch": steps_per_epoch,
            "planned_optimizer_steps": steps_per_epoch * EPOCHS,
            "fixed_final_epoch": EPOCHS,
            "early_stopping": False,
        },
        "training_graph": {
            "state_order": list(TRAIN_STATE_ORDER),
            "raw_backbone_cache": "one RGB and one SAR extraction per batch",
            "backbone_calls_per_batch": 2,
            "adapter_decoder_calls_per_batch": 3,
            "backward_calls_per_batch": 3,
            "optimizer_steps_per_batch": 1,
            "canonical_two_slot_states": list(TRAIN_STATE_ORDER),
        },
        "evaluation": {
            "raw_val_epochs": list(VALIDATION_EPOCHS),
            "raw_val_states": list(RAW_VAL_STATES),
            "checkpoint_policy": "fixed_epochs_no_best_selection",
            "primary_epoch": PRIMARY_EPOCH,
            "checkpoint_selection_support": "pooled_gt_present",
            "expected_val_selection_class_ids": VAL_SELECTION_CLASS_IDS,
            "checkpoint_roles": {
                fixed_snapshot_name(epoch): fixed_snapshot_role(epoch)
                for epoch in VALIDATION_EPOCHS
            },
            "primary_weights_checkpoint": fixed_snapshot_name(PRIMARY_EPOCH),
            "deployment_requires_bn_bank": True,
            "paired_endpoint_rule": "E15 weights for SAR and Full",
            "posthoc_calibration": {
                "primary": "Train-SAR",
                "paired_diagnostic": "Train-Full",
            },
        },
        "timing": {
            "train_epoch_wall_seconds": "CUDA-synchronized train-loop boundaries",
            "raw_val_wall_seconds": "CUDA-synchronized per state",
            "component_seconds": "host enqueue time; diagnostic only",
        },
    }


def backward_all_state_losses(model, criterion, rgb, sar, label):
    """Accumulate the exact mean of three state losses without stepping."""

    feature_started = time.perf_counter()
    backbone_outputs = model.extract_frozen_backbone_outputs(rgb, sar)
    feature_host_seconds = time.perf_counter() - feature_started
    if len(backbone_outputs) != 2:
        raise RuntimeError("Run D requires exactly two cached backbone outputs")

    loss_tensors = []
    state_host_seconds = {}
    for state in TRAIN_STATE_ORDER:
        state_started = time.perf_counter()
        availability = canonical_availability(
            state,
            batch_size=rgb.shape[0],
            device=rgb.device,
        )
        logits = model.forward_from_backbone_outputs(
            rgb,
            sar,
            backbone_outputs=backbone_outputs,
            availability=availability,
        )
        loss = criterion(logits, label)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {state} loss")
        (loss / len(TRAIN_STATE_ORDER)).backward()
        loss_tensors.append(loss.detach())
        state_host_seconds[state] = time.perf_counter() - state_started

    detached_losses = torch.stack(loss_tensors).cpu().tolist()
    state_losses = dict(zip(TRAIN_STATE_ORDER, detached_losses, strict=True))
    return {
        "loss": sum(state_losses.values()) / len(TRAIN_STATE_ORDER),
        "state_losses": state_losses,
        "instrumentation": {
            "backbone_calls": 2,
            "adapter_decoder_calls": 3,
            "backward_calls": 3,
            "feature_host_seconds": feature_host_seconds,
            "state_host_seconds": state_host_seconds,
        },
    }


def train_all_state_batch(model, criterion, optimizer, rgb, sar, label):
    """Run one three-backward, one-step Run D optimizer update."""

    optimizer.zero_grad(set_to_none=True)
    record = backward_all_state_losses(model, criterion, rgb, sar, label)
    optimizer_started = time.perf_counter()
    optimizer.step()
    record["instrumentation"]["optimizer_host_seconds"] = (
        time.perf_counter() - optimizer_started
    )
    record["instrumentation"]["optimizer_steps"] = 1
    return record


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    metadata,
    *,
    checkpoint_role,
    raw_validation=None,
):
    requires_bn_bank = checkpoint_role == "primary_weights_requires_bn_bank"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "run": "D",
            "seed": metadata["seed"],
            "protocol": metadata,
            "checkpoint_role": checkpoint_role,
            "selection_state": None,
            "deployment_state": "sar" if requires_bn_bank else None,
            "requires_bn_bank": requires_bn_bank,
            "selection_metric": None,
            "selection_score": None,
            "fixed_epoch_no_selection": True,
            "primary_cell": (
                metadata["primary_cell"] if requires_bn_bank else None
            ),
            "paired_diagnostic": (
                metadata["paired_diagnostic"] if requires_bn_bank else None
            ),
            "raw_validation": raw_validation,
        },
        path,
    )


def _synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    if args.window_size <= 0 or args.window_size % 16:
        raise ValueError("--window-size must be a positive multiple of 16")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers must be non-negative")
    resolve_seed_role(args.seed, args.seed_role)

    seed_everything(args.seed)
    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    if args.audit_only:
        audit_datasets(train_dataset, val_dataset)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("EarthMiss V2 Run D training requires a CUDA GPU")
    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    output_dir = Path(args.output_root) / f"run_d_seed{args.seed}"
    metrics_path = output_dir / "metrics.jsonl"
    existing_artifacts = tuple(
        path for path in run_artifact_paths(output_dir) if path.exists()
    )
    if existing_artifacts:
        raise FileExistsError(
            "EarthMiss V2 Run D does not support resume or overwrite; move the "
            f"interrupted run and restart from scratch: {existing_artifacts[0]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

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
        raw_logits=True,
    ).to(device)
    criterion = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=8),
        DiceLoss(smooth=0.05, ignore_index=8),
        1.0,
        1.0,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=SCHEDULER_T_MAX,
        eta_min=1e-7,
    )
    metadata = build_run_metadata(args, train_dataset, val_dataset, train_loader)

    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        _synchronize_if_cuda(device)
        epoch_started = time.perf_counter()
        loss_sum = 0.0
        state_loss_sums = {state: 0.0 for state in TRAIN_STATE_ORDER}
        feature_host_seconds = 0.0
        state_host_seconds = {state: 0.0 for state in TRAIN_STATE_ORDER}
        optimizer_host_seconds = 0.0
        backbone_calls = 0
        adapter_decoder_calls = 0
        backward_calls = 0
        optimizer_steps = 0
        batch_count = 0

        for rgb, sar, label in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            rgb = rgb.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            batch_record = train_all_state_batch(
                model,
                criterion,
                optimizer,
                rgb,
                sar,
                label,
            )

            loss_sum += batch_record["loss"]
            for state, loss in batch_record["state_losses"].items():
                state_loss_sums[state] += loss
            instrumentation = batch_record["instrumentation"]
            backbone_calls += instrumentation["backbone_calls"]
            adapter_decoder_calls += instrumentation["adapter_decoder_calls"]
            backward_calls += instrumentation["backward_calls"]
            optimizer_steps += instrumentation["optimizer_steps"]
            feature_host_seconds += instrumentation["feature_host_seconds"]
            for state, seconds in instrumentation["state_host_seconds"].items():
                state_host_seconds[state] += seconds
            optimizer_host_seconds += instrumentation["optimizer_host_seconds"]
            batch_count += 1

        _synchronize_if_cuda(device)
        epoch_wall_seconds = time.perf_counter() - epoch_started
        scheduler.step()
        if batch_count == 0:
            raise RuntimeError("EarthMiss Train loader produced no batches")
        expected_counts = (2 * batch_count, 3 * batch_count, 3 * batch_count, batch_count)
        actual_counts = (
            backbone_calls,
            adapter_decoder_calls,
            backward_calls,
            optimizer_steps,
        )
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Run D call contract changed: expected {expected_counts}, got {actual_counts}"
            )

        record = {
            "epoch": epoch,
            "train_loss": loss_sum / batch_count,
            "train_state_loss": {
                state: state_loss_sums[state] / batch_count
                for state in TRAIN_STATE_ORDER
            },
            "learning_rate": optimizer.param_groups[0]["lr"],
            "instrumentation": {
                "batches": batch_count,
                "backbone_calls": backbone_calls,
                "adapter_decoder_calls": adapter_decoder_calls,
                "backward_calls": backward_calls,
                "optimizer_steps": optimizer_steps,
                "train_epoch_wall_seconds": epoch_wall_seconds,
                "feature_host_seconds": feature_host_seconds,
                "state_host_seconds": state_host_seconds,
                "optimizer_host_seconds": optimizer_host_seconds,
            },
        }

        raw_validation = None
        if epoch in VALIDATION_EPOCHS:
            raw_validation = {}
            raw_val_wall_seconds = {}
            for state in RAW_VAL_STATES:
                _synchronize_if_cuda(device)
                validation_started = time.perf_counter()
                raw_validation[state] = evaluate(
                    model,
                    val_loader,
                    state,
                    device,
                    args.window_size,
                    args.batch_size * 4,
                )
                _synchronize_if_cuda(device)
                raw_val_wall_seconds[state] = (
                    time.perf_counter() - validation_started
                )
            record["raw_val"] = raw_validation
            record["raw_val_wall_seconds"] = raw_val_wall_seconds
            snapshot_path = output_dir / fixed_snapshot_name(epoch)
            save_checkpoint(
                snapshot_path,
                model,
                optimizer,
                scheduler,
                epoch,
                metadata,
                checkpoint_role=fixed_snapshot_role(epoch),
                raw_validation=raw_validation,
            )
            model.train()

        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record))


if __name__ == "__main__":
    main()
