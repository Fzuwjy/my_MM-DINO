"""Train the three EarthMiss missing-modality V1 baselines.

Run A trains with SAR only, Run B trains with both modalities, and Run C uses
homogeneous 50/50 Full/SAR batches. Validation is city-held-out EarthMiss Val.
This launcher is intentionally single-GPU and foreground-only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import build_dataset  # noqa: E402
from losses import DiceLoss, JointLoss, SoftCrossEntropyLoss  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from models.MMDINO.dino_segment import build_model  # noqa: E402
from utils.earthmiss_metrics import EarthMissMetrics  # noqa: E402
from utils.inference import slide_inference  # noqa: E402


DEFAULT_DATASET_ROOT = "/root/autodl-tmp/mm-dino/datasets/EarthMiss"
DEFAULT_WEIGHTS = (
    "/root/autodl-tmp/mm-dino/weights/"
    "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
DEFAULT_OUTPUT_ROOT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v1-cache-safe"
)
VAL_SELECTION_CLASS_IDS = list(range(7))
PROTOCOL_REVISION = "earthmiss_missing_v1_val_patience_v4"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("A", "B", "C"), required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument(
        "--early-stop-patience-evals",
        type=int,
        default=0,
        help=(
            "Stop after this many consecutive SAR validation checks without a "
            "strict mIoU improvement; 0 disables early stopping."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Build Train/Val manifests and inspect one sample without loading the model.",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(args):
    dataset_kwargs = {
        "dataset_root": args.dataset_root,
        "window_size": (args.window_size, args.window_size),
        "model_name": "DINOv3",
        "modality": "multi",
        "backbone_type": "dinov3_vits16",
    }
    train_dataset = build_dataset("EarthMiss", "train", **dataset_kwargs)
    val_dataset = build_dataset("EarthMiss", "val", **dataset_kwargs)
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=train_generator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def train_state(run, state_rng):
    if run == "A":
        return "sar"
    if run == "B":
        return "full"
    return "sar" if state_rng.random() < 0.5 else "full"


def validation_states(run):
    return ("sar",) if run == "A" else ("sar", "full")


def update_early_stopping_state(state, *, improved, epoch):
    updated = dict(state)
    if improved:
        updated["bad_validation_count"] = 0
        updated["best_epoch"] = epoch
    else:
        updated["bad_validation_count"] += 1
    return updated


def build_run_metadata(args, train_dataset, val_dataset, train_loader):
    steps_per_epoch = len(train_loader)
    checkpoint_roles = {"best_sar.pth": "primary_deployment"}
    if args.run != "A":
        checkpoint_roles["best_full.pth"] = "diagnostic_only"
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "run": args.run,
        "seed": args.seed,
        "train_tiles": len(train_dataset),
        "val_tiles": len(val_dataset),
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "train_policy": {"A": "sar", "B": "full", "C": "50/50 full-sar"}[
            args.run
        ],
        "model": {
            "segmentation_head": "raw_conv1x1",
            "released_segmentation_head": "conv_bn_relu",
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
        "data_loader": {
            "cached_source_arrays_immutable": True,
            "persistent_workers": args.num_workers > 0,
        },
        "budget": {
            "sampling": "one_crop_per_tile_per_epoch",
            "steps_per_epoch": steps_per_epoch,
            "planned_optimizer_steps": steps_per_epoch * args.epochs,
            "checkpoint_unit": "epoch",
        },
        "evaluation": {
            "checkpoint_selection_metric": "mIoU",
            "checkpoint_selection_support": "pooled_gt_present",
            "selection_split": "val_city_holdout",
            "expected_val_selection_class_ids": VAL_SELECTION_CLASS_IDS,
            "external_comparison_metric": "official_ever_mIoU",
            "external_comparison_support": "fixed_all_8_classes",
            "external_comparison_split": "test_city_holdout",
            "checkpoint_roles": checkpoint_roles,
            "paired_endpoint_rule": "same_checkpoint_and_epoch",
        },
        "early_stopping": {
            "selection_state": "sar",
            "strict_improvement": True,
            "patience_evaluations": args.early_stop_patience_evals,
            "evaluation_interval_epochs": args.eval_interval,
            "disabled": args.early_stop_patience_evals == 0,
        },
    }


@torch.no_grad()
def evaluate(model, loader, state, device, window_size, inference_batch_size):
    model.eval()
    evaluator = EarthMissMetrics()
    availability = canonical_availability(state, batch_size=1, device=device)
    stride = int(window_size * 2 / 3)
    for rgb, sar, label in tqdm(loader, desc=f"Val {state}", leave=False):
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        logits = slide_inference(
            rgb,
            model,
            n_output_channels=8,
            crop_size=(window_size, window_size),
            stride=(stride, stride),
            dsm=sar,
            availability=availability,
            batch_size=inference_batch_size,
        )
        evaluator.update(logits.argmax(dim=1), label)
    metrics = evaluator.compute()
    if metrics["selection_class_ids"] != VAL_SELECTION_CLASS_IDS:
        raise RuntimeError(
            "EarthMiss Val class support changed: expected "
            f"{VAL_SELECTION_CLASS_IDS}, got {metrics['selection_class_ids']}"
        )
    return metrics


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    best,
    metadata,
    *,
    checkpoint_role,
    selection_state=None,
    early_stopping_state=None,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "run": metadata["run"],
            "seed": metadata["seed"],
            "protocol": metadata,
            "checkpoint_role": checkpoint_role,
            "selection_state": selection_state,
            "selection_metric": "mIoU" if selection_state is not None else None,
            "selection_score": (
                best[selection_state] if selection_state is not None else None
            ),
            "early_stopping_state": early_stopping_state,
        },
        path,
    )


def ground_truth_pixel_counts(dataset):
    counts = np.zeros(8, dtype=np.int64)
    for sample in dataset.samples:
        label = dataset._read_label(sample.label_path)
        counts += np.bincount(label.reshape(-1), minlength=9)[:8]
    return counts.tolist()


def audit_datasets(train_dataset, val_dataset):
    rgb, sar, label = train_dataset[0]
    val_gt_pixels = ground_truth_pixel_counts(val_dataset)
    val_selection_class_ids = [
        class_id for class_id, count in enumerate(val_gt_pixels) if count > 0
    ]
    if val_selection_class_ids != VAL_SELECTION_CLASS_IDS:
        raise RuntimeError(
            "EarthMiss Val class support changed: expected "
            f"{VAL_SELECTION_CLASS_IDS}, got {val_selection_class_ids}"
        )
    summary = {
        "train_tiles": len(train_dataset),
        "val_tiles": len(val_dataset),
        "val_gt_pixels": val_gt_pixels,
        "val_selection_class_ids": val_selection_class_ids,
        "first_train_tile": {
            "city": train_dataset.samples[0].city,
            "tile_id": train_dataset.samples[0].tile_id,
            "rgb_shape": list(rgb.shape),
            "sar_shape": list(sar.shape),
            "label_shape": list(label.shape),
            "label_ids": sorted(label.unique().tolist()),
        },
    }
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    if args.window_size <= 0 or args.window_size % 16:
        raise ValueError("--window-size must be a positive multiple of 16")
    if args.epochs <= 0 or args.eval_interval <= 0:
        raise ValueError("--epochs and --eval-interval must be positive")
    if args.early_stop_patience_evals < 0:
        raise ValueError("--early-stop-patience-evals must be non-negative")

    seed_everything(args.seed)
    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    if args.audit_only:
        audit_datasets(train_dataset, val_dataset)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("EarthMiss V1 training requires a CUDA GPU")
    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    output_dir = Path(args.output_root) / f"run_{args.run.lower()}_seed{args.seed}"
    existing_artifacts = (
        output_dir / "metrics.jsonl",
        output_dir / "last.pth",
        output_dir / "best_sar.pth",
        output_dir / "best_full.pth",
    )
    if not args.resume and any(path.exists() for path in existing_artifacts):
        raise FileExistsError(
            f"Refusing to overwrite an existing EarthMiss run: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "last.pth"
    metrics_path = output_dir / "metrics.jsonl"

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
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    start_epoch = 1
    best = {state: float("-inf") for state in validation_states(args.run)}
    early_stopping_state = {"bad_validation_count": 0, "best_epoch": None}
    metadata = build_run_metadata(args, train_dataset, val_dataset, train_loader)

    if args.resume:
        checkpoint = torch.load(last_checkpoint, map_location=device)
        if checkpoint["run"] != args.run or checkpoint["seed"] != args.seed:
            raise ValueError("Resume checkpoint does not match --run/--seed")
        if checkpoint.get("protocol") != metadata:
            raise ValueError("Resume checkpoint does not match the frozen protocol")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint["best"]
        saved_early_stopping = checkpoint.get("early_stopping_state")
        if not isinstance(saved_early_stopping, dict):
            raise ValueError("Resume checkpoint lacks early-stopping state")
        early_stopping_state = dict(saved_early_stopping)
        if (
            args.early_stop_patience_evals > 0
            and early_stopping_state["bad_validation_count"]
            >= args.early_stop_patience_evals
        ):
            raise ValueError("Resume checkpoint has already triggered early stopping")

    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        state_rng = random.Random(args.seed * 100_000 + epoch)
        loss_sum = 0.0
        batch_count = 0
        for rgb, sar, label in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            state = train_state(args.run, state_rng)
            rgb = rgb.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            availability = canonical_availability(
                state, batch_size=rgb.shape[0], device=device
            )
            optimizer.zero_grad()
            logits = model(rgb, sar, availability=availability)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            batch_count += 1
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": loss_sum / batch_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            for state in validation_states(args.run):
                metrics = evaluate(
                    model,
                    val_loader,
                    state,
                    device,
                    args.window_size,
                    args.batch_size * 4,
                )
                record[state] = metrics
                improved = metrics["mIoU"] > best[state]
                if state == "sar":
                    early_stopping_state = update_early_stopping_state(
                        early_stopping_state,
                        improved=improved,
                        epoch=epoch,
                    )
                if improved:
                    best[state] = metrics["mIoU"]
                    save_checkpoint(
                        output_dir / f"best_{state}.pth",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        best,
                        metadata,
                        checkpoint_role=(
                            "primary_deployment"
                            if state == "sar"
                            else "diagnostic_only"
                        ),
                        selection_state=state,
                        early_stopping_state=early_stopping_state,
                    )
            record["early_stopping"] = {
                **early_stopping_state,
                "patience_evaluations": args.early_stop_patience_evals,
                "stop": (
                    args.early_stop_patience_evals > 0
                    and early_stopping_state["bad_validation_count"]
                    >= args.early_stop_patience_evals
                ),
            }
            model.train()

        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scheduler,
            epoch,
            best,
            metadata,
            checkpoint_role="resume_only",
            early_stopping_state=early_stopping_state,
        )
        print(json.dumps(record))
        if record.get("early_stopping", {}).get("stop", False):
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "epoch": epoch,
                        "selection_state": "sar",
                        "best_epoch": early_stopping_state["best_epoch"],
                        "best_mIoU": best["sar"],
                    }
                )
            )
            break


if __name__ == "__main__":
    main()
