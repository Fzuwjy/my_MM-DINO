"""Train task-aligned WHU-OPT-SAR missing-modality MM-DINO baselines.

The external task contract is optical+SAR available for training and SAR-only
deployment.  Run A is a SAR-only lower bound, Run B is naive Full-to-SAR, and
Run C uses homogeneous 50/50 Full/SAR batches.  All runs use the published
80/20 WHU split and fixed-epoch Test reporting without Test-based checkpoint
selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.io import imread
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
from utils.inference import slide_inference  # noqa: E402
from utils.pooled_segmentation_metrics import (  # noqa: E402
    PooledSegmentationMetrics,
)


DEFAULT_DATASET_ROOT = "/root/autodl-tmp/mm-dino/datasets/whu-opt-sar"
DEFAULT_WEIGHTS = (
    "/root/autodl-tmp/mm-dino/weights/"
    "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
DEFAULT_OUTPUT_ROOT = (
    "/root/autodl-tmp/mm-dino/outputs/whu-missing-baseline-nirrg"
)
CLASS_NAMES = (
    "Farmland",
    "City",
    "Village",
    "Water",
    "Forest",
    "Road",
    "Others",
)
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = NUM_CLASSES
PROTOCOL_REVISION = "whu_missing_baseline_nirrg_v1"


def parse_epoch_set(value):
    epochs = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not epochs or epochs[0] <= 0:
        raise argparse.ArgumentTypeError("epoch lists must contain positive integers")
    return epochs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("A", "B", "C"), required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--split-dir", default=str(REPO_ROOT / "splits" / "whu")
    )
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-size", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--eval-epochs", type=parse_epoch_set, default=(50,)
    )
    parser.add_argument(
        "--checkpoint-epochs", type=parse_epoch_set, default=(25, 50)
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run one real batch through SAR and Full forward/backward paths.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit data/splits/input bands without loading the model.",
    )
    return parser.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_state(run, state_rng):
    if run == "A":
        return "sar"
    if run == "B":
        return "full"
    return "sar" if state_rng.random() < 0.5 else "full"


def prepare_training_label(label, device):
    """Match the class-index dtype required by the released segmentation loss."""
    return label.to(device, dtype=torch.long, non_blocking=True)


def build_loaders(args):
    split_dir = Path(args.split_dir)
    common = {
        "dataset_root": args.dataset_root,
        "window_size": (args.window_size, args.window_size),
        "model_name": "DINOv3",
        "modality": "multi",
        "backbone_type": "dinov3_vits16",
        "optical_bands": "nir-r-g",
        "cache_size": args.cache_size,
        "mask_padding_ignore": True,
    }
    train_dataset = build_dataset(
        "WHU",
        "train",
        split_file=str(split_dir / "official_train.txt"),
        **common,
    )
    test_dataset = build_dataset(
        "WHU",
        "test",
        split_file=str(split_dir / "official_test.txt"),
        **common,
    )
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
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=True,
        persistent_workers=False,
    )
    return train_dataset, test_dataset, train_loader, test_loader, train_generator


def build_metadata(args, train_dataset, test_dataset, train_loader):
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "run": args.run,
        "seed": args.seed,
        "task_contract": {"train": "S+O allowed", "deployment": "S only"},
        "train_policy": {
            "A": "SAR-only lower bound",
            "B": "Full-only training, SAR deployment",
            "C": "batchwise 50/50 Full-SAR, SAR deployment",
        }[args.run],
        "data": {
            "dataset": "WHU-OPT-SAR",
            "split": "published 80 train / 20 test original scenes",
            "train_scenes": len(train_dataset.rgb_files),
            "test_scenes": len(test_dataset.rgb_files),
            "optical_external_contract": "NIR-R-G",
            "optical_band_indices": list(train_dataset.optical_band_indices),
            "sar_external_contract": "single-channel",
            "label_mapping": "0 ignore; 10..70 -> 0..6",
            "worker_cache_capacity": args.cache_size,
        },
        "model": {
            "backbone": "DINOv3 ViT-S/16 LVD",
            "backbone_frozen": True,
            "num_modalities": 2,
            "canonical_slots": ["optical", "sar"],
            "segmentation_head": "raw_conv1x1",
        },
        "normalization": {
            "optical": "ImageNet channel constants after NIR-R-G selection",
            "sar": "uint8 divided by 255; DINO input repeats one channel to three",
        },
        "augmentation": {
            "random_resize_ratio": [0.5, 2.0],
            "random_crop": [args.window_size, args.window_size],
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.5,
            "padding_label": IGNORE_INDEX,
        },
        "optimization": {
            "loss": "SoftCE(smoothing=0.05)+Dice(smoothing=0.05)",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.01,
            "scheduler": "CosineAnnealingLR",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "persistent_train_workers": args.num_workers > 0,
            "steps_per_epoch": len(train_loader),
            "planned_optimizer_steps": len(train_loader) * args.epochs,
        },
        "evaluation": {
            "split": "published 20-scene Test",
            "endpoint": "canonical SAR-only",
            "window_size": args.window_size,
            "stride": int(args.window_size * 2 / 3),
            "inference_batch_size": args.inference_batch_size,
            "aggregation": "pooled confusion matrix over 7 classes",
            "metrics": ["mIoU", "mF1"],
            "eval_epochs": list(args.eval_epochs),
            "checkpoint_epochs": list(args.checkpoint_epochs),
            "selection": "none; fixed final epoch is primary",
        },
        "comparability": {
            "MetaRS_WHU": "STARS-paper rerun only; no public MetaRS WHU recipe",
            "STARS_WHU": "paper task contract aligned; reported-only protocol",
            "MM_DINO_paper_WHU": "different task: Full train and Full test",
        },
    }


def audit_datasets(train_dataset, test_dataset):
    train_names = {Path(path).name for path in train_dataset.rgb_files}
    test_names = {Path(path).name for path in test_dataset.rgb_files}
    if len(train_names) != 80 or len(test_names) != 20 or train_names & test_names:
        raise RuntimeError("WHU published 80/20 split audit failed")

    raw_optical = imread(train_dataset.rgb_files[0])
    raw_sar = imread(train_dataset.sar_files[0])
    optical, sar, label = test_dataset[0]
    summary = {
        "train_original_scenes": len(train_names),
        "test_original_scenes": len(test_names),
        "train_dataset_samples_per_epoch": len(train_dataset),
        "split_overlap": 0,
        "first_raw": {
            "optical_shape": list(raw_optical.shape),
            "optical_dtype": str(raw_optical.dtype),
            "sar_shape": list(raw_sar.shape),
            "sar_dtype": str(raw_sar.dtype),
        },
        "external_input_contract": {
            "optical": "NIR-R-G",
            "optical_indices": list(train_dataset.optical_band_indices),
            "sar": "single-channel",
        },
        "first_tensor": {
            "optical_shape": list(optical.shape),
            "sar_shape": list(sar.shape),
            "label_shape": list(label.shape),
            "label_ids": sorted(np.unique(label).tolist()),
        },
    }
    if raw_optical.ndim != 3 or raw_optical.shape[2] != 4:
        raise RuntimeError("WHU optical source must be the official four-band TIFF")
    if raw_sar.ndim != 2 or sar.shape[0] != 1:
        raise RuntimeError("WHU SAR source must be single-channel")
    print(json.dumps(summary, indent=2))


@torch.no_grad()
def evaluate(model, loader, device, window_size, inference_batch_size):
    model.eval()
    evaluator = PooledSegmentationMetrics(NUM_CLASSES, IGNORE_INDEX)
    availability = canonical_availability("sar", batch_size=1, device=device)
    stride = int(window_size * 2 / 3)
    for optical, sar, label in tqdm(loader, desc="Test SAR", leave=False):
        optical = optical.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        logits = slide_inference(
            optical,
            model,
            n_output_channels=NUM_CLASSES,
            crop_size=(window_size, window_size),
            stride=(stride, stride),
            dsm=sar,
            availability=availability,
            batch_size=inference_batch_size,
        )
        evaluator.update(logits.argmax(dim=1), label)
    metrics = evaluator.compute()
    expected = list(range(NUM_CLASSES))
    if metrics["gt_present_class_ids"] != expected:
        raise RuntimeError(
            f"WHU Test class support changed: expected {expected}, "
            f"got {metrics['gt_present_class_ids']}"
        )
    metrics["class_names"] = list(CLASS_NAMES)
    return metrics


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    train_generator,
    epoch,
    metadata,
    *,
    role,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_generator_state": train_generator.get_state(),
            "epoch": epoch,
            "run": metadata["run"],
            "seed": metadata["seed"],
            "protocol": metadata,
            "checkpoint_role": role,
        },
        path,
    )


def smoke_training_paths(model, loader, criterion, device):
    model.train()
    optical, sar, label = next(iter(loader))
    optical = optical.to(device, non_blocking=True)
    sar = sar.to(device, non_blocking=True)
    label = prepare_training_label(label, device)
    result = {
        "optical_shape": list(optical.shape),
        "sar_shape": list(sar.shape),
        "label_shape": list(label.shape),
        "label_dtype": str(label.dtype),
        "paths": {},
    }
    for state in ("sar", "full"):
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        availability = canonical_availability(
            state, batch_size=optical.shape[0], device=device
        )
        logits = model(optical, sar, availability=availability)
        loss = criterion(logits, label)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {state} smoke loss")
        loss.backward()
        torch.cuda.synchronize(device)
        result["paths"][state] = {
            "logits_shape": list(logits.shape),
            "loss": float(loss.detach()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
    print(json.dumps(result, indent=2))


def main(argv=None):
    args = parse_args(argv)
    if args.audit_only and args.smoke_only:
        raise ValueError("--audit-only and --smoke-only are mutually exclusive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.window_size <= 0 or args.window_size % 16:
        raise ValueError("--window-size must be a positive multiple of 16")
    if args.cache_size <= 0 or args.num_workers < 0:
        raise ValueError("--cache-size must be positive and --num-workers non-negative")
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive")
    if max(args.eval_epochs + args.checkpoint_epochs) > args.epochs:
        raise ValueError("eval/checkpoint epochs cannot exceed --epochs")
    if args.epochs not in args.eval_epochs or args.epochs not in args.checkpoint_epochs:
        raise ValueError("the fixed final epoch must be evaluated and checkpointed")
    if not set(args.eval_epochs).issubset(args.checkpoint_epochs):
        raise ValueError("every evaluated epoch must also have a permanent checkpoint")

    seed_everything(args.seed)
    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
        train_generator,
    ) = build_loaders(args)
    if args.audit_only:
        audit_datasets(train_dataset, test_dataset)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("WHU missing-modality training requires a CUDA GPU")

    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    device = torch.device("cuda")
    model = build_model(
        model_name="DINOv3",
        backbone_weights=str(weights_path),
        backbone_type="dinov3_vits16",
        freeze_backbone=True,
        n_classes=NUM_CLASSES,
        use_lora=False,
        r=3,
        num_modalities=2,
        raw_logits=True,
    ).to(device)
    criterion = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=IGNORE_INDEX),
        DiceLoss(smooth=0.05, ignore_index=IGNORE_INDEX),
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
    if args.smoke_only:
        smoke_training_paths(model, train_loader, criterion, device)
        return

    output_dir = Path(args.output_root) / f"run_{args.run.lower()}_seed{args.seed}"
    last_path = output_dir / "last.pth"
    metrics_path = output_dir / "metrics.jsonl"
    if not args.resume and (last_path.exists() or metrics_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(args, train_dataset, test_dataset, train_loader)
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(last_path, map_location=device)
        if checkpoint.get("protocol") != metadata:
            raise ValueError("Resume checkpoint does not match the frozen protocol")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        train_generator.set_state(checkpoint["train_generator_state"])
        start_epoch = int(checkpoint["epoch"]) + 1

    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        state_rng = random.Random(args.seed * 100_000 + epoch)
        loss_sum = 0.0
        batch_count = 0
        state_batches = {"sar": 0, "full": 0}
        for optical, sar, label in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}"
        ):
            state = train_state(args.run, state_rng)
            state_batches[state] += 1
            optical = optical.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            label = prepare_training_label(label, device)
            availability = canonical_availability(
                state, batch_size=optical.shape[0], device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(optical, sar, availability=availability)
            loss = criterion(logits, label)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            batch_count += 1
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": loss_sum / batch_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "state_batches": state_batches,
        }
        if epoch in args.eval_epochs:
            record["sar_test"] = evaluate(
                model,
                test_loader,
                device,
                args.window_size,
                args.inference_batch_size,
            )
            model.train()

        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if epoch in args.checkpoint_epochs:
            save_checkpoint(
                output_dir / f"epoch_{epoch}.pth",
                model,
                optimizer,
                scheduler,
                train_generator,
                epoch,
                metadata,
                role="fixed_final_primary" if epoch == args.epochs else "fixed_diagnostic",
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            train_generator,
            epoch,
            metadata,
            role="resume_only",
        )
        print(json.dumps(record))


if __name__ == "__main__":
    main()
