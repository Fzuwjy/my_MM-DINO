"""Portable single-GPU and DDP training entry point for MM-DINO."""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dinov3.logging import setup_logging
import dinov3.distributed as distributed

from configs import get_cfg
from configs.common_cfg import DATASETS_ROOT, OUTPUT_ROOT, WEIGHTS_ROOT
from datasets import build_dataset
from utils.inference import slide_inference
from utils.metrics import metrics_from_confusion_matrix, update_confusion_matrix
from utils.runtime import (
    DistributedEvalSampler,
    append_jsonl,
    atomic_torch_save,
    autocast_context,
    checkpoint_payload,
    create_grad_scaler,
    forward_batch,
    get_device,
    git_metadata,
    initialize_distributed,
    load_training_checkpoint,
    reduce_confusion_matrix,
    write_json,
)
from utils.utils import set_seed


LOGGER_NAME = "dinov3seg"


def build_loaders(args, cfg):
    modality = "multi" if args.num_modalities > 1 else None
    common = {
        "window_size": cfg["window_size"],
        "model_name": args.model_name,
        "modality": modality,
        "backbone_type": args.backbone_type,
        "datasets_root": args.datasets_root,
        "cache_size": args.cache_size,
    }
    train_dataset = build_dataset(
        args.dataset_name,
        "train",
        split_file=args.train_split_file,
        **common,
    )
    eval_dataset = build_dataset(
        args.dataset_name,
        args.eval_split,
        split_file=args.eval_split_file,
        **common,
    )

    if distributed.is_enabled():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
            seed=args.seed,
        )
        eval_sampler = DistributedEvalSampler(eval_dataset)
        shuffle = False
    else:
        train_sampler = None
        eval_sampler = None
        shuffle = True

    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": args.pin_memory and torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        sampler=train_sampler,
        drop_last=False,
        **loader_kwargs,
    )
    # Sliding-window evaluation currently accepts one full image per loader batch.
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        sampler=eval_sampler,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, eval_loader


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, epoch, args, device):
    model.train()
    if distributed.is_enabled():
        loader.sampler.set_epoch(epoch)

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    num_batches = 0
    progress = tqdm(loader, disable=not distributed.is_main_process())

    for step, batch in enumerate(progress):
        group_start = (step // args.grad_accum_steps) * args.grad_accum_steps
        accumulation_divisor = min(
            args.grad_accum_steps,
            len(loader) - group_start,
        )
        should_step = (step + 1) % args.grad_accum_steps == 0 or step + 1 == len(loader)
        sync_context = (
            model.no_sync()
            if hasattr(model, "no_sync") and not should_step
            else nullcontext()
        )
        with sync_context:
            with autocast_context(device, args.amp_dtype):
                logits, label = forward_batch(model, batch, device, args.num_modalities)
                loss = loss_fn(logits, label)
                scaled_loss = loss / accumulation_divisor

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, step={step}: {loss.item()}"
                )
            scaler.scale(scaled_loss).backward()

        if should_step:
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach())
        num_batches += 1
        if distributed.is_main_process():
            progress.set_description(f"Epoch {epoch}/{args.epochs} Loss {loss.item():.4f}")

    totals = torch.tensor([total_loss, num_batches], dtype=torch.float64, device=device)
    if distributed.is_enabled():
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
    return float(totals[0] / totals[1].clamp_min(1))


@torch.no_grad()
def evaluate(model, loader, cfg, args, device):
    model.eval()
    labels = cfg["labels"]
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    window_size = cfg["window_size"]
    stride_value = max(1, int(window_size[0] * args.eval_stride_ratio))

    progress = tqdm(loader, disable=not distributed.is_main_process())
    for batch in progress:
        if args.num_modalities > 1:
            image, auxiliary, target = batch
            image = image.to(device, non_blocking=True)
            auxiliary = auxiliary.to(device, non_blocking=True)
        else:
            image, target = batch
            image = image.to(device, non_blocking=True)
            auxiliary = None

        with autocast_context(device, args.amp_dtype):
            prediction = slide_inference(
                image,
                model,
                dsm=auxiliary,
                n_output_channels=len(labels),
                crop_size=window_size,
                stride=(stride_value, stride_value),
                batch_size=args.inference_batch_size,
            )
        prediction = prediction.argmax(dim=1).numpy()
        update_confusion_matrix(confusion, prediction, target.numpy())

    confusion = reduce_confusion_matrix(confusion, device)
    miou, f1, kappa, accuracy = metrics_from_confusion_matrix(confusion, labels)
    return {
        "MIoU": float(miou),
        "F1": float(f1),
        "Kappa": float(kappa),
        "Acc": float(accuracy),
    }


def resolve_run_dir(args):
    if args.resume:
        return Path(args.resume).expanduser().resolve().parent
    root = Path(args.output_root).expanduser().resolve()
    run_name = args.run_name or (
        f"{args.dataset_name}_{args.backbone_type}_m{args.num_modalities}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    return root / args.model_name / run_name


def run(args):
    initialize_distributed("training")
    device = get_device()
    rank = distributed.get_rank() if distributed.is_enabled() else 0
    set_seed(args.seed + rank, deterministic=args.deterministic)

    if args.amp_dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16; use --amp-dtype fp16 or none")
    if args.eval_split == "test" and distributed.is_main_process():
        print(
            "WARNING: test split is being used during training. This is only suitable "
            "for reproducing the official protocol, not for formal model selection."
        )

    cfg = get_cfg(
        args.model_name,
        args.dataset_name,
        backbone_type=args.backbone_type,
        backbone_weights=args.backbone_weights,
        weights_root=args.weights_root,
        num_modalities=args.num_modalities,
        use_lora=args.use_lora,
        r=args.lora_rank,
        batch_size=args.batch_size,
        epochs=args.epochs,
        window_size=args.window_size,
        base_lr=args.learning_rate,
        weight_decay=args.weight_decay,
        scale_lr=args.scale_lr,
    )
    train_loader, eval_loader = build_loaders(args, cfg)

    model = cfg["model"].to(device)
    optimizer = cfg["optimizer"]
    scheduler = cfg["scheduler"]
    scaler = create_grad_scaler(device, args.amp_dtype)
    start_epoch = 1
    best_miou = 0.0
    if args.resume:
        start_epoch, best_miou, saved_args = load_training_checkpoint(
            Path(args.resume).expanduser().resolve(),
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        architecture_keys = (
            "model_name",
            "dataset_name",
            "num_modalities",
            "backbone_type",
            "use_lora",
            "lora_rank",
            "window_size",
            "epochs",
        )
        mismatches = {
            key: (saved_args[key], getattr(args, key))
            for key in architecture_keys
            if key in saved_args and saved_args[key] != getattr(args, key)
        }
        if mismatches:
            raise ValueError(f"Resume arguments do not match the checkpoint: {mismatches}")
        if start_epoch > args.epochs:
            raise ValueError(
                f"Checkpoint already reached epoch {start_epoch - 1}, but --epochs is {args.epochs}"
            )

    if distributed.is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=args.find_unused_parameters,
        )

    run_dir = resolve_run_dir(args)
    if distributed.is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(output=str(run_dir), level=logging.INFO, name=LOGGER_NAME)
        world_size = distributed.get_world_size() if distributed.is_enabled() else 1
        config_name = (
            f"resume_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if args.resume
            else "run_config.json"
        )
        write_json(
            run_dir / config_name,
            {
                **vars(args),
                "resolved_backbone_weights": cfg["backbone_weights"],
                "resolved_learning_rate": cfg["learning_rate"],
                "world_size": world_size,
                "effective_batch_size": args.batch_size * world_size * args.grad_accum_steps,
                "device": str(device),
                "git": git_metadata(PROJECT_ROOT),
            },
        )
    logger = logging.getLogger(LOGGER_NAME)

    for epoch in range(start_epoch, args.epochs + 1):
        average_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            cfg["loss_fn"],
            epoch,
            args,
            device,
        )
        scheduler.step()
        if distributed.is_main_process():
            logger.info("Epoch %d/%d - average loss %.6f", epoch, args.epochs, average_loss)
            append_jsonl(
                run_dir / "train_metrics.jsonl",
                {
                    "epoch": epoch,
                    "avg_loss": average_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
            )

        should_evaluate = epoch % args.eval_interval == 0 or epoch == args.epochs
        if should_evaluate:
            eval_metrics = evaluate(model, eval_loader, cfg, args, device)
            if distributed.is_main_process():
                eval_record = {"epoch": epoch, "split": args.eval_split, **eval_metrics}
                append_jsonl(run_dir / "eval_metrics.jsonl", eval_record)
                logger.info("Evaluation: %s", eval_record)
                if eval_metrics["MIoU"] > best_miou:
                    best_miou = eval_metrics["MIoU"]
                    atomic_torch_save(
                        checkpoint_payload(
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch,
                            best_miou,
                            args,
                        ),
                        run_dir / "best.pth",
                    )

        if distributed.is_main_process():
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_miou,
                args,
            )
            atomic_torch_save(payload, run_dir / "last.pth")
            if args.save_interval > 0 and epoch % args.save_interval == 0:
                atomic_torch_save(payload, run_dir / f"epoch_{epoch:03d}.pth")
        if distributed.is_enabled():
            torch.distributed.barrier()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train MM-DINO segmentation models")
    parser.add_argument("--model-name", default="DINOv3")
    parser.add_argument(
        "--dataset-name",
        choices=["WHU", "Potsdam", "Vaihingen", "EarthMiss", "YYYJ"],
        default="WHU",
    )
    parser.add_argument("--num-modalities", type=int, choices=[1, 2], default=2)
    parser.add_argument("--backbone-type", default="dinov3_vits16")
    parser.add_argument("--backbone-weights")
    parser.add_argument("--weights-root", default=str(WEIGHTS_ROOT))
    parser.add_argument("--datasets-root", default=str(DATASETS_ROOT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--train-split-file")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--eval-split-file")
    parser.add_argument("--run-name")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cache-size",
        type=int,
        default=2,
        help="Full images cached per DataLoader worker for WHU/EarthMiss; 0 disables",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Periodic checkpoint interval; 0 keeps only last.pth and best.pth",
    )
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--eval-stride-ratio", type=float, default=2.0 / 3.0)
    parser.add_argument("--amp-dtype", choices=["none", "fp16", "bf16"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume")

    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-rank", type=int, default=3)
    parser.add_argument("--scale-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)

    positive_ints = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "window_size": args.window_size,
        "grad_accum_steps": args.grad_accum_steps,
        "eval_interval": args.eval_interval,
        "inference_batch_size": args.inference_batch_size,
    }
    invalid = [name for name, value in positive_ints.items() if value <= 0]
    if invalid:
        parser.error(f"Expected positive values for: {', '.join(invalid)}")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.cache_size < 0:
        parser.error("--cache-size must be non-negative")
    if args.save_interval < 0:
        parser.error("--save-interval must be non-negative")
    if not 0 < args.eval_stride_ratio <= 1:
        parser.error("--eval-stride-ratio must be in (0, 1]")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        run(parsed_args)
    finally:
        if distributed.is_enabled():
            distributed.disable()
