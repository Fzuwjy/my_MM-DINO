"""Run one real-data MM-DINO forward, loss, backward and optimizer step."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = PROJECT_ROOT / "tasks" / "segmentation"
for path in (PROJECT_ROOT, SEGMENTATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs import get_cfg
from configs.common_cfg import DATASETS_ROOT, WEIGHTS_ROOT
from datasets import build_dataset
from utils.runtime import autocast_context, create_grad_scaler, forward_batch
from utils.utils import set_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="One-step real-data MM-DINO smoke test")
    parser.add_argument("--dataset-name", default="WHU", choices=["WHU"])
    parser.add_argument("--datasets-root", default=str(DATASETS_ROOT))
    parser.add_argument("--weights-root", default=str(WEIGHTS_ROOT))
    parser.add_argument("--backbone-weights")
    parser.add_argument("--backbone-type", default="dinov3_vits16")
    parser.add_argument("--num-modalities", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--train-split-file",
        default=str(PROJECT_ROOT / "splits" / "whu" / "official_train.txt"),
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache-size", type=int, default=1)
    parser.add_argument("--amp-dtype", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.window_size <= 0 or args.window_size % 16:
        parser.error("--window-size must be a positive multiple of 16")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.cache_size < 0:
        parser.error("--cache-size must be non-negative")
    return args


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("The segmentation smoke test requires a CUDA GPU")
    if args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bfloat16")

    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda", 0)
    cfg = get_cfg(
        "DINOv3",
        args.dataset_name,
        backbone_type=args.backbone_type,
        backbone_weights=args.backbone_weights,
        weights_root=args.weights_root,
        num_modalities=args.num_modalities,
        batch_size=args.batch_size,
        epochs=1,
        window_size=args.window_size,
        base_lr=1e-4,
        scale_lr=False,
    )
    dataset = build_dataset(
        args.dataset_name,
        "train",
        split_file=args.train_split_file,
        model_name="DINOv3",
        modality="multi" if args.num_modalities > 1 else None,
        backbone_type=args.backbone_type,
        datasets_root=args.datasets_root,
        window_size=cfg["window_size"],
        cache_size=args.cache_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    batch = next(iter(loader))

    model = cfg["model"].to(device).train()
    optimizer = cfg["optimizer"]
    scaler = create_grad_scaler(device, args.amp_dtype)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    with autocast_context(device, args.amp_dtype):
        logits, labels = forward_batch(model, batch, device, args.num_modalities)
        loss = cfg["loss_fn"](logits, labels)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite smoke-test loss: {loss.item()}")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    squared_norm = 0.0
    gradient_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            gradient_tensors += 1
            squared_norm += float(parameter.grad.detach().float().pow(2).sum())
    gradient_norm = math.sqrt(squared_norm)
    if not math.isfinite(gradient_norm) or gradient_norm == 0.0:
        raise FloatingPointError(f"Invalid gradient norm: {gradient_norm}")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - started
    inputs = batch[:-1]
    summary = {
        "status": "PASSED",
        "device": torch.cuda.get_device_name(device),
        "amp_dtype": args.amp_dtype,
        "input_shapes": [list(tensor.shape) for tensor in inputs],
        "label_shape": list(labels.shape),
        "label_dtype": str(labels.dtype),
        "label_values": torch.unique(labels).detach().cpu().tolist(),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "gradient_tensors": gradient_tensors,
        "gradient_norm": gradient_norm,
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "dataset_items": len(dataset),
        "backbone_weights": cfg["backbone_weights"],
    }
    print(json.dumps(summary, indent=2))


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
