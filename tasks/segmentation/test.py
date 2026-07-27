"""Final evaluation entry point for MM-DINO checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dinov3.distributed as distributed

from configs import get_cfg
from configs.common_cfg import DATASETS_ROOT, OUTPUT_ROOT, WEIGHTS_ROOT
from datasets import build_dataset
from utils import plot_confusion_matrix, save_prediction_as_image
from utils.inference import slide_inference
from utils.metrics import (
    metrics_from_confusion_matrix,
    per_class_metrics_from_confusion_matrix,
    update_confusion_matrix,
)
from utils.runtime import (
    DistributedEvalSampler,
    autocast_context,
    get_device,
    reduce_confusion_matrix,
    write_json,
)
from utils.utils import set_seed


def initialize_distributed():
    if not torch.cuda.is_available():
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise RuntimeError("Multi-process evaluation requires CUDA/NCCL")
        return
    os.environ.setdefault("NCCL_TIMEOUT", "1200")
    distributed.enable(overwrite=False, nccl_async_error_handling=True)


@torch.no_grad()
def run(args):
    initialize_distributed()
    device = get_device()
    rank = distributed.get_rank() if distributed.is_enabled() else 0
    set_seed(args.seed + rank, deterministic=True)

    cfg = get_cfg(
        args.model_name,
        args.dataset_name,
        backbone_type=args.backbone_type,
        backbone_weights=args.backbone_weights,
        weights_root=args.weights_root,
        num_modalities=args.num_modalities,
        use_lora=args.use_lora,
        r=args.lora_rank,
        batch_size=1,
        epochs=1,
        window_size=args.window_size,
        scale_lr=False,
    )
    model = cfg["model"].to(device)
    checkpoint = torch.load(
        Path(args.checkpoint_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    dataset = build_dataset(
        args.dataset_name,
        args.split,
        window_size=cfg["window_size"],
        model_name=args.model_name,
        modality="multi" if args.num_modalities > 1 else None,
        backbone_type=args.backbone_type,
        datasets_root=args.datasets_root,
        split_file=args.split_file,
    )
    sampler = DistributedEvalSampler(dataset) if distributed.is_enabled() else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else OUTPUT_ROOT / "evaluation" / f"{args.dataset_name}_{checkpoint_path.stem}"
    )
    if distributed.is_main_process() or args.save_predictions:
        output_dir.mkdir(parents=True, exist_ok=True)

    labels = cfg["labels"]
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    stride_value = max(1, int(cfg["window_size"][0] * args.eval_stride_ratio))
    progress = tqdm(loader, disable=not distributed.is_main_process())
    for sample_index, batch in enumerate(progress):
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
                crop_size=cfg["window_size"],
                stride=(stride_value, stride_value),
                batch_size=args.inference_batch_size,
            )
        prediction = prediction.argmax(dim=1).numpy()
        target_numpy = target.numpy()
        update_confusion_matrix(confusion, prediction, target_numpy)
        if args.save_predictions:
            save_prediction_as_image(
                prediction,
                target_numpy,
                str(output_dir),
                f"rank{rank}_{sample_index}",
                dataset_name=args.dataset_name,
            )

    confusion = reduce_confusion_matrix(confusion, device)
    miou, f1, kappa, accuracy = metrics_from_confusion_matrix(confusion, labels)
    result = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "MIoU": float(miou),
        "F1": float(f1),
        "Kappa": float(kappa),
        "Acc": float(accuracy),
        "per_class": per_class_metrics_from_confusion_matrix(confusion, labels),
        "confusion_matrix": confusion.tolist(),
    }
    if distributed.is_main_process():
        write_json(output_dir / "metrics.json", result)
        plot_confusion_matrix(
            confusion,
            labels,
            str(output_dir / "confusion_matrix.png"),
        )
        print(result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate an MM-DINO checkpoint")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model-name", default="DINOv3")
    parser.add_argument(
        "--dataset-name",
        choices=["WHU", "Potsdam", "Vaihingen", "EarthMiss", "YYYJ"],
        default="WHU",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--split-file")
    parser.add_argument("--num-modalities", type=int, choices=[1, 2], default=2)
    parser.add_argument("--backbone-type", default="dinov3_vits16")
    parser.add_argument("--backbone-weights")
    parser.add_argument("--weights-root", default=str(WEIGHTS_ROOT))
    parser.add_argument("--datasets-root", default=str(DATASETS_ROOT))
    parser.add_argument("--output-dir")
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--eval-stride-ratio", type=float, default=2.0 / 3.0)
    parser.add_argument("--amp-dtype", choices=["none", "fp16", "bf16"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lora-rank", type=int, default=3)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)
    if min(args.window_size, args.inference_batch_size) <= 0:
        parser.error("Window and inference batch sizes must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
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
