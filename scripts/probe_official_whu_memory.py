"""One-batch memory probes for the unmodified official WHU training path.

Run this script with ``torchrun --standalone --nproc_per_node=1``.  The training
phase performs exactly one FP32 batch-8 optimizer step.  The evaluation phase
runs official sliding-window inference on one test image and defaults to the
author's inference microbatch of ``batch_size * 4 == 32``.  Nothing is saved.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import dinov3.distributed as distributed  # noqa: E402
from configs import get_cfg  # noqa: E402
from datasets import build_dataset  # noqa: E402
from scripts.whu_cache_compat import CACHE_CAPACITY, install_whu_cache_compat  # noqa: E402
from scripts.whu_label_dtype_compat import install_whu_label_dtype_compat  # noqa: E402
from utils.inference import slide_inference  # noqa: E402
from utils.utils import set_seed  # noqa: E402


def preseed_model() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def memory_gib(value: int) -> float:
    return value / 1024**3


def report_memory(prefix: str, device: torch.device) -> None:
    torch.cuda.synchronize(device)
    print(f"{prefix}_peak_allocated_gib={memory_gib(torch.cuda.max_memory_allocated(device)):.3f}")
    print(f"{prefix}_peak_reserved_gib={memory_gib(torch.cuda.max_memory_reserved(device)):.3f}")


def build_official_components():
    os.environ["NCCL_TIMEOUT"] = "1200"
    distributed.enable(overwrite=True)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))

    cfg = get_cfg(
        "DINOv3",
        "WHU",
        num_modalities=2,
        use_lora=False,
        r=3,
        backbone_type="dinov3_vits16",
    )
    # Preserve the author's original post-construction seed reset.
    set_seed(SEED)
    return cfg, device


def wrap_model(cfg, device):
    model = cfg["model"].to(device)
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        find_unused_parameters=True,
    )


def probe_train(cfg, device) -> None:
    dataset = build_dataset(
        "WHU",
        "train",
        window_size=cfg["window_size"],
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
    )
    print(
        "train_cache_capacities="
        f"{dataset.rgb_cache.capacity},{dataset.label_cache.capacity},"
        f"{dataset.sar_cache.capacity}"
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
    )
    sampler.set_epoch(1)
    model = wrap_model(cfg, device)
    model.train()
    optimizer = cfg["optimizer"]
    loss_fn = cfg["loss_fn"]

    image, sar, label = next(iter(loader))
    print(f"train_batch_shapes={tuple(image.shape)},{tuple(sar.shape)},{tuple(label.shape)}")
    print(f"train_label_dtype={label.dtype}")
    image, sar, label = image.to(device), sar.to(device), label.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad()
    logits = model(image, sar)
    loss = loss_fn(logits, label)
    loss.backward()
    optimizer.step()
    print(f"train_loss={float(loss.detach()):.6f}")
    report_memory("train", device)


@torch.no_grad()
def probe_eval(cfg, device, inference_batch_size: int) -> None:
    dataset = build_dataset(
        "WHU",
        "test",
        window_size=cfg["window_size"],
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=1,
        persistent_workers=True,
    )
    model = wrap_model(cfg, device)
    model.eval()
    image, sar, _ = next(iter(loader))
    image, sar = image.to(device), sar.to(device)
    stride = int(cfg["window_size"][0] * 2 / 3)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    prediction = slide_inference(
        image,
        model,
        dsm=sar,
        n_output_channels=len(cfg["labels"]),
        crop_size=cfg["window_size"],
        stride=(stride, stride),
        batch_size=inference_batch_size,
    )
    print(f"eval_image_shape={tuple(image.shape)}")
    print(f"eval_prediction_shape={tuple(prediction.shape)}")
    print(f"eval_inference_batch_size={inference_batch_size}")
    report_memory("eval", device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "eval"), required=True)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the faithful WHU memory probe")
    preseed_model()
    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    cfg, device = build_official_components()
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(device)}")
    if args.phase == "train":
        probe_train(cfg, device)
    else:
        probe_eval(cfg, device, args.inference_batch_size)


if __name__ == "__main__":
    main()
