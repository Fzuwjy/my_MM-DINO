"""Evaluate a WHU ViT-L LoRA checkpoint with the released training protocol.

The public standalone ``test.py`` uses sliding-window microbatch 8, whereas the
released training loop evaluates every five epochs with ``batch_size * 4 ==
32``.  This external harness loads a checkpoint and directly calls the
unmodified ``train_multi.test`` function so comparison with training-time and
paper checkpoint results uses the same protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


SEED = 42
RELEASED_CONFIG_BATCH_SIZE = 8
RELEASED_INFERENCE_BATCH_SIZE = 32
REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"


def seed_model_initialization() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate WHU Table III Multi ViT-L LoRA checkpoint"
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation result: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    segmentation_root = str(SEGMENTATION_ROOT)
    repo_root = str(REPO_ROOT)
    if segmentation_root not in sys.path:
        sys.path.insert(0, segmentation_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import dinov3.distributed as distributed
    import train_multi as official_trainer
    from configs import get_cfg
    from datasets import build_dataset

    os.environ["NCCL_TIMEOUT"] = "1200"
    distributed.enable(overwrite=True)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)

    seed_model_initialization()
    cfg = get_cfg(
        "DINOv3",
        "WHU",
        num_modalities=2,
        use_lora=True,
        r=3,
        backbone_type="dinov3_vitl16",
    )
    if cfg.get("batch_size") != RELEASED_CONFIG_BATCH_SIZE:
        raise RuntimeError(
            f"Released config batch changed from {RELEASED_CONFIG_BATCH_SIZE} "
            f"to {cfg.get('batch_size')}"
        )

    dataset = build_dataset(
        "WHU",
        "test",
        window_size=cfg["window_size"],
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vitl16",
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=False
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=1,
        persistent_workers=True,
    )

    model = cfg["model"].to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("Expected a checkpoint dictionary containing 'model'")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    official_trainer.MODEL_NAME = "DINOv3"
    official_trainer.DATASET_NAME = "WHU"
    official_trainer.NUM_MODALITIES = 2

    print(f"checkpoint={checkpoint_path}")
    print("evaluation_function=train_multi.test")
    print(f"evaluation_inference_batch_size={RELEASED_INFERENCE_BATCH_SIZE}")
    metrics = official_trainer.test(
        model,
        loader,
        cfg,
        is_distributed=distributed.is_enabled(),
    )

    result = {
        "split": "test",
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "model_name": "DINOv3",
        "dataset_name": "WHU",
        "num_modalities": 2,
        "backbone_type": "dinov3_vitl16",
        "use_lora": True,
        "lora_rank": 3,
        "evaluation_function": "tasks/segmentation/train_multi.py:test",
        "evaluation_inference_batch_size": RELEASED_INFERENCE_BATCH_SIZE,
        **{name: float(value) for name, value in metrics.items()},
    }
    if distributed.is_main_process():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"evaluation_result={output_path}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
