"""Evaluate WHU auxiliary counterfactuals with the released test protocol.

Each invocation evaluates one condition so long-running server commands remain
explicit and recoverable.  The released task code is imported but not edited.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.aux_diagnostics_common import (
    AuxiliaryConditionDataset,
    ScaledSampleAdapter,
    modality_weight_summary,
)


SEED = 42
RELEASED_CONFIG_BATCH_SIZE = 8
RELEASED_INFERENCE_BATCH_SIZE = 32
CONDITIONS = (
    "normal",
    "aux-mean",
    "aux-shuffle",
    "aux-feature-off",
    "aux-weight-scale",
)


def seed_model_initialization(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one WHU auxiliary counterfactual condition"
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--aux-weight-scale", type=float)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optional non-formal smoke-test subset; omit for full evaluation",
    )
    return parser.parse_args()


def _condition_auxiliary_scale(args: argparse.Namespace) -> float | None:
    if args.condition == "aux-feature-off":
        if args.aux_weight_scale is not None:
            raise ValueError("aux-feature-off fixes the auxiliary scale at zero")
        return 0.0
    if args.condition == "aux-weight-scale":
        if args.aux_weight_scale is None:
            raise ValueError("aux-weight-scale requires --aux-weight-scale")
        if args.aux_weight_scale < 0.0:
            raise ValueError("--aux-weight-scale must be non-negative")
        return float(args.aux_weight_scale)
    if args.aux_weight_scale is not None:
        raise ValueError(
            "--aux-weight-scale is only valid with condition aux-weight-scale"
        )
    return None


def _dataset_metadata(dataset: Any) -> dict[str, Any]:
    if isinstance(dataset, AuxiliaryConditionDataset):
        return {
            "mode": dataset.mode,
            "permutation": dataset.permutation,
            "permutation_offset": dataset.permutation_offset,
        }
    return {"mode": "normal", "permutation": None, "permutation_offset": None}


def main() -> None:
    args = parse_args()
    auxiliary_scale = _condition_auxiliary_scale(args)
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation result: {output_path}")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")
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

    seed_model_initialization(args.seed)
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

    released_dataset = build_dataset(
        "WHU",
        "test",
        window_size=cfg["window_size"],
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vitl16",
    )
    if args.condition in ("aux-mean", "aux-shuffle"):
        dataset: Any = AuxiliaryConditionDataset(
            released_dataset,
            args.condition,
            seed=args.seed,
        )
    else:
        dataset = released_dataset
    condition_metadata = _dataset_metadata(dataset)
    full_dataset_length = len(dataset)
    if args.max_images is not None:
        dataset = torch.utils.data.Subset(
            dataset,
            list(range(min(args.max_images, len(dataset)))),
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

    released_weight_summary = modality_weight_summary(model.adapter)
    effective_weight_summary = released_weight_summary
    if auxiliary_scale is not None:
        effective_weight_summary = modality_weight_summary(
            model.adapter,
            auxiliary_scale=auxiliary_scale,
        )
        model.adapter = ScaledSampleAdapter(
            model.adapter,
            auxiliary_scale=auxiliary_scale,
        )

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
    print(f"condition={args.condition}")
    print("evaluation_function=train_multi.test")
    print(f"evaluation_inference_batch_size={RELEASED_INFERENCE_BATCH_SIZE}")
    print(json.dumps({"effective_weights": effective_weight_summary}, indent=2))
    metrics = official_trainer.test(
        model,
        loader,
        cfg,
        is_distributed=distributed.is_enabled(),
    )

    result = {
        "schema_version": 1,
        "split": "test",
        "formal_full_split": args.max_images is None,
        "evaluated_images": len(dataset),
        "full_dataset_images": full_dataset_length,
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "model_name": "DINOv3",
        "dataset_name": "WHU",
        "num_modalities": 2,
        "backbone_type": "dinov3_vitl16",
        "use_lora": True,
        "lora_rank": 3,
        "condition": args.condition,
        "seed": args.seed,
        "dataset_condition": condition_metadata,
        "auxiliary_scale": auxiliary_scale,
        "released_adapter_weights": released_weight_summary,
        "effective_adapter_weights": effective_weight_summary,
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
