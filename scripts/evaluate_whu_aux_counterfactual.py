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
from PIL import Image

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
MODEL_PROFILES = {
    "vitl-lora": {
        "backbone_type": "dinov3_vitl16",
        "use_lora": True,
        "lora_rank": 3,
    },
    "vits-frozen": {
        "backbone_type": "dinov3_vits16",
        "use_lora": False,
        "lora_rank": None,
    },
}
CONDITIONS = (
    "normal",
    "rgb-only-native",
    "aux-mean",
    "aux-shuffle",
    "aux-feature-off",
    "aux-weight-scale",
)


def _condition_input_modalities(condition: str) -> int:
    """Return the number of sensor inputs used by the inference call.

    ``rgb-only-native`` keeps the two-modality checkpoint architecture intact,
    but calls its released one-input forward path.  This skips SAR loading and
    backbone encoding, and it also intentionally exercises the model's native
    single-input Adapter/Decoder behavior instead of the multimodal SE fusion.
    """

    return 1 if condition == "rgb-only-native" else 2


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
    parser.add_argument(
        "--model-profile",
        choices=tuple(MODEL_PROFILES),
        default="vitl-lora",
        help="Locked architecture profile matching the evaluated checkpoint",
    )
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
            "permutation_strategy": dataset.permutation_strategy,
            "permutation_groups": dataset.permutation_groups,
        }
    return {
        "mode": "normal",
        "permutation": None,
        "permutation_offset": None,
        "permutation_strategy": None,
        "permutation_groups": None,
    }


def _whu_shuffle_group_keys(dataset: Any) -> list[str]:
    """Group WHU scenes by their paired native spatial size.

    A few released test scenes differ by one pixel in width. Shuffling only
    within equal-size groups preserves native pixels and avoids making resize
    or crop policy part of the counterfactual.
    """

    rgb_files = getattr(dataset, "rgb_files", None)
    auxiliary_files = getattr(dataset, "sar_files", None)
    if not rgb_files or not auxiliary_files:
        raise ValueError("WHU shuffle requires RGB and SAR source-file lists")
    if len(rgb_files) != len(auxiliary_files) or len(rgb_files) != len(dataset):
        raise ValueError("WHU RGB/SAR source lists must match the dataset length")

    group_keys: list[str] = []
    for index, (rgb_path, auxiliary_path) in enumerate(
        zip(rgb_files, auxiliary_files, strict=True)
    ):
        with (
            Image.open(rgb_path) as rgb_image,
            Image.open(auxiliary_path) as aux_image,
        ):
            if rgb_image.size != aux_image.size:
                raise ValueError(
                    f"Paired WHU RGB/SAR sizes differ at index {index}: "
                    f"{rgb_image.size} vs {aux_image.size}"
                )
            width, height = rgb_image.size
        group_keys.append(f"{height}x{width}")
    return group_keys


def main() -> None:
    args = parse_args()
    auxiliary_scale = _condition_auxiliary_scale(args)
    inference_num_modalities = _condition_input_modalities(args.condition)
    model_profile = MODEL_PROFILES[args.model_profile]
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
        use_lora=model_profile["use_lora"],
        r=model_profile["lora_rank"],
        backbone_type=model_profile["backbone_type"],
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
        modality="multi" if inference_num_modalities > 1 else None,
        backbone_type=model_profile["backbone_type"],
    )
    if args.condition in ("aux-mean", "aux-shuffle"):
        shuffle_group_keys = (
            _whu_shuffle_group_keys(released_dataset)
            if args.condition == "aux-shuffle"
            else None
        )
        dataset: Any = AuxiliaryConditionDataset(
            released_dataset,
            args.condition,
            seed=args.seed,
            shuffle_group_keys=shuffle_group_keys,
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
    effective_weight_summary = (
        released_weight_summary if inference_num_modalities > 1 else None
    )
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
    official_trainer.NUM_MODALITIES = inference_num_modalities

    print(f"checkpoint={checkpoint_path}")
    print(f"model_profile={args.model_profile}")
    print(f"condition={args.condition}")
    print(f"inference_num_modalities={inference_num_modalities}")
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
        "schema_version": 2,
        "split": "test",
        "formal_full_split": args.max_images is None,
        "evaluated_images": len(dataset),
        "full_dataset_images": full_dataset_length,
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "model_name": "DINOv3",
        "dataset_name": "WHU",
        "model_profile": args.model_profile,
        "num_modalities": 2,
        "checkpoint_num_modalities": 2,
        "inference_num_modalities": inference_num_modalities,
        "backbone_type": model_profile["backbone_type"],
        "use_lora": model_profile["use_lora"],
        "lora_rank": model_profile["lora_rank"],
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
