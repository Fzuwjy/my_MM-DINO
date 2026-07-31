"""Verify RGB-only fusion preservation against the feature-off endpoint.

This is a real-model smoke, not a metric evaluation.  It loads one trained
multimodal WHU checkpoint and asserts that:

1. the RGB tensor and label are identical in multimodal and RGB-only datasets;
2. two-input ``aux-feature-off`` logits equal one-input RGB logits routed
   through the trained multimodal Decoder slots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
for path in (REPO_ROOT, SEGMENTATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.aux_diagnostics_common import (
    FusionPreservingSingleInputDecoder,
    ScaledSampleAdapter,
)
from scripts.evaluate_whu_aux_counterfactual import MODEL_PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-checkpoint RGB-only fusion-equivalence smoke"
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--model-profile",
        choices=tuple(MODEL_PROFILES),
        default="vitl-lora",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=512)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    if args.crop_size <= 0 or args.crop_size % 16 != 0:
        raise ValueError("--crop-size must be a positive multiple of 16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from configs import get_cfg
    from datasets import build_dataset

    profile = MODEL_PROFILES[args.model_profile]
    seed_everything(args.seed)
    cfg = get_cfg(
        "DINOv3",
        "WHU",
        num_modalities=2,
        use_lora=profile["use_lora"],
        r=profile["lora_rank"],
        backbone_type=profile["backbone_type"],
    )

    common_dataset_args = {
        "window_size": cfg["window_size"],
        "model_name": "DINOv3",
        "backbone_type": profile["backbone_type"],
    }
    multimodal_dataset = build_dataset(
        "WHU", "test", modality="multi", **common_dataset_args
    )
    rgb_only_dataset = build_dataset(
        "WHU", "test", modality=None, **common_dataset_args
    )
    rgb_multi, sar, label_multi = multimodal_dataset[0]
    rgb_only, label_rgb_only = rgb_only_dataset[0]
    if not torch.equal(rgb_multi, rgb_only):
        raise AssertionError("Multimodal and RGB-only dataset RGB tensors differ")
    if not np.array_equal(np.asarray(label_multi), np.asarray(label_rgb_only)):
        raise AssertionError("Multimodal and RGB-only dataset labels differ")

    height = min(args.crop_size, int(rgb_multi.shape[-2]))
    width = min(args.crop_size, int(rgb_multi.shape[-1]))
    height -= height % 16
    width -= width % 16
    rgb = rgb_multi[:, :height, :width].unsqueeze(0).cuda()
    auxiliary = sar[:, :height, :width].unsqueeze(0).cuda()

    model = cfg["model"].cuda()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("Expected checkpoint['model']")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval()

    released_adapter = model.adapter
    released_decoder = model.decoder
    with torch.no_grad():
        model.adapter = ScaledSampleAdapter(
            released_adapter, auxiliary_scale=0.0
        )
        feature_off_logits = model(rgb, auxiliary)

        model.adapter = released_adapter
        model.decoder = FusionPreservingSingleInputDecoder(
            released_decoder, num_modalities=2
        )
        fusion_preserved_logits = model(rgb)

    difference = (feature_off_logits - fusion_preserved_logits).abs()
    exact_logits = bool(torch.equal(feature_off_logits, fusion_preserved_logits))
    exact_predictions = bool(
        torch.equal(
            feature_off_logits.argmax(dim=1),
            fusion_preserved_logits.argmax(dim=1),
        )
    )
    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "model_profile": args.model_profile,
        "crop_shape": list(rgb.shape),
        "rgb_dataset_tensor_exact": True,
        "label_dataset_array_exact": True,
        "feature_off_logits_sha256": tensor_sha256(feature_off_logits),
        "fusion_preserved_logits_sha256": tensor_sha256(fusion_preserved_logits),
        "exact_logits": exact_logits,
        "max_abs_logit_difference": float(difference.max().item()),
        "mean_abs_logit_difference": float(difference.mean().item()),
        "exact_argmax_predictions": exact_predictions,
        "sar_loaded_for_reference_only": True,
        "sar_encoded_by_fusion_preserved_path": False,
    }
    if not exact_logits or not exact_predictions:
        raise AssertionError(json.dumps(result, indent=2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"equivalence_result={output_path}")


if __name__ == "__main__":
    main()
