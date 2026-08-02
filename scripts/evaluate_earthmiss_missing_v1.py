"""Evaluate EarthMiss V1 endpoints from one frozen checkpoint.

The default Val run compares Full, canonical SAR-only, the historical
zero/no-renormalization intervention, and the native one-input SAR path.  All
endpoints use the same loaded checkpoint.  Test evaluation is available only
for the later external-comparison stage and reports the exact Ever metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import build_dataset  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from models.MMDINO.diagnostics import FeatureZeroSampleAdapter  # noqa: E402
from models.MMDINO.dino_segment import build_model  # noqa: E402
from utils.earthmiss_metrics import EarthMissMetrics  # noqa: E402
from utils.inference import slide_inference  # noqa: E402

from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
    VAL_SELECTION_CLASS_IDS,
)


ENDPOINTS = (
    "full",
    "sar-canonical",
    "sar-legacy-zero-no-renorm",
    "sar-one-input",
)
TEST_SELECTION_CLASS_IDS = list(range(8))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output")
    parser.add_argument(
        "--endpoints",
        nargs="+",
        choices=ENDPOINTS,
        default=list(ENDPOINTS),
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--allow-non-primary-test-checkpoint",
        action="store_true",
        help="Allow Test diagnostics from a checkpoint not tagged primary_deployment.",
    )
    return parser.parse_args()


def build_loader(args):
    dataset = build_dataset(
        "EarthMiss",
        args.split,
        dataset_root=args.dataset_root,
        window_size=(args.window_size, args.window_size),
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader


def _endpoint_call(endpoint, rgb, sar, model, window_size, batch_size, device):
    stride = int(window_size * 2 / 3)
    common = {
        "n_output_channels": 8,
        "crop_size": (window_size, window_size),
        "stride": (stride, stride),
        "batch_size": batch_size,
    }
    if endpoint == "sar-one-input":
        # The native one-input model path expects a DINOv3-compatible channel
        # count and deliberately bypasses the trained two-slot Decoder path.
        return slide_inference(
            sar.repeat(1, 3, 1, 1),
            model,
            **common,
        )

    availability = None
    if endpoint == "full":
        availability = canonical_availability("full", batch_size=1, device=device)
    elif endpoint == "sar-canonical":
        availability = canonical_availability("sar", batch_size=1, device=device)
    elif endpoint != "sar-legacy-zero-no-renorm":
        raise ValueError(f"Unknown endpoint: {endpoint}")

    return slide_inference(
        rgb,
        model,
        dsm=sar,
        availability=availability,
        **common,
    )


@torch.no_grad()
def evaluate_endpoint(model, loader, endpoint, device, window_size, batch_size):
    model.eval()
    original_adapter = model.adapter
    if endpoint == "sar-legacy-zero-no-renorm":
        model.adapter = FeatureZeroSampleAdapter(
            original_adapter,
            zero_modality_index=0,
        )

    evaluator = EarthMissMetrics()
    try:
        for rgb, sar, label in tqdm(loader, desc=endpoint, leave=False):
            rgb = rgb.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            logits = _endpoint_call(
                endpoint,
                rgb,
                sar,
                model,
                window_size,
                batch_size,
                device,
            )
            evaluator.update(logits.argmax(dim=1), label)
    finally:
        model.adapter = original_adapter
    return evaluator.compute()


def _default_output_path(checkpoint_path, split):
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}.earthmiss-{split}-endpoints.json"
    )


def _validate_checkpoint(checkpoint, args):
    role = checkpoint.get("checkpoint_role", "unregistered")
    selection_state = checkpoint.get("selection_state")
    if (
        args.split == "test"
        and (role != "primary_deployment" or selection_state != "sar")
        and not args.allow_non_primary_test_checkpoint
    ):
        raise ValueError(
            "External Test comparison requires a primary_deployment checkpoint "
            "selected on SAR Val mIoU (best_sar.pth); use "
            "--allow-non-primary-test-checkpoint only for explicit diagnostics"
        )
    protocol = checkpoint.get("protocol", {})
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("checkpoint_selection_support") != "pooled_gt_present":
        raise ValueError("Checkpoint predates the fixed EarthMiss selection metric")


def main():
    args = parse_args()
    if args.window_size <= 0 or args.window_size % 16:
        raise ValueError("--window-size must be a positive multiple of 16")
    if args.inference_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("EarthMiss endpoint evaluation requires a CUDA GPU")

    checkpoint_path = Path(args.checkpoint)
    weights_path = Path(args.backbone_weights)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _validate_checkpoint(checkpoint, args)
    dataset, loader = build_loader(args)

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
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    output_path = (
        Path(args.output)
        if args.output
        else _default_output_path(checkpoint_path, args.split)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "earthmiss_missing_v1_endpoints_v1",
        "split": args.split,
        "tiles": len(dataset),
        "checkpoint": {
            "path": str(checkpoint_path),
            "run": checkpoint.get("run"),
            "seed": checkpoint.get("seed"),
            "epoch": checkpoint.get("epoch"),
            "role": checkpoint.get("checkpoint_role", "unregistered"),
            "selection_state": checkpoint.get("selection_state"),
            "selection_metric": checkpoint.get("selection_metric"),
            "selection_score": checkpoint.get("selection_score"),
        },
        "fair_comparison_contract": {
            "same_checkpoint_for_all_endpoints": True,
            "internal_selection": "Val pooled_gt_present mIoU",
            "external_comparison": "Test official_ever_mIoU fixed_all_8_classes",
            "rgb_normalization": {
                "policy": dataset.rgb_normalization,
                "mean": list(dataset.imagenet_mean),
                "std": list(dataset.imagenet_std),
            },
            "sar_normalization": {
                "policy": "earthmiss_metars_dataset_stats",
                "mean": list(dataset.sar_mean),
                "std": list(dataset.sar_std),
            },
        },
        "endpoint_definitions": {
            "full": "canonical RGB+SAR",
            "sar-canonical": "skip RGB backbone, renormalize over SAR, keep two slots",
            "sar-legacy-zero-no-renorm": (
                "run both backbones, zero projected RGB, keep original denominator"
            ),
            "sar-one-input": (
                "SAR repeated to three DINO input channels, native one-input Decoder path"
            ),
        },
        "endpoints": {},
    }

    for endpoint in args.endpoints:
        metrics = evaluate_endpoint(
            model,
            loader,
            endpoint,
            device,
            args.window_size,
            args.inference_batch_size,
        )
        expected_support = (
            VAL_SELECTION_CLASS_IDS
            if args.split == "val"
            else TEST_SELECTION_CLASS_IDS
        )
        if metrics["selection_class_ids"] != expected_support:
            raise RuntimeError(
                f"EarthMiss {args.split} support changed: expected "
                f"{expected_support}, got {metrics['selection_class_ids']}"
            )
        result["endpoints"][endpoint] = metrics
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            f"{endpoint}: mIoU={metrics['mIoU']:.6f}, "
            f"official_ever_mIoU={metrics['official_ever_mIoU']:.5f}"
        )

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
