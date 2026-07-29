"""Evaluate full-test E0 equivalence for the WHU NAF-P2 experiment.

E0 means evaluation before any continuation update.  R0 is the released
MM-DINO model and R1 is the same checkpoint with the frozen NAF branch and an
exactly zero-initialized residual projection.  R0 and R1 must therefore make
identical predictions before a paired short continuation is allowed.

This is an evaluation-only foreground entry point.  It never updates or saves
model parameters and writes one immutable JSON result.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from configs import get_cfg  # noqa: E402
from datasets import build_dataset  # noqa: E402
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.inference import slide_inference  # noqa: E402
from utils.metrics import metrics  # noqa: E402
from utils.utils import set_seed  # noqa: E402


NUM_CLASSES = 7
BACKBONE_TYPE = "dinov3_vits16"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_baseline_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise TypeError("Baseline checkpoint must contain a 'model' state dict")
    return payload["model"]


def build_variant(args: argparse.Namespace, variant: str):
    if variant not in {"r0", "r1"}:
        raise ValueError(f"Unknown E0 variant: {variant}")

    # Seed before construction for reproducible diagnostics.  The checkpoint
    # then replaces all R0 parameters; R1 additionally loads the released NAF.
    set_seed(args.seed)
    use_naf = variant == "r1"
    cfg = get_cfg(
        "DINOv3",
        "WHU",
        num_modalities=2,
        use_lora=False,
        r=3,
        backbone_type=BACKBONE_TYPE,
        use_naf=use_naf,
        naf_checkpoint=str(args.naf_checkpoint) if use_naf else None,
        naf_guidance_size=args.guidance_size,
        naf_backend=args.naf_backend,
        naf_q_tile_shape=args.naf_q_tile,
        naf_kv_tile_shape=args.naf_kv_tile,
    )
    model = cfg["model"]
    state = load_baseline_state(args.baseline_checkpoint)
    if variant == "r0":
        model.load_state_dict(state, strict=True)
    else:
        incompatible = model.load_state_dict(state, strict=False)
        expected_missing = {
            key
            for key in model.state_dict()
            if key.startswith("adapter.naf.")
            or key == "adapter.naf_zero_conv.weight"
        }
        if set(incompatible.missing_keys) != expected_missing:
            raise AssertionError(
                "Baseline-to-R1 missing-key contract changed: "
                f"expected={sorted(expected_missing)}, "
                f"actual={sorted(incompatible.missing_keys)}"
            )
        if incompatible.unexpected_keys:
            raise AssertionError(
                f"Unexpected baseline keys for R1: {incompatible.unexpected_keys}"
            )
        if torch.count_nonzero(model.adapter.naf_zero_conv.weight).item() != 0:
            raise AssertionError("R1 ZeroConv is not exactly zero initialized")
        if any(parameter.requires_grad for parameter in model.adapter.naf.parameters()):
            raise AssertionError("R1 NAF parameters are not frozen")

    # Config-created optimizer/scheduler retain references to model parameters;
    # E0 is inference-only, so discard them explicitly.
    cfg["optimizer"] = None
    cfg["scheduler"] = None
    del state
    return model, cfg


def build_test_loader(args: argparse.Namespace, window_size: tuple[int, int]):
    dataset = build_dataset(
        "WHU",
        "test",
        window_size=window_size,
        model_name="DINOv3",
        modality="multi",
        backbone_type=BACKBONE_TYPE,
    )
    full_length = len(dataset)
    if args.max_images is not None:
        dataset = torch.utils.data.Subset(dataset, range(args.max_images))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return loader, full_length


def evaluate_variant(
    model: torch.nn.Module,
    cfg: dict,
    loader,
    device: torch.device,
    inference_batch_size: int,
) -> dict:
    model.to(device)
    model.eval()
    if hasattr(model.adapter, "naf") and model.adapter.naf is not None:
        if model.adapter.naf.training:
            raise AssertionError("Frozen NAF entered training mode during E0")

    predictions = []
    labels = []
    prediction_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    stride = tuple(int(size * 2 / 3) for size in cfg["window_size"])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for optical, sar, label in loader:
            optical = optical.to(device)
            sar = sar.to(device)
            scores = slide_inference(
                optical,
                model,
                dsm=sar,
                n_output_channels=NUM_CLASSES,
                crop_size=cfg["window_size"],
                stride=stride,
                batch_size=inference_batch_size,
            )
            prediction = scores.argmax(dim=1).numpy().astype(np.int64, copy=False)
            label_numpy = label.numpy().astype(np.int64, copy=False)
            prediction = np.ascontiguousarray(prediction)
            label_numpy = np.ascontiguousarray(label_numpy)
            prediction_digest.update(prediction.tobytes())
            label_digest.update(label_numpy.tobytes())

            flat_prediction = prediction.reshape(-1)
            flat_label = label_numpy.reshape(-1)
            valid = (flat_label >= 0) & (flat_label < NUM_CLASSES)
            encoded = flat_label[valid] * NUM_CLASSES + flat_prediction[valid]
            confusion += np.bincount(
                encoded, minlength=NUM_CLASSES**2
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            predictions.append(flat_prediction)
            labels.append(flat_label)
            del optical, sar, scores

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    concatenated_predictions = np.concatenate(predictions)
    concatenated_labels = np.concatenate(labels)
    miou, f1, kappa, accuracy = metrics(
        concatenated_predictions,
        concatenated_labels,
        cfg["labels"],
    )
    del concatenated_predictions, concatenated_labels, predictions, labels
    return {
        "num_images": len(loader.dataset),
        "prediction_sha256": prediction_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
        "confusion": confusion.tolist(),
        "MIoU": float(miou),
        "F1": float(f1),
        "Kappa": float(kappa),
        "Acc": float(accuracy),
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify full-test R0/R1 equivalence before NAF-P2 continuation"
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--naf-checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--guidance-size", type=int, default=224)
    parser.add_argument(
        "--naf-backend",
        choices=("cutlass-fna", "flex-fna"),
        default="cutlass-fna",
    )
    parser.add_argument("--naf-q-tile", type=int, nargs=2)
    parser.add_argument("--naf-kv-tile", type=int, nargs=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--expected-miou", type=float)
    parser.add_argument("--miou-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    for path in (args.baseline_checkpoint, args.naf_checkpoint):
        if not path.is_file():
            parser.error(f"checkpoint does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_path}")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.guidance_size <= 0:
        parser.error("--guidance-size must be positive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.miou_tolerance < 0:
        parser.error("--miou-tolerance cannot be negative")
    if (args.naf_q_tile is None) != (args.naf_kv_tile is None):
        parser.error("--naf-q-tile and --naf-kv-tile must be set together")
    if args.naf_q_tile is not None:
        if any(value <= 0 for value in (*args.naf_q_tile, *args.naf_kv_tile)):
            parser.error("NAF tile dimensions must be positive")
        args.naf_q_tile = tuple(args.naf_q_tile)
        args.naf_kv_tile = tuple(args.naf_kv_tile)
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WHU NAF E0 evaluation")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    device = torch.device(args.device)
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"baseline_checkpoint_sha256={file_sha256(args.baseline_checkpoint)}")
    print(f"naf_checkpoint_sha256={file_sha256(args.naf_checkpoint)}")
    print(f"evaluation_inference_batch_size={args.inference_batch_size}")

    results = {}
    full_test_length = None
    for variant in ("r0", "r1"):
        print(f"evaluating_variant={variant}")
        model, cfg = build_variant(args, variant)
        loader, variant_full_length = build_test_loader(args, cfg["window_size"])
        if full_test_length is None:
            full_test_length = variant_full_length
        elif full_test_length != variant_full_length:
            raise AssertionError("R0/R1 test-set lengths differ")
        results[variant] = evaluate_variant(
            model,
            cfg,
            loader,
            device,
            args.inference_batch_size,
        )
        model.to("cpu")
        del model, cfg, loader
        gc.collect()
        torch.cuda.empty_cache()

    prediction_equal = (
        results["r0"]["prediction_sha256"]
        == results["r1"]["prediction_sha256"]
    )
    label_equal = results["r0"]["label_sha256"] == results["r1"]["label_sha256"]
    confusion_equal = results["r0"]["confusion"] == results["r1"]["confusion"]
    metric_differences = {
        name: abs(results["r0"][name] - results["r1"][name])
        for name in ("MIoU", "F1", "Kappa", "Acc")
    }
    if not prediction_equal or not label_equal or not confusion_equal:
        raise AssertionError(
            "E0 failed: R0 and zero-gated R1 do not produce identical test predictions"
        )
    if any(value != 0.0 for value in metric_differences.values()):
        raise AssertionError(f"E0 metric mismatch: {metric_differences}")

    expected_miou_difference = None
    if args.expected_miou is not None:
        expected_miou_difference = abs(results["r0"]["MIoU"] - args.expected_miou)
        if expected_miou_difference > args.miou_tolerance:
            raise AssertionError(
                "R0 E0 differs from the expected historical baseline: "
                f"actual={results['r0']['MIoU']:.9f}, "
                f"expected={args.expected_miou:.9f}, "
                f"tolerance={args.miou_tolerance:.9f}"
            )

    output = {
        "status": "PASS",
        "scope": "full-test" if args.max_images is None else "subset-smoke",
        "full_test_length": full_test_length,
        "evaluated_images": results["r0"]["num_images"],
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": file_sha256(args.baseline_checkpoint),
        "naf_checkpoint": str(args.naf_checkpoint.resolve()),
        "naf_checkpoint_sha256": file_sha256(args.naf_checkpoint),
        "naf_backend": args.naf_backend,
        "naf_q_tile": args.naf_q_tile,
        "naf_kv_tile": args.naf_kv_tile,
        "guidance": f"optical_common_normalized_{args.guidance_size}x{args.guidance_size}",
        "inference_batch_size": args.inference_batch_size,
        "prediction_equal": prediction_equal,
        "label_equal": label_equal,
        "confusion_equal": confusion_equal,
        "metric_differences": metric_differences,
        "expected_miou": args.expected_miou,
        "expected_miou_tolerance": args.miou_tolerance,
        "expected_miou_difference": expected_miou_difference,
        **results,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"e0_result={args.output_path.resolve()}")
    print("e0_status=PASS")


if __name__ == "__main__":
    main()
