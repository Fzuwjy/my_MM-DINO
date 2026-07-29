"""Run the first-gate spatial error audit on the faithful WHU ViT-S E0.

This foreground entry point performs no training and does not modify model
parameters.  It saves lossless per-image prediction/label maps, verifies the
aggregate prediction against a trusted E0 JSON, and reports programmatic
boundary, semantic-component, mixed-patch, and oracle statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from configs import get_cfg  # noqa: E402
from datasets import build_dataset  # noqa: E402
from scripts.spatial_diagnostics_common import (  # noqa: E402
    baseline_summary,
    bootstrap_oracle_gain,
    diagnose_prediction,
    region_summary,
)
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.inference import slide_inference  # noqa: E402
from utils.utils import set_seed  # noqa: E402


BACKBONE_TYPE = "dinov3_vits16"
NUM_CLASSES = 7


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("r0", payload)
    required = {"prediction_sha256", "label_sha256", "confusion", "MIoU"}
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"E0 reference is missing fields: {missing}")
    return {"payload": payload, "result": result}


def validate_checkpoint_reference(
    reference: dict[str, Any], baseline_sha256: str
) -> bool:
    expected = reference["payload"].get("baseline_checkpoint_sha256")
    if expected is not None and baseline_sha256 != expected:
        raise AssertionError(
            "Baseline checkpoint SHA256 differs from E0 reference: "
            f"{baseline_sha256} != {expected}"
        )
    return expected is None or baseline_sha256 == expected


def load_model(checkpoint: Path, seed: int):
    set_seed(seed)
    cfg = get_cfg(
        "DINOv3",
        "WHU",
        num_modalities=2,
        use_lora=False,
        r=3,
        backbone_type=BACKBONE_TYPE,
        use_naf=False,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise TypeError("Baseline checkpoint must contain a 'model' state dict")
    cfg["model"].load_state_dict(payload["model"], strict=True)
    cfg["optimizer"] = None
    cfg["scheduler"] = None
    return cfg["model"], cfg


def build_test_loader(max_images: int | None, window_size: tuple[int, int]):
    base_dataset = build_dataset(
        "WHU",
        "test",
        window_size=window_size,
        model_name="DINOv3",
        modality="multi",
        backbone_type=BACKBONE_TYPE,
    )
    full_length = len(base_dataset)
    indices = list(range(full_length if max_images is None else max_images))
    if any(index >= full_length for index in indices):
        raise ValueError(
            f"--max-images exceeds test length: {max_images} > {full_length}"
        )
    dataset = torch.utils.data.Subset(base_dataset, indices)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    sample_names = [Path(base_dataset.rgb_files[index]).stem for index in indices]
    return loader, sample_names, full_length


def save_index_map(path: Path, array: np.ndarray) -> None:
    values = np.asarray(array)
    if values.ndim != 2:
        raise ValueError("index map must be 2-D")
    if np.any((values < 0) | (values > 255)):
        raise ValueError("index map cannot be represented losslessly as uint8")
    Image.fromarray(values.astype(np.uint8, copy=False)).save(
        path, format="PNG", compress_level=1
    )


def _as_int_stack(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("cannot stack an empty list")
    return np.stack([np.asarray(value, dtype=np.int64) for value in values])


def aggregate_diagnostics(
    images: list[dict[str, Any]],
    class_names: list[str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline_stack = _as_int_stack(
        [image["_baseline_confusion"] for image in images]
    )
    aggregate_baseline = baseline_stack.sum(axis=0)
    region_names = list(images[0]["_region_confusions"])
    if any(
        list(image["_region_confusions"]) != region_names for image in images
    ):
        raise AssertionError("per-image diagnostic regions differ")

    rng = np.random.default_rng(bootstrap_seed)
    sample_indices = rng.integers(
        0,
        len(images),
        size=(bootstrap_replicates, len(images)),
        endpoint=False,
    )
    regions = {}
    for region_name in region_names:
        region_stack = _as_int_stack(
            [image["_region_confusions"][region_name] for image in images]
        )
        summary = region_summary(
            aggregate_baseline, region_stack.sum(axis=0), class_names
        )
        summary["image_bootstrap"] = bootstrap_oracle_gain(
            baseline_stack, region_stack, sample_indices
        )
        regions[region_name] = summary
    return {
        "baseline": baseline_summary(aggregate_baseline, class_names),
        "regions": regions,
        "bootstrap": {
            "unit": "test image",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "interpretation": (
                "Descriptive image-cluster resampling interval, not a pixel-iid p-value."
            ),
        },
    }


def validate_against_e0(
    *,
    reference: dict[str, Any],
    baseline_sha256: str,
    evaluated_images: int,
    full_test_length: int,
    prediction_sha256: str,
    label_sha256: str,
    confusion: list[list[int]],
    miou: float,
    max_images: int | None,
    tolerance: float,
) -> dict[str, Any]:
    payload = reference["payload"]
    expected = reference["result"]
    checkpoint_equal = validate_checkpoint_reference(reference, baseline_sha256)

    validation = {
        "checkpoint_sha256_equal": checkpoint_equal,
        "full_prediction_checked": max_images is None,
    }
    if max_images is not None:
        return validation

    expected_images = payload.get("evaluated_images", full_test_length)
    checks = {
        "evaluated_images_equal": evaluated_images == expected_images,
        "prediction_sha256_equal": prediction_sha256
        == expected["prediction_sha256"],
        "label_sha256_equal": label_sha256 == expected["label_sha256"],
        "confusion_equal": confusion == expected["confusion"],
        "miou_within_tolerance": abs(miou - float(expected["MIoU"])) <= tolerance,
    }
    validation.update(checks)
    if not all(checks.values()):
        raise AssertionError(f"Spatial diagnostic failed E0 reproduction: {checks}")
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Programmatic spatial-error audit for faithful WHU ViT-S E0"
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-e0-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--miou-tolerance", type=float, default=1e-12)
    parser.add_argument("--boundary-radii", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument(
        "--component-area-thresholds",
        type=int,
        nargs="+",
        default=(256, 1024, 4096),
    )
    parser.add_argument(
        "--component-thickness-thresholds",
        type=int,
        nargs="+",
        default=(4, 8, 16),
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--union-boundary-radius", type=int, default=4)
    parser.add_argument("--union-component-area", type=int, default=1024)
    parser.add_argument("--union-component-thickness", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    args = parser.parse_args()

    for path in (args.baseline_checkpoint, args.expected_e0_json):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.prediction_dir.exists():
        parser.error(f"refusing to reuse prediction directory: {args.prediction_dir}")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.miou_tolerance < 0:
        parser.error("--miou-tolerance cannot be negative")
    if args.patch_size <= 0:
        parser.error("--patch-size must be positive")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates cannot be negative")
    if any(value < 0 for value in args.boundary_radii):
        parser.error("--boundary-radii must be non-negative")
    if any(value <= 0 for value in args.component_area_thresholds):
        parser.error("--component-area-thresholds must be positive")
    if any(value <= 0 for value in args.component_thickness_thresholds):
        parser.error("--component-thickness-thresholds must be positive")
    if args.union_boundary_radius not in args.boundary_radii:
        parser.error("--union-boundary-radius must occur in --boundary-radii")
    if args.union_component_area not in args.component_area_thresholds:
        parser.error(
            "--union-component-area must occur in --component-area-thresholds"
        )
    if args.union_component_thickness not in args.component_thickness_thresholds:
        parser.error(
            "--union-component-thickness must occur in "
            "--component-thickness-thresholds"
        )
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WHU ViT-S evaluation")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    reference = load_reference(args.expected_e0_json)
    baseline_sha = file_sha256(args.baseline_checkpoint)
    validate_checkpoint_reference(reference, baseline_sha)
    device = torch.device(args.device)
    model, cfg = load_model(args.baseline_checkpoint, args.seed)
    loader, sample_names, full_test_length = build_test_loader(
        args.max_images, cfg["window_size"]
    )
    model.to(device)
    model.eval()
    args.prediction_dir.mkdir(parents=True)

    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"baseline_checkpoint_sha256={baseline_sha}")
    print(f"evaluated_images={len(loader.dataset)}/{full_test_length}")
    print("diagnostic_mode=programmatic_spatial_error_first_gate")

    prediction_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    images: list[dict[str, Any]] = []
    region_definitions = None
    stride = tuple(int(size * 2 / 3) for size in cfg["window_size"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    with torch.inference_mode():
        for image_index, ((optical, sar, label), sample_name) in enumerate(
            zip(loader, sample_names, strict=True)
        ):
            image_started = time.perf_counter()
            scores = slide_inference(
                optical.to(device),
                model,
                dsm=sar.to(device),
                n_output_channels=NUM_CLASSES,
                crop_size=cfg["window_size"],
                stride=stride,
                batch_size=args.inference_batch_size,
            )
            prediction_i64 = np.ascontiguousarray(
                scores.argmax(dim=1).numpy().astype(np.int64, copy=False)
            )
            label_i64 = np.ascontiguousarray(
                label.numpy().astype(np.int64, copy=False)
            )
            prediction_digest.update(prediction_i64.tobytes())
            label_digest.update(label_i64.tobytes())
            prediction = prediction_i64[0]
            target = label_i64[0]

            artifact_stem = f"{image_index:03d}_{sample_name}"
            prediction_path = args.prediction_dir / f"{artifact_stem}_prediction.png"
            label_path = args.prediction_dir / f"{artifact_stem}_label.png"
            save_index_map(prediction_path, prediction)
            save_index_map(label_path, target)

            diagnostic = diagnose_prediction(
                prediction,
                target,
                cfg["labels"],
                boundary_radii=args.boundary_radii,
                component_area_thresholds=args.component_area_thresholds,
                component_thickness_thresholds=args.component_thickness_thresholds,
                patch_size=args.patch_size,
                union_boundary_radius=args.union_boundary_radius,
                union_component_area=args.union_component_area,
                union_component_thickness=args.union_component_thickness,
            )
            if region_definitions is None:
                region_definitions = diagnostic["definitions"]
            image_record = {
                "index": image_index,
                "sample_name": sample_name,
                "shape": list(target.shape),
                "prediction_path": str(prediction_path.resolve()),
                "label_path": str(label_path.resolve()),
                "baseline": baseline_summary(
                    diagnostic["baseline_confusion"], cfg["labels"]
                ),
                "regions": {
                    name: region_summary(
                        diagnostic["baseline_confusion"], confusion, cfg["labels"]
                    )
                    for name, confusion in diagnostic["region_confusions"].items()
                },
                "elapsed_seconds": time.perf_counter() - image_started,
                "_baseline_confusion": diagnostic["baseline_confusion"],
                "_region_confusions": diagnostic["region_confusions"],
            }
            images.append(image_record)
            print(
                f"image={image_index + 1}/{len(loader.dataset)} "
                f"name={sample_name} "
                f"miou={image_record['baseline']['miou_percent']:.6f}% "
                f"elapsed={image_record['elapsed_seconds']:.1f}s",
                flush=True,
            )
            del (
                scores,
                prediction_i64,
                label_i64,
                prediction,
                target,
                diagnostic,
                optical,
                sar,
                label,
            )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    aggregate = aggregate_diagnostics(
        images,
        cfg["labels"],
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    prediction_sha = prediction_digest.hexdigest()
    label_sha = label_digest.hexdigest()
    e0_validation = validate_against_e0(
        reference=reference,
        baseline_sha256=baseline_sha,
        evaluated_images=len(images),
        full_test_length=full_test_length,
        prediction_sha256=prediction_sha,
        label_sha256=label_sha,
        confusion=aggregate["baseline"]["confusion"],
        miou=aggregate["baseline"]["miou"],
        max_images=args.max_images,
        tolerance=args.miou_tolerance,
    )

    serializable_images = []
    for image in images:
        serializable_images.append(
            {
                key: value
                for key, value in image.items()
                if not key.startswith("_")
            }
        )
    output = {
        "status": "PASS",
        "scope": "full-test" if args.max_images is None else "subset-smoke",
        "scientific_scope": (
            "Describes error geometry and oracle headroom; does not by itself "
            "identify DINOv3 feature resolution as the cause."
        ),
        "full_test_length": full_test_length,
        "evaluated_images": len(images),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": baseline_sha,
        "expected_e0_json": str(args.expected_e0_json.resolve()),
        "prediction_sha256": prediction_sha,
        "label_sha256": label_sha,
        "e0_reference_validation": e0_validation,
        "inference": {
            "window_size": list(cfg["window_size"]),
            "stride": list(stride),
            "crop_batch_size": args.inference_batch_size,
            "device": str(device),
        },
        "region_parameters": {
            "boundary_radii": list(args.boundary_radii),
            "component_area_thresholds": list(args.component_area_thresholds),
            "component_thickness_thresholds": list(
                args.component_thickness_thresholds
            ),
            "patch_size": args.patch_size,
            "union_boundary_radius": args.union_boundary_radius,
            "union_component_area": args.union_component_area,
            "union_component_thickness": args.union_component_thickness,
        },
        "region_definitions": region_definitions,
        "aggregate": aggregate,
        "images": serializable_images,
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": output["status"],
        "scope": output["scope"],
        "evaluated_images": output["evaluated_images"],
        "baseline_miou_percent": aggregate["baseline"]["miou_percent"],
        "actionable_union": aggregate["regions"]["actionable_union"],
        "e0_reference_validation": e0_validation,
    }, ensure_ascii=False, indent=2))
    print(f"spatial_diagnostic_result={args.output_path.resolve()}")
    print("spatial_diagnostic_status=PASS")


if __name__ == "__main__":
    main()
