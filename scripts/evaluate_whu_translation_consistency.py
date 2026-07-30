"""Evaluate WHU ViT-S translation consistency against a 16-pixel control.

The script reuses the exact baseline prediction/label maps saved by
``diagnose_whu_spatial_errors.py``.  It shifts RGB and SAR together, runs only
the shifted conditions, inverse-aligns their predictions on one shared valid
field of view, and compares sub-patch shifts with a 16-pixel control.  It does
not train or save model parameters.
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

from scripts.diagnose_whu_spatial_errors import (  # noqa: E402
    build_test_loader,
    file_sha256,
    load_model,
)
from scripts.spatial_diagnostics_common import (  # noqa: E402
    bootstrap_paired_miou_delta,
    build_spatial_region_masks,
    class_ious_from_confusion,
    common_translation_slices,
    confusion_from_arrays,
    mean_iou_from_confusion,
)
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.inference import slide_inference  # noqa: E402


NUM_CLASSES = 7
REPORT_REGIONS = (
    "boundary_le_0px",
    "component_area_le_256px2",
    "component_thickness_le_4px",
    "actionable_union",
)


def translate_tensor(tensor: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate a BCHW tensor with zero fill and no wraparound."""

    if tensor.ndim != 4:
        raise ValueError("translation input must be BCHW")
    height, width = tensor.shape[-2:]
    dy, dx = int(dy), int(dx)
    source_y_start = max(0, -dy)
    source_y_end = min(height, height - dy)
    source_x_start = max(0, -dx)
    source_x_end = min(width, width - dx)
    if source_y_start >= source_y_end or source_x_start >= source_x_end:
        raise ValueError("translation magnitude must be smaller than the image")
    destination_y_start = source_y_start + dy
    destination_y_end = source_y_end + dy
    destination_x_start = source_x_start + dx
    destination_x_end = source_x_end + dx
    translated = torch.zeros_like(tensor)
    translated[
        ...,
        destination_y_start:destination_y_end,
        destination_x_start:destination_x_end,
    ] = tensor[
        ...,
        source_y_start:source_y_end,
        source_x_start:source_x_end,
    ]
    return translated


def load_spatial_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("scope") != "full-test":
        raise ValueError("spatial diagnostic reference must be a full-test PASS")
    validation = payload.get("e0_reference_validation", {})
    required_checks = (
        "checkpoint_sha256_equal",
        "prediction_sha256_equal",
        "label_sha256_equal",
        "confusion_equal",
        "miou_within_tolerance",
    )
    if not all(validation.get(name) is True for name in required_checks):
        raise ValueError(
            "spatial diagnostic reference did not strictly reproduce the E0 baseline"
        )
    if len(payload.get("images", [])) != payload.get("evaluated_images"):
        raise ValueError("spatial diagnostic image manifest is incomplete")
    return payload


def load_index_map(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.asarray(Image.open(path))
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"saved index map is not a 2-D integer image: {path}")
    return np.ascontiguousarray(values.astype(np.int64, copy=False))


def validate_saved_baseline_maps(
    reference: dict[str, Any], max_images: int | None
) -> dict[str, Any]:
    image_records = reference["images"]
    selected = image_records if max_images is None else image_records[:max_images]
    prediction_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    for image in selected:
        prediction = load_index_map(Path(image["prediction_path"]))[None]
        label = load_index_map(Path(image["label_path"]))[None]
        prediction_digest.update(prediction.tobytes())
        label_digest.update(label.tobytes())
    validation = {
        "evaluated_maps": len(selected),
        "full_hash_checked": max_images is None,
        "prediction_sha256": prediction_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
    }
    if max_images is None:
        validation["prediction_sha256_equal"] = (
            validation["prediction_sha256"] == reference["prediction_sha256"]
        )
        validation["label_sha256_equal"] = (
            validation["label_sha256"] == reference["label_sha256"]
        )
        if not (
            validation["prediction_sha256_equal"]
            and validation["label_sha256_equal"]
        ):
            raise AssertionError("saved prediction/label maps do not match E0 hashes")
    return validation


def paired_condition_summary(
    baseline_confusion: np.ndarray,
    shifted_confusion: np.ndarray,
    disagreement_pixels: int,
    valid_pixels: int,
    class_names: list[str],
) -> dict[str, Any]:
    baseline = np.asarray(baseline_confusion, dtype=np.int64)
    shifted = np.asarray(shifted_confusion, dtype=np.int64)
    baseline_miou = mean_iou_from_confusion(baseline)
    shifted_miou = mean_iou_from_confusion(shifted)
    baseline_ious = class_ious_from_confusion(baseline)
    shifted_ious = class_ious_from_confusion(shifted)
    return {
        "valid_pixels": int(valid_pixels),
        "prediction_disagreement_pixels": int(disagreement_pixels),
        "prediction_disagreement_rate": float(disagreement_pixels / valid_pixels),
        "baseline_confusion": baseline.tolist(),
        "shifted_confusion": shifted.tolist(),
        "baseline_miou_percent": float(baseline_miou * 100.0),
        "shifted_miou_percent": float(shifted_miou * 100.0),
        "shifted_minus_baseline_miou_pp": float(
            (shifted_miou - baseline_miou) * 100.0
        ),
        "class_shifted_minus_baseline_iou_pp": {
            name: (
                float((shifted_value - baseline_value) * 100.0)
                if np.isfinite(shifted_value) and np.isfinite(baseline_value)
                else None
            )
            for name, baseline_value, shifted_value in zip(
                class_names, baseline_ious, shifted_ious, strict=True
            )
        },
    }


def region_consistency_summary(counts: dict[str, int]) -> dict[str, Any]:
    pixels = counts["pixels"]
    if pixels <= 0:
        return {
            **counts,
            "coverage": None,
            "prediction_disagreement_rate": None,
            "baseline_error_rate": None,
            "shifted_error_rate": None,
            "shifted_minus_baseline_error_rate_pp": None,
        }
    baseline_rate = counts["baseline_errors"] / pixels
    shifted_rate = counts["shifted_errors"] / pixels
    return {
        **counts,
        "coverage": counts["pixels"] / counts["valid_pixels"],
        "prediction_disagreement_rate": counts["disagreement_pixels"] / pixels,
        "baseline_error_rate": baseline_rate,
        "shifted_error_rate": shifted_rate,
        "shifted_minus_baseline_error_rate_pp": (shifted_rate - baseline_rate)
        * 100.0,
    }


def aggregate_offset_records(
    records: list[dict[str, Any]],
    class_names: list[str],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    baseline_stack = np.stack(
        [record["_baseline_confusion"] for record in records]
    )
    shifted_stack = np.stack(
        [record["_shifted_confusion"] for record in records]
    )
    valid_pixels = sum(record["valid_pixels"] for record in records)
    disagreement = sum(record["disagreement_pixels"] for record in records)
    summary = paired_condition_summary(
        baseline_stack.sum(axis=0),
        shifted_stack.sum(axis=0),
        disagreement,
        valid_pixels,
        class_names,
    )
    summary["image_bootstrap_shifted_minus_baseline_miou_pp"] = (
        bootstrap_paired_miou_delta(
            baseline_stack, shifted_stack, bootstrap_indices
        )
    )
    summary["regions"] = {}
    for region_name in REPORT_REGIONS:
        counts = {
            key: sum(record["regions"][region_name][key] for record in records)
            for key in (
                "valid_pixels",
                "pixels",
                "disagreement_pixels",
                "baseline_errors",
                "shifted_errors",
            )
        }
        summary["regions"][region_name] = region_consistency_summary(counts)
    summary["_baseline_stack"] = baseline_stack
    summary["_shifted_stack"] = shifted_stack
    return summary


def phase_comparisons(
    aggregate: dict[int, dict[str, Any]],
    control_offset: int,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    control = aggregate[control_offset]
    result = {}
    for offset, condition in aggregate.items():
        if offset == control_offset:
            continue
        miou_difference = (
            condition["shifted_miou_percent"] - control["shifted_miou_percent"]
        )
        comparison = {
            "subpatch_minus_control_miou_pp": miou_difference,
            "image_bootstrap_subpatch_minus_control_miou_pp": (
                bootstrap_paired_miou_delta(
                    control["_shifted_stack"],
                    condition["_shifted_stack"],
                    bootstrap_indices,
                )
            ),
            "subpatch_minus_control_disagreement_rate_pp": (
                condition["prediction_disagreement_rate"]
                - control["prediction_disagreement_rate"]
            )
            * 100.0,
            "interpretation": (
                "Negative mIoU and positive disagreement differences mean the "
                "sub-patch shift is less equivariant than the 16px control."
            ),
            "regions": {},
        }
        for region_name in REPORT_REGIONS:
            candidate_region = condition["regions"][region_name]
            control_region = control["regions"][region_name]
            comparison["regions"][region_name] = {
                "subpatch_minus_control_disagreement_rate_pp": (
                    candidate_region["prediction_disagreement_rate"]
                    - control_region["prediction_disagreement_rate"]
                )
                * 100.0,
                "subpatch_minus_control_error_rate_delta_pp": (
                    candidate_region["shifted_minus_baseline_error_rate_pp"]
                    - control_region["shifted_minus_baseline_error_rate_pp"]
                ),
            }
        result[str(offset)] = comparison
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WHU ViT-S sub-patch translation consistency screen"
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--spatial-diagnostic-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", default=(1, 4, 8, 16))
    parser.add_argument("--control-offset", type=int, default=16)
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument("--valid-margin", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    args = parser.parse_args()

    for path in (args.baseline_checkpoint, args.spatial_diagnostic_json):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    args.offsets = tuple(sorted(set(args.offsets)))
    if not args.offsets or any(offset <= 0 for offset in args.offsets):
        parser.error("--offsets must contain positive integers")
    if args.control_offset not in args.offsets:
        parser.error("--control-offset must occur in --offsets")
    if args.valid_margin < 0:
        parser.error("--valid-margin cannot be negative")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WHU translation evaluation")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    reference = load_spatial_reference(args.spatial_diagnostic_json)
    baseline_sha = file_sha256(args.baseline_checkpoint)
    if baseline_sha != reference["baseline_checkpoint_sha256"]:
        raise AssertionError("baseline checkpoint differs from spatial reference")
    saved_map_validation = validate_saved_baseline_maps(reference, args.max_images)

    model, cfg = load_model(args.baseline_checkpoint, args.seed)
    if args.valid_margin < max(cfg["window_size"]):
        raise ValueError(
            "--valid-margin must be at least the 512px crop size so no retained "
            "sliding crop can attend to zero-filled image borders"
        )
    loader, sample_names, full_test_length = build_test_loader(
        args.max_images, cfg["window_size"]
    )
    selected_reference_images = reference["images"][: len(sample_names)]
    if [image["sample_name"] for image in selected_reference_images] != sample_names:
        raise AssertionError("current WHU test order differs from saved E0 manifest")

    shifts = {
        offset: ((0, offset) if args.axis == "x" else (offset, 0))
        for offset in args.offsets
    }
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    stride = tuple(int(size * 2 / 3) for size in cfg["window_size"])
    records_by_offset: dict[int, list[dict[str, Any]]] = {
        offset: [] for offset in args.offsets
    }
    shifted_digests = {
        offset: hashlib.sha256() for offset in args.offsets
    }

    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"baseline_checkpoint_sha256={baseline_sha}")
    print(f"translation_axis={args.axis}")
    print(f"translation_offsets={args.offsets}")
    print(f"translation_control_offset={args.control_offset}")
    print(f"valid_margin={args.valid_margin}")
    print(f"evaluated_images={len(loader.dataset)}/{full_test_length}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for image_index, ((optical, sar, dataset_label), manifest) in enumerate(
            zip(loader, selected_reference_images, strict=True)
        ):
            image_started = time.perf_counter()
            baseline_prediction = load_index_map(Path(manifest["prediction_path"]))
            saved_label = load_index_map(Path(manifest["label_path"]))
            current_label = np.ascontiguousarray(
                dataset_label.numpy()[0].astype(np.int64, copy=False)
            )
            if not np.array_equal(saved_label, current_label):
                raise AssertionError(
                    f"current label differs from saved E0 map: {manifest['sample_name']}"
                )
            original_slice, shifted_slices = common_translation_slices(
                saved_label.shape, tuple(shifts.values()), args.valid_margin
            )
            target = saved_label[original_slice]
            baseline = baseline_prediction[original_slice]
            valid = (target >= 0) & (target < NUM_CLASSES)
            baseline_confusion = confusion_from_arrays(
                baseline, target, NUM_CLASSES
            )
            region_masks, _ = build_spatial_region_masks(
                saved_label,
                NUM_CLASSES,
                boundary_radii=(0, 1, 2, 4, 8),
                component_area_thresholds=(256, 1024, 4096),
                component_thickness_thresholds=(4, 8, 16),
                patch_size=16,
                union_boundary_radius=0,
                union_component_area=256,
                union_component_thickness=4,
            )

            for offset in args.offsets:
                dy, dx = shifts[offset]
                translated_optical = translate_tensor(optical, dy, dx).to(device)
                translated_sar = translate_tensor(sar, dy, dx).to(device)
                scores = slide_inference(
                    translated_optical,
                    model,
                    dsm=translated_sar,
                    n_output_channels=NUM_CLASSES,
                    crop_size=cfg["window_size"],
                    stride=stride,
                    batch_size=args.inference_batch_size,
                )
                shifted_prediction_full = np.ascontiguousarray(
                    scores.argmax(dim=1).numpy().astype(np.int64, copy=False)
                )
                shifted_digests[offset].update(shifted_prediction_full.tobytes())
                shifted_prediction = shifted_prediction_full[0][shifted_slices[(dy, dx)]]
                shifted_confusion = confusion_from_arrays(
                    shifted_prediction, target, NUM_CLASSES
                )
                disagreement_mask = valid & (shifted_prediction != baseline)
                regions = {}
                for region_name in REPORT_REGIONS:
                    region_mask = region_masks[region_name][original_slice] & valid
                    regions[region_name] = {
                        "valid_pixels": int(valid.sum()),
                        "pixels": int(region_mask.sum()),
                        "disagreement_pixels": int(
                            np.count_nonzero(disagreement_mask & region_mask)
                        ),
                        "baseline_errors": int(
                            np.count_nonzero(
                                region_mask & (baseline != target)
                            )
                        ),
                        "shifted_errors": int(
                            np.count_nonzero(
                                region_mask & (shifted_prediction != target)
                            )
                        ),
                    }
                record = {
                    "image_index": image_index,
                    "sample_name": manifest["sample_name"],
                    "offset": offset,
                    "shift": [dy, dx],
                    "valid_shape": list(target.shape),
                    "valid_pixels": int(valid.sum()),
                    "disagreement_pixels": int(disagreement_mask.sum()),
                    "regions": regions,
                    "_baseline_confusion": baseline_confusion,
                    "_shifted_confusion": shifted_confusion,
                }
                records_by_offset[offset].append(record)
                condition = paired_condition_summary(
                    baseline_confusion,
                    shifted_confusion,
                    record["disagreement_pixels"],
                    record["valid_pixels"],
                    cfg["labels"],
                )
                print(
                    f"image={image_index + 1}/{len(loader.dataset)} "
                    f"name={manifest['sample_name']} offset={offset}px "
                    f"disagreement={condition['prediction_disagreement_rate'] * 100:.4f}% "
                    f"miou_delta={condition['shifted_minus_baseline_miou_pp']:+.4f}pp",
                    flush=True,
                )
                del (
                    translated_optical,
                    translated_sar,
                    scores,
                    shifted_prediction_full,
                    shifted_prediction,
                    shifted_confusion,
                    disagreement_mask,
                )
            del (
                baseline_prediction,
                saved_label,
                current_label,
                target,
                baseline,
                valid,
                baseline_confusion,
                region_masks,
                optical,
                sar,
                dataset_label,
            )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_indices = rng.integers(
        0,
        len(loader.dataset),
        size=(args.bootstrap_replicates, len(loader.dataset)),
        endpoint=False,
    )
    aggregate = {
        offset: aggregate_offset_records(
            records_by_offset[offset], cfg["labels"], bootstrap_indices
        )
        for offset in args.offsets
    }
    comparisons = phase_comparisons(
        aggregate, args.control_offset, bootstrap_indices
    )

    serializable_aggregate = {}
    for offset, condition in aggregate.items():
        serializable_aggregate[str(offset)] = {
            key: value
            for key, value in condition.items()
            if not key.startswith("_")
        }
        serializable_aggregate[str(offset)]["shifted_prediction_sha256"] = (
            shifted_digests[offset].hexdigest()
        )
    serializable_images = []
    for image_index, manifest in enumerate(selected_reference_images):
        image_conditions = {}
        for offset in args.offsets:
            record = records_by_offset[offset][image_index]
            image_conditions[str(offset)] = {
                key: value
                for key, value in record.items()
                if not key.startswith("_")
            }
            image_conditions[str(offset)]["summary"] = paired_condition_summary(
                record["_baseline_confusion"],
                record["_shifted_confusion"],
                record["disagreement_pixels"],
                record["valid_pixels"],
                cfg["labels"],
            )
        serializable_images.append(
            {
                "image_index": image_index,
                "sample_name": manifest["sample_name"],
                "conditions": image_conditions,
            }
        )

    output = {
        "status": "PASS",
        "scope": "full-test" if args.max_images is None else "subset-smoke",
        "scientific_scope": (
            "Tests translation/patch-phase sensitivity after excluding all crops "
            "that can attend to zero-filled image borders; it does not alone prove "
            "irreversible information loss in DINOv3 features."
        ),
        "full_test_length": full_test_length,
        "evaluated_images": len(loader.dataset),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": baseline_sha,
        "spatial_diagnostic_json": str(args.spatial_diagnostic_json.resolve()),
        "saved_map_validation": saved_map_validation,
        "translation": {
            "axis": args.axis,
            "offsets": list(args.offsets),
            "control_offset": args.control_offset,
            "fill": "zero outside translated image",
            "valid_margin": args.valid_margin,
            "valid_region": (
                "One shared intersection for all offsets; both original and shifted "
                "coordinates are at least one 512px crop from full-image borders."
            ),
        },
        "inference": {
            "window_size": list(cfg["window_size"]),
            "stride": list(stride),
            "crop_batch_size": args.inference_batch_size,
            "device": str(device),
        },
        "bootstrap": {
            "unit": "test image",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
        "aggregate": serializable_aggregate,
        "phase_comparisons_to_control": comparisons,
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
    concise = {
        "status": output["status"],
        "scope": output["scope"],
        "evaluated_images": output["evaluated_images"],
        "conditions": {
            str(offset): {
                "disagreement_percent": aggregate[offset][
                    "prediction_disagreement_rate"
                ]
                * 100.0,
                "shifted_minus_baseline_miou_pp": aggregate[offset][
                    "shifted_minus_baseline_miou_pp"
                ],
            }
            for offset in args.offsets
        },
        "phase_comparisons_to_control": comparisons,
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))
    print(f"translation_consistency_result={args.output_path.resolve()}")
    print("translation_consistency_status=PASS")


if __name__ == "__main__":
    main()
