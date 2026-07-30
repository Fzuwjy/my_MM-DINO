"""Zero-training WHU ViT-S phase-ensemble effect screen.

The primary condition averages aligned logits from the normal view and an
8-pixel x-shift.  An equal-size normal+16px ensemble is the control.  Only the
shared interior whose contributing 512px sliding crops cannot see translated
image padding is ensembled; the rest of each image remains the exact baseline.
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
from scripts.evaluate_whu_translation_consistency import (  # noqa: E402
    REPORT_REGIONS,
    load_spatial_reference,
    translate_tensor,
)
from scripts.spatial_diagnostics_common import (  # noqa: E402
    baseline_summary,
    bootstrap_paired_miou_delta,
    build_spatial_region_masks,
    common_translation_slices,
    confusion_from_arrays,
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
CITY_CLASS_INDEX = 1
ROAD_CLASS_INDEX = 5


def aligned_two_view_prediction(
    baseline_scores: torch.Tensor,
    shifted_scores: torch.Tensor,
    original_slice: tuple[slice, slice],
    shifted_slice: tuple[slice, slice],
) -> np.ndarray:
    """Use baseline outside the valid FOV and an equal-logit sum inside it."""

    if baseline_scores.ndim != 4 or shifted_scores.ndim != 4:
        raise ValueError("score tensors must be BCHW")
    if baseline_scores.shape != shifted_scores.shape:
        raise ValueError("baseline and shifted score shapes differ")
    if baseline_scores.shape[0] != 1:
        raise ValueError("phase ensemble currently expects one full image")
    prediction = np.ascontiguousarray(
        baseline_scores.argmax(dim=1).numpy().astype(np.int64, copy=False)
    )
    baseline_crop = baseline_scores[
        0, :, original_slice[0], original_slice[1]
    ]
    shifted_crop = shifted_scores[
        0, :, shifted_slice[0], shifted_slice[1]
    ]
    if baseline_crop.shape != shifted_crop.shape:
        raise ValueError("aligned score crops have different shapes")
    prediction[0][original_slice] = (
        baseline_crop + shifted_crop
    ).argmax(dim=0).numpy().astype(np.int64, copy=False)
    return prediction


def region_error_counts(
    baseline: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    region_masks: dict[str, np.ndarray],
    original_slice: tuple[slice, slice],
) -> dict[str, dict[str, int]]:
    result = {}
    for region_name in REPORT_REGIONS:
        region = region_masks[region_name][original_slice] & valid
        result[region_name] = {
            "valid_pixels": int(valid.sum()),
            "pixels": int(region.sum()),
            "baseline_errors": int(np.count_nonzero(region & (baseline != target))),
            "candidate_errors": int(
                np.count_nonzero(region & (candidate != target))
            ),
        }
    return result


def summarize_region_counts(counts: dict[str, int]) -> dict[str, Any]:
    pixels = counts["pixels"]
    if pixels <= 0:
        return {
            **counts,
            "coverage": None,
            "baseline_error_rate": None,
            "candidate_error_rate": None,
            "candidate_minus_baseline_error_rate_pp": None,
        }
    baseline_rate = counts["baseline_errors"] / pixels
    candidate_rate = counts["candidate_errors"] / pixels
    return {
        **counts,
        "coverage": pixels / counts["valid_pixels"],
        "baseline_error_rate": baseline_rate,
        "candidate_error_rate": candidate_rate,
        "candidate_minus_baseline_error_rate_pp": (
            candidate_rate - baseline_rate
        )
        * 100.0,
    }


def aggregate_candidate(
    records: list[dict[str, Any]],
    class_names: list[str],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    baseline_stack = np.stack([record["_baseline_confusion"] for record in records])
    candidate_stack = np.stack(
        [record["_candidate_confusion"] for record in records]
    )
    baseline = baseline_stack.sum(axis=0)
    candidate = candidate_stack.sum(axis=0)
    baseline_metrics = baseline_summary(baseline, class_names)
    candidate_metrics = baseline_summary(candidate, class_names)
    regions = {}
    for region_name in REPORT_REGIONS:
        counts = {
            key: sum(record["regions"][region_name][key] for record in records)
            for key in (
                "valid_pixels",
                "pixels",
                "baseline_errors",
                "candidate_errors",
            )
        }
        regions[region_name] = summarize_region_counts(counts)
    return {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_baseline_miou_pp": (
            candidate_metrics["miou_percent"] - baseline_metrics["miou_percent"]
        ),
        "class_candidate_minus_baseline_iou_pp": {
            name: (
                candidate_metrics["class_iou_percent"][name]
                - baseline_metrics["class_iou_percent"][name]
            )
            for name in class_names
        },
        "image_bootstrap_candidate_minus_baseline_miou_pp": (
            bootstrap_paired_miou_delta(
                baseline_stack, candidate_stack, bootstrap_indices
            )
        ),
        "regions": regions,
        "_candidate_stack": candidate_stack,
    }


def efficacy_decision(
    primary: dict[str, Any],
    control: dict[str, Any],
    class_names: list[str],
) -> dict[str, Any]:
    """Apply the prospectively fixed go/no-go rule to aggregate results."""

    primary_gain = primary["candidate_minus_baseline_miou_pp"]
    control_gain = control["candidate_minus_baseline_miou_pp"]
    class_deltas = primary["class_candidate_minus_baseline_iou_pp"]
    city_name = class_names[CITY_CLASS_INDEX]
    road_name = class_names[ROAD_CLASS_INDEX]
    city_delta = class_deltas[city_name]
    road_delta = class_deltas[road_name]
    checks = {
        "primary_absolute_gain_at_least_0_10pp": primary_gain >= 0.10,
        "primary_over_control_at_least_0_05pp": (
            primary_gain - control_gain >= 0.05
        ),
        "city_and_road_not_both_decrease": not (
            city_delta < 0.0 and road_delta < 0.0
        ),
    }
    return {
        "outcome": "GO" if all(checks.values()) else "STOP",
        "checks": checks,
        "observed": {
            "primary_absolute_gain_pp": primary_gain,
            "control_absolute_gain_pp": control_gain,
            "primary_over_control_pp": primary_gain - control_gain,
            "city_class": city_name,
            "city_iou_change_pp": city_delta,
            "road_class": road_name,
            "road_iou_change_pp": road_delta,
        },
        "interpretation": (
            "GO permits the next targeted method experiment; it does not establish "
            "a final method. STOP rejects naive equal-logit phase averaging, not the "
            "observed cross-backbone phase-sensitivity phenomenon."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WHU ViT-S normal+8px phase ensemble with normal+16px control"
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--spatial-diagnostic-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--subpatch-offset", type=int, default=8)
    parser.add_argument("--control-offset", type=int, default=16)
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument("--valid-margin", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--miou-tolerance", type=float, default=1e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    args = parser.parse_args()

    for path in (args.baseline_checkpoint, args.spatial_diagnostic_json):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.subpatch_offset <= 0 or args.control_offset <= 0:
        parser.error("translation offsets must be positive")
    if args.subpatch_offset >= args.control_offset:
        parser.error("sub-patch offset must be smaller than control offset")
    if args.valid_margin < 0:
        parser.error("--valid-margin cannot be negative")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.miou_tolerance < 0:
        parser.error("--miou-tolerance cannot be negative")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WHU phase ensemble evaluation")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    reference = load_spatial_reference(args.spatial_diagnostic_json)
    baseline_sha = file_sha256(args.baseline_checkpoint)
    if baseline_sha != reference["baseline_checkpoint_sha256"]:
        raise AssertionError("baseline checkpoint differs from spatial reference")
    model, cfg = load_model(args.baseline_checkpoint, args.seed)
    if args.valid_margin < max(cfg["window_size"]):
        raise ValueError("valid margin must be at least the 512px crop size")
    loader, sample_names, full_test_length = build_test_loader(
        args.max_images, cfg["window_size"]
    )
    manifest = reference["images"][: len(sample_names)]
    if [image["sample_name"] for image in manifest] != sample_names:
        raise AssertionError("current WHU test order differs from spatial reference")

    offsets = (args.subpatch_offset, args.control_offset)
    shifts = {
        offset: ((0, offset) if args.axis == "x" else (offset, 0))
        for offset in offsets
    }
    records_by_offset: dict[int, list[dict[str, Any]]] = {
        offset: [] for offset in offsets
    }
    candidate_digests = {offset: hashlib.sha256() for offset in offsets}
    baseline_digest = hashlib.sha256()
    label_digest = hashlib.sha256()

    device = torch.device(args.device)
    model.to(device)
    model.eval()
    stride = tuple(int(size * 2 / 3) for size in cfg["window_size"])
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"baseline_checkpoint_sha256={baseline_sha}")
    print(f"ensemble_axis={args.axis}")
    print(f"ensemble_offsets={offsets}")
    print(f"valid_margin={args.valid_margin}")
    print(f"evaluated_images={len(loader.dataset)}/{full_test_length}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for image_index, ((optical, sar, label_tensor), expected_image) in enumerate(
            zip(loader, manifest, strict=True)
        ):
            label_full = np.ascontiguousarray(
                label_tensor.numpy().astype(np.int64, copy=False)
            )
            label_digest.update(label_full.tobytes())
            baseline_scores = slide_inference(
                optical.to(device),
                model,
                dsm=sar.to(device),
                n_output_channels=NUM_CLASSES,
                crop_size=cfg["window_size"],
                stride=stride,
                batch_size=args.inference_batch_size,
            )
            baseline_prediction = np.ascontiguousarray(
                baseline_scores.argmax(dim=1).numpy().astype(np.int64, copy=False)
            )
            baseline_digest.update(baseline_prediction.tobytes())
            original_slice, shifted_slices = common_translation_slices(
                label_full[0].shape, tuple(shifts.values()), args.valid_margin
            )
            target_crop = label_full[0][original_slice]
            baseline_crop = baseline_prediction[0][original_slice]
            valid_crop = (target_crop >= 0) & (target_crop < NUM_CLASSES)
            region_masks, _ = build_spatial_region_masks(
                label_full[0],
                NUM_CLASSES,
                boundary_radii=(0, 1, 2, 4, 8),
                component_area_thresholds=(256, 1024, 4096),
                component_thickness_thresholds=(4, 8, 16),
                patch_size=16,
                union_boundary_radius=0,
                union_component_area=256,
                union_component_thickness=4,
            )
            baseline_confusion = confusion_from_arrays(
                baseline_prediction[0], label_full[0], NUM_CLASSES
            )

            for offset in offsets:
                dy, dx = shifts[offset]
                translated_optical = translate_tensor(optical, dy, dx).to(device)
                translated_sar = translate_tensor(sar, dy, dx).to(device)
                shifted_scores = slide_inference(
                    translated_optical,
                    model,
                    dsm=translated_sar,
                    n_output_channels=NUM_CLASSES,
                    crop_size=cfg["window_size"],
                    stride=stride,
                    batch_size=args.inference_batch_size,
                )
                candidate_prediction = aligned_two_view_prediction(
                    baseline_scores,
                    shifted_scores,
                    original_slice,
                    shifted_slices[(dy, dx)],
                )
                candidate_digests[offset].update(candidate_prediction.tobytes())
                candidate_confusion = confusion_from_arrays(
                    candidate_prediction[0], label_full[0], NUM_CLASSES
                )
                candidate_crop = candidate_prediction[0][original_slice]
                regions = region_error_counts(
                    baseline_crop,
                    candidate_crop,
                    target_crop,
                    valid_crop,
                    region_masks,
                    original_slice,
                )
                record = {
                    "image_index": image_index,
                    "sample_name": expected_image["sample_name"],
                    "offset": offset,
                    "regions": regions,
                    "_baseline_confusion": baseline_confusion,
                    "_candidate_confusion": candidate_confusion,
                }
                records_by_offset[offset].append(record)
                baseline_image = baseline_summary(
                    baseline_confusion, cfg["labels"]
                )["miou_percent"]
                candidate_image = baseline_summary(
                    candidate_confusion, cfg["labels"]
                )["miou_percent"]
                print(
                    f"image={image_index + 1}/{len(loader.dataset)} "
                    f"name={expected_image['sample_name']} ensemble=0+{offset}px "
                    f"miou_delta={candidate_image - baseline_image:+.4f}pp",
                    flush=True,
                )
                del (
                    translated_optical,
                    translated_sar,
                    shifted_scores,
                    candidate_prediction,
                    candidate_confusion,
                    candidate_crop,
                )
            del (
                baseline_scores,
                baseline_prediction,
                label_full,
                target_crop,
                baseline_crop,
                valid_crop,
                region_masks,
                baseline_confusion,
                optical,
                sar,
                label_tensor,
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
        offset: aggregate_candidate(
            records_by_offset[offset], cfg["labels"], bootstrap_indices
        )
        for offset in offsets
    }
    subpatch = aggregate[args.subpatch_offset]
    control = aggregate[args.control_offset]
    phase_stack_bootstrap = bootstrap_paired_miou_delta(
        control["_candidate_stack"],
        subpatch["_candidate_stack"],
        bootstrap_indices,
    )

    full_validation = {
        "checked": args.max_images is None,
        "prediction_sha256": baseline_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
    }
    if args.max_images is None:
        full_validation.update(
            {
                "prediction_sha256_equal": baseline_digest.hexdigest()
                == reference["prediction_sha256"],
                "label_sha256_equal": label_digest.hexdigest()
                == reference["label_sha256"],
                "confusion_equal": subpatch["baseline"]["confusion"]
                == reference["aggregate"]["baseline"]["confusion"],
                "miou_within_tolerance": abs(
                    subpatch["baseline"]["miou"]
                    - reference["aggregate"]["baseline"]["miou"]
                )
                <= args.miou_tolerance,
            }
        )
        if not all(
            full_validation[name]
            for name in (
                "prediction_sha256_equal",
                "label_sha256_equal",
                "confusion_equal",
                "miou_within_tolerance",
            )
        ):
            raise AssertionError(f"phase ensemble failed E0 validation: {full_validation}")

    serializable_aggregate = {}
    for offset, result in aggregate.items():
        serializable_aggregate[str(offset)] = {
            key: value for key, value in result.items() if not key.startswith("_")
        }
        serializable_aggregate[str(offset)]["candidate_prediction_sha256"] = (
            candidate_digests[offset].hexdigest()
        )
    primary_gain = subpatch["candidate_minus_baseline_miou_pp"]
    control_gain = control["candidate_minus_baseline_miou_pp"]
    decision = efficacy_decision(subpatch, control, cfg["labels"])
    output = {
        "status": "PASS",
        "scope": "full-test" if args.max_images is None else "subset-smoke",
        "scientific_scope": (
            "Zero-training direct-effect screen. A positive result shows usable "
            "phase diversity, not a deployable efficiency improvement or proof "
            "that DINOv3 features irreversibly lost spatial information."
        ),
        "full_test_length": full_test_length,
        "evaluated_images": len(loader.dataset),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": baseline_sha,
        "spatial_diagnostic_json": str(args.spatial_diagnostic_json.resolve()),
        "baseline_validation": full_validation,
        "protocol": {
            "primary": f"equal logits: normal + {args.subpatch_offset}px",
            "control": f"equal logits: normal + {args.control_offset}px",
            "axis": args.axis,
            "valid_margin": args.valid_margin,
            "outside_valid_region": "exact normal-view baseline logits",
            "inside_valid_region": "unscaled sum of two aligned logits",
        },
        "decision_rule": {
            "primary_absolute_gain_pp": 0.10,
            "primary_over_control_pp": 0.05,
            "class_guard": "city and road must not both decrease",
        },
        "efficacy_decision": decision,
        "aggregate": serializable_aggregate,
        "primary_minus_control": {
            "miou_pp": primary_gain - control_gain,
            "image_bootstrap_miou_pp": phase_stack_bootstrap,
            "class_iou_pp": {
                name: (
                    subpatch["candidate"]["class_iou_percent"][name]
                    - control["candidate"]["class_iou_percent"][name]
                )
                for name in cfg["labels"]
            },
        },
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
    print(
        json.dumps(
            {
                "status": output["status"],
                "scope": output["scope"],
                "evaluated_images": output["evaluated_images"],
                "baseline_validation": full_validation,
                "primary_gain_pp": primary_gain,
                "control_gain_pp": control_gain,
                "efficacy_decision": decision,
                "primary_minus_control": output["primary_minus_control"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"phase_ensemble_result={args.output_path.resolve()}")
    print("phase_ensemble_status=PASS")


if __name__ == "__main__":
    main()
