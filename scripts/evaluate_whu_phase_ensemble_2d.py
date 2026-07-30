"""Zero-training WHU ViT-S four-phase 2-D ensemble ceiling screen.

The primary condition averages aligned logits from (0, 0), (0, 8),
(8, 0), and (8, 8).  The equal-cost control uses the corresponding 16-pixel
grid-aligned shifts.  The script also reconstructs the completed normal+x8
two-view condition and validates it against its immutable full-test artifact.
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
from scripts.evaluate_whu_phase_ensemble import (  # noqa: E402
    aggregate_candidate,
    region_error_counts,
)
from scripts.evaluate_whu_translation_consistency import (  # noqa: E402
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


def four_phase_shifts(offset: int) -> tuple[tuple[int, int], ...]:
    """Return the fixed 2-D phase square for one positive offset."""

    offset = int(offset)
    if offset <= 0:
        raise ValueError("phase offset must be positive")
    return ((0, 0), (0, offset), (offset, 0), (offset, offset))


def prediction_from_aligned_score_sum(
    baseline_prediction: np.ndarray,
    aligned_score_sum: torch.Tensor,
    original_slice: tuple[slice, slice],
) -> np.ndarray:
    """Keep the baseline outside ``original_slice`` and fuse inside it."""

    baseline_prediction = np.asarray(baseline_prediction)
    if baseline_prediction.ndim != 3 or baseline_prediction.shape[0] != 1:
        raise ValueError("baseline prediction must be [1, H, W]")
    if aligned_score_sum.ndim != 3:
        raise ValueError("aligned score sum must be [class, H, W]")
    expected_shape = baseline_prediction[0][original_slice].shape
    if tuple(aligned_score_sum.shape[1:]) != expected_shape:
        raise ValueError("aligned score sum does not match the valid region")
    prediction = np.ascontiguousarray(baseline_prediction.copy())
    prediction[0][original_slice] = (
        aligned_score_sum.argmax(dim=0).numpy().astype(np.int64, copy=False)
    )
    return prediction


def load_two_phase_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("scope") != "full-test":
        raise ValueError("two-phase reference must be a full-test PASS")
    if payload.get("evaluated_images") != payload.get("full_test_length"):
        raise ValueError("two-phase reference does not cover the full test set")
    if payload.get("efficacy_decision", {}).get("outcome") != "GO":
        raise ValueError("two-phase reference did not pass its efficacy gate")
    validation = payload.get("baseline_validation", {})
    required = (
        "prediction_sha256_equal",
        "label_sha256_equal",
        "confusion_equal",
        "miou_within_tolerance",
    )
    if not validation.get("checked") or not all(
        validation.get(name) is True for name in required
    ):
        raise ValueError("two-phase reference lacks strict baseline validation")
    return payload


def ceiling_decision(
    primary_gain_pp: float,
    control_gain_pp: float,
    two_phase_gain_pp: float,
    primary_regions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the prospectively documented four-phase interpretation rules."""

    incremental = primary_gain_pp - two_phase_gain_pp
    over_control = primary_gain_pp - control_gain_pp
    strong = primary_gain_pp >= 0.20 and over_control >= 0.05
    limited = not strong and incremental >= 0.05
    saturated = not strong and not limited and incremental < 0.03
    if strong:
        outcome = "STRONG_GO"
    elif limited:
        outcome = "LIMITED_GO"
    elif saturated:
        outcome = "SATURATED"
    else:
        outcome = "BORDERLINE"

    small_delta = primary_regions["component_area_le_256px2"][
        "candidate_minus_baseline_error_rate_pp"
    ]
    thin_delta = primary_regions["component_thickness_le_4px"][
        "candidate_minus_baseline_error_rate_pp"
    ]
    return {
        "outcome": outcome,
        "checks": {
            "strong_absolute_gain_at_least_0_20pp": primary_gain_pp >= 0.20,
            "strong_primary_over_control_at_least_0_05pp": over_control >= 0.05,
            "limited_increment_over_two_phase_at_least_0_05pp": (
                incremental >= 0.05
            ),
            "saturation_increment_below_0_03pp": incremental < 0.03,
            "small_region_worsened": small_delta > 0.0,
            "thin_region_worsened": thin_delta > 0.0,
        },
        "observed": {
            "primary_absolute_gain_pp": primary_gain_pp,
            "control_absolute_gain_pp": control_gain_pp,
            "primary_over_control_pp": over_control,
            "two_phase_absolute_gain_pp": two_phase_gain_pp,
            "primary_over_two_phase_pp": incremental,
            "small_region_error_rate_change_pp": small_delta,
            "thin_region_error_rate_change_pp": thin_delta,
        },
        "interpretation": {
            "STRONG_GO": (
                "The four-phase ceiling is practically meaningful; proceed to a "
                "single-pass approximation method."
            ),
            "LIMITED_GO": (
                "Two-dimensional phase information adds value, but stop TTA "
                "expansion and allow only one low-cost distillation or selective-"
                "fusion screen."
            ),
            "SATURATED": (
                "The phase benefit saturates near the completed two-view result; "
                "do not test more shifts or views."
            ),
            "BORDERLINE": (
                "The incremental result falls in the predeclared 0.03-0.05pp "
                "gray zone; stop TTA expansion and judge only a minimal method "
                "screen against its cost."
            ),
        }[outcome],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "WHU ViT-S four-phase 8px 2-D ensemble with four-phase 16px control"
        )
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--spatial-diagnostic-json", type=Path, required=True)
    parser.add_argument("--two-phase-reference-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--subpatch-offset", type=int, default=8)
    parser.add_argument("--control-offset", type=int, default=16)
    parser.add_argument("--valid-margin", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--miou-tolerance", type=float, default=1e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    args = parser.parse_args()

    for path in (
        args.baseline_checkpoint,
        args.spatial_diagnostic_json,
        args.two_phase_reference_json,
    ):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.subpatch_offset <= 0 or args.control_offset <= 0:
        parser.error("phase offsets must be positive")
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


def serializable_aggregate(
    aggregate: dict[str, Any], candidate_digest: Any
) -> dict[str, Any]:
    result = {key: value for key, value in aggregate.items() if not key.startswith("_")}
    result["candidate_prediction_sha256"] = candidate_digest.hexdigest()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WHU phase ensemble evaluation")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    spatial_reference = load_spatial_reference(args.spatial_diagnostic_json)
    two_phase_reference = load_two_phase_reference(args.two_phase_reference_json)
    baseline_sha = file_sha256(args.baseline_checkpoint)
    if baseline_sha != spatial_reference["baseline_checkpoint_sha256"]:
        raise AssertionError("baseline checkpoint differs from spatial reference")
    if baseline_sha != two_phase_reference["baseline_checkpoint_sha256"]:
        raise AssertionError("baseline checkpoint differs from two-phase reference")

    model, cfg = load_model(args.baseline_checkpoint, args.seed)
    if args.valid_margin < max(cfg["window_size"]):
        raise ValueError("valid margin must be at least the 512px crop size")
    loader, sample_names, full_test_length = build_test_loader(
        args.max_images, cfg["window_size"]
    )
    manifest = spatial_reference["images"][: len(sample_names)]
    if [image["sample_name"] for image in manifest] != sample_names:
        raise AssertionError("current WHU test order differs from spatial reference")

    offsets = (args.subpatch_offset, args.control_offset)
    condition_shifts = {offset: four_phase_shifts(offset) for offset in offsets}
    nonzero_shifts = tuple(
        shift
        for offset in offsets
        for shift in condition_shifts[offset]
        if shift != (0, 0)
    )
    if len(set(nonzero_shifts)) != len(nonzero_shifts):
        raise ValueError("primary and control phase shifts unexpectedly overlap")

    records_by_offset: dict[int, list[dict[str, Any]]] = {
        offset: [] for offset in offsets
    }
    two_phase_records: list[dict[str, Any]] = []
    candidate_digests = {offset: hashlib.sha256() for offset in offsets}
    two_phase_digest = hashlib.sha256()
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
    print(f"primary_four_phases={condition_shifts[args.subpatch_offset]}")
    print(f"control_four_phases={condition_shifts[args.control_offset]}")
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
            baseline_confusion = confusion_from_arrays(
                baseline_prediction[0], label_full[0], NUM_CLASSES
            )

            original_slice, shifted_slices = common_translation_slices(
                label_full[0].shape, nonzero_shifts, args.valid_margin
            )
            x_original_slice, x_shifted_slices = common_translation_slices(
                label_full[0].shape,
                ((0, args.subpatch_offset), (0, args.control_offset)),
                args.valid_margin,
            )
            target_crop = label_full[0][original_slice]
            baseline_crop = baseline_prediction[0][original_slice]
            valid_crop = (target_crop >= 0) & (target_crop < NUM_CLASSES)
            x_target_crop = label_full[0][x_original_slice]
            x_baseline_crop = baseline_prediction[0][x_original_slice]
            x_valid_crop = (x_target_crop >= 0) & (x_target_crop < NUM_CLASSES)
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

            baseline_score_crop = baseline_scores[
                0, :, original_slice[0], original_slice[1]
            ]
            x_baseline_score_crop = baseline_scores[
                0, :, x_original_slice[0], x_original_slice[1]
            ].clone()
            score_sums = {
                args.subpatch_offset: baseline_score_crop.clone(),
                args.control_offset: baseline_score_crop.clone(),
            }
            del baseline_score_crop, baseline_scores
            two_phase_prediction = None
            two_phase_confusion = None
            two_phase_regions = None

            for shift in nonzero_shifts:
                dy, dx = shift
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
                aligned_crop = shifted_scores[
                    0,
                    :,
                    shifted_slices[shift][0],
                    shifted_slices[shift][1],
                ]
                for offset in offsets:
                    if shift in condition_shifts[offset]:
                        score_sums[offset].add_(aligned_crop)

                if shift == (0, args.subpatch_offset):
                    x_score_sum = (
                        x_baseline_score_crop
                        + shifted_scores[
                            0,
                            :,
                            x_shifted_slices[shift][0],
                            x_shifted_slices[shift][1],
                        ]
                    )
                    two_phase_prediction = prediction_from_aligned_score_sum(
                        baseline_prediction, x_score_sum, x_original_slice
                    )
                    two_phase_digest.update(two_phase_prediction.tobytes())
                    two_phase_confusion = confusion_from_arrays(
                        two_phase_prediction[0], label_full[0], NUM_CLASSES
                    )
                    two_phase_regions = region_error_counts(
                        x_baseline_crop,
                        two_phase_prediction[0][x_original_slice],
                        x_target_crop,
                        x_valid_crop,
                        region_masks,
                        x_original_slice,
                    )
                    del x_score_sum, x_baseline_score_crop

                del (
                    translated_optical,
                    translated_sar,
                    shifted_scores,
                    aligned_crop,
                )

            if (
                two_phase_prediction is None
                or two_phase_confusion is None
                or two_phase_regions is None
            ):
                raise AssertionError("failed to reconstruct the x-only two-phase view")
            two_phase_records.append(
                {
                    "image_index": image_index,
                    "sample_name": expected_image["sample_name"],
                    "regions": two_phase_regions,
                    "_baseline_confusion": baseline_confusion,
                    "_candidate_confusion": two_phase_confusion,
                }
            )

            baseline_image_miou = None
            for offset in offsets:
                candidate_prediction = prediction_from_aligned_score_sum(
                    baseline_prediction, score_sums[offset], original_slice
                )
                candidate_digests[offset].update(candidate_prediction.tobytes())
                candidate_confusion = confusion_from_arrays(
                    candidate_prediction[0], label_full[0], NUM_CLASSES
                )
                regions = region_error_counts(
                    baseline_crop,
                    candidate_prediction[0][original_slice],
                    target_crop,
                    valid_crop,
                    region_masks,
                    original_slice,
                )
                records_by_offset[offset].append(
                    {
                        "image_index": image_index,
                        "sample_name": expected_image["sample_name"],
                        "offset": offset,
                        "regions": regions,
                        "_baseline_confusion": baseline_confusion,
                        "_candidate_confusion": candidate_confusion,
                    }
                )
                if baseline_image_miou is None:
                    baseline_image_miou = baseline_summary(
                        baseline_confusion, cfg["labels"]
                    )["miou_percent"]
                candidate_image_miou = baseline_summary(
                    candidate_confusion, cfg["labels"]
                )["miou_percent"]
                print(
                    f"image={image_index + 1}/{len(loader.dataset)} "
                    f"name={expected_image['sample_name']} four_phase={offset}px "
                    f"miou_delta={candidate_image_miou - baseline_image_miou:+.4f}pp",
                    flush=True,
                )
                del candidate_prediction, candidate_confusion

            del (
                label_full,
                baseline_prediction,
                baseline_confusion,
                target_crop,
                baseline_crop,
                valid_crop,
                x_target_crop,
                x_baseline_crop,
                x_valid_crop,
                region_masks,
                score_sums,
                two_phase_prediction,
                two_phase_confusion,
                two_phase_regions,
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
    two_phase_aggregate = aggregate_candidate(
        two_phase_records, cfg["labels"], bootstrap_indices
    )
    primary = aggregate[args.subpatch_offset]
    control = aggregate[args.control_offset]
    primary_vs_control_bootstrap = bootstrap_paired_miou_delta(
        control["_candidate_stack"],
        primary["_candidate_stack"],
        bootstrap_indices,
    )
    primary_vs_two_phase_bootstrap = bootstrap_paired_miou_delta(
        two_phase_aggregate["_candidate_stack"],
        primary["_candidate_stack"],
        bootstrap_indices,
    )

    baseline_validation = {
        "checked": args.max_images is None,
        "prediction_sha256": baseline_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
    }
    two_phase_validation = {
        "checked": args.max_images is None,
        "candidate_prediction_sha256": two_phase_digest.hexdigest(),
    }
    if args.max_images is None:
        baseline_validation.update(
            {
                "prediction_sha256_equal": baseline_digest.hexdigest()
                == spatial_reference["prediction_sha256"],
                "label_sha256_equal": label_digest.hexdigest()
                == spatial_reference["label_sha256"],
                "confusion_equal": primary["baseline"]["confusion"]
                == spatial_reference["aggregate"]["baseline"]["confusion"],
                "miou_within_tolerance": abs(
                    primary["baseline"]["miou"]
                    - spatial_reference["aggregate"]["baseline"]["miou"]
                )
                <= args.miou_tolerance,
            }
        )
        two_phase_expected = two_phase_reference["aggregate"][
            str(args.subpatch_offset)
        ]
        two_phase_validation.update(
            {
                "candidate_prediction_sha256_equal": (
                    two_phase_digest.hexdigest()
                    == two_phase_expected["candidate_prediction_sha256"]
                ),
                "confusion_equal": two_phase_aggregate["candidate"]["confusion"]
                == two_phase_expected["candidate"]["confusion"],
                "miou_within_tolerance": abs(
                    two_phase_aggregate["candidate"]["miou"]
                    - two_phase_expected["candidate"]["miou"]
                )
                <= args.miou_tolerance,
            }
        )
        required_baseline_checks = (
            "prediction_sha256_equal",
            "label_sha256_equal",
            "confusion_equal",
            "miou_within_tolerance",
        )
        if not all(baseline_validation[name] for name in required_baseline_checks):
            raise AssertionError(
                f"four-phase evaluation failed E0 validation: {baseline_validation}"
            )
        required_two_phase_checks = (
            "candidate_prediction_sha256_equal",
            "confusion_equal",
            "miou_within_tolerance",
        )
        if not all(two_phase_validation[name] for name in required_two_phase_checks):
            raise AssertionError(
                "four-phase evaluation failed two-phase validation: "
                f"{two_phase_validation}"
            )

    primary_gain = primary["candidate_minus_baseline_miou_pp"]
    control_gain = control["candidate_minus_baseline_miou_pp"]
    two_phase_gain = two_phase_aggregate["candidate_minus_baseline_miou_pp"]
    decision = ceiling_decision(
        primary_gain, control_gain, two_phase_gain, primary["regions"]
    )
    output = {
        "status": "PASS",
        "scope": "full-test" if args.max_images is None else "subset-smoke",
        "scientific_scope": (
            "Zero-training 2-D phase-diversity ceiling. It tests whether x/y "
            "sub-patch phases add usable information beyond the completed x-only "
            "two-view result; it is not a deployable single-pass method."
        ),
        "full_test_length": full_test_length,
        "evaluated_images": len(loader.dataset),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": baseline_sha,
        "spatial_diagnostic_json": str(args.spatial_diagnostic_json.resolve()),
        "two_phase_reference_json": str(args.two_phase_reference_json.resolve()),
        "baseline_validation": baseline_validation,
        "two_phase_validation": two_phase_validation,
        "protocol": {
            "primary": [list(shift) for shift in condition_shifts[args.subpatch_offset]],
            "control": [list(shift) for shift in condition_shifts[args.control_offset]],
            "valid_margin": args.valid_margin,
            "outside_valid_region": "exact normal-view baseline logits",
            "inside_valid_region": "unscaled sum of four aligned logits",
            "same_common_valid_region_for_primary_and_control": True,
        },
        "decision_rule": {
            "strong_go": (
                "primary absolute gain >=0.20pp and primary-control >=0.05pp"
            ),
            "limited_go": "primary-two-phase >=0.05pp when strong-go is false",
            "saturated": "primary-two-phase <0.03pp",
            "borderline": "primary-two-phase in [0.03, 0.05)pp",
            "region_guard": "report whether small and thin error rates worsen",
        },
        "ceiling_decision": decision,
        "aggregate": {
            str(offset): serializable_aggregate(
                aggregate[offset], candidate_digests[offset]
            )
            for offset in offsets
        },
        "reconstructed_two_phase_x": serializable_aggregate(
            two_phase_aggregate, two_phase_digest
        ),
        "primary_minus_control": {
            "miou_pp": primary_gain - control_gain,
            "image_bootstrap_miou_pp": primary_vs_control_bootstrap,
            "class_iou_pp": {
                name: (
                    primary["candidate"]["class_iou_percent"][name]
                    - control["candidate"]["class_iou_percent"][name]
                )
                for name in cfg["labels"]
            },
        },
        "primary_minus_two_phase": {
            "miou_pp": primary_gain - two_phase_gain,
            "image_bootstrap_miou_pp": primary_vs_two_phase_bootstrap,
            "class_iou_pp": {
                name: (
                    primary["candidate"]["class_iou_percent"][name]
                    - two_phase_aggregate["candidate"]["class_iou_percent"][name]
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
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "scope": output["scope"],
                "evaluated_images": output["evaluated_images"],
                "baseline_validation": baseline_validation,
                "two_phase_validation": two_phase_validation,
                "primary_gain_pp": primary_gain,
                "control_gain_pp": control_gain,
                "two_phase_gain_pp": two_phase_gain,
                "primary_minus_control": output["primary_minus_control"],
                "primary_minus_two_phase": output["primary_minus_two_phase"],
                "ceiling_decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"phase_ensemble_2d_result={args.output_path.resolve()}")
    print("phase_ensemble_2d_status=PASS")


if __name__ == "__main__":
    main()
