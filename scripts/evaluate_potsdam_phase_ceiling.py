"""Zero-training Potsdam ViT-L spatial-phase ceiling evaluation.

This runner applies the accepted WHU phase protocol directly to the faithful
Potsdam multimodal checkpoint.  One acquisition produces the normal K1
baseline, the normal+x8 K2 condition, the four-view 8-pixel K4 ceiling, and a
matched four-view 16-pixel control.  It does not train or alter model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from configs import get_cfg  # noqa: E402
from datasets import build_dataset  # noqa: E402
from utils.inference import slide_inference  # noqa: E402
from utils.utils import set_seed  # noqa: E402


DATASET_NAME = "Potsdam"
MODEL_NAME = "DINOv3"
BACKBONE_TYPE = "dinov3_vitl16"
NUM_MODALITIES = 2
CLASS_NAMES = ("roads", "buildings", "low veg.", "trees", "cars", "clutter")
HEADLINE_CLASS_NAMES = CLASS_NAMES[:-1]
EXPECTED_FULL_BASELINE_MIOU_PERCENT = 86.10387847691046
PHASE_OFFSET = 8
CONTROL_OFFSET = 16
PHASE_NAMES = ("normal", "x", "y", "xy")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def four_phase_shifts(offset: int) -> tuple[tuple[int, int], ...]:
    offset = int(offset)
    if offset <= 0:
        raise ValueError("phase offset must be positive")
    return ((0, 0), (0, offset), (offset, 0), (offset, offset))


def translate_tensor(tensor: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate BCHW data with zero fill and no wraparound."""

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
    translated = torch.zeros_like(tensor)
    translated[
        ...,
        source_y_start + dy : source_y_end + dy,
        source_x_start + dx : source_x_end + dx,
    ] = tensor[
        ...,
        source_y_start:source_y_end,
        source_x_start:source_x_end,
    ]
    return translated


def common_translation_slices(
    shape: tuple[int, int],
    shifts: Sequence[tuple[int, int]],
    margin: int,
) -> tuple[tuple[slice, slice], dict[tuple[int, int], tuple[slice, slice]]]:
    """Return paired original/translated slices shared by all shifts."""

    if len(shape) != 2 or any(int(value) <= 0 for value in shape):
        raise ValueError("shape must contain positive height and width")
    if not shifts:
        raise ValueError("at least one shift is required")
    if margin < 0:
        raise ValueError("margin cannot be negative")
    height, width = (int(value) for value in shape)
    normalized = tuple((int(dy), int(dx)) for dy, dx in shifts)
    y_start = max([margin, *[margin - dy for dy, _ in normalized]])
    y_end = min(
        [height - margin, *[height - margin - dy for dy, _ in normalized]]
    )
    x_start = max([margin, *[margin - dx for _, dx in normalized]])
    x_end = min(
        [width - margin, *[width - margin - dx for _, dx in normalized]]
    )
    if y_start >= y_end or x_start >= x_end:
        raise ValueError("translation intersection is empty")
    original = (slice(y_start, y_end), slice(x_start, x_end))
    shifted = {
        (dy, dx): (
            slice(y_start + dy, y_end + dy),
            slice(x_start + dx, x_end + dx),
        )
        for dy, dx in normalized
    }
    return original, shifted


def prediction_from_score_sum(
    baseline_prediction: np.ndarray,
    score_sum: torch.Tensor,
    original_slice: tuple[slice, slice],
) -> np.ndarray:
    """Replace the common interior by fused logits and retain K1 outside."""

    baseline = np.asarray(baseline_prediction)
    if baseline.ndim != 2:
        raise ValueError("baseline prediction must be 2-D")
    if score_sum.ndim != 3 or score_sum.shape[0] != len(CLASS_NAMES):
        raise ValueError("score sum must have shape [6, H, W]")
    expected = baseline[original_slice].shape
    if tuple(score_sum.shape[1:]) != expected:
        raise ValueError("score sum does not match the replacement region")
    result = np.ascontiguousarray(baseline.copy())
    result[original_slice] = (
        score_sum.argmax(dim=0).numpy().astype(result.dtype, copy=False)
    )
    return result


def confusion_from_arrays(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must be matching 2-D arrays")
    valid = (target >= 0) & (target < len(CLASS_NAMES))
    if np.any(valid & ((prediction < 0) | (prediction >= len(CLASS_NAMES)))):
        raise ValueError("prediction contains an invalid class on a valid target")
    encoded = (
        target[valid].astype(np.int64, copy=False) * len(CLASS_NAMES)
        + prediction[valid].astype(np.int64, copy=False)
    )
    return np.bincount(
        encoded, minlength=len(CLASS_NAMES) ** 2
    ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))


def class_ious(confusion: np.ndarray) -> np.ndarray:
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ValueError("confusion matrix must be 6x6")
    diagonal = np.diag(matrix)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - diagonal
    values = np.full(len(CLASS_NAMES), np.nan, dtype=np.float64)
    np.divide(diagonal, union, out=values, where=union > 0)
    return values


def headline_miou(confusion: np.ndarray) -> float:
    """Match the released Potsdam metric by excluding the clutter class."""

    values = class_ious(confusion)[: len(HEADLINE_CLASS_NAMES)]
    return float(np.nanmean(values))


def metric_summary(confusion: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.int64)
    values = class_ious(matrix)
    return {
        "confusion": matrix.tolist(),
        "valid_pixels": int(matrix.sum()),
        "miou_percent": headline_miou(matrix) * 100.0,
        "class_iou_percent": {
            name: (float(value * 100.0) if np.isfinite(value) else None)
            for name, value in zip(CLASS_NAMES, values, strict=True)
        },
        "headline_classes": list(HEADLINE_CLASS_NAMES),
        "excluded_from_headline_miou": ["clutter"],
    }


def batch_headline_miou(confusions: np.ndarray) -> np.ndarray:
    matrices = np.asarray(confusions, dtype=np.float64)
    diagonal = np.diagonal(matrices, axis1=1, axis2=2)
    union = matrices.sum(axis=2) + matrices.sum(axis=1) - diagonal
    values = np.full(union.shape, np.nan, dtype=np.float64)
    np.divide(diagonal, union, out=values, where=union > 0)
    return np.nanmean(values[:, : len(HEADLINE_CLASS_NAMES)], axis=1)


def bootstrap_delta(
    reference_stack: np.ndarray,
    candidate_stack: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any] | None:
    reference = np.asarray(reference_stack, dtype=np.int64)
    candidate = np.asarray(candidate_stack, dtype=np.int64)
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("confusion stacks must match [image, 6, 6]")
    if reference.shape[0] < 2 or replicates <= 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, reference.shape[0], size=(replicates, reference.shape[0]), endpoint=False
    )
    reference_samples = reference[indices].sum(axis=1)
    candidate_samples = candidate[indices].sum(axis=1)
    deltas = (
        batch_headline_miou(candidate_samples)
        - batch_headline_miou(reference_samples)
    ) * 100.0
    lower, median, upper = np.percentile(deltas, [2.5, 50.0, 97.5])
    return {
        "unit": "test tile",
        "replicates": int(replicates),
        "seed": int(seed),
        "median_pp": float(median),
        "ci95_pp": [float(lower), float(upper)],
        "interpretation": "descriptive six-tile cluster bootstrap",
    }


def ceiling_interpretation(
    k2_gain_pp: float, k4_gain_pp: float, control_gain_pp: float
) -> dict[str, Any]:
    incremental = k4_gain_pp - k2_gain_pp
    over_control = k4_gain_pp - control_gain_pp
    if k4_gain_pp <= 0.0:
        outcome = "NO_POSITIVE_K4_CEILING"
        text = "The four-view 8px phase ceiling does not improve the Potsdam baseline."
    elif over_control >= 0.05:
        outcome = "POSITIVE_SUBPATCH_PHASE_CEILING"
        text = (
            "K4-8 improves K1 and beats the equal-cost 16px control by at least "
            "0.05pp, supporting a Potsdam sub-patch phase effect."
        )
    else:
        outcome = "POSITIVE_ENSEMBLE_CEILING_NOT_PHASE_SPECIFIC"
        text = (
            "K4-8 improves K1, but its advantage over the 16px control is below "
            "0.05pp; this establishes an ensemble gain, not the same sub-patch "
            "phase-specific property as WHU."
        )
    return {
        "outcome": outcome,
        "k2_x8_gain_pp": float(k2_gain_pp),
        "k4_8_gain_pp": float(k4_gain_pp),
        "k4_16_control_gain_pp": float(control_gain_pp),
        "k4_8_minus_k2_x8_pp": float(incremental),
        "k4_8_minus_k4_16_control_pp": float(over_control),
        "interpretation": text,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Potsdam ViT-L multimodal K1/K2/K4 spatial-phase ceiling"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--valid-margin", type=int, default=512)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    parser.add_argument("--baseline-miou-tolerance-pp", type=float, default=1e-8)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.valid_margin < 0:
        parser.error("--valid-margin cannot be negative")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates cannot be negative")
    if args.baseline_miou_tolerance_pp < 0:
        parser.error("--baseline-miou-tolerance-pp cannot be negative")
    return args


def load_faithful_model(checkpoint: Path, seed: int):
    set_seed(seed)
    cfg = get_cfg(
        MODEL_NAME,
        DATASET_NAME,
        num_modalities=NUM_MODALITIES,
        use_lora=False,
        r=3,
        backbone_type=BACKBONE_TYPE,
    )
    model = cfg["model"]
    if getattr(model, "use_lora", None) is not False:
        raise AssertionError("model construction unexpectedly enabled LoRA")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise TypeError("checkpoint must contain a model state dict")
    state = payload["model"]
    lora_markers = (
        ".linear_a_q.",
        ".linear_b_q.",
        ".linear_a_v.",
        ".linear_b_v.",
    )
    lora_keys = [key for key in state if any(marker in key for marker in lora_markers)]
    if lora_keys:
        raise AssertionError(f"checkpoint unexpectedly contains LoRA tensors: {lora_keys[:4]}")
    model.load_state_dict(state, strict=True)
    cfg["optimizer"] = None
    cfg["scheduler"] = None
    return model, cfg, {
        "top_level_keys": sorted(str(key) for key in payload),
        "state_tensor_count": len(state),
        "lora_tensor_count": len(lora_keys),
        "strict_load": True,
    }


def build_test_loader(max_images: int | None, window_size: tuple[int, int]):
    base_dataset = build_dataset(
        DATASET_NAME,
        "test",
        window_size=window_size,
        model_name=MODEL_NAME,
        modality="multi",
        backbone_type=BACKBONE_TYPE,
    )
    full_length = len(base_dataset)
    selected_length = full_length if max_images is None else max_images
    if selected_length > full_length:
        raise ValueError(f"--max-images exceeds test length: {selected_length} > {full_length}")
    indices = list(range(selected_length))
    dataset = torch.utils.data.Subset(base_dataset, indices)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    sample_names = [Path(base_dataset.data_files[index]).stem for index in indices]
    return loader, sample_names, full_length


def summarize_condition(
    records: list[dict[str, Any]], condition: str, baseline_stack: np.ndarray
) -> dict[str, Any]:
    stack = np.stack([record[condition] for record in records])
    summary = metric_summary(stack.sum(axis=0))
    baseline_miou = headline_miou(baseline_stack.sum(axis=0)) * 100.0
    summary["minus_k1_miou_pp"] = summary["miou_percent"] - baseline_miou
    summary["bootstrap_minus_k1"] = bootstrap_delta(
        baseline_stack, stack, records[0]["bootstrap_replicates"], records[0]["bootstrap_seed"]
    )
    summary["per_image_miou_percent"] = [
        metric_summary(confusion)["miou_percent"] for confusion in stack
    ]
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Potsdam phase ceiling")
    checkpoint_sha = file_sha256(args.checkpoint)
    model, cfg, checkpoint_audit = load_faithful_model(args.checkpoint, args.seed)
    if tuple(cfg["window_size"]) != (512, 512):
        raise AssertionError(f"unexpected crop size: {cfg['window_size']}")
    if tuple(cfg["labels"]) != CLASS_NAMES:
        raise AssertionError(f"unexpected Potsdam labels: {cfg['labels']}")
    if args.valid_margin < max(cfg["window_size"]):
        raise ValueError("valid margin must be at least the 512px crop size")

    loader, sample_names, full_test_length = build_test_loader(
        args.max_images, cfg["window_size"]
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    stride = tuple(int(size * 2 / 3) for size in cfg["window_size"])
    if stride != (341, 341):
        raise AssertionError(f"unexpected stride: {stride}")

    primary_shifts = four_phase_shifts(PHASE_OFFSET)
    control_shifts = four_phase_shifts(CONTROL_OFFSET)
    all_nonzero_shifts = tuple(
        shift
        for shifts in (primary_shifts, control_shifts)
        for shift in shifts
        if shift != (0, 0)
    )
    if len(set(all_nonzero_shifts)) != 6:
        raise AssertionError("primary and control nonzero phases must be distinct")

    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda={torch.version.cuda}", flush=True)
    print(f"gpu={torch.cuda.get_device_name(device)}", flush=True)
    print(f"checkpoint_sha256={checkpoint_sha}", flush=True)
    print(f"checkpoint_strict_load={checkpoint_audit['strict_load']}", flush=True)
    print("use_lora=False verified", flush=True)
    print(f"primary_phases={primary_shifts}", flush=True)
    print(f"control_phases={control_shifts}", flush=True)
    print(f"evaluated_images={len(loader.dataset)}/{full_test_length}", flush=True)

    records: list[dict[str, Any]] = []
    prediction_digests = {
        name: hashlib.sha256() for name in ("k1", "k2_x8", "k4_8", "k4_16")
    }
    label_digest = hashlib.sha256()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    with torch.inference_mode():
        for image_index, ((optical, dsm, label_tensor), sample_name) in enumerate(
            zip(loader, sample_names, strict=True)
        ):
            image_started = time.perf_counter()
            label = np.ascontiguousarray(
                label_tensor[0].numpy().astype(np.int16, copy=False)
            )
            shape = tuple(int(value) for value in label.shape)
            label_digest.update(label.tobytes())
            common_slice, shifted_slices = common_translation_slices(
                shape, all_nonzero_shifts, args.valid_margin
            )
            x_slice, x_shifted_slices = common_translation_slices(
                shape,
                ((0, PHASE_OFFSET), (0, CONTROL_OFFSET)),
                args.valid_margin,
            )

            print(
                f"image={image_index + 1}/{len(loader.dataset)} name={sample_name} "
                f"shape={shape} phase=normal start",
                flush=True,
            )
            baseline_scores = slide_inference(
                optical.to(device),
                model,
                dsm=dsm.to(device),
                n_output_channels=len(CLASS_NAMES),
                crop_size=cfg["window_size"],
                stride=stride,
                batch_size=args.inference_batch_size,
            )
            baseline_prediction = np.ascontiguousarray(
                baseline_scores[0].argmax(dim=0).numpy().astype(np.int16, copy=False)
            )
            prediction_digests["k1"].update(baseline_prediction.tobytes())
            score_sums = {
                PHASE_OFFSET: baseline_scores[
                    0, :, common_slice[0], common_slice[1]
                ].clone(),
                CONTROL_OFFSET: baseline_scores[
                    0, :, common_slice[0], common_slice[1]
                ].clone(),
            }
            k2_prediction = None

            for shift in all_nonzero_shifts:
                dy, dx = shift
                phase_name = f"dy{dy}_dx{dx}"
                print(
                    f"image={image_index + 1}/{len(loader.dataset)} "
                    f"name={sample_name} phase={phase_name} start",
                    flush=True,
                )
                phase_scores = slide_inference(
                    translate_tensor(optical, dy, dx).to(device),
                    model,
                    dsm=translate_tensor(dsm, dy, dx).to(device),
                    n_output_channels=len(CLASS_NAMES),
                    crop_size=cfg["window_size"],
                    stride=stride,
                    batch_size=args.inference_batch_size,
                )
                aligned = phase_scores[
                    0,
                    :,
                    shifted_slices[shift][0],
                    shifted_slices[shift][1],
                ]
                if shift in primary_shifts:
                    score_sums[PHASE_OFFSET].add_(aligned)
                if shift in control_shifts:
                    score_sums[CONTROL_OFFSET].add_(aligned)

                if shift == (0, PHASE_OFFSET):
                    x_score_sum = (
                        baseline_scores[0, :, x_slice[0], x_slice[1]]
                        + phase_scores[
                            0,
                            :,
                            x_shifted_slices[shift][0],
                            x_shifted_slices[shift][1],
                        ]
                    )
                    k2_prediction = prediction_from_score_sum(
                        baseline_prediction, x_score_sum, x_slice
                    )
                    prediction_digests["k2_x8"].update(k2_prediction.tobytes())
                    del x_score_sum
                del phase_scores, aligned

            if k2_prediction is None:
                raise AssertionError("failed to construct K2-x8")
            k4_prediction = prediction_from_score_sum(
                baseline_prediction, score_sums[PHASE_OFFSET], common_slice
            )
            control_prediction = prediction_from_score_sum(
                baseline_prediction, score_sums[CONTROL_OFFSET], common_slice
            )
            prediction_digests["k4_8"].update(k4_prediction.tobytes())
            prediction_digests["k4_16"].update(control_prediction.tobytes())

            image_confusions = {
                "k1": confusion_from_arrays(baseline_prediction, label),
                "k2_x8": confusion_from_arrays(k2_prediction, label),
                "k4_8": confusion_from_arrays(k4_prediction, label),
                "k4_16": confusion_from_arrays(control_prediction, label),
            }
            image_metrics = {
                name: metric_summary(confusion)["miou_percent"]
                for name, confusion in image_confusions.items()
            }
            records.append(
                {
                    **image_confusions,
                    "image_index": image_index,
                    "sample_name": sample_name,
                    "shape_hw": list(shape),
                    "bootstrap_replicates": args.bootstrap_replicates,
                    "bootstrap_seed": args.bootstrap_seed,
                }
            )
            print(
                f"image={image_index + 1}/{len(loader.dataset)} name={sample_name} "
                f"K1={image_metrics['k1']:.6f}% "
                f"K2x8={image_metrics['k2_x8']:.6f}% "
                f"K4x8={image_metrics['k4_8']:.6f}% "
                f"K4x16={image_metrics['k4_16']:.6f}% "
                f"seconds={time.perf_counter() - image_started:.1f}",
                flush=True,
            )
            del (
                optical,
                dsm,
                label_tensor,
                label,
                baseline_scores,
                baseline_prediction,
                k2_prediction,
                k4_prediction,
                control_prediction,
                score_sums,
            )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    baseline_stack = np.stack([record["k1"] for record in records])
    aggregate = {
        "k1": metric_summary(baseline_stack.sum(axis=0)),
        "k2_x8": summarize_condition(records, "k2_x8", baseline_stack),
        "k4_8": summarize_condition(records, "k4_8", baseline_stack),
        "k4_16_control": summarize_condition(records, "k4_16", baseline_stack),
    }
    aggregate["k1"]["minus_k1_miou_pp"] = 0.0
    aggregate["k1"]["per_image_miou_percent"] = [
        metric_summary(confusion)["miou_percent"] for confusion in baseline_stack
    ]

    baseline_delta = (
        aggregate["k1"]["miou_percent"] - EXPECTED_FULL_BASELINE_MIOU_PERCENT
    )
    baseline_validation = {
        "checked": args.max_images is None,
        "expected_miou_percent": EXPECTED_FULL_BASELINE_MIOU_PERCENT,
        "actual_miou_percent": aggregate["k1"]["miou_percent"],
        "actual_minus_expected_pp": baseline_delta,
        "tolerance_pp": args.baseline_miou_tolerance_pp,
        "within_tolerance": (
            abs(baseline_delta) <= args.baseline_miou_tolerance_pp
            if args.max_images is None
            else None
        ),
    }
    if args.max_images is None and not baseline_validation["within_tolerance"]:
        raise AssertionError(
            "K1 failed to reproduce the faithful epoch-35 metric: "
            f"{baseline_validation}"
        )

    interpretation = ceiling_interpretation(
        aggregate["k2_x8"]["minus_k1_miou_pp"],
        aggregate["k4_8"]["minus_k1_miou_pp"],
        aggregate["k4_16_control"]["minus_k1_miou_pp"],
    )
    output = {
        "status": "PASS",
        "scope": "full-test" if args.max_images is None else "subset-smoke",
        "scientific_scope": (
            "Zero-training optimistic spatial-phase ceiling on the faithful "
            "Potsdam multimodal ViT-L checkpoint; not a deployable method."
        ),
        "dataset": DATASET_NAME,
        "model": MODEL_NAME,
        "backbone": BACKBONE_TYPE,
        "num_modalities": NUM_MODALITIES,
        "use_lora": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_audit": checkpoint_audit,
        "full_test_length": full_test_length,
        "evaluated_images": len(records),
        "sample_manifest": [
            {
                "image_index": record["image_index"],
                "sample_name": record["sample_name"],
                "shape_hw": record["shape_hw"],
            }
            for record in records
        ],
        "protocol": {
            "crop_size_hw": list(cfg["window_size"]),
            "stride_hw": list(stride),
            "inference_batch_size": args.inference_batch_size,
            "primary_phases_dy_dx": [list(shift) for shift in primary_shifts],
            "control_phases_dy_dx": [list(shift) for shift in control_shifts],
            "translation": "positive zero-fill, RGB and DSM shifted together",
            "per_phase_inference": (
                "independent released count-normalized 512/341 sliding inference"
            ),
            "fusion": "equal-weight arithmetic mean of aligned float32 logits",
            "valid_margin": args.valid_margin,
            "outside_common_region": "retain exact K1 prediction",
            "metric": (
                "released Potsdam mIoU over roads/buildings/low vegetation/trees/cars; "
                "clutter excluded"
            ),
            "forward_equivalent_cost": {
                "k1": 1.0,
                "k2_x8": 2.0,
                "k4_8": 4.0,
                "k4_16_control": 4.0,
                "combined_unique_phase_acquisition": 7.0,
            },
        },
        "baseline_validation": baseline_validation,
        "aggregate": aggregate,
        "comparisons": {
            "k4_8_minus_k2_x8_pp": (
                aggregate["k4_8"]["miou_percent"]
                - aggregate["k2_x8"]["miou_percent"]
            ),
            "k4_8_minus_k4_16_control_pp": (
                aggregate["k4_8"]["miou_percent"]
                - aggregate["k4_16_control"]["miou_percent"]
            ),
            "k4_8_vs_k4_16_bootstrap": bootstrap_delta(
                np.stack([record["k4_16"] for record in records]),
                np.stack([record["k4_8"] for record in records]),
                args.bootstrap_replicates,
                args.bootstrap_seed,
            ),
        },
        "ceiling_interpretation": interpretation,
        "hashes": {
            "label_int16_sha256": label_digest.hexdigest(),
            **{
                f"{name}_prediction_int16_sha256": digest.hexdigest()
                for name, digest in prediction_digests.items()
            },
        },
        "reproducibility": {
            "git_revision": git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "seed": args.seed,
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
                "miou_percent": {
                    name: condition["miou_percent"]
                    for name, condition in aggregate.items()
                },
                "gain_pp": {
                    name: condition["minus_k1_miou_pp"]
                    for name, condition in aggregate.items()
                },
                "comparisons": output["comparisons"],
                "ceiling_interpretation": interpretation,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"potsdam_phase_ceiling_result={args.output_path.resolve()}", flush=True)
    print("potsdam_phase_ceiling_status=PASS", flush=True)


if __name__ == "__main__":
    main()
