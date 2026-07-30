"""Diagnose Stage-B1 live sparse batching against the sealed raw-crop cache.

This is a correctness diagnostic, not an evaluation.  For one cached image it
executes each shifted phase in three ways:

* ``compact`` packs only the selected K4-support crops into new batches, as the
  current sparse live primitive does;
* ``compact-pad-last`` keeps compact ordering but pads the last short model
  batch to the sealed batch size with duplicate crops, scattering only the real
  outputs;
* ``dense-preserving`` executes the original dense row-major batches and keeps
  only the selected outputs, matching the cache producer's batch membership.

Every selected raw crop is compared with its cached counterpart.  The runner
also compares phase accumulations and, when all three shifted phases are run,
the final K4 logits, predictions, confusion matrix, and mIoU.  Differences are
measured and reported; only invalid inputs, non-finite outputs, count-map
changes, or a broken cached reference fail closed.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
import sys

sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnose_whu_spatial_errors import (  # noqa: E402
    build_test_loader,
    load_model,
)
from scripts.evaluate_whu_phase_cached_replay import (  # noqa: E402
    _git_revision,
    _normal_mean,
    atomic_write_json,
    confusion_matrix,
    file_sha256,
    load_array_descriptor,
    load_b0,
    prepare_crop_stores,
    validate_cache_manifest,
)
from scripts.evaluate_whu_phase_closure import (  # noqa: E402
    load_stage_a,
    metric_summary,
)
from scripts.evaluate_whu_translation_consistency import (  # noqa: E402
    translate_tensor,
)
from scripts.phase_closure_common import (  # noqa: E402
    PHASE_NAMES,
    PHASE_SHIFTS,
    build_phase_closure_geometry,
)
from scripts.phase_sparse_replay_common import (  # noqa: E402
    accumulate_phase_crops,
    aligned_region,
    array_sha256,
    compose_policy_logits,
    endpoint_levels,
    expected_phase_crop_ids,
)
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)


ARTIFACT_TYPE = "whu_phase_sparse_batching_diagnostic"
SCHEMA_VERSION = 1
NUM_CLASSES = 7
MODES = ("compact", "compact-pad-last", "dense-preserving")
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-a-json", type=Path, required=True)
    parser.add_argument("--stage-b0-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=PHASE_NAMES,
        default=list(PHASE_NAMES),
    )
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=list(MODES)
    )
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--worst-crops", type=int, default=12)
    args = parser.parse_args()
    for path in (
        args.cache_manifest,
        args.baseline_checkpoint,
        args.stage_a_json,
        args.stage_b0_json,
    ):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    if args.worst_crops < 0:
        parser.error("--worst-crops must be non-negative")
    args.phases = tuple(dict.fromkeys(args.phases))
    args.modes = tuple(dict.fromkeys(args.modes))
    return args


def _difference_statistics(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Return deterministic numeric and argmax differences for one CHW array."""

    left = np.asarray(candidate)
    right = np.asarray(reference)
    if left.dtype != np.float32 or right.dtype != np.float32:
        raise TypeError("candidate and reference logits must be float32")
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("candidate and reference logits must share a CHW shape")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("candidate and reference logits must be finite")
    difference = np.abs(left - right)
    tolerance = atol + rtol * np.abs(right)
    exceed = difference > tolerance
    candidate_prediction = left.argmax(axis=0)
    reference_prediction = right.argmax(axis=0)
    flips = candidate_prediction != reference_prediction
    maximum_index = np.unravel_index(int(difference.argmax()), difference.shape)
    return {
        "array_equal": bool(np.array_equal(left, right)),
        "allclose": bool(not np.any(exceed)),
        "element_count": int(difference.size),
        "elements_exceeding_tolerance": int(np.count_nonzero(exceed)),
        "maximum_absolute_difference": float(difference[maximum_index]),
        "mean_absolute_difference": float(difference.mean(dtype=np.float64)),
        "root_mean_square_difference": float(
            np.sqrt(np.square(difference, dtype=np.float64).mean())
        ),
        "maximum_difference_chw": [int(value) for value in maximum_index],
        "candidate_at_maximum": float(left[maximum_index]),
        "reference_at_maximum": float(right[maximum_index]),
        "argmax_pixel_count": int(flips.size),
        "argmax_prediction_flips": int(np.count_nonzero(flips)),
    }


def _merge_difference_statistics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one crop-difference record is required")
    element_count = sum(int(item["element_count"]) for item in records)
    pixel_count = sum(int(item["argmax_pixel_count"]) for item in records)
    weighted_abs = sum(
        float(item["mean_absolute_difference"]) * int(item["element_count"])
        for item in records
    )
    weighted_square = sum(
        float(item["root_mean_square_difference"]) ** 2
        * int(item["element_count"])
        for item in records
    )
    maxima = np.asarray(
        [float(item["maximum_absolute_difference"]) for item in records],
        dtype=np.float64,
    )
    return {
        "crop_count": len(records),
        "array_equal_crop_count": sum(bool(item["array_equal"]) for item in records),
        "allclose_crop_count": sum(bool(item["allclose"]) for item in records),
        "crops_with_argmax_flips": sum(
            int(item["argmax_prediction_flips"]) > 0 for item in records
        ),
        "elements_exceeding_tolerance": sum(
            int(item["elements_exceeding_tolerance"]) for item in records
        ),
        "argmax_prediction_flips": sum(
            int(item["argmax_prediction_flips"]) for item in records
        ),
        "argmax_pixel_comparisons": pixel_count,
        "maximum_absolute_difference": float(maxima.max()),
        "median_crop_maximum_absolute_difference": float(np.median(maxima)),
        "p95_crop_maximum_absolute_difference": float(np.quantile(maxima, 0.95)),
        "mean_absolute_difference": float(weighted_abs / element_count),
        "root_mean_square_difference": float(
            np.sqrt(weighted_square / element_count)
        ),
    }


def _batch_shape_breakdown(
    records: Sequence[Mapping[str, Any]], batch_size: int
) -> dict[str, Any]:
    """Separate full model batches from a compact final partial batch."""

    result: dict[str, Any] = {}
    groups = {
        "full_model_batch": [
            item for item in records if int(item["model_batch_size"]) == batch_size
        ],
        "short_model_batch": [
            item for item in records if int(item["model_batch_size"]) < batch_size
        ],
        "padded_final_batch_real_outputs": [
            item for item in records if bool(item["batch_was_padded"])
        ],
    }
    for name, members in groups.items():
        result[name] = (
            _merge_difference_statistics(members)
            if members
            else {"crop_count": 0, "status": "NO_CROPS"}
        )
    return result


def _chunked_logit_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float,
    rtol: float,
    rows: int = 96,
) -> dict[str, Any]:
    """Measure large matching CHW arrays without materializing a full diff."""

    left = np.asarray(candidate)
    right = np.asarray(reference)
    if left.dtype != np.float32 or right.dtype != np.float32:
        raise TypeError("large logits must be float32")
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("large logits must share a CHW shape")
    total = 0
    exceed = 0
    sum_abs = 0.0
    sum_square = 0.0
    maximum = 0.0
    exact = True
    for y0 in range(0, left.shape[1], rows):
        y1 = min(y0 + rows, left.shape[1])
        left_chunk = left[:, y0:y1]
        right_chunk = right[:, y0:y1]
        if not np.all(np.isfinite(left_chunk)) or not np.all(np.isfinite(right_chunk)):
            raise ValueError("large logits contain non-finite values")
        delta = np.abs(left_chunk - right_chunk)
        exact = exact and bool(np.array_equal(left_chunk, right_chunk))
        exceed += int(np.count_nonzero(delta > atol + rtol * np.abs(right_chunk)))
        total += int(delta.size)
        sum_abs += float(delta.sum(dtype=np.float64))
        sum_square += float(np.square(delta, dtype=np.float64).sum())
        if delta.size:
            maximum = max(maximum, float(delta.max()))
    return {
        "array_equal": exact,
        "allclose": exceed == 0,
        "element_count": total,
        "elements_exceeding_tolerance": exceed,
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": sum_abs / total,
        "root_mean_square_difference": float(np.sqrt(sum_square / total)),
    }


def _phase_accumulation_difference(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    shift: tuple[int, int],
    common_bounds: tuple[int, int, int, int],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    candidate_sum = np.asarray(candidate["sum_logits"])
    reference_sum = np.asarray(reference["sum_logits"])
    candidate_count = np.asarray(candidate["count_mat"])
    reference_count = np.asarray(reference["count_mat"])
    if not np.array_equal(candidate_count, reference_count):
        raise AssertionError("live and cached phase count maps differ")
    candidate_common_sum = aligned_region(candidate_sum, shift, common_bounds)
    reference_common_sum = aligned_region(reference_sum, shift, common_bounds)
    count_common = aligned_region(candidate_count, shift, common_bounds)
    candidate_mean = np.zeros(candidate_common_sum.shape, dtype=np.float32)
    reference_mean = np.zeros(reference_common_sum.shape, dtype=np.float32)
    np.divide(
        candidate_common_sum,
        count_common[None],
        out=candidate_mean,
        where=count_common[None] > 0,
    )
    np.divide(
        reference_common_sum,
        count_common[None],
        out=reference_mean,
        where=count_common[None] > 0,
    )
    prediction_flips = int(
        np.count_nonzero(
            candidate_mean.argmax(axis=0) != reference_mean.argmax(axis=0)
        )
    )
    return {
        "count_map_equal": True,
        "count_map_sha256": array_sha256(candidate_count),
        "covered_common_pixels": int(np.count_nonzero(count_common > 0)),
        "common_sum_logits": _chunked_logit_difference(
            candidate_common_sum,
            reference_common_sum,
            atol=atol,
            rtol=rtol,
        ),
        "common_normalized_logits": _chunked_logit_difference(
            candidate_mean,
            reference_mean,
            atol=atol,
            rtol=rtol,
        ),
        "common_phase_argmax_prediction_flips": prediction_flips,
    }


def _original_dense_batch(crop_id: int, crop_count: int, batch_size: int) -> tuple[int, ...]:
    start = (int(crop_id) // batch_size) * batch_size
    return tuple(range(start, min(start + batch_size, crop_count)))


def _execute_mode(
    *,
    phase_inputs: torch.Tensor,
    phase_sar: torch.Tensor,
    model: torch.nn.Module,
    windows: Sequence[Sequence[int]],
    selected_ids: Sequence[int],
    reference_store: Mapping[int, np.ndarray],
    mode: str,
    batch_size: int,
    atol: float,
    rtol: float,
    worst_crops: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode not in MODES:
        raise ValueError(f"unknown execution mode: {mode}")
    selected = tuple(int(value) for value in selected_ids)
    selected_set = set(selected)
    crop_count = len(windows)
    if mode == "dense-preserving":
        base_ids = tuple(range(crop_count))
    else:
        base_ids = selected
    batch_plan: list[tuple[tuple[int, ...], int, bool]] = []
    for offset in range(0, len(base_ids), batch_size):
        real_ids = base_ids[offset : offset + batch_size]
        model_ids = real_ids
        padded = False
        if mode == "compact-pad-last" and len(real_ids) < batch_size:
            if not real_ids:
                continue
            padding = tuple(
                real_ids[index % len(real_ids)]
                for index in range(batch_size - len(real_ids))
            )
            model_ids = real_ids + padding
            padded = True
        batch_plan.append((tuple(model_ids), len(real_ids), padded))
    height, width = (int(value) for value in phase_inputs.shape[-2:])
    score_sum = phase_inputs.new_zeros((1, NUM_CLASSES, height, width))
    count = torch.zeros((1, 1, height, width), dtype=torch.int16, device=phase_inputs.device)
    crop_records: list[dict[str, Any]] = []
    observed: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (batch_ids, real_count, batch_was_padded) in enumerate(batch_plan):
            optical_batch = torch.cat(
                [
                    phase_inputs[:, :, int(w[0]) : int(w[1]), int(w[2]) : int(w[3])]
                    for w in (windows[crop_id] for crop_id in batch_ids)
                ],
                dim=0,
            )
            sar_batch = torch.cat(
                [
                    phase_sar[:, :, int(w[0]) : int(w[1]), int(w[2]) : int(w[3])]
                    for w in (windows[crop_id] for crop_id in batch_ids)
                ],
                dim=0,
            )
            decoded = model(optical_batch, sar_batch)
            if not isinstance(decoded, torch.Tensor):
                raise TypeError("sealed linear decoder must return a tensor")
            if tuple(decoded.shape) != (len(batch_ids), NUM_CLASSES, 512, 512):
                raise ValueError("model crop output shape differs from the sealed protocol")
            decoded = decoded.to(device=phase_inputs.device, dtype=torch.float32)
            if not torch.isfinite(decoded).all():
                raise FloatingPointError("live crop logits contain non-finite values")
            for slot, crop_id in enumerate(batch_ids[:real_count]):
                if crop_id not in selected_set:
                    continue
                y0, y1, x0, x1 = (int(value) for value in windows[crop_id])
                values = decoded[slot]
                score_sum[:, :, y0:y1, x0:x1] += values
                count[:, :, y0:y1, x0:x1] += 1
                candidate = np.ascontiguousarray(values.detach().cpu().numpy())
                reference = np.ascontiguousarray(reference_store[crop_id])
                difference = _difference_statistics(
                    candidate, reference, atol=atol, rtol=rtol
                )
                original_batch = _original_dense_batch(crop_id, crop_count, batch_size)
                crop_records.append(
                    {
                        "crop_id": int(crop_id),
                        "original_dense_batch_index": int(crop_id // batch_size),
                        "original_dense_batch_slot": int(crop_id % batch_size),
                        "original_dense_batch_ids": list(original_batch),
                        "live_batch_index": int(batch_index),
                        "live_batch_slot": int(slot),
                        "live_batch_ids": [int(value) for value in batch_ids],
                        "live_real_batch_ids": [
                            int(value) for value in batch_ids[:real_count]
                        ],
                        "model_batch_size": len(batch_ids),
                        "real_batch_size": int(real_count),
                        "batch_was_padded": bool(batch_was_padded),
                        "batch_membership_equal": tuple(batch_ids) == original_batch,
                        **difference,
                    }
                )
                observed.append(crop_id)
            del optical_batch, sar_batch, decoded
    if phase_inputs.is_cuda:
        torch.cuda.synchronize(phase_inputs.device)
    if tuple(observed) != selected:
        raise AssertionError("selected crop execution order differs from the plan")
    aggregate = {
        "sum_logits": np.ascontiguousarray(score_sum[0].cpu().numpy()),
        "count_mat": np.ascontiguousarray(count[0, 0].cpu().numpy()),
        "crop_ids": selected,
    }
    summary = _merge_difference_statistics(crop_records)
    worst = sorted(
        crop_records,
        key=lambda item: (
            int(item["argmax_prediction_flips"]),
            float(item["maximum_absolute_difference"]),
        ),
        reverse=True,
    )[:worst_crops]
    execution = {
        "mode": mode,
        "executed_crop_samples": sum(len(item[0]) for item in batch_plan),
        "selected_crop_samples": len(selected),
        "padding_crop_samples": sum(len(ids) - real for ids, real, _ in batch_plan),
        "batch_calls": len(batch_plan),
        "wall_seconds": float(time.perf_counter() - started),
        "selected_crop_ids": list(selected),
        "selected_crop_id_sha256": array_sha256(np.asarray(selected, dtype=np.int64)),
        "raw_crop_summary": summary,
        "batch_shape_breakdown": _batch_shape_breakdown(crop_records, batch_size),
        "worst_crops": worst,
        "all_crop_records": crop_records,
    }
    del score_sum, count
    return aggregate, execution


def _prediction_comparison(
    candidate_logits: np.ndarray,
    reference_logits: np.ndarray,
    target: np.ndarray,
    class_names: Sequence[str],
    *,
    common_bounds: tuple[int, int, int, int],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    candidate_prediction = np.ascontiguousarray(
        candidate_logits.argmax(axis=0).astype(np.int64, copy=False)
    )
    reference_prediction = np.ascontiguousarray(
        reference_logits.argmax(axis=0).astype(np.int64, copy=False)
    )
    flips = candidate_prediction != reference_prediction
    y0, y1, x0, x1 = common_bounds
    common_flips = flips[y0:y1, x0:x1]
    transition = np.bincount(
        reference_prediction[flips] * NUM_CLASSES + candidate_prediction[flips],
        minlength=NUM_CLASSES**2,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    reference_confusion = confusion_matrix(target, reference_prediction)
    candidate_confusion = confusion_matrix(target, candidate_prediction)
    reference_metrics = metric_summary(reference_confusion, class_names)
    candidate_metrics = metric_summary(candidate_confusion, class_names)
    flip_y, flip_x = np.nonzero(flips)
    if flip_y.size:
        reference_flip_logits = reference_logits[:, flip_y, flip_x]
        candidate_flip_logits = candidate_logits[:, flip_y, flip_x]
        reference_top2 = np.partition(reference_flip_logits, -2, axis=0)[-2:]
        candidate_top2 = np.partition(candidate_flip_logits, -2, axis=0)[-2:]
        reference_margin = reference_top2[-1] - reference_top2[-2]
        candidate_margin = candidate_top2[-1] - candidate_top2[-2]

        def _margin_summary(values: np.ndarray) -> dict[str, Any]:
            return {
                "count": int(values.size),
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "maximum": float(values.max()),
                "count_le_1e-6": int(np.count_nonzero(values <= 1e-6)),
                "count_le_1e-5": int(np.count_nonzero(values <= 1e-5)),
                "count_le_1e-4": int(np.count_nonzero(values <= 1e-4)),
                "count_le_1e-3": int(np.count_nonzero(values <= 1e-3)),
                "count_le_1e-2": int(np.count_nonzero(values <= 1e-2)),
            }

        sample_order = np.argsort(reference_margin, kind="stable")[:24]
        flip_examples = []
        for index in sample_order:
            y = int(flip_y[index])
            x = int(flip_x[index])
            reference_class = int(reference_prediction[y, x])
            candidate_class = int(candidate_prediction[y, x])
            flip_examples.append(
                {
                    "y": y,
                    "x": x,
                    "reference_class_index": reference_class,
                    "reference_class_name": str(class_names[reference_class]),
                    "candidate_class_index": candidate_class,
                    "candidate_class_name": str(class_names[candidate_class]),
                    "reference_top1_top2_margin": float(reference_margin[index]),
                    "candidate_top1_top2_margin": float(candidate_margin[index]),
                    "reference_winner_logit": float(
                        reference_logits[reference_class, y, x]
                    ),
                    "reference_candidate_class_logit": float(
                        reference_logits[candidate_class, y, x]
                    ),
                    "candidate_winner_logit": float(
                        candidate_logits[candidate_class, y, x]
                    ),
                    "candidate_reference_class_logit": float(
                        candidate_logits[reference_class, y, x]
                    ),
                    "maximum_class_logit_absolute_difference": float(
                        np.abs(
                            candidate_logits[:, y, x] - reference_logits[:, y, x]
                        ).max()
                    ),
                }
            )
        flip_margin = {
            "reference_top1_top2": _margin_summary(reference_margin),
            "candidate_top1_top2": _margin_summary(candidate_margin),
            "examples_sorted_by_reference_margin": flip_examples,
        }
    else:
        flip_margin = {
            "reference_top1_top2": {"count": 0},
            "candidate_top1_top2": {"count": 0},
            "examples_sorted_by_reference_margin": [],
        }
    return {
        "logits": _chunked_logit_difference(
            candidate_logits, reference_logits, atol=atol, rtol=rtol
        ),
        "prediction_equal": bool(not np.any(flips)),
        "prediction_flip_count": int(np.count_nonzero(flips)),
        "prediction_flip_count_common": int(np.count_nonzero(common_flips)),
        "prediction_pixel_count": int(flips.size),
        "reference_prediction_sha256": array_sha256(reference_prediction),
        "candidate_prediction_sha256": array_sha256(candidate_prediction),
        "reference_to_candidate_flip_matrix": transition,
        "flip_margin_diagnostic": flip_margin,
        "reference_confusion": reference_confusion,
        "candidate_confusion": candidate_confusion,
        "confusion_delta_candidate_minus_reference": (
            candidate_confusion - reference_confusion
        ),
        "confusion_equal": bool(np.array_equal(candidate_confusion, reference_confusion)),
        "reference_metrics": reference_metrics,
        "candidate_metrics": candidate_metrics,
        "miou_delta_percentage_points": float(
            candidate_metrics["miou_percent"] - reference_metrics["miou_percent"]
        ),
    }


def classify_diagnostic(
    mode_records: Mapping[str, Mapping[str, Any]], *, final_available: bool
) -> dict[str, Any]:
    """Classify the narrow batching diagnosis without making a science claim."""

    def _raw_crop_drift(record: Mapping[str, Any]) -> bool:
        phases = record["phases"]
        return any(
            int(item["execution"]["raw_crop_summary"]["array_equal_crop_count"])
            != int(item["execution"]["raw_crop_summary"]["crop_count"])
            for item in phases.values()
        )

    def _endpoint_equal(record: Mapping[str, Any]) -> bool | None:
        if not final_available:
            return None
        return bool(record["final_k4"]["prediction_equal"])

    compact = mode_records.get("compact")
    padded = mode_records.get("compact-pad-last")
    preserving = mode_records.get("dense-preserving")
    compact_raw_drift = _raw_crop_drift(compact) if compact is not None else None
    padded_raw_drift = _raw_crop_drift(padded) if padded is not None else None
    preserving_raw_drift = _raw_crop_drift(preserving) if preserving is not None else None
    compact_endpoint_equal = _endpoint_equal(compact) if compact is not None else None
    padded_endpoint_equal = _endpoint_equal(padded) if padded is not None else None
    preserving_endpoint_equal = (
        _endpoint_equal(preserving) if preserving is not None else None
    )
    compact_diverged = bool(compact_raw_drift) or compact_endpoint_equal is False
    padded_diverged = bool(padded_raw_drift) or padded_endpoint_equal is False
    preserving_diverged = (
        bool(preserving_raw_drift) or preserving_endpoint_equal is False
    )
    if compact is None or padded is None or preserving is None:
        outcome = "INCOMPLETE_MODE_COMPARISON"
        meaning = (
            "Compact, padded-compact, and dense-preserving modes are all required "
            "for attribution."
        )
    elif preserving_diverged:
        outcome = "NOT_ISOLATED_DENSE_PRESERVING_ALSO_DIFFERS"
        meaning = (
            "The original dense batch membership did not recover the cache, so batch "
            "compaction alone is not an adequate explanation."
        )
    elif compact_diverged and not padded_diverged and (
        not final_available or compact_endpoint_equal is False
    ) and (not final_available or padded_endpoint_equal is True):
        outcome = "CONFIRMED_FINAL_PARTIAL_BATCH_SHAPE_SENSITIVITY"
        meaning = (
            "Padding the final short compact batch recovers the cache, isolating the "
            "difference to model batch shape rather than selected-crop membership."
        )
    elif compact_diverged and padded_diverged and (
        not final_available or compact_endpoint_equal is False
    ) and (not final_available or preserving_endpoint_equal is True):
        outcome = "CONFIRMED_COMPACT_BATCH_MEMBERSHIP_SENSITIVITY"
        meaning = (
            "Restoring only the final batch size is insufficient, while preserving "
            "the original dense batch membership recovers the cached reference."
        )
    elif compact_diverged:
        outcome = "COMPACT_NUMERICAL_DRIFT_WITHOUT_K4_ENDPOINT_FLIP"
        meaning = (
            "Compact batching changes raw crop logits, but this run did not change the "
            "final K4 prediction."
        )
    else:
        outcome = "BATCHING_MISMATCH_NOT_REPRODUCED"
        meaning = "Neither execution mode materially differed from the cached crops."
    return {
        "outcome": outcome,
        "meaning": meaning,
        "compact_raw_crop_numerical_drift": compact_raw_drift,
        "compact_padded_raw_crop_numerical_drift": padded_raw_drift,
        "dense_preserving_raw_crop_numerical_drift": preserving_raw_drift,
        "compact_mode_diverged_including_final_prediction": compact_diverged,
        "compact_padded_mode_diverged_including_final_prediction": padded_diverged,
        "dense_preserving_mode_diverged_including_final_prediction": preserving_diverged,
        "compact_final_k4_prediction_equal": compact_endpoint_equal,
        "compact_padded_final_k4_prediction_equal": padded_endpoint_equal,
        "dense_preserving_final_k4_prediction_equal": preserving_endpoint_equal,
        "scientific_decision_evaluated": False,
        "authorizes_training": False,
        "authorizes_full_b1": False,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this live-model diagnostic")

    stage_a = load_stage_a(args.stage_a_json)
    b0 = load_b0(args.stage_b0_json, args.stage_a_json)
    geometry = build_phase_closure_geometry(stage_a["images"], stage_a["cells"])
    cache = validate_cache_manifest(
        args.cache_manifest,
        args.stage_a_json,
        args.stage_b0_json,
        stage_a,
        b0,
        geometry,
    )
    checkpoint_sha = file_sha256(args.baseline_checkpoint)
    if checkpoint_sha != cache["baseline_checkpoint"]["sha256"]:
        raise ValueError("supplied checkpoint differs from the sealed cache")
    cached_batch_size = int(cache["reproducibility"]["inference_batch_size"])
    if args.inference_batch_size != cached_batch_size:
        raise ValueError(
            "diagnostic batch size must match the cache dense schedule: "
            f"{args.inference_batch_size} != {cached_batch_size}"
        )
    if args.seed != int(cache["reproducibility"]["seed"]):
        raise ValueError("diagnostic seed must match the cache")

    image = cache["image"]
    image_index = int(image["image_index"])
    height, width = geometry["image_shapes"][image_index]
    common_bounds = tuple(int(value) for value in geometry["common_bounds"][image_index])
    label = load_array_descriptor(
        args.cache_manifest.resolve().parent,
        image["label"],
        name="label",
        expected_dtype=np.dtype("int64"),
        expected_shape=(height, width),
        expected_relative_path="labels/label.npy",
    )
    stores, store_audit = prepare_crop_stores(
        args.cache_manifest, cache, b0, geometry
    )
    windows = geometry["windows_by_image"][image_index]
    k4_levels = endpoint_levels(geometry, 4)
    selected_by_phase = expected_phase_crop_ids(
        k4_levels, geometry, image_index
    )

    reference_normal = accumulate_phase_crops(
        stores["normal"],
        tuple(stores["normal"]),
        geometry,
        image_index,
        num_classes=NUM_CLASSES,
    )
    normal_logits = _normal_mean(reference_normal)
    reference_phases: dict[str, Mapping[str, Any]] = {}
    for phase_name in PHASE_NAMES:
        reference_phases[phase_name] = accumulate_phase_crops(
            stores[phase_name],
            selected_by_phase[phase_name],
            geometry,
            image_index,
            num_classes=NUM_CLASSES,
        )
    reference_k4 = compose_policy_logits(
        normal_logits, reference_phases, k4_levels, geometry, image_index
    )
    expected_k4_sha = stage_a["images"][image_index]["prediction_sha256"]["k4"]
    if reference_k4["prediction_sha256"] != expected_k4_sha:
        raise AssertionError("cached raw crops no longer reproduce the Stage-A K4 anchor")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    model, cfg = load_model(args.baseline_checkpoint, args.seed)
    if tuple(cfg["window_size"]) != tuple(geometry["crop_size"]):
        raise ValueError("model crop size differs from the sealed geometry")
    loader, sample_names, full_test_length = build_test_loader(1, cfg["window_size"])
    if full_test_length != stage_a["full_test_length"]:
        raise ValueError("current test length differs from Stage A")
    if sample_names != [image["sample_name"]] or image_index != 0:
        raise ValueError("diagnostic currently requires the sealed first cached image")
    optical, sar, live_label = next(iter(loader))
    optical_sha = array_sha256(np.ascontiguousarray(optical.numpy(), dtype=np.float32))
    sar_sha = array_sha256(np.ascontiguousarray(sar.numpy(), dtype=np.float32))
    live_label_np = np.ascontiguousarray(live_label.numpy()[0], dtype=np.int64)
    if optical_sha != image["decoded_tensor_sha256"]["optical"]:
        raise ValueError("live optical tensor differs from the cached provenance")
    if sar_sha != image["decoded_tensor_sha256"]["sar"]:
        raise ValueError("live SAR tensor differs from the cached provenance")
    if array_sha256(live_label_np) != image["decoded_tensor_sha256"]["label_int64"]:
        raise ValueError("live label tensor differs from the cached provenance")
    if not np.array_equal(live_label_np, label):
        raise AssertionError("live and cached labels differ")

    model.to(device)
    model.eval()
    mode_records: dict[str, dict[str, Any]] = {
        mode: {"phases": {}} for mode in args.modes
    }
    candidate_phases: dict[str, dict[str, Mapping[str, Any]]] = {
        mode: {} for mode in args.modes
    }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for phase_name in args.phases:
        dy, dx = PHASE_SHIFTS[phase_name]
        phase_optical = translate_tensor(optical, dy, dx).to(device)
        phase_sar = translate_tensor(sar, dy, dx).to(device)
        for mode in args.modes:
            aggregate, execution = _execute_mode(
                phase_inputs=phase_optical,
                phase_sar=phase_sar,
                model=model,
                windows=windows,
                selected_ids=selected_by_phase[phase_name],
                reference_store=stores[phase_name],
                mode=mode,
                batch_size=args.inference_batch_size,
                atol=args.atol,
                rtol=args.rtol,
                worst_crops=args.worst_crops,
            )
            phase_difference = _phase_accumulation_difference(
                aggregate,
                reference_phases[phase_name],
                shift=PHASE_SHIFTS[phase_name],
                common_bounds=common_bounds,
                atol=args.atol,
                rtol=args.rtol,
            )
            candidate_phases[mode][phase_name] = aggregate
            mode_records[mode]["phases"][phase_name] = {
                "execution": execution,
                "phase_accumulation": phase_difference,
            }
            print(
                f"phase={phase_name} mode={mode} "
                f"crop_flips={execution['raw_crop_summary']['argmax_prediction_flips']} "
                f"crop_max_abs={execution['raw_crop_summary']['maximum_absolute_difference']:.8g} "
                f"phase_flips={phase_difference['common_phase_argmax_prediction_flips']}",
                flush=True,
            )
        del phase_optical, phase_sar
        torch.cuda.empty_cache()

    final_available = set(args.phases) == set(PHASE_NAMES)
    if final_available:
        for mode in args.modes:
            candidate_k4 = compose_policy_logits(
                normal_logits,
                candidate_phases[mode],
                k4_levels,
                geometry,
                image_index,
            )
            final = _prediction_comparison(
                candidate_k4["logits"],
                reference_k4["logits"],
                label,
                stage_a["class_names"],
                common_bounds=common_bounds,
                atol=args.atol,
                rtol=args.rtol,
            )
            mode_records[mode]["final_k4"] = final
            print(
                f"final=K4 mode={mode} flips={final['prediction_flip_count']} "
                f"miou_delta={final['miou_delta_percentage_points']:+.8f}pp "
                f"logit_max_abs={final['logits']['maximum_absolute_difference']:.8g}",
                flush=True,
            )
            del candidate_k4
    else:
        for mode in args.modes:
            mode_records[mode]["final_k4"] = {
                "status": "NOT_EVALUATED_REQUIRES_ALL_THREE_PHASES"
            }

    decision = classify_diagnostic(mode_records, final_available=final_available)
    payload = {
        "status": "PASS_DIAGNOSTIC_COMPLETED",
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "first-image-implementation-diagnostic-only",
        "source_artifacts": {
            "cache_manifest": {
                "path": str(args.cache_manifest.resolve()),
                "sha256": file_sha256(args.cache_manifest),
            },
            "stage_a": {
                "path": str(args.stage_a_json.resolve()),
                "sha256": file_sha256(args.stage_a_json),
            },
            "stage_b0": {
                "path": str(args.stage_b0_json.resolve()),
                "sha256": file_sha256(args.stage_b0_json),
            },
            "baseline_checkpoint": {
                "path": str(args.baseline_checkpoint.resolve()),
                "sha256": checkpoint_sha,
            },
        },
        "sample": {
            "image_index": image_index,
            "sample_name": image["sample_name"],
            "shape_hw": [height, width],
        },
        "protocol": {
            "phases": list(args.phases),
            "modes": list(args.modes),
            "compact_definition": "selected crop ids repacked into contiguous batches",
            "compact_pad_last_definition": (
                "compact batches with the final short batch padded by duplicate real "
                "crop ids; only the original real slots are scattered"
            ),
            "dense_preserving_definition": (
                "all original row-major dense batches executed; only selected outputs kept"
            ),
            "batch_size": args.inference_batch_size,
            "atol": args.atol,
            "rtol": args.rtol,
            "normal_logits": "sealed raw-cache K1 replay; current live K1 already passed",
            "policy": "support-pruned K4 endpoint",
        },
        "cache_store_audit": store_audit,
        "reference_k4": {
            "prediction_sha256": reference_k4["prediction_sha256"],
            "stage_a_prediction_sha256": expected_k4_sha,
            "equal": True,
            "metrics": metric_summary(
                confusion_matrix(label, reference_k4["prediction"]),
                stage_a["class_names"],
            ),
        },
        "modes": mode_records,
        "diagnostic_decision": decision,
        "explicit_non_claims": [
            "no efficacy or method-selection conclusion",
            "no latency or deployment conclusion",
            "no full-test result",
            "no authorization to train",
        ],
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "seed": args.seed,
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "runtime_seconds": float(time.perf_counter() - started),
        },
    }
    atomic_write_json(args.output_path, payload)
    print(f"diagnostic_outcome={decision['outcome']}", flush=True)
    print(f"diagnostic_result={args.output_path.resolve()}", flush=True)
    print("phase_sparse_batching_diagnostic_status=PASS", flush=True)


if __name__ == "__main__":
    main()
