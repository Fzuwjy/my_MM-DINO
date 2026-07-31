"""Collect H4-A overlap and H4-B K2-observed response statistics together.

Normal K1 is executed exactly once per image.  Its crop outputs feed the H4-A
overlap accumulator and reconstruct the K1 aggregate.  The exact uniform-K2
x8 dependency closure is then executed with the validated fixed-batch sparse
primitive.  Padding outputs count as physical model samples but are discarded.

Cache mode replays the existing first-image normal/x8 raw crops without model
forwards.  Live full-test mode is the only data-acquisition run needed for both
offline screens.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scripts.cache_whu_phase_teacher import file_sha256  # noqa: E402
from scripts.evaluate_whu_h4a_overlap_statistics import (  # noqa: E402
    NUM_CLASSES,
    SEALED_BATCH_SIZE,
    SEALED_SEED,
    _cache_inputs,
    _confusion,
    _image_result,
    _live_model_and_dataset,
    _load_live_image,
    _ownership_bounds,
    _resolve_cache_path,
    _run_live_crops,
    _stage_cells_by_image,
    _validate_stage_protocol,
    _windows,
)
from scripts.evaluate_whu_phase_closure import (  # noqa: E402
    atomic_write_json,
    load_stage_a,
)
from scripts.evaluate_whu_translation_consistency import (  # noqa: E402
    translate_tensor,
)
from scripts.phase_closure_common import (  # noqa: E402
    build_phase_closure_geometry,
)
from scripts.phase_overlap_common import (  # noqa: E402
    RESPONSE_SCORE_NAMES,
    LogitSlideAccumulator,
    OverlapDisagreementAccumulator,
    array_sha256,
    finite_or_none,
    phase_response_cell_summaries,
)
from scripts.phase_sparse_live_common import (  # noqa: E402
    forward_selected_phase_key_crops,
)
from scripts.phase_sparse_replay_common import (  # noqa: E402
    expected_phase_crop_ids,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "whu_h4ab_k1_x8_response_statistics"
X8_SHIFT = (0, 8)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect combined WHU H4-A/H4-B response statistics"
    )
    parser.add_argument("--stage-a-json", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--raw-cache-manifest", type=Path)
    mode.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--standalone-h4a-json", type=Path)
    args = parser.parse_args(argv)
    for name in ("stage_a_json",):
        if not getattr(args, name).is_file():
            parser.error(f"--{name.replace('_', '-')} does not exist")
    if args.raw_cache_manifest is not None and not args.raw_cache_manifest.is_file():
        parser.error("--raw-cache-manifest does not exist")
    if args.baseline_checkpoint is not None and not args.baseline_checkpoint.is_file():
        parser.error("--baseline-checkpoint does not exist")
    if args.standalone_h4a_json is not None and not args.standalone_h4a_json.is_file():
        parser.error("--standalone-h4a-json does not exist")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.inference_batch_size <= 0 or args.seed < 0:
        parser.error("batch size must be positive and seed non-negative")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.raw_cache_manifest is not None and args.max_images not in (None, 1):
        parser.error("raw cache mode contains exactly one image")
    if args.baseline_checkpoint is not None and args.standalone_h4a_json is not None:
        parser.error("--standalone-h4a-json is cache-smoke-only")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _common_bounds(reference: Mapping[str, Any]) -> tuple[int, int, int, int]:
    mapping = reference.get("common_bounds")
    if not isinstance(mapping, Mapping):
        raise ValueError("Stage-A image lacks common bounds")
    result = tuple(
        int(mapping[name]) for name in ("y_start", "y_stop", "x_start", "x_stop")
    )
    y0, y1, x0, x1 = result
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError("Stage-A common bounds are invalid")
    return result


def _normal_common_logits(
    accumulator: OverlapDisagreementAccumulator,
    bounds: Sequence[int],
) -> np.ndarray:
    y0, y1, x0, x1 = (int(value) for value in bounds)
    count = accumulator.crop_count[y0:y1, x0:x1]
    if np.any(count == 0):
        raise AssertionError("normal K1 common support is uncovered")
    result = np.array(
        accumulator.logit_sum[:, y0:y1, x0:x1],
        dtype=np.float32,
        copy=True,
        order="C",
    )
    result /= count[None]
    return result


def _uniform_k2_levels(
    stage_a: Mapping[str, Any], geometry: Mapping[str, Any]
) -> np.ndarray:
    levels = np.ones(len(stage_a["cells"]), dtype=np.int64)
    levels[np.asarray(geometry["eligible"], dtype=np.bool_)] = 2
    return levels


def _cache_phase_iterator(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    phase_name: str,
    expected_ids: Sequence[int],
    windows: Sequence[Sequence[int]],
) -> Iterator[tuple[np.ndarray, tuple[int, int, int, int]]]:
    image = manifest.get("image", {})
    phase = image.get("phases", {}).get(phase_name)
    if not isinstance(phase, Mapping):
        raise ValueError(f"raw cache lacks phase {phase_name}")
    records = phase.get("crops")
    selected = tuple(int(value) for value in expected_ids)
    if not isinstance(records, list) or len(records) != len(selected):
        raise ValueError(f"raw-cache {phase_name} crop list is incomplete")
    root = manifest_path.resolve().parent
    for position, (record, crop_id) in enumerate(zip(records, selected, strict=True)):
        if int(record.get("local_crop_id", -1)) != crop_id:
            raise ValueError(f"raw-cache {phase_name} crop id differs")
        window = tuple(int(value) for value in windows[crop_id])
        if tuple(int(value) for value in record.get("window_yxyx", [])) != window:
            raise ValueError(f"raw-cache {phase_name} crop window differs")
        path = _resolve_cache_path(
            root, record.get("path"), field=f"{phase_name} crop {crop_id}"
        )
        crop = np.load(path, allow_pickle=False)
        if crop.dtype != np.float32 or crop.shape != (NUM_CLASSES, 512, 512):
            raise ValueError(f"raw-cache {phase_name} crop shape/dtype differs")
        if array_sha256(crop) != record.get("array_sha256"):
            raise ValueError(f"raw-cache {phase_name} crop hash differs")
        yield crop, window
        if (position + 1) % SEALED_BATCH_SIZE == 0 or position + 1 == len(records):
            print(
                f"cache_{phase_name}_crops={position + 1}/{len(records)}",
                flush=True,
            )


def _aligned_x8_from_cache(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    expected_ids: Sequence[int],
    windows: Sequence[Sequence[int]],
    image_shape: Sequence[int],
    common_bounds: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    accumulator = LogitSlideAccumulator(image_shape, NUM_CLASSES)
    started = time.perf_counter()
    for crop, window in _cache_phase_iterator(
        manifest_path=manifest_path,
        manifest=manifest,
        phase_name="x8",
        expected_ids=expected_ids,
        windows=windows,
    ):
        accumulator.add_crop(crop, window)
    y0, y1, x0, x1 = (int(value) for value in common_bounds)
    aligned = accumulator.normalized_logits((y0, y1, x0 + 8, x1 + 8))
    record = {
        "selected_real_x8_crops": len(expected_ids),
        "processed_x8_model_samples_including_padding": None,
        "padding_crop_samples": None,
        "model_forward_executed": False,
        "cache_replay_seconds": float(time.perf_counter() - started),
        "host_accumulator_storage_nbytes": accumulator.storage_nbytes(),
    }
    del accumulator
    return aligned, record


def _aligned_x8_live(
    *,
    model: torch.nn.Module,
    optical: torch.Tensor,
    sar: torch.Tensor,
    windows: Sequence[Sequence[int]],
    selected_ids: Sequence[int],
    common_bounds: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    shifted_optical = translate_tensor(optical, *X8_SHIFT).to(device)
    shifted_sar = translate_tensor(sar, *X8_SHIFT).to(device)
    execution = forward_selected_phase_key_crops(
        {"x8": (shifted_optical, shifted_sar)},
        model,
        windows,
        {"x8": selected_ids},
        phase_order=("x8",),
        n_output_channels=NUM_CLASSES,
        batch_size=batch_size,
        decoder_head_type="linear",
    )
    phase = execution["phases"]["x8"]
    y0, y1, x0, x1 = (int(value) for value in common_bounds)
    shifted_bounds = (y0, y1, x0 + 8, x1 + 8)
    sy0, sy1, sx0, sx1 = shifted_bounds
    count = np.asarray(phase["count_mat"])[sy0:sy1, sx0:sx1]
    if np.any(count == 0):
        raise AssertionError("live x8 closure does not cover aligned common support")
    aligned = np.array(
        np.asarray(phase["sum_logits"])[:, sy0:sy1, sx0:sx1],
        dtype=np.float32,
        copy=True,
        order="C",
    )
    aligned /= count[None]
    expected_processed = math.ceil(len(selected_ids) / batch_size) * batch_size
    if execution["selected_crop_samples"] != len(selected_ids):
        raise AssertionError("live x8 selected-crop count differs")
    if execution["model_forward_crop_samples"] != expected_processed:
        raise AssertionError("live x8 physical model-sample count differs")
    if execution["padding_crop_samples"] != expected_processed - len(selected_ids):
        raise AssertionError("live x8 padding count differs")
    if any(size != batch_size for size in execution["model_batch_sizes"]):
        raise AssertionError("live x8 model batch shape is not frozen")
    record = {
        "selected_real_x8_crops": int(execution["selected_crop_samples"]),
        "processed_x8_model_samples_including_padding": int(
            execution["model_forward_crop_samples"]
        ),
        "padding_crop_samples": int(execution["padding_crop_samples"]),
        "batch_calls": int(execution["batch_calls"]),
        "model_batch_sizes": list(execution["model_batch_sizes"]),
        "padding_policy": execution["padding_policy"],
        "model_forward_executed": True,
        "wall_seconds": float(execution["wall_seconds"]),
    }
    del shifted_optical, shifted_sar, execution, phase
    return aligned, record


def _validate_k2_endpoint(
    *,
    k1_prediction: np.ndarray,
    k1_logits: np.ndarray,
    kx_logits: np.ndarray,
    common_bounds: Sequence[int],
    reference: Mapping[str, Any],
    label: np.ndarray,
) -> dict[str, Any]:
    y0, y1, x0, x1 = (int(value) for value in common_bounds)
    prediction = np.ascontiguousarray(k1_prediction.copy())
    prediction[y0:y1, x0:x1] = (k1_logits + kx_logits).argmax(axis=0)
    computed_sha = array_sha256(prediction)
    expected_sha = reference.get("prediction_sha256", {}).get("matched_k2")
    if computed_sha != expected_sha:
        raise AssertionError(
            f"combined matched K2 SHA differs: {computed_sha} != {expected_sha}"
        )
    confusion = _confusion(prediction, label)
    expected_confusion = np.asarray(reference.get("confusion", {}).get("matched_k2"))
    if not np.array_equal(confusion, expected_confusion):
        raise AssertionError("combined matched K2 confusion differs from Stage A")
    return {
        "computed_prediction_sha256": computed_sha,
        "stage_a_prediction_sha256": expected_sha,
        "prediction_equal": True,
        "computed_confusion": confusion.tolist(),
        "stage_a_confusion_equal": True,
    }


def _attach_response_scores(
    *,
    cell_records: list[dict[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    k1_logits: np.ndarray,
    kx_logits: np.ndarray,
    common_bounds: Sequence[int],
) -> dict[str, Any]:
    response = phase_response_cell_summaries(
        k1_logits,
        kx_logits,
        _ownership_bounds(cells),
        common_bounds,
    )
    if len(cell_records) != len(cells):
        raise AssertionError("H4-A and H4-B cell counts differ")
    for local_index, record in enumerate(cell_records):
        response_scores = {
            name: finite_or_none(response["score_means"][name][local_index])
            for name in RESPONSE_SCORE_NAMES
        }
        valid = int(response["valid_pixels"][local_index])
        area = int(response["common_intersection_pixels"][local_index])
        record["k2_response_scores"] = response_scores
        record["k2_response_support"] = {
            "valid_pixels": valid,
            "common_intersection_pixels": area,
            "valid_fraction": float(valid / area) if area else None,
        }
    return response["map_diagnostics"]


def _attach_paired_evaluation_outcomes(
    *,
    cell_records: list[dict[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    k1_prediction: np.ndarray,
    k1_logits: np.ndarray,
    kx_logits: np.ndarray,
    common_bounds: Sequence[int],
    label: np.ndarray,
) -> None:
    """Attach GT-backed report-only fix/break counts, never usable as features."""

    y0, y1, x0, x1 = (int(value) for value in common_bounds)
    k2_common = (k1_logits + kx_logits).argmax(axis=0)
    if k2_common.shape != (y1 - y0, x1 - x0):
        raise ValueError("paired K2 common prediction shape differs")
    for local_index, (record, cell) in enumerate(
        zip(cell_records, cells, strict=True)
    ):
        cy0, cy1, cx0, cx1 = (int(value) for value in cell["ownership_yxyx"])
        iy0, iy1 = max(cy0, y0), min(cy1, y1)
        ix0, ix1 = max(cx0, x0), min(cx1, x1)
        if iy0 >= iy1 or ix0 >= ix1:
            outcome = {
                "valid_pixels": 0,
                "fixed_k1_errors": 0,
                "broken_k1_correct": 0,
                "net_correct": 0,
                "prediction_changed": 0,
            }
        else:
            target = label[iy0:iy1, ix0:ix1]
            first = k1_prediction[iy0:iy1, ix0:ix1]
            second = k2_common[
                iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0
            ]
            valid = (target >= 0) & (target < NUM_CLASSES)
            first_correct = valid & (first == target)
            second_correct = valid & (second == target)
            fixed = (~first_correct) & second_correct & valid
            broken = first_correct & (~second_correct) & valid
            outcome = {
                "valid_pixels": int(np.count_nonzero(valid)),
                "fixed_k1_errors": int(np.count_nonzero(fixed)),
                "broken_k1_correct": int(np.count_nonzero(broken)),
                "net_correct": int(np.count_nonzero(fixed) - np.count_nonzero(broken)),
                "prediction_changed": int(
                    np.count_nonzero(valid & (first != second))
                ),
            }
        record["evaluation_only_k1_to_k2"] = {
            **outcome,
            "role": (
                "GT-backed report-only mechanism audit; forbidden as a feature, "
                "threshold, q selector, or held-out action input"
            ),
            "local_index": local_index,
        }


def _compare_standalone_h4a(
    standalone_path: Path | None,
    combined_cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if standalone_path is None:
        return None
    standalone = _read_json(standalone_path)
    source_cells = standalone.get("cells")
    if not isinstance(source_cells, list) or len(source_cells) != len(combined_cells):
        raise ValueError("standalone H4-A cell count differs from combined smoke")
    compared = 0
    maximum_difference = 0.0
    for old, new in zip(source_cells, combined_cells, strict=True):
        if old.get("cell_index") != new.get("cell_index"):
            raise ValueError("standalone H4-A cell identity differs")
        for name, old_value in old.get("scores", {}).items():
            new_value = new.get("scores", {}).get(name)
            if old_value is None or new_value is None:
                if old_value != new_value:
                    raise AssertionError("standalone H4-A null score differs")
                continue
            difference = abs(float(old_value) - float(new_value))
            maximum_difference = max(maximum_difference, difference)
            compared += 1
            if difference != 0.0:
                raise AssertionError("combined H4-A score differs from standalone smoke")
    return {
        "path": str(standalone_path.resolve()),
        "sha256": file_sha256(standalone_path),
        "compared_numeric_scores": compared,
        "maximum_absolute_difference": maximum_difference,
        "bitwise_numeric_equality": True,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    stage_a_sha = file_sha256(args.stage_a_json)
    stage_a = load_stage_a(args.stage_a_json)
    _validate_stage_protocol(stage_a, args)
    cells_by_image = _stage_cells_by_image(stage_a)
    geometry = build_phase_closure_geometry(stage_a["images"], stage_a["cells"])
    uniform_levels = _uniform_k2_levels(stage_a, geometry)
    full_count = len(stage_a["images"])
    mode = "cache-replay" if args.raw_cache_manifest is not None else "live-k1-x8"
    image_records: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    standalone_validation = None
    peak_cuda_memory = None

    if args.raw_cache_manifest is not None:
        manifest, image_index, label, normal_iterator, provenance = _cache_inputs(
            args.raw_cache_manifest,
            stage_a=stage_a,
            stage_a_sha=stage_a_sha,
            cells_by_image=cells_by_image,
        )
        expected_checkpoint_sha = stage_a.get("baseline_checkpoint_sha256")
        cached_checkpoint_sha = provenance.get("baseline_checkpoint", {}).get(
            "sha256"
        )
        if cached_checkpoint_sha != expected_checkpoint_sha:
            raise ValueError(
                "raw-cache checkpoint SHA differs from the Stage-A baseline"
            )
        reference = stage_a["images"][image_index]
        cells = cells_by_image[image_index]
        windows = _windows(cells)
        selected_ids = expected_phase_crop_ids(
            uniform_levels, geometry, image_index
        )["x8"]
        normal = OverlapDisagreementAccumulator(reference["full_shape_hw"], NUM_CLASSES)
        normal_started = time.perf_counter()
        for crop, window in normal_iterator:
            normal.add_crop(crop, window)
        normal_seconds = time.perf_counter() - normal_started
        normal_prediction = normal.endpoint_prediction()
        image_record, cell_records = _image_result(
            accumulator=normal,
            image_index=image_index,
            reference=reference,
            cells=cells,
            label=label,
            crop_source_runtime_seconds=normal_seconds,
        )
        # Finalize the larger H4-A temporaries before retaining the common K1
        # logit copy needed by H4-B.  This keeps the combined host-memory peak
        # below the otherwise equivalent reverse ordering.
        k1_logits = _normal_common_logits(normal, _common_bounds(reference))
        del normal
        kx_logits, x8_record = _aligned_x8_from_cache(
            manifest_path=args.raw_cache_manifest,
            manifest=manifest,
            expected_ids=selected_ids,
            windows=windows,
            image_shape=reference["full_shape_hw"],
            common_bounds=_common_bounds(reference),
        )
        image_record["matched_k2_endpoint_validation"] = _validate_k2_endpoint(
            k1_prediction=normal_prediction,
            k1_logits=k1_logits,
            kx_logits=kx_logits,
            common_bounds=_common_bounds(reference),
            reference=reference,
            label=label,
        )
        image_record["x8_execution"] = x8_record
        image_record["k2_response_map_diagnostics"] = _attach_response_scores(
            cell_records=cell_records,
            cells=cells,
            k1_logits=k1_logits,
            kx_logits=kx_logits,
            common_bounds=_common_bounds(reference),
        )
        _attach_paired_evaluation_outcomes(
            cell_records=cell_records,
            cells=cells,
            k1_prediction=normal_prediction,
            k1_logits=k1_logits,
            kx_logits=kx_logits,
            common_bounds=_common_bounds(reference),
            label=label,
        )
        standalone_validation = _compare_standalone_h4a(
            args.standalone_h4a_json, cell_records
        )
        image_records.append(image_record)
        all_cells.extend(cell_records)
        del k1_logits, kx_logits, normal_prediction, label
    else:
        device = torch.device(args.device)
        model, _, dataset, checkpoint_sha = _live_model_and_dataset(
            args.baseline_checkpoint, args.seed, device
        )
        if checkpoint_sha != stage_a.get("baseline_checkpoint_sha256"):
            raise ValueError(
                "baseline checkpoint SHA differs from Stage A; refusing to run images"
            )
        if len(dataset) != full_count:
            raise ValueError("WHU test length differs from Stage A")
        limit = full_count if args.max_images is None else min(args.max_images, full_count)
        provenance = {
            "mode": "single-normal-plus-sparse-x8-combined-live-collection",
            "baseline_checkpoint": {
                "path": str(args.baseline_checkpoint.resolve()),
                "sha256": checkpoint_sha,
            },
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        for image_index in range(limit):
            reference = stage_a["images"][image_index]
            cells = cells_by_image[image_index]
            windows = _windows(cells)
            selected_ids = expected_phase_crop_ids(
                uniform_levels, geometry, image_index
            )["x8"]
            optical, sar, label = _load_live_image(dataset, image_index, reference)
            normal = OverlapDisagreementAccumulator(reference["full_shape_hw"], NUM_CLASSES)
            print(
                f"image={image_index + 1}/{limit} sample={reference['sample_name']} "
                f"normal_crops={len(windows)} x8_real_crops={len(selected_ids)}",
                flush=True,
            )
            normal_seconds = _run_live_crops(
                model=model,
                optical=optical,
                sar=sar,
                windows=windows,
                batch_size=args.inference_batch_size,
                device=device,
                accumulator=normal,
            )
            normal_prediction = normal.endpoint_prediction()
            image_record, cell_records = _image_result(
                accumulator=normal,
                image_index=image_index,
                reference=reference,
                cells=cells,
                label=label,
                crop_source_runtime_seconds=normal_seconds,
            )
            k1_logits = _normal_common_logits(normal, _common_bounds(reference))
            del normal
            gc.collect()
            kx_logits, x8_record = _aligned_x8_live(
                model=model,
                optical=optical,
                sar=sar,
                windows=windows,
                selected_ids=selected_ids,
                common_bounds=_common_bounds(reference),
                batch_size=args.inference_batch_size,
                device=device,
            )
            image_record["matched_k2_endpoint_validation"] = _validate_k2_endpoint(
                k1_prediction=normal_prediction,
                k1_logits=k1_logits,
                kx_logits=kx_logits,
                common_bounds=_common_bounds(reference),
                reference=reference,
                label=label,
            )
            image_record["x8_execution"] = x8_record
            image_record["k2_response_map_diagnostics"] = _attach_response_scores(
                cell_records=cell_records,
                cells=cells,
                k1_logits=k1_logits,
                kx_logits=kx_logits,
                common_bounds=_common_bounds(reference),
            )
            _attach_paired_evaluation_outcomes(
                cell_records=cell_records,
                cells=cells,
                k1_prediction=normal_prediction,
                k1_logits=k1_logits,
                kx_logits=kx_logits,
                common_bounds=_common_bounds(reference),
                label=label,
            )
            image_records.append(image_record)
            all_cells.extend(cell_records)
            print(
                f"image={image_index + 1}/{limit} K1=PASS K2=PASS "
                f"normal_seconds={normal_seconds:.3f} "
                f"x8_seconds={x8_record['wall_seconds']:.3f}",
                flush=True,
            )
            del optical, sar, label, k1_logits, kx_logits, normal_prediction
            gc.collect()
        peak_cuda_memory = int(torch.cuda.max_memory_allocated(device))

    evaluated = len(image_records)
    full_scope = mode == "live-k1-x8" and evaluated == full_count
    x8_real = sum(int(record["x8_execution"]["selected_real_x8_crops"]) for record in image_records)
    x8_processed_values = [
        record["x8_execution"]["processed_x8_model_samples_including_padding"]
        for record in image_records
    ]
    x8_processed = (
        sum(int(value) for value in x8_processed_values)
        if all(value is not None for value in x8_processed_values)
        else None
    )
    baseline_samples = sum(int(record["crop_grid"]["crop_count"]) for record in image_records)
    physical_cost = (
        float((baseline_samples + x8_processed) / baseline_samples)
        if x8_processed is not None
        else None
    )
    if full_scope:
        if baseline_samples != 3520 or x8_real != 2520 or x8_processed != 2560:
            raise AssertionError(
                "full WHU K1/x8 sample accounting differs from the frozen geometry"
            )
        if abs(float(physical_cost) - (6080.0 / 3520.0)) > 1e-12:
            raise AssertionError("full WHU physical K2 cost differs from 1.727273x")
    output = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "PASS",
        "status_meaning": (
            "Every processed image exactly reproduced Stage-A K1 and matched-K2 "
            "prediction SHA/confusion before cell response statistics were serialized."
        ),
        "scope": (
            "full-test"
            if full_scope
            else (
                "single-image-cache-correctness"
                if mode == "cache-replay"
                else "live-prefix-smoke"
            )
        ),
        "scientific_scope": (
            "combined H4-A/H4-B score extraction for separate frozen offline screens"
            if full_scope
            else "implementation correctness only; no scientific Go/No-Go"
        ),
        "execution_mode": mode,
        "full_test_length": full_count,
        "evaluated_images": evaluated,
        "source_stage_a": {
            "path": str(args.stage_a_json.resolve()),
            "sha256": stage_a_sha,
            "artifact_type": stage_a["artifact_type"],
            "schema_version": stage_a["schema_version"],
        },
        "source_execution": provenance,
        "protocol": {
            "normal_execution_count": "exactly once per image",
            "normal_role": "shared by H4-A overlap scores and H4-B z1",
            "x8_shift_dy_dx": list(X8_SHIFT),
            "x8_route": "uniform-K2 exact common-support dependency closure",
            "x8_padding": (
                "fixed batch=8 cyclic duplicate padding; padding outputs discarded "
                "from sum/count but counted as physical model samples"
            ),
            "k2_endpoint": "argmax(z1+zx), equivalent to argmax((z1+zx)/2)",
            "k2_probability": (
                "softmax((z1+zx)/2); never softmax(z1+zx), preventing a false "
                "temperature-doubling confidence gain"
            ),
            "h4b_primary_score": RESPONSE_SCORE_NAMES[0],
            "h4b_auxiliary_score": RESPONSE_SCORE_NAMES[1],
            "h4b_report_only_scores": list(RESPONSE_SCORE_NAMES[2:]),
            "gt_use": (
                "labels verify frozen endpoints and attach isolated report-only paired "
                "fix/break outcomes; no response feature or action input uses GT"
            ),
        },
        "collection_cost": {
            "baseline_normal_real_and_model_samples": baseline_samples,
            "x8_selected_real_crops": x8_real,
            "x8_processed_model_samples_including_padding": x8_processed,
            "physical_model_sample_cost_ratio": physical_cost,
            "cache_mode_physical_cost_is_none": mode == "cache-replay",
        },
        "images": image_records,
        "cells": all_cells,
        "standalone_h4a_equality": standalone_validation,
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "h4a_runner_sha256": file_sha256(
                REPO_ROOT / "scripts" / "evaluate_whu_h4a_overlap_statistics.py"
            ),
            "common_module_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_overlap_common.py"
            ),
            "sparse_live_common_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_sparse_live_common.py"
            ),
            "seed": args.seed,
            "inference_batch_size": args.inference_batch_size,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device if mode == "live-k1-x8" else None,
            "peak_cuda_memory_bytes": peak_cuda_memory,
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "created_at_utc": utc_now(),
        "next_step": (
            "Run the separate frozen H4-A and H4-B offline screens."
            if full_scope
            else "After independent review, run one foreground full-test combined collection."
        ),
        "explicit_non_claims": [
            "Extraction alone establishes H4-A or H4-B predictability.",
            "H4-B depends on H4-A passing.",
            "Response diagnostics are pure causal phase measures.",
            "A utility head or K2->K4 router is implemented or authorized.",
        ],
    }
    atomic_write_json(args.output_path, output)
    print(f"h4ab_response_result={args.output_path.resolve()}", flush=True)
    print("h4ab_response_status=PASS", flush=True)


if __name__ == "__main__":
    main()
