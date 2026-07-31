"""Extract H4-A normal-slide response disagreement for WHU ownership cells.

Two execution modes share the exact same host-side accumulator:

* ``--raw-cache-manifest`` replays the existing single-image Stage-B1a raw
  normal-crop cache without a model forward.  This is correctness smoke only.
* ``--baseline-checkpoint`` executes the sealed normal K1 crop schedule and
  streams crop logits to the accumulator.  A complete 20-image run produces
  the score artifact consumed by the separate offline H4-A screen.

Neither mode changes the released ``slide_inference`` function or model.
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

from scripts.cache_whu_phase_teacher import (  # noqa: E402
    build_full_image_dataset,
    file_sha256,
)
from scripts.evaluate_whu_phase_closure import (  # noqa: E402
    atomic_write_json,
    load_stage_a,
)
from scripts.phase_overlap_common import (  # noqa: E402
    SCORE_NAMES,
    OverlapDisagreementAccumulator,
    array_sha256,
    finite_or_none,
)
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "whu_h4a_k1_overlap_disagreement"
EXPECTED_CACHE_TYPE = "whu_phase_crop_logits_correctness_cache"
NUM_CLASSES = 7
SEALED_BATCH_SIZE = 8
SEALED_SEED = 42


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
        description="Extract WHU H4-A intra-slide response disagreement"
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
    args = parser.parse_args(argv)
    if not args.stage_a_json.is_file():
        parser.error(f"--stage-a-json does not exist: {args.stage_a_json}")
    if (
        args.raw_cache_manifest is not None
        and not args.raw_cache_manifest.is_file()
    ):
        parser.error(
            f"--raw-cache-manifest does not exist: {args.raw_cache_manifest}"
        )
    if (
        args.baseline_checkpoint is not None
        and not args.baseline_checkpoint.is_file()
    ):
        parser.error(
            f"--baseline-checkpoint does not exist: {args.baseline_checkpoint}"
        )
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.raw_cache_manifest is not None and args.max_images not in (None, 1):
        parser.error("raw-cache correctness mode contains exactly one image")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _stage_cells_by_image(stage_a: Mapping[str, Any]) -> tuple[tuple[dict[str, Any], ...], ...]:
    image_count = len(stage_a["images"])
    grouped: list[list[dict[str, Any]]] = [[] for _ in range(image_count)]
    for global_index, raw in enumerate(stage_a["cells"]):
        if not isinstance(raw, dict):
            raise TypeError("Stage-A cells must be objects")
        image_index = int(raw.get("image_index", -1))
        if image_index < 0 or image_index >= image_count:
            raise ValueError("Stage-A cell image index is invalid")
        if int(raw.get("cell_index", -1)) != global_index:
            raise ValueError("Stage-A global cell indices are not contiguous")
        grouped[image_index].append(raw)
    result = []
    for image_index, cells in enumerate(grouped):
        expected = int(stage_a["images"][image_index]["ownership_cell_count"])
        if len(cells) != expected:
            raise ValueError("Stage-A ownership-cell count differs from image metadata")
        for local_index, cell in enumerate(cells):
            if int(cell.get("local_crop_id", -1)) != local_index:
                raise ValueError("Stage-A local crop ids are not canonical")
        result.append(tuple(cells))
    return tuple(result)


def _windows(cells: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, int, int, int], ...]:
    result = []
    for cell in cells:
        raw = cell.get("window_yxyx")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
            raise ValueError("Stage-A cell lacks a crop window")
        result.append(tuple(int(value) for value in raw))
    return tuple(result)


def _ownership_bounds(cells: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = np.asarray([cell.get("ownership_yxyx") for cell in cells])
    if values.shape != (len(cells), 4) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Stage-A ownership bounds are invalid")
    return np.ascontiguousarray(values, dtype=np.int64)


def _confusion(prediction: np.ndarray, label: np.ndarray) -> np.ndarray:
    pred = np.asarray(prediction)
    target = np.asarray(label)
    if pred.shape != target.shape:
        raise ValueError("prediction and label shapes differ")
    valid = (target >= 0) & (target < NUM_CLASSES)
    if np.any((pred[valid] < 0) | (pred[valid] >= NUM_CLASSES)):
        raise ValueError("prediction contains an invalid class")
    encoded = NUM_CLASSES * target[valid].astype(np.int64) + pred[valid]
    return np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(
        NUM_CLASSES, NUM_CLASSES
    )


def _resolve_cache_path(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must remain relative to the cache root")
    root_resolved = root.resolve()
    path = (root_resolved / relative).resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{field} escapes the cache root") from error
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _validate_stage_protocol(stage_a: Mapping[str, Any], args: argparse.Namespace) -> None:
    if len(stage_a.get("class_names", [])) != NUM_CLASSES:
        raise ValueError("H4-A requires the sealed seven-class WHU artifact")
    reproducibility = stage_a.get("reproducibility", {})
    if args.inference_batch_size != reproducibility.get("inference_batch_size"):
        raise ValueError("batch size must exactly match Stage A")
    if args.seed != reproducibility.get("seed"):
        raise ValueError("seed must exactly match Stage A")
    if args.inference_batch_size != SEALED_BATCH_SIZE or args.seed != SEALED_SEED:
        raise ValueError("H4-A protocol is frozen to batch=8 and seed=42")


def _cache_inputs(
    path: Path,
    *,
    stage_a: Mapping[str, Any],
    stage_a_sha: str,
    cells_by_image: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[
    dict[str, Any],
    int,
    np.ndarray,
    Iterator[tuple[np.ndarray, tuple[int, int, int, int]]],
    dict[str, Any],
]:
    manifest = _read_json(path)
    checks = {
        "artifact_type": manifest.get("artifact_type") == EXPECTED_CACHE_TYPE,
        "schema_version": manifest.get("schema_version") == 1,
        "status": manifest.get("status") == "PASS",
        "scope": manifest.get("scope") == "single-image-correctness-only",
        "source_stage_a": manifest.get("source_stage_a", {}).get("sha256")
        == stage_a_sha,
        "raw_float32": manifest.get("protocol", {}).get("raw_crop_storage_dtype")
        == "float32",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"raw-cache binding failed: {failed}")
    image = manifest.get("image")
    if not isinstance(image, Mapping):
        raise ValueError("raw cache lacks its image record")
    image_index = int(image.get("image_index", -1))
    if image_index < 0 or image_index >= len(stage_a["images"]):
        raise ValueError("raw-cache image index is invalid")
    reference = stage_a["images"][image_index]
    if image.get("sample_name") != reference.get("sample_name"):
        raise ValueError("raw-cache sample differs from Stage A")
    if image.get("full_shape_hw") != reference.get("full_shape_hw"):
        raise ValueError("raw-cache shape differs from Stage A")

    cache_root = path.resolve().parent
    label_record = image.get("label")
    if not isinstance(label_record, Mapping):
        raise ValueError("raw cache lacks its decoded label")
    label_path = _resolve_cache_path(cache_root, label_record.get("path"), field="label")
    label = np.load(label_path, allow_pickle=False)
    if label.dtype != np.int64 or list(label.shape) != image.get("full_shape_hw"):
        raise ValueError("raw-cache label metadata is invalid")
    if array_sha256(label) != label_record.get("array_sha256"):
        raise ValueError("raw-cache label array hash differs")

    normal = image.get("phases", {}).get("normal")
    if not isinstance(normal, Mapping):
        raise ValueError("raw cache lacks its normal phase")
    records = normal.get("crops")
    expected_windows = _windows(cells_by_image[image_index])
    if not isinstance(records, list) or len(records) != len(expected_windows):
        raise ValueError("raw-cache normal crop list is incomplete")

    def iterator() -> Iterator[tuple[np.ndarray, tuple[int, int, int, int]]]:
        for crop_id, (record, expected_window) in enumerate(
            zip(records, expected_windows, strict=True)
        ):
            if int(record.get("local_crop_id", -1)) != crop_id:
                raise ValueError("raw-cache crop ids are not canonical")
            if tuple(int(value) for value in record.get("window_yxyx", [])) != expected_window:
                raise ValueError("raw-cache crop window differs from Stage A")
            crop_path = _resolve_cache_path(
                cache_root, record.get("path"), field=f"normal crop {crop_id}"
            )
            crop = np.load(crop_path, allow_pickle=False)
            if crop.dtype != np.float32 or crop.shape != (NUM_CLASSES, 512, 512):
                raise ValueError("raw-cache crop dtype/shape is invalid")
            if array_sha256(crop) != record.get("array_sha256"):
                raise ValueError(f"raw-cache crop {crop_id} array hash differs")
            yield crop, expected_window
            if (crop_id + 1) % SEALED_BATCH_SIZE == 0 or crop_id + 1 == len(records):
                print(
                    f"cache_replay_crops={crop_id + 1}/{len(records)}",
                    flush=True,
                )

    provenance = {
        "mode": "existing-single-image-raw-cache-replay",
        "raw_cache_manifest": {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "artifact_type": manifest["artifact_type"],
            "schema_version": manifest["schema_version"],
        },
        "baseline_checkpoint": manifest.get("baseline_checkpoint"),
    }
    return manifest, image_index, np.ascontiguousarray(label), iterator(), provenance


def _live_model_and_dataset(
    checkpoint: Path, seed: int, device: torch.device
) -> tuple[torch.nn.Module, Any, Any, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for live H4-A extraction")
    if device.type != "cuda":
        raise ValueError("live H4-A extraction requires a CUDA device")
    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    dataset = build_full_image_dataset("test")
    from scripts.diagnose_whu_spatial_errors import load_model

    model, cfg = load_model(checkpoint, seed)
    if tuple(cfg["window_size"]) != (512, 512):
        raise ValueError("model crop size differs from sealed 512x512")
    if len(cfg["labels"]) != NUM_CLASSES:
        raise ValueError("model class count differs from sealed WHU")
    model.to(device)
    model.eval()
    return model, cfg, dataset, file_sha256(checkpoint)


def _load_live_image(dataset: Any, image_index: int, reference: Mapping[str, Any]):
    if int(reference.get("dataset_index", -1)) != image_index:
        raise ValueError("formal Stage-A dataset index differs from loader position")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [image_index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    optical, sar, label_tensor = next(iter(loader))
    shape = tuple(int(value) for value in reference["full_shape_hw"])
    if tuple(optical.shape[-2:]) != shape or tuple(sar.shape[-2:]) != shape:
        raise ValueError("decoded RGB/SAR shape differs from Stage A")
    label = np.ascontiguousarray(label_tensor.numpy()[0], dtype=np.int64)
    if label.shape != shape:
        raise ValueError("decoded label shape differs from Stage A")
    sample_name = str(reference["sample_name"])
    if Path(dataset.rgb_files[image_index]).stem != sample_name:
        raise ValueError("decoded dataset sample differs from Stage A")
    return optical, sar, label


def _run_live_crops(
    *,
    model: torch.nn.Module,
    optical: torch.Tensor,
    sar: torch.Tensor,
    windows: Sequence[Sequence[int]],
    batch_size: int,
    device: torch.device,
    accumulator: OverlapDisagreementAccumulator,
) -> float:
    optical_gpu = optical.to(device)
    sar_gpu = sar.to(device)
    started = time.perf_counter()
    batch_count = math.ceil(len(windows) / batch_size)
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(windows), batch_size)):
            batch_windows = windows[start : start + batch_size]
            optical_crops = []
            sar_crops = []
            for y0, y1, x0, x1 in batch_windows:
                optical_crops.append(optical_gpu[:, :, y0:y1, x0:x1])
                sar_crops.append(sar_gpu[:, :, y0:y1, x0:x1])
            prediction = model(
                torch.cat(optical_crops, dim=0),
                torch.cat(sar_crops, dim=0),
            )
            expected = (len(batch_windows), NUM_CLASSES, 512, 512)
            if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != expected:
                raise ValueError(
                    f"model crop output {getattr(prediction, 'shape', None)} differs from {expected}"
                )
            raw = np.ascontiguousarray(
                prediction.detach().to(dtype=torch.float32).cpu().numpy(),
                dtype=np.float32,
            )
            accumulator.add_batch(raw, batch_windows)
            print(
                f"live_batch={batch_index + 1}/{batch_count} "
                f"crops={min(start + len(batch_windows), len(windows))}/{len(windows)}",
                flush=True,
            )
            del prediction, raw, optical_crops, sar_crops
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    del optical_gpu, sar_gpu
    return float(elapsed)


def _image_result(
    *,
    accumulator: OverlapDisagreementAccumulator,
    image_index: int,
    reference: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    label: np.ndarray,
    crop_source_runtime_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction = accumulator.endpoint_prediction()
    computed_sha = array_sha256(prediction)
    expected_sha = reference.get("prediction_sha256", {}).get("k1")
    if computed_sha != expected_sha:
        raise AssertionError(
            f"H4-A K1 prediction differs from Stage A: {computed_sha} != {expected_sha}"
        )
    confusion = _confusion(prediction, label)
    expected_confusion = np.asarray(reference.get("confusion", {}).get("k1"))
    if not np.array_equal(confusion, expected_confusion):
        raise AssertionError("H4-A K1 confusion differs from Stage A")

    common_mapping = reference.get("common_bounds")
    common_bounds = tuple(
        int(common_mapping[name])
        for name in ("y_start", "y_stop", "x_start", "x_stop")
    )
    summaries = accumulator.cell_summaries(
        _ownership_bounds(cells), common_bounds
    )
    cell_records = []
    for local_index, cell in enumerate(cells):
        scores = {
            name: finite_or_none(summaries["score_means"][name][local_index])
            for name in SCORE_NAMES
        }
        valid_pixels = {
            name: int(summaries["score_valid_pixels"][name][local_index])
            for name in SCORE_NAMES
        }
        intersection_pixels = {
            name: int(
                summaries["score_common_intersection_pixels"][name][local_index]
            )
            for name in SCORE_NAMES
        }
        report_only_top10 = finite_or_none(
            summaries["report_only"][
                "overlap_jsd_normalized_top10pct_mean"
            ][local_index]
        )
        cell_records.append(
            {
                "cell_index": int(cell["cell_index"]),
                "image_index": image_index,
                "local_crop_id": local_index,
                "geometry_eligible": bool(cell["geometry_eligible"]),
                "scores": scores,
                "valid_pixels": valid_pixels,
                "common_intersection_pixels": intersection_pixels,
                "valid_fractions": {
                    name: (
                        float(valid_pixels[name] / intersection_pixels[name])
                        if intersection_pixels[name]
                        else None
                    )
                    for name in SCORE_NAMES
                },
                "report_only": {
                    "overlap_jsd_normalized_top10pct_mean": report_only_top10,
                    "role": summaries["report_only"]["role"],
                },
            }
        )
    image_record = {
        "image_index": image_index,
        "loader_position": int(reference["loader_position"]),
        "dataset_index": int(reference["dataset_index"]),
        "sample_name": reference["sample_name"],
        "full_shape_hw": reference["full_shape_hw"],
        "common_bounds": common_mapping,
        "crop_grid": reference["crop_grid"],
        "endpoint_validation": {
            "computed_k1_prediction_sha256": computed_sha,
            "stage_a_k1_prediction_sha256": expected_sha,
            "prediction_equal": True,
            "computed_confusion": confusion.tolist(),
            "stage_a_confusion_equal": True,
        },
        "coverage": summaries["coverage"],
        "residue_metadata": accumulator.residue_metadata(16),
        "host_accumulator_storage_nbytes": accumulator.storage_nbytes(),
        "crop_source_runtime_seconds": crop_source_runtime_seconds,
    }
    return image_record, cell_records


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    stage_a_sha = file_sha256(args.stage_a_json)
    stage_a = load_stage_a(args.stage_a_json)
    _validate_stage_protocol(stage_a, args)
    cells_by_image = _stage_cells_by_image(stage_a)
    full_image_count = len(stage_a["images"])
    mode = "cache-replay" if args.raw_cache_manifest is not None else "live-k1"

    image_records: list[dict[str, Any]] = []
    cell_records: list[dict[str, Any]] = []
    source_provenance: dict[str, Any]
    peak_cuda_memory = None

    if args.raw_cache_manifest is not None:
        _, image_index, label, crop_iterator, source_provenance = _cache_inputs(
            args.raw_cache_manifest,
            stage_a=stage_a,
            stage_a_sha=stage_a_sha,
            cells_by_image=cells_by_image,
        )
        reference = stage_a["images"][image_index]
        cells = cells_by_image[image_index]
        accumulator = OverlapDisagreementAccumulator(
            reference["full_shape_hw"], NUM_CLASSES
        )
        replay_started = time.perf_counter()
        for crop, window in crop_iterator:
            accumulator.add_crop(crop, window)
        replay_seconds = time.perf_counter() - replay_started
        image_record, records = _image_result(
            accumulator=accumulator,
            image_index=image_index,
            reference=reference,
            cells=cells,
            label=label,
            crop_source_runtime_seconds=replay_seconds,
        )
        image_records.append(image_record)
        cell_records.extend(records)
        del accumulator, label
    else:
        device = torch.device(args.device)
        model, _, dataset, checkpoint_sha = _live_model_and_dataset(
            args.baseline_checkpoint, args.seed, device
        )
        if len(dataset) != full_image_count:
            raise ValueError("WHU test length differs from Stage A")
        limit = (
            full_image_count
            if args.max_images is None
            else min(args.max_images, full_image_count)
        )
        source_provenance = {
            "mode": "sealed-live-normal-k1-streaming-statistics",
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
            optical, sar, label = _load_live_image(
                dataset, image_index, reference
            )
            accumulator = OverlapDisagreementAccumulator(
                reference["full_shape_hw"], NUM_CLASSES
            )
            print(
                f"image={image_index + 1}/{limit} sample={reference['sample_name']} "
                f"shape={tuple(reference['full_shape_hw'])} crops={len(windows)}",
                flush=True,
            )
            image_runtime = _run_live_crops(
                model=model,
                optical=optical,
                sar=sar,
                windows=windows,
                batch_size=args.inference_batch_size,
                device=device,
                accumulator=accumulator,
            )
            image_record, records = _image_result(
                accumulator=accumulator,
                image_index=image_index,
                reference=reference,
                cells=cells,
                label=label,
                crop_source_runtime_seconds=image_runtime,
            )
            image_records.append(image_record)
            cell_records.extend(records)
            print(
                f"image={image_index + 1}/{limit} endpoint=PASS "
                f"overlap_fraction={image_record['coverage']['overlap_fraction']:.6f} "
                f"runtime={image_runtime:.3f}s",
                flush=True,
            )
            del accumulator, optical, sar, label
            gc.collect()
        peak_cuda_memory = int(torch.cuda.max_memory_allocated(device))

    evaluated_images = len(image_records)
    full_scope = mode == "live-k1" and evaluated_images == full_image_count
    scope = (
        "full-test"
        if full_scope
        else (
            "single-image-cache-correctness"
            if mode == "cache-replay"
            else "live-prefix-smoke"
        )
    )
    protocol = {
        "hypothesis": "H4-A post-hoc independent mechanism screen",
        "input_visibility": "normal K1 raw overlapping crop responses only",
        "crop_size_hw": [512, 512],
        "stride_hw": [341, 341],
        "crop_order": "sealed Stage-A row-major",
        "inference_batch_size": args.inference_batch_size,
        "raw_crop_dtype": "float32",
        "aggregation_endpoint": "sum crop logits then count-normalize; argmax unchanged",
        "scores": {
            SCORE_NAMES[0]: (
                "generalized probability JSD divided by log(crop_count), "
                "cell mean over pixels with crop_count>=2; primary score"
            ),
            SCORE_NAMES[1]: (
                "1-max_class_vote_fraction normalized by 1-1/crop_count, "
                "cell mean over pixels with crop_count>=2"
            ),
            SCORE_NAMES[2]: (
                "4*q*(1-q) for per-crop semantic-boundary votes, cell mean "
                "where at least two crop-interior boundary observations exist"
            ),
        },
        "report_only": {
            "overlap_jsd_normalized_top10pct_mean": (
                "cell upper 10% mean on the normalized JSD map; frozen dilution "
                "diagnostic only and forbidden from H4-A ranking/selection"
            )
        },
        "residue_role": (
            "origin modulo 16 is stored as mechanism metadata, not screened as "
            "a separate selector"
        ),
        "gt_use": "labels used only for endpoint confusion equality, never score construction",
        "scientific_limit": (
            "cache/prefix modes are correctness smoke; full-test official-test "
            "scores remain exploratory method-selection data"
        ),
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "PASS",
        "status_meaning": (
            "All processed normal K1 endpoints exactly reproduce Stage A; cell "
            "response statistics were serialized without raw-crop persistence."
        ),
        "scope": scope,
        "scientific_scope": (
            "H4-A full-test score extraction for offline screening"
            if full_scope
            else "implementation correctness only; no scientific Go/No-Go"
        ),
        "execution_mode": mode,
        "full_test_length": full_image_count,
        "evaluated_images": evaluated_images,
        "source_stage_a": {
            "path": str(args.stage_a_json.resolve()),
            "sha256": stage_a_sha,
            "artifact_type": stage_a["artifact_type"],
            "schema_version": stage_a["schema_version"],
            "git_revision": stage_a.get("reproducibility", {}).get(
                "git_revision"
            ),
        },
        "source_execution": source_provenance,
        "protocol": protocol,
        "images": image_records,
        "cells": cell_records,
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "common_module_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_overlap_common.py"
            ),
            "seed": args.seed,
            "inference_batch_size": args.inference_batch_size,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device if mode == "live-k1" else None,
            "gpu": (
                torch.cuda.get_device_name(torch.device(args.device))
                if mode == "live-k1"
                else None
            ),
            "peak_cuda_memory_bytes": peak_cuda_memory,
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "created_at_utc": utc_now(),
        "next_step": (
            "Run the separate frozen offline H4-A screen."
            if full_scope
            else (
                "If cache correctness PASS is independently reviewed, run one "
                "foreground full-test live K1 extraction."
            )
        ),
        "explicit_non_claims": [
            "No H4-A predictability or routing gain is established by extraction.",
            "Intra-slide disagreement is not claimed to be a pure causal phase measure.",
            "The formal H3 No-Go is unchanged.",
            "No K2-observed router or utility head is implemented or authorized.",
        ],
    }
    atomic_write_json(args.output_path, output)
    print(f"h4a_overlap_result={args.output_path.resolve()}", flush=True)
    print("h4a_overlap_status=PASS", flush=True)


if __name__ == "__main__":
    main()
