"""Cache one WHU image as raw crops for Stage-B1 correctness replay.

This is deliberately a correctness-only cache producer.  Every phase still
executes the complete row-major 512/341 sliding-window schedule with the same
dense batches used by Stage A.  The normal phase persists every raw float32
crop; shifted phases persist only the K4 common-support dependency closure.
No shared inference code is modified and no training is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
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
    array_sha256,
    atomic_save_npy,
    atomic_write_json,
    build_full_image_dataset,
    canonical_json_sha256,
    file_sha256,
)
from scripts.evaluate_whu_translation_consistency import (  # noqa: E402
    translate_tensor,
)
from scripts.phase_closure_common import (  # noqa: E402
    PHASE_NAMES,
    PHASE_SHIFTS,
    SEALED_CROP_SIZE,
    SEALED_STRIDE,
    build_phase_closure_geometry,
)
from scripts.phase_sparse_replay_common import (  # noqa: E402
    analytic_count_map,
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


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "whu_phase_crop_logits_correctness_cache"
NUM_CLASSES = 7
PHASE_ORDER = ("normal", *PHASE_NAMES)
PHASE_SHIFTS_WITH_NORMAL = {"normal": (0, 0), **PHASE_SHIFTS}


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


def _crop_ids_sha256(crop_ids: Sequence[int]) -> str:
    values = tuple(int(value) for value in crop_ids)
    if values != tuple(sorted(set(values))):
        raise ValueError("crop ids must be unique and strictly increasing")
    encoded = np.asarray(values, dtype="<i8")
    return hashlib.sha256(encoded.tobytes()).hexdigest()


def _hash_descriptor(array: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(array)
    return {
        "dtype": str(values.dtype),
        "shape": [int(value) for value in values.shape],
        "array_sha256": array_sha256(values),
        "nbytes": int(values.nbytes),
    }


def _bounds_mapping(bounds: Sequence[int]) -> dict[str, int]:
    y0, y1, x0, x1 = (int(value) for value in bounds)
    return {"y_start": y0, "y_stop": y1, "x_start": x0, "x_stop": x1}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _validate_sources(
    stage_a_path: Path,
    stage_b0_path: Path,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str, str]:
    stage_a_sha = file_sha256(stage_a_path)
    stage_b0_sha = file_sha256(stage_b0_path)
    checkpoint_sha = file_sha256(checkpoint_path)
    stage_a = _load_json(stage_a_path)
    stage_b0 = _load_json(stage_b0_path)

    stage_a_checks = {
        "status": stage_a.get("status") == "PASS",
        "artifact_type": stage_a.get("artifact_type")
        == "whu_phase_utility_stage_a",
        "schema_version": stage_a.get("schema_version") == 3,
        "scope": stage_a.get("scope") == "full-test",
        "complete": stage_a.get("evaluated_images")
        == stage_a.get("full_test_length"),
        "checkpoint": stage_a.get("baseline_checkpoint_sha256")
        == checkpoint_sha,
    }
    failed = [name for name, passed in stage_a_checks.items() if not passed]
    if failed:
        raise ValueError(f"Stage-A binding failed: {failed}")

    source_stage_a = stage_b0.get("source_stage_a", {})
    decision = stage_b0.get("stage_b0_decision", {})
    stage_b0_checks = {
        "status": stage_b0.get("status") == "PASS",
        "artifact_type": stage_b0.get("artifact_type")
        == "whu_phase_closure_stage_b0",
        "schema_version": stage_b0.get("schema_version") == 1,
        "scope": stage_b0.get("scope") == "full-test",
        "stage_a_sha256": source_stage_a.get("sha256") == stage_a_sha,
        "stage_a_artifact_type": source_stage_a.get("artifact_type")
        == stage_a.get("artifact_type"),
        "stage_a_schema_version": source_stage_a.get("schema_version")
        == stage_a.get("schema_version"),
        "b1_authorized": decision.get("b1_implementation_and_smoke_authorized")
        is True,
    }
    failed = [name for name, passed in stage_b0_checks.items() if not passed]
    if failed:
        raise ValueError(f"Stage-B0 binding failed: {failed}")

    expected_phases = [[0, 0], [0, 8], [8, 0], [8, 8]]
    if stage_a.get("protocol", {}).get("teacher_phases_dy_dx") != expected_phases:
        raise ValueError("Stage-A phase order/shifts differ from the sealed protocol")
    if stage_a.get("protocol", {}).get("crop_size_hw") != list(SEALED_CROP_SIZE):
        raise ValueError("Stage-A crop size differs from the sealed protocol")
    if stage_a.get("protocol", {}).get("stride_hw") != list(SEALED_STRIDE):
        raise ValueError("Stage-A stride differs from the sealed protocol")
    expected_b0_shifts = {
        name: list(PHASE_SHIFTS[name]) for name in PHASE_NAMES
    }
    if stage_b0.get("protocol", {}).get("phase_shifts_dy_dx") != expected_b0_shifts:
        raise ValueError("Stage-B0 phase shifts differ from the sealed protocol")
    return stage_a, stage_b0, stage_a_sha, stage_b0_sha, checkpoint_sha


def _select_image(
    stage_a: Mapping[str, Any],
    *,
    loader_position: int | None,
    sample_name: str | None,
) -> tuple[int, dict[str, Any]]:
    images = stage_a.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("Stage-A images are missing")
    if sample_name is not None:
        matches = [
            (index, image)
            for index, image in enumerate(images)
            if image.get("sample_name") == sample_name
        ]
        if len(matches) != 1:
            raise ValueError("--sample-name must match exactly one Stage-A image")
        image_index, image = matches[0]
    else:
        image_index = 0 if loader_position is None else int(loader_position)
        if image_index < 0 or image_index >= len(images):
            raise IndexError("--loader-position is outside Stage-A images")
        image = images[image_index]
    if image.get("loader_position") != image_index:
        raise ValueError("Stage-A images are not in contiguous loader order")
    return image_index, dict(image)


def _k4_crop_ids(
    stage_a: Mapping[str, Any],
    stage_b0: Mapping[str, Any],
    image_index: int,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    images = stage_a.get("images")
    cells = stage_a.get("cells")
    if not isinstance(images, list) or not isinstance(cells, list):
        raise ValueError("Stage-A geometry records are missing")
    geometry = build_phase_closure_geometry(images, cells)
    levels = endpoint_levels(geometry, 4)
    expected = expected_phase_crop_ids(levels, geometry, image_index)

    per_image = (
        stage_b0.get("minimal_common_support_endpoints", {})
        .get("k4", {})
        .get("closure", {})
        .get("per_image")
    )
    if not isinstance(per_image, list) or image_index >= len(per_image):
        raise ValueError("Stage-B0 K4 per-image closure is missing")
    b0_record = per_image[image_index]
    if (
        b0_record.get("image_index") != image_index
        or b0_record.get("sample_name") != geometry["sample_names"][image_index]
    ):
        raise ValueError("Stage-B0 K4 image identity differs from Stage A")
    for phase_name in PHASE_NAMES:
        ids = tuple(int(value) for value in b0_record["phase_crop_ids"][phase_name])
        digest = _crop_ids_sha256(ids)
        if ids != expected[phase_name]:
            raise ValueError(f"Stage-B0 {phase_name} crop ids differ from geometry")
        if digest != b0_record["phase_crop_id_sha256"][phase_name]:
            raise ValueError(f"Stage-B0 {phase_name} crop-id SHA256 is invalid")

    common_bounds = tuple(int(value) for value in geometry["common_bounds"][image_index])
    dense_count = analytic_count_map(geometry, image_index)
    for phase_name in PHASE_NAMES:
        sparse_count = analytic_count_map(
            geometry, image_index, expected[phase_name]
        )
        dy, dx = PHASE_SHIFTS[phase_name]
        y0, y1, x0, x1 = common_bounds
        shifted = (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))
        if not np.array_equal(sparse_count[shifted], dense_count[shifted]):
            raise AssertionError(
                f"{phase_name} K4 dependency closure is incomplete on common support"
            )
    return geometry, expected


def _protocol() -> dict[str, Any]:
    return {
        "phase_order": list(PHASE_ORDER),
        "phase_shifts_dy_dx": {
            name: list(PHASE_SHIFTS_WITH_NORMAL[name]) for name in PHASE_ORDER
        },
        "crop_size_hw": list(SEALED_CROP_SIZE),
        "stride_hw": list(SEALED_STRIDE),
        "dense_crop_order": (
            "all row-major Stage-A windows; local_crop_id is the dense sequence index"
        ),
        "dense_batching": (
            "every phase executes all dense crops in contiguous original batches; "
            "support-pruned crops are filtered only after their dense batch forward"
        ),
        "modality_pairing": "RGB and SAR use the identical shifted canvas and crop window",
        "translation": "positive zero-fill shifted[y+dy,x+dx]=original[y,x]",
        "persisted_crop_policy": {
            "normal": "all dense raw crops",
            "x8_y8_xy8": "Stage-B0-validated K4 common-support closure only",
        },
        "raw_crop_storage_dtype": "float32",
        "dense_common_hash_scope": (
            "phase sum/count/normalized logits aligned to original common bounds; "
            "hash descriptors only, arrays are reconstructed from raw crops"
        ),
        "dense_common_count_dtype": "int16",
        "crop_id_sha256_encoding": "little-endian int64 bytes",
        "scientific_limit": (
            "single-image implementation correctness only; not a scientific gain, "
            "latency, structure, or full-test result"
        ),
    }


def _environment(device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "git_revision": _git_revision(),
        "runner_sha256": file_sha256(Path(__file__)),
        "phase_closure_common_sha256": file_sha256(
            REPO_ROOT / "scripts" / "phase_closure_common.py"
        ),
        "phase_sparse_replay_common_sha256": file_sha256(
            REPO_ROOT / "scripts" / "phase_sparse_replay_common.py"
        ),
        "seed": int(args.seed),
        "inference_batch_size": int(args.inference_batch_size),
        "device": str(device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def _persist_or_validate_array(
    path: Path, array: np.ndarray, *, resume: bool
) -> dict[str, Any]:
    values = np.ascontiguousarray(array)
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.int64)):
        raise TypeError("cache arrays must be raw float32 logits or int64 labels")
    if not path.exists():
        return atomic_save_npy(path, values)
    if not resume:
        raise FileExistsError(f"refusing to overwrite array: {path}")
    cached = np.load(path, allow_pickle=False)
    if cached.dtype != values.dtype or cached.shape != values.shape:
        raise ValueError(f"resume array metadata differs: {path}")
    if array_sha256(cached) != array_sha256(values):
        raise ValueError(f"resume array values differ: {path}")
    return {
        "dtype": str(cached.dtype),
        "shape": [int(value) for value in cached.shape],
        "array_sha256": array_sha256(cached),
        "file_sha256": file_sha256(path),
        "nbytes": int(cached.nbytes),
    }


def _persist_partial(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(
        output_dir / "manifest.partial.json", manifest, replace=True
    )


def _validate_resume_manifest(
    manifest: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    keys = (
        "schema_version",
        "artifact_type",
        "scope",
        "scientific_scope",
        "source_stage_a",
        "source_stage_b0",
        "baseline_checkpoint",
        "protocol",
        "protocol_sha256",
        "reproducibility",
    )
    failed = [name for name in keys if manifest.get(name) != expected.get(name)]
    if manifest.get("status") != "IN_PROGRESS":
        failed.append("status")
    expected_image = expected["image"]
    actual_image = manifest.get("image", {})
    image_keys = (
        "image_index",
        "loader_position",
        "dataset_index",
        "sample_name",
        "full_shape_hw",
        "common_bounds",
        "crop_grid",
        "source",
        "source_file_sha256",
        "decoded_tensor_sha256",
    )
    failed.extend(
        f"image.{name}"
        for name in image_keys
        if actual_image.get(name) != expected_image.get(name)
    )
    if failed:
        raise ValueError(f"partial cache is incompatible with this run: {failed}")


def _crop_record(
    output_dir: Path,
    phase_name: str,
    crop_id: int,
    window: Sequence[int],
    batch_size: int,
    logits: np.ndarray,
    *,
    resume: bool,
) -> dict[str, Any]:
    relative = Path("crops") / phase_name / f"crop_{crop_id:04d}.npy"
    descriptor = _persist_or_validate_array(
        output_dir / relative, logits, resume=resume
    )
    return {
        "local_crop_id": int(crop_id),
        "window_yxyx": [int(value) for value in window],
        "dense_sequence_index": int(crop_id),
        "dense_batch_index": int(crop_id // batch_size),
        "dense_batch_slot": int(crop_id % batch_size),
        "path": relative.as_posix(),
        **descriptor,
    }


def _run_dense_phase(
    *,
    model: torch.nn.Module,
    optical: torch.Tensor,
    sar: torch.Tensor,
    windows: Sequence[Sequence[int]],
    common_bounds: Sequence[int],
    phase_name: str,
    shift: tuple[int, int],
    persisted_ids: Sequence[int],
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    resume: bool,
) -> tuple[dict[str, Any], torch.Tensor, np.ndarray | None]:
    if optical.ndim != 4 or sar.ndim != 4 or optical.shape[0] != 1 or sar.shape[0] != 1:
        raise ValueError("RGB and SAR tensors must be single-image BCHW")
    if optical.shape[-2:] != sar.shape[-2:]:
        raise ValueError("RGB and SAR canvas shapes differ")
    height, width = (int(value) for value in optical.shape[-2:])
    dy, dx = shift
    if phase_name == "normal":
        phase_optical = optical.to(device)
        phase_sar = sar.to(device)
    else:
        phase_optical = translate_tensor(optical, dy, dx).to(device)
        phase_sar = translate_tensor(sar, dy, dx).to(device)

    score_sum = phase_optical.new_zeros((1, NUM_CLASSES, height, width))
    count_mat = phase_optical.new_zeros((1, 1, height, width)).to(torch.int8)
    selected = tuple(int(value) for value in persisted_ids)
    if selected != tuple(sorted(set(selected))):
        raise ValueError("persisted crop ids must be unique and strictly increasing")
    selected_set = set(selected)
    crop_records: list[dict[str, Any]] = []
    dense_batch_count = math.ceil(len(windows) / batch_size)
    started = time.perf_counter()

    for batch_index, batch_start in enumerate(range(0, len(windows), batch_size)):
        batch_windows = windows[batch_start : batch_start + batch_size]
        optical_crops = []
        sar_crops = []
        for window in batch_windows:
            y0, y1, x0, x1 = (int(value) for value in window)
            optical_crops.append(phase_optical[:, :, y0:y1, x0:x1])
            sar_crops.append(phase_sar[:, :, y0:y1, x0:x1])
        batch_optical = torch.cat(optical_crops, dim=0)
        batch_sar = torch.cat(sar_crops, dim=0)
        batch_predictions = model(batch_optical, batch_sar)
        if not isinstance(batch_predictions, torch.Tensor):
            raise TypeError("sealed linear decoder must return a tensor")
        expected_shape = (len(batch_windows), NUM_CLASSES, *SEALED_CROP_SIZE)
        if tuple(batch_predictions.shape) != expected_shape:
            raise ValueError(
                f"model crop output shape {tuple(batch_predictions.shape)} "
                f"differs from {expected_shape}"
            )
        batch_predictions = batch_predictions.to(device=device, dtype=torch.float32)
        if not torch.isfinite(batch_predictions).all():
            raise FloatingPointError(f"{phase_name} crop logits are non-finite")

        for slot, window in enumerate(batch_windows):
            crop_id = batch_start + slot
            y0, y1, x0, x1 = (int(value) for value in window)
            score_sum[:, :, y0:y1, x0:x1] += batch_predictions[slot]
            count_mat[:, :, y0:y1, x0:x1] += 1
            if crop_id in selected_set:
                logits = np.ascontiguousarray(
                    batch_predictions[slot].detach().cpu().numpy(),
                    dtype=np.float32,
                )
                crop_records.append(
                    _crop_record(
                        output_dir,
                        phase_name,
                        crop_id,
                        window,
                        batch_size,
                        logits,
                        resume=resume,
                    )
                )
        print(
            f"phase={phase_name} dense_batch={batch_index + 1}/{dense_batch_count} "
            f"dense_crops={min(batch_start + len(batch_windows), len(windows))}/"
            f"{len(windows)} persisted={len(crop_records)}/{len(selected)}",
            flush=True,
        )
        del batch_optical, batch_sar, batch_predictions

    if tuple(record["local_crop_id"] for record in crop_records) != selected:
        raise AssertionError(f"{phase_name} persisted crop order is incomplete")
    if torch.any(count_mat == 0):
        raise AssertionError(f"{phase_name} dense count_mat has uncovered pixels")

    y0, y1, x0, x1 = (int(value) for value in common_bounds)
    shifted_bounds = (y0 + dy, y1 + dy, x0 + dx, x1 + dx)
    sy0, sy1, sx0, sx1 = shifted_bounds
    common_sum = score_sum[0, :, sy0:sy1, sx0:sx1].detach().cpu()
    common_count = (
        count_mat[0, 0, sy0:sy1, sx0:sx1]
        .detach()
        .cpu()
        .to(torch.int16)
    )
    common_sum_np = np.ascontiguousarray(common_sum.numpy(), dtype=np.float32)
    common_count_np = np.ascontiguousarray(common_count.numpy(), dtype=np.int16)
    common_mean_np = np.zeros(common_sum_np.shape, dtype=np.float32)
    np.divide(
        common_sum_np,
        common_count_np[None],
        out=common_mean_np,
        where=common_count_np[None] > 0,
    )
    common_mean = torch.from_numpy(common_mean_np)
    full_prediction = None
    if phase_name == "normal":
        # Match slide_inference's CPU normalization before argmax without
        # materializing another full [1,C,H,W] host tensor at once.
        prediction_rows = []
        for row_start in range(0, height, 256):
            row_stop = min(row_start + 256, height)
            normalized_rows = (
                score_sum[:, :, row_start:row_stop, :].detach().cpu()
                / count_mat[:, :, row_start:row_stop, :].detach().cpu()
            )
            prediction_rows.append(
                normalized_rows.argmax(dim=1)
                .numpy()
                .astype(np.int64, copy=False)
            )
        full_prediction = np.ascontiguousarray(
            np.concatenate(prediction_rows, axis=1)
        )

    torch.cuda.synchronize(device)
    phase_record = {
        "shift_dy_dx": [int(dy), int(dx)],
        "dense_crop_count": int(len(windows)),
        "dense_batch_count": int(dense_batch_count),
        "persisted_crop_count": int(len(selected)),
        "persisted_crop_ids": list(selected),
        "persisted_crop_id_sha256": _crop_ids_sha256(selected),
        "crops": crop_records,
        "dense_common": {
            "original_bounds_yxyx": [y0, y1, x0, x1],
            "shifted_bounds_yxyx": list(shifted_bounds),
            "sum_logits": _hash_descriptor(common_sum_np),
            "count_mat": _hash_descriptor(common_count_np),
            "normalized_logits": _hash_descriptor(common_mean_np),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    del phase_optical, phase_sar, score_sum, count_mat, common_sum, common_count
    return phase_record, common_mean, full_prediction


def _prediction_validation(
    prediction: np.ndarray, expected_sha256: str, endpoint: str
) -> dict[str, Any]:
    computed = array_sha256(prediction)
    if computed != expected_sha256:
        raise AssertionError(
            f"{endpoint} prediction SHA256 differs from Stage A: "
            f"{computed} != {expected_sha256}"
        )
    return {
        "computed_prediction_sha256": computed,
        "stage_a_prediction_sha256": expected_sha256,
        "equal": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache raw dense-order phase crops for one Stage-B1 correctness image"
        )
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-a-json", type=Path, required=True)
    parser.add_argument("--stage-b0-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--loader-position", type=int)
    selection.add_argument("--sample-name")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for name in ("baseline_checkpoint", "stage_a_json", "stage_b0_json"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"--{name.replace('_', '-')} does not exist: {path}")
    if args.loader_position is not None and args.loader_position < 0:
        parser.error("--loader-position must be non-negative")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.resume:
        if not args.output_dir.is_dir():
            parser.error(f"--resume directory does not exist: {args.output_dir}")
        if (args.output_dir / "manifest.json").exists():
            parser.error("cache is already complete")
        if not (args.output_dir / "manifest.partial.json").is_file():
            parser.error("--resume requires manifest.partial.json")
    elif args.output_dir.exists():
        parser.error(f"refusing to overwrite output directory: {args.output_dir}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stage-B1 correctness cache")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select CUDA for exact reference execution")

    stage_a, stage_b0, stage_a_sha, stage_b0_sha, checkpoint_sha = (
        _validate_sources(
            args.stage_a_json, args.stage_b0_json, args.baseline_checkpoint
        )
    )
    expected_batch = stage_a.get("reproducibility", {}).get(
        "inference_batch_size"
    )
    expected_seed = stage_a.get("reproducibility", {}).get("seed")
    if args.inference_batch_size != expected_batch:
        raise ValueError(
            "batch size must match Stage A for dense batch-slot equivalence: "
            f"{args.inference_batch_size} != {expected_batch}"
        )
    if args.seed != expected_seed:
        raise ValueError(
            f"seed must match Stage A: {args.seed} != {expected_seed}"
        )

    image_index, image_reference = _select_image(
        stage_a,
        loader_position=args.loader_position,
        sample_name=args.sample_name,
    )
    geometry, extra_crop_ids = _k4_crop_ids(
        stage_a, stage_b0, image_index
    )
    windows = geometry["windows_by_image"][image_index]
    full_shape = tuple(int(value) for value in geometry["image_shapes"][image_index])
    common_bounds = tuple(
        int(value) for value in geometry["common_bounds"][image_index]
    )

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    dataset = build_full_image_dataset("test")
    dataset_index = int(image_reference["dataset_index"])
    if dataset_index != image_index:
        raise ValueError(
            "formal full-test Stage-A dataset_index must equal loader_position"
        )
    if dataset_index < 0 or dataset_index >= len(dataset):
        raise IndexError("Stage-A dataset_index is outside the WHU test dataset")
    sample_name = str(image_reference["sample_name"])
    source = {
        "rgb_file": str(Path(dataset.rgb_files[dataset_index]).resolve()),
        "sar_file": str(Path(dataset.sar_files[dataset_index]).resolve()),
        "label_file": str(Path(dataset.label_files[dataset_index]).resolve()),
    }
    if Path(source["rgb_file"]).stem != sample_name:
        raise ValueError("dataset RGB identity differs from Stage A")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [dataset_index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    optical, sar, label_tensor = next(iter(loader))
    label_batched = np.ascontiguousarray(
        label_tensor.numpy().astype(np.int64, copy=False)
    )
    if label_batched.shape != (1, *full_shape):
        raise ValueError("decoded label shape differs from Stage A")
    label = np.ascontiguousarray(label_batched[0])
    if tuple(optical.shape[-2:]) != full_shape or tuple(sar.shape[-2:]) != full_shape:
        raise ValueError("decoded RGB/SAR shapes differ from Stage A")

    source_file_sha256 = {
        "rgb": file_sha256(Path(source["rgb_file"])),
        "sar": file_sha256(Path(source["sar_file"])),
        "label": file_sha256(Path(source["label_file"])),
    }
    decoded_tensor_sha256 = {
        "optical": array_sha256(
            np.ascontiguousarray(optical.numpy(), dtype=np.float32)
        ),
        "sar": array_sha256(np.ascontiguousarray(sar.numpy(), dtype=np.float32)),
        "label_int64": array_sha256(label),
    }
    protocol = _protocol()
    reproducibility = _environment(device, args)
    base_manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "IN_PROGRESS",
        "scope": "single-image-correctness-only",
        "scientific_scope": (
            "Stage-B1 cached replay implementation audit; all phases execute the "
            "full dense schedule, but this artifact cannot establish gain or latency"
        ),
        "source_stage_a": {
            "path": str(args.stage_a_json.resolve()),
            "sha256": stage_a_sha,
            "artifact_type": stage_a["artifact_type"],
            "schema_version": stage_a["schema_version"],
            "git_revision": stage_a.get("reproducibility", {}).get("git_revision"),
        },
        "source_stage_b0": {
            "path": str(args.stage_b0_json.resolve()),
            "sha256": stage_b0_sha,
            "artifact_type": stage_b0["artifact_type"],
            "schema_version": stage_b0["schema_version"],
            "git_revision": stage_b0.get("reproducibility", {}).get("git_revision"),
        },
        "baseline_checkpoint": {
            "path": str(args.baseline_checkpoint.resolve()),
            "sha256": checkpoint_sha,
        },
        "protocol": protocol,
        "protocol_sha256": canonical_json_sha256(protocol),
        "reproducibility": reproducibility,
        "image": {
            "image_index": int(image_index),
            "loader_position": int(image_reference["loader_position"]),
            "dataset_index": dataset_index,
            "sample_name": sample_name,
            "full_shape_hw": list(full_shape),
            "common_bounds": _bounds_mapping(common_bounds),
            "crop_grid": image_reference["crop_grid"],
            "source": source,
            "source_file_sha256": source_file_sha256,
            "decoded_tensor_sha256": decoded_tensor_sha256,
            "label": None,
            "phases": {},
            "endpoint_validation": {},
        },
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
    }

    partial_path = args.output_dir / "manifest.partial.json"
    if args.resume:
        manifest = _load_json(partial_path)
        _validate_resume_manifest(manifest, base_manifest)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = base_manifest
        _persist_partial(args.output_dir, manifest)

    relative_label = Path("labels") / "label.npy"
    label_descriptor = _persist_or_validate_array(
        args.output_dir / relative_label, label, resume=args.resume
    )
    label_record = {"path": relative_label.as_posix(), **label_descriptor}
    if manifest["image"].get("label") not in (None, label_record):
        raise ValueError("resume label descriptor differs")
    manifest["image"]["label"] = label_record
    _persist_partial(args.output_dir, manifest)

    from scripts.diagnose_whu_spatial_errors import load_model

    model, cfg = load_model(args.baseline_checkpoint, args.seed)
    if tuple(cfg["window_size"]) != SEALED_CROP_SIZE:
        raise ValueError("model window size differs from the sealed 512x512 crop")
    if len(cfg["labels"]) != NUM_CLASSES:
        raise ValueError("model class count differs from the sealed WHU protocol")
    model.to(device)
    model.eval()

    normal_ids = tuple(range(len(windows)))
    persisted_by_phase = {"normal": normal_ids, **extra_crop_ids}
    print(
        f"sample={sample_name} image_index={image_index} dense_crops={len(windows)} "
        + " ".join(
            f"{name}_persisted={len(persisted_by_phase[name])}"
            for name in PHASE_ORDER
        ),
        flush=True,
    )
    print(f"checkpoint_sha256={checkpoint_sha}", flush=True)

    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_prediction: np.ndarray | None = None
    fused_common: torch.Tensor | None = None
    expected_prediction_hashes = image_reference.get("prediction_sha256", {})
    with torch.inference_mode():
        for phase_name in PHASE_ORDER:
            phase_record, common_mean, phase_prediction = _run_dense_phase(
                model=model,
                optical=optical,
                sar=sar,
                windows=windows,
                common_bounds=common_bounds,
                phase_name=phase_name,
                shift=PHASE_SHIFTS_WITH_NORMAL[phase_name],
                persisted_ids=persisted_by_phase[phase_name],
                output_dir=args.output_dir,
                batch_size=args.inference_batch_size,
                device=device,
                resume=args.resume,
            )
            old_record = manifest["image"]["phases"].get(phase_name)
            if old_record is not None:
                old_deterministic = dict(old_record)
                new_deterministic = dict(phase_record)
                old_deterministic.pop("runtime_seconds", None)
                new_deterministic.pop("runtime_seconds", None)
                if old_deterministic != new_deterministic:
                    raise ValueError(f"resume {phase_name} phase record differs")
                phase_record = old_record
            manifest["image"]["phases"][phase_name] = phase_record

            if phase_name == "normal":
                if phase_prediction is None:
                    raise AssertionError("normal phase did not return a prediction")
                baseline_prediction = phase_prediction
                fused_common = common_mean.clone()
                manifest["image"]["endpoint_validation"]["k1"] = (
                    _prediction_validation(
                        baseline_prediction,
                        expected_prediction_hashes["k1"],
                        "k1",
                    )
                )
            else:
                if fused_common is None or baseline_prediction is None:
                    raise AssertionError("normal phase must execute first")
                fused_common.add_(common_mean)
                endpoint = "matched_k2" if phase_name == "x8" else None
                if phase_name == "xy8":
                    endpoint = "k4"
                if endpoint is not None:
                    prediction = np.ascontiguousarray(baseline_prediction.copy())
                    y0, y1, x0, x1 = common_bounds
                    prediction[0, y0:y1, x0:x1] = (
                        fused_common.argmax(dim=0)
                        .numpy()
                        .astype(np.int64, copy=False)
                    )
                    manifest["image"]["endpoint_validation"][endpoint] = (
                        _prediction_validation(
                            prediction,
                            expected_prediction_hashes[endpoint],
                            endpoint,
                        )
                    )
            _persist_partial(args.output_dir, manifest)
            del common_mean

    manifest["status"] = "PASS"
    manifest["status_meaning"] = (
        "All four phases executed their full Stage-A dense batch schedule; raw "
        "crop files, source bindings, K4 closure, and K1/K2/K4 endpoint hashes passed"
    )
    manifest["completed_at_utc"] = utc_now()
    manifest["runtime_seconds"] = float(time.perf_counter() - started)
    manifest["peak_cuda_memory_bytes"] = int(
        torch.cuda.max_memory_allocated(device)
    )
    complete_path = args.output_dir / "manifest.json"
    atomic_write_json(complete_path, manifest)
    partial_path.unlink()
    print(f"phase_crop_cache_manifest={complete_path.resolve()}", flush=True)
    print("phase_crop_cache_status=PASS", flush=True)


if __name__ == "__main__":
    main()
