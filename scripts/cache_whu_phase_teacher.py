"""Cache the exact WHU four-phase teacher logits on full source images.

The cache deliberately reproduces the completed zero-training ceiling protocol:
each translated phase is evaluated by independent 512/341 sliding inference,
overlap logits are normalized inside that phase, the four phase maps are aligned
to one physical region, and their logits are averaged.  Only the fused teacher
crop is persisted; normal-view logits are never cached.

Long runs are resumable only when ``--resume`` is explicit.  Every array and
metadata file is first written to a unique ``.part`` file and atomically moved
into place.  ``manifest.partial.json`` is updated after every completed image
and is promoted to ``manifest.json`` only after the requested cache is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from scripts.evaluate_whu_phase_ensemble_2d import four_phase_shifts  # noqa: E402
from scripts.evaluate_whu_translation_consistency import translate_tensor  # noqa: E402
from scripts.spatial_diagnostics_common import common_translation_slices  # noqa: E402
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.inference import slide_inference  # noqa: E402


BACKBONE_TYPE = "dinov3_vits16"
NUM_CLASSES = 7
CROP_SIZE = (512, 512)
STRIDE = (341, 341)
PHASE_OFFSET = 8
CONTROL_OFFSET = 16
VALID_MARGIN = 512
SCHEMA_VERSION = 1
ARTIFACT_TYPE = "whu_exact_four_phase_teacher_logits"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_artifact_key(index: int, sample_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_name)).strip("._")
    if not safe_name:
        safe_name = "sample"
    return f"{int(index):04d}_{safe_name}"


def build_phase_protocol() -> dict[str, Any]:
    primary = four_phase_shifts(PHASE_OFFSET)
    control = four_phase_shifts(CONTROL_OFFSET)
    common_nonzero = tuple(
        shift for shifts in (primary, control) for shift in shifts if shift != (0, 0)
    )
    return {
        "teacher_phases_dy_dx": [list(shift) for shift in primary],
        "common_alignment_shifts_dy_dx": [
            list(shift) for shift in common_nonzero
        ],
        "control_offset_used_only_for_common_region": CONTROL_OFFSET,
        "crop_size_hw": list(CROP_SIZE),
        "stride_hw": list(STRIDE),
        "valid_margin": VALID_MARGIN,
        "translation": (
            "positive zero-fill translation: shifted[y+dy,x+dx]=original[y,x]"
        ),
        "per_phase_inference": (
            "independent full-image sliding inference; overlapping crop logits "
            "are summed and divided by that phase's count_mat"
        ),
        "alignment": "shifted logits are sampled at original coordinates plus dy/dx",
        "phase_fusion": "arithmetic mean of four aligned float32 logit maps",
        "cached_tensor": "only fused teacher logits inside common bounds",
        "normal_logits_cached": False,
        "formal_equivalence": (
            "matches evaluate_whu_phase_ensemble_2d.py primary geometry, including "
            "the common valid region jointly fixed by the 8px and 16px conditions"
        ),
    }


def phase_common_slices(
    shape: tuple[int, int],
) -> tuple[tuple[slice, slice], dict[tuple[int, int], tuple[slice, slice]]]:
    protocol = build_phase_protocol()
    shifts = tuple(
        tuple(int(value) for value in shift)
        for shift in protocol["common_alignment_shifts_dy_dx"]
    )
    return common_translation_slices(shape, shifts, VALID_MARGIN)


def bounds_from_slice(
    full_shape: tuple[int, int], region: tuple[slice, slice]
) -> dict[str, int]:
    height, width = (int(value) for value in full_shape)
    y_slice, x_slice = region
    y_start, y_stop, y_step = y_slice.indices(height)
    x_start, x_stop, x_step = x_slice.indices(width)
    if y_step != 1 or x_step != 1:
        raise ValueError("phase teacher bounds must use unit-stride slices")
    return {
        "y_start": y_start,
        "y_stop": y_stop,
        "x_start": x_start,
        "x_stop": x_stop,
    }


def aligned_phase_crop(
    scores: torch.Tensor,
    shift: tuple[int, int],
    original_slice: tuple[slice, slice],
    shifted_slices: Mapping[tuple[int, int], tuple[slice, slice]],
) -> torch.Tensor:
    """Return one CHW phase logit crop aligned to the original region."""

    if scores.ndim != 4 or scores.shape[0] != 1:
        raise ValueError("phase scores must have shape [1, C, H, W]")
    shift = (int(shift[0]), int(shift[1]))
    region = original_slice if shift == (0, 0) else shifted_slices[shift]
    crop = scores[0, :, region[0], region[1]]
    expected = scores[0, :, original_slice[0], original_slice[1]].shape
    if crop.shape != expected:
        raise AssertionError(
            f"aligned phase shape mismatch for {shift}: {crop.shape} != {expected}"
        )
    return crop


def accumulate_aligned_phase(
    score_sum: torch.Tensor | None,
    scores: torch.Tensor,
    shift: tuple[int, int],
    original_slice: tuple[slice, slice],
    shifted_slices: Mapping[tuple[int, int], tuple[slice, slice]],
) -> torch.Tensor:
    """Accumulate one aligned phase without retaining its full-image map."""

    crop = aligned_phase_crop(scores, shift, original_slice, shifted_slices)
    if score_sum is None:
        return crop.clone()
    if score_sum.shape != crop.shape:
        raise ValueError(
            f"phase accumulator shape mismatch: {score_sum.shape} != {crop.shape}"
        )
    score_sum.add_(crop)
    return score_sum


def phase_arithmetic_mean(score_sum: torch.Tensor, phase_count: int) -> torch.Tensor:
    """Finalize the exact, equal-weight arithmetic mean of aligned phase logits."""

    if score_sum.ndim != 3:
        raise ValueError("aligned phase sum must be CHW")
    if phase_count <= 0:
        raise ValueError("phase_count must be positive")
    return score_sum.mul_(1.0 / float(phase_count))


def atomic_save_npy(path: Path, array: np.ndarray) -> dict[str, Any]:
    """Write one NumPy array atomically without replacing an existing artifact."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite array: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    contiguous = np.ascontiguousarray(array)
    try:
        with temporary.open("xb") as stream:
            np.save(stream, contiguous, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        artifact = {
            "dtype": str(contiguous.dtype),
            "shape": [int(value) for value in contiguous.shape],
            "array_sha256": array_sha256(contiguous),
            "file_sha256": file_sha256(temporary),
            "nbytes": int(contiguous.nbytes),
        }
        os.replace(temporary, path)
        return artifact
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(
    path: Path, payload: Mapping[str, Any], *, replace: bool = False
) -> str:
    """Write UTF-8 JSON atomically; replacement must be explicitly authorized."""

    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        digest = file_sha256(temporary)
        os.replace(temporary, path)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def build_full_image_dataset(split: str):
    """Build WHU with the requested file list but deterministic full-image access."""

    if split not in {"train", "test"}:
        raise ValueError("WHU split must be 'train' or 'test'")
    dataset = build_dataset(
        "WHU",
        split,
        window_size=CROP_SIZE,
        model_name="DINOv3",
        modality="multi",
        backbone_type=BACKBONE_TYPE,
    )
    # WHU_Dataset uses data_type only to select random augmented training crops.
    # The filenames were already fixed by build_dataset, so switching this flag
    # gives deterministic full-image tensors from the selected train/test list.
    dataset.data_type = "test"
    return dataset


def selected_indices(
    full_length: int, max_images: int | None, smoke: bool
) -> list[int]:
    if full_length <= 0:
        raise ValueError("WHU split is empty")
    if smoke and max_images is not None:
        raise ValueError("--smoke and --max-images are mutually exclusive")
    limit = 1 if smoke else max_images
    if limit is None:
        limit = full_length
    if limit <= 0 or limit > full_length:
        raise ValueError(
            f"requested image count must be in [1, {full_length}], got {limit}"
        )
    return list(range(limit))


def dataset_selection(dataset, indices: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {
            "index": int(index),
            "sample_name": Path(dataset.rgb_files[index]).stem,
            "rgb_file": str(Path(dataset.rgb_files[index]).resolve()),
            "sar_file": str(Path(dataset.sar_files[index]).resolve()),
            "label_file": str(Path(dataset.label_files[index]).resolve()),
        }
        for index in indices
    ]


def image_hw(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return int(height), int(width)


def estimate_cache(selection: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    images = []
    total_bytes = 0
    for sample in selection:
        source_shapes = {
            "rgb": image_hw(Path(sample["rgb_file"])),
            "sar": image_hw(Path(sample["sar_file"])),
            "label": image_hw(Path(sample["label_file"])),
        }
        if len(set(source_shapes.values())) != 1:
            raise ValueError(
                f"RGB/SAR/label shapes differ for {sample['sample_name']}: "
                f"{source_shapes}"
            )
        shape = source_shapes["rgb"]
        original_slice, _ = phase_common_slices(shape)
        bounds = bounds_from_slice(shape, original_slice)
        crop_height = bounds["y_stop"] - bounds["y_start"]
        crop_width = bounds["x_stop"] - bounds["x_start"]
        nbytes = NUM_CLASSES * crop_height * crop_width * np.dtype(np.float16).itemsize
        total_bytes += nbytes
        images.append(
            {
                "index": sample["index"],
                "sample_name": sample["sample_name"],
                "full_shape_hw": list(shape),
                "source_shapes_hw": {
                    name: list(value) for name, value in source_shapes.items()
                },
                "bounds": bounds,
                "teacher_shape_chw": [NUM_CLASSES, crop_height, crop_width],
                "float16_nbytes": int(nbytes),
            }
        )
    return {
        "status": "ESTIMATE_ONLY",
        "model_loaded": False,
        "image_count": len(images),
        "total_float16_nbytes": int(total_bytes),
        "total_float16_gib": float(total_bytes / 1024**3),
        "protocol": build_phase_protocol(),
        "images": images,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache exact full-image WHU four-phase 8px teacher logits"
    )
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    if args.smoke and args.max_images is not None:
        parser.error("--smoke and --max-images are mutually exclusive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.estimate_only:
        if args.resume:
            parser.error("--estimate-only cannot be combined with --resume")
        return args
    if args.baseline_checkpoint is None or not args.baseline_checkpoint.is_file():
        parser.error("--baseline-checkpoint must name an existing file")
    if args.output_dir is None:
        parser.error("--output-dir is required unless --estimate-only is used")
    if args.output_dir.exists() and not args.resume:
        parser.error(
            f"output already exists; refuse silent reuse (use --resume only for "
            f"a compatible partial cache): {args.output_dir}"
        )
    if args.resume and not args.output_dir.is_dir():
        parser.error(f"resume directory does not exist: {args.output_dir}")
    return args


def initial_manifest(
    args: argparse.Namespace,
    full_length: int,
    selection: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    protocol = build_phase_protocol()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "IN_PROGRESS",
        "scope": (
            "subset-smoke"
            if args.smoke or args.max_images is not None
            else "full-split"
        ),
        "split": args.split,
        "full_dataset_length": int(full_length),
        "requested_images": [dict(sample) for sample in selection],
        "baseline_checkpoint": {
            "path": str(args.baseline_checkpoint.resolve()),
            "sha256": checkpoint_sha256,
        },
        "protocol": protocol,
        "protocol_sha256": canonical_json_sha256(protocol),
        "execution": {
            "seed": int(args.seed),
            "inference_batch_size": int(args.inference_batch_size),
        },
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "images": [],
    }


def validate_resume_manifest(
    manifest: Mapping[str, Any],
    args: argparse.Namespace,
    full_length: int,
    selection: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
) -> None:
    expected_protocol = build_phase_protocol()
    checks = {
        "schema_version": manifest.get("schema_version") == SCHEMA_VERSION,
        "artifact_type": manifest.get("artifact_type") == ARTIFACT_TYPE,
        "status": manifest.get("status") in {"IN_PROGRESS", "PASS"},
        "split": manifest.get("split") == args.split,
        "full_dataset_length": manifest.get("full_dataset_length") == full_length,
        "requested_images": manifest.get("requested_images")
        == [dict(sample) for sample in selection],
        "checkpoint_sha256": manifest.get("baseline_checkpoint", {}).get("sha256")
        == checkpoint_sha256,
        "protocol": manifest.get("protocol") == expected_protocol,
        "protocol_sha256": manifest.get("protocol_sha256")
        == canonical_json_sha256(expected_protocol),
        "seed": manifest.get("execution", {}).get("seed") == args.seed,
        "inference_batch_size": manifest.get("execution", {}).get(
            "inference_batch_size"
        )
        == args.inference_batch_size,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"partial cache is incompatible with this run: {failed}")


def validate_cached_record(
    output_dir: Path,
    record: Mapping[str, Any],
    expected_sample: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    if record.get("index") != expected_sample["index"]:
        raise ValueError("cached image index differs from requested selection")
    if record.get("sample_name") != expected_sample["sample_name"]:
        raise ValueError("cached sample name differs from requested selection")
    logits = record.get("logits", {})
    npy_path = output_dir / str(logits.get("path", ""))
    metadata_path = output_dir / str(logits.get("metadata_path", ""))
    if not npy_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"cached record is incomplete: {record.get('sample_name')}")
    if file_sha256(npy_path) != logits.get("file_sha256"):
        raise ValueError(f"cached array SHA256 mismatch: {npy_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol") != protocol:
        raise ValueError(f"cached sidecar protocol mismatch: {metadata_path}")
    sidecar_record = metadata.get("record", {})
    if not isinstance(sidecar_record, Mapping):
        raise ValueError(f"cached sidecar record is invalid: {metadata_path}")
    if dict(sidecar_record) != dict(record):
        raise ValueError(f"cached sidecar record differs from manifest: {metadata_path}")
    if sidecar_record.get("logits", {}).get("file_sha256") != logits.get(
        "file_sha256"
    ):
        raise ValueError(f"cached sidecar SHA256 mismatch: {metadata_path}")
    expected_source_hashes = record.get("source_file_sha256", {})
    for field, source_key in (
        ("rgb", "rgb_file"),
        ("sar", "sar_file"),
        ("label", "label_file"),
    ):
        source_path = Path(expected_sample[source_key])
        if file_sha256(source_path) != expected_source_hashes.get(field):
            raise ValueError(f"source {field} changed since cache creation: {source_path}")


def recover_completed_sidecar(
    output_dir: Path,
    expected_sample: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Recover an image completed just before an interrupted manifest update."""

    key = safe_artifact_key(expected_sample["index"], expected_sample["sample_name"])
    npy_path = output_dir / "logits" / f"{key}.npy"
    metadata_path = output_dir / "metadata" / f"{key}.json"
    if not npy_path.exists() and not metadata_path.exists():
        return None
    if npy_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        record = metadata.get("record")
        if not isinstance(record, dict):
            raise ValueError(f"orphan sidecar lacks a recoverable record: {metadata_path}")
        validate_cached_record(output_dir, record, expected_sample, protocol)
        return record

    # ``--resume`` explicitly authorizes removal of an incomplete artifact that
    # was never committed to the partial manifest.  Targets are deterministic,
    # per-image files under the requested cache directory only.
    npy_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    return None


def persist_partial_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(
        output_dir / "manifest.partial.json", manifest, replace=True
    )


def teacher_record(
    output_dir: Path,
    sample: Mapping[str, Any],
    label: np.ndarray,
    full_shape: tuple[int, int],
    bounds: Mapping[str, int],
    teacher: np.ndarray,
    protocol: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    key = safe_artifact_key(sample["index"], sample["sample_name"])
    relative_npy = Path("logits") / f"{key}.npy"
    relative_metadata = Path("metadata") / f"{key}.json"
    artifact = atomic_save_npy(output_dir / relative_npy, teacher)
    source_file_sha256 = {
        "rgb": file_sha256(Path(sample["rgb_file"])),
        "sar": file_sha256(Path(sample["sar_file"])),
        "label": file_sha256(Path(sample["label_file"])),
    }
    record = {
        "index": sample["index"],
        "sample_name": sample["sample_name"],
        "source": {
            "rgb_file": sample["rgb_file"],
            "sar_file": sample["sar_file"],
            "label_file": sample["label_file"],
        },
        "source_file_sha256": source_file_sha256,
        "source_label_file_sha256": source_file_sha256["label"],
        "label_sha256": array_sha256(label.astype(np.int64, copy=False)),
        "full_shape_hw": [int(value) for value in full_shape],
        "bounds": {name: int(value) for name, value in bounds.items()},
        "logits": {
            "path": relative_npy.as_posix(),
            "metadata_path": relative_metadata.as_posix(),
            **artifact,
        },
    }
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "baseline_checkpoint_sha256": checkpoint_sha256,
        "protocol": dict(protocol),
        "protocol_sha256": canonical_json_sha256(protocol),
        "record": record,
    }
    atomic_write_json(output_dir / relative_metadata, sidecar)
    return record


def main() -> None:
    args = parse_args()
    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    dataset = build_full_image_dataset(args.split)
    indices = selected_indices(len(dataset), args.max_images, args.smoke)
    selection = dataset_selection(dataset, indices)

    if args.estimate_only:
        print(json.dumps(estimate_cache(selection), ensure_ascii=False, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact WHU phase-teacher caching")

    checkpoint_sha256 = file_sha256(args.baseline_checkpoint)
    partial_path = args.output_dir / "manifest.partial.json"
    complete_path = args.output_dir / "manifest.json"
    if args.resume:
        if complete_path.exists():
            raise FileExistsError(f"cache is already complete: {complete_path}")
        if not partial_path.is_file():
            raise FileNotFoundError(f"resume manifest is missing: {partial_path}")
        manifest = json.loads(partial_path.read_text(encoding="utf-8"))
        validate_resume_manifest(
            manifest,
            args,
            len(dataset),
            selection,
            checkpoint_sha256,
        )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = initial_manifest(
            args, len(dataset), selection, checkpoint_sha256
        )
        persist_partial_manifest(args.output_dir, manifest)

    protocol = manifest["protocol"]
    completed_by_index = {record["index"]: record for record in manifest["images"]}
    for sample in selection:
        record = completed_by_index.get(sample["index"])
        if record is not None:
            validate_cached_record(args.output_dir, record, sample, protocol)
            print(
                f"resume_verified index={sample['index']} name={sample['sample_name']}",
                flush=True,
            )
            continue
        recovered = recover_completed_sidecar(args.output_dir, sample, protocol)
        if recovered is not None:
            manifest["images"].append(recovered)
            manifest["images"].sort(key=lambda item: item["index"])
            persist_partial_manifest(args.output_dir, manifest)
            completed_by_index[sample["index"]] = recovered
            print(
                f"resume_recovered index={sample['index']} name={sample['sample_name']}",
                flush=True,
            )

    # Importing/building the model is intentionally delayed until after preflight
    # and resume validation, so --estimate-only never loads weights.
    from scripts.diagnose_whu_spatial_errors import load_model

    model, cfg = load_model(args.baseline_checkpoint, args.seed)
    if tuple(cfg["window_size"]) != CROP_SIZE:
        raise ValueError(
            f"formal teacher requires 512x512 crops, got {cfg['window_size']}"
        )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    remaining_selection = [
        sample for sample in selection if sample["index"] not in completed_by_index
    ]
    remaining_indices = [sample["index"] for sample in remaining_selection]
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, remaining_indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for (optical, sar, label_tensor), sample in zip(
            loader, remaining_selection, strict=True
        ):
            label = np.ascontiguousarray(
                label_tensor[0].numpy().astype(np.int64, copy=False)
            )
            full_shape = tuple(int(value) for value in label.shape)
            if tuple(optical.shape[-2:]) != full_shape or tuple(sar.shape[-2:]) != full_shape:
                raise AssertionError("RGB, SAR, and label full-image shapes differ")
            original_slice, shifted_slices = phase_common_slices(full_shape)
            bounds = bounds_from_slice(full_shape, original_slice)
            teacher_sum = None

            for shift in four_phase_shifts(PHASE_OFFSET):
                dy, dx = shift
                if shift == (0, 0):
                    phase_optical = optical.to(device)
                    phase_sar = sar.to(device)
                else:
                    phase_optical = translate_tensor(optical, dy, dx).to(device)
                    phase_sar = translate_tensor(sar, dy, dx).to(device)
                phase_scores = slide_inference(
                    phase_optical,
                    model,
                    dsm=phase_sar,
                    n_output_channels=NUM_CLASSES,
                    crop_size=CROP_SIZE,
                    stride=STRIDE,
                    batch_size=args.inference_batch_size,
                )
                teacher_sum = accumulate_aligned_phase(
                    teacher_sum,
                    phase_scores,
                    shift,
                    original_slice,
                    shifted_slices,
                )
                del phase_optical, phase_sar, phase_scores

            if teacher_sum is None:
                raise AssertionError("four-phase teacher accumulation is empty")
            teacher_sum = phase_arithmetic_mean(
                teacher_sum, len(four_phase_shifts(PHASE_OFFSET))
            )
            teacher = np.ascontiguousarray(
                teacher_sum.numpy().astype(np.float16, copy=False)
            )
            expected_shape = (
                NUM_CLASSES,
                bounds["y_stop"] - bounds["y_start"],
                bounds["x_stop"] - bounds["x_start"],
            )
            if teacher.shape != expected_shape:
                raise AssertionError(
                    f"teacher crop shape differs from bounds: {teacher.shape} != "
                    f"{expected_shape}"
                )
            record = teacher_record(
                args.output_dir,
                sample,
                label,
                full_shape,
                bounds,
                teacher,
                protocol,
                checkpoint_sha256,
            )
            manifest["images"].append(record)
            manifest["images"].sort(key=lambda item: item["index"])
            persist_partial_manifest(args.output_dir, manifest)
            completed_by_index[sample["index"]] = record
            print(
                f"cached={len(manifest['images'])}/{len(selection)} "
                f"name={sample['sample_name']} shape={teacher.shape} "
                f"file_sha256={record['logits']['file_sha256']}",
                flush=True,
            )
            del label, teacher_sum, teacher, optical, sar, label_tensor

    if len(manifest["images"]) != len(selection):
        raise AssertionError("cache completed without all requested image records")
    manifest["status"] = "PASS"
    manifest["completed_at_utc"] = utc_now()
    manifest["runtime"] = {
        "elapsed_seconds_this_process": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }
    persist_partial_manifest(args.output_dir, manifest)
    if complete_path.exists():
        raise FileExistsError(f"refusing to overwrite completed manifest: {complete_path}")
    os.replace(partial_path, complete_path)
    print(f"phase_teacher_manifest={complete_path.resolve()}")
    print("phase_teacher_cache_status=PASS")


if __name__ == "__main__":
    main()
