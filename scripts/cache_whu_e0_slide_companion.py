"""Cache normal-phase WHU sliding logits bound to a sealed phase teacher.

This artifact is deliberately separate from ``cache_whu_phase_teacher.py``.
The bound teacher remains a schema-v1, fused-only cache whose protocol says
``normal_logits_cached=False``.  This script evaluates only the untranslated
``(0, 0)`` view with the same 512/341 count-normalized full-image sliding
operator, crops it to each teacher record's exact common bounds, and persists
the result in an independently versioned companion cache.

Long runs are resumable only when ``--resume`` is explicit.  Arrays and JSON
sidecars are written atomically, a partial manifest is committed after every
image, and the completed manifest is published only after every bound teacher
record has been verified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))


from scripts.cache_whu_phase_teacher import (
    ARTIFACT_TYPE as TEACHER_ARTIFACT_TYPE,
    CROP_SIZE,
    NUM_CLASSES,
    SCHEMA_VERSION as TEACHER_SCHEMA_VERSION,
    STRIDE,
    array_sha256,
    atomic_save_npy,
    atomic_write_json,
    bounds_from_slice,
    build_full_image_dataset,
    build_phase_protocol,
    canonical_json_sha256,
    dataset_selection,
    file_sha256,
    phase_common_slices,
    safe_artifact_key,
    utc_now,
)
from scripts.whu_cache_compat import CACHE_CAPACITY, install_whu_cache_compat
from scripts.whu_label_dtype_compat import install_whu_label_dtype_compat
from utils.inference import slide_inference


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "whu_e0_slide_companion_logits"
NORMAL_PHASE = (0, 0)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class E0SlideCompanionRecord:
    """Validated runner-facing descriptor for one companion logit map."""

    index: int
    sample_name: str
    full_shape_hw: tuple[int, int]
    bounds_yxyx: tuple[int, int, int, int]
    logits_path: Path
    logits_shape: tuple[int, int, int]
    logits_array_sha256: str
    logits_file_sha256: str
    teacher_record_sha256: str


def _require_sha256(value: Any, field: str) -> str:
    result = str(value)
    if SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return result


def _shape_hw(value: Any, field: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain [height, width]")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{field} must contain positive dimensions")
    return result


def _shape_chw(value: Any, field: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must contain [channels, height, width]")
    result = tuple(int(item) for item in value)
    if any(item <= 0 for item in result):
        raise ValueError(f"{field} must contain positive dimensions")
    return result


def _bounds_yxyx(value: Any, field: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a coordinate mapping")
    try:
        result = tuple(
            int(value[key]) for key in ("y_start", "y_stop", "x_start", "x_stop")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{field} is missing integer y/x bounds") from error
    y0, y1, x0, x1 = result
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"{field} is invalid: {result}")
    return result


def _bounds_dict(bounds: tuple[int, int, int, int]) -> dict[str, int]:
    return dict(
        zip(
            ("y_start", "y_stop", "x_start", "x_stop"),
            (int(value) for value in bounds),
            strict=True,
        )
    )


def _resolve_artifact_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    relative = Path(value)
    path = relative if relative.is_absolute() else root / relative
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path.resolve()


def _teacher_record_sha256(record: Mapping[str, Any]) -> str:
    return canonical_json_sha256({"teacher_record": dict(record)})


def build_e0_slide_protocol() -> dict[str, Any]:
    """Return the standalone normal-phase cache protocol."""

    return {
        "phase_dy_dx": list(NORMAL_PHASE),
        "input_view": "untranslated normal phase",
        "crop_size_hw": list(CROP_SIZE),
        "stride_hw": list(STRIDE),
        "inference": (
            "independent full-image sliding inference; overlapping crop logits "
            "are summed and divided by the normal phase count_mat"
        ),
        "spatial_region": (
            "exact common original-coordinate bounds copied from the bound "
            "schema-v1 four-phase teacher record"
        ),
        "alignment": "normal logits are sampled directly in original coordinates",
        "cached_tensor": "normal-phase count-normalized sliding logits inside common bounds",
        "storage_dtype": "float16",
        "loss_dtype": "runner converts cached logits to float32",
        "teacher_cache_mutated": False,
    }


def _validate_teacher_record(record: Mapping[str, Any], expected_index: int) -> None:
    if int(record.get("index", -1)) != expected_index:
        raise ValueError("teacher image indices must be contiguous from zero")
    sample_name = record.get("sample_name")
    if not isinstance(sample_name, str) or not sample_name:
        raise ValueError("teacher image record lacks sample_name")
    full_shape = _shape_hw(record.get("full_shape_hw"), "teacher full_shape_hw")
    bounds = _bounds_yxyx(record.get("bounds"), "teacher bounds")
    original_slice, _ = phase_common_slices(full_shape)
    expected_bounds = _bounds_yxyx(
        bounds_from_slice(full_shape, original_slice), "expected teacher bounds"
    )
    if bounds != expected_bounds:
        raise ValueError(
            f"teacher common bounds differ for {sample_name}: {bounds} != {expected_bounds}"
        )
    source_hashes = record.get("source_file_sha256")
    if not isinstance(source_hashes, Mapping):
        raise ValueError(f"teacher source hashes missing: {sample_name}")
    for name in ("rgb", "sar", "label"):
        _require_sha256(source_hashes.get(name), f"teacher {name} source SHA")
    _require_sha256(record.get("label_sha256"), "teacher decoded label SHA")

    logits = record.get("logits")
    if not isinstance(logits, Mapping):
        raise ValueError(f"teacher logits descriptor missing: {sample_name}")
    if logits.get("dtype") != "float16":
        raise ValueError(f"teacher logits dtype differs from float16: {sample_name}")
    logits_shape = _shape_chw(logits.get("shape"), "teacher logits shape")
    expected_shape = (
        NUM_CLASSES,
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
    )
    if logits_shape != expected_shape:
        raise ValueError(
            f"teacher logits/bounds mismatch for {sample_name}: "
            f"{logits_shape} != {expected_shape}"
        )
    _require_sha256(logits.get("array_sha256"), "teacher logits array SHA")
    _require_sha256(logits.get("file_sha256"), "teacher logits file SHA")


def read_bound_teacher_manifest(path: Path) -> dict[str, Any]:
    """Read and strictly validate the immutable teacher manifest semantics."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"teacher manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError("bound teacher manifest must have PASS status")
    if int(payload.get("schema_version", -1)) != TEACHER_SCHEMA_VERSION:
        raise ValueError("bound teacher manifest must remain schema-v1")
    if payload.get("artifact_type") != TEACHER_ARTIFACT_TYPE:
        raise ValueError("bound manifest is not the exact four-phase teacher artifact")
    protocol = payload.get("protocol")
    expected_protocol = build_phase_protocol()
    if not isinstance(protocol, Mapping) or dict(protocol) != expected_protocol:
        raise ValueError("bound teacher protocol differs from the sealed teacher protocol")
    if payload.get("protocol_sha256") != canonical_json_sha256(expected_protocol):
        raise ValueError("bound teacher protocol SHA is inconsistent")
    if protocol.get("normal_logits_cached") is not False:
        raise ValueError("bound teacher must remain a fused-only cache")
    split = payload.get("split")
    expected_full_length = {"train": 80, "test": 20}.get(split)
    if expected_full_length is None:
        raise ValueError("bound teacher split must be train or test")
    if int(payload.get("full_dataset_length", -1)) != expected_full_length:
        raise ValueError("bound teacher full dataset length is inconsistent")
    if payload.get("scope") not in {"subset-smoke", "full-split"}:
        raise ValueError("bound teacher scope is not a released cache scope")
    checkpoint = payload.get("baseline_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("bound teacher lacks baseline checkpoint metadata")
    _require_sha256(checkpoint.get("sha256"), "teacher baseline checkpoint SHA")
    images = payload.get("images")
    requested = payload.get("requested_images")
    if not isinstance(images, list) or not images:
        raise ValueError("bound teacher manifest contains no images")
    if not isinstance(requested, list) or len(requested) != len(images):
        raise ValueError("bound teacher requested/completed image counts differ")
    for expected_index, record in enumerate(images):
        if not isinstance(record, Mapping):
            raise ValueError("bound teacher image record must be a mapping")
        _validate_teacher_record(record, expected_index)
        request = requested[expected_index]
        if not isinstance(request, Mapping):
            raise ValueError("bound teacher requested image record must be a mapping")
        if int(request.get("index", -1)) != expected_index:
            raise ValueError("bound teacher requested image indices are inconsistent")
        if request.get("sample_name") != record.get("sample_name"):
            raise ValueError("bound teacher requested/completed sample names differ")
    return payload


def teacher_manifest_binding(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal the exact teacher file and the scientific identities used by it."""

    checkpoint = payload["baseline_checkpoint"]
    protocol = payload["protocol"]
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path.resolve()),
        "schema_version": int(payload["schema_version"]),
        "artifact_type": str(payload["artifact_type"]),
        "scope": str(payload.get("scope")),
        "split": str(payload["split"]),
        "full_dataset_length": int(payload["full_dataset_length"]),
        "image_count": len(payload["images"]),
        "protocol_sha256": canonical_json_sha256(protocol),
        "baseline_checkpoint_sha256": _require_sha256(
            checkpoint.get("sha256"), "teacher baseline checkpoint SHA"
        ),
        "image_records_sha256": canonical_json_sha256(
            {"images": list(payload["images"])}
        ),
    }


def _selection_from_teacher(dataset: Any, teacher: Mapping[str, Any]) -> list[dict[str, Any]]:
    if len(dataset) != int(teacher["full_dataset_length"]):
        raise ValueError("local WHU split length differs from the bound teacher")
    local = {
        int(item["index"]): item
        for item in dataset_selection(dataset, range(len(dataset)))
    }
    selection: list[dict[str, Any]] = []
    for teacher_record in teacher["images"]:
        index = int(teacher_record["index"])
        sample = dict(local[index])
        if sample["sample_name"] != teacher_record["sample_name"]:
            raise ValueError(
                f"local/teacher sample mismatch at index {index}: "
                f"{sample['sample_name']} != {teacher_record['sample_name']}"
            )
        sample.update(
            {
                "full_shape_hw": list(teacher_record["full_shape_hw"]),
                "bounds": dict(teacher_record["bounds"]),
                "label_sha256": str(teacher_record["label_sha256"]),
                "source_file_sha256": dict(teacher_record["source_file_sha256"]),
                "teacher_record_sha256": _teacher_record_sha256(teacher_record),
            }
        )
        selection.append(sample)
    return selection


def _validate_sample_sources(sample: Mapping[str, Any]) -> dict[str, str]:
    actual = {
        "rgb": file_sha256(Path(sample["rgb_file"])),
        "sar": file_sha256(Path(sample["sar_file"])),
        "label": file_sha256(Path(sample["label_file"])),
    }
    if actual != dict(sample["source_file_sha256"]):
        raise ValueError(f"source files differ from teacher: {sample['sample_name']}")
    return actual


def initial_manifest(
    args: argparse.Namespace,
    *,
    full_length: int,
    selection: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = build_e0_slide_protocol()
    requested_images = [
        {
            "index": int(sample["index"]),
            "sample_name": str(sample["sample_name"]),
            "rgb_file": str(sample["rgb_file"]),
            "sar_file": str(sample["sar_file"]),
            "label_file": str(sample["label_file"]),
            "teacher_record_sha256": str(sample["teacher_record_sha256"]),
        }
        for sample in selection
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "IN_PROGRESS",
        "scope": str(binding["scope"]),
        "split": str(binding["split"]),
        "full_dataset_length": int(full_length),
        "requested_images": requested_images,
        "baseline_checkpoint": {
            "path": str(args.baseline_checkpoint.resolve()),
            "sha256": checkpoint_sha256,
        },
        "bound_teacher_manifest": dict(binding),
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
    *,
    full_length: int,
    selection: Sequence[Mapping[str, Any]],
    checkpoint_sha256: str,
    binding: Mapping[str, Any],
) -> None:
    expected = initial_manifest(
        args,
        full_length=full_length,
        selection=selection,
        checkpoint_sha256=checkpoint_sha256,
        binding=binding,
    )
    checks = {
        "schema_version": manifest.get("schema_version") == SCHEMA_VERSION,
        "artifact_type": manifest.get("artifact_type") == ARTIFACT_TYPE,
        "status": manifest.get("status") in {"IN_PROGRESS", "PASS"},
        "scope": manifest.get("scope") == expected["scope"],
        "split": manifest.get("split") == expected["split"],
        "full_dataset_length": manifest.get("full_dataset_length") == full_length,
        "requested_images": manifest.get("requested_images")
        == expected["requested_images"],
        "checkpoint_sha256": manifest.get("baseline_checkpoint", {}).get("sha256")
        == checkpoint_sha256,
        "bound_teacher_manifest": manifest.get("bound_teacher_manifest")
        == dict(binding),
        "protocol": manifest.get("protocol") == expected["protocol"],
        "protocol_sha256": manifest.get("protocol_sha256")
        == expected["protocol_sha256"],
        "seed": manifest.get("execution", {}).get("seed") == args.seed,
        "inference_batch_size": manifest.get("execution", {}).get(
            "inference_batch_size"
        )
        == args.inference_batch_size,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"partial companion cache is incompatible: {failed}")


def _descriptor(record: Mapping[str, Any]) -> Mapping[str, Any]:
    descriptor = record.get("e0_slide_logits")
    if not isinstance(descriptor, Mapping):
        raise ValueError("companion record lacks e0_slide_logits descriptor")
    return descriptor


def validate_cached_record(
    output_dir: Path,
    record: Mapping[str, Any],
    expected_sample: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    if int(record.get("index", -1)) != int(expected_sample["index"]):
        raise ValueError("cached companion image index differs")
    if record.get("sample_name") != expected_sample["sample_name"]:
        raise ValueError("cached companion sample name differs")
    if record.get("teacher_record_sha256") != expected_sample["teacher_record_sha256"]:
        raise ValueError("cached companion teacher record binding differs")
    if record.get("source_file_sha256") != expected_sample["source_file_sha256"]:
        raise ValueError("cached companion source hashes differ")
    if record.get("label_sha256") != expected_sample["label_sha256"]:
        raise ValueError("cached companion label SHA differs")
    if _shape_hw(record.get("full_shape_hw"), "cached companion full_shape_hw") != _shape_hw(
        expected_sample["full_shape_hw"], "expected companion full_shape_hw"
    ):
        raise ValueError("cached companion full shape differs")
    bounds = _bounds_yxyx(record.get("bounds"), "cached companion bounds")
    if bounds != _bounds_yxyx(expected_sample["bounds"], "expected companion bounds"):
        raise ValueError("cached companion bounds differ")
    descriptor = _descriptor(record)
    if descriptor.get("dtype") != "float16":
        raise ValueError("cached companion dtype differs from float16")
    shape = _shape_chw(descriptor.get("shape"), "cached companion logits shape")
    expected_shape = (
        NUM_CLASSES,
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
    )
    if shape != expected_shape:
        raise ValueError("cached companion logits/bounds mismatch")
    _require_sha256(descriptor.get("array_sha256"), "cached companion array SHA")
    _require_sha256(descriptor.get("file_sha256"), "cached companion file SHA")
    npy_path = output_dir / str(descriptor.get("path", ""))
    metadata_path = output_dir / str(descriptor.get("metadata_path", ""))
    if not npy_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"cached companion record is incomplete: {record.get('sample_name')}"
        )
    if file_sha256(npy_path) != descriptor.get("file_sha256"):
        raise ValueError(f"cached companion file SHA mismatch: {npy_path}")
    array = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float16 or tuple(array.shape) != shape:
        raise ValueError(f"cached companion array metadata mismatch: {npy_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("cached companion sidecar schema differs")
    if metadata.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("cached companion sidecar artifact type differs")
    if metadata.get("protocol") != protocol:
        raise ValueError("cached companion sidecar protocol differs")
    if metadata.get("protocol_sha256") != canonical_json_sha256(protocol):
        raise ValueError("cached companion sidecar protocol SHA differs")
    if metadata.get("bound_teacher_manifest") != dict(binding):
        raise ValueError("cached companion sidecar teacher binding differs")
    if metadata.get("baseline_checkpoint_sha256") != binding[
        "baseline_checkpoint_sha256"
    ]:
        raise ValueError("cached companion sidecar checkpoint differs")
    if metadata.get("record") != dict(record):
        raise ValueError("cached companion sidecar record differs from manifest")
    _validate_sample_sources(expected_sample)


def recover_completed_sidecar(
    output_dir: Path,
    expected_sample: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Recover an atomic per-image commit omitted from a partial manifest."""

    key = safe_artifact_key(expected_sample["index"], expected_sample["sample_name"])
    npy_path = output_dir / "e0_slide_logits" / f"{key}.npy"
    metadata_path = output_dir / "metadata" / f"{key}.json"
    if not npy_path.exists() and not metadata_path.exists():
        return None
    if npy_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        record = metadata.get("record")
        if not isinstance(record, dict):
            raise ValueError(f"orphan companion sidecar is invalid: {metadata_path}")
        validate_cached_record(
            output_dir,
            record,
            expected_sample,
            protocol=protocol,
            binding=binding,
        )
        return record

    # Only explicit --resume reaches this path.  These deterministic per-image
    # artifacts live strictly below the requested companion output directory.
    npy_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
    return None


def persist_partial_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = utc_now()
    atomic_write_json(
        output_dir / "manifest.partial.json", manifest, replace=True
    )


def companion_record(
    output_dir: Path,
    sample: Mapping[str, Any],
    *,
    label: np.ndarray,
    full_shape: tuple[int, int],
    bounds: Mapping[str, int],
    e0_slide: np.ndarray,
    protocol: Mapping[str, Any],
    binding: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Atomically persist one companion array and its sealed sidecar."""

    expected_full_shape = _shape_hw(sample["full_shape_hw"], "sample full_shape_hw")
    if tuple(full_shape) != expected_full_shape:
        raise ValueError("source full shape differs from bound teacher record")
    expected_bounds = _bounds_yxyx(sample["bounds"], "sample teacher bounds")
    actual_bounds = _bounds_yxyx(bounds, "companion bounds")
    if actual_bounds != expected_bounds:
        raise ValueError("companion bounds differ from bound teacher record")
    label_digest = array_sha256(np.asarray(label, dtype=np.int64))
    if label_digest != sample["label_sha256"]:
        raise ValueError("decoded label differs from bound teacher record")
    source_hashes = _validate_sample_sources(sample)
    expected_shape = (
        NUM_CLASSES,
        actual_bounds[1] - actual_bounds[0],
        actual_bounds[3] - actual_bounds[2],
    )
    value = np.ascontiguousarray(e0_slide)
    if value.dtype != np.float16 or tuple(value.shape) != expected_shape:
        raise ValueError(
            f"companion logits must be float16 {expected_shape}, got "
            f"{value.dtype} {value.shape}"
        )

    key = safe_artifact_key(sample["index"], sample["sample_name"])
    relative_npy = Path("e0_slide_logits") / f"{key}.npy"
    relative_metadata = Path("metadata") / f"{key}.json"
    artifact = atomic_save_npy(output_dir / relative_npy, value)
    record = {
        "index": int(sample["index"]),
        "sample_name": str(sample["sample_name"]),
        "teacher_record_sha256": str(sample["teacher_record_sha256"]),
        "source": {
            "rgb_file": str(sample["rgb_file"]),
            "sar_file": str(sample["sar_file"]),
            "label_file": str(sample["label_file"]),
        },
        "source_file_sha256": source_hashes,
        "label_sha256": label_digest,
        "full_shape_hw": [int(value) for value in full_shape],
        "bounds": _bounds_dict(actual_bounds),
        "e0_slide_logits": {
            "path": relative_npy.as_posix(),
            "metadata_path": relative_metadata.as_posix(),
            **artifact,
        },
    }
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "baseline_checkpoint_sha256": checkpoint_sha256,
        "bound_teacher_manifest": dict(binding),
        "protocol": dict(protocol),
        "protocol_sha256": canonical_json_sha256(protocol),
        "record": record,
    }
    atomic_write_json(output_dir / relative_metadata, sidecar)
    return record


def crop_normal_slide_logits(
    scores: torch.Tensor, original_slice: tuple[slice, slice]
) -> torch.Tensor:
    """Copy one full-resolution normal-phase map inside teacher common bounds."""

    if scores.ndim != 4 or scores.shape[0] != 1:
        raise ValueError("normal slide scores must have shape [1, C, H, W]")
    if int(scores.shape[1]) != NUM_CLASSES:
        raise ValueError(f"normal slide scores must contain {NUM_CLASSES} classes")
    crop = scores[0, :, original_slice[0], original_slice[1]]
    if crop.ndim != 3 or crop.shape[-2] <= 0 or crop.shape[-1] <= 0:
        raise ValueError("normal slide common crop is empty")
    return crop.detach().cpu().contiguous()


def _validate_companion_record_against_teacher(
    record: Mapping[str, Any],
    teacher_record: Mapping[str, Any],
    *,
    root: Path,
    protocol: Mapping[str, Any],
    binding: Mapping[str, Any],
    verify_artifacts: bool,
) -> E0SlideCompanionRecord:
    index = int(record.get("index", -1))
    sample_name = str(record.get("sample_name", ""))
    if index != int(teacher_record["index"]):
        raise ValueError("companion/teacher image index differs")
    if sample_name != teacher_record["sample_name"]:
        raise ValueError("companion/teacher sample name differs")
    teacher_record_sha = _teacher_record_sha256(teacher_record)
    if record.get("teacher_record_sha256") != teacher_record_sha:
        raise ValueError(f"companion teacher-record SHA differs: {sample_name}")
    if record.get("source_file_sha256") != teacher_record.get("source_file_sha256"):
        raise ValueError(f"companion/teacher source hashes differ: {sample_name}")
    if record.get("label_sha256") != teacher_record.get("label_sha256"):
        raise ValueError(f"companion/teacher label SHA differs: {sample_name}")
    full_shape = _shape_hw(record.get("full_shape_hw"), "companion full_shape_hw")
    teacher_full_shape = _shape_hw(
        teacher_record.get("full_shape_hw"), "teacher full_shape_hw"
    )
    if full_shape != teacher_full_shape:
        raise ValueError(f"companion/teacher full shape differs: {sample_name}")
    bounds = _bounds_yxyx(record.get("bounds"), "companion bounds")
    if bounds != _bounds_yxyx(teacher_record.get("bounds"), "teacher bounds"):
        raise ValueError(f"companion/teacher bounds differ: {sample_name}")

    descriptor = _descriptor(record)
    if descriptor.get("dtype") != "float16":
        raise ValueError(f"companion dtype differs from float16: {sample_name}")
    shape = _shape_chw(descriptor.get("shape"), "companion logits shape")
    expected_shape = (
        NUM_CLASSES,
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
    )
    if shape != expected_shape:
        raise ValueError(f"companion logits/bounds mismatch: {sample_name}")
    array_sha = _require_sha256(
        descriptor.get("array_sha256"), "companion logits array SHA"
    )
    file_sha = _require_sha256(
        descriptor.get("file_sha256"), "companion logits file SHA"
    )
    logits_path = _resolve_artifact_path(
        root, descriptor.get("path"), "companion logits path"
    )
    metadata_path = _resolve_artifact_path(
        root, descriptor.get("metadata_path"), "companion metadata path"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("companion sidecar schema differs")
    if metadata.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("companion sidecar artifact type differs")
    if metadata.get("bound_teacher_manifest") != dict(binding):
        raise ValueError("companion sidecar teacher binding differs")
    if metadata.get("protocol") != dict(protocol):
        raise ValueError("companion sidecar protocol differs")
    if metadata.get("protocol_sha256") != canonical_json_sha256(protocol):
        raise ValueError("companion sidecar protocol SHA differs")
    if metadata.get("record") != dict(record):
        raise ValueError("companion sidecar record differs from manifest")
    if verify_artifacts:
        if file_sha256(logits_path) != file_sha:
            raise ValueError(f"companion logits file SHA mismatch: {sample_name}")
        value = np.load(logits_path, mmap_mode="r", allow_pickle=False)
        if value.dtype != np.float16 or tuple(value.shape) != shape:
            raise ValueError(f"companion logits array metadata mismatch: {sample_name}")
        if array_sha256(value) != array_sha:
            raise ValueError(f"companion logits array SHA mismatch: {sample_name}")
    return E0SlideCompanionRecord(
        index=index,
        sample_name=sample_name,
        full_shape_hw=full_shape,
        bounds_yxyx=bounds,
        logits_path=logits_path,
        logits_shape=shape,
        logits_array_sha256=array_sha,
        logits_file_sha256=file_sha,
        teacher_record_sha256=teacher_record_sha,
    )


def load_e0_slide_companion_records(
    companion_manifest_path: Path,
    teacher_manifest_path: Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    verify_artifacts: bool = False,
) -> dict[int, E0SlideCompanionRecord]:
    """Validate a PASS companion and return immutable records keyed by image index.

    The function is intentionally independent of the training runner.  The
    runner can use ``image_index``, ``crop_y``, and ``crop_x`` with each returned
    record's common bounds to crop the exact physical E0-slide target, while
    lazily checking ``logits_array_sha256`` when the array is first opened.
    """

    teacher_manifest_path = teacher_manifest_path.resolve()
    teacher = read_bound_teacher_manifest(teacher_manifest_path)
    binding = teacher_manifest_binding(teacher_manifest_path, teacher)
    if expected_checkpoint_sha256 is not None:
        expected_checkpoint_sha256 = _require_sha256(
            expected_checkpoint_sha256, "expected checkpoint SHA"
        )
        if binding["baseline_checkpoint_sha256"] != expected_checkpoint_sha256:
            raise ValueError("companion teacher was generated from another checkpoint")

    companion_manifest_path = companion_manifest_path.resolve()
    if not companion_manifest_path.is_file():
        raise FileNotFoundError(
            f"companion manifest does not exist: {companion_manifest_path}"
        )
    payload = json.loads(companion_manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError("companion manifest must have PASS status")
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("companion manifest schema differs")
    if payload.get("artifact_type") != ARTIFACT_TYPE:
        raise ValueError("unexpected companion artifact type")
    if payload.get("bound_teacher_manifest") != binding:
        raise ValueError("companion is not bound to the supplied teacher manifest")
    protocol = payload.get("protocol")
    expected_protocol = build_e0_slide_protocol()
    if not isinstance(protocol, Mapping) or dict(protocol) != expected_protocol:
        raise ValueError("companion protocol differs")
    if payload.get("protocol_sha256") != canonical_json_sha256(expected_protocol):
        raise ValueError("companion protocol SHA is inconsistent")
    checkpoint = payload.get("baseline_checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("sha256") != binding[
        "baseline_checkpoint_sha256"
    ]:
        raise ValueError("companion checkpoint differs from bound teacher")
    if payload.get("scope") != binding["scope"]:
        raise ValueError("companion/teacher scope differs")
    if payload.get("split") != binding["split"]:
        raise ValueError("companion/teacher split differs")
    if int(payload.get("full_dataset_length", -1)) != binding[
        "full_dataset_length"
    ]:
        raise ValueError("companion/teacher full dataset length differs")

    images = payload.get("images")
    teacher_images = teacher["images"]
    if not isinstance(images, list) or len(images) != len(teacher_images):
        raise ValueError("companion/teacher image counts differ")
    requested = payload.get("requested_images")
    if not isinstance(requested, list) or len(requested) != len(teacher_images):
        raise ValueError("companion requested/completed image counts differ")
    for expected_index, (request, teacher_record) in enumerate(
        zip(requested, teacher_images, strict=True)
    ):
        if not isinstance(request, Mapping):
            raise ValueError("companion requested image record must be a mapping")
        expected_request = {
            "index": expected_index,
            "sample_name": teacher_record["sample_name"],
            "teacher_record_sha256": _teacher_record_sha256(teacher_record),
        }
        if any(request.get(name) != value for name, value in expected_request.items()):
            raise ValueError("companion requested image differs from bound teacher")
    result: dict[int, E0SlideCompanionRecord] = {}
    root = companion_manifest_path.parent
    for expected_index, (record, teacher_record) in enumerate(
        zip(images, teacher_images, strict=True)
    ):
        if not isinstance(record, Mapping):
            raise ValueError("companion image record must be a mapping")
        if int(record.get("index", -1)) != expected_index:
            raise ValueError("companion image indices must be contiguous from zero")
        validated = _validate_companion_record_against_teacher(
            record,
            teacher_record,
            root=root,
            protocol=expected_protocol,
            binding=binding,
            verify_artifacts=verify_artifacts,
        )
        result[validated.index] = validated
    if set(result) != set(range(len(teacher_images))):
        raise ValueError("companion image index set differs from bound teacher")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache exact normal-phase full-image WHU sliding logits as a "
            "standalone companion to an immutable phase-teacher manifest"
        )
    )
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if not args.teacher_manifest.is_file():
        parser.error("--teacher-manifest must name an existing PASS manifest")
    if not args.baseline_checkpoint.is_file():
        parser.error("--baseline-checkpoint must name an existing file")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    teacher_root = args.teacher_manifest.resolve().parent
    output = args.output_dir.resolve()
    if output == teacher_root or teacher_root in output.parents:
        parser.error("--output-dir must not be the teacher cache or one of its children")
    if args.output_dir.exists() and not args.resume:
        parser.error(
            "output already exists; use --resume only for a compatible partial cache"
        )
    if args.resume and not args.output_dir.is_dir():
        parser.error("--resume requires an existing companion output directory")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    teacher = read_bound_teacher_manifest(args.teacher_manifest)
    binding = teacher_manifest_binding(args.teacher_manifest, teacher)
    checkpoint_sha256 = file_sha256(args.baseline_checkpoint)
    if checkpoint_sha256 != binding["baseline_checkpoint_sha256"]:
        raise ValueError("baseline checkpoint differs from the bound teacher")

    dataset = build_full_image_dataset(str(teacher["split"]))
    selection = _selection_from_teacher(dataset, teacher)
    partial_path = args.output_dir / "manifest.partial.json"
    complete_path = args.output_dir / "manifest.json"
    if args.resume:
        if complete_path.exists():
            raise FileExistsError(f"companion cache is already complete: {complete_path}")
        if not partial_path.is_file():
            raise FileNotFoundError(f"resume manifest is missing: {partial_path}")
        manifest = json.loads(partial_path.read_text(encoding="utf-8"))
        validate_resume_manifest(
            manifest,
            args,
            full_length=len(dataset),
            selection=selection,
            checkpoint_sha256=checkpoint_sha256,
            binding=binding,
        )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = initial_manifest(
            args,
            full_length=len(dataset),
            selection=selection,
            checkpoint_sha256=checkpoint_sha256,
            binding=binding,
        )
        persist_partial_manifest(args.output_dir, manifest)

    protocol = manifest["protocol"]
    completed_by_index = {int(record["index"]): record for record in manifest["images"]}
    for sample in selection:
        record = completed_by_index.get(int(sample["index"]))
        if record is not None:
            validate_cached_record(
                args.output_dir,
                record,
                sample,
                protocol=protocol,
                binding=binding,
            )
            print(
                f"resume_verified index={sample['index']} name={sample['sample_name']}",
                flush=True,
            )
            continue
        recovered = recover_completed_sidecar(
            args.output_dir,
            sample,
            protocol=protocol,
            binding=binding,
        )
        if recovered is not None:
            manifest["images"].append(recovered)
            manifest["images"].sort(key=lambda item: item["index"])
            persist_partial_manifest(args.output_dir, manifest)
            completed_by_index[int(sample["index"])] = recovered
            print(
                f"resume_recovered index={sample['index']} name={sample['sample_name']}",
                flush=True,
            )

    remaining_selection = [
        sample for sample in selection if int(sample["index"]) not in completed_by_index
    ]
    if remaining_selection and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact E0-slide companion caching")

    started = time.perf_counter()
    peak_allocated_gib = 0.0
    peak_reserved_gib = 0.0
    if remaining_selection:
        from scripts.diagnose_whu_spatial_errors import load_model

        model, cfg = load_model(args.baseline_checkpoint, args.seed)
        if tuple(cfg["window_size"]) != CROP_SIZE:
            raise ValueError(
                f"E0-slide companion requires 512x512 crops, got {cfg['window_size']}"
            )
        device = torch.device(args.device)
        model.to(device)
        model.eval()
        remaining_indices = [int(sample["index"]) for sample in remaining_selection]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, remaining_indices),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
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
                if tuple(optical.shape[-2:]) != full_shape or tuple(
                    sar.shape[-2:]
                ) != full_shape:
                    raise AssertionError("RGB, SAR, and label full-image shapes differ")
                original_slice, _ = phase_common_slices(full_shape)
                bounds = bounds_from_slice(full_shape, original_slice)
                if _bounds_yxyx(bounds, "actual common bounds") != _bounds_yxyx(
                    sample["bounds"], "teacher common bounds"
                ):
                    raise AssertionError("actual common bounds differ from teacher record")
                scores = slide_inference(
                    optical.to(device),
                    model,
                    dsm=sar.to(device),
                    n_output_channels=NUM_CLASSES,
                    crop_size=CROP_SIZE,
                    stride=STRIDE,
                    batch_size=args.inference_batch_size,
                )
                normal_crop = crop_normal_slide_logits(scores, original_slice)
                e0_slide = np.ascontiguousarray(
                    normal_crop.numpy().astype(np.float16, copy=False)
                )
                record = companion_record(
                    args.output_dir,
                    sample,
                    label=label,
                    full_shape=full_shape,
                    bounds=bounds,
                    e0_slide=e0_slide,
                    protocol=protocol,
                    binding=binding,
                    checkpoint_sha256=checkpoint_sha256,
                )
                manifest["images"].append(record)
                manifest["images"].sort(key=lambda item: item["index"])
                persist_partial_manifest(args.output_dir, manifest)
                completed_by_index[int(sample["index"])] = record
                print(
                    f"cached={len(manifest['images'])}/{len(selection)} "
                    f"name={sample['sample_name']} shape={e0_slide.shape} "
                    f"file_sha256={record['e0_slide_logits']['file_sha256']}",
                    flush=True,
                )
                del optical, sar, label_tensor, label, scores, normal_crop, e0_slide
        peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3

    if len(manifest["images"]) != len(selection):
        raise AssertionError("companion completed without every bound teacher image")
    manifest["status"] = "PASS"
    manifest["completed_at_utc"] = utc_now()
    manifest["runtime"] = {
        "elapsed_seconds_this_process": time.perf_counter() - started,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
    }
    persist_partial_manifest(args.output_dir, manifest)
    if complete_path.exists():
        raise FileExistsError(f"refusing to overwrite completed manifest: {complete_path}")
    os.replace(partial_path, complete_path)
    print(f"e0_slide_companion_manifest={complete_path.resolve()}")
    print("e0_slide_companion_cache_status=PASS")


if __name__ == "__main__":
    main()
