"""Paired WHU single-phase distillation screen from a sealed phase teacher.

This runner is deliberately separate from the released MM-DINO trainer.  It
keeps the verified ViT-S E0 model frozen in ``eval`` mode, obtains E0 logits and
its final decoder P2 feature once per batch, and updates two identically
initialized low-capacity correction branches:

* R0: released CE+Dice supervision only;
* R1: the same loss plus a unit-weight masked KL to a cached four-phase teacher.

Training crops are addressed in the coordinates of the original 80 WHU
training images.  They are sampled wholly inside the teacher's exact bounds,
never padded, rescaled, or reflected.  Avoiding reflection keeps each offline
teacher target equal to the exact teacher evaluated on that physical source
image rather than assuming flip equivariance.  The teacher cache and full-image
structure-mask manifests are immutable inputs.  Formal evaluation is
single-phase sliding-window inference at epochs 5, 10, and 15.
``--stop-after-epoch`` and ``--resume-checkpoint`` allow those stages to be run
separately without changing the fixed 15-epoch cosine path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from skimage.io import imread
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TVF
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from configs import get_cfg  # noqa: E402
from datasets import build_dataset  # noqa: E402
from scripts.evaluate_whu_naf_e0 import (  # noqa: E402
    BACKBONE_TYPE,
    build_test_loader,
    file_sha256,
    load_baseline_state,
)
from scripts.phase_distillation_common import (  # noqa: E402
    FrozenE0P2Extractor,
    PhaseCorrectionBranch,
    masked_kl_divergence,
    teacher_gain_mask,
)
from scripts.spatial_diagnostics_common import (  # noqa: E402
    build_spatial_region_masks,
    class_ious_from_confusion,
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
from utils.utils import set_seed  # noqa: E402


NUM_CLASSES = 7
CROP_SIZE = 512
INFERENCE_STRIDE = (341, 341)
VALID_MARGIN = 512
CONTROL_OFFSET = 16
PROTOCOL_EPOCHS = 15
EVALUATION_EPOCHS = (5, 10, 15)
TEACHER_PHASES = ((0, 0), (0, 8), (8, 0), (8, 8))
COMMON_ALIGNMENT_SHIFTS = (
    (0, 8),
    (8, 0),
    (8, 8),
    (0, 16),
    (16, 0),
    (16, 16),
)
KD_WEIGHT = 1.0
KD_TEMPERATURE = 1.0
GAIN_MARGIN = 0.0
IGNORE_INDEX = NUM_CLASSES
DEFAULT_TRAIN_SAMPLES_PER_EPOCH = 3200
DEFAULT_SOURCE_CACHE_SIZE = 16
REGION_NAMES = (
    "boundary_le_0px",
    "component_area_le_256px2",
    "component_thickness_le_4px",
    "actionable_union",
)
COMMON_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
COMMON_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def numpy_array_sha256(array: np.ndarray) -> str:
    """Hash an array in C order without materializing a full-size copy."""

    value = np.asarray(array)
    digest = hashlib.sha256()
    if value.ndim == 0:
        digest.update(np.ascontiguousarray(value).tobytes())
    elif value.ndim == 1:
        digest.update(np.ascontiguousarray(value).tobytes())
    else:
        for leading_slice in value:
            digest.update(np.ascontiguousarray(leading_slice).tobytes())
    return digest.hexdigest()


def named_tensor_sha256(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def e0_state_fingerprints(model: nn.Module) -> dict[str, str]:
    batch_norm_buffer_names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            for buffer_name, _ in module.named_buffers(recurse=False):
                batch_norm_buffer_names.add(prefix + buffer_name)
    named_buffers = dict(model.named_buffers())
    return {
        "parameters_sha256": named_tensor_sha256(model.named_parameters()),
        "batch_norm_buffers_sha256": named_tensor_sha256(
            (name, named_buffers[name]) for name in batch_norm_buffer_names
        ),
    }


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        )


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def epoch_artifact_paths(output_dir: Path, epoch: int) -> dict[str, Path]:
    if epoch <= 0 or epoch > PROTOCOL_EPOCHS:
        raise ValueError(f"epoch must be within 1..{PROTOCOL_EPOCHS}: {epoch}")
    paths = {
        "train": output_dir / f"train_e{epoch}.json",
        "checkpoint": output_dir / f"checkpoint_e{epoch}.pth",
        "commit": output_dir / f"epoch_e{epoch}_commit.json",
    }
    if epoch in EVALUATION_EPOCHS:
        paths["evaluation"] = output_dir / f"evaluation_e{epoch}.json"
    return paths


def verify_epoch_commit(output_dir: Path, epoch: int) -> dict[str, Any]:
    """Verify the last-written marker and every artifact it seals."""

    paths = epoch_artifact_paths(output_dir, epoch)
    marker_path = paths["commit"]
    if not marker_path.is_file():
        raise RuntimeError(f"epoch {epoch} lacks a commit marker: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "COMMITTED" or int(marker.get("epoch", -1)) != epoch:
        raise RuntimeError(f"invalid epoch commit marker: {marker_path}")
    declared = marker.get("files")
    expected_keys = set(paths) - {"commit"}
    if not isinstance(declared, Mapping) or set(declared) != expected_keys:
        raise RuntimeError(f"epoch {epoch} commit artifact set changed")
    for key in sorted(expected_keys):
        descriptor = declared.get(key)
        expected_path = paths[key].resolve()
        if not isinstance(descriptor, Mapping):
            raise RuntimeError(f"epoch {epoch} {key} descriptor is invalid")
        if descriptor.get("path") != paths[key].name:
            raise RuntimeError(f"epoch {epoch} {key} path changed")
        if not expected_path.is_file():
            raise RuntimeError(f"epoch {epoch} committed {key} is missing")
        if descriptor.get("sha256") != file_sha256(expected_path):
            raise RuntimeError(f"epoch {epoch} committed {key} hash differs")
    return marker


def publish_epoch_commit(output_dir: Path, epoch: int) -> Path:
    """Publish one epoch atomically enough for deterministic crash recovery."""

    paths = epoch_artifact_paths(output_dir, epoch)
    marker_path = paths.pop("commit")
    if marker_path.exists():
        raise FileExistsError(f"refusing to overwrite epoch commit: {marker_path}")
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot commit epoch {epoch}; missing artifacts: {missing}")
    marker = {
        "status": "COMMITTED",
        "epoch": epoch,
        "files": {
            key: {"path": path.name, "sha256": file_sha256(path)}
            for key, path in sorted(paths.items())
        },
    }
    write_json_atomic(marker_path, marker)
    verify_epoch_commit(output_dir, epoch)
    return marker_path


def committed_epochs(output_dir: Path) -> list[int]:
    result: list[int] = []
    for path in output_dir.glob("epoch_e*_commit.json"):
        middle = path.name.removeprefix("epoch_e").removesuffix("_commit.json")
        if middle.isdigit():
            result.append(int(middle))
    return sorted(result)


def rebuild_metric_indexes(output_dir: Path) -> None:
    """Rebuild convenience JSONL files only from fully committed epochs."""

    train_records: list[Mapping[str, Any]] = []
    evaluation_records: list[Mapping[str, Any]] = []
    for epoch in committed_epochs(output_dir):
        verify_epoch_commit(output_dir, epoch)
        paths = epoch_artifact_paths(output_dir, epoch)
        train_records.append(json.loads(paths["train"].read_text(encoding="utf-8")))
        evaluation_path = paths.get("evaluation")
        if evaluation_path is not None:
            evaluation_records.append(
                json.loads(evaluation_path.read_text(encoding="utf-8"))
            )
    write_jsonl_atomic(output_dir / "train_metrics.jsonl", train_records)
    write_jsonl_atomic(output_dir / "evaluation_metrics.jsonl", evaluation_records)


def _manifest_records(payload: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    for key in ("images", "records", "samples"):
        records = payload.get(key)
        if isinstance(records, list):
            if not all(isinstance(record, dict) for record in records):
                raise TypeError(f"{path}: manifest {key} must contain objects")
            return records
    raise ValueError(f"{path}: manifest must contain an images list")


def _normalized_sample_name(value: Any) -> str:
    return Path(str(value)).stem.lower()


def _resolve_manifest_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest field {field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"manifest field {field} does not exist: {path}")
    return path.resolve()


def _shape_hw(value: Any, field: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain [height, width]")
    shape = tuple(int(item) for item in value)
    if any(item <= 0 for item in shape):
        raise ValueError(f"{field} must contain positive dimensions")
    return shape


def _teacher_bounds(record: Mapping[str, Any]) -> tuple[int, int, int, int]:
    bounds = record.get("bounds")
    if not isinstance(bounds, Mapping):
        raise ValueError("teacher image record lacks bounds")
    values = tuple(
        int(bounds[key]) for key in ("y_start", "y_stop", "x_start", "x_stop")
    )
    y0, y1, x0, x1 = values
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"invalid teacher bounds: {values}")
    return values


def expected_teacher_bounds(full_shape_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return the common 8/16-phase region used by the accepted reference."""

    height, width = (int(value) for value in full_shape_hw)
    return (
        VALID_MARGIN,
        height - VALID_MARGIN - CONTROL_OFFSET,
        VALID_MARGIN,
        width - VALID_MARGIN - CONTROL_OFFSET,
    )


def validate_teacher_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject any cache whose geometry differs from the accepted 8/16 run."""

    phases = protocol.get("teacher_phases_dy_dx")
    normalized_phases = tuple(tuple(int(v) for v in shift) for shift in phases or ())
    if normalized_phases != TEACHER_PHASES:
        raise RuntimeError(f"teacher phase contract changed: {normalized_phases}")
    if protocol.get("phase_fusion") != (
        "arithmetic mean of four aligned float32 logit maps"
    ):
        raise RuntimeError("teacher logits must be the arithmetic mean of four phases")
    if protocol.get("per_phase_inference") != (
        "independent full-image sliding inference; overlapping crop logits "
        "are summed and divided by that phase's count_mat"
    ):
        raise RuntimeError("teacher cache must use independent count-normalized sliding inference")
    if protocol.get("normal_logits_cached") is not False:
        raise RuntimeError("teacher cache must contain only the fused teacher target")
    if _shape_hw(protocol.get("crop_size_hw"), "teacher crop size") != (
        CROP_SIZE,
        CROP_SIZE,
    ):
        raise RuntimeError("teacher cache must use 512x512 inference crops")
    if _shape_hw(protocol.get("stride_hw"), "teacher stride") != INFERENCE_STRIDE:
        raise RuntimeError("teacher cache must use the accepted 341x341 stride")
    if int(protocol.get("valid_margin", -1)) != VALID_MARGIN:
        raise RuntimeError("teacher cache must use the accepted 512px valid margin")
    if int(protocol.get("control_offset_used_only_for_common_region", -1)) != CONTROL_OFFSET:
        raise RuntimeError("teacher cache must retain the accepted 16px control bound")
    common_shifts = protocol.get("common_alignment_shifts_dy_dx")
    normalized_common_shifts = tuple(
        tuple(int(v) for v in shift) for shift in common_shifts or ()
    )
    if normalized_common_shifts != COMMON_ALIGNMENT_SHIFTS:
        raise RuntimeError(
            "teacher cache common-region shifts differ from the accepted 8/16 comparison"
        )


@dataclass(frozen=True)
class PhaseImageRecord:
    index: int
    sample_name: str
    optical_path: Path
    sar_path: Path
    label_path: Path
    full_shape_hw: tuple[int, int]
    label_sha256: str
    bounds_yxyx: tuple[int, int, int, int]
    teacher_logits_path: Path
    teacher_logits_shape: tuple[int, int, int]
    teacher_logits_array_sha256: str | None
    structure_mask_path: Path
    structure_mask_shape: tuple[int, int]
    structure_mask_bounds_yxyx: tuple[int, int, int, int]
    structure_mask_array_sha256: str


def read_phase_reference(path: Path, baseline_sha256: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = (
        payload.get("status") == "PASS"
        and payload.get("scope") == "full-test"
        and int(payload.get("evaluated_images", -1))
        == int(payload.get("full_test_length", -2))
        and payload.get("baseline_checkpoint_sha256") == baseline_sha256
    )
    if not required:
        raise RuntimeError("phase reference is not a matching full-test PASS")
    validation = payload.get("baseline_validation")
    if not isinstance(validation, Mapping) or not all(
        validation.get(key) is True
        for key in (
            "checked",
            "prediction_sha256_equal",
            "label_sha256_equal",
            "confusion_equal",
            "miou_within_tolerance",
        )
    ):
        raise RuntimeError("phase reference lacks strict E0 validation")
    try:
        baseline = payload["aggregate"]["8"]["baseline"]
        expected_miou_percent = float(baseline["miou_percent"])
        expected_confusion = baseline["confusion"]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("phase reference lacks aggregate['8'].baseline") from error
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "expected_e0_miou_percent": expected_miou_percent,
        "expected_e0_confusion": expected_confusion,
        "expected_prediction_sha256": validation["prediction_sha256"],
        "expected_label_sha256": validation["label_sha256"],
    }


def _mask_file_and_shape(
    root: Path, record: Mapping[str, Any]
) -> tuple[Path, tuple[int, int], tuple[int, int, int, int]]:
    descriptor: Any = record.get("mask")
    if not isinstance(descriptor, Mapping):
        descriptor = record.get("structure_mask")
    if isinstance(descriptor, Mapping):
        relative = descriptor.get("path")
        shape_value = descriptor.get("shape")
        bounds_value = descriptor.get("bounds", record.get("bounds"))
    else:
        relative = record.get("mask_path") or record.get("structure_mask_path")
        shape_value = record.get("mask_shape") or record.get("shape")
        bounds_value = record.get("bounds")
    path = _resolve_manifest_path(root, relative, "structure mask path")
    shape = _shape_hw(shape_value, "structure mask shape")
    if bounds_value is None:
        bounds = (0, shape[0], 0, shape[1])
    elif isinstance(bounds_value, Mapping):
        bounds = tuple(
            int(bounds_value[key])
            for key in ("y_start", "y_stop", "x_start", "x_stop")
        )
    else:
        if not isinstance(bounds_value, (list, tuple)) or len(bounds_value) != 4:
            raise ValueError("structure mask bounds must contain four coordinates")
        bounds = tuple(int(value) for value in bounds_value)
    if shape != (bounds[1] - bounds[0], bounds[3] - bounds[2]):
        raise ValueError("structure mask shape does not match its full-image bounds")
    return path, shape, bounds


def build_phase_records(
    teacher_manifest_path: Path,
    structure_manifest_path: Path,
    source_dataset: Any,
    baseline_sha256: str,
    *,
    require_full_split: bool,
) -> tuple[list[PhaseImageRecord], dict[str, Any]]:
    teacher = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    structure = json.loads(structure_manifest_path.read_text(encoding="utf-8"))
    if teacher.get("status") != "PASS" or int(teacher.get("schema_version", -1)) != 1:
        raise RuntimeError("teacher manifest must be schema-v1 PASS")
    if structure.get("status") != "PASS" or int(structure.get("schema_version", -1)) != 1:
        raise RuntimeError("structure manifest must be schema-v1 PASS")
    if teacher.get("artifact_type") != "whu_exact_four_phase_teacher_logits":
        raise RuntimeError("unexpected teacher manifest artifact type")
    if structure.get("artifact_type") != "whu_full_image_phase_structure_mask":
        raise RuntimeError("unexpected structure manifest artifact type")
    if teacher.get("split") != "train" or structure.get("split") != "train":
        raise RuntimeError("teacher and structure manifests must use the train split")
    if teacher.get("scope") != structure.get("scope"):
        raise RuntimeError("teacher and structure manifest scopes differ")
    if int(teacher.get("full_dataset_length", -1)) != 80 or int(
        structure.get("full_dataset_length", -1)
    ) != 80:
        raise RuntimeError("manifests must originate from the official 80-image train split")
    if require_full_split and teacher.get("scope") != "full-split":
        raise RuntimeError("formal training requires full-split manifests")
    checkpoint = teacher.get("baseline_checkpoint")
    teacher_baseline_sha = (
        checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None
    )
    if teacher_baseline_sha != baseline_sha256:
        raise RuntimeError("teacher cache was made from a different E0 checkpoint")
    protocol = teacher.get("protocol")
    if not isinstance(protocol, Mapping) or teacher.get(
        "protocol_sha256"
    ) != canonical_json_sha256(protocol):
        raise RuntimeError("teacher manifest protocol hash is inconsistent")
    structure_protocol = structure.get("protocol")
    if not isinstance(structure_protocol, Mapping) or structure.get(
        "protocol_sha256"
    ) != canonical_json_sha256(structure_protocol):
        raise RuntimeError("structure manifest protocol hash is inconsistent")
    expected_structure_fields = {
        "valid_class_indices": list(range(NUM_CLASSES)),
        "ignore_class_indices": [IGNORE_INDEX],
        "connectivity": 8,
        "small_definition": "component area <= 256 pixels",
        "thin_definition": "2 * component_area / perimeter_pixel_count <= 4",
    }
    if any(
        structure_protocol.get(name) != value
        for name, value in expected_structure_fields.items()
    ) or structure_protocol.get("encoding", {}).get("dtype") != "uint8":
        raise RuntimeError("structure mask scientific protocol differs from V1")
    validate_teacher_protocol(protocol)

    teacher_records = _manifest_records(teacher, teacher_manifest_path)
    structure_records = _manifest_records(structure, structure_manifest_path)
    if not teacher_records or len(teacher_records) != len(structure_records):
        raise RuntimeError("teacher/mask manifests must contain the same non-zero count")
    if require_full_split and len(teacher_records) != 80:
        raise RuntimeError("formal paired training requires all 80 WHU train images")
    structure_by_name = {
        _normalized_sample_name(record.get("sample_name")): record
        for record in structure_records
    }
    teacher_root = teacher_manifest_path.resolve().parent
    structure_root = structure_manifest_path.resolve().parent

    source_by_name: dict[str, tuple[Path, Path, Path]] = {}
    for optical, sar, label in zip(
        source_dataset.rgb_files,
        source_dataset.sar_files,
        source_dataset.label_files,
        strict=True,
    ):
        source_by_name[_normalized_sample_name(optical)] = (
            Path(optical).resolve(),
            Path(sar).resolve(),
            Path(label).resolve(),
        )
    if len(source_by_name) != 80:
        raise RuntimeError("WHU source dataset does not expose the official 80 images")

    result: list[PhaseImageRecord] = []
    for expected_index, teacher_record in enumerate(teacher_records):
        index = int(teacher_record.get("index", -1))
        if index != expected_index:
            raise RuntimeError("teacher manifest image indices must be contiguous")
        sample_name = str(teacher_record.get("sample_name", ""))
        key = _normalized_sample_name(sample_name)
        if key not in source_by_name or key not in structure_by_name:
            raise RuntimeError(f"manifest/source sample mismatch: {sample_name}")
        structure_record = structure_by_name[key]
        if int(structure_record.get("index", index)) != index:
            raise RuntimeError(f"teacher/mask index mismatch: {sample_name}")
        full_shape = _shape_hw(teacher_record.get("full_shape_hw"), "full_shape_hw")
        structure_full_shape = _shape_hw(
            structure_record.get("full_shape_hw"), "structure full_shape_hw"
        )
        if full_shape != structure_full_shape:
            raise RuntimeError(f"teacher/mask image shape mismatch: {sample_name}")
        label_sha = str(teacher_record.get("label_sha256", ""))
        if not label_sha or label_sha != str(structure_record.get("label_sha256", "")):
            raise RuntimeError(f"teacher/mask label hash mismatch: {sample_name}")
        declared_source_hashes = teacher_record.get("source_file_sha256")
        if not isinstance(declared_source_hashes, Mapping):
            raise RuntimeError(f"teacher source hashes missing: {sample_name}")
        actual_source_hashes = {
            "rgb": file_sha256(source_by_name[key][0]),
            "sar": file_sha256(source_by_name[key][1]),
            "label": file_sha256(source_by_name[key][2]),
        }
        if dict(declared_source_hashes) != actual_source_hashes:
            raise RuntimeError(f"teacher/source file hash mismatch: {sample_name}")
        if structure_record.get("source_label_file_sha256") != actual_source_hashes["label"]:
            raise RuntimeError(f"structure/source label file hash mismatch: {sample_name}")
        bounds = _teacher_bounds(teacher_record)
        expected_bounds = expected_teacher_bounds(full_shape)
        if bounds != expected_bounds:
            raise RuntimeError(
                f"teacher bounds differ from the accepted common region for {sample_name}: "
                f"{bounds} != {expected_bounds}"
            )
        if bounds[1] > full_shape[0] or bounds[3] > full_shape[1]:
            raise RuntimeError(f"teacher bounds exceed full image: {sample_name}")
        if bounds[1] - bounds[0] < CROP_SIZE or bounds[3] - bounds[2] < CROP_SIZE:
            raise RuntimeError(f"teacher exact region is smaller than 512: {sample_name}")
        logits = teacher_record.get("logits")
        if not isinstance(logits, Mapping):
            raise RuntimeError(f"teacher logits descriptor missing: {sample_name}")
        logits_path = _resolve_manifest_path(
            teacher_root, logits.get("path"), "teacher logits path"
        )
        if str(logits.get("dtype")) != "float16":
            raise RuntimeError(f"teacher cache must be float16: {sample_name}")
        logits_shape_value = logits.get("shape")
        if not isinstance(logits_shape_value, (list, tuple)) or len(logits_shape_value) != 3:
            raise ValueError("teacher logits shape must be [C,H,W]")
        logits_shape = tuple(int(value) for value in logits_shape_value)
        expected_logits_shape = (
            NUM_CLASSES,
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
        )
        if logits_shape != expected_logits_shape:
            raise RuntimeError(
                f"teacher logits/bounds mismatch for {sample_name}: "
                f"{logits_shape} != {expected_logits_shape}"
            )
        mask_path, mask_shape, mask_bounds = _mask_file_and_shape(
            structure_root, structure_record
        )
        mask_descriptor = structure_record.get("mask")
        if not isinstance(mask_descriptor, Mapping):
            mask_descriptor = structure_record.get("structure_mask")
        if not isinstance(mask_descriptor, Mapping):
            mask_descriptor = structure_record
        if str(mask_descriptor.get("dtype")) != "uint8":
            raise RuntimeError(f"structure cache must be uint8: {sample_name}")
        mask_array_sha = mask_descriptor.get("array_sha256")
        if not isinstance(mask_array_sha, str) or not mask_array_sha:
            raise RuntimeError(f"structure cache lacks array SHA: {sample_name}")
        result.append(
            PhaseImageRecord(
                index=index,
                sample_name=sample_name,
                optical_path=source_by_name[key][0],
                sar_path=source_by_name[key][1],
                label_path=source_by_name[key][2],
                full_shape_hw=full_shape,
                label_sha256=label_sha,
                bounds_yxyx=bounds,
                teacher_logits_path=logits_path,
                teacher_logits_shape=logits_shape,
                teacher_logits_array_sha256=(
                    str(logits.get("array_sha256"))
                    if logits.get("array_sha256") is not None
                    else None
                ),
                structure_mask_path=mask_path,
                structure_mask_shape=mask_shape,
                structure_mask_bounds_yxyx=mask_bounds,
                structure_mask_array_sha256=mask_array_sha,
            )
        )
    return result, {
        "scope": teacher.get("scope"),
        "split": teacher.get("split"),
        "image_count": len(result),
        "teacher_manifest": str(teacher_manifest_path.resolve()),
        "teacher_manifest_sha256": file_sha256(teacher_manifest_path),
        "teacher_protocol_sha256": canonical_json_sha256(protocol),
        "structure_manifest": str(structure_manifest_path.resolve()),
        "structure_manifest_sha256": file_sha256(structure_manifest_path),
    }


class SmallArrayCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("array cache capacity must be positive")
        self.capacity = int(capacity)
        self._values: OrderedDict[Path, np.ndarray] = OrderedDict()

    def get(self, path: Path, *, mmap: bool = False) -> np.ndarray:
        path = path.resolve()
        if path in self._values:
            self._values.move_to_end(path)
            return self._values[path]
        if path.suffix.lower() != ".npy":
            raise ValueError(f"immutable cache arrays must be .npy: {path}")
        value = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        self._values[path] = value
        if len(self._values) > self.capacity:
            self._values.popitem(last=False)
        return value


def decode_whu_label(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    decoded = raw.astype(np.int32) // 10 - 1
    decoded[decoded == -1] = IGNORE_INDEX
    return np.ascontiguousarray(decoded, dtype=np.int64)


def label_sha256(label: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(label).tobytes()).hexdigest()


def normalize_common_optical(optical_hwc: np.ndarray) -> np.ndarray:
    """Match WHU_Dataset's released ``normalize_type='common'`` conversion."""

    optical = np.asarray(optical_hwc)
    if optical.ndim != 3 or optical.shape[2] < 3:
        raise ValueError("optical input must be HWC with at least three channels")
    if optical.dtype != np.uint8:
        raise TypeError(f"released WHU optical conversion requires uint8, got {optical.dtype}")
    tensor = TVF.to_tensor(np.ascontiguousarray(optical[:, :, :3]))
    tensor = TVF.normalize(
        tensor,
        COMMON_MEAN[:, 0, 0].tolist(),
        COMMON_STD[:, 0, 0].tolist(),
    )
    return np.ascontiguousarray(tensor.numpy())


def exact_spatial_crop(
    array: np.ndarray, y: int, x: int, size: int = CROP_SIZE
) -> np.ndarray:
    """Copy an in-bounds spatial crop; padding or truncated crops are forbidden."""

    value = np.asarray(array)
    if value.ndim < 2 or y < 0 or x < 0 or size <= 0:
        raise ValueError("invalid spatial crop request")
    if y + size > value.shape[0] or x + size > value.shape[1]:
        raise ValueError("spatial crop would require padding")
    crop = np.asarray(value[y : y + size, x : x + size, ...]).copy()
    if crop.shape[:2] != (size, size):
        raise AssertionError("exact crop returned a truncated spatial shape")
    return crop


def crop_encoded_structure_mask(
    encoded: np.ndarray,
    bounds: tuple[int, int, int, int],
    y: int,
    x: int,
    size: int = CROP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    y0, y1, x0, x1 = bounds
    if y < y0 or x < x0 or y + size > y1 or x + size > x1:
        raise ValueError("requested crop lies outside structure-mask bounds")
    local_y = y - y0
    local_x = x - x0
    crop = np.asarray(encoded[local_y : local_y + size, local_x : local_x + size])
    if crop.shape != (size, size):
        raise ValueError("structure mask crop has the wrong shape")
    crop = crop.astype(np.uint8, copy=False)
    return (crop & np.uint8(1)) != 0, (crop & np.uint8(2)) != 0


class WHUPhaseCropDataset(Dataset):
    """Deterministic per-epoch random 512 crops in exact full-image coordinates."""

    def __init__(
        self,
        records: Sequence[PhaseImageRecord],
        *,
        seed: int,
        length: int = DEFAULT_TRAIN_SAMPLES_PER_EPOCH,
        source_cache_size: int = DEFAULT_SOURCE_CACHE_SIZE,
    ):
        if not records:
            raise ValueError("phase dataset requires at least one WHU image")
        if length <= 0:
            raise ValueError("phase dataset length must be positive")
        self.records = tuple(records)
        self.seed = int(seed)
        self.length = int(length)
        self.epoch = 0
        # Capacity is measured in full WHU images per modality.  Keeping three
        # independent LRUs prevents RGB/SAR/label entries from evicting one
        # another and degenerating to roughly capacity/3 source images.
        self._source_caches = {
            "optical": SmallArrayCache(source_cache_size),
            "sar": SmallArrayCache(source_cache_size),
            "label": SmallArrayCache(source_cache_size),
        }
        self._cache_arrays = SmallArrayCache(max(source_cache_size, 16))
        self._verified_labels: set[int] = set()
        self._verified_teacher_arrays: set[int] = set()
        self._verified_structure_arrays: set[int] = set()

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.length

    def _rng(self, index: int) -> random.Random:
        seed = self.seed + self.epoch * 1_000_003 + int(index) * 9_973
        return random.Random(seed)

    @staticmethod
    def _read_source(path: Path) -> np.ndarray:
        value = imread(path)
        return np.asarray(value)

    def _source(self, kind: str, path: Path) -> np.ndarray:
        # TIFF sources are not .npy, so retain a small explicit LRU here.
        source_cache = self._source_caches[kind]
        cache = source_cache._values
        resolved = path.resolve()
        if resolved in cache:
            cache.move_to_end(resolved)
            return cache[resolved]
        value = self._read_source(resolved)
        cache[resolved] = value
        if len(cache) > source_cache.capacity:
            cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= self.length:
            raise IndexError(index)
        record = self.records[index % len(self.records)]
        rng = self._rng(index)
        y0, y1, x0, x1 = record.bounds_yxyx
        y = rng.randint(y0, y1 - CROP_SIZE)
        x = rng.randint(x0, x1 - CROP_SIZE)

        optical_full = self._source("optical", record.optical_path)
        sar_full = self._source("sar", record.sar_path)
        raw_label_full = self._source("label", record.label_path)
        if tuple(optical_full.shape[:2]) != record.full_shape_hw:
            raise RuntimeError(f"optical shape mismatch: {record.sample_name}")
        if tuple(sar_full.shape[:2]) != record.full_shape_hw:
            raise RuntimeError(f"SAR shape mismatch: {record.sample_name}")
        if tuple(raw_label_full.shape[:2]) != record.full_shape_hw:
            raise RuntimeError(f"label shape mismatch: {record.sample_name}")
        if optical_full.dtype != np.uint8 or sar_full.dtype != np.uint8:
            raise TypeError(
                f"released WHU conversion requires uint8 RGB/SAR: {record.sample_name}"
            )
        if record.index not in self._verified_labels:
            full_decoded_label = decode_whu_label(raw_label_full)
            if label_sha256(full_decoded_label) != record.label_sha256:
                raise RuntimeError(f"full label hash mismatch: {record.sample_name}")
            self._verified_labels.add(record.index)
            del full_decoded_label

        teacher = self._cache_arrays.get(record.teacher_logits_path, mmap=True)
        if tuple(teacher.shape) != record.teacher_logits_shape:
            raise RuntimeError(f"teacher array shape mismatch: {record.sample_name}")
        if teacher.dtype != np.float16:
            raise RuntimeError(f"teacher array dtype mismatch: {record.sample_name}")
        if record.index not in self._verified_teacher_arrays:
            if not record.teacher_logits_array_sha256:
                raise RuntimeError(f"teacher array SHA missing: {record.sample_name}")
            if numpy_array_sha256(teacher) != record.teacher_logits_array_sha256:
                raise RuntimeError(f"teacher array SHA mismatch: {record.sample_name}")
            self._verified_teacher_arrays.add(record.index)
        teacher_local_y = y - y0
        teacher_local_x = x - x0
        teacher_crop = np.asarray(
            teacher[
                :,
                teacher_local_y : teacher_local_y + CROP_SIZE,
                teacher_local_x : teacher_local_x + CROP_SIZE,
            ],
            dtype=np.float32,
        ).copy()
        if teacher_crop.shape != (NUM_CLASSES, CROP_SIZE, CROP_SIZE):
            raise RuntimeError(f"teacher crop shape mismatch: {record.sample_name}")

        encoded_mask = self._cache_arrays.get(record.structure_mask_path, mmap=True)
        if tuple(encoded_mask.shape) != record.structure_mask_shape:
            raise RuntimeError(f"structure array shape mismatch: {record.sample_name}")
        if encoded_mask.dtype != np.uint8:
            raise RuntimeError(f"structure array dtype mismatch: {record.sample_name}")
        if record.index not in self._verified_structure_arrays:
            if numpy_array_sha256(encoded_mask) != record.structure_mask_array_sha256:
                raise RuntimeError(f"structure array SHA mismatch: {record.sample_name}")
            self._verified_structure_arrays.add(record.index)
        small, thin = crop_encoded_structure_mask(
            encoded_mask, record.structure_mask_bounds_yxyx, y, x
        )

        optical_crop = exact_spatial_crop(optical_full, y, x)[:, :, :3]
        sar_crop = exact_spatial_crop(sar_full, y, x)
        raw_label_crop = exact_spatial_crop(raw_label_full, y, x)
        label_crop = decode_whu_label(raw_label_crop)
        if optical_crop.shape != (CROP_SIZE, CROP_SIZE, 3):
            raise RuntimeError("no-padding optical crop contract failed")
        if sar_crop.shape[:2] != (CROP_SIZE, CROP_SIZE):
            raise RuntimeError("no-padding SAR crop contract failed")

        # Do not reflect the source crop: flip(T(I)) is not guaranteed to equal
        # T(flip(I)) for the phase-sensitive sealed teacher.  The first screen
        # therefore uses only random physical crop locations.
        flip_h = False
        flip_v = False
        optical_crop, sar_crop, label_crop, small, thin = (
            np.ascontiguousarray(value)
            for value in (optical_crop, sar_crop, label_crop, small, thin)
        )
        teacher_crop = np.ascontiguousarray(teacher_crop)

        optical_chw = normalize_common_optical(optical_crop)
        if sar_crop.ndim == 2:
            sar_chw = sar_crop[None]
        elif sar_crop.ndim == 3:
            sar_chw = sar_crop.transpose(2, 0, 1)
        else:
            raise RuntimeError(f"unsupported SAR rank: {sar_crop.ndim}")
        sar_chw = sar_chw.astype(np.float32) / 255.0
        return {
            "optical": torch.from_numpy(np.ascontiguousarray(optical_chw)),
            "sar": torch.from_numpy(np.ascontiguousarray(sar_chw)),
            "label": torch.from_numpy(label_crop.astype(np.int64, copy=False)),
            "teacher_logits": torch.from_numpy(teacher_crop),
            "small_mask": torch.from_numpy(small.astype(bool, copy=False)),
            "thin_mask": torch.from_numpy(thin.astype(bool, copy=False)),
            "image_index": record.index,
            "sample_name": record.sample_name,
            "crop_y": y,
            "crop_x": x,
            "flip_h": flip_h,
            "flip_v": flip_v,
        }


def build_e0(args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any]]:
    set_seed(args.seed)
    cfg = get_cfg(
        "DINOv3",
        "WHU",
        num_modalities=2,
        use_lora=False,
        r=3,
        backbone_type=BACKBONE_TYPE,
        use_naf=False,
    )
    model = cfg["model"]
    model.load_state_dict(load_baseline_state(args.baseline_checkpoint), strict=True)
    model.requires_grad_(False)
    model.eval()
    cfg["optimizer"] = None
    cfg["scheduler"] = None
    return model, cfg


class PairedEvaluationModel(nn.Module):
    """Return E0, R0, and R1 logits while sharing one frozen E0 forward."""

    def __init__(
        self,
        extractor: FrozenE0P2Extractor,
        r0_branch: PhaseCorrectionBranch,
        r1_branch: PhaseCorrectionBranch,
    ):
        super().__init__()
        self.extractor = extractor
        self.r0_branch = r0_branch
        self.r1_branch = r1_branch

    def train(self, mode: bool = True):
        super().train(False)
        self.extractor.eval()
        self.r0_branch.train(mode)
        self.r1_branch.train(mode)
        return self

    def forward(self, *modalities: torch.Tensor) -> torch.Tensor:
        base_logits, p2 = self.extractor(*modalities)
        r0_delta = F.interpolate(
            self.r0_branch(p2),
            size=base_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        r1_delta = F.interpolate(
            self.r1_branch(p2),
            size=base_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.cat(
            (base_logits, base_logits + r0_delta, base_logits + r1_delta), dim=1
        )


def build_optimizers(
    r0_branch: nn.Module,
    r1_branch: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    torch.optim.lr_scheduler.CosineAnnealingLR,
]:
    optimizer0 = torch.optim.AdamW(
        r0_branch.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    optimizer1 = torch.optim.AdamW(
        r1_branch.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer0, T_max=PROTOCOL_EPOCHS, eta_min=1e-7
    )
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer1, T_max=PROTOCOL_EPOCHS, eta_min=1e-7
    )
    return optimizer0, optimizer1, scheduler0, scheduler1


def correction_logits(
    base_logits: torch.Tensor,
    p2: torch.Tensor,
    branch: nn.Module,
) -> torch.Tensor:
    delta = branch(p2)
    if delta.shape[-2:] != base_logits.shape[-2:]:
        delta = F.interpolate(
            delta,
            size=base_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return base_logits + delta


def count_mask_statistics(
    counters: dict[str, Any],
    *,
    labels: torch.Tensor,
    base_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    gain: torch.Tensor,
    small: torch.Tensor,
    thin: torch.Tensor,
) -> None:
    valid = (labels >= 0) & (labels < NUM_CLASSES)
    teacher_correct = teacher_logits.argmax(dim=1).eq(labels) & valid
    base_correct = base_logits.argmax(dim=1).eq(labels) & valid
    protected = small | thin
    counters["valid"] += int(valid.sum())
    counters["gain"] += int((gain & valid).sum())
    counters["gain_small"] += int((gain & small & valid).sum())
    counters["gain_thin"] += int((gain & thin & valid).sum())
    counters["gain_small_or_thin"] += int((gain & protected & valid).sum())
    counters["small"] += int((small & valid).sum())
    counters["thin"] += int((thin & valid).sum())
    counters["small_or_thin"] += int((protected & valid).sum())
    quadrants = {
        "teacher_correct_base_wrong": teacher_correct & ~base_correct,
        "both_correct": teacher_correct & base_correct,
        "teacher_wrong_base_correct": ~teacher_correct & base_correct & valid,
        "both_wrong": ~teacher_correct & ~base_correct & valid,
    }
    for name, mask in quadrants.items():
        counters["quadrants"][name] += int(mask.sum())
        counters["gain_quadrants"][name] += int((mask & gain).sum())
    for class_index in range(NUM_CLASSES):
        class_mask = valid & labels.eq(class_index)
        counters["per_class_valid"][class_index] += int(class_mask.sum())
        counters["per_class_gain"][class_index] += int((class_mask & gain).sum())


def empty_mask_counters() -> dict[str, Any]:
    return {
        "valid": 0,
        "gain": 0,
        "gain_small": 0,
        "gain_thin": 0,
        "gain_small_or_thin": 0,
        "small": 0,
        "thin": 0,
        "small_or_thin": 0,
        "quadrants": {
            "teacher_correct_base_wrong": 0,
            "both_correct": 0,
            "teacher_wrong_base_correct": 0,
            "both_wrong": 0,
        },
        "gain_quadrants": {
            "teacher_correct_base_wrong": 0,
            "both_correct": 0,
            "teacher_wrong_base_correct": 0,
            "both_wrong": 0,
        },
        "per_class_valid": [0] * NUM_CLASSES,
        "per_class_gain": [0] * NUM_CLASSES,
    }


def ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def serialize_mask_counters(counters: Mapping[str, Any], labels: Sequence[str]) -> dict:
    return {
        "valid_pixels": counters["valid"],
        "gain_pixels": counters["gain"],
        "gain_coverage": ratio(counters["gain"], counters["valid"]),
        "small_pixels": counters["small"],
        "thin_pixels": counters["thin"],
        "small_or_thin_pixels": counters["small_or_thin"],
        "gain_on_small_pixels": counters["gain_small"],
        "gain_on_thin_pixels": counters["gain_thin"],
        "gain_on_small_or_thin_pixels": counters["gain_small_or_thin"],
        "gain_coverage_within_small": ratio(counters["gain_small"], counters["small"]),
        "gain_coverage_within_thin": ratio(counters["gain_thin"], counters["thin"]),
        "gain_coverage_within_small_or_thin": ratio(
            counters["gain_small_or_thin"], counters["small_or_thin"]
        ),
        "per_class": {
            str(labels[index]): {
                "valid_pixels": counters["per_class_valid"][index],
                "gain_pixels": counters["per_class_gain"][index],
                "gain_coverage": ratio(
                    counters["per_class_gain"][index],
                    counters["per_class_valid"][index],
                ),
            }
            for index in range(NUM_CLASSES)
        },
        "correctness_quadrants": {
            name: {
                "pixels": value,
                "fraction_of_valid": ratio(value, counters["valid"]),
                "gain_pixels": counters["gain_quadrants"][name],
                "gain_fraction_within_quadrant": ratio(
                    counters["gain_quadrants"][name], value
                ),
            }
            for name, value in counters["quadrants"].items()
        },
    }


def metrics_from_confusion(
    confusion: np.ndarray, labels: Sequence[str]
) -> dict[str, Any]:
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError("confusion must be 7x7")
    total = int(confusion.sum())
    diagonal = np.diag(confusion).astype(np.float64)
    row = confusion.sum(axis=1).astype(np.float64)
    column = confusion.sum(axis=0).astype(np.float64)
    ious = class_ious_from_confusion(confusion)
    f1 = np.zeros(NUM_CLASSES, dtype=np.float64)
    np.divide(2.0 * diagonal, row + column, out=f1, where=(row + column) > 0)
    accuracy = float(diagonal.sum() * 100.0 / total) if total else float("nan")
    pa = float(diagonal.sum() / total) if total else float("nan")
    pe = (
        float(np.sum(row * column) / float(total * total))
        if total
        else float("nan")
    )
    kappa = float((pa - pe) / (1.0 - pe)) if total and pe != 1.0 else float("nan")
    return {
        "confusion": confusion.tolist(),
        "valid_pixels": total,
        "errors": int(total - diagonal.sum()),
        "MIoU": mean_iou_from_confusion(confusion),
        "miou_percent": mean_iou_from_confusion(confusion) * 100.0,
        "F1": float(np.nanmean(f1)),
        "Kappa": kappa,
        "Acc": accuracy,
        "class_iou_percent": {
            str(name): float(ious[index] * 100.0)
            for index, name in enumerate(labels)
        },
    }


def empty_evaluation_accumulator() -> dict[str, Any]:
    return {
        "confusion": np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64),
        "prediction_digest": hashlib.sha256(),
        "regions": {
            name: {"pixels": 0, "errors": 0} for name in REGION_NAMES
        },
    }


def add_prediction_to_evaluation(
    accumulator: dict[str, Any],
    prediction: np.ndarray,
    target: np.ndarray,
    region_masks: Mapping[str, np.ndarray],
) -> None:
    prediction = np.ascontiguousarray(prediction, dtype=np.int64)
    target = np.ascontiguousarray(target, dtype=np.int64)
    accumulator["prediction_digest"].update(prediction.tobytes())
    accumulator["confusion"] += confusion_from_arrays(
        prediction, target, NUM_CLASSES
    )
    valid = (target >= 0) & (target < NUM_CLASSES)
    errors = valid & prediction.__ne__(target)
    for name in REGION_NAMES:
        mask = np.asarray(region_masks[name], dtype=bool) & valid
        accumulator["regions"][name]["pixels"] += int(mask.sum())
        accumulator["regions"][name]["errors"] += int((errors & mask).sum())


def serialize_evaluation_accumulator(
    accumulator: Mapping[str, Any], labels: Sequence[str]
) -> dict[str, Any]:
    summary = metrics_from_confusion(accumulator["confusion"], labels)
    summary["prediction_sha256"] = accumulator["prediction_digest"].hexdigest()
    summary["regions"] = {
        name: {
            "pixels": values["pixels"],
            "errors": values["errors"],
            "error_rate": ratio(values["errors"], values["pixels"]),
            "error_rate_percent": (
                ratio(values["errors"], values["pixels"]) * 100.0
                if values["pixels"]
                else None
            ),
        }
        for name, values in accumulator["regions"].items()
    }
    return summary


def paired_differences(
    e0: Mapping[str, Any],
    r0: Mapping[str, Any],
    r1: Mapping[str, Any],
    labels: Sequence[str],
) -> dict[str, Any]:
    region_differences = {}
    for name in REGION_NAMES:
        r0_rate = r0["regions"][name]["error_rate_percent"]
        r1_rate = r1["regions"][name]["error_rate_percent"]
        e0_rate = e0["regions"][name]["error_rate_percent"]
        region_differences[name] = {
            "r1_minus_r0_error_rate_pp": (
                r1_rate - r0_rate
                if r1_rate is not None and r0_rate is not None
                else None
            ),
            "r1_minus_e0_error_rate_pp": (
                r1_rate - e0_rate
                if r1_rate is not None and e0_rate is not None
                else None
            ),
            "r0_minus_e0_error_rate_pp": (
                r0_rate - e0_rate
                if r0_rate is not None and e0_rate is not None
                else None
            ),
        }
    return {
        "r1_minus_r0_miou_pp": float(r1["miou_percent"] - r0["miou_percent"]),
        "r1_minus_e0_miou_pp": float(r1["miou_percent"] - e0["miou_percent"]),
        "r0_minus_e0_miou_pp": float(r0["miou_percent"] - e0["miou_percent"]),
        "class_r1_minus_r0_iou_pp": {
            str(name): float(
                r1["class_iou_percent"][str(name)]
                - r0["class_iou_percent"][str(name)]
            )
            for name in labels
        },
        "regions": region_differences,
    }


def stage_decision(epoch: int, differences: Mapping[str, Any]) -> dict[str, Any]:
    paired = float(differences["r1_minus_r0_miou_pp"])
    absolute = float(differences["r1_minus_e0_miou_pp"])
    small_delta = differences["regions"]["component_area_le_256px2"][
        "r1_minus_r0_error_rate_pp"
    ]
    thin_delta = differences["regions"]["component_thickness_le_4px"][
        "r1_minus_r0_error_rate_pp"
    ]
    small_thin_guard = (
        small_delta is not None
        and thin_delta is not None
        and float(small_delta) <= 0.0
        and float(thin_delta) <= 0.0
    )
    if epoch == 5:
        outcome = "EARLY_TREND_ONLY"
        explanation = (
            "E5 verifies learning behavior only; it cannot independently reject the route."
        )
    elif epoch == 10:
        if paired >= 0.05 and absolute >= 0.05:
            outcome = "CONTINUE_TO_E15"
            explanation = "Both the paired 0.05pp trend gate and absolute E0 gate pass."
        else:
            outcome = "PAUSE_FOR_REVIEW"
            explanation = (
                "At least one E10 gate misses; inspect loss/mask/gradient trends before "
                "deciding whether to spend E11-E15."
            )
    elif epoch == 15:
        if paired >= 0.10 and absolute >= 0.10 and small_thin_guard:
            outcome = "GO"
            explanation = (
                "Both 0.10pp mIoU gates pass and neither small nor thin error "
                "rate worsens relative to R0."
            )
        elif paired >= 0.05 and absolute >= 0.05:
            outcome = "GRAY"
            explanation = (
                "The route beats E0 and has a paired signal, but misses the 0.10pp GO gate."
            )
        else:
            outcome = "NO_GO"
            explanation = "The final paired/absolute dual gate does not pass."
    else:
        outcome = "UNSCHEDULED_EVALUATION"
        explanation = "No prospective decision is defined outside E5/E10/E15."
    return {
        "epoch": int(epoch),
        "outcome": outcome,
        "interpretation_scope": (
            "Exploratory internal resource decision only. The official test split "
            "helped form this route, so GO is not an unbiased paper-level confirmation."
        ),
        "paired_gate": {
            "metric": "R1-R0 mIoU",
            "actual_pp": paired,
            "e10_threshold_pp": 0.05,
            "e15_go_threshold_pp": 0.10,
        },
        "absolute_gate": {
            "metric": "R1-E0 mIoU",
            "actual_pp": absolute,
            "e10_threshold_pp": 0.05,
            "e15_go_threshold_pp": 0.10,
        },
        "small_thin_guard": {
            "passes": small_thin_guard,
            "rule": "R1-R0 error-rate delta <=0.0pp for both regions",
            "small_r1_minus_r0_error_rate_pp": small_delta,
            "thin_r1_minus_r0_error_rate_pp": thin_delta,
        },
        "explanation": explanation,
    }


def evaluate_paired(
    model: PairedEvaluationModel,
    cfg: Mapping[str, Any],
    loader: DataLoader,
    *,
    device: torch.device,
    inference_batch_size: int,
    phase_reference: Mapping[str, Any],
) -> dict[str, Any]:
    model.eval()
    model.extractor.eval()
    accumulators = {
        "e0": empty_evaluation_accumulator(),
        "r0": empty_evaluation_accumulator(),
        "r1": empty_evaluation_accumulator(),
    }
    label_digest = hashlib.sha256()
    stride = tuple(int(size * 2 / 3) for size in cfg["window_size"])
    if stride != INFERENCE_STRIDE:
        raise RuntimeError(f"formal evaluation stride changed: {stride}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.no_grad():
        for optical, sar, label in tqdm(loader, desc="paired single-phase eval"):
            optical = optical.to(device)
            sar = sar.to(device)
            scores = slide_inference(
                optical,
                model,
                dsm=sar,
                n_output_channels=NUM_CLASSES * 3,
                crop_size=cfg["window_size"],
                stride=stride,
                batch_size=inference_batch_size,
            )
            e0_scores, r0_scores, r1_scores = scores.split(NUM_CLASSES, dim=1)
            target = np.ascontiguousarray(label[0].numpy().astype(np.int64, copy=False))
            label_digest.update(target.tobytes())
            region_masks, _ = build_spatial_region_masks(
                target,
                NUM_CLASSES,
                boundary_radii=(0, 1, 2, 4, 8),
                component_area_thresholds=(256, 1024, 4096),
                component_thickness_thresholds=(4, 8, 16),
                patch_size=16,
                union_boundary_radius=0,
                union_component_area=256,
                union_component_thickness=4,
            )
            for name, variant_scores in (
                ("e0", e0_scores),
                ("r0", r0_scores),
                ("r1", r1_scores),
            ):
                prediction = np.ascontiguousarray(
                    variant_scores.argmax(dim=1)[0].numpy().astype(np.int64, copy=False)
                )
                add_prediction_to_evaluation(
                    accumulators[name], prediction, target, region_masks
                )
            del optical, sar, scores, e0_scores, r0_scores, r1_scores
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    results = {
        name: serialize_evaluation_accumulator(accumulator, cfg["labels"])
        for name, accumulator in accumulators.items()
    }
    actual_label_hash = label_digest.hexdigest()
    e0_validation = {
        "miou_percent_actual": results["e0"]["miou_percent"],
        "miou_percent_expected": phase_reference["expected_e0_miou_percent"],
        "miou_difference_pp": float(
            results["e0"]["miou_percent"]
            - phase_reference["expected_e0_miou_percent"]
        ),
        "miou_within_1e-8pp": math.isclose(
            results["e0"]["miou_percent"],
            phase_reference["expected_e0_miou_percent"],
            rel_tol=0.0,
            abs_tol=1e-8,
        ),
        "confusion_equal": (
            results["e0"]["confusion"] == phase_reference["expected_e0_confusion"]
        ),
        "prediction_sha256_equal": (
            results["e0"]["prediction_sha256"]
            == phase_reference["expected_prediction_sha256"]
        ),
        "label_sha256_equal": (
            actual_label_hash == phase_reference["expected_label_sha256"]
        ),
        "actual_label_sha256": actual_label_hash,
    }
    if not all(
        e0_validation[key]
        for key in (
            "miou_within_1e-8pp",
            "confusion_equal",
            "prediction_sha256_equal",
            "label_sha256_equal",
        )
    ):
        raise AssertionError(f"paired evaluation failed sealed E0: {e0_validation}")
    differences = paired_differences(
        results["e0"], results["r0"], results["r1"], cfg["labels"]
    )
    return {
        "num_images": len(loader.dataset),
        "e0_validation": e0_validation,
        **results,
        "differences": differences,
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        },
    }


def batch_input_hashes(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "optical_sha256": tensor_sha256(batch["optical"]),
        "sar_sha256": tensor_sha256(batch["sar"]),
        "label_sha256": tensor_sha256(batch["label"]),
        "teacher_logits_sha256": tensor_sha256(batch["teacher_logits"]),
        "small_mask_sha256": tensor_sha256(batch["small_mask"]),
        "thin_mask_sha256": tensor_sha256(batch["thin_mask"]),
        "image_indices": batch["image_index"].tolist(),
        "sample_names": list(batch["sample_name"]),
        "crop_y": batch["crop_y"].tolist(),
        "crop_x": batch["crop_x"].tolist(),
        "flip_h": batch["flip_h"].tolist(),
        "flip_v": batch["flip_v"].tolist(),
    }


def prepare_training_batch(
    batch: Mapping[str, Any],
    *,
    extractor: FrozenE0P2Extractor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    optical = batch["optical"].to(device, non_blocking=False)
    sar = batch["sar"].to(device, non_blocking=False)
    labels = batch["label"].to(device, non_blocking=False)
    teacher_logits = batch["teacher_logits"].to(
        device, dtype=torch.float32, non_blocking=False
    ).detach()
    small = batch["small_mask"].to(device, dtype=torch.bool)
    thin = batch["thin_mask"].to(device, dtype=torch.bool)

    extractor.eval()
    base_logits, p2 = extractor(optical, sar)
    if base_logits.requires_grad or p2.requires_grad:
        raise AssertionError("frozen E0 returned a grad-enabled tensor")
    if teacher_logits.shape != base_logits.shape:
        raise RuntimeError(
            f"teacher/base shapes differ: {teacher_logits.shape} != {base_logits.shape}"
        )
    valid = (labels >= 0) & (labels < NUM_CLASSES)
    gain = teacher_gain_mask(
        teacher_logits,
        base_logits,
        labels,
        delta=GAIN_MARGIN,
        valid_mask=valid,
        ignore_index=IGNORE_INDEX,
    )
    kd_mask = gain & valid
    del optical, sar
    return {
        "labels": labels,
        "teacher_logits": teacher_logits,
        "small": small,
        "thin": thin,
        "base_logits": base_logits,
        "p2": p2,
        "valid": valid,
        "gain": gain,
        "kd_mask": kd_mask,
    }


def optimize_prepared_batch(
    prepared: Mapping[str, torch.Tensor],
    *,
    extractor: FrozenE0P2Extractor,
    r0_branch: nn.Module,
    r1_branch: nn.Module,
    optimizer0: torch.optim.Optimizer,
    optimizer1: torch.optim.Optimizer,
    supervised_loss: nn.Module,
    counters: dict[str, Any],
    count_statistics: bool,
) -> dict[str, float | int]:
    labels = prepared["labels"]
    teacher_logits = prepared["teacher_logits"]
    small = prepared["small"]
    thin = prepared["thin"]
    base_logits = prepared["base_logits"]
    p2 = prepared["p2"]
    valid = prepared["valid"]
    gain = prepared["gain"]
    kd_mask = prepared["kd_mask"]
    if count_statistics:
        count_mask_statistics(
            counters,
            labels=labels,
            base_logits=base_logits,
            teacher_logits=teacher_logits,
            gain=gain,
            small=small,
            thin=thin,
        )

    r0_branch.train()
    r1_branch.train()
    optimizer0.zero_grad(set_to_none=True)
    optimizer1.zero_grad(set_to_none=True)
    r0_logits = correction_logits(base_logits, p2, r0_branch)
    r1_logits = correction_logits(base_logits, p2, r1_branch)
    r0_supervised = supervised_loss(r0_logits, labels)
    r1_supervised = supervised_loss(r1_logits, labels)
    kd_loss = masked_kl_divergence(
        r1_logits,
        teacher_logits,
        kd_mask,
        temperature=KD_TEMPERATURE,
    )
    r0_loss = r0_supervised
    r1_loss = r1_supervised + KD_WEIGHT * kd_loss
    total_loss = r0_loss + r1_loss
    if not torch.isfinite(total_loss):
        raise RuntimeError("non-finite paired training loss")
    total_loss.backward()
    if any(parameter.grad is not None for parameter in extractor.parameters()):
        raise AssertionError("frozen E0 accumulated gradients")
    for name, branch in (("R0", r0_branch), ("R1", r1_branch)):
        finite_gradients = [
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in branch.parameters()
        ]
        if not any(finite_gradients) or not all(
            ok for ok, parameter in zip(finite_gradients, branch.parameters()) if parameter.grad is not None
        ):
            raise AssertionError(f"{name} branch lacks finite gradients")
    optimizer0.step()
    optimizer1.step()
    result = {
        "r0_supervised": float(r0_supervised.detach()),
        "r1_supervised": float(r1_supervised.detach()),
        "kd": float(kd_loss.detach()),
        "r0_total": float(r0_loss.detach()),
        "r1_total": float(r1_loss.detach()),
        "gain_pixels": int(gain.sum()),
        "kd_pixels": int(kd_mask.sum()),
        "valid_pixels": int(valid.sum()),
    }
    del (
        r0_logits,
        r1_logits,
        r0_supervised,
        r1_supervised,
        kd_loss,
        r0_loss,
        r1_loss,
        total_loss,
    )
    return result


def train_one_batch(
    batch: Mapping[str, Any],
    *,
    extractor: FrozenE0P2Extractor,
    r0_branch: nn.Module,
    r1_branch: nn.Module,
    optimizer0: torch.optim.Optimizer,
    optimizer1: torch.optim.Optimizer,
    supervised_loss: nn.Module,
    device: torch.device,
    counters: dict[str, Any],
) -> dict[str, float | int]:
    prepared = prepare_training_batch(batch, extractor=extractor, device=device)
    result = optimize_prepared_batch(
        prepared,
        extractor=extractor,
        r0_branch=r0_branch,
        r1_branch=r1_branch,
        optimizer0=optimizer0,
        optimizer1=optimizer1,
        supervised_loss=supervised_loss,
        counters=counters,
        count_statistics=True,
    )
    del prepared
    return result


def average_step_metrics(total: Mapping[str, float], steps: int) -> dict[str, float]:
    if steps <= 0:
        raise ValueError("cannot average zero steps")
    return {name: float(value / steps) for name, value in total.items()}


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "loader_generator": loader_generator.get_state(),
    }


def restore_rng_state(state: Mapping[str, Any], loader_generator: torch.Generator) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "loader_generator"}
    if set(state) != required:
        raise RuntimeError(f"checkpoint RNG keys changed: {sorted(state)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    loader_generator.set_state(state["loader_generator"])


def save_paired_checkpoint(
    path: Path,
    *,
    epoch: int,
    r0_branch: nn.Module,
    r1_branch: nn.Module,
    optimizer0: torch.optim.Optimizer,
    optimizer1: torch.optim.Optimizer,
    scheduler0: torch.optim.lr_scheduler.LRScheduler,
    scheduler1: torch.optim.lr_scheduler.LRScheduler,
    loader_generator: torch.Generator,
    baseline_sha256: str,
    protocol: Mapping[str, Any],
    allow_replace_incomplete: bool = False,
) -> None:
    if path.exists() and not allow_replace_incomplete:
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    payload = {
        "format": "whu-phase-distillation-paired-v1",
        "epoch": int(epoch),
        "r0_branch": r0_branch.state_dict(),
        "r1_branch": r1_branch.state_dict(),
        "optimizer_r0": optimizer0.state_dict(),
        "optimizer_r1": optimizer1.state_dict(),
        "scheduler_r0": scheduler0.state_dict(),
        "scheduler_r1": scheduler1.state_dict(),
        "rng": capture_rng_state(loader_generator),
        "baseline_checkpoint_sha256": baseline_sha256,
        "protocol": dict(protocol),
        "protocol_sha256": canonical_json_sha256(protocol),
    }
    temporary = path.with_name(
        f"{path.name}.{time.time_ns()}.part"
    )
    torch.save(payload, temporary)
    temporary.replace(path)


def load_paired_checkpoint(
    path: Path,
    *,
    r0_branch: nn.Module,
    r1_branch: nn.Module,
    optimizer0: torch.optim.Optimizer,
    optimizer1: torch.optim.Optimizer,
    scheduler0: torch.optim.lr_scheduler.LRScheduler,
    scheduler1: torch.optim.lr_scheduler.LRScheduler,
    loader_generator: torch.Generator,
    baseline_sha256: str,
    protocol: Mapping[str, Any],
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("resume checkpoint must contain a mapping")
    if payload.get("format") != "whu-phase-distillation-paired-v1":
        raise RuntimeError("resume checkpoint format changed")
    if payload.get("baseline_checkpoint_sha256") != baseline_sha256:
        raise RuntimeError("resume checkpoint belongs to a different E0")
    expected_protocol_sha = canonical_json_sha256(protocol)
    if payload.get("protocol_sha256") != expected_protocol_sha:
        raise RuntimeError("resume checkpoint scientific protocol differs")
    if canonical_json_sha256(payload.get("protocol")) != expected_protocol_sha:
        raise RuntimeError("resume checkpoint protocol payload is internally inconsistent")
    r0_branch.load_state_dict(payload["r0_branch"], strict=True)
    r1_branch.load_state_dict(payload["r1_branch"], strict=True)
    optimizer0.load_state_dict(payload["optimizer_r0"])
    optimizer1.load_state_dict(payload["optimizer_r1"])
    scheduler0.load_state_dict(payload["scheduler_r0"])
    scheduler1.load_state_dict(payload["scheduler_r1"])
    restore_rng_state(payload["rng"], loader_generator)
    epoch = int(payload.get("epoch", 0))
    if epoch <= 0 or epoch > PROTOCOL_EPOCHS:
        raise RuntimeError(f"resume epoch is outside 1..{PROTOCOL_EPOCHS}: {epoch}")
    return epoch


def branch_metadata(branch: nn.Module) -> dict[str, Any]:
    parameters = list(branch.parameters())
    return {
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "parameter_tensors": len(parameters),
        "state_sha256": named_tensor_sha256(branch.state_dict().items()),
        "nonzero_parameters": int(
            sum(torch.count_nonzero(parameter.detach()).item() for parameter in parameters)
        ),
    }


def build_scientific_protocol(
    args: argparse.Namespace,
    *,
    baseline_sha256: str,
    phase_reference: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    initial_branch: Mapping[str, Any],
) -> dict[str, Any]:
    objective_mask_mode = (
        args.objective_mask if args.mode == "objective" else None
    )
    kd_mask_description = (
        "valid AND teacher argmax equals the ground-truth label AND frozen E0 "
        "argmax differs from the ground-truth label; gain is retained only for "
        "routing diagnostics"
        if objective_mask_mode == "correction"
        else (
            "gain AND valid; small/thin pixels remain eligible when the teacher "
            "has lower true-class CE"
        )
    )
    return {
        "name": "WHU paired single-phase phase-distillation V1",
        "execution_mode": args.mode,
        "objective_mask_mode": objective_mask_mode,
        "evidence_scope": (
            "Exploratory test-selected method screen for deciding whether the route "
            "deserves more resources; not unbiased paper-level evidence."
        ),
        "paper_confirmation_requirement": (
            "If adopted for a paper, rebuild a clean train/validation/test protocol "
            "and rerun E0, teacher generation, R0, and R1 from the beginning."
        ),
        "git_commit": git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": baseline_sha256,
        "phase_reference": dict(phase_reference),
        "immutable_manifests": dict(manifest_metadata),
        "dataset": {
            "split": (
                "all official 80 train images"
                if int(manifest_metadata["image_count"]) == 80
                else f"diagnostic subset of {int(manifest_metadata['image_count'])} train image(s)"
            ),
            "manifest_scope": manifest_metadata["scope"],
            "train_image_count": int(manifest_metadata["image_count"]),
            "samples_per_epoch": args.train_samples_per_epoch,
            "crop_size_hw": [CROP_SIZE, CROP_SIZE],
            "coordinate_system": "original full-image y/x",
            "sampling_region": "uniform crop origin wholly inside exact teacher bounds",
            "padding": "forbidden",
            "scale_augmentation": "disabled",
            "augmentation": (
                "disabled; random physical crop location only, so cached T(I) is not "
                "approximated by flip(T(I)) for a reflected input"
            ),
            "optical_normalization": {
                "type": "common/ImageNet",
                "mean": COMMON_MEAN[:, 0, 0].tolist(),
                "std": COMMON_STD[:, 0, 0].tolist(),
            },
            "sar_scaling": "float32 / 255",
            "batch_size": args.batch_size,
            "num_workers": 0,
            "seed": args.seed,
        },
        "teacher": {
            "phases_dy_dx": [list(shift) for shift in TEACHER_PHASES],
            "cached_logits": "detached arithmetic mean, float16 storage, float32 loss",
            "target_augmentation": "none",
            "temperature": KD_TEMPERATURE,
            "gain_mask": "teacher true-class CE < E0 true-class CE",
            "gain_margin": GAIN_MARGIN,
            "kd_mask": kd_mask_description,
            "empty_mask_loss": 0.0,
            "kl_normalization": "valid KD-mask pixels",
            "lambda": KD_WEIGHT,
        },
        "students": {
            "E0": "frozen once; eval for every forward; shared base logits and P2",
            "R0": (
                "combined diagnostic branch: released CE+Dice + masked phase KL"
                if args.mode == "objective"
                else "zero-initialized correction branch + released CE+Dice"
            ),
            "R1": (
                "KD-only diagnostic branch from the identical initialization"
                if args.mode == "objective"
                else "identical correction branch + released CE+Dice + masked phase KL"
            ),
            "comparison": (
                "KD-only capacity versus the unchanged combined objective"
                if args.mode == "objective"
                else "R1-R0, with a separate R1-E0 absolute gate"
            ),
            "initial_branch": dict(initial_branch),
            "mode_specific_arm_mapping": (
                {
                    "R0_slot": "combined CE+Dice+KD diagnostic arm",
                    "R1_slot": "KD-only capacity diagnostic arm",
                }
                if args.mode == "objective"
                else {
                    "R0_slot": "R0 supervised-only arm",
                    "R1_slot": "R1 supervised-plus-KD arm",
                }
            ),
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": "CosineAnnealingLR",
            "T_max": PROTOCOL_EPOCHS,
            "eta_min": 1e-7,
            "protocol_epochs": PROTOCOL_EPOCHS,
            "formal_evaluation_epochs": list(EVALUATION_EPOCHS),
            "checkpoint_frequency_epochs": 1,
        },
        "formal_evaluation": {
            "split": "official 20-image test",
            "mode": "single-phase count-normalized sliding inference",
            "crop_size_hw": [CROP_SIZE, CROP_SIZE],
            "stride_hw": [341, 341],
            "three_outputs_share_one_E0_forward": True,
            "inference_batch_size": args.inference_batch_size,
            "prospective_gates": {
                "E5": "trend only, never independent NO-GO",
                "E10": "R1-R0 >=0.05pp AND R1-E0 >=0.05pp to continue without review",
                "E15_GO": (
                    "R1-R0 >=0.10pp AND R1-E0 >=0.10pp AND neither small nor "
                    "thin error rate worsens relative to R0"
                ),
            },
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired WHU R0/R1 single-phase phase distillation"
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--phase-reference-json", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--structure-mask-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("formal", "smoke", "overfit", "objective"),
        default="formal",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=PROTOCOL_EPOCHS)
    parser.add_argument(
        "--stop-after-epoch", type=int, choices=EVALUATION_EPOCHS, default=5
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument(
        "--train-samples-per-epoch",
        type=int,
        default=DEFAULT_TRAIN_SAMPLES_PER_EPOCH,
    )
    parser.add_argument(
        "--source-cache-size",
        type=int,
        default=DEFAULT_SOURCE_CACHE_SIZE,
        help="full-image LRU capacity per modality (RGB, SAR, and label)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--overfit-steps", type=int, default=100)
    parser.add_argument(
        "--objective-mask",
        choices=("gain", "correction"),
        default="gain",
        help=(
            "KD routing for objective mode: the original lower-CE gain mask or "
            "only pixels where the teacher corrects a frozen-E0 error"
        ),
    )
    args = parser.parse_args(argv)

    for path in (
        args.baseline_checkpoint,
        args.phase_reference_json,
        args.teacher_manifest,
        args.structure_mask_manifest,
    ):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        parser.error(f"resume checkpoint does not exist: {args.resume_checkpoint}")
    if args.mode == "formal":
        if args.epochs != PROTOCOL_EPOCHS:
            parser.error("formal --epochs is fixed at 15; use --stop-after-epoch to stage")
        if args.resume_checkpoint is not None and not args.output_dir.is_dir():
            parser.error("resume requires the existing original --output-dir")
    else:
        if args.resume_checkpoint is not None:
            parser.error("resume is only defined for formal mode")
        if args.output_dir.exists():
            parser.error(f"refusing to reuse output directory: {args.output_dir}")
    if args.batch_size <= 0 or args.inference_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay non-negative")
    if args.hidden_channels <= 0 or args.train_samples_per_epoch <= 0:
        parser.error("hidden channels and train samples must be positive")
    if args.source_cache_size <= 0:
        parser.error("source cache size must be positive")
    if not 1 <= args.smoke_steps <= 5:
        parser.error("--smoke-steps must be within 1..5")
    if args.overfit_steps <= 0:
        parser.error("--overfit-steps must be positive")
    if args.mode == "objective" and args.overfit_steps != 100:
        parser.error("objective mode is pre-registered at exactly 100 steps")
    if args.mode != "objective" and args.objective_mask != "gain":
        parser.error("--objective-mask correction is only defined for objective mode")
    return args


def verify_initial_zero_pair(
    dataset: WHUPhaseCropDataset,
    *,
    extractor: FrozenE0P2Extractor,
    r0_branch: nn.Module,
    r1_branch: nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    from torch.utils.data._utils.collate import default_collate

    preview = default_collate(
        [dataset[index] for index in range(min(batch_size, len(dataset)))]
    )
    optical = preview["optical"].to(device)
    sar = preview["sar"].to(device)
    extractor.eval()
    r0_branch.eval()
    r1_branch.eval()
    with torch.no_grad():
        base_logits, p2 = extractor(optical, sar)
        delta0 = r0_branch(p2)
        delta1 = r1_branch(p2)
        if torch.count_nonzero(delta0).item() != 0:
            raise AssertionError("R0 is not exactly zero at initialization")
        if not torch.equal(delta0, delta1):
            raise AssertionError("R0/R1 initial correction outputs differ")
        r0_logits = correction_logits(base_logits, p2, r0_branch)
        r1_logits = correction_logits(base_logits, p2, r1_branch)
        if not torch.equal(base_logits, r0_logits) or not torch.equal(
            base_logits, r1_logits
        ):
            raise AssertionError("zero branches do not reproduce exact E0 logits")
    result = {
        "checked_batch_size": int(optical.shape[0]),
        "base_logits_shape": list(base_logits.shape),
        "p2_shape": list(p2.shape),
        "r0_delta_exact_zeros": True,
        "r1_delta_exact_zeros": True,
        "r0_r1_output_equal": True,
        "r0_e0_logits_exact": True,
        "r1_e0_logits_exact": True,
    }
    del optical, sar, base_logits, p2, delta0, delta1, r0_logits, r1_logits
    return result


def initialize_or_resume_output(
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
) -> None:
    config_path = args.output_dir / "run_config.json"
    protocol_sha = canonical_json_sha256(protocol)
    if args.resume_checkpoint is None:
        if not args.output_dir.exists():
            args.output_dir.mkdir(parents=True, exist_ok=False)
            write_json_atomic(
                config_path,
                {
                    "status": "CONFIGURED",
                    "mode": args.mode,
                    "protocol": dict(protocol),
                    "protocol_sha256": protocol_sha,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
            )
            return
        if args.mode != "formal" or not args.output_dir.is_dir():
            raise RuntimeError(f"refusing to reuse output directory: {args.output_dir}")
        if not config_path.is_file():
            raise RuntimeError("incomplete formal recovery lacks run_config.json")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != protocol_sha or canonical_json_sha256(
            existing.get("protocol")
        ) != protocol_sha:
            raise RuntimeError("incomplete formal recovery protocol differs")
        completed = committed_epochs(args.output_dir)
        if completed:
            raise RuntimeError(
                f"formal output already has committed epoch {completed[-1]}; "
                "resume from its sealed checkpoint"
            )
        # No epoch was ever published.  Recreate the deterministic initial state
        # and allow epoch 1 to overwrite only uncommitted generated artifacts.
        return
    if args.resume_checkpoint.resolve().parent != args.output_dir.resolve():
        raise RuntimeError("resume checkpoint must be inside the original output directory")
    if not config_path.is_file():
        raise RuntimeError("resume output directory lacks run_config.json")
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    if existing.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("resume run_config protocol differs")
    if canonical_json_sha256(existing.get("protocol")) != protocol_sha:
        raise RuntimeError("resume run_config is internally inconsistent")


def validate_resume_commit(
    output_dir: Path,
    epoch: int,
    checkpoint_path: Path,
) -> None:
    """Require resume to start at the latest fully committed epoch."""

    verify_epoch_commit(output_dir, epoch)
    expected_checkpoint = epoch_artifact_paths(output_dir, epoch)["checkpoint"]
    if checkpoint_path.resolve() != expected_checkpoint.resolve():
        raise RuntimeError(
            f"resume checkpoint is not the sealed epoch-{epoch} checkpoint"
        )
    completed = committed_epochs(output_dir)
    if not completed or completed[-1] != epoch:
        raise RuntimeError(
            f"resume must use latest committed epoch {completed[-1] if completed else None}"
        )
    rebuild_metric_indexes(output_dir)


def checkpoint_epoch_from_path(path: Path) -> int:
    name = path.name
    if not name.startswith("checkpoint_e") or not name.endswith(".pth"):
        raise RuntimeError(f"unexpected resume checkpoint name: {name}")
    value = name.removeprefix("checkpoint_e").removesuffix(".pth")
    if not value.isdigit():
        raise RuntimeError(f"unexpected resume checkpoint name: {name}")
    return int(value)


def objective_gradient_diagnostics(
    branch: nn.Module,
    prepared: Mapping[str, torch.Tensor],
    supervised_loss: nn.Module,
) -> dict[str, Any]:
    """Measure supervision/KD gradient scale and alignment on one fixed state."""

    branch.train()
    parameters = tuple(branch.named_parameters())
    logits = correction_logits(
        prepared["base_logits"], prepared["p2"], branch
    )
    supervised = supervised_loss(logits, prepared["labels"])
    kd = masked_kl_divergence(
        logits,
        prepared["teacher_logits"],
        prepared["kd_mask"],
        temperature=KD_TEMPERATURE,
    )
    tensors = tuple(parameter for _, parameter in parameters)
    supervised_gradients = torch.autograd.grad(
        supervised,
        tensors,
        retain_graph=True,
        allow_unused=False,
    )
    kd_gradients = torch.autograd.grad(
        kd,
        tensors,
        allow_unused=False,
    )

    def group_stats(group: str) -> dict[str, Any]:
        selected: list[tuple[torch.Tensor, torch.Tensor]] = []
        for (name, _), supervised_gradient, kd_gradient in zip(
            parameters, supervised_gradients, kd_gradients, strict=True
        ):
            is_output = name.startswith("output_projection.")
            if group == "output_head" and not is_output:
                continue
            if group == "upstream" and is_output:
                continue
            if supervised_gradient is None or kd_gradient is None:
                raise AssertionError(f"gradient graph is disconnected at {name}")
            selected.append(
                (
                    supervised_gradient.detach().float().reshape(-1),
                    kd_gradient.detach().float().reshape(-1),
                )
            )
        supervised_sq = sum(
            float((supervised_gradient * supervised_gradient).sum())
            for supervised_gradient, _ in selected
        )
        kd_sq = sum(
            float((kd_gradient * kd_gradient).sum())
            for _, kd_gradient in selected
        )
        dot = sum(
            float((supervised_gradient * kd_gradient).sum())
            for supervised_gradient, kd_gradient in selected
        )
        supervised_norm = math.sqrt(max(supervised_sq, 0.0))
        kd_norm = math.sqrt(max(kd_sq, 0.0))
        denominator = supervised_norm * kd_norm
        combined_dot_kd = dot + KD_WEIGHT * kd_sq
        return {
            "supervised_norm": supervised_norm,
            "kd_norm": kd_norm,
            "kd_over_supervised_norm": (
                kd_norm / supervised_norm if supervised_norm > 0.0 else None
            ),
            "lambda_equal_norm": (
                supervised_norm / kd_norm if kd_norm > 0.0 else None
            ),
            "cosine": dot / denominator if denominator > 0.0 else None,
            "supervised_dot_kd": dot,
            "euclidean_combined_dot_kd_at_lambda_1": combined_dot_kd,
            "first_order_euclidean_step_reduces_kd": combined_dot_kd > 0.0,
            "parameter_values": int(sum(pair[0].numel() for pair in selected)),
        }

    result = {
        "supervised_loss": float(supervised.detach()),
        "kd_loss": float(kd.detach()),
        "groups": {
            group: group_stats(group)
            for group in ("all", "output_head", "upstream")
        },
    }
    del logits, supervised, kd, supervised_gradients, kd_gradients
    return result


def objective_loss_snapshot(
    branch: nn.Module,
    prepared: Mapping[str, torch.Tensor],
    supervised_loss: nn.Module,
) -> dict[str, float]:
    branch.eval()
    with torch.no_grad():
        logits = correction_logits(
            prepared["base_logits"], prepared["p2"], branch
        )
        supervised = supervised_loss(logits, prepared["labels"])
        kd = masked_kl_divergence(
            logits,
            prepared["teacher_logits"],
            prepared["kd_mask"],
            temperature=KD_TEMPERATURE,
        )
    return {
        "supervised": float(supervised),
        "kd": float(kd),
        "combined": float(supervised + KD_WEIGHT * kd),
    }


def kd_capacity_decision(initial_kd: float, last_ten_kd: Sequence[float]) -> dict[str, Any]:
    if not math.isfinite(initial_kd) or initial_kd <= 0.0:
        raise ValueError("initial KD must be positive and finite")
    if not last_ten_kd or any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in last_ten_kd
    ):
        raise ValueError("last-ten KD values must be finite and non-negative")
    median = float(statistics.median(float(value) for value in last_ten_kd))
    reduction = 1.0 - median / float(initial_kd)
    if reduction >= 0.50:
        outcome = "CLEAR_CAPACITY"
    elif reduction >= 0.20:
        outcome = "WEAK_CAPACITY"
    else:
        outcome = "NO_DEMONSTRATED_CAPACITY"
    return {
        "outcome": outcome,
        "initial_kd": float(initial_kd),
        "last_ten_median_kd": median,
        "reduction_fraction": reduction,
        "reduction_percent": reduction * 100.0,
        "thresholds": {
            "weak_capacity_minimum_reduction_percent": 20.0,
            "clear_capacity_minimum_reduction_percent": 50.0,
            "nature": (
                "pre-registered resource-screen heuristics for this fixed batch, "
                "optimizer, and 100-step budget; not a universal capacity theorem"
            ),
        },
    }


def select_objective_kd_mask(
    prepared: Mapping[str, torch.Tensor],
    mode: str,
) -> torch.Tensor:
    if mode == "gain":
        return prepared["kd_mask"].to(dtype=torch.bool)
    if mode != "correction":
        raise ValueError(f"unknown objective KD mask mode: {mode}")
    labels = prepared["labels"]
    valid = prepared["valid"]
    teacher_correct = prepared["teacher_logits"].argmax(dim=1).eq(labels)
    baseline_wrong = prepared["base_logits"].argmax(dim=1).ne(labels)
    return valid & teacher_correct & baseline_wrong


def active_kd_mask_statistics(
    prepared: Mapping[str, torch.Tensor],
    labels: Sequence[str],
) -> dict[str, Any]:
    active = prepared["kd_mask"].to(dtype=torch.bool)
    valid = prepared["valid"].to(dtype=torch.bool)
    small = prepared["small"].to(dtype=torch.bool)
    thin = prepared["thin"].to(dtype=torch.bool)
    target = prepared["labels"]
    teacher_correct = prepared["teacher_logits"].argmax(dim=1).eq(target)
    baseline_correct = prepared["base_logits"].argmax(dim=1).eq(target)
    active_count = int(active.sum())
    valid_count = int(valid.sum())

    def ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator > 0 else None

    quadrants = {
        "teacher_correct_base_wrong": teacher_correct & ~baseline_correct & valid,
        "both_correct": teacher_correct & baseline_correct & valid,
        "teacher_wrong_base_correct": ~teacher_correct & baseline_correct & valid,
        "both_wrong": ~teacher_correct & ~baseline_correct & valid,
    }
    return {
        "valid_pixels": valid_count,
        "active_pixels": active_count,
        "active_coverage": ratio(active_count, valid_count),
        "active_on_small_pixels": int((active & small & valid).sum()),
        "active_on_thin_pixels": int((active & thin & valid).sum()),
        "per_class": {
            str(class_name): {
                "valid_pixels": int(((target == index) & valid).sum()),
                "active_pixels": int(((target == index) & active).sum()),
            }
            for index, class_name in enumerate(labels)
        },
        "correctness_quadrants": {
            name: {
                "active_pixels": int((active & quadrant).sum()),
                "fraction_of_active": ratio(
                    int((active & quadrant).sum()), active_count
                ),
            }
            for name, quadrant in quadrants.items()
        },
    }


def run_diagnostic_mode(
    args: argparse.Namespace,
    *,
    dataset: WHUPhaseCropDataset,
    loader: DataLoader,
    extractor: FrozenE0P2Extractor,
    r0_branch: nn.Module,
    r1_branch: nn.Module,
    optimizer0: torch.optim.Optimizer,
    optimizer1: torch.optim.Optimizer,
    supervised_loss: nn.Module,
    device: torch.device,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    dataset.set_epoch(0)
    iterator = iter(loader)
    fixed_batch = next(iterator) if args.mode == "overfit" else None
    fixed_prepared = (
        prepare_training_batch(fixed_batch, extractor=extractor, device=device)
        if fixed_batch is not None
        else None
    )
    steps = args.overfit_steps if args.mode == "overfit" else args.smoke_steps
    counters = empty_mask_counters()
    step_records: list[dict[str, Any]] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        if fixed_batch is not None:
            batch = fixed_batch
        else:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
        if fixed_prepared is not None:
            metrics = optimize_prepared_batch(
                fixed_prepared,
                extractor=extractor,
                r0_branch=r0_branch,
                r1_branch=r1_branch,
                optimizer0=optimizer0,
                optimizer1=optimizer1,
                supervised_loss=supervised_loss,
                counters=counters,
                count_statistics=step == 1,
            )
        else:
            metrics = train_one_batch(
                batch,
                extractor=extractor,
                r0_branch=r0_branch,
                r1_branch=r1_branch,
                optimizer0=optimizer0,
                optimizer1=optimizer1,
                supervised_loss=supervised_loss,
                device=device,
                counters=counters,
            )
        record = {"step": step, **metrics}
        if step == 1:
            record["batch"] = batch_input_hashes(batch)
        step_records.append(record)
        append_jsonl(args.output_dir / "diagnostic_steps.jsonl", record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    torch.cuda.synchronize(device)
    kd_values = [float(record["kd"]) for record in step_records]
    total_kd_pixels = sum(int(record["kd_pixels"]) for record in step_records)
    if total_kd_pixels <= 0:
        raise RuntimeError("diagnostic batches contain zero teacher-gain pixels")
    joint_objective_check = None
    if args.mode == "overfit":
        last_ten_median = float(statistics.median(kd_values[-10:]))
        joint_objective_check = {
            "initial_kd": kd_values[0],
            "final_kd": kd_values[-1],
            "minimum_kd": min(kd_values),
            "final_below_initial": kd_values[-1] < kd_values[0],
            "last_ten_median_kd": last_ten_median,
            "last_ten_median_below_initial": last_ten_median < kd_values[0],
            "interpretation": (
                "This is CE+Dice+KD joint-objective behavior, not a branch-capacity "
                "test. Use objective mode for a separate KD-only capacity arm."
            ),
        }
    result = {
        "status": "PASS",
        "mode": args.mode,
        "steps": steps,
        "mask_statistics": serialize_mask_counters(counters, cfg["labels"]),
        "joint_objective_check": joint_objective_check,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        },
    }
    del fixed_prepared
    return result


def run_objective_diagnostic(
    args: argparse.Namespace,
    *,
    dataset: WHUPhaseCropDataset,
    loader: DataLoader,
    extractor: FrozenE0P2Extractor,
    combined_branch: nn.Module,
    kd_only_branch: nn.Module,
    combined_optimizer: torch.optim.Optimizer,
    kd_only_optimizer: torch.optim.Optimizer,
    supervised_loss: nn.Module,
    device: torch.device,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate branch capacity from CE+Dice/KD scale and conflict."""

    dataset.set_epoch(0)
    fixed_batch = next(iter(loader))
    prepared = prepare_training_batch(
        fixed_batch,
        extractor=extractor,
        device=device,
    )
    routing_counters = empty_mask_counters()
    count_mask_statistics(
        routing_counters,
        labels=prepared["labels"],
        base_logits=prepared["base_logits"],
        teacher_logits=prepared["teacher_logits"],
        gain=prepared["gain"],
        small=prepared["small"],
        thin=prepared["thin"],
    )
    prepared["kd_mask"] = select_objective_kd_mask(
        prepared, args.objective_mask
    )
    if int(prepared["kd_mask"].sum()) <= 0:
        raise RuntimeError(
            f"objective diagnostic fixed batch has zero {args.objective_mask} KD pixels"
        )
    combined_initial_metadata = branch_metadata(combined_branch)
    kd_only_initial_metadata = branch_metadata(kd_only_branch)
    if combined_initial_metadata != kd_only_initial_metadata:
        raise AssertionError("objective diagnostic branches do not start identically")

    active_statistics = active_kd_mask_statistics(prepared, cfg["labels"])
    fixed_batch_hashes = batch_input_hashes(fixed_batch)
    initial_combined = objective_loss_snapshot(
        combined_branch, prepared, supervised_loss
    )
    initial_kd_only = objective_loss_snapshot(
        kd_only_branch, prepared, supervised_loss
    )
    if initial_combined != initial_kd_only:
        raise AssertionError("objective diagnostic initial losses differ")
    gradient_probes = {
        "step_0": {
            "updates_completed": 0,
            "combined_branch": objective_gradient_diagnostics(
                combined_branch, prepared, supervised_loss
            ),
            "kd_only_branch": objective_gradient_diagnostics(
                kd_only_branch, prepared, supervised_loss
            ),
        }
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for step in range(1, args.overfit_steps + 1):
        combined_branch.train()
        kd_only_branch.train()
        combined_optimizer.zero_grad(set_to_none=True)
        kd_only_optimizer.zero_grad(set_to_none=True)

        combined_logits = correction_logits(
            prepared["base_logits"], prepared["p2"], combined_branch
        )
        combined_supervised = supervised_loss(
            combined_logits, prepared["labels"]
        )
        combined_kd = masked_kl_divergence(
            combined_logits,
            prepared["teacher_logits"],
            prepared["kd_mask"],
            temperature=KD_TEMPERATURE,
        )
        combined_total = combined_supervised + KD_WEIGHT * combined_kd
        if not torch.isfinite(combined_total):
            raise RuntimeError("non-finite combined objective diagnostic loss")
        combined_total.backward()
        combined_optimizer.step()

        kd_only_logits = correction_logits(
            prepared["base_logits"], prepared["p2"], kd_only_branch
        )
        kd_only_kd = masked_kl_divergence(
            kd_only_logits,
            prepared["teacher_logits"],
            prepared["kd_mask"],
            temperature=KD_TEMPERATURE,
        )
        if not torch.isfinite(kd_only_kd):
            raise RuntimeError("non-finite KD-only objective diagnostic loss")
        kd_only_kd.backward()
        kd_only_optimizer.step()

        combined_after = objective_loss_snapshot(
            combined_branch, prepared, supervised_loss
        )
        kd_only_after = objective_loss_snapshot(
            kd_only_branch, prepared, supervised_loss
        )
        record = {
            "step": step,
            "combined_branch": combined_after,
            "kd_only_branch": kd_only_after,
        }
        if step == 1:
            record["batch"] = fixed_batch_hashes
        records.append(record)
        append_jsonl(args.output_dir / "objective_steps.jsonl", record)
        if step in (1, 10, args.overfit_steps):
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
        if step == 10:
            gradient_probes["step_10"] = {
                "updates_completed": 10,
                "combined_branch": objective_gradient_diagnostics(
                    combined_branch, prepared, supervised_loss
                ),
                "kd_only_branch": objective_gradient_diagnostics(
                    kd_only_branch, prepared, supervised_loss
                ),
            }

    torch.cuda.synchronize(device)
    kd_only_values = [
        float(record["kd_only_branch"]["kd"]) for record in records
    ]
    combined_values = [
        float(record["combined_branch"]["kd"]) for record in records
    ]
    decision = kd_capacity_decision(
        float(initial_kd_only["kd"]),
        kd_only_values[-10:],
    )
    routing_statistics = serialize_mask_counters(
        routing_counters, cfg["labels"]
    )
    result = {
        "status": "PASS",
        "mode": "objective",
        "objective_mask_mode": args.objective_mask,
        "steps": args.overfit_steps,
        "scientific_question": (
            "Does the fixed P2 branch have fixed-batch KD capacity under the "
            "selected oracle routing mask independently of the released CE+Dice "
            "objective, and how do the two gradients interact?"
        ),
        "fixed_variables": {
            "same_cached_batch": True,
            "same_zero_initialization": True,
            "objective_mask_mode": args.objective_mask,
            "broad_gain_mask_used_for_loss": args.objective_mask == "gain",
            "mask_uses_ground_truth": args.objective_mask == "correction",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "temperature": KD_TEMPERATURE,
            "lambda_in_combined_branch": KD_WEIGHT,
            "hyperparameter_sweep": False,
        },
        "fixed_batch": fixed_batch_hashes,
        "initial_branch_metadata": {
            "combined_branch": combined_initial_metadata,
            "kd_only_branch": kd_only_initial_metadata,
        },
        "initial": {
            "combined_branch": initial_combined,
            "kd_only_branch": initial_kd_only,
        },
        "capacity_decision": decision,
        "combined_kd_behavior": {
            "final_kd": combined_values[-1],
            "minimum_kd": min(combined_values),
            "minimum_step": combined_values.index(min(combined_values)) + 1,
            "last_ten_median_kd": float(statistics.median(combined_values[-10:])),
        },
        "kd_only_behavior": {
            "final_kd": kd_only_values[-1],
            "minimum_kd": min(kd_only_values),
            "minimum_step": kd_only_values.index(min(kd_only_values)) + 1,
        },
        "gradient_probes": gradient_probes,
        # Keep the original key as a broad-gain compatibility alias.  In
        # correction mode it is routing context, not the active loss mask.
        "mask_statistics": routing_statistics,
        "routing_gain_statistics": routing_statistics,
        "active_kd_mask_statistics": active_statistics,
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        },
    }
    del prepared
    return result


def committed_stage_result(output_dir: Path, epoch: int) -> dict[str, Any]:
    """Reconstruct a stage result if shutdown happened after epoch commit."""

    if epoch not in EVALUATION_EPOCHS:
        raise RuntimeError(f"epoch {epoch} is not a formal stage boundary")
    verify_epoch_commit(output_dir, epoch)
    paths = epoch_artifact_paths(output_dir, epoch)
    train_record = json.loads(paths["train"].read_text(encoding="utf-8"))
    evaluation = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
    if int(train_record.get("epoch", -1)) != epoch or int(
        evaluation.get("epoch", -1)
    ) != epoch:
        raise RuntimeError(f"committed epoch-{epoch} records have wrong epoch values")
    rebuild_metric_indexes(output_dir)
    return {
        "status": "PASS",
        "mode": "formal",
        "resumed_from_epoch": epoch,
        "stopped_after_epoch": epoch,
        "reconstructed_from_committed_epoch": True,
        "last_train": train_record,
        "last_evaluation": evaluation,
        "checkpoint": str(paths["checkpoint"].resolve()),
    }


def run_formal(
    args: argparse.Namespace,
    *,
    dataset: WHUPhaseCropDataset,
    loader: DataLoader,
    loader_generator: torch.Generator,
    test_loader: DataLoader,
    extractor: FrozenE0P2Extractor,
    r0_branch: nn.Module,
    r1_branch: nn.Module,
    optimizer0: torch.optim.Optimizer,
    optimizer1: torch.optim.Optimizer,
    scheduler0: torch.optim.lr_scheduler.LRScheduler,
    scheduler1: torch.optim.lr_scheduler.LRScheduler,
    supervised_loss: nn.Module,
    device: torch.device,
    cfg: Mapping[str, Any],
    protocol: Mapping[str, Any],
    baseline_sha256: str,
    phase_reference: Mapping[str, Any],
    start_epoch: int,
    expected_e0_state: Mapping[str, str],
) -> dict[str, Any]:
    if args.stop_after_epoch < start_epoch:
        raise RuntimeError(
            f"stop-after epoch {args.stop_after_epoch} precedes resume epoch {start_epoch}"
        )
    if args.stop_after_epoch == start_epoch:
        return committed_stage_result(args.output_dir, start_epoch)
    evaluation_model = PairedEvaluationModel(extractor, r0_branch, r1_branch)
    evaluation_records: list[dict[str, Any]] = []
    last_train_record = None
    for epoch in range(start_epoch + 1, args.stop_after_epoch + 1):
        artifact_paths = epoch_artifact_paths(args.output_dir, epoch)
        if artifact_paths["commit"].exists():
            verify_epoch_commit(args.output_dir, epoch)
            raise RuntimeError(
                f"epoch {epoch} is already committed; resume from its checkpoint"
            )
        dataset.set_epoch(epoch)
        counters = empty_mask_counters()
        totals = {
            "r0_supervised": 0.0,
            "r1_supervised": 0.0,
            "kd": 0.0,
            "r0_total": 0.0,
            "r1_total": 0.0,
            "gain_pixels": 0.0,
            "kd_pixels": 0.0,
            "valid_pixels": 0.0,
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        steps = 0
        first_batch_hashes = None
        iterator = tqdm(loader, desc=f"paired epoch {epoch}/{PROTOCOL_EPOCHS}")
        for batch_index, batch in enumerate(iterator):
            if batch_index == 0:
                first_batch_hashes = batch_input_hashes(batch)
            metrics = train_one_batch(
                batch,
                extractor=extractor,
                r0_branch=r0_branch,
                r1_branch=r1_branch,
                optimizer0=optimizer0,
                optimizer1=optimizer1,
                supervised_loss=supervised_loss,
                device=device,
                counters=counters,
            )
            for name, value in metrics.items():
                totals[name] += float(value)
            steps += 1
            iterator.set_postfix(
                r0=f"{metrics['r0_total']:.4f}",
                r1=f"{metrics['r1_total']:.4f}",
                kd=f"{metrics['kd']:.4f}",
            )
        if steps == 0:
            raise RuntimeError("formal epoch executed zero optimizer steps")
        scheduler0.step()
        scheduler1.step()
        torch.cuda.synchronize(device)
        last_train_record = {
            "epoch": epoch,
            "steps": steps,
            "averages": average_step_metrics(totals, steps),
            "mask_statistics": serialize_mask_counters(counters, cfg["labels"]),
            "first_batch": first_batch_hashes,
            "learning_rate_after_scheduler": {
                "R0": optimizer0.param_groups[0]["lr"],
                "R1": optimizer1.param_groups[0]["lr"],
            },
            "branch_state": {
                "R0": branch_metadata(r0_branch),
                "R1": branch_metadata(r1_branch),
            },
            "runtime": {
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            },
        }
        evaluation = None
        if epoch in EVALUATION_EPOCHS:
            evaluation = evaluate_paired(
                evaluation_model,
                cfg,
                test_loader,
                device=device,
                inference_batch_size=args.inference_batch_size,
                phase_reference=phase_reference,
            )
            evaluation["epoch"] = epoch
            evaluation["decision"] = stage_decision(epoch, evaluation["differences"])
        current_e0_state = e0_state_fingerprints(extractor.base_model)
        if current_e0_state != dict(expected_e0_state):
            raise AssertionError(f"frozen E0 changed during epoch {epoch}")
        last_train_record["e0_state_unchanged"] = True
        write_json_atomic(artifact_paths["train"], last_train_record)
        if evaluation is not None:
            write_json_atomic(artifact_paths["evaluation"], evaluation)
        save_paired_checkpoint(
            artifact_paths["checkpoint"],
            epoch=epoch,
            r0_branch=r0_branch,
            r1_branch=r1_branch,
            optimizer0=optimizer0,
            optimizer1=optimizer1,
            scheduler0=scheduler0,
            scheduler1=scheduler1,
            loader_generator=loader_generator,
            baseline_sha256=baseline_sha256,
            protocol=protocol,
            allow_replace_incomplete=True,
        )
        publish_epoch_commit(args.output_dir, epoch)
        rebuild_metric_indexes(args.output_dir)
        print(json.dumps(last_train_record, ensure_ascii=False, sort_keys=True), flush=True)
        if evaluation is not None:
            evaluation_records.append(evaluation)
            print(json.dumps(evaluation["decision"], ensure_ascii=False), flush=True)
    if not evaluation_records or evaluation_records[-1]["epoch"] != args.stop_after_epoch:
        raise AssertionError("formal stage must stop on a scheduled evaluated epoch")
    return {
        "status": "PASS",
        "mode": "formal",
        "resumed_from_epoch": start_epoch,
        "stopped_after_epoch": args.stop_after_epoch,
        "last_train": last_train_record,
        "last_evaluation": evaluation_records[-1],
        "checkpoint": str(
            (args.output_dir / f"checkpoint_e{args.stop_after_epoch}.pth").resolve()
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WHU phase distillation")
    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    baseline_sha256 = file_sha256(args.baseline_checkpoint)
    phase_reference = read_phase_reference(
        args.phase_reference_json, baseline_sha256
    )

    e0, cfg = build_e0(args)
    e0_before = e0_state_fingerprints(e0)
    source_dataset = build_dataset(
        "WHU",
        "train",
        window_size=cfg["window_size"],
        model_name="DINOv3",
        modality="multi",
        backbone_type=BACKBONE_TYPE,
    )
    records, manifest_metadata = build_phase_records(
        args.teacher_manifest,
        args.structure_mask_manifest,
        source_dataset,
        baseline_sha256,
        require_full_split=args.mode == "formal",
    )
    dataset = WHUPhaseCropDataset(
        records,
        seed=args.seed,
        length=args.train_samples_per_epoch,
        source_cache_size=args.source_cache_size,
    )
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=loader_generator,
    )

    device = torch.device(args.device)
    e0.to(device)
    e0.eval()
    extractor = FrozenE0P2Extractor(e0)
    branch_seed = args.seed + 104_729
    set_seed(branch_seed)
    p2_channels = int(getattr(e0.decoder, "out_channels", 256))
    r0_branch = PhaseCorrectionBranch(
        in_channels=p2_channels,
        hidden_channels=args.hidden_channels,
        num_classes=NUM_CLASSES,
    ).to(device)
    r1_branch = copy.deepcopy(r0_branch).to(device)
    initial_r0 = branch_metadata(r0_branch)
    initial_r1 = branch_metadata(r1_branch)
    if initial_r0 != initial_r1:
        raise AssertionError("deep-copied R0/R1 initial states differ")
    initial_pair = {
        "branch_seed": branch_seed,
        "R0": initial_r0,
        "R1": initial_r1,
        "state_equal": True,
    }
    optimizer0, optimizer1, scheduler0, scheduler1 = build_optimizers(
        r0_branch,
        r1_branch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    protocol = build_scientific_protocol(
        args,
        baseline_sha256=baseline_sha256,
        phase_reference=phase_reference,
        manifest_metadata=manifest_metadata,
        initial_branch=initial_pair,
    )
    initialize_or_resume_output(args, protocol)

    start_epoch = 0
    if args.resume_checkpoint is None:
        dataset.set_epoch(0)
        initial_equivalence = verify_initial_zero_pair(
            dataset,
            extractor=extractor,
            r0_branch=r0_branch,
            r1_branch=r1_branch,
            device=device,
            batch_size=args.batch_size,
        )
        write_json_atomic(
            args.output_dir / "initial_equivalence.json", initial_equivalence
        )
    else:
        declared_resume_epoch = checkpoint_epoch_from_path(args.resume_checkpoint)
        validate_resume_commit(
            args.output_dir,
            declared_resume_epoch,
            args.resume_checkpoint,
        )
        start_epoch = load_paired_checkpoint(
            args.resume_checkpoint,
            r0_branch=r0_branch,
            r1_branch=r1_branch,
            optimizer0=optimizer0,
            optimizer1=optimizer1,
            scheduler0=scheduler0,
            scheduler1=scheduler1,
            loader_generator=loader_generator,
            baseline_sha256=baseline_sha256,
            protocol=protocol,
        )
        if start_epoch != declared_resume_epoch:
            raise RuntimeError("resume checkpoint filename/epoch payload mismatch")

    print(
        json.dumps(
            {
                "mode": args.mode,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "baseline_checkpoint_sha256": baseline_sha256,
                "protocol_sha256": canonical_json_sha256(protocol),
                "start_epoch": start_epoch,
                "stop_after_epoch": args.stop_after_epoch if args.mode == "formal" else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if args.mode == "formal":
        test_loader, full_test_length = build_test_loader(
            SimpleNamespace(max_images=None), cfg["window_size"]
        )
        if full_test_length != 20 or len(test_loader.dataset) != 20:
            raise RuntimeError("formal evaluation requires the complete 20-image test split")
        result = run_formal(
            args,
            dataset=dataset,
            loader=loader,
            loader_generator=loader_generator,
            test_loader=test_loader,
            extractor=extractor,
            r0_branch=r0_branch,
            r1_branch=r1_branch,
            optimizer0=optimizer0,
            optimizer1=optimizer1,
            scheduler0=scheduler0,
            scheduler1=scheduler1,
            supervised_loss=cfg["loss_fn"],
            device=device,
            cfg=cfg,
            protocol=protocol,
            baseline_sha256=baseline_sha256,
            phase_reference=phase_reference,
            start_epoch=start_epoch,
            expected_e0_state=e0_before,
        )
        summary_path = args.output_dir / f"stage_e{args.stop_after_epoch}_summary.json"
    elif args.mode == "objective":
        result = run_objective_diagnostic(
            args,
            dataset=dataset,
            loader=loader,
            extractor=extractor,
            combined_branch=r0_branch,
            kd_only_branch=r1_branch,
            combined_optimizer=optimizer0,
            kd_only_optimizer=optimizer1,
            supervised_loss=cfg["loss_fn"],
            device=device,
            cfg=cfg,
        )
        summary_path = args.output_dir / "objective_summary.json"
    else:
        result = run_diagnostic_mode(
            args,
            dataset=dataset,
            loader=loader,
            extractor=extractor,
            r0_branch=r0_branch,
            r1_branch=r1_branch,
            optimizer0=optimizer0,
            optimizer1=optimizer1,
            supervised_loss=cfg["loss_fn"],
            device=device,
            cfg=cfg,
        )
        summary_path = args.output_dir / f"{args.mode}_summary.json"

    e0_after = e0_state_fingerprints(e0)
    e0_unchanged = e0_before == e0_after
    e0_audit = {
        "before": e0_before,
        "after": e0_after,
        "unchanged": e0_unchanged,
        "all_parameters_frozen": not any(
            parameter.requires_grad for parameter in e0.parameters()
        ),
        "all_parameter_gradients_absent": not any(
            parameter.grad is not None for parameter in e0.parameters()
        ),
        "e0_training_flag": e0.training,
    }
    if not e0_unchanged or e0.training:
        raise AssertionError(f"frozen E0 state changed: {e0_audit}")
    result["e0_state_audit"] = e0_audit
    result["protocol_sha256"] = canonical_json_sha256(protocol)
    result["output_dir"] = str(args.output_dir.resolve())
    write_json_atomic(summary_path, result)
    print(f"phase_distillation_result={summary_path.resolve()}")
    print(f"phase_distillation_status=PASS mode={args.mode}")


if __name__ == "__main__":
    main()
