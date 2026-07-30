"""Replay a one-image Stage-B1 phase-crop cache without model execution.

This is a correctness-only runner.  It validates an immutable raw per-crop
logit cache against the sealed Stage-A and formal Stage-B0 artifacts, then
replays K1, support-pruned K2x/K4, and the preregistered middle policy.  An
optional first-image replay of the frozen B0 rescue policy is also available.

No outcome emitted here is a scientific GO.  The runner checks crop closure,
phase-local accumulation, inverse alignment, ownership masking, predictions,
and confusions; live model execution, structure gates, and latency remain for
Stage B1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_whu_phase_closure import load_stage_a  # noqa: E402
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
    crop_key_sha256,
    endpoint_levels,
    expected_phase_crop_ids,
    frozen_middle_levels,
    validate_phase_replay_on_routed_pixels,
)


ARTIFACT_TYPE = "whu_phase_cached_replay_correctness"
SCHEMA_VERSION = 1
CACHE_ARTIFACT_TYPE = "whu_phase_crop_logits_correctness_cache"
CACHE_SCHEMA_VERSION = 1
B0_ARTIFACT_TYPE = "whu_phase_closure_stage_b0"
B0_SCHEMA_VERSION = 1
STAGE_A_SCHEMA_VERSION = 3
PHASE_ORDER = ("normal", *PHASE_NAMES)
NUM_CLASSES = 7
FORMAL_B0_OUTCOME = "PROVISIONAL_GO_B1_KNOWN_B0_GATES_PASS"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correctness-only Stage-B1 cached sparse-phase replay"
    )
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--stage-a-json", type=Path, required=True)
    parser.add_argument("--stage-b0-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--include-first-image-rescue",
        action="store_true",
        help="also replay the frozen B0 rescue policy on cached image zero",
    )
    args = parser.parse_args()
    for path in (args.cache_manifest, args.stage_a_json, args.stage_b0_json):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    return args


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _integer_sequence(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence")
    return tuple(_integer(item, f"{name}[{index}]") for index, item in enumerate(value))


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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_cache_path(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TypeError(f"{name}.path must be a non-empty string")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{name}.path must be a safe relative path")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{name}.path escapes the cache directory") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"cached array does not exist: {resolved}")
    return resolved


def _dtype_name(dtype: np.dtype[Any]) -> str:
    return np.dtype(dtype).name


def load_array_descriptor(
    cache_root: Path,
    descriptor: Mapping[str, Any],
    *,
    name: str,
    expected_dtype: np.dtype[Any],
    expected_shape: Sequence[int],
    expected_relative_path: str | None = None,
    mmap: bool = False,
) -> np.ndarray:
    if not isinstance(descriptor, Mapping):
        raise TypeError(f"{name} descriptor must be an object")
    relative = descriptor.get("path")
    if expected_relative_path is not None and relative != expected_relative_path:
        raise ValueError(f"{name}.path differs from the sealed layout")
    path = _resolve_cache_path(cache_root, relative, name)
    if path.suffix != ".npy":
        raise ValueError(f"{name} must be an .npy file")
    expected_file_sha = _require_sha256(
        descriptor.get("file_sha256"), f"{name}.file_sha256"
    )
    if file_sha256(path) != expected_file_sha:
        raise ValueError(f"{name} file SHA256 mismatch")
    value = np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)
    dtype = np.dtype(expected_dtype)
    shape = tuple(int(item) for item in expected_shape)
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise ValueError(f"{name} array dtype/shape differs from the protocol")
    if descriptor.get("dtype") != _dtype_name(dtype):
        raise ValueError(f"{name} descriptor dtype mismatch")
    if _integer_sequence(descriptor.get("shape"), f"{name}.shape") != shape:
        raise ValueError(f"{name} descriptor shape mismatch")
    if _integer(descriptor.get("nbytes"), f"{name}.nbytes") != value.nbytes:
        raise ValueError(f"{name} descriptor nbytes mismatch")
    expected_array_sha = _require_sha256(
        descriptor.get("array_sha256"), f"{name}.array_sha256"
    )
    if array_sha256(value) != expected_array_sha:
        raise ValueError(f"{name} array SHA256 mismatch")
    return value


def _crop_id_sha256(crop_ids: Sequence[int]) -> str:
    values = np.asarray(tuple(int(value) for value in crop_ids), dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


class CropArrayStore(Mapping[int, np.ndarray]):
    """A validated lazy mapping from local crop id to raw float32 logits."""

    def __init__(
        self,
        cache_root: Path,
        phase_name: str,
        records: Mapping[int, Mapping[str, Any]],
        windows: Sequence[Sequence[int]],
    ) -> None:
        self.cache_root = cache_root
        self.phase_name = phase_name
        self.records = dict(records)
        self.windows = tuple(tuple(int(value) for value in item) for item in windows)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self.records))

    def __getitem__(self, crop_id: int) -> np.ndarray:
        crop_id = int(crop_id)
        if crop_id not in self.records:
            raise KeyError(crop_id)
        y0, y1, x0, x1 = self.windows[crop_id]
        descriptor = self.records[crop_id]
        return load_array_descriptor(
            self.cache_root,
            descriptor,
            name=f"{self.phase_name}.crop[{crop_id}]",
            expected_dtype=np.dtype("float32"),
            expected_shape=(NUM_CLASSES, y1 - y0, x1 - x0),
            expected_relative_path=(
                f"crops/{self.phase_name}/crop_{crop_id:04d}.npy"
            ),
            mmap=True,
        )


def load_b0(path: Path, stage_a_path: Path) -> dict[str, Any]:
    payload = load_json_strict(path)
    if payload.get("artifact_type") != B0_ARTIFACT_TYPE:
        raise ValueError("input is not a Stage-B0 closure artifact")
    if payload.get("schema_version") != B0_SCHEMA_VERSION:
        raise ValueError("Stage-B0 schema version differs from the replay contract")
    if payload.get("status") != "PASS" or payload.get("scope") != "full-test":
        raise ValueError("Stage-B0 input must be a full-test completed artifact")
    source = payload.get("source_stage_a")
    if not isinstance(source, Mapping):
        raise TypeError("Stage-B0 source_stage_a must be an object")
    if _require_sha256(source.get("sha256"), "B0 source Stage-A SHA") != file_sha256(
        stage_a_path
    ):
        raise ValueError("Stage-B0 points to a different Stage-A artifact")
    if source.get("schema_version") != STAGE_A_SCHEMA_VERSION:
        raise ValueError("Stage-B0 source Stage-A schema differs")
    decision = payload.get("stage_b0_decision")
    if not isinstance(decision, Mapping):
        raise TypeError("Stage-B0 decision must be an object")
    if (
        decision.get("outcome") != FORMAL_B0_OUTCOME
        or decision.get("scientific_decision_evaluated") is not True
        or decision.get("known_checks_passed") is not True
        or decision.get("b1_implementation_and_smoke_authorized") is not True
        or decision.get("full_stage_b_scientific_pass") is not False
    ):
        raise ValueError("Stage-B0 did not issue the frozen provisional B1 authorization")
    protocol = payload.get("protocol", {})
    if (
        protocol.get("formal_random_control") is not True
        or protocol.get("random_replicates") != 1000
        or protocol.get("random_seed") != 20260730
    ):
        raise ValueError("Stage-B0 random control differs from the formal protocol")
    oracle = payload.get("exact_cost_a2_oracle")
    if not isinstance(oracle, Mapping):
        raise TypeError("Stage-B0 exact oracle must be an object")
    if not isinstance(oracle.get("levels_by_cell"), list):
        raise ValueError("Stage-B0 exact oracle lacks levels_by_cell")
    return payload


def _validate_source_link(
    descriptor: Any,
    *,
    name: str,
    expected_path: Path,
    expected_type: str,
    expected_schema: int,
) -> None:
    if not isinstance(descriptor, Mapping):
        raise TypeError(f"cache {name} must be an object")
    if _require_sha256(descriptor.get("sha256"), f"cache {name} SHA") != file_sha256(
        expected_path
    ):
        raise ValueError(f"cache {name} SHA differs from the supplied artifact")
    if descriptor.get("artifact_type") != expected_type:
        raise ValueError(f"cache {name} artifact type mismatch")
    if descriptor.get("schema_version") != expected_schema:
        raise ValueError(f"cache {name} schema mismatch")


def _bounds_tuple(value: Any, name: str) -> tuple[int, int, int, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    result = tuple(
        _integer(value.get(key), f"{name}.{key}")
        for key in ("y_start", "y_stop", "x_start", "x_stop")
    )
    if result[0] >= result[1] or result[2] >= result[3]:
        raise ValueError(f"{name} is an empty rectangle")
    return result


def validate_cache_manifest(
    path: Path,
    stage_a_path: Path,
    b0_path: Path,
    stage_a: Mapping[str, Any],
    b0: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    payload = load_json_strict(path)
    if payload.get("artifact_type") != CACHE_ARTIFACT_TYPE:
        raise ValueError("input is not a phase-crop correctness cache")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("phase-crop cache schema version differs")
    if payload.get("status") != "PASS":
        raise ValueError("phase-crop cache is not complete")
    _validate_source_link(
        payload.get("source_stage_a"),
        name="source_stage_a",
        expected_path=stage_a_path,
        expected_type="whu_phase_utility_stage_a",
        expected_schema=STAGE_A_SCHEMA_VERSION,
    )
    _validate_source_link(
        payload.get("source_stage_b0"),
        name="source_stage_b0",
        expected_path=b0_path,
        expected_type=B0_ARTIFACT_TYPE,
        expected_schema=B0_SCHEMA_VERSION,
    )
    checkpoint = payload.get("baseline_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("cache baseline_checkpoint must be an object")
    checkpoint_sha = _require_sha256(
        checkpoint.get("sha256"), "cache baseline checkpoint SHA"
    )
    if checkpoint_sha != stage_a.get("baseline_checkpoint_sha256"):
        raise ValueError("cache checkpoint differs from Stage A")

    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TypeError("cache protocol must be an object")
    if protocol.get("phase_order") != list(PHASE_ORDER):
        raise ValueError("cache phase order differs from the replay contract")
    expected_shifts = {
        "normal": [0, 0],
        **{name: list(PHASE_SHIFTS[name]) for name in PHASE_NAMES},
    }
    if protocol.get("phase_shifts_dy_dx") != expected_shifts:
        raise ValueError("cache phase shifts differ from the sealed protocol")
    if protocol.get("crop_size_hw") != [512, 512] or protocol.get(
        "stride_hw"
    ) != [341, 341]:
        raise ValueError("cache crop/stride geometry differs from 512/341")
    if protocol.get("raw_crop_storage_dtype") != "float32":
        raise ValueError("cache raw crop dtype must be float32")
    if protocol.get("dense_common_count_dtype") != "int16":
        raise ValueError("cache count dtype must be int16")
    if protocol.get("crop_id_sha256_encoding") != "little-endian int64 bytes":
        raise ValueError("cache crop-id digest encoding differs")
    if _require_sha256(payload.get("protocol_sha256"), "cache protocol SHA") != canonical_json_sha256(protocol):
        raise ValueError("cache protocol SHA256 mismatch")

    image = payload.get("image")
    if not isinstance(image, Mapping):
        raise TypeError("cache image must be an object")
    image_index = _integer(image.get("image_index"), "cache image_index")
    for field in ("loader_position", "dataset_index"):
        if _integer(image.get(field), f"cache image.{field}") != image_index:
            raise ValueError(f"cache image {field} differs from image_index")
    if image_index >= len(stage_a["images"]):
        raise ValueError("cached image index lies outside Stage A")
    reference_image = stage_a["images"][image_index]
    if image.get("sample_name") != reference_image.get("sample_name"):
        raise ValueError("cache sample name differs from Stage A")
    if image.get("full_shape_hw") != reference_image.get("full_shape_hw"):
        raise ValueError("cache image shape differs from Stage A")
    if image.get("common_bounds") != reference_image.get("common_bounds"):
        raise ValueError("cache common bounds differ from Stage A")
    if image.get("crop_grid") != reference_image.get("crop_grid"):
        raise ValueError("cache crop grid differs from Stage A")
    if _bounds_tuple(image.get("common_bounds"), "cache common_bounds") != tuple(
        geometry["common_bounds"][image_index]
    ):
        raise ValueError("cache common bounds differ from rebuilt geometry")
    source_hashes = image.get("source_file_sha256")
    decoded_hashes = image.get("decoded_tensor_sha256")
    if not isinstance(source_hashes, Mapping) or not isinstance(
        decoded_hashes, Mapping
    ):
        raise TypeError("cache image source/decoded SHA records must be objects")
    for key in ("rgb", "sar", "label"):
        _require_sha256(source_hashes.get(key), f"source_file_sha256.{key}")
    for key in ("optical", "sar", "label_int64"):
        _require_sha256(decoded_hashes.get(key), f"decoded_tensor_sha256.{key}")
    endpoint_validation = image.get("endpoint_validation")
    if not isinstance(endpoint_validation, Mapping) or set(endpoint_validation) != {
        "k1",
        "matched_k2",
        "k4",
    }:
        raise ValueError("cache endpoint validation must contain K1/K2x/K4")
    for endpoint, reference_key in (
        ("k1", "k1"),
        ("matched_k2", "matched_k2"),
        ("k4", "k4"),
    ):
        record = endpoint_validation[endpoint]
        if not isinstance(record, Mapping) or record.get("equal") is not True:
            raise ValueError(f"cache endpoint {endpoint} did not validate")
        expected_prediction_sha = _require_sha256(
            reference_image["prediction_sha256"][reference_key],
            f"Stage-A {endpoint} prediction SHA",
        )
        if (
            _require_sha256(
                record.get("computed_prediction_sha256"),
                f"cache {endpoint} computed prediction SHA",
            )
            != expected_prediction_sha
            or _require_sha256(
                record.get("stage_a_prediction_sha256"),
                f"cache {endpoint} Stage-A prediction SHA",
            )
            != expected_prediction_sha
        ):
            raise ValueError(f"cache endpoint {endpoint} SHA differs from Stage A")
    return payload


def _validate_hash_descriptor(
    descriptor: Any,
    *,
    name: str,
    expected_dtype: str,
    expected_shape: Sequence[int],
    actual: np.ndarray,
) -> None:
    if not isinstance(descriptor, Mapping):
        raise TypeError(f"{name} must be a hash descriptor")
    if descriptor.get("dtype") != expected_dtype:
        raise ValueError(f"{name} dtype mismatch")
    if _integer_sequence(descriptor.get("shape"), f"{name}.shape") != tuple(
        expected_shape
    ):
        raise ValueError(f"{name} shape mismatch")
    if _integer(descriptor.get("nbytes"), f"{name}.nbytes") != int(actual.nbytes):
        raise ValueError(f"{name} nbytes mismatch")
    if _require_sha256(descriptor.get("array_sha256"), f"{name}.array_sha256") != array_sha256(
        actual
    ):
        raise ValueError(f"{name} array SHA mismatch")


def validate_dense_common(
    phase_name: str,
    descriptor: Mapping[str, Any],
    dense: Mapping[str, Any],
    geometry: Mapping[str, Any],
    image_index: int,
) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise TypeError(f"{phase_name}.dense_common must be an object")
    common = tuple(int(value) for value in geometry["common_bounds"][image_index])
    shift = (0, 0) if phase_name == "normal" else PHASE_SHIFTS[phase_name]
    shifted = (common[0] + shift[0], common[1] + shift[0], common[2] + shift[1], common[3] + shift[1])
    original_value = descriptor.get("original_bounds_yxyx", descriptor.get("original_bounds"))
    shifted_value = descriptor.get("shifted_bounds_yxyx", descriptor.get("shifted_bounds"))
    if _integer_sequence(original_value, f"{phase_name}.dense_common.original_bounds") != common:
        raise ValueError(f"{phase_name} dense-common original bounds mismatch")
    if _integer_sequence(shifted_value, f"{phase_name}.dense_common.shifted_bounds") != shifted:
        raise ValueError(f"{phase_name} dense-common shifted bounds mismatch")
    score_sum = aligned_region(dense["sum_logits"], shift, common)
    count = aligned_region(dense["count_mat"], shift, common)
    normalized = np.zeros(score_sum.shape, dtype=np.float32)
    np.divide(score_sum, count[None], out=normalized, where=count[None] > 0)
    if not np.all(count > 0):
        raise AssertionError(f"{phase_name} dense common support is not covered")
    _validate_hash_descriptor(
        descriptor.get("sum_logits"),
        name=f"{phase_name}.dense_common.sum_logits",
        expected_dtype="float32",
        expected_shape=score_sum.shape,
        actual=score_sum,
    )
    _validate_hash_descriptor(
        descriptor.get("count_mat"),
        name=f"{phase_name}.dense_common.count_mat",
        expected_dtype="int16",
        expected_shape=count.shape,
        actual=count,
    )
    _validate_hash_descriptor(
        descriptor.get("normalized_logits"),
        name=f"{phase_name}.dense_common.normalized_logits",
        expected_dtype="float32",
        expected_shape=normalized.shape,
        actual=normalized,
    )
    return {
        "original_bounds_yxyx": common,
        "shifted_bounds_yxyx": shifted,
        "sum_logits_sha256": array_sha256(score_sum),
        "count_mat_sha256": array_sha256(count),
        "normalized_logits_sha256": array_sha256(normalized),
        "positive_count_everywhere": True,
    }


def prepare_crop_stores(
    cache_manifest_path: Path,
    cache: Mapping[str, Any],
    b0: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> tuple[dict[str, CropArrayStore], dict[str, Any]]:
    cache_root = cache_manifest_path.resolve().parent
    image = cache["image"]
    image_index = int(image["image_index"])
    phases = image.get("phases")
    if not isinstance(phases, Mapping) or tuple(phases) != PHASE_ORDER:
        raise ValueError("cache image phases must preserve normal/x8/y8/xy8 order")
    windows = geometry["windows_by_image"][image_index]
    crop_count = len(windows)
    reproducibility = cache.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        raise TypeError("cache reproducibility must be an object")
    batch_size = _integer(
        reproducibility.get("inference_batch_size"),
        "cache inference_batch_size",
        minimum=1,
    )
    expected_batch_count = (crop_count + batch_size - 1) // batch_size
    k4_closure = b0["minimal_common_support_endpoints"]["k4"]["closure"][
        "per_image"
    ][image_index]
    stores: dict[str, CropArrayStore] = {}
    audit: dict[str, Any] = {}
    for phase_name in PHASE_ORDER:
        phase = phases[phase_name]
        if not isinstance(phase, Mapping):
            raise TypeError(f"cache phase {phase_name} must be an object")
        expected_shift = (0, 0) if phase_name == "normal" else PHASE_SHIFTS[phase_name]
        if _integer_sequence(phase.get("shift_dy_dx"), f"{phase_name}.shift") != expected_shift:
            raise ValueError(f"cache phase {phase_name} shift mismatch")
        if _integer(phase.get("dense_crop_count"), f"{phase_name}.dense_crop_count", minimum=1) != crop_count:
            raise ValueError(f"cache phase {phase_name} dense crop count mismatch")
        if _integer(
            phase.get("dense_batch_count"),
            f"{phase_name}.dense_batch_count",
            minimum=1,
        ) != expected_batch_count:
            raise ValueError(f"cache phase {phase_name} dense batch count mismatch")
        ids = _integer_sequence(phase.get("persisted_crop_ids"), f"{phase_name}.persisted_crop_ids")
        if ids != tuple(sorted(set(ids))) or any(value >= crop_count for value in ids):
            raise ValueError(f"cache phase {phase_name} crop ids are not canonical")
        if _integer(
            phase.get("persisted_crop_count"),
            f"{phase_name}.persisted_crop_count",
        ) != len(ids):
            raise ValueError(f"cache phase {phase_name} persisted count mismatch")
        expected_ids = (
            tuple(range(crop_count))
            if phase_name == "normal"
            else tuple(int(value) for value in k4_closure["phase_crop_ids"][phase_name])
        )
        if ids != expected_ids:
            raise ValueError(f"cache phase {phase_name} ids differ from B0 K4 closure")
        digest = _crop_id_sha256(ids)
        if _require_sha256(phase.get("persisted_crop_id_sha256"), f"{phase_name}.crop-id SHA") != digest:
            raise ValueError(f"cache phase {phase_name} crop-id SHA mismatch")
        if phase_name != "normal" and digest != k4_closure["phase_crop_id_sha256"][phase_name]:
            raise ValueError(f"cache phase {phase_name} crop-id SHA differs from B0")
        crops = phase.get("crops")
        if not isinstance(crops, list) or len(crops) != len(ids):
            raise ValueError(f"cache phase {phase_name} crop records are incomplete")
        records: dict[int, Mapping[str, Any]] = {}
        batch_slots: list[tuple[int, int]] = []
        for expected_id, record in zip(ids, crops, strict=True):
            if not isinstance(record, Mapping):
                raise TypeError(f"{phase_name} crop record must be an object")
            crop_id = _integer(record.get("local_crop_id"), f"{phase_name}.crop_id")
            if crop_id != expected_id:
                raise ValueError(f"{phase_name} crop records are not in canonical order")
            if _integer(
                record.get("dense_sequence_index"),
                f"{phase_name}.dense_sequence_index",
            ) != crop_id:
                raise ValueError(f"{phase_name} dense sequence index differs from crop id")
            if _integer_sequence(record.get("window_yxyx"), f"{phase_name}.window") != tuple(windows[crop_id]):
                raise ValueError(f"{phase_name} crop window differs from Stage A")
            batch_index = _integer(record.get("dense_batch_index"), f"{phase_name}.batch_index")
            batch_slot = _integer(record.get("dense_batch_slot"), f"{phase_name}.batch_slot")
            if batch_index != crop_id // batch_size or batch_slot != crop_id % batch_size:
                raise ValueError(f"{phase_name} dense batch metadata differs from crop order")
            batch_slots.append((batch_index, batch_slot))
            records[crop_id] = record
        if len(set(batch_slots)) != len(batch_slots) or batch_slots != sorted(batch_slots):
            raise ValueError(f"{phase_name} dense batch/slot metadata is inconsistent")
        store = CropArrayStore(cache_root, phase_name, records, windows)
        # Force every descriptor/file/hash through validation before replay.
        for crop_id in ids:
            value = store[crop_id]
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{phase_name} crop {crop_id} contains non-finite logits")
        stores[phase_name] = store
        audit[phase_name] = {
            "dense_crop_count": crop_count,
            "persisted_crop_count": len(ids),
            "persisted_crop_ids": ids,
            "persisted_crop_id_sha256": digest,
            "all_crop_records_validated": True,
        }
    return stores, audit


def confusion_matrix(label: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.asarray(label)
    candidate = np.asarray(prediction)
    if target.ndim == 3 and target.shape[0] == 1:
        target = target[0]
    if target.shape != candidate.shape or target.ndim != 2:
        raise ValueError("label/prediction shapes differ")
    if candidate.dtype != np.int64 or np.any((candidate < 0) | (candidate >= NUM_CLASSES)):
        raise ValueError("prediction contains an invalid class")
    valid = (target >= 0) & (target < NUM_CLASSES)
    encoded = target[valid].astype(np.int64) * NUM_CLASSES + candidate[valid]
    return np.bincount(encoded, minlength=NUM_CLASSES**2).reshape(
        NUM_CLASSES, NUM_CLASSES
    )


def expected_image_confusion(
    stage_a: Mapping[str, Any], levels: np.ndarray, image_index: int
) -> np.ndarray:
    image = stage_a["images"][image_index]
    result = np.asarray(image["confusion"]["k1"], dtype=np.int64).copy()
    for cell in stage_a["cells"]:
        if int(cell["image_index"]) != image_index:
            continue
        cell_index = int(cell["cell_index"])
        level = int(levels[cell_index])
        result += np.asarray(cell["confusion"][f"k{level}"], dtype=np.int64)
        result -= np.asarray(cell["confusion"]["k1"], dtype=np.int64)
    if np.any(result < 0):
        raise AssertionError("expected routed image confusion became negative")
    return result


def _normal_mean(normal_dense: Mapping[str, Any]) -> np.ndarray:
    score_sum = np.asarray(normal_dense["sum_logits"])
    count = np.asarray(normal_dense["count_mat"])
    if not np.all(count > 0):
        raise AssertionError("normal crops do not cover the full image")
    result = np.zeros(score_sum.shape, dtype=np.float32)
    np.divide(score_sum, count[None], out=result)
    return result


def _structured_keys(keys: Sequence[tuple[int, str, int]]) -> list[dict[str, Any]]:
    return [
        {
            "image_index": int(image_index),
            "phase_name": str(phase_name),
            "local_crop_id": int(crop_id),
        }
        for image_index, phase_name, crop_id in keys
    ]


def replay_policy(
    *,
    policy_name: str,
    levels: np.ndarray,
    image_index: int,
    stage_a: Mapping[str, Any],
    b0: Mapping[str, Any],
    geometry: Mapping[str, Any],
    stores: Mapping[str, CropArrayStore],
    dense_phases: Mapping[str, Mapping[str, Any]],
    normal_logits: np.ndarray,
    label: np.ndarray,
    closure_reference: Mapping[str, Any] | None,
    prediction_reference_key: str | None,
) -> dict[str, Any]:
    expected_ids = expected_phase_crop_ids(levels, geometry, image_index)
    expected_keys = tuple(
        (image_index, phase_name, crop_id)
        for phase_name in PHASE_NAMES
        for crop_id in expected_ids[phase_name]
    )
    expected_normal_keys = tuple(
        (image_index, "normal", crop_id)
        for crop_id in range(int(geometry["baseline_crop_counts"][image_index]))
    )
    observed_normal_keys = tuple(
        (image_index, "normal", int(crop_id)) for crop_id in stores["normal"]
    )
    if observed_normal_keys != expected_normal_keys:
        raise AssertionError(f"{policy_name} normal crop keys differ structurally")
    if closure_reference is not None:
        for phase_name in PHASE_NAMES:
            referenced = tuple(
                int(value) for value in closure_reference["phase_crop_ids"][phase_name]
            )
            if expected_ids[phase_name] != referenced:
                raise AssertionError(
                    f"{policy_name} expected {phase_name} keys differ from B0"
                )
    sparse_phases: dict[str, Mapping[str, Any]] = {}
    phase_checks: dict[str, Any] = {}
    actual_keys: list[tuple[int, str, int]] = []
    level_map = None
    common_bounds = tuple(int(value) for value in geometry["common_bounds"][image_index])
    for phase_name in PHASE_NAMES:
        crop_ids = expected_ids[phase_name]
        if not crop_ids:
            continue
        sparse = accumulate_phase_crops(
            stores[phase_name],
            crop_ids,
            geometry,
            image_index,
            num_classes=NUM_CLASSES,
        )
        sparse_phases[phase_name] = sparse
        if level_map is None:
            # compose_policy_logits independently rasterizes the same map below.
            from scripts.phase_sparse_replay_common import policy_level_map

            level_map = policy_level_map(levels, geometry, image_index)
        phase_checks[phase_name] = validate_phase_replay_on_routed_pixels(
            sparse,
            dense_phases[phase_name],
            level_map,
            phase_name=phase_name,
            shift=PHASE_SHIFTS[phase_name],
            common_bounds=common_bounds,
            compare_sum_logits=True,
        )
        actual_keys.extend(
            (image_index, phase_name, int(crop_id))
            for crop_id in sparse["crop_ids"]
        )
    actual_keys_tuple = tuple(actual_keys)
    if actual_keys_tuple != expected_keys:
        raise AssertionError(f"{policy_name} executed crop keys differ structurally")
    expected_model_keys = expected_normal_keys + expected_keys
    observed_model_keys = observed_normal_keys + actual_keys_tuple
    if observed_model_keys != expected_model_keys:
        raise AssertionError(f"{policy_name} complete model crop keys differ")

    dense_routed = compose_policy_logits(
        normal_logits,
        dense_phases,
        levels,
        geometry,
        image_index,
    )
    sparse_routed = compose_policy_logits(
        normal_logits,
        sparse_phases,
        levels,
        geometry,
        image_index,
    )
    if not np.array_equal(sparse_routed["level_map"], dense_routed["level_map"]):
        raise AssertionError(f"{policy_name} sparse/dense level maps differ")
    if not np.array_equal(sparse_routed["logits"], dense_routed["logits"]):
        raise AssertionError(f"{policy_name} sparse/dense routed logits are not bit-exact")
    if not np.array_equal(sparse_routed["prediction"], dense_routed["prediction"]):
        raise AssertionError(f"{policy_name} sparse/dense predictions differ")
    observed_confusion = confusion_matrix(label, sparse_routed["prediction"])
    expected_confusion = expected_image_confusion(stage_a, levels, image_index)
    if not np.array_equal(observed_confusion, expected_confusion):
        raise AssertionError(f"{policy_name} confusion differs from Stage-A cell replay")
    prediction_sha = sparse_routed["prediction_sha256"]
    reference_sha = None
    if prediction_reference_key is not None:
        reference_sha = _require_sha256(
            stage_a["images"][image_index]["prediction_sha256"][
                prediction_reference_key
            ],
            f"Stage-A {prediction_reference_key} prediction SHA",
        )
        if prediction_sha != reference_sha:
            raise AssertionError(
                f"{policy_name} prediction SHA differs from the sealed Stage-A endpoint"
            )
    local_levels = levels[np.asarray(geometry["cells_by_image"][image_index])]
    return {
        "role": "correctness anchor only; not a scientific metric or GO decision",
        "levels_on_cached_image": {
            f"k{level}": int(np.count_nonzero(local_levels == level))
            for level in (1, 2, 4)
        },
        "expected_extra_phase_crop_keys": _structured_keys(expected_keys),
        "observed_extra_phase_crop_keys": _structured_keys(actual_keys_tuple),
        "extra_phase_crop_key_sha256": crop_key_sha256(expected_keys),
        "structured_crop_keys_equal": True,
        "expected_model_crop_keys": _structured_keys(expected_model_keys),
        "observed_model_crop_keys": _structured_keys(observed_model_keys),
        "model_crop_key_sha256": crop_key_sha256(expected_model_keys),
        "complete_model_crop_keys_equal": True,
        "phase_sparse_dense_checks": phase_checks,
        "sparse_dense_level_map_equal": True,
        "sparse_dense_routed_logits_bit_exact": True,
        "sparse_dense_prediction_equal": True,
        "routed_logits_sha256": sparse_routed["logits_sha256"],
        "prediction_sha256": prediction_sha,
        "reference_prediction_sha256": reference_sha,
        "reference_prediction_sha256_equal": (
            True if reference_sha is not None else None
        ),
        "confusion": observed_confusion,
        "expected_confusion": expected_confusion,
        "confusion_equal": True,
        "valid_pixels": int(observed_confusion.sum()),
        "errors": int(observed_confusion.sum() - np.diag(observed_confusion).sum()),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    stage_a = load_stage_a(args.stage_a_json)
    if stage_a.get("schema_version") != STAGE_A_SCHEMA_VERSION:
        raise ValueError("Stage-A schema differs from the replay contract")
    b0 = load_b0(args.stage_b0_json, args.stage_a_json)
    geometry = build_phase_closure_geometry(stage_a["images"], stage_a["cells"])
    if len(b0["exact_cost_a2_oracle"]["levels_by_cell"]) != len(stage_a["cells"]):
        raise ValueError("B0 rescue levels differ from the Stage-A cell count")
    cache = validate_cache_manifest(
        args.cache_manifest,
        args.stage_a_json,
        args.stage_b0_json,
        stage_a,
        b0,
        geometry,
    )
    image = cache["image"]
    image_index = int(image["image_index"])
    cache_root = args.cache_manifest.resolve().parent
    height, width = geometry["image_shapes"][image_index]
    label = load_array_descriptor(
        cache_root,
        image.get("label"),
        name="label",
        expected_dtype=np.dtype("int64"),
        expected_shape=(height, width),
        expected_relative_path="labels/label.npy",
    )
    if array_sha256(label) != image["decoded_tensor_sha256"]["label_int64"]:
        raise ValueError("cached label differs from decoded label provenance SHA")
    stores, crop_record_audit = prepare_crop_stores(
        args.cache_manifest, cache, b0, geometry
    )

    dense_phases: dict[str, Mapping[str, Any]] = {}
    dense_common_audit: dict[str, Any] = {}
    for phase_name in PHASE_ORDER:
        ids = tuple(int(value) for value in cache["image"]["phases"][phase_name]["persisted_crop_ids"])
        dense = accumulate_phase_crops(
            stores[phase_name], ids, geometry, image_index, num_classes=NUM_CLASSES
        )
        dense_common_audit[phase_name] = validate_dense_common(
            phase_name,
            cache["image"]["phases"][phase_name]["dense_common"],
            dense,
            geometry,
            image_index,
        )
        dense_phases[phase_name] = dense
    normal_logits = _normal_mean(dense_phases.pop("normal"))

    endpoint_specs = {
        "k1": (endpoint_levels(geometry, 1), "k1", "k1"),
        "eligible_k2x": (
            endpoint_levels(geometry, 2),
            "matched_k2",
            "matched_k2",
        ),
        "eligible_k4": (endpoint_levels(geometry, 4), "k4", "k4"),
    }
    policies: dict[str, Any] = {}
    for policy_name, (levels, closure_key, prediction_key) in endpoint_specs.items():
        reference = b0["minimal_common_support_endpoints"][closure_key]["closure"][
            "per_image"
        ][image_index]
        policies[policy_name] = replay_policy(
            policy_name=policy_name,
            levels=np.asarray(levels, dtype=np.int64),
            image_index=image_index,
            stage_a=stage_a,
            b0=b0,
            geometry=geometry,
            stores=stores,
            dense_phases=dense_phases,
            normal_logits=normal_logits,
            label=label,
            closure_reference=reference,
            prediction_reference_key=prediction_key,
        )

    middle_matches = [
        index
        for index, name in enumerate(geometry["sample_names"])
        if name == "NH49E001014"
    ]
    if middle_matches != [image_index]:
        raise ValueError(
            "the one-image correctness cache must contain the frozen middle sample"
        )
    middle = frozen_middle_levels(geometry)
    policies["frozen_middle_subset"] = replay_policy(
        policy_name="frozen_middle_subset",
        levels=middle,
        image_index=image_index,
        stage_a=stage_a,
        b0=b0,
        geometry=geometry,
        stores=stores,
        dense_phases=dense_phases,
        normal_logits=normal_logits,
        label=label,
        closure_reference=None,
        prediction_reference_key=None,
    )

    if args.include_first_image_rescue:
        if image_index != 0:
            raise ValueError("first-image rescue replay requires cached image_index=0")
        rescue_levels = np.asarray(
            b0["exact_cost_a2_oracle"]["levels_by_cell"], dtype=np.int64
        )
        rescue_reference = b0["exact_cost_a2_oracle"]["closure"]["per_image"][0]
        policies["b0_rescue_first_image"] = replay_policy(
            policy_name="b0_rescue_first_image",
            levels=rescue_levels,
            image_index=0,
            stage_a=stage_a,
            b0=b0,
            geometry=geometry,
            stores=stores,
            dense_phases=dense_phases,
            normal_logits=normal_logits,
            label=label,
            closure_reference=rescue_reference,
            prediction_reference_key=None,
        )

    output = {
        "status": "PASS",
        "status_meaning": (
            "All requested cached-replay integrity and correctness assertions passed; "
            "PASS is an execution status and not a Stage-B scientific decision."
        ),
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "single-image-correctness-only",
        "scientific_scope": (
            "Cached raw-logit replay only; no live model execution, structure gate, "
            "latency gate, router prediction, H2 confirmation, or scientific GO."
        ),
        "stage_b1_decision": {
            "outcome": "NOT_EVALUATED_CORRECTNESS_ONLY",
            "full_stage_b_scientific_pass": False,
            "h2_confirmed": False,
        },
        "sources": {
            "cache_manifest": {
                "path": str(args.cache_manifest.resolve()),
                "sha256": file_sha256(args.cache_manifest),
                "artifact_type": cache["artifact_type"],
                "schema_version": cache["schema_version"],
            },
            "stage_a": {
                "path": str(args.stage_a_json.resolve()),
                "sha256": file_sha256(args.stage_a_json),
                "git_revision": stage_a["reproducibility"]["git_revision"],
            },
            "stage_b0": {
                "path": str(args.stage_b0_json.resolve()),
                "sha256": file_sha256(args.stage_b0_json),
                "git_revision": b0["reproducibility"]["git_revision"],
                "outcome": b0["stage_b0_decision"]["outcome"],
            },
        },
        "image": {
            "image_index": image_index,
            "sample_name": image["sample_name"],
            "full_shape_hw": [height, width],
            "common_bounds_yxyx": list(geometry["common_bounds"][image_index]),
            "label_sha256": array_sha256(label),
        },
        "cache_integrity": {
            "crop_records": crop_record_audit,
            "dense_common": dense_common_audit,
            "all_declared_arrays_and_files_sha256_validated": True,
        },
        "policies": policies,
        "correctness_checks": {
            "source_sha_and_schema": True,
            "crop_records_and_windows": True,
            "b0_closure_keys": True,
            "dense_common_replay": True,
            "phase_sparse_dense_count_and_sum_bit_exact_on_routed_pixels": True,
            "ownership_mask_blocks_closure_spill": True,
            "sparse_dense_predictions": True,
            "stage_a_endpoint_prediction_sha": True,
            "stage_a_cell_replay_confusions": True,
        },
        "next_step": (
            "Use the same frozen crop keys/action levels for live-model correctness; "
            "then close aggregate small/thin and support-pruned K4 latency before "
            "H2 can be evaluated."
        ),
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "replay_common_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_sparse_replay_common.py"
            ),
            "numpy": np.__version__,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_write_json(args.output_path, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "scope": output["scope"],
                "sample_name": image["sample_name"],
                "policies": list(policies),
                "stage_b1_outcome": output["stage_b1_decision"]["outcome"],
                "h2_confirmed": False,
                "output": str(args.output_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
