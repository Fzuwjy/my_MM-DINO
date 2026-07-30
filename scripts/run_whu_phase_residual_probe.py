"""Run the sealed matched-operator WHU phase-residual P2 probe.

This is an independent successor to the V1 objective diagnostic.  It does not
reinterpret or modify that experiment.  The old correction-only run is used
only as an immutable identity anchor for the random crops, P2 branch
initialization, and optimizer hyperparameters.

The scientific target is

    center(T_slide - E0_slide),

where both cached tensors use the same full-image 512/341 count-normalized
sliding operator and the same common bounds.  Two deep-copied branches are
optimized on one fixed batch for exactly 100 steps:

* Arm A minimizes the target residual on oracle repair pixels only;
* Arm B adds an equally weighted zero-residual loss on every other valid pixel.

Fixed-crop behavior is diagnostic only.  The resource decision is made from
Arm B at step 100 after running the actual normal-phase full-image sliding
path on the same cached training image.  Consequently a PASS is only a cheap
same-image capacity/selectivity signal; it is not test-set improvement,
generalization, phase causality, or an unlabeled routing result.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from scripts.cache_whu_e0_slide_companion import (  # noqa: E402
    E0SlideCompanionRecord,
    load_e0_slide_companion_records,
)
from scripts.cache_whu_phase_teacher import build_full_image_dataset  # noqa: E402
from scripts.phase_distillation_common import (  # noqa: E402
    FrozenE0P2Extractor,
    PhaseCorrectionBranch,
)
from scripts.phase_residual_common import (  # noqa: E402
    PhaseResidualGateThresholds,
    center_class_logits,
    fix_mask_rms_scale,
    full_resolution_centered_delta,
    matched_phase_residual_target,
    matched_residual_masks,
    phase_residual_behavior_statistics,
    phase_residual_probe_gate,
    scaled_phase_residual_losses,
)
from scripts.run_whu_phase_distillation import (  # noqa: E402
    BACKBONE_TYPE,
    CROP_SIZE,
    DEFAULT_SOURCE_CACHE_SIZE,
    DEFAULT_TRAIN_SAMPLES_PER_EPOCH,
    INFERENCE_STRIDE,
    NUM_CLASSES,
    PhaseImageRecord,
    SmallArrayCache,
    WHUPhaseCropDataset,
    batch_input_hashes,
    branch_metadata,
    build_e0,
    build_phase_records,
    canonical_json_sha256,
    e0_state_fingerprints,
    file_sha256,
    git_commit,
    label_sha256,
    numpy_array_sha256,
    tensor_sha256,
    write_json_atomic,
)
from scripts.spatial_diagnostics_common import confusion_from_arrays  # noqa: E402
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.inference import slide_inference  # noqa: E402
from utils.utils import set_seed  # noqa: E402


PROBE_STEPS = 100
SMOOTH_L1_BETA = 1.0
BRANCH_SEED_OFFSET = 104_729
PROTOCOL_VERSION = 1
ARTIFACT_TYPE = "whu_matched_phase_residual_p2_probe"
GATE_THRESHOLDS = PhaseResidualGateThresholds(
    minimum_routed_fix_rate=0.50,
    maximum_broken_per_routed_fixed=0.25,
    minimum_miou_delta_pp=0.0,
)


def _json_payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _normalized_name(value: Any) -> str:
    return Path(str(value)).stem.lower()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def validate_reference_objective(
    summary_path: Path,
    *,
    baseline_sha256: str,
    teacher_manifest_sha256: str,
    structure_manifest_sha256: str,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, Any]:
    """Validate and summarize the old correction-only identity anchor."""

    summary = _json_payload(summary_path)
    config_path = summary_path.resolve().parent / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"reference objective run_config.json is missing beside {summary_path}"
        )
    config = _json_payload(config_path)
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise RuntimeError("reference run_config lacks its scientific protocol")
    protocol_sha = canonical_json_sha256(protocol)
    failed: list[str] = []
    if summary.get("status") != "PASS":
        failed.append("summary status is not PASS")
    if summary.get("mode") != "objective":
        failed.append("summary mode is not objective")
    if summary.get("objective_mask_mode") != "correction":
        failed.append("objective mask is not correction-only")
    if int(summary.get("steps", -1)) != PROBE_STEPS:
        failed.append("reference did not complete exactly 100 steps")
    if summary.get("protocol_sha256") != protocol_sha:
        failed.append("summary/run_config protocol hashes differ")
    if config.get("protocol_sha256") != protocol_sha:
        failed.append("run_config protocol hash is inconsistent")
    if protocol.get("baseline_checkpoint_sha256") != baseline_sha256:
        failed.append("reference E0 checkpoint differs")
    if protocol.get("execution_mode") != "objective":
        failed.append("reference protocol is not objective mode")
    if protocol.get("objective_mask_mode") != "correction":
        failed.append("reference protocol is not correction-only")
    manifests = protocol.get("immutable_manifests")
    if not isinstance(manifests, Mapping):
        failed.append("reference immutable manifests are missing")
    else:
        if manifests.get("teacher_manifest_sha256") != teacher_manifest_sha256:
            failed.append("reference teacher manifest differs")
        if manifests.get("structure_manifest_sha256") != structure_manifest_sha256:
            failed.append("reference structure manifest differs")
    fixed_variables = summary.get("fixed_variables")
    if not isinstance(fixed_variables, Mapping):
        failed.append("reference fixed_variables are missing")
    else:
        if float(fixed_variables.get("learning_rate", float("nan"))) != float(
            learning_rate
        ):
            failed.append("reference learning rate differs")
        if float(fixed_variables.get("weight_decay", float("nan"))) != float(
            weight_decay
        ):
            failed.append("reference weight decay differs")
        if fixed_variables.get("same_cached_batch") is not True:
            failed.append("reference did not seal one cached batch")
        if fixed_variables.get("same_zero_initialization") is not True:
            failed.append("reference did not seal paired initialization")
    e0_audit = summary.get("e0_state_audit")
    if not isinstance(e0_audit, Mapping) or not all(
        e0_audit.get(key) is expected
        for key, expected in (
            ("unchanged", True),
            ("all_parameters_frozen", True),
            ("all_parameter_gradients_absent", True),
            ("e0_training_flag", False),
        )
    ):
        failed.append("reference frozen-E0 audit is incomplete")
    fixed_batch = summary.get("fixed_batch")
    initial_metadata = summary.get("initial_branch_metadata")
    if not isinstance(fixed_batch, Mapping):
        failed.append("reference fixed batch is missing")
    if not isinstance(initial_metadata, Mapping) or not isinstance(
        initial_metadata.get("combined_branch"), Mapping
    ):
        failed.append("reference initial branch metadata is missing")
    if failed:
        raise RuntimeError("invalid correction-only reference: " + "; ".join(failed))
    return {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "run_config_path": str(config_path.resolve()),
        "run_config_sha256": file_sha256(config_path),
        "protocol_sha256": protocol_sha,
        "fixed_batch": dict(fixed_batch),
        "initial_branch_metadata": dict(initial_metadata["combined_branch"]),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
    }


def validate_fixed_batch_identity(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if dict(actual) != dict(expected):
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise RuntimeError(
            "new probe does not reproduce the old fixed batch; differing fields: "
            + ", ".join(differing)
        )


def validate_companion_alignment(
    phase_records: Sequence[PhaseImageRecord],
    companion_records: Mapping[int, E0SlideCompanionRecord],
) -> None:
    if len(phase_records) != 1:
        raise RuntimeError(
            "this sealed probe is a one-image exploratory screen; expected exactly "
            f"one teacher record, got {len(phase_records)}"
        )
    if set(companion_records) != {record.index for record in phase_records}:
        raise RuntimeError("teacher and E0-slide companion image indices differ")
    for record in phase_records:
        companion = companion_records[record.index]
        checks = {
            "sample_name": _normalized_name(companion.sample_name)
            == _normalized_name(record.sample_name),
            "full_shape": tuple(companion.full_shape_hw) == tuple(record.full_shape_hw),
            "bounds": tuple(companion.bounds_yxyx) == tuple(record.bounds_yxyx),
            "logits_shape": tuple(companion.logits_shape)
            == tuple(record.teacher_logits_shape),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                f"teacher/E0-slide alignment differs for {record.sample_name}: {failed}"
            )


def companion_crops_for_batch(
    batch: Mapping[str, Any],
    companion_records: Mapping[int, E0SlideCompanionRecord],
    *,
    array_cache: SmallArrayCache,
) -> torch.Tensor:
    """Load exact float32 E0-slide crops at the old batch coordinates."""

    image_indices = [int(value) for value in batch["image_index"].tolist()]
    crop_y = [int(value) for value in batch["crop_y"].tolist()]
    crop_x = [int(value) for value in batch["crop_x"].tolist()]
    crops: list[np.ndarray] = []
    verified: set[int] = set()
    for index, y, x in zip(image_indices, crop_y, crop_x, strict=True):
        if index not in companion_records:
            raise RuntimeError(f"fixed batch references uncached E0-slide image {index}")
        record = companion_records[index]
        value = array_cache.get(record.logits_path, mmap=True)
        if value.dtype != np.float16 or tuple(value.shape) != tuple(record.logits_shape):
            raise RuntimeError(f"E0-slide array metadata mismatch: {record.sample_name}")
        if index not in verified:
            if numpy_array_sha256(value) != record.logits_array_sha256:
                raise RuntimeError(f"E0-slide array SHA mismatch: {record.sample_name}")
            verified.add(index)
        y0, y1, x0, x1 = (int(v) for v in record.bounds_yxyx)
        if y < y0 or x < x0 or y + CROP_SIZE > y1 or x + CROP_SIZE > x1:
            raise RuntimeError("fixed crop lies outside the E0-slide companion bounds")
        local_y, local_x = y - y0, x - x0
        crop = np.asarray(
            value[
                :,
                local_y : local_y + CROP_SIZE,
                local_x : local_x + CROP_SIZE,
            ],
            dtype=np.float32,
        ).copy()
        if crop.shape != (NUM_CLASSES, CROP_SIZE, CROP_SIZE):
            raise RuntimeError("E0-slide crop shape differs from 7x512x512")
        crops.append(np.ascontiguousarray(crop))
    return torch.from_numpy(np.stack(crops, axis=0))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 100-step matched WHU phase-residual P2 probe"
    )
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--e0-slide-manifest", type=Path, required=True)
    parser.add_argument("--structure-mask-manifest", type=Path, required=True)
    parser.add_argument("--reference-objective-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        "--source-cache-size", type=int, default=DEFAULT_SOURCE_CACHE_SIZE
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    for path in (
        args.baseline_checkpoint,
        args.teacher_manifest,
        args.e0_slide_manifest,
        args.structure_mask_manifest,
        args.reference_objective_summary,
    ):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
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
    return args


def _selected_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 4 or mask.shape != (
        values.shape[0],
        values.shape[2],
        values.shape[3],
    ):
        raise ValueError("selected-value tensor/mask shapes differ")
    if mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("selected-value mask must be non-empty bool")
    return values.permute(0, 2, 3, 1)[mask].reshape(-1)


def magnitude_statistics(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    divisor: float = 1.0,
) -> dict[str, Any]:
    """Report deterministic absolute-value summaries; never choose a scale."""

    if not math.isfinite(float(divisor)) or divisor <= 0:
        raise ValueError("statistics divisor must be finite and positive")
    selected = (
        _selected_values(values.detach().to(dtype=torch.float32), mask)
        .abs()
        .to(device="cpu", dtype=torch.float64)
        / float(divisor)
    )
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("magnitude statistics contain non-finite values")
    quantiles = torch.quantile(
        selected, torch.tensor([0.50, 0.90, 0.95, 0.99], dtype=torch.float64)
    )
    return {
        "selected_values": int(selected.numel()),
        "mean_abs": float(selected.mean().item()),
        "p50_abs": float(quantiles[0].item()),
        "p90_abs": float(quantiles[1].item()),
        "p95_abs": float(quantiles[2].item()),
        "p99_abs": float(quantiles[3].item()),
        "rms": float(torch.sqrt(selected.square().mean()).item()),
        "divisor": float(divisor),
    }


def residual_loss_tensors(
    branch: nn.Module,
    *,
    p2: torch.Tensor,
    reference_logits: torch.Tensor,
    target: torch.Tensor,
    fix_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    low_delta = branch(p2)
    delta = full_resolution_centered_delta(low_delta, reference_logits)
    losses = scaled_phase_residual_losses(
        delta,
        target,
        fix_mask,
        keep_mask,
        scale=scale,
        beta=SMOOTH_L1_BETA,
    )
    return delta, losses


def residual_loss_snapshot(
    branch: nn.Module,
    *,
    p2: torch.Tensor,
    reference_logits: torch.Tensor,
    target: torch.Tensor,
    fix_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    scale: float,
) -> dict[str, float]:
    branch.eval()
    with torch.no_grad():
        _, losses = residual_loss_tensors(
            branch,
            p2=p2,
            reference_logits=reference_logits,
            target=target,
            fix_mask=fix_mask,
            keep_mask=keep_mask,
            scale=scale,
        )
    result = {name: float(value.item()) for name, value in losses.items()}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("residual loss snapshot is non-finite")
    return result


def gradient_interaction(
    branch: nn.Module,
    *,
    p2: torch.Tensor,
    reference_logits: torch.Tensor,
    target: torch.Tensor,
    fix_mask: torch.Tensor,
    keep_mask: torch.Tensor,
    scale: float,
) -> dict[str, Any]:
    """Report fix/keep gradients without changing optimizer or parameter grads."""

    branch.train()
    parameters = tuple(parameter for parameter in branch.parameters() if parameter.requires_grad)
    _, losses = residual_loss_tensors(
        branch,
        p2=p2,
        reference_logits=reference_logits,
        target=target,
        fix_mask=fix_mask,
        keep_mask=keep_mask,
        scale=scale,
    )
    fix_gradients = torch.autograd.grad(
        losses["fix"], parameters, retain_graph=True, allow_unused=True
    )
    keep_gradients = torch.autograd.grad(
        losses["keep"], parameters, retain_graph=False, allow_unused=True
    )

    def flattened(gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
        return torch.cat(
            [
                (
                    gradient.detach().reshape(-1)
                    if gradient is not None
                    else torch.zeros_like(parameter).reshape(-1)
                )
                for parameter, gradient in zip(parameters, gradients, strict=True)
            ]
        )

    fix_vector = flattened(fix_gradients)
    keep_vector = flattened(keep_gradients)
    fix_norm = float(torch.linalg.vector_norm(fix_vector).item())
    keep_norm = float(torch.linalg.vector_norm(keep_vector).item())
    cosine = None
    if fix_norm > 0.0 and keep_norm > 0.0:
        cosine = float(
            torch.dot(fix_vector, keep_vector).item() / (fix_norm * keep_norm)
        )
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "fix_gradient_norm": fix_norm,
        "keep_gradient_norm": keep_norm,
        "cosine": cosine,
        "parameter_values": int(fix_vector.numel()),
    }


def behavior_for_branch(
    branch: nn.Module,
    *,
    p2: torch.Tensor,
    e0_slide_logits: torch.Tensor,
    teacher_slide_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    branch.eval()
    with torch.no_grad():
        low_delta = branch(p2)
        delta = full_resolution_centered_delta(low_delta, e0_slide_logits)
        corrected = e0_slide_logits + delta
        behavior = phase_residual_behavior_statistics(
            e0_slide_logits,
            teacher_slide_logits,
            corrected,
            labels,
        )
    return behavior, delta.detach()


def region_change_statistics(
    e0_logits: torch.Tensor,
    corrected_logits: torch.Tensor,
    labels: torch.Tensor,
    region_mask: torch.Tensor,
) -> dict[str, Any]:
    valid = labels.ge(0) & labels.lt(NUM_CLASSES) & region_mask
    e0_prediction = e0_logits.argmax(dim=1)
    corrected_prediction = corrected_logits.argmax(dim=1)
    e0_correct = valid & e0_prediction.eq(labels)
    corrected_correct = valid & corrected_prediction.eq(labels)
    e0_errors = int((valid & ~e0_correct).sum().item())
    corrected_errors = int((valid & ~corrected_correct).sum().item())
    return {
        "pixels": int(valid.sum().item()),
        "e0_errors": e0_errors,
        "corrected_errors": corrected_errors,
        "error_delta": corrected_errors - e0_errors,
        "globally_fixed_pixels": int((valid & ~e0_correct & corrected_correct).sum()),
        "broken_pixels": int((valid & e0_correct & ~corrected_correct).sum()),
        "changed_prediction_pixels": int(
            (valid & corrected_prediction.ne(e0_prediction)).sum()
        ),
    }


class SlidingArmBModel(nn.Module):
    """Return E0 and E0+center(delta) for one shared crop forward."""

    def __init__(
        self, extractor: FrozenE0P2Extractor, branch: PhaseCorrectionBranch
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.branch = branch

    def train(self, mode: bool = True):
        super().train(False)
        self.extractor.eval()
        self.branch.eval()
        return self

    def forward(self, *modalities: torch.Tensor) -> torch.Tensor:
        base_logits, p2 = self.extractor(*modalities)
        delta = full_resolution_centered_delta(self.branch(p2), base_logits)
        return torch.cat((base_logits, base_logits + delta), dim=1)


def _common_structure_crop(
    encoded: np.ndarray,
    mask_bounds: tuple[int, int, int, int],
    common_bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    mask_y0, mask_y1, mask_x0, mask_x1 = mask_bounds
    y0, y1, x0, x1 = common_bounds
    if y0 < mask_y0 or x0 < mask_x0 or y1 > mask_y1 or x1 > mask_x1:
        raise RuntimeError("common bounds lie outside structure-mask bounds")
    crop = np.asarray(
        encoded[y0 - mask_y0 : y1 - mask_y0, x0 - mask_x0 : x1 - mask_x0]
    )
    if crop.shape != (y1 - y0, x1 - x0):
        raise RuntimeError("structure-mask common crop has the wrong shape")
    crop = crop.astype(np.uint8, copy=False)
    return (crop & np.uint8(1)) != 0, (crop & np.uint8(2)) != 0


def full_label_to_numpy(label: torch.Tensor | np.ndarray) -> np.ndarray:
    """Normalize direct-dataset or DataLoader labels to contiguous int64 HW."""

    if isinstance(label, torch.Tensor):
        value = label.detach().cpu().numpy()
    else:
        value = np.asarray(label)
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.integer):
        raise TypeError("full-image WHU label must be a 2-D integer array")
    return np.ascontiguousarray(value.astype(np.int64, copy=False))


def exact_full_slide_evaluation(
    *,
    phase_record: PhaseImageRecord,
    companion_record: E0SlideCompanionRecord,
    extractor: FrozenE0P2Extractor,
    arm_b: PhaseCorrectionBranch,
    device: torch.device,
    inference_batch_size: int,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Evaluate step-100 Arm B through the actual normal 512/341 path."""

    full_dataset = build_full_image_dataset("train")
    if phase_record.index >= len(full_dataset):
        raise RuntimeError("cached image index is outside the WHU train split")
    if _normalized_name(full_dataset.rgb_files[phase_record.index]) != _normalized_name(
        phase_record.sample_name
    ):
        raise RuntimeError("full-slide source image differs from the sealed cache")
    optical, sar, label_value = full_dataset[phase_record.index]
    if optical.ndim != 3 or sar.ndim != 3:
        raise RuntimeError("unexpected full-image WHU tensor ranks")
    label_full = full_label_to_numpy(label_value)
    full_shape = tuple(int(value) for value in label_full.shape)
    if full_shape != tuple(phase_record.full_shape_hw):
        raise RuntimeError("full-slide label shape differs from the cache")
    if label_sha256(label_full) != phase_record.label_sha256:
        raise RuntimeError("full-slide decoded label hash differs from the cache")

    model = SlidingArmBModel(extractor, arm_b).to(device)
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        paired_scores = slide_inference(
            optical.unsqueeze(0).to(device),
            model,
            dsm=sar.unsqueeze(0).to(device),
            n_output_channels=NUM_CLASSES * 2,
            crop_size=(CROP_SIZE, CROP_SIZE),
            stride=INFERENCE_STRIDE,
            batch_size=inference_batch_size,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if paired_scores.shape[1] != NUM_CLASSES * 2:
        raise RuntimeError("paired full-slide output does not have 14 channels")
    base_full = paired_scores[:, :NUM_CLASSES].contiguous()
    corrected_full = paired_scores[:, NUM_CLASSES:].contiguous()
    y0, y1, x0, x1 = (int(value) for value in phase_record.bounds_yxyx)
    base = base_full[:, :, y0:y1, x0:x1].contiguous()
    corrected = corrected_full[:, :, y0:y1, x0:x1].contiguous()
    aggregated_delta = (corrected - base).contiguous()

    teacher_array = np.load(
        phase_record.teacher_logits_path, mmap_mode="r", allow_pickle=False
    )
    e0_array = np.load(
        companion_record.logits_path, mmap_mode="r", allow_pickle=False
    )
    structure_array = np.load(
        phase_record.structure_mask_path, mmap_mode="r", allow_pickle=False
    )
    if numpy_array_sha256(teacher_array) != phase_record.teacher_logits_array_sha256:
        raise RuntimeError("full-slide teacher array SHA differs")
    if numpy_array_sha256(e0_array) != companion_record.logits_array_sha256:
        raise RuntimeError("full-slide E0 companion array SHA differs")
    if numpy_array_sha256(structure_array) != phase_record.structure_mask_array_sha256:
        raise RuntimeError("full-slide structure array SHA differs")
    teacher = torch.from_numpy(
        np.asarray(teacher_array, dtype=np.float32).copy()
    ).unsqueeze(0)
    cached_e0 = torch.from_numpy(
        np.asarray(e0_array, dtype=np.float32).copy()
    ).unsqueeze(0)
    labels = torch.from_numpy(label_full[y0:y1, x0:x1].copy()).unsqueeze(0)
    expected_shape = teacher.shape
    if cached_e0.shape != expected_shape or base.shape != expected_shape:
        raise RuntimeError("teacher, cached E0, and live E0 common shapes differ")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (teacher, cached_e0, base, corrected, aggregated_delta)
    ):
        raise RuntimeError("full-slide teacher/E0/correction tensors must be finite")

    live_prediction = base.argmax(dim=1)
    cached_prediction = cached_e0.argmax(dim=1)
    prediction_equal = torch.equal(live_prediction, cached_prediction)
    float16_roundtrip_equal = torch.equal(
        base.to(dtype=torch.float16), cached_e0.to(dtype=torch.float16)
    )
    cached_confusion = confusion_from_arrays(
        cached_prediction[0].numpy(), labels[0].numpy(), NUM_CLASSES
    )
    live_confusion = confusion_from_arrays(
        live_prediction[0].numpy(), labels[0].numpy(), NUM_CLASSES
    )
    confusion_equal = np.array_equal(cached_confusion, live_confusion)
    difference = (base - cached_e0).abs()
    centered_difference = base - cached_e0
    centered_difference = centered_difference - centered_difference.mean(
        dim=1, keepdim=True
    )
    rounded_mismatch = base.to(dtype=torch.float16).ne(
        cached_e0.to(dtype=torch.float16)
    )
    base_validation = {
        "float16_roundtrip_logits_equal": float16_roundtrip_equal,
        "float16_roundtrip_mismatch_values": int(rounded_mismatch.sum().item()),
        "float16_roundtrip_mismatch_fraction": float(
            rounded_mismatch.to(dtype=torch.float32).mean().item()
        ),
        "prediction_equal": prediction_equal,
        "confusion_equal": confusion_equal,
        "cached_prediction_sha256": tensor_sha256(cached_prediction),
        "live_prediction_sha256": tensor_sha256(live_prediction),
        "cached_confusion": cached_confusion.tolist(),
        "live_confusion": live_confusion.tolist(),
        "logit_max_abs_difference_from_float16_cache": float(difference.max()),
        "logit_mean_abs_difference_from_float16_cache": float(difference.mean()),
        "centered_logit_max_abs_difference": float(centered_difference.abs().max()),
        "centered_logit_mean_abs_difference": float(centered_difference.abs().mean()),
        "centered_logit_rms_difference": float(
            torch.sqrt(centered_difference.square().mean())
        ),
    }
    if not float16_roundtrip_equal or not prediction_equal or not confusion_equal:
        raise RuntimeError(
            "live normal-phase 512/341 E0 does not exactly reproduce the "
            "float16 companion cache: "
            + json.dumps(base_validation, ensure_ascii=False, sort_keys=True)
        )

    behavior = phase_residual_behavior_statistics(
        cached_e0, teacher, corrected, labels
    )
    ground_truth_support = [
        int((labels == class_index).sum().item()) for class_index in range(NUM_CLASSES)
    ]
    if any(count <= 0 for count in ground_truth_support):
        raise RuntimeError(
            "the one-image common bounds do not contain all seven WHU classes; "
            "refuse an ambiguous partial-class mIoU resource gate"
        )
    gate = phase_residual_probe_gate(behavior, GATE_THRESHOLDS)
    small_np, thin_np = _common_structure_crop(
        structure_array,
        phase_record.structure_mask_bounds_yxyx,
        phase_record.bounds_yxyx,
    )
    small = torch.from_numpy(np.ascontiguousarray(small_np)).unsqueeze(0)
    thin = torch.from_numpy(np.ascontiguousarray(thin_np)).unsqueeze(0)
    valid = labels.ge(0) & labels.lt(NUM_CLASSES)
    e0_prediction = cached_e0.argmax(dim=1)
    teacher_prediction = teacher.argmax(dim=1)
    corrected_prediction = corrected.argmax(dim=1)
    both_wrong = (
        valid
        & e0_prediction.ne(labels)
        & teacher_prediction.ne(labels)
    )
    both_wrong_report = {
        "pixels": int(both_wrong.sum()),
        "corrected_to_ground_truth": int(
            (both_wrong & corrected_prediction.eq(labels)).sum()
        ),
        "wrong_class_changed": int(
            (
                both_wrong
                & corrected_prediction.ne(labels)
                & corrected_prediction.ne(e0_prediction)
            ).sum()
        ),
    }
    return {
        "scope": "same cached training image; exact matched normal 512/341 sliding path",
        "sample": {
            "index": int(phase_record.index),
            "sample_name": phase_record.sample_name,
            "full_shape_hw": list(full_shape),
            "common_bounds_yxyx": list(phase_record.bounds_yxyx),
        },
        "operator": {
            "phase": [0, 0],
            "crop_size_hw": [CROP_SIZE, CROP_SIZE],
            "stride_hw": list(INFERENCE_STRIDE),
            "count_normalization": True,
            "per_crop_delta_centered_across_classes": True,
            "arm_b_step": PROBE_STEPS,
        },
        "base_validation": base_validation,
        "metric_identity": {
            "ground_truth_support_pixels_per_class": ground_truth_support,
            "all_seven_ground_truth_classes_present": True,
            "miou_equivalence": (
                "fixed GT-support and the repository union>0 class set are "
                "identical because all seven classes have ground-truth support"
            ),
        },
        "behavior": behavior,
        "resource_gate": gate,
        "regions": {
            "small_component": region_change_statistics(
                cached_e0, corrected, labels, small
            ),
            "thin_component": region_change_statistics(
                cached_e0, corrected, labels, thin
            ),
            "both_teacher_and_e0_wrong": both_wrong_report,
        },
        "tensor_sha256": {
            "live_e0_logits_float32": tensor_sha256(base),
            "cached_e0_logits_loaded_float32": tensor_sha256(cached_e0),
            "teacher_logits_loaded_float32": tensor_sha256(teacher),
            "aggregated_centered_delta_float32": tensor_sha256(aggregated_delta),
            "corrected_logits_float32": tensor_sha256(corrected),
            "corrected_prediction": tensor_sha256(corrected_prediction),
            "label": tensor_sha256(labels),
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        },
        "class_names": list(class_names),
    }


def build_probe_protocol(
    args: argparse.Namespace,
    *,
    baseline_sha256: str,
    manifest_metadata: Mapping[str, Any],
    companion_manifest: Mapping[str, Any],
    reference: Mapping[str, Any],
    initial_branch: Mapping[str, Any],
    fixed_batch: Mapping[str, Any],
    target_audit: Mapping[str, Any],
) -> dict[str, Any]:
    companion_protocol = companion_manifest.get("protocol")
    if not isinstance(companion_protocol, Mapping):
        raise RuntimeError("E0-slide companion manifest lacks its protocol")
    companion_protocol_sha = canonical_json_sha256(companion_protocol)
    if companion_manifest.get("protocol_sha256") != companion_protocol_sha:
        raise RuntimeError("E0-slide companion protocol hash is inconsistent")
    companion_execution = companion_manifest.get("execution")
    if not isinstance(companion_execution, Mapping):
        raise RuntimeError("E0-slide companion manifest lacks execution metadata")
    if int(companion_execution.get("inference_batch_size", -1)) != int(
        args.inference_batch_size
    ):
        raise RuntimeError(
            "live full-slide inference batch size must match the companion cache"
        )
    teacher_manifest = _json_payload(args.teacher_manifest)
    teacher_execution = teacher_manifest.get("execution")
    if not isinstance(teacher_execution, Mapping):
        raise RuntimeError("teacher manifest lacks execution metadata")
    return {
        "artifact_type": ARTIFACT_TYPE,
        "protocol_version": PROTOCOL_VERSION,
        "name": "WHU matched-operator phase-residual P2 probe V2",
        "git_commit": git_commit(),
        "evidence_scope": (
            "Fixed one-image exploratory capacity/selectivity screen. A PASS only "
            "permits a separately pre-registered short-training experiment; it is "
            "not a test-set gain, generalization result, phase-causal proof, or "
            "unlabeled routing result."
        ),
        "paper_confirmation_requirement": (
            "If this route enters a paper, rebuild a clean train/validation/test "
            "protocol and rerun all method selection and comparisons from scratch."
        ),
        "baseline_checkpoint": {
            "path": str(args.baseline_checkpoint.resolve()),
            "sha256": baseline_sha256,
        },
        "immutable_inputs": {
            "teacher_manifest": manifest_metadata["teacher_manifest"],
            "teacher_manifest_sha256": manifest_metadata[
                "teacher_manifest_sha256"
            ],
            "teacher_protocol_sha256": manifest_metadata[
                "teacher_protocol_sha256"
            ],
            "structure_manifest": manifest_metadata["structure_manifest"],
            "structure_manifest_sha256": manifest_metadata[
                "structure_manifest_sha256"
            ],
            "e0_slide_manifest": str(args.e0_slide_manifest.resolve()),
            "e0_slide_manifest_sha256": file_sha256(args.e0_slide_manifest),
            "e0_slide_protocol_sha256": companion_protocol_sha,
            "reference_objective": {
                key: value
                for key, value in reference.items()
                if key not in {"fixed_batch", "initial_branch_metadata"}
            },
        },
        "matched_operator": {
            "teacher": (
                "four independently count-normalized 512/341 full-image phases, "
                "aligned and arithmetic-mean fused inside common bounds"
            ),
            "baseline": (
                "normal (0,0) full-image 512/341 sliding inference with identical "
                "count normalization and common bounds"
            ),
            "target": "per-pixel class-centered T_slide - E0_slide",
            "forbidden_old_target": (
                "T_slide - E0_crop is not used because it mixes phase correction "
                "with slide/crop context"
            ),
            "companion_protocol": dict(companion_protocol),
            "inference_batch_sizes": {
                "cached_teacher": int(
                    teacher_execution.get("inference_batch_size", -1)
                ),
                "cached_e0_slide_companion": int(
                    companion_execution["inference_batch_size"]
                ),
                "live_step100_arm_b": int(args.inference_batch_size),
            },
        },
        "fixed_batch": dict(fixed_batch),
        "target_and_masks": dict(target_audit),
        "branch": {
            "input": "frozen normal-phase final decoder P2 from the fixed crop",
            "architecture": "unchanged V1 PhaseCorrectionBranch",
            "hidden_channels": int(args.hidden_channels),
            "initial": dict(initial_branch),
            "predicted_delta": (
                "bilinear upsample to 512x512 with align_corners=False, then "
                "per-pixel center across classes"
            ),
        },
        "paired_arms": {
            "Arm_A_fix_only": "L_fix",
            "Arm_B_fix_keep": "L_fix + L_keep",
            "M_fix": (
                "valid AND argmax(T_slide)==GT AND argmax(E0_slide)!=GT"
            ),
            "M_keep": "valid AND NOT M_fix",
            "group_weighting": (
                "L_fix and L_keep are independently averaged over their own "
                "pixels x classes; Arm B uses an explicit 1:1 sum"
            ),
            "paired_identity": (
                "same fixed batch, deep-copied exact initial state, independent "
                "fresh optimizers with identical hyperparameters"
            ),
        },
        "loss": {
            "scale": (
                "one fixed float32-target RMS over M_fix x classes, accumulated "
                "in float64 before step 0; shared by both arms"
            ),
            "smooth_l1_beta_in_scaled_units": SMOOTH_L1_BETA,
            "target_scale_sweep": False,
            "additional_CE_Dice_KL": False,
        },
        "optimization": {
            "steps": PROBE_STEPS,
            "optimizer": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "scheduler": None,
            "batch_reuse": "the exact same fixed batch at all 100 steps",
            "best_step_selection": False,
            "decision_step": PROBE_STEPS,
            "seed": int(args.seed),
            "branch_seed": int(args.seed + BRANCH_SEED_OFFSET),
        },
        "evaluation": {
            "fixed_crop_proxy": (
                "E0_slide_crop + centered delta; diagnostic only and never the "
                "final GO gate"
            ),
            "hard_behavior_gate": (
                "step-100 Arm B on the same one-image exact normal-phase 512/341 "
                "full-slide path"
            ),
            "thresholds": asdict(GATE_THRESHOLDS),
            "inference_batch_size": int(args.inference_batch_size),
            "live_e0_cache_validity": (
                "live normal-phase logits converted to float16 must be exactly "
                "equal to the companion array; prediction and confusion must "
                "also be exactly equal"
            ),
            "hard_checks": [
                "non-empty M_fix",
                "routed repair rate >= 50%",
                "broken E0-correct pixels <= 25% of routed fixed pixels",
                "7-class mIoU strictly greater than matched E0_slide",
            ],
            "arm_a_diagnostic": (
                "R_fix_A = 1 - final L_fix / initial L_fix >= 50%; newly "
                "registered resource heuristic, not the old KL 50% gate"
            ),
            "report_only": [
                "R_fix_B",
                "leakage mean/p95/p99",
                "gradient norms and cosine",
                "net fixed-minus-broken",
                "per-class, small, thin, and both-wrong behavior",
                "fixed-crop proxy mIoU",
            ],
        },
    }


def final_resource_decision(
    full_slide_gate: Mapping[str, Any], arm_a_capacity_pass: bool
) -> dict[str, Any]:
    behavior_pass = full_slide_gate.get("passes") is True
    return {
        "outcome": (
            "ELIGIBLE_FOR_SEPARATELY_PREREGISTERED_SHORT_TRAINING"
            if behavior_pass
            else "RESOURCE_NO_GO_FOR_CURRENT_P2_MATCHED_RESIDUAL_PROBE"
        ),
        "behavior_gate_pass": behavior_pass,
        "arm_a_capacity_diagnostic_pass": bool(arm_a_capacity_pass),
        "arm_a_is_not_an_additional_behavior_gate": True,
        "interpretation": (
            "Step-100 Arm B showed sufficient same-image matched-slide repair, "
            "selectivity, and mIoU behavior to justify a separate short-training "
            "screen. This is not evidence of generalization or phase causality."
            if behavior_pass
            else (
                "Under the predeclared 100-step budget, the current P2 branch and "
                "matched residual objective did not satisfy the same-image exact "
                "full-slide resource gate. This is a resource No-Go for this "
                "candidate, not a theorem about all phase or high-resolution routes."
            )
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the WHU phase-residual probe")
    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    baseline_sha256 = file_sha256(args.baseline_checkpoint)

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
    phase_records, manifest_metadata = build_phase_records(
        args.teacher_manifest,
        args.structure_mask_manifest,
        source_dataset,
        baseline_sha256,
        require_full_split=False,
    )
    companion_records = load_e0_slide_companion_records(
        args.e0_slide_manifest,
        args.teacher_manifest,
        expected_checkpoint_sha256=baseline_sha256,
        verify_artifacts=True,
    )
    validate_companion_alignment(phase_records, companion_records)
    companion_manifest = _json_payload(args.e0_slide_manifest)
    reference = validate_reference_objective(
        args.reference_objective_summary,
        baseline_sha256=baseline_sha256,
        teacher_manifest_sha256=manifest_metadata["teacher_manifest_sha256"],
        structure_manifest_sha256=manifest_metadata["structure_manifest_sha256"],
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    dataset = WHUPhaseCropDataset(
        phase_records,
        seed=args.seed,
        length=args.train_samples_per_epoch,
        source_cache_size=args.source_cache_size,
    )
    dataset.set_epoch(0)
    loader_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=loader_generator,
    )
    fixed_batch = next(iter(loader))
    old_batch_hashes = batch_input_hashes(fixed_batch)
    validate_fixed_batch_identity(old_batch_hashes, reference["fixed_batch"])
    e0_slide_batch = companion_crops_for_batch(
        fixed_batch,
        companion_records,
        array_cache=SmallArrayCache(max(args.source_cache_size, 16)),
    )

    device = torch.device(args.device)
    e0.to(device).eval()
    extractor = FrozenE0P2Extractor(e0)
    optical = fixed_batch["optical"].to(device)
    sar = fixed_batch["sar"].to(device)
    labels = fixed_batch["label"].to(device)
    teacher = fixed_batch["teacher_logits"].to(device, dtype=torch.float32)
    e0_slide = e0_slide_batch.to(device, dtype=torch.float32)
    with torch.no_grad():
        e0_crop, p2 = extractor(optical, sar)
    if e0_crop.requires_grad or p2.requires_grad:
        raise AssertionError("frozen E0 fixed-batch tensors require gradients")
    if teacher.shape != e0_slide.shape or labels.shape != teacher.shape[:1] + teacher.shape[2:]:
        raise RuntimeError("fixed teacher/E0-slide/label tensor shapes differ")

    target = matched_phase_residual_target(teacher, e0_slide)
    fix_mask, keep_mask = matched_residual_masks(teacher, e0_slide, labels)
    valid = labels.ge(0) & labels.lt(NUM_CLASSES)
    if not torch.equal(fix_mask | keep_mask, valid):
        raise AssertionError("M_fix and M_keep do not cover every valid pixel")
    if bool((fix_mask & keep_mask).any()):
        raise AssertionError("M_fix and M_keep overlap")
    fix_pixels = int(fix_mask.sum().item())
    keep_pixels = int(keep_mask.sum().item())
    if fix_pixels <= 0 or keep_pixels <= 0:
        raise RuntimeError("residual probe requires non-empty fix and keep masks")
    scale = fix_mask_rms_scale(target, fix_mask)
    # Match fix_mask_rms_scale's defensive float32 re-centering before the
    # report; both RMS reductions then consume the same CPU-float64 values.
    target_statistics = magnitude_statistics(
        center_class_logits(target.detach().to(dtype=torch.float32)), fix_mask
    )
    if not math.isclose(
        target_statistics["rms"], scale, rel_tol=0.0, abs_tol=1e-15
    ):
        raise AssertionError("reported target RMS differs from the fixed loss scale")

    branch_seed = args.seed + BRANCH_SEED_OFFSET
    set_seed(branch_seed)
    p2_channels = int(getattr(e0.decoder, "out_channels", 256))
    arm_a = PhaseCorrectionBranch(
        in_channels=p2_channels,
        hidden_channels=args.hidden_channels,
        num_classes=NUM_CLASSES,
    ).to(device)
    arm_b = copy.deepcopy(arm_a).to(device)
    initial_a_metadata = branch_metadata(arm_a)
    initial_b_metadata = branch_metadata(arm_b)
    if initial_a_metadata != initial_b_metadata:
        raise AssertionError("paired residual branches do not start identically")
    if initial_a_metadata != reference["initial_branch_metadata"]:
        raise RuntimeError("residual branch initialization differs from the V1 anchor")

    optimizer_a = torch.optim.AdamW(
        arm_a.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    optimizer_b = torch.optim.AdamW(
        arm_b.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if optimizer_a.state or optimizer_b.state:
        raise AssertionError("fresh AdamW optimizers unexpectedly contain state")

    initial_a = residual_loss_snapshot(
        arm_a,
        p2=p2,
        reference_logits=e0_slide,
        target=target,
        fix_mask=fix_mask,
        keep_mask=keep_mask,
        scale=scale,
    )
    initial_b = residual_loss_snapshot(
        arm_b,
        p2=p2,
        reference_logits=e0_slide,
        target=target,
        fix_mask=fix_mask,
        keep_mask=keep_mask,
        scale=scale,
    )
    if initial_a != initial_b:
        raise AssertionError("paired residual branches have different initial losses")
    initial_behavior, initial_delta = behavior_for_branch(
        arm_b,
        p2=p2,
        e0_slide_logits=e0_slide,
        teacher_slide_logits=teacher,
        labels=labels,
    )
    if torch.count_nonzero(initial_delta).item() != 0:
        raise AssertionError("residual branch is not exactly zero at initialization")
    if initial_behavior["corrected"]["confusion"] != initial_behavior["e0"]["confusion"]:
        raise AssertionError("zero residual does not reproduce fixed-crop E0-slide")

    fixed_batch_audit = {
        **old_batch_hashes,
        "e0_slide_logits_sha256": tensor_sha256(e0_slide_batch),
        "e0_crop_logits_sha256": tensor_sha256(e0_crop),
        "p2_sha256": tensor_sha256(p2),
        "matched_target_sha256": tensor_sha256(target),
        "fix_mask_sha256": tensor_sha256(fix_mask),
        "keep_mask_sha256": tensor_sha256(keep_mask),
    }
    target_audit = {
        "formula": "center_per_pixel_across_classes(T_slide - E0_slide)",
        "target_dtype": str(target.dtype),
        "target_sha256": tensor_sha256(target),
        "fix_mask_sha256": tensor_sha256(fix_mask),
        "keep_mask_sha256": tensor_sha256(keep_mask),
        "valid_pixels": int(valid.sum().item()),
        "fix_pixels": fix_pixels,
        "keep_pixels": keep_pixels,
        "partition_is_exact": True,
        "scale_rule": "RMS over centered target on M_fix x classes",
        "scale": scale,
        "magnitude_statistics_report_only": target_statistics,
    }
    protocol = build_probe_protocol(
        args,
        baseline_sha256=baseline_sha256,
        manifest_metadata=manifest_metadata,
        companion_manifest=companion_manifest,
        reference=reference,
        initial_branch=initial_a_metadata,
        fixed_batch=fixed_batch_audit,
        target_audit=target_audit,
    )
    protocol_sha = canonical_json_sha256(protocol)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        args.output_dir / "run_config.json",
        {
            "status": "CONFIGURED_BEFORE_STEP_0",
            "artifact_type": ARTIFACT_TYPE,
            "protocol": protocol,
            "protocol_sha256": protocol_sha,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
        },
    )
    write_json_atomic(
        args.output_dir / "initial_state.json",
        {
            "fixed_batch": fixed_batch_audit,
            "target_audit": target_audit,
            "initial_branch_metadata": {
                "Arm_A": initial_a_metadata,
                "Arm_B": initial_b_metadata,
            },
            "initial_losses": {"Arm_A": initial_a, "Arm_B": initial_b},
            "initial_fixed_crop_proxy": initial_behavior,
            "initial_gradient_interaction_Arm_B": gradient_interaction(
                arm_b,
                p2=p2,
                reference_logits=e0_slide,
                target=target,
                fix_mask=fix_mask,
                keep_mask=keep_mask,
                scale=scale,
            ),
        },
    )

    print(
        json.dumps(
            {
                "status": "SEALED",
                "protocol_sha256": protocol_sha,
                "steps": PROBE_STEPS,
                "fix_pixels": fix_pixels,
                "keep_pixels": keep_pixels,
                "scale": scale,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    for step in range(1, PROBE_STEPS + 1):
        arm_a.train()
        optimizer_a.zero_grad(set_to_none=True)
        _, a_losses = residual_loss_tensors(
            arm_a,
            p2=p2,
            reference_logits=e0_slide,
            target=target,
            fix_mask=fix_mask,
            keep_mask=keep_mask,
            scale=scale,
        )
        if not torch.isfinite(a_losses["fix"]):
            raise RuntimeError(f"non-finite Arm A loss at step {step}")
        a_losses["fix"].backward()
        optimizer_a.step()

        arm_b.train()
        optimizer_b.zero_grad(set_to_none=True)
        _, b_losses = residual_loss_tensors(
            arm_b,
            p2=p2,
            reference_logits=e0_slide,
            target=target,
            fix_mask=fix_mask,
            keep_mask=keep_mask,
            scale=scale,
        )
        if not torch.isfinite(b_losses["total"]):
            raise RuntimeError(f"non-finite Arm B loss at step {step}")
        b_losses["total"].backward()
        optimizer_b.step()
        record: dict[str, Any] = {
            "step": step,
            "Arm_A_used_for_update": {
                "fix": float(a_losses["fix"].detach()),
            },
            "Arm_B_used_for_update": {
                name: float(value.detach()) for name, value in b_losses.items()
            },
        }
        if step in {1, 10, PROBE_STEPS}:
            record["post_update"] = {
                "Arm_A": residual_loss_snapshot(
                    arm_a,
                    p2=p2,
                    reference_logits=e0_slide,
                    target=target,
                    fix_mask=fix_mask,
                    keep_mask=keep_mask,
                    scale=scale,
                ),
                "Arm_B": residual_loss_snapshot(
                    arm_b,
                    p2=p2,
                    reference_logits=e0_slide,
                    target=target,
                    fix_mask=fix_mask,
                    keep_mask=keep_mask,
                    scale=scale,
                ),
            }
        _append_jsonl(args.output_dir / "steps.jsonl", record)
        if step in {1, 10, PROBE_STEPS}:
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
    torch.cuda.synchronize(device)
    training_runtime = {
        "elapsed_seconds": time.perf_counter() - training_started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
    }

    final_a = residual_loss_snapshot(
        arm_a,
        p2=p2,
        reference_logits=e0_slide,
        target=target,
        fix_mask=fix_mask,
        keep_mask=keep_mask,
        scale=scale,
    )
    final_b = residual_loss_snapshot(
        arm_b,
        p2=p2,
        reference_logits=e0_slide,
        target=target,
        fix_mask=fix_mask,
        keep_mask=keep_mask,
        scale=scale,
    )
    r_fix_a = 1.0 - final_a["fix"] / initial_a["fix"]
    r_fix_b = 1.0 - final_b["fix"] / initial_b["fix"]
    arm_a_capacity_pass = r_fix_a >= 0.50
    proxy_a, delta_a = behavior_for_branch(
        arm_a,
        p2=p2,
        e0_slide_logits=e0_slide,
        teacher_slide_logits=teacher,
        labels=labels,
    )
    proxy_b, delta_b = behavior_for_branch(
        arm_b,
        p2=p2,
        e0_slide_logits=e0_slide,
        teacher_slide_logits=teacher,
        labels=labels,
    )
    final_gradient = gradient_interaction(
        arm_b,
        p2=p2,
        reference_logits=e0_slide,
        target=target,
        fix_mask=fix_mask,
        keep_mask=keep_mask,
        scale=scale,
    )
    fixed_crop_result = {
        "scope": "diagnostic proxy only; not used by the final resource gate",
        "Arm_A": {
            "initial_losses": initial_a,
            "final_losses": final_a,
            "R_fix": r_fix_a,
            "R_fix_at_least_50_percent": arm_a_capacity_pass,
            "behavior": proxy_a,
            "keep_leakage_in_scale_units": magnitude_statistics(
                delta_a, keep_mask, divisor=scale
            ),
        },
        "Arm_B": {
            "initial_losses": initial_b,
            "final_losses": final_b,
            "R_fix_report_only": r_fix_b,
            "behavior": proxy_b,
            "keep_leakage_in_scale_units_report_only": magnitude_statistics(
                delta_b, keep_mask, divisor=scale
            ),
            "final_gradient_interaction_report_only": final_gradient,
        },
    }
    write_json_atomic(args.output_dir / "fixed_crop_result.json", fixed_crop_result)

    checkpoint_path = args.output_dir / "branches_step100.pth"
    _atomic_torch_save(
        checkpoint_path,
        {
            "artifact_type": ARTIFACT_TYPE,
            "protocol_sha256": protocol_sha,
            "step": PROBE_STEPS,
            "baseline_checkpoint_sha256": baseline_sha256,
            "Arm_A_state_dict": {
                name: value.detach().cpu() for name, value in arm_a.state_dict().items()
            },
            "Arm_B_state_dict": {
                name: value.detach().cpu() for name, value in arm_b.state_dict().items()
            },
        },
    )

    # Fixed-batch tensors are no longer needed; release them before full-slide inference.
    del (
        optical,
        sar,
        e0_crop,
        p2,
        teacher,
        e0_slide,
        target,
        fix_mask,
        keep_mask,
        initial_delta,
        delta_a,
        delta_b,
        fixed_batch,
        e0_slide_batch,
        loader,
        dataset,
    )
    torch.cuda.empty_cache()
    full_slide = exact_full_slide_evaluation(
        phase_record=phase_records[0],
        companion_record=companion_records[phase_records[0].index],
        extractor=extractor,
        arm_b=arm_b,
        device=device,
        inference_batch_size=args.inference_batch_size,
        class_names=cfg["labels"],
    )
    write_json_atomic(args.output_dir / "full_slide_result.json", full_slide)

    e0_after = e0_state_fingerprints(e0)
    e0_audit = {
        "before": e0_before,
        "after": e0_after,
        "unchanged": e0_before == e0_after,
        "all_parameters_frozen": not any(
            parameter.requires_grad for parameter in e0.parameters()
        ),
        "all_parameter_gradients_absent": not any(
            parameter.grad is not None for parameter in e0.parameters()
        ),
        "e0_training_flag": e0.training,
    }
    if not all(
        (
            e0_audit["unchanged"],
            e0_audit["all_parameters_frozen"],
            e0_audit["all_parameter_gradients_absent"],
            not e0_audit["e0_training_flag"],
        )
    ):
        raise AssertionError(f"frozen E0 audit failed: {e0_audit}")
    decision = final_resource_decision(
        full_slide["resource_gate"], arm_a_capacity_pass
    )
    summary = {
        "status": "PASS",
        "status_meaning": "execution and all validity checks completed",
        "artifact_type": ARTIFACT_TYPE,
        "protocol_sha256": protocol_sha,
        "steps": PROBE_STEPS,
        "fixed_batch": fixed_batch_audit,
        "target_audit": target_audit,
        "paired_initial_state_equal": initial_a_metadata == initial_b_metadata,
        "initial_branch_metadata": initial_a_metadata,
        "final_branch_metadata": {
            "Arm_A": branch_metadata(arm_a),
            "Arm_B": branch_metadata(arm_b),
        },
        "fixed_crop_result": fixed_crop_result,
        "full_slide_result": full_slide,
        "decision": decision,
        "e0_state_audit": e0_audit,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": file_sha256(checkpoint_path),
        },
        "runtime": {
            "training": training_runtime,
            "full_slide": full_slide["runtime"],
        },
    }
    summary_path = args.output_dir / "summary.json"
    write_json_atomic(summary_path, summary)
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True), flush=True)
    print(f"phase_residual_result={summary_path.resolve()}", flush=True)
    print("phase_residual_execution_status=PASS", flush=True)
    print(f"phase_residual_decision={decision['outcome']}", flush=True)


if __name__ == "__main__":
    main()
