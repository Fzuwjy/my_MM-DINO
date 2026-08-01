"""Train only the V4-C WHU optical-stem candidate against a sealed V4-A base.

The clean baseline is the existing ``mask-ignore`` V4-A run.  This runner does
not retrain it.  Before training, it constructs an in-memory clean model and
the ``mask-ignore+optical-stem`` candidate from the same seed, verifies every
shared parameter bitwise, verifies a fixed eval-mode step-0 prediction probe,
then releases the clean model and trains only the candidate.

The scheduler horizon is always 50 epochs.  ``--stop-after-epoch`` is only a
screen stop and never shortens the cosine schedule.  Formal use is locked to a
single torchrun process to preserve the accepted WHU pairing contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from configs import get_cfg  # noqa: E402
from scripts import run_whu_v4_a_screen as a_runner  # noqa: E402
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.utils import set_seed  # noqa: E402


NUM_CLASSES = a_runner.NUM_CLASSES
IGNORE_INDEX = a_runner.IGNORE_INDEX
PROTOCOL_EPOCHS = a_runner.PROTOCOL_EPOCHS
DEFAULT_EVALUATION_EPOCHS = a_runner.DEFAULT_EVALUATION_EPOCHS
BACKBONE_TYPE = a_runner.BACKBONE_TYPE
CANDIDATE_VARIANT = "mask-ignore+optical-stem"
STEM_PARAMETER_PREFIX = "decoder.optical_stem."
DEFAULT_STEM_SEED = 104771
DEFAULT_PROBE_SIZE = 64


def tensor_bitwise_sha256(tensor: torch.Tensor) -> str:
    """Hash dtype, shape, and exact tensor bytes."""

    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def keyed_tensor_records(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "sha256": tensor_bitwise_sha256(tensor),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in sorted(tensors.items())
    }


def audit_shared_initialization(
    clean_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
) -> dict[str, Any]:
    """Require bitwise identity for every parameter shared by clean and C."""

    clean_parameters = dict(clean_model.named_parameters())
    candidate_parameters = dict(candidate_model.named_parameters())
    clean_names = set(clean_parameters)
    candidate_names = set(candidate_parameters)
    missing_from_candidate = sorted(clean_names - candidate_names)
    candidate_only = sorted(candidate_names - clean_names)
    unexpected_candidate_only = [
        name for name in candidate_only if not name.startswith(STEM_PARAMETER_PREFIX)
    ]
    if missing_from_candidate:
        raise RuntimeError(
            f"C initialization lost clean parameters: {missing_from_candidate}"
        )
    if not candidate_only:
        raise RuntimeError("C initialization has no optical-stem parameters")
    if unexpected_candidate_only:
        raise RuntimeError(
            "C initialization added parameters outside the optical stem: "
            f"{unexpected_candidate_only}"
        )

    per_key: dict[str, dict[str, Any]] = {}
    mismatched: list[str] = []
    for name in sorted(clean_names):
        clean_value = clean_parameters[name]
        candidate_value = candidate_parameters[name]
        equal = bool(torch.equal(clean_value, candidate_value))
        clean_sha = tensor_bitwise_sha256(clean_value)
        candidate_sha = tensor_bitwise_sha256(candidate_value)
        per_key[name] = {
            "clean_sha256": clean_sha,
            "candidate_sha256": candidate_sha,
            "bitwise_equal": equal,
            "shape": list(clean_value.shape),
            "dtype": str(clean_value.dtype),
        }
        if not equal or clean_sha != candidate_sha:
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(f"C shared initialization differs: {mismatched}")

    clean_buffers = dict(clean_model.named_buffers())
    candidate_buffers = dict(candidate_model.named_buffers())
    shared_buffer_names = sorted(set(clean_buffers) & set(candidate_buffers))
    buffer_mismatches = [
        name
        for name in shared_buffer_names
        if not torch.equal(clean_buffers[name], candidate_buffers[name])
    ]
    if buffer_mismatches:
        raise RuntimeError(f"C shared buffers differ: {buffer_mismatches}")

    return {
        "contract": "per-key shared tensors, never full-model SHA equality",
        "shared_parameter_count": len(per_key),
        "shared_parameters_bitwise_equal": True,
        "shared_parameter_mismatches": [],
        "shared_parameter_records": per_key,
        "candidate_only_parameter_names": candidate_only,
        "candidate_only_parameter_records": keyed_tensor_records(
            {name: candidate_parameters[name] for name in candidate_only}
        ),
        "shared_buffer_count": len(shared_buffer_names),
        "shared_buffers_bitwise_equal": True,
        "shared_buffer_mismatches": [],
    }


def audit_optimizer_membership(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Verify that every trainable stem parameter occurs in exactly one group."""

    named_parameters = dict(model.named_parameters())
    name_by_id = {id(parameter): name for name, parameter in named_parameters.items()}
    counts = {name: 0 for name in named_parameters}
    unknown_parameter_ids: list[int] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = name_by_id.get(id(parameter))
            if name is None:
                unknown_parameter_ids.append(id(parameter))
            else:
                counts[name] += 1

    stem_names = sorted(
        name
        for name, parameter in named_parameters.items()
        if name.startswith(STEM_PARAMETER_PREFIX) and parameter.requires_grad
    )
    if not stem_names:
        raise RuntimeError("candidate optimizer audit found no trainable optical stem")
    stem_counts = {name: counts[name] for name in stem_names}
    invalid = {name: count for name, count in stem_counts.items() if count != 1}
    if unknown_parameter_ids:
        raise RuntimeError("optimizer contains parameters not owned by the candidate model")
    if invalid:
        raise RuntimeError(
            f"optical-stem optimizer membership is not exactly once: {invalid}"
        )
    return {
        "stem_trainable_parameter_names": stem_names,
        "stem_optimizer_membership_count_by_name": stem_counts,
        "stem_parameters_present_exactly_once": True,
        "optimizer_group_count": len(optimizer.param_groups),
    }


def fixed_probe_inputs(size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    optical_count = 3 * size * size
    optical = torch.arange(optical_count, dtype=torch.float32).reshape(
        1, 3, size, size
    )
    optical = ((optical % 257.0) / 128.0) - 1.0
    sar = torch.arange(size * size, dtype=torch.float32).reshape(1, 1, size, size)
    sar = (sar % 251.0) / 250.0
    return optical.to(device), sar.to(device)


def audit_step0_probe(
    clean_model: torch.nn.Module,
    candidate_model: torch.nn.Module,
    *,
    device: torch.device,
    size: int,
) -> dict[str, Any]:
    """Require exact eval predictions before C receives any optimizer step."""

    clean_model.eval()
    candidate_model.eval()
    optical, sar = fixed_probe_inputs(size, device)
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    try:
        with torch.inference_mode():
            clean_prediction = clean_model(optical, sar)
            candidate_prediction = candidate_model(optical, sar)
    finally:
        # The probe is an audit, not part of training.  Restore even if a future
        # model implementation introduces an eval-time random operation.
        torch.set_rng_state(cpu_rng_before)
        if cuda_rng_before:
            torch.cuda.set_rng_state_all(cuda_rng_before)
    cpu_rng_after = torch.get_rng_state()
    cuda_rng_after = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )
    cpu_rng_equal = bool(torch.equal(cpu_rng_before, cpu_rng_after))
    cuda_rng_equal = len(cuda_rng_before) == len(cuda_rng_after) and all(
        torch.equal(before, after)
        for before, after in zip(cuda_rng_before, cuda_rng_after, strict=True)
    )
    if not cpu_rng_equal or not cuda_rng_equal:
        raise RuntimeError("C step-0 probe failed to restore CPU/CUDA RNG state")
    equal = bool(torch.equal(clean_prediction, candidate_prediction))
    clean_sha = tensor_bitwise_sha256(clean_prediction)
    candidate_sha = tensor_bitwise_sha256(candidate_prediction)
    if not equal or clean_sha != candidate_sha:
        max_abs = float((candidate_prediction - clean_prediction).abs().max().cpu())
        raise RuntimeError(
            "C step-0 eval prediction is not bitwise equal to clean: "
            f"max_abs={max_abs} clean_sha={clean_sha} candidate_sha={candidate_sha}"
        )
    return {
        "probe_kind": "fixed_synthetic_eval_forward",
        "probe_size": [size, size],
        "optical_sha256": tensor_bitwise_sha256(optical),
        "sar_sha256": tensor_bitwise_sha256(sar),
        "prediction_shape": list(clean_prediction.shape),
        "clean_prediction_sha256": clean_sha,
        "candidate_prediction_sha256": candidate_sha,
        "prediction_torch_equal": True,
        "prediction_sha256_equal": True,
        "cpu_rng_state_restored": cpu_rng_equal,
        "cuda_rng_state_restored": cuda_rng_equal,
        "rng_contract": "probe restores CPU and all CUDA RNG states before training",
    }


def candidate_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "artifact_type": "whu_v4_c_optical_stem_screen",
        "schema_version": 1,
        "variant": CANDIDATE_VARIANT,
        "clean_baseline_variant": "mask-ignore",
        "model_name": "DINOv3",
        "dataset_name": "WHU",
        "num_modalities": 2,
        "backbone_type": BACKBONE_TYPE,
        "use_lora": False,
        "use_optical_stem": True,
        "optical_stem_location": "post-ACFM-L0-pre-FRN-single-injection",
        "optical_stem_seed": args.optical_stem_seed,
        "seed": args.seed,
        "scheduler_horizon_epochs": PROTOCOL_EPOCHS,
        "stop_after_epoch": args.stop_after_epoch,
        "evaluation_epochs": list(args.evaluation_epochs),
        "mask_padding_ignore": True,
        "mask_fill": IGNORE_INDEX,
        "aux_fill": 0,
        "loss_change": "none",
        "soft_ce_residual_confound": (
            "ignore positions are zeroed before a mean over all pixels"
        ),
        "train_batch_size_per_gpu": 8,
        "train_workers": args.num_workers,
        "inference_batch_size": args.inference_batch_size,
        "max_train_batches": args.max_train_batches,
        "max_test_images": args.max_test_images,
        "step0_probe_size": args.step0_probe_size,
        "clean_reference_dir": (
            None
            if args.clean_reference_dir is None
            else str(args.clean_reference_dir)
        ),
        "scope": "smoke" if args.smoke else "formal-screen",
    }


CLEAN_REFERENCE_PROTOCOL_FIELDS = (
    "seed",
    "scheduler_horizon_epochs",
    "stop_after_epoch",
    "evaluation_epochs",
    "model_name",
    "dataset_name",
    "num_modalities",
    "backbone_type",
    "use_lora",
    "train_batch_size_per_gpu",
    "train_workers",
    "inference_batch_size",
    "max_train_batches",
    "max_test_images",
    "scope",
    "mask_padding_ignore",
    "mask_fill",
    "aux_fill",
)


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def load_and_validate_clean_reference(
    reference_dir: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    reference = read_json_object(reference_dir / "protocol.json")
    if reference.get("variant") != "mask-ignore":
        raise RuntimeError("clean reference is not the V4-A mask-ignore variant")
    mismatches = {
        field: (reference.get(field), protocol.get(field))
        for field in CLEAN_REFERENCE_PROTOCOL_FIELDS
        if reference.get(field) != protocol.get(field)
    }
    if mismatches:
        raise RuntimeError(f"clean reference protocol differs: {mismatches}")
    for epoch in range(1, int(protocol["stop_after_epoch"]) + 1):
        path = reference_dir / f"train_e{epoch}.json"
        if not path.is_file():
            raise FileNotFoundError(f"clean reference lacks epoch {epoch}: {path}")
    for epoch in protocol["evaluation_epochs"]:
        path = reference_dir / f"evaluation_e{int(epoch)}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"clean reference lacks evaluation epoch {epoch}: {path}"
            )
    return reference


def validate_clean_reference_initialization(
    reference: Mapping[str, Any], clean_initial_state_sha256: str
) -> None:
    expected = reference.get("initial_model_state_sha256")
    if expected != clean_initial_state_sha256:
        raise RuntimeError(
            "in-process clean initialization does not reproduce sealed V4-A clean: "
            f"reference={expected} actual={clean_initial_state_sha256}"
        )


def validate_clean_reference_dataset(
    reference: Mapping[str, Any],
    *,
    train_dataset_length: int,
    full_test_length: int,
    evaluated_test_length: int,
) -> None:
    actual = {
        "train_dataset_length": train_dataset_length,
        "full_test_length": full_test_length,
        "evaluated_test_length": evaluated_test_length,
    }
    mismatches = {
        field: (reference.get(field), value)
        for field, value in actual.items()
        if reference.get(field) != value
    }
    if mismatches:
        raise RuntimeError(f"clean reference dataset geometry differs: {mismatches}")


def _first_trace_signature(record: Mapping[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (
            item.get("pair_sha256"),
            item.get("optical_sha256"),
            item.get("sar_sha256"),
            item.get("raw_label_sha256"),
            item.get("normalized_label_sha256"),
        )
        for item in record.get("first_batch_trace", [])
    ]


def validate_epoch_against_clean_reference(
    training: Mapping[str, Any], reference_path: Path
) -> dict[str, Any]:
    reference = read_json_object(reference_path)
    fields = ("paired_data_sha256", "raw_label_sha256")
    mismatches = {
        field: (reference.get(field), training.get(field))
        for field in fields
        if reference.get(field) != training.get(field)
    }
    if _first_trace_signature(reference) != _first_trace_signature(training):
        mismatches["first_batch_trace"] = ("reference", "candidate")
    if mismatches:
        raise RuntimeError(
            f"candidate data stream differs from clean reference: {mismatches}"
        )
    return {
        "reference_path": str(reference_path),
        "paired_data_sha256_equal": True,
        "raw_label_sha256_equal": True,
        "first_batch_trace_equal": True,
    }


def _get_cfg(*, use_optical_stem: bool, args: argparse.Namespace):
    return get_cfg(
        "DINOv3",
        "WHU",
        num_modalities=2,
        use_lora=False,
        r=3,
        backbone_type=BACKBONE_TYPE,
        use_naf=False,
        use_optical_stem=use_optical_stem,
        optical_stem_seed=args.optical_stem_seed,
    )


def build_audited_candidate(args: argparse.Namespace, device: torch.device):
    """Construct clean+C for audits, then return only the C training state."""

    set_seed(args.seed)
    clean_cfg = _get_cfg(use_optical_stem=False, args=args)
    clean_model = clean_cfg["model"]
    if int(clean_cfg["epochs"]) != PROTOCOL_EPOCHS:
        raise RuntimeError("clean scheduler horizon is no longer 50 epochs")

    # Reset before candidate construction.  Optical stem initialization must be
    # isolated inside the model so the shared released initialization is exact.
    set_seed(args.seed)
    candidate_cfg = _get_cfg(use_optical_stem=True, args=args)
    candidate_model = candidate_cfg["model"]
    candidate_optimizer = candidate_cfg["optimizer"]
    candidate_scheduler = candidate_cfg["scheduler"]
    if int(candidate_cfg["epochs"]) != PROTOCOL_EPOCHS:
        raise RuntimeError("candidate scheduler horizon is no longer 50 epochs")
    if int(candidate_scheduler.T_max) != PROTOCOL_EPOCHS:
        raise RuntimeError("candidate cosine scheduler T_max must remain 50")

    shared_audit = audit_shared_initialization(clean_model, candidate_model)
    optimizer_audit = audit_optimizer_membership(
        candidate_model, candidate_optimizer
    )
    clean_initial_state_sha256 = a_runner.named_tensor_sha256(
        clean_model.state_dict().items()
    )
    candidate_initial_state_sha256 = a_runner.named_tensor_sha256(
        candidate_model.state_dict().items()
    )

    clean_model = clean_model.to(device)
    candidate_model = candidate_model.to(device)
    step0_probe = audit_step0_probe(
        clean_model,
        candidate_model,
        device=device,
        size=args.step0_probe_size,
    )

    # The clean model is an initialization witness only.  Never train it here.
    del clean_cfg, clean_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        candidate_cfg,
        candidate_model,
        candidate_optimizer,
        candidate_scheduler,
        clean_initial_state_sha256,
        candidate_initial_state_sha256,
        shared_audit,
        optimizer_audit,
        step0_probe,
    )


def build_candidate_loaders(args: argparse.Namespace, cfg: Mapping[str, Any]):
    data_args = SimpleNamespace(**vars(args))
    data_args.variant = "mask-ignore"
    return a_runner.build_loaders(data_args, cfg)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def optical_stem(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = unwrap_model(model)
    stem = getattr(getattr(unwrapped, "decoder", None), "optical_stem", None)
    if stem is None:
        raise RuntimeError("V4-C candidate lost decoder.optical_stem")
    return stem


def _tensor_readout(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float()
    return {
        "rms": float(torch.sqrt(torch.mean(value.square())).cpu()),
        "abs_max": float(value.abs().max().cpu()),
        "nonzero_fraction": float(torch.count_nonzero(value).cpu()) / value.numel(),
        "finite": bool(torch.isfinite(value).all().cpu()),
    }


def _gradient_l2(parameters: Sequence[torch.nn.Parameter]) -> tuple[float, bool]:
    squared = 0.0
    finite = True
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        finite = finite and bool(torch.isfinite(gradient).all().cpu())
        squared += float(gradient.square().sum().cpu())
    return math.sqrt(squared), finite


def train_candidate_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    epoch: int,
    max_batches: int | None,
) -> dict[str, Any]:
    """V4-A training loop plus bounded optical-stem mechanism readouts."""

    model.train()
    loader.sampler.set_epoch(epoch)
    pair_digest = hashlib.sha256()
    raw_label_digest = hashlib.sha256()
    first_batch_trace: list[dict[str, Any]] = []
    mechanism_batches: list[dict[str, Any]] = []
    class_counts = np.zeros(NUM_CLASSES + 1, dtype=np.int64)
    loss_total = 0.0
    batch_count = 0
    started = time.perf_counter()

    stem = optical_stem(model)
    projection_parameters = list(stem.projection.parameters())
    upstream_parameters = list(stem.features.parameters())
    captured_output: dict[str, torch.Tensor | None] = {"value": None}
    capture_enabled = {"value": False}

    def capture_stem_output(_module, _inputs, output):
        if capture_enabled["value"]:
            captured_output["value"] = output.detach()

    hook = stem.register_forward_hook(capture_stem_output)
    try:
        iterations = tqdm(loader, desc=f"train C e{epoch}/50", dynamic_ncols=True)
        for batch_index, (optical_cpu, sar_cpu, label_cpu) in enumerate(
            iterations, start=1
        ):
            if max_batches is not None and batch_index > max_batches:
                break
            fingerprint = a_runner.batch_pair_fingerprint(
                optical_cpu, sar_cpu, label_cpu
            )
            pair_digest.update(bytes.fromhex(fingerprint))
            a_runner.update_tensor_digest(
                raw_label_digest, f"label_{batch_index}", label_cpu
            )
            counts = torch.bincount(
                label_cpu.reshape(-1).to(torch.int64),
                minlength=NUM_CLASSES + 1,
            )[: NUM_CLASSES + 1]
            class_counts += counts.numpy()
            if batch_index <= 10:
                first_batch_trace.append(
                    {
                        "batch": batch_index,
                        "pair_sha256": fingerprint,
                        "optical_sha256": a_runner.tensor_sha256(optical_cpu),
                        "sar_sha256": a_runner.tensor_sha256(sar_cpu),
                        "raw_label_sha256": a_runner.tensor_sha256(label_cpu),
                        "normalized_label_sha256": a_runner.tensor_sha256(
                            a_runner.normalized_pair_label(label_cpu)
                        ),
                    }
                )

            optical = optical_cpu.to(device, non_blocking=False)
            sar = sar_cpu.to(device, non_blocking=False)
            label = label_cpu.to(device, non_blocking=False)
            optimizer.zero_grad()
            capture_enabled["value"] = batch_index <= 10
            captured_output["value"] = None
            logits = model(optical, sar)
            loss = loss_fn(logits, label)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch} batch={batch_index}"
                )
            loss.backward()

            if batch_index <= 10:
                output = captured_output["value"]
                if output is None:
                    raise RuntimeError("optical-stem forward hook captured no output")
                projection_grad_l2, projection_grad_finite = _gradient_l2(
                    projection_parameters
                )
                upstream_grad_l2, upstream_grad_finite = _gradient_l2(
                    upstream_parameters
                )
                mechanism_batches.append(
                    {
                        "batch": batch_index,
                        "stem_output": _tensor_readout(output),
                        "projection_gradient_l2": projection_grad_l2,
                        "projection_gradient_finite": projection_grad_finite,
                        "upstream_gradient_l2": upstream_grad_l2,
                        "upstream_gradient_finite": upstream_grad_finite,
                    }
                )
            optimizer.step()
            capture_enabled["value"] = False
            value = float(loss.detach().cpu())
            loss_total += value
            batch_count += 1
            iterations.set_postfix(loss=f"{value:.4f}")
    finally:
        hook.remove()

    if batch_count == 0:
        raise RuntimeError("training epoch produced zero batches")
    valid_pixels = int(class_counts[:NUM_CLASSES].sum())
    all_pixels = int(class_counts.sum())
    all_finite = all(
        record["stem_output"]["finite"]
        and record["projection_gradient_finite"]
        and record["upstream_gradient_finite"]
        for record in mechanism_batches
    )
    mean_output_rms = (
        sum(record["stem_output"]["rms"] for record in mechanism_batches)
        / len(mechanism_batches)
        if mechanism_batches
        else None
    )
    mechanism_summary = {
        "observed_batches": len(mechanism_batches),
        "all_readouts_finite": all_finite,
        "mean_stem_output_rms_first_ten": mean_output_rms,
        "stem_output_nonzero_batches": sum(
            record["stem_output"]["nonzero_fraction"] > 0.0
            for record in mechanism_batches
        ),
        "projection_gradient_nonzero_batches": sum(
            record["projection_gradient_l2"] > 0.0
            for record in mechanism_batches
        ),
        "upstream_gradient_nonzero_batches": sum(
            record["upstream_gradient_l2"] > 0.0
            for record in mechanism_batches
        ),
        "first_batch_zero_output": bool(
            mechanism_batches
            and mechanism_batches[0]["stem_output"]["nonzero_fraction"] == 0.0
        ),
        "first_batch_projection_gradient_nonzero": bool(
            mechanism_batches
            and mechanism_batches[0]["projection_gradient_l2"] > 0.0
        ),
        "first_batch_upstream_gradient_delayed": bool(
            mechanism_batches
            and mechanism_batches[0]["upstream_gradient_l2"] == 0.0
        ),
        "interpretation": (
            "zero projection makes the first stem output and upstream gradient zero; "
            "projection gradients should be nonzero first, then open the upstream path"
        ),
    }
    return {
        "epoch": epoch,
        "variant": CANDIDATE_VARIANT,
        "average_loss": loss_total / batch_count,
        "batches": batch_count,
        "elapsed_seconds": time.perf_counter() - started,
        "paired_data_sha256": pair_digest.hexdigest(),
        "raw_label_sha256": raw_label_digest.hexdigest(),
        "first_batch_trace": first_batch_trace,
        "label_pixel_counts_0_to_ignore": class_counts.tolist(),
        "valid_pixels": valid_pixels,
        "all_pixels": all_pixels,
        "valid_fraction": valid_pixels / all_pixels,
        "optical_stem_mechanism": {
            "summary": mechanism_summary,
            "first_ten_batches": mechanism_batches,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(CANDIDATE_VARIANT,),
        required=True,
        help="explicitly identifies C; the clean mask-ignore baseline is reused",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--clean-reference-dir",
        type=Path,
        help=(
            "sealed V4-A mask-ignore directory; required for formal candidate-only "
            "training and checked before and after every epoch"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optical-stem-seed", type=int, default=DEFAULT_STEM_SEED)
    parser.add_argument("--step0-probe-size", type=int, default=DEFAULT_PROBE_SIZE)
    parser.add_argument("--stop-after-epoch", type=int, default=15)
    parser.add_argument(
        "--evaluation-epochs",
        type=a_runner.parse_epoch_list,
        default=DEFAULT_EVALUATION_EPOCHS,
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-test-images", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.stop_after_epoch = 1
        args.evaluation_epochs = (1,)
        args.max_train_batches = 1
        args.max_test_images = 1
        args.num_workers = 0
    elif args.clean_reference_dir is None:
        parser.error("formal V4-C requires --clean-reference-dir")
    if not 1 <= args.stop_after_epoch <= PROTOCOL_EPOCHS:
        parser.error("--stop-after-epoch must be within 1..50")
    if any(epoch > args.stop_after_epoch for epoch in args.evaluation_epochs):
        parser.error("evaluation epoch exceeds --stop-after-epoch")
    if args.num_workers < 0 or args.inference_batch_size <= 0:
        parser.error("worker count must be non-negative and inference batch positive")
    if args.step0_probe_size < 32 or args.step0_probe_size % 16:
        parser.error("--step0-probe-size must be >=32 and divisible by 16")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    a_runner.distributed.enable(overwrite=True)
    a_runner.require_single_process()
    local_rank = a_runner.get_local_rank()
    if not torch.cuda.is_available():
        raise RuntimeError("V4-C requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    protocol = candidate_protocol(args)
    clean_reference_protocol = None
    if args.clean_reference_dir is not None:
        clean_reference_protocol = load_and_validate_clean_reference(
            args.clean_reference_dir, protocol
        )
    dirty = a_runner.git_is_dirty()
    if dirty and not args.smoke:
        raise RuntimeError(
            "formal V4-C run requires a clean committed worktree; use --smoke "
            "for pre-commit diagnostics"
        )
    protocol = {
        **protocol,
        "git_commit": a_runner.git_commit(),
        "git_dirty": dirty,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    a_runner.prepare_output_dir(args.output_dir, protocol)
    print(
        f"V4-C start candidate={CANDIDATE_VARIANT} clean_baseline=mask-ignore "
        f"seed={args.seed} stop={args.stop_after_epoch} scope={protocol['scope']}",
        flush=True,
    )

    (
        cfg,
        model,
        optimizer,
        scheduler,
        clean_initial_state_sha256,
        candidate_initial_state_sha256,
        shared_audit,
        optimizer_audit,
        step0_probe,
    ) = build_audited_candidate(args, device)
    if clean_reference_protocol is not None:
        validate_clean_reference_initialization(
            clean_reference_protocol, clean_initial_state_sha256
        )
    train_loader, test_loader, test_names, full_test_length, loader_generator = (
        build_candidate_loaders(args, cfg)
    )
    if clean_reference_protocol is not None:
        validate_clean_reference_dataset(
            clean_reference_protocol,
            train_dataset_length=len(train_loader.dataset),
            full_test_length=full_test_length,
            evaluated_test_length=len(test_loader.dataset),
        )
    protocol_update = {
        **protocol,
        "clean_baseline_initial_state_sha256": clean_initial_state_sha256,
        "candidate_initial_model_state_sha256": candidate_initial_state_sha256,
        "full_model_sha_contract": (
            "candidate full SHA is recorded but never compared with clean because "
            "the candidate intentionally adds stem parameters"
        ),
        "clean_reference_protocol_sha256": (
            None
            if args.clean_reference_dir is None
            else a_runner.file_sha256(args.clean_reference_dir / "protocol.json")
        ),
        "shared_initialization_audit": shared_audit,
        "optimizer_membership_audit": optimizer_audit,
        "step0_prediction_probe": step0_probe,
        "train_dataset_length": len(train_loader.dataset),
        "full_test_length": full_test_length,
        "evaluated_test_length": len(test_loader.dataset),
    }
    a_runner.write_json_atomic(args.output_dir / "protocol.json", protocol_update)
    print(
        "initialization PASS shared_parameters_bitwise_equal=true "
        "step0_prediction_sha_equal=true stem_optimizer_exactly_once=true",
        flush=True,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    evaluations: list[dict[str, Any]] = []
    for epoch in range(1, args.stop_after_epoch + 1):
        print(f"epoch={epoch}/{args.stop_after_epoch} C train start", flush=True)
        training = train_candidate_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=cfg["loss_fn"],
            device=device,
            epoch=epoch,
            max_batches=args.max_train_batches,
        )
        if args.clean_reference_dir is not None:
            training["clean_reference_pair_audit"] = (
                validate_epoch_against_clean_reference(
                    training,
                    args.clean_reference_dir / f"train_e{epoch}.json",
                )
            )
        scheduler.step()
        training["learning_rates_after_scheduler_step"] = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        a_runner.write_json_atomic(
            args.output_dir / f"train_e{epoch}.json", training
        )
        mechanism = training["optical_stem_mechanism"]["summary"]
        print(
            f"epoch={epoch} train PASS loss={training['average_loss']:.6f} "
            f"data_sha256={training['paired_data_sha256']} "
            f"stem_output_nonzero={mechanism['stem_output_nonzero_batches']}/"
            f"{mechanism['observed_batches']} projection_grad_nonzero="
            f"{mechanism['projection_gradient_nonzero_batches']}/"
            f"{mechanism['observed_batches']}",
            flush=True,
        )

        if epoch in args.evaluation_epochs:
            evaluation = a_runner.evaluate(
                model=model,
                loader=test_loader,
                sample_names=test_names,
                cfg=cfg,
                device=device,
                inference_batch_size=args.inference_batch_size,
                epoch=epoch,
            )
            evaluation["variant"] = CANDIDATE_VARIANT
            a_runner.write_json_atomic(
                args.output_dir / f"evaluation_e{epoch}.json", evaluation
            )
            checkpoint_path = args.output_dir / f"checkpoint_e{epoch}.pth"
            a_runner.torch_save_atomic(
                checkpoint_path,
                a_runner.checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loader_generator=loader_generator,
                    epoch=epoch,
                    protocol=protocol_update,
                ),
            )
            evaluations.append(
                {
                    "epoch": epoch,
                    "miou_percent": evaluation["aggregate"]["miou_percent"],
                    "evaluation_path": f"evaluation_e{epoch}.json",
                    "evaluation_sha256": a_runner.file_sha256(
                        args.output_dir / f"evaluation_e{epoch}.json"
                    ),
                    "checkpoint_path": checkpoint_path.name,
                    "checkpoint_sha256": a_runner.file_sha256(checkpoint_path),
                }
            )
            print(
                f"epoch={epoch} evaluation PASS "
                f"mIoU={evaluation['aggregate']['miou_percent']:.6f}%",
                flush=True,
            )

    summary = {
        "status": "PASS",
        "outcome": "PENDING_C_VS_SEALED_BASELINE_COMPARISON",
        "artifact_type": "whu_v4_c_optical_stem_screen_summary",
        "variant": CANDIDATE_VARIANT,
        "clean_baseline_variant": "mask-ignore",
        "scope": protocol["scope"],
        "git_commit": protocol["git_commit"],
        "clean_baseline_initial_state_sha256": clean_initial_state_sha256,
        "candidate_initial_model_state_sha256": candidate_initial_state_sha256,
        "stop_after_epoch": args.stop_after_epoch,
        "evaluations": evaluations,
    }
    a_runner.write_json_atomic(args.output_dir / "summary.json", summary)
    print(
        f"PASS summary={args.output_dir / 'summary.json'} "
        "outcome=PENDING_C_VS_SEALED_BASELINE_COMPARISON",
        flush=True,
    )


if __name__ == "__main__":
    main()
