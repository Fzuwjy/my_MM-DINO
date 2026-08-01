"""Run the V4 E15->E30 three-arm epoch-boundary restart kill test.

This runner intentionally does *not* claim to reproduce an uninterrupted E30
trajectory.  It restores all state present in each arm's sealed E15 checkpoint,
but the original persistent DataLoader workers, their RNG state, cache,
prefetch, and iterator state cannot be recovered.  The official, clean, and C
arms therefore start fresh workers from matched restored loader-generator state
at the E16 boundary.

Run the clean arm first.  Both official and C then bind to that completed clean
continuation and audit every E16..E30 data stream online.  The official arm is
unconditional: it is not gated on C-minus-clean.  Formal runs evaluate E20,
E25, and E30 so the comparison can distinguish persistence from decay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scripts import run_whu_v4_a_screen as a_runner  # noqa: E402
from scripts import run_whu_v4_c_screen as c_runner  # noqa: E402
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.utils import set_seed  # noqa: E402


OFFICIAL_VARIANT = "official"
CLEAN_VARIANT = "mask-ignore"
CANDIDATE_VARIANT = c_runner.CANDIDATE_VARIANT
VARIANTS = (OFFICIAL_VARIANT, CLEAN_VARIANT, CANDIDATE_VARIANT)
SOURCE_EPOCH = 15
TARGET_EPOCH = 30
FIRST_CONTINUATION_EPOCH = SOURCE_EPOCH + 1
PROTOCOL_EPOCHS = a_runner.PROTOCOL_EPOCHS
CONTINUATION_MODE = "epoch_boundary_paired_restart_kill_test"
ARTIFACT_TYPE = "whu_v4_three_arm_epoch_boundary_restart"
FORMAL_EVALUATION_EPOCHS = (20, 25, 30)


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _resolved_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise RuntimeError(f"{field} is not a non-empty path")
    return Path(value).expanduser().resolve()


def _sealed_e15_artifacts(source_dir: Path) -> dict[str, Any]:
    """Seal the exact clean E15 checkpoint/evaluation selected by its summary."""

    protocol_path = source_dir / "protocol.json"
    summary_path = source_dir / "summary.json"
    protocol = read_json_object(protocol_path)
    summary = read_json_object(summary_path)
    if protocol.get("variant") != CLEAN_VARIANT:
        raise RuntimeError("candidate clean reference is not mask-ignore")
    if summary.get("status") != "PASS" or summary.get("variant") != CLEAN_VARIANT:
        raise RuntimeError("candidate clean reference summary is not sealed PASS")
    record = _summary_evaluation_record(summary, SOURCE_EPOCH)
    checkpoint_path = source_dir / str(record.get("checkpoint_path"))
    evaluation_path = source_dir / f"evaluation_e{SOURCE_EPOCH}.json"
    if not checkpoint_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError("candidate clean E15 reference artifacts are missing")
    checkpoint_sha256 = a_runner.file_sha256(checkpoint_path)
    evaluation_sha256 = a_runner.file_sha256(evaluation_path)
    if checkpoint_sha256 != record.get("checkpoint_sha256"):
        raise RuntimeError("candidate clean E15 checkpoint SHA256 differs")
    if evaluation_sha256 != record.get("evaluation_sha256"):
        raise RuntimeError("candidate clean E15 evaluation SHA256 differs")
    return {
        "clean_reference_dir": str(source_dir),
        "clean_reference_protocol_sha256": a_runner.file_sha256(protocol_path),
        "clean_reference_git_commit": protocol.get("git_commit"),
        "clean_reference_checkpoint_sha256": checkpoint_sha256,
        "clean_reference_evaluation_sha256": evaluation_sha256,
    }


def _pickle_sha256(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def rng_state_fingerprints(rng_state: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "loader_generator",
    )
    missing = [name for name in required if name not in rng_state]
    if missing:
        raise RuntimeError(f"checkpoint lacks RNG state: {missing}")
    cuda_states = rng_state["torch_cuda"]
    if not isinstance(cuda_states, (list, tuple)):
        raise TypeError("torch_cuda RNG state must be a list")
    result = {
        "python_sha256": _pickle_sha256(rng_state["python"]),
        "numpy_sha256": _pickle_sha256(rng_state["numpy"]),
        "torch_cpu_sha256": c_runner.tensor_bitwise_sha256(
            rng_state["torch_cpu"]
        ),
        "torch_cuda_sha256": [
            c_runner.tensor_bitwise_sha256(state) for state in cuda_states
        ],
        "loader_generator_sha256": c_runner.tensor_bitwise_sha256(
            rng_state["loader_generator"]
        ),
    }
    result["combined_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return result


def restore_rng_state(
    rng_state: Mapping[str, Any], loader_generator: torch.Generator
) -> dict[str, Any]:
    expected = rng_state_fingerprints(rng_state)
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    torch.cuda.set_rng_state_all(list(rng_state["torch_cuda"]))
    loader_generator.set_state(rng_state["loader_generator"])
    actual = rng_state_fingerprints(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "loader_generator": loader_generator.get_state(),
        }
    )
    if actual != expected:
        raise RuntimeError("failed to restore the sealed continuation RNG state")
    return actual


def _summary_evaluation_record(
    summary: Mapping[str, Any], epoch: int
) -> Mapping[str, Any]:
    records = summary.get("evaluations")
    if not isinstance(records, list):
        raise RuntimeError("source summary lacks evaluations")
    matches = [record for record in records if int(record.get("epoch", -1)) == epoch]
    if len(matches) != 1:
        raise RuntimeError(f"source summary lacks one sealed E{epoch} record")
    return matches[0]


SOURCE_PROTOCOL_FIELDS = (
    "variant",
    "model_name",
    "dataset_name",
    "num_modalities",
    "backbone_type",
    "use_lora",
    "seed",
    "scheduler_horizon_epochs",
    "stop_after_epoch",
    "evaluation_epochs",
    "mask_padding_ignore",
    "mask_fill",
    "aux_fill",
    "train_batch_size_per_gpu",
    "train_workers",
    "inference_batch_size",
    "max_train_batches",
    "max_test_images",
    "scope",
    "git_commit",
    "initial_model_state_sha256",
    "train_dataset_length",
    "full_test_length",
    "evaluated_test_length",
    "loss_change",
    "soft_ce_residual_confound",
)


def validate_source_protocol(
    protocol: Mapping[str, Any], *, variant: str, smoke: bool
) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown continuation variant: {variant}")
    if protocol.get("variant") != variant:
        raise RuntimeError(
            f"source variant differs: {protocol.get('variant')} != {variant}"
        )
    expected = {
        "model_name": "DINOv3",
        "dataset_name": "WHU",
        "num_modalities": 2,
        "backbone_type": a_runner.BACKBONE_TYPE,
        "use_lora": False,
        "seed": 42,
        "scheduler_horizon_epochs": PROTOCOL_EPOCHS,
        "stop_after_epoch": SOURCE_EPOCH,
        "mask_padding_ignore": variant != OFFICIAL_VARIANT,
        "mask_fill": (
            0 if variant == OFFICIAL_VARIANT else a_runner.IGNORE_INDEX
        ),
        "aux_fill": 0,
        "train_batch_size_per_gpu": 8,
        "train_workers": 4,
        "inference_batch_size": 32,
        "max_train_batches": None,
        "max_test_images": None,
        "scope": "formal-screen",
    }
    mismatches = {
        field: (expected_value, protocol.get(field))
        for field, expected_value in expected.items()
        if protocol.get(field) != expected_value
    }
    if protocol.get("evaluation_epochs") != [5, 10, 15]:
        mismatches["evaluation_epochs"] = (
            [5, 10, 15],
            protocol.get("evaluation_epochs"),
        )
    if variant == CANDIDATE_VARIANT:
        candidate_expected = {
            "use_optical_stem": True,
            "optical_stem_location": (
                "post-ACFM-L0-pre-FRN-single-injection"
            ),
            "optical_stem_seed": c_runner.DEFAULT_STEM_SEED,
            "clean_baseline_variant": CLEAN_VARIANT,
        }
        mismatches.update(
            {
                field: (value, protocol.get(field))
                for field, value in candidate_expected.items()
                if protocol.get(field) != value
            }
        )
    if mismatches:
        scope = "smoke source" if smoke else "formal source"
        raise RuntimeError(f"{scope} protocol differs: {mismatches}")


def load_sealed_source(
    source_dir: Path, *, variant: str, smoke: bool
) -> tuple[dict[str, Any], dict[str, Any], Path, str, dict[str, Any]]:
    source_dir = source_dir.resolve()
    protocol_path = source_dir / "protocol.json"
    summary_path = source_dir / "summary.json"
    protocol = read_json_object(protocol_path)
    summary = read_json_object(summary_path)
    validate_source_protocol(protocol, variant=variant, smoke=smoke)
    if summary.get("status") != "PASS" or summary.get("variant") != variant:
        raise RuntimeError("source summary did not seal the requested variant")
    record = _summary_evaluation_record(summary, SOURCE_EPOCH)
    checkpoint_path = source_dir / str(record.get("checkpoint_path"))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"source checkpoint is missing: {checkpoint_path}")
    checkpoint_sha256 = a_runner.file_sha256(checkpoint_path)
    if checkpoint_sha256 != record.get("checkpoint_sha256"):
        raise RuntimeError("source checkpoint SHA256 differs from sealed summary")
    evaluation_path = source_dir / f"evaluation_e{SOURCE_EPOCH}.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"source evaluation is missing: {evaluation_path}")
    if a_runner.file_sha256(evaluation_path) != record.get("evaluation_sha256"):
        raise RuntimeError("source evaluation SHA256 differs from sealed summary")
    if variant == CANDIDATE_VARIANT:
        comparison_path = source_dir / "comparison.json"
        comparison = read_json_object(comparison_path)
        comparison_expected = {
            "status": "PASS",
            "artifact_type": "whu_v4_c_sealed_baseline_comparison",
            "outcome": "PASS_C_E15_EXTEND_TO_E30",
            "scope": "formal-screen",
            "candidate_git_commit": protocol.get("git_commit"),
            "seed": protocol.get("seed"),
        }
        comparison_mismatches = {
            field: (expected, comparison.get(field))
            for field, expected in comparison_expected.items()
            if comparison.get(field) != expected
        }
        if comparison_mismatches:
            raise RuntimeError(
                "C source was not authorized for E30 extension: "
                f"{comparison_mismatches}"
            )
        bindings = comparison.get("baseline_bindings")
        if not isinstance(bindings, Mapping):
            raise RuntimeError("C E15 comparison lacks baseline bindings")
        clean_reference_dir = _resolved_path(
            protocol.get("clean_reference_dir"),
            field="C source clean_reference_dir",
        )
        bound_clean_dir = _resolved_path(
            bindings.get("clean_dir"), field="C comparison clean_dir"
        )
        bound_candidate_dir = _resolved_path(
            bindings.get("candidate_dir"), field="C comparison candidate_dir"
        )
        if bound_clean_dir != clean_reference_dir or bound_candidate_dir != source_dir:
            raise RuntimeError("C E15 comparison baseline bindings changed")
        candidate_clean_lineage = _sealed_e15_artifacts(clean_reference_dir)
        if candidate_clean_lineage["clean_reference_protocol_sha256"] != protocol.get(
            "clean_reference_protocol_sha256"
        ):
            raise RuntimeError("C source clean-reference protocol SHA256 differs")
        if candidate_clean_lineage["clean_reference_git_commit"] != comparison.get(
            "sealed_v4_a_git_commit"
        ):
            raise RuntimeError("C comparison clean-reference commit differs")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("source checkpoint payload is not a mapping")
    if int(checkpoint.get("epoch", -1)) != SOURCE_EPOCH:
        raise RuntimeError("source checkpoint epoch is not E15")
    internal_protocol = checkpoint.get("protocol")
    if not isinstance(internal_protocol, Mapping):
        raise RuntimeError("source checkpoint lacks its protocol")
    protocol_mismatches = {
        field: (protocol.get(field), internal_protocol.get(field))
        for field in SOURCE_PROTOCOL_FIELDS
        if protocol.get(field) != internal_protocol.get(field)
    }
    if protocol_mismatches:
        raise RuntimeError(
            f"checkpoint/source protocol differs: {protocol_mismatches}"
        )
    scheduler_state = checkpoint.get("scheduler")
    if not isinstance(scheduler_state, Mapping):
        raise RuntimeError("source checkpoint lacks scheduler state")
    if (
        int(scheduler_state.get("T_max", -1)) != PROTOCOL_EPOCHS
        or int(scheduler_state.get("last_epoch", -1)) != SOURCE_EPOCH
    ):
        raise RuntimeError("source scheduler is not Cosine(T_max=50,last_epoch=15)")
    if not isinstance(checkpoint.get("optimizer"), Mapping):
        raise RuntimeError("source checkpoint lacks optimizer state")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise RuntimeError("source checkpoint lacks model state")
    rng_fingerprints = rng_state_fingerprints(checkpoint.get("rng_state", {}))
    seals = {
        "source_dir_resolved": str(source_dir),
        "source_git_commit": protocol.get("git_commit"),
        "source_protocol_sha256": a_runner.file_sha256(protocol_path),
        "source_evaluation_sha256": a_runner.file_sha256(evaluation_path),
        "restart_rng_fingerprints": rng_fingerprints,
    }
    if variant == CANDIDATE_VARIANT:
        seals["candidate_e15_comparison_sha256"] = a_runner.file_sha256(
            comparison_path
        )
        seals["candidate_clean_lineage"] = candidate_clean_lineage
    return protocol, checkpoint, checkpoint_path, checkpoint_sha256, seals


def build_training_state(
    *, variant: str, source_protocol: Mapping[str, Any]
) -> tuple[Mapping[str, Any], torch.nn.Module, torch.optim.Optimizer, Any]:
    seed = int(source_protocol["seed"])
    if variant in (OFFICIAL_VARIANT, CLEAN_VARIANT):
        return a_runner.build_training_state(SimpleNamespace(seed=seed))
    set_seed(seed)
    args = SimpleNamespace(
        optical_stem_seed=int(source_protocol["optical_stem_seed"])
    )
    cfg = c_runner._get_cfg(use_optical_stem=True, args=args)
    scheduler = cfg["scheduler"]
    if int(cfg["epochs"]) != PROTOCOL_EPOCHS or int(scheduler.T_max) != PROTOCOL_EPOCHS:
        raise RuntimeError("candidate continuation scheduler horizon changed")
    c_runner.audit_optimizer_membership(cfg["model"], cfg["optimizer"])
    return cfg, cfg["model"], cfg["optimizer"], scheduler


def build_loaders(
    *,
    variant: str,
    source_protocol: Mapping[str, Any],
    cfg: Mapping[str, Any],
    num_workers: int,
    max_test_images: int | None,
):
    data_variant = (
        OFFICIAL_VARIANT if variant == OFFICIAL_VARIANT else CLEAN_VARIANT
    )
    args = SimpleNamespace(
        # The C model intentionally shares the clean mask-ignore data path.
        variant=data_variant,
        seed=int(source_protocol["seed"]),
        num_workers=num_workers,
        max_test_images=max_test_images,
    )
    return a_runner.build_loaders(args, cfg)


def _move_optimizer_moments_or_fail(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    mismatches: list[str] = []
    for parameter, state in optimizer.state.items():
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name)
            if isinstance(value, torch.Tensor) and value.device != parameter.device:
                mismatches.append(
                    f"{name}:{value.device}!={parameter.device}"
                )
        if parameter.device != device:
            mismatches.append(f"parameter:{parameter.device}!={device}")
    if mismatches:
        raise RuntimeError(f"optimizer moments are on the wrong device: {mismatches}")


def load_training_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpoint: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    _move_optimizer_moments_or_fail(optimizer, device)
    if int(scheduler.T_max) != PROTOCOL_EPOCHS or int(scheduler.last_epoch) != SOURCE_EPOCH:
        raise RuntimeError("restored scheduler has an off-by-one or horizon error")
    last_lrs = [float(value) for value in scheduler.get_last_lr()]
    optimizer_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if last_lrs != optimizer_lrs:
        raise RuntimeError(
            f"restored optimizer/scheduler LR differs: {optimizer_lrs} != {last_lrs}"
        )
    source_model_sha = a_runner.named_tensor_sha256(checkpoint["model"].items())
    loaded_model_sha = a_runner.named_tensor_sha256(unwrapped.state_dict().items())
    if loaded_model_sha != source_model_sha:
        raise RuntimeError("strict-loaded model tensors differ from source checkpoint")
    return {
        "source_model_state_sha256": source_model_sha,
        "loaded_model_state_sha256": loaded_model_sha,
        "strict_model_load": True,
        "optimizer_moments_on_parameter_device": True,
        "scheduler_t_max": int(scheduler.T_max),
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "learning_rates": optimizer_lrs,
    }


PAIR_PROTOCOL_FIELDS = (
    "artifact_type",
    "continuation_mode",
    "source_epoch",
    "first_continuation_epoch",
    "target_epoch",
    "stop_after_epoch",
    "evaluation_epochs",
    "not_equivalent_to_uninterrupted",
    "persistent_worker_state_restored",
    "three_arm_policy",
    "official_arm_execution_policy",
    "scientific_scope",
    "model_name",
    "dataset_name",
    "num_modalities",
    "backbone_type",
    "use_lora",
    "seed",
    "scheduler_horizon_epochs",
    "train_batch_size_per_gpu",
    "train_workers",
    "inference_batch_size",
    "max_train_batches",
    "max_test_images",
    "scope",
    "git_commit",
)

OFFICIAL_CLEAN_SOURCE_COMMON_FIELDS = (
    "model_name",
    "dataset_name",
    "num_modalities",
    "backbone_type",
    "use_lora",
    "seed",
    "scheduler_horizon_epochs",
    "stop_after_epoch",
    "evaluation_epochs",
    "aux_fill",
    "train_batch_size_per_gpu",
    "train_workers",
    "inference_batch_size",
    "max_train_batches",
    "max_test_images",
    "scope",
    "git_commit",
    "initial_model_state_sha256",
    "train_dataset_length",
    "full_test_length",
    "evaluated_test_length",
    "loss_change",
    "soft_ce_residual_confound",
)


def validate_completed_clean_continuation(
    clean_dir: Path,
    *,
    protocol: Mapping[str, Any],
    restart_rng_fingerprints: Mapping[str, Any],
    candidate_clean_lineage: Mapping[str, Any] | None = None,
    source_protocol: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    clean_protocol = read_json_object(clean_dir / "protocol.json")
    clean_summary = read_json_object(clean_dir / "summary.json")
    if clean_protocol.get("variant") != CLEAN_VARIANT:
        raise RuntimeError("paired continuation is not the clean variant")
    if clean_summary.get("status") != "PASS":
        raise RuntimeError("paired clean continuation is incomplete")
    mismatches = {
        field: (clean_protocol.get(field), protocol.get(field))
        for field in PAIR_PROTOCOL_FIELDS
        if clean_protocol.get(field) != protocol.get(field)
    }
    if mismatches:
        raise RuntimeError(f"paired clean continuation protocol differs: {mismatches}")
    if clean_protocol.get("restart_rng_fingerprints") != dict(
        restart_rng_fingerprints
    ):
        raise RuntimeError("paired source checkpoint RNG states differ")
    variant = protocol.get("variant")
    if variant == CANDIDATE_VARIANT:
        if not isinstance(candidate_clean_lineage, Mapping):
            raise RuntimeError("C continuation lacks its sealed clean lineage")
        lineage_checks = {
            "source_dir_resolved": candidate_clean_lineage.get(
                "clean_reference_dir"
            ),
            "source_protocol_sha256": candidate_clean_lineage.get(
                "clean_reference_protocol_sha256"
            ),
            "source_git_commit": candidate_clean_lineage.get(
                "clean_reference_git_commit"
            ),
            "source_checkpoint_sha256": candidate_clean_lineage.get(
                "clean_reference_checkpoint_sha256"
            ),
            "source_evaluation_sha256": candidate_clean_lineage.get(
                "clean_reference_evaluation_sha256"
            ),
        }
        lineage_mismatches = {
            field: (expected, clean_protocol.get(field))
            for field, expected in lineage_checks.items()
            if expected is None or clean_protocol.get(field) != expected
        }
        if lineage_mismatches:
            raise RuntimeError(
                "paired clean continuation is not the C source's sealed clean "
                f"lineage: {lineage_mismatches}"
            )
    elif variant == OFFICIAL_VARIANT:
        if not isinstance(source_protocol, Mapping):
            raise RuntimeError("official continuation lacks its source protocol")
        validate_source_protocol(
            source_protocol, variant=OFFICIAL_VARIANT, smoke=False
        )
        clean_source_dir = _resolved_path(
            clean_protocol.get("source_dir_resolved"),
            field="paired clean source_dir_resolved",
        )
        clean_source_protocol = read_json_object(
            clean_source_dir / "protocol.json"
        )
        validate_source_protocol(
            clean_source_protocol, variant=CLEAN_VARIANT, smoke=False
        )
        clean_source_seals = _sealed_e15_artifacts(clean_source_dir)
        seal_expectations = {
            "source_protocol_sha256": clean_source_seals[
                "clean_reference_protocol_sha256"
            ],
            "source_git_commit": clean_source_seals[
                "clean_reference_git_commit"
            ],
            "source_checkpoint_sha256": clean_source_seals[
                "clean_reference_checkpoint_sha256"
            ],
            "source_evaluation_sha256": clean_source_seals[
                "clean_reference_evaluation_sha256"
            ],
        }
        seal_mismatches = {
            field: (expected, clean_protocol.get(field))
            for field, expected in seal_expectations.items()
            if clean_protocol.get(field) != expected
        }
        if seal_mismatches:
            raise RuntimeError(
                "paired clean continuation source seals differ: "
                f"{seal_mismatches}"
            )
        missing_common_fields = [
            field
            for field in OFFICIAL_CLEAN_SOURCE_COMMON_FIELDS
            if field not in clean_source_protocol or field not in source_protocol
        ]
        if missing_common_fields:
            raise RuntimeError(
                "official/clean E15 source protocols lack paired fields: "
                f"{missing_common_fields}"
            )
        source_mismatches = {
            field: (clean_source_protocol.get(field), source_protocol.get(field))
            for field in OFFICIAL_CLEAN_SOURCE_COMMON_FIELDS
            if clean_source_protocol.get(field) != source_protocol.get(field)
        }
        if source_mismatches:
            raise RuntimeError(
                "official/clean E15 source protocols are not paired: "
                f"{source_mismatches}"
            )
    else:
        raise RuntimeError(
            "only official or C may bind to a completed clean continuation"
        )
    for epoch in range(FIRST_CONTINUATION_EPOCH, int(protocol["stop_after_epoch"]) + 1):
        if not (clean_dir / f"train_e{epoch}.json").is_file():
            raise FileNotFoundError(f"paired clean continuation lacks E{epoch}")
    for epoch in protocol["evaluation_epochs"]:
        if not (clean_dir / f"evaluation_e{int(epoch)}.json").is_file():
            raise FileNotFoundError(f"paired clean continuation lacks eval E{epoch}")
    return clean_protocol


def validate_epoch_pair(
    training: Mapping[str, Any],
    clean_train_path: Path,
    *,
    expected_trace_count: int,
    variant: str = CANDIDATE_VARIANT,
) -> dict[str, Any]:
    if variant not in (OFFICIAL_VARIANT, CANDIDATE_VARIANT):
        raise ValueError("epoch pairing is only defined against official or C")
    clean = read_json_object(clean_train_path)
    fields = ["paired_data_sha256"]
    if variant == CANDIDATE_VARIANT:
        fields.append("raw_label_sha256")
    mismatches = {
        field: (clean.get(field), training.get(field))
        for field in fields
        if clean.get(field) != training.get(field)
    }
    clean_trace = clean.get("first_batch_trace")
    candidate_trace = training.get("first_batch_trace")
    if not isinstance(clean_trace, list) or not isinstance(candidate_trace, list):
        mismatches["first_batch_trace_type"] = (
            type(clean_trace).__name__,
            type(candidate_trace).__name__,
        )
    elif (
        len(clean_trace) != expected_trace_count
        or len(candidate_trace) != expected_trace_count
    ):
        mismatches["first_batch_trace_count"] = (
            len(clean_trace),
            len(candidate_trace),
            expected_trace_count,
        )
    elif variant == CANDIDATE_VARIANT and clean_trace != candidate_trace:
        mismatches["first_batch_trace"] = ("clean", "candidate")
    elif variant == OFFICIAL_VARIANT:
        paired_trace_fields = (
            "batch",
            "pair_sha256",
            "optical_sha256",
            "sar_sha256",
            "normalized_label_sha256",
        )
        for index, (clean_item, official_item) in enumerate(
            zip(clean_trace, candidate_trace, strict=True), start=1
        ):
            for field in paired_trace_fields:
                if clean_item.get(field) != official_item.get(field):
                    mismatches[f"first_batch_trace[{index}].{field}"] = (
                        clean_item.get(field),
                        official_item.get(field),
                    )
    raw_label_equal = clean.get("raw_label_sha256") == training.get(
        "raw_label_sha256"
    )
    if (
        variant == OFFICIAL_VARIANT
        and expected_trace_count == 10
        and raw_label_equal
    ):
        mismatches["raw_label_sha256"] = (
            "different_by_official-vs-ignore_padding_design",
            "equal",
        )
    if mismatches:
        raise RuntimeError(f"continuation epoch data pairing differs: {mismatches}")
    raw_trace_difference_count = sum(
        clean_item.get("raw_label_sha256")
        != paired_item.get("raw_label_sha256")
        for clean_item, paired_item in zip(
            clean_trace, candidate_trace, strict=True
        )
    )
    return {
        "clean_train_path": str(clean_train_path),
        "paired_variant": variant,
        "paired_data_sha256_equal": True,
        "raw_label_sha256_equal": raw_label_equal,
        "raw_label_relation": (
            "must_equal"
            if variant == CANDIDATE_VARIANT
            else "intentional_official_vs_ignore_padding_difference"
        ),
        "raw_label_difference_required": (
            variant == OFFICIAL_VARIANT and expected_trace_count == 10
        ),
        "raw_label_trace_difference_count": raw_trace_difference_count,
        "first_batch_trace_equal": clean_trace == candidate_trace,
        "first_batch_trace_pair_fields_equal": True,
        "first_batch_trace_count": expected_trace_count,
    }


def continuation_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loader_generator: torch.Generator,
    epoch: int,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    payload = a_runner.checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader_generator=loader_generator,
        epoch=epoch,
        protocol=protocol,
    )
    payload["resume_contract"] = (
        "epoch-boundary paired restart: model/optimizer/scheduler/main RNG and "
        "loader generator restored; persistent worker/cache/prefetch state was not"
    )
    payload["not_equivalent_to_uninterrupted"] = True
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--paired-clean-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.variant in (OFFICIAL_VARIANT, CANDIDATE_VARIANT)
        and args.paired_clean_dir is None
    ):
        parser.error(
            "official and C continuations require --paired-clean-dir; run the "
            "clean arm first"
        )
    if args.variant == CLEAN_VARIANT and args.paired_clean_dir is not None:
        parser.error("clean continuation must not receive --paired-clean-dir")
    if args.num_workers < 0 or args.inference_batch_size <= 0:
        parser.error("workers must be non-negative and inference batch positive")
    if not args.smoke and args.num_workers != 4:
        parser.error("formal paired restart is frozen to --num-workers=4")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    a_runner.distributed.enable(overwrite=True)
    a_runner.require_single_process()
    local_rank = a_runner.get_local_rank()
    if not torch.cuda.is_available():
        raise RuntimeError("V4-C E30 continuation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)

    dirty = a_runner.git_is_dirty()
    if dirty and not args.smoke:
        raise RuntimeError("formal continuation requires a clean committed worktree")
    source_protocol, checkpoint, checkpoint_path, checkpoint_sha, seals = (
        load_sealed_source(
            args.source_dir,
            variant=args.variant,
            smoke=args.smoke,
        )
    )
    stop_after_epoch = FIRST_CONTINUATION_EPOCH if args.smoke else TARGET_EPOCH
    evaluation_epochs = (
        [stop_after_epoch] if args.smoke else list(FORMAL_EVALUATION_EPOCHS)
    )
    max_train_batches = 1 if args.smoke else None
    max_test_images = 1 if args.smoke else None
    protocol = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": 1,
        "continuation_mode": CONTINUATION_MODE,
        "variant": args.variant,
        "source_variant": args.variant,
        "source_epoch": SOURCE_EPOCH,
        "first_continuation_epoch": FIRST_CONTINUATION_EPOCH,
        "target_epoch": TARGET_EPOCH,
        "stop_after_epoch": stop_after_epoch,
        "evaluation_epochs": evaluation_epochs,
        "scheduler_horizon_epochs": PROTOCOL_EPOCHS,
        "not_equivalent_to_uninterrupted": True,
        "persistent_worker_state_restored": False,
        "three_arm_policy": "official_clean_C_all_required",
        "official_arm_execution_policy": "unconditional",
        "unrestored_state": [
            "worker_python_numpy_torch_rng",
            "worker_lru_cache",
            "prefetch_and_iterator_state",
            "ddp_reducer_state",
        ],
        "restored_state": [
            "model",
            "optimizer",
            "scheduler",
            "main_python_numpy_torch_cpu_cuda_rng",
            "loader_generator",
        ],
        "scientific_scope": (
            "three-arm E15-to-E30 epoch-boundary restart kill test with "
            "E20/E25/E30 shape readings; not an uninterrupted E30 result"
        ),
        "model_name": source_protocol["model_name"],
        "dataset_name": source_protocol["dataset_name"],
        "num_modalities": source_protocol["num_modalities"],
        "backbone_type": source_protocol["backbone_type"],
        "use_lora": source_protocol["use_lora"],
        "use_optical_stem": args.variant == CANDIDATE_VARIANT,
        "optical_stem_location": source_protocol.get("optical_stem_location"),
        "optical_stem_seed": source_protocol.get("optical_stem_seed"),
        "seed": source_protocol["seed"],
        "mask_padding_ignore": args.variant != OFFICIAL_VARIANT,
        "mask_fill": (
            0 if args.variant == OFFICIAL_VARIANT else a_runner.IGNORE_INDEX
        ),
        "aux_fill": 0,
        "loss_change": "none",
        "clean_baseline_variant": (
            CLEAN_VARIANT if args.variant == CANDIDATE_VARIANT else None
        ),
        "train_batch_size_per_gpu": source_protocol["train_batch_size_per_gpu"],
        "train_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
        "inference_batch_size": args.inference_batch_size,
        "max_train_batches": max_train_batches,
        "max_test_images": max_test_images,
        "scope": "smoke" if args.smoke else "formal-restart-screen",
        "git_commit": a_runner.git_commit(),
        "git_dirty": dirty,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "source_dir": seals["source_dir_resolved"],
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_sha,
        "paired_clean_dir": (
            None if args.paired_clean_dir is None else str(args.paired_clean_dir)
        ),
        **seals,
    }
    if args.paired_clean_dir is not None:
        candidate_clean_lineage = seals.get("candidate_clean_lineage")
        validate_completed_clean_continuation(
            args.paired_clean_dir,
            protocol=protocol,
            restart_rng_fingerprints=seals["restart_rng_fingerprints"],
            candidate_clean_lineage=candidate_clean_lineage,
            source_protocol=source_protocol,
        )

    cfg, model, optimizer, scheduler = build_training_state(
        variant=args.variant,
        source_protocol=source_protocol,
    )
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    train_loader, test_loader, test_names, full_test_length, loader_generator = (
        build_loaders(
            variant=args.variant,
            source_protocol=source_protocol,
            cfg=cfg,
            num_workers=args.num_workers,
            max_test_images=max_test_images,
        )
    )
    load_audit = load_training_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint=checkpoint,
        device=device,
    )
    if args.variant == CANDIDATE_VARIANT:
        c_runner.audit_optimizer_membership(
            model.module, optimizer
        )
    protocol = {
        **protocol,
        "source_load_audit": load_audit,
        "train_dataset_length": len(train_loader.dataset),
        "full_test_length": full_test_length,
        "evaluated_test_length": len(test_loader.dataset),
    }
    if int(source_protocol["train_dataset_length"]) != len(train_loader.dataset):
        raise RuntimeError("continuation train dataset length changed")
    if int(source_protocol["full_test_length"]) != full_test_length:
        raise RuntimeError("continuation test dataset length changed")
    a_runner.prepare_output_dir(args.output_dir, protocol)
    print(
        f"V4 three-arm restart start variant={args.variant} "
        f"source=E{SOURCE_EPOCH} stop=E{stop_after_epoch} scope={protocol['scope']}",
        flush=True,
    )
    print(
        f"source_checkpoint_sha256={checkpoint_sha} strict_load=true "
        f"scheduler_last_epoch={scheduler.last_epoch}",
        flush=True,
    )

    restored_rng = restore_rng_state(checkpoint["rng_state"], loader_generator)
    if restored_rng != seals["restart_rng_fingerprints"]:
        raise RuntimeError("restored continuation RNG fingerprint changed")
    training_records: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for epoch in range(FIRST_CONTINUATION_EPOCH, stop_after_epoch + 1):
        print(f"epoch={epoch}/{TARGET_EPOCH} restart train start", flush=True)
        if args.variant == CANDIDATE_VARIANT:
            training = c_runner.train_candidate_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=cfg["loss_fn"],
                device=device,
                epoch=epoch,
                max_batches=max_train_batches,
            )
        else:
            training = a_runner.train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=cfg["loss_fn"],
                device=device,
                epoch=epoch,
                max_batches=max_train_batches,
            )
            training["variant"] = args.variant
        if args.paired_clean_dir is not None:
            training["paired_restart_audit"] = validate_epoch_pair(
                training,
                args.paired_clean_dir / f"train_e{epoch}.json",
                expected_trace_count=1 if args.smoke else 10,
                variant=args.variant,
            )
        scheduler.step()
        training["learning_rates_after_scheduler_step"] = [
            float(group["lr"]) for group in optimizer.param_groups
        ]
        training["continuation_mode"] = CONTINUATION_MODE
        a_runner.write_json_atomic(
            args.output_dir / f"train_e{epoch}.json", training
        )
        training_records.append(
            {
                "epoch": epoch,
                "paired_data_sha256": training["paired_data_sha256"],
                "train_path": f"train_e{epoch}.json",
                "train_sha256": a_runner.file_sha256(
                    args.output_dir / f"train_e{epoch}.json"
                ),
            }
        )
        print(
            f"epoch={epoch} train PASS loss={training['average_loss']:.6f} "
            f"data_sha256={training['paired_data_sha256']}",
            flush=True,
        )

        if epoch in evaluation_epochs:
            evaluation = a_runner.evaluate(
                model=model,
                loader=test_loader,
                sample_names=test_names,
                cfg=cfg,
                device=device,
                inference_batch_size=args.inference_batch_size,
                epoch=epoch,
            )
            evaluation["variant"] = args.variant
            evaluation["continuation_mode"] = CONTINUATION_MODE
            evaluation_path = args.output_dir / f"evaluation_e{epoch}.json"
            a_runner.write_json_atomic(evaluation_path, evaluation)
            checkpoint_out = args.output_dir / f"checkpoint_e{epoch}.pth"
            a_runner.torch_save_atomic(
                checkpoint_out,
                continuation_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loader_generator=loader_generator,
                    epoch=epoch,
                    protocol=protocol,
                ),
            )
            evaluations.append(
                {
                    "epoch": epoch,
                    "miou_percent": evaluation["aggregate"]["miou_percent"],
                    "evaluation_path": evaluation_path.name,
                    "evaluation_sha256": a_runner.file_sha256(evaluation_path),
                    "checkpoint_path": checkpoint_out.name,
                    "checkpoint_sha256": a_runner.file_sha256(checkpoint_out),
                }
            )
            print(
                f"epoch={epoch} evaluation PASS "
                f"mIoU={evaluation['aggregate']['miou_percent']:.6f}%",
                flush=True,
            )

    if args.smoke and args.variant == CANDIDATE_VARIANT:
        mechanism = training["optical_stem_mechanism"]["summary"]
        if not (
            mechanism["all_readouts_finite"]
            and mechanism["stem_output_nonzero_batches"] == 1
            and mechanism["projection_gradient_nonzero_batches"] == 1
            and mechanism["upstream_gradient_nonzero_batches"] == 1
        ):
            raise RuntimeError("resumed C stem was not fully active on the smoke batch")
    if args.smoke:
        outcome = "PASS_RESTART_SMOKE_CONTRACT"
    elif args.variant == CLEAN_VARIANT:
        outcome = "COMPLETES_REQUIRED_CLEAN_RESTART_ARM"
    elif args.variant == OFFICIAL_VARIANT:
        outcome = "COMPLETES_UNCONDITIONAL_OFFICIAL_RESTART_ARM"
    else:
        outcome = "COMPLETES_REQUIRED_C_RESTART_ARM"
    summary = {
        "status": "PASS",
        "outcome": outcome,
        "scientific_decision": "NONE",
        "artifact_type": f"{ARTIFACT_TYPE}_summary",
        "continuation_mode": CONTINUATION_MODE,
        "variant": args.variant,
        "scope": protocol["scope"],
        "git_commit": protocol["git_commit"],
        "source_checkpoint_sha256": checkpoint_sha,
        "source_epoch": SOURCE_EPOCH,
        "stop_after_epoch": stop_after_epoch,
        "not_equivalent_to_uninterrupted": True,
        "training": training_records,
        "evaluations": evaluations,
    }
    a_runner.write_json_atomic(args.output_dir / "summary.json", summary)
    print(
        f"PASS summary={args.output_dir / 'summary.json'} "
        f"outcome={summary['outcome']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
