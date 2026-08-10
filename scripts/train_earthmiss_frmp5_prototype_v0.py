"""Train the matched EarthMiss FRM-P5 prototype-transfer v0 arms.

P0 is the matched Run-C objective, P1 adds a SAR class-separation control, and
P2 adds privileged Full-state-to-SAR batch Prototype InfoNCE.  All arms use the
released Run-C deployment graph; P1/P2 add no parameter, buffer, or inference
operation.  Formal runs are fixed to 50 epochs with no early stopping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import build_dataset  # noqa: E402
from losses import DiceLoss, JointLoss, SoftCrossEntropyLoss  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from models.MMDINO.dino_segment import build_model  # noqa: E402
from utils.earthmiss_metrics import EarthMissMetrics  # noqa: E402
from utils.inference import slide_inference  # noqa: E402
from scripts.earthmiss_frmp5_prototype_common import (  # noqa: E402
    PrototypeBatchStatistics,
    assert_batchnorm_buffers_equal,
    preserve_rng_state,
    prototype_transfer_infonce,
    prototype_weight_multiplier,
    restore_batchnorm_buffers,
    restore_rng_state,
    snapshot_batchnorm_buffers,
    snapshot_rng_state,
    temporary_batchnorm_eval,
)
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
    VAL_SELECTION_CLASS_IDS,
    audit_datasets,
    seed_everything,
)


ARMS = ("p0", "p1", "p2")
PROTOTYPE_ARMS = ("p1", "p2")
PROTOCOL_REVISION = "earthmiss_frmp5_prototype_transfer_v0"
HEALTH_SCHEMA = "earthmiss_frmp5_prototype_health_v0"
DEFAULT_OUTPUT_ROOT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-frmp5-prototype-v0"
)
EPOCHS = 50
BATCH_SIZE = 8
WINDOW_SIZE = 512
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
EVAL_INTERVAL = 5
ELIGIBLE_SELECTION_EPOCHS = tuple(range(15, EPOCHS + 1, EVAL_INTERVAL))
GRADIENT_CALIBRATION_BATCHES = 32
GRADIENT_TARGET_RATIO = 0.10
MINIMUM_RAW_SUPPORT = 2.0
TEMPERATURE = 0.1
NUM_CLASSES = 8
IGNORE_INDEX = 8
CALIBRATION_SEED_OFFSET = 20_260_811


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-size", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--health-report",
        help="Formal Train-only BN/prototype health report; required for P1/P2.",
    )
    parser.add_argument(
        "--confirm-health-gates-passed",
        action="store_true",
        help="Record the explicit decision to proceed after reviewing the report.",
    )
    parser.add_argument(
        "--smoke-batches",
        type=int,
        default=0,
        help="Run a non-formal foreground optimizer smoke test and exit.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.num_workers < 0 or args.cache_size < 0:
        raise ValueError("loader counts must be non-negative")
    if args.smoke_batches < 0:
        raise ValueError("--smoke-batches must be non-negative")
    if args.resume and args.smoke_batches:
        raise ValueError("smoke runs cannot resume")
    if args.arm in PROTOTYPE_ARMS and not args.audit_only and not args.smoke_batches:
        if not args.health_report:
            raise ValueError(f"{args.arm} requires --health-report")
        if not args.confirm_health_gates_passed:
            raise ValueError(
                f"{args.arm} requires --confirm-health-gates-passed"
            )


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_state_sha256(model) -> str:
    """Hash the complete initialized deployment state without changing it."""

    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        hasher.update(name.encode("utf-8"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(str(tuple(value.shape)).encode("ascii"))
        hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def validate_health_report(path: str | Path | None):
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"health report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != HEALTH_SCHEMA:
        raise ValueError("unexpected FRM-P5 health report schema")
    if report.get("formal") is not True:
        raise ValueError("a smoke health report cannot unlock formal training")
    if report.get("training_was_performed") is not False:
        raise ValueError("health report must be a zero-training diagnostic")
    if report.get("decision", {}).get("training_allowed") is not True:
        raise ValueError("FRM-P5 health report did not pass its frozen gates")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "schema": report["schema"],
        "checkpoint": report.get("checkpoint"),
        "manual_review_confirmed": True,
    }


def _dataset_kwargs(args, *, cache_size):
    return {
        "dataset_root": args.dataset_root,
        "window_size": (WINDOW_SIZE, WINDOW_SIZE),
        "model_name": "DINOv3",
        "modality": "multi",
        "backbone_type": "dinov3_vits16",
        "cache_size": cache_size,
    }


def build_loaders(args):
    train_dataset = build_dataset(
        "EarthMiss",
        "train",
        **_dataset_kwargs(args, cache_size=args.cache_size),
    )
    val_dataset = build_dataset(
        "EarthMiss",
        "val",
        **_dataset_kwargs(args, cache_size=args.cache_size),
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=train_generator,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return train_dataset, val_dataset, train_loader, val_loader, train_generator


def build_calibration_loader(args):
    dataset = build_dataset(
        "EarthMiss",
        "train",
        **_dataset_kwargs(args, cache_size=0),
    )
    generator = torch.Generator().manual_seed(
        args.seed + CALIBRATION_SEED_OFFSET
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        generator=generator,
    )
    return loader


def train_state(state_rng: random.Random) -> str:
    return "sar" if state_rng.random() < 0.5 else "full"


def build_model_instance(weights_path, device):
    model = build_model(
        model_name="DINOv3",
        backbone_weights=str(weights_path),
        backbone_type="dinov3_vits16",
        freeze_backbone=True,
        n_classes=NUM_CLASSES,
        use_lora=False,
        r=3,
        num_modalities=2,
        raw_logits=True,
    ).to(device)
    return model


@torch.no_grad()
def evaluate_with_cities(model, loader, dataset, state, device):
    """Evaluate one endpoint once while accumulating pooled and city metrics."""

    if len(loader) != len(dataset):
        raise RuntimeError("city-aware Val evaluation requires batch_size=1")
    model.eval()
    pooled = EarthMissMetrics()
    by_city = {}
    availability = canonical_availability(state, batch_size=1, device=device)
    stride = int(WINDOW_SIZE * 2 / 3)
    for index, (rgb, sar, label) in enumerate(
        tqdm(loader, desc=f"Val {state}", leave=False)
    ):
        city = dataset.samples[index].city
        evaluator = by_city.setdefault(city, EarthMissMetrics())
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        logits = slide_inference(
            rgb,
            model,
            n_output_channels=NUM_CLASSES,
            crop_size=(WINDOW_SIZE, WINDOW_SIZE),
            stride=(stride, stride),
            dsm=sar,
            availability=availability,
            batch_size=BATCH_SIZE * 4,
        )
        prediction = logits.argmax(dim=1)
        pooled.update(prediction, label)
        evaluator.update(prediction, label)
    pooled_metrics = pooled.compute()
    if pooled_metrics["selection_class_ids"] != VAL_SELECTION_CLASS_IDS:
        raise RuntimeError(
            "EarthMiss Val class support changed: expected "
            f"{VAL_SELECTION_CLASS_IDS}, got "
            f"{pooled_metrics['selection_class_ids']}"
        )
    return {
        "pooled": pooled_metrics,
        "by_city": {
            city: evaluator.compute() for city, evaluator in sorted(by_city.items())
        },
    }


def build_metadata(args, train_dataset, val_dataset, train_loader, health_record):
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "formal": args.smoke_batches == 0,
        "arm": args.arm,
        "seed": args.seed,
        "method_identity": (
            "privileged Full-state fused post-FRM-P5 to canonical SAR-state "
            "batch prototype transfer"
        ),
        "not_optical_only_teacher": True,
        "training_start": {
            "policy": "same_common_DINOv3_MM-DINO_initialization_per_seed",
            "warm_start_from_run_c_e15": False,
            "shared_e1_e5_checkpoint_fork": False,
        },
        "data": {
            "train_tiles": len(train_dataset),
            "val_tiles": len(val_dataset),
            "window_size": WINDOW_SIZE,
            "batch_size": BATCH_SIZE,
            "steps_per_epoch": len(train_loader),
            "num_workers": args.num_workers,
            "cache_size": args.cache_size,
            "persistent_workers": args.num_workers > 0,
            "augmentation": [
                "random_512_crop",
                "horizontal_flip_p0.5",
                "vertical_flip_p0.5",
                "rotate90_p0.5",
            ],
        },
        "model": {
            "base": "released DINOv3 SampleAdapter + Decoder Run C",
            "frozen_backbone": True,
            "raw_logits": True,
            "new_parameters": 0,
            "new_buffers": 0,
            "deployment_graph": "exactly Run C",
            "feature": "state-level post-FRM P5 before SEFusion",
            "feature_channels": 256,
            "canonical_adapter_slots_are_duplicate_states": True,
        },
        "prototype": {
            "enabled": args.arm in PROTOTYPE_ARMS,
            "arm_semantics": {
                "p0": "matched Run C objective",
                "p1": "SAR class-prototype separation control",
                "p2": "Full-state anchor to SAR-state query",
            }[args.arm],
            "gt_area_occupancy": "exact geometric area average",
            "ignore_pixels": "zero occupancy and no valid-pixel renormalization",
            "purity_weight": "a^2",
            "support_rule": f"sum(a)>={MINIMUM_RAW_SUPPORT}",
            "aggregation": "whole homogeneous batch",
            "class_weighting": "equal over supported classes",
            "minimum_supported_classes": 2,
            "loss": "Prototype InfoNCE",
            "temperature": TEMPERATURE,
            "anchor_stop_gradient": True,
            "projector_or_bank": False,
        },
        "auxiliary_forward": {
            "only_on_sar_batches_with_positive_schedule_weight": True,
            "frozen_backbone_outputs_cached_once_for_rgb_and_sar": True,
            "primary_sar_segmentation_bn": "train mode; sole persistent update",
            "query_and_anchor_frm_bn": "eval mode with shared persistent buffers",
            "rng_preserved": True,
            "bn_buffers_asserted_unchanged": True,
        },
        "optimization": {
            "epochs": EPOCHS,
            "early_stopping": False,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": EPOCHS,
            "segmentation_loss": "SoftCrossEntropy(0.05)+Dice(0.05), ignore=8",
            "prototype_schedule": {
                "epochs_1_5": 0.0,
                "epochs_6_10": "linear ramp",
                "epochs_11_50": "calibrated fixed weight",
            },
            "gradient_calibration": {
                "epoch": 6,
                "split": "Train only",
                "effective_sar_batches": GRADIENT_CALIBRATION_BATCHES,
                "target_joint_median_ratio": GRADIENT_TARGET_RATIO,
                "groups": ["whole Adapter", "whole FRM (active P5 grads)"],
                "validation_or_test_used": False,
            },
        },
        "evaluation": {
            "split": "val_city_holdout",
            "states": ["sar", "full"],
            "interval_epochs": EVAL_INTERVAL,
            "eligible_selection_epochs": list(ELIGIBLE_SELECTION_EPOCHS),
            "trajectory_only_epochs": [5, 10],
            "selection_state": "sar",
            "selection_metric": "mIoU",
            "selection_support": "pooled_gt_present",
            "expected_selection_class_ids": VAL_SELECTION_CLASS_IDS,
            "paired_full_endpoint_same_checkpoint": True,
            "test_is_not_used": True,
        },
        "pre_registered_screen": {
            "seed42_p2_minus_p0_sar_val_pp_minimum": 0.50,
            "seed42_p2_minus_p0_full_val_pp_minimum": -0.50,
            "seed42_nonnegative_val_cities_minimum": "2_of_3",
            "conditional_p2_minus_p1_sar_val_pp_minimum": 0.30,
            "failure_action": "archive_v0_without_tau_lambda_scale_or_purity_sweep",
        },
        "determinism": {
            "arm_data_trace_recorded": True,
            "auxiliary_model_rng_isolated": True,
            "resume_saves_main_rng_and_loader_generator": True,
            "bitwise_resume_claim": False,
            "reason": "persistent worker augmentation and prefetch state is not serialized",
        },
        "health_report": health_record,
        "smoke_batches": args.smoke_batches,
    }


def output_directory(args):
    prefix = "smoke" if args.smoke_batches else "run"
    return Path(args.output_root) / f"{prefix}_{args.arm}_seed{args.seed}"


def _zero_from(logits):
    return logits.sum() * 0.0


def auxiliary_prototype_features(
    model,
    rgb,
    sar,
    backbone_outputs,
    *,
    arm,
):
    if arm not in PROTOTYPE_ARMS:
        raise ValueError("auxiliary prototype extraction requires P1 or P2")
    batch_size = rgb.shape[0]
    sar_availability = canonical_availability(
        "sar", batch_size=batch_size, device=rgb.device
    )
    full_availability = canonical_availability(
        "full", batch_size=batch_size, device=rgb.device
    )
    before = snapshot_batchnorm_buffers(model.decoder.frm)
    with preserve_rng_state(), temporary_batchnorm_eval(model.decoder.frm):
        query = model.extract_state_frm_p5_from_backbone_outputs(
            rgb,
            sar,
            backbone_outputs=backbone_outputs,
            availability=sar_availability,
        )
        if arm == "p1":
            anchor = query.detach()
        else:
            with torch.no_grad():
                anchor = model.extract_state_frm_p5_from_backbone_outputs(
                    rgb,
                    sar,
                    backbone_outputs=backbone_outputs,
                    availability=full_availability,
                )
    assert_batchnorm_buffers_equal(before, model.decoder.frm)
    return query, anchor


def forward_training_losses(
    model,
    criterion,
    rgb,
    sar,
    label,
    *,
    state,
    arm,
    prototype_weight,
):
    if state not in ("full", "sar"):
        raise ValueError(f"unsupported training state: {state}")
    if arm not in ARMS:
        raise ValueError(f"unsupported arm: {arm}")
    availability = canonical_availability(
        state,
        batch_size=rgb.shape[0],
        device=rgb.device,
    )
    use_prototype = (
        arm in PROTOTYPE_ARMS and state == "sar" and prototype_weight > 0.0
    )
    if use_prototype:
        backbone_outputs = model.extract_frozen_backbone_outputs(rgb, sar)
        logits = model.forward_from_backbone_outputs(
            rgb,
            sar,
            backbone_outputs=backbone_outputs,
            availability=availability,
        )
        query, anchor = auxiliary_prototype_features(
            model,
            rgb,
            sar,
            backbone_outputs,
            arm=arm,
        )
        prototype_loss, statistics = prototype_transfer_infonce(
            query,
            anchor,
            label,
            num_classes=NUM_CLASSES,
            ignore_index=IGNORE_INDEX,
            minimum_raw_support=MINIMUM_RAW_SUPPORT,
            temperature=TEMPERATURE,
        )
    else:
        logits = model(rgb, sar, availability=availability)
        prototype_loss = _zero_from(logits)
        statistics = None
    segmentation_loss = criterion(logits, label)
    total_loss = segmentation_loss + prototype_weight * prototype_loss
    return total_loss, segmentation_loss, prototype_loss, statistics


def _gradient_norm(grads):
    terms = [grad.detach().float().square().sum() for grad in grads if grad is not None]
    if not terms:
        return 0.0
    return float(torch.sqrt(torch.stack(terms).sum()).cpu())


def active_frm_p5_parameters(frm):
    """Return exactly the FRM modules that contribute to the s5 output."""

    if not hasattr(frm, "conv_scales") or not hasattr(frm, "conv_aggregation_s5"):
        raise RuntimeError("unexpected FRM structure for P5 gradient calibration")
    modules = [
        frm.conv_scales[f"conv_scale5_c{source_scale}"]
        for source_scale in range(2, 6)
    ]
    modules.append(frm.conv_aggregation_s5)
    parameters = []
    seen = set()
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return tuple(parameters)


def calibrate_prototype_weight(model, criterion, loader, *, arm, device):
    if arm not in PROTOTYPE_ARMS:
        raise ValueError("gradient calibration requires P1 or P2")
    was_training = model.training
    adapter_parameters = tuple(
        parameter for parameter in model.adapter.parameters() if parameter.requires_grad
    )
    frm_parameters = active_frm_p5_parameters(model.decoder.frm)
    parameters = adapter_parameters + frm_parameters
    if not adapter_parameters or not frm_parameters:
        raise RuntimeError("gradient calibration parameter groups are empty")

    ratios = {"adapter": [], "frm": []}
    effective_batches = 0
    processed_batches = 0
    skipped_batches = 0
    model.train()
    with preserve_rng_state():
        for rgb, sar, label in loader:
            processed_batches += 1
            rgb = rgb.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            bn_before = snapshot_batchnorm_buffers(model)
            try:
                _, segmentation_loss, prototype_loss, statistics = (
                    forward_training_losses(
                        model,
                        criterion,
                        rgb,
                        sar,
                        label,
                        state="sar",
                        arm=arm,
                        prototype_weight=1.0,
                    )
                )
                if statistics is None or int(statistics.support_mask.sum()) < 2:
                    skipped_batches += 1
                    continue
                segmentation_grads = torch.autograd.grad(
                    segmentation_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                prototype_grads = torch.autograd.grad(
                    prototype_loss,
                    parameters,
                    allow_unused=True,
                )
                split = len(adapter_parameters)
                group_grads = {
                    "adapter": (
                        segmentation_grads[:split],
                        prototype_grads[:split],
                    ),
                    "frm": (
                        segmentation_grads[split:],
                        prototype_grads[split:],
                    ),
                }
                for name, (seg_grads, proto_grads) in group_grads.items():
                    seg_norm = _gradient_norm(seg_grads)
                    proto_norm = _gradient_norm(proto_grads)
                    if not math.isfinite(seg_norm) or not math.isfinite(proto_norm):
                        raise RuntimeError("non-finite gradient norm during calibration")
                    if seg_norm <= 0.0 or proto_norm <= 0.0:
                        raise RuntimeError(
                            f"zero {name} gradient norm during prototype calibration"
                        )
                    ratios[name].append(proto_norm / seg_norm)
                effective_batches += 1
            finally:
                restore_batchnorm_buffers(bn_before, model)
                model.zero_grad(set_to_none=True)
            if effective_batches >= GRADIENT_CALIBRATION_BATCHES:
                break
    model.train(was_training)

    if effective_batches != GRADIENT_CALIBRATION_BATCHES:
        raise RuntimeError(
            "insufficient supported Train batches for gradient calibration: "
            f"{effective_batches}/{GRADIENT_CALIBRATION_BATCHES}"
        )
    joint = ratios["adapter"] + ratios["frm"]
    unscaled_median = float(np.median(np.asarray(joint, dtype=np.float64)))
    if not math.isfinite(unscaled_median) or unscaled_median <= 1e-8:
        raise RuntimeError("prototype gradient ratio is non-finite or degenerate")
    calibrated_weight = GRADIENT_TARGET_RATIO / unscaled_median
    if not math.isfinite(calibrated_weight) or calibrated_weight <= 0.0:
        raise RuntimeError("calibrated prototype weight is invalid")
    return calibrated_weight, {
        "effective_batches": effective_batches,
        "processed_batches": processed_batches,
        "skipped_batches": skipped_batches,
        "target_joint_median_ratio": GRADIENT_TARGET_RATIO,
        "unscaled_joint_median_ratio": unscaled_median,
        "calibrated_weight": calibrated_weight,
        "per_group_unscaled_ratio": {
            name: {
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
            for name, values in ratios.items()
        },
    }


class PrototypeEpochAccumulator:
    def __init__(self):
        self.batches = 0
        self.skipped_batches = 0
        self.loss_sum = 0.0
        self.class_inclusion = np.zeros(NUM_CLASSES, dtype=np.int64)
        self.raw_support = np.zeros(NUM_CLASSES, dtype=np.float64)
        self.purity_mass = np.zeros(NUM_CLASSES, dtype=np.float64)
        self.ess = np.zeros(NUM_CLASSES, dtype=np.float64)
        self.positive_sum = 0.0
        self.positive_count = 0
        self.offdiag_sum = 0.0
        self.offdiag_count = 0

    def update(self, loss: torch.Tensor, stats: PrototypeBatchStatistics):
        self.batches += 1
        self.loss_sum += float(loss.detach())
        mask = stats.support_mask.detach().cpu().numpy().astype(bool)
        self.class_inclusion += mask.astype(np.int64)
        self.raw_support += stats.raw_support.detach().cpu().numpy()
        self.purity_mass += stats.purity_mass.detach().cpu().numpy()
        self.ess += stats.effective_sample_size.detach().cpu().numpy()
        if mask.sum() < 2:
            self.skipped_batches += 1
        positive = stats.positive_cosine.detach().cpu()
        if positive.numel():
            self.positive_sum += float(positive.sum())
            self.positive_count += positive.numel()
        offdiag = float(stats.off_diagonal_cosine_mean.detach().cpu())
        if math.isfinite(offdiag):
            self.offdiag_sum += offdiag
            self.offdiag_count += 1

    def summary(self):
        if self.batches == 0:
            return None
        return {
            "batches": self.batches,
            "skipped_batches": self.skipped_batches,
            "mean_loss": self.loss_sum / self.batches,
            "class_inclusion_batches": self.class_inclusion.tolist(),
            "mean_raw_support": (self.raw_support / self.batches).tolist(),
            "mean_purity_mass": (self.purity_mass / self.batches).tolist(),
            "mean_effective_sample_size": (self.ess / self.batches).tolist(),
            "mean_positive_cosine": (
                self.positive_sum / self.positive_count
                if self.positive_count
                else None
            ),
            "mean_off_diagonal_cosine": (
                self.offdiag_sum / self.offdiag_count
                if self.offdiag_count
                else None
            ),
        }


def update_batch_trace(hasher, *, state, rgb, sar, label):
    hasher.update(state.encode("ascii"))
    tensors = (
        rgb[:, :, ::64, ::64],
        sar[:, :, ::64, ::64],
        label[:, ::32, ::32],
    )
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        hasher.update(str(array.shape).encode("ascii"))
        hasher.update(array.tobytes())


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
    epoch,
    best_score,
    best_epoch,
    metadata,
    train_generator,
    calibrated_weight,
    calibration_record,
    checkpoint_role,
    selection_score=None,
    paired_validation=None,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "arm": metadata["arm"],
            "seed": metadata["seed"],
            "run": "c",
            "protocol": metadata,
            "checkpoint_role": checkpoint_role,
            "selection_state": "sar" if selection_score is not None else None,
            "selection_metric": "mIoU" if selection_score is not None else None,
            "selection_score": selection_score,
            "paired_validation": paired_validation,
            "calibrated_prototype_weight": calibrated_weight,
            "gradient_calibration": calibration_record,
            "train_generator_state": train_generator.get_state(),
            "rng_state": snapshot_rng_state(),
        },
        path,
    )


def run_smoke(
    args,
    model,
    criterion,
    optimizer,
    train_loader,
    device,
    output_dir,
    metadata,
):
    model.train()
    accumulator = PrototypeEpochAccumulator()
    records = []
    for batch_index, (rgb, sar, label) in enumerate(train_loader, start=1):
        if batch_index > args.smoke_batches:
            break
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        state = "sar" if args.arm in PROTOTYPE_ARMS else "full"
        weight = 0.1 if args.arm in PROTOTYPE_ARMS else 0.0
        optimizer.zero_grad(set_to_none=True)
        total, segmentation, prototype, statistics = forward_training_losses(
            model,
            criterion,
            rgb,
            sar,
            label,
            state=state,
            arm=args.arm,
            prototype_weight=weight,
        )
        if not torch.isfinite(total):
            raise RuntimeError("non-finite smoke loss")
        total.backward()
        optimizer.step()
        if statistics is not None:
            accumulator.update(prototype, statistics)
        records.append(
            {
                "batch": batch_index,
                "state": state,
                "total_loss": float(total.detach()),
                "segmentation_loss": float(segmentation.detach()),
                "prototype_loss": float(prototype.detach()),
            }
        )
    if len(records) != args.smoke_batches:
        raise RuntimeError("training loader ended before requested smoke batches")
    report = {
        "schema": "earthmiss_frmp5_prototype_smoke_v0",
        "formal": False,
        "training_was_performed": True,
        "protocol": metadata,
        "batches": records,
        "prototype": accumulator.summary(),
    }
    (output_dir / "smoke.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, allow_nan=False))


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    seed_everything(args.seed)
    health_record = (
        validate_health_report(args.health_report)
        if args.arm in PROTOTYPE_ARMS
        and not args.audit_only
        and not args.smoke_batches
        else None
    )

    output_dir = output_directory(args)
    if not args.audit_only:
        existing = (
            output_dir / "run.json",
            output_dir / "metrics.jsonl",
            output_dir / "last.pth",
            output_dir / "best_sar.pth",
            output_dir / "smoke.json",
            output_dir / "epoch_50.pth",
        )
        if not args.resume and any(path.exists() for path in existing):
            raise FileExistsError(f"refusing to overwrite FRM-P5 run: {output_dir}")

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
        train_generator,
    ) = build_loaders(args)
    if args.audit_only:
        audit_datasets(train_dataset, val_dataset)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("EarthMiss FRM-P5 training requires CUDA")
    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(
        args,
        train_dataset,
        val_dataset,
        train_loader,
        health_record,
    )
    device = torch.device("cuda")
    model = build_model_instance(weights_path, device)
    metadata["initial_model_state_sha256"] = model_state_sha256(model)
    criterion = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=IGNORE_INDEX),
        DiceLoss(smooth=0.05, ignore_index=IGNORE_INDEX),
        1.0,
        1.0,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-7,
    )

    if args.smoke_batches:
        (output_dir / "run.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        run_smoke(
            args,
            model,
            criterion,
            optimizer,
            train_loader,
            device,
            output_dir,
            metadata,
        )
        return

    metrics_path = output_dir / "metrics.jsonl"
    last_path = output_dir / "last.pth"
    start_epoch = 1
    best_score = float("-inf")
    best_epoch = None
    calibrated_weight = 0.0 if args.arm == "p0" else None
    calibration_record = None
    if args.resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol") != metadata:
            raise ValueError("resume checkpoint does not match frozen protocol")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_epoch = checkpoint["best_epoch"]
        calibrated_weight = checkpoint.get("calibrated_prototype_weight")
        calibration_record = checkpoint.get("gradient_calibration")
        train_generator.set_state(checkpoint["train_generator_state"])
        restore_rng_state(checkpoint["rng_state"])
        if start_epoch > 6 and args.arm in PROTOTYPE_ARMS:
            if calibrated_weight is None or calibration_record is None:
                raise ValueError("resume checkpoint lacks gradient calibration")

    run_path = output_dir / "run.json"
    if args.resume:
        if not run_path.is_file():
            raise FileNotFoundError("resume run lacks run.json")
        if json.loads(run_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("run.json does not match frozen resume protocol")
    else:
        run_path.write_text(
            json.dumps(metadata, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    for epoch in range(start_epoch, EPOCHS + 1):
        if epoch == 6 and args.arm in PROTOTYPE_ARMS and calibrated_weight is None:
            calibration_loader = build_calibration_loader(args)
            calibrated_weight, calibration_record = calibrate_prototype_weight(
                model,
                criterion,
                calibration_loader,
                arm=args.arm,
                device=device,
            )
            print(json.dumps({"event": "gradient_calibration", **calibration_record}))

        model.train()
        state_rng = random.Random(args.seed * 100_000 + epoch)
        multiplier = prototype_weight_multiplier(epoch)
        epoch_weight = (
            0.0
            if args.arm == "p0" or calibrated_weight is None
            else calibrated_weight * multiplier
        )
        total_sum = 0.0
        segmentation_sum = 0.0
        prototype_sum = 0.0
        batch_count = 0
        state_counts = {"sar": 0, "full": 0}
        prototype_accumulator = PrototypeEpochAccumulator()
        trace = hashlib.sha256()
        for rgb, sar, label in tqdm(
            train_loader,
            desc=f"{args.arm} epoch {epoch}/{EPOCHS}",
        ):
            state = train_state(state_rng)
            state_counts[state] += 1
            update_batch_trace(
                trace,
                state=state,
                rgb=rgb,
                sar=sar,
                label=label,
            )
            rgb = rgb.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            total, segmentation, prototype, statistics = forward_training_losses(
                model,
                criterion,
                rgb,
                sar,
                label,
                state=state,
                arm=args.arm,
                prototype_weight=epoch_weight,
            )
            if not torch.isfinite(total):
                raise RuntimeError("non-finite training loss")
            total.backward()
            optimizer.step()
            total_sum += float(total.detach())
            segmentation_sum += float(segmentation.detach())
            prototype_sum += float(prototype.detach())
            batch_count += 1
            if statistics is not None:
                prototype_accumulator.update(prototype, statistics)
        if batch_count == 0:
            raise RuntimeError("training loader produced no batches")
        scheduler.step()

        record = {
            "epoch": epoch,
            "train": {
                "mean_total_loss": total_sum / batch_count,
                "mean_segmentation_loss": segmentation_sum / batch_count,
                "mean_prototype_loss_all_batches": prototype_sum / batch_count,
                "prototype_weight": epoch_weight,
                "prototype_schedule_multiplier": multiplier,
                "state_batches": state_counts,
                "data_trace_sha256": trace.hexdigest(),
                "prototype": prototype_accumulator.summary(),
            },
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_calibration": calibration_record if epoch == 6 else None,
        }

        paired_validation = None
        if epoch % EVAL_INTERVAL == 0:
            sar_metrics = evaluate_with_cities(
                model,
                val_loader,
                val_dataset,
                "sar",
                device,
            )
            full_metrics = evaluate_with_cities(
                model,
                val_loader,
                val_dataset,
                "full",
                device,
            )
            paired_validation = {"sar": sar_metrics, "full": full_metrics}
            record["validation"] = paired_validation
            record["eligible_for_selection"] = epoch in ELIGIBLE_SELECTION_EPOCHS
            sar_score = sar_metrics["pooled"]["mIoU"]
            if epoch in ELIGIBLE_SELECTION_EPOCHS and sar_score > best_score:
                best_score = sar_score
                best_epoch = epoch
                save_checkpoint(
                    output_dir / "best_sar.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    metadata=metadata,
                    train_generator=train_generator,
                    calibrated_weight=calibrated_weight,
                    calibration_record=calibration_record,
                    checkpoint_role="primary_deployment",
                    selection_score=best_score,
                    paired_validation=paired_validation,
                )
            model.train()

        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_score=best_score,
            best_epoch=best_epoch,
            metadata=metadata,
            train_generator=train_generator,
            calibrated_weight=calibrated_weight,
            calibration_record=calibration_record,
            checkpoint_role="resume_only",
            paired_validation=paired_validation,
        )
        if epoch == EPOCHS:
            save_checkpoint(
                output_dir / "epoch_50.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                metadata=metadata,
                train_generator=train_generator,
                calibrated_weight=calibrated_weight,
                calibration_record=calibration_record,
                checkpoint_role="fixed_budget_endpoint",
                paired_validation=paired_validation,
            )
        print(json.dumps(record, allow_nan=False))

    if best_epoch is None:
        raise RuntimeError("no eligible SAR validation checkpoint was selected")


if __name__ == "__main__":
    main()
