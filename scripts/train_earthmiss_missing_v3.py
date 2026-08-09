"""Run the MetaRS-aligned EarthMiss V3 A/teacher/residual experiments.

This is intentionally a Test-developed released-protocol runner.  It trains on
the official Train split, evaluates only three fixed completed-step snapshots
on the official Test split, and records every result.  Residual arms freeze the
entire base model in eval mode and optimize only the BN-free SAR residual head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
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

from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
    seed_everything,
)


ARMS = ("a", "teacher-full", "r-ce", "r-priv")
RESIDUAL_ARMS = ("r-ce", "r-priv")
MAX_STEPS = 15_000
CHECKPOINT_STEPS = (6_620, 13_240, MAX_STEPS)
BATCH_SIZE = 8
WINDOW_SIZE = 512
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 0.05
POLY_POWER = 0.9
PRIVILEGED_WEIGHT = 1.0
PRIVILEGED_TEMPERATURE = 1.0
TEST_CLASS_IDS = list(range(8))
CLASS_NAMES = (
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agricultural",
    "Playground",
)
CACHE_SIZE = 64
PROTOCOL_REVISION = "earthmiss_missing_v3_metars_released_v2"
ZERO_TRAINING_GATE_SCHEMA = "earthmiss_missing_v3_zero_training_gates_v2"
DEFAULT_OUTPUT_ROOT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v3-metars-released"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--cache-size",
        type=int,
        default=CACHE_SIZE,
        help="Per-worker, per-modality EarthMiss LRU capacity.",
    )
    parser.add_argument(
        "--base-checkpoint",
        help="Test-selected V3 Run A checkpoint; required by residual arms.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        help="Test-selected V3 Full-teacher checkpoint; required by R-Priv.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--diagnostic-report",
        help="Formal zero-training gate report; required before training.",
    )
    parser.add_argument(
        "--confirm-zero-training-gates-passed",
        action="store_true",
        help="Record the explicit manual decision to proceed after report review.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if getattr(args, "cache_size", CACHE_SIZE) < 0:
        raise ValueError("--cache-size must be non-negative")
    if args.arm in RESIDUAL_ARMS and not args.base_checkpoint:
        raise ValueError(f"{args.arm} requires --base-checkpoint")
    if args.arm == "r-priv" and not args.teacher_checkpoint:
        raise ValueError("r-priv requires --teacher-checkpoint")
    if not getattr(args, "audit_only", False):
        if not getattr(args, "diagnostic_report", None):
            raise ValueError("training requires --diagnostic-report")
        if not getattr(args, "confirm_zero_training_gates_passed", False):
            raise ValueError(
                "training requires --confirm-zero-training-gates-passed"
            )


def validate_zero_training_gate_report(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"zero-training diagnostic not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != ZERO_TRAINING_GATE_SCHEMA:
        raise ValueError("unexpected zero-training diagnostic schema")
    if report.get("formal") is not True:
        raise ValueError("a smoke diagnostic cannot unlock V3 training")
    if report.get("training_was_performed") is not False:
        raise ValueError("gate report is not a zero-training diagnostic")
    expected_splits = {"test"}
    if set(report.get("splits", {})) != expected_splits:
        raise ValueError("gate report must contain the complete Test diagnosis")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "schema": report["schema"],
        "manual_review_confirmed": True,
    }


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_loaders(args):
    dataset_kwargs = {
        "dataset_root": args.dataset_root,
        "window_size": (WINDOW_SIZE, WINDOW_SIZE),
        "model_name": "DINOv3",
        "modality": "multi",
        "backbone_type": "dinov3_vits16",
        "cache_size": args.cache_size,
    }
    train_dataset = build_dataset("EarthMiss", "train", **dataset_kwargs)
    test_dataset = build_dataset("EarthMiss", "test", **dataset_kwargs)
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
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(args.num_workers, 2),
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return train_dataset, test_dataset, train_loader, test_loader, train_generator


def build_model_instance(weights_path, *, residual, seed, device):
    return build_model(
        model_name="DINOv3",
        backbone_weights=str(weights_path),
        backbone_type="dinov3_vits16",
        freeze_backbone=True,
        n_classes=8,
        use_lora=False,
        r=3,
        num_modalities=2,
        raw_logits=True,
        use_sar_logit_residual=residual,
        sar_logit_residual_seed=seed,
        sar_logit_residual_channels=64,
    ).to(device)


def _load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise ValueError(f"Checkpoint lacks a model state dict: {path}")
    return checkpoint


def load_base_into_residual(model, checkpoint):
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    expected_missing = {
        key
        for key in model.state_dict()
        if key.startswith("decoder.sar_logit_residual.")
    }
    if set(missing) != expected_missing or unexpected:
        raise ValueError(
            "Base checkpoint does not match the V3 residual model: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model.freeze_base_for_sar_logit_residual()


def snapshot_batchnorm_buffers(model):
    return {
        name: {
            "running_mean": module.running_mean.detach().cpu().clone(),
            "running_var": module.running_var.detach().cpu().clone(),
            "num_batches_tracked": module.num_batches_tracked.detach().cpu().clone(),
        }
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }


def assert_batchnorm_buffers_equal(expected, model):
    actual = snapshot_batchnorm_buffers(model)
    if actual.keys() != expected.keys():
        raise RuntimeError("Base BatchNorm module set changed")
    for name in expected:
        for field in expected[name]:
            if not torch.equal(expected[name][field], actual[name][field]):
                raise RuntimeError(f"Frozen BatchNorm buffer changed: {name}.{field}")


def privileged_masks(base_logits, teacher_logits, target):
    """Return the fixed teacher/base masks used by R-Priv and its audit."""

    num_classes = base_logits.shape[1]
    if teacher_logits.shape != base_logits.shape:
        raise ValueError("teacher and frozen-base logits must have equal shapes")
    if target.shape != base_logits.shape[:1] + base_logits.shape[2:]:
        raise ValueError("target shape does not match privileged logits")
    valid = (target >= 0) & (target < num_classes)
    teacher_prediction = teacher_logits.argmax(dim=1)
    base_prediction = base_logits.argmax(dim=1)
    base_correct = valid & (base_prediction == target)
    teacher_correct = valid & (teacher_prediction == target)
    return {
        "valid": valid,
        "base_error": valid & ~base_correct,
        "correction": teacher_correct & ~base_correct,
        "harmful": base_correct & ~teacher_correct,
    }


class PrivilegedMaskAccumulator:
    """Cumulative, resume-safe R-Priv mask statistics with no extra forward."""

    def __init__(self, num_classes=8):
        self.num_classes = num_classes
        self.batches = 0
        self.valid_by_class = torch.zeros(num_classes, dtype=torch.int64)
        self.base_error_by_class = torch.zeros(num_classes, dtype=torch.int64)
        self.correction_by_class = torch.zeros(num_classes, dtype=torch.int64)
        self.harmful_by_class = torch.zeros(num_classes, dtype=torch.int64)

    @staticmethod
    def _counts(target, mask, num_classes):
        return torch.bincount(
            target[mask].detach().to("cpu", torch.int64),
            minlength=num_classes,
        )

    def update(self, target, masks):
        required = {"valid", "base_error", "correction", "harmful"}
        if set(masks) != required:
            raise ValueError("privileged mask bundle is incomplete")
        self.batches += 1
        self.valid_by_class += self._counts(
            target, masks["valid"], self.num_classes
        )
        self.base_error_by_class += self._counts(
            target, masks["base_error"], self.num_classes
        )
        self.correction_by_class += self._counts(
            target, masks["correction"], self.num_classes
        )
        self.harmful_by_class += self._counts(
            target, masks["harmful"], self.num_classes
        )

    def state_dict(self):
        return {
            "num_classes": self.num_classes,
            "batches": self.batches,
            "valid_by_class": self.valid_by_class.clone(),
            "base_error_by_class": self.base_error_by_class.clone(),
            "correction_by_class": self.correction_by_class.clone(),
            "harmful_by_class": self.harmful_by_class.clone(),
        }

    def load_state_dict(self, state):
        if state.get("num_classes") != self.num_classes:
            raise ValueError("privileged-mask class count changed on resume")
        self.batches = int(state["batches"])
        for field in (
            "valid_by_class",
            "base_error_by_class",
            "correction_by_class",
            "harmful_by_class",
        ):
            value = torch.as_tensor(state[field], dtype=torch.int64).cpu()
            if value.shape != (self.num_classes,):
                raise ValueError(f"invalid privileged-mask state: {field}")
            setattr(self, field, value.clone())

    @staticmethod
    def _ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    def summary(self):
        valid = int(self.valid_by_class.sum())
        base_error = int(self.base_error_by_class.sum())
        correction = int(self.correction_by_class.sum())
        harmful = int(self.harmful_by_class.sum())
        by_class = []
        for class_id in range(self.num_classes):
            class_valid = int(self.valid_by_class[class_id])
            class_error = int(self.base_error_by_class[class_id])
            class_correction = int(self.correction_by_class[class_id])
            class_harmful = int(self.harmful_by_class[class_id])
            by_class.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "valid_pixels": class_valid,
                    "base_error_pixels": class_error,
                    "teacher_correct_base_wrong_pixels": class_correction,
                    "teacher_wrong_base_correct_pixels": class_harmful,
                    "q_abs": self._ratio(class_correction, class_valid),
                    "q_cov": self._ratio(class_correction, class_error),
                }
            )
        return {
            "batches": self.batches,
            "valid_pixels": valid,
            "base_error_pixels": base_error,
            "teacher_correct_base_wrong_pixels": correction,
            "teacher_wrong_base_correct_pixels": harmful,
            "teacher_minus_base_net_correct_pixels": correction - harmful,
            "q_abs": self._ratio(correction, valid),
            "q_cov": self._ratio(correction, base_error),
            "by_class": by_class,
        }


def reliable_privileged_loss(
    student_logits,
    base_logits,
    teacher_logits,
    target,
    *,
    temperature=PRIVILEGED_TEMPERATURE,
    correction_mask=None,
):
    """KL only where frozen Full is correct and frozen SAR is wrong."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if correction_mask is None:
        correction_mask = privileged_masks(
            base_logits, teacher_logits, target
        )["correction"]
    if correction_mask.shape != target.shape or correction_mask.dtype != torch.bool:
        raise ValueError("correction_mask must be a boolean target-shaped tensor")
    if not correction_mask.any():
        return student_logits.sum() * 0.0, 0
    teacher_probability = F.softmax(
        teacher_logits.detach() / temperature,
        dim=1,
    )
    student_log_probability = F.log_softmax(
        student_logits / temperature,
        dim=1,
    )
    per_pixel = F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="none",
    ).sum(dim=1) * (temperature**2)
    return per_pixel[correction_mask].mean(), int(correction_mask.sum())


@torch.no_grad()
def evaluate_test(model, loader, state, device):
    model.eval()
    evaluator = EarthMissMetrics()
    availability = canonical_availability(state, batch_size=1, device=device)
    stride = int(WINDOW_SIZE * 2 / 3)
    for rgb, sar, label in tqdm(loader, desc=f"Test {state}", leave=False):
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        logits = slide_inference(
            rgb,
            model,
            n_output_channels=8,
            crop_size=(WINDOW_SIZE, WINDOW_SIZE),
            stride=(stride, stride),
            dsm=sar,
            availability=availability,
            batch_size=BATCH_SIZE * 4,
        )
        evaluator.update(logits.argmax(dim=1), label)
    metrics = evaluator.compute()
    if metrics["selection_class_ids"] != TEST_CLASS_IDS:
        raise RuntimeError(
            "EarthMiss Test support changed: "
            f"{metrics['selection_class_ids']} != {TEST_CLASS_IDS}"
        )
    return metrics


def build_metadata(
    args,
    train_dataset,
    test_dataset,
    train_loader,
    gate_record,
):
    base_path = Path(args.base_checkpoint) if args.base_checkpoint else None
    teacher_path = (
        Path(args.teacher_checkpoint) if args.teacher_checkpoint else None
    )
    endpoint = "full" if args.arm == "teacher-full" else "sar"
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "arm": args.arm,
        "seed": args.seed,
        "train_tiles": len(train_dataset),
        "test_tiles": len(test_dataset),
        "train_steps_per_pass": len(train_loader),
        "model": {
            "backbone": "dinov3_vits16_lvd1689m_frozen",
            "segmentation_head": "raw_conv1x1",
            "sar_logit_residual": args.arm in RESIDUAL_ARMS,
            "residual_location": "post_neck_p2_after_all_existing_bn",
            "residual_normalization": "groupnorm_only",
            "residual_final_projection": "zero_initialized",
        },
        "initialization": {
            "base_checkpoint": str(base_path) if base_path else None,
            "base_checkpoint_sha256": file_sha256(base_path) if base_path else None,
            "teacher_checkpoint": str(teacher_path) if teacher_path else None,
            "teacher_checkpoint_sha256": (
                file_sha256(teacher_path) if teacher_path else None
            ),
        },
        "zero_training_gate": gate_record,
        "training": {
            "endpoint": endpoint,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "PolynomialLR",
            "scheduler_power": POLY_POWER,
            "optimizer_steps": MAX_STEPS,
            "data_loader_workers": args.num_workers,
            "per_worker_per_modality_cache_size": args.cache_size,
            "base_mode_for_residual_arms": "eval_frozen",
            "trainable_scope": (
                "sar_logit_residual_only"
                if args.arm in RESIDUAL_ARMS
                else "adapter_decoder_with_frozen_backbone"
            ),
            "privileged_weight": (
                PRIVILEGED_WEIGHT if args.arm == "r-priv" else 0.0
            ),
            "privileged_mask": (
                "full_teacher_correct_and_frozen_sar_wrong"
                if args.arm == "r-priv"
                else None
            ),
            "privileged_mask_audit": (
                "cumulative_total_and_per_class_saved_at_candidates_and_resume"
                if args.arm == "r-priv"
                else None
            ),
        },
        "evaluation": {
            "selection_split": "test_city_holdout_test_developed",
            "selection_endpoint": endpoint,
            "selection_metric": "official_ever_mIoU_fixed_8_classes",
            "completed_step_candidates": list(CHECKPOINT_STEPS),
            "inference": "fp32_sliding_512_stride341",
            "bn_state": "raw_online_checkpoint_no_posthoc_bank",
            "all_candidate_scores_reported": True,
        },
    }


def save_snapshot(
    path,
    model,
    step,
    score,
    metrics,
    metadata,
    privileged_mask_summary=None,
):
    torch.save(
        {
            "model": model.state_dict(),
            "completed_optimizer_steps": step,
            "run": metadata["arm"],
            "seed": metadata["seed"],
            "protocol": metadata,
            "checkpoint_role": "test_candidate_released_protocol",
            "selection_state": metadata["evaluation"]["selection_endpoint"],
            "selection_metric": "official_ever_mIoU",
            "selection_score": score,
            "test_metrics": metrics,
            "privileged_mask_summary": privileged_mask_summary,
        },
        path,
    )


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_resume(
    path,
    model,
    optimizer,
    scheduler,
    completed_steps,
    completed_passes,
    train_generator,
    metadata,
    privileged_mask_state,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "completed_optimizer_steps": completed_steps,
            "completed_train_passes": completed_passes,
            "train_generator_state": train_generator.get_state(),
            "rng_state": rng_state(),
            "protocol": metadata,
            "privileged_mask_state": privileged_mask_state,
        },
        path,
    )


def main():
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    (
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
        train_generator,
    ) = build_loaders(args)
    if args.audit_only:
        print(
            json.dumps(
                {
                    "train_tiles": len(train_dataset),
                    "test_tiles": len(test_dataset),
                    "steps_per_pass": len(train_loader),
                    "fixed_checkpoint_steps": CHECKPOINT_STEPS,
                    "cache_size": args.cache_size,
                },
                indent=2,
            )
        )
        return
    gate_record = validate_zero_training_gate_report(args.diagnostic_report)
    if not torch.cuda.is_available():
        raise RuntimeError("EarthMiss V3 training requires a CUDA GPU")
    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    device = torch.device("cuda")
    residual_arm = args.arm in RESIDUAL_ARMS
    model = build_model_instance(
        weights_path,
        residual=residual_arm,
        seed=args.seed,
        device=device,
    )
    if residual_arm:
        base_checkpoint = _load_checkpoint(args.base_checkpoint, device)
        trainable_parameters = load_base_into_residual(model, base_checkpoint)
    else:
        trainable_parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )

    teacher = None
    if args.arm == "r-priv":
        teacher = build_model_instance(
            weights_path,
            residual=False,
            seed=args.seed,
            device=device,
        )
        teacher_checkpoint = _load_checkpoint(args.teacher_checkpoint, device)
        teacher.load_state_dict(teacher_checkpoint["model"], strict=True)
        teacher.requires_grad_(False)
        teacher.eval()

    criterion = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=8),
        DiceLoss(smooth=0.05, ignore_index=8),
        1.0,
        1.0,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=MAX_STEPS,
        power=POLY_POWER,
    )

    output_dir = Path(args.output_root) / f"run_{args.arm}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    resume_path = output_dir / "last.pth"
    best_path = output_dir / "best_test.json"
    metadata = build_metadata(
        args,
        train_dataset,
        test_dataset,
        train_loader,
        gate_record,
    )
    if not args.resume and (metrics_path.exists() or resume_path.exists()):
        raise FileExistsError(f"Refusing to overwrite V3 run: {output_dir}")
    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    completed_steps = 0
    completed_passes = 0
    best_score = float("-inf")
    privileged_mask_accumulator = (
        PrivilegedMaskAccumulator() if args.arm == "r-priv" else None
    )
    if args.resume:
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume.get("protocol") != metadata:
            raise ValueError("Resume checkpoint does not match the frozen V3 protocol")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        completed_steps = int(resume["completed_optimizer_steps"])
        completed_passes = int(resume["completed_train_passes"])
        train_generator.set_state(resume["train_generator_state"])
        restore_rng_state(resume["rng_state"])
        saved_mask_state = resume.get("privileged_mask_state")
        if privileged_mask_accumulator is not None:
            if not isinstance(saved_mask_state, dict):
                raise ValueError("R-Priv resume lacks privileged-mask state")
            privileged_mask_accumulator.load_state_dict(saved_mask_state)
        elif saved_mask_state is not None:
            raise ValueError("non-R-Priv resume contains privileged-mask state")
        if best_path.is_file():
            best_score = json.loads(best_path.read_text(encoding="utf-8"))["score"]

    frozen_bn = snapshot_batchnorm_buffers(model) if residual_arm else None
    sar_state = canonical_availability("sar", batch_size=BATCH_SIZE, device=device)
    full_state = canonical_availability("full", batch_size=BATCH_SIZE, device=device)
    train_state = full_state if args.arm == "teacher-full" else sar_state
    endpoint = "full" if args.arm == "teacher-full" else "sar"

    while completed_steps < MAX_STEPS:
        if residual_arm:
            model.freeze_base_for_sar_logit_residual()
        else:
            model.train()
        for rgb, sar, label in tqdm(
            train_loader,
            desc=f"{args.arm} pass {completed_passes + 1}",
        ):
            rgb = rgb.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            batch_size = rgb.shape[0]
            availability = train_state[:1].expand(batch_size, -1)

            optimizer.zero_grad(set_to_none=True)
            if args.arm == "r-priv":
                base_logits, logits = model.forward_sar_residual_components(
                    rgb, sar
                )
            else:
                logits = model(rgb, sar, availability=availability)
                base_logits = None
            segmentation_loss = criterion(logits, label)
            privileged_loss = logits.sum() * 0.0
            correction_pixels = 0
            mask_bundle = None
            if args.arm == "r-priv":
                with torch.no_grad():
                    teacher_logits = teacher(
                        rgb,
                        sar,
                        availability=canonical_availability(
                            "full", batch_size=batch_size, device=device
                        ),
                    )
                mask_bundle = privileged_masks(
                    base_logits,
                    teacher_logits,
                    label,
                )
                privileged_mask_accumulator.update(label, mask_bundle)
                privileged_loss, correction_pixels = reliable_privileged_loss(
                    logits,
                    base_logits,
                    teacher_logits,
                    label,
                    correction_mask=mask_bundle["correction"],
                )
            loss = segmentation_loss + PRIVILEGED_WEIGHT * privileged_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            completed_steps += 1

            if completed_steps in CHECKPOINT_STEPS:
                if residual_arm:
                    assert_batchnorm_buffers_equal(frozen_bn, model)
                metrics = evaluate_test(model, test_loader, endpoint, device)
                score = metrics["official_ever_mIoU"]
                snapshot_name = f"step_{completed_steps}.pth"
                save_snapshot(
                    output_dir / snapshot_name,
                    model,
                    completed_steps,
                    score,
                    metrics,
                    metadata,
                    (
                        privileged_mask_accumulator.summary()
                        if privileged_mask_accumulator is not None
                        else None
                    ),
                )
                record = {
                    "completed_optimizer_steps": completed_steps,
                    "endpoint": endpoint,
                    "official_ever_mIoU": score,
                    "metrics": metrics,
                    "last_train_batch": {
                        "loss": float(loss.detach()),
                        "segmentation_loss": float(segmentation_loss.detach()),
                        "privileged_loss": float(privileged_loss.detach()),
                        "correction_pixels": correction_pixels,
                    },
                    "privileged_mask_summary": (
                        privileged_mask_accumulator.summary()
                        if privileged_mask_accumulator is not None
                        else None
                    ),
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                if score > best_score:
                    best_score = score
                    best_path.write_text(
                        json.dumps(
                            {
                                "step": completed_steps,
                                "score": score,
                                "checkpoint": snapshot_name,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                if residual_arm:
                    model.freeze_base_for_sar_logit_residual()
                else:
                    model.train()

            if completed_steps >= MAX_STEPS:
                break

        completed_passes += 1
        if residual_arm:
            assert_batchnorm_buffers_equal(frozen_bn, model)
        save_resume(
            resume_path,
            model,
            optimizer,
            scheduler,
            completed_steps,
            completed_passes,
            train_generator,
            metadata,
            (
                privileged_mask_accumulator.state_dict()
                if privileged_mask_accumulator is not None
                else None
            ),
        )

    print(best_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
