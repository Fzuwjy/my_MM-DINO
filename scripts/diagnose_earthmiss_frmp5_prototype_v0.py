"""Train-only BN and class-geometry health gate for F2S-CSPT v0."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import build_dataset  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from models.MMDINO.dino_segment import build_model  # noqa: E402
from scripts.earthmiss_frmp5_prototype_common import (  # noqa: E402
    assert_batchnorm_buffers_equal,
    batch_class_prototypes,
    preserve_rng_state,
    restore_batchnorm_buffers,
    snapshot_batchnorm_buffers,
    supported_prototype_separation,
    temporary_batchnorm_eval,
)
from scripts.train_earthmiss_frmp5_prototype_v0 import (  # noqa: E402
    BATCH_SIZE,
    HEALTH_SCHEMA,
    IGNORE_INDEX,
    MINIMUM_RAW_SUPPORT,
    NUM_CLASSES,
    WINDOW_SIZE,
    file_sha256,
)
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
    seed_everything,
)


EXPECTED_RUN_C_CHECKPOINT_SHA256 = (
    "b038dfcfe771ca5c67da500acc2b88e74f066496cac332fc3b76d4377b73dff9"
)
EXPECTED_RUN_C_EPOCH = 15
EXPECTED_RUN_C_SEED = 42
PROBE_BATCHES = 32
PROBE_SEED = 20_260_811
MIN_VALID_GEOMETRY_FRACTION = 0.8
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-frmp5-prototype-v0/"
    "run-c-e15-train-health.json"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args(argv)


def validate_args(args):
    if args.smoke_batches < 0:
        raise ValueError("--smoke-batches must be non-negative")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite health report: {output}")


def load_run_c_model(checkpoint_path, weights_path, device, *, formal):
    checkpoint_path = Path(checkpoint_path)
    weights_path = Path(weights_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Run C checkpoint not found: {checkpoint_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
    checkpoint_sha = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    record = {
        "path": str(checkpoint_path),
        "sha256": checkpoint_sha,
        "run": str(checkpoint.get("run", "")).lower(),
        "epoch": checkpoint.get("epoch"),
        "seed": checkpoint.get("seed"),
        "selection_state": checkpoint.get("selection_state"),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
    }
    if record["run"] != "c" or record["selection_state"] != "sar":
        raise ValueError("health gate requires the primary Run C SAR checkpoint")
    if formal:
        expected = {
            "sha256": EXPECTED_RUN_C_CHECKPOINT_SHA256,
            "epoch": EXPECTED_RUN_C_EPOCH,
            "seed": EXPECTED_RUN_C_SEED,
            "checkpoint_role": "primary_deployment",
        }
        for key, value in expected.items():
            if record[key] != value:
                raise ValueError(
                    f"formal health checkpoint {key} mismatch: "
                    f"expected {value!r}, got {record[key]!r}"
                )

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
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.requires_grad_(True)
    model.backbone.requires_grad_(False)
    return model, record


def build_probe_loader(args, batches):
    dataset = build_dataset(
        "EarthMiss",
        "train",
        dataset_root=args.dataset_root,
        window_size=(WINDOW_SIZE, WINDOW_SIZE),
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
        cache_size=0,
    )
    generator = torch.Generator().manual_seed(PROBE_SEED)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        generator=generator,
    )
    if len(loader) < batches:
        raise RuntimeError("Train split is too short for the health probe")
    return dataset, loader


def _cross_state_geometry(full_batch, sar_batch):
    support = full_batch.support_mask & sar_batch.support_mask
    full = full_batch.prototypes[support]
    sar = sar_batch.prototypes[support]
    if full.shape[0] < 2:
        return None
    cosine = sar @ full.transpose(0, 1)
    positive = cosine.diagonal()
    off_diagonal_mask = ~torch.eye(
        cosine.shape[0], dtype=torch.bool, device=cosine.device
    )
    negative_rows = cosine.masked_fill(~off_diagonal_mask, float("-inf"))
    positive_margin = positive - negative_rows.max(dim=1).values
    return {
        "supported_classes": int(full.shape[0]),
        "full_separation": float(supported_prototype_separation(full_batch)),
        "sar_separation": float(supported_prototype_separation(sar_batch)),
        "positive_cross_state_cosine": float(positive.mean()),
        "positive_nearest_negative_margin": float(positive_margin.mean()),
    }


def extract_mode_features(model, rgb, sar, outputs, *, mode):
    batch_size = rgb.shape[0]
    availability = {
        state: canonical_availability(
            state, batch_size=batch_size, device=rgb.device
        )
        for state in ("full", "sar")
    }
    baseline = snapshot_batchnorm_buffers(model.decoder.frm)
    features = {}
    if mode == "eval":
        model.eval()
        with torch.no_grad(), preserve_rng_state():
            for state in ("full", "sar"):
                features[state] = model.extract_state_frm_p5_from_backbone_outputs(
                    rgb,
                    sar,
                    backbone_outputs=outputs,
                    availability=availability[state],
                )
    elif mode == "train":
        model.train()
        model.backbone.eval()
        with torch.no_grad(), preserve_rng_state():
            for state in ("full", "sar"):
                restore_batchnorm_buffers(baseline, model.decoder.frm)
                features[state] = model.extract_state_frm_p5_from_backbone_outputs(
                    rgb,
                    sar,
                    backbone_outputs=outputs,
                    availability=availability[state],
                )
        restore_batchnorm_buffers(baseline, model.decoder.frm)
    else:
        raise ValueError(f"unsupported BN mode: {mode}")
    assert_batchnorm_buffers_equal(baseline, model.decoder.frm)
    return features


class GeometryAccumulator:
    def __init__(self):
        self.rows = []

    def update(self, row):
        if row is not None and all(math.isfinite(value) for value in row.values()):
            self.rows.append(row)

    def summary(self):
        if not self.rows:
            return None
        keys = self.rows[0].keys()
        means = {
            key: float(np.mean([row[key] for row in self.rows])) for key in keys
        }
        means["valid_batches"] = len(self.rows)
        means["full_minus_sar_separation"] = (
            means["full_separation"] - means["sar_separation"]
        )
        return means


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    formal = args.smoke_batches == 0
    requested_batches = PROBE_BATCHES if formal else args.smoke_batches
    if requested_batches <= 0:
        raise ValueError("smoke health probe requires positive --smoke-batches")
    seed_everything(PROBE_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("FRM-P5 health probe requires CUDA")
    device = torch.device("cuda")
    model, checkpoint_record = load_run_c_model(
        args.checkpoint,
        args.backbone_weights,
        device,
        formal=formal,
    )
    dataset, loader = build_probe_loader(args, requested_batches)
    accumulators = {mode: GeometryAccumulator() for mode in ("train", "eval")}
    inclusion = np.zeros(NUM_CLASSES, dtype=np.int64)
    raw_support_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    purity_mass_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    ess_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    processed = 0

    for rgb, sar, target in loader:
        if processed >= requested_batches:
            break
        processed += 1
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        model.eval()
        outputs = model.extract_frozen_backbone_outputs(rgb, sar)
        for mode in ("train", "eval"):
            features = extract_mode_features(model, rgb, sar, outputs, mode=mode)
            full_batch = batch_class_prototypes(
                features["full"],
                target,
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
                minimum_raw_support=MINIMUM_RAW_SUPPORT,
            )
            sar_batch = batch_class_prototypes(
                features["sar"],
                target,
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
                minimum_raw_support=MINIMUM_RAW_SUPPORT,
            )
            accumulators[mode].update(_cross_state_geometry(full_batch, sar_batch))
            if mode == "eval":
                mask = full_batch.support_mask.detach().cpu().numpy().astype(bool)
                inclusion += mask.astype(np.int64)
                raw_support_sum += full_batch.raw_support.detach().cpu().numpy()
                purity_mass_sum += full_batch.purity_mass.detach().cpu().numpy()
                ess_sum += full_batch.effective_sample_size.detach().cpu().numpy()

    summaries = {mode: acc.summary() for mode, acc in accumulators.items()}
    minimum_valid_batches = math.ceil(
        requested_batches * MIN_VALID_GEOMETRY_FRACTION
    )
    gates = {
        "complete_budget": processed == requested_batches,
        "sufficient_train_bn_geometry_batches": (
            summaries["train"] is not None
            and summaries["train"]["valid_batches"] >= minimum_valid_batches
        ),
        "sufficient_eval_bn_geometry_batches": (
            summaries["eval"] is not None
            and summaries["eval"]["valid_batches"] >= minimum_valid_batches
        ),
        "train_bn_full_separation_exceeds_sar": (
            summaries["train"] is not None
            and summaries["train"]["full_minus_sar_separation"] > 0.0
        ),
        "eval_bn_full_separation_exceeds_sar": (
            summaries["eval"] is not None
            and summaries["eval"]["full_minus_sar_separation"] > 0.0
        ),
    }
    training_allowed = formal and all(gates.values())
    report = {
        "schema": HEALTH_SCHEMA,
        "formal": formal,
        "training_was_performed": False,
        "checkpoint": checkpoint_record,
        "protocol": {
            "split": "official Train only",
            "seed": PROBE_SEED,
            "batch_size": BATCH_SIZE,
            "requested_batches": requested_batches,
            "processed_batches": processed,
            "minimum_valid_geometry_batches": minimum_valid_batches,
            "train_transform": True,
            "num_workers": 0,
            "feature": "Full-state/SAR-state post-FRM P5",
            "bn_modes": ["state-specific train batch statistics", "shared eval buffers"],
            "minimum_raw_support": MINIMUM_RAW_SUPPORT,
            "purity_weight": "a^2",
        },
        "geometry": summaries,
        "support": {
            "class_inclusion_batches": inclusion.tolist(),
            "mean_raw_support": (raw_support_sum / processed).tolist(),
            "mean_purity_mass": (purity_mass_sum / processed).tolist(),
            "mean_effective_sample_size": (ess_sum / processed).tolist(),
            "all_classes_observed_at_least_once": bool(np.all(inclusion > 0)),
        },
        "decision": {
            "gates": gates,
            "training_allowed": training_allowed,
            "support_is_reported_not_tuned": True,
            "manual_review_still_required": True,
        },
        "dataset": {
            "train_tiles": len(dataset),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
