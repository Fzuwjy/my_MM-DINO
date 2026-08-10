"""Train the isolated EarthMiss P5-to-P4 residual-flow experiment.

The structural and optimization baseline is Run C.  A later Run-A control is
accepted by the same launcher only after C+FAM passes its pre-registered Val
screen.  Both candidates start from the same common DINOv3/MM-DINO
initialization as their corresponding baseline; C-E15 is diagnostic evidence,
not a warm-start checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from losses import DiceLoss, JointLoss, SoftCrossEntropyLoss  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from models.MMDINO.dino_segment import build_model  # noqa: E402
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
    VAL_SELECTION_CLASS_IDS,
    audit_datasets,
    build_run_metadata as build_baseline_run_metadata,
    build_loaders,
    evaluate,
    save_checkpoint,
    seed_everything,
    train_state,
    update_early_stopping_state,
    validation_states,
)


SEED = 42
FAM_SEED = 42
FLOW_CHANNELS = 128
PROTOCOL_REVISION = "earthmiss_prn_p5p4_residual_flow_v1"
DEFAULT_OUTPUT_ROOT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-prn-p5p4-fam-v1"
)
REFERENCE_SOURCES = {
    "torchcv": {
        "url": "https://github.com/donnyyou/torchcv.git",
        "commit": "5cb5203fd6afaefd644f0a22de83bef5684da7bd",
        "implementation": "model/seg/nets/sfnet.py::AlignModule",
    },
    "sfsegnets": {
        "url": "https://github.com/lxtGH/SFSegNets.git",
        "commit": "ba475bafbfde7ce0caf94b3d839574ab0b9b953a",
        "implementation": "network/nn/operators.py::AlignedModule",
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        choices=("A", "C"),
        required=True,
        help="Run C is primary; Run A is a conditional attribution control.",
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fam-seed", type=int, default=FAM_SEED)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument(
        "--early-stop-patience-evals",
        type=int,
        default=0,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit Train/Val data without constructing the model.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.window_size <= 0 or args.window_size % 16:
        raise ValueError("--window-size must be a positive multiple of 16")
    if args.epochs <= 0 or args.eval_interval <= 0:
        raise ValueError("--epochs and --eval-interval must be positive")
    if args.early_stop_patience_evals < 0:
        raise ValueError("--early-stop-patience-evals must be non-negative")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid loader configuration")


def build_run_metadata(args, train_dataset, val_dataset, train_loader):
    metadata = build_baseline_run_metadata(
        args,
        train_dataset,
        val_dataset,
        train_loader,
    )
    states = validation_states(args.run)
    checkpoint_roles = {"best_sar.pth": "primary_deployment"}
    if "full" in states:
        checkpoint_roles["best_full.pth"] = "diagnostic_only"
    metadata.update({
        "protocol_revision": PROTOCOL_REVISION,
        "experiment": "prn_p5_to_p4_residual_flow_alignment",
        "fam_seed": args.fam_seed,
        "training_start": {
            "policy": "same_common_initialization_as_corresponding_baseline",
            "warm_start_from_c_e15": False,
            "c_e15_role": "diagnostic_evidence_only",
        },
        "model": {
            **metadata["model"],
            "base": "DINOv3 SampleAdapter + Decoder (Run C architecture)",
            "frozen_backbone": True,
            "raw_logits": True,
            "fam": {
                "enabled": True,
                "location": "PRN P5 nearest resize output before P4 concat",
                "direction": "coarse_P5_to_fine_P4_only",
                "formula": (
                    "nearest(P5)+warp(P5,predicted_flow)-warp(P5,zero_flow)"
                ),
                "flow_channels": FLOW_CHANNELS,
                "kernel_size": 3,
                "flow_units": "P4_target_grid_pixels",
                "align_corners": False,
                "padding_mode": "border",
                "flow_predictor_initialization": "zeros",
                "normalization": "none",
                "learnable_gate": False,
            },
            "forbidden_combinations": [
                "FSD-Down",
                "multi-scale FAM",
                "bidirectional warp",
                "optical stem",
                "SAR logit residual",
                "prototype loss",
            ],
        },
        "reference_sources": REFERENCE_SOURCES,
        "data_and_optimization": {
            "augmentation": "see inherited augmentation contract",
            "loss": "SoftCrossEntropy(0.05)+Dice(0.05), ignore=8",
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.01,
            "scheduler": "CosineAnnealingLR",
            "steps_per_epoch": len(train_loader),
            "planned_optimizer_steps": len(train_loader) * args.epochs,
        },
        "evaluation": {
            **metadata["evaluation"],
            "selection_state": "sar",
            "selection_split": "val_city_holdout",
            "checkpoint_roles": checkpoint_roles,
            "conditional_control": (
                "run A+FAM only after C+FAM passes the Val screen"
            ),
        },
        "early_stopping": {
            "selection_state": "sar",
            "strict_improvement": True,
            "patience_evaluations": args.early_stop_patience_evals,
            "evaluation_interval_epochs": args.eval_interval,
            "disabled": args.early_stop_patience_evals == 0,
        },
        "determinism": {
            "same_data_trace_and_common_initialization_required": True,
            "step_zero_run_c_equivalence_required": True,
            "bitwise_training_checkpoint_reproducibility_required": False,
            "reason": "CUDA grid_sample backward may be nondeterministic",
        },
    })
    return metadata


def build_fam_model(args, weights_path, device):
    model = build_model(
        model_name="DINOv3",
        backbone_weights=str(weights_path),
        backbone_type="dinov3_vits16",
        freeze_backbone=True,
        n_classes=8,
        use_lora=False,
        r=3,
        num_modalities=2,
        raw_logits=True,
        use_prn_p5_p4_fam=True,
        prn_p5_p4_fam_seed=args.fam_seed,
        prn_p5_p4_fam_flow_channels=FLOW_CHANNELS,
    ).to(device)
    fam = model.decoder.neck.p5_p4_fam
    if fam is None or not model.decoder.use_prn_p5_p4_fam:
        raise RuntimeError("the requested P5-to-P4 FAM was not constructed")
    if any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in fam.modules()
    ):
        raise RuntimeError("the FAM experiment must not add BatchNorm")
    if torch.count_nonzero(fam.flow_predictor.weight).item() != 0:
        raise RuntimeError("the FAM flow predictor is not zero initialized")
    return model


def output_directory(args):
    return Path(args.output_root) / f"fam_run_{args.run.lower()}_seed{args.seed}"


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    seed_everything(args.seed)
    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    if args.audit_only:
        audit_datasets(train_dataset, val_dataset)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("the EarthMiss P5-to-P4 FAM experiment requires CUDA")
    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    output_dir = output_directory(args)
    existing_artifacts = (
        output_dir / "run.json",
        output_dir / "metrics.jsonl",
        output_dir / "last.pth",
        output_dir / "best_sar.pth",
        output_dir / "best_full.pth",
    )
    if not args.resume and any(path.exists() for path in existing_artifacts):
        raise FileExistsError(
            f"Refusing to overwrite an existing EarthMiss FAM run: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = output_dir / "last.pth"
    metrics_path = output_dir / "metrics.jsonl"

    device = torch.device("cuda")
    model = build_fam_model(args, weights_path, device)
    criterion = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=8),
        DiceLoss(smooth=0.05, ignore_index=8),
        1.0,
        1.0,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    start_epoch = 1
    best = {state: float("-inf") for state in validation_states(args.run)}
    early_stopping_state = {"bad_validation_count": 0, "best_epoch": None}
    metadata = build_run_metadata(args, train_dataset, val_dataset, train_loader)

    if args.resume:
        checkpoint = torch.load(last_checkpoint, map_location=device)
        if checkpoint["run"] != args.run or checkpoint["seed"] != args.seed:
            raise ValueError("Resume checkpoint does not match --run/--seed")
        if checkpoint.get("protocol") != metadata:
            raise ValueError("Resume checkpoint does not match the frozen protocol")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint["best"]
        saved_early_stopping = checkpoint.get("early_stopping_state")
        if not isinstance(saved_early_stopping, dict):
            raise ValueError("Resume checkpoint lacks early-stopping state")
        early_stopping_state = dict(saved_early_stopping)
        if (
            args.early_stop_patience_evals > 0
            and early_stopping_state["bad_validation_count"]
            >= args.early_stop_patience_evals
        ):
            raise ValueError("Resume checkpoint has already triggered early stopping")

    (output_dir / "run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        state_rng = random.Random(args.seed * 100_000 + epoch)
        loss_sum = 0.0
        batch_count = 0
        for rgb, sar, label in tqdm(
            train_loader, desc=f"FAM epoch {epoch}/{args.epochs}"
        ):
            state = train_state(args.run, state_rng)
            rgb = rgb.to(device, non_blocking=True)
            sar = sar.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            availability = canonical_availability(
                state, batch_size=rgb.shape[0], device=device
            )
            optimizer.zero_grad()
            logits = model(rgb, sar, availability=availability)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            batch_count += 1
        if batch_count == 0:
            raise RuntimeError("EarthMiss FAM training loader produced no batches")
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": loss_sum / batch_count,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            for state in validation_states(args.run):
                metrics = evaluate(
                    model,
                    val_loader,
                    state,
                    device,
                    args.window_size,
                    args.batch_size * 4,
                )
                record[state] = metrics
                improved = metrics["mIoU"] > best[state]
                if state == "sar":
                    early_stopping_state = update_early_stopping_state(
                        early_stopping_state,
                        improved=improved,
                        epoch=epoch,
                    )
                if improved:
                    best[state] = metrics["mIoU"]
                    save_checkpoint(
                        output_dir / f"best_{state}.pth",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        best,
                        metadata,
                        checkpoint_role=(
                            "primary_deployment"
                            if state == "sar"
                            else "diagnostic_only"
                        ),
                        selection_state=state,
                        early_stopping_state=early_stopping_state,
                    )
            record["early_stopping"] = {
                **early_stopping_state,
                "patience_evaluations": args.early_stop_patience_evals,
                "stop": (
                    args.early_stop_patience_evals > 0
                    and early_stopping_state["bad_validation_count"]
                    >= args.early_stop_patience_evals
                ),
            }
            model.train()

        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            scheduler,
            epoch,
            best,
            metadata,
            checkpoint_role="resume_only",
            early_stopping_state=early_stopping_state,
        )
        print(json.dumps(record))
        if record.get("early_stopping", {}).get("stop", False):
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "epoch": epoch,
                        "selection_state": "sar",
                        "best_epoch": early_stopping_state["best_epoch"],
                        "best_mIoU": best["sar"],
                    }
                )
            )
            break


if __name__ == "__main__":
    main()
