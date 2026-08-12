"""Locate causal Run C bottlenecks with read-only Full-feature interventions.

For every deterministic EarthMiss Val crop, the frozen Run C checkpoint first
decodes canonical Full and captures internal tensors.  Canonical SAR is then
decoded with exactly one selected stage interpolated toward the paired Full
tensor.  No parameter, optimizer, BatchNorm buffer, Test sample, or raw feature
cache is written.  The first screen should use alpha=1; response curves are a
conditional follow-up for stages that improve pooled SAR Val mIoU.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from itertools import islice
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import build_dataset  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from utils.earthmiss_metrics import EarthMissMetrics  # noqa: E402

from scripts.diagnose_earthmiss_missing_v3 import load_frozen_model  # noqa: E402
from scripts.diagnose_earthmiss_scale_transition import (  # noqa: E402
    EXPECTED_RUN_C_CHECKPOINT_SHA256,
    EXPECTED_RUN_C_EPOCH,
    EXPECTED_RUN_C_SEED,
    EXPECTED_VAL_SELECTION_CLASS_IDS,
    EXPECTED_VAL_TILES,
    assert_buffers_unchanged,
    iter_crop_batches,
    sliding_window_coordinates,
    snapshot_batchnorm_buffers,
    verify_cached_forward_equivalence,
)
from scripts.earthmiss_causal_diagnostics_common import (  # noqa: E402
    NUM_CLASSES,
    ORACLE_SCREEN_STAGES,
    ORACLE_STAGE_ORDER,
    OracleStageIntervention,
    VariantLogitStitcher,
)
from scripts.earthmiss_scale_transition_common import write_json_exclusive  # noqa: E402
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
)


SCHEMA = "earthmiss_missing_causal_oracle_v1"
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/"
    "run-c-e15-val-oracle-screen.json"
)
WINDOW_SIZE = 512
STRIDE = 341


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=ORACLE_STAGE_ORDER,
        default=list(ORACLE_SCREEN_STAGES),
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--smoke-tiles",
        type=int,
        default=0,
        help="Non-formal deterministic Val prefix; zero means all 277 tiles.",
    )
    parser.add_argument("--skip-cached-equivalence-check", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers < 0 or args.smoke_tiles < 0:
        raise ValueError("worker/smoke counts must be non-negative")
    if len(args.stages) != len(set(args.stages)):
        raise ValueError("--stages must be unique")
    if len(args.alphas) != len(set(args.alphas)):
        raise ValueError("--alphas must be unique")
    if any(not 0.0 < value <= 1.0 for value in args.alphas):
        raise ValueError("oracle alphas must be in (0,1]")
    if len(args.stages) > 1 and args.alphas != [1.0]:
        raise ValueError(
            "multi-stage screen is frozen to alpha=1; response curves must "
            "select exactly one stage"
        )


def _variant_name(stage: str, alpha: float) -> str:
    return f"{stage}@alpha={alpha:.2f}"


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _build_loader(args: argparse.Namespace):
    dataset = build_dataset(
        "EarthMiss",
        "val",
        dataset_root=args.dataset_root,
        window_size=(WINDOW_SIZE, WINDOW_SIZE),
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
        cache_size=0,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader


def _metric_summary(
    pooled: dict[str, EarthMissMetrics],
    by_city: dict[str, dict[str, EarthMissMetrics]],
) -> dict[str, Any]:
    return {
        "pooled": {name: metric.compute() for name, metric in pooled.items()},
        "by_city": {
            city: {name: metric.compute() for name, metric in variants.items()}
            for city, variants in sorted(by_city.items())
        },
    }


@torch.inference_mode()
def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite oracle report: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("oracle intervention requires CUDA")
    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    device = torch.device("cuda")
    model, checkpoint = load_frozen_model(
        args.checkpoint,
        "c",
        weights_path,
        device,
    )
    expected_binding = {
        "sha256": EXPECTED_RUN_C_CHECKPOINT_SHA256,
        "epoch": EXPECTED_RUN_C_EPOCH,
        "seed": EXPECTED_RUN_C_SEED,
    }
    observed_binding = {key: checkpoint.get(key) for key in expected_binding}
    if observed_binding != expected_binding:
        raise ValueError(
            f"oracle is bound to preserved Run C E15: expected "
            f"{expected_binding}, got {observed_binding}"
        )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("oracle model must be frozen and eval-mode")

    dataset, loader = _build_loader(args)
    if len(dataset) != EXPECTED_VAL_TILES:
        raise RuntimeError(
            f"EarthMiss Val manifest changed: {len(dataset)} != {EXPECTED_VAL_TILES}"
        )
    processed_tiles = min(args.smoke_tiles, len(dataset)) if args.smoke_tiles else len(dataset)
    variants = (
        "full",
        "sar",
        *(
            _variant_name(stage, alpha)
            for stage in args.stages
            for alpha in args.alphas
        ),
    )
    pooled = {name: EarthMissMetrics() for name in variants}
    by_city: dict[str, dict[str, EarthMissMetrics]] = {}
    buffers_before = snapshot_batchnorm_buffers(model)
    cached_equivalence = None
    crop_windows = 0
    exact_control_stages = {
        "adapter.all",
        "frm.all",
        "se.all",
        "prn.all",
        "head.logits",
    }
    exact_controls = {
        _variant_name(stage, alpha): True
        for stage in args.stages
        for alpha in args.alphas
        if alpha == 1.0 and stage in exact_control_stages
    }
    full_availability = canonical_availability("full", batch_size=1, device=device)
    sar_availability = canonical_availability("sar", batch_size=1, device=device)

    with OracleStageIntervention(model) as intervention:
        progress = tqdm(
            islice(loader, processed_tiles),
            total=processed_tiles,
            desc="oracle-intervention",
        )
        for tile_index, (rgb, sar, target) in enumerate(progress):
            sample = dataset.samples[tile_index]
            city = str(sample.city)
            city_metrics = by_city.setdefault(
                city,
                {name: EarthMissMetrics() for name in variants},
            )
            height, width = target.shape[-2:]
            coordinates = sliding_window_coordinates(
                height,
                width,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
            )
            stitcher = VariantLogitStitcher(height, width, NUM_CLASSES, variants)
            for batch_coordinates, rgb_cpu, sar_cpu, _ in iter_crop_batches(
                rgb,
                sar,
                target,
                coordinates,
                batch_size=1,
            ):
                crop_windows += 1
                rgb_crop = rgb_cpu.to(device, non_blocking=True)
                sar_crop = sar_cpu.to(device, non_blocking=True)
                backbone_outputs = model.extract_frozen_backbone_outputs(
                    rgb_crop,
                    sar_crop,
                )
                if cached_equivalence is None and not args.skip_cached_equivalence_check:
                    cached_equivalence = verify_cached_forward_equivalence(
                        model,
                        rgb_crop,
                        sar_crop,
                        backbone_outputs,
                    )

                def decode(availability):
                    return model.forward_from_backbone_outputs(
                        rgb_crop,
                        sar_crop,
                        backbone_outputs=backbone_outputs,
                        availability=availability,
                    )

                full_logits = intervention.capture_full(
                    lambda: decode(full_availability)
                )
                sar_logits = decode(sar_availability)
                crop_logits: dict[str, torch.Tensor] = {
                    "full": full_logits,
                    "sar": sar_logits,
                }
                for stage in args.stages:
                    for alpha in args.alphas:
                        name = _variant_name(stage, alpha)
                        values = intervention.intervene(
                            stage,
                            alpha,
                            lambda: decode(sar_availability),
                        )
                        crop_logits[name] = values
                        if name in exact_controls:
                            exact_controls[name] &= torch.equal(values, full_logits)
                stitcher.add(batch_coordinates, crop_logits)

            logits = stitcher.finalize()
            target_cpu = target.cpu()
            for name, values in logits.items():
                prediction = values.argmax(dim=1)
                pooled[name].update(prediction, target_cpu)
                city_metrics[name].update(prediction, target_cpu)

    metrics = _metric_summary(pooled, by_city)
    selection_ids = metrics["pooled"]["sar"]["selection_class_ids"]
    formal = (
        args.smoke_tiles == 0
        and tuple(args.stages) == ORACLE_SCREEN_STAGES
        and args.alphas == [1.0]
    )
    if formal and selection_ids != EXPECTED_VAL_SELECTION_CLASS_IDS:
        raise RuntimeError(
            f"Val GT support changed: {selection_ids} != "
            f"{EXPECTED_VAL_SELECTION_CLASS_IDS}"
        )
    sar_miou = metrics["pooled"]["sar"]["mIoU"]
    deltas = {}
    for stage in args.stages:
        for alpha in args.alphas:
            name = _variant_name(stage, alpha)
            deltas[name] = {
                "pooled_sar_mIoU_delta_pp": 100.0
                * (metrics["pooled"][name]["mIoU"] - sar_miou),
                "by_city_sar_mIoU_delta_pp": {
                    city: 100.0
                    * (
                        city_metrics[name]["mIoU"]
                        - city_metrics["sar"]["mIoU"]
                    )
                    for city, city_metrics in metrics["by_city"].items()
                },
                "descriptive_positive_delta": metrics["pooled"][name]["mIoU"] > sar_miou,
            }
            city_values = deltas[name]["by_city_sar_mIoU_delta_pp"]
            deltas[name]["nonnegative_cities"] = sum(
                value >= 0.0 for value in city_values.values()
            )
            deltas[name]["actionable_causal_signal"] = (
                deltas[name]["pooled_sar_mIoU_delta_pp"] >= 0.25
                and deltas[name]["nonnegative_cities"] >= 2
            )

    report = {
        "schema": SCHEMA,
        "formal": formal,
        "training_was_performed": False,
        "split": "Val only; Test not accessed",
        "git_head": _git_head(),
        "checkpoint": checkpoint,
        "protocol": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "crop_batch_size": 1,
            "stages": list(args.stages),
            "alphas": list(args.alphas),
            "intervention": "SAR_stage <- (1-alpha)*SAR_stage + alpha*Full_stage",
            "paired_cached_backbone": True,
            "selection": "none; observation-only diagnostic",
            "screen_rule": "multi-stage screen is alpha=1 only",
            "actionable_screen_rule": (
                "pooled SAR gain >=0.25 pp and at least 2/3 Val cities nonnegative"
            ),
            "response_curve_rule": "only one preselected positive stage per follow-up",
        },
        "processed": {
            "tiles": processed_tiles,
            "expected_val_tiles": len(dataset),
            "crop_windows": crop_windows,
        },
        "cached_forward_equivalence": cached_equivalence,
        "batchnorm_audit": assert_buffers_unchanged(buffers_before, model),
        "exact_full_controls": exact_controls,
        "metrics": metrics,
        "intervention_vs_sar": deltas,
        "interpretation": (
            "A positive alpha=1 effect shows downstream causal sufficiency of "
            "the paired Full tensor at that intervention site. It does not show "
            "that the tensor is predictable from SAR or trainably transferable."
        ),
    }
    write_json_exclusive(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = diagnose(args)
    print(
        {
            "output": args.output,
            "formal": report["formal"],
            "tiles": report["processed"]["tiles"],
            "deltas": report["intervention_vs_sar"],
        }
    )


if __name__ == "__main__":
    main()
