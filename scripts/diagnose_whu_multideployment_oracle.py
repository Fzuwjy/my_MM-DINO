"""Read-only Full-feature oracle audit for all WHU deployment endpoints.

The preserved Run C checkpoint is evaluated as Full, Optical-only (``rgb``),
and SAR-only.  For each missing endpoint, exactly one internal stage is replaced
by the paired Full-state tensor.  The historical checkpoint saw Full/SAR but not
Optical-only during training, so this official-Test audit is descriptive and is
never eligible to select a publishable method.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from models.MMDINO.availability import canonical_availability  # noqa: E402

from scripts.diagnose_earthmiss_scale_transition import (  # noqa: E402
    assert_buffers_unchanged,
    snapshot_batchnorm_buffers,
)
from scripts.earthmiss_causal_diagnostics_common import (  # noqa: E402
    ORACLE_SCREEN_STAGES,
    ORACLE_STAGE_ORDER,
    OracleStageIntervention,
)
from scripts.earthmiss_scale_transition_common import write_json_exclusive  # noqa: E402
from scripts.whu_multideployment_diagnostics_common import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_DATASET_ROOT,
    DEFAULT_RUN_C_CHECKPOINT,
    DEFAULT_WEIGHTS,
    ENDPOINTS,
    MISSING_ENDPOINTS,
    NUM_CLASSES,
    build_whu_scene_dataset,
    deterministic_coordinate_subset,
    fixed_grid_coordinates,
    load_historical_run_c,
    new_metric,
    split_names,
    verify_cached_equivalence_all_endpoints,
)


SCHEMA = "whu_multideployment_causal_oracle_v1"
DEFAULT_SPLIT = str(REPO_ROOT / "splits" / "whu" / "official_test.txt")
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/mm-dino/outputs/whu-multideployment-causal-audit/"
    "historical-run-c-e50-oracle.json"
)
WINDOW_SIZE = 512
SEED = 20260817


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_RUN_C_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-file", default=DEFAULT_SPLIT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stages", nargs="+", choices=ORACLE_STAGE_ORDER,
        default=list(ORACLE_SCREEN_STAGES),
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-crops-per-scene", type=int, default=0,
        help="Zero uses every disjoint 512 crop; positive values are non-formal.",
    )
    parser.add_argument("--smoke-scenes", type=int, default=0)
    parser.add_argument("--skip-cached-equivalence-check", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.num_workers < 0 or args.smoke_scenes < 0 or args.max_crops_per_scene < 0:
        raise ValueError("worker/smoke/crop counts must be non-negative")
    if len(args.stages) != len(set(args.stages)):
        raise ValueError("--stages must be unique")
    if len(args.alphas) != len(set(args.alphas)):
        raise ValueError("--alphas must be unique")
    if any(not 0.0 < alpha <= 1.0 for alpha in args.alphas):
        raise ValueError("oracle alphas must be in (0,1]")
    if len(args.stages) > 1 and args.alphas != [1.0]:
        raise ValueError("multi-stage screen is frozen to alpha=1")


def _variant(endpoint: str, stage: str, alpha: float) -> str:
    return f"{endpoint}|{stage}@alpha={alpha:.2f}"


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _compute_metrics(metrics):
    return {name: value.compute() for name, value in metrics.items()}


@torch.inference_mode()
def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite oracle report: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("WHU oracle audit requires CUDA")
    names = split_names(args.split_file)
    official_names = split_names(DEFAULT_SPLIT)
    if names != official_names:
        raise ValueError(
            "historical Run C audit is bound to official_test.txt; a clean "
            "development-Val checkpoint requires a separate protocol"
        )

    device = torch.device("cuda")
    model, checkpoint = load_historical_run_c(
        args.checkpoint, args.backbone_weights, device, freeze_model=True,
    )
    dataset = build_whu_scene_dataset(args.dataset_root, args.split_file)
    if len(dataset) != 20 or len(dataset.rgb_files) != 20:
        raise RuntimeError("WHU official Test manifest changed")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
    )
    processed_scenes = min(args.smoke_scenes, len(dataset)) if args.smoke_scenes else len(dataset)
    variants = tuple(ENDPOINTS) + tuple(
        _variant(endpoint, stage, alpha)
        for endpoint in MISSING_ENDPOINTS
        for stage in args.stages
        for alpha in args.alphas
    )
    pooled = {name: new_metric() for name in variants}
    by_scene = {}
    buffers_before = snapshot_batchnorm_buffers(model)
    cached_equivalence = None
    crop_count = 0
    covered_pixels = 0
    native_pixels = 0
    exact_stage_names = {"adapter.all", "frm.all", "se.all", "prn.all", "head.logits"}
    exact_controls = {
        _variant(endpoint, stage, alpha): True
        for endpoint in MISSING_ENDPOINTS
        for stage in args.stages
        for alpha in args.alphas
        if stage in exact_stage_names and alpha == 1.0
    }
    availability = {
        endpoint: canonical_availability(endpoint, batch_size=1, device=device)
        for endpoint in ENDPOINTS
    }

    with OracleStageIntervention(model) as intervention:
        progress = tqdm(
            islice(loader, processed_scenes), total=processed_scenes,
            desc="whu-multideployment-oracle",
        )
        for scene_index, (optical, sar, target) in enumerate(progress):
            scene_name = Path(dataset.rgb_files[scene_index]).name
            scene_metrics = {name: new_metric() for name in variants}
            by_scene[scene_name] = scene_metrics
            height, width = target.shape[-2:]
            coordinates = fixed_grid_coordinates(height, width, window_size=WINDOW_SIZE)
            coordinates = deterministic_coordinate_subset(
                coordinates, args.max_crops_per_scene, SEED + scene_index,
            )
            native_pixels += height * width
            covered_pixels += len(coordinates) * WINDOW_SIZE * WINDOW_SIZE
            for y1, y2, x1, x2 in coordinates:
                crop_count += 1
                optical_crop = optical[:, :, y1:y2, x1:x2].to(device, non_blocking=True)
                sar_crop = sar[:, :, y1:y2, x1:x2].to(device, non_blocking=True)
                target_crop = target[:, y1:y2, x1:x2].to("cpu")
                backbone_outputs = model.extract_frozen_backbone_outputs(
                    optical_crop, sar_crop,
                )
                if cached_equivalence is None and not args.skip_cached_equivalence_check:
                    cached_equivalence = verify_cached_equivalence_all_endpoints(
                        model, optical_crop, sar_crop, backbone_outputs,
                    )

                def decode(endpoint: str):
                    return model.forward_from_backbone_outputs(
                        optical_crop, sar_crop, backbone_outputs=backbone_outputs,
                        availability=availability[endpoint],
                    )

                full_logits = intervention.capture_full(lambda: decode("full"))
                logits = {
                    "full": full_logits,
                    "rgb": decode("rgb"),
                    "sar": decode("sar"),
                }
                for endpoint in MISSING_ENDPOINTS:
                    for stage in args.stages:
                        for alpha in args.alphas:
                            name = _variant(endpoint, stage, alpha)
                            value = intervention.intervene(
                                stage, alpha, lambda endpoint=endpoint: decode(endpoint),
                            )
                            logits[name] = value
                            if name in exact_controls:
                                exact_controls[name] &= torch.equal(value, full_logits)
                for name, value in logits.items():
                    prediction = value.argmax(dim=1).to("cpu")
                    pooled[name].update(prediction, target_crop)
                    scene_metrics[name].update(prediction, target_crop)

    metrics = {
        "pooled": _compute_metrics(pooled),
        "by_scene": {
            scene: _compute_metrics(values) for scene, values in sorted(by_scene.items())
        },
    }
    complete = (
        args.smoke_scenes == 0
        and args.max_crops_per_scene == 0
        and tuple(args.stages) == ORACLE_SCREEN_STAGES
        and args.alphas == [1.0]
    )
    if complete and metrics["pooled"]["full"]["gt_present_class_ids"] != list(range(NUM_CLASSES)):
        raise RuntimeError("WHU official Test GT class support changed")
    deltas = {}
    for endpoint in MISSING_ENDPOINTS:
        endpoint_deltas = {}
        baseline = metrics["pooled"][endpoint]["mIoU"]
        for stage in args.stages:
            for alpha in args.alphas:
                name = _variant(endpoint, stage, alpha)
                scene_deltas = {
                    scene: 100.0 * (
                        values[name]["mIoU"] - values[endpoint]["mIoU"]
                    )
                    for scene, values in metrics["by_scene"].items()
                }
                endpoint_deltas[name] = {
                    "pooled_mIoU_delta_pp": 100.0 * (
                        metrics["pooled"][name]["mIoU"] - baseline
                    ),
                    "by_scene_mIoU_delta_pp": scene_deltas,
                    "nonnegative_scene_fraction": sum(
                        value >= 0.0 for value in scene_deltas.values()
                    ) / len(scene_deltas),
                    "descriptive_causal_signal": (
                        100.0 * (metrics["pooled"][name]["mIoU"] - baseline) >= 0.25
                        and sum(value >= 0.0 for value in scene_deltas.values())
                        / len(scene_deltas) >= 0.60
                    ),
                    "actionable_for_method_selection": False,
                }
        deltas[endpoint] = endpoint_deltas

    report = {
        "schema": SCHEMA,
        "formal": complete,
        "training_was_performed": False,
        "method_selection_eligible": False,
        "split": "official 20-scene Test; development-exposed historical diagnostic",
        "git_head": _git_head(),
        "checkpoint": checkpoint,
        "protocol": {
            "deployment_endpoints": list(ENDPOINTS),
            "source_endpoint": "full",
            "target_endpoints": list(MISSING_ENDPOINTS),
            "window_size": WINDOW_SIZE,
            "crop_policy": "disjoint full 512 crops; border remainders excluded",
            "stages": list(args.stages),
            "alphas": list(args.alphas),
            "intervention": "target <- (1-alpha)*target + alpha*paired Full",
            "selection_warning": (
                "Run C never trained Optical-only and official Test cannot be used "
                "to choose an innovation. Results are mechanism description only."
            ),
        },
        "processed": {
            "scenes": processed_scenes,
            "expected_scenes": len(dataset),
            "crops": crop_count,
            "covered_pixel_fraction": covered_pixels / native_pixels,
        },
        "class_names": list(CLASS_NAMES),
        "cached_forward_equivalence": cached_equivalence,
        "batchnorm_audit": assert_buffers_unchanged(buffers_before, model),
        "exact_full_controls": exact_controls,
        "metrics": metrics,
        "intervention_vs_endpoint": deltas,
        "interpretation": (
            "A positive replacement proves downstream sufficiency of paired Full "
            "features at that site. It does not prove missing-endpoint predictability "
            "or trainable transfer, and this Test audit cannot select a method."
        ),
    }
    write_json_exclusive(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = diagnose(args)
    print({
        "output": args.output,
        "formal": report["formal"],
        "method_selection_eligible": report["method_selection_eligible"],
        "scenes": report["processed"]["scenes"],
    })


if __name__ == "__main__":
    main()
