"""Evaluate one frozen EarthMiss P5-to-P4 FAM checkpoint.

The primary comparison uses the SAR-selected checkpoint and evaluates canonical
Full and canonical SAR from that same checkpoint.  In addition to pooled and
per-city segmentation metrics, the report records descriptive displacement and
residual-correction statistics from the single deployed FAM connection.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from models.MMDINO.dino_segment import build_model  # noqa: E402
from models.MMDINO.semantic_flow import (  # noqa: E402
    target_pixel_flow_grid,
)
from scripts.earthmiss_scale_transition_common import (  # noqa: E402
    semantic_boundary_mask,
    write_json_exclusive,
)
from scripts.evaluate_earthmiss_missing_v1 import (  # noqa: E402
    TEST_SELECTION_CLASS_IDS,
    _endpoint_call,
    build_loader,
)
from scripts.train_earthmiss_fam_p5p4_v1 import (  # noqa: E402
    DEFAULT_WEIGHTS,
    FLOW_CHANNELS,
    PROTOCOL_REVISION,
    REFERENCE_SOURCES,
)
from tasks.segmentation.utils.earthmiss_metrics import (  # noqa: E402
    EarthMissMetrics,
)
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    VAL_SELECTION_CLASS_IDS,
)


ENDPOINTS = ("full", "sar-canonical")
EXPECTED_VAL_TILES = 277
REPORT_SCHEMA = "earthmiss_prn_p5p4_fam_evaluation_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output")
    parser.add_argument(
        "--endpoints",
        nargs="+",
        choices=ENDPOINTS,
        default=list(ENDPOINTS),
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--allow-non-primary-checkpoint",
        action="store_true",
        help="Permit an explicitly diagnostic checkpoint instead of best_sar.pth.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.window_size <= 0 or args.window_size % 16:
        raise ValueError("--window-size must be a positive multiple of 16")
    if args.inference_batch_size <= 0 or args.num_workers < 0:
        raise ValueError("invalid evaluation loader configuration")
    if len(set(args.endpoints)) != len(args.endpoints):
        raise ValueError("--endpoints must not contain duplicates")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(checkpoint, *, allow_non_primary=False):
    protocol = checkpoint.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("checkpoint lacks the frozen FAM protocol")
    if protocol.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("checkpoint does not use the P5-to-P4 FAM protocol")
    if protocol.get("experiment") != "prn_p5_to_p4_residual_flow_alignment":
        raise ValueError("checkpoint belongs to a different experiment")
    if checkpoint.get("run") not in {"A", "C"}:
        raise ValueError("checkpoint run must be A or C")
    fam = protocol.get("model", {}).get("fam", {})
    expected_fam = {
        "enabled": True,
        "direction": "coarse_P5_to_fine_P4_only",
        "flow_channels": FLOW_CHANNELS,
        "align_corners": False,
        "padding_mode": "border",
        "learnable_gate": False,
    }
    mismatched = {
        key: (fam.get(key), value)
        for key, value in expected_fam.items()
        if fam.get(key) != value
    }
    if mismatched:
        raise ValueError(f"checkpoint FAM contract changed: {mismatched}")
    evaluation = protocol.get("evaluation", {})
    if evaluation.get("checkpoint_selection_support") != "pooled_gt_present":
        raise ValueError("checkpoint predates the fixed EarthMiss selection metric")
    if not allow_non_primary and (
        checkpoint.get("checkpoint_role") != "primary_deployment"
        or checkpoint.get("selection_state") != "sar"
    ):
        raise ValueError(
            "evaluation requires the SAR-selected primary_deployment checkpoint"
        )


class FlowAccumulator:
    """Accumulate exact flow quantiles and global RMS ratios on CPU."""

    def __init__(self):
        self.calls = 0
        self.vectors = 0
        self.dx_sum = 0.0
        self.dy_sum = 0.0
        self.out_of_grid = 0
        self.correction_square_sum = 0.0
        self.reference_square_sum = 0.0
        self.feature_values = 0
        self._magnitudes = []

    def update(self, high, low, aligned, module):
        output_size = tuple(low.shape[-2:])
        reference = module.zero_flow_reference(high, output_size)
        flow = module.predict_flow(reference, low)
        grid = target_pixel_flow_grid(flow)
        baseline = F.interpolate(high, size=output_size, mode="nearest")
        correction = aligned - baseline

        detached_flow = flow.detach().to(torch.float32)
        magnitude = torch.linalg.vector_norm(detached_flow, dim=1)
        self._magnitudes.append(magnitude.reshape(-1).cpu())
        self.calls += 1
        self.vectors += magnitude.numel()
        self.dx_sum += float(detached_flow[:, 0].sum())
        self.dy_sum += float(detached_flow[:, 1].sum())
        self.out_of_grid += int((grid.detach().abs() > 1).any(dim=-1).sum())
        self.correction_square_sum += float(
            correction.detach().to(torch.float32).square().sum()
        )
        self.reference_square_sum += float(
            reference.detach().to(torch.float32).square().sum()
        )
        self.feature_values += correction.numel()

    def compute(self):
        if not self.calls or not self.vectors or not self.feature_values:
            raise RuntimeError("the FAM diagnostics hook recorded no activations")
        magnitude = torch.cat(self._magnitudes)
        correction_rms = (
            self.correction_square_sum / self.feature_values
        ) ** 0.5
        reference_rms = (
            self.reference_square_sum / self.feature_values
        ) ** 0.5
        return {
            "forward_calls": self.calls,
            "flow_vectors": self.vectors,
            "flow_units": "P4_target_grid_pixels",
            "magnitude": {
                "mean": float(magnitude.mean()),
                "p50": float(torch.quantile(magnitude, 0.50)),
                "p95": float(torch.quantile(magnitude, 0.95)),
                "max": float(magnitude.max()),
            },
            "mean_dx": self.dx_sum / self.vectors,
            "mean_dy": self.dy_sum / self.vectors,
            "grid_out_of_bounds_ratio": self.out_of_grid / self.vectors,
            "correction_rms": correction_rms,
            "zero_flow_reference_rms": reference_rms,
            "correction_to_reference_rms_ratio": (
                correction_rms / max(reference_rms, 1e-12)
            ),
        }


class FlowDiagnostics:
    def __init__(self, module):
        self.module = module
        self.pooled = FlowAccumulator()
        self.by_city = defaultdict(FlowAccumulator)
        self.current_city = None
        self.handle = None

    def __enter__(self):
        def record(_module, inputs, output):
            if self.current_city is None:
                raise RuntimeError("flow hook has no active EarthMiss city")
            high, low = inputs
            self.pooled.update(high, low, output, self.module)
            self.by_city[self.current_city].update(
                high, low, output, self.module
            )

        self.handle = self.module.register_forward_hook(record)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.current_city = None

    def compute(self):
        return {
            "pooled": self.pooled.compute(),
            "by_city": {
                city: accumulator.compute()
                for city, accumulator in sorted(self.by_city.items())
            },
        }


@torch.inference_mode()
def evaluate_endpoint_with_flow(
    model,
    dataset,
    loader,
    endpoint,
    device,
    window_size,
    inference_batch_size,
):
    model.eval()
    pooled_metrics = EarthMissMetrics()
    city_metrics = defaultdict(EarthMissMetrics)
    pooled_regions = {
        "boundary": EarthMissMetrics(),
        "interior": EarthMissMetrics(),
    }
    city_regions = defaultdict(
        lambda: {
            "boundary": EarthMissMetrics(),
            "interior": EarthMissMetrics(),
        }
    )
    diagnostics = FlowDiagnostics(model.decoder.neck.p5_p4_fam)
    tile_seconds = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with diagnostics:
        for index, (rgb, sar, label) in enumerate(
            tqdm(loader, desc=endpoint, leave=False)
        ):
            sample = dataset.samples[index]
            diagnostics.current_city = sample.city
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            logits = _endpoint_call(
                endpoint,
                rgb.to(device, non_blocking=True),
                sar.to(device, non_blocking=True),
                model,
                window_size,
                inference_batch_size,
                device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tile_seconds.append(time.perf_counter() - started)
            prediction = logits.argmax(dim=1)
            pooled_metrics.update(prediction, label)
            city_metrics[sample.city].update(prediction, label)
            boundary = semantic_boundary_mask(label)
            valid = (label >= 0) & (label < 8)
            region_masks = {
                "boundary": boundary,
                "interior": valid & ~boundary,
            }
            for region, mask in region_masks.items():
                region_target = torch.where(mask, label, torch.full_like(label, 8))
                pooled_regions[region].update(prediction, region_target)
                city_regions[sample.city][region].update(
                    prediction, region_target
                )

    def compute_regions(regions):
        return {
            region: evaluator.compute()
            for region, evaluator in regions.items()
        }

    timing = torch.tensor(tile_seconds, dtype=torch.float64)
    return {
        "metrics": pooled_metrics.compute(),
        "by_city": {
            city: evaluator.compute()
            for city, evaluator in sorted(city_metrics.items())
        },
        "regions": {
            "definition": (
                "GT four-neighbour class transitions mark both adjacent pixels; "
                "interior is valid GT excluding that boundary"
            ),
            "pooled": compute_regions(pooled_regions),
            "by_city": {
                city: compute_regions(regions)
                for city, regions in sorted(city_regions.items())
            },
        },
        "flow": diagnostics.compute(),
        "instrumented_runtime": {
            "warning": (
                "includes flow-statistic hooks, CPU stitching, and data transfer; "
                "not the deployment latency gate"
            ),
            "tiles": len(tile_seconds),
            "total_seconds": float(timing.sum()),
            "mean_seconds_per_tile": float(timing.mean()),
            "p50_seconds_per_tile": float(torch.quantile(timing, 0.50)),
            "p95_seconds_per_tile": float(torch.quantile(timing, 0.95)),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
    }


def build_fam_model(checkpoint, weights_path, device):
    protocol = checkpoint["protocol"]
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
        prn_p5_p4_fam_seed=protocol["fam_seed"],
        prn_p5_p4_fam_flow_channels=FLOW_CHANNELS,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _default_output_path(checkpoint_path, split):
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}.earthmiss-{split}-fam-evaluation.json"
    )


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    output_path = (
        Path(args.output)
        if args.output
        else _default_output_path(Path(args.checkpoint), args.split)
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("EarthMiss FAM evaluation requires a CUDA GPU")

    checkpoint_path = Path(args.checkpoint)
    weights_path = Path(args.backbone_weights)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_checkpoint(
        checkpoint,
        allow_non_primary=args.allow_non_primary_checkpoint,
    )
    dataset, loader = build_loader(args)
    if args.split == "val" and len(dataset) != EXPECTED_VAL_TILES:
        raise RuntimeError(
            f"EarthMiss Val manifest changed: {len(dataset)} != {EXPECTED_VAL_TILES}"
        )

    device = torch.device("cuda")
    model = build_fam_model(checkpoint, weights_path, device)
    fam = model.decoder.neck.p5_p4_fam
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    fam_parameters = sum(parameter.numel() for parameter in fam.parameters())
    target_height = args.window_size // 16
    target_width = args.window_size // 16
    fam_conv_macs = target_height * target_width * (
        2 * 256 * FLOW_CHANNELS
        + 2 * (2 * FLOW_CHANNELS) * 3 * 3
    )
    result = {
        "schema": REPORT_SCHEMA,
        "split": args.split,
        "tiles": len(dataset),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "run": checkpoint.get("run"),
            "seed": checkpoint.get("seed"),
            "fam_seed": checkpoint["protocol"].get("fam_seed"),
            "epoch": checkpoint.get("epoch"),
            "role": checkpoint.get("checkpoint_role"),
            "selection_state": checkpoint.get("selection_state"),
            "selection_score": checkpoint.get("selection_score"),
        },
        "protocol": {
            "revision": PROTOCOL_REVISION,
            "same_checkpoint_for_all_endpoints": True,
            "endpoints": {
                "full": "canonical RGB+SAR",
                "sar-canonical": (
                    "skip RGB backbone, renormalize over SAR, keep two slots"
                ),
            },
            "inference": {
                "window_size": args.window_size,
                "stride": int(args.window_size * 2 / 3),
                "batch_size": args.inference_batch_size,
                "overlap_rule": "uniform logit mean",
            },
            "flow_statistics": {
                "scope": "overlapping crop forwards, not stitched whole-tile flow",
                "quantiles": "exact over crop-grid displacement magnitudes",
                "out_of_bounds": "normalized sampling coordinate outside [-1,1]",
                "causal_status": "descriptive; segmentation metrics decide the arm",
            },
            "reference_sources": REFERENCE_SOURCES,
        },
        "model_complexity": {
            "total_parameters": total_parameters,
            "fam_parameters": fam_parameters,
            "base_parameters_by_subtraction": total_parameters - fam_parameters,
            "fam_parameter_fraction": fam_parameters / total_parameters,
            "fam_conv_macs_per_window": fam_conv_macs,
            "window_size": args.window_size,
            "mac_scope": (
                "two 1x1 projections plus the 3x3 flow predictor; excludes "
                "grid construction and bilinear sampling"
            ),
            "deployment_latency_gate": (
                "deferred until the accuracy screen passes; requires a paired "
                "Run-C versus C+FAM benchmark without diagnostic hooks"
            ),
        },
        "endpoints": {},
    }

    expected_support = (
        VAL_SELECTION_CLASS_IDS
        if args.split == "val"
        else TEST_SELECTION_CLASS_IDS
    )
    for endpoint in args.endpoints:
        endpoint_result = evaluate_endpoint_with_flow(
            model,
            dataset,
            loader,
            endpoint,
            device,
            args.window_size,
            args.inference_batch_size,
        )
        support = endpoint_result["metrics"]["selection_class_ids"]
        if support != expected_support:
            raise RuntimeError(
                f"EarthMiss {args.split} support changed: "
                f"{support} != {expected_support}"
            )
        result["endpoints"][endpoint] = endpoint_result
        print(
            f"{endpoint}: mIoU={endpoint_result['metrics']['mIoU']:.6f}, "
            f"flow_mean={endpoint_result['flow']['pooled']['magnitude']['mean']:.6f}"
        )

    write_json_exclusive(output_path, result)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
