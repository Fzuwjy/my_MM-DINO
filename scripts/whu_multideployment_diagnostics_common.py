"""Shared contracts for WHU multi-deployment causal diagnostics.

The historical WHU Run C checkpoint was trained with Full/SAR batches and is
therefore only a descriptive anchor for the unseen Optical-only endpoint.  The
helpers in this file make that limitation explicit while keeping the model,
checkpoint, input bands, and official scene manifests strictly auditable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import build_dataset  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from models.MMDINO.dino_segment import build_model  # noqa: E402
from utils.pooled_segmentation_metrics import PooledSegmentationMetrics  # noqa: E402


DEFAULT_DATASET_ROOT = "/root/autodl-tmp/mm-dino/datasets/whu-opt-sar"
DEFAULT_WEIGHTS = (
    "/root/autodl-tmp/mm-dino/weights/"
    "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
DEFAULT_RUN_C_CHECKPOINT = (
    "/root/autodl-tmp/mm-dino/outputs/whu-missing-baseline-nirrg/"
    "run_c_seed42/epoch_50.pth"
)
EXPECTED_RUN_C_SHA256 = (
    "e82913830972d11d0c856d972ba49edd06830728688623034ce993cd36717824"
)
EXPECTED_RUN_C_EPOCH = 50
EXPECTED_RUN_C_SEED = 42
EXPECTED_RUN_C_PROTOCOL = "whu_missing_baseline_nirrg_v1"
EXPECTED_RUN_C_ROLE = "fixed_final_primary"

CLASS_NAMES = (
    "Farmland",
    "City",
    "Village",
    "Water",
    "Forest",
    "Road",
    "Others",
)
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = NUM_CLASSES
ENDPOINTS = ("full", "rgb", "sar")
MISSING_ENDPOINTS = ("rgb", "sar")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_binding(checkpoint: dict[str, Any], path: str | Path) -> dict[str, Any]:
    protocol = checkpoint.get("protocol")
    revision = protocol.get("protocol_revision") if isinstance(protocol, dict) else None
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "run": str(checkpoint.get("run", "")).upper(),
        "epoch": int(checkpoint.get("epoch", -1)),
        "seed": int(checkpoint.get("seed", -1)),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
        "protocol_revision": revision,
    }


def load_historical_run_c(
    checkpoint_path: str | Path,
    backbone_weights: str | Path,
    device: torch.device,
    *,
    freeze_model: bool,
):
    checkpoint_path = Path(checkpoint_path)
    backbone_weights = Path(backbone_weights)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"WHU checkpoint not found: {checkpoint_path}")
    if not backbone_weights.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {backbone_weights}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    binding = checkpoint_binding(checkpoint, checkpoint_path)
    expected = {
        "sha256": EXPECTED_RUN_C_SHA256,
        "run": "C",
        "epoch": EXPECTED_RUN_C_EPOCH,
        "seed": EXPECTED_RUN_C_SEED,
        "checkpoint_role": EXPECTED_RUN_C_ROLE,
        "protocol_revision": EXPECTED_RUN_C_PROTOCOL,
    }
    observed = {key: binding[key] for key in expected}
    if observed != expected:
        raise ValueError(f"WHU historical Run C binding mismatch: {observed} != {expected}")

    model = build_model(
        model_name="DINOv3",
        backbone_weights=str(backbone_weights),
        backbone_type="dinov3_vits16",
        freeze_backbone=True,
        n_classes=NUM_CLASSES,
        use_lora=False,
        r=3,
        num_modalities=2,
        raw_logits=True,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if freeze_model:
        model.requires_grad_(False)
        model.eval()
    return model, binding


def build_whu_scene_dataset(
    dataset_root: str | Path,
    split_file: str | Path,
    *,
    window_size: int = 512,
    cache_size: int = 0,
):
    return build_dataset(
        "WHU",
        "test",
        dataset_root=str(dataset_root),
        split_file=str(split_file),
        window_size=(window_size, window_size),
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
        optical_bands="nir-r-g",
        cache_size=cache_size,
    )


def split_names(path: str | Path) -> tuple[str, ...]:
    names = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not names or len(names) != len(set(names)):
        raise ValueError(f"invalid or duplicate WHU split entries: {path}")
    return names


def new_metric() -> PooledSegmentationMetrics:
    return PooledSegmentationMetrics(NUM_CLASSES, IGNORE_INDEX)


@torch.inference_mode()
def verify_cached_equivalence_all_endpoints(
    model: torch.nn.Module,
    optical: torch.Tensor,
    sar: torch.Tensor,
    backbone_outputs: tuple,
) -> dict[str, Any]:
    result = {}
    for endpoint in ENDPOINTS:
        availability = canonical_availability(
            endpoint,
            batch_size=optical.shape[0],
            device=optical.device,
        )
        cached = model.forward_from_backbone_outputs(
            optical,
            sar,
            backbone_outputs=backbone_outputs,
            availability=availability,
        )
        direct = model(optical, sar, availability=availability)
        maximum = float((cached - direct).abs().max())
        if not torch.equal(cached, direct):
            raise RuntimeError(
                f"cached {endpoint} forward differs from direct; max={maximum}"
            )
        result[endpoint] = {"exact_equal": True, "max_abs_delta": maximum}
    return result


def fixed_grid_coordinates(
    height: int,
    width: int,
    *,
    window_size: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return disjoint full-size crops; border remainders are deliberately excluded."""

    if height < window_size or width < window_size:
        raise ValueError("WHU scene is smaller than the diagnostic crop")
    return tuple(
        (y, y + window_size, x, x + window_size)
        for y in range(0, height - window_size + 1, window_size)
        for x in range(0, width - window_size + 1, window_size)
    )


def deterministic_coordinate_subset(
    coordinates: Sequence[tuple[int, int, int, int]],
    maximum: int,
    seed: int,
) -> tuple[tuple[int, int, int, int], ...]:
    if maximum <= 0 or maximum >= len(coordinates):
        return tuple(coordinates)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randperm(len(coordinates), generator=generator)[:maximum]
    return tuple(coordinates[int(index)] for index in indices.sort().values)


@dataclass(frozen=True)
class EndpointRecoverabilityExamples:
    features: torch.Tensor
    targets: torch.Tensor
    class_ids: torch.Tensor
    pure_cells: int
    endpoint_wrong_cells: int


def exact_whu_class_occupancy(
    target: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 3:
        raise ValueError("target must have shape BHW or B1HW")
    height, width = target.shape[-2:]
    out_h, out_w = map(int, output_size)
    if height % out_h or width % out_w:
        raise ValueError("target must be exactly divisible by feature grid")
    invalid = (target < 0) | (target > IGNORE_INDEX)
    if invalid.any():
        raise ValueError("target contains invalid WHU labels")
    valid = (target >= 0) & (target < NUM_CLASSES)
    safe = target.clamp(0, NUM_CLASSES - 1)
    one_hot = F.one_hot(safe.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()
    one_hot *= valid.unsqueeze(1)
    return F.avg_pool2d(
        one_hot,
        kernel_size=(height // out_h, width // out_w),
        stride=(height // out_h, width // out_w),
    )


def endpoint_recoverability_examples(
    endpoint_feature: torch.Tensor,
    full_logits: torch.Tensor,
    endpoint_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    purity_threshold: float = 0.75,
) -> EndpointRecoverabilityExamples:
    if endpoint_feature.ndim != 4:
        raise ValueError("endpoint feature must be BCHW")
    if full_logits.shape != endpoint_logits.shape or full_logits.ndim != 4:
        raise ValueError("Full and endpoint logits must have equal BCHW shape")
    if not 0.5 < purity_threshold <= 1.0:
        raise ValueError("purity threshold must be in (0.5,1]")
    output_size = tuple(endpoint_feature.shape[-2:])
    occupancy = exact_whu_class_occupancy(target, output_size)
    purity, class_ids = occupancy.max(dim=1)
    pure = purity >= purity_threshold
    pooled_full = F.adaptive_avg_pool2d(full_logits.float(), output_size).argmax(1)
    pooled_endpoint = F.adaptive_avg_pool2d(
        endpoint_logits.float(), output_size
    ).argmax(1)
    endpoint_wrong = pure & (pooled_endpoint != class_ids)
    recoverable = pooled_full == class_ids
    features = endpoint_feature.permute(0, 2, 3, 1)[endpoint_wrong].float()
    return EndpointRecoverabilityExamples(
        features=features,
        targets=recoverable[endpoint_wrong].to(torch.int64),
        class_ids=class_ids[endpoint_wrong].to(torch.int64),
        pure_cells=int(pure.sum()),
        endpoint_wrong_cells=int(endpoint_wrong.sum()),
    )
