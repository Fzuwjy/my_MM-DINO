"""Reduced ViT-L LoRA confirmation of the WHU patch-phase signal.

Only the strongest ViT-S sub-patch condition (8px) and the phase-preserving
16px control are evaluated.  The normal prediction is also recomputed so a
full-test run must reproduce the trusted 55.5821% ViT-L LoRA checkpoint before
the translation comparison is accepted.  No training or checkpoint writes
occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from configs import get_cfg  # noqa: E402
from datasets import build_dataset  # noqa: E402
from scripts.diagnose_whu_spatial_errors import file_sha256  # noqa: E402
from scripts.evaluate_whu_translation_consistency import (  # noqa: E402
    REPORT_REGIONS,
    aggregate_offset_records,
    paired_condition_summary,
    phase_comparisons,
    translate_tensor,
)
from scripts.spatial_diagnostics_common import (  # noqa: E402
    build_spatial_region_masks,
    common_translation_slices,
    confusion_from_arrays,
    mean_iou_from_confusion,
)
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.inference import slide_inference  # noqa: E402
from utils.utils import set_seed  # noqa: E402


BACKBONE_TYPE = "dinov3_vitl16"
NUM_CLASSES = 7


def load_evaluation_reference(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "model_name": "DINOv3",
        "dataset_name": "WHU",
        "num_modalities": 2,
        "backbone_type": BACKBONE_TYPE,
        "use_lora": True,
        "lora_rank": 3,
        "evaluation_inference_batch_size": 32,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"ViT-L evaluation reference changed: {mismatches}")
    if not isinstance(payload.get("MIoU"), (int, float)):
        raise ValueError("ViT-L evaluation reference has no numeric MIoU")
    return payload


def load_vitl_model(checkpoint: Path, seed: int):
    set_seed(seed)
    cfg = get_cfg(
        "DINOv3",
        "WHU",
        num_modalities=2,
        use_lora=True,
        r=3,
        backbone_type=BACKBONE_TYPE,
        use_naf=False,
    )
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise TypeError("ViT-L checkpoint must contain a 'model' state dict")
    cfg["model"].load_state_dict(payload["model"], strict=True)
    cfg["optimizer"] = None
    cfg["scheduler"] = None
    return cfg["model"], cfg


def build_vitl_loader(max_images: int | None, window_size: tuple[int, int]):
    base_dataset = build_dataset(
        "WHU",
        "test",
        window_size=window_size,
        model_name="DINOv3",
        modality="multi",
        backbone_type=BACKBONE_TYPE,
    )
    full_length = len(base_dataset)
    selected_length = full_length if max_images is None else max_images
    if selected_length > full_length:
        raise ValueError(
            f"--max-images exceeds test length: {selected_length} > {full_length}"
        )
    indices = list(range(selected_length))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(base_dataset, indices),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    names = [Path(base_dataset.rgb_files[index]).stem for index in indices]
    return loader, names, full_length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduced WHU ViT-L LoRA 8px-versus-16px translation confirmation"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-evaluation-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--subpatch-offset", type=int, default=8)
    parser.add_argument("--control-offset", type=int, default=16)
    parser.add_argument("--axis", choices=("x", "y"), default="x")
    parser.add_argument("--valid-margin", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--miou-tolerance", type=float, default=1e-12)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    args = parser.parse_args()

    for path in (args.checkpoint, args.expected_evaluation_json):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.subpatch_offset <= 0 or args.control_offset <= 0:
        parser.error("translation offsets must be positive")
    if args.subpatch_offset >= args.control_offset:
        parser.error("sub-patch offset must be smaller than control offset")
    if args.valid_margin < 0:
        parser.error("--valid-margin cannot be negative")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.miou_tolerance < 0:
        parser.error("--miou-tolerance cannot be negative")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ViT-L translation confirmation")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    reference = load_evaluation_reference(args.expected_evaluation_json)
    checkpoint_sha = file_sha256(args.checkpoint)
    reference_checkpoint = Path(reference["checkpoint"])
    if reference_checkpoint.resolve() != args.checkpoint.resolve():
        raise AssertionError(
            "checkpoint path differs from trusted ViT-L evaluation reference"
        )

    model, cfg = load_vitl_model(args.checkpoint, args.seed)
    if args.inference_batch_size != reference["evaluation_inference_batch_size"]:
        raise ValueError(
            "ViT-L confirmation must retain the trusted crop microbatch 32"
        )
    if args.valid_margin < max(cfg["window_size"]):
        raise ValueError("valid margin must be at least the 512px crop size")
    loader, sample_names, full_test_length = build_vitl_loader(
        args.max_images, cfg["window_size"]
    )

    offsets = (args.subpatch_offset, args.control_offset)
    shifts = {
        offset: ((0, offset) if args.axis == "x" else (offset, 0))
        for offset in offsets
    }
    records_by_offset: dict[int, list[dict[str, Any]]] = {
        offset: [] for offset in offsets
    }
    shifted_digests = {offset: hashlib.sha256() for offset in offsets}
    normal_prediction_digest = hashlib.sha256()
    label_digest = hashlib.sha256()
    full_baseline_confusions: list[np.ndarray] = []
    per_image_records: list[dict[str, Any]] = []

    device = torch.device(args.device)
    model.to(device)
    model.eval()
    stride = tuple(int(size * 2 / 3) for size in cfg["window_size"])
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"checkpoint_sha256={checkpoint_sha}")
    print(f"translation_axis={args.axis}")
    print(f"translation_offsets={offsets}")
    print(f"valid_margin={args.valid_margin}")
    print(f"evaluated_images={len(loader.dataset)}/{full_test_length}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for image_index, ((optical, sar, label_tensor), sample_name) in enumerate(
            zip(loader, sample_names, strict=True)
        ):
            label_full = np.ascontiguousarray(
                label_tensor.numpy().astype(np.int64, copy=False)
            )
            label_digest.update(label_full.tobytes())
            normal_scores = slide_inference(
                optical.to(device),
                model,
                dsm=sar.to(device),
                n_output_channels=NUM_CLASSES,
                crop_size=cfg["window_size"],
                stride=stride,
                batch_size=args.inference_batch_size,
            )
            normal_prediction_full = np.ascontiguousarray(
                normal_scores.argmax(dim=1).numpy().astype(np.int64, copy=False)
            )
            normal_prediction_digest.update(normal_prediction_full.tobytes())
            full_baseline_confusions.append(
                confusion_from_arrays(
                    normal_prediction_full[0], label_full[0], NUM_CLASSES
                )
            )

            original_slice, shifted_slices = common_translation_slices(
                label_full[0].shape, tuple(shifts.values()), args.valid_margin
            )
            target = label_full[0][original_slice]
            baseline = normal_prediction_full[0][original_slice]
            valid = (target >= 0) & (target < NUM_CLASSES)
            baseline_confusion = confusion_from_arrays(
                baseline, target, NUM_CLASSES
            )
            region_masks, _ = build_spatial_region_masks(
                label_full[0],
                NUM_CLASSES,
                boundary_radii=(0, 1, 2, 4, 8),
                component_area_thresholds=(256, 1024, 4096),
                component_thickness_thresholds=(4, 8, 16),
                patch_size=16,
                union_boundary_radius=0,
                union_component_area=256,
                union_component_thickness=4,
            )
            image_record = {
                "image_index": image_index,
                "sample_name": sample_name,
                "conditions": {},
            }

            for offset in offsets:
                dy, dx = shifts[offset]
                translated_optical = translate_tensor(optical, dy, dx).to(device)
                translated_sar = translate_tensor(sar, dy, dx).to(device)
                shifted_scores = slide_inference(
                    translated_optical,
                    model,
                    dsm=translated_sar,
                    n_output_channels=NUM_CLASSES,
                    crop_size=cfg["window_size"],
                    stride=stride,
                    batch_size=args.inference_batch_size,
                )
                shifted_full = np.ascontiguousarray(
                    shifted_scores.argmax(dim=1).numpy().astype(np.int64, copy=False)
                )
                shifted_digests[offset].update(shifted_full.tobytes())
                shifted = shifted_full[0][shifted_slices[(dy, dx)]]
                shifted_confusion = confusion_from_arrays(
                    shifted, target, NUM_CLASSES
                )
                disagreement = valid & (shifted != baseline)
                regions = {}
                for region_name in REPORT_REGIONS:
                    region_mask = region_masks[region_name][original_slice] & valid
                    regions[region_name] = {
                        "valid_pixels": int(valid.sum()),
                        "pixels": int(region_mask.sum()),
                        "disagreement_pixels": int(
                            np.count_nonzero(disagreement & region_mask)
                        ),
                        "baseline_errors": int(
                            np.count_nonzero(region_mask & (baseline != target))
                        ),
                        "shifted_errors": int(
                            np.count_nonzero(region_mask & (shifted != target))
                        ),
                    }
                record = {
                    "image_index": image_index,
                    "sample_name": sample_name,
                    "offset": offset,
                    "shift": [dy, dx],
                    "valid_shape": list(target.shape),
                    "valid_pixels": int(valid.sum()),
                    "disagreement_pixels": int(disagreement.sum()),
                    "regions": regions,
                    "_baseline_confusion": baseline_confusion,
                    "_shifted_confusion": shifted_confusion,
                }
                records_by_offset[offset].append(record)
                condition = paired_condition_summary(
                    baseline_confusion,
                    shifted_confusion,
                    record["disagreement_pixels"],
                    record["valid_pixels"],
                    cfg["labels"],
                )
                image_record["conditions"][str(offset)] = {
                    key: value
                    for key, value in record.items()
                    if not key.startswith("_")
                }
                image_record["conditions"][str(offset)]["summary"] = condition
                print(
                    f"image={image_index + 1}/{len(loader.dataset)} "
                    f"name={sample_name} offset={offset}px "
                    f"disagreement={condition['prediction_disagreement_rate'] * 100:.4f}% "
                    f"miou_delta={condition['shifted_minus_baseline_miou_pp']:+.4f}pp",
                    flush=True,
                )
                del (
                    translated_optical,
                    translated_sar,
                    shifted_scores,
                    shifted_full,
                    shifted,
                    shifted_confusion,
                    disagreement,
                )
            per_image_records.append(image_record)
            del (
                normal_scores,
                normal_prediction_full,
                label_full,
                target,
                baseline,
                valid,
                baseline_confusion,
                region_masks,
                optical,
                sar,
                label_tensor,
            )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    aggregate_full_baseline = np.stack(full_baseline_confusions).sum(axis=0)
    full_baseline_miou = mean_iou_from_confusion(aggregate_full_baseline)
    full_reference_validation = {
        "checked": args.max_images is None,
        "expected_miou": float(reference["MIoU"]),
        "actual_miou": full_baseline_miou,
        "difference": abs(full_baseline_miou - float(reference["MIoU"])),
    }
    if args.max_images is None:
        full_reference_validation["miou_within_tolerance"] = (
            full_reference_validation["difference"] <= args.miou_tolerance
        )
        if not full_reference_validation["miou_within_tolerance"]:
            raise AssertionError(
                f"ViT-L normal baseline failed reproduction: {full_reference_validation}"
            )

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_indices = rng.integers(
        0,
        len(loader.dataset),
        size=(args.bootstrap_replicates, len(loader.dataset)),
        endpoint=False,
    )
    aggregate = {
        offset: aggregate_offset_records(
            records_by_offset[offset], cfg["labels"], bootstrap_indices
        )
        for offset in offsets
    }
    comparisons = phase_comparisons(
        aggregate, args.control_offset, bootstrap_indices
    )
    serializable_aggregate = {}
    for offset, condition in aggregate.items():
        serializable_aggregate[str(offset)] = {
            key: value
            for key, value in condition.items()
            if not key.startswith("_")
        }
        serializable_aggregate[str(offset)]["shifted_prediction_sha256"] = (
            shifted_digests[offset].hexdigest()
        )

    output = {
        "status": "PASS",
        "scope": "full-test" if args.max_images is None else "subset-smoke",
        "scientific_scope": (
            "Reduced cross-capacity confirmation of the ViT-S patch-phase signal; "
            "ViT-S and ViT-L differ in both capacity and LoRA adaptation, so this "
            "is a generality check rather than an isolated capacity experiment."
        ),
        "full_test_length": full_test_length,
        "evaluated_images": len(loader.dataset),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "expected_evaluation_json": str(args.expected_evaluation_json.resolve()),
        "normal_prediction_sha256": normal_prediction_digest.hexdigest(),
        "label_sha256": label_digest.hexdigest(),
        "normal_full_confusion": aggregate_full_baseline.tolist(),
        "normal_full_reference_validation": full_reference_validation,
        "model": {
            "backbone_type": BACKBONE_TYPE,
            "use_lora": True,
            "lora_rank": 3,
        },
        "translation": {
            "axis": args.axis,
            "subpatch_offset": args.subpatch_offset,
            "control_offset": args.control_offset,
            "valid_margin": args.valid_margin,
            "valid_region": (
                "One shared intersection; original and shifted coordinates stay "
                "at least one 512px crop from full-image borders."
            ),
        },
        "inference": {
            "window_size": list(cfg["window_size"]),
            "stride": list(stride),
            "crop_batch_size": args.inference_batch_size,
            "device": str(device),
        },
        "bootstrap": {
            "unit": "test image",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
        "aggregate": serializable_aggregate,
        "phase_comparisons_to_control": comparisons,
        "images": per_image_records,
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "scope": output["scope"],
                "evaluated_images": output["evaluated_images"],
                "normal_full_reference_validation": full_reference_validation,
                "phase_comparisons_to_control": comparisons,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"vitl_translation_result={args.output_path.resolve()}")
    print("vitl_translation_status=PASS")


if __name__ == "__main__":
    main()
