"""Prepare full-image WHU small/thin structure masks for phase distillation.

Components are computed class-wise on complete label maps with 8-connectivity.
The persisted uint8 value uses bit 0 for component area <= 256 pixels and bit 1
for thickness proxy ``2 * area / perimeter_pixel_count <= 4``.  Class index 7
is the released WHU ignore value and never receives either bit.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from skimage.io import imread


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scripts.cache_whu_phase_teacher import (  # noqa: E402
    NUM_CLASSES,
    array_sha256,
    atomic_save_npy,
    atomic_write_json,
    build_full_image_dataset,
    canonical_json_sha256,
    dataset_selection,
    file_sha256,
    safe_artifact_key,
    selected_indices,
    utc_now,
)
from scripts.spatial_diagnostics_common import component_geometry_masks  # noqa: E402
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "whu_full_image_phase_structure_mask"
SMALL_BIT = np.uint8(1)
THIN_BIT = np.uint8(2)
AREA_THRESHOLD = 256
THICKNESS_THRESHOLD = 4


def build_mask_protocol() -> dict[str, Any]:
    return {
        "source": "complete WHU semantic label image before training crops",
        "valid_class_indices": list(range(NUM_CLASSES)),
        "ignore_class_indices": [NUM_CLASSES],
        "component_semantics": "class-wise semantic regions, not object instances",
        "connectivity": 8,
        "perimeter": (
            "component pixels removed by 8-neighbour binary erosion with "
            "border_value=0"
        ),
        "small_definition": f"component area <= {AREA_THRESHOLD} pixels",
        "thickness_proxy": "2 * component_area / perimeter_pixel_count",
        "thin_definition": (
            f"2 * component_area / perimeter_pixel_count <= {THICKNESS_THRESHOLD}"
        ),
        "encoding": {
            "dtype": "uint8",
            "bit_0_value_1": "small component",
            "bit_1_value_2": "thin component",
            "value_3": "both small and thin",
        },
    }


def load_whu_label(path: Path) -> np.ndarray:
    """Decode one label with the exact released WHU index conversion."""

    label = imread(path).astype(np.int32)
    if label.ndim != 2:
        raise ValueError(f"WHU label must be 2-D, got {label.shape}: {path}")
    label = label / 10 - 1
    label[label == -1] = NUM_CLASSES
    return np.ascontiguousarray(label.astype(np.int64))


def encode_structure_mask(
    label: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return uint8 bit mask plus component-algorithm metadata and counts."""

    label = np.asarray(label)
    if label.ndim != 2 or not np.issubdtype(label.dtype, np.integer):
        raise ValueError("label must be a 2-D integer array")
    masks, component_metadata = component_geometry_masks(
        label,
        NUM_CLASSES,
        area_thresholds=(AREA_THRESHOLD,),
        thickness_thresholds=(THICKNESS_THRESHOLD,),
    )
    small = masks[f"component_area_le_{AREA_THRESHOLD}px2"]
    thin = masks[f"component_thickness_le_{THICKNESS_THRESHOLD}px"]
    encoded = np.zeros(label.shape, dtype=np.uint8)
    encoded[small] |= SMALL_BIT
    encoded[thin] |= THIN_BIT
    valid = (label >= 0) & (label < NUM_CLASSES)
    if np.any(encoded[~valid] != 0):
        raise AssertionError("structure bits leaked onto ignored label pixels")
    stats = {
        "valid_pixels": int(valid.sum()),
        "small_pixels": int(small.sum()),
        "thin_pixels": int(thin.sum()),
        "small_and_thin_pixels": int((small & thin).sum()),
        "small_or_thin_pixels": int((small | thin).sum()),
        "component_algorithm": component_metadata,
    }
    return np.ascontiguousarray(encoded), stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full-image WHU small/thin bit masks"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.max_images is not None:
        parser.error("--smoke and --max-images are mutually exclusive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if args.output_dir.exists():
        parser.error(f"refusing to overwrite output directory: {args.output_dir}")
    return args


def mask_record(
    staging_dir: Path,
    sample: Mapping[str, Any],
    label: np.ndarray,
    encoded: np.ndarray,
    stats: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    key = safe_artifact_key(sample["index"], sample["sample_name"])
    relative_mask = Path("masks") / f"{key}.npy"
    relative_metadata = Path("metadata") / f"{key}.json"
    artifact = atomic_save_npy(staging_dir / relative_mask, encoded)
    record = {
        "index": sample["index"],
        "sample_name": sample["sample_name"],
        "source_label_file": sample["label_file"],
        "source_label_file_sha256": file_sha256(Path(sample["label_file"])),
        "label_sha256": array_sha256(label.astype(np.int64, copy=False)),
        "full_shape_hw": [int(value) for value in label.shape],
        "mask": {
            "path": relative_mask.as_posix(),
            "metadata_path": relative_metadata.as_posix(),
            **artifact,
        },
        "counts": dict(stats),
    }
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "protocol": dict(protocol),
        "protocol_sha256": canonical_json_sha256(protocol),
        "record": record,
    }
    atomic_write_json(staging_dir / relative_metadata, sidecar)
    return record


def main() -> None:
    args = parse_args()
    install_whu_cache_compat(CACHE_CAPACITY)
    dataset = build_full_image_dataset(args.split)
    indices = selected_indices(len(dataset), args.max_images, args.smoke)
    selection = dataset_selection(dataset, indices)
    protocol = build_mask_protocol()

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = args.output_dir.with_name(
        f".{args.output_dir.name}.partial-{uuid.uuid4().hex}"
    )
    staging_dir.mkdir(exist_ok=False)
    try:
        records = []
        for sample in selection:
            label = load_whu_label(Path(sample["label_file"]))
            encoded, stats = encode_structure_mask(label)
            record = mask_record(
                staging_dir, sample, label, encoded, stats, protocol
            )
            records.append(record)
            print(
                f"prepared={len(records)}/{len(selection)} "
                f"name={sample['sample_name']} small={stats['small_pixels']} "
                f"thin={stats['thin_pixels']}",
                flush=True,
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "status": "PASS",
            "scope": (
                "subset-smoke"
                if args.smoke or args.max_images is not None
                else "full-split"
            ),
            "split": args.split,
            "full_dataset_length": len(dataset),
            "requested_images": selection,
            "protocol": protocol,
            "protocol_sha256": canonical_json_sha256(protocol),
            "created_at_utc": utc_now(),
            "images": records,
        }
        atomic_write_json(staging_dir / "manifest.json", manifest)
        if args.output_dir.exists():
            raise FileExistsError(
                f"refusing to replace output created concurrently: {args.output_dir}"
            )
        os.rename(staging_dir, args.output_dir)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    print(f"phase_mask_manifest={(args.output_dir / 'manifest.json').resolve()}")
    print("phase_mask_status=PASS")


if __name__ == "__main__":
    main()
