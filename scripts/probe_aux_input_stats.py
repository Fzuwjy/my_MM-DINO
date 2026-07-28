"""Audit auxiliary-modality values, pairing, and coarse geometric alignment.

This script is read-only.  It uses the released dataset builders and writes a
JSON report plus optional PNG inspection artifacts outside the dataset tree.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.aux_diagnostics_common import (
    array_summary,
    edge_alignment_summary,
    tensor_summary,
    to_grayscale,
)


SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit MM-DINO auxiliary inputs without changing data"
    )
    parser.add_argument(
        "--dataset",
        choices=("WHU", "Vaihingen", "Potsdam"),
        required=True,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=("train", "test"),
    )
    parser.add_argument("--samples-per-split", type=int, default=4)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--ms-root-dir", type=Path, default=Path("/home/yyyjvm"))
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def _configure_imports(ms_root_dir: Path) -> Any:
    segmentation_root = str(SEGMENTATION_ROOT)
    repo_root = str(REPO_ROOT)
    if segmentation_root not in sys.path:
        sys.path.insert(0, segmentation_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import configs.common_cfg as common_cfg
    import datasets as released_datasets

    common_cfg.MS_ROOT_DIR = str(ms_root_dir)
    released_datasets.MS_ROOT_DIR = str(ms_root_dir)
    return released_datasets.build_dataset


def _paths_for_dataset(dataset: Any) -> list[tuple[Path, Path, Path]]:
    if hasattr(dataset, "rgb_files"):
        rgb_files = dataset.rgb_files
        aux_files = dataset.sar_files
    else:
        rgb_files = dataset.data_files
        aux_files = dataset.dsm_files
    label_files = dataset.label_files
    if not (len(rgb_files) == len(aux_files) == len(label_files)):
        raise RuntimeError("Released dataset file lists have different lengths")
    return [
        (Path(rgb), Path(auxiliary), Path(label))
        for rgb, auxiliary, label in zip(
            rgb_files, aux_files, label_files, strict=True
        )
    ]


def _sample_indices(length: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("samples-per-split must be positive")
    count = min(length, count)
    return [int(value) for value in np.linspace(0, length - 1, count)]


def _pil_record(path: Path, *, include_distribution: bool) -> dict[str, Any]:
    with Image.open(path) as image:
        record: dict[str, Any] = {
            "path": str(path),
            "exists": True,
            "format": image.format,
            "mode": image.mode,
            "size": list(image.size),
            "extrema": image.getextrema(),
        }
        if include_distribution:
            record["distribution"] = array_summary(np.asarray(image))
        return record


def _percentile_uint8(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(array, [1.0, 99.0])
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    scaled = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _save_artifacts(
    rgb: torch.Tensor,
    auxiliary: torch.Tensor,
    output_dir: Path,
    stem: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    max_side = 1024
    height, width = rgb.shape[-2:]
    top = max((height - max_side) // 2, 0)
    left = max((width - max_side) // 2, 0)
    crop_height = min(height, max_side)
    crop_width = min(width, max_side)
    rgb = rgb[:, top : top + crop_height, left : left + crop_width]
    auxiliary = auxiliary[:, top : top + crop_height, left : left + crop_width]
    rgb_array = np.moveaxis(rgb.detach().cpu().numpy(), 0, -1)
    aux_array = to_grayscale(auxiliary.detach().cpu().numpy())
    rgb_vis = np.stack(
        [_percentile_uint8(rgb_array[..., channel]) for channel in range(3)],
        axis=-1,
    )
    aux_vis = _percentile_uint8(aux_array)
    aux_color = np.stack(
        [aux_vis, np.zeros_like(aux_vis), 255 - aux_vis], axis=-1
    )
    overlay = np.round(0.65 * rgb_vis + 0.35 * aux_color).astype(np.uint8)

    rgb_path = output_dir / f"{stem}_rgb.png"
    aux_path = output_dir / f"{stem}_aux.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    existing = [path for path in (rgb_path, aux_path, overlay_path) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite artifacts: {existing}")
    Image.fromarray(rgb_vis, mode="RGB").save(rgb_path)
    Image.fromarray(aux_vis, mode="L").save(aux_path)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)
    return {
        "rgb": str(rgb_path),
        "auxiliary": str(aux_path),
        "overlay": str(overlay_path),
    }


def _clear_released_caches(dataset: Any) -> None:
    for name in ("data_cache", "label_cache", "dsm_cache"):
        cache = getattr(dataset, name, None)
        if isinstance(cache, dict):
            cache.clear()
    for name in ("rgb_cache", "label_cache", "sar_cache"):
        cache = getattr(dataset, name, None)
        mapping = getattr(cache, "cache", None)
        if mapping is not None:
            mapping.clear()


def _audit_split(
    build_dataset: Any,
    dataset_name: str,
    split: str,
    *,
    window_size: int,
    sample_count: int,
    seed: int,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = build_dataset(
        dataset_name,
        split,
        window_size=(window_size, window_size),
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vitl16",
    )

    file_triplets = _paths_for_dataset(dataset)
    raw_indices = _sample_indices(len(file_triplets), sample_count)
    raw_records = []
    for index in raw_indices:
        rgb_path, aux_path, label_path = file_triplets[index]
        rgb_record = _pil_record(rgb_path, include_distribution=False)
        aux_record = _pil_record(aux_path, include_distribution=True)
        label_record = _pil_record(label_path, include_distribution=False)
        raw_records.append(
            {
                "index": index,
                "rgb": rgb_record,
                "auxiliary": aux_record,
                "label": label_record,
                "same_spatial_size": (
                    rgb_record["size"]
                    == aux_record["size"]
                    == label_record["size"]
                ),
                "same_basename_whu": (
                    rgb_path.name == aux_path.name == label_path.name
                    if dataset_name == "WHU"
                    else None
                ),
            }
        )

    item_records = []
    for sample_number, dataset_index in enumerate(
        _sample_indices(len(dataset), sample_count)
    ):
        item = dataset[dataset_index]
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            raise RuntimeError("Expected released multimodal dataset item")
        rgb, auxiliary, label = item
        rgb_numpy = rgb.detach().cpu().numpy()
        auxiliary_numpy = auxiliary.detach().cpu().numpy()
        geometry: dict[str, Any]
        try:
            geometry = edge_alignment_summary(rgb_numpy, auxiliary_numpy)
        except ValueError as error:
            geometry = {"error": str(error)}
        artifacts = None
        if artifact_dir is not None:
            artifacts = _save_artifacts(
                rgb,
                auxiliary,
                artifact_dir,
                f"{dataset_name.lower()}_{split}_{sample_number:02d}",
            )
        item_records.append(
            {
                "requested_index": dataset_index,
                "rgb": tensor_summary(rgb),
                "auxiliary": tensor_summary(auxiliary),
                "label": array_summary(np.asarray(label)),
                "same_spatial_shape": (
                    list(rgb.shape[-2:])
                    == list(auxiliary.shape[-2:])
                    == list(np.asarray(label).shape[-2:])
                ),
                "edge_alignment": geometry,
                "artifacts": artifacts,
            }
        )
        _clear_released_caches(dataset)

    return {
        "split": split,
        "dataset_length": len(dataset),
        "source_file_count": len(file_triplets),
        "raw_files": raw_records,
        "dataset_items": item_records,
    }


def main() -> None:
    args = parse_args()
    output_path = args.output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {output_path}")
    artifact_dir = (
        args.artifact_dir.expanduser().resolve() if args.artifact_dir else None
    )
    build_dataset = _configure_imports(args.ms_root_dir.expanduser().resolve())

    result = {
        "schema_version": 1,
        "dataset": args.dataset,
        "seed": args.seed,
        "window_size": args.window_size,
        "ms_root_dir": str(args.ms_root_dir.expanduser().resolve()),
        "read_only": True,
        "splits": [
            _audit_split(
                build_dataset,
                args.dataset,
                split,
                window_size=args.window_size,
                sample_count=args.samples_per_split,
                seed=args.seed,
                artifact_dir=artifact_dir,
            )
            for split in args.splits
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"audit_report={output_path}")


if __name__ == "__main__":
    main()
