"""Shared, host-independent configuration for segmentation experiments."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


# These defaults keep generated data out of the source tree tracked by Git while
# still making a fresh clone usable without editing Python files. On a server,
# set the corresponding environment variables or use the CLI overrides.
DATASETS_ROOT = _path_from_env("MM_DINO_DATASETS_ROOT", PROJECT_ROOT / "data")
WEIGHTS_ROOT = _path_from_env("MM_DINO_WEIGHTS_ROOT", PROJECT_ROOT / "weights")
OUTPUT_ROOT = _path_from_env("MM_DINO_OUTPUT_ROOT", PROJECT_ROOT / "outputs")


BACKBONE_WEIGHT_FILES = {
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    "dinov3_vit7b16": "dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
}


DATASET_LABELS = {
    "Vaihingen": [
        "roads",
        "buildings",
        "low veg.",
        "trees",
        "cars",
        "clutter",
    ],
    "Potsdam": [
        "roads",
        "buildings",
        "low veg.",
        "trees",
        "cars",
        "clutter",
    ],
    "YYYJ": [
        "地基建设",
        "基础结构建设",
        "封顶厂房",
        "封顶楼房",
        "施工道路",
        "硬化道路",
        "风电施工",
        "风电",
        "光伏",
        "推填土",
        "体育场地",
        "临时棚房",
        "自建房",
        "专属设施",
        "未定义",
    ],
    "EarthMiss": [
        "Background",
        "Building",
        "Road",
        "Water",
        "Barren",
        "Forest",
        "Agricultural",
        "Playground",
    ],
    "WHU": ["农田", "城市", "村庄", "水体", "森林", "道路", "其他"],
}


def get_labels(dataset_name: str | None = None) -> list[str]:
    if dataset_name is None:
        raise ValueError("Please specify a dataset")
    try:
        return list(DATASET_LABELS[dataset_name])
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_LABELS))
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}"
        ) from exc
