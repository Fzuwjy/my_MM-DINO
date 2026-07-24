"""Dataset factory with portable roots and explicit split handling."""

from __future__ import annotations

from pathlib import Path

from configs.common_cfg import DATASETS_ROOT

from .EarthMiss_dataset import EarthMiss_Dataset
from .ISPRS_dataset import ISRPS_Dataset
from .WHU_dataset import WHU_Dataset
from .YYYJ_dataset import YYYJ_Dataset


VALID_SPLITS = {"train", "val", "test"}


def _read_split(path: Path, cast=str) -> list:
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path}. Formal training requires an explicit "
            "validation split; use --eval-split test only to reproduce the official protocol."
        )
    with path.open("r", encoding="utf-8") as handle:
        return [cast(line.strip()) for line in handle if line.strip()]


def _resolve_split(default_ids, data_type: str, split_file, cast=str):
    if split_file:
        return _read_split(Path(split_file).expanduser().resolve(), cast=cast)
    if data_type == "val":
        raise ValueError(
            "This dataset has no official validation split. Pass --eval-split-file "
            "with tile/image identifiers, or use --eval-split test only for official reproduction."
        )
    return default_ids


def build_dataset(dataset_name, data_type="test", **kwargs):
    if data_type not in VALID_SPLITS:
        raise ValueError(f"Invalid data split '{data_type}'. Expected one of {sorted(VALID_SPLITS)}")

    model_name = kwargs.get("model_name")
    backbone_type = kwargs.get("backbone_type")
    normalize_type = (
        "geo"
        if model_name == "DINOv3"
        and backbone_type in ["dinov3_vitl16", "dinov3_vit7b16"]
        else "common"
    )
    datasets_root = Path(kwargs.get("datasets_root") or DATASETS_ROOT).expanduser().resolve()
    split_file = kwargs.get("split_file")
    is_multi = kwargs.get("modality") == "multi"
    window_size = kwargs.get("window_size", (224, 224))
    cache_size = int(kwargs.get("cache_size", 2))
    if cache_size < 0:
        raise ValueError("cache_size must be non-negative")

    if dataset_name == "Potsdam":
        test_ids = ["4_10", "5_11", "2_11", "3_10", "6_11", "7_12"]
        train_ids = [
            "6_10", "7_10", "2_12", "3_11", "2_10", "7_8", "5_10",
            "3_12", "5_12", "7_11", "7_9", "6_9", "7_7", "4_12",
            "6_8", "6_12", "6_7", "4_11",
        ]
        ids = _resolve_split(
            test_ids if data_type == "test" else train_ids,
            data_type,
            split_file,
        )
        root = datasets_root / "ISPRS_dataset" / "Potsdam"
        return ISRPS_Dataset(
            ids=ids,
            data_dir=str(root / "2_Ortho_RGB" / "top_potsdam_{}_RGB.tif"),
            label_dir=str(
                root
                / "5_Labels_for_participants_no_Boundary"
                / "top_potsdam_{}_label_noBoundary.tif"
            ),
            dsm_dir=(
                str(
                    root
                    / "1_DSM_normalisation"
                    / "dsm_potsdam_{}_normalized_lastools.jpg"
                )
                if is_multi
                else None
            ),
            dataset_name=dataset_name,
            data_type=data_type,
            window_size=window_size,
            normalize_type=normalize_type,
        )

    if dataset_name == "Vaihingen":
        test_ids = [5, 21, 15, 30]
        train_ids = [1, 3, 23, 26, 7, 11, 13, 28, 17, 32, 34, 37]
        ids = _resolve_split(
            test_ids if data_type == "test" else train_ids,
            data_type,
            split_file,
            cast=int,
        )
        root = datasets_root / "ISPRS_dataset" / "Vaihingen"
        return ISRPS_Dataset(
            ids=ids,
            data_dir=str(root / "top" / "top_mosaic_09cm_area{}.tif"),
            label_dir=str(
                root
                / "gts_eroded_for_participants"
                / "top_mosaic_09cm_area{}_noBoundary.tif"
            ),
            dsm_dir=(
                str(root / "dsm" / "dsm_09cm_matching_area{}.tif")
                if is_multi
                else None
            ),
            dataset_name=dataset_name,
            data_type=data_type,
            window_size=window_size,
            normalize_type=normalize_type,
        )

    if dataset_name == "YYYJ":
        root = datasets_root / "YYYJ_dataset"
        source_dir = root / ("new_test" if data_type == "test" else "new_train")
        if split_file:
            ids = _read_split(Path(split_file).expanduser().resolve())
        elif data_type == "val":
            raise ValueError("YYYJ validation requires --eval-split-file")
        else:
            if not source_dir.is_dir():
                raise FileNotFoundError(f"Dataset directory not found: {source_dir}")
            ids = sorted(path.stem for path in source_dir.glob("*.tif"))
        return YYYJ_Dataset(
            ids=ids,
            data_dir=str(source_dir / "{}.tif"),
            label_dir=str(source_dir / "label_masks" / "{}.tif"),
            data_type=data_type,
            window_size=window_size,
            normalize_type=normalize_type,
        )

    if dataset_name == "EarthMiss":
        train_cities = [
            "Singapore", "Nanjing", "America-Eugene", "America-Louisville",
            "French-Paris", "Netherlands-Rotterdam", "Morocco-Casablanca",
        ]
        test_cities = ["Japan-Hakodate", "America-NewYork", "Peru-Callao"]
        cities = _resolve_split(
            test_cities if data_type == "test" else train_cities,
            data_type,
            split_file,
        )
        root = datasets_root / "EarthMiss"
        return EarthMiss_Dataset(
            citys=cities,
            rgb_dir=str(root / "{}" / "images" / "RGB"),
            sar_dir=str(root / "{}" / "images" / "SAR") if is_multi else None,
            label_dir=str(root / "{}" / "masks"),
            data_type=data_type,
            window_size=window_size,
            normalize_type=normalize_type,
            cache_size=cache_size,
        )

    if dataset_name == "WHU":
        root = datasets_root / "whu-opt-sar"
        list_path = (
            Path(split_file).expanduser().resolve()
            if split_file
            else root / f"{data_type}_list.txt"
        )
        filenames = _read_split(list_path)
        return WHU_Dataset(
            filenames=filenames,
            rgb_dir=str(root / "optical" / "{}"),
            label_dir=str(root / "lbl" / "{}"),
            sar_dir=str(root / "sar" / "{}") if is_multi else None,
            data_type=data_type,
            window_size=window_size,
            normalize_type=normalize_type,
            cache_size=cache_size,
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")
