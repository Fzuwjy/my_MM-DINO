"""Validate or launch the faithful MetaRS recipe with the true Val split.

The official EarthMiss source tree is imported as a pinned, read-only
dependency.  This script does not copy or modify its scientific implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "metars_clean.py"
DEFAULT_OFFICIAL_ROOT = Path(
    "/root/autodl-tmp/mm-dino/reference/EarthMiss-edcd0374"
)
DEFAULT_DATASET_ROOT = Path("/root/autodl-tmp/mm-dino/datasets/EarthMiss")
DEFAULT_TORCH_HOME = Path("/root/autodl-tmp/mm-dino/cache/torch")
RESNET50_FILENAME = "resnet50-19c8e357.pth"
RESNET50_SHA256_PREFIX = "19c8e357"
EXPECTED_COUNTS = {"train": 2641, "val": 277, "test": 437}


def _enter_official_tree(root: Path) -> None:
    root = root.resolve()
    required = (
        root / "train.py",
        root / "configs" / "baseline" / "MetaRS.py",
        root / "data" / "EarthMiss.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"invalid pinned EarthMiss tree; missing: {missing}")
    os.chdir(root)
    sys.path.insert(0, str(root))


def _files(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".tif", ".png"}
    )


def _normalized_names(files: list[Path], marker: str) -> set[str]:
    return {item.name.replace(marker, "") for item in files}


def inspect_dataset(dataset_root: Path, splits: dict[str, list[str]]) -> dict:
    report: dict[str, dict] = {}
    seen_city_files: set[tuple[str, str]] = set()
    for split, cities in splits.items():
        split_total = 0
        city_counts = {}
        split_keys: set[tuple[str, str]] = set()
        for city in cities:
            city_root = dataset_root / city
            sar = _files(city_root / "images" / "SAR")
            rgb = _files(city_root / "images" / "RGB")
            masks = _files(city_root / "masks")
            sar_names = _normalized_names(sar, "_SAR")
            rgb_names = _normalized_names(rgb, "")
            mask_names = _normalized_names(masks, "_mask")
            if not sar_names == rgb_names == mask_names:
                raise ValueError(f"unaligned SAR/RGB/mask files in {city}")
            city_counts[city] = len(rgb_names)
            split_total += len(rgb_names)
            split_keys.update((city, name) for name in rgb_names)
        if split_total != EXPECTED_COUNTS[split]:
            raise ValueError(
                f"{split} has {split_total} samples, expected {EXPECTED_COUNTS[split]}"
            )
        overlap = seen_city_files.intersection(split_keys)
        if overlap:
            raise ValueError(f"split overlap detected: {sorted(overlap)[:3]}")
        seen_city_files.update(split_keys)
        report[split] = {
            "cities": cities,
            "city_counts": city_counts,
            "samples": split_total,
        }
    return report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pretrain(torch_home: Path, require: bool) -> dict:
    path = torch_home / "hub" / "checkpoints" / RESNET50_FILENAME
    if not path.is_file():
        if require:
            raise FileNotFoundError(
                f"missing ImageNet initialization: {path}\n"
                "Run scripts/prepare_metars_pretrain.py before training."
            )
        return {"path": str(path), "present": False}
    digest = sha256(path)
    if not digest.startswith(RESNET50_SHA256_PREFIX):
        raise ValueError(f"unexpected SHA256 for {path}: {digest}")
    return {
        "path": str(path),
        "present": True,
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def validate_config(cfg, dataset_root: Path, splits: dict[str, list[str]]) -> None:
    for split, cities in splits.items():
        actual_images = list(cfg.data[split].params.image_dir)
        actual_masks = list(cfg.data[split].params.mask_dir)
        expected_images = [str(dataset_root / city / "images") for city in cities]
        expected_masks = [str(dataset_root / city / "masks") for city in cities]
        if actual_images != expected_images or actual_masks != expected_masks:
            raise ValueError(f"{split} paths do not match the released city split")
    if (
        list(cfg.model.params.data.params.image_dir)
        != list(cfg.data.val.params.image_dir)
        or list(cfg.model.params.data.params.mask_dir)
        != list(cfg.data.val.params.mask_dir)
    ):
        raise ValueError("MetaRS MMR loader is not tied to data.val")
    if cfg.data.train.params.batch_size != 4:
        raise ValueError("per-rank batch must remain 4 for the official two-rank recipe")
    if cfg.train.num_iters != 15000 or cfg.learning_rate.params.max_iters != 15000:
        raise ValueError("official 15000-iteration schedule was changed")
    if cfg.model.params.begin_mmr_iter != 1600:
        raise ValueError("official begin_mmr_iter=1600 was changed")


def main() -> None:
    import ever as er
    from ever.core.config import import_config

    parser = er.trainer.get_default_parser()
    parser.add_argument("--official-code-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--check-only", action="store_true")
    parser.set_defaults(config_path=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    args.dataset_root = args.dataset_root.resolve()
    args.torch_home = args.torch_home.resolve()
    os.environ["EARTHMISS_ROOT"] = str(args.dataset_root)
    os.environ["TORCH_HOME"] = str(args.torch_home)
    _enter_official_tree(args.official_code_root)

    from configs.metadata.EarthMiss import test_cities, train_cities, val_cities

    splits = {
        "train": list(train_cities),
        "val": list(val_cities),
        "test": list(test_cities),
    }
    cfg = import_config(str(Path(args.config_path).resolve()))
    if args.opts:
        cfg.update_from_list(args.opts)
    validate_config(cfg, args.dataset_root, splits)
    report = {
        "protocol": "MetaRS-clean: official recipe with data.val corrected to true Val",
        "official_code_root": str(args.official_code_root.resolve()),
        "config_path": str(Path(args.config_path).resolve()),
        "dataset": inspect_dataset(args.dataset_root, splits),
        "pretrain": inspect_pretrain(args.torch_home, require=not args.check_only),
        "effective_recipe": {
            "world_size": 2,
            "batch_per_rank": 4,
            "global_batch": 8,
            "iterations": 15000,
            "begin_mmr_iter": 1600,
            "seed": 2333,
        },
    }
    if args.check_only:
        print(json.dumps(report, indent=2))
        return

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("training requires GPU mode; use --check-only without a GPU")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise RuntimeError(
            f"official effective recipe requires WORLD_SIZE=2, got {world_size}"
        )
    if args.model_dir is None:
        raise ValueError("--model_dir is required for training")
    if args.local_rank is None:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    # These imports register the official loader/model and reuse the official
    # evaluator and seeding behavior unchanged.
    from data.EarthMiss import EarthM3DALoader  # noqa: F401
    from module.baseline.MetaRS import MetaRS  # noqa: F401
    from train import register_evaluate_fn, seed_torch

    seed_torch(2333)
    trainer = er.trainer.TRAINER[args.trainer](args)()
    trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])


if __name__ == "__main__":
    main()
