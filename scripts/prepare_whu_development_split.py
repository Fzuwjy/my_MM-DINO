"""Create a deterministic group-disjoint WHU development Train/Val split.

Only the published 80-scene Train manifest and GT label histograms are read.
Model predictions, SAR, Optical, and the official 20-scene Test manifest never
enter the selection objective.  Output is written to a new directory and is
intended to be reviewed and committed before any new baseline is trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from skimage.io import imread


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = "/root/autodl-tmp/mm-dino/datasets/whu-opt-sar"
DEFAULT_OFFICIAL_TRAIN = str(REPO_ROOT / "splits" / "whu" / "official_train.txt")
DEFAULT_OUTPUT_DIR = (
    "/root/autodl-tmp/mm-dino/outputs/whu-development-split-v1"
)
NUM_CLASSES = 7
SEED = 20260817
TARGET_VAL_SCENES = 16
DEFAULT_TRIALS = 20_000
GROUP_PATTERN = re.compile(r"^([A-Z]{2}\d{2}E\d{3})\d{3}\.tif$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--official-train", default=DEFAULT_OFFICIAL_TRAIN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-scenes", type=int, default=TARGET_VAL_SCENES)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def map_sheet_group(filename: str) -> str:
    match = GROUP_PATTERN.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"unrecognized WHU map-sheet filename: {filename}")
    return match.group(1)


def macro_region(group: str) -> str:
    return group[:4]


def _read_names(path: str | Path) -> tuple[str, ...]:
    names = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(names) != 80 or len(names) != len(set(names)):
        raise RuntimeError("published WHU Train manifest must contain 80 unique scenes")
    return names


def label_histogram(path: str | Path) -> np.ndarray:
    label = np.asarray(imread(path))
    if label.ndim != 2:
        raise ValueError(f"WHU label must be 2-D: {path}")
    valid_values = {0, 10, 20, 30, 40, 50, 60, 70}
    observed = set(np.unique(label).tolist())
    if not observed.issubset(valid_values):
        raise ValueError(f"unexpected WHU label ids {sorted(observed - valid_values)}")
    return np.asarray([(label == 10 * (index + 1)).sum() for index in range(NUM_CLASSES)], dtype=np.int64)


def _random_exact_subset(
    group_counts: Mapping[str, int],
    target: int,
    generator: np.random.Generator,
) -> tuple[str, ...] | None:
    groups = list(group_counts)
    generator.shuffle(groups)
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group in groups:
        count = int(group_counts[group])
        for current in sorted(tuple(reachable), reverse=True):
            updated = current + count
            if updated <= target and updated not in reachable:
                reachable[updated] = reachable[current] + (group,)
    value = reachable.get(target)
    return tuple(sorted(value)) if value is not None else None


def _distribution(histogram: np.ndarray) -> np.ndarray:
    total = float(histogram.sum())
    if total <= 0:
        raise ValueError("class histogram has no valid pixels")
    return histogram.astype(np.float64) / total


def choose_validation_groups(
    group_histograms: Mapping[str, np.ndarray],
    group_counts: Mapping[str, int],
    *,
    target_scenes: int,
    trials: int,
    seed: int,
) -> tuple[tuple[str, ...], dict[str, float]]:
    if set(group_histograms) != set(group_counts):
        raise ValueError("group histogram/count manifests differ")
    if trials <= 0:
        raise ValueError("trials must be positive")
    all_hist = sum((value for value in group_histograms.values()), np.zeros(NUM_CLASSES, dtype=np.int64))
    all_distribution = _distribution(all_hist)
    required_regions = {macro_region(group) for group in group_counts}
    generator = np.random.default_rng(seed)
    best = None
    seen = set()
    for _ in range(trials):
        selected = _random_exact_subset(group_counts, target_scenes, generator)
        if selected is None or selected in seen:
            continue
        seen.add(selected)
        if {macro_region(group) for group in selected} != required_regions:
            continue
        val_hist = sum(
            (group_histograms[group] for group in selected),
            np.zeros(NUM_CLASSES, dtype=np.int64),
        )
        train_hist = all_hist - val_hist
        if (val_hist == 0).any() or (train_hist == 0).any():
            continue
        val_distribution = _distribution(val_hist)
        train_distribution = _distribution(train_hist)
        val_max_abs = float(np.max(np.abs(val_distribution - all_distribution)))
        train_max_abs = float(np.max(np.abs(train_distribution - all_distribution)))
        val_l1 = float(np.abs(val_distribution - all_distribution).sum())
        score = val_max_abs + 0.5 * train_max_abs + 0.1 * val_l1
        candidate = (score, selected, {
            "score": score,
            "val_max_abs_class_fraction_delta": val_max_abs,
            "train_max_abs_class_fraction_delta": train_max_abs,
            "val_l1_class_fraction_delta": val_l1,
            "unique_candidates": len(seen),
        })
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("no feasible group-disjoint WHU development split found")
    best[2]["unique_candidates"] = len(seen)
    return best[1], best[2]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.val_scenes <= 0 or args.val_scenes >= 80:
        raise ValueError("--val-scenes must be in [1,79]")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite split directory: {output_dir}")
    names = _read_names(args.official_train)
    groups: dict[str, list[str]] = {}
    group_histograms: dict[str, np.ndarray] = {}
    scene_histograms = {}
    for name in names:
        group = map_sheet_group(name)
        histogram = label_histogram(Path(args.dataset_root) / "lbl" / name)
        groups.setdefault(group, []).append(name)
        group_histograms[group] = group_histograms.get(
            group, np.zeros(NUM_CLASSES, dtype=np.int64)
        ) + histogram
        scene_histograms[name] = histogram
    group_counts = {group: len(values) for group, values in groups.items()}
    val_groups, objective = choose_validation_groups(
        group_histograms, group_counts, target_scenes=args.val_scenes,
        trials=args.trials, seed=args.seed,
    )
    val_group_set = set(val_groups)
    val_names = tuple(name for name in names if map_sheet_group(name) in val_group_set)
    train_names = tuple(name for name in names if map_sheet_group(name) not in val_group_set)
    if len(val_names) != args.val_scenes or len(train_names) + len(val_names) != 80:
        raise RuntimeError("development split scene counts changed")
    if {map_sheet_group(name) for name in train_names} & val_group_set:
        raise RuntimeError("map-sheet group leakage detected")
    train_text = "\n".join(train_names) + "\n"
    val_text = "\n".join(val_names) + "\n"
    all_hist = sum(scene_histograms.values(), np.zeros(NUM_CLASSES, dtype=np.int64))
    train_hist = sum(
        (scene_histograms[name] for name in train_names),
        np.zeros(NUM_CLASSES, dtype=np.int64),
    )
    val_hist = all_hist - train_hist
    manifest = {
        "schema": "whu_development_split_v1",
        "seed": args.seed,
        "selection_inputs": "official Train filenames + GT class pixel histograms only",
        "official_test_was_accessed": False,
        "group_rule": "map-sheet prefix before final three-digit tile id",
        "train_scenes": len(train_names),
        "val_scenes": len(val_names),
        "train_groups": sorted({map_sheet_group(name) for name in train_names}),
        "val_groups": list(val_groups),
        "group_overlap": [],
        "macro_regions": sorted({macro_region(group) for group in groups}),
        "class_pixels": {
            "all": all_hist.tolist(),
            "train": train_hist.tolist(),
            "val": val_hist.tolist(),
        },
        "class_fractions": {
            "all": _distribution(all_hist).tolist(),
            "train": _distribution(train_hist).tolist(),
            "val": _distribution(val_hist).tolist(),
        },
        "objective": objective,
        "files": {
            "development_train.txt": _sha256_text(train_text),
            "development_val.txt": _sha256_text(val_text),
        },
    }
    output_dir.mkdir(parents=True)
    (output_dir / "development_train.txt").write_text(train_text, encoding="utf-8")
    (output_dir / "development_val.txt").write_text(val_text, encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
