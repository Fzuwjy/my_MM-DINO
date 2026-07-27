"""Validate and optionally create the paths hard-coded by official MM-DINO.

The official Python sources remain untouched.  This script only creates narrow
symbolic links from the authors' expected paths to the server repository,
dataset, pretrained weight, split lists, and persistent output directory.
Existing non-matching paths are never removed or overwritten.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


BACKBONE_FILENAMES = {
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
}
AUTHOR_ROOT = Path("/home/yyyjvm")


def same_file_contents(left: Path, right: Path) -> bool:
    return left.is_file() and right.is_file() and left.read_bytes() == right.read_bytes()


def ensure_symlink(link: Path, target: Path, *, apply: bool, is_dir: bool) -> None:
    target = target.resolve()
    if os.path.lexists(link):
        if link.is_symlink() and link.resolve() == target:
            print(f"OK link: {link} -> {target}")
            return
        if not is_dir and same_file_contents(link, target):
            print(f"OK file: {link} matches {target}")
            return
        raise RuntimeError(
            f"Refusing to replace existing non-matching path: {link}. "
            "Inspect it manually before continuing."
        )

    print(f"PLAN link: {link} -> {target}")
    if apply:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=is_dir)


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_dataset(dataset_root: Path, names: list[str]) -> None:
    missing = []
    for name in names:
        for subdir in ("optical", "sar", "lbl"):
            candidate = dataset_root / subdir / name
            if not candidate.is_file():
                missing.append(str(candidate))
                if len(missing) >= 10:
                    break
        if len(missing) >= 10:
            break
    if missing:
        raise FileNotFoundError(
            "Dataset is incomplete; first missing paths:\n" + "\n".join(missing)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Create missing links after validation")
    parser.add_argument(
        "--backbone-type",
        choices=tuple(BACKBONE_FILENAMES),
        default="dinov3_vits16",
    )
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/mm-dino/datasets/whu-opt-sar")
    parser.add_argument("--weights-root", default="/root/autodl-tmp/mm-dino/weights")
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/mm-dino/outputs/faithful-whu-author-protocol",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    weights_root = Path(args.weights_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    train_split = repo_root / "splits" / "whu" / "official_train.txt"
    test_split = repo_root / "splits" / "whu" / "official_test.txt"
    backbone_filename = BACKBONE_FILENAMES[args.backbone_type]
    backbone = weights_root / backbone_filename

    for required in (dataset_root, train_split, test_split, backbone):
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    train_names = read_names(train_split)
    test_names = read_names(test_split)
    if len(train_names) != 80 or len(test_names) != 20:
        raise RuntimeError(
            f"Expected official 80/20 split, got {len(train_names)}/{len(test_names)}"
        )
    if set(train_names) & set(test_names):
        raise RuntimeError("Official train and test lists overlap")
    validate_dataset(dataset_root, train_names + test_names)

    ensure_symlink(dataset_root / "train_list.txt", train_split, apply=args.apply, is_dir=False)
    ensure_symlink(dataset_root / "test_list.txt", test_split, apply=args.apply, is_dir=False)
    ensure_symlink(
        AUTHOR_ROOT / "SS-datasets" / "whu-opt-sar",
        dataset_root,
        apply=args.apply,
        is_dir=True,
    )
    ensure_symlink(
        AUTHOR_ROOT / "Checkpoints" / "facebook" / backbone_filename,
        backbone,
        apply=args.apply,
        is_dir=False,
    )
    ensure_symlink(
        AUTHOR_ROOT / "SS-projects" / "dinov3",
        repo_root,
        apply=args.apply,
        is_dir=True,
    )
    ensure_symlink(
        repo_root / "tasks" / "segmentation" / "logs",
        output_root,
        apply=args.apply,
        is_dir=True,
    )

    if args.apply:
        output_root.mkdir(parents=True, exist_ok=True)
        print("faithful_whu_paths=READY")
    else:
        print("dry_run=PASSED; rerun with --apply to create the planned links")


if __name__ == "__main__":
    main()
