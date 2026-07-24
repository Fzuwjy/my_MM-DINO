"""Create a deterministic, group-disjoint train/validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def read_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    if not names:
        raise ValueError(f"No image names found in {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate image names found in {path}")
    return names


def choose_validation_groups(group_sizes: dict[str, int], target: int, seed: int):
    """Subset-sum selection with seeded tie-breaking."""
    groups = list(group_sizes)
    random.Random(seed).shuffle(groups)
    states = {0: ()}
    for group in groups:
        additions = {
            count + group_sizes[group]: chosen + (group,)
            for count, chosen in list(states.items())
        }
        for count, chosen in additions.items():
            states.setdefault(count, chosen)
    selected_count = min(states, key=lambda count: (abs(count - target), count < target, count))
    return set(states[selected_count])


def digest_lines(lines: list[str]) -> str:
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_lines(path: Path, lines: list[str], force: bool):
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force if intentional")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Official training list")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--manifest-output")
    parser.add_argument("--group-prefix-length", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.group_prefix_length <= 0:
        parser.error("--group-prefix-length must be positive")
    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be in (0, 1)")
    return args


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    names = read_names(input_path)

    grouped = defaultdict(list)
    for name in names:
        key = Path(name).stem[: args.group_prefix_length]
        grouped[key].append(name)
    if len(grouped) < 2:
        raise ValueError("The grouping rule produced fewer than two groups")

    target = round(len(names) * args.val_ratio)
    validation_groups = choose_validation_groups(
        {group: len(items) for group, items in grouped.items()},
        target,
        args.seed,
    )
    train_names = [
        name
        for name in names
        if Path(name).stem[: args.group_prefix_length] not in validation_groups
    ]
    val_names = [
        name
        for name in names
        if Path(name).stem[: args.group_prefix_length] in validation_groups
    ]
    if not train_names or not val_names:
        raise RuntimeError("Generated an empty train or validation split")

    train_output = Path(args.train_output).expanduser().resolve()
    val_output = Path(args.val_output).expanduser().resolve()
    manifest_output = (
        Path(args.manifest_output).expanduser().resolve()
        if args.manifest_output
        else val_output.with_suffix(val_output.suffix + ".manifest.json")
    )
    if not args.force:
        existing = [path for path in (train_output, val_output, manifest_output) if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing outputs: "
                + ", ".join(str(path) for path in existing)
                + "; pass --force if intentional"
            )
    write_lines(train_output, train_names, args.force)
    write_lines(val_output, val_names, args.force)

    manifest = {
        "source": str(input_path),
        "source_sha256": digest_lines(names),
        "seed": args.seed,
        "val_ratio_requested": args.val_ratio,
        "group_prefix_length": args.group_prefix_length,
        "validation_groups": sorted(validation_groups),
        "train_count": len(train_names),
        "val_count": len(val_names),
        "train_sha256": digest_lines(train_names),
        "val_sha256": digest_lines(val_names),
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
