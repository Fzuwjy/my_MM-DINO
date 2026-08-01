"""Write a non-destructive adjudication sidecar for a V4-A comparison."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_whu_v4_a_runs import (
    A_DECISION_RULE_VERSION,
    decision,
    read_json,
)
from scripts.run_whu_v4_a_screen import write_json_atomic


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adjudicate(source_path: Path) -> dict[str, Any]:
    source = read_json(source_path)
    if source.get("status") != "PASS":
        raise RuntimeError("source comparison did not pass its execution contract")
    if source.get("artifact_type") != "whu_v4_a_paired_comparison":
        raise RuntimeError("source is not a V4-A paired comparison")
    epochs = source.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise RuntimeError("source comparison lacks epoch results")

    corrected = decision(str(source.get("scope")), epochs)
    raw_metrics = [
        {
            "epoch": record.get("epoch"),
            "official_miou_percent": record.get("official_miou_percent"),
            "mask_ignore_miou_percent": record.get("candidate_miou_percent"),
            "mask_ignore_minus_official_pp": record.get(
                "candidate_minus_official_pp"
            ),
            "paired_bootstrap": record.get("paired_bootstrap"),
        }
        for record in epochs
    ]
    return {
        "status": "PASS",
        "artifact_type": "whu_v4_a_adjudication_sidecar",
        "adjudication_rule_version": A_DECISION_RULE_VERSION,
        "source": {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
            "status": source["status"],
            "outcome": source.get("outcome"),
            "scientific_decision": source.get("scientific_decision"),
        },
        "reason": (
            "The original rule conflated E15 effect magnitude with trajectory and "
            "therefore mislabeled a strong positive but declining A result as low gain."
        ),
        "source_scientific_outcome_superseded": True,
        "raw_metrics": raw_metrics,
        "corrected": corrected,
        "baseline_decision": corrected["scientific_decision"],
        "durability": corrected.get("durability"),
        "scope_note": (
            "A is a training-protocol baseline selection, not a method contribution; "
            "its gain must not be counted as the C effect."
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.comparison_path.is_file():
        parser.error(f"comparison does not exist: {args.comparison_path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = adjudicate(args.comparison_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_path, payload)
    print(
        f"PASS outcome={payload['corrected']['outcome']} "
        f"output={args.output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
