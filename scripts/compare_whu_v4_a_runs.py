"""Validate and compare one official/mask-ignore V4-A run pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_whu_v4_a_screen import (
    confusion_summary,
    write_json_atomic,
)


EXTEND_E15_THRESHOLD_PP = 0.05
NONINFERIOR_TOLERANCE_PP = -0.05
FAITHFUL_HISTORICAL_BEST_MIOU_PERCENT = 54.145880


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected an object: {path}")
    return payload


def validate_protocol_pair(
    official: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    if official.get("variant") != "official":
        raise RuntimeError("first run is not the official variant")
    if candidate.get("variant") != "mask-ignore":
        raise RuntimeError("second run is not the mask-ignore variant")
    fields = (
        "schema_version",
        "git_commit",
        "seed",
        "scheduler_horizon_epochs",
        "stop_after_epoch",
        "evaluation_epochs",
        "model_name",
        "dataset_name",
        "num_modalities",
        "backbone_type",
        "use_lora",
        "train_batch_size_per_gpu",
        "train_workers",
        "inference_batch_size",
        "max_train_batches",
        "max_test_images",
        "scope",
        "initial_model_state_sha256",
        "train_dataset_length",
        "full_test_length",
        "evaluated_test_length",
    )
    mismatches = {
        field: (official.get(field), candidate.get(field))
        for field in fields
        if official.get(field) != candidate.get(field)
    }
    if mismatches:
        raise RuntimeError(f"paired protocol differs: {mismatches}")


def per_image_confusions(evaluation: Mapping[str, Any]) -> np.ndarray:
    records = evaluation.get("per_image")
    if not isinstance(records, list) or not records:
        raise RuntimeError("evaluation lacks per-image confusion")
    result = np.asarray([record["confusion"] for record in records], dtype=np.int64)
    if result.ndim != 3 or result.shape[1:] != (7, 7):
        raise RuntimeError("per-image confusion shape changed")
    return result


def pooled_miou(confusions: np.ndarray) -> float:
    return float(confusion_summary(confusions.sum(axis=0))["miou_percent"])


def paired_bootstrap_ci(
    official: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if official.shape != candidate.shape:
        raise RuntimeError("paired per-image confusion arrays differ in shape")
    rng = np.random.default_rng(seed)
    image_count = official.shape[0]
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = rng.integers(0, image_count, size=image_count)
        deltas[replicate] = pooled_miou(candidate[indices]) - pooled_miou(
            official[indices]
        )
    low, high = np.percentile(deltas, (2.5, 97.5))
    return {
        "replicates": replicates,
        "seed": seed,
        "low_pp": float(low),
        "high_pp": float(high),
        "descriptive_positive_fraction": float(np.mean(deltas > 0.0)),
        "interpretation": (
            "exploratory paired stability only; official test has been reused for selection"
        ),
    }


def compare_epoch(
    official_dir: Path,
    candidate_dir: Path,
    epoch: int,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    official_train = read_json(official_dir / f"train_e{epoch}.json")
    candidate_train = read_json(candidate_dir / f"train_e{epoch}.json")
    if official_train.get("paired_data_sha256") != candidate_train.get(
        "paired_data_sha256"
    ):
        raise RuntimeError(f"epoch {epoch} paired data stream differs")
    official_trace = [
        item.get("pair_sha256") for item in official_train.get("first_batch_trace", [])
    ]
    candidate_trace = [
        item.get("pair_sha256") for item in candidate_train.get("first_batch_trace", [])
    ]
    if official_trace != candidate_trace:
        raise RuntimeError(f"epoch {epoch} first-batch trace differs")

    official_eval = read_json(official_dir / f"evaluation_e{epoch}.json")
    candidate_eval = read_json(candidate_dir / f"evaluation_e{epoch}.json")
    if official_eval.get("label_sha256") != candidate_eval.get("label_sha256"):
        raise RuntimeError(f"epoch {epoch} test label stream differs")
    official_names = [item.get("sample_name") for item in official_eval["per_image"]]
    candidate_names = [item.get("sample_name") for item in candidate_eval["per_image"]]
    if official_names != candidate_names:
        raise RuntimeError(f"epoch {epoch} test image order differs")

    official_confusions = per_image_confusions(official_eval)
    candidate_confusions = per_image_confusions(candidate_eval)
    official_miou = pooled_miou(official_confusions)
    candidate_miou = pooled_miou(candidate_confusions)
    official_class = official_eval["aggregate"]["class_iou_percent"]
    candidate_class = candidate_eval["aggregate"]["class_iou_percent"]
    class_delta = {
        name: (
            None
            if official_class[name] is None or candidate_class[name] is None
            else float(candidate_class[name] - official_class[name])
        )
        for name in official_class
    }
    return {
        "epoch": epoch,
        "paired_data_sha256": official_train["paired_data_sha256"],
        "official_raw_label_sha256": official_train["raw_label_sha256"],
        "candidate_raw_label_sha256": candidate_train["raw_label_sha256"],
        "raw_label_stream_changed": (
            official_train["raw_label_sha256"] != candidate_train["raw_label_sha256"]
        ),
        "official_valid_fraction": official_train["valid_fraction"],
        "candidate_valid_fraction": candidate_train["valid_fraction"],
        "official_miou_percent": official_miou,
        "candidate_miou_percent": candidate_miou,
        "candidate_minus_official_pp": candidate_miou - official_miou,
        "class_iou_delta_pp": class_delta,
        "paired_bootstrap": paired_bootstrap_ci(
            official_confusions,
            candidate_confusions,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + epoch,
        ),
    }


def decision(scope: str, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if scope == "smoke":
        return {
            "outcome": "PASS_SMOKE_PAIR_CONTRACT",
            "scientific_decision": "NONE",
        }
    by_epoch = {int(record["epoch"]): record for record in results}
    if 15 not in by_epoch:
        return {
            "outcome": "PENDING_E15",
            "scientific_decision": "NONE",
        }
    delta15 = float(by_epoch[15]["candidate_minus_official_pp"])
    if not bool(by_epoch[15]["raw_label_stream_changed"]):
        return {
            "outcome": "FAIL_A_VARIANT_NOT_EXERCISED",
            "scientific_decision": "CHECK_PADDING_SAMPLING_AND_RUNNER",
        }
    rising = 10 not in by_epoch or delta15 > float(
        by_epoch[10]["candidate_minus_official_pp"]
    )
    if delta15 >= EXTEND_E15_THRESHOLD_PP and rising:
        outcome = "PASS_A_E15_EXTEND_TO_E30"
        action = "EXTEND_ONCE_TO_E30"
    elif delta15 >= NONINFERIOR_TOLERANCE_PP:
        outcome = "STOP_A_LOW_OR_FLAT_GAIN"
        action = "DO_NOT_FORCE_R_MASK_IGNORE_AS_METHOD_BASE"
    else:
        outcome = "STOP_A_NEGATIVE"
        action = "KEEP_R_OFFICIAL_AS_METHOD_BASE"
    return {
        "outcome": outcome,
        "scientific_decision": action,
        "delta15_pp": delta15,
        "e15_threshold_pp": EXTEND_E15_THRESHOLD_PP,
        "rising_from_e10": rising,
        "noninferior_tolerance_pp": NONINFERIOR_TOLERANCE_PP,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--mask-ignore-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    args = parser.parse_args(argv)
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.bootstrap_replicates <= 0:
        parser.error("bootstrap replicates must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    official_protocol = read_json(args.official_dir / "protocol.json")
    candidate_protocol = read_json(args.mask_ignore_dir / "protocol.json")
    validate_protocol_pair(official_protocol, candidate_protocol)
    epochs = tuple(int(value) for value in official_protocol["evaluation_epochs"])
    results = [
        compare_epoch(
            args.official_dir,
            args.mask_ignore_dir,
            epoch,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        for epoch in epochs
    ]
    verdict = decision(str(official_protocol["scope"]), results)
    payload = {
        "status": "PASS",
        "artifact_type": "whu_v4_a_paired_comparison",
        "git_commit": official_protocol["git_commit"],
        "seed": official_protocol["seed"],
        "scope": official_protocol["scope"],
        "faithful_historical_best_miou_percent": (
            FAITHFUL_HISTORICAL_BEST_MIOU_PERCENT
        ),
        "initial_model_state_sha256": official_protocol[
            "initial_model_state_sha256"
        ],
        "epochs": results,
        **verdict,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_path, payload)
    print(
        f"PASS outcome={payload['outcome']} output={args.output_path}", flush=True
    )
    for record in results:
        print(
            f"epoch={record['epoch']} official={record['official_miou_percent']:.6f}% "
            f"mask_ignore={record['candidate_miou_percent']:.6f}% "
            f"delta={record['candidate_minus_official_pp']:+.6f}pp",
            flush=True,
        )


if __name__ == "__main__":
    main()
