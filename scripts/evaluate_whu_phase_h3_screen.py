"""Run the frozen offline H3 K1->K2 predictability screen.

This runner never loads the model or dataset.  It cross-fits one of three
predeclared K1-only scores and four coarse q values with whole-image
leave-one-out selection.  The held-out action map is built from score and
public phase-closure geometry before its labels are used for evaluation.

The resulting K1/K2 policy is only an exploratory official-test screen.  A
pass can authorize one live small/thin and same-primitive latency check; it
cannot establish a deployable router or a paper-ready H3 result.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_whu_phase_closure import (  # noqa: E402
    atomic_write_json,
    file_sha256,
    load_stage_a,
    metric_summary,
)
from scripts.phase_closure_common import (  # noqa: E402
    build_phase_closure_geometry,
    confusion_for_levels,
    mean_iou_from_confusion,
    summarize_closure,
)
from scripts.phase_h3_screen_common import (  # noqa: E402
    physical_cost_summary,
    random_exact_cost_k2_levels,
    score_ranked_k2_levels,
)


ARTIFACT_TYPE = "whu_phase_h3_k1_to_k2_screen"
SCHEMA_VERSION = 1
EXPECTED_STAGE_B0_TYPE = "whu_phase_closure_stage_b0"
EXPECTED_LATENCY_TYPE = "whu_phase_sparse_latency_b1"
EXPECTED_LATENCY_OUTCOME = (
    "PASS_FULL_STAGE_B_H2_GT_ORACLE_EXECUTION_FEASIBILITY"
)
SCORE_SPECS = (
    ("entropy", "k1_entropy"),
    ("negative_margin", "k1_negative_margin"),
    ("predicted_boundary_density", "k1_predicted_boundary_density"),
)
Q_VALUES = (0.10, 0.20, 1.0 / 3.0, 0.50)
RANDOM_REPLICATES = 1000
RANDOM_SEED = 20260731
INFERENCE_BATCH_SIZE = 8
MAX_PHYSICAL_COST = 2.0
MIN_GAIN_OVER_K2 = 0.0005
MIN_K4_GAIN_RETENTION = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline whole-image-LOO H3 K1->K2 score screen"
    )
    parser.add_argument("--stage-a-json", type=Path, required=True)
    parser.add_argument("--stage-b0-json", type=Path, required=True)
    parser.add_argument("--latency-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--random-replicates", type=int, default=RANDOM_REPLICATES
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    for name in ("stage_a_json", "stage_b0_json", "latency_json"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"input artifact does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.random_replicates <= 0:
        parser.error("--random-replicates must be positive")
    if args.random_seed < 0:
        parser.error("--random-seed must be non-negative")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def load_stage_b0(path: Path, *, stage_a_sha256: str) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("artifact_type") != EXPECTED_STAGE_B0_TYPE:
        raise ValueError("input is not a WHU Stage-B0 closure artifact")
    if payload.get("schema_version") != 1 or payload.get("status") != "PASS":
        raise ValueError("H3 requires the PASS Stage-B0 schema version 1 artifact")
    source = payload.get("source_stage_a")
    if not isinstance(source, Mapping) or source.get("sha256") != stage_a_sha256:
        raise ValueError("Stage-B0 source does not match the supplied Stage-A artifact")
    oracle = payload.get("exact_cost_a2_oracle")
    if not isinstance(oracle, Mapping):
        raise ValueError("Stage-B0 artifact lacks its exact-cost oracle")
    levels = oracle.get("levels_by_cell")
    if not isinstance(levels, list):
        raise ValueError("Stage-B0 oracle levels are missing")
    decision = payload.get("stage_b0_decision")
    if not isinstance(decision, Mapping) or decision.get("known_checks_passed") is not True:
        raise ValueError("Stage-B0 known scientific gates did not pass")
    return payload


def load_latency_confirmation(
    path: Path,
    *,
    stage_a_sha256: str,
    stage_b0_sha256: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("artifact_type") != EXPECTED_LATENCY_TYPE:
        raise ValueError("input is not a WHU Stage-B latency artifact")
    if payload.get("schema_version") != 1 or payload.get("status") != "PASS":
        raise ValueError("H3 requires the PASS Stage-B latency schema version 1")
    sources = payload.get("source_artifacts")
    if not isinstance(sources, Mapping):
        raise ValueError("latency artifact lacks source provenance")
    if sources.get("stage_a", {}).get("sha256") != stage_a_sha256:
        raise ValueError("latency Stage-A source does not match the supplied artifact")
    if sources.get("stage_b0", {}).get("sha256") != stage_b0_sha256:
        raise ValueError("latency Stage-B0 source does not match the supplied artifact")
    decision = payload.get("latency_decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("evaluated") is not True
        or decision.get("passed") is not True
        or decision.get("full_stage_b_scientific_pass") is not True
        or decision.get("h2_confirmed") is not True
        or decision.get("outcome") != EXPECTED_LATENCY_OUTCOME
    ):
        raise ValueError("latency artifact does not confirm H2/full Stage B")
    return payload


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _confusion(value: Any, *, name: str, num_classes: int) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.shape != (num_classes, num_classes):
        raise ValueError(f"{name} has the wrong confusion shape")
    if not np.issubdtype(matrix.dtype, np.integer) or np.any(matrix < 0):
        raise ValueError(f"{name} must be a non-negative integer confusion")
    return np.ascontiguousarray(matrix, dtype=np.int64)


def _score_arrays(
    cells: Sequence[Mapping[str, Any]], eligible: np.ndarray
) -> dict[str, np.ndarray]:
    if eligible.shape != (len(cells),) or eligible.dtype != np.bool_:
        raise TypeError("eligible must be a bool vector matching cells")
    result = {}
    for public_name, cell_key in SCORE_SPECS:
        values = np.full(len(cells), -np.inf, dtype=np.float64)
        for index, (cell, is_eligible) in enumerate(
            zip(cells, eligible, strict=True)
        ):
            scores = cell.get("scores")
            if not isinstance(scores, Mapping):
                raise TypeError("every cell must contain a score mapping")
            raw = scores.get(cell_key)
            if is_eligible:
                if isinstance(raw, (bool, np.bool_)) or not isinstance(
                    raw, (int, float, np.number)
                ):
                    raise TypeError(f"eligible cell lacks numeric {cell_key}")
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError(f"eligible cell has non-finite {cell_key}")
                values[index] = value
            elif raw is not None:
                raise ValueError(f"ineligible cell unexpectedly has {cell_key}")
        result[public_name] = values
    return result


def _image_confusions_for_levels(
    stage_a: Mapping[str, Any],
    geometry: Mapping[str, Any],
    levels: np.ndarray,
) -> tuple[np.ndarray, ...]:
    cells = stage_a["cells"]
    class_count = len(stage_a["class_names"])
    values = []
    for image_index, cell_indices_raw in enumerate(geometry["cells_by_image"]):
        cell_indices = np.asarray(cell_indices_raw, dtype=np.int64)
        image_k1 = _confusion(
            stage_a["images"][image_index]["confusion"]["k1"],
            name=f"images[{image_index}].confusion.k1",
            num_classes=class_count,
        )
        image_cells = [cells[int(index)] for index in cell_indices]
        values.append(
            confusion_for_levels(image_cells, levels[cell_indices], image_k1)
        )
    return tuple(values)


def _pooled_miou(
    confusions: Sequence[np.ndarray], image_indices: Sequence[int]
) -> float:
    indices = tuple(int(value) for value in image_indices)
    if not indices:
        raise ValueError("at least one image is required for pooled mIoU")
    pooled = np.zeros_like(confusions[indices[0]], dtype=np.int64)
    for image_index in indices:
        pooled += confusions[image_index]
    return mean_iou_from_confusion(pooled)


def _subset_physical_cost(
    cost: Mapping[str, Any], image_indices: Sequence[int]
) -> float:
    per_image = cost.get("per_image")
    if not isinstance(per_image, Sequence):
        raise TypeError("physical cost summary lacks per_image records")
    baseline = 0
    processed = 0
    for image_index in image_indices:
        record = per_image[int(image_index)]
        baseline += int(record["baseline_crop_samples"])
        processed += int(record["processed_x8_crop_samples_including_padding"])
    if baseline <= 0:
        raise ValueError("physical cost baseline must be positive")
    return float((baseline + processed) / baseline)


def _configuration_key(
    *, train_miou: float, train_cost: float, score_order: int, q: float
) -> tuple[float, float, int, float]:
    """Lower tuple is better; exact ties follow the frozen protocol."""

    return (-float(train_miou), float(train_cost), int(score_order), float(q))


def cross_fit_policy(
    stage_a: Mapping[str, Any],
    geometry: Mapping[str, Any],
    score_arrays: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Select one frozen score/q on 19 images and apply it to the held-out image."""

    image_count = len(stage_a["images"])
    all_image_indices = tuple(range(image_count))
    candidates = []
    for score_order, (score_name, _) in enumerate(SCORE_SPECS):
        score_values = score_arrays[score_name]
        for q in Q_VALUES:
            policy = score_ranked_k2_levels(score_values, q, geometry)
            levels = np.asarray(policy["levels_by_cell"], dtype=np.int64)
            closure = policy.get("closure") or summarize_closure(
                levels, geometry, include_crop_ids=True
            )
            cost = physical_cost_summary(
                levels, geometry, batch_size=batch_size
            )
            candidates.append(
                {
                    "score_name": score_name,
                    "score_order": score_order,
                    "q": float(q),
                    "levels_by_cell": levels,
                    "selection": policy,
                    "closure": closure,
                    "cost": cost,
                    "image_confusions": _image_confusions_for_levels(
                        stage_a, geometry, levels
                    ),
                }
            )

    final_levels = np.ones(len(stage_a["cells"]), dtype=np.int64)
    fold_records = []
    chosen_configurations = Counter()
    for heldout_index in all_image_indices:
        train_indices = tuple(
            value for value in all_image_indices if value != heldout_index
        )
        ranked = []
        for candidate in candidates:
            per_image = candidate["cost"]["per_image"]
            if any(
                float(per_image[index]["physical_model_sample_cost_ratio"])
                > MAX_PHYSICAL_COST + 1e-12
                for index in train_indices
            ):
                continue
            train_cost = _subset_physical_cost(candidate["cost"], train_indices)
            if train_cost > MAX_PHYSICAL_COST + 1e-12:
                continue
            train_miou = _pooled_miou(
                candidate["image_confusions"], train_indices
            )
            ranked.append(
                (
                    _configuration_key(
                        train_miou=train_miou,
                        train_cost=train_cost,
                        score_order=candidate["score_order"],
                        q=candidate["q"],
                    ),
                    candidate,
                    train_miou,
                    train_cost,
                )
            )
        if not ranked:
            raise AssertionError("no frozen score/q candidate satisfies the cost cap")
        _, selected, train_miou, train_cost = min(ranked, key=lambda item: item[0])
        cell_indices = np.asarray(
            geometry["cells_by_image"][heldout_index], dtype=np.int64
        )
        # This is the leakage boundary: held-out actions are copied before the
        # held-out confusion is referenced by this fold's evaluation record.
        final_levels[cell_indices] = selected["levels_by_cell"][cell_indices]
        config_key = f"{selected['score_name']}@{selected['q']:.12g}"
        chosen_configurations[config_key] += 1
        heldout_confusion = selected["image_confusions"][heldout_index]
        heldout_metrics = metric_summary(
            heldout_confusion, stage_a["class_names"]
        )
        fold_records.append(
            {
                "heldout_image_index": heldout_index,
                "heldout_sample_name": stage_a["images"][heldout_index][
                    "sample_name"
                ],
                "selected_score": selected["score_name"],
                "selected_q": selected["q"],
                "fit_images": list(train_indices),
                "fit_pooled_miou": train_miou,
                "fit_physical_cost_ratio": train_cost,
                "heldout_selected_counts": selected["closure"]["per_image"]
                [heldout_index]["selected_counts"],
                "heldout_logical_unique_x8_crops": selected["cost"]["per_image"]
                [heldout_index]["selected_unique_x8_crop_samples"],
                "heldout_processed_x8_crops": selected["cost"]["per_image"]
                [heldout_index]["processed_x8_crop_samples_including_padding"],
                "heldout_physical_cost_ratio": selected["cost"]["per_image"]
                [heldout_index]["physical_model_sample_cost_ratio"],
                "heldout_full_image": heldout_metrics,
            }
        )

    final_confusions = _image_confusions_for_levels(stage_a, geometry, final_levels)
    pooled = np.zeros_like(final_confusions[0])
    for value in final_confusions:
        pooled += value
    closure = summarize_closure(final_levels, geometry, include_crop_ids=True)
    cost = physical_cost_summary(final_levels, geometry, batch_size=batch_size)
    return {
        "levels_by_cell": final_levels,
        "closure": closure,
        "cost": cost,
        "full_confusion": pooled,
        "image_confusions": final_confusions,
        "folds": fold_records,
        "chosen_configuration_counts": dict(sorted(chosen_configurations.items())),
        "candidate_family": candidates,
    }


def exact_cost_random_control(
    stage_a: Mapping[str, Any],
    geometry: Mapping[str, Any],
    *,
    target_unique_x8_by_image: Sequence[int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = np.empty(replicates, dtype=np.float64)
    attempts_by_replicate = []
    matched_physical_cost = None
    for replicate in range(replicates):
        walk = random_exact_cost_k2_levels(
            geometry,
            target_unique_x8_by_image,
            seed=seed,
            replicate=replicate,
        )
        levels = np.asarray(walk["levels_by_cell"], dtype=np.int64)
        if replicate == 0:
            matched_physical_cost = physical_cost_summary(
                levels, geometry, batch_size=INFERENCE_BATCH_SIZE
            )
            for image_index, target in enumerate(target_unique_x8_by_image):
                record = matched_physical_cost["per_image"][image_index]
                expected_processed = (
                    0
                    if int(target) == 0
                    else math.ceil(int(target) / INFERENCE_BATCH_SIZE)
                    * INFERENCE_BATCH_SIZE
                )
                if (
                    int(record["selected_unique_x8_crop_samples"]) != int(target)
                    or int(record["processed_x8_crop_samples_including_padding"])
                    != expected_processed
                ):
                    raise AssertionError(
                        "random control does not physically match the target cost"
                    )
        confusion = confusion_for_levels(
            stage_a["cells"],
            levels,
            stage_a["aggregate"]["endpoints"]["k1"]["full_image"][
                "confusion"
            ],
        )
        values[replicate] = mean_iou_from_confusion(confusion)
        attempts_by_replicate.append(walk["attempts_by_image"])
    return {
        "role": (
            "GT-free random K1->K2 priorities matched to every image's realized "
            "cross-fitted x8 closure cost"
        ),
        "replicates": int(replicates),
        "seed": int(seed),
        "quantile_method": "numpy default linear percentile",
        "target_unique_x8_by_image": [
            int(value) for value in target_unique_x8_by_image
        ],
        "construction_attempts_by_replicate_and_image": attempts_by_replicate,
        "matched_physical_cost": matched_physical_cost,
        "replicate_values": values,
        "miou": {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        },
    }


def _descriptive_candidate_family(
    candidates: Sequence[Mapping[str, Any]], class_names: Sequence[str]
) -> list[dict[str, Any]]:
    result = []
    for candidate in candidates:
        pooled = np.zeros_like(candidate["image_confusions"][0])
        for confusion in candidate["image_confusions"]:
            pooled += confusion
        result.append(
            {
                "score_name": candidate["score_name"],
                "q": candidate["q"],
                "role": "full-test descriptive only; never used by the cross-fitted gate",
                "full_image": metric_summary(pooled, class_names),
                "cost": candidate["cost"],
                "selection": {
                    key: value
                    for key, value in candidate["selection"].items()
                    if key != "closure"
                },
            }
        )
    return result


def screen_gate(
    *,
    candidate_miou: float,
    k1_miou: float,
    matched_k2_miou: float,
    k4_miou: float,
    cost: Mapping[str, Any],
    random_control: Mapping[str, Any],
    formal_random_control: bool,
) -> dict[str, Any]:
    denominator = k4_miou - k1_miou
    if denominator <= 0:
        raise ValueError("K4 must improve over K1 for the frozen retention gate")
    retention = (candidate_miou - k1_miou) / denominator
    candidate_minus_k2 = candidate_miou - matched_k2_miou
    random_p95 = float(random_control["miou"]["p95"])
    checks = {
        "overall_physical_cost_at_most_2x": (
            float(cost["physical_model_sample_cost_ratio"])
            <= MAX_PHYSICAL_COST + 1e-12
        ),
        "every_image_physical_cost_at_most_2x": all(
            float(record["physical_model_sample_cost_ratio"])
            <= MAX_PHYSICAL_COST + 1e-12
            for record in cost["per_image"]
        ),
        "gain_over_matched_k2_at_least_0_05pp": (
            candidate_minus_k2 >= MIN_GAIN_OVER_K2 - 1e-12
        ),
        "retains_at_least_70pct_of_k4_gain": (
            retention >= MIN_K4_GAIN_RETENTION - 1e-12
        ),
        "strictly_above_matched_exact_cost_random_p95": (
            candidate_miou > random_p95
        ),
        "small_error_not_worse_than_matched_k2": None,
        "thin_error_not_worse_than_matched_k2": None,
    }
    known_checks = [value for value in checks.values() if value is not None]
    known_passed = bool(all(known_checks))
    evaluated = bool(formal_random_control)
    if not evaluated:
        outcome = "NOT_EVALUATED_NONFORMAL_RANDOM_CONTROL"
    elif known_passed:
        outcome = "PROVISIONAL_GO_ONE_LIVE_CONFIRMATION_KNOWN_GATES_PASS"
    else:
        outcome = "STOP_SIMPLE_K1_SCORE_SCREEN_KNOWN_GATE_FAILED"
    return {
        "outcome": outcome,
        "scientific_decision_evaluated": evaluated,
        "known_checks_passed": known_passed if evaluated else None,
        "complete": False,
        "one_live_confirmation_authorized": bool(evaluated and known_passed),
        "h3_confirmed": False,
        "checks": checks,
        "observed": {
            "candidate_miou": candidate_miou,
            "candidate_miou_percent": candidate_miou * 100.0,
            "candidate_minus_k1_pp": (candidate_miou - k1_miou) * 100.0,
            "candidate_minus_matched_k2_pp": candidate_minus_k2 * 100.0,
            "candidate_minus_k4_pp": (candidate_miou - k4_miou) * 100.0,
            "k4_gain_retention": retention,
            "overall_physical_cost": cost["physical_model_sample_cost_ratio"],
            "maximum_per_image_physical_cost": max(
                float(record["physical_model_sample_cost_ratio"])
                for record in cost["per_image"]
            ),
            "random_p95_miou": random_p95,
            "candidate_minus_random_p95_pp": (
                candidate_miou - random_p95
            )
            * 100.0,
        },
        "thresholds": {
            "maximum_overall_and_per_image_physical_cost": MAX_PHYSICAL_COST,
            "minimum_gain_over_matched_k2_miou_fraction": MIN_GAIN_OVER_K2,
            "minimum_gain_over_matched_k2_pp": MIN_GAIN_OVER_K2 * 100.0,
            "minimum_k4_gain_retention": MIN_K4_GAIN_RETENTION,
            "random": "strictly above matched exact-realized-cost random p95",
            "structure": (
                "aggregate small/thin error rate no worse than matched K2; "
                "unavailable in compact Stage-A cells"
            ),
        },
        "interpretation": (
            "The nonformal random control cannot make a scientific decision."
            if not evaluated
            else (
                "Known-gate PASS can authorize one frozen live small/thin and "
                "same-primitive latency confirmation only. It does not establish H3."
                if known_passed
                else "A known-gate failure stops the three simple K1 scores without "
                "expanding q or combining scores."
            )
        ),
    }


def next_step_for_decision(decision: Mapping[str, Any]) -> str:
    if decision.get("scientific_decision_evaluated") is not True:
        return (
            "Smoke only: rerun the frozen 1,000-replicate random control before "
            "making a scientific or stop/go decision."
        )
    if decision.get("one_live_confirmation_authorized") is True:
        return "Run one frozen live small/thin and same-primitive latency confirmation."
    return "Stop the three simple K1 scores; do not expand q or combine them."


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    stage_a_sha = file_sha256(args.stage_a_json)
    stage_b0_sha = file_sha256(args.stage_b0_json)
    latency_sha = file_sha256(args.latency_json)
    stage_a = load_stage_a(args.stage_a_json)
    stage_b0 = load_stage_b0(args.stage_b0_json, stage_a_sha256=stage_a_sha)
    latency = load_latency_confirmation(
        args.latency_json,
        stage_a_sha256=stage_a_sha,
        stage_b0_sha256=stage_b0_sha,
    )
    if len(stage_b0["exact_cost_a2_oracle"]["levels_by_cell"]) != len(
        stage_a["cells"]
    ):
        raise ValueError("Stage-B0 oracle level count differs from Stage A")

    geometry = build_phase_closure_geometry(stage_a["images"], stage_a["cells"])
    eligible = np.asarray(geometry["eligible"], dtype=bool)
    scores = _score_arrays(stage_a["cells"], eligible)
    crossfit_started = time.perf_counter()
    crossfit = cross_fit_policy(
        stage_a,
        geometry,
        scores,
        batch_size=INFERENCE_BATCH_SIZE,
    )
    crossfit_seconds = time.perf_counter() - crossfit_started

    class_names = stage_a["class_names"]
    endpoints = stage_a["aggregate"]["endpoints"]
    k1_confusion = _confusion(
        endpoints["k1"]["full_image"]["confusion"],
        name="aggregate K1",
        num_classes=len(class_names),
    )
    k2_confusion = _confusion(
        endpoints["matched_k2"]["full_image"]["confusion"],
        name="aggregate matched K2",
        num_classes=len(class_names),
    )
    k4_confusion = _confusion(
        endpoints["k4"]["full_image"]["confusion"],
        name="aggregate K4",
        num_classes=len(class_names),
    )
    all_k1_levels = np.ones(len(stage_a["cells"]), dtype=np.int64)
    all_k2_levels = all_k1_levels.copy()
    all_k2_levels[eligible] = 2
    reconstructed_k1 = confusion_for_levels(
        stage_a["cells"], all_k1_levels, k1_confusion
    )
    reconstructed_k2 = confusion_for_levels(
        stage_a["cells"], all_k2_levels, k1_confusion
    )
    reconstructed_candidate = confusion_for_levels(
        stage_a["cells"], crossfit["levels_by_cell"], k1_confusion
    )
    if not np.array_equal(reconstructed_k1, k1_confusion):
        raise AssertionError("all-K1 reconstruction differs from sealed K1")
    if not np.array_equal(reconstructed_k2, k2_confusion):
        raise AssertionError("all-eligible-K2 reconstruction differs from matched K2")
    if not np.array_equal(reconstructed_candidate, crossfit["full_confusion"]):
        raise AssertionError(
            "cross-fitted per-image confusion differs from aggregate reconstruction"
        )

    target_costs = [
        int(record["selected_unique_x8_crop_samples"])
        for record in crossfit["cost"]["per_image"]
    ]
    random_started = time.perf_counter()
    random_control = exact_cost_random_control(
        stage_a,
        geometry,
        target_unique_x8_by_image=target_costs,
        replicates=args.random_replicates,
        seed=args.random_seed,
    )
    random_seconds = time.perf_counter() - random_started
    formal_random = bool(
        args.random_replicates == RANDOM_REPLICATES
        and args.random_seed == RANDOM_SEED
    )

    candidate_metrics = metric_summary(
        crossfit["full_confusion"], class_names
    )
    k1_metrics = metric_summary(k1_confusion, class_names)
    k2_metrics = metric_summary(k2_confusion, class_names)
    k4_metrics = metric_summary(k4_confusion, class_names)
    decision = screen_gate(
        candidate_miou=candidate_metrics["miou"],
        k1_miou=k1_metrics["miou"],
        matched_k2_miou=k2_metrics["miou"],
        k4_miou=k4_metrics["miou"],
        cost=crossfit["cost"],
        random_control=random_control,
        formal_random_control=formal_random,
    )

    output = {
        "status": "PASS",
        "status_meaning": (
            "Input, provenance, whole-image cross-fit, closure, physical-cost, "
            "matched-random, and serialization checks completed. Scientific "
            "outcome is in h3_screen_decision and is not implied by status."
        ),
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "official-test exploratory whole-image leave-one-out",
        "scientific_scope": (
            "first H3 screen of K1-visible scores for K1->fixed-x8-K2 only; "
            "no K2->K4 decision, model forward, training, or paper-ready validation"
        ),
        "source_artifacts": {
            "stage_a": {
                "path": str(args.stage_a_json.resolve()),
                "sha256": stage_a_sha,
            },
            "stage_b0": {
                "path": str(args.stage_b0_json.resolve()),
                "sha256": stage_b0_sha,
            },
            "full_stage_b_latency": {
                "path": str(args.latency_json.resolve()),
                "sha256": latency_sha,
                "outcome": latency["latency_decision"]["outcome"],
            },
        },
        "protocol": {
            "route": "K1 -> fixed x8 K2 only",
            "score_order_and_direction": [
                {"name": name, "cell_key": key, "higher_is_promoted_first": True}
                for name, key in SCORE_SPECS
            ],
            "q_values": list(Q_VALUES),
            "q_denominator": "geometry-eligible cells independently per image",
            "selection": (
                "top floor(q*eligible) by descending score, lower global cell "
                "index tie-break, then exhaustive zero-cost x8-closure spill"
            ),
            "cross_fit": (
                "20-fold whole-image leave-one-out; choose on 19-image pooled mIoU, "
                "then lower physical cost, fixed score order, lower q"
            ),
            "physical_cost": (
                "normal K1 crops plus per-image x8 unique closure padded once to "
                "fixed batch=8; padding is counted as model compute"
            ),
            "random_control": (
                "GT-free random priorities independently match each held-out "
                "policy image's realized unique x8 closure cost"
            ),
            "random_replicates": args.random_replicates,
            "random_seed": args.random_seed,
            "formal_random_control": formal_random,
        },
        "provenance": {
            "feature_uses_ground_truth": False,
            "fit_uses_ground_truth": True,
            "fit_ground_truth_scope": "other 19 images only",
            "heldout_action_uses_ground_truth": False,
            "heldout_action_selection_excludes_heldout_ground_truth": True,
            "heldout_evaluation_occurs_after_fold_action_freeze": True,
            "forbidden_as_features": [
                "labels or valid-label mask",
                "cell confusion or error masks",
                "B0 oracle levels/action trace/utility",
                "K2 or K4 outputs",
                "image/sample identity",
            ],
        },
        "geometry": {
            "images": len(stage_a["images"]),
            "cells": len(stage_a["cells"]),
            "eligible_cells": int(np.count_nonzero(eligible)),
            "baseline_k1_crop_forwards": int(
                np.asarray(geometry["baseline_crop_counts"]).sum()
            ),
        },
        "endpoint_validation": {
            "all_k1_reconstructs_sealed_k1": True,
            "all_eligible_k2_reconstructs_matched_k2": True,
            "cross_fitted_per_image_confusion_matches_aggregate_reconstruction": True,
            "k1": k1_metrics,
            "matched_k2": k2_metrics,
            "k4": k4_metrics,
        },
        "descriptive_fixed_configuration_family": _descriptive_candidate_family(
            crossfit["candidate_family"], class_names
        ),
        "cross_fitted_policy": {
            "levels_by_cell": crossfit["levels_by_cell"],
            "chosen_configuration_counts": crossfit[
                "chosen_configuration_counts"
            ],
            "folds": crossfit["folds"],
            "closure": crossfit["closure"],
            "cost": crossfit["cost"],
            "full_image": candidate_metrics,
        },
        "matched_exact_cost_random_control": random_control,
        "h3_screen_decision": decision,
        "explicit_non_claims": [
            "H3 is confirmed",
            "a deployable router exists",
            "K2->K4 is predictable",
            "small/thin structure is closed for this policy",
            "latency including this policy has been measured",
            "official-test exploratory selection is paper-ready validation",
        ],
        "next_step": next_step_for_decision(decision),
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "common_module_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_h3_screen_common.py"
            ),
            "numpy": np.__version__,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
        },
        "runtime": {
            "cross_fit_seconds": crossfit_seconds,
            "random_control_seconds": random_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    atomic_write_json(args.output_path, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "outcome": decision["outcome"],
                "candidate_miou_percent": candidate_metrics["miou_percent"],
                "candidate_minus_matched_k2_pp": decision["observed"][
                    "candidate_minus_matched_k2_pp"
                ],
                "k4_gain_retention": decision["observed"][
                    "k4_gain_retention"
                ],
                "random_p95_miou_percent": decision["observed"][
                    "random_p95_miou"
                ]
                * 100.0,
                "physical_cost": decision["observed"][
                    "overall_physical_cost"
                ],
                "output_path": str(args.output_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
