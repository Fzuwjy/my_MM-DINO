"""Frozen offline H4-B K2-first response-arbitration screen.

The x8 phase has already been acquired uniformly, so every policy has the same
1.727273x physical inference cost.  A policy only chooses whether each eligible
ownership cell emits K1 or matched K2.  The primary signed entropy response is
the sole formal score; margin is auxiliary and disagreement features are
report-only diagnostics.
"""

from __future__ import annotations

import argparse
import json
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
from scripts.evaluate_whu_phase_h3_screen import (  # noqa: E402
    INFERENCE_BATCH_SIZE,
    MAX_PHYSICAL_COST,
    MIN_GAIN_OVER_K2,
    MIN_K4_GAIN_RETENTION,
    _confusion,
    _image_confusions_for_levels,
    _pooled_miou,
    load_latency_confirmation,
    load_stage_b0,
)
from scripts.phase_closure_common import (  # noqa: E402
    build_phase_closure_geometry,
    confusion_for_levels,
    mean_iou_from_confusion,
)
from scripts.phase_h3_screen_common import physical_cost_summary  # noqa: E402
from scripts.phase_overlap_common import RESPONSE_SCORE_NAMES  # noqa: E402


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "whu_h4b_k2_response_arbitration_screen"
EXPECTED_STATISTICS_TYPE = "whu_h4ab_k1_x8_response_statistics"
PRIMARY_SPEC = ("entropy_gain", RESPONSE_SCORE_NAMES[0])
AUXILIARY_SPEC = ("margin_gain", RESPONSE_SCORE_NAMES[1])
REPORT_ONLY_KEYS = RESPONSE_SCORE_NAMES[2:]
Q_VALUES = (0.10, 0.20, 1.0 / 3.0, 0.50, 2.0 / 3.0)
RANDOM_REPLICATES = 1000
RANDOM_SEED = 20260803


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline whole-image H4-B K1/K2 response arbitration"
    )
    parser.add_argument("--stage-a-json", type=Path, required=True)
    parser.add_argument("--stage-b0-json", type=Path, required=True)
    parser.add_argument("--latency-json", type=Path, required=True)
    parser.add_argument("--response-statistics-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--random-replicates", type=int, default=RANDOM_REPLICATES
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)
    for name in (
        "stage_a_json",
        "stage_b0_json",
        "latency_json",
        "response_statistics_json",
    ):
        if not getattr(args, name).is_file():
            parser.error(f"input artifact does not exist: {getattr(args, name)}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.random_replicates <= 0 or args.random_seed < 0:
        parser.error("random replicates must be positive and seed non-negative")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
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


def _join_response_scores(
    stage_a: Mapping[str, Any],
    statistics: Mapping[str, Any],
    *,
    stage_a_sha: str,
) -> dict[str, Any]:
    checks = {
        "artifact_type": statistics.get("artifact_type") == EXPECTED_STATISTICS_TYPE,
        "schema_version": statistics.get("schema_version") == 1,
        "status": statistics.get("status") == "PASS",
        "scope": statistics.get("scope") == "full-test",
        "mode": statistics.get("execution_mode") == "live-k1-x8",
        "complete": statistics.get("evaluated_images")
        == statistics.get("full_test_length")
        == len(stage_a["images"]),
        "source": statistics.get("source_stage_a", {}).get("sha256")
        == stage_a_sha,
        "checkpoint": statistics.get("source_execution", {})
        .get("baseline_checkpoint", {})
        .get("sha256")
        == stage_a.get("baseline_checkpoint_sha256"),
        "k1_once": statistics.get("protocol", {}).get("normal_execution_count")
        == "exactly once per image",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"H4-B response-statistics binding failed: {failed}")
    statistic_cells = statistics.get("cells")
    if not isinstance(statistic_cells, list) or len(statistic_cells) != len(
        stage_a["cells"]
    ):
        raise ValueError("H4-B response cell count differs from Stage A")

    joined = dict(stage_a)
    joined_cells = []
    for index, (base, response) in enumerate(
        zip(stage_a["cells"], statistic_cells, strict=True)
    ):
        for key in ("cell_index", "image_index", "local_crop_id", "geometry_eligible"):
            expected = index if key == "cell_index" else base.get(key)
            if response.get(key) != expected:
                raise ValueError(f"H4-B cell {index} identity differs at {key}")
        raw_scores = response.get("k2_response_scores")
        if not isinstance(raw_scores, Mapping):
            raise ValueError(f"H4-B cell {index} lacks response scores")
        copied = dict(base)
        copied_scores = dict(base.get("scores", {}))
        for key in RESPONSE_SCORE_NAMES:
            raw = raw_scores.get(key)
            if bool(base["geometry_eligible"]):
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise ValueError(f"eligible H4-B cell {index} lacks {key}")
                value = float(raw)
                if not np.isfinite(value):
                    raise ValueError(f"eligible H4-B cell {index} has non-finite {key}")
                copied_scores[key] = value
            else:
                copied_scores[key] = None
        copied["scores"] = copied_scores
        joined_cells.append(copied)
    joined["cells"] = joined_cells
    return joined


def _score_vector(
    cells: Sequence[Mapping[str, Any]], eligible: np.ndarray, key: str
) -> np.ndarray:
    values = np.full(len(cells), -np.inf, dtype=np.float64)
    for index, cell in enumerate(cells):
        raw = cell.get("scores", {}).get(key)
        if eligible[index]:
            if raw is None or not np.isfinite(float(raw)):
                raise ValueError(f"eligible cell lacks finite score {key}")
            values[index] = float(raw)
        elif raw is not None:
            raise ValueError(f"ineligible cell unexpectedly exposes {key}")
    return values


def _validate_combined_collection_ledger(
    statistics: Mapping[str, Any], *, image_count: int
) -> None:
    """Bind the formal screen to the frozen full-WHU acquisition ledger."""

    collection_cost = statistics.get("collection_cost")
    if not isinstance(collection_cost, Mapping):
        raise ValueError("H4-B combined collection lacks its cost ledger")
    expected_collection = {
        "baseline_normal_real_and_model_samples": 3520,
        "x8_selected_real_crops": 2520,
        "x8_processed_model_samples_including_padding": 2560,
    }
    failed = [
        name
        for name, expected in expected_collection.items()
        if collection_cost.get(name) != expected
    ]
    raw_ratio = collection_cost.get("physical_model_sample_cost_ratio")
    if (
        isinstance(raw_ratio, bool)
        or not isinstance(raw_ratio, (int, float))
        or abs(float(raw_ratio) - 6080 / 3520) > 1e-12
    ):
        failed.append("physical_model_sample_cost_ratio")

    image_validations = statistics.get("images")
    if not isinstance(image_validations, list) or len(image_validations) != image_count:
        failed.append("per_image_endpoint_validation_count")
    else:
        for image_index, record in enumerate(image_validations):
            endpoint = record.get("endpoint_validation")
            matched = record.get("matched_k2_endpoint_validation")
            if not (
                record.get("image_index") == image_index
                and isinstance(endpoint, Mapping)
                and endpoint.get("prediction_equal") is True
                and endpoint.get("stage_a_confusion_equal") is True
                and isinstance(matched, Mapping)
                and matched.get("prediction_equal") is True
                and matched.get("stage_a_confusion_equal") is True
            ):
                failed.append(f"image_{image_index}_endpoint_validation")
    if failed:
        raise ValueError(f"H4-B combined collection validation failed: {failed}")


def response_ranked_levels(
    scores: np.ndarray,
    q: float,
    geometry: Mapping[str, Any],
) -> np.ndarray:
    """Choose a fixed fraction of each image's eligible cells to emit K2."""

    values = np.asarray(scores, dtype=np.float64)
    eligible = np.asarray(geometry["eligible"], dtype=np.bool_)
    if values.shape != eligible.shape or not 0.0 <= float(q) <= 1.0:
        raise ValueError("response scores/q are invalid")
    levels = np.ones(len(values), dtype=np.int64)
    for raw_indices in geometry["cells_by_image"]:
        indices = np.asarray(raw_indices, dtype=np.int64)
        candidates = indices[eligible[indices]]
        count = int(np.floor(float(q) * len(candidates)))
        ranked = sorted(candidates.tolist(), key=lambda index: (-values[index], index))
        if count:
            levels[np.asarray(ranked[:count], dtype=np.int64)] = 2
    return levels


def cross_fit_response_policy(
    stage_a: Mapping[str, Any],
    geometry: Mapping[str, Any],
    scores: np.ndarray,
) -> dict[str, Any]:
    candidates = []
    for q in Q_VALUES:
        levels = response_ranked_levels(scores, q, geometry)
        candidates.append(
            {
                "q": float(q),
                "levels": levels,
                "image_confusions": _image_confusions_for_levels(
                    stage_a, geometry, levels
                ),
            }
        )
    image_count = len(stage_a["images"])
    final_levels = np.ones(len(stage_a["cells"]), dtype=np.int64)
    folds = []
    chosen = Counter()
    for heldout in range(image_count):
        fit = tuple(index for index in range(image_count) if index != heldout)
        ranked = []
        for candidate in candidates:
            fit_miou = _pooled_miou(candidate["image_confusions"], fit)
            # No q-dependent compute saving remains after uniform x8 acquisition.
            # Exact mIoU ties therefore prefer the higher-K2 policy.
            ranked.append(((-fit_miou, -candidate["q"]), candidate, fit_miou))
        _, selected, fit_miou = min(ranked, key=lambda item: item[0])
        indices = np.asarray(geometry["cells_by_image"][heldout], dtype=np.int64)
        final_levels[indices] = selected["levels"][indices]
        chosen[f"q={selected['q']:.12g}"] += 1
        folds.append(
            {
                "heldout_image_index": heldout,
                "heldout_sample_name": stage_a["images"][heldout]["sample_name"],
                "fit_images": list(fit),
                "selected_q": selected["q"],
                "fit_pooled_miou": fit_miou,
                "heldout_k2_output_cells": int(
                    np.count_nonzero(final_levels[indices] == 2)
                ),
                "heldout_eligible_cells": int(
                    np.count_nonzero(np.asarray(geometry["eligible"])[indices])
                ),
            }
        )
    image_confusions = _image_confusions_for_levels(stage_a, geometry, final_levels)
    pooled = np.zeros_like(image_confusions[0])
    for confusion in image_confusions:
        pooled += confusion
    return {
        "levels_by_cell": final_levels,
        "image_confusions": image_confusions,
        "full_confusion": pooled,
        "folds": folds,
        "chosen_q_counts": dict(sorted(chosen.items())),
        "fixed_q_candidates": candidates,
    }


def matched_action_count_random(
    stage_a: Mapping[str, Any],
    geometry: Mapping[str, Any],
    target_levels: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    eligible = np.asarray(geometry["eligible"], dtype=np.bool_)
    target = np.asarray(target_levels, dtype=np.int64)
    target_counts = []
    for raw_indices in geometry["cells_by_image"]:
        indices = np.asarray(raw_indices, dtype=np.int64)
        target_counts.append(int(np.count_nonzero(target[indices] == 2)))
    values = np.empty(replicates, dtype=np.float64)
    first_levels = None
    for replicate in range(replicates):
        levels = np.ones(len(target), dtype=np.int64)
        for image_index, raw_indices in enumerate(geometry["cells_by_image"]):
            indices = np.asarray(raw_indices, dtype=np.int64)
            candidates = indices[eligible[indices]]
            count = target_counts[image_index]
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), int(replicate), int(image_index)])
            )
            selected = rng.choice(candidates, size=count, replace=False)
            levels[selected] = 2
            if int(np.count_nonzero(levels[indices] == 2)) != count:
                raise AssertionError("random K2 output count differs from target")
        confusion = confusion_for_levels(
            stage_a["cells"],
            levels,
            stage_a["aggregate"]["endpoints"]["k1"]["full_image"]["confusion"],
        )
        values[replicate] = mean_iou_from_confusion(confusion)
        if replicate == 0:
            first_levels = levels.tolist()
    return {
        "role": (
            "matched-action-count random: x8 compute is already uniform/fixed, so "
            "each image matches only the candidate's K2 output-cell count"
        ),
        "replicates": replicates,
        "seed": seed,
        "target_k2_output_cells_by_image": target_counts,
        "first_replicate_levels_by_cell": first_levels,
        "replicate_values": values,
        "miou": {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        },
    }


def _candidate_summary(
    policy: Mapping[str, Any], class_names: Sequence[str]
) -> list[dict[str, Any]]:
    result = []
    for candidate in policy["fixed_q_candidates"]:
        pooled = np.zeros_like(candidate["image_confusions"][0])
        for confusion in candidate["image_confusions"]:
            pooled += confusion
        result.append(
            {
                "q": candidate["q"],
                "role": "full-test descriptive only; not used on held-out actions",
                "full_image": metric_summary(pooled, class_names),
            }
        )
    return result


def _decision(
    *,
    candidate_miou: float,
    endpoints: Mapping[str, Mapping[str, Any]],
    physical_cost: Mapping[str, Any],
    random_control: Mapping[str, Any],
    formal: bool,
) -> dict[str, Any]:
    k1 = float(endpoints["k1"]["miou"])
    k2 = float(endpoints["matched_k2"]["miou"])
    k4 = float(endpoints["k4"]["miou"])
    if k4 <= k1:
        raise ValueError("H4-B gain retention requires K4 to improve K1")
    retention = (candidate_miou - k1) / (k4 - k1)
    random_p95 = float(random_control["miou"]["p95"])
    checks = {
        "fixed_uniform_k2_physical_cost_at_most_2x": float(
            physical_cost["physical_model_sample_cost_ratio"]
        )
        <= MAX_PHYSICAL_COST + 1e-12,
        "gain_over_uniform_matched_k2_at_least_0_05pp": candidate_miou - k2
        >= MIN_GAIN_OVER_K2 - 1e-12,
        "retains_at_least_70pct_of_k4_gain": retention
        >= MIN_K4_GAIN_RETENTION - 1e-12,
        "strictly_above_matched_action_count_random_p95": candidate_miou
        > random_p95,
        "small_error_not_worse_than_matched_k2": None,
        "thin_error_not_worse_than_matched_k2": None,
    }
    known_passed = bool(all(value for value in checks.values() if value is not None))
    mechanism_checks = {
        "strictly_improves_uniform_matched_k2": candidate_miou > k2,
        "strictly_above_matched_action_count_random_p95": candidate_miou
        > random_p95,
    }
    mechanism_passed = bool(all(mechanism_checks.values()))
    if not formal:
        outcome = "NOT_EVALUATED_NONFORMAL_RANDOM_CONTROL"
    elif known_passed:
        outcome = "PROVISIONAL_GO_H4B_LIVE_STRUCTURE_CONFIRMATION"
    elif mechanism_passed:
        outcome = "GO_H4C_RESPONSE_SIGNAL_ONLY"
    else:
        outcome = "STOP_H4B_SIMPLE_K2_RESPONSE_RULES"
    return {
        "outcome": outcome,
        "scientific_decision_evaluated": formal,
        "known_checks_passed": known_passed if formal else None,
        "mechanism_signal_passed": mechanism_passed if formal else None,
        "live_structure_confirmation_authorized": bool(formal and known_passed),
        "h4c_implementation_authorized": bool(
            formal and mechanism_passed and not known_passed
        ),
        "h4c_after_live_structure_confirmation_eligible": bool(
            formal and mechanism_passed and known_passed
        ),
        "checks": checks,
        "mechanism_checks": mechanism_checks,
        "observed": {
            "candidate_miou": candidate_miou,
            "candidate_miou_percent": candidate_miou * 100.0,
            "candidate_minus_k1_pp": (candidate_miou - k1) * 100.0,
            "candidate_minus_matched_k2_pp": (candidate_miou - k2) * 100.0,
            "candidate_minus_k4_pp": (candidate_miou - k4) * 100.0,
            "k4_gain_retention": retention,
            "physical_model_sample_cost_ratio": physical_cost[
                "physical_model_sample_cost_ratio"
            ],
            "random_p95_miou": random_p95,
            "candidate_minus_random_p95_pp": (candidate_miou - random_p95)
            * 100.0,
        },
    }


def _endpoint_metrics(stage_a: Mapping[str, Any]) -> dict[str, Any]:
    endpoints = stage_a["aggregate"]["endpoints"]
    class_names = stage_a["class_names"]
    return {
        name: metric_summary(
            _confusion(
                endpoints[key]["full_image"]["confusion"],
                name=name,
                num_classes=len(class_names),
            ),
            class_names,
        )
        for name, key in (("k1", "k1"), ("matched_k2", "matched_k2"), ("k4", "k4"))
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    stage_a_sha = file_sha256(args.stage_a_json)
    stage_b0_sha = file_sha256(args.stage_b0_json)
    latency_sha = file_sha256(args.latency_json)
    statistics_sha = file_sha256(args.response_statistics_json)
    stage_a = load_stage_a(args.stage_a_json)
    stage_b0 = load_stage_b0(args.stage_b0_json, stage_a_sha256=stage_a_sha)
    latency = load_latency_confirmation(
        args.latency_json,
        stage_a_sha256=stage_a_sha,
        stage_b0_sha256=stage_b0_sha,
    )
    statistics = _read_json(args.response_statistics_json)
    joined = _join_response_scores(stage_a, statistics, stage_a_sha=stage_a_sha)
    geometry = build_phase_closure_geometry(joined["images"], joined["cells"])
    eligible = np.asarray(geometry["eligible"], dtype=np.bool_)
    endpoints = _endpoint_metrics(joined)

    _validate_combined_collection_ledger(
        statistics, image_count=len(joined["images"])
    )

    all_k2 = np.ones(len(joined["cells"]), dtype=np.int64)
    all_k2[eligible] = 2
    reconstructed_k2 = confusion_for_levels(
        joined["cells"],
        all_k2,
        joined["aggregate"]["endpoints"]["k1"]["full_image"]["confusion"],
    )
    expected_k2 = _confusion(
        joined["aggregate"]["endpoints"]["matched_k2"]["full_image"]["confusion"],
        name="aggregate matched K2",
        num_classes=len(joined["class_names"]),
    )
    if not np.array_equal(reconstructed_k2, expected_k2):
        raise AssertionError("H4-B cell reconstruction does not reproduce matched K2")
    physical_cost = physical_cost_summary(
        all_k2, geometry, batch_size=INFERENCE_BATCH_SIZE
    )
    if abs(float(physical_cost["physical_model_sample_cost_ratio"]) - 6080 / 3520) > 1e-12:
        raise AssertionError("uniform K2 physical cost differs from 1.727273x")

    screen_results = {}
    for role, spec in (("primary", PRIMARY_SPEC), ("auxiliary", AUXILIARY_SPEC)):
        screen_started = time.perf_counter()
        scores = _score_vector(joined["cells"], eligible, spec[1])
        policy = cross_fit_response_policy(joined, geometry, scores)
        random_control = matched_action_count_random(
            joined,
            geometry,
            policy["levels_by_cell"],
            replicates=args.random_replicates,
            seed=args.random_seed,
        )
        screen_results[role] = {
            "score_name": spec[0],
            "cell_key": spec[1],
            "role": (
                "sole formal H4-B decision score"
                if role == "primary"
                else "auxiliary only; cannot rescue a failed primary score"
            ),
            "cross_fitted_policy": {
                "levels_by_cell": policy["levels_by_cell"],
                "chosen_q_counts": policy["chosen_q_counts"],
                "folds": policy["folds"],
                "full_image": metric_summary(
                    policy["full_confusion"], joined["class_names"]
                ),
            },
            "descriptive_fixed_q_family": _candidate_summary(
                policy, joined["class_names"]
            ),
            "matched_action_count_random": random_control,
            "runtime_seconds": float(time.perf_counter() - screen_started),
        }
    primary = screen_results["primary"]
    formal = (
        args.random_replicates == RANDOM_REPLICATES
        and args.random_seed == RANDOM_SEED
    )
    decision = _decision(
        candidate_miou=float(primary["cross_fitted_policy"]["full_image"]["miou"]),
        endpoints=endpoints,
        physical_cost=physical_cost,
        random_control=primary["matched_action_count_random"],
        formal=formal,
    )

    if decision["live_structure_confirmation_authorized"]:
        next_step = "Run one frozen live small/thin confirmation for the primary policy."
    elif decision["h4c_implementation_authorized"]:
        next_step = (
            "The primary response has a real but sub-threshold signal; authorize only "
            "the frozen H4-C K2->K4/one-small-head step."
        )
    else:
        next_step = (
            "Stop H4-B simple response arbitration; auxiliary margin cannot rescue it, "
            "and H4-C is not authorized."
        )
    output = {
        "status": "PASS",
        "status_meaning": (
            "Full-test response binding, whole-image cross-fit, fixed K2 cost, "
            "matched-action random, and reconstruction checks completed. Scientific "
            "outcome is in h4b_decision."
        ),
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "official-test exploratory whole-image leave-one-out",
        "scientific_scope": (
            "one independently pre-authorized H4-B zero-training K1/K2 output "
            "arbitration screen; no K2->K4 route or utility-head training"
        ),
        "source_artifacts": {
            "stage_a": {"path": str(args.stage_a_json.resolve()), "sha256": stage_a_sha},
            "stage_b0": {"path": str(args.stage_b0_json.resolve()), "sha256": stage_b0_sha},
            "full_stage_b_latency": {
                "path": str(args.latency_json.resolve()),
                "sha256": latency_sha,
                "outcome": latency["latency_decision"]["outcome"],
            },
            "combined_response_statistics": {
                "path": str(args.response_statistics_json.resolve()),
                "sha256": statistics_sha,
            },
        },
        "protocol": {
            "information_state": "uniform x8 already acquired; choose K1 or K2 output",
            "primary": {"name": PRIMARY_SPEC[0], "cell_key": PRIMARY_SPEC[1]},
            "auxiliary": {"name": AUXILIARY_SPEC[0], "cell_key": AUXILIARY_SPEC[1]},
            "report_only": list(REPORT_ONLY_KEYS),
            "auxiliary_or_report_only_can_rescue_primary": False,
            "q_values": list(Q_VALUES),
            "q_selection": (
                "whole-image LOO on other 19-image pooled mIoU; exact ties prefer "
                "higher q because all q share the same uniform-x8 compute"
            ),
            "random_control": (
                "per-image matched K2-output-cell count, not matched cost; x8 cost "
                "is already fixed and identical for every action policy"
            ),
            "random_replicates": args.random_replicates,
            "random_seed": args.random_seed,
            "formal_random_control": formal,
        },
        "uniform_k2_physical_cost": physical_cost,
        "endpoint_validation": endpoints,
        "score_screens": screen_results,
        "h4b_decision": decision,
        "next_step": next_step,
        "explicit_non_claims": [
            "H4-B was selected because H4-A passed.",
            "Margin or disagreement may rescue a failed entropy primary.",
            "Matched-action-count random is an exact-cost random control.",
            "Small/thin structure is closed before a live confirmation.",
            "Official-test exploratory cross-fit is paper-ready validation.",
        ],
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "statistics_runner_sha256": file_sha256(
                REPO_ROOT / "scripts" / "evaluate_whu_h4ab_response_statistics.py"
            ),
            "inference_batch_size": INFERENCE_BATCH_SIZE,
            "numpy": np.__version__,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    atomic_write_json(args.output_path, output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "outcome": decision["outcome"],
                "primary_miou_percent": primary["cross_fitted_policy"]["full_image"][
                    "miou_percent"
                ],
                "primary_minus_k2_pp": decision["observed"][
                    "candidate_minus_matched_k2_pp"
                ],
                "primary_minus_random_p95_pp": decision["observed"][
                    "candidate_minus_random_p95_pp"
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(f"h4b_screen_result={args.output_path.resolve()}", flush=True)
    print("h4b_screen_status=PASS", flush=True)


if __name__ == "__main__":
    main()
