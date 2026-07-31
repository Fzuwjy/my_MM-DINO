"""Offline whole-image H4-A screen for intra-slide response disagreement.

The full-test score artifact is joined to the sealed Stage-A cells by immutable
cell indices.  Each of the three frozen response scores is screened separately;
the primary normalized overlap JSD alone controls the H4-A/H4-B decision.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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
    Q_VALUES,
    _confusion,
    _descriptive_candidate_family,
    _score_arrays,
    cross_fit_policy,
    exact_cost_random_control,
    load_latency_confirmation,
    load_stage_b0,
)
from scripts.phase_closure_common import (  # noqa: E402
    build_phase_closure_geometry,
    confusion_for_levels,
)
from scripts.phase_overlap_common import SCORE_NAMES  # noqa: E402


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "whu_h4a_k1_overlap_disagreement_screen"
EXPECTED_SCORE_TYPE = "whu_h4a_k1_overlap_disagreement"
SCORE_SPECS = (
    ("overlap_jsd", SCORE_NAMES[0]),
    ("argmax_vote_disagreement", SCORE_NAMES[1]),
    ("boundary_response_disagreement", SCORE_NAMES[2]),
)
PRIMARY_SCORE = SCORE_SPECS[0][0]
RANDOM_REPLICATES = 1000
RANDOM_SEED = 20260802
RIDGE_MIOU_REFERENCE = 0.54244027
RIDGE_DEFICIT_TO_K2 = 0.00019663


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline H4-A whole-image response-disagreement screen"
    )
    parser.add_argument("--stage-a-json", type=Path, required=True)
    parser.add_argument("--stage-b0-json", type=Path, required=True)
    parser.add_argument("--latency-json", type=Path, required=True)
    parser.add_argument("--overlap-statistics-json", type=Path, required=True)
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
        "overlap_statistics_json",
    ):
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


def _join_scores(
    stage_a: Mapping[str, Any],
    statistics: Mapping[str, Any],
    *,
    stage_a_sha: str,
) -> dict[str, Any]:
    checks = {
        "artifact_type": statistics.get("artifact_type") == EXPECTED_SCORE_TYPE,
        "schema_version": statistics.get("schema_version") == 1,
        "status": statistics.get("status") == "PASS",
        "scope": statistics.get("scope") == "full-test",
        "execution_mode": statistics.get("execution_mode") == "live-k1",
        "complete": statistics.get("evaluated_images")
        == statistics.get("full_test_length")
        == len(stage_a["images"]),
        "source_stage_a": statistics.get("source_stage_a", {}).get("sha256")
        == stage_a_sha,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"H4-A statistics binding failed: {failed}")
    score_cells = statistics.get("cells")
    if not isinstance(score_cells, list) or len(score_cells) != len(stage_a["cells"]):
        raise ValueError("H4-A score-cell count differs from Stage A")

    joined = dict(stage_a)
    joined_cells = []
    for index, (stage_cell, score_cell) in enumerate(
        zip(stage_a["cells"], score_cells, strict=True)
    ):
        identity_checks = {
            "cell_index": score_cell.get("cell_index") == index,
            "image_index": score_cell.get("image_index")
            == stage_cell.get("image_index"),
            "local_crop_id": score_cell.get("local_crop_id")
            == stage_cell.get("local_crop_id"),
            "geometry_eligible": score_cell.get("geometry_eligible")
            == stage_cell.get("geometry_eligible"),
        }
        if not all(identity_checks.values()):
            failed_identity = [
                name for name, passed in identity_checks.items() if not passed
            ]
            raise ValueError(
                f"H4-A cell {index} identity differs: {failed_identity}"
            )
        response_scores = score_cell.get("scores")
        if not isinstance(response_scores, Mapping):
            raise ValueError(f"H4-A cell {index} lacks response scores")
        copied = dict(stage_cell)
        copied_scores = dict(stage_cell.get("scores", {}))
        for _, key in SCORE_SPECS:
            raw = response_scores.get(key)
            if bool(stage_cell["geometry_eligible"]):
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise ValueError(
                        f"eligible H4-A cell {index} lacks numeric {key}"
                    )
                value = float(raw)
                if not np.isfinite(value):
                    raise ValueError(
                        f"eligible H4-A cell {index} has non-finite {key}"
                    )
                copied_scores[key] = value
            else:
                # The extractor can legitimately observe some common support in
                # an ineligible ownership cell.  The routing geometry remains
                # authoritative, so those values are explicitly hidden here.
                copied_scores[key] = None
        copied["scores"] = copied_scores
        joined_cells.append(copied)
    joined["cells"] = joined_cells
    return joined


def _endpoint_metrics(stage_a: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    class_names = stage_a["class_names"]
    endpoints = stage_a["aggregate"]["endpoints"]
    return {
        name: metric_summary(
            _confusion(
                endpoints[key]["full_image"]["confusion"],
                name=f"aggregate {name}",
                num_classes=len(class_names),
            ),
            class_names,
        )
        for name, key in (
            ("k1", "k1"),
            ("matched_k2", "matched_k2"),
            ("k4", "k4"),
        )
    }


def _strong_gate(
    *,
    candidate_miou: float,
    endpoints: Mapping[str, Mapping[str, Any]],
    cost: Mapping[str, Any],
    random_control: Mapping[str, Any],
    formal: bool,
) -> dict[str, Any]:
    k1 = float(endpoints["k1"]["miou"])
    k2 = float(endpoints["matched_k2"]["miou"])
    k4 = float(endpoints["k4"]["miou"])
    denominator = k4 - k1
    if denominator <= 0:
        raise ValueError("sealed K4 must improve over K1")
    retention = (candidate_miou - k1) / denominator
    random_p95 = float(random_control["miou"]["p95"])
    checks = {
        "overall_physical_cost_at_most_2x": float(
            cost["physical_model_sample_cost_ratio"]
        )
        <= MAX_PHYSICAL_COST + 1e-12,
        "every_image_physical_cost_at_most_2x": all(
            float(record["physical_model_sample_cost_ratio"])
            <= MAX_PHYSICAL_COST + 1e-12
            for record in cost["per_image"]
        ),
        "gain_over_matched_k2_at_least_0_05pp": candidate_miou - k2
        >= MIN_GAIN_OVER_K2 - 1e-12,
        "retains_at_least_70pct_of_k4_gain": retention
        >= MIN_K4_GAIN_RETENTION - 1e-12,
        "strictly_above_matched_exact_cost_random_p95": candidate_miou
        > random_p95,
        "small_error_not_worse_than_matched_k2": None,
        "thin_error_not_worse_than_matched_k2": None,
    }
    known_passed = bool(all(value for value in checks.values() if value is not None))
    return {
        "formal_random_control": formal,
        "known_checks_passed": known_passed if formal else None,
        "complete": False,
        "checks": checks,
        "observed": {
            "candidate_miou": candidate_miou,
            "candidate_miou_percent": candidate_miou * 100.0,
            "candidate_minus_k1_pp": (candidate_miou - k1) * 100.0,
            "candidate_minus_matched_k2_pp": (candidate_miou - k2) * 100.0,
            "candidate_minus_k4_pp": (candidate_miou - k4) * 100.0,
            "k4_gain_retention": retention,
            "overall_physical_cost": cost["physical_model_sample_cost_ratio"],
            "maximum_per_image_physical_cost": max(
                float(record["physical_model_sample_cost_ratio"])
                for record in cost["per_image"]
            ),
            "random_p95_miou": random_p95,
            "candidate_minus_random_p95_pp": (candidate_miou - random_p95)
            * 100.0,
        },
    }


def _screen_one(
    *,
    joined_stage_a: Mapping[str, Any],
    geometry: Mapping[str, Any],
    eligible: np.ndarray,
    score_spec: tuple[str, str],
    endpoints: Mapping[str, Mapping[str, Any]],
    random_replicates: int,
    random_seed: int,
) -> dict[str, Any]:
    score_arrays = _score_arrays(
        joined_stage_a["cells"], eligible, (score_spec,)
    )
    crossfit_started = time.perf_counter()
    crossfit = cross_fit_policy(
        joined_stage_a,
        geometry,
        score_arrays,
        batch_size=INFERENCE_BATCH_SIZE,
        score_specs=(score_spec,),
    )
    crossfit_seconds = time.perf_counter() - crossfit_started

    class_names = joined_stage_a["class_names"]
    candidate_metrics = metric_summary(crossfit["full_confusion"], class_names)
    k1_confusion = _confusion(
        joined_stage_a["aggregate"]["endpoints"]["k1"]["full_image"][
            "confusion"
        ],
        name="aggregate K1",
        num_classes=len(class_names),
    )
    reconstructed = confusion_for_levels(
        joined_stage_a["cells"], crossfit["levels_by_cell"], k1_confusion
    )
    if not np.array_equal(reconstructed, crossfit["full_confusion"]):
        raise AssertionError("H4-A cross-fitted confusion reconstruction differs")

    target_costs = [
        int(record["selected_unique_x8_crop_samples"])
        for record in crossfit["cost"]["per_image"]
    ]
    random_started = time.perf_counter()
    random_control = exact_cost_random_control(
        joined_stage_a,
        geometry,
        target_unique_x8_by_image=target_costs,
        replicates=random_replicates,
        seed=random_seed,
    )
    random_seconds = time.perf_counter() - random_started
    formal = random_replicates == RANDOM_REPLICATES and random_seed == RANDOM_SEED
    gate = _strong_gate(
        candidate_miou=float(candidate_metrics["miou"]),
        endpoints=endpoints,
        cost=crossfit["cost"],
        random_control=random_control,
        formal=formal,
    )
    return {
        "score_name": score_spec[0],
        "cell_key": score_spec[1],
        "role": (
            "primary decision score"
            if score_spec[0] == PRIMARY_SCORE
            else "secondary mechanism diagnostic; cannot select the final route"
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
        "descriptive_fixed_q_family": _descriptive_candidate_family(
            crossfit["candidate_family"], class_names
        ),
        "matched_exact_cost_random_control": random_control,
        "strong_gate": gate,
        "runtime": {
            "cross_fit_seconds": crossfit_seconds,
            "random_control_seconds": random_seconds,
        },
    }


def _primary_decision(primary: Mapping[str, Any]) -> dict[str, Any]:
    gate = primary["strong_gate"]
    policy = primary["cross_fitted_policy"]
    observed = gate["observed"]
    formal = gate["formal_random_control"] is True
    selected_qs = [float(fold["selected_q"]) for fold in policy["folds"]]
    nonmaximum_q = any(value < max(Q_VALUES) - 1e-12 for value in selected_qs)
    candidate_miou = float(policy["full_image"]["miou"])
    candidate_minus_k2_fraction = observed["candidate_minus_matched_k2_pp"] / 100.0
    mechanism_checks = {
        "strictly_above_own_matched_random_p95": observed[
            "candidate_minus_random_p95_pp"
        ]
        > 0,
        "not_below_failed_ridge_miou": candidate_miou
        >= RIDGE_MIOU_REFERENCE - 1e-12,
        "nonmaximum_q_selected_or_k2_deficit_smaller_than_ridge": (
            nonmaximum_q
            or candidate_minus_k2_fraction > -RIDGE_DEFICIT_TO_K2 + 1e-12
        ),
    }
    mechanism_passed = bool(all(mechanism_checks.values())) if formal else None
    strong_passed = gate["known_checks_passed"] is True
    if not formal:
        outcome = "NOT_EVALUATED_NONFORMAL_RANDOM_CONTROL"
    elif strong_passed:
        outcome = "PROVISIONAL_GO_H4A_ONE_LIVE_STRUCTURE_LATENCY_CONFIRMATION"
    elif mechanism_passed:
        outcome = "GO_H4B_RESPONSE_MECHANISM_SIGNAL_ONLY"
    else:
        outcome = "STOP_H4A_K1_OVERLAP_RESPONSE_NO_SIGNAL"
    return {
        "outcome": outcome,
        "scientific_decision_evaluated": formal,
        "strong_known_gate_passed": strong_passed if formal else None,
        "mechanism_signal_gate_passed": mechanism_passed,
        "h4a_confirmed": False,
        "h4b_implementation_authorized": bool(
            formal and (strong_passed or mechanism_passed)
        ),
        "live_h4a_structure_latency_authorized": bool(formal and strong_passed),
        "mechanism_checks": mechanism_checks,
        "mechanism_observed": {
            "candidate_miou_percent": candidate_miou * 100.0,
            "failed_ridge_reference_miou_percent": RIDGE_MIOU_REFERENCE * 100.0,
            "candidate_minus_failed_ridge_pp": (
                candidate_miou - RIDGE_MIOU_REFERENCE
            )
            * 100.0,
            "selected_qs": selected_qs,
            "at_least_one_nonmaximum_q": nonmaximum_q,
            "candidate_deficit_to_k2_pp": min(
                0.0, observed["candidate_minus_matched_k2_pp"]
            ),
            "failed_ridge_deficit_to_k2_pp": -RIDGE_DEFICIT_TO_K2 * 100.0,
        },
        "strong_gate": gate,
        "interpretation": (
            "Only the preregistered primary overlap JSD controls this decision. "
            "Auxiliary scores cannot rescue a failed primary screen."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    stage_a_sha = file_sha256(args.stage_a_json)
    stage_b0_sha = file_sha256(args.stage_b0_json)
    latency_sha = file_sha256(args.latency_json)
    statistics_sha = file_sha256(args.overlap_statistics_json)
    stage_a = load_stage_a(args.stage_a_json)
    stage_b0 = load_stage_b0(args.stage_b0_json, stage_a_sha256=stage_a_sha)
    latency = load_latency_confirmation(
        args.latency_json,
        stage_a_sha256=stage_a_sha,
        stage_b0_sha256=stage_b0_sha,
    )
    statistics = _read_json(args.overlap_statistics_json)
    joined = _join_scores(
        stage_a, statistics, stage_a_sha=stage_a_sha
    )
    geometry = build_phase_closure_geometry(joined["images"], joined["cells"])
    eligible = np.asarray(geometry["eligible"], dtype=bool)
    endpoints = _endpoint_metrics(joined)

    results = []
    for score_spec in SCORE_SPECS:
        print(f"screen_score={score_spec[0]}", flush=True)
        results.append(
            _screen_one(
                joined_stage_a=joined,
                geometry=geometry,
                eligible=eligible,
                score_spec=score_spec,
                endpoints=endpoints,
                random_replicates=args.random_replicates,
                random_seed=args.random_seed,
            )
        )
    by_name = {record["score_name"]: record for record in results}
    decision = _primary_decision(by_name[PRIMARY_SCORE])

    if decision["live_h4a_structure_latency_authorized"]:
        next_step = (
            "Run one frozen live small/thin and same-primitive latency confirmation "
            "for the primary overlap-JSD policy."
        )
    elif decision["h4b_implementation_authorized"]:
        next_step = (
            "Implement only H4-B K2-observed response statistics and discrete K1/K2 "
            "arbitration; do not train a utility head yet."
        )
    elif decision["scientific_decision_evaluated"]:
        next_step = (
            "Stop K1-only overlap routing; do not train a K1 utility head or let "
            "secondary scores rescue the failed primary screen."
        )
    else:
        next_step = "Rerun the frozen 1,000-replicate random control before deciding."

    output = {
        "status": "PASS",
        "status_meaning": (
            "Artifact joining, frozen whole-image cross-fit, exact closure, physical "
            "cost, matched random, and endpoint checks completed. The scientific "
            "outcome is in h4a_decision."
        ),
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "official-test exploratory whole-image leave-one-out",
        "scientific_scope": (
            "post-hoc H4-A normal-K1 intra-slide response screen; no K2-visible "
            "feature, training, K2->K4 route, or paper-ready validation"
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
            "h4a_overlap_statistics": {
                "path": str(args.overlap_statistics_json.resolve()),
                "sha256": statistics_sha,
                "runner_git_revision": statistics.get("reproducibility", {}).get(
                    "git_revision"
                ),
            },
        },
        "protocol": {
            "hypothesis_role": (
                "H4 is an independent post-hoc hypothesis and does not modify the "
                "formal H3 No-Go"
            ),
            "route": "K1 -> fixed x8 K2 only",
            "scores": [
                {
                    "name": name,
                    "cell_key": key,
                    "higher_is_promoted_first": True,
                    "role": "primary" if name == PRIMARY_SCORE else "secondary",
                }
                for name, key in SCORE_SPECS
            ],
            "primary_score": PRIMARY_SCORE,
            "auxiliary_cannot_rescue_primary": True,
            "score_combination_or_weight_tuning": False,
            "q_values": list(Q_VALUES),
            "cross_fit": (
                "each score independently uses 20-fold whole-image LOO; q chosen "
                "on the other 19 images by pooled mIoU then lower cost/lower q"
            ),
            "physical_cost": (
                "normal K1 plus per-image x8 exact unique closure, padded once to "
                "fixed batch=8 with padding counted as model samples"
            ),
            "random_replicates": args.random_replicates,
            "random_seed": args.random_seed,
            "formal_random_control": args.random_replicates == RANDOM_REPLICATES
            and args.random_seed == RANDOM_SEED,
            "strong_thresholds": {
                "maximum_overall_and_per_image_cost": MAX_PHYSICAL_COST,
                "minimum_gain_over_matched_k2_pp": MIN_GAIN_OVER_K2 * 100.0,
                "minimum_k4_gain_retention": MIN_K4_GAIN_RETENTION,
                "random": "strictly above own matched exact-cost random p95",
                "structure": "small/thin live check deferred until known gates pass",
            },
            "mechanism_thresholds": {
                "random": "strictly above own matched exact-cost random p95",
                "minimum_miou_percent": RIDGE_MIOU_REFERENCE * 100.0,
                "nonmaximum_q_or_smaller_k2_deficit_than_ridge_pp": RIDGE_DEFICIT_TO_K2
                * 100.0,
            },
        },
        "geometry": {
            "images": len(joined["images"]),
            "cells": len(joined["cells"]),
            "eligible_cells": int(np.count_nonzero(eligible)),
            "baseline_k1_crop_forwards": int(
                np.asarray(geometry["baseline_crop_counts"]).sum()
            ),
        },
        "endpoint_validation": endpoints,
        "independent_score_screens": results,
        "h4a_decision": decision,
        "next_step": next_step,
        "explicit_non_claims": [
            "The formal H3 No-Go has changed.",
            "Auxiliary scores may be selected after seeing the primary result.",
            "Intra-slide disagreement is a pure causal patch-phase measure.",
            "H4-A is deployable before live small/thin and latency confirmation.",
            "Official-test exploratory cross-fit is paper-ready validation.",
        ],
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "statistics_runner_sha256": file_sha256(
                REPO_ROOT / "scripts" / "evaluate_whu_h4a_overlap_statistics.py"
            ),
            "statistics_common_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_overlap_common.py"
            ),
            "closure_common_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_closure_common.py"
            ),
            "h3_screen_common_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_h3_screen_common.py"
            ),
            "numpy": np.__version__,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }
    atomic_write_json(args.output_path, output)
    primary = by_name[PRIMARY_SCORE]
    print(
        json.dumps(
            {
                "status": output["status"],
                "outcome": decision["outcome"],
                "primary_score": PRIMARY_SCORE,
                "candidate_miou_percent": primary["cross_fitted_policy"][
                    "full_image"
                ]["miou_percent"],
                "candidate_minus_matched_k2_pp": primary["strong_gate"][
                    "observed"
                ]["candidate_minus_matched_k2_pp"],
                "candidate_minus_random_p95_pp": primary["strong_gate"][
                    "observed"
                ]["candidate_minus_random_p95_pp"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(f"h4a_screen_result={args.output_path.resolve()}", flush=True)
    print("h4a_screen_status=PASS", flush=True)


if __name__ == "__main__":
    main()
