"""Run the local Stage-B0 exact phase-crop closure audit.

This runner consumes the immutable full-test Stage-A JSON and never loads the
model or dataset.  It first measures the dependency closure of the selected
Stage-A A2/per-image action map, then runs the separately declared A2 rescue
oracle using marginal *unique crop forwards* under independent per-image 2x
caps.  This rescue was motivated after Stage A and is frozen here before the
formal B0 artifact; it is not presented as part of the original Stage-A
preregistration.  A GT-free random-priority walk is evaluated at the exact
realized per-image candidate costs.

Schema-v3 Stage-A artifacts do not contain per-cell K2 small/thin counts.  In
that case B0 can reject on any known failed gate, but otherwise emits only a
provisional B1-smoke authorization; it must not claim a complete Stage-B pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.phase_closure_common import (  # noqa: E402
    PHASE_NAMES,
    PHASE_SHIFTS,
    build_phase_closure_geometry,
    confusion_for_levels,
    exact_cost_hierarchical_oracle,
    mean_iou_from_confusion,
    random_priority_control,
    summarize_closure,
)


ARTIFACT_TYPE = "whu_phase_closure_stage_b0"
SCHEMA_VERSION = 1
EXPECTED_STAGE_A_TYPE = "whu_phase_utility_stage_a"
EXPECTED_STAGE_A_ROUTE = "a2_hierarchical_x"
REQUIRED_REFERENCE_CHECKS = (
    "k1_prediction_sha256_equal",
    "label_sha256_equal",
    "k1_confusion_equal",
    "k1_miou_within_tolerance",
    "legacy_k2_prediction_sha256_equal",
    "legacy_k2_confusion_equal",
    "legacy_k2_miou_within_tolerance",
    "k4_prediction_sha256_equal",
    "k4_confusion_equal",
    "k4_miou_within_tolerance",
)
RANDOM_REPLICATES = 1000
RANDOM_SEED = 20260730
MIN_RETENTION = 0.70
MIN_GAIN_OVER_K2 = 0.0005  # mIoU fraction == +0.05 percentage points
MAX_FORWARD_EQUIVALENT_COST = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local WHU Stage-B0 exact shifted-phase crop-closure audit"
    )
    parser.add_argument("--stage-a-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--random-replicates", type=int, default=RANDOM_REPLICATES)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if not args.stage_a_json.is_file():
        parser.error(f"Stage-A artifact does not exist: {args.stage_a_json}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite output: {args.output_path}")
    if args.random_replicates <= 0:
        parser.error("--random-replicates must be positive")
    if args.random_seed < 0:
        parser.error("--random-seed must be non-negative")
    return args


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_stage_a(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != EXPECTED_STAGE_A_TYPE:
        raise ValueError("input is not a WHU Stage-A phase-utility artifact")
    if payload.get("schema_version") != 3:
        raise ValueError("Stage B0 requires the sealed Stage-A schema version 3")
    if payload.get("status") != "PASS" or payload.get("scope") != "full-test":
        raise ValueError("Stage B0 requires the full-test PASS Stage-A artifact")
    if payload.get("evaluated_images") != payload.get("full_test_length"):
        raise ValueError("Stage-A artifact does not cover the full test set")
    decision = payload.get("stage_a_decision", {})
    if (
        decision.get("outcome") != "GO_STAGE_B_A2_HIERARCHICAL_X"
        or decision.get("passed") is not True
        or decision.get("stage_b_authorized") is not True
        or decision.get("selected_route") != EXPECTED_STAGE_A_ROUTE
    ):
        raise ValueError("Stage A did not authorize the frozen A2 Stage-B route")
    validation = payload.get("reference_validation", {})
    if validation.get("checked") is not True or not all(
        validation.get(name) is True for name in REQUIRED_REFERENCE_CHECKS
    ):
        raise ValueError("Stage-A sealed references are not fully validated")
    protocol = payload.get("protocol", {})
    if protocol.get("teacher_phases_dy_dx") != [
        [0, 0],
        [0, 8],
        [8, 0],
        [8, 8],
    ]:
        raise ValueError("Stage-A phase set differs from the frozen x/y/xy route")
    if protocol.get("crop_size_hw") != [512, 512] or protocol.get(
        "stride_hw"
    ) != [341, 341]:
        raise ValueError("Stage-A sliding geometry differs from 512/341")
    if protocol.get("valid_margin") != 512:
        raise ValueError("Stage-A valid margin differs from the sealed 512 pixels")
    if not isinstance(payload.get("images"), list) or not isinstance(
        payload.get("cells"), list
    ):
        raise ValueError("Stage-A artifact lacks images or cells")
    if len(payload["cells"]) != payload.get("aggregate", {}).get("total_cells"):
        raise ValueError("Stage-A cell count is inconsistent")
    route = payload.get("formal_stage_a_routes", {}).get(
        EXPECTED_STAGE_A_ROUTE, {}
    )
    per_image = route.get("per_image", {})
    if per_image.get("gate", {}).get("passed") is not True:
        raise ValueError("frozen Stage-A A2/per-image route did not pass")
    levels = per_image.get("greedy_point", {}).get("levels_by_cell")
    if not isinstance(levels, list) or len(levels) != len(payload["cells"]):
        raise ValueError("Stage-A A2/per-image action map is missing")
    return payload


def metric_summary(
    confusion: Sequence[Sequence[int]] | np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("confusion shape differs from class names")
    diagonal = np.diag(matrix)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - diagonal
    supported = union > 0
    iou = np.zeros(len(class_names), dtype=np.float64)
    np.divide(diagonal, union, out=iou, where=supported)
    miou = mean_iou_from_confusion(matrix)
    valid = int(matrix.sum())
    errors = int(valid - diagonal.sum())
    return {
        "confusion": matrix,
        "valid_pixels": valid,
        "errors": errors,
        "error_rate": float(errors / valid) if valid else None,
        "miou": miou,
        "miou_percent": float(miou * 100.0),
        "class_iou_percent": {
            str(name): (float(iou[index] * 100.0) if supported[index] else None)
            for index, name in enumerate(class_names)
        },
    }


def _endpoint_levels(level: int, eligible: np.ndarray) -> np.ndarray:
    levels = np.ones(len(eligible), dtype=np.int64)
    levels[eligible] = int(level)
    return levels


def endpoint_closure_validation(
    stage_a: Mapping[str, Any], geometry: Mapping[str, Any]
) -> dict[str, Any]:
    cells = stage_a["cells"]
    eligible = np.asarray(geometry["eligible"], dtype=bool)
    full_k1 = np.asarray(
        stage_a["aggregate"]["endpoints"]["k1"]["full_image"]["confusion"],
        dtype=np.int64,
    )
    levels_by_name = {
        "k1": np.ones(len(cells), dtype=np.int64),
        "matched_k2": _endpoint_levels(2, eligible),
        "k4": _endpoint_levels(4, eligible),
    }
    result = {}
    for name, levels in levels_by_name.items():
        confusion = confusion_for_levels(cells, levels, full_k1)
        expected = np.asarray(
            stage_a["aggregate"]["endpoints"][name]["full_image"]["confusion"],
            dtype=np.int64,
        )
        equal = bool(np.array_equal(confusion, expected))
        if not equal:
            raise AssertionError(f"minimal closure endpoint {name} changed confusion")
        result[name] = {
            "confusion_equal": equal,
            "miou": mean_iou_from_confusion(confusion),
            "closure": summarize_closure(levels, geometry),
        }
    return result


def fixed_policy_audit(
    stage_a: Mapping[str, Any], geometry: Mapping[str, Any]
) -> dict[str, Any]:
    point = stage_a["formal_stage_a_routes"][EXPECTED_STAGE_A_ROUTE][
        "per_image"
    ]["greedy_point"]
    levels = np.asarray(point["levels_by_cell"], dtype=np.int64)
    full_k1 = stage_a["aggregate"]["endpoints"]["k1"]["full_image"][
        "confusion"
    ]
    confusion = confusion_for_levels(stage_a["cells"], levels, full_k1)
    expected = np.asarray(point["full_image"]["confusion"], dtype=np.int64)
    if not np.array_equal(confusion, expected):
        raise AssertionError("fixed Stage-A action map no longer reproduces its result")
    closure = summarize_closure(levels, geometry)
    per_image_pass = all(
        item["forward_equivalent_cost"] <= MAX_FORWARD_EQUIVALENT_COST
        for item in closure["per_image"]
    )
    checks = {
        "overall_exact_cost_at_most_2x": (
            closure["forward_equivalent_cost"] <= MAX_FORWARD_EQUIVALENT_COST
        ),
        "every_image_exact_cost_at_most_2x": per_image_pass,
    }
    return {
        "role": (
            "diagnose Stage-A proxy optimism only; failure does not replace the "
            "separately frozen exact-cost-aware rescue oracle"
        ),
        "levels_by_cell": levels,
        "closure": closure,
        "full_miou": mean_iou_from_confusion(confusion),
        "checks": checks,
        "passed_exact_cost": bool(all(checks.values())),
        "outcome": (
            "FIXED_STAGE_A_MAP_EXACT_COST_PASS"
            if all(checks.values())
            else "FIXED_STAGE_A_MAP_FAILS_OVERLAP_CLOSURE"
        ),
    }


def per_image_stability(
    stage_a: Mapping[str, Any], levels: np.ndarray
) -> dict[str, Any]:
    cells = stage_a["cells"]
    records = []
    for image in stage_a["images"]:
        image_index = int(image["loader_position"])
        cell_indices = [
            index
            for index, cell in enumerate(cells)
            if int(cell["image_index"]) == image_index
        ]
        k1 = np.asarray(image["confusion"]["k1"], dtype=np.int64)
        candidate = k1.copy()
        for cell_index in cell_indices:
            cell = cells[cell_index]
            chosen = np.asarray(
                cell["confusion"][f"k{int(levels[cell_index])}"], dtype=np.int64
            )
            candidate += chosen - np.asarray(cell["confusion"]["k1"], dtype=np.int64)
        k2 = np.asarray(image["confusion"]["matched_k2"], dtype=np.int64)
        k4 = np.asarray(image["confusion"]["k4"], dtype=np.int64)
        candidate_miou = mean_iou_from_confusion(candidate)
        k1_miou = mean_iou_from_confusion(k1)
        k2_miou = mean_iou_from_confusion(k2)
        k4_miou = mean_iou_from_confusion(k4)
        records.append(
            {
                "image_index": image_index,
                "sample_name": image["sample_name"],
                "candidate_miou": candidate_miou,
                "candidate_minus_k1_pp": float((candidate_miou - k1_miou) * 100),
                "candidate_minus_k2_pp": float((candidate_miou - k2_miou) * 100),
                "candidate_minus_k4_pp": float((candidate_miou - k4_miou) * 100),
                "candidate_errors": int(candidate.sum() - np.diag(candidate).sum()),
                "k2_errors": int(k2.sum() - np.diag(k2).sum()),
            }
        )
    deltas = np.asarray(
        [record["candidate_minus_k2_pp"] for record in records], dtype=np.float64
    )
    return {
        "per_image": records,
        "summary": {
            "images_above_k1": int(
                sum(record["candidate_minus_k1_pp"] > 0 for record in records)
            ),
            "images_above_k2": int(np.count_nonzero(deltas > 0)),
            "images_not_above_k2": int(np.count_nonzero(deltas <= 0)),
            "candidate_minus_k2_pp_minimum": float(deltas.min()),
            "candidate_minus_k2_pp_median": float(np.median(deltas)),
            "candidate_minus_k2_pp_maximum": float(deltas.max()),
        },
    }


def rescue_gate(
    stage_a: Mapping[str, Any],
    oracle: Mapping[str, Any],
    random_control: Mapping[str, Any],
    *,
    formal_random_control: bool = True,
) -> dict[str, Any]:
    endpoints = stage_a["aggregate"]["endpoints"]
    k1_miou = float(endpoints["k1"]["full_image"]["miou"])
    k2_miou = float(endpoints["matched_k2"]["full_image"]["miou"])
    k4_miou = float(endpoints["k4"]["full_image"]["miou"])
    candidate_miou = float(oracle["full_miou"])
    denominator = k4_miou - k1_miou
    if denominator <= 0:
        raise ValueError("sealed K4 does not improve over K1")
    retention = (candidate_miou - k1_miou) / denominator
    closure = oracle["closure"]
    observed = {
        "candidate_miou": candidate_miou,
        "candidate_miou_percent": candidate_miou * 100.0,
        "candidate_minus_k1_pp": (candidate_miou - k1_miou) * 100.0,
        "candidate_minus_matched_k2_pp": (candidate_miou - k2_miou) * 100.0,
        "k4_gain_retention": retention,
        "overall_exact_cost": closure["forward_equivalent_cost"],
        "maximum_per_image_exact_cost": max(
            item["forward_equivalent_cost"] for item in closure["per_image"]
        ),
        "random_p95_miou": float(random_control["miou"]["p95"]),
        "candidate_minus_random_p95_pp": (
            candidate_miou - float(random_control["miou"]["p95"])
        )
        * 100.0,
    }
    checks: dict[str, bool | None] = {
        "overall_exact_cost_at_most_2x": (
            observed["overall_exact_cost"] <= MAX_FORWARD_EQUIVALENT_COST
        ),
        "every_image_exact_cost_at_most_2x": all(
            item["forward_equivalent_cost"] <= MAX_FORWARD_EQUIVALENT_COST
            for item in closure["per_image"]
        ),
        "retains_at_least_70pct_of_k4_gain": retention >= MIN_RETENTION,
        "gain_over_matched_k2_at_least_0_05pp": (
            candidate_miou - k2_miou >= MIN_GAIN_OVER_K2
        ),
        "strictly_above_exact_cost_random_p95": (
            candidate_miou > float(random_control["miou"]["p95"])
        ),
    }
    regions = oracle.get("regions")
    structure = {
        "available": regions is not None,
        "reason_if_missing": (
            None
            if regions is not None
            else (
                "schema-v3 Stage-A compact cells omit per-level region errors; "
                "B1 live evaluation must close aggregate small/thin before a "
                "complete Stage-B decision"
            )
        ),
        "candidate": regions,
        "matched_k2": endpoints["matched_k2"]["common_support"]["regions"],
    }
    if regions is None:
        checks["small_error_not_worse_than_matched_k2"] = None
        checks["thin_error_not_worse_than_matched_k2"] = None
    else:
        for region_name in ("small", "thin"):
            candidate_rate = float(regions[region_name]["error_rate"])
            reference_rate = float(
                endpoints["matched_k2"]["common_support"]["regions"][region_name][
                    "error_rate"
                ]
            )
            observed[f"{region_name}_minus_matched_k2_error_rate_pp"] = (
                candidate_rate - reference_rate
            ) * 100.0
            checks[f"{region_name}_error_not_worse_than_matched_k2"] = (
                candidate_rate <= reference_rate
            )

    known_checks = [value for value in checks.values() if value is not None]
    known_pass = all(known_checks)
    complete = all(value is not None for value in checks.values())
    failed_checks = {name for name, value in checks.items() if value is False}
    structure_check_names = {
        "small_error_not_worse_than_matched_k2",
        "thin_error_not_worse_than_matched_k2",
    }
    if not formal_random_control:
        outcome = "NOT_EVALUATED_NONFORMAL_RANDOM"
        b1_smoke_authorized = False
    elif failed_checks and failed_checks.issubset(structure_check_names):
        outcome = "STOP_STAGE_B0_STRUCTURE_GATE_FAILED"
        b1_smoke_authorized = False
    elif failed_checks:
        outcome = "STOP_STAGE_B0_KNOWN_GATE_FAILED"
        b1_smoke_authorized = False
    elif not complete:
        outcome = "PROVISIONAL_GO_B1_KNOWN_B0_GATES_PASS"
        b1_smoke_authorized = True
    else:
        outcome = "GO_B1_EXACT_SPARSE_EXECUTION"
        b1_smoke_authorized = True
    return {
        "outcome": outcome,
        "scientific_decision_evaluated": formal_random_control,
        "complete": complete and formal_random_control,
        "known_checks_passed": known_pass,
        "b1_implementation_and_smoke_authorized": b1_smoke_authorized,
        "full_stage_b_scientific_pass": False,
        "checks": checks,
        "observed": observed,
        "thresholds": {
            "maximum_overall_and_per_image_exact_cost": MAX_FORWARD_EQUIVALENT_COST,
            "minimum_k4_gain_retention": MIN_RETENTION,
            "minimum_gain_over_matched_k2_miou_fraction": MIN_GAIN_OVER_K2,
            "minimum_gain_over_matched_k2_pp": MIN_GAIN_OVER_K2 * 100.0,
            "random": "strictly above exact-realized-cost random-priority p95",
            "structure": "aggregate small/thin error rate no worse than matched K2x",
        },
        "structure": structure,
        "interpretation": (
            (
                "A non-default random replicate count or seed is development-only "
                "and cannot authorize B1. "
                if not formal_random_control
                else "B0 may authorize only B1 implementation/smoke. "
            )
            + "H2 has provisional geometry-level support but remains unconfirmed "
            "until sparse execution, endpoint/middle closure, structure, and "
            "latency checks all pass."
        ),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    stage_a = load_stage_a(args.stage_a_json)
    geometry = build_phase_closure_geometry(stage_a["images"], stage_a["cells"])
    endpoints = endpoint_closure_validation(stage_a, geometry)
    fixed = fixed_policy_audit(stage_a, geometry)
    class_names = stage_a["class_names"]
    full_k1_confusion = stage_a["aggregate"]["endpoints"]["k1"]["full_image"][
        "confusion"
    ]

    oracle_started = time.perf_counter()
    oracle = exact_cost_hierarchical_oracle(
        stage_a["cells"],
        geometry,
        full_k1_confusion,
        extra_crop_cap_by_image=geometry["baseline_crop_counts"],
        num_classes=len(class_names),
    )
    oracle_seconds = time.perf_counter() - oracle_started
    oracle_metrics = metric_summary(oracle["full_confusion"], class_names)
    stability = per_image_stability(stage_a, oracle["levels_by_cell"])

    random_started = time.perf_counter()
    random_control = random_priority_control(
        stage_a["cells"],
        geometry,
        full_k1_confusion,
        extra_crop_cap_by_image=oracle["used_extra_crop_forwards_by_image"],
        replicates=args.random_replicates,
        seed=args.random_seed,
    )
    random_seconds = time.perf_counter() - random_started
    formal_random_control = (
        args.random_replicates == RANDOM_REPLICATES
        and args.random_seed == RANDOM_SEED
    )
    gate = rescue_gate(
        stage_a,
        oracle,
        random_control,
        formal_random_control=formal_random_control,
    )
    total_seconds = time.perf_counter() - started

    output = {
        "status": "PASS",
        "status_meaning": (
            "Local geometry, manifest, endpoint, exact-cost greedy, random-control, "
            "and serialization checks completed. Scientific completeness is in "
            "stage_b0_decision and is not implied by status."
        ),
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "full-test",
        "scientific_scope": (
            "Stage-B0 geometry-only exact shifted-phase crop-closure feasibility; "
            "no model forward, latency, sparse logits, or router generalization"
        ),
        "source_stage_a": {
            "path": str(args.stage_a_json.resolve()),
            "sha256": file_sha256(args.stage_a_json),
            "artifact_type": stage_a["artifact_type"],
            "schema_version": stage_a["schema_version"],
            "git_revision": stage_a["reproducibility"]["git_revision"],
            "selected_route": stage_a["stage_a_decision"]["selected_route"],
        },
        "protocol": {
            "route": "A2 hierarchical K1 -> fixed x8 K2 -> x8/y8/xy8 K4",
            "phase_shifts_dy_dx": {
                name: list(PHASE_SHIFTS[name]) for name in PHASE_NAMES
            },
            "routed_region": (
                "half-open ownership rectangle intersected with sealed K4 common support"
            ),
            "dependency": (
                "translate the entire routed region by the phase shift and include "
                "every same-phase 512/341 window with positive-area intersection"
            ),
            "deduplication_key": "(image_index, phase_name, local_crop_id)",
            "common_support_outside_behavior": "retain K1",
            "invalid_label_holes_reduce_closure": False,
            "fixed_map_role": "diagnostic only",
            "formal_b0_candidate": "per-image-capped exact-cost-aware A2 oracle",
            "zero_cost_action_rule": (
                "strictly positive zero-cost actions first by absolute gain, then "
                "lower global cell index"
            ),
            "paid_action_rule": (
                "strictly positive current pooled-mIoU gain per newly required crop, "
                "then lower global cell index"
            ),
            "bridge_rule": "never cross a nonpositive K1->K2x bridge",
            "random_rule": (
                "GT-free fixed random transition priorities; independent per-image "
                "walks are retried until their realized unique-crop cost exactly "
                "equals the candidate; zero-cost fitting actions remain eligible"
            ),
            "random_replicates": args.random_replicates,
            "random_seed": args.random_seed,
            "formal_random_control": formal_random_control,
            "formal_random_requirement": {
                "replicates": RANDOM_REPLICATES,
                "seed": RANDOM_SEED,
            },
        },
        "geometry": {
            "images": len(stage_a["images"]),
            "cells": len(stage_a["cells"]),
            "eligible_cells": int(np.count_nonzero(geometry["eligible"])),
            "baseline_k1_crop_forwards": int(
                np.asarray(geometry["baseline_crop_counts"]).sum()
            ),
        },
        "minimal_common_support_endpoints": endpoints,
        "fixed_stage_a_policy": fixed,
        "exact_cost_a2_oracle": {
            "role": (
                "GT-informed optimistic executable-cost feasibility oracle; not a "
                "deployable router or an optimality proof"
            ),
            "selection_rule": oracle["selection_rule"],
            "levels_by_cell": oracle["levels_by_cell"],
            "action_trace": oracle["actions"],
            "caps_by_image": oracle["caps_by_image"],
            "closure": oracle["closure"],
            "full_image": oracle_metrics,
            "regions": oracle["regions"],
            "stability": stability,
        },
        "exact_cost_random_control": random_control,
        "stage_b0_decision": gate,
        "next_step": (
            "Implement B1 sparse runner and run one-image correctness smoke only "
            "if stage_b0_decision authorizes it; B1 must close K1/K2x/K4 plus a "
            "frozen middle subset, aggregate small/thin, exact crop keys, final "
            "confusion, and support-pruned K4 latency before H2 can pass."
        ),
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "common_module_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_closure_common.py"
            ),
            "numpy": np.__version__,
        },
        "runtime": {
            "oracle_seconds": oracle_seconds,
            "random_control_seconds": random_seconds,
            "total_seconds": total_seconds,
        },
    }
    atomic_write_json(args.output_path, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "fixed_stage_a_map_exact_cost": fixed["closure"][
                    "forward_equivalent_cost"
                ],
                "exact_oracle_cost": oracle["closure"]["forward_equivalent_cost"],
                "exact_oracle_miou_percent": oracle_metrics["miou_percent"],
                "exact_oracle_minus_k2_pp": gate["observed"][
                    "candidate_minus_matched_k2_pp"
                ],
                "random_p95_miou_percent": random_control["miou"]["p95"] * 100.0,
                "stage_b0_outcome": gate["outcome"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"phase_closure_result={args.output_path.resolve()}", flush=True)
    print("phase_closure_status=PASS", flush=True)


if __name__ == "__main__":
    main()
