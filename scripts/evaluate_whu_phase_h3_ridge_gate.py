"""Run the one allowed nested-LOO ridge gate after simple H3 scores fail.

The three input features and ridge hyperparameter are fixed.  Every outer
held-out image is scored by a model fit on the other 19 images.  Its q value
is selected only by an inner 19-fold whole-image cross-fit.  The target is a
fit-fold singleton K1-to-K2 pooled-mIoU gain, never the Stage-B0 action map.

This remains an exploratory official-test screen.  Even a pass can authorize
only one frozen live structure/latency confirmation; it cannot establish H3.
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
    Q_VALUES,
    _confusion,
    _subset_physical_cost,
    exact_cost_random_control,
    load_latency_confirmation,
    load_stage_b0,
    screen_gate,
)
from scripts.phase_closure_common import (  # noqa: E402
    build_phase_closure_geometry,
    confusion_for_levels,
    mean_iou_from_confusion,
    summarize_closure,
)
from scripts.phase_h3_ridge_common import (  # noqa: E402
    FEATURE_NAMES,
    RIDGE_LAMBDA,
    fit_ridge_gate,
    predict_ridge_gate,
)
from scripts.phase_h3_screen_common import (  # noqa: E402
    physical_cost_summary,
    score_ranked_k2_levels_for_image,
)


ARTIFACT_TYPE = "whu_phase_h3_ridge_k1_to_k2_gate"
SCHEMA_VERSION = 1
EXPECTED_SIMPLE_TYPE = "whu_phase_h3_k1_to_k2_screen"
EXPECTED_SIMPLE_OUTCOME = "STOP_SIMPLE_K1_SCORE_SCREEN_KNOWN_GATE_FAILED"
RANDOM_REPLICATES = 1000
RANDOM_SEED = 20260801
EXPECTED_IMAGE_COUNT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested whole-image-LOO H3 ridge K1-to-K2 gate"
    )
    parser.add_argument("--stage-a-json", type=Path, required=True)
    parser.add_argument("--stage-b0-json", type=Path, required=True)
    parser.add_argument("--latency-json", type=Path, required=True)
    parser.add_argument("--simple-h3-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--random-replicates", type=int, default=RANDOM_REPLICATES
    )
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    for name in (
        "stage_a_json",
        "stage_b0_json",
        "latency_json",
        "simple_h3_json",
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


def load_simple_h3_failure(
    path: Path,
    *,
    stage_a_sha256: str,
    stage_b0_sha256: str,
    latency_sha256: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("artifact_type") != EXPECTED_SIMPLE_TYPE:
        raise ValueError("input is not the simple H3 K1-to-K2 screen artifact")
    if payload.get("schema_version") != 1 or payload.get("status") != "PASS":
        raise ValueError("ridge gate requires the PASS simple-H3 schema version 1")
    sources = payload.get("source_artifacts")
    if not isinstance(sources, Mapping):
        raise ValueError("simple-H3 artifact lacks source provenance")
    expected = {
        "stage_a": stage_a_sha256,
        "stage_b0": stage_b0_sha256,
        "full_stage_b_latency": latency_sha256,
    }
    for name, sha256 in expected.items():
        source = sources.get(name)
        if not isinstance(source, Mapping) or source.get("sha256") != sha256:
            raise ValueError(f"simple-H3 {name} source does not match")
    protocol = payload.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("formal_random_control") is not True
        or protocol.get("random_replicates") != 1000
        or protocol.get("random_seed") != 20260731
    ):
        raise ValueError("simple-H3 screen did not use its frozen random control")
    decision = payload.get("h3_screen_decision")
    if (
        not isinstance(decision, Mapping)
        or decision.get("scientific_decision_evaluated") is not True
        or decision.get("known_checks_passed") is not False
        or decision.get("one_live_confirmation_authorized") is not False
        or decision.get("outcome") != EXPECTED_SIMPLE_OUTCOME
    ):
        raise ValueError("simple-H3 screen did not formally fail as required")
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


def _full_k1_confusions(stage_a: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    class_count = len(stage_a["class_names"])
    return tuple(
        _confusion(
            image["confusion"]["k1"],
            name=f"images[{image_index}].confusion.k1",
            num_classes=class_count,
        )
        for image_index, image in enumerate(stage_a["images"])
    )


def _image_confusion_for_levels(
    stage_a: Mapping[str, Any],
    geometry: Mapping[str, Any],
    levels: np.ndarray,
    image_index: int,
) -> np.ndarray:
    """Reconstruct exactly one named image; no other image confusion is read."""

    cell_indices = np.asarray(
        geometry["cells_by_image"][image_index], dtype=np.int64
    )
    image_cells = [stage_a["cells"][int(index)] for index in cell_indices]
    image_k1 = _confusion(
        stage_a["images"][image_index]["confusion"]["k1"],
        name=f"images[{image_index}].confusion.k1",
        num_classes=len(stage_a["class_names"]),
    )
    return confusion_for_levels(image_cells, levels[cell_indices], image_k1)


def _selection_scores(
    prediction: Mapping[str, Any],
    *,
    cell_count: int,
) -> np.ndarray:
    """Make a finite global vector; only the requested image is ever ranked."""

    raw = prediction.get("scores_by_cell")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("ridge prediction lacks scores_by_cell")
    if len(raw) != cell_count:
        raise ValueError("ridge prediction score length differs from cells")
    scores = np.zeros(cell_count, dtype=np.float64)
    predicted = set(int(value) for value in prediction["predicted_cell_indices"])
    for cell_index, value in enumerate(raw):
        if cell_index in predicted:
            if value is None or not np.isfinite(float(value)):
                raise ValueError("a predicted ridge score is not finite")
            scores[cell_index] = float(value)
        elif value is not None:
            raise ValueError("ridge prediction populated an unrequested cell")
    if len(predicted) != len(prediction["predicted_cell_indices"]):
        raise ValueError("ridge prediction contains duplicate cell indices")
    return scores


def _compact_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Retain enough to audit a fit without duplicating thousands of indices."""

    target = model["target_audit"]
    return {
        "model_type": model["model_type"],
        "schema_version": model["schema_version"],
        "feature_names": model["feature_names"],
        "feature_mean": model["feature_mean"],
        "feature_scale": model["feature_scale"],
        "constant_feature_mask": model["constant_feature_mask"],
        "standardized_coefficients": model["standardized_coefficients"],
        "raw_coefficients": model["raw_coefficients"],
        "raw_intercept": model["raw_intercept"],
        "target_mean": model["target_mean"],
        "ridge_lambda": model["ridge_lambda"],
        "fit_image_indices": model["fit_image_indices"],
        "fit_cell_count": model["fit_cell_count"],
        "target_audit": {
            key: value
            for key, value in target.items()
            if key != "pooled_full_k1_confusion"
        },
        "feature_audit": model["feature_audit"],
        "training_diagnostics": model["training_diagnostics"],
    }


def _q_selection_key(
    *, pooled_miou: float, processed_x8_crops: int, q: float
) -> tuple[float, int, float]:
    return (-float(pooled_miou), int(processed_x8_crops), float(q))


def _levels_for_prediction(
    prediction: Mapping[str, Any],
    *,
    q: float,
    geometry: Mapping[str, Any],
    image_index: int,
) -> dict[str, Any]:
    scores = _selection_scores(
        prediction, cell_count=len(geometry["image_ids"])
    )
    return score_ranked_k2_levels_for_image(
        scores, q, geometry, image_index
    )


def _assert_subset_cost(
    cost: Mapping[str, Any], image_indices: Sequence[int]
) -> tuple[float, int]:
    ratio = _subset_physical_cost(cost, image_indices)
    if ratio > MAX_PHYSICAL_COST + 1e-12:
        raise AssertionError("a frozen ridge q exceeds the 2x physical cost cap")
    processed = 0
    for image_index in image_indices:
        record = cost["per_image"][int(image_index)]
        if (
            float(record["physical_model_sample_cost_ratio"])
            > MAX_PHYSICAL_COST + 1e-12
        ):
            raise AssertionError("a ridge image exceeds the 2x physical cost cap")
        processed += int(record["processed_x8_crop_samples_including_padding"])
    return ratio, processed


def nested_cross_fit_ridge(
    stage_a: Mapping[str, Any],
    geometry: Mapping[str, Any],
    full_k1_by_image: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Run 20 outer folds, each with 19 inner LOO fits plus one final fit."""

    image_count = len(stage_a["images"])
    if image_count != EXPECTED_IMAGE_COUNT:
        raise ValueError(f"formal ridge gate requires {EXPECTED_IMAGE_COUNT} images")
    cells = stage_a["cells"]
    class_count = len(stage_a["class_names"])
    all_images = tuple(range(image_count))
    final_levels = np.ones(len(cells), dtype=np.int64)
    outer_records = []
    chosen_q_counts = Counter()
    fit_count = 0

    for outer_index in all_images:
        outer_train = tuple(value for value in all_images if value != outer_index)
        inner_levels = {
            q: np.ones(len(cells), dtype=np.int64) for q in Q_VALUES
        }
        inner_records = []
        for inner_index in outer_train:
            inner_fit = tuple(
                value for value in outer_train if value != inner_index
            )
            model = fit_ridge_gate(
                cells,
                full_k1_by_image,
                inner_fit,
                num_classes=class_count,
                ridge_lambda=RIDGE_LAMBDA,
            )
            fit_count += 1
            prediction = predict_ridge_gate(model, cells, (inner_index,))
            inner_cell_indices = np.asarray(
                geometry["cells_by_image"][inner_index], dtype=np.int64
            )
            per_q = {}
            for q in Q_VALUES:
                selection = _levels_for_prediction(
                    prediction,
                    q=q,
                    geometry=geometry,
                    image_index=inner_index,
                )
                inner_levels[q][inner_cell_indices] = np.asarray(
                    selection["levels_by_cell"], dtype=np.int64
                )[inner_cell_indices]
                per_q[f"{q:.12g}"] = {
                    "initial_selected_count": int(
                        len(selection["initial_selected_indices"])
                    ),
                    "zero_cost_selected_count": int(
                        len(selection["zero_cost_selected_indices"])
                    ),
                    "unique_x8_crop_count": int(
                        selection["closure"]["per_image"][inner_index][
                            "extra_crop_forwards_by_phase"
                        ]["x8"]
                    ),
                }
            inner_records.append(
                {
                    "inner_heldout_image_index": inner_index,
                    "inner_fit_image_indices": list(inner_fit),
                    "model": _compact_model(model),
                    "prediction": {
                        key: prediction[key]
                        for key in (
                            "predict_image_indices",
                            "predicted_cell_indices",
                            "prediction_sha256_float64_le",
                        )
                    },
                    "per_q_selection": per_q,
                }
            )

        q_records = []
        for q in Q_VALUES:
            levels = inner_levels[q]
            pooled = np.zeros_like(full_k1_by_image[outer_train[0]])
            for image_index in outer_train:
                pooled += _image_confusion_for_levels(
                    stage_a, geometry, levels, image_index
                )
            pooled_miou = mean_iou_from_confusion(pooled)
            cost = physical_cost_summary(
                levels, geometry, batch_size=INFERENCE_BATCH_SIZE
            )
            physical_ratio, processed = _assert_subset_cost(cost, outer_train)
            q_records.append(
                {
                    "q": float(q),
                    "inner_cross_fitted_pooled_miou": pooled_miou,
                    "inner_cross_fitted_pooled_miou_percent": pooled_miou * 100.0,
                    "inner_physical_cost_ratio": physical_ratio,
                    "inner_processed_x8_crop_samples": processed,
                }
            )
        selected_q_record = min(
            q_records,
            key=lambda record: _q_selection_key(
                pooled_miou=record["inner_cross_fitted_pooled_miou"],
                processed_x8_crops=record["inner_processed_x8_crop_samples"],
                q=record["q"],
            ),
        )
        selected_q = float(selected_q_record["q"])
        chosen_q_counts[f"{selected_q:.12g}"] += 1

        final_model = fit_ridge_gate(
            cells,
            full_k1_by_image,
            outer_train,
            num_classes=class_count,
            ridge_lambda=RIDGE_LAMBDA,
        )
        fit_count += 1
        outer_prediction = predict_ridge_gate(
            final_model, cells, (outer_index,)
        )
        outer_selection = _levels_for_prediction(
            outer_prediction,
            q=selected_q,
            geometry=geometry,
            image_index=outer_index,
        )
        outer_cell_indices = np.asarray(
            geometry["cells_by_image"][outer_index], dtype=np.int64
        )
        # Outer GT is not referenced until this action copy is complete.
        final_levels[outer_cell_indices] = np.asarray(
            outer_selection["levels_by_cell"], dtype=np.int64
        )[outer_cell_indices]
        outer_confusion = _image_confusion_for_levels(
            stage_a, geometry, final_levels, outer_index
        )
        outer_cost = physical_cost_summary(
            np.asarray(outer_selection["levels_by_cell"], dtype=np.int64),
            geometry,
            batch_size=INFERENCE_BATCH_SIZE,
        )["per_image"][outer_index]
        if (
            float(outer_cost["physical_model_sample_cost_ratio"])
            > MAX_PHYSICAL_COST + 1e-12
        ):
            raise AssertionError("outer ridge action exceeds the physical cap")
        outer_records.append(
            {
                "outer_heldout_image_index": outer_index,
                "outer_heldout_sample_name": stage_a["images"][outer_index][
                    "sample_name"
                ],
                "outer_train_image_indices": list(outer_train),
                "selected_q": selected_q,
                "inner_q_curve": q_records,
                "inner_folds": inner_records,
                "final_model": _compact_model(final_model),
                "outer_prediction": {
                    key: outer_prediction[key]
                    for key in (
                        "predict_image_indices",
                        "predicted_cell_indices",
                        "prediction_sha256_float64_le",
                    )
                },
                "outer_selection": {
                    "initial_selected_indices": outer_selection[
                        "initial_selected_indices"
                    ],
                    "zero_cost_selected_indices": outer_selection[
                        "zero_cost_selected_indices"
                    ],
                },
                "outer_cost": outer_cost,
                "outer_full_image": metric_summary(
                    outer_confusion, stage_a["class_names"]
                ),
            }
        )

    expected_fits = image_count * image_count
    if fit_count != expected_fits:
        raise AssertionError(
            f"nested ridge fit count {fit_count} differs from {expected_fits}"
        )
    image_confusions = tuple(
        _image_confusion_for_levels(stage_a, geometry, final_levels, image_index)
        for image_index in all_images
    )
    pooled = np.zeros_like(image_confusions[0])
    for confusion in image_confusions:
        pooled += confusion
    full_k1 = np.zeros_like(full_k1_by_image[0])
    for confusion in full_k1_by_image:
        full_k1 += confusion
    reconstructed = confusion_for_levels(cells, final_levels, full_k1)
    if not np.array_equal(pooled, reconstructed):
        raise AssertionError(
            "ridge per-image confusion differs from aggregate reconstruction"
        )
    closure = summarize_closure(final_levels, geometry, include_crop_ids=True)
    cost = physical_cost_summary(
        final_levels, geometry, batch_size=INFERENCE_BATCH_SIZE
    )
    for record, fold in zip(cost["per_image"], outer_records, strict=True):
        if (
            int(record["selected_unique_x8_crop_samples"])
            != int(fold["outer_cost"]["selected_unique_x8_crop_samples"])
            or int(record["processed_x8_crop_samples_including_padding"])
            != int(
                fold["outer_cost"][
                    "processed_x8_crop_samples_including_padding"
                ]
            )
        ):
            raise AssertionError("outer fold cost differs from final closure")
    return {
        "levels_by_cell": final_levels,
        "full_confusion": reconstructed,
        "closure": closure,
        "cost": cost,
        "outer_folds": outer_records,
        "chosen_q_counts": dict(sorted(chosen_q_counts.items())),
        "ridge_fit_count": fit_count,
    }


def ridge_gate_decision(
    *,
    candidate_miou: float,
    k1_miou: float,
    matched_k2_miou: float,
    k4_miou: float,
    cost: Mapping[str, Any],
    random_control: Mapping[str, Any],
    formal_random_control: bool,
) -> dict[str, Any]:
    decision = screen_gate(
        candidate_miou=candidate_miou,
        k1_miou=k1_miou,
        matched_k2_miou=matched_k2_miou,
        k4_miou=k4_miou,
        cost=cost,
        random_control=random_control,
        formal_random_control=formal_random_control,
    )
    if not formal_random_control:
        decision["outcome"] = "NOT_EVALUATED_NONFORMAL_RIDGE_RANDOM_CONTROL"
        decision["interpretation"] = (
            "Smoke only; the ridge gate requires its frozen 1,000-replicate "
            "random control before any stop/go decision."
        )
    elif decision["known_checks_passed"]:
        decision["outcome"] = (
            "PROVISIONAL_GO_ONE_RIDGE_LIVE_CONFIRMATION_KNOWN_GATES_PASS"
        )
        decision["interpretation"] = (
            "The only ridge gate passed known offline checks and may receive one "
            "frozen live structure/latency confirmation; H3 is not confirmed."
        )
    else:
        decision["outcome"] = "STOP_H3_RIDGE_GATE_KNOWN_GATE_FAILED"
        decision["interpretation"] = (
            "The only allowed ridge gate failed; end H3 without changing target, "
            "features, model, q grid, or thresholds."
        )
    return decision


def next_step_for_ridge(decision: Mapping[str, Any]) -> str:
    if decision.get("scientific_decision_evaluated") is not True:
        return "Run the frozen formal random control before deciding."
    if decision.get("one_live_confirmation_authorized") is True:
        return "Run one frozen live small/thin and same-primitive latency confirmation."
    return "End H3; do not add features, models, targets, or q values."


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    stage_a_sha = file_sha256(args.stage_a_json)
    stage_b0_sha = file_sha256(args.stage_b0_json)
    latency_sha = file_sha256(args.latency_json)
    simple_sha = file_sha256(args.simple_h3_json)
    stage_a = load_stage_a(args.stage_a_json)
    stage_b0 = load_stage_b0(args.stage_b0_json, stage_a_sha256=stage_a_sha)
    latency = load_latency_confirmation(
        args.latency_json,
        stage_a_sha256=stage_a_sha,
        stage_b0_sha256=stage_b0_sha,
    )
    simple = load_simple_h3_failure(
        args.simple_h3_json,
        stage_a_sha256=stage_a_sha,
        stage_b0_sha256=stage_b0_sha,
        latency_sha256=latency_sha,
    )
    if len(stage_a["images"]) != EXPECTED_IMAGE_COUNT:
        raise ValueError("formal ridge gate requires the frozen 20-image test set")
    geometry = build_phase_closure_geometry(stage_a["images"], stage_a["cells"])
    full_k1_by_image = _full_k1_confusions(stage_a)

    nested_started = time.perf_counter()
    nested = nested_cross_fit_ridge(stage_a, geometry, full_k1_by_image)
    nested_seconds = time.perf_counter() - nested_started

    class_names = stage_a["class_names"]
    endpoints = stage_a["aggregate"]["endpoints"]
    k1 = _confusion(
        endpoints["k1"]["full_image"]["confusion"],
        name="aggregate K1",
        num_classes=len(class_names),
    )
    k2 = _confusion(
        endpoints["matched_k2"]["full_image"]["confusion"],
        name="aggregate matched K2",
        num_classes=len(class_names),
    )
    k4 = _confusion(
        endpoints["k4"]["full_image"]["confusion"],
        name="aggregate K4",
        num_classes=len(class_names),
    )
    if not np.array_equal(sum(full_k1_by_image, np.zeros_like(k1)), k1):
        raise AssertionError("per-image full K1 confusions do not sum to aggregate K1")

    target_costs = [
        int(record["selected_unique_x8_crop_samples"])
        for record in nested["cost"]["per_image"]
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
    candidate_metrics = metric_summary(nested["full_confusion"], class_names)
    k1_metrics = metric_summary(k1, class_names)
    k2_metrics = metric_summary(k2, class_names)
    k4_metrics = metric_summary(k4, class_names)
    decision = ridge_gate_decision(
        candidate_miou=candidate_metrics["miou"],
        k1_miou=k1_metrics["miou"],
        matched_k2_miou=k2_metrics["miou"],
        k4_miou=k4_metrics["miou"],
        cost=nested["cost"],
        random_control=random_control,
        formal_random_control=formal_random,
    )

    output = {
        "status": "PASS",
        "status_meaning": (
            "Input, nested whole-image cross-fit, target provenance, exact closure, "
            "physical cost, matched random, and serialization checks completed. "
            "Scientific outcome is in ridge_gate_decision."
        ),
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "official-test exploratory strict nested whole-image leave-one-out",
        "scientific_scope": (
            "the one pre-authorized fixed ridge gate for K1-visible K1->K2 "
            "selection after simple-score failure; no K2->K4 or live execution"
        ),
        "source_artifacts": {
            "stage_a": {"path": str(args.stage_a_json.resolve()), "sha256": stage_a_sha},
            "stage_b0": {"path": str(args.stage_b0_json.resolve()), "sha256": stage_b0_sha},
            "full_stage_b_latency": {
                "path": str(args.latency_json.resolve()),
                "sha256": latency_sha,
                "outcome": latency["latency_decision"]["outcome"],
            },
            "simple_h3_failure": {
                "path": str(args.simple_h3_json.resolve()),
                "sha256": simple_sha,
                "outcome": simple["h3_screen_decision"]["outcome"],
            },
        },
        "protocol": {
            "features": list(FEATURE_NAMES),
            "target": (
                "fit-fold singleton pooled-full-image-mIoU gain from replacing "
                "one cell K1 confusion by K2 confusion"
            ),
            "model": "standardized three-feature ridge with centered target",
            "ridge_lambda_sum_sse": RIDGE_LAMBDA,
            "feature_standardization_ddof": 0,
            "outer_folds": EXPECTED_IMAGE_COUNT,
            "inner_folds_per_outer": EXPECTED_IMAGE_COUNT - 1,
            "expected_total_ridge_fits": EXPECTED_IMAGE_COUNT**2,
            "q_values": list(Q_VALUES),
            "q_selection": (
                "inner cross-fitted 19-image pooled mIoU, then integer processed "
                "x8 samples, then lower q"
            ),
            "selection": (
                "per-image top floor(q*eligible), lower global-index tie, then "
                "all zero-cost x8 closure spill"
            ),
            "physical_cost": "per-image unique x8 closure with fixed batch=8 padding",
            "random_replicates": args.random_replicates,
            "random_seed": args.random_seed,
            "formal_random_control": formal_random,
        },
        "provenance": {
            "feature_uses_ground_truth": False,
            "target_uses_fit_fold_ground_truth": True,
            "uses_stored_stage_a_oracle_score": False,
            "uses_stage_b0_levels_or_action_trace": False,
            "outer_action_uses_outer_ground_truth": False,
            "outer_q_uses_outer_ground_truth": False,
            "outer_fit_or_standardizer_uses_outer_features": False,
            "official_test_role": "exploratory method selection only",
        },
        "geometry": {
            "images": len(stage_a["images"]),
            "cells": len(stage_a["cells"]),
            "eligible_cells": int(np.count_nonzero(geometry["eligible"])),
            "baseline_k1_crop_forwards": int(
                np.asarray(geometry["baseline_crop_counts"]).sum()
            ),
        },
        "endpoint_validation": {
            "per_image_full_k1_sums_to_aggregate": True,
            "nested_per_image_confusion_matches_aggregate_reconstruction": True,
            "k1": k1_metrics,
            "matched_k2": k2_metrics,
            "k4": k4_metrics,
        },
        "nested_cross_fitted_policy": {
            "levels_by_cell": nested["levels_by_cell"],
            "ridge_fit_count": nested["ridge_fit_count"],
            "chosen_q_counts": nested["chosen_q_counts"],
            "outer_folds": nested["outer_folds"],
            "closure": nested["closure"],
            "cost": nested["cost"],
            "full_image": candidate_metrics,
        },
        "matched_exact_cost_random_control": random_control,
        "ridge_gate_decision": decision,
        "explicit_non_claims": [
            "H3 is confirmed",
            "a deployable router exists",
            "K2->K4 is predictable",
            "small/thin structure is closed for this ridge policy",
            "ridge-policy latency has been measured",
            "official-test nested cross-fit is independent paper-ready validation",
        ],
        "next_step": next_step_for_ridge(decision),
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "ridge_common_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_h3_ridge_common.py"
            ),
            "simple_common_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_h3_screen_common.py"
            ),
            "numpy": np.__version__,
            "inference_batch_size": INFERENCE_BATCH_SIZE,
        },
        "runtime": {
            "nested_cross_fit_seconds": nested_seconds,
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
                "k4_gain_retention": decision["observed"]["k4_gain_retention"],
                "random_p95_miou_percent": decision["observed"][
                    "random_p95_miou"
                ]
                * 100.0,
                "physical_cost": decision["observed"]["overall_physical_cost"],
                "chosen_q_counts": nested["chosen_q_counts"],
                "ridge_fit_count": nested["ridge_fit_count"],
                "output_path": str(args.output_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
