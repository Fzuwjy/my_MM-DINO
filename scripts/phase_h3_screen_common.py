"""Deployment-safe offline helpers for the first H3 K1-to-K2 screen.

This module deliberately has no target, confusion, prediction-quality, or
K4-facing API.  It turns caller-supplied K1-visible scores into a K1/K2 cell
map, audits its exact x8 dependency closure, and builds exact-cost random
controls without observing ground truth.

The score policy is frozen as follows, independently for every image:

1. rank geometry-eligible cells by descending score and then global cell index;
2. select exactly ``floor(q * eligible_count)`` ranked cells;
3. form their x8 unique-crop closure; and
4. promote every remaining eligible cell whose x8 dependency is already fully
   covered.  These spill promotions are genuinely zero-cost.

Physical accounting mirrors the live sparse executor: the normal K1 workload
is unchanged, while each image's selected x8 pool is padded to a fixed model
batch size.  Padding outputs are discarded but their model samples count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

import numpy as np

from scripts.phase_closure_common import PHASE_NAMES, summarize_closure


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _fraction(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return result


def _sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        return tuple(value.tolist())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


def _validate_geometry(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the closure fields used by all public helpers.

    ``phase_closure_common`` intentionally accepts a compact, trusted internal
    geometry mapping.  H3 artifacts are longer-lived external inputs, so this
    layer validates their index partition and dependency-mask bounds before any
    routing decision is made.
    """

    if not isinstance(geometry, Mapping):
        raise TypeError("geometry must be a mapping")
    if tuple(geometry.get("phase_names", ())) != PHASE_NAMES:
        raise ValueError(f"geometry phase_names must equal {PHASE_NAMES}")

    baseline_raw = np.asarray(geometry.get("baseline_crop_counts"))
    if (
        baseline_raw.ndim != 1
        or baseline_raw.size == 0
        or not np.issubdtype(baseline_raw.dtype, np.integer)
        or baseline_raw.dtype == np.bool_
    ):
        raise TypeError("baseline_crop_counts must be a non-empty integer vector")
    baseline = np.ascontiguousarray(baseline_raw, dtype=np.int64)
    if np.any(baseline <= 0):
        raise ValueError("baseline_crop_counts must be positive")
    image_count = len(baseline)

    sample_names = _sequence(geometry.get("sample_names"), name="sample_names")
    if len(sample_names) != image_count:
        raise ValueError("sample_names length differs from image count")
    if any(not isinstance(name, str) or not name for name in sample_names):
        raise ValueError("sample_names must contain non-empty strings")

    image_ids_raw = np.asarray(geometry.get("image_ids"))
    if (
        image_ids_raw.ndim != 1
        or image_ids_raw.size == 0
        or not np.issubdtype(image_ids_raw.dtype, np.integer)
        or image_ids_raw.dtype == np.bool_
    ):
        raise TypeError("image_ids must be a non-empty integer vector")
    image_ids = np.ascontiguousarray(image_ids_raw, dtype=np.int64)
    if np.any(image_ids < 0) or np.any(image_ids >= image_count):
        raise ValueError("image_ids contain an out-of-range image index")
    cell_count = len(image_ids)

    eligible_raw = np.asarray(geometry.get("eligible"))
    if eligible_raw.shape != (cell_count,) or eligible_raw.dtype != np.bool_:
        raise TypeError("eligible must be a bool vector matching image_ids")
    eligible = np.ascontiguousarray(eligible_raw, dtype=bool)

    groups = _sequence(geometry.get("cells_by_image"), name="cells_by_image")
    if len(groups) != image_count:
        raise ValueError("cells_by_image length differs from image count")
    normalized_groups: list[tuple[int, ...]] = []
    seen: list[int] = []
    for image_index, raw_group in enumerate(groups):
        group = _sequence(raw_group, name=f"cells_by_image[{image_index}]")
        normalized: list[int] = []
        for position, raw_index in enumerate(group):
            cell_index = _integer(
                raw_index,
                name=f"cells_by_image[{image_index}][{position}]",
            )
            if cell_index >= cell_count:
                raise ValueError("cells_by_image contains an out-of-range cell index")
            if int(image_ids[cell_index]) != image_index:
                raise ValueError("cells_by_image disagrees with image_ids")
            normalized.append(cell_index)
            seen.append(cell_index)
        normalized_groups.append(tuple(normalized))
    if len(seen) != cell_count or sorted(seen) != list(range(cell_count)):
        raise ValueError("cells_by_image must partition every global cell exactly once")

    dependencies = geometry.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise TypeError("dependencies must be a mapping")
    normalized_dependencies: dict[str, tuple[int, ...]] = {}
    for phase_name in PHASE_NAMES:
        masks = _sequence(
            dependencies.get(phase_name), name=f"dependencies.{phase_name}"
        )
        if len(masks) != cell_count:
            raise ValueError(f"dependencies.{phase_name} length differs from cell count")
        normalized_masks: list[int] = []
        for cell_index, raw_mask in enumerate(masks):
            mask = _integer(
                raw_mask, name=f"dependencies.{phase_name}[{cell_index}]"
            )
            image_index = int(image_ids[cell_index])
            if mask >> int(baseline[image_index]):
                raise ValueError(
                    f"dependencies.{phase_name}[{cell_index}] references a crop "
                    "outside its image"
                )
            normalized_masks.append(mask)
        normalized_dependencies[phase_name] = tuple(normalized_masks)

    return {
        "baseline_crop_counts": baseline,
        "sample_names": sample_names,
        "image_ids": image_ids,
        "eligible": eligible,
        "cells_by_image": tuple(normalized_groups),
        "dependencies": normalized_dependencies,
        "cell_count": cell_count,
        "image_count": image_count,
    }


def _validate_scores(
    scores: Sequence[float | None] | np.ndarray,
    *,
    cell_count: int,
    eligible: np.ndarray,
) -> np.ndarray:
    raw = (
        tuple(scores.tolist())
        if isinstance(scores, np.ndarray)
        else _sequence(scores, name="scores")
    )
    if len(raw) != cell_count:
        raise ValueError(f"scores must have length {cell_count}")
    values = np.full(cell_count, -np.inf, dtype=np.float64)
    for cell_index, value in enumerate(raw):
        if value is None:
            if eligible[cell_index]:
                raise ValueError("every geometry-eligible cell must have a score")
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError("scores must contain real numbers or None for ineligible cells")
        score = float(value)
        if eligible[cell_index]:
            if not math.isfinite(score):
                raise ValueError("geometry-eligible scores must be finite")
        elif math.isnan(score) or math.isinf(score) and score > 0:
            raise ValueError("ineligible scores may be finite, -inf, or None")
        values[cell_index] = score
    return values


def _score_selection_for_image(
    values: np.ndarray,
    q: float,
    validated: Mapping[str, Any],
    image_index: int,
) -> dict[str, Any]:
    group = np.asarray(validated["cells_by_image"][image_index], dtype=np.int64)
    eligible = validated["eligible"]
    eligible_indices = group[eligible[group]]
    quota = int(math.floor(q * len(eligible_indices)))
    if quota:
        order = np.lexsort((eligible_indices, -values[eligible_indices]))
        initial = eligible_indices[order[:quota]]
    else:
        initial = np.empty(0, dtype=np.int64)

    dependencies = validated["dependencies"]["x8"]
    used_mask = 0
    for cell_index in initial:
        used_mask |= dependencies[int(cell_index)]
    initial_set = {int(index) for index in initial}
    spill = np.asarray(
        [
            int(cell_index)
            for cell_index in eligible_indices
            if int(cell_index) not in initial_set
            and dependencies[int(cell_index)] & ~used_mask == 0
        ],
        dtype=np.int64,
    )
    # Spill order is an audit convention, not a routing choice.
    spill.sort()
    return {
        "image_index": image_index,
        "sample_name": validated["sample_names"][image_index],
        "eligible_count": int(len(eligible_indices)),
        "initial_quota_count": quota,
        "initial_selected_indices": initial.copy(),
        "zero_cost_selected_indices": spill,
        "selected_count": int(len(initial) + len(spill)),
        "unique_x8_crop_count": int(used_mask.bit_count()),
        "x8_crop_mask": used_mask,
    }


def score_ranked_k2_levels_for_image(
    scores: Sequence[float | None] | np.ndarray,
    q: float,
    geometry: Mapping[str, Any],
    image_index: int,
) -> dict[str, Any]:
    """Build one image's score-ranked K1/K2 map in global cell coordinates.

    Cells from every other image remain K1.  This makes the returned
    ``levels_by_cell`` directly valid for :func:`summarize_closure` and avoids
    ambiguity when global cell indices are not contiguous within an image.
    """

    validated = _validate_geometry(geometry)
    fraction = _fraction(q, name="q")
    index = _integer(image_index, name="image_index")
    if index >= validated["image_count"]:
        raise ValueError("image_index is out of range")
    values = _validate_scores(
        scores,
        cell_count=validated["cell_count"],
        eligible=validated["eligible"],
    )
    record = _score_selection_for_image(values, fraction, validated, index)
    levels = np.ones(validated["cell_count"], dtype=np.int64)
    levels[record["initial_selected_indices"]] = 2
    levels[record["zero_cost_selected_indices"]] = 2
    closure = summarize_closure(levels, geometry, include_crop_ids=False)
    realized = closure["per_image"][index]["extra_crop_forwards_by_phase"]
    if realized != {"x8": record["unique_x8_crop_count"], "y8": 0, "xy8": 0}:
        raise AssertionError("score selection disagrees with its x8 closure")
    return {
        "selection_rule": (
            "per-image descending score, lower global-index tie, floor(q*eligible), "
            "then all fully-covered zero-cost x8 spill cells"
        ),
        "q": fraction,
        "levels_by_cell": levels,
        "initial_selected_indices": record["initial_selected_indices"],
        "zero_cost_selected_indices": record["zero_cost_selected_indices"],
        "per_image": (record,),
        "closure": closure,
    }


def score_ranked_k2_levels(
    scores: Sequence[float | None] | np.ndarray,
    q: float,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the frozen score-ranked K1/K2 map independently for every image."""

    validated = _validate_geometry(geometry)
    fraction = _fraction(q, name="q")
    values = _validate_scores(
        scores,
        cell_count=validated["cell_count"],
        eligible=validated["eligible"],
    )
    levels = np.ones(validated["cell_count"], dtype=np.int64)
    records = tuple(
        _score_selection_for_image(values, fraction, validated, image_index)
        for image_index in range(validated["image_count"])
    )
    for record in records:
        levels[record["initial_selected_indices"]] = 2
        levels[record["zero_cost_selected_indices"]] = 2
    closure = summarize_closure(levels, geometry, include_crop_ids=False)
    for record, realized in zip(records, closure["per_image"], strict=True):
        phase_counts = realized["extra_crop_forwards_by_phase"]
        expected = {"x8": record["unique_x8_crop_count"], "y8": 0, "xy8": 0}
        if phase_counts != expected:
            raise AssertionError("score selection disagrees with its x8 closure")
    initial = np.concatenate(
        [record["initial_selected_indices"] for record in records]
    ).astype(np.int64, copy=False)
    spill = np.concatenate(
        [record["zero_cost_selected_indices"] for record in records]
    ).astype(np.int64, copy=False)
    return {
        "selection_rule": (
            "per-image descending score, lower global-index tie, floor(q*eligible), "
            "then all fully-covered zero-cost x8 spill cells"
        ),
        "q": fraction,
        "levels_by_cell": levels,
        "initial_selected_indices": initial,
        "zero_cost_selected_indices": spill,
        "per_image": records,
        "closure": closure,
    }


def physical_cost_summary(
    levels: Sequence[int] | np.ndarray,
    geometry: Mapping[str, Any],
    batch_size: int = 8,
) -> dict[str, Any]:
    """Summarize logical x8 closure cost and fixed-batch physical cost.

    Only shifted K2 samples are padded.  The normal K1 slide is the unchanged
    baseline workload.  A zero-sized shifted pool schedules no model call and
    therefore has zero padding.
    """

    _validate_geometry(geometry)
    size = _integer(batch_size, name="batch_size", minimum=1)
    raw_levels = np.asarray(levels)
    if (
        raw_levels.ndim != 1
        or not np.issubdtype(raw_levels.dtype, np.integer)
        or raw_levels.dtype == np.bool_
    ):
        raise TypeError("levels must be a one-dimensional integer vector")
    if np.any(~np.isin(raw_levels, (1, 2))):
        raise ValueError("H3 K2 physical accounting accepts only K1/K2 levels")
    closure = summarize_closure(raw_levels, geometry, include_crop_ids=False)
    if closure["extra_crop_forwards_by_phase"]["y8"] or closure[
        "extra_crop_forwards_by_phase"
    ]["xy8"]:
        raise AssertionError("a K1/K2 map unexpectedly scheduled non-x8 phases")

    per_image: list[dict[str, Any]] = []
    for image in closure["per_image"]:
        baseline = int(image["baseline_k1_crop_forwards"])
        selected = int(image["extra_crop_forwards_by_phase"]["x8"])
        processed = 0 if selected == 0 else math.ceil(selected / size) * size
        padding = processed - selected
        per_image.append(
            {
                "image_index": int(image["image_index"]),
                "sample_name": image["sample_name"],
                "baseline_crop_samples": baseline,
                "selected_unique_x8_crop_samples": selected,
                "processed_x8_crop_samples_including_padding": processed,
                "padding_x8_crop_samples": padding,
                "shifted_batch_calls": processed // size,
                "logical_unique_crop_cost_ratio": float(
                    (baseline + selected) / baseline
                ),
                "physical_model_sample_cost_ratio": float(
                    (baseline + processed) / baseline
                ),
            }
        )

    baseline_total = sum(item["baseline_crop_samples"] for item in per_image)
    selected_total = sum(
        item["selected_unique_x8_crop_samples"] for item in per_image
    )
    processed_total = sum(
        item["processed_x8_crop_samples_including_padding"] for item in per_image
    )
    padding_total = sum(item["padding_x8_crop_samples"] for item in per_image)
    if selected_total != int(closure["extra_crop_forwards"]):
        raise AssertionError("physical ledger disagrees with summarize_closure")
    if processed_total != selected_total + padding_total:
        raise AssertionError("physical padding ledger is inconsistent")
    return {
        "batch_size": size,
        "padding_policy": (
            "per-image global x8 pool; cyclic final-real-batch padding; padding "
            "outputs discarded but physical model samples counted"
        ),
        "baseline_crop_samples": baseline_total,
        "selected_unique_x8_crop_samples": selected_total,
        "processed_x8_crop_samples_including_padding": processed_total,
        "padding_x8_crop_samples": padding_total,
        "shifted_batch_calls": processed_total // size,
        "logical_unique_crop_cost_ratio": float(
            (baseline_total + selected_total) / baseline_total
        ),
        "physical_model_sample_cost_ratio": float(
            (baseline_total + processed_total) / baseline_total
        ),
        "maximum_per_image_logical_cost_ratio": max(
            item["logical_unique_crop_cost_ratio"] for item in per_image
        ),
        "maximum_per_image_physical_cost_ratio": max(
            item["physical_model_sample_cost_ratio"] for item in per_image
        ),
        "padding_is_counted_as_model_compute": True,
        "per_image": tuple(per_image),
        "closure": closure,
    }


def _validate_targets(
    values: Sequence[int] | np.ndarray,
    *,
    baseline: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(values)
    if (
        raw.shape != baseline.shape
        or not np.issubdtype(raw.dtype, np.integer)
        or raw.dtype == np.bool_
    ):
        raise TypeError("target_unique_x8_by_image must be an integer image vector")
    targets = np.ascontiguousarray(raw, dtype=np.int64)
    if np.any(targets < 0):
        raise ValueError("target_unique_x8_by_image must be non-negative")
    if np.any(targets > baseline):
        raise ValueError("an x8 unique-crop target exceeds its image crop count")
    return targets


def random_exact_cost_k2_levels(
    geometry: Mapping[str, Any],
    target_unique_x8_by_image: Sequence[int] | np.ndarray,
    seed: int,
    replicate: int,
    *,
    max_attempts: int = 128,
) -> dict[str, Any]:
    """Build a reproducible GT-free K1/K2 random-priority exact-cost control.

    Each attempt samples one priority permutation per image.  The walk accepts
    a cell when its marginal unique x8 crop cost fits the remaining target.
    After the priority pass it exhausts every now-covered zero-cost cell.
    Under-spending is never accepted: retries are deterministic, and exhausting
    ``max_attempts`` raises instead of silently emitting a cheaper control.
    """

    validated = _validate_geometry(geometry)
    targets = _validate_targets(
        target_unique_x8_by_image,
        baseline=validated["baseline_crop_counts"],
    )
    base_seed = _integer(seed, name="seed")
    replicate_index = _integer(replicate, name="replicate")
    attempt_limit = _integer(max_attempts, name="max_attempts", minimum=1)
    levels = np.ones(validated["cell_count"], dtype=np.int64)
    attempts = np.zeros(validated["image_count"], dtype=np.int64)
    accepted_records: list[dict[str, Any]] = []
    dependencies = validated["dependencies"]["x8"]
    eligible = validated["eligible"]

    for image_index, raw_group in enumerate(validated["cells_by_image"]):
        group = np.asarray(raw_group, dtype=np.int64)
        candidates = group[eligible[group]]
        target = int(targets[image_index])
        available_mask = 0
        for cell_index in candidates:
            available_mask |= dependencies[int(cell_index)]
        if target > available_mask.bit_count():
            raise RuntimeError(
                f"image {image_index} cannot exactly spend x8 target {target}: "
                "eligible dependency union is smaller"
            )

        rng = np.random.default_rng(
            np.random.SeedSequence([base_seed, replicate_index, image_index])
        )
        accepted: dict[str, Any] | None = None
        for attempt in range(1, attempt_limit + 1):
            order = rng.permutation(candidates)
            used_mask = 0
            selected: list[int] = []
            selected_set: set[int] = set()
            for raw_cell_index in order:
                cell_index = int(raw_cell_index)
                addition = dependencies[cell_index] & ~used_mask
                if used_mask.bit_count() + addition.bit_count() <= target:
                    used_mask |= addition
                    selected.append(cell_index)
                    selected_set.add(cell_index)

            spill = [
                int(cell_index)
                for cell_index in candidates
                if int(cell_index) not in selected_set
                and dependencies[int(cell_index)] & ~used_mask == 0
            ]
            spill.sort()
            selected.extend(spill)
            if used_mask.bit_count() == target:
                accepted = {
                    "image_index": image_index,
                    "sample_name": validated["sample_names"][image_index],
                    "target_unique_x8_crop_count": target,
                    "realized_unique_x8_crop_count": int(used_mask.bit_count()),
                    "attempt": attempt,
                    "priority_order": np.asarray(order, dtype=np.int64),
                    "selected_indices": np.asarray(selected, dtype=np.int64),
                    "final_zero_cost_selected_indices": np.asarray(
                        spill, dtype=np.int64
                    ),
                }
                break
        if accepted is None:
            raise RuntimeError(
                f"random K2 priority walk could not exactly spend image "
                f"{image_index} x8 target {target} after {attempt_limit} attempts"
            )
        levels[accepted["selected_indices"]] = 2
        attempts[image_index] = int(accepted["attempt"])
        accepted_records.append(accepted)

    closure = summarize_closure(levels, geometry, include_crop_ids=False)
    realized = np.asarray(
        [
            image["extra_crop_forwards_by_phase"]["x8"]
            for image in closure["per_image"]
        ],
        dtype=np.int64,
    )
    if not np.array_equal(realized, targets):
        raise AssertionError("random K2 walk did not match every exact x8 target")
    if closure["extra_crop_forwards_by_phase"]["y8"] or closure[
        "extra_crop_forwards_by_phase"
    ]["xy8"]:
        raise AssertionError("random K2 walk unexpectedly scheduled non-x8 phases")
    return {
        "selection_rule": (
            "GT-free per-image random-priority K1-to-K2 walk with exact unique-x8 "
            "target, deterministic retries, and final zero-cost exhaustion"
        ),
        "seed": base_seed,
        "replicate": replicate_index,
        "max_attempts": attempt_limit,
        "levels_by_cell": levels,
        "attempts_by_image": attempts,
        "target_unique_x8_by_image": targets,
        "per_image": tuple(accepted_records),
        "closure": closure,
    }


__all__ = (
    "physical_cost_summary",
    "random_exact_cost_k2_levels",
    "score_ranked_k2_levels",
    "score_ranked_k2_levels_for_image",
)
