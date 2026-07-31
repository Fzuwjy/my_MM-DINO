"""Streaming statistics for normal-slide crop-response disagreement.

The public MM-DINO ``slide_inference`` averages overlapping crop logits and
therefore discards the variation between crop responses.  This module keeps
that released endpoint unchanged while accumulating three GT-free diagnostics:

* normalized generalized Jensen-Shannon divergence between crop probabilities;
* normalized argmax-vote disagreement;
* semantic-boundary response disagreement.

The implementation is deliberately NumPy-only.  A live runner can copy each
sealed float32 crop output to host memory and discard it immediately, while a
cache replay can feed the exact same accumulator from persisted raw crops.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np


SCORE_NAMES = (
    "overlap_jsd_normalized",
    "argmax_vote_disagreement_normalized",
    "boundary_response_disagreement",
)

RESPONSE_SCORE_NAMES = (
    "k1_minus_k2_entropy",
    "k2_minus_k1_margin",
    "k1_kx_jsd_normalized",
    "k1_kx_argmax_flip_rate",
    "k1_k2_argmax_flip_rate",
)


def array_sha256(array: np.ndarray) -> str:
    """Hash a contiguous array using the same byte-level convention as Stage A."""

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Return class-axis float32 softmax probabilities for one raw crop."""

    values = np.asarray(logits)
    if values.ndim != 3:
        raise ValueError("crop logits must have shape [class,height,width]")
    if values.dtype != np.float32:
        raise TypeError("crop logits must preserve the sealed float32 dtype")
    if values.shape[0] < 2 or min(values.shape[1:]) <= 0:
        raise ValueError("crop logits have an invalid shape")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("crop logits contain non-finite values")

    shifted = np.array(values, dtype=np.float32, copy=True, order="C")
    shifted -= shifted.max(axis=0, keepdims=True)
    np.exp(shifted, out=shifted)
    denominator = shifted.sum(axis=0, keepdims=True, dtype=np.float32)
    if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError("softmax denominator is invalid")
    shifted /= denominator
    return shifted


def semantic_boundary_observations(
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-crop boundary bits and their valid-observation mask.

    A four-neighbour semantic boundary is evaluated only at crop-interior
    pixels.  The outer one-pixel ring has incomplete crop-local neighbourhoods
    and is excluded rather than being interpreted as a non-boundary vote.
    """

    values = np.asarray(prediction)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("prediction must be a non-empty 2D array")
    boundary = np.zeros(values.shape, dtype=np.uint8)
    valid = np.zeros(values.shape, dtype=np.uint8)
    if values.shape[0] < 3 or values.shape[1] < 3:
        return boundary, valid

    center = values[1:-1, 1:-1]
    boundary[1:-1, 1:-1] = (
        (center != values[:-2, 1:-1])
        | (center != values[2:, 1:-1])
        | (center != values[1:-1, :-2])
        | (center != values[1:-1, 2:])
    )
    valid[1:-1, 1:-1] = 1
    return boundary, valid


def _bounds4(value: Sequence[int], *, name: str) -> tuple[int, int, int, int]:
    if len(value) != 4:
        raise ValueError(f"{name} must contain y0,y1,x0,x1")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integer bounds") from error
    y0, y1, x0, x1 = result
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError(f"{name} is invalid: {result}")
    return result


def _masked_cell_reduction(
    values: np.ndarray,
    valid: np.ndarray,
    cell_bounds: np.ndarray,
    common_bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce a common-support score map over global ownership rectangles."""

    score = np.asarray(values)
    mask = np.asarray(valid)
    bounds = np.asarray(cell_bounds)
    cy0, cy1, cx0, cx1 = common_bounds
    expected_shape = (cy1 - cy0, cx1 - cx0)
    if score.shape != expected_shape or mask.shape != expected_shape:
        raise ValueError("score/mask shape differs from common support")
    if score.ndim != 2 or mask.dtype != np.bool_:
        raise TypeError("score must be 2D and valid mask must be bool")
    if bounds.ndim != 2 or bounds.shape[1] != 4:
        raise ValueError("cell bounds must have shape [cell,4]")
    if np.any(~np.isfinite(score)):
        raise ValueError("score map contains non-finite values")

    means = np.full(len(bounds), np.nan, dtype=np.float64)
    counts = np.zeros(len(bounds), dtype=np.int64)
    areas = np.zeros(len(bounds), dtype=np.int64)
    for index, raw in enumerate(bounds):
        y0, y1, x0, x1 = (int(item) for item in raw)
        iy0, iy1 = max(y0, cy0), min(y1, cy1)
        ix0, ix1 = max(x0, cx0), min(x1, cx1)
        if iy0 >= iy1 or ix0 >= ix1:
            continue
        local = (
            slice(iy0 - cy0, iy1 - cy0),
            slice(ix0 - cx0, ix1 - cx0),
        )
        local_mask = mask[local]
        count = int(np.count_nonzero(local_mask))
        areas[index] = int(local_mask.size)
        counts[index] = count
        if count:
            means[index] = float(
                np.mean(score[local][local_mask], dtype=np.float64)
            )
    return means, counts, areas


def _masked_cell_top_fraction_mean(
    values: np.ndarray,
    valid: np.ndarray,
    cell_bounds: np.ndarray,
    common_bounds: tuple[int, int, int, int],
    *,
    fraction: float,
) -> np.ndarray:
    """Return a report-only upper-tail mean for every ownership rectangle."""

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0,1]")
    score = np.asarray(values)
    mask = np.asarray(valid)
    bounds = np.asarray(cell_bounds)
    cy0, cy1, cx0, cx1 = common_bounds
    if score.shape != mask.shape or mask.dtype != np.bool_:
        raise ValueError("tail score and bool mask must have identical shapes")
    result = np.full(len(bounds), np.nan, dtype=np.float64)
    for index, raw in enumerate(bounds):
        y0, y1, x0, x1 = (int(item) for item in raw)
        iy0, iy1 = max(y0, cy0), min(y1, cy1)
        ix0, ix1 = max(x0, cx0), min(x1, cx1)
        if iy0 >= iy1 or ix0 >= ix1:
            continue
        local = (
            slice(iy0 - cy0, iy1 - cy0),
            slice(ix0 - cx0, ix1 - cx0),
        )
        selected = score[local][mask[local]]
        if not len(selected):
            continue
        count = max(1, int(math.ceil(len(selected) * float(fraction))))
        split = len(selected) - count
        upper = np.partition(selected, split)[split:]
        result[index] = float(np.mean(upper, dtype=np.float64))
    return result


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    return -(
        probabilities
        * np.log(np.clip(probabilities, np.float32(1e-12), None))
    ).sum(axis=0, dtype=np.float32)


def _margin(probabilities: np.ndarray) -> np.ndarray:
    partitioned = np.partition(probabilities, -2, axis=0)
    return (
        partitioned[-1].astype(np.float32, copy=False)
        - partitioned[-2].astype(np.float32, copy=False)
    )


def phase_response_cell_summaries(
    k1_logits: np.ndarray,
    kx_logits: np.ndarray,
    cell_bounds: np.ndarray,
    common_bounds: Sequence[int],
) -> dict[str, Any]:
    """Reduce frozen K1/x8/K2 response maps over ownership cells.

    ``K2`` follows the released arithmetic mean of aligned float32 logits.
    Entropy and margin are computed after softmax of that mean.  JSD and class
    flips are response-strength diagnostics only; the signed entropy change is
    the preregistered H4-B primary direction.
    """

    first = np.asarray(k1_logits)
    shifted = np.asarray(kx_logits)
    if first.dtype != np.float32 or shifted.dtype != np.float32:
        raise TypeError("aligned K1/x8 logits must be float32")
    if first.shape != shifted.shape or first.ndim != 3:
        raise ValueError("aligned K1/x8 logits must share [class,height,width]")
    cy0, cy1, cx0, cx1 = _bounds4(common_bounds, name="common bounds")
    if first.shape[1:] != (cy1 - cy0, cx1 - cx0):
        raise ValueError("aligned logits shape differs from common bounds")

    p1 = stable_softmax(first)
    px = stable_softmax(shifted)
    k2_logits = np.asarray((first + shifted) * np.float32(0.5), dtype=np.float32)
    p2 = stable_softmax(k2_logits)
    entropy1 = _entropy(p1)
    entropyx = _entropy(px)
    entropy2 = _entropy(p2)
    mixture = np.asarray((p1 + px) * np.float32(0.5), dtype=np.float32)
    jsd = np.maximum(
        _entropy(mixture) - np.float32(0.5) * (entropy1 + entropyx), 0.0
    ) / np.float32(math.log(2.0))
    np.clip(jsd, 0.0, 1.0, out=jsd)
    prediction1 = first.argmax(axis=0)
    predictionx = shifted.argmax(axis=0)
    prediction2 = k2_logits.argmax(axis=0)
    maps = {
        RESPONSE_SCORE_NAMES[0]: entropy1 - entropy2,
        RESPONSE_SCORE_NAMES[1]: _margin(p2) - _margin(p1),
        RESPONSE_SCORE_NAMES[2]: jsd,
        RESPONSE_SCORE_NAMES[3]: (prediction1 != predictionx).astype(np.float32),
        RESPONSE_SCORE_NAMES[4]: (prediction1 != prediction2).astype(np.float32),
    }
    valid = np.ones(first.shape[1:], dtype=np.bool_)
    means = {}
    counts = None
    areas = None
    for name, values in maps.items():
        score_means, score_counts, score_areas = _masked_cell_reduction(
            np.asarray(values, dtype=np.float32),
            valid,
            np.asarray(cell_bounds),
            (cy0, cy1, cx0, cx1),
        )
        means[name] = score_means
        if counts is None:
            counts, areas = score_counts, score_areas
        elif not np.array_equal(counts, score_counts) or not np.array_equal(
            areas, score_areas
        ):
            raise AssertionError("phase-response cell supports differ by score")
    return {
        "score_means": means,
        "valid_pixels": counts,
        "common_intersection_pixels": areas,
        "map_diagnostics": {
            name: {
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values, dtype=np.float64)),
            }
            for name, values in maps.items()
        },
    }


class LogitSlideAccumulator:
    """Minimal float32 sum/count accumulator for one shifted-canvas phase."""

    def __init__(self, image_shape: Sequence[int], num_classes: int) -> None:
        if len(image_shape) != 2:
            raise ValueError("image_shape must contain height,width")
        self.height, self.width = (int(value) for value in image_shape)
        self.num_classes = int(num_classes)
        if min(self.height, self.width) <= 0 or self.num_classes < 2:
            raise ValueError("image shape and class count must be positive")
        self.logit_sum = np.zeros(
            (self.num_classes, self.height, self.width), dtype=np.float32
        )
        self.crop_count = np.zeros((self.height, self.width), dtype=np.uint8)
        self.windows: list[tuple[int, int, int, int]] = []

    def add_crop(self, logits: np.ndarray, window: Sequence[int]) -> None:
        y0, y1, x0, x1 = _bounds4(window, name="crop window")
        if y1 > self.height or x1 > self.width:
            raise ValueError("crop window exceeds the image")
        values = np.asarray(logits)
        expected = (self.num_classes, y1 - y0, x1 - x0)
        if values.dtype != np.float32 or values.shape != expected:
            raise ValueError("shifted crop logits must preserve float32 shape")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("shifted crop logits contain non-finite values")
        region = (slice(y0, y1), slice(x0, x1))
        self.logit_sum[:, region[0], region[1]] += values
        self.crop_count[region] += 1
        self.windows.append((y0, y1, x0, x1))

    def add_batch(
        self, logits: np.ndarray, windows: Sequence[Sequence[int]]
    ) -> None:
        values = np.asarray(logits)
        if values.ndim != 4 or len(values) != len(windows):
            raise ValueError("shifted batch and windows have different shapes")
        for crop, window in zip(values, windows, strict=True):
            self.add_crop(crop, window)

    def normalized_logits(self, bounds: Sequence[int]) -> np.ndarray:
        y0, y1, x0, x1 = _bounds4(bounds, name="normalization bounds")
        if y1 > self.height or x1 > self.width:
            raise ValueError("normalization bounds exceed the image")
        count = self.crop_count[y0:y1, x0:x1]
        if np.any(count == 0):
            raise AssertionError("requested shifted support contains uncovered pixels")
        result = np.array(
            self.logit_sum[:, y0:y1, x0:x1],
            dtype=np.float32,
            copy=True,
            order="C",
        )
        result /= count[None]
        return result

    def storage_nbytes(self) -> int:
        return int(self.logit_sum.nbytes + self.crop_count.nbytes)


class OverlapDisagreementAccumulator:
    """Accumulate raw normal-slide crops without persisting them."""

    def __init__(self, image_shape: Sequence[int], num_classes: int) -> None:
        if len(image_shape) != 2:
            raise ValueError("image_shape must contain height,width")
        self.height, self.width = (int(value) for value in image_shape)
        self.num_classes = int(num_classes)
        if min(self.height, self.width) <= 0 or self.num_classes < 2:
            raise ValueError("image shape and class count must be positive")

        shape = (self.height, self.width)
        class_shape = (self.num_classes, *shape)
        self.logit_sum = np.zeros(class_shape, dtype=np.float32)
        self.probability_sum = np.zeros(class_shape, dtype=np.float32)
        self.entropy_sum = np.zeros(shape, dtype=np.float32)
        self.class_votes = np.zeros(class_shape, dtype=np.uint8)
        self.boundary_votes = np.zeros(shape, dtype=np.uint8)
        self.boundary_observations = np.zeros(shape, dtype=np.uint8)
        self.crop_count = np.zeros(shape, dtype=np.uint8)
        self.windows: list[tuple[int, int, int, int]] = []

    def add_crop(
        self, logits: np.ndarray, window: Sequence[int]
    ) -> None:
        """Add one crop in the exact released row-major accumulation order."""

        y0, y1, x0, x1 = _bounds4(window, name="crop window")
        if y1 > self.height or x1 > self.width:
            raise ValueError("crop window exceeds the image")
        values = np.asarray(logits)
        expected = (self.num_classes, y1 - y0, x1 - x0)
        if values.shape != expected:
            raise ValueError(f"crop logits shape {values.shape} differs from {expected}")
        probabilities = stable_softmax(values)
        entropy = -(
            probabilities
            * np.log(np.clip(probabilities, np.float32(1e-12), None))
        ).sum(axis=0, dtype=np.float32)
        prediction = values.argmax(axis=0)
        boundary, boundary_valid = semantic_boundary_observations(prediction)

        region = (slice(y0, y1), slice(x0, x1))
        self.logit_sum[:, region[0], region[1]] += values
        self.probability_sum[:, region[0], region[1]] += probabilities
        self.entropy_sum[region] += entropy
        for class_index in range(self.num_classes):
            self.class_votes[class_index, region[0], region[1]] += (
                prediction == class_index
            )
        self.boundary_votes[region] += boundary
        self.boundary_observations[region] += boundary_valid
        self.crop_count[region] += 1
        self.windows.append((y0, y1, x0, x1))

    def add_batch(
        self,
        logits: np.ndarray,
        windows: Sequence[Sequence[int]],
    ) -> None:
        values = np.asarray(logits)
        if values.ndim != 4 or values.shape[1] != self.num_classes:
            raise ValueError("batch logits must have shape [batch,class,height,width]")
        if len(values) != len(windows):
            raise ValueError("batch logits and windows have different lengths")
        for crop, window in zip(values, windows, strict=True):
            self.add_crop(crop, window)

    def endpoint_prediction(self) -> np.ndarray:
        """Return the released count-normalized-logit argmax as int64."""

        if np.any(self.crop_count == 0):
            raise AssertionError("normal slide has uncovered pixels")
        # Dividing every class at a pixel by the same positive crop count does
        # not change argmax.  Avoiding the full normalized-logit temporary also
        # keeps the live runner's host-memory peak bounded.
        return np.ascontiguousarray(
            self.logit_sum.argmax(axis=0).astype(np.int64, copy=False)
        )

    def residue_metadata(self, patch_size: int = 16) -> dict[str, Any]:
        size = int(patch_size)
        if size <= 0:
            raise ValueError("patch_size must be positive")
        residues = [(y0 % size, x0 % size) for y0, _, x0, _ in self.windows]
        return {
            "patch_size": size,
            "per_crop_origin_residue_yx": [list(value) for value in residues],
            "unique_origin_residue_yx": [
                list(value) for value in sorted(set(residues))
            ],
            "unique_origin_residue_count": len(set(residues)),
            "role": "mechanism metadata only; not a separately screened score",
        }

    def cell_summaries(
        self,
        cell_bounds: np.ndarray,
        common_bounds: Sequence[int],
    ) -> dict[str, Any]:
        """Finalize the three frozen maps and ownership-cell reductions."""

        if np.any(self.crop_count == 0):
            raise AssertionError("normal slide has uncovered pixels")
        cy0, cy1, cx0, cx1 = _bounds4(common_bounds, name="common bounds")
        if cy1 > self.height or cx1 > self.width:
            raise ValueError("common bounds exceed the image")
        common = (slice(cy0, cy1), slice(cx0, cx1))
        count = self.crop_count[common]
        overlap_valid = count >= 2

        mean_probabilities = np.array(
            self.probability_sum[:, common[0], common[1]],
            dtype=np.float32,
            copy=True,
            order="C",
        )
        mean_probabilities /= count[None]
        entropy_of_mean = -(
            mean_probabilities
            * np.log(np.clip(mean_probabilities, np.float32(1e-12), None))
        ).sum(axis=0, dtype=np.float32)
        mean_entropy = self.entropy_sum[common] / count
        jsd = np.maximum(entropy_of_mean - mean_entropy, 0.0)
        jsd_normalizer = np.ones(count.shape, dtype=np.float32)
        jsd_normalizer[overlap_valid] = np.log(
            count[overlap_valid].astype(np.float32)
        )
        jsd_normalized = np.zeros(count.shape, dtype=np.float32)
        np.divide(
            jsd,
            jsd_normalizer,
            out=jsd_normalized,
            where=overlap_valid,
        )
        np.clip(jsd_normalized, 0.0, 1.0, out=jsd_normalized)

        maximum_votes = self.class_votes[:, common[0], common[1]].max(axis=0)
        vote_raw = 1.0 - maximum_votes.astype(np.float32) / count
        vote_maximum = np.zeros(count.shape, dtype=np.float32)
        vote_maximum[overlap_valid] = (
            1.0 - 1.0 / count[overlap_valid].astype(np.float32)
        )
        vote_normalized = np.zeros(count.shape, dtype=np.float32)
        np.divide(
            vote_raw,
            vote_maximum,
            out=vote_normalized,
            where=overlap_valid & (vote_maximum > 0),
        )
        np.clip(vote_normalized, 0.0, 1.0, out=vote_normalized)

        boundary_count = self.boundary_observations[common]
        boundary_valid = boundary_count >= 2
        boundary_probability = np.zeros(count.shape, dtype=np.float32)
        np.divide(
            self.boundary_votes[common],
            boundary_count,
            out=boundary_probability,
            where=boundary_count > 0,
        )
        boundary_disagreement = (
            4.0 * boundary_probability * (1.0 - boundary_probability)
        ).astype(np.float32, copy=False)
        boundary_disagreement[~boundary_valid] = 0.0

        maps_and_masks = {
            SCORE_NAMES[0]: (jsd_normalized, overlap_valid),
            SCORE_NAMES[1]: (vote_normalized, overlap_valid),
            SCORE_NAMES[2]: (boundary_disagreement, boundary_valid),
        }
        score_means = {}
        score_counts = {}
        score_areas = {}
        for name, (score, valid) in maps_and_masks.items():
            means, counts, areas = _masked_cell_reduction(
                score,
                valid,
                np.asarray(cell_bounds),
                (cy0, cy1, cx0, cx1),
            )
            score_means[name] = means
            score_counts[name] = counts
            score_areas[name] = areas

        jsd_top10 = _masked_cell_top_fraction_mean(
            jsd_normalized,
            overlap_valid,
            np.asarray(cell_bounds),
            (cy0, cy1, cx0, cx1),
            fraction=0.10,
        )

        common_histogram = {
            str(value): int(np.count_nonzero(count == value))
            for value in sorted(int(item) for item in np.unique(count))
        }
        return {
            "score_means": score_means,
            "score_valid_pixels": score_counts,
            "score_common_intersection_pixels": score_areas,
            "report_only": {
                "overlap_jsd_normalized_top10pct_mean": jsd_top10,
                "role": (
                    "frozen dilution diagnostic only; never used for ranking, "
                    "q selection, random control, or H4-A Go/No-Go"
                ),
            },
            "coverage": {
                "common_pixels": int(count.size),
                "overlap_pixels": int(np.count_nonzero(overlap_valid)),
                "overlap_fraction": float(np.mean(overlap_valid)),
                "boundary_multi_observation_pixels": int(
                    np.count_nonzero(boundary_valid)
                ),
                "boundary_multi_observation_fraction": float(
                    np.mean(boundary_valid)
                ),
                "crop_count_histogram": common_histogram,
                "mean_crop_count": float(np.mean(count, dtype=np.float64)),
            },
        }

    def storage_nbytes(self) -> dict[str, int]:
        arrays = {
            "logit_sum": self.logit_sum,
            "probability_sum": self.probability_sum,
            "entropy_sum": self.entropy_sum,
            "class_votes": self.class_votes,
            "boundary_votes": self.boundary_votes,
            "boundary_observations": self.boundary_observations,
            "crop_count": self.crop_count,
        }
        result = {name: int(value.nbytes) for name, value in arrays.items()}
        result["total"] = int(sum(result.values()))
        return result


def finite_or_none(value: float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None
