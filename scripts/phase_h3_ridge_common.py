"""Leakage-safe mathematics for the single preregistered H3 ridge gate.

The gate predicts a geometry-eligible cell's local K1-to-K2 value from exactly
three K1-visible statistics.  Its regression target is recomputed inside every
fit fold from that fold's *full-image* pooled K1 confusion.  In particular,
this module never reads a stored singleton-oracle score or a Stage-B0 action
map.

Prediction is intentionally a separate operation.  It rejects image overlap
with the model's fit images and reads only the three K1 feature fields from
the requested prediction images; held-out cell confusions are not touched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from numbers import Real
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "k1_entropy",
    "k1_negative_margin",
    "k1_predicted_boundary_density",
)
RIDGE_LAMBDA = 1.0
MODEL_TYPE = "h3_k1_to_k2_singleton_ridge"
MODEL_SCHEMA_VERSION = 1


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            raise ValueError(f"{name} must not be scalar")
        return tuple(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)


def _validated_basic_cells(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], np.ndarray, np.ndarray, int]:
    records = _sequence(cells, name="cells")
    if not records:
        raise ValueError("cells must not be empty")
    image_ids = np.empty(len(records), dtype=np.int64)
    eligible = np.empty(len(records), dtype=bool)
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"cells[{position}] must be a mapping")
        cell_index = _integer(
            record.get("cell_index"), name=f"cells[{position}].cell_index"
        )
        if cell_index != position:
            raise ValueError("cells must be in contiguous global cell-index order")
        image_ids[position] = _integer(
            record.get("image_index"), name=f"cells[{position}].image_index"
        )
        raw_eligible = record.get("geometry_eligible")
        if not isinstance(raw_eligible, (bool, np.bool_)):
            raise TypeError(
                f"cells[{position}].geometry_eligible must be boolean"
            )
        eligible[position] = bool(raw_eligible)

    image_count = int(image_ids.max()) + 1
    observed = np.unique(image_ids)
    if not np.array_equal(observed, np.arange(image_count, dtype=np.int64)):
        raise ValueError("cell image indices must be contiguous from zero")
    return records, image_ids, eligible, image_count


def _validated_image_indices(
    image_indices: Sequence[int] | np.ndarray,
    *,
    name: str,
    image_count: int,
) -> tuple[int, ...]:
    raw = _sequence(image_indices, name=name)
    if not raw:
        raise ValueError(f"{name} must not be empty")
    normalized = tuple(
        _integer(value, name=f"{name}[{position}]")
        for position, value in enumerate(raw)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    if any(value >= image_count for value in normalized):
        raise ValueError(f"{name} contains an out-of-range image index")
    # A canonical order prevents fold metadata from depending on caller order.
    return tuple(sorted(normalized))


def _num_classes(value: Any) -> int:
    return _integer(value, name="num_classes", minimum=1)


def _confusion(value: Any, *, name: str, num_classes: int) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.shape != (num_classes, num_classes):
        raise ValueError(
            f"{name} must have shape ({num_classes}, {num_classes})"
        )
    if matrix.dtype == np.bool_ or not np.issubdtype(matrix.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    result = np.ascontiguousarray(matrix, dtype=np.int64)
    if np.any(result < 0):
        raise ValueError(f"{name} must be non-negative")
    return result


def _mean_iou(confusion: np.ndarray) -> float:
    diagonal = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - diagonal
    supported = union > 0
    if not np.any(supported):
        raise ValueError("confusion has no supported class")
    return float(np.mean(diagonal[supported] / union[supported]))


def _cell_confusions(
    record: Mapping[str, Any],
    *,
    cell_index: int,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    confusions = record.get("confusion")
    if not isinstance(confusions, Mapping):
        raise TypeError(f"cells[{cell_index}].confusion must be a mapping")
    k1 = _confusion(
        confusions.get("k1"),
        name=f"cells[{cell_index}].confusion.k1",
        num_classes=num_classes,
    )
    k2 = _confusion(
        confusions.get("k2"),
        name=f"cells[{cell_index}].confusion.k2",
        num_classes=num_classes,
    )
    if not np.array_equal(k1.sum(axis=1), k2.sum(axis=1)):
        raise ValueError(
            f"cells[{cell_index}] K1/K2 confusions disagree on target support"
        )
    return k1, k2


def _feature_row(record: Mapping[str, Any], *, cell_index: int) -> np.ndarray:
    scores = record.get("scores")
    if not isinstance(scores, Mapping):
        raise TypeError(f"cells[{cell_index}].scores must be a mapping")
    result = np.empty(len(FEATURE_NAMES), dtype=np.float64)
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        value = scores.get(feature_name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(
                f"cells[{cell_index}].scores.{feature_name} must be numeric"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(
                f"cells[{cell_index}].scores.{feature_name} must be finite"
            )
        result[feature_index] = numeric
    return result


def _eligible_indices_for_images(
    image_ids: np.ndarray,
    eligible: np.ndarray,
    image_indices: Sequence[int],
) -> np.ndarray:
    selected = np.isin(image_ids, np.asarray(image_indices, dtype=np.int64))
    return np.flatnonzero(selected & eligible).astype(np.int64, copy=False)


def _sha256_array(values: np.ndarray, *, dtype: str) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def singleton_k1_to_k2_targets(
    cells: Sequence[Mapping[str, Any]],
    full_k1_confusion_by_image: Sequence[Sequence[Sequence[int]] | np.ndarray],
    fit_image_indices: Sequence[int] | np.ndarray,
    *,
    num_classes: int,
) -> dict[str, Any]:
    """Recompute fit-fold singleton K1-to-K2 pooled-mIoU gains.

    ``full_k1_confusion_by_image`` must contain one full-image reference per
    image and is indexed by ``image_index``.  Only entries named by
    ``fit_image_indices`` are read.  This is deliberate: a malformed or
    different held-out reference cannot affect a fit fold.
    """

    classes = _num_classes(num_classes)
    records, image_ids, eligible, image_count = _validated_basic_cells(cells)
    fit_images = _validated_image_indices(
        fit_image_indices,
        name="fit_image_indices",
        image_count=image_count,
    )
    references = _sequence(
        full_k1_confusion_by_image, name="full_k1_confusion_by_image"
    )
    if len(references) != image_count:
        raise ValueError(
            "full_k1_confusion_by_image length must equal the cell image count"
        )

    pooled = np.zeros((classes, classes), dtype=np.int64)
    fit_cell_indices = _eligible_indices_for_images(
        image_ids, eligible, fit_images
    )
    if fit_cell_indices.size == 0:
        raise ValueError("fit images contain no geometry-eligible cells")

    # Validate target confusions only for fit images.  Held-out confusion is
    # not an inference input and must not leak through eager validation.
    fit_confusions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for image_index in fit_images:
        reference = _confusion(
            references[image_index],
            name=f"full_k1_confusion_by_image[{image_index}]",
            num_classes=classes,
        )
        common_k1 = np.zeros_like(reference)
        for cell_index in np.flatnonzero(image_ids == image_index):
            k1, k2 = _cell_confusions(
                records[int(cell_index)],
                cell_index=int(cell_index),
                num_classes=classes,
            )
            common_k1 += k1
            if eligible[cell_index]:
                fit_confusions[int(cell_index)] = (k1, k2)
        if np.any(reference < common_k1):
            raise ValueError(
                f"full K1 confusion for image {image_index} does not contain "
                "all of its cell K1 confusion"
            )
        pooled += reference

    baseline_miou = _mean_iou(pooled)
    targets = np.empty(len(fit_cell_indices), dtype=np.float64)
    for position, raw_cell_index in enumerate(fit_cell_indices):
        cell_index = int(raw_cell_index)
        k1, k2 = fit_confusions[cell_index]
        candidate = pooled + k2 - k1
        if np.any(candidate < 0):
            raise ValueError(
                f"cells[{cell_index}] K1-to-K2 delta makes pooled confusion negative"
            )
        targets[position] = _mean_iou(candidate) - baseline_miou
    if not np.all(np.isfinite(targets)):
        raise AssertionError("singleton target computation produced a non-finite value")

    return {
        "fit_image_indices": fit_images,
        "fit_cell_indices": fit_cell_indices,
        "pooled_full_k1_confusion": pooled,
        "pooled_full_k1_miou": baseline_miou,
        "targets": targets,
    }

def fit_ridge_gate(
    cells: Sequence[Mapping[str, Any]],
    full_k1_confusion_by_image: Sequence[Sequence[Sequence[int]] | np.ndarray],
    fit_image_indices: Sequence[int] | np.ndarray,
    *,
    num_classes: int,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> dict[str, Any]:
    """Fit the one allowed standardized three-feature ridge model."""

    if isinstance(ridge_lambda, (bool, np.bool_)) or not isinstance(
        ridge_lambda, Real
    ):
        raise TypeError("ridge_lambda must be numeric")
    penalty = float(ridge_lambda)
    if not math.isfinite(penalty) or penalty != RIDGE_LAMBDA:
        raise ValueError("the preregistered ridge_lambda is fixed at 1.0")

    records, _, _, _ = _validated_basic_cells(cells)
    target = singleton_k1_to_k2_targets(
        records,
        full_k1_confusion_by_image,
        fit_image_indices,
        num_classes=num_classes,
    )
    fit_cells = np.asarray(target["fit_cell_indices"], dtype=np.int64)
    features = np.vstack(
        [
            _feature_row(records[int(cell_index)], cell_index=int(cell_index))
            for cell_index in fit_cells
        ]
    )
    target_values = np.asarray(target["targets"], dtype=np.float64)

    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0, ddof=0)
    constant_mask = feature_std == 0.0
    feature_scale = feature_std.copy()
    feature_scale[constant_mask] = 1.0
    standardized = (features - feature_mean) / feature_scale
    centered_target = target_values - target_values.mean()

    gram = standardized.T @ standardized
    coefficients = np.linalg.solve(
        gram + penalty * np.eye(len(FEATURE_NAMES), dtype=np.float64),
        standardized.T @ centered_target,
    )
    coefficients[constant_mask] = 0.0
    target_mean = float(target_values.mean())
    training_predictions = target_mean + standardized @ coefficients
    raw_coefficients = coefficients / feature_scale
    raw_intercept = float(target_mean - feature_mean @ raw_coefficients)

    residuals = training_predictions - target_values
    training_mse = float(np.mean(residuals * residuals))
    target_variance = float(np.mean(centered_target * centered_target))
    training_r2 = (
        None
        if target_variance == 0.0
        else float(1.0 - training_mse / target_variance)
    )

    pooled = np.asarray(target["pooled_full_k1_confusion"], dtype=np.int64)
    return {
        "model_type": MODEL_TYPE,
        "schema_version": MODEL_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "feature_mean": tuple(float(value) for value in feature_mean),
        "feature_scale": tuple(float(value) for value in feature_scale),
        "constant_feature_mask": tuple(bool(value) for value in constant_mask),
        "standardized_coefficients": tuple(
            float(value) for value in coefficients
        ),
        "raw_coefficients": tuple(float(value) for value in raw_coefficients),
        "raw_intercept": raw_intercept,
        "target_mean": target_mean,
        "ridge_lambda": penalty,
        "intercept_penalized": False,
        "feature_standardization_ddof": 0,
        "target_centered": True,
        "fit_image_indices": target["fit_image_indices"],
        "fit_cell_indices": tuple(int(value) for value in fit_cells),
        "fit_cell_count": int(len(fit_cells)),
        "num_classes": int(pooled.shape[0]),
        "target_audit": {
            "definition": (
                "mIoU(pooled full-image fit K1 confusion + cell K2 - cell K1) "
                "- mIoU(pooled full-image fit K1 confusion)"
            ),
            "pooled_full_k1_confusion": pooled.tolist(),
            "pooled_full_k1_miou": float(target["pooled_full_k1_miou"]),
            "minimum": float(target_values.min()),
            "maximum": float(target_values.max()),
            "mean": target_mean,
            "standard_deviation": float(target_values.std(ddof=0)),
            "target_sha256_float64_le": _sha256_array(
                target_values, dtype="<f8"
            ),
            "fit_cell_indices_sha256_int64_le": _sha256_array(
                fit_cells, dtype="<i8"
            ),
        },
        "feature_audit": {
            "matrix_sha256_float64_le": _sha256_array(features, dtype="<f8"),
            "allowed_sources": tuple(f"scores.{name}" for name in FEATURE_NAMES),
            "uses_image_identity": False,
            "uses_coordinates": False,
            "uses_stored_singleton_oracle_score": False,
            "uses_stage_b0_action_map": False,
            "uses_k2_or_k4_output_features": False,
        },
        "training_diagnostics": {
            "prediction_mean": float(training_predictions.mean()),
            "mean_squared_error": training_mse,
            "r_squared": training_r2,
            "prediction_sha256_float64_le": _sha256_array(
                training_predictions, dtype="<f8"
            ),
        },
    }


def _model_vector(
    model: Mapping[str, Any], *, name: str, length: int
) -> np.ndarray:
    raw = np.asarray(model.get(name))
    if raw.shape != (length,):
        raise ValueError(f"model {name} must have length {length}")
    if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"model {name} must be numeric")
    result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"model {name} must be finite")
    return result


def _validated_model(
    model: Mapping[str, Any], *, image_count: int
) -> dict[str, Any]:
    if not isinstance(model, Mapping):
        raise TypeError("model must be a mapping")
    if model.get("model_type") != MODEL_TYPE:
        raise ValueError(f"model_type must be {MODEL_TYPE!r}")
    if model.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported ridge model schema version")
    if tuple(model.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("model feature_names differ from the frozen feature set")
    if model.get("ridge_lambda") != RIDGE_LAMBDA:
        raise ValueError("model ridge_lambda differs from the frozen value 1.0")
    if model.get("intercept_penalized") is not False:
        raise ValueError("model must use an unpenalized intercept")
    if model.get("target_centered") is not True:
        raise ValueError("model must have a centered training target")
    if model.get("feature_standardization_ddof") != 0:
        raise ValueError("model must use population feature standard deviations")

    fit_images = _validated_image_indices(
        model.get("fit_image_indices"),
        name="model.fit_image_indices",
        image_count=image_count,
    )
    mean = _model_vector(model, name="feature_mean", length=len(FEATURE_NAMES))
    scale = _model_vector(model, name="feature_scale", length=len(FEATURE_NAMES))
    if np.any(scale <= 0.0):
        raise ValueError("model feature_scale must be positive")
    coefficients = _model_vector(
        model, name="standardized_coefficients", length=len(FEATURE_NAMES)
    )
    target_mean = model.get("target_mean")
    if isinstance(target_mean, (bool, np.bool_)) or not isinstance(
        target_mean, Real
    ):
        raise TypeError("model target_mean must be numeric")
    target_mean = float(target_mean)
    if not math.isfinite(target_mean):
        raise ValueError("model target_mean must be finite")
    return {
        "fit_image_indices": fit_images,
        "feature_mean": mean,
        "feature_scale": scale,
        "standardized_coefficients": coefficients,
        "target_mean": target_mean,
    }


def predict_ridge_gate(
    model: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    predict_image_indices: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Score only eligible cells in disjoint, explicitly named images.

    The returned ``scores_by_cell`` has global cell length.  Entries outside
    the requested prediction images, and ineligible entries inside them, are
    ``None``.  Every index in ``predicted_cell_indices`` has a finite score.
    """

    records, image_ids, eligible, image_count = _validated_basic_cells(cells)
    validated_model = _validated_model(model, image_count=image_count)
    predict_images = _validated_image_indices(
        predict_image_indices,
        name="predict_image_indices",
        image_count=image_count,
    )
    overlap = sorted(
        set(validated_model["fit_image_indices"]).intersection(predict_images)
    )
    if overlap:
        raise ValueError(
            f"predict images overlap model fit images: {overlap}"
        )

    predicted_cells = _eligible_indices_for_images(
        image_ids, eligible, predict_images
    )
    if predicted_cells.size == 0:
        raise ValueError("predict images contain no geometry-eligible cells")
    features = np.vstack(
        [
            _feature_row(records[int(cell_index)], cell_index=int(cell_index))
            for cell_index in predicted_cells
        ]
    )
    standardized = (
        features - validated_model["feature_mean"]
    ) / validated_model["feature_scale"]
    predicted = (
        validated_model["target_mean"]
        + standardized @ validated_model["standardized_coefficients"]
    )
    if not np.all(np.isfinite(predicted)):
        raise AssertionError("ridge prediction produced a non-finite score")

    scores_by_cell: list[float | None] = [None] * len(records)
    for cell_index, score in zip(predicted_cells, predicted, strict=True):
        scores_by_cell[int(cell_index)] = float(score)
    return {
        "model_type": MODEL_TYPE,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "model_fit_image_indices": validated_model["fit_image_indices"],
        "predict_image_indices": predict_images,
        "predicted_cell_indices": tuple(int(value) for value in predicted_cells),
        "scores_by_cell": tuple(scores_by_cell),
        "prediction_sha256_float64_le": _sha256_array(predicted, dtype="<f8"),
    }
