"""Pooled semantic-segmentation metrics for the eight EarthMiss classes."""

import numpy as np
import torch


EVER_EPS = 1e-7
EVER_DECIMALS = 5


def _official_ever_metrics(confusion, decimals=EVER_DECIMALS):
    """Reproduce ``ever.metric.PixelMetric.summary_all`` for IoU and F1."""
    # Ever accumulates its sparse confusion matrix in float32, so retaining
    # that dtype matters for exact five-decimal benchmark reporting.
    matrix = np.asarray(confusion, dtype=np.float32)
    true_count = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = np.diag(matrix)

    class_iou = np.round(
        true_positive
        / (true_count + predicted_count - true_positive + EVER_EPS),
        decimals,
    )
    precision = true_positive / (predicted_count + EVER_EPS)
    recall = true_positive / (true_count + EVER_EPS)
    class_f1 = np.round(
        2.0 * precision * recall / (precision + recall + EVER_EPS),
        decimals,
    )
    miou = np.round(class_iou.mean(), decimals)
    mf1 = np.round(class_f1.mean(), decimals)

    def reported(value):
        return float(f"{float(value):.{decimals}f}")

    return {
        "mIoU": reported(miou),
        "mF1": reported(mf1),
        "class_iou": [reported(value) for value in class_iou],
        "class_f1": [reported(value) for value in class_f1],
    }


class EarthMissMetrics:
    def __init__(self, num_classes=8, ignore_index=8):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64
        )
        self.official_ever_confusion = torch.zeros(
            (num_classes, num_classes), dtype=torch.float32
        )

    def update(self, prediction, target):
        prediction = torch.as_tensor(prediction).detach().to("cpu", torch.int64)
        target = torch.as_tensor(target).detach().to("cpu", torch.int64)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {prediction.shape} vs {target.shape}"
            )
        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
        )
        if not valid.any():
            return
        valid_prediction = prediction[valid]
        if ((valid_prediction < 0) | (valid_prediction >= self.num_classes)).any():
            raise ValueError("Prediction contains an invalid EarthMiss class id")
        flat = target[valid] * self.num_classes + valid_prediction
        batch_confusion = torch.bincount(
            flat, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += batch_confusion
        # Match Ever's per-forward float32 sparse-matrix accumulation order.
        self.official_ever_confusion += batch_confusion.to(torch.float32)

    def compute(self):
        matrix = self.confusion.to(torch.float64)
        true_count = matrix.sum(dim=1)
        predicted_count = matrix.sum(dim=0)
        true_positive = matrix.diag()

        selection_support = true_count > 0
        if not selection_support.any():
            raise ValueError("EarthMiss metrics require valid ground-truth pixels")

        iou_denominator = true_count + predicted_count - true_positive
        f1_denominator = true_count + predicted_count
        class_iou = torch.full((self.num_classes,), torch.nan, dtype=torch.float64)
        class_f1 = torch.full((self.num_classes,), torch.nan, dtype=torch.float64)
        iou_supported = iou_denominator > 0
        f1_supported = f1_denominator > 0
        class_iou[iou_supported] = (
            true_positive[iou_supported] / iou_denominator[iou_supported]
        )
        class_f1[f1_supported] = (
            2.0 * true_positive[f1_supported] / f1_denominator[f1_supported]
        )

        total = matrix.sum()
        accuracy = true_positive.sum() / total
        official = _official_ever_metrics(self.official_ever_confusion.numpy())
        selection_class_ids = torch.nonzero(
            selection_support, as_tuple=False
        ).flatten().tolist()
        return {
            # This fixed, GT-only support is the sole checkpoint-selection metric.
            "mIoU": float(class_iou[selection_support].mean()),
            "mF1": float(class_f1[selection_support].mean()),
            "selection_policy": "pooled_gt_present",
            "selection_class_ids": selection_class_ids,
            "selection_class_count": len(selection_class_ids),
            # Exact EarthMiss/MetaRS public evaluator convention: all classes,
            # EPS=1e-7, per-class rounding before the fixed-class macro mean.
            "official_ever_mIoU": official["mIoU"],
            "official_ever_mF1": official["mF1"],
            "official_ever_class_iou": official["class_iou"],
            "official_ever_class_f1": official["class_f1"],
            "official_ever_class_count": self.num_classes,
            "accuracy": float(accuracy),
            "class_iou": [
                float(value) if supported else None
                for value, supported in zip(class_iou, iou_supported, strict=True)
            ],
            "class_f1": [
                float(value) if supported else None
                for value, supported in zip(class_f1, f1_supported, strict=True)
            ],
            "gt_pixels": [int(value) for value in true_count],
            "predicted_pixels": [int(value) for value in predicted_count],
            "confusion": self.confusion.tolist(),
            "valid_pixels": int(self.confusion.sum()),
        }
