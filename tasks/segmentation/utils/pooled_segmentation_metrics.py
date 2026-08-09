"""Pooled confusion-matrix metrics for fixed-class segmentation benchmarks."""

from __future__ import annotations

import torch


class PooledSegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64
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
            raise ValueError("Prediction contains an invalid class id")

        flat = target[valid] * self.num_classes + valid_prediction
        self.confusion += torch.bincount(
            flat, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        matrix = self.confusion.to(torch.float64)
        true_count = matrix.sum(dim=1)
        predicted_count = matrix.sum(dim=0)
        true_positive = matrix.diag()
        gt_present = true_count > 0
        if not gt_present.any():
            raise ValueError("Metrics require at least one valid ground-truth pixel")

        iou_denominator = true_count + predicted_count - true_positive
        f1_denominator = true_count + predicted_count
        class_iou = torch.zeros(self.num_classes, dtype=torch.float64)
        class_f1 = torch.zeros(self.num_classes, dtype=torch.float64)
        iou_supported = iou_denominator > 0
        f1_supported = f1_denominator > 0
        class_iou[iou_supported] = (
            true_positive[iou_supported] / iou_denominator[iou_supported]
        )
        class_f1[f1_supported] = (
            2.0 * true_positive[f1_supported] / f1_denominator[f1_supported]
        )

        total = matrix.sum()
        return {
            "mIoU": float(class_iou.mean()),
            "mF1": float(class_f1.mean()),
            "accuracy": float(true_positive.sum() / total),
            "class_iou": [float(value) for value in class_iou],
            "class_f1": [float(value) for value in class_f1],
            "gt_present_class_ids": torch.nonzero(
                gt_present, as_tuple=False
            ).flatten().tolist(),
            "gt_pixels": [int(value) for value in true_count],
            "predicted_pixels": [int(value) for value in predicted_count],
            "valid_pixels": int(total),
            "confusion": self.confusion.tolist(),
        }
