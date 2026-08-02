"""Pooled semantic-segmentation metrics for the eight EarthMiss classes."""

import torch


class EarthMissMetrics:
    def __init__(self, num_classes=8, ignore_index=8):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64
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
        self.confusion += torch.bincount(
            flat, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self):
        matrix = self.confusion.to(torch.float64)
        true_count = matrix.sum(dim=1)
        predicted_count = matrix.sum(dim=0)
        true_positive = matrix.diag()

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
        accuracy = true_positive.sum() / total if total > 0 else torch.tensor(float("nan"))
        return {
            "mIoU": float(torch.nanmean(class_iou)),
            "mF1": float(torch.nanmean(class_f1)),
            "accuracy": float(accuracy),
            "class_iou": [float(value) for value in class_iou],
            "class_f1": [float(value) for value in class_f1],
            "confusion": self.confusion.tolist(),
            "valid_pixels": int(self.confusion.sum()),
        }
