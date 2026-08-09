"""CPU contracts for the EarthMiss V3 zero-training gates."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from scripts.diagnose_earthmiss_missing_v3 import (
    NUM_CLASSES,
    SingleClassBiasAccumulator,
    TeacherPairScope,
    fixed_eight_class_metrics,
    summarize_bias_sweep,
    summarize_test_bias_upper_bound,
    validate_args,
)
from tasks.segmentation.utils.earthmiss_metrics import EarthMissMetrics


class EarthMissV3ZeroTrainingTest(unittest.TestCase):
    def test_argument_contract_rejects_invalid_runtime_controls(self):
        with self.assertRaisesRegex(ValueError, "num-workers"):
            validate_args(
                SimpleNamespace(
                    num_workers=-1,
                    inference_batch_size=1,
                    smoke_tiles=0,
                )
            )
        with self.assertRaisesRegex(ValueError, "smoke-tiles"):
            validate_args(
                SimpleNamespace(
                    num_workers=0,
                    inference_batch_size=1,
                    smoke_tiles=-1,
                )
            )

    def test_teacher_scope_counts_correctable_and_harmful_pixels(self):
        target = torch.tensor([[[0, 1, 2, 8]]])
        student = torch.zeros(1, NUM_CLASSES, 1, 4)
        teacher = torch.zeros_like(student)
        student[0, 1, 0, 0] = 4.0  # wrong, teacher corrects class 0
        teacher[0, 0, 0, 0] = 4.0
        student[0, 1, 0, 1] = 4.0  # student correct, teacher harms
        teacher[0, 2, 0, 1] = 4.0
        student[0, 2, 0, 2] = 4.0  # both correct
        teacher[0, 2, 0, 2] = 4.0

        scope = TeacherPairScope()
        scope.update(student, teacher, target)
        result = scope.summary()

        self.assertEqual(result["valid_pixels"], 3)
        self.assertEqual(result["student_error_pixels"], 1)
        self.assertEqual(result["teacher_correct_student_wrong_pixels"], 1)
        self.assertEqual(result["teacher_wrong_student_correct_pixels"], 1)
        self.assertEqual(result["q_cov"], 1.0)
        self.assertGreater(result["oracle_gain_over_student_pp"], 0.0)

    def test_bias_histogram_matches_direct_argmax_including_ties(self):
        logits = torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0], [0.5, 2.0, -1.0]],
                    [[1.0, 1.0, 0.0], [0.5, 0.0, -1.0]],
                    [[0.0, 0.0, 1.0], [0.5, 0.0, -1.0]],
                    [[0.0, 0.0, 0.0], [0.5, 0.0, -1.0]],
                    [[0.0, 0.0, 0.0], [0.5, 0.0, -1.0]],
                    [[0.0, 0.0, 0.0], [0.5, 0.0, -1.0]],
                    [[0.0, 0.0, 0.0], [0.5, 0.0, -1.0]],
                    [[0.0, 0.0, 0.0], [0.5, 0.0, -1.0]],
                ]
            ]
        )
        target = torch.tensor([[[0, 1, 2], [3, 4, 8]]])
        grid = (-1.0, 0.0, 1.0)
        accumulator = SingleClassBiasAccumulator(grid)
        accumulator.update(logits, target)

        for class_id in range(NUM_CLASSES):
            for bias_index, bias in enumerate(grid):
                adjusted = logits.clone()
                adjusted[:, class_id] += bias
                evaluator = EarthMissMetrics()
                evaluator.update(adjusted.argmax(dim=1), target)
                self.assertTrue(
                    torch.equal(
                        accumulator.confusion_for(class_id, bias_index),
                        evaluator.confusion,
                    ),
                    (class_id, bias),
                )
        accumulator.assert_zero_matches_baseline()

    def test_train_fitted_and_test_oracle_bias_are_reported_separately(self):
        train = SingleClassBiasAccumulator((-1.0, 0.0, 1.0))
        test = SingleClassBiasAccumulator((-1.0, 0.0, 1.0))
        target = torch.tensor([[[0, 1], [2, 3]]])
        train_logits = torch.zeros(1, NUM_CLASSES, 2, 2)
        test_logits = torch.zeros_like(train_logits)
        for class_id in range(4):
            train_logits[0, class_id, class_id // 2, class_id % 2] = 0.5
            test_logits[0, class_id, class_id // 2, class_id % 2] = 0.5
        train.update(train_logits, target)
        test.update(test_logits, target)

        report = summarize_bias_sweep(train, test)

        self.assertEqual(report["scope"], "one_class_bias_at_a_time")
        self.assertTrue(report["test_oracle_is_test_developed_diagnostic_only"])
        self.assertEqual(len(report["per_class"]), NUM_CLASSES)
        self.assertIn("best_train_fitted_single_class", report)
        self.assertIn("best_test_oracle_single_class", report)

        test_only = summarize_test_bias_upper_bound(test)
        self.assertEqual(
            test_only["scope"],
            "one_class_bias_at_a_time_on_complete_test",
        )
        self.assertFalse(test_only["independent_test_claim_allowed"])
        self.assertNotIn("best_train_fitted_single_class", test_only)

    def test_fixed_metric_always_averages_all_eight_classes(self):
        confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
        confusion[0, 0] = 10
        result = fixed_eight_class_metrics(confusion)
        self.assertAlmostEqual(result["mIoU_percent"], 12.5)


if __name__ == "__main__":
    unittest.main()
