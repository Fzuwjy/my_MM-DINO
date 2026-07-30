"""CPU contract tests for the independent matched phase-residual math."""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from scripts.phase_residual_common import (
    NUM_CLASSES,
    PhaseResidualGateThresholds,
    center_class_logits,
    fix_mask_rms_scale,
    full_resolution_centered_delta,
    matched_phase_residual_target,
    matched_residual_masks,
    mean_iou_from_confusion,
    phase_residual_behavior_statistics,
    phase_residual_probe_gate,
    scaled_phase_residual_losses,
    semantic_confusion_matrix,
)


def logits_from_predictions(prediction: torch.Tensor) -> torch.Tensor:
    return (
        F.one_hot(prediction.to(torch.int64), num_classes=NUM_CLASSES)
        .permute(0, 3, 1, 2)
        .to(torch.float32)
        .mul(4.0)
    )


class PhaseResidualCommonTests(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(23)

    def test_class_center_is_per_pixel_and_shift_invariant(self) -> None:
        logits = torch.randn(2, NUM_CLASSES, 3, 4, dtype=torch.float64)
        shift = torch.randn(2, 1, 3, 4, dtype=torch.float64)
        centered = center_class_logits(logits)
        shifted = center_class_logits(logits + shift)
        self.assertTrue(torch.allclose(centered, shifted, atol=1e-12, rtol=0.0))
        self.assertTrue(
            torch.allclose(
                centered.mean(dim=1),
                torch.zeros_like(centered[:, 0]),
                atol=1e-12,
                rtol=0.0,
            )
        )

    def test_full_resolution_delta_is_bilinear_then_centered(self) -> None:
        low = torch.randn(1, NUM_CLASSES, 2, 3, requires_grad=True)
        reference = torch.zeros(1, NUM_CLASSES, 5, 4)
        actual = full_resolution_centered_delta(low, reference)
        expected = center_class_logits(
            F.interpolate(low, size=(5, 4), mode="bilinear", align_corners=False)
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(tuple(actual.shape), tuple(reference.shape))
        actual.square().mean().backward()
        self.assertIsNotNone(low.grad)
        self.assertGreater(torch.count_nonzero(low.grad).item(), 0)

    def test_matched_target_recovers_teacher_and_ignores_input_gauges(self) -> None:
        e0 = torch.randn(2, NUM_CLASSES, 3, 2)
        teacher = torch.randn_like(e0)
        target = matched_phase_residual_target(teacher, e0)
        corrected = e0 + target
        self.assertTrue(
            torch.allclose(
                corrected.softmax(dim=1),
                teacher.softmax(dim=1),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        teacher_shift = torch.randn(2, 1, 3, 2)
        e0_shift = torch.randn(2, 1, 3, 2)
        shifted_target = matched_phase_residual_target(
            teacher + teacher_shift, e0 + e0_shift
        )
        self.assertTrue(torch.allclose(target, shifted_target, atol=5e-7, rtol=0.0))
        self.assertFalse(target.requires_grad)

    def test_masks_partition_valid_and_keep_both_wrong_pixels(self) -> None:
        labels = torch.tensor([[[0, 1, 2, 7, -1]]])
        e0_prediction = torch.tensor([[[1, 1, 3, 0, 0]]])
        teacher_prediction = torch.tensor([[[0, 2, 3, 0, 0]]])
        e0 = logits_from_predictions(e0_prediction)
        teacher = logits_from_predictions(teacher_prediction)
        fix, keep = matched_residual_masks(teacher, e0, labels)
        expected_fix = torch.tensor([[[True, False, False, False, False]]])
        expected_keep = torch.tensor([[[False, True, True, False, False]]])
        self.assertTrue(torch.equal(fix, expected_fix))
        self.assertTrue(torch.equal(keep, expected_keep))
        self.assertFalse(bool((fix & keep).any()))
        self.assertTrue(torch.equal(fix | keep, labels.ge(0) & labels.lt(NUM_CLASSES)))

        external_valid = torch.tensor([[[True, True, False, True, True]]])
        restricted_fix, restricted_keep = matched_residual_masks(
            teacher, e0, labels, valid_mask=external_valid
        )
        self.assertTrue(torch.equal(restricted_fix, expected_fix))
        self.assertFalse(bool(restricted_keep[0, 0, 2]))

    def test_rms_scaled_losses_use_independent_pixel_class_means(self) -> None:
        target = torch.tensor([[[[3.0, 0.0]], [[-3.0, 0.0]]]])
        fix = torch.tensor([[[True, False]]])
        keep = torch.tensor([[[False, True]]])
        scale = fix_mask_rms_scale(target, fix)
        self.assertEqual(scale, 3.0)

        prediction = torch.tensor(
            [[[[5.0, 6.0]], [[5.0, 4.0]]]], requires_grad=True
        )
        losses = scaled_phase_residual_losses(
            prediction,
            target,
            fix,
            keep,
            scale=scale,
            beta=1.0,
        )
        # Centering removes the common +5 shift.  The fix residual is +/-1 RMS
        # units, while keep leakage is +/-1/3 RMS units.
        self.assertTrue(torch.allclose(losses["fix"], torch.tensor(0.5)))
        self.assertTrue(
            torch.allclose(losses["keep"], torch.tensor(1.0 / 18.0), atol=1e-7)
        )
        self.assertTrue(
            torch.allclose(losses["total"], losses["fix"] + losses["keep"])
        )
        losses["total"].backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_rms_and_independent_losses_reject_degenerate_masks(self) -> None:
        target = torch.zeros(1, NUM_CLASSES, 1, 1)
        empty = torch.zeros(1, 1, 1, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            fix_mask_rms_scale(target, empty)
        nonempty = ~empty
        with self.assertRaisesRegex(ValueError, "positive"):
            fix_mask_rms_scale(target, nonempty)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            scaled_phase_residual_losses(
                target,
                target,
                nonempty,
                nonempty,
                scale=1.0,
            )

    def test_fix_rms_scale_is_cpu_float64_reproducible(self) -> None:
        target = torch.tensor(
            [[[[0.1, -0.3]], [[-0.2, 0.4]], [[0.1, -0.1]]]],
            dtype=torch.float32,
        )
        mask = torch.tensor([[[True, True]]])
        centered = center_class_logits(target)
        selected = centered.permute(0, 2, 3, 1)[mask].reshape(-1).double()
        expected = float(torch.sqrt(selected.square().mean()).item())
        self.assertEqual(fix_mask_rms_scale(target, mask), expected)

    def test_miou_uses_repository_union_supported_class_set(self) -> None:
        labels = torch.tensor([[[0, 1]]])
        prediction = torch.tensor([[[0, 2]]])
        confusion = semantic_confusion_matrix(prediction, labels)
        # Class 2 has a false positive and therefore a non-zero union even
        # without target support.  It must be included and penalized.
        self.assertAlmostEqual(mean_iou_from_confusion(confusion), 1.0 / 3.0)

    def test_behavior_statistics_and_preregistered_gate_pass(self) -> None:
        labels = torch.tensor(
            [[[class_index for class_index in range(NUM_CLASSES) for _ in range(2)]]]
        )
        e0_prediction = labels.clone()
        e0_prediction[0, 0, 0] = 1
        e0_prediction[0, 0, 2] = 2
        teacher_prediction = labels.clone()
        corrected_prediction = labels.clone()
        statistics = phase_residual_behavior_statistics(
            logits_from_predictions(e0_prediction),
            logits_from_predictions(teacher_prediction),
            logits_from_predictions(corrected_prediction),
            labels,
        )
        self.assertEqual(statistics["fix_pixels"], 2)
        self.assertEqual(statistics["keep_pixels"], 12)
        self.assertEqual(statistics["routed_fixed_pixels"], 2)
        self.assertEqual(statistics["global_fixed_pixels"], 2)
        self.assertEqual(statistics["broken_pixels"], 0)
        self.assertEqual(statistics["routed_net_pixels"], 2)
        self.assertEqual(statistics["global_net_correct_pixels"], 2)
        self.assertGreater(statistics["corrected_minus_e0_miou_pp"], 0.0)
        self.assertEqual(statistics["e0"]["target_present_class_count"], 7)

        decision = phase_residual_probe_gate(statistics)
        self.assertEqual(decision["outcome"], "PASS")
        self.assertTrue(decision["passes"])
        self.assertTrue(all(decision["checks"].values()))
        self.assertNotIn("positive_routed_net", decision["checks"])
        self.assertIn("same-one-image exact matched-full-slide", decision["scope"])

    def test_gate_uses_counts_and_strict_miou_improvement(self) -> None:
        passing = {
            "fix_pixels": 4,
            "routed_fixed_pixels": 2,
            "broken_pixels": 0,
            "routed_net_pixels": 2,
            "corrected_minus_e0_miou_pp": 0.01,
        }
        self.assertTrue(phase_residual_probe_gate(passing)["passes"])

        excessive_damage = copy.deepcopy(passing)
        excessive_damage["broken_pixels"] = 1
        excessive_damage["routed_net_pixels"] = 1
        result = phase_residual_probe_gate(excessive_damage)
        self.assertFalse(result["passes"])
        self.assertFalse(result["checks"]["maximum_broken_per_routed_fixed"])

        tied_miou = copy.deepcopy(passing)
        tied_miou["corrected_minus_e0_miou_pp"] = 0.0
        self.assertFalse(phase_residual_probe_gate(tied_miou)["passes"])

        strict_threshold = PhaseResidualGateThresholds(minimum_miou_delta_pp=0.02)
        self.assertFalse(
            phase_residual_probe_gate(passing, thresholds=strict_threshold)["passes"]
        )


if __name__ == "__main__":
    unittest.main()
