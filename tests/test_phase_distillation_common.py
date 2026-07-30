"""CPU guards for the common spatial-phase distillation components."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from scripts.phase_distillation_common import (
    FrozenE0P2Extractor,
    PhaseCorrectionBranch,
    SinglePhaseCorrectedModel,
    batchnorm_buffer_sha256,
    masked_kl_divergence,
    masked_teacher_student_kl,
    module_state_sha256,
    parameter_sha256,
    snapshot_batchnorm_buffers,
    snapshot_module_state,
    snapshot_named_parameters,
    teacher_gain_mask,
    tensor_snapshot_sha256,
    tensor_snapshots_equal,
)


class _ToyNeck(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 256, kernel_size=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        p2 = self.projection(inputs)
        return [p2, F.avg_pool2d(p2, 2), F.avg_pool2d(p2, 4)]


class _ToyDecoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.neck = _ToyNeck()
        self.out_conv = nn.Sequential(
            nn.Conv2d(256, 7, kernel_size=1, bias=False),
            nn.BatchNorm2d(7),
            nn.ReLU(inplace=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        p2, _, _ = self.neck(inputs)
        low_resolution = self.out_conv(p2)
        return F.interpolate(
            low_resolution, size=(8, 8), mode="bilinear", align_corners=False
        )


class _ToyE0(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.decoder = _ToyDecoder()

    def forward(self, optical: torch.Tensor, sar: torch.Tensor | None = None) -> torch.Tensor:
        if sar is not None:
            optical = optical + sar[:, :3] * 0.0
        return self.decoder(optical)


class PhaseDistillationCommonTests(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_zero_initialized_single_phase_model_is_exactly_e0(self) -> None:
        base = _ToyE0().eval()
        optical = torch.randn(2, 3, 4, 4)
        sar = torch.randn(2, 3, 4, 4)
        with torch.no_grad():
            expected = base(optical, sar)

        corrected = SinglePhaseCorrectedModel(base)
        base_state_before = module_state_sha256(base)
        batchnorm_before = batchnorm_buffer_sha256(base)
        corrected.train()
        self.assertTrue(corrected.training)
        self.assertTrue(corrected.correction_branch.training)
        self.assertFalse(corrected.base_model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))
        self.assertTrue(
            all(
                not child.training
                for child in base.modules()
                if isinstance(child, nn.modules.batchnorm._BatchNorm)
            )
        )

        actual = corrected(optical, sar)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(torch.count_nonzero(corrected.correction_branch.output_projection.weight), 0)
        self.assertEqual(torch.count_nonzero(corrected.correction_branch.output_projection.bias), 0)

        actual.mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        output_gradient = corrected.correction_branch.output_projection.weight.grad
        self.assertIsNotNone(output_gradient)
        self.assertGreater(torch.count_nonzero(output_gradient).item(), 0)
        self.assertEqual(module_state_sha256(base), base_state_before)
        self.assertEqual(batchnorm_buffer_sha256(base), batchnorm_before)

    def test_extractor_hook_can_be_removed(self) -> None:
        extractor = FrozenE0P2Extractor(_ToyE0())
        logits, p2 = extractor(torch.randn(1, 3, 4, 4))
        self.assertEqual(tuple(logits.shape), (1, 7, 8, 8))
        self.assertEqual(tuple(p2.shape), (1, 256, 4, 4))
        self.assertFalse(logits.requires_grad)
        self.assertFalse(p2.requires_grad)
        extractor.remove_hook()
        extractor.remove_hook()
        self.assertFalse(extractor.hook_is_active)
        with self.assertRaisesRegex(RuntimeError, "hook has been removed"):
            extractor(torch.randn(1, 3, 4, 4))

    def test_zero_head_delays_upstream_gradient_until_second_step(self) -> None:
        branch = PhaseCorrectionBranch()
        optimizer = torch.optim.SGD(branch.parameters(), lr=0.05)
        p2 = torch.randn(2, 256, 4, 4)
        target = torch.randn(2, 7, 4, 4)

        first_loss = F.mse_loss(branch(p2), target)
        first_loss.backward()
        output_gradient = branch.output_projection.weight.grad
        self.assertIsNotNone(output_gradient)
        self.assertGreater(torch.count_nonzero(output_gradient).item(), 0)
        upstream_parameters = [
            parameter
            for name, parameter in branch.named_parameters()
            if not name.startswith("output_projection.")
        ]
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
                for parameter in upstream_parameters
            )
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertGreater(
            torch.count_nonzero(branch.output_projection.weight).item(), 0
        )
        second_loss = F.mse_loss(branch(p2), target)
        second_loss.backward()
        projection_gradient = branch.input_projection[0].weight.grad
        self.assertIsNotNone(projection_gradient)
        self.assertGreater(torch.count_nonzero(projection_gradient).item(), 0)

    def test_teacher_gain_mask_is_label_aware_and_ignores_seven(self) -> None:
        baseline = torch.tensor(
            [[[[0.0, 2.0, 4.0]], [[2.0, 0.0, 0.0]]]], dtype=torch.float32
        )
        teacher = torch.tensor(
            [[[[2.0, 0.0, 0.0]], [[0.0, 2.0, 4.0]]]], dtype=torch.float32
        )
        labels = torch.tensor([[[0, 0, 7]]], dtype=torch.long)
        mask = teacher_gain_mask(teacher, baseline, labels, ignore_index=7)
        expected = torch.tensor([[[True, False, False]]])
        self.assertTrue(torch.equal(mask, expected))
        self.assertFalse(mask.requires_grad)

        strict = teacher_gain_mask(
            teacher, baseline, labels, margin=3.0, ignore_index=7
        )
        self.assertFalse(bool(strict.any()))

        constrained = teacher_gain_mask(
            teacher,
            baseline,
            labels,
            delta=0.0,
            valid_mask=torch.tensor([[[False, True, True]]]),
            ignore_index=7,
        )
        self.assertFalse(bool(constrained.any()))

    def test_teacher_gain_rejects_non_ignore_out_of_range_label(self) -> None:
        logits = torch.zeros(1, 2, 1, 1)
        labels = torch.tensor([[[6]]])
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            teacher_gain_mask(logits, logits, labels, ignore_index=7)

    def test_masked_kl_matches_manual_valid_pixel_mean(self) -> None:
        student = torch.tensor(
            [[[[1.0, -1.0]], [[0.0, 2.0]]]], requires_grad=True
        )
        teacher = torch.tensor(
            [[[[0.0, 3.0]], [[2.0, -2.0]]]], requires_grad=True
        )
        mask = torch.tensor([[[True, False]]])
        loss = masked_teacher_student_kl(student, teacher, mask, temperature=1.0)

        teacher_log_prob = F.log_softmax(teacher.detach()[:, :, :, :1], dim=1)
        expected = (
            teacher_log_prob.exp()
            * (
                teacher_log_prob
                - F.log_softmax(student[:, :, :, :1], dim=1)
            )
        ).sum()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-7, rtol=0.0))
        alias_loss = masked_kl_divergence(student, teacher, mask, temperature=1.0)
        self.assertTrue(torch.equal(loss, alias_loss))
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_empty_mask_kl_is_differentiable_zero(self) -> None:
        student = torch.randn(2, 7, 3, 4, requires_grad=True)
        teacher = torch.randn_like(student)
        loss = masked_teacher_student_kl(
            student, teacher, torch.zeros(2, 3, 4, dtype=torch.bool)
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))

    def test_state_and_batchnorm_snapshots_detect_mutation(self) -> None:
        model = _ToyE0().eval()
        state_before = snapshot_module_state(model)
        parameters_before = snapshot_named_parameters(model)
        batchnorm_before = snapshot_batchnorm_buffers(model)
        state_hash = module_state_sha256(model)
        parameter_hash = parameter_sha256(model)
        batchnorm_hash = batchnorm_buffer_sha256(model)
        self.assertEqual(state_hash, tensor_snapshot_sha256(state_before))
        self.assertEqual(parameter_hash, tensor_snapshot_sha256(parameters_before))
        self.assertEqual(batchnorm_hash, tensor_snapshot_sha256(batchnorm_before))

        batchnorm = model.decoder.out_conv[1]
        batchnorm.running_mean.add_(1.0)
        self.assertFalse(
            tensor_snapshots_equal(batchnorm_before, snapshot_batchnorm_buffers(model))
        )
        self.assertNotEqual(batchnorm_hash, batchnorm_buffer_sha256(model))
        self.assertNotEqual(state_hash, module_state_sha256(model))
        self.assertTrue(
            tensor_snapshots_equal(parameters_before, snapshot_named_parameters(model))
        )


if __name__ == "__main__":
    unittest.main()
