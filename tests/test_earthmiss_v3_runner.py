"""CPU contracts for the MetaRS-aligned EarthMiss V3 runner."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json
import tempfile
import unittest

import torch
from torch import nn

from scripts.train_earthmiss_missing_v3 import (
    BATCH_SIZE,
    CHECKPOINT_STEPS,
    MAX_STEPS,
    assert_batchnorm_buffers_equal,
    load_base_into_residual,
    reliable_privileged_loss,
    snapshot_batchnorm_buffers,
    validate_args,
    validate_zero_training_gate_report,
)


class _ResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2))
        self.decoder = nn.Module()
        self.decoder.sar_logit_residual = nn.Conv2d(2, 3, 1)
        self.use_sar_logit_residual = True

    def freeze_base_for_sar_logit_residual(self):
        self.requires_grad_(False)
        self.decoder.sar_logit_residual.requires_grad_(True)
        self.eval()
        return tuple(self.decoder.sar_logit_residual.parameters())


class EarthMissV3RunnerTest(unittest.TestCase):
    def test_budget_and_fixed_test_candidates_are_frozen(self):
        self.assertEqual(BATCH_SIZE, 8)
        self.assertEqual(MAX_STEPS, 15_000)
        self.assertEqual(CHECKPOINT_STEPS, (6_620, 13_240, 15_000))

    def test_residual_arm_arguments_require_frozen_inputs(self):
        with self.assertRaisesRegex(ValueError, "base-checkpoint"):
            validate_args(
                SimpleNamespace(
                    arm="r-ce",
                    base_checkpoint=None,
                    teacher_checkpoint=None,
                    num_workers=0,
                    audit_only=True,
                )
            )
        with self.assertRaisesRegex(ValueError, "teacher-checkpoint"):
            validate_args(
                SimpleNamespace(
                    arm="r-priv",
                    base_checkpoint="a.pth",
                    teacher_checkpoint=None,
                    num_workers=0,
                    audit_only=True,
                )
            )

    def test_every_training_arm_requires_the_zero_training_gate(self):
        with self.assertRaisesRegex(ValueError, "diagnostic-report"):
            validate_args(
                SimpleNamespace(
                    arm="a",
                    base_checkpoint=None,
                    teacher_checkpoint=None,
                    num_workers=0,
                    audit_only=False,
                    diagnostic_report=None,
                    confirm_zero_training_gates_passed=False,
                )
            )

    def test_only_a_complete_formal_gate_report_unlocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.json"
            report = {
                "schema": "earthmiss_missing_v3_zero_training_gates_v2",
                "formal": True,
                "training_was_performed": False,
                "splits": {"test": {}},
            }
            path.write_text(json.dumps(report), encoding="utf-8")
            record = validate_zero_training_gate_report(path)
            self.assertEqual(record["schema"], report["schema"])
            self.assertTrue(record["manual_review_confirmed"])

            report["formal"] = False
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "smoke"):
                validate_zero_training_gate_report(path)

    def test_reliable_privileged_loss_uses_only_correctable_pixels(self):
        # Pixel 0: teacher correct / base wrong -> selected.
        # Pixel 1: teacher wrong -> excluded. Pixel 2: base correct -> excluded.
        target = torch.tensor([[[0, 1, 2]]])
        base = torch.tensor(
            [[[[0.0, 2.0, 0.0]], [[2.0, 0.0, 0.0]], [[0.0, 0.0, 2.0]]]]
        )
        teacher = torch.tensor(
            [[[[3.0, 3.0, 0.0]], [[0.0, 0.0, 0.0]], [[0.0, 0.0, 3.0]]]]
        )
        student = torch.zeros_like(base, requires_grad=True)
        loss, selected = reliable_privileged_loss(
            student, base, teacher, target
        )
        self.assertEqual(selected, 1)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertGreater(float(student.grad[:, :, :, 0].abs().sum()), 0.0)
        self.assertEqual(float(student.grad[:, :, :, 1:].abs().sum()), 0.0)

    def test_empty_privileged_mask_preserves_a_differentiable_zero(self):
        logits = torch.randn(1, 3, 2, 2, requires_grad=True)
        target = torch.zeros(1, 2, 2, dtype=torch.long)
        teacher = torch.zeros_like(logits)
        base = torch.zeros_like(logits)
        loss, selected = reliable_privileged_loss(
            logits, base, teacher, target
        )
        self.assertEqual(selected, 0)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits.grad)))

    def test_base_checkpoint_load_allows_only_new_residual_keys(self):
        base = _ResidualModel()
        base_state = {
            key: value
            for key, value in base.state_dict().items()
            if not key.startswith("decoder.sar_logit_residual.")
        }
        candidate = _ResidualModel()
        trainable = load_base_into_residual(candidate, {"model": base_state})
        self.assertEqual(
            {id(parameter) for parameter in trainable},
            {
                id(parameter)
                for parameter in candidate.decoder.sar_logit_residual.parameters()
            },
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in candidate.named_parameters()
                if not name.startswith("decoder.sar_logit_residual.")
            )
        )

    def test_bn_snapshot_detects_any_frozen_buffer_update(self):
        model = _ResidualModel().eval()
        snapshot = snapshot_batchnorm_buffers(model)
        assert_batchnorm_buffers_equal(snapshot, model)
        model.base[1].running_mean.add_(1.0)
        with self.assertRaisesRegex(RuntimeError, "running_mean"):
            assert_batchnorm_buffers_equal(snapshot, model)


if __name__ == "__main__":
    unittest.main()
