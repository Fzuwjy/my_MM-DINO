"""Contracts for the frozen-base, BN-free EarthMiss V3 SAR residual."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F
from torch import nn

SEGMENTATION_ROOT = Path(__file__).resolve().parents[1] / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))

from models.MMDINO.Decoder import (  # noqa: E402
    Decoder,
    SARLogitResidualHead,
)
from models.MMDINO.availability import canonical_availability  # noqa: E402
from models.MMDINO.dino_segment import DINOSegmentModule  # noqa: E402


def _decoder(*, residual: bool) -> Decoder:
    return Decoder(
        n_classes=3,
        in_channels=[4, 4, 4, 4],
        out_channels=16,
        num_modalities=2,
        raw_logits=True,
        use_sar_logit_residual=residual,
        sar_logit_residual_seed=701,
        sar_logit_residual_channels=16,
    )


def _modalities(batch_size: int = 2) -> tuple[list[torch.Tensor], ...]:
    sizes = (8, 4, 2, 1)
    return tuple(
        [torch.randn(batch_size, 4, size, size) for size in sizes]
        for _ in range(2)
    )


class EarthMissV3SARResidualTest(unittest.TestCase):
    def test_branch_contains_no_batchnorm_and_starts_at_zero(self):
        branch = SARLogitResidualHead(16, 3, hidden_channels=16)
        self.assertFalse(
            any(isinstance(module, nn.modules.batchnorm._BatchNorm)
                for module in branch.modules())
        )
        output = branch(torch.randn(2, 16, 8, 8))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_opt_in_keeps_base_rng_and_common_state_identical(self):
        torch.manual_seed(41)
        base = _decoder(residual=False)
        base_next_random = torch.rand(8)

        torch.manual_seed(41)
        candidate = _decoder(residual=True)
        candidate_next_random = torch.rand(8)

        self.assertTrue(torch.equal(base_next_random, candidate_next_random))
        base_state = base.state_dict()
        candidate_state = candidate.state_dict()
        for key, value in base_state.items():
            self.assertTrue(torch.equal(value, candidate_state[key]), key)

    def test_step_zero_sar_and_full_outputs_are_bitwise_equal(self):
        torch.manual_seed(43)
        base = _decoder(residual=False).eval()
        torch.manual_seed(43)
        candidate = _decoder(residual=True).eval()
        modalities = _modalities()

        with torch.inference_mode():
            base_output = base(*modalities)
            candidate_full = candidate(*modalities)
            candidate_sar = candidate(
                *modalities,
                apply_sar_logit_residual=True,
            )
            exported_base, exported_sar = candidate(
                *modalities,
                apply_sar_logit_residual=True,
                return_base_logits=True,
            )
        self.assertTrue(torch.equal(base_output, candidate_full))
        self.assertTrue(torch.equal(base_output, candidate_sar))
        self.assertTrue(torch.equal(base_output, exported_base))
        self.assertTrue(torch.equal(base_output, exported_sar))

    def test_zero_projection_is_trainable_without_base_gradients(self):
        decoder = _decoder(residual=True).eval()
        decoder.requires_grad_(False)
        decoder.sar_logit_residual.requires_grad_(True)
        modalities = _modalities()
        logits = decoder(*modalities, apply_sar_logit_residual=True)
        labels = torch.randint(0, 3, logits.shape[:1] + logits.shape[2:])
        F.cross_entropy(logits, labels).backward()

        branch = decoder.sar_logit_residual
        self.assertGreater(float(branch.projection.weight.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for name, parameter in decoder.named_parameters()
                if not name.startswith("sar_logit_residual."))
        )

    def test_freeze_helper_leaves_only_residual_trainable_and_bn_in_eval(self):
        model = DINOSegmentModule.__new__(DINOSegmentModule)
        nn.Module.__init__(model)
        model.use_sar_logit_residual = True
        model.decoder = _decoder(residual=True)
        parameters = model.freeze_base_for_sar_logit_residual()

        self.assertEqual(
            {id(parameter) for parameter in parameters},
            {
                id(parameter)
                for parameter in model.decoder.sar_logit_residual.parameters()
            },
        )
        self.assertTrue(all(parameter.requires_grad for parameter in parameters))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if not name.startswith("decoder.sar_logit_residual.")
            )
        )
        self.assertTrue(
            all(
                not module.training
                for module in model.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
            )
        )

    def test_canonical_sar_enables_branch_and_full_bypasses_it(self):
        class Adapter(nn.Module):
            def forward(
                self,
                *features,
                patch_h,
                patch_w,
                guidance,
                modality_indices,
            ):
                batch = features[0][0].shape[0]
                slot = [
                    torch.zeros(batch, 4, patch_h, patch_w)
                    for _ in range(4)
                ]
                return [list(slot), list(slot)]

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.flags = []

            def forward(self, *slots, apply_sar_logit_residual=False):
                self.flags.append(apply_sar_logit_residual)
                batch = slots[0][0].shape[0]
                return torch.zeros(batch, 3, 2, 2)

        model = DINOSegmentModule.__new__(DINOSegmentModule)
        nn.Module.__init__(model)
        model.num_modalities = 2
        model.use_optical_stem = False
        model.use_sar_logit_residual = True
        model.adapter = Adapter()
        model.decoder = Decoder()

        rgb = torch.zeros(1, 3, 32, 32)
        sar = torch.zeros(1, 1, 32, 32)
        outputs = tuple(
            tuple(torch.zeros(1, 4, 4) for _ in range(4))
            for _ in range(2)
        )
        model.forward_from_backbone_outputs(
            rgb,
            sar,
            backbone_outputs=outputs,
            availability=canonical_availability("sar", batch_size=1),
        )
        model.forward_from_backbone_outputs(
            rgb,
            sar,
            backbone_outputs=outputs,
            availability=canonical_availability("full", batch_size=1),
        )
        self.assertEqual(model.decoder.flags, [True, False])


if __name__ == "__main__":
    unittest.main()
