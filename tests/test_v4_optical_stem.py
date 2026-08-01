"""Focused contracts for the V4-C post-ACFM optical spatial path."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from tasks.segmentation.models.MMDINO.Decoder import Decoder, OpticalSpatialStem


def _decoder(*, use_optical_stem: bool) -> Decoder:
    return Decoder(
        n_classes=3,
        in_channels=[4, 4, 4, 4],
        out_channels=16,
        num_modalities=2,
        use_optical_stem=use_optical_stem,
        optical_stem_seed=104771,
    )


def _modalities(batch_size: int = 2) -> tuple[list[torch.Tensor], ...]:
    sizes = (8, 4, 2, 1)
    return tuple(
        [torch.randn(batch_size, 4, size, size) for size in sizes]
        for _ in range(2)
    )


class _IdentityFRM(nn.Module):
    def forward(self, *features):
        return features


class _SumFusion(nn.Module):
    def forward(self, *features):
        return sum(features)


class _CaptureNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = None

    def forward(self, features):
        self.features = features
        return features[:3]


class _CountingStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, guidance):
        self.calls += 1
        return torch.ones(
            guidance.shape[0], 16, 2, 2,
            dtype=guidance.dtype,
            device=guidance.device,
        )


class V4OpticalStemTest(unittest.TestCase):
    def test_default_state_dict_is_unchanged_and_stem_rng_is_isolated(self):
        torch.manual_seed(13)
        released = _decoder(use_optical_stem=False)
        released_next_random = torch.rand(8)

        torch.manual_seed(13)
        candidate = _decoder(use_optical_stem=True)
        candidate_next_random = torch.rand(8)

        released_state = released.state_dict()
        candidate_state = candidate.state_dict()
        self.assertFalse(any("optical_stem" in key for key in released_state))
        self.assertTrue(any("optical_stem" in key for key in candidate_state))
        _decoder(use_optical_stem=False).load_state_dict(
            released_state, strict=True
        )
        self.assertTrue(torch.equal(released_next_random, candidate_next_random))
        for key, value in released_state.items():
            self.assertTrue(torch.equal(value, candidate_state[key]), key)

    def test_stem_shape_and_input_contract(self):
        stem = OpticalSpatialStem(out_channels=16)
        output = stem(torch.randn(2, 3, 32, 48))
        self.assertEqual(output.shape, (2, 16, 8, 12))
        self.assertTrue(torch.count_nonzero(output) == 0)
        self.assertEqual(int(torch.count_nonzero(stem.projection.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(stem.projection.bias)), 0)

        with self.assertRaisesRegex(ValueError, r"\[B, 3, H, W\]"):
            stem(torch.randn(2, 1, 32, 48))
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            stem(torch.randn(2, 3, 31, 48))

    def test_step_zero_prediction_is_bitwise_equal(self):
        torch.manual_seed(29)
        released = _decoder(use_optical_stem=False).eval()
        torch.manual_seed(29)
        candidate = _decoder(use_optical_stem=True).eval()

        torch.manual_seed(31)
        modalities = _modalities()
        guidance = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            released_output = released(*modalities)
            candidate_output = candidate(*modalities, guidance=guidance)
        self.assertTrue(torch.equal(released_output, candidate_output))

    def test_zero_projection_delays_upstream_stem_gradient_one_step(self):
        torch.manual_seed(37)
        candidate = _decoder(use_optical_stem=True).train()
        modalities = _modalities()
        guidance = torch.randn(2, 3, 32, 32)
        logits = candidate(*modalities, guidance=guidance)
        labels = torch.randint(0, logits.shape[1], logits.shape[:1] + logits.shape[2:])
        F.cross_entropy(logits, labels).backward()

        stem = candidate.optical_stem
        self.assertIsNotNone(stem)
        self.assertGreater(float(stem.projection.weight.grad.abs().sum()), 0.0)
        first_conv = stem.features[0]
        self.assertIsNotNone(first_conv.weight.grad)
        self.assertEqual(float(first_conv.weight.grad.abs().sum()), 0.0)

        with torch.no_grad():
            stem.projection.weight.add_(
                stem.projection.weight.grad, alpha=-1.0e-3
            )
            stem.projection.bias.add_(stem.projection.bias.grad, alpha=-1.0e-3)
        candidate.zero_grad(set_to_none=True)
        second_logits = candidate(*modalities, guidance=guidance)
        F.cross_entropy(second_logits, labels).backward()
        self.assertIsNotNone(first_conv.weight.grad)
        self.assertGreater(float(first_conv.weight.grad.abs().sum()), 0.0)

    def test_optical_feature_is_injected_once_after_l0_fusion(self):
        decoder = _decoder(use_optical_stem=True)
        decoder.frm = _IdentityFRM()
        decoder.fusion1 = _SumFusion()
        decoder.fusion2 = _SumFusion()
        decoder.fusion3 = _SumFusion()
        decoder.fusion4 = _SumFusion()
        decoder.neck = _CaptureNeck()
        decoder.out_conv = nn.Identity()
        decoder.optical_stem = _CountingStem()

        first = [torch.full((1, 16, 2, 2), 2.0) for _ in range(4)]
        second = [torch.full((1, 16, 2, 2), 3.0) for _ in range(4)]
        output = decoder(first, second, guidance=torch.zeros(1, 3, 32, 32))

        self.assertEqual(decoder.optical_stem.calls, 1)
        self.assertTrue(torch.equal(output, torch.full_like(output, 6.0)))
        self.assertTrue(
            torch.equal(decoder.neck.features[1], torch.full_like(output, 5.0))
        )


if __name__ == "__main__":
    unittest.main()
