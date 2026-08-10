"""CPU contracts for the opt-in PRN P5-to-P4 flow alignment."""

from __future__ import annotations

import copy
import unittest

import torch
import torch.nn.functional as F

from tasks.segmentation.models.MMDINO.Decoder import (
    Decoder,
    ProgressiveRefinementNeck,
)
from tasks.segmentation.models.MMDINO.semantic_flow import (
    ResidualFlowAlignment,
    flow_warp,
    target_pixel_flow_grid,
)


class SemanticFlowCoordinateTests(unittest.TestCase):
    def test_target_pixel_flow_grid_validates_shape_and_dtype(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            target_pixel_flow_grid(torch.zeros(1, 3, 4, 4))
        with self.assertRaisesRegex(TypeError, "floating"):
            target_pixel_flow_grid(torch.zeros(1, 2, 4, 4, dtype=torch.long))

    def test_zero_flow_matches_pixel_center_bilinear_resize(self):
        source = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5)
        flow = torch.zeros(1, 2, 6, 10)
        actual = flow_warp(source, flow, padding_mode="border")
        expected = F.interpolate(
            source,
            size=(6, 10),
            mode="bilinear",
            align_corners=False,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

    def test_positive_sampling_dx_moves_an_impulse_left(self):
        source = torch.zeros(1, 1, 5, 5)
        source[0, 0, 2, 2] = 1
        flow = torch.zeros(1, 2, 5, 5)
        flow[:, 0] = 1
        actual = flow_warp(source, flow, padding_mode="border")
        maximum = torch.nonzero(actual == actual.max(), as_tuple=False)
        self.assertEqual(maximum.tolist(), [[0, 0, 2, 1]])
        self.assertEqual(actual.max().item(), 1.0)

    def test_border_padding_cannot_suppress_constant_features_by_sampling_outside(self):
        source = torch.ones(1, 1, 3, 3)
        flow = torch.full((1, 2, 3, 3), 100.0)
        actual = flow_warp(source, flow, padding_mode="border")
        self.assertTrue(torch.equal(actual, source))


class ResidualFlowAlignmentTests(unittest.TestCase):
    def _inputs(self):
        generator = torch.Generator().manual_seed(11)
        high = torch.randn(2, 4, 3, 5, generator=generator)
        low = torch.randn(2, 4, 6, 10, generator=generator)
        baseline = F.interpolate(high, size=low.shape[-2:], mode="nearest")
        return high, low, baseline

    def test_zero_initialization_is_exactly_the_nearest_baseline(self):
        module = ResidualFlowAlignment(4, flow_channels=2)
        high, low, baseline = self._inputs()
        actual = module(high, low, baseline_high=baseline)
        self.assertTrue(torch.equal(actual, baseline))
        self.assertEqual(
            torch.count_nonzero(module.flow_predictor.weight).item(), 0
        )

    def test_flow_predictor_gets_first_backward_gradient_then_projections_learn(self):
        module = ResidualFlowAlignment(4, flow_channels=2)
        high, low, baseline = self._inputs()
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

        module(high, low, baseline_high=baseline).square().mean().backward()
        first_flow_gradient = module.flow_predictor.weight.grad
        self.assertGreater(first_flow_gradient.abs().sum().item(), 0)
        self.assertEqual(module.high_projection.weight.grad.abs().sum().item(), 0)
        self.assertEqual(module.low_projection.weight.grad.abs().sum().item(), 0)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        module(high, low, baseline_high=baseline).square().mean().backward()
        self.assertGreater(
            module.high_projection.weight.grad.abs().sum().item(), 0
        )
        self.assertGreater(module.low_projection.weight.grad.abs().sum().item(), 0)

    def test_module_contains_no_batchnorm(self):
        module = ResidualFlowAlignment(8, flow_channels=4)
        self.assertFalse(
            any(isinstance(child, torch.nn.modules.batchnorm._BatchNorm)
                for child in module.modules())
        )

    def test_invalid_baseline_shape_is_rejected(self):
        module = ResidualFlowAlignment(4, flow_channels=2)
        high, low, _ = self._inputs()
        with self.assertRaisesRegex(ValueError, "wrong shape"):
            module(high, low, baseline_high=torch.zeros(2, 4, 5, 10))


class ProgressiveRefinementNeckFAMTests(unittest.TestCase):
    @staticmethod
    def _features():
        generator = torch.Generator().manual_seed(31)
        return [
            torch.randn(1, 4, 16, 16, generator=generator),
            torch.randn(1, 4, 8, 8, generator=generator),
            torch.randn(1, 4, 4, 4, generator=generator),
            torch.randn(1, 4, 2, 2, generator=generator),
        ]

    def test_attached_zero_init_fam_preserves_complete_neck_outputs(self):
        torch.manual_seed(7)
        baseline_neck = ProgressiveRefinementNeck([4, 4, 4, 4])
        candidate_neck = copy.deepcopy(baseline_neck)
        candidate_neck.attach_p5_p4_fam(
            ResidualFlowAlignment(4, flow_channels=2)
        )
        baseline_neck.eval()
        candidate_neck.eval()

        features = self._features()
        with torch.inference_mode():
            expected = baseline_neck(features)
            actual = candidate_neck(features)
        for left, right in zip(expected, actual, strict=True):
            self.assertTrue(torch.equal(left, right))

    def test_fam_is_called_once_and_other_top_down_resizes_are_unchanged(self):
        neck = ProgressiveRefinementNeck([4, 4, 4, 4])
        fam = ResidualFlowAlignment(4, flow_channels=2)
        neck.attach_p5_p4_fam(fam)
        calls = []
        handle = fam.register_forward_hook(
            lambda _module, inputs, output: calls.append(
                (tuple(inputs[0].shape), tuple(inputs[1].shape), tuple(output.shape))
            )
        )
        try:
            neck(self._features())
        finally:
            handle.remove()
        self.assertEqual(calls, [((1, 4, 2, 2), (1, 4, 4, 4), (1, 4, 4, 4))])

    def test_duplicate_attachment_is_rejected(self):
        neck = ProgressiveRefinementNeck([4, 4, 4, 4])
        neck.attach_p5_p4_fam(ResidualFlowAlignment(4, flow_channels=2))
        with self.assertRaisesRegex(RuntimeError, "already attached"):
            neck.attach_p5_p4_fam(ResidualFlowAlignment(4, flow_channels=2))


class DecoderFAMConstructionTests(unittest.TestCase):
    @staticmethod
    def _decoder(*, fam: bool, seed: int = 19, **kwargs):
        return Decoder(
            n_classes=3,
            in_channels=[16, 16, 16, 16],
            out_channels=16,
            num_modalities=2,
            raw_logits=True,
            use_prn_p5_p4_fam=fam,
            prn_p5_p4_fam_seed=seed,
            prn_p5_p4_fam_flow_channels=8,
            **kwargs,
        )

    def test_enabling_fam_preserves_base_initialization_and_rng_successor(self):
        torch.manual_seed(101)
        baseline = self._decoder(fam=False)
        baseline_rng = torch.get_rng_state().clone()

        torch.manual_seed(101)
        candidate = self._decoder(fam=True)
        candidate_rng = torch.get_rng_state().clone()

        baseline_state = baseline.state_dict()
        candidate_state = candidate.state_dict()
        self.assertTrue(torch.equal(baseline_rng, candidate_rng))
        self.assertTrue(set(baseline_state).issubset(candidate_state))
        for key, value in baseline_state.items():
            self.assertTrue(torch.equal(value, candidate_state[key]), key)
        new_keys = set(candidate_state) - set(baseline_state)
        self.assertEqual(
            new_keys,
            {
                "neck.p5_p4_fam.high_projection.weight",
                "neck.p5_p4_fam.low_projection.weight",
                "neck.p5_p4_fam.flow_predictor.weight",
            },
        )

    def test_fam_seed_changes_only_projection_initialization(self):
        torch.manual_seed(211)
        first = self._decoder(fam=True, seed=1).state_dict()
        torch.manual_seed(211)
        second = self._decoder(fam=True, seed=2).state_dict()
        differing = {key for key in first if not torch.equal(first[key], second[key])}
        self.assertEqual(
            differing,
            {
                "neck.p5_p4_fam.high_projection.weight",
                "neck.p5_p4_fam.low_projection.weight",
            },
        )

    def test_zero_init_fam_preserves_complete_decoder_logits(self):
        torch.manual_seed(313)
        baseline = self._decoder(fam=False).eval()
        torch.manual_seed(313)
        candidate = self._decoder(fam=True).eval()
        generator = torch.Generator().manual_seed(17)
        modality = (
            torch.randn(1, 16, 16, 16, generator=generator),
            torch.randn(1, 16, 8, 8, generator=generator),
            torch.randn(1, 16, 4, 4, generator=generator),
            torch.randn(1, 16, 2, 2, generator=generator),
        )
        with torch.inference_mode():
            expected = baseline(modality, modality)
            actual = candidate(modality, modality)
        self.assertTrue(torch.equal(actual, expected))

    def test_baseline_checkpoint_has_only_the_three_registered_missing_keys(self):
        torch.manual_seed(419)
        baseline_state = self._decoder(fam=False).state_dict()
        torch.manual_seed(419)
        candidate = self._decoder(fam=True)
        incompatible = candidate.load_state_dict(baseline_state, strict=False)
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(
            set(incompatible.missing_keys),
            {
                "neck.p5_p4_fam.high_projection.weight",
                "neck.p5_p4_fam.low_projection.weight",
                "neck.p5_p4_fam.flow_predictor.weight",
            },
        )

    def test_fam_cannot_be_combined_with_existing_opt_in_branches(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self._decoder(fam=True, use_sar_logit_residual=True)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self._decoder(fam=True, use_optical_stem=True)


if __name__ == "__main__":
    unittest.main()
