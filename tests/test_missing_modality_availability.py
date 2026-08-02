"""Checks for the two-slot availability path used by EarthMiss V1."""

from pathlib import Path
import sys
import unittest

import torch


SEGMENTATION_ROOT = Path(__file__).resolve().parents[1] / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))

from models.MMDINO.availability import (  # noqa: E402
    active_modality_indices,
    canonical_availability,
)
from models.MMDINO.diagnostics import FeatureZeroSampleAdapter  # noqa: E402
from models.MMDINO.sample_adapter import SampleAdapter  # noqa: E402
from models.MMDINO.dino_segment import DINOSegmentModule  # noqa: E402
from utils.inference import slide_inference  # noqa: E402


class AvailabilityTest(unittest.TestCase):
    def test_named_states_use_fixed_rgb_sar_slots(self):
        self.assertEqual(canonical_availability("full", batch_size=1).tolist(), [[True, True]])
        self.assertEqual(canonical_availability("sar", batch_size=2).tolist(), [[False, True]] * 2)
        self.assertEqual(canonical_availability("rgb", batch_size=1).tolist(), [[True, False]])

    def test_batch_state_must_be_homogeneous_and_nonempty(self):
        with self.assertRaisesRegex(ValueError, "one availability state per batch"):
            active_modality_indices(
                torch.tensor([[True, True], [False, True]]),
                batch_size=2,
                num_modalities=2,
            )
        with self.assertRaisesRegex(ValueError, "At least one modality"):
            active_modality_indices(
                torch.tensor([[False, False]]),
                batch_size=1,
                num_modalities=2,
            )

    def test_single_active_branch_is_normalized_over_that_branch_only(self):
        adapter = SampleAdapter(
            in_channels=4,
            out_channels=[2, 2, 2, 2],
            num_modalities=2,
        )
        sar_features = tuple(torch.randn(1, 4, 4) for _ in range(4))

        outputs = adapter(
            sar_features,
            patch_h=2,
            patch_w=2,
            modality_indices=(1,),
        )

        self.assertEqual(len(outputs), 2)
        self.assertEqual([len(slot) for slot in outputs], [4, 4])
        for first_slot, second_slot in zip(*outputs):
            self.assertTrue(torch.equal(first_slot, second_slot))

    def test_full_availability_matches_the_legacy_full_adapter_path(self):
        torch.manual_seed(7)
        adapter = SampleAdapter(
            in_channels=4,
            out_channels=[2, 2, 2, 2],
            num_modalities=2,
        )
        rgb_features = tuple(torch.randn(1, 4, 4) for _ in range(4))
        sar_features = tuple(torch.randn(1, 4, 4) for _ in range(4))

        legacy = adapter(rgb_features, sar_features, patch_h=2, patch_w=2)
        canonical = adapter(
            rgb_features,
            sar_features,
            patch_h=2,
            patch_w=2,
            modality_indices=(0, 1),
        )

        for legacy_slot, canonical_slot in zip(legacy, canonical):
            for legacy_feature, canonical_feature in zip(legacy_slot, canonical_slot):
                self.assertTrue(torch.equal(legacy_feature, canonical_feature))

    def test_feature_zero_no_mask_is_exactly_the_released_full_path(self):
        torch.manual_seed(19)
        adapter = SampleAdapter(
            in_channels=4,
            out_channels=[2, 2, 2, 2],
            num_modalities=2,
        )
        rgb_features = tuple(torch.randn(1, 4, 4) for _ in range(4))
        sar_features = tuple(torch.randn(1, 4, 4) for _ in range(4))

        expected = adapter(
            rgb_features,
            sar_features,
            patch_h=2,
            patch_w=2,
        )
        actual = FeatureZeroSampleAdapter(
            adapter,
            zero_modality_index=None,
        )(
            rgb_features,
            sar_features,
            patch_h=2,
            patch_w=2,
        )

        for expected_slot, actual_slot in zip(expected, actual, strict=True):
            for expected_feature, actual_feature in zip(
                expected_slot, actual_slot, strict=True
            ):
                self.assertTrue(torch.equal(expected_feature, actual_feature))

    def test_feature_zero_keeps_original_denominator_and_two_slots(self):
        torch.manual_seed(23)
        adapter = SampleAdapter(
            in_channels=4,
            out_channels=[2, 2, 2, 2],
            num_modalities=2,
        )
        adapter.modality_weights["weight_modality_0"].data.fill_(-0.4)
        adapter.modality_weights["weight_modality_1"].data.fill_(0.9)
        rgb_features = tuple(torch.randn(1, 4, 4) for _ in range(4))
        sar_features = tuple(torch.randn(1, 4, 4) for _ in range(4))

        actual = FeatureZeroSampleAdapter(
            adapter,
            zero_modality_index=0,
        )(
            rgb_features,
            sar_features,
            patch_h=2,
            patch_w=2,
        )

        weights = [
            torch.sigmoid(adapter.modality_weights[f"weight_modality_{index}"])
            for index in range(2)
        ]
        sar_weight_with_original_denominator = weights[1] / sum(weights)
        self.assertEqual(len(actual), 2)
        self.assertEqual([len(slot) for slot in actual], [4, 4])
        for layer_index, sar_feature in enumerate(sar_features):
            sar_processed = sar_feature.permute(0, 2, 1).reshape(1, 4, 2, 2)
            sar_processed = adapter.projects[layer_index](sar_processed)
            sar_processed = adapter.resize_layers[layer_index](sar_processed)
            expected = sar_weight_with_original_denominator * sar_processed
            torch.testing.assert_close(
                actual[0][layer_index], expected, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                actual[1][layer_index], expected, rtol=0.0, atol=0.0
            )
            self.assertFalse(torch.equal(actual[0][layer_index], sar_processed))

    def test_feature_zero_model_path_runs_both_backbones_and_two_decoder_slots(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def get_intermediate_layers(self, tensor, n):
                self.calls.append(tensor.detach().clone())
                batch = tensor.shape[0]
                value = tensor.mean(dim=(1, 2, 3), keepdim=True)
                return tuple(
                    value.reshape(batch, 1, 1).expand(batch, 4, 4)
                    for _ in range(4)
                )

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.slot_count = None

            def forward(self, *slots):
                self.slot_count = len(slots)
                batch = slots[0][0].shape[0]
                return torch.zeros(batch, 8, 32, 32)

        model = DINOSegmentModule.__new__(DINOSegmentModule)
        torch.nn.Module.__init__(model)
        model.num_modalities = 2
        model.use_optical_stem = False
        model.backbone_type = "dinov3_vits16"
        model.backbone = Backbone()
        model.adapter = FeatureZeroSampleAdapter(
            SampleAdapter(
                in_channels=4,
                out_channels=[2, 2, 2, 2],
                num_modalities=2,
            ),
            zero_modality_index=0,
        )
        model.decoder = Decoder()

        output = model(
            torch.full((2, 3, 32, 32), 11.0),
            torch.full((2, 1, 32, 32), 23.0),
        )

        self.assertEqual(tuple(output.shape), (2, 8, 32, 32))
        self.assertEqual(len(model.backbone.calls), 2)
        self.assertEqual(model.decoder.slot_count, 2)

    def test_sar_only_skips_rgb_backbone_and_keeps_two_decoder_slots(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def get_intermediate_layers(self, tensor, n):
                self.calls.append(tensor.detach().clone())
                batch = tensor.shape[0]
                return tuple(torch.zeros(batch, 4, 2) for _ in range(4))

        class Adapter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.indices = None

            def forward(self, *features, patch_h, patch_w, guidance, modality_indices):
                self.indices = tuple(modality_indices)
                batch = features[0][0].shape[0]
                fused = [torch.zeros(batch, 2, patch_h, patch_w) for _ in range(4)]
                return [fused, list(fused)]

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.slot_count = None

            def forward(self, *slots):
                self.slot_count = len(slots)
                batch = slots[0][0].shape[0]
                return torch.zeros(batch, 8, 32, 32)

        model = DINOSegmentModule.__new__(DINOSegmentModule)
        torch.nn.Module.__init__(model)
        model.num_modalities = 2
        model.use_optical_stem = False
        model.backbone_type = "dinov3_vits16"
        model.backbone = Backbone()
        model.adapter = Adapter()
        model.decoder = Decoder()

        rgb = torch.full((2, 3, 32, 32), 11.0)
        sar = torch.full((2, 1, 32, 32), 23.0)
        output = model(
            rgb,
            sar,
            availability=canonical_availability("sar", batch_size=2),
        )

        self.assertEqual(tuple(output.shape), (2, 8, 32, 32))
        self.assertEqual(len(model.backbone.calls), 1)
        self.assertEqual(tuple(model.backbone.calls[0].shape), (2, 3, 32, 32))
        self.assertTrue(torch.all(model.backbone.calls[0] == 23.0))
        self.assertEqual(model.adapter.indices, (1,))
        self.assertEqual(model.decoder.slot_count, 2)

    def test_sliding_inference_expands_availability_to_crop_batch(self):
        class Segmenter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.availability_shapes = []

            def forward(self, rgb, sar, availability):
                self.availability_shapes.append(tuple(availability.shape))
                self.asserted_state = availability.detach().cpu().tolist()
                return torch.zeros(rgb.shape[0], 3, *rgb.shape[-2:])

        segmenter = Segmenter()
        rgb = torch.zeros(1, 3, 4, 4)
        sar = torch.zeros(1, 1, 4, 4)
        output = slide_inference(
            rgb,
            segmenter,
            n_output_channels=3,
            crop_size=(2, 2),
            stride=(2, 2),
            dsm=sar,
            availability=canonical_availability("sar", batch_size=1),
            batch_size=2,
        )

        self.assertEqual(tuple(output.shape), (1, 3, 4, 4))
        self.assertEqual(segmenter.availability_shapes, [(2, 2), (2, 2)])
        self.assertEqual(segmenter.asserted_state, [[False, True], [False, True]])


if __name__ == "__main__":
    unittest.main()
