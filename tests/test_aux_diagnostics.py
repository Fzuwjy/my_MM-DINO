"""Unit tests for read-only auxiliary diagnostic helpers."""

from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn as nn

from scripts.aux_diagnostics_common import (
    AuxiliaryConditionDataset,
    FusionPreservingSingleInputDecoder,
    ScaledSampleAdapter,
    array_summary,
    edge_alignment_summary,
    fixed_derangement,
    grouped_derangement,
    modality_weight_summary,
)
from scripts.evaluate_whu_aux_counterfactual import (
    MODEL_PROFILES,
    _condition_auxiliary_scale,
    _condition_input_modalities,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyMultimodalDataset:
    def __init__(self, length: int = 5):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rgb = torch.full((3, 2, 2), float(index))
        auxiliary = torch.tensor(
            [[[float(index), float(index + 1)], [float(index + 2), float(index + 3)]]]
        )
        label = np.full((2, 2), index, dtype=np.int64)
        return rgb, auxiliary, label


class ReferenceAdapter(nn.Module):
    """Small exact analogue of the released multimodal SampleAdapter."""

    def __init__(self):
        super().__init__()
        self.projects = nn.ModuleList([nn.Conv2d(2, 2, kernel_size=1)])
        self.resize_layers = nn.ModuleList([nn.Identity()])
        self.num_modalities = 2
        self.modality_weights = nn.ParameterDict(
            {
                "weight_modality_0": nn.Parameter(torch.tensor([0.2])),
                "weight_modality_1": nn.Parameter(torch.tensor([0.8])),
            }
        )

    def forward(self, *features_list, patch_h=None, patch_w=None):
        if len(features_list) == 1:
            outputs = []
            for layer_index, feature in enumerate(features_list[0]):
                feature = feature.permute(0, 2, 1).reshape(
                    feature.shape[0], feature.shape[-1], patch_h, patch_w
                )
                outputs.append(self.projects[layer_index](feature))
            return outputs
        outputs = [[] for _ in features_list]
        for layer_index, modality_features in enumerate(zip(*features_list)):
            processed = []
            for feature in modality_features:
                feature = feature.permute(0, 2, 1).reshape(
                    feature.shape[0], feature.shape[-1], patch_h, patch_w
                )
                processed.append(self.projects[layer_index](feature))
            weights = [
                torch.sigmoid(self.modality_weights[f"weight_modality_{index}"])
                for index in range(len(processed))
            ]
            total = sum(weights)
            normalized = [weight / total for weight in weights]
            fused = sum(
                weight * feature
                for weight, feature in zip(normalized, processed, strict=True)
            )
            for output in outputs:
                output.append(fused)
        return outputs


class SumDecoder(nn.Module):
    def forward(self, *modalities):
        return sum(modality[0] for modality in modalities)


class AuxDiagnosticsTest(unittest.TestCase):
    def test_array_summary_reports_quantization(self):
        values = np.array([[0, 0, 1, 1]], dtype=np.uint8)
        summary = array_summary(values)
        self.assertEqual(summary["shape"], [1, 4])
        self.assertEqual(summary["unique_count_sampled"], 2)
        self.assertEqual(summary["nonzero_ratio"], 0.5)

    def test_fixed_derangement_is_deterministic_and_has_no_fixed_points(self):
        first, offset = fixed_derangement(20, 42)
        second, second_offset = fixed_derangement(20, 42)
        self.assertEqual(first, second)
        self.assertEqual(offset, second_offset)
        self.assertTrue(all(index != value for index, value in enumerate(first)))
        self.assertEqual(sorted(first), list(range(20)))

    def test_aux_mean_removes_spatial_structure(self):
        dataset = AuxiliaryConditionDataset(TinyMultimodalDataset(), "aux-mean")
        _, auxiliary, _ = dataset[2]
        self.assertTrue(torch.equal(auxiliary, torch.full_like(auxiliary, 3.5)))

    def test_aux_shuffle_uses_a_different_item(self):
        dataset = AuxiliaryConditionDataset(
            TinyMultimodalDataset(), "aux-shuffle", seed=42
        )
        for index in range(len(dataset)):
            _, auxiliary, _ = dataset[index]
            self.assertNotEqual(float(auxiliary[0, 0, 0]), float(index))

    def test_grouped_derangement_preserves_compatibility(self):
        group_keys = ["wide", "wide", "wide", "narrow", "narrow"]
        permutation, metadata = grouped_derangement(group_keys, seed=42)
        self.assertEqual(sorted(permutation), list(range(len(group_keys))))
        for index, shuffled_index in enumerate(permutation):
            self.assertNotEqual(index, shuffled_index)
            self.assertEqual(group_keys[index], group_keys[shuffled_index])
        self.assertEqual([len(group["indices"]) for group in metadata], [3, 2])

    def test_grouped_derangement_rejects_singleton_groups(self):
        with self.assertRaisesRegex(ValueError, "at least 2 items"):
            grouped_derangement(["wide", "wide", "narrow"], seed=42)

    def test_aux_shuffle_accepts_compatible_group_keys(self):
        group_keys = ["a", "a", "a", "b", "b"]
        dataset = AuxiliaryConditionDataset(
            TinyMultimodalDataset(),
            "aux-shuffle",
            seed=42,
            shuffle_group_keys=group_keys,
        )
        self.assertEqual(dataset.permutation_strategy, "grouped-cyclic-derangement")
        for index, shuffled_index in enumerate(dataset.permutation):
            self.assertEqual(group_keys[index], group_keys[shuffled_index])

    def test_scaled_adapter_scale_one_matches_released_behavior(self):
        torch.manual_seed(7)
        adapter = ReferenceAdapter()
        rgb = [torch.randn(2, 4, 2)]
        auxiliary = [torch.randn(2, 4, 2)]
        expected = adapter(rgb, auxiliary, patch_h=2, patch_w=2)
        actual = ScaledSampleAdapter(adapter, auxiliary_scale=1.0)(
            rgb, auxiliary, patch_h=2, patch_w=2
        )
        for expected_modality, actual_modality in zip(
            expected, actual, strict=True
        ):
            torch.testing.assert_close(expected_modality[0], actual_modality[0])

    def test_scaled_adapter_zero_removes_aux_and_renormalizes_rgb(self):
        torch.manual_seed(9)
        adapter = ReferenceAdapter()
        rgb = [torch.randn(1, 4, 2)]
        auxiliary = [torch.randn(1, 4, 2)]
        actual = ScaledSampleAdapter(adapter, auxiliary_scale=0.0)(
            rgb, auxiliary, patch_h=2, patch_w=2
        )
        expected = adapter.projects[0](
            rgb[0].permute(0, 2, 1).reshape(1, 2, 2, 2)
        )
        torch.testing.assert_close(actual[0][0], expected)
        torch.testing.assert_close(actual[1][0], expected)

    def test_modality_weight_summary_applies_aux_scale_before_normalization(self):
        adapter = ReferenceAdapter()
        normal = modality_weight_summary(adapter)
        disabled = modality_weight_summary(adapter, auxiliary_scale=0.0)
        self.assertAlmostEqual(sum(normal["effective_normalized"]), 1.0)
        self.assertEqual(disabled["effective_normalized"], [1.0, 0.0])

    def test_edge_alignment_finds_known_shift(self):
        rgb = np.zeros((16, 16), dtype=np.float32)
        rgb[4:10, 5:11] = 1.0
        auxiliary = np.roll(rgb, shift=(2, -1), axis=(0, 1))
        summary = edge_alignment_summary(
            rgb, auxiliary, offsets=(-2, -1, 0, 1, 2)
        )
        self.assertIsNotNone(summary["best"])
        self.assertNotEqual(
            (summary["best"]["dy"], summary["best"]["dx"]), (0, 0)
        )

    def test_counterfactual_scale_argument_validation(self):
        class Args:
            condition = "aux-feature-off"
            aux_weight_scale = None

        self.assertEqual(_condition_auxiliary_scale(Args()), 0.0)
        Args.condition = "aux-weight-scale"
        Args.aux_weight_scale = 2.0
        self.assertEqual(_condition_auxiliary_scale(Args()), 2.0)

    def test_native_rgb_only_condition_uses_one_inference_modality(self):
        self.assertEqual(_condition_input_modalities("rgb-only-native"), 1)
        self.assertEqual(
            _condition_input_modalities("rgb-only-fusion-preserved"), 1
        )
        self.assertEqual(_condition_input_modalities("normal"), 2)

        class Args:
            condition = "rgb-only-native"
            aux_weight_scale = None

        self.assertIsNone(_condition_auxiliary_scale(Args()))

    def test_rgb_only_fusion_preserved_matches_feature_off_endpoint(self):
        torch.manual_seed(13)
        adapter = ReferenceAdapter()
        rgb = [torch.randn(1, 4, 2)]
        auxiliary = [torch.randn(1, 4, 2)]

        feature_off = ScaledSampleAdapter(adapter, auxiliary_scale=0.0)(
            rgb, auxiliary, patch_h=2, patch_w=2
        )
        rgb_only = adapter(rgb, patch_h=2, patch_w=2)

        decoder = SumDecoder()
        expected = decoder(*feature_off)
        actual = FusionPreservingSingleInputDecoder(
            decoder, num_modalities=2
        )(rgb_only)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_checkpoint_profiles_lock_our_two_multimodal_runs(self):
        self.assertEqual(
            MODEL_PROFILES["vitl-lora"],
            {
                "backbone_type": "dinov3_vitl16",
                "use_lora": True,
                "lora_rank": 3,
            },
        )
        self.assertEqual(
            MODEL_PROFILES["vits-frozen"],
            {
                "backbone_type": "dinov3_vits16",
                "use_lora": False,
                "lora_rank": None,
            },
        )

    def test_released_transforms_share_geometry_and_preserve_label_interpolation(self):
        source = (
            REPO_ROOT
            / "tasks"
            / "segmentation"
            / "utils"
            / "transform.py"
        ).read_text(encoding="utf-8")
        self.assertIn("mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)", source)
        self.assertIn("dsm = cv2.resize(dsm, (ow, oh), interpolation=cv2.INTER_LINEAR)", source)
        self.assertIn("mask = mask.crop((x, y, x + size, y + size))", source)
        self.assertIn("dsm = dsm.crop((x, y, x + size, y + size))", source)


if __name__ == "__main__":
    unittest.main()
