"""Synthetic tests for programmatic spatial-error statistics."""

import unittest

import numpy as np
import torch

from scripts.spatial_diagnostics_common import (
    bootstrap_oracle_gain,
    bootstrap_paired_miou_delta,
    build_spatial_region_masks,
    component_geometry_masks,
    confusion_from_arrays,
    common_translation_slices,
    image_origin_mixed_patch_mask,
    mean_iou_from_confusion,
    oracle_confusion,
    region_summary,
    semantic_boundary_mask,
)
from scripts.evaluate_whu_translation_consistency import translate_tensor
from scripts.evaluate_whu_phase_ensemble import (
    aligned_two_view_prediction,
    efficacy_decision,
)


class SpatialDiagnosticsTest(unittest.TestCase):
    def test_confusion_excludes_ignore_target(self):
        target = np.array([[0, 1], [2, 7]], dtype=np.int64)
        prediction = np.array([[0, 0], [2, 1]], dtype=np.int64)
        confusion = confusion_from_arrays(prediction, target, num_classes=3)
        np.testing.assert_array_equal(
            confusion,
            np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]]),
        )

    def test_boundary_marks_both_sides_of_transition(self):
        target = np.array(
            [[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int64
        )
        boundary = semantic_boundary_mask(target, num_classes=2)
        expected = np.array(
            [[False, True, True, False], [False, True, True, False]]
        )
        np.testing.assert_array_equal(boundary, expected)

    def test_mixed_patch_mask_is_image_origin_aligned(self):
        target = np.array(
            [
                [0, 0, 1, 1],
                [0, 1, 1, 1],
                [2, 2, 2, 2],
                [2, 2, 2, 2],
            ],
            dtype=np.int64,
        )
        actual = image_origin_mixed_patch_mask(target, 3, patch_size=2)
        expected = np.zeros_like(target, dtype=bool)
        expected[:2, :2] = True
        np.testing.assert_array_equal(actual, expected)

    def test_component_area_mask_selects_small_semantic_region(self):
        target = np.zeros((6, 6), dtype=np.int64)
        target[0, 0] = 1
        target[3:6, 3:6] = 1
        masks, metadata = component_geometry_masks(
            target,
            num_classes=2,
            area_thresholds=(1, 9),
            thickness_thresholds=(2, 8),
        )
        expected_small = np.zeros_like(target, dtype=bool)
        expected_small[0, 0] = True
        np.testing.assert_array_equal(masks["component_area_le_1px2"], expected_small)
        self.assertEqual(metadata["component_semantics"], "class-wise semantic regions, not object instances")

    def test_oracle_repairs_only_selected_confusion(self):
        baseline = np.array([[8, 2], [1, 9]], dtype=np.int64)
        region = np.array([[1, 2], [0, 1]], dtype=np.int64)
        oracle = oracle_confusion(baseline, region)
        np.testing.assert_array_equal(oracle, np.array([[10, 0], [1, 9]]))
        self.assertGreater(
            mean_iou_from_confusion(oracle), mean_iou_from_confusion(baseline)
        )

    def test_region_summary_reports_coverage_risk_and_class_gain(self):
        baseline = np.array([[8, 2], [1, 9]], dtype=np.int64)
        region = np.array([[1, 1], [0, 1]], dtype=np.int64)
        summary = region_summary(baseline, region, ["a", "b"])
        self.assertEqual(summary["pixels"], 3)
        self.assertEqual(summary["errors"], 1)
        self.assertAlmostEqual(summary["coverage"], 0.15)
        self.assertGreater(summary["relative_error_risk"], 1.0)
        self.assertGreater(summary["error_enrichment_over_global"], 1.0)
        self.assertGreater(summary["oracle_gain_pp"], 0.0)
        self.assertGreater(summary["per_class"]["a"]["relative_error_risk"], 1.0)
        self.assertGreater(summary["per_class"]["a"]["oracle_iou_gain_pp"], 0.0)

    def test_actionable_union_is_boolean_union_not_added_scores(self):
        target = np.zeros((8, 8), dtype=np.int64)
        target[:, 4:] = 1
        target[0, 0] = 2
        masks, metadata = build_spatial_region_masks(
            target,
            num_classes=3,
            boundary_radii=(1,),
            component_area_thresholds=(1,),
            component_thickness_thresholds=(2,),
            patch_size=4,
            union_boundary_radius=1,
            union_component_area=1,
            union_component_thickness=2,
        )
        expected = (
            masks["boundary_le_1px"]
            | masks["component_area_le_1px2"]
            | masks["component_thickness_le_2px"]
        )
        np.testing.assert_array_equal(masks["actionable_union"], expected)
        self.assertEqual(len(metadata["actionable_union_sources"]), 3)

    def test_bootstrap_is_deterministic_for_fixed_samples(self):
        baselines = np.array(
            [
                [[8, 2], [1, 9]],
                [[7, 3], [2, 8]],
            ],
            dtype=np.int64,
        )
        regions = np.array(
            [
                [[1, 2], [0, 1]],
                [[1, 1], [1, 1]],
            ],
            dtype=np.int64,
        )
        samples = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int64)
        first = bootstrap_oracle_gain(baselines, regions, samples)
        second = bootstrap_oracle_gain(baselines, regions, samples)
        self.assertEqual(first, second)
        self.assertEqual(first["replicates"], 3)

    def test_common_translation_slices_use_one_shared_field_of_view(self):
        original, shifted = common_translation_slices(
            (100, 120), ((0, 1), (0, 8), (0, 16)), margin=10
        )
        self.assertEqual(original, (slice(10, 90), slice(10, 94)))
        self.assertEqual(shifted[(0, 1)], (slice(10, 90), slice(11, 95)))
        self.assertEqual(shifted[(0, 16)], (slice(10, 90), slice(26, 110)))

    def test_translate_and_inverse_slices_recover_original_values(self):
        source = torch.arange(1 * 1 * 8 * 10).reshape(1, 1, 8, 10)
        translated = translate_tensor(source, dy=1, dx=2)
        original_slice, shifted = common_translation_slices(
            (8, 10), ((1, 2),), margin=1
        )
        np.testing.assert_array_equal(
            source.numpy()[0, 0][original_slice],
            translated.numpy()[0, 0][shifted[(1, 2)]],
        )
        self.assertTrue(
            torch.equal(
                translated[..., 0, :], torch.zeros_like(translated[..., 0, :])
            )
        )

    def test_paired_miou_bootstrap_reports_candidate_minus_reference(self):
        references = np.array(
            [
                [[8, 2], [1, 9]],
                [[7, 3], [2, 8]],
            ],
            dtype=np.int64,
        )
        candidates = np.array(
            [
                [[9, 1], [1, 9]],
                [[8, 2], [1, 9]],
            ],
            dtype=np.int64,
        )
        samples = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int64)
        result = bootstrap_paired_miou_delta(references, candidates, samples)
        self.assertGreater(result["median_pp"], 0.0)
        self.assertGreater(result["ci95_pp"][0], 0.0)

    def test_phase_ensemble_changes_only_the_shared_valid_region(self):
        baseline = torch.zeros((1, 2, 4, 6), dtype=torch.float32)
        baseline[:, 0] = 2.0
        shifted = torch.zeros_like(baseline)
        shifted[:, 1] = 4.0
        original = (slice(1, 3), slice(1, 4))
        shifted_slice = (slice(1, 3), slice(2, 5))
        prediction = aligned_two_view_prediction(
            baseline, shifted, original, shifted_slice
        )
        expected = np.zeros((1, 4, 6), dtype=np.int64)
        expected[0][original] = 1
        np.testing.assert_array_equal(prediction, expected)

    def test_phase_ensemble_efficacy_rule_is_prospective_and_conjunctive(self):
        names = ["farm", "city", "village", "water", "forest", "road", "other"]
        primary = {
            "candidate_minus_baseline_miou_pp": 0.12,
            "class_candidate_minus_baseline_iou_pp": {
                name: 0.0 for name in names
            },
        }
        control = {"candidate_minus_baseline_miou_pp": 0.06}
        self.assertEqual(
            efficacy_decision(primary, control, names)["outcome"], "GO"
        )
        primary["class_candidate_minus_baseline_iou_pp"]["city"] = -0.01
        primary["class_candidate_minus_baseline_iou_pp"]["road"] = -0.01
        self.assertEqual(
            efficacy_decision(primary, control, names)["outcome"], "STOP"
        )


if __name__ == "__main__":
    unittest.main()
