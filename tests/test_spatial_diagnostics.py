"""Synthetic tests for programmatic spatial-error statistics."""

import unittest

import numpy as np

from scripts.spatial_diagnostics_common import (
    bootstrap_oracle_gain,
    build_spatial_region_masks,
    component_geometry_masks,
    confusion_from_arrays,
    image_origin_mixed_patch_mask,
    mean_iou_from_confusion,
    oracle_confusion,
    region_summary,
    semantic_boundary_mask,
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


if __name__ == "__main__":
    unittest.main()
