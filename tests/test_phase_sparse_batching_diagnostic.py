"""CPU tests for the Stage-B1 sparse batch-composition diagnostic helpers."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.diagnose_whu_phase_sparse_batching import (
    _chunked_logit_difference,
    _difference_statistics,
    _merge_difference_statistics,
    _prediction_comparison,
    classify_diagnostic,
)


class SparseBatchingDiagnosticTest(unittest.TestCase):
    def test_crop_difference_counts_numeric_and_argmax_changes(self):
        reference = np.zeros((2, 2, 3), dtype=np.float32)
        reference[0] = 1.0
        candidate = reference.copy()
        candidate[0, 1, 2] = 0.0
        candidate[1, 1, 2] = 2.0
        result = _difference_statistics(
            candidate, reference, atol=1e-6, rtol=0.0
        )
        self.assertFalse(result["array_equal"])
        self.assertFalse(result["allclose"])
        self.assertEqual(result["elements_exceeding_tolerance"], 2)
        self.assertEqual(result["argmax_prediction_flips"], 1)
        self.assertEqual(result["maximum_absolute_difference"], 2.0)

    def test_chunked_difference_matches_small_exact_case(self):
        reference = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4)
        candidate = reference.copy()
        candidate[2, 4, 3] += np.float32(0.25)
        result = _chunked_logit_difference(
            candidate, reference, atol=0.1, rtol=0.0, rows=2
        )
        self.assertFalse(result["array_equal"])
        self.assertFalse(result["allclose"])
        self.assertEqual(result["elements_exceeding_tolerance"], 1)
        self.assertAlmostEqual(result["maximum_absolute_difference"], 0.25)

    def test_merged_crop_summary_is_weighted_by_elements(self):
        first = {
            "array_equal": True,
            "allclose": True,
            "element_count": 4,
            "mean_absolute_difference": 0.0,
            "root_mean_square_difference": 0.0,
            "maximum_absolute_difference": 0.0,
            "argmax_prediction_flips": 0,
            "argmax_pixel_count": 2,
            "elements_exceeding_tolerance": 0,
        }
        second = {
            **first,
            "array_equal": False,
            "allclose": False,
            "element_count": 12,
            "mean_absolute_difference": 2.0,
            "root_mean_square_difference": 2.0,
            "maximum_absolute_difference": 2.0,
            "argmax_prediction_flips": 3,
            "argmax_pixel_count": 6,
            "elements_exceeding_tolerance": 5,
        }
        result = _merge_difference_statistics([first, second])
        self.assertEqual(result["array_equal_crop_count"], 1)
        self.assertEqual(result["allclose_crop_count"], 1)
        self.assertEqual(result["argmax_prediction_flips"], 3)
        self.assertAlmostEqual(result["mean_absolute_difference"], 1.5)

    def test_classification_separates_partial_shape_from_membership(self):
        def record(flips: int, allclose: bool, final_equal: bool):
            return {
                "phases": {
                    "x8": {
                        "execution": {
                            "raw_crop_summary": {
                                "argmax_prediction_flips": flips,
                                "array_equal_crop_count": int(flips == 0),
                                "allclose_crop_count": int(allclose),
                                "crop_count": 1,
                            }
                        }
                    }
                },
                "final_k4": {"prediction_equal": final_equal},
            }

        result = classify_diagnostic(
            {
                "compact": record(1, False, False),
                "compact-pad-last": record(0, True, True),
                "dense-preserving": record(0, True, True),
            },
            final_available=True,
        )
        self.assertEqual(
            result["outcome"],
            "CONFIRMED_FINAL_PARTIAL_BATCH_SHAPE_SENSITIVITY",
        )

        membership = classify_diagnostic(
            {
                "compact": record(1, False, False),
                "compact-pad-last": record(1, False, False),
                "dense-preserving": record(0, True, True),
            },
            final_available=True,
        )
        self.assertEqual(
            membership["outcome"],
            "CONFIRMED_COMPACT_BATCH_MEMBERSHIP_SENSITIVITY",
        )

        unresolved = classify_diagnostic(
            {
                "compact": record(1, False, False),
                "compact-pad-last": record(1, False, False),
                "dense-preserving": record(1, False, False),
            },
            final_available=True,
        )
        self.assertEqual(
            unresolved["outcome"],
            "NOT_ISOLATED_DENSE_PRESERVING_ALSO_DIFFERS",
        )

    def test_allclose_but_nonexact_crop_is_still_numerical_drift(self):
        def record(exact: bool):
            return {
                "phases": {
                    "x8": {
                        "execution": {
                            "raw_crop_summary": {
                                "argmax_prediction_flips": 0,
                                "array_equal_crop_count": int(exact),
                                "allclose_crop_count": 1,
                                "crop_count": 1,
                            }
                        }
                    }
                },
                "final_k4": {"prediction_equal": exact},
            }

        result = classify_diagnostic(
            {
                "compact": record(False),
                "compact-pad-last": record(True),
                "dense-preserving": record(True),
            },
            final_available=True,
        )
        self.assertTrue(result["compact_raw_crop_numerical_drift"])
        self.assertEqual(
            result["outcome"],
            "CONFIRMED_FINAL_PARTIAL_BATCH_SHAPE_SENSITIVITY",
        )

    def test_final_flip_records_margin_and_coordinates(self):
        reference = np.zeros((7, 2, 2), dtype=np.float32)
        candidate = reference.copy()
        reference[0] = 1.0
        candidate[0, 1, 1] = 0.999
        candidate[1, 1, 1] = 1.001
        target = np.zeros((2, 2), dtype=np.int64)
        result = _prediction_comparison(
            candidate,
            reference,
            target,
            [f"class_{index}" for index in range(7)],
            common_bounds=(0, 2, 0, 2),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertEqual(result["prediction_flip_count"], 1)
        examples = result["flip_margin_diagnostic"][
            "examples_sorted_by_reference_margin"
        ]
        self.assertEqual((examples[0]["y"], examples[0]["x"]), (1, 1))
        self.assertEqual(examples[0]["reference_class_index"], 0)
        self.assertEqual(examples[0]["candidate_class_index"], 1)


if __name__ == "__main__":
    unittest.main()
