"""Synthetic checks for intra-slide crop-response statistics."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.phase_overlap_common import (
    SCORE_NAMES,
    OverlapDisagreementAccumulator,
    semantic_boundary_observations,
    stable_softmax,
)


def _binary_logits(prediction: np.ndarray, magnitude: float = 8.0) -> np.ndarray:
    labels = np.asarray(prediction, dtype=np.int64)
    result = np.full((2, *labels.shape), -magnitude, dtype=np.float32)
    for class_index in range(2):
        result[class_index][labels == class_index] = magnitude
    return result


class StableSoftmaxTests(unittest.TestCase):
    def test_probabilities_are_float32_finite_and_normalized(self):
        logits = np.asarray(
            [
                [[1000.0, -1000.0], [2.0, 0.0]],
                [[999.0, -999.0], [0.0, 2.0]],
            ],
            dtype=np.float32,
        )
        probabilities = stable_softmax(logits)
        self.assertEqual(probabilities.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(probabilities)))
        np.testing.assert_allclose(
            probabilities.sum(axis=0), np.ones((2, 2)), rtol=0, atol=1e-7
        )

    def test_rejects_non_float32_and_nonfinite_inputs(self):
        with self.assertRaisesRegex(TypeError, "float32"):
            stable_softmax(np.zeros((2, 2, 2), dtype=np.float64))
        values = np.zeros((2, 2, 2), dtype=np.float32)
        values[0, 0, 0] = np.nan
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            stable_softmax(values)


class BoundaryObservationTests(unittest.TestCase):
    def test_outer_ring_is_invalid_and_internal_change_is_symmetric(self):
        prediction = np.asarray(
            [
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
            ],
            dtype=np.int64,
        )
        boundary, valid = semantic_boundary_observations(prediction)
        self.assertTrue(np.all(valid[0] == 0))
        self.assertTrue(np.all(valid[-1] == 0))
        self.assertTrue(np.all(valid[:, 0] == 0))
        self.assertTrue(np.all(valid[:, -1] == 0))
        self.assertTrue(np.all(boundary[1:-1, 1:3] == 1))
        self.assertTrue(np.all(boundary[1:-1, 3] == 0))


class OverlapAccumulatorTests(unittest.TestCase):
    def test_opposing_overlap_votes_produce_high_jsd_and_disagreement(self):
        accumulator = OverlapDisagreementAccumulator((3, 4), 2)
        crop0 = _binary_logits(np.zeros((3, 3), dtype=np.int64))
        crop1 = _binary_logits(np.ones((3, 3), dtype=np.int64))
        accumulator.add_crop(crop0, (0, 3, 0, 3))
        accumulator.add_crop(crop1, (0, 3, 1, 4))

        result = accumulator.cell_summaries(
            np.asarray([[0, 3, 0, 4]], dtype=np.int64),
            (0, 3, 0, 4),
        )
        jsd = result["score_means"][SCORE_NAMES[0]][0]
        votes = result["score_means"][SCORE_NAMES[1]][0]
        self.assertGreater(jsd, 0.99)
        self.assertEqual(votes, 1.0)
        self.assertEqual(result["coverage"]["overlap_pixels"], 6)
        self.assertAlmostEqual(result["coverage"]["overlap_fraction"], 0.5)
        self.assertEqual(
            result["score_valid_pixels"][SCORE_NAMES[0]].tolist(), [6]
        )
        self.assertEqual(
            result["score_common_intersection_pixels"][SCORE_NAMES[0]].tolist(),
            [12],
        )

        endpoint = accumulator.endpoint_prediction()
        self.assertEqual(endpoint.dtype, np.int64)
        self.assertTrue(np.all(endpoint[:, 0] == 0))
        self.assertTrue(np.all(endpoint[:, -1] == 1))

    def test_identical_crop_responses_have_zero_disagreement(self):
        accumulator = OverlapDisagreementAccumulator((4, 4), 2)
        prediction = np.zeros((4, 4), dtype=np.int64)
        logits = _binary_logits(prediction)
        accumulator.add_crop(logits, (0, 4, 0, 4))
        accumulator.add_crop(logits, (0, 4, 0, 4))
        result = accumulator.cell_summaries(
            np.asarray([[0, 4, 0, 4]], dtype=np.int64), (0, 4, 0, 4)
        )
        for name in SCORE_NAMES:
            self.assertAlmostEqual(result["score_means"][name][0], 0.0)

    def test_boundary_response_disagreement_uses_only_valid_observations(self):
        accumulator = OverlapDisagreementAccumulator((5, 5), 2)
        constant = np.zeros((5, 5), dtype=np.int64)
        split = constant.copy()
        split[:, 2:] = 1
        accumulator.add_crop(_binary_logits(constant), (0, 5, 0, 5))
        accumulator.add_crop(_binary_logits(split), (0, 5, 0, 5))
        result = accumulator.cell_summaries(
            np.asarray([[0, 5, 0, 5]], dtype=np.int64), (0, 5, 0, 5)
        )
        boundary_name = SCORE_NAMES[2]
        self.assertEqual(result["score_valid_pixels"][boundary_name][0], 9)
        # Six interior pixels lie on one of the two sides of the split.  One
        # crop votes boundary and the other does not, hence 4q(1-q)=1 there.
        self.assertAlmostEqual(
            result["score_means"][boundary_name][0], 6.0 / 9.0
        )

    def test_residue_metadata_is_descriptive_only(self):
        accumulator = OverlapDisagreementAccumulator((3, 4), 2)
        logits = _binary_logits(np.zeros((3, 3), dtype=np.int64))
        accumulator.add_crop(logits, (0, 3, 0, 3))
        accumulator.add_crop(logits, (0, 3, 1, 4))
        metadata = accumulator.residue_metadata(patch_size=2)
        self.assertEqual(metadata["unique_origin_residue_count"], 2)
        self.assertIn("not a separately screened score", metadata["role"])

    def test_rejects_uncovered_endpoint_and_wrong_crop_shape(self):
        accumulator = OverlapDisagreementAccumulator((4, 4), 2)
        with self.assertRaisesRegex(AssertionError, "uncovered"):
            accumulator.endpoint_prediction()
        with self.assertRaisesRegex(ValueError, "differs"):
            accumulator.add_crop(
                np.zeros((2, 2, 2), dtype=np.float32), (0, 3, 0, 3)
            )


if __name__ == "__main__":
    unittest.main()
