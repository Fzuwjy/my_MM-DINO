"""CPU contracts for the EarthMiss scale-transition diagnostic."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from models.MMDINO.Decoder import Decoder  # noqa: E402
from models.MMDINO.sample_adapter import SampleAdapter  # noqa: E402
from scripts.diagnose_earthmiss_scale_transition import (  # noqa: E402
    ActivationRecorder,
    EXPECTED_RUN_C_CHECKPOINT_SHA256,
    EXPECTED_RUN_C_EPOCH,
    EXPECTED_RUN_C_SEED,
    EXPECTED_VAL_SELECTION_CLASS_IDS,
    EXPECTED_VAL_TILES,
    LogitStitcher,
    iter_crop_batches,
    sliding_window_coordinates,
    validate_args,
)
from scripts.earthmiss_scale_transition_common import (  # noqa: E402
    CROSS_SCALE_STAGE_ORDER,
    ScaleTransitionAccumulator,
    bootstrap_city_cluster_mean_ci,
    bootstrap_mean_ci,
    centered_linear_cka,
    downsample_semantic_regions,
    feature_pair_batch_statistics,
    native_semantic_region_area_weights,
    orthogonal_haar_energies,
    pearson_correlation,
    segmentation_region_statistics,
    semantic_boundary_mask,
    strict_json_text,
    summarize_amplification,
    summarize_cross_scale_alignment_degradation,
    summarize_error_correlations,
    write_json_exclusive,
)
from scripts.spatial_diagnostics_common import (  # noqa: E402
    semantic_boundary_mask as numpy_semantic_boundary_mask,
)


class TinyReleasedPath(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = SampleAdapter(
            in_channels=4,
            out_channels=[16, 16, 16, 16],
            num_modalities=2,
        )
        self.decoder = Decoder(
            n_classes=3,
            in_channels=[16, 16, 16, 16],
            out_channels=16,
            num_modalities=2,
            raw_logits=True,
        )

    def decode(self, *features, active_indices):
        slots = self.adapter(
            *features,
            patch_h=4,
            patch_w=4,
            guidance=None,
            modality_indices=active_indices,
        )
        logits = self.decoder(*slots)
        return F.interpolate(logits, size=(64, 64), mode="bilinear")


def _features(value: float) -> tuple[torch.Tensor, ...]:
    base = torch.arange(16 * 4, dtype=torch.float32).reshape(1, 16, 4)
    return tuple(base / 100.0 + value + index for index in range(4))


class FeatureMetricTest(unittest.TestCase):
    def test_identical_and_scaled_features_have_unit_cka_and_cosine(self):
        target = torch.zeros(1, 8, 8, dtype=torch.int64)
        left = torch.randn(1, 4, 8, 8)

        identical = feature_pair_batch_statistics(left, left, target)
        valid = identical.regions["valid"]
        self.assertEqual(valid["cosine_distance_sum"], 0.0)
        self.assertEqual(valid["diff_squared_sum"], 0.0)
        self.assertAlmostEqual(valid["subsampled_cka"], 1.0)

        scaled = feature_pair_batch_statistics(left, 3.0 * left, target)
        self.assertAlmostEqual(
            scaled.regions["valid"]["cosine_distance_sum"],
            0.0,
            places=5,
        )
        self.assertAlmostEqual(scaled.regions["valid"]["subsampled_cka"], 1.0)

    def test_centered_cka_matches_independent_formula(self):
        left = torch.tensor(
            [[1.0, 2.0], [2.0, -1.0], [4.0, 3.0], [-2.0, 1.0]]
        )
        right = torch.tensor(
            [[0.0, 1.0], [3.0, 2.0], [2.0, -2.0], [-1.0, 4.0]]
        )
        left_centered = left.double() - left.double().mean(dim=0, keepdim=True)
        right_centered = right.double() - right.double().mean(dim=0, keepdim=True)
        cross = left_centered.T @ right_centered
        expected = cross.square().sum() / torch.sqrt(
            (left_centered.T @ left_centered).square().sum()
            * (right_centered.T @ right_centered).square().sum()
        )
        actual, points, channels = centered_linear_cka(left, right)
        self.assertEqual((points, channels), (4, 2))
        self.assertAlmostEqual(actual, float(expected), places=12)

    def test_cka_is_none_for_constant_features(self):
        value, points, channels = centered_linear_cka(
            torch.ones(4, 3), torch.ones(4, 3)
        )
        self.assertIsNone(value)
        self.assertEqual((points, channels), (4, 3))

    def test_boundary_only_difference_does_not_leak_into_interior(self):
        target = torch.zeros(1, 8, 8, dtype=torch.int64)
        target[:, :, 4:] = 1
        left = torch.zeros(1, 2, 8, 8)
        left[:, 0] = 1.0
        right = left.clone()
        boundary = semantic_boundary_mask(target)
        right[:, 0][boundary] = -1.0

        result = feature_pair_batch_statistics(left, right, target)
        self.assertGreater(result.regions["boundary"]["cosine_distance_sum"], 0)
        self.assertEqual(result.regions["interior"]["cosine_distance_sum"], 0.0)
        self.assertEqual(result.regions["interior"]["diff_squared_sum"], 0.0)

    def test_shape_and_finite_contracts(self):
        target = torch.zeros(1, 4, 4, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            feature_pair_batch_statistics(
                torch.zeros(1, 2, 4, 4),
                torch.zeros(1, 2, 2, 2),
                target,
            )
        invalid = torch.zeros(1, 2, 4, 4)
        invalid[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            feature_pair_batch_statistics(invalid, invalid, target)

        with self.assertRaisesRegex(ValueError, "exactly one crop window"):
            feature_pair_batch_statistics(
                torch.zeros(2, 2, 4, 4),
                torch.zeros(2, 2, 4, 4),
                torch.zeros(2, 4, 4, dtype=torch.int64),
            )


class BoundaryAndHaarTest(unittest.TestCase):
    def test_torch_boundary_matches_frozen_numpy_definition(self):
        target = np.array(
            [
                [0, 0, 1, 1],
                [0, 8, 1, 2],
                [3, 3, 2, 2],
            ],
            dtype=np.int64,
        )
        expected = numpy_semantic_boundary_mask(target, 8)
        actual = semantic_boundary_mask(torch.from_numpy(target))[0].numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_downsample_preserves_boundary_and_excludes_ignore_cells(self):
        target = torch.zeros(1, 8, 8, dtype=torch.int64)
        target[:, :, 4:] = 1
        target[:, :2, :2] = 8
        labels, masks = downsample_semantic_regions(target, (4, 4))
        self.assertEqual(tuple(labels.shape), (1, 4, 4))
        self.assertFalse(bool(masks["valid"][0, 0, 0]))
        self.assertTrue(bool(masks["boundary"].any()))
        self.assertFalse(bool((masks["boundary"] & masks["interior"]).any()))

    def test_native_support_area_weights_are_invariant_across_scales(self):
        target = torch.zeros(1, 8, 8, dtype=torch.int64)
        target[:, :, 4:] = 1
        target[:, :2, :2] = 8
        target[:, 6:, 6:] = 2
        native_boundary = semantic_boundary_mask(target)
        native_valid = (target >= 0) & (target < 8)
        expected_region_weights = {
            "valid": float(native_valid.sum()),
            "boundary": float((native_boundary & native_valid).sum()),
            "interior": float((native_valid & ~native_boundary).sum()),
        }
        expected_class_weights = torch.tensor(
            [
                float(((target == class_id) & native_valid & ~native_boundary).sum())
                for class_id in range(8)
            ],
            dtype=torch.float64,
        )

        for output_size in ((8, 8), (4, 4), (2, 2), (1, 1)):
            regions, classes = native_semantic_region_area_weights(
                target, output_size
            )
            for name, expected in expected_region_weights.items():
                self.assertEqual(float(regions[name].sum()), expected)
            torch.testing.assert_close(
                classes.sum(dim=(0, 2, 3)),
                expected_class_weights,
                rtol=0.0,
                atol=0.0,
            )

        fine = feature_pair_batch_statistics(
            torch.ones(1, 2, 8, 8),
            torch.ones(1, 2, 8, 8),
            target,
        )
        coarse = feature_pair_batch_statistics(
            torch.ones(1, 2, 4, 4),
            torch.ones(1, 2, 4, 4),
            target,
        )
        for name in ("valid", "boundary", "interior"):
            self.assertEqual(
                fine.regions[name]["native_pixel_weight"],
                coarse.regions[name]["native_pixel_weight"],
            )

        with self.assertRaisesRegex(ValueError, "exactly divide"):
            native_semantic_region_area_weights(target, (3, 4))

    def test_haar_constant_impulse_checkerboard_and_zero(self):
        constant = orthogonal_haar_energies(torch.ones(1, 1, 4, 4))
        self.assertEqual(constant["ll"], 16.0)
        self.assertEqual(sum(constant[band] for band in ("lh", "hl", "hh")), 0.0)

        impulse = torch.zeros(1, 1, 2, 2)
        impulse[0, 0, 0, 0] = 1.0
        impulse_energy = orthogonal_haar_energies(impulse)
        for value in impulse_energy.values():
            self.assertAlmostEqual(value, 0.25)
        self.assertAlmostEqual(sum(impulse_energy.values()), 1.0)

        checkerboard = torch.tensor([[[[1.0, -1.0], [-1.0, 1.0]]]])
        checkerboard_energy = orthogonal_haar_energies(checkerboard)
        self.assertEqual(checkerboard_energy["hh"], 4.0)
        self.assertEqual(
            sum(
                checkerboard_energy[band]
                for band in ("ll", "lh", "hl")
            ),
            0.0,
        )

        zero = orthogonal_haar_energies(torch.zeros(1, 1, 4, 4))
        self.assertEqual(sum(zero.values()), 0.0)
        with self.assertRaisesRegex(ValueError, "must be even"):
            orthogonal_haar_energies(torch.zeros(1, 1, 3, 4))


class SlidingAndRecorderTest(unittest.TestCase):
    def test_coordinates_and_crop_batches_match_release_order(self):
        coordinates = sliding_window_coordinates(
            700, 800, window_size=512, stride=341
        )
        self.assertEqual(
            coordinates,
            (
                (0, 512, 0, 512),
                (0, 512, 288, 800),
                (188, 700, 0, 512),
                (188, 700, 288, 800),
            ),
        )
        rgb = torch.arange(700 * 800).reshape(1, 1, 700, 800)
        sar = rgb + 10
        target = torch.zeros(1, 700, 800, dtype=torch.int64)
        batches = list(
            iter_crop_batches(
                rgb, sar, target, coordinates, batch_size=3
            )
        )
        self.assertEqual([batch[1].shape[0] for batch in batches], [3, 1])
        self.assertEqual(tuple(batches[0][1].shape[-2:]), (512, 512))

    def test_logit_stitcher_averages_overlap(self):
        coordinates = ((0, 4, 0, 4), (0, 4, 2, 6))
        stitcher = LogitStitcher(4, 6, 2)
        full = torch.stack((torch.ones(2, 4, 4), 3 * torch.ones(2, 4, 4)))
        sar = 2 * full
        stitcher.add(coordinates, full, sar)
        result = stitcher.finalize()
        self.assertTrue(torch.equal(result["full"][0, :, :, :2], torch.ones(2, 4, 2)))
        self.assertTrue(torch.equal(result["full"][0, :, :, 2:4], 2 * torch.ones(2, 4, 2)))
        self.assertTrue(torch.equal(result["full"][0, :, :, 4:], 3 * torch.ones(2, 4, 2)))
        self.assertTrue(torch.equal(result["sar"], 2 * result["full"]))

    def test_recorder_is_transparent_and_enforces_canonical_call_contract(self):
        torch.manual_seed(11)
        model = TinyReleasedPath().eval()
        rgb_features = _features(0.5)
        sar_features = _features(1.5)
        expected_full = model.decode(
            rgb_features, sar_features, active_indices=(0, 1)
        )
        expected_sar = model.decode(sar_features, active_indices=(1,))

        with ActivationRecorder(model) as recorder:
            actual_full, full_stages, full_audit = recorder.run(
                "full",
                (0, 1),
                lambda: model.decode(
                    rgb_features, sar_features, active_indices=(0, 1)
                ),
            )
            actual_sar, sar_stages, sar_audit = recorder.run(
                "sar",
                (1,),
                lambda: model.decode(sar_features, active_indices=(1,)),
            )
        torch.testing.assert_close(actual_full, expected_full, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_sar, expected_sar, rtol=0.0, atol=0.0)
        self.assertEqual(set(full_stages), set(sar_stages))
        self.assertEqual(full_audit["call_counts"]["project.P2"], 2)
        self.assertEqual(sar_audit["call_counts"]["project.P2"], 1)
        self.assertEqual(full_audit["call_counts"]["frm"], 2)
        self.assertEqual(sar_audit["call_counts"]["frm"], 2)

        for stages in (full_stages, sar_stages):
            torch.testing.assert_close(
                stages["adapter.pre_resize.P4"],
                stages["adapter.fused.P4"],
                rtol=0.0,
                atol=0.0,
            )
            transitions = (
                (
                    "decoder.se_fused.P5",
                    "decoder.se_fused.P4",
                    "prn.resized.P5_to_P4",
                    "prn.concat.P4",
                ),
                (
                    "prn.td.P4",
                    "decoder.se_fused.P3",
                    "prn.resized.P4_to_P3",
                    "prn.concat.P3",
                ),
                (
                    "prn.td.P3",
                    "decoder.se_fused.P2",
                    "prn.resized.P3_to_P2",
                    "prn.concat.P2",
                ),
            )
            for coarse_key, lateral_key, resized_key, concat_key in transitions:
                expected_resized = F.interpolate(
                    stages[coarse_key],
                    size=stages[lateral_key].shape[-2:],
                    mode="nearest",
                )
                torch.testing.assert_close(
                    stages[resized_key],
                    expected_resized,
                    rtol=0.0,
                    atol=0.0,
                )
                torch.testing.assert_close(
                    stages[concat_key],
                    torch.cat((expected_resized, stages[lateral_key]), dim=1),
                    rtol=0.0,
                    atol=0.0,
                )

        recorded = full_stages["logits.final"].clone()
        actual_full.add_(10.0)
        self.assertTrue(torch.equal(full_stages["logits.final"], recorded))
        counts_after_exit = copy.deepcopy(recorder.counts)
        _ = model.decode(rgb_features, sar_features, active_indices=(0, 1))
        self.assertEqual(recorder.counts, counts_after_exit)


class AggregationAndOutputTest(unittest.TestCase):
    def test_aggregation_bootstrap_and_amplification_are_deterministic(self):
        target = torch.zeros(1, 8, 8, dtype=torch.int64)
        target[:, :, 4:] = 1
        left = torch.randn(1, 4, 8, 8)
        before = feature_pair_batch_statistics(left, left + 0.1, target)
        after = feature_pair_batch_statistics(left, left + 0.2, target)
        accumulator = ScaleTransitionAccumulator(
            stage_order=("adapter.pre_resize.P4", "adapter.fused.P4")
        )
        for tile_index in range(3):
            accumulator.update(
                "adapter.pre_resize.P4",
                before,
                city="A",
                tile_id=str(tile_index),
            )
            accumulator.update(
                "adapter.fused.P4",
                after,
                city="A",
                tile_id=str(tile_index),
            )
        first = accumulator.summary(bootstrap_resamples=100, bootstrap_seed=7)
        second = accumulator.summary(bootstrap_resamples=100, bootstrap_seed=7)
        self.assertEqual(first, second)
        amplification = summarize_amplification(
            first["by_tile"],
            left_label="full",
            right_label="sar",
            resamples=100,
            seed=7,
        )
        identity = amplification["adapter_identity_P4_control"]
        self.assertGreater(
            identity["metrics"]["valid_relative_rms_delta"]["mean"], 0
        )
        self.assertEqual(identity["analysis_role"], "negative_control")
        self.assertEqual(
            amplification["prn_nearest_P5_to_P4"]["analysis_role"],
            "descriptive_control",
        )
        self.assertNotIn(
            "prn_nearest_P5_to_P4",
            amplification["inference_policy"]["primary_edges"],
        )
        self.assertEqual(
            amplification["inference_policy"]["fam_primary_test"],
            "cross_scale_alignment.prn.cross_scale.P5_to_P4",
        )
        self.assertNotIn("NaN", strict_json_text({"summary": first}))

    def test_cross_scale_alignment_uses_resized_coarse_lateral_pairs(self):
        torch.manual_seed(31)
        target = torch.zeros(1, 8, 8, dtype=torch.int64)
        target[:, :, 4:] = 1
        stage = CROSS_SCALE_STAGE_ORDER[0]
        full_accumulator = ScaleTransitionAccumulator(
            stage_order=CROSS_SCALE_STAGE_ORDER,
            left_label="resized_coarse",
            right_label="same_grid_lateral",
        )
        sar_accumulator = ScaleTransitionAccumulator(
            stage_order=CROSS_SCALE_STAGE_ORDER,
            left_label="resized_coarse",
            right_label="same_grid_lateral",
        )
        for city in ("A", "B", "C"):
            for tile_index in range(2):
                coarse = torch.randn(1, 4, 8, 8)
                full_pair = feature_pair_batch_statistics(coarse, coarse, target)
                sar_pair = feature_pair_batch_statistics(
                    coarse,
                    torch.roll(coarse, shifts=1, dims=-1),
                    target,
                )
                full_accumulator.update(
                    stage, full_pair, city=city, tile_id=str(tile_index)
                )
                sar_accumulator.update(
                    stage, sar_pair, city=city, tile_id=str(tile_index)
                )
        full = full_accumulator.summary(
            bootstrap_resamples=50, bootstrap_seed=5
        )
        sar = sar_accumulator.summary(
            bootstrap_resamples=50, bootstrap_seed=5
        )
        first = summarize_cross_scale_alignment_degradation(
            full["by_tile"], sar["by_tile"], resamples=50, seed=7
        )
        second = summarize_cross_scale_alignment_degradation(
            full["by_tile"], sar["by_tile"], resamples=50, seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(first[stage]["analysis_role"], "primary")
        self.assertEqual(
            first[stage]["pair_within_each_endpoint"],
            {"left": "resized_coarse", "right": "same_grid_lateral"},
        )
        metrics = first[stage]["metrics"]
        self.assertGreater(
            metrics["valid_cosine_distance_sar_minus_full"]["mean"], 0
        )
        self.assertLess(
            metrics[
                "valid_mean_per_window_subsampled_linear_cka_sar_minus_full"
            ]["mean"],
            0,
        )
        self.assertEqual(
            first["inference_policy"]["primary_stage"],
            "prn.cross_scale.P5_to_P4",
        )

    def test_bootstrap_and_pearson_handle_small_or_constant_inputs(self):
        self.assertEqual(
            bootstrap_mean_ci([], resamples=10, seed=1),
            {"n": 0, "mean": None, "ci95": [None, None]},
        )
        self.assertIsNone(pearson_correlation([1, 1, 1], [1, 2, 3])["r"])
        self.assertAlmostEqual(
            pearson_correlation([1, 2, 3], [2, 4, 6])["r"], 1.0
        )
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            pearson_correlation([1, 2, 3], [1, 2])

    def test_city_cluster_bootstrap_and_correlations_are_deterministic(self):
        values_by_city = {
            "A": [1.0, 2.0],
            "B": [3.0, 4.0],
            "C": [5.0, 6.0],
        }
        first_mean = bootstrap_city_cluster_mean_ci(
            values_by_city, resamples=100, seed=19
        )
        second_mean = bootstrap_city_cluster_mean_ci(
            values_by_city, resamples=100, seed=19
        )
        self.assertEqual(first_mean, second_mean)
        self.assertEqual(first_mean["cities"], 3)

        stage = "adapter.pre_resize.P2"
        by_tile = {}
        outcomes = {}
        for city_index, city in enumerate(("A", "B", "C")):
            for tile_index in range(2):
                value = float(city_index * 2 + tile_index + 1) / 10.0
                tile_key = f"{city}/{tile_index}"
                by_tile[tile_key] = {
                    stage: {
                        "regions": {
                            "valid": {"cosine_distance": value},
                            "boundary": {"cosine_distance": value},
                        },
                        "frequency": {
                            "full": {
                                "representation_grid_high_frequency_fraction": 0.1
                            },
                            "sar": {
                                "representation_grid_high_frequency_fraction": (
                                    0.1 + value
                                )
                            },
                        },
                    }
                }
                relative = {
                    "right_minus_left_error_rate": value,
                    "left_correct_right_wrong_rate": value,
                }
                outcomes[tile_key] = {
                    "regions": {
                        "valid": {"relative_degradation": relative},
                        "boundary": {"relative_degradation": relative},
                    }
                }

        first = summarize_error_correlations(
            by_tile, outcomes, resamples=100, seed=23
        )
        second = summarize_error_correlations(
            by_tile, outcomes, resamples=100, seed=23
        )
        self.assertEqual(first, second)
        correlation = first[stage][
            "valid_gap_vs_relative_error_degradation"
        ]
        self.assertAlmostEqual(correlation["raw"]["r"], 1.0)
        self.assertAlmostEqual(correlation["city_demeaned"]["r"], 1.0)
        self.assertEqual(correlation["raw"]["cities"], 3)
        self.assertGreater(correlation["raw"]["valid_bootstrap_resamples"], 0)

    def test_segmentation_regions_and_class_counts(self):
        target = torch.tensor([[[0, 0, 1, 1], [0, 0, 1, 1]]])
        full = torch.zeros(1, 2, 2, 4)
        sar = torch.zeros_like(full)
        full[:, 0] = 1.0
        sar[:, 0] = 1.0
        full[:, 1, :, 2:] = 2.0
        sar[:, 1, :, 3:] = 2.0
        result = segmentation_region_statistics(full, sar, target, num_classes=2)
        self.assertEqual(result["regions"]["valid"]["pixels"], 8)
        self.assertEqual(result["regions"]["valid"]["full"]["error_pixels"], 0)
        self.assertEqual(result["regions"]["valid"]["sar"]["error_pixels"], 2)
        relative = result["regions"]["valid"]["relative_degradation"]
        self.assertEqual(relative["right_minus_left_error_rate"], 0.25)
        self.assertEqual(relative["left_correct_right_wrong_pixels"], 2)
        self.assertEqual(relative["left_correct_right_wrong_rate"], 0.25)
        self.assertEqual(relative["right_correct_left_wrong_pixels"], 0)
        self.assertEqual(relative["right_correct_left_wrong_rate"], 0.0)
        self.assertEqual(result["by_class"][1]["pixels"], 4)

    def test_json_writer_is_byte_stable_and_refuses_overwrite(self):
        payload = {"z": None, "a": {"value": 1.25}}
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_json_exclusive(first, payload)
            write_json_exclusive(second, payload)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_json_exclusive(first, payload)

    def test_json_writer_does_not_overwrite_a_concurrent_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            barrier = threading.Barrier(2)

            def attempt(writer_id: int):
                barrier.wait()
                try:
                    write_json_exclusive(output, {"writer": writer_id})
                except FileExistsError:
                    return "exists", writer_id
                return "written", writer_id

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(attempt, (1, 2)))
            winners = [writer_id for status, writer_id in results if status == "written"]
            losers = [writer_id for status, writer_id in results if status == "exists"]
            self.assertEqual(len(winners), 1)
            self.assertEqual(len(losers), 1)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"writer": winners[0]},
            )

    def test_formal_manifest_and_checkpoint_constants_are_frozen(self):
        self.assertEqual(EXPECTED_VAL_TILES, 277)
        self.assertEqual(EXPECTED_VAL_SELECTION_CLASS_IDS, list(range(7)))
        self.assertEqual(EXPECTED_RUN_C_EPOCH, 15)
        self.assertEqual(EXPECTED_RUN_C_SEED, 42)
        self.assertEqual(
            EXPECTED_RUN_C_CHECKPOINT_SHA256,
            "b038dfcfe771ca5c67da500acc2b88e74f066496cac332fc3b76d4377b73dff9",
        )

    def test_runtime_argument_contract(self):
        base = dict(
            window_size=512,
            stride=341,
            feature_batch_size=1,
            num_workers=0,
            smoke_tiles=0,
            cka_max_points=128,
            cka_max_channels=128,
            bootstrap_resamples=10,
        )
        validate_args(SimpleNamespace(**base))
        with self.assertRaisesRegex(ValueError, "multiple of 64"):
            validate_args(SimpleNamespace(**{**base, "window_size": 500}))
        with self.assertRaisesRegex(ValueError, "multiple of 64"):
            validate_args(SimpleNamespace(**{**base, "window_size": 96}))
        with self.assertRaisesRegex(ValueError, "stride"):
            validate_args(SimpleNamespace(**{**base, "stride": 513}))
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            validate_args(SimpleNamespace(**{**base, "feature_batch_size": 0}))
        with self.assertRaisesRegex(ValueError, "exactly 1"):
            validate_args(SimpleNamespace(**{**base, "feature_batch_size": 2}))


if __name__ == "__main__":
    unittest.main()
