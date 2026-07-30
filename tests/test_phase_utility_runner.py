import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import scripts.evaluate_whu_phase_utility_audit as phase_utility_runner
from scripts.evaluate_whu_phase_utility_audit import (
    DECISION_Q,
    Q_VALUES,
    _finite_or_none,
    _reference_checks,
    arbitrate_stage_a_routes,
    atomic_write_json,
    cell_means_from_common_map,
    finite_oracle_envelope,
    fixed_policy_stability,
    grouped_singleton_miou_scores,
    grouped_top_q_indices,
    load_four_phase_reference,
    matched_action_random_control,
    k2_axis_diagnostic,
    restrict_score_to_geometry,
    route_decision_from_point,
    slide_window_manifest,
    split_spatial_region_metadata,
    stage_a_decision,
)
from scripts.phase_utility_common import (
    CellScoreVector,
    aggregate_cell_assignment,
    cell_phase_statistics,
    oracle_cell_miou_gain_scores,
)
from scripts.spatial_diagnostics_common import baseline_summary


class PhaseUtilityRunnerTests(unittest.TestCase):
    def test_spatial_region_metadata_splits_dynamic_anchor_count(self):
        first = {
            "boundary": {"anchor_pixels": 12, "definition": "fixed"},
            "components": {"connectivity": 8},
            "mixed_patch": {"patch_size": 16},
            "actionable_union_sources": ["boundary_le_0px"],
        }
        second = {
            **first,
            "boundary": {"anchor_pixels": 37, "definition": "fixed"},
        }
        first_definitions, first_image = split_spatial_region_metadata(first)
        second_definitions, second_image = split_spatial_region_metadata(second)
        self.assertEqual(first_definitions, second_definitions)
        self.assertEqual(first_image, {"boundary_anchor_pixels": 12})
        self.assertEqual(second_image, {"boundary_anchor_pixels": 37})
        self.assertEqual(first["boundary"]["anchor_pixels"], 12)
        self.assertNotIn("anchor_pixels", first_definitions["boundary"])

    def test_actual_region_definitions_are_invariant_across_targets(self):
        uniform = np.zeros((8, 8), dtype=np.int64)
        divided = uniform.copy()
        divided[:, 4:] = 1
        _, uniform_metadata = phase_utility_runner.build_spatial_region_masks(
            uniform, 7
        )
        _, divided_metadata = phase_utility_runner.build_spatial_region_masks(
            divided, 7
        )
        uniform_definitions, uniform_image = split_spatial_region_metadata(
            uniform_metadata
        )
        divided_definitions, divided_image = split_spatial_region_metadata(
            divided_metadata
        )
        self.assertEqual(uniform_definitions, divided_definitions)
        self.assertNotEqual(
            uniform_image["boundary_anchor_pixels"],
            divided_image["boundary_anchor_pixels"],
        )

    def test_nonfinite_empty_cell_score_serializes_as_null(self):
        self.assertIsNone(_finite_or_none(-np.inf))
        self.assertEqual(_finite_or_none(1.25), 1.25)

    def test_geometry_restriction_is_shared_by_oracle_and_deployment_scores(self):
        eligible = np.array([False, True, True], dtype=bool)
        oracle = restrict_score_to_geometry(
            CellScoreVector("oracle", np.array([9.0, 1.0, -1.0]), True),
            eligible,
        )
        deployment = restrict_score_to_geometry(
            CellScoreVector("entropy", np.array([3.0, 2.0, 1.0]), False),
            eligible,
        )
        self.assertTrue(np.isneginf(oracle.values[0]))
        self.assertTrue(np.isneginf(deployment.values[0]))
        self.assertTrue(oracle.uses_ground_truth)
        self.assertFalse(deployment.uses_ground_truth)

    def test_atomic_json_is_strict_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_write_json(path, {"value": 1.0})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1.0})
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                atomic_write_json(path, {"value": 2.0})
            invalid = Path(directory) / "invalid.json"
            with self.assertRaises(ValueError):
                atomic_write_json(invalid, {"value": -np.inf})
            self.assertFalse(invalid.exists())

    def test_slide_manifest_reproduces_end_backtracking(self):
        cases = {
            512: [0],
            853: [0, 341],
            854: [0, 341, 342],
            1100: [0, 341, 588],
        }
        for length, expected in cases.items():
            with self.subTest(length=length):
                manifest = slide_window_manifest((length, length))
                self.assertEqual(manifest["row_starts"], expected)
                self.assertEqual(manifest["column_starts"], expected)
                self.assertEqual(
                    len(manifest["windows"]), len(expected) * len(expected)
                )
                self.assertEqual(manifest["windows"][0].tolist(), [0, 512, 0, 512])
                self.assertEqual(
                    manifest["windows"][-1].tolist(),
                    [length - 512, length, length - 512, length],
                )

    def test_slide_manifest_rejects_small_sealed_image(self):
        with self.assertRaisesRegex(ValueError, "both image axes"):
            slide_window_manifest((511, 800))

    def test_cell_means_intersect_common_support_without_gt(self):
        cells = np.array(
            [
                [0, 4, 0, 4],
                [0, 4, 4, 8],
                [4, 8, 0, 4],
                [4, 8, 4, 8],
            ],
            dtype=np.int64,
        )
        common = np.arange(16, dtype=np.float32).reshape(4, 4)
        values = cell_means_from_common_map(cells, common, (2, 6, 2, 6))
        self.assertEqual(values[0], np.mean(common[:2, :2]))
        self.assertEqual(values[1], np.mean(common[:2, 2:]))
        self.assertEqual(values[2], np.mean(common[2:, :2]))
        self.assertEqual(values[3], np.mean(common[2:, 2:]))

    def test_cell_means_marks_nonintersecting_cell_negative_infinity(self):
        cells = np.array([[0, 2, 0, 2], [2, 4, 2, 4]], dtype=np.int64)
        values = cell_means_from_common_map(
            cells, np.ones((2, 2), dtype=np.float32), (2, 4, 2, 4)
        )
        self.assertTrue(np.isneginf(values[0]))
        self.assertEqual(values[1], 1.0)

    def test_singleton_oracle_uses_full_reference_confusion(self):
        full = np.zeros((7, 7), dtype=np.int64)
        full[0, 0] = 99
        full[0, 1] = 1
        full[1, 1] = 100
        cell_k1 = np.zeros((7, 7), dtype=np.int64)
        cell_k4 = np.zeros((7, 7), dtype=np.int64)
        cell_k1[0, 1] = 1
        cell_k4[0, 0] = 1
        stats = (
            {
                "cell_index": 0,
                "confusion": {"k1": cell_k1, "k2": cell_k1, "k4": cell_k4},
            },
        )
        score = oracle_cell_miou_gain_scores(
            stats, num_classes=7, reference_confusion=full
        )
        self.assertTrue(score.uses_ground_truth)
        self.assertGreater(score.values[0], 0.0)

    def test_grouped_top_q_applies_exact_budget_inside_each_image(self):
        score = CellScoreVector(
            "synthetic",
            np.array([1.0, 9.0, 8.0, 4.0, 3.0, 2.0, 1.0]),
            False,
        )
        groups = np.array([7, 7, 7, 9, 9, 9, 9], dtype=np.int64)
        selected = grouped_top_q_indices(
            score, 0.5, groups, deployment=True
        )
        np.testing.assert_array_equal(selected, np.array([1, 3, 4]))
        self.assertEqual(np.count_nonzero(groups[selected] == 7), 1)
        self.assertEqual(np.count_nonzero(groups[selected] == 9), 2)

    @staticmethod
    def _two_group_cell_stats():
        bounds = np.array(
            [[0, 1, index, index + 1] for index in range(4)],
            dtype=np.int64,
        )
        target = np.array([[0, 1, 0, 1]], dtype=np.int64)
        k1 = np.array([[1, 0, 1, 0]], dtype=np.int64)
        k4 = target.copy()
        return cell_phase_statistics(bounds, k1, k4, k4, target, 7)

    def test_grouped_singleton_scores_use_each_images_own_reference(self):
        stats = self._two_group_cell_stats()
        groups = np.array([7, 7, 9, 9], dtype=np.int64)
        references = {}
        for group, diagonal in ((7, (100, 1)), (9, (1, 100))):
            indices = np.flatnonzero(groups == group)
            reference = sum(
                (stats[index]["confusion"]["k1"] for index in indices),
                start=np.zeros((7, 7), dtype=np.int64),
            )
            reference[0, 0] += diagonal[0]
            reference[1, 1] += diagonal[1]
            references[group] = reference

        actual = grouped_singleton_miou_scores(stats, groups, references)
        expected = np.empty(4, dtype=np.float64)
        for group in (7, 9):
            indices = np.flatnonzero(groups == group)
            local_stats = []
            for local_index, global_index in enumerate(indices):
                record = dict(stats[int(global_index)])
                record["cell_index"] = local_index
                local_stats.append(record)
            local = oracle_cell_miou_gain_scores(
                local_stats,
                num_classes=7,
                reference_confusion=references[group],
            )
            expected[indices] = local.values

        np.testing.assert_allclose(actual.values, expected)
        self.assertNotAlmostEqual(actual.values[0], actual.values[2])

    def test_matched_random_preserves_group_counts_and_nests_k4(self):
        bounds = np.array(
            [[0, 1, index, index + 1] for index in range(6)],
            dtype=np.int64,
        )
        target = np.array([[0, 1, 0, 1, 0, 1]], dtype=np.int64)
        k1 = np.array([[1, 0, 1, 0, 1, 0]], dtype=np.int64)
        k2 = target.copy()
        k4 = target.copy()
        stats = cell_phase_statistics(bounds, k1, k2, k4, target, 7)
        full_k1 = sum(
            (record["confusion"]["k1"] for record in stats),
            start=np.zeros((7, 7), dtype=np.int64),
        )
        groups = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        target_levels = np.array([1, 2, 4, 4, 4, 2], dtype=np.int64)
        eligible = np.array([True, False, True, True, True, True])
        generated_levels = []
        real_aggregate = aggregate_cell_assignment
        real_exact_random = phase_utility_runner._exact_count_random_indices

        def recording_aggregate(cell_stats, levels, *, num_classes):
            generated_levels.append(np.asarray(levels, dtype=np.int64).copy())
            return real_aggregate(cell_stats, levels, num_classes=num_classes)

        with patch.object(phase_utility_runner, "RANDOM_REPLICATES", 4), patch.object(
            phase_utility_runner,
            "aggregate_cell_assignment",
            side_effect=recording_aggregate,
        ), patch.object(
            phase_utility_runner,
            "_exact_count_random_indices",
            wraps=real_exact_random,
        ) as exact_random:
            control = matched_action_random_control(
                stats,
                full_k1,
                target_levels,
                eligible,
                groups,
                route_kind="a2_hierarchical_x",
            )

        self.assertEqual(
            control["target_counts_by_group"],
            {
                0: {"k2_only": 1, "k4": 1},
                1: {"k2_only": 1, "k4": 2},
            },
        )
        self.assertEqual(control["replicates"], 4)
        # Only the one validation aggregation remains; replicate evaluation
        # is vectorized from the exact promoted/K4 index sets.
        self.assertEqual(len(generated_levels), 1)
        np.testing.assert_array_equal(generated_levels[0], np.ones(6))

        # Every group/replicate makes two selections. The second selection's
        # pool is exactly the first selection's promoted K2-or-K4 set, which
        # is the implementation-level guarantee that K4 is nested under K2.
        exact_calls = exact_random.call_args_list
        self.assertEqual(len(exact_calls), 4 * 2 * 2)
        expected_by_group = [
            control["target_counts_by_group"][0],
            control["target_counts_by_group"][1],
        ]
        for call_index in range(0, len(exact_calls), 2):
            promoted_call = exact_calls[call_index]
            k4_call = exact_calls[call_index + 1]
            expected = expected_by_group[(call_index // 2) % 2]
            self.assertEqual(
                promoted_call.args[1], expected["k2_only"] + expected["k4"]
            )
            self.assertEqual(k4_call.args[0], promoted_call.args[1])
            self.assertEqual(k4_call.args[1], expected["k4"])

    def test_matched_random_keeps_explicit_a2_provenance_without_k2_only_cells(self):
        stats = self._two_group_cell_stats()
        full_k1 = sum(
            (record["confusion"]["k1"] for record in stats),
            start=np.zeros((7, 7), dtype=np.int64),
        )
        with patch.object(phase_utility_runner, "RANDOM_REPLICATES", 2):
            control = matched_action_random_control(
                stats,
                full_k1,
                np.ones(len(stats), dtype=np.int64),
                np.ones(len(stats), dtype=bool),
                np.zeros(len(stats), dtype=np.int64),
                route_kind="a2_hierarchical_x",
            )
        self.assertEqual(control["route_kind"], "a2_hierarchical_x")
        self.assertEqual(control["cost"]["scheme"], "hierarchical_k1_k2_k4")

    def test_matched_random_fails_closed_if_geometry_pool_is_too_small(self):
        stats = self._two_group_cell_stats()
        full_k1 = sum(
            (record["confusion"]["k1"] for record in stats),
            start=np.zeros((7, 7), dtype=np.int64),
        )
        with patch.object(phase_utility_runner, "RANDOM_REPLICATES", 1):
            with self.assertRaisesRegex(ValueError, "geometry-eligible pool"):
                matched_action_random_control(
                    stats,
                    full_k1,
                    np.array([4, 1, 1, 1], dtype=np.int64),
                    np.zeros(len(stats), dtype=bool),
                    np.zeros(len(stats), dtype=np.int64),
                    route_kind="a1_binary",
                )

    def test_finite_oracle_envelope_uses_higher_observed_miou(self):
        net = []
        singleton = []
        for index, q in enumerate(Q_VALUES):
            net.append({"requested_q": q, "full_image": {"miou": 0.5 + index / 100}})
            singleton.append(
                {"requested_q": q, "full_image": {"miou": 0.55 - index / 100}}
            )
        envelope = finite_oracle_envelope(
            {"net_correct": net, "singleton_global_miou": singleton}
        )
        self.assertEqual(envelope[0]["source_ranking"], "singleton_global_miou")
        self.assertEqual(envelope[-1]["source_ranking"], "net_correct")

    @staticmethod
    def _decision_fixture(oracle_miou=0.58, random_p95=0.57):
        endpoint = lambda miou, small, thin: {  # noqa: E731
            "full_image": {"miou": miou},
            "common_support": {
                "regions": {
                    "small": {"error_rate": small},
                    "thin": {"error_rate": thin},
                }
            },
        }
        endpoints = {
            "k1": endpoint(0.50, 0.25, 0.35),
            "matched_k2": endpoint(0.55, 0.20, 0.30),
            "k4": endpoint(0.60, 0.19, 0.29),
        }
        oracle = [
            {
                "requested_q": DECISION_Q,
                "source_ranking": "net_correct",
                "cost": {"forward_equivalent_cost": 2.0},
                "full_image": {"miou": oracle_miou},
                "common_support": {
                    "regions": {
                        "small": {"error_rate": 0.19},
                        "thin": {"error_rate": 0.29},
                    }
                },
            }
        ]
        random = [
            {"requested_q": DECISION_Q, "miou": {"p95": random_p95}}
        ]
        return endpoints, oracle, random

    def test_stage_a_decision_requires_all_fixed_gates(self):
        endpoints, oracle, random = self._decision_fixture()
        decision = stage_a_decision(endpoints, oracle, random, formal=True)
        self.assertEqual(decision["outcome"], "PASS_FEASIBILITY")
        self.assertTrue(decision["checks"]["outperforms_equal_budget_random_p95"])

        endpoints, oracle, random = self._decision_fixture(random_p95=0.58)
        decision = stage_a_decision(endpoints, oracle, random, formal=True)
        self.assertEqual(decision["outcome"], "FAIL_FEASIBILITY")
        self.assertFalse(decision["passed"])

    def test_route_decision_reports_route_and_scope_before_arbitration(self):
        endpoints, oracle, random = self._decision_fixture()
        decision = route_decision_from_point(
            endpoints,
            oracle[0],
            random[0],
            formal=True,
            route_name="A2_hierarchical_k1_k2x_k4",
            budget_scope="per-image",
        )
        self.assertEqual(decision["outcome"], "PASS_FEASIBILITY")
        self.assertTrue(decision["passed"])
        self.assertEqual(decision["route_name"], "A2_hierarchical_k1_k2x_k4")
        self.assertEqual(decision["budget_scope"], "per-image")

    def test_stage_a_arbitration_is_a2_first_and_requires_both_scopes(self):
        def decisions(a1_global, a1_per_image, a2_global, a2_per_image):
            return {
                "a1_binary": {
                    "global": {"passed": a1_global},
                    "per_image": {"passed": a1_per_image},
                },
                "a2_hierarchical_x": {
                    "global": {"passed": a2_global},
                    "per_image": {"passed": a2_per_image},
                },
            }

        both_pass = arbitrate_stage_a_routes(
            decisions(True, True, True, True), formal=True
        )
        self.assertEqual(both_pass["selected_route"], "a2_hierarchical_x")
        self.assertEqual(both_pass["outcome"], "GO_STAGE_B_A2_HIERARCHICAL_X")

        fallback = arbitrate_stage_a_routes(
            decisions(True, True, False, False), formal=True
        )
        self.assertEqual(fallback["selected_route"], "a1_binary")
        self.assertEqual(fallback["outcome"], "GO_STAGE_B_A1_BINARY")

        pooled_only = arbitrate_stage_a_routes(
            decisions(True, False, False, False), formal=True
        )
        self.assertEqual(
            pooled_only["outcome"], "POOLED_ONLY_NO_WINDOW_STAGE_B"
        )
        self.assertFalse(pooled_only["stage_b_authorized"])

        stopped = arbitrate_stage_a_routes(
            decisions(False, True, False, False), formal=True
        )
        self.assertEqual(
            stopped["outcome"], "NO_GO_STOP_CURRENT_PHASE_ON_DEMAND_ROUTE"
        )
        self.assertFalse(stopped["stage_b_authorized"])

    def test_fixed_policy_stability_is_descriptive_and_uses_fixed_loo_policy(self):
        stats = self._two_group_cell_stats()
        groups = np.array([7, 7, 9, 9], dtype=np.int64)
        levels = np.array([1, 4, 1, 4], dtype=np.int64)
        full_by_group = {}
        for group in (7, 9):
            indices = np.flatnonzero(groups == group)
            full_by_group[group] = {
                level: sum(
                    (stats[index]["confusion"][level] for index in indices),
                    start=np.zeros((7, 7), dtype=np.int64),
                )
                for level in ("k1", "k2", "k4")
            }
        result = fixed_policy_stability(stats, levels, groups, full_by_group)
        self.assertIn("descriptive stability only", result["role"])
        self.assertEqual(result["summary"]["image_count"], 2)
        self.assertEqual(len(result["per_image"]), 2)
        self.assertEqual(len(result["leave_one_image_out_fixed_policy"]), 2)
        self.assertTrue(
            all(
                row["fixed_policy_not_reoptimized"]
                for row in result["leave_one_image_out_fixed_policy"]
            )
        )

    def test_k2y_axis_diagnostic_never_selects_an_axis(self):
        x = np.zeros((7, 7), dtype=np.int64)
        y = np.zeros((7, 7), dtype=np.int64)
        x[0, 0], x[0, 1] = 9, 1
        y[0, 0], y[0, 1] = 10, 0
        result = k2_axis_diagnostic(
            [
                {
                    "loader_position": 0,
                    "sample_name": "synthetic",
                    "confusion": {
                        "matched_k2": x.tolist(),
                        "matched_k2_y": y.tolist(),
                    },
                }
            ]
        )
        self.assertIn("never enters A2", result["role"])
        self.assertEqual(result["summary"]["k2y_better_image_count"], 1)
        self.assertGreater(result["per_image"][0]["k2y_minus_k2x_pp"], 0.0)

    def test_subset_never_makes_scientific_decision(self):
        decision = stage_a_decision({}, [], [], formal=False)
        self.assertEqual(decision["outcome"], "NOT_EVALUATED_SUBSET")
        self.assertIsNone(decision["passed"])

    def test_load_four_phase_reference_checks_sealed_protocol(self):
        payload = {
            "status": "PASS",
            "scope": "full-test",
            "evaluated_images": 20,
            "full_test_length": 20,
            "protocol": {
                "primary": [[0, 0], [0, 8], [8, 0], [8, 8]],
                "valid_margin": 512,
                "same_common_valid_region_for_primary_and_control": True,
            },
            "baseline_validation": {
                "checked": True,
                "prediction_sha256_equal": True,
                "label_sha256_equal": True,
                "confusion_equal": True,
                "miou_within_tolerance": True,
            },
            "two_phase_validation": {
                "checked": True,
                "candidate_prediction_sha256_equal": True,
                "confusion_equal": True,
                "miou_within_tolerance": True,
            },
            "aggregate": {
                "8": {
                    "candidate": {"miou": 0.5, "confusion": []},
                    "candidate_prediction_sha256": "abc",
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(load_four_phase_reference(path)["status"], "PASS")
            payload["protocol"]["valid_margin"] = 511
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid margin"):
                load_four_phase_reference(path)

    def test_full_reference_checks_cover_all_three_sealed_endpoints(self):
        labels = [f"class_{index}" for index in range(7)]
        k1 = np.zeros((7, 7), dtype=np.int64)
        k2 = np.zeros((7, 7), dtype=np.int64)
        k4 = np.zeros((7, 7), dtype=np.int64)
        k1[0, 0], k2[0, 0], k4[0, 0] = 10, 11, 12
        k1_metrics = baseline_summary(k1, labels)
        k2_metrics = baseline_summary(k2, labels)
        k4_metrics = baseline_summary(k4, labels)
        validation = _reference_checks(
            full_test=True,
            spatial_reference={
                "prediction_sha256": "k1",
                "label_sha256": "label",
                "aggregate": {"baseline": k1_metrics},
            },
            two_phase_reference={
                "aggregate": {
                    "8": {
                        "candidate_prediction_sha256": "k2",
                        "candidate": k2_metrics,
                    }
                }
            },
            four_phase_reference={
                "aggregate": {
                    "8": {
                        "candidate_prediction_sha256": "k4",
                        "candidate": k4_metrics,
                    }
                }
            },
            baseline_digest="k1",
            label_digest="label",
            legacy_k2_digest="k2",
            k4_digest="k4",
            k1_confusion=k1,
            legacy_k2_confusion=k2,
            k4_confusion=k4,
            class_names=labels,
            tolerance=0.0,
        )
        self.assertTrue(validation["checked"])
        self.assertTrue(all(value for key, value in validation.items() if key != "checked"))


if __name__ == "__main__":
    unittest.main()
