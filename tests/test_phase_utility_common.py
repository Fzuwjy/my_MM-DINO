"""Synthetic tests for the zero-training phase-utility statistics layer."""

import unittest

import numpy as np

from scripts.phase_utility_common import (
    CellScoreVector,
    PhaseUtilityGateThresholds,
    aggregate_cell_assignment,
    binary_greedy_oracle,
    binary_group_budget_oracle,
    binary_phase_cost,
    binary_selection_curve,
    build_cell_ownership,
    cell_phase_statistics,
    confusion_from_arrays,
    deployment_score_vector,
    deterministic_random_indices,
    deterministic_top_q_indices,
    evaluate_phase_utility_gate,
    gain_retention,
    hierarchical_greedy_oracle,
    hierarchical_group_budget_oracle,
    hierarchical_phase_cost,
    mean_iou_from_confusion,
    oracle_cell_miou_gain_scores,
    oracle_net_correct_scores,
    phase_transition_counts,
    reconstruct_mixed_prediction,
    validate_cell_partition,
)


def _horizontal_windows(length, starts, crop=512):
    return np.array(
        [[0, 1, start, start + crop] for start in starts], dtype=np.int64
    )


class CellPartitionTest(unittest.TestCase):
    def test_midpoint_bounds_cover_end_aligned_854_axis(self):
        windows = _horizontal_windows(854, (0, 341, 342))
        bounds = build_cell_ownership((1, 854), windows)
        np.testing.assert_array_equal(
            bounds,
            np.array(
                [[0, 1, 0, 426], [0, 1, 426, 597], [0, 1, 597, 854]],
                dtype=np.int64,
            ),
        )
        areas = validate_cell_partition(bounds, windows, (1, 854))
        self.assertEqual(int(areas.sum()), 854)

    def test_midpoint_bounds_cover_end_aligned_1100_axis(self):
        windows = _horizontal_windows(1100, (0, 341, 588))
        bounds = build_cell_ownership((1, 1100), windows)
        np.testing.assert_array_equal(
            bounds[:, 2:],
            np.array([[0, 426], [426, 720], [720, 1100]], dtype=np.int64),
        )
        for bound, window in zip(bounds, windows, strict=True):
            self.assertGreaterEqual(bound[2], window[2])
            self.assertLessEqual(bound[3], window[3])

    def test_cartesian_cells_follow_input_window_order(self):
        windows = np.array(
            [
                [2, 4, 2, 4],
                [0, 2, 0, 2],
                [2, 4, 0, 2],
                [0, 2, 2, 4],
            ],
            dtype=np.int64,
        )
        bounds = build_cell_ownership((4, 4), windows)
        np.testing.assert_array_equal(bounds, windows)

    def test_partition_rejects_gap_and_cell_outside_source_window(self):
        windows = np.array(
            [[0, 1, 0, 3], [0, 1, 1, 4]], dtype=np.int64
        )
        with self.assertRaisesRegex(ValueError, "end-aligned"):
            build_cell_ownership((1, 4), windows[:1])
        invalid = np.array([[0, 1, 1, 4], [0, 1, 0, 1]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "source window"):
            validate_cell_partition(invalid, windows, (1, 4))


class MetricAndCellStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.bounds = np.array(
            [
                [0, 1, 0, 2],
                [1, 2, 0, 1],
                [1, 2, 1, 2],
            ],
            dtype=np.int64,
        )
        self.target = np.array([[0, 1], [1, 0]], dtype=np.int64)
        self.k1 = np.array([[1, 1], [0, 0]], dtype=np.int64)
        self.k2 = np.array([[0, 0], [1, 1]], dtype=np.int64)
        self.k4 = np.array([[0, 1], [0, 0]], dtype=np.int64)
        self.valid = np.array([[True, True], [True, False]])
        self.small = np.array([[True, False], [True, False]])
        self.thin = np.array([[False, True], [False, False]])
        self.stats = cell_phase_statistics(
            self.bounds,
            self.k1,
            self.k2,
            self.k4,
            self.target,
            2,
            valid_mask=self.valid,
            small_mask=self.small,
            thin_mask=self.thin,
        )

    def test_confusion_ignores_invalid_target_and_miou_uses_supported_classes(self):
        target = np.array([[0, 1], [255, 1]], dtype=np.int64)
        prediction = np.array([[0, 0], [99, 1]], dtype=np.int64)
        confusion = confusion_from_arrays(prediction, target, 3)
        np.testing.assert_array_equal(
            confusion,
            np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=np.int64),
        )
        # Class 2 has union zero and is excluded, rather than contributing 0.
        self.assertAlmostEqual(mean_iou_from_confusion(confusion), 0.5)
        self.assertEqual(mean_iou_from_confusion(np.diag([3, 0, 0])), 1.0)

    def test_transition_counts_distinguish_fixes_and_breaks(self):
        target = np.array([[0, 1, 0, 1]])
        reference = np.array([[1, 1, 0, 0]])
        candidate = np.array([[0, 0, 1, 0]])
        counts = phase_transition_counts(reference, candidate, target, 2)
        self.assertEqual(counts["fixed"], 1)
        self.assertEqual(counts["broken"], 2)
        self.assertEqual(counts["net_correct"], -1)
        self.assertEqual(counts["changed"], 3)

    def test_cell_statistics_keep_zero_evaluation_pixel_outer_cell(self):
        self.assertEqual(len(self.stats), 3)
        self.assertEqual(self.stats[0]["pixels"], 2)
        self.assertEqual(self.stats[1]["pixels"], 1)
        self.assertEqual(self.stats[2]["pixels"], 0)
        np.testing.assert_array_equal(
            self.stats[2]["confusion"]["k4"], np.zeros((2, 2), dtype=np.int64)
        )
        transition = self.stats[0]["transitions"]["k1_to_k4"]
        self.assertEqual(transition["all"]["fixed"], 1)
        self.assertEqual(transition["all"]["broken"], 0)
        self.assertEqual(transition["small"]["fixed"], 1)
        self.assertEqual(transition["thin"]["fixed"], 0)

    def test_direct_aggregation_matches_whole_image_reconstruction(self):
        levels = np.array([1, 2, 4], dtype=np.int64)
        aggregate = aggregate_cell_assignment(
            self.stats, levels, num_classes=2
        )
        reconstructed = reconstruct_mixed_prediction(
            self.bounds, self.k1, self.k2, self.k4, levels
        )
        expected_confusion = confusion_from_arrays(
            reconstructed, self.target, 2, mask=self.valid
        )
        np.testing.assert_array_equal(aggregate["confusion"], expected_confusion)
        np.testing.assert_array_equal(
            reconstructed, np.array([[1, 1], [1, 0]], dtype=np.int64)
        )
        self.assertEqual(aggregate["regions"]["all"]["errors"], 1)
        self.assertEqual(aggregate["regions"]["small"]["errors"], 1)
        self.assertEqual(aggregate["regions"]["thin"]["errors"], 0)

    def test_oracle_scores_are_marked_as_ground_truth_dependent(self):
        net = oracle_net_correct_scores(self.stats)
        self.assertTrue(net.uses_ground_truth)
        np.testing.assert_array_equal(net.values, np.array([1.0, 0.0, 0.0]))
        singleton = oracle_cell_miou_gain_scores(
            self.stats, num_classes=2
        )
        self.assertTrue(singleton.uses_ground_truth)
        self.assertTrue(np.all(np.isfinite(singleton.values)))

    def test_singleton_miou_uses_full_reference_confusion(self):
        common_reference = sum(
            (record["confusion"]["k1"] for record in self.stats),
            start=np.zeros((2, 2), dtype=np.int64),
        )
        full_reference = common_reference + np.array([[10, 0], [0, 5]])
        scores = oracle_cell_miou_gain_scores(
            self.stats,
            num_classes=2,
            reference_confusion=full_reference,
        )
        switched = (
            full_reference
            + self.stats[0]["confusion"]["k4"]
            - self.stats[0]["confusion"]["k1"]
        )
        expected = (
            mean_iou_from_confusion(switched)
            - mean_iou_from_confusion(full_reference)
        )
        self.assertAlmostEqual(scores.values[0], expected)
        invalid_reference = common_reference.copy()
        invalid_reference[0, 1] -= 1
        with self.assertRaisesRegex(ValueError, "contain all cell"):
            oracle_cell_miou_gain_scores(
                self.stats,
                num_classes=2,
                reference_confusion=invalid_reference,
            )


class DynamicGreedyOracleTest(unittest.TestCase):
    @staticmethod
    def _binary_dynamic_fixture():
        bounds = np.array(
            [[0, 1, 0, 1], [0, 1, 1, 2]], dtype=np.int64
        )
        target = np.array([[0, 1]], dtype=np.int64)
        k1 = np.array([[1, 0]], dtype=np.int64)
        k2 = np.array([[0, 1]], dtype=np.int64)
        k4 = np.array([[0, 1]], dtype=np.int64)
        stats = cell_phase_statistics(bounds, k1, k2, k4, target, 2)
        full_k1 = np.array([[8, 2], [2, 8]], dtype=np.int64)
        return stats, full_k1

    @staticmethod
    def _hierarchical_fixture():
        bounds = np.array(
            [[0, 1, 0, 2], [0, 1, 2, 4]], dtype=np.int64
        )
        target = np.array([[0, 0, 1, 1]], dtype=np.int64)
        k1 = np.array([[1, 1, 0, 0]], dtype=np.int64)
        k2 = np.array([[0, 1, 1, 0]], dtype=np.int64)
        k4 = np.array([[0, 0, 1, 1]], dtype=np.int64)
        return cell_phase_statistics(bounds, k1, k2, k4, target, 2)

    def test_binary_oracle_recomputes_full_miou_and_obeys_budget(self):
        stats, full_k1 = self._binary_dynamic_fixture()
        oracle = binary_greedy_oracle(
            stats,
            (0, 2, 3, 6),
            num_classes=2,
            reference_confusion=full_k1,
        )
        self.assertEqual([action["cell_index"] for action in oracle["actions"]], [0, 1])
        snapshots = oracle["snapshots"]
        np.testing.assert_array_equal(snapshots[0]["levels_by_cell"], [1, 1])
        np.testing.assert_array_equal(snapshots[1]["levels_by_cell"], [1, 1])
        np.testing.assert_array_equal(snapshots[2]["levels_by_cell"], [4, 1])
        np.testing.assert_array_equal(snapshots[3]["levels_by_cell"], [4, 4])
        for snapshot in snapshots:
            self.assertLessEqual(
                snapshot["actual_extra_proxy_cost"],
                snapshot["requested_extra_proxy_cost"],
            )

        cell_one_delta = (
            stats[1]["confusion"]["k4"]
            - stats[1]["confusion"]["k1"]
        )
        initial_cell_one_gain = (
            mean_iou_from_confusion(full_k1 + cell_one_delta)
            - mean_iou_from_confusion(full_k1)
        )
        # The second gain is evaluated after cell 0 changed the full confusion,
        # not copied from a static singleton ranking.
        self.assertNotAlmostEqual(
            oracle["actions"][1]["full_miou_gain"], initial_cell_one_gain
        )
        expected_full = full_k1.copy()
        for record in stats:
            expected_full += (
                record["confusion"]["k4"] - record["confusion"]["k1"]
            )
        np.testing.assert_array_equal(
            snapshots[-1]["aggregate"]["full_confusion"], expected_full
        )

    def test_binary_oracle_stops_when_no_positive_marginal_exists(self):
        bounds = np.array([[0, 1, 0, 1]], dtype=np.int64)
        target = np.array([[0]], dtype=np.int64)
        k1 = np.array([[0]], dtype=np.int64)
        k2 = np.array([[0]], dtype=np.int64)
        k4 = np.array([[1]], dtype=np.int64)
        stats = cell_phase_statistics(bounds, k1, k2, k4, target, 2)
        oracle = binary_greedy_oracle(stats, (3,), num_classes=2)
        self.assertEqual(oracle["actions"], ())
        self.assertEqual(
            oracle["snapshots"][0]["actual_extra_proxy_cost"], 0
        )
        np.testing.assert_array_equal(
            oracle["snapshots"][0]["levels_by_cell"], [1]
        )

    def test_hierarchical_path_enforces_precedence_and_reaches_endpoints(self):
        stats = self._hierarchical_fixture()
        oracle = hierarchical_greedy_oracle(
            stats, tuple(range(7)), num_classes=2
        )
        seen_k2 = set()
        for action in oracle["actions"]:
            cell = action["cell_index"]
            if action["from_level"] == 1:
                self.assertEqual(action["to_level"], 2)
                seen_k2.add(cell)
            else:
                self.assertEqual(action["from_level"], 2)
                self.assertEqual(action["to_level"], 4)
                self.assertIn(cell, seen_k2)
            self.assertGreater(action["full_miou_gain"], 0.0)
        np.testing.assert_array_equal(
            oracle["snapshots"][0]["levels_by_cell"], [1, 1]
        )
        np.testing.assert_array_equal(
            oracle["path_endpoint"]["levels_by_cell"], [4, 4]
        )
        for snapshot in oracle["snapshots"]:
            self.assertLessEqual(
                snapshot["actual_extra_proxy_cost"],
                snapshot["requested_extra_proxy_cost"],
            )
        self.assertEqual(
            oracle["snapshots"][3]["explored_through_budget_cost"], 2
        )

    def test_hierarchical_oracle_does_not_cross_nonpositive_k2_bridge(self):
        bounds = np.array([[0, 1, 0, 1]], dtype=np.int64)
        target = np.array([[0]], dtype=np.int64)
        k1 = np.array([[0]], dtype=np.int64)
        k2 = np.array([[1]], dtype=np.int64)
        k4 = np.array([[0]], dtype=np.int64)
        stats = cell_phase_statistics(bounds, k1, k2, k4, target, 2)
        oracle = hierarchical_greedy_oracle(stats, (3,), num_classes=2)
        self.assertEqual(oracle["actions"], ())
        np.testing.assert_array_equal(
            oracle["path_endpoint"]["levels_by_cell"], [1]
        )

    def test_hierarchical_budget_filters_unaffordable_action_before_ranking(self):
        bounds = np.array(
            [[0, 1, 0, 4], [0, 1, 4, 5]], dtype=np.int64
        )
        target = np.zeros((1, 5), dtype=np.int64)
        k1 = np.ones((1, 5), dtype=np.int64)
        k2 = np.array([[0, 1, 1, 1, 0]], dtype=np.int64)
        k4 = np.zeros((1, 5), dtype=np.int64)
        stats = cell_phase_statistics(bounds, k1, k2, k4, target, 2)
        full_k1 = np.array([[5, 5], [0, 0]], dtype=np.int64)

        global_result = hierarchical_greedy_oracle(
            stats,
            (2,),
            num_classes=2,
            reference_confusion=full_k1,
        )
        snapshot = global_result["snapshots"][0]
        # After cell 0 reaches K2, its +2 K4 action has the best ratio but no
        # longer fits.  The +1 K2 action for cell 1 must still be considered.
        np.testing.assert_array_equal(snapshot["levels_by_cell"], [2, 2])
        self.assertEqual(snapshot["actual_extra_proxy_cost"], 2)
        self.assertAlmostEqual(snapshot["aggregate"]["full_miou"], 0.35)

        capped = hierarchical_group_budget_oracle(
            stats,
            ("g", "g"),
            {"g": 2},
            num_classes=2,
            reference_confusion_by_group={"g": full_k1},
        )
        np.testing.assert_array_equal(
            capped["snapshot"]["levels_by_cell"],
            snapshot["levels_by_cell"],
        )
        np.testing.assert_array_equal(
            capped["snapshot"]["aggregate"]["full_confusion"],
            snapshot["aggregate"]["full_confusion"],
        )

    def test_hierarchical_oracle_ranks_gain_per_cost_not_absolute_gain(self):
        bounds = np.array(
            [[0, 1, 0, 6], [0, 1, 6, 8]], dtype=np.int64
        )
        target = np.zeros((1, 8), dtype=np.int64)
        k1 = np.ones((1, 8), dtype=np.int64)
        k2 = np.array([[0, 0, 0, 1, 1, 1, 0, 0]], dtype=np.int64)
        k4 = np.zeros((1, 8), dtype=np.int64)
        stats = cell_phase_statistics(bounds, k1, k2, k4, target, 2)
        oracle = hierarchical_greedy_oracle(
            stats,
            (6,),
            num_classes=2,
            reference_confusion=np.array([[2, 8], [0, 0]], dtype=np.int64),
        )
        actions = oracle["actions"]
        self.assertEqual(
            [
                (action["cell_index"], action["from_level"], action["to_level"])
                for action in actions
            ],
            [(0, 1, 2), (1, 1, 2), (0, 2, 4)],
        )
        # After action 1, cell 0's K2->K4 absolute gain is larger, but its
        # cost-normalized gain is lower than cell 1's K1->K2 action.
        self.assertGreater(actions[2]["full_miou_gain"], actions[1]["full_miou_gain"])

    def test_group_caps_compete_on_one_global_confusion(self):
        stats, _ = self._binary_dynamic_fixture()
        reference_a = np.array([[4, 1], [0, 4]], dtype=np.int64)
        reference_b = np.array([[4, 0], [1, 4]], dtype=np.int64)
        global_reference = reference_a + reference_b
        result = binary_group_budget_oracle(
            stats,
            ("a", "b"),
            {"a": 3, "b": 0},
            num_classes=2,
            reference_confusion_by_group={"a": reference_a, "b": reference_b},
        )
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["actions"][0]["group_id"], "a")
        self.assertAlmostEqual(
            result["actions"][0]["full_miou_before"],
            mean_iou_from_confusion(global_reference),
        )
        usage = {group["group_id"]: group for group in result["groups"]}
        self.assertEqual(usage["a"]["used_extra_proxy_cost"], 3)
        self.assertEqual(usage["b"]["used_extra_proxy_cost"], 0)
        np.testing.assert_array_equal(
            result["snapshot"]["levels_by_cell"], [4, 1]
        )

    def test_group_caps_use_global_objective_even_when_both_groups_can_act(self):
        bounds = np.array(
            [[0, 1, 0, 8], [0, 1, 8, 16]], dtype=np.int64
        )
        target = np.array(
            [[0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1]],
            dtype=np.int64,
        )
        k1 = np.array(
            [[0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0]],
            dtype=np.int64,
        )
        k4 = np.array(
            [[1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1]],
            dtype=np.int64,
        )
        stats = cell_phase_statistics(bounds, k1, k1, k4, target, 2)
        result = binary_group_budget_oracle(
            stats,
            ("a", "b"),
            {"a": 3, "b": 3},
            num_classes=2,
        )
        # Both actions improve their own image in isolation, but B is harmful
        # to the pooled confusion after A.  One global objective selects A only.
        self.assertEqual(
            [(action["cell_index"], action["group_id"]) for action in result["actions"]],
            [(0, "a")],
        )
        np.testing.assert_array_equal(result["levels_by_cell"], [4, 1])
        usage = {group["group_id"]: group for group in result["groups"]}
        self.assertEqual(usage["a"]["used_extra_proxy_cost"], 3)
        self.assertEqual(usage["b"]["used_extra_proxy_cost"], 0)

    def test_group_caps_prevent_budget_borrowing_and_merge_assignments(self):
        bounds = np.array(
            [
                [0, 1, 0, 1],
                [0, 1, 1, 2],
                [0, 1, 2, 3],
                [0, 1, 3, 4],
            ],
            dtype=np.int64,
        )
        target = np.array([[0, 1, 0, 1]], dtype=np.int64)
        k1 = np.array([[1, 0, 1, 0]], dtype=np.int64)
        k2 = target.copy()
        k4 = target.copy()
        stats = cell_phase_statistics(bounds, k1, k2, k4, target, 2)
        result = binary_group_budget_oracle(
            stats,
            ("a", "a", "b", "b"),
            {"a": 3, "b": 3},
            num_classes=2,
        )
        usage = {group["group_id"]: group for group in result["groups"]}
        self.assertEqual(usage["a"]["selected_counts"]["k4"], 1)
        self.assertEqual(usage["b"]["selected_counts"]["k4"], 1)
        self.assertEqual(result["snapshot"]["actual_extra_proxy_cost"], 6)
        self.assertEqual(
            int(np.count_nonzero(result["snapshot"]["levels_by_cell"] == 4)),
            2,
        )

    def test_hierarchical_group_quota_is_enforced_globally(self):
        stats = self._hierarchical_fixture()
        result = hierarchical_group_budget_oracle(
            stats,
            ("a", "b"),
            {"a": 1, "b": 1},
            num_classes=2,
        )
        np.testing.assert_array_equal(
            result["snapshot"]["levels_by_cell"], [2, 2]
        )
        self.assertEqual(
            [group["used_extra_proxy_cost"] for group in result["groups"]],
            [1, 1],
        )
        self.assertTrue(
            all(action["full_miou_gain"] > 0.0 for action in result["actions"])
        )


class SelectionCostAndGateTest(unittest.TestCase):
    def test_top_q_is_floor_budgeted_and_ties_use_lower_cell_index(self):
        scores = deployment_score_vector("entropy", [1.0, -np.inf, 1.0, 2.0])
        np.testing.assert_array_equal(
            deterministic_top_q_indices(scores, 0.5, deployment=True),
            np.array([3, 0]),
        )
        self.assertEqual(
            deterministic_top_q_indices(scores, 0.24, deployment=True).size, 0
        )
        self.assertEqual(
            deterministic_top_q_indices(scores, 1.0, deployment=True).size, 4
        )

    def test_deployment_selection_rejects_oracle_scores(self):
        oracle = CellScoreVector("oracle", np.array([1.0, 0.0]), True)
        with self.assertRaisesRegex(ValueError, "must not use ground-truth"):
            deterministic_top_q_indices(oracle, 0.5, deployment=True)
        with self.assertRaises(ValueError):
            CellScoreVector("bad", np.array([0.0, np.nan]), False)
        with self.assertRaises(ValueError):
            CellScoreVector("bad", np.array([0.0, np.inf]), False)

    def test_random_control_is_deterministic_and_budget_matched(self):
        first = deterministic_random_indices(20, 0.25, seed=42, replicate=3)
        second = deterministic_random_indices(20, 0.25, seed=42, replicate=3)
        other = deterministic_random_indices(20, 0.25, seed=42, replicate=4)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 5)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(np.all(first[:-1] < first[1:]))

    def test_geometry_aware_random_uses_eligible_cells_before_outer_cells(self):
        eligible = np.array([False, True, True, False, False], dtype=bool)
        selected = deterministic_random_indices(
            5, 0.4, seed=42, replicate=0, eligible_mask=eligible
        )
        np.testing.assert_array_equal(selected, np.array([1, 2]))
        overflow = deterministic_random_indices(
            5, 0.8, seed=42, replicate=0, eligible_mask=eligible
        )
        self.assertEqual(len(overflow), 4)
        self.assertTrue({1, 2}.issubset(set(overflow.tolist())))
        with self.assertRaisesRegex(TypeError, "bool vector"):
            deterministic_random_indices(
                5, 0.4, seed=42, eligible_mask=np.ones(5, dtype=np.int64)
            )

    def test_binary_and_hierarchical_cost_endpoints(self):
        self.assertEqual(binary_phase_cost(0, 3)["forward_equivalent_cost"], 1.0)
        self.assertEqual(binary_phase_cost(1, 3)["forward_equivalent_cost"], 2.0)
        self.assertEqual(binary_phase_cost(3, 3)["forward_equivalent_cost"], 4.0)
        self.assertEqual(
            hierarchical_phase_cost(0, 0, 4)["forward_equivalent_cost"], 1.0
        )
        self.assertEqual(
            hierarchical_phase_cost(4, 0, 4)["forward_equivalent_cost"], 2.0
        )
        self.assertEqual(
            hierarchical_phase_cost(4, 4, 4)["forward_equivalent_cost"], 4.0
        )
        self.assertEqual(
            hierarchical_phase_cost(2, 1, 4)["forward_equivalent_cost"], 2.0
        )
        with self.assertRaisesRegex(ValueError, "subset"):
            hierarchical_phase_cost(1, 2, 4)

    def test_retention_and_preregistered_gate_are_conjunctive(self):
        self.assertAlmostEqual(gain_retention(0.50, 0.60, 0.57), 0.70)
        passing = evaluate_phase_utility_gate(
            k1_miou=0.50,
            k2_miou=0.55,
            k4_miou=0.60,
            mixed_miou=0.57,
            forward_equivalent_cost=2.0,
            k2_small_error_rate=0.20,
            mixed_small_error_rate=0.20,
            k2_thin_error_rate=0.30,
            mixed_thin_error_rate=0.29,
            random_p95_miou=0.56,
        )
        self.assertEqual(passing["outcome"], "GO")
        self.assertTrue(all(passing["checks"].values()))

        failing = evaluate_phase_utility_gate(
            k1_miou=0.50,
            k2_miou=0.55,
            k4_miou=0.60,
            mixed_miou=0.57,
            forward_equivalent_cost=2.01,
            k2_small_error_rate=0.20,
            mixed_small_error_rate=0.21,
            k2_thin_error_rate=0.30,
            mixed_thin_error_rate=0.29,
            random_p95_miou=0.56,
        )
        self.assertEqual(failing["outcome"], "NO_GO")
        self.assertFalse(failing["checks"]["cost_within_budget"])
        self.assertFalse(failing["checks"]["small_not_worse_than_k2"])

    def test_gate_fails_closed_without_positive_k4_gain_or_region_support(self):
        result = evaluate_phase_utility_gate(
            k1_miou=0.50,
            k2_miou=0.50,
            k4_miou=0.49,
            mixed_miou=0.50,
            forward_equivalent_cost=1.0,
            k2_small_error_rate=None,
            mixed_small_error_rate=None,
            k2_thin_error_rate=None,
            mixed_thin_error_rate=None,
            random_p95_miou=0.49,
        )
        self.assertFalse(result["passed"])
        self.assertIsNone(result["observed"]["k4_gain_retention"])
        with self.assertRaisesRegex(ValueError, "K4 mIoU"):
            gain_retention(0.5, 0.5, 0.5)

    def test_binary_curve_uses_cell_aggregation_at_endpoints(self):
        bounds = np.array([[0, 1, 0, 1], [0, 1, 1, 2]], dtype=np.int64)
        target = np.array([[0, 1]])
        k1 = np.array([[1, 1]])
        k2 = np.array([[0, 0]])
        k4 = np.array([[0, 1]])
        stats = cell_phase_statistics(bounds, k1, k2, k4, target, 2)
        scores = oracle_net_correct_scores(stats)
        curve = binary_selection_curve(
            stats, scores, (0.0, 1.0), num_classes=2
        )
        np.testing.assert_array_equal(
            curve[0]["aggregate"]["confusion"],
            confusion_from_arrays(k1, target, 2),
        )
        np.testing.assert_array_equal(
            curve[1]["aggregate"]["confusion"],
            confusion_from_arrays(k4, target, 2),
        )
        self.assertEqual(curve[0]["cost"]["forward_equivalent_cost"], 1.0)
        self.assertEqual(curve[1]["cost"]["forward_equivalent_cost"], 4.0)

    def test_custom_gate_thresholds_are_honored(self):
        thresholds = PhaseUtilityGateThresholds(min_k4_gain_retention=0.8)
        result = evaluate_phase_utility_gate(
            k1_miou=0.50,
            k2_miou=0.54,
            k4_miou=0.60,
            mixed_miou=0.57,
            forward_equivalent_cost=2.0,
            k2_small_error_rate=0.2,
            mixed_small_error_rate=0.2,
            k2_thin_error_rate=0.2,
            mixed_thin_error_rate=0.2,
            random_p95_miou=0.56,
            thresholds=thresholds,
        )
        self.assertFalse(result["checks"]["retains_k4_gain"])


if __name__ == "__main__":
    unittest.main()
