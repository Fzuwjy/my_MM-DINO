"""CPU contracts for the EarthMiss FRM-P5 prototype-transfer experiment."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from models.MMDINO.Decoder import Decoder  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from models.MMDINO.dino_segment import DINOSegmentModule  # noqa: E402
from models.MMDINO.sample_adapter import SampleAdapter  # noqa: E402
from scripts.earthmiss_frmp5_prototype_common import (  # noqa: E402
    assert_batchnorm_buffers_equal,
    batch_class_prototypes,
    exact_area_occupancy,
    preserve_rng_state,
    prototype_transfer_infonce,
    prototype_weight_multiplier,
    snapshot_batchnorm_buffers,
    temporary_batchnorm_eval,
)
from scripts import train_earthmiss_frmp5_prototype_v0 as runner  # noqa: E402
from scripts import diagnose_earthmiss_frmp5_prototype_v0 as diagnostic  # noqa: E402
from scripts import analyze_earthmiss_frmp5_prototype_v0 as analyzer  # noqa: E402


class _TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.calls = 0

    def get_intermediate_layers(self, tensor, n):
        self.calls += 1
        batch = tensor.shape[0]
        value = torch.nn.functional.adaptive_avg_pool2d(
            tensor.mean(dim=1, keepdim=True),
            (4, 4),
        )
        value = value.flatten(2).transpose(1, 2).expand(batch, 16, 4)
        return tuple(value + float(index) for index in range(4))


def _tiny_state_model():
    model = DINOSegmentModule.__new__(DINOSegmentModule)
    torch.nn.Module.__init__(model)
    model.num_modalities = 2
    model.use_optical_stem = False
    model.use_sar_logit_residual = False
    model.backbone_type = "dinov3_vits16"
    model.backbone = _TinyBackbone()
    model.adapter = SampleAdapter(
        in_channels=4,
        out_channels=[16, 16, 16, 16],
        num_modalities=2,
    )
    model.decoder = Decoder(
        n_classes=3,
        in_channels=[16, 16, 16, 16],
        out_channels=16,
        num_modalities=2,
        raw_logits=True,
    )
    return model


class AreaOccupancyTest(unittest.TestCase):
    def test_exact_occupancy_keeps_ignore_as_empty_area(self):
        target = torch.tensor(
            [[
                [0, 0, 1, 1],
                [0, 8, 1, 1],
                [2, 2, 3, 3],
                [2, 2, 3, 8],
            ]]
        )
        occupancy = exact_area_occupancy(
            target,
            (2, 2),
            num_classes=4,
            ignore_index=8,
        )
        self.assertEqual(tuple(occupancy.shape), (1, 4, 2, 2))
        self.assertEqual(float(occupancy[0, 0, 0, 0]), 0.75)
        self.assertEqual(float(occupancy[0, 1, 0, 1]), 1.0)
        self.assertEqual(float(occupancy[0, 2, 1, 0]), 1.0)
        self.assertEqual(float(occupancy[0, 3, 1, 1]), 0.75)
        self.assertEqual(float(occupancy[:, :, 0, 0].sum()), 0.75)

    def test_area_mapping_rejects_non_divisible_grid_and_bad_ids(self):
        with self.assertRaisesRegex(ValueError, "exactly divisible"):
            exact_area_occupancy(torch.zeros(1, 5, 4), (2, 2))
        invalid = torch.zeros(1, 4, 4, dtype=torch.long)
        invalid[0, 0, 0] = 9
        with self.assertRaisesRegex(ValueError, "invalid class ids"):
            exact_area_occupancy(invalid, (2, 2))


class PrototypeLossTest(unittest.TestCase):
    @staticmethod
    def _two_class_case():
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[:, :, 2:] = 1
        feature = torch.zeros(1, 2, 2, 2)
        feature[:, 0, :, 0] = 1.0
        feature[:, 1, :, 1] = 1.0
        return feature, target

    def test_correct_class_pairing_has_lower_loss_than_swapped_anchor(self):
        query, target = self._two_class_case()
        correct, stats = prototype_transfer_infonce(
            query,
            query,
            target,
            num_classes=2,
            minimum_raw_support=1.0,
        )
        swapped = query.flip(1)
        wrong, _ = prototype_transfer_infonce(
            query,
            swapped,
            target,
            num_classes=2,
            minimum_raw_support=1.0,
        )
        self.assertLess(float(correct), float(wrong))
        torch.testing.assert_close(stats.positive_cosine, torch.ones(2))
        self.assertEqual(stats.support_mask.tolist(), [True, True])

    def test_anchor_is_detached_but_sar_self_control_has_negative_gradients(self):
        query, target = self._two_class_case()
        query = (query + 0.2).requires_grad_()
        anchor = query.detach().clone().requires_grad_()
        loss, stats = prototype_transfer_infonce(
            query,
            anchor,
            target,
            num_classes=2,
            minimum_raw_support=1.0,
        )
        loss.backward()
        self.assertIsNotNone(query.grad)
        self.assertGreater(float(query.grad.abs().sum()), 0.0)
        self.assertIsNone(anchor.grad)
        torch.testing.assert_close(stats.positive_cosine, torch.ones(2))

    def test_raw_area_support_precedes_squared_purity_mass(self):
        target = torch.tensor(
            [[
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1],
            ]]
        )
        query = torch.randn(1, 3, 2, 2, requires_grad=True)
        loss, stats = prototype_transfer_infonce(
            query,
            query.detach(),
            target,
            num_classes=2,
            minimum_raw_support=2.0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats.support_mask.tolist(), [True, True])
        self.assertTrue(torch.all(stats.purity_mass < stats.raw_support))
        self.assertTrue(torch.all(stats.effective_sample_size > 0))

    def test_single_supported_class_skips_with_connected_zero(self):
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        query = torch.randn(1, 3, 2, 2, requires_grad=True)
        loss, stats = prototype_transfer_infonce(
            query,
            query.detach(),
            target,
            num_classes=2,
            minimum_raw_support=1.0,
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(stats.support_mask.tolist(), [True, False])
        loss.backward()
        torch.testing.assert_close(query.grad, torch.zeros_like(query))


class StateSafetyTest(unittest.TestCase):
    def test_temporary_bn_eval_preserves_buffers_flags_and_gradients(self):
        module = torch.nn.Sequential(
            torch.nn.Conv2d(2, 2, 1),
            torch.nn.BatchNorm2d(2),
        )
        module.train()
        before = snapshot_batchnorm_buffers(module)
        value = torch.randn(2, 2, 3, 3, requires_grad=True)
        with temporary_batchnorm_eval(module):
            self.assertFalse(module[1].training)
            module(value).sum().backward()
        self.assertTrue(module[1].training)
        self.assertIsNotNone(value.grad)
        self.assertIsNotNone(module[1].weight.grad)
        assert_batchnorm_buffers_equal(before, module)

    def test_rng_context_restores_python_numpy_and_torch(self):
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)
        expected = (random.random(), np.random.rand(), torch.rand(()))
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)
        with preserve_rng_state():
            random.random()
            np.random.rand()
            torch.rand(17)
        actual = (random.random(), np.random.rand(), torch.rand(()))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0.0, atol=0.0)

    def test_frozen_weight_schedule(self):
        self.assertEqual([prototype_weight_multiplier(e) for e in range(1, 6)], [0.0] * 5)
        self.assertEqual(prototype_weight_multiplier(6), 0.2)
        self.assertEqual(prototype_weight_multiplier(10), 1.0)
        self.assertEqual(prototype_weight_multiplier(50), 1.0)


class ModelFeatureApiTest(unittest.TestCase):
    def test_state_feature_api_is_opt_in_and_backpropagates_only_from_query(self):
        torch.manual_seed(19)
        model = _tiny_state_model()
        model.eval()
        original_keys = tuple(model.state_dict())
        rgb = torch.full((2, 3, 64, 64), 2.0)
        sar = torch.full((2, 1, 64, 64), 5.0)
        outputs = model.extract_frozen_backbone_outputs(rgb, sar)
        sar_state = canonical_availability("sar", batch_size=2)
        full_state = canonical_availability("full", batch_size=2)

        sar_p5 = model.extract_state_frm_p5_from_backbone_outputs(
            rgb,
            sar,
            backbone_outputs=outputs,
            availability=sar_state,
        )
        with torch.no_grad():
            full_p5 = model.extract_state_frm_p5_from_backbone_outputs(
                rgb,
                sar,
                backbone_outputs=outputs,
                availability=full_state,
            )
        self.assertEqual(tuple(sar_p5.shape), (2, 16, 2, 2))
        self.assertTrue(sar_p5.requires_grad)
        self.assertFalse(full_p5.requires_grad)
        self.assertFalse(torch.equal(sar_p5, full_p5))
        self.assertEqual(tuple(model.state_dict()), original_keys)

        sar_p5.square().mean().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.adapter.parameters())
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.decoder.frm.parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.decoder.fusion4.parameters())
        )

    def test_auxiliary_feature_call_does_not_change_eval_logits_or_bn(self):
        torch.manual_seed(23)
        model = _tiny_state_model().eval()
        reference = copy.deepcopy(model)
        rgb = torch.randn(2, 3, 64, 64)
        sar = torch.randn(2, 1, 64, 64)
        state = canonical_availability("sar", batch_size=2)
        outputs = model.extract_frozen_backbone_outputs(rgb, sar)
        bn_before = snapshot_batchnorm_buffers(model)
        _ = model.extract_state_frm_p5_from_backbone_outputs(
            rgb,
            sar,
            backbone_outputs=outputs,
            availability=state,
        )
        assert_batchnorm_buffers_equal(bn_before, model)
        actual = model(rgb, sar, availability=state)
        expected = reference(rgb, sar, availability=state)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


class _ToySegmentationLoss(torch.nn.Module):
    def forward(self, logits, target):
        del target
        return logits.square().mean()


def _training_inputs(batch_size=2):
    rgb = torch.randn(batch_size, 3, 64, 64)
    sar = torch.randn(batch_size, 1, 64, 64)
    target = torch.zeros(batch_size, 64, 64, dtype=torch.long)
    target[:, :, 32:] = 1
    return rgb, sar, target


class RunnerContractTest(unittest.TestCase):
    def test_formal_prototype_arms_require_health_but_smoke_is_exempt(self):
        args = runner.parse_args(["--arm", "p2"])
        with self.assertRaisesRegex(ValueError, "health-report"):
            runner.validate_args(args)
        smoke = runner.parse_args(["--arm", "p2", "--smoke-batches", "1"])
        runner.validate_args(smoke)
        p0 = runner.parse_args(["--arm", "p0"])
        runner.validate_args(p0)

    def test_zero_weight_p2_is_exactly_the_p0_forward_path(self):
        torch.manual_seed(31)
        p0_model = _tiny_state_model().train()
        p2_model = copy.deepcopy(p0_model).train()
        rgb, sar, target = _training_inputs()
        criterion = _ToySegmentationLoss()
        p0_before = snapshot_batchnorm_buffers(p0_model)
        p2_before = snapshot_batchnorm_buffers(p2_model)
        self.assertEqual(p0_before.keys(), p2_before.keys())

        p0 = runner.forward_training_losses(
            p0_model,
            criterion,
            rgb,
            sar,
            target,
            state="sar",
            arm="p0",
            prototype_weight=0.0,
        )
        p2 = runner.forward_training_losses(
            p2_model,
            criterion,
            rgb,
            sar,
            target,
            state="sar",
            arm="p2",
            prototype_weight=0.0,
        )
        for left, right in zip(p0[:3], p2[:3]):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        self.assertIsNone(p0[3])
        self.assertIsNone(p2[3])
        self.assertEqual(p0_model.backbone.calls, 1)
        self.assertEqual(p2_model.backbone.calls, 1)
        p0_buffers = snapshot_batchnorm_buffers(p0_model)
        p2_buffers = snapshot_batchnorm_buffers(p2_model)
        for name in p0_buffers:
            for field in p0_buffers[name]:
                torch.testing.assert_close(
                    p0_buffers[name][field],
                    p2_buffers[name][field],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_p2_uses_cached_two_backbones_and_preserves_auxiliary_bn(self):
        torch.manual_seed(37)
        model = _tiny_state_model().train()
        rgb, sar, target = _training_inputs()
        criterion = _ToySegmentationLoss()
        _, _, prototype, statistics = runner.forward_training_losses(
            model,
            criterion,
            rgb,
            sar,
            target,
            state="sar",
            arm="p2",
            prototype_weight=0.1,
        )
        self.assertEqual(model.backbone.calls, 2)
        self.assertTrue(torch.isfinite(prototype))
        self.assertIsNotNone(statistics)
        self.assertEqual(int(statistics.support_mask.sum()), 2)
        self.assertTrue(all(module.training for module in model.decoder.frm.modules()))

    def test_gradient_calibration_is_finite_and_state_preserving(self):
        torch.manual_seed(41)
        model = _tiny_state_model().train()
        criterion = _ToySegmentationLoss()
        batches = [_training_inputs() for _ in range(3)]
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        bn_before = snapshot_batchnorm_buffers(model)
        with patch.object(runner, "GRADIENT_CALIBRATION_BATCHES", 2):
            weight, record = runner.calibrate_prototype_weight(
                model,
                criterion,
                batches,
                arm="p2",
                device=torch.device("cpu"),
            )
        self.assertTrue(math.isfinite(weight))
        self.assertGreater(weight, 0.0)
        self.assertEqual(record["effective_batches"], 2)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter,
                parameters_before[name],
                rtol=0.0,
                atol=0.0,
            )
        assert_batchnorm_buffers_equal(bn_before, model)
        self.assertTrue(model.training)

    def test_gradient_budget_uses_only_the_active_frm_p5_subgraph(self):
        model = _tiny_state_model()
        actual = {id(parameter) for parameter in runner.active_frm_p5_parameters(model.decoder.frm)}
        expected = {
            id(parameter)
            for source_scale in range(2, 6)
            for parameter in model.decoder.frm.conv_scales[
                f"conv_scale5_c{source_scale}"
            ].parameters()
        }
        expected.update(
            id(parameter)
            for parameter in model.decoder.frm.conv_aggregation_s5.parameters()
        )
        self.assertEqual(actual, expected)
        excluded = {
            id(parameter)
            for parameter in model.decoder.frm.conv_aggregation_s4.parameters()
        }
        self.assertTrue(actual.isdisjoint(excluded))

    def test_batch_trace_is_deterministic_and_state_sensitive(self):
        import hashlib

        rgb, sar, target = _training_inputs()
        first = hashlib.sha256()
        second = hashlib.sha256()
        third = hashlib.sha256()
        runner.update_batch_trace(
            first, state="sar", rgb=rgb, sar=sar, label=target
        )
        runner.update_batch_trace(
            second, state="sar", rgb=rgb, sar=sar, label=target
        )
        runner.update_batch_trace(
            third, state="full", rgb=rgb, sar=sar, label=target
        )
        self.assertEqual(first.hexdigest(), second.hexdigest())
        self.assertNotEqual(first.hexdigest(), third.hexdigest())

    def test_frozen_protocol_has_no_early_selection_epochs(self):
        self.assertEqual(runner.ELIGIBLE_SELECTION_EPOCHS[0], 15)
        self.assertEqual(runner.ELIGIBLE_SELECTION_EPOCHS[-1], 50)
        self.assertNotIn(5, runner.ELIGIBLE_SELECTION_EPOCHS)
        self.assertNotIn(10, runner.ELIGIBLE_SELECTION_EPOCHS)

    def test_health_report_is_a_strict_training_unlock(self):
        report = {
            "schema": runner.HEALTH_SCHEMA,
            "formal": True,
            "training_was_performed": False,
            "decision": {"training_allowed": True},
            "checkpoint": {"sha256": "fixed"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            record = runner.validate_health_report(path)
            self.assertEqual(record["schema"], runner.HEALTH_SCHEMA)
            self.assertTrue(record["manual_review_confirmed"])
            report["decision"]["training_allowed"] = False
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not pass"):
                runner.validate_health_report(path)

    def test_geometry_row_detects_full_class_separation_advantage(self):
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        target[:, :, 2:] = 1
        full = torch.zeros(1, 2, 2, 2)
        full[:, 0, :, 0] = 1.0
        full[:, 1, :, 1] = 1.0
        sar = torch.zeros_like(full)
        sar[:, 0, :, 0] = 1.0
        sar[:, 0, :, 1] = 0.8
        sar[:, 1, :, 1] = 0.2
        full_batch = batch_class_prototypes(
            full,
            target,
            num_classes=2,
            minimum_raw_support=1.0,
        )
        sar_batch = batch_class_prototypes(
            sar,
            target,
            num_classes=2,
            minimum_raw_support=1.0,
        )
        row = diagnostic._cross_state_geometry(full_batch, sar_batch)
        self.assertGreater(row["full_separation"], row["sar_separation"])
        accumulator = diagnostic.GeometryAccumulator()
        accumulator.update(row)
        summary = accumulator.summary()
        self.assertGreater(summary["full_minus_sar_separation"], 0.0)


def _synthetic_arm(arm, sar_score, full_score, *, trace="same"):
    records = []
    for epoch in range(1, runner.EPOCHS + 1):
        record = {
            "epoch": epoch,
            "train": {"data_trace_sha256": f"{trace}-{epoch}"},
        }
        if epoch % runner.EVAL_INTERVAL == 0:
            record["eligible_for_selection"] = (
                epoch in runner.ELIGIBLE_SELECTION_EPOCHS
            )
            record["validation"] = {
                state: {
                    "pooled": {"mIoU": score},
                    "by_city": {
                        "America": {"mIoU": score},
                        "Australia": {"mIoU": score - 0.01},
                        "Russian": {"mIoU": score + 0.01},
                    },
                }
                for state, score in (("sar", sar_score), ("full", full_score))
            }
        records.append(record)
    selected = max(
        [record for record in records if record.get("eligible_for_selection")],
        key=lambda record: record["validation"]["sar"]["pooled"]["mIoU"],
    )
    return {
        "arm": arm,
        "protocol": {
            "arm": arm,
            "seed": 42,
            "initial_model_state_sha256": "common",
        },
        "records": records,
        "selected": selected,
    }


class DecisionAnalyzerTest(unittest.TestCase):
    def test_frozen_val_rules_require_p0_and_p1_gains(self):
        p0 = _synthetic_arm("p0", 0.25, 0.34)
        p2 = _synthetic_arm("p2", 0.26, 0.339)
        p1 = _synthetic_arm("p1", 0.255, 0.338)
        report = analyzer.analyze(p0, p2, p1)
        self.assertTrue(report["p2_vs_p0"]["passed"])
        self.assertTrue(report["p2_vs_p1"]["passed"])
        self.assertEqual(
            report["decision"],
            "proceed_to_seeds_43_44_and_matched_sar_only_a",
        )

    def test_trace_mismatch_invalidates_matched_attribution(self):
        p0 = _synthetic_arm("p0", 0.25, 0.34, trace="p0")
        p2 = _synthetic_arm("p2", 0.26, 0.34, trace="p2")
        report = analyzer.analyze(p0, p2)
        self.assertFalse(
            report["p2_vs_p0"]["gates"]["all_epoch_data_trace_equal"]
        )
        self.assertEqual(report["decision"], "archive_v0_without_followup_sweep")


if __name__ == "__main__":
    unittest.main()
