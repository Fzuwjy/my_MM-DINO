"""CPU protocol tests for the EarthMiss P5-to-P4 FAM experiment arm."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.evaluate_earthmiss_fam_p5p4_v1 import (
    FlowAccumulator,
    parse_args as parse_evaluation_args,
    validate_args as validate_evaluation_args,
    validate_checkpoint,
)
from scripts.train_earthmiss_fam_p5p4_v1 import (
    FLOW_CHANNELS,
    PROTOCOL_REVISION,
    build_run_metadata,
    output_directory,
    parse_args as parse_training_args,
    validate_args as validate_training_args,
)
from tasks.segmentation.models.MMDINO.semantic_flow import ResidualFlowAlignment


class _FakeDataset:
    rgb_normalization = "imagenet"
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    sar_mean = (0.5,)
    sar_std = (0.25,)

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


class _FakeLoader:
    def __len__(self):
        return 3


def _training_args(run="C"):
    return SimpleNamespace(
        run=run,
        seed=42,
        fam_seed=42,
        epochs=50,
        batch_size=8,
        num_workers=4,
        window_size=512,
        learning_rate=1e-4,
        eval_interval=5,
        early_stop_patience_evals=0,
        output_root="/tmp/fam",
    )


def _metadata(run="C"):
    return build_run_metadata(
        _training_args(run),
        _FakeDataset(10),
        _FakeDataset(4),
        _FakeLoader(),
    )


class TrainingProtocolTests(unittest.TestCase):
    def test_parser_requires_an_explicit_primary_or_control_run(self):
        self.assertEqual(parse_training_args(["--run", "C"]).run, "C")
        self.assertEqual(parse_training_args(["--run", "A"]).run, "A")

    def test_metadata_inherits_baseline_and_freezes_the_fam_contract(self):
        metadata = _metadata()
        self.assertEqual(metadata["protocol_revision"], PROTOCOL_REVISION)
        self.assertEqual(metadata["augmentation"]["random_crop"], [512, 512])
        self.assertEqual(
            metadata["evaluation"]["checkpoint_selection_support"],
            "pooled_gt_present",
        )
        self.assertFalse(metadata["training_start"]["warm_start_from_c_e15"])
        fam = metadata["model"]["fam"]
        self.assertEqual(fam["flow_channels"], FLOW_CHANNELS)
        self.assertEqual(fam["padding_mode"], "border")
        self.assertFalse(fam["align_corners"])
        self.assertFalse(fam["learnable_gate"])
        self.assertIn("warp(P5,zero_flow)", fam["formula"])
        self.assertTrue(
            metadata["determinism"][
                "same_data_trace_and_common_initialization_required"
            ]
        )

    def test_run_a_is_marked_conditional_and_has_no_full_checkpoint(self):
        metadata = _metadata("A")
        roles = metadata["evaluation"]["checkpoint_roles"]
        self.assertEqual(roles, {"best_sar.pth": "primary_deployment"})
        self.assertIn("after C+FAM passes", metadata["evaluation"]["conditional_control"])

    def test_argument_validation_and_output_directory(self):
        args = _training_args()
        validate_training_args(args)
        self.assertEqual(
            output_directory(args),
            Path("/tmp/fam/fam_run_c_seed42"),
        )
        args.window_size = 510
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            validate_training_args(args)


class EvaluationProtocolTests(unittest.TestCase):
    @staticmethod
    def _checkpoint(role="primary_deployment", selection="sar"):
        return {
            "run": "C",
            "checkpoint_role": role,
            "selection_state": selection,
            "protocol": _metadata(),
        }

    def test_evaluator_defaults_to_the_paired_canonical_endpoints(self):
        args = parse_evaluation_args(["--checkpoint", "candidate.pth"])
        self.assertEqual(args.endpoints, ["full", "sar-canonical"])
        validate_evaluation_args(args)

    def test_checkpoint_validator_requires_frozen_protocol_and_sar_selection(self):
        validate_checkpoint(self._checkpoint())
        with self.assertRaisesRegex(ValueError, "SAR-selected"):
            validate_checkpoint(self._checkpoint(role="diagnostic_only"))
        diagnostic = self._checkpoint(role="diagnostic_only", selection="full")
        validate_checkpoint(diagnostic, allow_non_primary=True)

    def test_checkpoint_validator_rejects_semantic_drift(self):
        checkpoint = self._checkpoint()
        checkpoint["protocol"]["model"]["fam"]["padding_mode"] = "zeros"
        with self.assertRaisesRegex(ValueError, "contract changed"):
            validate_checkpoint(checkpoint)

    def test_evaluation_rejects_duplicate_endpoints(self):
        args = parse_evaluation_args(
            [
                "--checkpoint",
                "candidate.pth",
                "--endpoints",
                "full",
                "full",
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validate_evaluation_args(args)


class FlowAccumulatorTests(unittest.TestCase):
    def test_zero_initialized_module_reports_zero_displacement_and_correction(self):
        torch.manual_seed(3)
        module = ResidualFlowAlignment(4, flow_channels=2)
        high = torch.randn(2, 4, 3, 3)
        low = torch.randn(2, 4, 6, 6)
        aligned = module(high, low)
        accumulator = FlowAccumulator()
        accumulator.update(high, low, aligned, module)
        summary = accumulator.compute()
        self.assertEqual(summary["forward_calls"], 1)
        self.assertEqual(summary["flow_vectors"], 72)
        self.assertEqual(summary["magnitude"]["max"], 0.0)
        self.assertEqual(summary["grid_out_of_bounds_ratio"], 0.0)
        self.assertEqual(summary["correction_rms"], 0.0)

    def test_empty_accumulator_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no activations"):
            FlowAccumulator().compute()


if __name__ == "__main__":
    unittest.main()
