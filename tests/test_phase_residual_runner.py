"""CPU checks for the matched phase-residual runner's sealed glue logic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.cache_whu_e0_slide_companion import E0SlideCompanionRecord
from scripts.phase_residual_common import center_class_logits, fix_mask_rms_scale
from scripts.run_whu_phase_distillation import SmallArrayCache, canonical_json_sha256
from scripts.run_whu_phase_residual_probe import (
    companion_crops_for_batch,
    final_resource_decision,
    magnitude_statistics,
    validate_fixed_batch_identity,
    validate_reference_objective,
)


class PhaseResidualRunnerTests(unittest.TestCase):
    def test_magnitude_statistics_report_rms_without_selecting_scale(self):
        values = torch.tensor(
            [[[[3.0, 0.0]], [[4.0, 0.0]]]], dtype=torch.float32
        )
        mask = torch.tensor([[[True, False]]])
        result = magnitude_statistics(values, mask, divisor=5.0)
        self.assertEqual(result["selected_values"], 2)
        self.assertAlmostEqual(result["rms"], np.sqrt((0.6**2 + 0.8**2) / 2))
        self.assertEqual(result["divisor"], 5.0)
        with self.assertRaises(ValueError):
            magnitude_statistics(values, torch.zeros_like(mask))
        centered = center_class_logits(values)
        scale = fix_mask_rms_scale(centered, mask)
        audited = magnitude_statistics(center_class_logits(centered), mask)
        self.assertEqual(audited["rms"], scale)

    def test_companion_crop_uses_original_batch_coordinates_and_float32(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logits = np.broadcast_to(
                np.arange(512, dtype=np.float16)[None, None, :],
                (7, 512, 512),
            ).copy()
            path = root / "e0.npy"
            np.save(path, logits)
            import hashlib

            array_sha = hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest()
            record = E0SlideCompanionRecord(
                index=0,
                sample_name="tile",
                full_shape_hw=(512, 512),
                bounds_yxyx=(0, 512, 0, 512),
                logits_path=path,
                logits_shape=tuple(logits.shape),
                logits_array_sha256=array_sha,
                logits_file_sha256="a" * 64,
                teacher_record_sha256="b" * 64,
            )
            batch = {
                "image_index": torch.tensor([0]),
                "crop_y": torch.tensor([0]),
                "crop_x": torch.tensor([0]),
            }
            actual = companion_crops_for_batch(
                batch, {0: record}, array_cache=SmallArrayCache(1)
            )
            self.assertEqual(tuple(actual.shape), (1, 7, 512, 512))
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual[0], torch.from_numpy(logits).float())

    def test_old_fixed_batch_anchor_is_exact_and_fail_closed(self):
        expected = {"optical_sha256": "a", "crop_y": [1, 2]}
        validate_fixed_batch_identity(dict(expected), expected)
        with self.assertRaisesRegex(RuntimeError, "crop_y"):
            validate_fixed_batch_identity(
                {"optical_sha256": "a", "crop_y": [1, 3]}, expected
            )

    def test_reference_validator_seals_correction_run_and_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_sha = "a" * 64
            teacher_sha = "b" * 64
            structure_sha = "c" * 64
            protocol = {
                "execution_mode": "objective",
                "objective_mask_mode": "correction",
                "baseline_checkpoint_sha256": baseline_sha,
                "immutable_manifests": {
                    "teacher_manifest_sha256": teacher_sha,
                    "structure_manifest_sha256": structure_sha,
                },
            }
            protocol_sha = canonical_json_sha256(protocol)
            config = {"protocol": protocol, "protocol_sha256": protocol_sha}
            summary = {
                "status": "PASS",
                "mode": "objective",
                "objective_mask_mode": "correction",
                "steps": 100,
                "protocol_sha256": protocol_sha,
                "fixed_variables": {
                    "learning_rate": 1e-4,
                    "weight_decay": 0.01,
                    "same_cached_batch": True,
                    "same_zero_initialization": True,
                },
                "fixed_batch": {"crop_y": [1]},
                "initial_branch_metadata": {
                    "combined_branch": {"state_sha256": "d" * 64}
                },
                "e0_state_audit": {
                    "unchanged": True,
                    "all_parameters_frozen": True,
                    "all_parameter_gradients_absent": True,
                    "e0_training_flag": False,
                },
            }
            (root / "run_config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            summary_path = root / "objective_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            result = validate_reference_objective(
                summary_path,
                baseline_sha256=baseline_sha,
                teacher_manifest_sha256=teacher_sha,
                structure_manifest_sha256=structure_sha,
                learning_rate=1e-4,
                weight_decay=0.01,
            )
            self.assertEqual(result["fixed_batch"], {"crop_y": [1]})
            with self.assertRaisesRegex(RuntimeError, "teacher manifest"):
                validate_reference_objective(
                    summary_path,
                    baseline_sha256=baseline_sha,
                    teacher_manifest_sha256="e" * 64,
                    structure_manifest_sha256=structure_sha,
                    learning_rate=1e-4,
                    weight_decay=0.01,
                )

    def test_actual_full_slide_gate_is_primary_even_if_arm_a_diagnostic_fails(self):
        passing_gate = {"passes": True}
        decision = final_resource_decision(passing_gate, arm_a_capacity_pass=False)
        self.assertEqual(
            decision["outcome"],
            "ELIGIBLE_FOR_SEPARATELY_PREREGISTERED_SHORT_TRAINING",
        )
        self.assertFalse(decision["arm_a_capacity_diagnostic_pass"])
        self.assertTrue(decision["arm_a_is_not_an_additional_behavior_gate"])
        failing = final_resource_decision({"passes": False}, True)
        self.assertIn("RESOURCE_NO_GO", failing["outcome"])


if __name__ == "__main__":
    unittest.main()
