import copy
import gc
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.transforms import functional as TVF

from scripts.cache_whu_phase_teacher import build_phase_protocol
from scripts.phase_distillation_common import PhaseCorrectionBranch
from scripts.run_whu_phase_distillation import (
    COMMON_MEAN,
    COMMON_STD,
    PhaseImageRecord,
    WHUPhaseCropDataset,
    build_optimizers,
    checkpoint_epoch_from_path,
    committed_stage_result,
    crop_encoded_structure_mask,
    decode_whu_label,
    exact_spatial_crop,
    epoch_artifact_paths,
    expected_teacher_bounds,
    initialize_or_resume_output,
    kd_capacity_decision,
    normalize_common_optical,
    objective_gradient_diagnostics,
    publish_epoch_commit,
    rebuild_metric_indexes,
    load_paired_checkpoint,
    save_paired_checkpoint,
    stage_decision,
    validate_resume_commit,
    validate_teacher_protocol,
    verify_epoch_commit,
)


class PhaseDistillationRunnerTests(unittest.TestCase):
    def test_common_normalization_matches_released_tensor_formula(self):
        optical = np.array([[[0, 127, 255], [255, 64, 32]]], dtype=np.uint8)
        actual = normalize_common_optical(optical)
        expected = TVF.normalize(
            TVF.to_tensor(optical),
            COMMON_MEAN[:, 0, 0].tolist(),
            COMMON_STD[:, 0, 0].tolist(),
        ).numpy()
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        with self.assertRaises(TypeError):
            normalize_common_optical(optical.astype(np.float32))

    def test_whu_label_decode_preserves_ignore_contract(self):
        raw = np.array([[10, 20, 70, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(
            decode_whu_label(raw), np.array([[0, 1, 6, 7]], dtype=np.int64)
        )

    def test_exact_crop_never_pads_or_truncates(self):
        full = np.arange(6 * 7 * 3, dtype=np.int32).reshape(6, 7, 3)
        crop = exact_spatial_crop(full, y=2, x=3, size=4)
        np.testing.assert_array_equal(crop, full[2:6, 3:7])
        crop[0, 0, 0] = -1
        self.assertNotEqual(full[2, 3, 0], -1)
        with self.assertRaisesRegex(ValueError, "padding"):
            exact_spatial_crop(full, y=3, x=3, size=4)

    def test_teacher_protocol_and_bounds_are_sealed_to_accepted_geometry(self):
        protocol = build_phase_protocol()
        validate_teacher_protocol(protocol)
        self.assertEqual(
            expected_teacher_bounds((2048, 2304)),
            (512, 1520, 512, 1776),
        )
        for field, replacement in (
            ("stride_hw", [256, 256]),
            ("valid_margin", 256),
            ("control_offset_used_only_for_common_region", 8),
            ("normal_logits_cached", True),
        ):
            changed = copy.deepcopy(protocol)
            changed[field] = replacement
            with self.assertRaises(RuntimeError, msg=field):
                validate_teacher_protocol(changed)
        changed = copy.deepcopy(protocol)
        changed["common_alignment_shifts_dy_dx"] = changed[
            "common_alignment_shifts_dy_dx"
        ][:3]
        with self.assertRaises(RuntimeError):
            validate_teacher_protocol(changed)

    def test_diagnostic_dataset_accepts_one_manifest_image(self):
        record = PhaseImageRecord(
            index=0,
            sample_name="sample",
            optical_path=Path("rgb.tif"),
            sar_path=Path("sar.tif"),
            label_path=Path("label.tif"),
            full_shape_hw=(2048, 2048),
            label_sha256="a" * 64,
            bounds_yxyx=(512, 1536, 512, 1536),
            teacher_logits_path=Path("teacher.npy"),
            teacher_logits_shape=(7, 1024, 1024),
            teacher_logits_array_sha256="b" * 64,
            structure_mask_path=Path("mask.npy"),
            structure_mask_shape=(2048, 2048),
            structure_mask_bounds_yxyx=(0, 2048, 0, 2048),
            structure_mask_array_sha256="c" * 64,
        )
        dataset = WHUPhaseCropDataset([record], seed=42, length=3)
        self.assertEqual(len(dataset), 3)

    def test_first_screen_does_not_approximate_reflected_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = np.broadcast_to(
                np.arange(512, dtype=np.float16)[None, None, :],
                (7, 512, 512),
            ).copy()
            structure = np.zeros((512, 512), dtype=np.uint8)
            teacher_path = root / "teacher.npy"
            structure_path = root / "structure.npy"
            np.save(teacher_path, teacher)
            np.save(structure_path, structure)
            record = PhaseImageRecord(
                index=0,
                sample_name="sample",
                optical_path=root / "rgb.tif",
                sar_path=root / "sar.tif",
                label_path=root / "label.tif",
                full_shape_hw=(512, 512),
                label_sha256="unused",
                bounds_yxyx=(0, 512, 0, 512),
                teacher_logits_path=teacher_path,
                teacher_logits_shape=teacher.shape,
                teacher_logits_array_sha256="unused",
                structure_mask_path=structure_path,
                structure_mask_shape=structure.shape,
                structure_mask_bounds_yxyx=(0, 512, 0, 512),
                structure_mask_array_sha256="unused",
            )
            dataset = WHUPhaseCropDataset([record], seed=42, length=1)
            sources = {
                "optical": np.zeros((512, 512, 3), dtype=np.uint8),
                "sar": np.zeros((512, 512), dtype=np.uint8),
                "label": np.full((512, 512), 10, dtype=np.uint8),
            }
            dataset._source = lambda kind, path: sources[kind]
            dataset._verified_labels.add(0)
            dataset._verified_teacher_arrays.add(0)
            dataset._verified_structure_arrays.add(0)
            item = dataset[0]
            self.assertFalse(item["flip_h"])
            self.assertFalse(item["flip_v"])
            np.testing.assert_array_equal(item["teacher_logits"].numpy(), teacher)
            dataset._cache_arrays._values.clear()
            del item, dataset
            gc.collect()

    def test_epoch_commit_seals_artifacts_and_rebuilds_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = epoch_artifact_paths(output_dir, 5)
            paths["train"].write_text(
                json.dumps({"epoch": 5, "loss": 1.0}), encoding="utf-8"
            )
            paths["evaluation"].write_text(
                json.dumps({"epoch": 5, "miou": 54.2}), encoding="utf-8"
            )
            paths["checkpoint"].write_bytes(b"checkpoint")
            publish_epoch_commit(output_dir, 5)
            verify_epoch_commit(output_dir, 5)
            validate_resume_commit(output_dir, 5, paths["checkpoint"])
            self.assertEqual(checkpoint_epoch_from_path(paths["checkpoint"]), 5)
            with self.assertRaises(RuntimeError):
                checkpoint_epoch_from_path(output_dir / "latest.pth")
            rebuild_metric_indexes(output_dir)
            stage = committed_stage_result(output_dir, 5)
            self.assertTrue(stage["reconstructed_from_committed_epoch"])
            self.assertEqual(stage["stopped_after_epoch"], 5)
            self.assertIn('"epoch": 5', (output_dir / "train_metrics.jsonl").read_text())
            self.assertIn(
                '"miou": 54.2',
                (output_dir / "evaluation_metrics.jsonl").read_text(),
            )
            paths["checkpoint"].write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "hash differs"):
                verify_epoch_commit(output_dir, 5)

    def test_formal_epoch_one_precommit_directory_can_restart_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "formal"
            args = SimpleNamespace(
                output_dir=output_dir,
                resume_checkpoint=None,
                mode="formal",
            )
            protocol = {"name": "synthetic"}
            initialize_or_resume_output(args, protocol)
            (output_dir / "checkpoint_e1.pth").write_bytes(b"incomplete")
            initialize_or_resume_output(args, protocol)

            paths = epoch_artifact_paths(output_dir, 1)
            paths["train"].write_text(json.dumps({"epoch": 1}), encoding="utf-8")
            paths["checkpoint"].write_bytes(b"complete")
            publish_epoch_commit(output_dir, 1)
            with self.assertRaisesRegex(RuntimeError, "resume from"):
                initialize_or_resume_output(args, protocol)

    def test_structure_crop_uses_full_image_coordinates_and_bits(self):
        encoded = np.zeros((8, 9), dtype=np.uint8)
        encoded[2:6, 3:7] = 1
        encoded[4:8, 5:9] |= 2
        small, thin = crop_encoded_structure_mask(
            encoded, (10, 18, 20, 29), y=12, x=23, size=4
        )
        np.testing.assert_array_equal(small, encoded[2:6, 3:7] & 1 != 0)
        np.testing.assert_array_equal(thin, encoded[2:6, 3:7] & 2 != 0)

    def test_kd_capacity_decision_uses_last_ten_median_thresholds(self):
        self.assertEqual(
            kd_capacity_decision(1.0, [0.49] * 10)["outcome"],
            "CLEAR_CAPACITY",
        )
        self.assertEqual(
            kd_capacity_decision(1.0, [0.70] * 10)["outcome"],
            "WEAK_CAPACITY",
        )
        self.assertEqual(
            kd_capacity_decision(1.0, [0.90] * 10)["outcome"],
            "NO_DEMONSTRATED_CAPACITY",
        )

    def test_initial_gradient_diagnostic_separates_zero_head_and_upstream(self):
        torch.manual_seed(3)
        branch = PhaseCorrectionBranch(
            in_channels=4, hidden_channels=4, num_classes=2
        )
        base = torch.randn(1, 2, 8, 8)
        teacher = base.clone()
        teacher[:, 0] += 0.5
        prepared = {
            "base_logits": base,
            "p2": torch.randn(1, 4, 4, 4),
            "labels": torch.randint(0, 2, (1, 8, 8)),
            "teacher_logits": teacher,
            "kd_mask": torch.ones(1, 8, 8, dtype=torch.bool),
        }
        result = objective_gradient_diagnostics(
            branch,
            prepared,
            torch.nn.CrossEntropyLoss(),
        )
        self.assertGreater(result["groups"]["output_head"]["supervised_norm"], 0.0)
        self.assertGreater(result["groups"]["output_head"]["kd_norm"], 0.0)
        self.assertEqual(result["groups"]["upstream"]["supervised_norm"], 0.0)
        self.assertEqual(result["groups"]["upstream"]["kd_norm"], 0.0)
        cosine = result["groups"]["all"]["cosine"]
        self.assertGreaterEqual(cosine, -1.0)
        self.assertLessEqual(cosine, 1.0)

    @staticmethod
    def _differences(pair, absolute, small=0.0, thin=0.0):
        return {
            "r1_minus_r0_miou_pp": pair,
            "r1_minus_e0_miou_pp": absolute,
            "regions": {
                "component_area_le_256px2": {
                    "r1_minus_r0_error_rate_pp": small
                },
                "component_thickness_le_4px": {
                    "r1_minus_r0_error_rate_pp": thin
                },
            },
        }

    def test_e5_never_makes_final_no_go(self):
        decision = stage_decision(5, self._differences(-1.0, -1.0))
        self.assertEqual(decision["outcome"], "EARLY_TREND_ONLY")

    def test_e15_requires_both_point_one_gates_and_structure_guard(self):
        self.assertEqual(
            stage_decision(15, self._differences(0.10, 0.10))["outcome"],
            "GO",
        )
        self.assertNotEqual(
            stage_decision(15, self._differences(0.10, 0.099))["outcome"],
            "GO",
        )

    def test_checkpoint_contains_only_paired_state_and_resumes_epoch(self):
        r0 = PhaseCorrectionBranch(in_channels=4, hidden_channels=4, num_classes=2)
        r1 = copy.deepcopy(r0)
        optimizer0, optimizer1, scheduler0, scheduler1 = build_optimizers(
            r0, r1, learning_rate=1e-4, weight_decay=0.01
        )
        generator = torch.Generator().manual_seed(7)
        protocol = {"name": "synthetic"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_e15.pth"
            save_paired_checkpoint(
                path,
                epoch=15,
                r0_branch=r0,
                r1_branch=r1,
                optimizer0=optimizer0,
                optimizer1=optimizer1,
                scheduler0=scheduler0,
                scheduler1=scheduler1,
                loader_generator=generator,
                baseline_sha256="a" * 64,
                protocol=protocol,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertNotIn("e0_model", payload)
            self.assertNotIn("model", payload)

            new_r0 = PhaseCorrectionBranch(
                in_channels=4, hidden_channels=4, num_classes=2
            )
            new_r1 = copy.deepcopy(new_r0)
            new_o0, new_o1, new_s0, new_s1 = build_optimizers(
                new_r0, new_r1, learning_rate=1e-4, weight_decay=0.01
            )
            new_generator = torch.Generator().manual_seed(99)
            epoch = load_paired_checkpoint(
                path,
                r0_branch=new_r0,
                r1_branch=new_r1,
                optimizer0=new_o0,
                optimizer1=new_o1,
                scheduler0=new_s0,
                scheduler1=new_s1,
                loader_generator=new_generator,
                baseline_sha256="a" * 64,
                protocol=protocol,
            )
            self.assertEqual(epoch, 15)
            for expected, actual in zip(r0.parameters(), new_r0.parameters()):
                self.assertTrue(torch.equal(expected, actual))
        self.assertNotEqual(
            stage_decision(15, self._differences(0.10, 0.10, small=0.001))[
                "outcome"
            ],
            "GO",
        )


if __name__ == "__main__":
    unittest.main()
