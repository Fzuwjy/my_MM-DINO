"""Synthetic checks for the standalone E0-slide companion cache."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from scripts.cache_whu_e0_slide_companion import (
    ARTIFACT_TYPE,
    SCHEMA_VERSION,
    build_e0_slide_protocol,
    companion_record,
    crop_normal_slide_logits,
    initial_manifest,
    load_e0_slide_companion_records,
    read_bound_teacher_manifest,
    recover_completed_sidecar,
    teacher_manifest_binding,
    validate_resume_manifest,
)
from scripts.cache_whu_phase_teacher import (
    ARTIFACT_TYPE as TEACHER_ARTIFACT_TYPE,
    array_sha256,
    atomic_write_json,
    bounds_from_slice,
    build_phase_protocol,
    canonical_json_sha256,
    file_sha256,
    phase_common_slices,
)


class E0SlideCompanionCacheTest(unittest.TestCase):
    def _fixture(self, root: Path) -> dict:
        checkpoint = root / "baseline.pth"
        checkpoint.write_bytes(b"sealed checkpoint")
        sources = {}
        source_hashes = {}
        for name, payload in (
            ("rgb", b"rgb source"),
            ("sar", b"sar source"),
            ("label", b"label source"),
        ):
            path = root / f"{name}.tif"
            path.write_bytes(payload)
            sources[f"{name}_file"] = str(path.resolve())
            source_hashes[name] = file_sha256(path)

        full_shape = (1056, 1056)
        original_slice, _ = phase_common_slices(full_shape)
        bounds = bounds_from_slice(full_shape, original_slice)
        self.assertEqual(
            bounds,
            {"y_start": 512, "y_stop": 528, "x_start": 512, "x_stop": 528},
        )
        label = np.zeros(full_shape, dtype=np.int64)
        teacher_record = {
            "index": 0,
            "sample_name": "tile",
            "source": dict(sources),
            "source_file_sha256": source_hashes,
            "source_label_file_sha256": source_hashes["label"],
            "label_sha256": array_sha256(label),
            "full_shape_hw": list(full_shape),
            "bounds": bounds,
            "logits": {
                "path": "logits/teacher.npy",
                "metadata_path": "metadata/teacher.json",
                "dtype": "float16",
                "shape": [7, 16, 16],
                "array_sha256": "a" * 64,
                "file_sha256": "b" * 64,
                "nbytes": 7 * 16 * 16 * 2,
            },
        }
        teacher_protocol = build_phase_protocol()
        teacher_manifest = {
            "schema_version": 1,
            "artifact_type": TEACHER_ARTIFACT_TYPE,
            "status": "PASS",
            "scope": "subset-smoke",
            "split": "train",
            "full_dataset_length": 80,
            "requested_images": [
                {
                    "index": 0,
                    "sample_name": "tile",
                    **sources,
                }
            ],
            "baseline_checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": file_sha256(checkpoint),
            },
            "protocol": teacher_protocol,
            "protocol_sha256": canonical_json_sha256(teacher_protocol),
            "images": [teacher_record],
        }
        teacher_path = root / "teacher_manifest.json"
        atomic_write_json(teacher_path, teacher_manifest)
        validated_teacher = read_bound_teacher_manifest(teacher_path)
        binding = teacher_manifest_binding(teacher_path, validated_teacher)

        teacher_record_sha = canonical_json_sha256(
            {"teacher_record": teacher_record}
        )
        sample = {
            "index": 0,
            "sample_name": "tile",
            **sources,
            "full_shape_hw": list(full_shape),
            "bounds": bounds,
            "label_sha256": teacher_record["label_sha256"],
            "source_file_sha256": source_hashes,
            "teacher_record_sha256": teacher_record_sha,
        }
        output_dir = root / "companion"
        args = SimpleNamespace(
            baseline_checkpoint=checkpoint,
            output_dir=output_dir,
            seed=42,
            inference_batch_size=8,
        )
        protocol = build_e0_slide_protocol()
        e0_slide = np.arange(7 * 16 * 16, dtype=np.float16).reshape(7, 16, 16)
        return {
            "checkpoint": checkpoint,
            "teacher_path": teacher_path,
            "teacher_manifest": teacher_manifest,
            "teacher_record": teacher_record,
            "binding": binding,
            "label": label,
            "sample": sample,
            "args": args,
            "protocol": protocol,
            "e0_slide": e0_slide,
            "output_dir": output_dir,
        }

    def _persist_complete_companion(self, fixture: dict) -> tuple[Path, dict]:
        record = companion_record(
            fixture["output_dir"],
            fixture["sample"],
            label=fixture["label"],
            full_shape=tuple(fixture["label"].shape),
            bounds=fixture["sample"]["bounds"],
            e0_slide=fixture["e0_slide"],
            protocol=fixture["protocol"],
            binding=fixture["binding"],
            checkpoint_sha256=file_sha256(fixture["checkpoint"]),
        )
        manifest = initial_manifest(
            fixture["args"],
            full_length=80,
            selection=[fixture["sample"]],
            checkpoint_sha256=file_sha256(fixture["checkpoint"]),
            binding=fixture["binding"],
        )
        manifest["images"] = [record]
        manifest["status"] = "PASS"
        manifest_path = fixture["output_dir"] / "manifest.json"
        atomic_write_json(manifest_path, manifest)
        return manifest_path, record

    def test_protocol_is_independent_and_keeps_old_teacher_fused_only(self):
        old_protocol = build_phase_protocol()
        companion_protocol = build_e0_slide_protocol()
        self.assertFalse(old_protocol["normal_logits_cached"])
        self.assertEqual(companion_protocol["phase_dy_dx"], [0, 0])
        self.assertEqual(companion_protocol["crop_size_hw"], [512, 512])
        self.assertEqual(companion_protocol["stride_hw"], [341, 341])
        self.assertFalse(companion_protocol["teacher_cache_mutated"])
        self.assertEqual(SCHEMA_VERSION, 1)
        self.assertEqual(ARTIFACT_TYPE, "whu_e0_slide_companion_logits")
        self.assertNotEqual(ARTIFACT_TYPE, TEACHER_ARTIFACT_TYPE)

    def test_normal_slide_crop_is_full_resolution_and_original_coordinate(self):
        scores = torch.arange(1 * 7 * 5 * 6, dtype=torch.float32).reshape(
            1, 7, 5, 6
        )
        crop = crop_normal_slide_logits(scores, (slice(1, 4), slice(2, 6)))
        self.assertEqual(tuple(crop.shape), (7, 3, 4))
        torch.testing.assert_close(crop, scores[0, :, 1:4, 2:6])
        self.assertFalse(crop.requires_grad)
        with self.assertRaisesRegex(ValueError, "7 classes"):
            crop_normal_slide_logits(scores[:, :6], (slice(1, 4), slice(2, 6)))

    def test_loader_returns_runner_facing_records_and_verifies_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            manifest_path, record = self._persist_complete_companion(fixture)
            records = load_e0_slide_companion_records(
                manifest_path,
                fixture["teacher_path"],
                expected_checkpoint_sha256=file_sha256(fixture["checkpoint"]),
                verify_artifacts=True,
            )
            self.assertEqual(set(records), {0})
            loaded = records[0]
            self.assertEqual(loaded.index, 0)
            self.assertEqual(loaded.sample_name, "tile")
            self.assertEqual(loaded.full_shape_hw, (1056, 1056))
            self.assertEqual(loaded.bounds_yxyx, (512, 528, 512, 528))
            self.assertEqual(loaded.logits_shape, (7, 16, 16))
            self.assertEqual(
                loaded.logits_array_sha256,
                record["e0_slide_logits"]["array_sha256"],
            )
            np.testing.assert_array_equal(
                np.load(loaded.logits_path, allow_pickle=False), fixture["e0_slide"]
            )

    def test_loader_rejects_teacher_file_change_even_when_teacher_stays_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            manifest_path, _ = self._persist_complete_companion(fixture)
            changed = copy.deepcopy(fixture["teacher_manifest"])
            changed["runtime"] = {"harmless_but_changes_manifest_sha": True}
            fixture["teacher_path"].write_text(
                json.dumps(changed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not bound"):
                load_e0_slide_companion_records(
                    manifest_path, fixture["teacher_path"]
                )

    def test_resume_rejects_changed_execution_and_recovers_atomic_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            _, record = self._persist_complete_companion(fixture)
            partial = initial_manifest(
                fixture["args"],
                full_length=80,
                selection=[fixture["sample"]],
                checkpoint_sha256=file_sha256(fixture["checkpoint"]),
                binding=fixture["binding"],
            )
            validate_resume_manifest(
                partial,
                fixture["args"],
                full_length=80,
                selection=[fixture["sample"]],
                checkpoint_sha256=file_sha256(fixture["checkpoint"]),
                binding=fixture["binding"],
            )
            recovered = recover_completed_sidecar(
                fixture["output_dir"],
                fixture["sample"],
                protocol=fixture["protocol"],
                binding=fixture["binding"],
            )
            self.assertEqual(recovered, record)

            changed_args = copy.copy(fixture["args"])
            changed_args.inference_batch_size = 4
            with self.assertRaisesRegex(ValueError, "inference_batch_size"):
                validate_resume_manifest(
                    partial,
                    changed_args,
                    full_length=80,
                    selection=[fixture["sample"]],
                    checkpoint_sha256=file_sha256(fixture["checkpoint"]),
                    binding=fixture["binding"],
                )

    def test_explicit_resume_removes_only_an_incomplete_per_image_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            incomplete = (
                fixture["output_dir"] / "e0_slide_logits" / "0000_tile.npy"
            )
            incomplete.parent.mkdir(parents=True)
            incomplete.write_bytes(b"interrupted array")
            recovered = recover_completed_sidecar(
                fixture["output_dir"],
                fixture["sample"],
                protocol=fixture["protocol"],
                binding=fixture["binding"],
            )
            self.assertIsNone(recovered)
            self.assertFalse(incomplete.exists())
            self.assertTrue(fixture["teacher_path"].is_file())

    def test_bound_teacher_protocol_cannot_be_redefined_as_caching_normal_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            changed = copy.deepcopy(fixture["teacher_manifest"])
            changed["protocol"]["normal_logits_cached"] = True
            changed["protocol_sha256"] = canonical_json_sha256(changed["protocol"])
            fixture["teacher_path"].write_text(
                json.dumps(changed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sealed teacher protocol"):
                read_bound_teacher_manifest(fixture["teacher_path"])


if __name__ == "__main__":
    unittest.main()
