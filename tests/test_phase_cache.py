"""Synthetic checks for exact phase-teacher caching and structure masks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.cache_whu_phase_teacher import (
    accumulate_aligned_phase,
    atomic_save_npy,
    bounds_from_slice,
    build_phase_protocol,
    file_sha256,
    phase_arithmetic_mean,
    phase_common_slices,
    recover_completed_sidecar,
    teacher_record,
    validate_cached_record,
)
from scripts.prepare_whu_phase_masks import encode_structure_mask


class PhaseTeacherCacheTest(unittest.TestCase):
    def test_common_bounds_match_formal_8_16_region(self):
        original_slice, shifted = phase_common_slices((2048, 2304))
        self.assertEqual(
            bounds_from_slice((2048, 2304), original_slice),
            {"y_start": 512, "y_stop": 1520, "x_start": 512, "x_stop": 1776},
        )
        self.assertEqual(
            shifted[(8, 8)], (slice(520, 1528), slice(520, 1784))
        )
        self.assertEqual(
            shifted[(16, 16)], (slice(528, 1536), slice(528, 1792))
        )
        protocol = build_phase_protocol()
        self.assertEqual(
            protocol["teacher_phases_dy_dx"],
            [[0, 0], [0, 8], [8, 0], [8, 8]],
        )
        self.assertEqual(protocol["stride_hw"], [341, 341])
        self.assertFalse(protocol["normal_logits_cached"])

    def test_aligned_four_phase_logits_use_arithmetic_mean(self):
        shape = (1100, 1100)
        original_slice, shifted = phase_common_slices(shape)
        score_sum = None
        for value, shift in enumerate(((0, 0), (0, 8), (8, 0), (8, 8)), start=1):
            scores = torch.zeros((1, 2, *shape), dtype=torch.float32)
            region = original_slice if shift == (0, 0) else shifted[shift]
            scores[0, 0, region[0], region[1]] = float(value)
            scores[0, 1, region[0], region[1]] = float(value * 10)
            score_sum = accumulate_aligned_phase(
                score_sum, scores, shift, original_slice, shifted
            )
        teacher = phase_arithmetic_mean(score_sum, 4)
        self.assertEqual(tuple(teacher.shape), (2, 60, 60))
        torch.testing.assert_close(teacher[0], torch.full((60, 60), 2.5))
        torch.testing.assert_close(teacher[1], torch.full((60, 60), 25.0))

    def test_atomic_npy_refuses_overwrite_and_hashes_final_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.npy"
            array = np.arange(12, dtype=np.float16).reshape(3, 2, 2)
            metadata = atomic_save_npy(path, array)
            self.assertEqual(metadata["file_sha256"], file_sha256(path))
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), array)
            with self.assertRaises(FileExistsError):
                atomic_save_npy(path, array)

    def test_sidecar_recovery_verifies_all_three_source_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {}
            for name, payload in (
                ("rgb", b"rgb"),
                ("sar", b"sar"),
                ("label", b"label"),
            ):
                path = root / f"{name}.tif"
                path.write_bytes(payload)
                sources[f"{name}_file"] = str(path.resolve())
            sample = {"index": 0, "sample_name": "tile", **sources}
            label = np.zeros((4, 5), dtype=np.int64)
            teacher = np.zeros((7, 2, 3), dtype=np.float16)
            protocol = build_phase_protocol()
            record = teacher_record(
                root,
                sample,
                label,
                (4, 5),
                {"y_start": 1, "y_stop": 3, "x_start": 1, "x_stop": 4},
                teacher,
                protocol,
                "checkpoint-sha",
            )
            validate_cached_record(root, record, sample, protocol)
            recovered = recover_completed_sidecar(root, sample, protocol)
            self.assertEqual(recovered, record)
            Path(sample["rgb_file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source rgb changed"):
                validate_cached_record(root, record, sample, protocol)

    def test_structure_bits_distinguish_small_and_thin(self):
        label = np.full((340, 340), 7, dtype=np.int64)
        label[10:20, 10:20] = 0  # area 100, thickness 200/36 > 4
        label[30, 20:320] = 1  # area 300, thickness 2
        encoded, stats = encode_structure_mask(label)
        self.assertTrue(np.all(encoded[10:20, 10:20] == 1))
        self.assertTrue(np.all(encoded[30, 20:320] == 2))
        self.assertEqual(encoded[0, 0], 0)
        self.assertEqual(stats["small_pixels"], 100)
        self.assertEqual(stats["thin_pixels"], 300)
        self.assertEqual(stats["small_and_thin_pixels"], 0)
        self.assertEqual(stats["component_algorithm"]["connectivity"], 8)


if __name__ == "__main__":
    unittest.main()
