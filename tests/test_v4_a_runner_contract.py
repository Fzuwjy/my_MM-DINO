"""Pure unit tests for the V4-A runner and paired decision contract."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.compare_whu_v4_a_runs import (
    decision,
    paired_bootstrap_ci,
    validate_protocol_pair,
)
from scripts.run_whu_v4_a_screen import (
    PROTOCOL_EPOCHS,
    batch_pair_fingerprint,
    scientific_protocol,
)


class V4ARunnerContractTest(unittest.TestCase):
    def _args(self, variant: str) -> SimpleNamespace:
        return SimpleNamespace(
            variant=variant,
            seed=42,
            stop_after_epoch=15,
            evaluation_epochs=(5, 10, 15),
            num_workers=4,
            inference_batch_size=32,
            max_train_batches=None,
            max_test_images=None,
            smoke=False,
        )

    def test_pair_hash_ignores_only_the_intentional_mask_fill_change(self):
        optical = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
        sar = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        official = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.long)
        mask_ignore = torch.tensor([[[7, 1], [2, 7]]], dtype=torch.long)
        self.assertEqual(
            batch_pair_fingerprint(optical, sar, official),
            batch_pair_fingerprint(optical, sar, mask_ignore),
        )

        changed_geometry = optical.clone()
        changed_geometry[0, 0, 0, 0] += 1
        self.assertNotEqual(
            batch_pair_fingerprint(optical, sar, official),
            batch_pair_fingerprint(changed_geometry, sar, official),
        )

    def test_protocol_keeps_fifty_epoch_scheduler_horizon(self):
        official = scientific_protocol(self._args("official"))
        candidate = scientific_protocol(self._args("mask-ignore"))
        self.assertEqual(official["scheduler_horizon_epochs"], PROTOCOL_EPOCHS)
        self.assertEqual(candidate["scheduler_horizon_epochs"], PROTOCOL_EPOCHS)
        self.assertFalse(official["mask_padding_ignore"])
        self.assertTrue(candidate["mask_padding_ignore"])
        self.assertEqual(official["aux_fill"], 0)
        self.assertEqual(candidate["aux_fill"], 0)

    def test_protocol_pair_rejects_initialization_or_seed_drift(self):
        official = scientific_protocol(self._args("official"))
        candidate = scientific_protocol(self._args("mask-ignore"))
        shared = {
            "git_commit": "abc",
            "initial_model_state_sha256": "sealed",
            "train_dataset_length": 3200,
            "full_test_length": 20,
            "evaluated_test_length": 20,
        }
        official.update(shared)
        candidate.update(shared)
        validate_protocol_pair(official, candidate)
        candidate["seed"] = 43
        with self.assertRaisesRegex(RuntimeError, "protocol differs"):
            validate_protocol_pair(official, candidate)

    def test_e15_practical_gate_does_not_promote_tiny_positive(self):
        weak = decision(
            "formal-screen",
            [
                {
                    "epoch": 10,
                    "candidate_minus_official_pp": 0.01,
                    "raw_label_stream_changed": True,
                },
                {
                    "epoch": 15,
                    "candidate_minus_official_pp": 0.02,
                    "raw_label_stream_changed": True,
                },
            ],
        )
        self.assertEqual(weak["outcome"], "STOP_A_LOW_OR_FLAT_GAIN")

        useful = decision(
            "formal-screen",
            [
                {
                    "epoch": 10,
                    "candidate_minus_official_pp": 0.03,
                    "raw_label_stream_changed": True,
                },
                {
                    "epoch": 15,
                    "candidate_minus_official_pp": 0.06,
                    "raw_label_stream_changed": True,
                },
            ],
        )
        self.assertEqual(useful["outcome"], "PASS_A_E15_EXTEND_TO_E30")

    def test_bootstrap_recomputes_pooled_confusion(self):
        official = np.stack(
            [np.eye(7, dtype=np.int64) * 8, np.eye(7, dtype=np.int64) * 9]
        )
        candidate = official.copy()
        official[0, 0, 0] -= 2
        official[0, 0, 1] += 2
        official[1, 1, 1] -= 2
        official[1, 1, 0] += 2
        result = paired_bootstrap_ci(
            official, candidate, replicates=100, seed=7
        )
        self.assertGreater(result["low_pp"], 0.0)
        self.assertEqual(result["replicates"], 100)


if __name__ == "__main__":
    unittest.main()
