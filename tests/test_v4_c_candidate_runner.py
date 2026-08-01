"""Candidate-only runner contracts that do not require the WHU dataset."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from scripts.run_whu_v4_c_screen import (
    audit_optimizer_membership,
    audit_shared_initialization,
    audit_step0_probe,
    load_and_validate_clean_reference,
    validate_clean_reference_initialization,
    validate_epoch_against_clean_reference,
)


class _MiniModel(nn.Module):
    def __init__(self, *, with_stem: bool):
        super().__init__()
        self.shared = nn.Linear(3, 4)
        self.decoder = nn.Module()
        if with_stem:
            self.decoder.optical_stem = nn.Linear(2, 2)


class _ProbeModel(nn.Module):
    def forward(self, optical, sar):
        # Deliberately consume RNG to prove that the audit restores it.
        torch.rand(1)
        return optical[:, :1] + sar


def _protocol() -> dict:
    return {
        "variant": "mask-ignore+optical-stem",
        "seed": 42,
        "scheduler_horizon_epochs": 50,
        "stop_after_epoch": 1,
        "evaluation_epochs": [1],
        "model_name": "DINOv3",
        "dataset_name": "WHU",
        "num_modalities": 2,
        "backbone_type": "dinov3_vits16",
        "use_lora": False,
        "train_batch_size_per_gpu": 8,
        "train_workers": 0,
        "inference_batch_size": 32,
        "max_train_batches": 1,
        "max_test_images": 1,
        "scope": "smoke",
        "mask_padding_ignore": True,
        "mask_fill": 7,
        "aux_fill": 0,
    }


class V4CCandidateRunnerTest(unittest.TestCase):
    def test_shared_parameters_and_optimizer_membership_are_per_key(self):
        torch.manual_seed(9)
        clean = _MiniModel(with_stem=False)
        torch.manual_seed(9)
        candidate = _MiniModel(with_stem=True)
        shared = audit_shared_initialization(clean, candidate)
        self.assertTrue(shared["shared_parameters_bitwise_equal"])
        self.assertNotIn("full_model_sha256", shared)
        optimizer = torch.optim.AdamW(candidate.parameters())
        membership = audit_optimizer_membership(candidate, optimizer)
        self.assertTrue(membership["stem_parameters_present_exactly_once"])

    def test_step0_probe_restores_rng_and_prediction_sha(self):
        before = torch.get_rng_state().clone()
        audit = audit_step0_probe(
            _ProbeModel(),
            _ProbeModel(),
            device=torch.device("cpu"),
            size=32,
        )
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(audit["prediction_torch_equal"])
        self.assertTrue(audit["prediction_sha256_equal"])
        self.assertTrue(audit["cpu_rng_state_restored"])

    def test_clean_reference_is_checked_before_training(self):
        candidate = _protocol()
        reference = {
            **candidate,
            "variant": "mask-ignore",
            "initial_model_state_sha256": "sealed",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text(
                json.dumps(reference), encoding="utf-8"
            )
            (root / "train_e1.json").write_text("{}", encoding="utf-8")
            (root / "evaluation_e1.json").write_text("{}", encoding="utf-8")
            loaded = load_and_validate_clean_reference(root, candidate)
        validate_clean_reference_initialization(loaded, "sealed")
        with self.assertRaisesRegex(RuntimeError, "does not reproduce"):
            validate_clean_reference_initialization(loaded, "drift")

    def test_epoch_pairing_checks_full_hash_and_first_trace(self):
        trace = [
            {
                "pair_sha256": "pair",
                "optical_sha256": "rgb",
                "sar_sha256": "sar",
                "raw_label_sha256": "label",
                "normalized_label_sha256": "normalized",
            }
        ]
        training = {
            "paired_data_sha256": "data",
            "raw_label_sha256": "raw",
            "first_batch_trace": trace,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_e1.json"
            path.write_text(json.dumps(training), encoding="utf-8")
            audit = validate_epoch_against_clean_reference(training, path)
            self.assertTrue(audit["first_batch_trace_equal"])
            changed = {**training, "raw_label_sha256": "changed"}
            with self.assertRaisesRegex(RuntimeError, "data stream differs"):
                validate_epoch_against_clean_reference(changed, path)


if __name__ == "__main__":
    unittest.main()
