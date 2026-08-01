"""Pure unit contracts for the V4-C E15->E30 restart runner."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from scripts.run_whu_v4_c_e30_continuation import (
    ARTIFACT_TYPE,
    CANDIDATE_VARIANT,
    CLEAN_VARIANT,
    CONTINUATION_MODE,
    load_sealed_source,
    load_training_state,
    restore_rng_state,
    rng_state_fingerprints,
    validate_completed_clean_continuation,
    validate_epoch_pair,
    validate_source_protocol,
)
from scripts.run_whu_v4_a_screen import file_sha256


def _source_protocol(variant: str = CLEAN_VARIANT) -> dict:
    protocol = {
        "variant": variant,
        "model_name": "DINOv3",
        "dataset_name": "WHU",
        "num_modalities": 2,
        "backbone_type": "dinov3_vits16",
        "use_lora": False,
        "seed": 42,
        "scheduler_horizon_epochs": 50,
        "stop_after_epoch": 15,
        "evaluation_epochs": [5, 10, 15],
        "mask_padding_ignore": True,
        "mask_fill": 7,
        "aux_fill": 0,
        "train_batch_size_per_gpu": 8,
        "train_workers": 4,
        "inference_batch_size": 32,
        "max_train_batches": None,
        "max_test_images": None,
        "scope": "formal-screen",
        "git_commit": "sealed-source-commit",
    }
    if variant == CANDIDATE_VARIANT:
        protocol.update(
            {
                "use_optical_stem": True,
                "optical_stem_location": (
                    "post-ACFM-L0-pre-FRN-single-injection"
                ),
                "optical_stem_seed": 104771,
                "clean_baseline_variant": CLEAN_VARIANT,
            }
        )
    return protocol


def _rng_state(seed: int = 19) -> dict:
    previous_python = random.getstate()
    previous_numpy = np.random.get_state()
    previous_cpu = torch.get_rng_state().clone()
    previous_cuda = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        generator = torch.Generator().manual_seed(seed + 1)
        torch.rand(7, generator=generator)
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else []
            ),
            "loader_generator": generator.get_state().clone(),
        }
    finally:
        random.setstate(previous_python)
        np.random.set_state(previous_numpy)
        torch.set_rng_state(previous_cpu)
        if previous_cuda:
            torch.cuda.set_rng_state_all(previous_cuda)


def _write_sealed_source(
    root: Path,
    *,
    protocol: dict | None = None,
    checkpoint_updates: dict | None = None,
    directory_name: str = "source",
) -> tuple[Path, dict]:
    source = root / directory_name
    source.mkdir()
    protocol = deepcopy(protocol or _source_protocol())
    (source / "protocol.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    evaluation = source / "evaluation_e15.json"
    evaluation.write_text(json.dumps({"epoch": 15}), encoding="utf-8")
    checkpoint = {
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {}, "param_groups": []},
        "scheduler": {"T_max": 50, "last_epoch": 15},
        "epoch": 15,
        "protocol": deepcopy(protocol),
        "rng_state": _rng_state(),
    }
    if checkpoint_updates:
        checkpoint.update(deepcopy(checkpoint_updates))
    checkpoint_path = source / "checkpoint_e15.pth"
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "status": "PASS",
        "variant": protocol["variant"],
        "evaluations": [
            {
                "epoch": 15,
                "checkpoint_path": checkpoint_path.name,
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "evaluation_sha256": file_sha256(evaluation),
            }
        ],
    }
    (source / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return source, summary


def _trace() -> list[dict]:
    return [
        {
            "pair_sha256": "pair",
            "optical_sha256": "optical",
            "sar_sha256": "sar",
            "raw_label_sha256": "raw-label",
            "normalized_label_sha256": "normalized-label",
        }
    ]


def _pair_protocol(variant: str) -> dict:
    return {
        "artifact_type": ARTIFACT_TYPE,
        "continuation_mode": CONTINUATION_MODE,
        "variant": variant,
        "source_epoch": 15,
        "target_epoch": 30,
        "stop_after_epoch": 16,
        "evaluation_epochs": [16],
        "not_equivalent_to_uninterrupted": True,
        "seed": 42,
        "scheduler_horizon_epochs": 50,
        "train_batch_size_per_gpu": 8,
        "train_workers": 0,
        "inference_batch_size": 32,
        "max_train_batches": 1,
        "max_test_images": 1,
        "scope": "smoke",
        "git_commit": "same-continuation-commit",
    }


def _one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    x: torch.Tensor,
    target: torch.Tensor,
) -> None:
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
    scheduler.step()


class V4CE30ContinuationTest(unittest.TestCase):
    def test_rng_fingerprint_and_restore_cover_every_saved_state(self):
        sealed = _rng_state(31)
        expected = rng_state_fingerprints(sealed)
        self.assertEqual(len(expected["combined_sha256"]), 64)
        loader_generator = torch.Generator().manual_seed(999)

        previous_python = random.getstate()
        previous_numpy = np.random.get_state()
        previous_cpu = torch.get_rng_state().clone()
        previous_cuda = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        )
        try:
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            restored = restore_rng_state(sealed, loader_generator)
            self.assertEqual(restored, expected)
            self.assertEqual(
                rng_state_fingerprints(
                    {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all(),
                        "loader_generator": loader_generator.get_state(),
                    }
                ),
                expected,
            )
        finally:
            random.setstate(previous_python)
            np.random.set_state(previous_numpy)
            torch.set_rng_state(previous_cpu)
            if previous_cuda:
                torch.cuda.set_rng_state_all(previous_cuda)

        missing = deepcopy(sealed)
        del missing["loader_generator"]
        with self.assertRaisesRegex(RuntimeError, "lacks RNG state"):
            rng_state_fingerprints(missing)
        wrong_cuda = deepcopy(sealed)
        wrong_cuda["torch_cuda"] = sealed["torch_cpu"]
        with self.assertRaisesRegex(TypeError, "must be a list"):
            rng_state_fingerprints(wrong_cuda)

    def test_source_protocol_and_checkpoint_seals_reject_drift(self):
        validate_source_protocol(
            _source_protocol(), variant=CLEAN_VARIANT, smoke=False
        )
        candidate = _source_protocol(CANDIDATE_VARIANT)
        validate_source_protocol(
            candidate, variant=CANDIDATE_VARIANT, smoke=False
        )

        drift = deepcopy(candidate)
        drift["optical_stem_seed"] += 1
        with self.assertRaisesRegex(RuntimeError, "protocol differs"):
            validate_source_protocol(
                drift, variant=CANDIDATE_VARIANT, smoke=False
            )

        with tempfile.TemporaryDirectory() as directory:
            source, _ = _write_sealed_source(Path(directory))
            loaded = load_sealed_source(
                source, variant=CLEAN_VARIANT, smoke=False
            )
            self.assertEqual(loaded[1]["epoch"], 15)
            with (source / "checkpoint_e15.pth").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "SHA256 differs"):
                load_sealed_source(source, variant=CLEAN_VARIANT, smoke=False)

        bad_checkpoints = (
            ({"epoch": 14}, "epoch is not E15"),
            (
                {"scheduler": {"T_max": 30, "last_epoch": 15}},
                "scheduler is not Cosine",
            ),
            ({"rng_state": {}}, "lacks RNG state"),
        )
        for updates, message in bad_checkpoints:
            with self.subTest(updates=updates):
                with tempfile.TemporaryDirectory() as directory:
                    source, _ = _write_sealed_source(
                        Path(directory), checkpoint_updates=updates
                    )
                    with self.assertRaisesRegex(RuntimeError, message):
                        load_sealed_source(
                            source, variant=CLEAN_VARIANT, smoke=False
                        )

    def test_candidate_source_binds_the_exact_clean_e15_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean, _ = _write_sealed_source(
                root, directory_name="clean-source"
            )
            candidate_protocol = _source_protocol(CANDIDATE_VARIANT)
            candidate_protocol.update(
                {
                    "clean_reference_dir": str(clean.resolve()),
                    "clean_reference_protocol_sha256": file_sha256(
                        clean / "protocol.json"
                    ),
                }
            )
            candidate, _ = _write_sealed_source(
                root,
                protocol=candidate_protocol,
                directory_name="candidate-source",
            )
            comparison = {
                "status": "PASS",
                "artifact_type": "whu_v4_c_sealed_baseline_comparison",
                "outcome": "PASS_C_E15_EXTEND_TO_E30",
                "scope": "formal-screen",
                "candidate_git_commit": candidate_protocol["git_commit"],
                "sealed_v4_a_git_commit": _source_protocol()["git_commit"],
                "seed": 42,
                "baseline_bindings": {
                    "clean_dir": str(clean.resolve()),
                    "candidate_dir": str(candidate.resolve()),
                },
            }
            comparison_path = candidate / "comparison.json"
            comparison_path.write_text(
                json.dumps(comparison), encoding="utf-8"
            )
            loaded = load_sealed_source(
                candidate, variant=CANDIDATE_VARIANT, smoke=False
            )
            lineage = loaded[4]["candidate_clean_lineage"]
            self.assertEqual(
                lineage["clean_reference_checkpoint_sha256"],
                file_sha256(clean / "checkpoint_e15.pth"),
            )

            comparison["baseline_bindings"]["clean_dir"] = str(candidate)
            comparison_path.write_text(
                json.dumps(comparison), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "bindings changed"):
                load_sealed_source(
                    candidate, variant=CANDIDATE_VARIANT, smoke=False
                )

    def test_clean_candidate_pairing_rejects_rng_hash_or_epoch_data_drift(self):
        fingerprints = rng_state_fingerprints(_rng_state(7))
        candidate_protocol = _pair_protocol(CANDIDATE_VARIANT)
        lineage = {
            "clean_reference_dir": "/sealed/clean",
            "clean_reference_protocol_sha256": "clean-protocol-sha",
            "clean_reference_git_commit": "clean-source-commit",
            "clean_reference_checkpoint_sha256": "clean-checkpoint-sha",
            "clean_reference_evaluation_sha256": "clean-evaluation-sha",
        }
        with tempfile.TemporaryDirectory() as directory:
            clean = Path(directory) / "clean"
            clean.mkdir()
            clean_protocol = _pair_protocol(CLEAN_VARIANT)
            clean_protocol["restart_rng_fingerprints"] = fingerprints
            clean_protocol.update(
                {
                    "source_dir_resolved": lineage["clean_reference_dir"],
                    "source_protocol_sha256": lineage[
                        "clean_reference_protocol_sha256"
                    ],
                    "source_git_commit": lineage[
                        "clean_reference_git_commit"
                    ],
                    "source_checkpoint_sha256": lineage[
                        "clean_reference_checkpoint_sha256"
                    ],
                    "source_evaluation_sha256": lineage[
                        "clean_reference_evaluation_sha256"
                    ],
                }
            )
            (clean / "protocol.json").write_text(
                json.dumps(clean_protocol), encoding="utf-8"
            )
            (clean / "summary.json").write_text(
                json.dumps({"status": "PASS"}), encoding="utf-8"
            )
            training = {
                "paired_data_sha256": "paired",
                "raw_label_sha256": "raw",
                "first_batch_trace": _trace(),
            }
            (clean / "train_e16.json").write_text(
                json.dumps(training), encoding="utf-8"
            )
            (clean / "evaluation_e16.json").write_text("{}", encoding="utf-8")

            validate_completed_clean_continuation(
                clean,
                protocol=candidate_protocol,
                restart_rng_fingerprints=fingerprints,
                candidate_clean_lineage=lineage,
            )
            validate_epoch_pair(
                training,
                clean / "train_e16.json",
                expected_trace_count=1,
            )

            changed_rng = deepcopy(fingerprints)
            changed_rng["combined_sha256"] = "different"
            with self.assertRaisesRegex(RuntimeError, "RNG states differ"):
                validate_completed_clean_continuation(
                    clean,
                    protocol=candidate_protocol,
                    restart_rng_fingerprints=changed_rng,
                    candidate_clean_lineage=lineage,
                )

            changed_lineage = deepcopy(lineage)
            changed_lineage["clean_reference_checkpoint_sha256"] = "different"
            with self.assertRaisesRegex(RuntimeError, "sealed clean lineage"):
                validate_completed_clean_continuation(
                    clean,
                    protocol=candidate_protocol,
                    restart_rng_fingerprints=fingerprints,
                    candidate_clean_lineage=changed_lineage,
                )

            changed_training = deepcopy(training)
            changed_training["first_batch_trace"][0]["sar_sha256"] = "different"
            with self.assertRaisesRegex(RuntimeError, "pairing differs"):
                validate_epoch_pair(
                    changed_training,
                    clean / "train_e16.json",
                    expected_trace_count=1,
                )

            with self.assertRaisesRegex(RuntimeError, "pairing differs"):
                validate_epoch_pair(
                    training,
                    clean / "train_e16.json",
                    expected_trace_count=10,
                )

    def test_e15_optimizer_scheduler_restore_has_no_off_by_one(self):
        torch.manual_seed(123)
        reference = nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-7
        )
        x = torch.tensor([[0.2, -0.3, 0.7], [0.8, 0.1, -0.4]])
        target = torch.tensor([[0.4, -0.2], [0.3, 0.6]])
        for _ in range(15):
            _one_step(reference, optimizer, scheduler, x, target)
        self.assertEqual(scheduler.last_epoch, 15)

        checkpoint = {
            "model": deepcopy(reference.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict()),
            "scheduler": deepcopy(scheduler.state_dict()),
        }
        torch.manual_seed(999)
        resumed = nn.Linear(3, 2)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=9e-3)
        resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            resumed_optimizer, T_max=50, eta_min=1e-7
        )
        audit = load_training_state(
            model=resumed,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            checkpoint=checkpoint,
            device=torch.device("cpu"),
        )
        self.assertEqual(audit["scheduler_last_epoch"], 15)
        self.assertEqual(
            audit["learning_rates"],
            [float(group["lr"]) for group in optimizer.param_groups],
        )

        # The first resumed update is E16; the scheduler advances exactly once.
        _one_step(reference, optimizer, scheduler, x, target)
        _one_step(resumed, resumed_optimizer, resumed_scheduler, x, target)
        self.assertEqual(scheduler.last_epoch, 16)
        self.assertEqual(resumed_scheduler.last_epoch, 16)
        self.assertEqual(
            scheduler.get_last_lr(), resumed_scheduler.get_last_lr()
        )
        for expected, actual in zip(
            reference.parameters(), resumed.parameters(), strict=True
        ):
            self.assertTrue(torch.equal(expected, actual))

        for field, value in (("last_epoch", 14), ("T_max", 30)):
            bad = deepcopy(checkpoint)
            bad["scheduler"][field] = value
            model = nn.Linear(3, 2)
            bad_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            bad_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                bad_optimizer, T_max=50, eta_min=1e-7
            )
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "off-by-one|horizon"):
                    load_training_state(
                        model=model,
                        optimizer=bad_optimizer,
                        scheduler=bad_scheduler,
                        checkpoint=bad,
                        device=torch.device("cpu"),
                    )


if __name__ == "__main__":
    unittest.main()
