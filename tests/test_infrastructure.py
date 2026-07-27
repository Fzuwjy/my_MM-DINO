from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import metrics_from_confusion_matrix, update_confusion_matrix
from utils.runtime import (
    DistributedEvalSampler,
    atomic_torch_save,
    checkpoint_payload,
    create_grad_scaler,
    load_training_checkpoint,
)
from scripts.make_grouped_split import choose_validation_groups
from scripts.preflight import (
    architecture_is_supported,
    numeric_version,
    parse_args as parse_preflight_args,
    sanitize_thread_environment,
)
from datasets import build_dataset


class InfrastructureTests(unittest.TestCase):
    def test_preflight_uses_runtime_root_environment_variables(self):
        with patch.dict(
            os.environ,
            {
                "MM_DINO_DATASETS_ROOT": "/runtime/datasets",
                "MM_DINO_WEIGHTS_ROOT": "/runtime/weights",
            },
        ):
            args = parse_preflight_args([])
        self.assertEqual(args.datasets_root, "/runtime/datasets")
        self.assertEqual(args.weights_root, "/runtime/weights")

    def test_preflight_sanitizes_invalid_thread_count(self):
        warnings = []
        with patch.dict(os.environ, {"OMP_NUM_THREADS": ""}):
            sanitize_thread_environment(warnings)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
        self.assertEqual(len(warnings), 1)

    def test_preflight_parses_pytorch_local_version(self):
        self.assertEqual(numeric_version("2.7.1+cu128"), (2, 7, 1))
        self.assertEqual(numeric_version("12.8"), (12, 8, 0))
        self.assertEqual(numeric_version("unknown"), ())

    def test_preflight_accepts_native_or_ptx_gpu_architecture(self):
        self.assertTrue(architecture_is_supported({"sm_120"}, (12, 0)))
        self.assertTrue(architecture_is_supported({"compute_120"}, (12, 0)))
        self.assertFalse(architecture_is_supported({"sm_90"}, (12, 0)))

    def test_distributed_eval_sampler_has_no_duplicates(self):
        dataset = list(range(11))
        partitions = [
            list(DistributedEvalSampler(dataset, num_replicas=3, rank=rank))
            for rank in range(3)
        ]
        flattened = [index for partition in partitions for index in partition]
        self.assertEqual(sorted(flattened), list(range(len(dataset))))
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_incremental_confusion_matrix_ignores_out_of_range_targets(self):
        confusion = np.zeros((3, 3), dtype=np.int64)
        prediction = np.array([[0, 2, 1, 1]])
        target = np.array([[0, 1, 2, 3]])  # class 3 is the ignore index
        update_confusion_matrix(confusion, prediction, target)
        expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(confusion, expected)
        metrics = metrics_from_confusion_matrix(confusion, ["a", "b", "c"])
        self.assertTrue(all(np.isfinite(value) for value in metrics))

    def test_atomic_torch_save_produces_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "checkpoint.pth"
            atomic_torch_save({"value": torch.tensor([1, 2, 3])}, path)
            loaded = torch.load(path, weights_only=False)
            torch.testing.assert_close(loaded["value"], torch.tensor([1, 2, 3]))
            self.assertFalse(path.with_suffix(".pth.tmp").exists())

    def test_training_checkpoint_restores_state_and_epoch(self):
        device = torch.device("cpu")
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        scaler = create_grad_scaler(device, "none")
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "resume.pth"
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch=2,
                    best_miou=0.5,
                    args={},
                ),
                path,
            )
            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
            restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                restored_optimizer, T_max=5
            )
            start_epoch, best_miou, saved_args = load_training_checkpoint(
                path,
                restored_model,
                restored_optimizer,
                restored_scheduler,
                create_grad_scaler(device, "none"),
                device,
            )
            self.assertEqual(start_epoch, 3)
            self.assertEqual(best_miou, 0.5)
            self.assertEqual(saved_args, {})
            for expected, actual in zip(model.parameters(), restored_model.parameters()):
                torch.testing.assert_close(expected, actual)

    def test_grouped_split_hits_target_without_partial_groups(self):
        sizes = {"a": 5, "b": 4, "c": 3, "d": 2}
        selected = choose_validation_groups(sizes, target=7, seed=42)
        self.assertEqual(sum(sizes[group] for group in selected), 7)
        self.assertTrue(selected.issubset(sizes))

    def test_committed_whu_official_split_is_disjoint(self):
        split_root = REPO_ROOT / "splits" / "whu"
        train = (split_root / "official_train.txt").read_text(encoding="utf-8").splitlines()
        test = (split_root / "official_test.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)
        self.assertEqual(len(train), len(set(train)))
        self.assertEqual(len(test), len(set(test)))
        self.assertTrue(set(train).isdisjoint(test))

    def test_whu_dataset_factory_forwards_bounded_cache(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            datasets_root = Path(temporary_dir)
            dataset_root = datasets_root / "whu-opt-sar"
            dataset_root.mkdir()
            (dataset_root / "train_list.txt").write_text("sample.tif\n", encoding="utf-8")
            with patch("datasets.WHU_Dataset", side_effect=lambda **kwargs: kwargs):
                options = build_dataset(
                    "WHU",
                    "train",
                    datasets_root=datasets_root,
                    cache_size=2,
                    modality="multi",
                )
            self.assertEqual(options["cache_size"], 2)
            self.assertTrue(options["sar_dir"].endswith(str(Path("sar") / "{}")))


if __name__ == "__main__":
    unittest.main()
