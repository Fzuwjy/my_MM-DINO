"""Guards for the public WHU training protocol used by the reproduction branch."""

from pathlib import Path
import unittest

import numpy as np
import torch

from scripts.whu_cache_compat import CACHE_CAPACITY, set_dataset_cache_capacity
from scripts.whu_label_dtype_compat import label_to_int64
from scripts.prepare_faithful_whu import BACKBONE_FILENAMES
from scripts.evaluate_whu_vitl_lora import (
    RELEASED_CONFIG_BATCH_SIZE,
    RELEASED_INFERENCE_BATCH_SIZE,
)
from scripts.run_whu_vitl_lora_accumulated import (
    GradientAccumulationController,
    GradientScaledLoss,
    validate_effective_batch,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FaithfulWhuContractTest(unittest.TestCase):
    def test_official_split_is_80_20_and_disjoint(self):
        split_root = REPO_ROOT / "splits" / "whu"
        train = set((split_root / "official_train.txt").read_text().splitlines())
        test = set((split_root / "official_test.txt").read_text().splitlines())
        train.discard("")
        test.discard("")
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)
        self.assertFalse(train & test)

    def test_released_config_values_are_unchanged(self):
        config = (
            REPO_ROOT / "tasks" / "segmentation" / "configs" / "MMDINO.py"
        ).read_text(encoding="utf-8")
        self.assertIn("base_lr = 1e-4", config)
        self.assertIn("batch_size = 8", config)
        self.assertIn("epochs = 50", config)
        self.assertIn("window_size = (512, 512)", config)
        self.assertIn("weight_decay=0.01", config)
        self.assertIn("eta_min=1e-7", config)

    def test_released_training_selection_is_unchanged(self):
        trainer = (
            REPO_ROOT / "tasks" / "segmentation" / "train_multi.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"test",', trainer)
        self.assertIn("find_unused_parameters=True", trainer)
        self.assertIn("save_interval = 5", trainer)
        self.assertIn("if mIoU > best_IoU:", trainer)
        self.assertIn('batch_size=cfg.get("batch_size", 4) * 4', trainer)
        self.assertNotIn("autocast", trainer)
        self.assertNotIn("GradScaler", trainer)
        self.assertNotIn("grad_accum", trainer)

    def test_probe_backbones_match_released_config(self):
        self.assertEqual(
            BACKBONE_FILENAMES["dinov3_vits16"],
            "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        )
        self.assertEqual(
            BACKBONE_FILENAMES["dinov3_vitl16"],
            "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        )

    def test_vitl_lora_evaluator_uses_training_time_inference_batch(self):
        self.assertEqual(RELEASED_CONFIG_BATCH_SIZE, 8)
        self.assertEqual(RELEASED_INFERENCE_BATCH_SIZE, 32)
        evaluator = (
            REPO_ROOT / "scripts" / "evaluate_whu_vitl_lora.py"
        ).read_text(encoding="utf-8")
        self.assertIn("official_trainer.test(", evaluator)
        self.assertNotIn("metrics_print_version", evaluator)

    def test_accumulation_keeps_released_effective_batch(self):
        validate_effective_batch(4, 2)
        with self.assertRaises(ValueError):
            validate_effective_batch(4, 1)

    def test_accumulation_gates_optimizer_calls(self):
        class Optimizer:
            def __init__(self):
                self.zero_grad_calls = 0
                self.step_calls = 0

            def zero_grad(self):
                self.zero_grad_calls += 1

            def step(self):
                self.step_calls += 1

        optimizer = Optimizer()
        controller = GradientAccumulationController(optimizer, 2)
        for _ in range(4):
            optimizer.zero_grad()
            optimizer.step()

        self.assertEqual(optimizer.zero_grad_calls, 2)
        self.assertEqual(optimizer.step_calls, 2)
        self.assertEqual(controller.micro_steps, 4)
        self.assertEqual(controller.optimizer_steps, 2)

    def test_accumulation_scales_gradient_not_reported_loss(self):
        value = torch.tensor(2.0, requires_grad=True)
        loss_fn = GradientScaledLoss(lambda item: item.square(), 2)
        loss = loss_fn(value)
        self.assertEqual(float(loss.detach()), 4.0)
        loss.backward()
        self.assertEqual(float(value.grad), 2.0)

    def test_accumulation_preserves_one_adamw_step_per_effective_batch(self):
        value = torch.nn.Parameter(torch.tensor(2.0))
        optimizer = torch.optim.AdamW([value], lr=0.1, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        controller = GradientAccumulationController(optimizer, 2)
        loss_fn = GradientScaledLoss(lambda item: item.square(), 2)

        for _ in range(2):
            optimizer.zero_grad()
            loss_fn(value).backward()
            optimizer.step()
        scheduler.step()

        self.assertEqual(controller.micro_steps, 2)
        self.assertEqual(controller.optimizer_steps, 1)
        self.assertEqual(int(optimizer.state[value]["step"]), 1)

    def test_whu_dtype_compat_changes_dtype_not_values(self):
        label = np.array([[0, 1, 6, 7]], dtype=np.int32)
        converted = label_to_int64(label)
        self.assertEqual(converted.dtype, np.int64)
        np.testing.assert_array_equal(converted, label)

        dataset_source = (
            REPO_ROOT / "tasks" / "segmentation" / "datasets" / "WHU_dataset.py"
        ).read_text(encoding="utf-8")
        self.assertIn("label = label.astype(np.int32)", dataset_source)

    def test_whu_cache_compat_changes_only_capacities(self):
        class Cache:
            def __init__(self):
                self.capacity = 100
                self.cache = {"sentinel": object()}

        class Dataset:
            cache_size = 100
            rgb_cache = Cache()
            label_cache = Cache()
            sar_cache = Cache()

        dataset = Dataset()
        sentinel_objects = tuple(
            cache.cache["sentinel"]
            for cache in (dataset.rgb_cache, dataset.label_cache, dataset.sar_cache)
        )
        set_dataset_cache_capacity(dataset, CACHE_CAPACITY)

        self.assertEqual(dataset.cache_size, CACHE_CAPACITY)
        self.assertEqual(dataset.rgb_cache.capacity, CACHE_CAPACITY)
        self.assertEqual(dataset.label_cache.capacity, CACHE_CAPACITY)
        self.assertEqual(dataset.sar_cache.capacity, CACHE_CAPACITY)
        self.assertEqual(
            sentinel_objects,
            tuple(
                cache.cache["sentinel"]
                for cache in (dataset.rgb_cache, dataset.label_cache, dataset.sar_cache)
            ),
        )


if __name__ == "__main__":
    unittest.main()
