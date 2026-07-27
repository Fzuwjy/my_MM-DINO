"""Guards for the public WHU training protocol used by the reproduction branch."""

from pathlib import Path
import unittest

import numpy as np

from scripts.whu_cache_compat import set_dataset_cache_capacity
from scripts.whu_label_dtype_compat import label_to_int64


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
        set_dataset_cache_capacity(dataset, 2)

        self.assertEqual(dataset.cache_size, 2)
        self.assertEqual(dataset.rgb_cache.capacity, 2)
        self.assertEqual(dataset.label_cache.capacity, 2)
        self.assertEqual(dataset.sar_cache.capacity, 2)
        self.assertEqual(
            sentinel_objects,
            tuple(
                cache.cache["sentinel"]
                for cache in (dataset.rgb_cache, dataset.label_cache, dataset.sar_cache)
            ),
        )


if __name__ == "__main__":
    unittest.main()
