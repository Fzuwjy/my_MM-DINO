"""Guards for the public WHU training protocol used by the reproduction branch."""

from pathlib import Path
import unittest


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


if __name__ == "__main__":
    unittest.main()
