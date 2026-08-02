"""Small policy checks for the EarthMiss V1 launcher."""

import random
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from scripts.train_earthmiss_missing_v1 import (
    build_run_metadata,
    train_state,
    validation_states,
)


class EarthMissV1RunnerTest(unittest.TestCase):
    def test_run_policies_and_validation_targets(self):
        rng = random.Random(42)
        self.assertEqual(train_state("A", rng), "sar")
        self.assertEqual(train_state("B", rng), "full")
        self.assertEqual(validation_states("A"), ("sar",))
        self.assertEqual(validation_states("B"), ("sar", "full"))
        self.assertEqual(validation_states("C"), ("sar", "full"))

    def test_run_c_uses_only_full_and_sar_and_is_deterministic(self):
        first_rng = random.Random(99)
        second_rng = random.Random(99)
        first = [train_state("C", first_rng) for _ in range(20)]
        second = [train_state("C", second_rng) for _ in range(20)]
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"full", "sar"})

    def test_run_metadata_freezes_normalization_and_optimizer_budget(self):
        args = SimpleNamespace(
            run="C", seed=42, window_size=512, batch_size=8, epochs=50
        )
        train_dataset = MagicMock()
        train_dataset.__len__.return_value = 2641
        train_dataset.rgb_normalization = "dinov3_lvd_imagenet"
        train_dataset.imagenet_mean = (0.485, 0.456, 0.406)
        train_dataset.imagenet_std = (0.229, 0.224, 0.225)
        train_dataset.sar_mean = (63.30051921735858 / 255.0,)
        train_dataset.sar_std = (68.20405016 / 255.0,)
        val_dataset = MagicMock()
        val_dataset.__len__.return_value = 300
        train_loader = MagicMock()
        train_loader.__len__.return_value = 331

        metadata = build_run_metadata(
            args, train_dataset, val_dataset, train_loader
        )

        self.assertEqual(
            metadata["normalization"]["rgb"]["policy"],
            "dinov3_lvd_imagenet",
        )
        self.assertEqual(
            metadata["budget"],
            {
                "sampling": "one_crop_per_tile_per_epoch",
                "steps_per_epoch": 331,
                "planned_optimizer_steps": 16550,
                "checkpoint_unit": "epoch",
            },
        )


if __name__ == "__main__":
    unittest.main()
