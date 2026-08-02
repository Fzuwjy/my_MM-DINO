"""Small policy checks for the EarthMiss V1 launcher."""

import random
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.evaluate_earthmiss_missing_v1 import _validate_checkpoint

from scripts.train_earthmiss_missing_v1 import (
    build_run_metadata,
    ground_truth_pixel_counts,
    save_checkpoint,
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
        self.assertEqual(
            metadata["evaluation"],
            {
                "checkpoint_selection_metric": "mIoU",
                "checkpoint_selection_support": "pooled_gt_present",
                "selection_split": "val_city_holdout",
                "expected_val_selection_class_ids": list(range(7)),
                "external_comparison_metric": "official_ever_mIoU",
                "external_comparison_support": "fixed_all_8_classes",
                "external_comparison_split": "test_city_holdout",
                "checkpoint_roles": {
                    "best_sar.pth": "primary_deployment",
                    "best_full.pth": "diagnostic_only",
                },
                "paired_endpoint_rule": "same_checkpoint_and_epoch",
            },
        )

    def test_best_checkpoint_records_its_scientific_role(self):
        model = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        metadata = {"run": "B", "seed": 42}
        best = {"sar": 0.25, "full": 0.5}

        with patch("scripts.train_earthmiss_missing_v1.torch.save") as save:
            save_checkpoint(
                "best_sar.pth",
                model,
                optimizer,
                scheduler,
                10,
                best,
                metadata,
                checkpoint_role="primary_deployment",
                selection_state="sar",
            )

        payload = save.call_args.args[0]
        self.assertEqual(payload["checkpoint_role"], "primary_deployment")
        self.assertEqual(payload["selection_state"], "sar")
        self.assertEqual(payload["selection_metric"], "mIoU")
        self.assertEqual(payload["selection_score"], 0.25)

    def test_external_test_rejects_a_diagnostic_checkpoint_by_default(self):
        checkpoint = {
            "checkpoint_role": "diagnostic_only",
            "protocol": {
                "evaluation": {
                    "checkpoint_selection_support": "pooled_gt_present"
                }
            },
        }
        args = SimpleNamespace(
            split="test", allow_non_primary_test_checkpoint=False
        )

        with self.assertRaisesRegex(ValueError, "primary_deployment"):
            _validate_checkpoint(checkpoint, args)

        checkpoint["checkpoint_role"] = "primary_deployment"
        checkpoint["selection_state"] = "sar"
        _validate_checkpoint(checkpoint, args)

    def test_ground_truth_support_audit_ignores_no_data(self):
        dataset = MagicMock()
        dataset.samples = [
            SimpleNamespace(label_path="first"),
            SimpleNamespace(label_path="second"),
        ]
        dataset._read_label.side_effect = (
            np.array([[0, 1, 8]], dtype=np.int64),
            np.array([[1, 7, 8]], dtype=np.int64),
        )

        self.assertEqual(
            ground_truth_pixel_counts(dataset),
            [1, 2, 0, 0, 0, 0, 0, 1],
        )


if __name__ == "__main__":
    unittest.main()
