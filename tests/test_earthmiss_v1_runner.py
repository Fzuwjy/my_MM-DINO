"""Small policy checks for the EarthMiss V1 launcher."""

import random
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.evaluate_earthmiss_missing_v1 import (
    _checkpoint_uses_raw_logits,
    _validate_checkpoint,
)

from scripts.train_earthmiss_missing_v1 import (
    build_run_metadata,
    ground_truth_pixel_counts,
    save_checkpoint,
    train_state,
    update_early_stopping_state,
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
            run="C",
            seed=42,
            window_size=512,
            batch_size=8,
            num_workers=4,
            epochs=50,
            eval_interval=5,
            early_stop_patience_evals=3,
            full_probability=0.5,
            decoder_normalization="batchnorm",
            decoder_groupnorm_groups=32,
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
            metadata["model"],
            {
                "segmentation_head": "raw_conv1x1",
                "released_segmentation_head": "conv_bn_relu",
                "decoder_normalization": "batchnorm",
                "decoder_groupnorm_groups": 32,
                "normalization_scope": "trainable Decoder only; frozen DINO LayerNorm unchanged",
            },
        )
        self.assertEqual(
            metadata["budget"],
            {
                "sampling": "one_crop_per_tile_per_epoch",
                "steps_per_epoch": 331,
                "planned_optimizer_steps": 16550,
                "checkpoint_unit": "epoch",
                "normalization_ratio_arms_share_total_steps": True,
            },
        )
        self.assertEqual(
            metadata["data_loader"],
            {
                "cached_source_arrays_immutable": True,
                "persistent_workers": True,
            },
        )
        self.assertEqual(
            metadata["evaluation"],
            {
                "checkpoint_selection_metric": "fixed_final_epoch",
                "checkpoint_selection_support": "pooled_gt_present",
                "selection_split": "val_city_holdout",
                "expected_val_selection_class_ids": list(range(7)),
                "external_comparison_metric": "official_ever_mIoU",
                "external_comparison_support": "fixed_all_8_classes",
                "external_comparison_split": "test_city_holdout",
                "checkpoint_roles": {
                    "epoch_50.pth": "fixed_final_primary",
                    "best_sar.pth": "val_selection_diagnostic",
                    "best_full.pth": "diagnostic_only",
                },
                "paired_endpoint_rule": "same_checkpoint_and_epoch",
                "val_best_is_diagnostic_only": True,
            },
        )
        self.assertEqual(
            metadata["early_stopping"],
            {
                "selection_state": "sar",
                "strict_improvement": True,
                "patience_evaluations": 3,
                "evaluation_interval_epochs": 5,
                "disabled": False,
            },
        )

    def test_early_stopping_patience_resets_only_on_strict_sar_improvement(self):
        state = {"bad_validation_count": 0, "best_epoch": None}
        state = update_early_stopping_state(state, improved=True, epoch=5)
        self.assertEqual(state, {"bad_validation_count": 0, "best_epoch": 5})

        state = update_early_stopping_state(state, improved=False, epoch=10)
        state = update_early_stopping_state(state, improved=False, epoch=15)
        self.assertEqual(state, {"bad_validation_count": 2, "best_epoch": 5})

        state = update_early_stopping_state(state, improved=True, epoch=20)
        self.assertEqual(state, {"bad_validation_count": 0, "best_epoch": 20})

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

    def test_external_test_accepts_registered_fixed_final_factorial_checkpoint(self):
        checkpoint = {
            "checkpoint_role": "fixed_final_primary",
            "epoch": 50,
            "selection_state": None,
            "protocol": {
                "protocol_revision": "earthmiss_norm_ratio_factorial_v1",
                "epochs": 50,
                "evaluation": {
                    "checkpoint_selection_support": "pooled_gt_present"
                },
            },
        }
        args = SimpleNamespace(
            split="test", allow_non_primary_test_checkpoint=False
        )
        _validate_checkpoint(checkpoint, args)

        checkpoint["epoch"] = 45
        with self.assertRaisesRegex(ValueError, "registered primary"):
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

    def test_checkpoint_head_contract_is_backward_compatible_and_explicit(self):
        released = {"protocol": {}}
        corrected = {
            "protocol": {"model": {"segmentation_head": "raw_conv1x1"}}
        }
        unknown = {
            "protocol": {"model": {"segmentation_head": "mystery"}}
        }

        self.assertFalse(_checkpoint_uses_raw_logits(released))
        self.assertTrue(_checkpoint_uses_raw_logits(corrected))
        with self.assertRaisesRegex(ValueError, "Unknown checkpoint"):
            _checkpoint_uses_raw_logits(unknown)


if __name__ == "__main__":
    unittest.main()
