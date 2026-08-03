"""CPU contract tests for fixed-checkpoint EarthMiss V2 BN calibration."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from scripts.evaluate_earthmiss_v2_bn_calibration import (
    DEFAULT_CALIBRATION_STATES,
    EXPECTED_VAL_TILES,
    OFFICIAL_VAL_CITIES,
    build_deployment_composition,
    build_calibration_loader,
    build_evaluation_plan,
    calibrate_decoder_bn,
    decoder_batchnorm_modules,
    evaluate_val_by_city,
    learned_parameter_sha256,
    load_decoder_bn_buffers,
    snapshot_decoder_bn_buffers,
    source_checkpoint_selection_metadata,
    validate_output_paths,
)


class _ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.projection = nn.Conv2d(1, 2, kernel_size=1, bias=False)

    def forward(self, *slots):
        # Canonical MM-DINO keeps two Decoder slots even when one sensor is
        # unavailable.  The shared FRM BN is therefore called twice.
        calibrated = [self.bn(value) for value in slots]
        return self.projection(sum(calibrated) / len(calibrated))


class _ToySegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        # These standard BNs are deliberately outside Decoder and must stay in
        # eval mode with unchanged running buffers during calibration.
        self.backbone_bn = nn.BatchNorm2d(1, eps=1e-12)
        self.adapter_bn = nn.BatchNorm2d(1, eps=1e-12)
        self.decoder = _ToyDecoder()
        self.scalar = nn.Parameter(torch.tensor(1.0))

    def forward(self, rgb, sar, availability=None):
        if availability is None:
            value = (rgb + sar) / 2.0
        elif bool(availability[0, 0]) and bool(availability[0, 1]):
            value = (rgb + sar) / 2.0
        elif bool(availability[0, 0]):
            value = rgb
        else:
            value = sar
        value = self.backbone_bn(value)
        value = self.adapter_bn(value)
        return self.decoder(value, value) * self.scalar


class _ConstantPairDataset(torch.utils.data.Dataset):
    def __init__(self, sar_values):
        self.sar_values = list(sar_values)

    def __len__(self):
        return len(self.sar_values)

    def __getitem__(self, index):
        rgb = torch.full((1, 2, 2), 10.0 + index)
        sar = torch.full((1, 2, 2), float(self.sar_values[index]))
        label = torch.zeros((2, 2), dtype=torch.int64)
        return rgb, sar, label


class _RandomCropProxyDataset(torch.utils.data.Dataset):
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        # EarthMiss crops use Python's random module inside __getitem__.
        crop_draw = random.random()
        rgb = torch.full((1, 1, 1), crop_draw)
        sar = torch.full((1, 1, 1), float(index))
        label = torch.tensor([[index]], dtype=torch.int64)
        return rgb, sar, label


class _CityDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.predictions = (
            torch.tensor([[0, 1], [0, 1]]),
            torch.ones((2, 2), dtype=torch.int64),
            torch.ones((2, 2), dtype=torch.int64),
        )
        self.labels = (
            torch.tensor([[0, 1], [0, 1]]),
            torch.zeros((2, 2), dtype=torch.int64),
            torch.ones((2, 2), dtype=torch.int64),
        )

    def __len__(self):
        return 3

    def __getitem__(self, index):
        rgb = self.predictions[index].to(torch.float32).unsqueeze(0)
        sar = torch.zeros_like(rgb)
        return rgb, sar, self.labels[index]


def _prediction_from_rgb(
    model,
    rgb,
    sar,
    state,
    *,
    device,
    window_size,
    inference_batch_size,
):
    del model, sar, state, device, window_size, inference_batch_size
    prediction = rgb[:, 0].to(torch.int64)
    logits = torch.full(
        (prediction.shape[0], 8, *prediction.shape[-2:]),
        -10.0,
        dtype=torch.float32,
    )
    return logits.scatter_(1, prediction.unsqueeze(1), 10.0)


class EarthMissV2BNCalibrationTest(unittest.TestCase):
    def test_hash_excludes_buffers_and_supports_scalar_parameters(self):
        model = _ToySegmenter()
        original = learned_parameter_sha256(model)
        model.decoder.bn.running_mean.add_(123.0)
        self.assertEqual(learned_parameter_sha256(model), original)
        with torch.no_grad():
            model.scalar.add_(1.0)
        self.assertNotEqual(learned_parameter_sha256(model), original)

    def test_only_decoder_standard_bn_is_selected(self):
        model = _ToySegmenter()
        modules = decoder_batchnorm_modules(model, expected_count=1)
        self.assertEqual(list(modules), ["decoder.bn"])
        self.assertIs(modules["decoder.bn"], model.decoder.bn)

        model.decoder.bn = nn.SyncBatchNorm(1)
        with self.assertRaisesRegex(TypeError, "standard nn.BatchNorm2d"):
            decoder_batchnorm_modules(model)

    def test_calibration_updates_only_decoder_buffers_and_no_parameters(self):
        model = _ToySegmenter()
        loader = torch.utils.data.DataLoader(
            _ConstantPairDataset([1.0, 1.0, 3.0, 3.0]),
            batch_size=2,
            shuffle=False,
        )
        outside_before = {
            "backbone": model.backbone_bn.running_mean.clone(),
            "adapter": model.adapter_bn.running_mean.clone(),
        }
        parameter_hash = learned_parameter_sha256(model)

        calibrated = calibrate_decoder_bn(
            model,
            loader,
            state="sar",
            device="cpu",
            expected_batches=2,
            expected_tiles=4,
            expected_bn_count=1,
            expected_updates_per_bn=4,
        )

        self.assertEqual(calibrated["logical_batches"], 2)
        self.assertEqual(calibrated["tiles"], 4)
        self.assertEqual(calibrated["canonical_decoder_slots_per_logical_batch"], 2)
        self.assertEqual(calibrated["bn_updates_per_module"], 4)
        self.assertAlmostEqual(
            calibrated["buffers"]["decoder.bn"]["running_mean"].item(),
            2.0,
            places=5,
        )
        self.assertEqual(
            calibrated["buffers"]["decoder.bn"]["num_batches_tracked"].item(),
            4,
        )
        self.assertTrue(torch.equal(model.backbone_bn.running_mean, outside_before["backbone"]))
        self.assertTrue(torch.equal(model.adapter_bn.running_mean, outside_before["adapter"]))
        self.assertFalse(model.backbone_bn.training)
        self.assertFalse(model.adapter_bn.training)
        self.assertFalse(model.decoder.bn.training)
        self.assertEqual(model.decoder.bn.momentum, 0.1)
        self.assertEqual(learned_parameter_sha256(model), parameter_hash)

    def test_buffer_banks_round_trip_without_touching_parameters(self):
        model = _ToySegmenter()
        original = snapshot_decoder_bn_buffers(model, expected_count=1)
        parameter_hash = learned_parameter_sha256(model)
        model.decoder.bn.running_mean.fill_(9.0)
        changed = snapshot_decoder_bn_buffers(model, expected_count=1)

        load_decoder_bn_buffers(
            model,
            original,
            expected_parameter_sha256=parameter_hash,
            expected_count=1,
        )
        self.assertTrue(
            torch.equal(
                model.decoder.bn.running_mean,
                original["decoder.bn"]["running_mean"],
            )
        )
        load_decoder_bn_buffers(model, changed, expected_count=1)
        self.assertEqual(model.decoder.bn.running_mean.item(), 9.0)

        with torch.no_grad():
            model.scalar.add_(1.0)
        with self.assertRaisesRegex(ValueError, "Learned-parameter hash"):
            load_decoder_bn_buffers(
                model,
                original,
                expected_parameter_sha256=parameter_hash,
                expected_count=1,
            )

    def test_rebuilt_loader_repeats_shuffle_drop_and_crop_stream(self):
        dataset = _RandomCropProxyDataset(17)

        def consume():
            loader = build_calibration_loader(
                dataset,
                seed=42,
                num_workers=0,
                batch_size=8,
                pin_memory=False,
            )
            records = []
            for rgb, sar, _label in loader:
                records.extend(zip(sar.flatten().tolist(), rgb.flatten().tolist()))
            return len(loader), records

        first_batches, first = consume()
        second_batches, second = consume()
        self.assertEqual(first_batches, 2)
        self.assertEqual(len(first), 16)
        self.assertEqual(first, second)

    def test_fixed_evaluation_plan_predeclares_primary_cells(self):
        self.assertEqual(DEFAULT_CALIBRATION_STATES, ("sar", "full"))
        self.assertEqual(EXPECTED_VAL_TILES, 277)
        e15 = build_evaluation_plan(
            ("sar", "full"),
            ("original", "sar", "rgb", "full"),
            checkpoint_epoch=15,
        )
        by_cell = {(row["endpoint"], row["condition"]): row for row in e15}
        self.assertEqual(by_cell[("sar", "matched_state")]["cell_role"], "primary")
        self.assertEqual(
            by_cell[("full", "matched_state")]["cell_role"],
            "paired_diagnostic",
        )
        self.assertEqual(by_cell[("sar", "wrong_state")]["bn_bank"], "full")
        self.assertEqual(by_cell[("full", "wrong_state")]["bn_bank"], "sar")
        self.assertEqual(by_cell[("sar", "original")]["cell_role"], "background")

        e10 = build_evaluation_plan(
            ("sar",), ("original", "sar", "full"), checkpoint_epoch=10
        )
        self.assertTrue(all(row["cell_role"] == "background" for row in e10))

        rgb = build_evaluation_plan(
            ("rgb",),
            ("original", "sar", "rgb"),
            checkpoint_epoch=15,
        )
        self.assertEqual(
            [(row["condition"], row["bn_bank"]) for row in rgb],
            [
                ("original", "original"),
                ("wrong_state", "sar"),
                ("matched_state", "rgb"),
            ],
        )
        self.assertTrue(all(row["cell_role"] == "background" for row in rgb))
        with self.assertRaisesRegex(ValueError, "missing BN bank 'rgb'"):
            build_evaluation_plan(
                ("rgb",),
                ("original", "sar", "full"),
            )

    def test_source_checkpoint_selection_is_preserved_without_reinterpretation(self):
        v1_best = {
            "selection_state": "sar",
            "selection_metric": "mIoU",
            "selection_score": 0.25505,
        }
        self.assertEqual(
            source_checkpoint_selection_metadata(v1_best),
            {
                "source_checkpoint_role": None,
                "source_selection_state": "sar",
                "source_selection_metric": "mIoU",
                "source_selection_score": 0.25505,
                "source_fixed_epoch_no_selection": None,
            },
        )

        v2_fixed = {
            "checkpoint_role": "primary_weights_requires_bn_bank",
            "selection_state": "sar",
            "selection_metric": None,
            "selection_score": None,
            "fixed_epoch_no_selection": True,
        }
        self.assertEqual(
            source_checkpoint_selection_metadata(v2_fixed),
            {
                "source_checkpoint_role": "primary_weights_requires_bn_bank",
                "source_selection_state": "sar",
                "source_selection_metric": None,
                "source_selection_score": None,
                "source_fixed_epoch_no_selection": True,
            },
        )

    def test_val_evaluator_reports_pooled_and_each_official_city(self):
        dataset = _CityDataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        result = evaluate_val_by_city(
            nn.Identity(),
            loader,
            OFFICIAL_VAL_CITIES,
            state="sar",
            device="cpu",
            window_size=2,
            inference_batch_size=1,
            expected_pooled_class_ids=(0, 1),
            infer_fn=_prediction_from_rgb,
        )

        self.assertEqual(result["tiles"], 3)
        self.assertEqual(list(result["cities"]), list(OFFICIAL_VAL_CITIES))
        self.assertEqual(result["pooled"]["confusion"][0][:2], [2, 4])
        self.assertEqual(result["pooled"]["confusion"][1][:2], [0, 6])
        self.assertAlmostEqual(result["pooled"]["mIoU"], (1.0 / 3.0 + 0.6) / 2.0)
        self.assertEqual(result["cities"][OFFICIAL_VAL_CITIES[0]]["tiles"], 1)
        self.assertEqual(
            result["cities"][OFFICIAL_VAL_CITIES[1]]["confusion"][0][:2],
            [0, 4],
        )

        shuffled = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        with self.assertRaisesRegex(ValueError, "SequentialSampler"):
            evaluate_val_by_city(
                nn.Identity(),
                shuffled,
                OFFICIAL_VAL_CITIES,
                state="sar",
                device="cpu",
                window_size=2,
                inference_batch_size=1,
                expected_pooled_class_ids=(0, 1),
                infer_fn=_prediction_from_rgb,
            )

        with self.assertRaisesRegex(ValueError, "pooled Val class support changed"):
            evaluate_val_by_city(
                nn.Identity(),
                loader,
                OFFICIAL_VAL_CITIES,
                state="sar",
                device="cpu",
                window_size=2,
                inference_batch_size=1,
                expected_pooled_class_ids=(0, 1, 2),
                infer_fn=_prediction_from_rgb,
            )

    def test_primary_sar_artifact_is_checkpoint_plus_bn_bank(self):
        composition = build_deployment_composition(
            source_checkpoint_sha256="a" * 64,
            bn_bank_file_sha256="b" * 64,
            matched_sar_bank_sha256="c" * 64,
        )
        self.assertEqual(composition["role"], "primary_deployable_composite")
        self.assertEqual(composition["endpoint"], "sar")
        self.assertEqual(composition["bn_condition"], "matched_state")
        self.assertEqual(composition["source_checkpoint_sha256"], "a" * 64)
        self.assertEqual(composition["bn_bank_file_sha256"], "b" * 64)
        self.assertEqual(composition["bn_bank_entry"], "sar")
        self.assertEqual(composition["bn_bank_entry_sha256"], "c" * 64)
        self.assertFalse(composition["contains_full_model_copy"])
        self.assertTrue(composition["requires_both_components"])
        self.assertNotEqual(composition["composition_sha256"], "a" * 64)

        changed_bank = build_deployment_composition(
            source_checkpoint_sha256="a" * 64,
            bn_bank_file_sha256="d" * 64,
            matched_sar_bank_sha256="c" * 64,
        )
        self.assertNotEqual(
            composition["composition_sha256"],
            changed_bank["composition_sha256"],
        )

    def test_calibration_budget_mismatch_fails_before_updates(self):
        model = _ToySegmenter()
        loader = torch.utils.data.DataLoader(
            _ConstantPairDataset([1.0, 2.0]), batch_size=2
        )
        with self.assertRaisesRegex(ValueError, "1 batches, expected 2"):
            calibrate_decoder_bn(
                model,
                loader,
                state="sar",
                device="cpu",
                expected_batches=2,
                expected_bn_count=1,
            )

    def test_outputs_cannot_overwrite_checkpoint_or_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "fixed.pth"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "report.json"
            bank = root / "bank.pt"
            validate_output_paths(checkpoint, output, bank)

            with self.assertRaisesRegex(ValueError, "different files"):
                validate_output_paths(checkpoint, checkpoint, bank)

            output.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                validate_output_paths(checkpoint, output, bank)


if __name__ == "__main__":
    unittest.main()
