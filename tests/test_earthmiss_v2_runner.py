"""CPU contract tests for the fixed EarthMiss V2 three-state runner."""

import copy
from contextlib import redirect_stderr
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from scripts.evaluate_earthmiss_missing_v1 import _validate_checkpoint
from scripts.train_earthmiss_missing_v2 import (
    EPOCHS,
    PRIMARY_EPOCH,
    RAW_VAL_STATES,
    SCHEDULER_T_MAX,
    TRAIN_STATE_ORDER,
    VALIDATION_EPOCHS,
    backward_all_state_losses,
    build_run_metadata,
    fixed_snapshot_name,
    fixed_snapshot_role,
    parse_args,
    resolve_seed_role,
    run_artifact_paths,
    save_checkpoint,
    train_all_state_batch,
)
from tasks.segmentation.models.MMDINO.availability import canonical_availability
from tasks.segmentation.models.MMDINO.dino_segment import DINOSegmentModule


class _TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "frozen_anchor",
            torch.nn.Parameter(torch.tensor(1.0), requires_grad=False),
        )
        self.calls = []

    def get_intermediate_layers(self, tensor, n):
        self.calls.append(tensor.detach().clone())
        batch = tensor.shape[0]
        tokens = tensor[:, :1, :2, :2].reshape(batch, 4, 1)
        return tuple(tokens + float(index) for index in range(4))


class _TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.75))
        self.calls = []

    def forward(
        self,
        *features,
        patch_h,
        patch_w,
        guidance=None,
        modality_indices=None,
    ):
        if modality_indices is None:
            modality_indices = tuple(range(len(features)))
        modality_indices = tuple(modality_indices)
        self.calls.append((modality_indices, len(features)))
        active_maps = [
            output[0].squeeze(-1).reshape(output[0].shape[0], 1, 2, 2)
            for output in features
        ]
        fused = self.scale * sum(active_maps) / len(active_maps)
        pyramid = [fused for _ in range(4)]
        return [pyramid, list(pyramid)]


class _TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(1, momentum=0.1)
        self.out = torch.nn.Conv2d(1, 2, kernel_size=1, bias=False)
        self.slot_counts = []

    def forward(self, *slots):
        self.slot_counts.append(len(slots))
        normalized_slots = [self.bn(slot[0]) for slot in slots]
        fused = sum(normalized_slots) / len(normalized_slots)
        return self.out(fused)


class _SquaredLoss(torch.nn.Module):
    def forward(self, logits, target):
        return (logits - target).square().mean()


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0
        self.zero_calls = 0

    def zero_grad(self, *args, **kwargs):
        self.zero_calls += 1
        return super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


def _tiny_model():
    model = DINOSegmentModule.__new__(DINOSegmentModule)
    torch.nn.Module.__init__(model)
    model.num_modalities = 2
    model.use_optical_stem = False
    model.backbone_type = "dinov3_vits16"
    model.backbone = _TinyBackbone()
    model.adapter = _TinyAdapter()
    model.decoder = _TinyDecoder()
    return model


def _inputs():
    base = torch.arange(2 * 32 * 32, dtype=torch.float32).reshape(2, 1, 32, 32)
    rgb = torch.cat((base, base + 3.0, base + 7.0), dim=1) / 100.0
    sar = (base.flip(-1) + 11.0) / 70.0
    target = torch.zeros(2, 2, 32, 32)
    return rgb, sar, target


def _cached_logits(model, rgb, sar):
    outputs = model.extract_frozen_backbone_outputs(rgb, sar)
    logits = {}
    for state in TRAIN_STATE_ORDER:
        logits[state] = model.forward_from_backbone_outputs(
            rgb,
            sar,
            backbone_outputs=outputs,
            availability=canonical_availability(
                state,
                batch_size=rgb.shape[0],
            ),
        )
    return logits


class EarthMissV2RunnerTest(unittest.TestCase):
    def test_cached_three_state_logits_and_bn_match_naive_forwards(self):
        torch.manual_seed(8)
        cached_model = _tiny_model()
        naive_model = copy.deepcopy(cached_model)
        cached_model.train()
        naive_model.train()
        rgb, sar, _ = _inputs()

        cached_logits = _cached_logits(cached_model, rgb, sar)
        naive_logits = {}
        for state in TRAIN_STATE_ORDER:
            naive_logits[state] = naive_model(
                rgb,
                sar,
                availability=canonical_availability(
                    state,
                    batch_size=rgb.shape[0],
                ),
            )

        self.assertEqual(len(cached_model.backbone.calls), 2)
        self.assertEqual(len(naive_model.backbone.calls), 4)
        self.assertEqual(
            cached_model.adapter.calls,
            [((1,), 1), ((0,), 1), ((0, 1), 2)],
        )
        self.assertEqual(cached_model.decoder.slot_counts, [2, 2, 2])
        for state in TRAIN_STATE_ORDER:
            torch.testing.assert_close(
                cached_logits[state],
                naive_logits[state],
                rtol=0.0,
                atol=0.0,
            )
        torch.testing.assert_close(
            cached_model.decoder.bn.running_mean,
            naive_model.decoder.bn.running_mean,
        )
        torch.testing.assert_close(
            cached_model.decoder.bn.running_var,
            naive_model.decoder.bn.running_var,
        )
        self.assertEqual(
            cached_model.decoder.bn.num_batches_tracked.item(),
            naive_model.decoder.bn.num_batches_tracked.item(),
        )
        self.assertEqual(cached_model.decoder.bn.num_batches_tracked.item(), 6)

    def test_cached_api_checks_all_slots_but_old_forward_checks_only_active_slots(self):
        model = _tiny_model()
        model.eval()
        rgb = torch.zeros(2, 3, 48, 48)
        sar = torch.zeros(2, 1, 32, 32)

        logits = model(
            rgb,
            sar,
            availability=canonical_availability("sar", batch_size=2),
        )
        self.assertEqual(tuple(logits.shape), (2, 2, 32, 32))
        with self.assertRaisesRegex(ValueError, "same spatial shape"):
            model.extract_frozen_backbone_outputs(rgb, sar)
        with self.assertRaisesRegex(ValueError, "same spatial shape"):
            model(
                rgb,
                sar,
                availability=canonical_availability("full", batch_size=2),
            )

    def test_three_sequential_scaled_backwards_equal_one_combined_backward(self):
        torch.manual_seed(17)
        sequential = _tiny_model()
        combined = copy.deepcopy(sequential)
        sequential.train()
        combined.train()
        rgb, sar, target = _inputs()
        criterion = _SquaredLoss()

        sequential.zero_grad(set_to_none=True)
        record = backward_all_state_losses(
            sequential,
            criterion,
            rgb,
            sar,
            target,
        )

        combined.zero_grad(set_to_none=True)
        combined_outputs = combined.extract_frozen_backbone_outputs(rgb, sar)
        combined_losses = []
        for state in TRAIN_STATE_ORDER:
            logits = combined.forward_from_backbone_outputs(
                rgb,
                sar,
                backbone_outputs=combined_outputs,
                availability=canonical_availability(
                    state,
                    batch_size=rgb.shape[0],
                ),
            )
            combined_losses.append(criterion(logits, target))
        (sum(combined_losses) / len(combined_losses)).backward()

        self.assertEqual(record["instrumentation"]["backbone_calls"], 2)
        self.assertEqual(record["instrumentation"]["backward_calls"], 3)
        combined_parameters = dict(combined.named_parameters())
        for name, parameter in sequential.named_parameters():
            if not parameter.requires_grad:
                continue
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(
                parameter.grad,
                combined_parameters[name].grad,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_batch_helper_executes_one_optimizer_step(self):
        torch.manual_seed(29)
        model = _tiny_model()
        model.train()
        rgb, sar, target = _inputs()
        optimizer = _CountingSGD(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )

        record = train_all_state_batch(
            model,
            _SquaredLoss(),
            optimizer,
            rgb,
            sar,
            target,
        )

        self.assertEqual(optimizer.zero_calls, 1)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(record["instrumentation"]["optimizer_steps"], 1)
        self.assertEqual(len(model.backbone.calls), 2)

    def test_protocol_is_fixed_e20_with_e15_as_the_only_primary(self):
        args = SimpleNamespace(
            seed=42,
            window_size=512,
            batch_size=8,
            num_workers=4,
            learning_rate=1e-4,
            seed_role="auto",
        )
        train_dataset = MagicMock()
        train_dataset.__len__.return_value = 2641
        train_dataset.rgb_normalization = "dinov3_lvd_imagenet"
        train_dataset.imagenet_mean = (0.485, 0.456, 0.406)
        train_dataset.imagenet_std = (0.229, 0.224, 0.225)
        train_dataset.sar_mean = (63.30051921735858 / 255.0,)
        train_dataset.sar_std = (68.20405016 / 255.0,)
        val_dataset = MagicMock()
        val_dataset.__len__.return_value = 277
        train_loader = MagicMock()
        train_loader.__len__.return_value = 331

        metadata = build_run_metadata(
            args,
            train_dataset,
            val_dataset,
            train_loader,
        )

        self.assertEqual(EPOCHS, 20)
        self.assertEqual(SCHEDULER_T_MAX, 50)
        self.assertEqual(VALIDATION_EPOCHS, (10, 15, 20))
        self.assertEqual(PRIMARY_EPOCH, 15)
        self.assertEqual(RAW_VAL_STATES, ("sar", "full"))
        self.assertEqual(metadata["primary_cell"], "E15/SAR/Train-SAR")
        self.assertEqual(
            metadata["paired_diagnostic"],
            "E15/Full/Train-Full",
        )
        self.assertEqual(metadata["seed_role"], "exploratory_screen")
        self.assertFalse(metadata["resume_supported"])
        self.assertEqual(
            metadata["interrupted_run_policy"],
            "restart_from_scratch",
        )
        self.assertFalse(metadata["budget"]["early_stopping"])
        self.assertEqual(metadata["scheduler"]["T_max"], 50)
        self.assertEqual(
            metadata["training_graph"]["state_order"],
            ["sar", "rgb", "full"],
        )
        self.assertEqual(
            metadata["evaluation"]["checkpoint_policy"],
            "fixed_epochs_no_best_selection",
        )
        self.assertEqual(fixed_snapshot_name(15), "fixed_e15.pth")
        self.assertEqual(fixed_snapshot_role(10), "diagnostic_only")
        self.assertEqual(
            fixed_snapshot_role(15),
            "primary_weights_requires_bn_bank",
        )
        self.assertEqual(fixed_snapshot_role(20), "diagnostic_only")
        self.assertEqual(
            metadata["evaluation"]["primary_weights_checkpoint"],
            "fixed_e15.pth",
        )
        self.assertTrue(
            metadata["evaluation"]["deployment_requires_bn_bank"]
        )
        with self.assertRaisesRegex(ValueError, "not a frozen"):
            fixed_snapshot_role(5)

    def test_seed_roles_are_explicit_for_screen_and_confirmation(self):
        self.assertEqual(resolve_seed_role(42), "exploratory_screen")
        self.assertEqual(resolve_seed_role(43), "confirmatory")
        self.assertEqual(resolve_seed_role(44), "confirmatory")
        self.assertEqual(
            resolve_seed_role(45, "confirmatory"),
            "confirmatory",
        )
        with self.assertRaisesRegex(ValueError, "reserved"):
            resolve_seed_role(42, "confirmatory")
        with self.assertRaisesRegex(ValueError, "reserved"):
            resolve_seed_role(43, "exploratory_screen")
        with self.assertRaisesRegex(ValueError, "explicit role"):
            resolve_seed_role(45)

    def test_resume_flag_is_rejected_and_run_json_blocks_reuse(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parse_args(["--resume"])
        self.assertEqual(raised.exception.code, 2)

        artifact_names = [
            path.name for path in run_artifact_paths(Path("run_d_seed42"))
        ]
        self.assertEqual(
            artifact_names,
            [
                "run.json",
                "metrics.jsonl",
                "fixed_e10.pth",
                "fixed_e15.pth",
                "fixed_e20.pth",
            ],
        )

    def test_raw_e15_checkpoint_is_weights_only_until_bn_bank_is_attached(self):
        model = MagicMock()
        optimizer = MagicMock()
        scheduler = MagicMock()
        metadata = {
            "seed": 42,
            "primary_cell": "E15/SAR/Train-SAR",
            "paired_diagnostic": "E15/Full/Train-Full",
        }
        raw_validation = {"sar": {"mIoU": 0.2}, "full": {"mIoU": 0.4}}

        with patch("scripts.train_earthmiss_missing_v2.torch.save") as save:
            save_checkpoint(
                "fixed_e15.pth",
                model,
                optimizer,
                scheduler,
                15,
                metadata,
                checkpoint_role="primary_weights_requires_bn_bank",
                raw_validation=raw_validation,
            )

        payload = save.call_args.args[0]
        self.assertEqual(
            payload["checkpoint_role"],
            "primary_weights_requires_bn_bank",
        )
        self.assertIsNone(payload["selection_state"])
        self.assertEqual(payload["deployment_state"], "sar")
        self.assertTrue(payload["requires_bn_bank"])
        self.assertIsNone(payload["selection_metric"])
        self.assertIsNone(payload["selection_score"])
        self.assertTrue(payload["fixed_epoch_no_selection"])
        self.assertEqual(payload["primary_cell"], "E15/SAR/Train-SAR")
        self.assertEqual(payload["raw_validation"], raw_validation)

        args = SimpleNamespace(
            split="test",
            allow_non_primary_test_checkpoint=False,
        )
        with self.assertRaisesRegex(ValueError, "primary_deployment"):
            _validate_checkpoint(payload, args)


if __name__ == "__main__":
    unittest.main()
