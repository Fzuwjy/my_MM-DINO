"""Fail-fast server smoke for the frozen-NAF P2 residual experiment.

This is deliberately not a training launcher. It verifies four prerequisites
before a paired five-epoch R0/R1 screen is allowed:

1. the epoch-45 baseline checkpoint loads strictly;
2. the released standalone NAF checkpoint loads strictly;
3. a zero-initialized residual branch is exactly equivalent to the baseline;
4. two real batch-8 training steps fit and open the intended gradient gate.

Run on one GPU with plain ``python`` (not torchrun).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from pathlib import Path
from typing import Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from configs import get_cfg  # noqa: E402
from datasets import build_dataset  # noqa: E402
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.utils import set_seed  # noqa: E402


NUM_CLASSES = 7


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def memory_gib(value: int) -> float:
    return value / 1024**3


def load_baseline_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise TypeError("Baseline checkpoint must contain a 'model' state dict")
    return payload["model"]


def build_models(args: argparse.Namespace):
    common = dict(
        num_modalities=2,
        use_lora=False,
        r=3,
        backbone_type="dinov3_vits16",
    )
    baseline_cfg = get_cfg("DINOv3", "WHU", use_naf=False, **common)
    naf_cfg = get_cfg(
        "DINOv3",
        "WHU",
        use_naf=True,
        naf_checkpoint=str(args.naf_checkpoint),
        naf_guidance_size=args.guidance_size,
        **common,
    )
    baseline = baseline_cfg["model"]
    naf_model = naf_cfg["model"]
    state = load_baseline_state(args.baseline_checkpoint)

    baseline.load_state_dict(state, strict=True)
    incompatible = naf_model.load_state_dict(state, strict=False)
    expected_missing = {
        key
        for key in naf_model.state_dict()
        if key.startswith("adapter.naf.")
        or key == "adapter.naf_zero_conv.weight"
    }
    actual_missing = set(incompatible.missing_keys)
    if incompatible.unexpected_keys:
        raise AssertionError(
            f"Unexpected baseline checkpoint keys: {incompatible.unexpected_keys}"
        )
    if actual_missing != expected_missing:
        raise AssertionError(
            "Baseline-to-NAF missing-key contract changed: "
            f"expected={sorted(expected_missing)}, actual={sorted(actual_missing)}"
        )
    if torch.count_nonzero(naf_model.adapter.naf_zero_conv.weight).item() != 0:
        raise AssertionError("NAF residual projection is not exactly zero initialized")
    if any(parameter.requires_grad for parameter in naf_model.adapter.naf.parameters()):
        raise AssertionError("Released NAF parameters must all be frozen")
    if sum(p.numel() for p in naf_model.adapter.naf.parameters()) != 662_528:
        raise AssertionError("Unexpected released NAF parameter count")
    if len(naf_model.adapter.naf.state_dict()) != 37:
        raise AssertionError("Unexpected released NAF state-dict structure")

    # The optimizers created by get_cfg belong to the full-run baseline and are
    # intentionally discarded. The smoke builds the experimental groups only
    # after both checkpoint loads.
    baseline_cfg["optimizer"] = None
    baseline_cfg["scheduler"] = None
    naf_cfg["optimizer"] = None
    naf_cfg["scheduler"] = None
    del state
    print("baseline_checkpoint_strict_load=true")
    print("naf_checkpoint_strict_load=true")
    print(f"naf_missing_keys_from_baseline={len(actual_missing)}")
    return baseline_cfg, naf_cfg


def fetch_real_batches(args: argparse.Namespace, window_size: tuple[int, int]):
    # Seed only after construction and checkpoint loading. With num_workers=0,
    # the released Python-random crop/augmentation stream stays in this process.
    set_seed(args.seed)
    dataset = build_dataset(
        "WHU",
        "train",
        window_size=window_size,
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    iterator = iter(loader)
    batches = [next(iterator), next(iterator)]
    print(
        "train_cache_capacities="
        f"{dataset.rgb_cache.capacity},{dataset.label_cache.capacity},"
        f"{dataset.sar_cache.capacity}"
    )
    for index, (optical, sar, label) in enumerate(batches):
        print(
            f"batch_{index}_shapes="
            f"{tuple(optical.shape)},{tuple(sar.shape)},{tuple(label.shape)}"
        )
        print(
            f"batch_{index}_sha256="
            f"optical:{tensor_sha256(optical)},"
            f"sar:{tensor_sha256(sar)},label:{tensor_sha256(label)}"
        )
        if label.dtype != torch.int64:
            raise AssertionError(f"Expected int64 labels, received {label.dtype}")
    return batches


def confusion(prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    prediction = prediction.reshape(-1).to(torch.int64)
    label = label.reshape(-1).to(torch.int64)
    valid = (label >= 0) & (label < NUM_CLASSES)
    encoded = label[valid] * NUM_CLASSES + prediction[valid]
    return torch.bincount(encoded, minlength=NUM_CLASSES**2).reshape(
        NUM_CLASSES, NUM_CLASSES
    )


def check_naf_repeatability(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    guidance_size: int,
) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    values = torch.randn(1, 256, 32, 32, generator=generator).to(device)
    guidance = torch.nn.functional.interpolate(
        batch[0][:1].to(device),
        size=(guidance_size, guidance_size),
        mode="bilinear",
        align_corners=False,
    )
    naf = model.adapter.naf.to(device)
    naf.eval()
    with torch.inference_mode():
        first = naf(guidance, values, (128, 128))
        second = naf(guidance, values, (128, 128))
    if not torch.isfinite(first).all():
        raise AssertionError("Standalone NAF output is non-finite")
    max_abs = float((first - second).abs().max())
    print(
        "naf_probe_shapes="
        f"(1,3,{guidance_size},{guidance_size}),"
        "(1,256,32,32),(1,256,128,128)"
    )
    print(f"naf_eval_repeat_max_abs={max_abs:.12g}")
    if not torch.equal(first, second):
        raise AssertionError("Frozen NAF is not exactly repeatable in eval mode")
    print("naf_eval_repeat_exact=true")
    del values, guidance, first, second


def inference_snapshot(
    model: torch.nn.Module,
    batch,
    loss_fn,
    device: torch.device,
):
    model.to(device)
    model.eval()
    optical, sar, label = (value.to(device) for value in batch)
    with torch.inference_mode():
        logits = model(optical, sar)
        loss = loss_fn(logits, label)
        prediction = logits.argmax(dim=1)
    torch.cuda.synchronize(device)
    snapshot = (
        logits.cpu(),
        loss.cpu(),
        prediction.cpu(),
        confusion(prediction.cpu(), label.cpu()),
    )
    del optical, sar, label, logits, loss, prediction
    return snapshot


def check_e0_exact(
    baseline_cfg,
    naf_cfg,
    batch,
    device: torch.device,
) -> None:
    loss_fn = naf_cfg["loss_fn"]
    baseline = baseline_cfg["model"]
    baseline_snapshot = inference_snapshot(baseline, batch, loss_fn, device)
    baseline.to("cpu")
    del baseline_cfg, baseline
    gc.collect()
    torch.cuda.empty_cache()

    naf_model = naf_cfg["model"]
    naf_snapshot = inference_snapshot(naf_model, batch, loss_fn, device)
    baseline_logits, baseline_loss, baseline_pred, baseline_confusion = (
        baseline_snapshot
    )
    naf_logits, naf_loss, naf_pred, naf_confusion = naf_snapshot
    absolute = (baseline_logits - naf_logits).abs()
    max_abs = float(absolute.max())
    mean_abs = float(absolute.mean())
    loss_abs = abs(float(baseline_loss) - float(naf_loss))
    pred_mismatch = int(torch.count_nonzero(baseline_pred != naf_pred))
    confusion_equal = torch.equal(baseline_confusion, naf_confusion)

    print(f"e0_logits_max_abs={max_abs:.12g}")
    print(f"e0_logits_mean_abs={mean_abs:.12g}")
    print(f"e0_loss_abs={loss_abs:.12g}")
    print(f"e0_prediction_mismatch={pred_mismatch}")
    print(f"e0_confusion_equal={str(confusion_equal).lower()}")
    print(f"e0_confusion={baseline_confusion.tolist()}")
    if not torch.equal(baseline_logits, naf_logits):
        raise AssertionError("E0 failed: baseline and zero-NAF logits differ")
    if loss_abs != 0.0:
        raise AssertionError("E0 failed: baseline and zero-NAF losses differ")
    if pred_mismatch != 0 or not confusion_equal:
        raise AssertionError("E0 failed: predictions or confusion matrix differ")
    print("e0_exact_equivalence=true")


def experimental_optimizer(model: torch.nn.Module):
    zero_weight = model.adapter.naf_zero_conv.weight
    old_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "adapter.naf_zero_conv.weight":
            continue
        old_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {"params": old_parameters, "lr": 1e-5, "weight_decay": 0.01},
            {"params": [zero_weight], "lr": 1e-4, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=5, eta_min=1e-7
    )
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if optimized_ids != trainable_ids:
        raise AssertionError("Experimental optimizer does not exactly cover trainable params")
    naf_ids = {id(p) for p in model.adapter.naf.parameters()}
    if optimized_ids & naf_ids:
        raise AssertionError("Frozen NAF parameters leaked into the optimizer")
    print(f"optimizer_old_parameter_tensors={len(old_parameters)}")
    print("optimizer_old_lr=1e-05,weight_decay=0.01")
    print("optimizer_zero_lr=0.0001,weight_decay=0.0")
    return optimizer, scheduler


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def gradients_are_none(parameters: Iterable[torch.nn.Parameter]) -> bool:
    return all(parameter.grad is None for parameter in parameters)


def check_two_training_steps(
    naf_cfg,
    batches,
    device: torch.device,
) -> None:
    model = naf_cfg["model"]
    loss_fn = naf_cfg["loss_fn"]
    model.to(device)
    optimizer, scheduler = experimental_optimizer(model)
    captured: dict[str, object] = {}

    def capture_values(_module, inputs):
        values = inputs[1]
        values.retain_grad()
        captured["values"] = values

    def capture_delta(_module, inputs):
        delta = inputs[0]
        captured["delta_mean_abs"] = float(delta.detach().abs().mean())
        captured["delta_max_abs"] = float(delta.detach().abs().max())

    naf_hook = model.adapter.naf.register_forward_pre_hook(capture_values)
    zero_hook = model.adapter.naf_zero_conv.register_forward_pre_hook(capture_delta)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    start_allocated = torch.cuda.memory_allocated(device)
    zero_weight = model.adapter.naf_zero_conv.weight
    try:
        for step, batch in enumerate(batches, start=1):
            captured.clear()
            model.train()
            if model.adapter.naf.training:
                raise AssertionError("Outer model.train() enabled frozen NAF")
            if model.adapter.naf.image_encoder.rope.training:
                raise AssertionError("Outer model.train() enabled frozen NAF RoPE")

            optical, sar, label = (value.to(device) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(optical, sar)
            loss = loss_fn(logits, label)
            loss.backward()
            if not torch.isfinite(loss):
                raise AssertionError(f"Non-finite smoke loss at step {step}")
            if zero_weight.grad is None or not torch.isfinite(zero_weight.grad).all():
                raise AssertionError(f"Invalid zero-conv gradient at step {step}")
            zero_grad_norm = float(zero_weight.grad.norm())
            values = captured.get("values")
            if not isinstance(values, torch.Tensor) or values.grad is None:
                raise AssertionError("NAF low-resolution value gradient was not captured")
            if not torch.isfinite(values.grad).all():
                raise AssertionError(f"Invalid NAF value gradient at step {step}")
            value_grad_norm = float(values.grad.norm())
            if zero_grad_norm <= 0.0:
                raise AssertionError(f"Zero-conv received no gradient at step {step}")
            if step == 1 and value_grad_norm != 0.0:
                raise AssertionError(
                    "Step 1 NAF value path should be closed by exact zero weights"
                )
            if step == 2 and value_grad_norm <= 0.0:
                raise AssertionError("Step 2 NAF value path did not open")
            if not gradients_are_none(model.adapter.naf.parameters()):
                raise AssertionError("Frozen NAF accumulated parameter gradients")

            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            zero_weight_norm = float(zero_weight.detach().norm())
            if zero_weight_norm <= 0.0:
                raise AssertionError("Zero-conv weights did not update")
            print(f"step_{step}_loss={float(loss.detach()):.9f}")
            print(f"step_{step}_zero_grad_norm={zero_grad_norm:.12g}")
            print(f"step_{step}_value_grad_norm={value_grad_norm:.12g}")
            print(f"step_{step}_zero_weight_norm={zero_weight_norm:.12g}")
            print(
                f"step_{step}_delta_abs="
                f"mean:{captured['delta_mean_abs']:.12g},"
                f"max:{captured['delta_max_abs']:.12g}"
            )
            del optical, sar, label, logits, loss, values

            if step == 1:
                zero_state = optimizer.state.get(zero_weight, {})
                if "exp_avg" not in zero_state or "exp_avg_sq" not in zero_state:
                    raise AssertionError("AdamW lazy state was not allocated for zero-conv")
                print(
                    "step_1_optimizer_state_mib="
                    f"{optimizer_state_bytes(optimizer) / 1024**2:.3f}"
                )
    finally:
        naf_hook.remove()
        zero_hook.remove()

    torch.cuda.synchronize(device)
    print(f"train_start_allocated_gib={memory_gib(start_allocated):.3f}")
    print(
        "train_two_step_peak_allocated_gib="
        f"{memory_gib(torch.cuda.max_memory_allocated(device)):.3f}"
    )
    print(
        "train_two_step_peak_reserved_gib="
        f"{memory_gib(torch.cuda.max_memory_reserved(device)):.3f}"
    )
    print("two_real_training_steps=true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--naf-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--guidance-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for path in (args.baseline_checkpoint, args.naf_checkpoint):
        if not path.is_file():
            parser.error(f"checkpoint does not exist: {path}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.guidance_size <= 0:
        parser.error("--guidance-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NAF P2 smoke")
    device = torch.device(args.device)
    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)

    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"baseline_checkpoint_sha256={file_sha256(args.baseline_checkpoint)}")
    print(f"naf_checkpoint_sha256={file_sha256(args.naf_checkpoint)}")
    print(f"naf_guidance=optical_common_normalized_{args.guidance_size}x{args.guidance_size}")

    baseline_cfg, naf_cfg = build_models(args)
    import natten  # Imported lazily by the NAF-enabled model above.

    print(f"natten={getattr(natten, '__version__', 'unknown')}")
    print(f"natten_has_libnatten={getattr(natten, 'HAS_LIBNATTEN', 'unknown')}")
    batches = fetch_real_batches(args, naf_cfg["window_size"])
    check_naf_repeatability(
        naf_cfg["model"], batches[0], device, args.guidance_size
    )
    check_e0_exact(baseline_cfg, naf_cfg, batches[0], device)
    check_two_training_steps(naf_cfg, batches, device)
    print("smoke_status=PASS")


if __name__ == "__main__":
    main()
