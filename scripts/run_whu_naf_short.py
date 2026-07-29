"""Paired five-epoch WHU continuation for the NAF-P2 quick screen.

Run this foreground entry point once with ``--variant r0`` and once with
``--variant r1``.  Both variants start from the same sealed epoch-45
checkpoint.  R1 alone adds the frozen released NAF and a zero-initialized
1x1 residual projection.  A passing full-test E0 JSON is required so this
script cannot accidentally train from an unverified starting point.

This script is intentionally separate from the faithful released trainer:
it is an experimental continuation protocol, not an official reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from scripts.evaluate_whu_naf_e0 import (  # noqa: E402
    BACKBONE_TYPE,
    build_test_loader,
    build_variant,
    evaluate_variant,
    file_sha256,
)
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)
from utils.utils import set_seed  # noqa: E402


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_verified_e0(args: argparse.Namespace) -> tuple[dict, str, str]:
    e0 = json.loads(args.e0_result.read_text(encoding="utf-8"))
    baseline_sha = file_sha256(args.baseline_checkpoint)
    naf_sha = file_sha256(args.naf_checkpoint)
    required_true = (
        e0.get("status") == "PASS"
        and e0.get("scope") == "full-test"
        and e0.get("prediction_equal") is True
        and e0.get("label_equal") is True
        and e0.get("confusion_equal") is True
        and e0.get("evaluated_images") == e0.get("full_test_length")
    )
    if not required_true:
        raise RuntimeError("Continuation requires a passing full-test E0 result")
    if e0.get("baseline_checkpoint_sha256") != baseline_sha:
        raise RuntimeError("Baseline checkpoint differs from the verified E0 object")
    if e0.get("naf_checkpoint_sha256") != naf_sha:
        raise RuntimeError("NAF checkpoint differs from the verified E0 object")
    if e0.get("naf_backend") != args.naf_backend:
        raise RuntimeError("NAF backend differs from the verified E0 protocol")
    if e0.get("naf_q_tile") != list(args.naf_q_tile or []):
        raise RuntimeError("NAF query tile differs from the verified E0 protocol")
    if e0.get("naf_kv_tile") != list(args.naf_kv_tile or []):
        raise RuntimeError("NAF key/value tile differs from the verified E0 protocol")
    if any(float(value) != 0.0 for value in e0.get("metric_differences", {}).values()):
        raise RuntimeError("Verified E0 result contains non-zero metric differences")
    return e0, baseline_sha, naf_sha


def build_train_loader(args: argparse.Namespace, window_size: tuple[int, int]):
    # This reset happens after model construction and both checkpoint loads.
    # num_workers=0 keeps WHU's Python-random source/crop/augmentation stream in
    # this process; a separate torch Generator controls shuffling.
    set_seed(args.seed)
    dataset = build_dataset(
        "WHU",
        "train",
        window_size=window_size,
        model_name="DINOv3",
        modality="multi",
        backbone_type=BACKBONE_TYPE,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )
    return loader


def build_optimizer(
    model: torch.nn.Module,
    variant: str,
    old_lr: float,
    zero_lr: float,
    weight_decay: float,
    epochs: int,
):
    zero_weight = None
    old_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "adapter.naf_zero_conv.weight":
            if variant != "r1":
                raise AssertionError("R0 unexpectedly contains a trainable ZeroConv")
            zero_weight = parameter
            continue
        old_parameters.append(parameter)

    groups = [
        {
            "params": old_parameters,
            "lr": old_lr,
            "weight_decay": weight_decay,
            "group_name": "old_parameters",
        }
    ]
    if variant == "r1":
        if zero_weight is None:
            raise AssertionError("R1 ZeroConv parameter was not found")
        groups.append(
            {
                "params": [zero_weight],
                "lr": zero_lr,
                "weight_decay": 0.0,
                "group_name": "naf_zero_conv",
            }
        )

    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-7,
    )
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if optimized_ids != trainable_ids:
        raise AssertionError("Optimizer does not exactly cover trainable parameters")
    if variant == "r1":
        naf_ids = {id(parameter) for parameter in model.adapter.naf.parameters()}
        if optimized_ids & naf_ids:
            raise AssertionError("Frozen NAF parameters leaked into the optimizer")
    return optimizer, scheduler, zero_weight, len(old_parameters)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    run_config: dict,
) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "protocol": run_config,
        },
        temporary,
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one side of the paired WHU NAF-P2 five-epoch screen"
    )
    parser.add_argument("--variant", choices=("r0", "r1"), required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--naf-checkpoint", type=Path, required=True)
    parser.add_argument("--e0-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval-epochs", type=int, nargs="+", default=(2, 5))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--old-lr", type=float, default=1e-5)
    parser.add_argument("--zero-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--guidance-size", type=int, default=224)
    parser.add_argument(
        "--naf-backend",
        choices=("cutlass-fna", "flex-fna"),
        default="cutlass-fna",
    )
    parser.add_argument("--naf-q-tile", type=int, nargs=2)
    parser.add_argument("--naf-kv-tile", type=int, nargs=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-batches", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="smoke-only cap; omit for the formal continuation",
    )
    parser.add_argument(
        "--max-test-images",
        type=int,
        help="smoke-only cap; omit for full E2/E5 evaluation",
    )
    args = parser.parse_args()

    for path in (args.baseline_checkpoint, args.naf_checkpoint, args.e0_result):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.output_dir.exists():
        parser.error(f"refusing to reuse output directory: {args.output_dir}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.inference_batch_size <= 0:
        parser.error("epochs and batch sizes must be positive")
    if args.old_lr <= 0 or args.zero_lr <= 0 or args.weight_decay < 0:
        parser.error("learning rates must be positive and weight decay non-negative")
    if args.hash_batches < 0:
        parser.error("--hash-batches cannot be negative")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        parser.error("--max-train-batches must be positive")
    if args.max_test_images is not None and args.max_test_images <= 0:
        parser.error("--max-test-images must be positive")
    if not args.eval_epochs:
        parser.error("at least one evaluation epoch is required")
    if any(epoch <= 0 or epoch > args.epochs for epoch in args.eval_epochs):
        parser.error("evaluation epochs must fall within 1..epochs")
    if len(set(args.eval_epochs)) != len(args.eval_epochs):
        parser.error("evaluation epochs must be unique")
    if (args.naf_q_tile is None) != (args.naf_kv_tile is None):
        parser.error("--naf-q-tile and --naf-kv-tile must be set together")
    if args.naf_q_tile is not None:
        if any(value <= 0 for value in (*args.naf_q_tile, *args.naf_kv_tile)):
            parser.error("NAF tile dimensions must be positive")
        args.naf_q_tile = tuple(args.naf_q_tile)
        args.naf_kv_tile = tuple(args.naf_kv_tile)
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NAF-P2 short continuation")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    e0, baseline_sha, naf_sha = read_verified_e0(args)
    commit = git_commit()
    scope = (
        "smoke"
        if args.max_train_batches is not None or args.max_test_images is not None
        else "formal-paired-short"
    )
    args.output_dir.mkdir(parents=True)

    model, cfg = build_variant(args, args.variant)
    train_loader = build_train_loader(args, cfg["window_size"])
    test_loader, full_test_length = build_test_loader(
        SimpleNamespace(max_images=args.max_test_images),
        cfg["window_size"],
    )
    optimizer, scheduler, zero_weight, old_parameter_tensors = build_optimizer(
        model,
        args.variant,
        args.old_lr,
        args.zero_lr,
        args.weight_decay,
        args.epochs,
    )
    device = torch.device(args.device)
    model.to(device)

    run_config = {
        "scope": scope,
        "variant": args.variant,
        "git_commit": commit,
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "baseline_checkpoint_sha256": baseline_sha,
        "naf_checkpoint": str(args.naf_checkpoint.resolve()),
        "naf_checkpoint_sha256": naf_sha,
        "verified_e0_result": str(args.e0_result.resolve()),
        "verified_e0_result_sha256": file_sha256(args.e0_result),
        "e0_miou": float(e0["r0"]["MIoU"]),
        "epochs": args.epochs,
        "eval_epochs": sorted(args.eval_epochs),
        "batch_size": args.batch_size,
        "inference_batch_size": args.inference_batch_size,
        "old_lr": args.old_lr,
        "zero_lr": args.zero_lr if args.variant == "r1" else None,
        "old_weight_decay": args.weight_decay,
        "zero_weight_decay": 0.0 if args.variant == "r1" else None,
        "scheduler": "CosineAnnealingLR",
        "scheduler_t_max": args.epochs,
        "scheduler_eta_min": 1e-7,
        "seed_reset_after_model_load": args.seed,
        "train_num_workers": 0,
        "train_shuffle_generator_seed": args.seed,
        "hash_batches": args.hash_batches,
        "train_batches_per_epoch": len(train_loader),
        "max_train_batches": args.max_train_batches,
        "test_images": len(test_loader.dataset),
        "full_test_length": full_test_length,
        "max_test_images": args.max_test_images,
        "naf_backend": args.naf_backend,
        "naf_q_tile": args.naf_q_tile,
        "naf_kv_tile": args.naf_kv_tile,
        "guidance_size": args.guidance_size,
        "old_parameter_tensors": old_parameter_tensors,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_config, ensure_ascii=False, indent=2))

    train_metrics_path = args.output_dir / "train_metrics.jsonl"
    eval_metrics_path = args.output_dir / "eval_metrics.jsonl"
    batch_hashes_path = args.output_dir / "epoch1_batch_hashes.jsonl"
    loss_fn = cfg["loss_fn"]
    eval_epochs = set(args.eval_epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.variant == "r1":
            if model.adapter.naf.training:
                raise AssertionError("Outer model.train() enabled frozen NAF")
            if any(parameter.requires_grad for parameter in model.adapter.naf.parameters()):
                raise AssertionError("Frozen NAF parameters became trainable")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        loss_sum = 0.0
        steps = 0
        iterator = tqdm(train_loader, desc=f"{args.variant} epoch {epoch}/{args.epochs}")
        for batch_index, (optical, sar, label) in enumerate(iterator):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            if epoch == 1 and batch_index < args.hash_batches:
                append_jsonl(
                    batch_hashes_path,
                    {
                        "epoch": epoch,
                        "batch_index": batch_index,
                        "optical_sha256": tensor_sha256(optical),
                        "sar_sha256": tensor_sha256(sar),
                        "label_sha256": tensor_sha256(label),
                        "optical_shape": list(optical.shape),
                        "sar_shape": list(sar.shape),
                        "label_shape": list(label.shape),
                    },
                )

            optical = optical.to(device)
            sar = sar.to(device)
            label = label.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(optical, sar)
            loss = loss_fn(logits, label)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch={epoch}, batch={batch_index}"
                )
            loss.backward()
            if args.variant == "r1":
                if any(parameter.grad is not None for parameter in model.adapter.naf.parameters()):
                    raise AssertionError("Frozen NAF accumulated parameter gradients")
                if zero_weight.grad is None or not torch.isfinite(zero_weight.grad).all():
                    raise AssertionError("ZeroConv gradient is missing or non-finite")
            optimizer.step()
            loss_value = float(loss.detach())
            loss_sum += loss_value
            steps += 1
            iterator.set_postfix(loss=f"{loss_value:.4f}")
            del optical, sar, label, logits, loss

        if steps == 0:
            raise RuntimeError("Training epoch executed zero optimizer steps")
        scheduler.step()
        torch.cuda.synchronize(device)
        train_record = {
            "epoch": epoch,
            "steps": steps,
            "average_loss": loss_sum / steps,
            "learning_rates": {
                group.get("group_name", f"group_{index}"): group["lr"]
                for index, group in enumerate(optimizer.param_groups)
            },
            "zero_weight_norm": (
                float(zero_weight.detach().norm()) if zero_weight is not None else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
        append_jsonl(train_metrics_path, train_record)
        print(json.dumps(train_record, ensure_ascii=False, sort_keys=True))

        if epoch in eval_epochs:
            evaluation = evaluate_variant(
                model,
                cfg,
                test_loader,
                device,
                args.inference_batch_size,
            )
            evaluation["epoch"] = epoch
            evaluation["gain_over_e0"] = evaluation["MIoU"] - run_config["e0_miou"]
            append_jsonl(eval_metrics_path, evaluation)
            print(json.dumps(evaluation, ensure_ascii=False, sort_keys=True))
            save_checkpoint(
                args.output_dir / f"checkpoint_e{epoch}.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                run_config,
            )

    print(f"run_output={args.output_dir.resolve()}")
    print(f"run_status=PASS variant={args.variant} scope={scope}")


if __name__ == "__main__":
    main()
