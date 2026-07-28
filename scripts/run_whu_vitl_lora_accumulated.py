"""Run a released WHU ViT-L LoRA path with a memory-compatible batch.

The public trainer remains untouched.  This external launcher makes only the
minimum batching adaptation required by a 31.4 GiB RTX 5090:

* the training DataLoader uses microbatch 4 instead of released batch 8;
* every loss gradient is divided by 2;
* ``zero_grad`` and ``optimizer.step`` run once per two microbatches;
* the config still reports batch 8, so released sliding-window evaluation
  continues to use inference microbatch ``8 * 4 == 32``.

This is a compatibility reproduction, not an exact batch-8 reproduction.
BatchNorm statistics and non-sample-separable losses can depend on microbatch
boundaries even when the optimizer effective batch remains 8.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import runpy
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


SEED = 42
RELEASED_BATCH_SIZE = 8
RELEASED_INFERENCE_BATCH_SIZE = 32
REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TRAINER = REPO_ROOT / "tasks" / "segmentation" / "train_multi.py"


def scientific_configuration(num_modalities: int) -> dict[str, Any]:
    """Return the locked WHU ViT-L LoRA target for one or two modalities."""

    if num_modalities not in (1, 2):
        raise ValueError("WHU compatibility training supports 1 or 2 modalities")
    return {
        "model_name": "DINOv3",
        "dataset_name": "WHU",
        "num_modalities": num_modalities,
        "use_lora": True,
        "r": 3,
        "backbone_type": "dinov3_vitl16",
    }


def seed_model_initialization() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_effective_batch(micro_batch_size: int, grad_accum_steps: int) -> None:
    if micro_batch_size <= 0 or grad_accum_steps <= 0:
        raise ValueError("Microbatch size and accumulation steps must be positive")
    effective_batch = micro_batch_size * grad_accum_steps
    if effective_batch != RELEASED_BATCH_SIZE:
        raise ValueError(
            f"Expected effective batch {RELEASED_BATCH_SIZE}, got "
            f"{micro_batch_size} * {grad_accum_steps} = {effective_batch}"
        )


class GradientScaledLoss:
    """Keep the reported loss unchanged while scaling its backward gradient."""

    def __init__(self, loss_fn: Callable[..., torch.Tensor], accumulation_steps: int):
        self.loss_fn = loss_fn
        self.accumulation_steps = accumulation_steps

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        loss = self.loss_fn(*args, **kwargs)
        if torch.is_grad_enabled() and loss.requires_grad:
            loss.register_hook(
                lambda gradient: gradient / self.accumulation_steps
            )
        return loss


class GradientAccumulationController:
    """Gate an existing optimizer without changing its state representation."""

    def __init__(self, optimizer: Any, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.micro_steps = 0
        self.optimizer_steps = 0
        self._zero_grad = optimizer.zero_grad
        self._step = optimizer.step
        optimizer.zero_grad = self.zero_grad

        def gated_step(*args: Any, **kwargs: Any) -> Any:
            return self.step(*args, **kwargs)

        if getattr(self._step, "_wrapped_by_lr_sched", False):
            gated_step._wrapped_by_lr_sched = True
        optimizer.step = gated_step

    def zero_grad(self, *args: Any, **kwargs: Any) -> Any:
        if self.micro_steps % self.accumulation_steps == 0:
            return self._zero_grad(*args, **kwargs)
        return None

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self.micro_steps += 1
        if self.micro_steps % self.accumulation_steps != 0:
            return None
        self.optimizer_steps += 1
        return self._step(*args, **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WHU Table III ViT-L LoRA compatibility trainer"
    )
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument(
        "--num-modalities",
        type=int,
        choices=(1, 2),
        default=2,
        help="1 for the RGB-only control; 2 preserves the multimodal target",
    )
    args = parser.parse_args()
    try:
        validate_effective_batch(args.micro_batch_size, args.grad_accum_steps)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    if not OFFICIAL_TRAINER.is_file():
        raise FileNotFoundError(f"Official trainer not found: {OFFICIAL_TRAINER}")
    if os.environ.get("WORLD_SIZE") != "1" or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "Launch with torchrun --standalone --nproc_per_node=1 to preserve "
            "the released one-process DDP path"
        )

    segmentation_root = str(OFFICIAL_TRAINER.parent)
    repo_root = str(REPO_ROOT)
    if segmentation_root not in sys.path:
        sys.path.insert(0, segmentation_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    seed_model_initialization()

    from scripts.whu_cache_compat import CACHE_CAPACITY, install_whu_cache_compat
    from scripts.whu_label_dtype_compat import install_whu_label_dtype_compat

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)

    configs_module = importlib.import_module("configs")
    clean_logs_module = importlib.import_module("utils.clean_logs")
    move_files_module = importlib.import_module("utils.move_files")

    original_get_cfg = configs_module.get_cfg
    original_data_loader = torch.utils.data.DataLoader
    original_clean_logs = clean_logs_module.clean_logs
    original_move_files = move_files_module.move_files
    runtime: dict[str, Any] = {}

    def get_accumulated_cfg(model_name=None, dataset_name=None, **kwargs):
        expected = scientific_configuration(args.num_modalities)
        actual = {
            "model_name": model_name,
            "dataset_name": dataset_name,
        }
        for name in ("num_modalities", "use_lora", "r", "backbone_type"):
            actual[name] = kwargs.get(name)
        if actual != expected:
            raise RuntimeError(f"Refusing unexpected scientific configuration: {actual}")

        cfg = original_get_cfg(model_name, dataset_name, **kwargs)
        if cfg.get("batch_size") != RELEASED_BATCH_SIZE:
            raise RuntimeError(
                f"Released config batch changed from {RELEASED_BATCH_SIZE} to "
                f"{cfg.get('batch_size')}"
            )
        runtime["controller"] = GradientAccumulationController(
            cfg["optimizer"], args.grad_accum_steps
        )
        cfg["loss_fn"] = GradientScaledLoss(cfg["loss_fn"], args.grad_accum_steps)
        return cfg

    def compatibility_data_loader(dataset, *loader_args, **loader_kwargs):
        is_whu_train = (
            dataset.__class__.__name__ == "WHU_Dataset"
            and getattr(dataset, "data_type", None) == "train"
        )
        if is_whu_train:
            if loader_args:
                raise RuntimeError("Expected released DataLoader batch_size as a keyword")
            if loader_kwargs.get("batch_size") != RELEASED_BATCH_SIZE:
                raise RuntimeError(
                    "Released training DataLoader batch size changed unexpectedly"
                )
            loader_kwargs["batch_size"] = args.micro_batch_size

        loader = original_data_loader(dataset, *loader_args, **loader_kwargs)
        if is_whu_train:
            micro_batches = len(loader)
            if micro_batches % args.grad_accum_steps != 0:
                raise RuntimeError(
                    f"{micro_batches} microbatches do not divide evenly into "
                    f"groups of {args.grad_accum_steps}"
                )
            runtime["micro_batches_per_epoch"] = micro_batches
            runtime["optimizer_steps_per_epoch"] = (
                micro_batches // args.grad_accum_steps
            )
            print(f"train_micro_batches_per_epoch={micro_batches}")
            print(
                "train_optimizer_steps_per_epoch="
                f"{runtime['optimizer_steps_per_epoch']}"
            )
        return loader

    def preserve_existing_logs(*_args, **_kwargs):
        print("compatibility_cleanup=DISABLED_TO_PRESERVE_EXISTING_RUNS")

    def move_files_with_protocol(src_dir, dst_dir, exclude_names):
        result = original_move_files(src_dir, dst_dir, exclude_names)
        run_dir = Path(dst_dir).resolve().parent
        modality_label = "RGB-only" if args.num_modalities == 1 else "Multi"
        protocol = {
            "protocol": (
                f"WHU Table III {modality_label} ViT-L LoRA "
                "compatibility reproduction"
            ),
            "fidelity": "not exact released batching",
            "seed": SEED,
            "precision": "FP32",
            "released_batch_size": RELEASED_BATCH_SIZE,
            "training_micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.grad_accum_steps,
            "effective_batch_size": args.micro_batch_size * args.grad_accum_steps,
            "evaluation_inference_batch_size": RELEASED_INFERENCE_BATCH_SIZE,
            "backbone_type": "dinov3_vitl16",
            "use_lora": True,
            "lora_rank": 3,
            "num_modalities": args.num_modalities,
            "train_micro_batches_per_epoch": runtime.get("micro_batches_per_epoch"),
            "optimizer_steps_per_epoch": runtime.get("optimizer_steps_per_epoch"),
            "compatibility_notes": [
                "The released trainer source is unmodified.",
                "BatchNorm statistics use microbatch 4.",
                "The released loss is evaluated separately on each microbatch.",
                "Existing run cleanup is disabled to preserve reproduction artifacts.",
            ],
        }
        protocol_path = run_dir / "compatibility_protocol.json"
        protocol_path.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"compatibility_protocol={protocol_path}")
        return result

    configs_module.get_cfg = get_accumulated_cfg
    torch.utils.data.DataLoader = compatibility_data_loader
    clean_logs_module.clean_logs = preserve_existing_logs
    move_files_module.move_files = move_files_with_protocol

    print("compatibility_mode=WHU_VITL_LORA_MICROBATCH_ACCUMULATION")
    print(f"training_micro_batch_size={args.micro_batch_size}")
    print(f"gradient_accumulation_steps={args.grad_accum_steps}")
    print(f"effective_batch_size={RELEASED_BATCH_SIZE}")
    print(f"evaluation_inference_batch_size={RELEASED_INFERENCE_BATCH_SIZE}")
    print(f"num_modalities={args.num_modalities}")
    print(f"whu_cache_capacity={CACHE_CAPACITY}")

    target = scientific_configuration(args.num_modalities)
    official_args = [
        "--model-name",
        target["model_name"],
        "--dataset-name",
        target["dataset_name"],
        "--num-modalities",
        str(target["num_modalities"]),
        "--use-lora",
        str(target["use_lora"]),
        "--r",
        str(target["r"]),
        "--backbone-type",
        target["backbone_type"],
    ]
    sys.argv = [str(OFFICIAL_TRAINER), *official_args]
    try:
        runpy.run_path(str(OFFICIAL_TRAINER), run_name="__main__")
    finally:
        configs_module.get_cfg = original_get_cfg
        torch.utils.data.DataLoader = original_data_loader
        clean_logs_module.clean_logs = original_clean_logs
        move_files_module.move_files = original_move_files

    controller = runtime.get("controller")
    if controller is None:
        raise RuntimeError("Gradient accumulation controller was not installed")
    if controller.micro_steps % args.grad_accum_steps != 0:
        raise RuntimeError("Training ended with an incomplete accumulation group")
    print(f"completed_micro_steps={controller.micro_steps}")
    print(f"completed_optimizer_steps={controller.optimizer_steps}")


if __name__ == "__main__":
    main()
