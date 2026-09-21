"""Run the unmodified official trainer with deterministic model initialization.

This launcher sets seed 42 before the official script constructs the model.  The
official script then executes unchanged and calls ``set_seed(42)`` again at its
original location, preserving its data-loader and augmentation seed sequence.
All command-line arguments are passed through verbatim.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch


SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TRAINER = REPO_ROOT / "tasks" / "segmentation" / "train_multi.py"
SINGLE_RANK_EVAL_ANCHOR = "is_distributed=distributed.is_enabled())"
SINGLE_RANK_EVAL_REPLACEMENT = (
    "is_distributed=(distributed.is_enabled() and "
    "distributed.get_world_size() > 1))"
)
LAUNCHER_LORA_RANK_OPTION = "--lora-rank"
OFFICIAL_LORA_RANK_OPTION = "--r"


def single_rank_eval_compatible_source(source: str) -> str:
    """Select the released local metrics path when DDP has one rank."""

    matches = source.count(SINGLE_RANK_EVAL_ANCHOR)
    if matches != 1:
        raise RuntimeError(
            "Expected exactly one released evaluation call anchor, found "
            f"{matches}; refusing to patch an unknown trainer revision"
        )
    return source.replace(
        SINGLE_RANK_EVAL_ANCHOR,
        SINGLE_RANK_EVAL_REPLACEMENT,
        1,
    )


def seed_model_initialization() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def translate_launcher_arguments(arguments: list[str]) -> list[str]:
    """Translate launcher-only arguments after torchrun has parsed its CLI."""

    translated: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == LAUNCHER_LORA_RANK_OPTION:
            if index + 1 >= len(arguments):
                raise ValueError(f"{LAUNCHER_LORA_RANK_OPTION} requires a value")
            translated.extend((OFFICIAL_LORA_RANK_OPTION, arguments[index + 1]))
            index += 2
            continue
        if argument.startswith(f"{LAUNCHER_LORA_RANK_OPTION}="):
            _, value = argument.split("=", 1)
            if not value:
                raise ValueError(f"{LAUNCHER_LORA_RANK_OPTION} requires a value")
            translated.extend((OFFICIAL_LORA_RANK_OPTION, value))
            index += 1
            continue
        translated.append(argument)
        index += 1
    return translated


def main() -> None:
    if not OFFICIAL_TRAINER.is_file():
        raise FileNotFoundError(f"Official trainer not found: {OFFICIAL_TRAINER}")

    seed_model_initialization()
    segmentation_root = str(OFFICIAL_TRAINER.parent)
    repo_root = str(REPO_ROOT)
    if segmentation_root not in sys.path:
        sys.path.insert(0, segmentation_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from scripts.whu_label_dtype_compat import install_whu_label_dtype_compat
    from scripts.whu_cache_compat import CACHE_CAPACITY, install_whu_cache_compat

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    print(
        "WHU compatibility: training labels=int64, "
        f"per-worker full-image cache capacity={CACHE_CAPACITY}"
    )
    sys.argv = [
        str(OFFICIAL_TRAINER),
        *translate_launcher_arguments(sys.argv[1:]),
    ]
    source = OFFICIAL_TRAINER.read_text(encoding="utf-8")
    compatible_source = single_rank_eval_compatible_source(source)
    trainer_globals = {
        "__name__": "__main__",
        "__file__": str(OFFICIAL_TRAINER),
        "__package__": None,
        "__cached__": None,
    }
    print("single_rank_eval_gpu_all_gather=BYPASSED")
    exec(
        compile(compatible_source, str(OFFICIAL_TRAINER), "exec"),
        trainer_globals,
    )


if __name__ == "__main__":
    main()
