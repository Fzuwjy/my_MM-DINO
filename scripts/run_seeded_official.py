"""Run the unmodified official trainer with deterministic model initialization.

This launcher sets seed 42 before the official script constructs the model.  The
official script then executes unchanged and calls ``set_seed(42)`` again at its
original location, preserving its data-loader and augmentation seed sequence.
All command-line arguments are passed through verbatim.
"""

from __future__ import annotations

import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TRAINER = REPO_ROOT / "tasks" / "segmentation" / "train_multi.py"


def seed_model_initialization() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    sys.argv = [str(OFFICIAL_TRAINER), *sys.argv[1:]]
    runpy.run_path(str(OFFICIAL_TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
