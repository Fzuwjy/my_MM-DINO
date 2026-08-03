"""MetaRS-clean training configuration.

This derives from the pinned official EarthMiss configuration at runtime.  The
only protocol correction is that ``data.val`` (also consumed by MetaRS when it
builds the MMR mask matrix) points to the released validation cities instead
of the test cities.  The per-rank train batch size is the effective value from
the official two-GPU launcher: 4 x 2 ranks = global batch size 8.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

from configs.baseline.MetaRS import config as _official_config
from configs.metadata.EarthMiss import test_cities, train_cities, val_cities


dataset_root = Path(
    os.environ.get(
        "EARTHMISS_ROOT", "/root/autodl-tmp/mm-dino/datasets/EarthMiss"
    )
).resolve()
train_batch_per_rank = int(os.environ.get("METARS_TRAIN_BATCH_PER_RANK", "4"))


def _dirs(cities: list[str], leaf: str) -> list[str]:
    return [str(dataset_root / city / leaf) for city in cities]


config = copy.deepcopy(_official_config)

config["data"]["train"]["params"]["image_dir"] = _dirs(train_cities, "images")
config["data"]["train"]["params"]["mask_dir"] = _dirs(train_cities, "masks")
config["data"]["train"]["params"]["batch_size"] = train_batch_per_rank

config["data"]["val"]["params"]["image_dir"] = _dirs(val_cities, "images")
config["data"]["val"]["params"]["mask_dir"] = _dirs(val_cities, "masks")

config["data"]["test"]["params"]["image_dir"] = _dirs(test_cities, "images")
config["data"]["test"]["params"]["mask_dir"] = _dirs(test_cities, "masks")

# MetaRS.conduct_mask_matrix() reads this nested loader configuration at step
# 1600.  Keep it tied to the corrected validation loader.
config["model"]["params"]["data"] = config["data"]["val"]
