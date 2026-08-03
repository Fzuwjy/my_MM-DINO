"""Faithful runtime-path version of the released MetaRS training config.

This deliberately preserves the public implementation's behavior: ``data.val``
and the MMR mask-matrix loader both use the Test cities.  It is the artifact
reproduction baseline, not the corrected MetaRS-clean protocol.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

from configs.baseline.MetaRS import config as _official_config
from configs.metadata.EarthMiss import test_cities, train_cities


dataset_root = Path(
    os.environ.get(
        "EARTHMISS_ROOT", "/root/autodl-tmp/mm-dino/datasets/EarthMiss"
    )
).resolve()
train_batch_per_rank = int(os.environ.get("METARS_TRAIN_BATCH_PER_RANK", "4"))
test_batch_per_rank = int(os.environ.get("METARS_TEST_BATCH_PER_RANK", "16"))
save_ckpt_interval_epoch = int(
    os.environ.get("METARS_SAVE_CKPT_INTERVAL_EPOCH", "20")
)


def _dirs(cities: list[str], leaf: str) -> list[str]:
    return [str(dataset_root / city / leaf) for city in cities]


config = copy.deepcopy(_official_config)

config["data"]["train"]["params"]["image_dir"] = _dirs(train_cities, "images")
config["data"]["train"]["params"]["mask_dir"] = _dirs(train_cities, "masks")
config["data"]["train"]["params"]["batch_size"] = train_batch_per_rank

config["data"]["val"]["params"]["image_dir"] = _dirs(test_cities, "images")
config["data"]["val"]["params"]["mask_dir"] = _dirs(test_cities, "masks")

config["data"]["test"]["params"]["image_dir"] = _dirs(test_cities, "images")
config["data"]["test"]["params"]["mask_dir"] = _dirs(test_cities, "masks")
config["data"]["test"]["params"]["batch_size"] = test_batch_per_rank
config["train"]["save_ckpt_interval_epoch"] = save_ckpt_interval_epoch

config["model"]["params"]["data"] = config["data"]["val"]
