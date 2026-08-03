from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = ROOT / "scripts" / "train_metars_clean.py"
    spec = importlib.util.spec_from_file_location("train_metars_clean", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_effective_official_recipe_constants() -> None:
    runner = _load_runner()
    assert runner.EXPECTED_COUNTS == {"train": 2641, "val": 277, "test": 437}
    assert runner.RESNET50_FILENAME == "resnet50-19c8e357.pth"
    assert runner.RESNET50_SHA256_PREFIX == "19c8e357"


def test_decode_smoke_covers_every_split() -> None:
    runner = _load_runner()
    assert ("train", "val", "test") in runner.decode_samples.__code__.co_consts


def test_clean_config_changes_val_and_effective_batch_only() -> None:
    text = (ROOT / "configs" / "metars_clean.py").read_text(encoding="utf-8")
    assert 'config["data"]["val"]["params"]["image_dir"] = _dirs(val_cities, "images")' in text
    assert 'config["model"]["params"]["data"] = config["data"]["val"]' in text
    assert 'config["data"]["train"]["params"]["batch_size"] = 4' in text
