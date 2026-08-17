from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import hashlib
import json

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from scripts.diagnose_whu_multideployment_gradients import (  # noqa: E402
    PAIRS,
    _rename_pair_statistics,
    parse_args as parse_gradient_args,
)
from scripts.diagnose_whu_multideployment_oracle import (  # noqa: E402
    _variant,
    parse_args as parse_oracle_args,
    validate_args as validate_oracle_args,
)
from scripts.diagnose_whu_multideployment_recoverability import (  # noqa: E402
    parse_args as parse_recoverability_args,
)
from scripts.prepare_whu_development_split import (  # noqa: E402
    choose_validation_groups,
    map_sheet_group,
)
from scripts.whu_multideployment_diagnostics_common import (  # noqa: E402
    ENDPOINTS,
    IGNORE_INDEX,
    MISSING_ENDPOINTS,
    NUM_CLASSES,
    deterministic_coordinate_subset,
    endpoint_recoverability_examples,
    exact_whu_class_occupancy,
    fixed_grid_coordinates,
    split_names,
)


def test_three_endpoint_contract_is_explicit():
    assert ENDPOINTS == ("full", "rgb", "sar")
    assert MISSING_ENDPOINTS == ("rgb", "sar")
    assert PAIRS == (("full", "rgb"), ("full", "sar"), ("rgb", "sar"))


def test_oracle_defaults_cover_both_missing_endpoints_without_selection_claim():
    args = parse_oracle_args([])
    assert args.alphas == [1.0]
    assert args.smoke_scenes == 0
    assert args.max_crops_per_scene == 0
    assert _variant("rgb", "frm.P5", 1.0) == "rgb|frm.P5@alpha=1.00"
    assert _variant("sar", "frm.P5", 1.0) == "sar|frm.P5@alpha=1.00"
    validate_oracle_args(args)


def test_oracle_rejects_multistage_response_curve():
    args = parse_oracle_args(
        ["--stages", "adapter.P2", "adapter.P3", "--alphas", "0.5"]
    )
    with pytest.raises(ValueError, match="frozen to alpha=1"):
        validate_oracle_args(args)


def test_fixed_grid_is_disjoint_and_excludes_border_remainders():
    coordinates = fixed_grid_coordinates(3704, 5556, window_size=512)
    assert len(coordinates) == 7 * 10
    assert coordinates[0] == (0, 512, 0, 512)
    assert coordinates[-1] == (3072, 3584, 4608, 5120)
    covered = set()
    for y1, y2, x1, x2 in coordinates:
        key = (y1 // 512, x1 // 512)
        assert key not in covered
        covered.add(key)
        assert y2 - y1 == 512 and x2 - x1 == 512


def test_coordinate_subsample_is_deterministic_and_keeps_source_order():
    coordinates = fixed_grid_coordinates(1024, 1536, window_size=512)
    left = deterministic_coordinate_subset(coordinates, 3, 17)
    right = deterministic_coordinate_subset(coordinates, 3, 17)
    assert left == right
    assert len(left) == 3
    assert [coordinates.index(value) for value in left] == sorted(
        coordinates.index(value) for value in left
    )


def test_exact_occupancy_preserves_ignore_as_empty_area():
    target = torch.tensor(
        [[[0, 0, IGNORE_INDEX, IGNORE_INDEX], [0, 1, 1, IGNORE_INDEX],
          [2, 2, 2, 2], [2, 2, 2, 2]]]
    )
    occupancy = exact_whu_class_occupancy(target, (2, 2))
    assert occupancy.shape == (1, NUM_CLASSES, 2, 2)
    assert occupancy[0, 0, 0, 0].item() == pytest.approx(0.75)
    assert occupancy[0, 1, 0, 0].item() == pytest.approx(0.25)
    assert occupancy[0, :, 0, 1].sum().item() == pytest.approx(0.25)
    assert occupancy[0, 2, 1, 0].item() == pytest.approx(1.0)


def test_endpoint_recoverability_selects_only_wrong_endpoint_cells():
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    feature = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    full_logits = torch.zeros((1, NUM_CLASSES, 4, 4))
    endpoint_logits = torch.zeros_like(full_logits)
    full_logits[:, 0] = 4.0
    endpoint_logits[:, 0] = 4.0
    endpoint_logits[:, 1, :2, :2] = 8.0
    examples = endpoint_recoverability_examples(
        feature, full_logits, endpoint_logits, target,
    )
    assert examples.pure_cells == 4
    assert examples.endpoint_wrong_cells == 1
    assert examples.features.shape == (1, 2)
    assert examples.targets.tolist() == [1]
    assert examples.class_ids.tolist() == [0]


def test_pair_statistic_names_do_not_mislabel_optical_as_sar():
    values = _rename_pair_statistics({
        "parameter_tensors": 1,
        "active_both": 1,
        "full_only": 0,
        "sar_only": 0,
        "full_norm": 2.0,
        "sar_norm": 3.0,
        "sar_to_full_norm_ratio": 1.5,
        "dot": 1.0,
        "cosine": 0.1,
        "conflict": False,
        "negative_tensor_dot_fraction": 0.0,
    })
    assert values["left_norm"] == 2.0
    assert values["right_norm"] == 3.0
    assert values["right_to_left_norm_ratio"] == 1.5
    assert "sar_norm" not in values


def test_split_manifest_rejects_duplicates():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "split.txt"
        path.write_text("a.tif\na.tif\n", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            split_names(path)


def test_formal_defaults_are_frozen_for_gradient_and_recoverability():
    gradient = parse_gradient_args([])
    recoverability = parse_recoverability_args([])
    assert gradient.batches == 32
    assert set(gradient.bn_modes) == {"train", "eval"}
    assert recoverability.train_scenes == 16
    assert recoverability.test_scenes == 0
    assert recoverability.crops_per_scene == 8
    assert recoverability.max_examples == 100_000


def test_whu_map_sheet_group_excludes_final_tile_id():
    assert map_sheet_group("NH49E006014.tif") == "NH49E006"
    assert map_sheet_group("NI49E024020.tif") == "NI49E024"
    with pytest.raises(ValueError, match="unrecognized"):
        map_sheet_group("tile.tif")


def test_development_split_selection_is_exact_group_disjoint_and_deterministic():
    groups = {
        "NH49E001": 2,
        "NH49E002": 2,
        "NH50E001": 2,
        "NH50E002": 2,
        "NI49E001": 2,
        "NI49E002": 2,
    }
    histograms = {
        group: torch.tensor([10 + index, 20, 30, 40, 50, 60, 70]).numpy()
        for index, group in enumerate(groups)
    }
    left, left_objective = choose_validation_groups(
        histograms, groups, target_scenes=6, trials=500, seed=17,
    )
    right, right_objective = choose_validation_groups(
        histograms, groups, target_scenes=6, trials=500, seed=17,
    )
    assert left == right
    assert left_objective == right_objective
    assert sum(groups[group] for group in left) == 6
    assert {group[:4] for group in left} == {"NH49", "NH50", "NI49"}


def test_frozen_development_split_partitions_official_train_without_group_leakage():
    split_dir = REPO_ROOT / "splits" / "whu"
    official = set(split_names(split_dir / "official_train.txt"))
    train = split_names(split_dir / "development_train.txt")
    val = split_names(split_dir / "development_val.txt")
    manifest = json.loads(
        (split_dir / "development_manifest.json").read_text(encoding="utf-8")
    )
    assert len(train) == 64 and len(val) == 16
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == official
    assert {map_sheet_group(name) for name in train}.isdisjoint(
        map_sheet_group(name) for name in val
    )
    assert manifest["official_test_was_accessed"] is False
    assert manifest["group_overlap"] == []
    for filename in ("development_train.txt", "development_val.txt"):
        digest = hashlib.sha256((split_dir / filename).read_bytes()).hexdigest()
        assert digest == manifest["files"][filename]
