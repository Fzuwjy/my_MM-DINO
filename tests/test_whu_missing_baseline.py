from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets.WHU_dataset import WHU_Dataset  # noqa: E402
from losses import DiceLoss, JointLoss, SoftCrossEntropyLoss  # noqa: E402
from scripts.train_whu_missing_baseline import (  # noqa: E402
    parse_args,
    prepare_training_label,
    train_state,
)
from utils.pooled_segmentation_metrics import (  # noqa: E402
    PooledSegmentationMetrics,
)


def write_whu_sample(root, channels=4):
    for directory in ("optical", "sar", "lbl"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    optical = np.zeros((8, 9, channels), dtype=np.uint8)
    for channel in range(channels):
        optical[..., channel] = 10 * (channel + 1)
    sar = np.full((8, 9), 50, dtype=np.uint8)
    label = np.full((8, 9), 10, dtype=np.uint8)
    label[0, 0] = 0
    label[0, 1] = 70
    tifffile.imwrite(root / "optical" / "tile.tif", optical, photometric="rgb")
    tifffile.imwrite(root / "sar" / "tile.tif", sar)
    tifffile.imwrite(root / "lbl" / "tile.tif", label)


def build_test_dataset(root, optical_bands="nir-r-g"):
    return WHU_Dataset(
        filenames=["tile.tif"],
        rgb_dir=str(root / "optical" / "{}"),
        label_dir=str(root / "lbl" / "{}"),
        sar_dir=str(root / "sar" / "{}"),
        data_type="test",
        normalize_type=None,
        optical_bands=optical_bands,
    )


def test_whu_nir_r_g_selection_and_single_channel_sar(tmp_path):
    write_whu_sample(tmp_path)
    dataset = build_test_dataset(tmp_path)
    optical, sar, label = dataset[0]

    assert dataset.optical_band_indices == (3, 0, 1)
    assert optical.shape == (3, 8, 9)
    assert sar.shape == (1, 8, 9)
    torch.testing.assert_close(
        optical[:, 1, 1], torch.tensor([40.0, 10.0, 20.0]) / 255.0
    )
    torch.testing.assert_close(sar[:, 1, 1], torch.tensor([50.0]) / 255.0)
    assert label[0, 0].item() == 7
    assert label[0, 1].item() == 6
    assert label[1, 1].item() == 0


def test_whu_nir_r_g_requires_four_bands(tmp_path):
    write_whu_sample(tmp_path, channels=3)
    dataset = build_test_dataset(tmp_path)
    with pytest.raises(ValueError, match="requires at least 4 bands"):
        dataset[0]


def test_whu_missing_baseline_states_are_explicit():
    assert train_state("A", random_stub(0.9)) == "sar"
    assert train_state("B", random_stub(0.1)) == "full"
    assert train_state("C", random_stub(0.49)) == "sar"
    assert train_state("C", random_stub(0.50)) == "full"


def test_whu_smoke_flag_is_explicit():
    args = parse_args(["--run", "A", "--smoke-only"])
    assert args.smoke_only
    assert not args.audit_only


class random_stub:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def test_whu_training_label_matches_joint_loss_contract():
    label = prepare_training_label(
        torch.tensor([[[0, 1], [6, 7]]], dtype=torch.int32),
        torch.device("cpu"),
    )
    logits = torch.randn(1, 7, 2, 2, requires_grad=True)
    criterion = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=7),
        DiceLoss(smooth=0.05, ignore_index=7),
        1.0,
        1.0,
    )

    assert label.dtype == torch.int64
    loss = criterion(logits, label)
    assert torch.isfinite(loss)
    loss.backward()


def test_pooled_metrics_use_all_fixed_classes():
    evaluator = PooledSegmentationMetrics(num_classes=3, ignore_index=3)
    target = torch.tensor([[0, 1, 2, 3]])
    prediction = torch.tensor([[0, 0, 2, 1]])
    evaluator.update(prediction, target)
    result = evaluator.compute()

    assert result["gt_present_class_ids"] == [0, 1, 2]
    assert result["class_iou"] == pytest.approx([0.5, 0.0, 1.0])
    assert result["mIoU"] == pytest.approx(0.5)
    assert result["mF1"] == pytest.approx((2 / 3 + 0 + 1) / 3)
