"""Executable checks for the EarthMiss data contract."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT / "tasks" / "segmentation" / "datasets" / "EarthMiss_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("earthmiss_dataset_under_test", MODULE_PATH)
EARTHMISS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EARTHMISS
SPEC.loader.exec_module(EARTHMISS)

EarthMiss_Dataset = EARTHMISS.EarthMiss_Dataset

sys.path.insert(0, str(REPO_ROOT / "tasks" / "segmentation"))
from datasets import EARTHMISS_CITIES, build_dataset  # noqa: E402
from utils.earthmiss_metrics import EarthMissMetrics  # noqa: E402


def _save_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _write_tile(root: Path, city: str, tile_id: str, rgb_value: int, sar_value: int):
    rgb = np.full((4, 5, 3), rgb_value, dtype=np.uint8)
    sar = np.full((4, 5), sar_value, dtype=np.uint8)
    mask = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 1],
            [2, 3, 4, 5, 6],
            [7, 8, 0, 1, 2],
        ],
        dtype=np.uint8,
    )
    _save_tiff(root / city / "images" / "RGB" / f"{tile_id}.tif", rgb)
    _save_tiff(root / city / "images" / "SAR" / f"{tile_id}_SAR.tif", sar)
    _save_tiff(root / city / "masks" / f"{tile_id}_mask.tif", mask)


def _dataset(root: Path, *, data_type="val", with_sar=True, window_size=(4, 5)):
    template = str(root / "{}")
    return EarthMiss_Dataset(
        citys=["City"],
        rgb_dir=template + "/images/RGB",
        sar_dir=template + "/images/SAR" if with_sar else None,
        label_dir=template + "/masks",
        data_type=data_type,
        window_size=window_size,
        normalize_type=None,
        cache_size=2,
    )


class EarthMissDatasetTest(unittest.TestCase):
    def test_manifest_pairs_by_tile_id_and_has_real_epoch_length(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tile(root, "City", "tile_b", 255, 255)
            _write_tile(root, "City", "tile_a", 0, 0)

            dataset = _dataset(root)

            self.assertEqual(len(dataset), 2)
            self.assertEqual([sample.tile_id for sample in dataset.samples], ["tile_a", "tile_b"])
            self.assertTrue(dataset.samples[0].sar_path.endswith("tile_a_SAR.tif"))
            self.assertTrue(dataset.samples[0].label_path.endswith("tile_a_mask.tif"))

    def test_scale_label_mapping_and_sar_normalization_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tile(root, "City", "tile", 255, 255)

            rgb, sar, label = _dataset(root)[0]

            self.assertEqual(rgb.dtype, torch.float32)
            self.assertTrue(torch.equal(rgb, torch.ones_like(rgb)))
            self.assertEqual(label.dtype, torch.int64)
            self.assertEqual(label[0].tolist(), [8, 0, 1, 2, 3])
            expected_sar = (1.0 - EARTHMISS.EARTHMISS_SAR_MEAN[0]) / EARTHMISS.EARTHMISS_SAR_STD[0]
            self.assertTrue(torch.allclose(sar, torch.full_like(sar, expected_sar)))

    def test_rgb_only_mode_does_not_require_a_sar_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tile(root, "City", "tile", 127, 63)

            rgb, label = _dataset(root, with_sar=False)[0]

            self.assertEqual(tuple(rgb.shape), (3, 4, 5))
            self.assertEqual(tuple(label.shape), (4, 5))

    def test_mismatched_pairing_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tile(root, "City", "tile", 127, 63)
            (root / "City" / "images" / "SAR" / "tile_SAR.tif").unlink()
            _save_tiff(
                root / "City" / "images" / "SAR" / "other_SAR.tif",
                np.zeros((4, 5), dtype=np.uint8),
            )

            with self.assertRaisesRegex(ValueError, "pairing mismatch"):
                _dataset(root)

    def test_train_crop_keeps_shapes_and_valid_label_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tile(root, "City", "tile", 127, 63)

            dataset = _dataset(root, data_type="train", window_size=(3, 3))
            rgb, sar, label = dataset[0]

            self.assertEqual(tuple(rgb.shape), (3, 3, 3))
            self.assertEqual(tuple(sar.shape), (1, 3, 3))
            self.assertEqual(tuple(label.shape), (3, 3))
            self.assertTrue(set(label.unique().tolist()).issubset(set(range(9))))

    def test_formal_geometric_transforms_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_tile(root, "City", "tile", 127, 63)
            dataset = _dataset(root, data_type="train", window_size=(4, 5))
            rgb, sar, label = dataset._load_sample(0)

            with (
                patch.object(EARTHMISS.random, "randint", return_value=0),
                patch.object(
                    EARTHMISS.random,
                    "random",
                    side_effect=(0.1, 0.1, 0.1),
                ) as probability,
                patch.object(EARTHMISS.random, "randrange", return_value=1),
            ):
                transformed = dataset._train_transform(rgb, sar, label)

            expected_rgb = np.rot90(np.flip(np.flip(rgb, 1), 0), 1)
            expected_sar = np.rot90(np.flip(np.flip(sar, 1), 0), 1)
            expected_label = np.rot90(np.flip(np.flip(label, 1), 0), 1)
            np.testing.assert_array_equal(transformed[0], expected_rgb)
            np.testing.assert_array_equal(transformed[1], expected_sar)
            np.testing.assert_array_equal(transformed[2], expected_label)
            self.assertEqual(probability.call_count, 3)

    def test_evaluator_pools_confusion_and_ignores_no_data(self):
        evaluator = EarthMissMetrics(num_classes=3, ignore_index=8)
        evaluator.update(
            torch.tensor([[0, 1], [2, 0]]),
            torch.tensor([[0, 1], [8, 2]]),
        )
        evaluator.update(
            torch.tensor([[2, 1]]),
            torch.tensor([[2, 0]]),
        )

        metrics = evaluator.compute()

        self.assertEqual(metrics["valid_pixels"], 5)
        self.assertEqual(
            metrics["confusion"],
            [[1, 1, 0], [0, 1, 0], [1, 0, 1]],
        )
        self.assertAlmostEqual(metrics["mIoU"], (1 / 3 + 1 / 2 + 1 / 2) / 3)

    def test_builder_uses_the_released_city_held_out_validation_split(self):
        self.assertEqual(
            EARTHMISS_CITIES["val"],
            ["Australia-PortHedland", "America-Pake", "Russian-Engels"],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for city in EARTHMISS_CITIES["val"]:
                _write_tile(root, city, "tile", 127, 63)

            dataset = build_dataset(
                "EarthMiss",
                "val",
                dataset_root=str(root),
                model_name="DINOv3",
                backbone_type="dinov3_vits16",
                modality="multi",
            )

            self.assertEqual(len(dataset), 3)
            self.assertEqual(
                [sample.city for sample in dataset.samples],
                EARTHMISS_CITIES["val"],
            )


if __name__ == "__main__":
    unittest.main()
