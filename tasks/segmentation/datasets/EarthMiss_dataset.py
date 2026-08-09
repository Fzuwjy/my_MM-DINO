import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from skimage.io import imread


palette = {
    0: (255, 255, 255),
    1: (255, 0, 0),
    2: (255, 255, 0),
    3: (0, 0, 255),
    4: (159, 129, 183),
    5: (0, 255, 0),
    6: (255, 195, 128),
    7: (165, 0, 165),
    8: (0, 0, 0),
}

invert_palette = {value: key for key, value in palette.items()}

# The ViT-S/LVD backbone stays on its pretraining input convention.  This is a
# deliberate MM-DINO choice; it is not the dataset-specific MetaRS RGB policy.
DINOV3_LVD_RGB_MEAN = (0.485, 0.456, 0.406)
DINOV3_LVD_RGB_STD = (0.229, 0.224, 0.225)

# Statistics released with the EarthMiss/MetaRS metadata, expressed on [0, 1].
EARTHMISS_SAR_MEAN = (63.30051921735858 / 255.0,)
EARTHMISS_SAR_STD = (68.20405016 / 255.0,)


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


@dataclass(frozen=True)
class EarthMissSample:
    city: str
    tile_id: str
    rgb_path: str
    label_path: str
    sar_path: Optional[str]


def _index_tiffs(directory: str) -> dict[str, str]:
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise FileNotFoundError(f"EarthMiss directory does not exist: {directory}")

    indexed = {}
    for path in sorted(directory_path.glob("*.tif")):
        key = path.stem
        if key in indexed:
            raise ValueError(f"Duplicate EarthMiss tile id {key!r} in {directory}")
        indexed[key] = str(path)
    return indexed


def _match_modality(reference_ids, candidates, suffix, modality_name, city):
    matched = {}
    consumed = set()
    missing = []
    ambiguous = []
    for tile_id in sorted(reference_ids):
        candidate_ids = (tile_id, f"{tile_id}{suffix}")
        present = [candidate_id for candidate_id in candidate_ids if candidate_id in candidates]
        if not present:
            missing.append(tile_id)
            continue
        if len(present) != 1:
            ambiguous.append((tile_id, present))
            continue
        candidate_id = present[0]
        if candidate_id in consumed:
            ambiguous.append((tile_id, present))
            continue
        consumed.add(candidate_id)
        matched[tile_id] = candidates[candidate_id]

    extra = sorted(set(candidates) - consumed)
    if missing or ambiguous or extra:
        raise ValueError(
            f"EarthMiss pairing mismatch in {city} for {modality_name}: "
            f"missing={missing[:5]}, ambiguous={ambiguous[:5]}, extra={extra[:5]}"
        )
    return matched


def _paired_city_samples(city, rgb_dir, label_dir, sar_dir=None):
    rgb = _index_tiffs(rgb_dir.format(city))
    raw_labels = _index_tiffs(label_dir.format(city))
    raw_sar = _index_tiffs(sar_dir.format(city)) if sar_dir is not None else None

    reference_ids = set(rgb)
    labels = _match_modality(reference_ids, raw_labels, "_mask", "mask", city)
    sar = (
        _match_modality(reference_ids, raw_sar, "_SAR", "SAR", city)
        if raw_sar is not None
        else None
    )

    return [
        EarthMissSample(
            city=city,
            tile_id=tile_id,
            rgb_path=rgb[tile_id],
            label_path=labels[tile_id],
            sar_path=sar[tile_id] if sar is not None else None,
        )
        for tile_id in sorted(reference_ids)
    ]


class EarthMiss_Dataset(torch.utils.data.Dataset):
    """Paired EarthMiss RGB/SAR semantic-segmentation tiles."""

    def __init__(
        self,
        citys,
        rgb_dir,
        label_dir,
        data_type,
        window_size=(224, 224),
        normalize_type=None,
        sar_dir=None,
        cache_size=500,
        apply_train_transform=None,
    ):
        super().__init__()
        if data_type not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported EarthMiss split: {data_type!r}")

        self.data_type = data_type
        self.apply_train_transform = (
            data_type == "train"
            if apply_train_transform is None
            else bool(apply_train_transform)
        )
        if data_type != "train" and self.apply_train_transform:
            raise ValueError(
                "train transforms can only be enabled for the EarthMiss train split"
            )
        self.window_size = tuple(window_size)
        if len(self.window_size) != 2 or any(size <= 0 for size in self.window_size):
            raise ValueError("EarthMiss window_size must contain two positive values")
        if self.apply_train_transform and self.window_size[0] != self.window_size[1]:
            raise ValueError(
                "EarthMiss training crop must be square because RandomRotate90 "
                "can exchange height and width"
            )
        self.cache_size = cache_size
        self.samples = []
        for city in citys:
            self.samples.extend(_paired_city_samples(city, rgb_dir, label_dir, sar_dir))
        if not self.samples:
            raise ValueError(f"No EarthMiss samples found for split {data_type!r}")

        # Keep the public lists used by older scripts, but derive all three from one manifest.
        self.rgb_files = [sample.rgb_path for sample in self.samples]
        self.label_files = [sample.label_path for sample in self.samples]
        self.sar_files = [sample.sar_path for sample in self.samples if sample.sar_path]

        self.rgb_cache = LRUCache(cache_size)
        self.label_cache = LRUCache(cache_size)
        self.sar_cache = LRUCache(cache_size)

        if normalize_type == "geo":
            self.imagenet_mean = (0.430, 0.411, 0.296)
            self.imagenet_std = (0.213, 0.156, 0.143)
            self.rgb_normalization = "dinov3_satellite"
        elif normalize_type == "common":
            self.imagenet_mean = DINOV3_LVD_RGB_MEAN
            self.imagenet_std = DINOV3_LVD_RGB_STD
            self.rgb_normalization = "dinov3_lvd_imagenet"
        else:
            self.imagenet_mean = None
            self.imagenet_std = None
            self.rgb_normalization = "none"
        self.sar_mean = EARTHMISS_SAR_MEAN
        self.sar_std = EARTHMISS_SAR_STD

    def __len__(self):
        # One logical epoch visits every tile once; DataLoader shuffling controls order.
        return len(self.samples)

    @staticmethod
    def _normalize(tensor, mean, std):
        mean_tensor = tensor.new_tensor(mean).view(-1, 1, 1)
        std_tensor = tensor.new_tensor(std).view(-1, 1, 1)
        return tensor.sub(mean_tensor).div(std_tensor)

    @staticmethod
    def _read_sensor(path, name):
        array = imread(path)
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.complexfloating
        ):
            raise ValueError(
                f"EarthMiss {name} must be real-valued, got {array.dtype} at {path}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"EarthMiss {name} contains non-finite values at {path}")
        # Released RGB/SAR TIFFs are float32 in their original radiometric
        # 0-255-scale convention and may contain values outside [0, 255].
        # Preserve those values; normalization below is an affine transform,
        # not a clipping or uint8 conversion.
        return array.astype(np.float32, copy=False)

    @staticmethod
    def _read_label(path):
        raw = imread(path)
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"EarthMiss mask must be integer-valued, got {raw.dtype} at {path}")
        if raw.ndim != 2:
            raise ValueError(f"EarthMiss mask must be 2-D, got shape {raw.shape} at {path}")
        values = np.unique(raw)
        invalid = values[(values < 0) | (values > 8)]
        if invalid.size:
            raise ValueError(f"EarthMiss mask has invalid raw labels {invalid.tolist()} at {path}")
        # Raw 0 is no-data; raw 1..8 become the eight training classes 0..7.
        return np.where(raw == 0, 8, raw - 1).astype(np.int64, copy=False)

    def _load_sample(self, idx):
        sample = self.samples[idx]

        rgb = self.rgb_cache.get(idx)
        if rgb is None:
            rgb = self._read_sensor(sample.rgb_path, "RGB")
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(
                    f"EarthMiss RGB must have shape HxWx3, got {rgb.shape} at {sample.rgb_path}"
                )
            self.rgb_cache.put(idx, rgb)

        label = self.label_cache.get(idx)
        if label is None:
            label = self._read_label(sample.label_path)
            self.label_cache.put(idx, label)

        sar = None
        if sample.sar_path is not None:
            sar = self.sar_cache.get(idx)
            if sar is None:
                sar = self._read_sensor(sample.sar_path, "SAR")
                if sar.ndim == 3 and sar.shape[2] == 1:
                    sar = sar[:, :, 0]
                if sar.ndim != 2:
                    raise ValueError(
                        f"EarthMiss SAR must have shape HxW, got {sar.shape} at {sample.sar_path}"
                    )
                self.sar_cache.put(idx, sar)

        spatial_shapes = {rgb.shape[:2], label.shape[:2]}
        if sar is not None:
            spatial_shapes.add(sar.shape[:2])
        if len(spatial_shapes) != 1:
            raise ValueError(f"EarthMiss modalities are not aligned for {sample.city}/{sample.tile_id}")
        return rgb, sar, label

    def _train_transform(self, rgb, sar, label):
        crop_h, crop_w = self.window_size
        image_h, image_w = rgb.shape[:2]
        if image_h < crop_h or image_w < crop_w:
            raise ValueError(
                f"EarthMiss tile {image_h}x{image_w} is smaller than crop {crop_h}x{crop_w}"
            )

        top = random.randint(0, image_h - crop_h)
        left = random.randint(0, image_w - crop_w)
        row = slice(top, top + crop_h)
        col = slice(left, left + crop_w)
        rgb = rgb[row, col]
        label = label[row, col]
        sar = sar[row, col] if sar is not None else None

        # The formal EarthMiss config applies all three transforms independently.
        if random.random() < 0.5:
            rgb, label = np.flip(rgb, 1), np.flip(label, 1)
            sar = np.flip(sar, 1) if sar is not None else None
        if random.random() < 0.5:
            rgb, label = np.flip(rgb, 0), np.flip(label, 0)
            sar = np.flip(sar, 0) if sar is not None else None
        if random.random() < 0.5:
            turns = random.randrange(4)
            rgb, label = np.rot90(rgb, turns), np.rot90(label, turns)
            sar = np.rot90(sar, turns) if sar is not None else None
        return rgb, sar, label

    def __getitem__(self, idx):
        rgb, sar, label = self._load_sample(idx)
        if self.apply_train_transform:
            rgb, sar, label = self._train_transform(rgb, sar, label)

        # Make the input scale explicit before normalization.
        rgb = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
        rgb = torch.from_numpy(rgb).div_(255.0)
        if self.imagenet_mean is not None:
            rgb = self._normalize(rgb, self.imagenet_mean, self.imagenet_std)

        label = torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64))
        if sar is None:
            return rgb, label

        # ``sar`` may be the exact array stored in the LRU cache.  A full-size
        # validation tile is already contiguous, so ``ascontiguousarray`` can
        # otherwise return a view backed by that cache entry.  The in-place
        # scaling below would then divide the cached tile by 255 again on every
        # validation pass (persistent DataLoader workers keep the cache alive).
        sar = np.array(sar[None, :, :], dtype=np.float32, order="C", copy=True)
        sar = torch.from_numpy(sar).div_(255.0)
        sar = self._normalize(sar, self.sar_mean, self.sar_std)
        return rgb, sar, label

    @staticmethod
    def convert_from_color(arr_3d, palette=invert_palette):
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
        for color, class_id in palette.items():
            matches = np.all(arr_3d == np.array(color).reshape(1, 1, 3), axis=2)
            arr_2d[matches] = class_id
        return arr_2d

    @staticmethod
    def get_random_pos(img, window_shape):
        crop_h, crop_w = window_shape
        image_h, image_w = img.shape[:2]
        if image_h < crop_h or image_w < crop_w:
            raise ValueError("Image is smaller than the requested crop")
        top = random.randint(0, image_h - crop_h)
        left = random.randint(0, image_w - crop_w)
        return top, top + crop_h, left, left + crop_w
