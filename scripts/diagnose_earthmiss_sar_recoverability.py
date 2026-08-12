"""Test whether paired Full corrections are linearly predictable from SAR P5.

The frozen Run C model supplies canonical Full/SAR logits and the canonical SAR
post-FRM P5 feature.  On semantically pure P5 cells where SAR is wrong, a fixed
linear probe predicts whether Full is correct.  Train cities fit the probe and
the three held-out Val cities evaluate it.  Class-conditional prevalence and a
within-class shuffled-target probe prevent ordinary class difficulty from being
misreported as Full-specific recoverability.  MM-DINO is never updated and Test
is never accessed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import EARTHMISS_CITIES, build_dataset  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402

from scripts.diagnose_earthmiss_missing_v3 import load_frozen_model  # noqa: E402
from scripts.diagnose_earthmiss_scale_transition import (  # noqa: E402
    EXPECTED_RUN_C_CHECKPOINT_SHA256,
    EXPECTED_RUN_C_EPOCH,
    EXPECTED_RUN_C_SEED,
    EXPECTED_VAL_TILES,
    assert_buffers_unchanged,
    iter_crop_batches,
    sliding_window_coordinates,
    snapshot_batchnorm_buffers,
    verify_cached_forward_equivalence,
)
from scripts.earthmiss_causal_diagnostics_common import (  # noqa: E402
    RecoverabilityExamples,
    recoverability_examples,
    stable_seed,
)
from scripts.earthmiss_scale_transition_common import write_json_exclusive  # noqa: E402
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
)


SCHEMA = "earthmiss_sar_full_recoverability_probe_v1"
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-causal-diagnostics/"
    "run-c-e15-sar-p5-recoverability.json"
)
EXPECTED_TRAIN_TILES = 2641
WINDOW_SIZE = 512
PROBE_STRIDE = 512
PURITY_THRESHOLD = 0.75
TRAIN_TILES_PER_CITY = 64
MAX_EXAMPLES_PER_SPLIT = 200_000
SEED = 20260812


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--train-tiles-per-city", type=int, default=TRAIN_TILES_PER_CITY)
    parser.add_argument(
        "--val-tiles",
        type=int,
        default=0,
        help="Non-formal deterministic Val prefix; zero means all 277 tiles.",
    )
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES_PER_SPLIT)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-cached-equivalence-check", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.train_tiles_per_city <= 0:
        raise ValueError("--train-tiles-per-city must be positive")
    if args.val_tiles < 0 or args.num_workers < 0:
        raise ValueError("Val/worker counts must be non-negative")
    if args.max_examples < 100:
        raise ValueError("--max-examples must be at least 100")


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class _IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        rgb, sar, target = self.dataset[source_index]
        return source_index, rgb, sar, target


class _FRMP5Capture:
    def __init__(self, model) -> None:
        self.model = model
        self.handle = None
        self.active = False
        self.calls = 0
        self.value = None

    def __enter__(self):
        self.handle = self.model.decoder.frm.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _hook(self, module, inputs, output):
        if not self.active:
            return None
        self.calls += 1
        if not isinstance(output, (list, tuple)) or len(output) != 4:
            raise RuntimeError("FRM must return four scales")
        value = output[3].detach().clone()
        if self.calls == 1:
            self.value = value
        elif self.calls == 2:
            if not torch.equal(self.value, value):
                raise RuntimeError("duplicate canonical FRM P5 slots differ")
        else:
            raise RuntimeError("FRM called more than twice")
        return None

    def run(self, forward):
        self.calls = 0
        self.value = None
        self.active = True
        try:
            logits = forward()
        finally:
            self.active = False
        if self.calls != 2 or self.value is None:
            raise RuntimeError("FRM P5 capture call contract changed")
        return logits, self.value


def _build_dataset(args: argparse.Namespace, split: str):
    return build_dataset(
        "EarthMiss",
        split,
        dataset_root=args.dataset_root,
        window_size=(WINDOW_SIZE, WINDOW_SIZE),
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
        apply_train_transform=False,
        cache_size=0,
    )


def _stratified_train_indices(dataset, per_city: int, seed: int) -> tuple[int, ...]:
    by_city: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(dataset.samples):
        by_city[str(sample.city)].append(index)
    result = []
    for city in EARTHMISS_CITIES["train"]:
        indices = list(by_city[city])
        random.Random(stable_seed(seed, city)).shuffle(indices)
        if len(indices) < per_city:
            raise RuntimeError(f"city {city} has fewer than {per_city} tiles")
        result.extend(indices[:per_city])
    return tuple(result)


def _subsample(
    features: torch.Tensor,
    targets: torch.Tensor,
    class_ids: torch.Tensor,
    city_ids: torch.Tensor,
    tile_ids: torch.Tensor,
    maximum: int,
    seed: int,
):
    if features.shape[0] <= maximum:
        return features, targets, class_ids, city_ids, tile_ids
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(features.shape[0], generator=generator)[:maximum]
    return (
        features[indices],
        targets[indices],
        class_ids[indices],
        city_ids[indices],
        tile_ids[indices],
    )


@torch.inference_mode()
def _extract_split(
    model,
    capture,
    dataset,
    indices: Sequence[int],
    args,
    device,
    city_to_id,
    equivalence_holder,
):
    subset = _IndexedSubset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    features = []
    targets = []
    class_ids = []
    city_ids = []
    tile_ids = []
    tile_keys = []
    pure_cells = 0
    sar_wrong_cells = 0
    crop_windows = 0
    for local_tile_id, (source_index, rgb, sar, target) in enumerate(
        tqdm(loader, desc=f"recoverability-{dataset.data_type}")
    ):
        index = int(source_index.item())
        sample = dataset.samples[index]
        city_id = city_to_id[str(sample.city)]
        tile_keys.append(f"{sample.city}/{sample.tile_id}")
        if target.shape[-2] % WINDOW_SIZE or target.shape[-1] % WINDOW_SIZE:
            raise RuntimeError(
                "recoverability probe requires native tiles divisible by 512 "
                "to guarantee non-overlapping coverage"
            )
        coordinates = sliding_window_coordinates(
            target.shape[-2],
            target.shape[-1],
            window_size=WINDOW_SIZE,
            stride=PROBE_STRIDE,
        )
        for _, rgb_cpu, sar_cpu, target_cpu in iter_crop_batches(
            rgb,
            sar,
            target,
            coordinates,
            batch_size=1,
        ):
            crop_windows += 1
            rgb_crop = rgb_cpu.to(device, non_blocking=True)
            sar_crop = sar_cpu.to(device, non_blocking=True)
            target_crop = target_cpu.to(device, non_blocking=True)
            backbone_outputs = model.extract_frozen_backbone_outputs(rgb_crop, sar_crop)
            if equivalence_holder[0] is None and not args.skip_cached_equivalence_check:
                equivalence_holder[0] = verify_cached_forward_equivalence(
                    model,
                    rgb_crop,
                    sar_crop,
                    backbone_outputs,
                )
            full_availability = canonical_availability(
                "full", batch_size=1, device=device
            )
            sar_availability = canonical_availability(
                "sar", batch_size=1, device=device
            )
            full_logits = model.forward_from_backbone_outputs(
                rgb_crop,
                sar_crop,
                backbone_outputs=backbone_outputs,
                availability=full_availability,
            )
            sar_logits, sar_p5 = capture.run(
                lambda: model.forward_from_backbone_outputs(
                    rgb_crop,
                    sar_crop,
                    backbone_outputs=backbone_outputs,
                    availability=sar_availability,
                )
            )
            examples: RecoverabilityExamples = recoverability_examples(
                sar_p5,
                full_logits,
                sar_logits,
                target_crop,
                purity_threshold=PURITY_THRESHOLD,
            )
            pure_cells += examples.pure_cells
            sar_wrong_cells += examples.sar_wrong_cells
            if examples.features.shape[0]:
                count = examples.features.shape[0]
                features.append(examples.features.to("cpu", torch.float16))
                targets.append(examples.targets.to("cpu", torch.int8))
                class_ids.append(examples.class_ids.to("cpu", torch.int8))
                city_ids.append(torch.full((count,), city_id, dtype=torch.int16))
                tile_ids.append(
                    torch.full((count,), local_tile_id, dtype=torch.int32)
                )
    if not features:
        raise RuntimeError(f"no recoverability examples extracted from {dataset.data_type}")
    joined = (
        torch.cat(features),
        torch.cat(targets).long(),
        torch.cat(class_ids).long(),
        torch.cat(city_ids).long(),
        torch.cat(tile_ids).long(),
    )
    joined = _subsample(*joined, args.max_examples, stable_seed(args.seed, dataset.data_type))
    return joined, {
        "tiles": len(indices),
        "crop_windows": crop_windows,
        "pure_cells": pure_cells,
        "sar_wrong_cells": sar_wrong_cells,
        "retained_examples": int(joined[0].shape[0]),
        "recoverable_prevalence": float(joined[1].float().mean()),
        "tile_keys": tile_keys,
    }


def _binary_metrics(target, score) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    result = {
        "n": int(target.size),
        "positives": int(target.sum()),
        "prevalence": float(target.mean()) if target.size else None,
        "average_precision": None,
        "roc_auc": None,
        "brier": None,
    }
    if not target.size:
        return result
    result["brier"] = float(brier_score_loss(target, score))
    if np.unique(target).size == 2:
        result["average_precision"] = float(average_precision_score(target, score))
        result["roc_auc"] = float(roc_auc_score(target, score))
    return result


def _fit_probe(features, targets, seed):
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if np.unique(targets).size != 2:
        raise RuntimeError("probe Train targets must contain both outcomes")
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-5,
            random_state=seed,
        ),
    )
    model.fit(features, targets)
    return model


def _within_class_shuffle(targets, class_ids, seed):
    result = np.asarray(targets).copy()
    generator = np.random.default_rng(seed)
    for class_id in range(8):
        indices = np.flatnonzero(np.asarray(class_ids) == class_id)
        result[indices] = result[generator.permutation(indices)]
    return result


def _class_prior_scores(train_target, train_class, eval_class):
    priors = np.empty(8, dtype=np.float64)
    global_prior = (float(np.sum(train_target)) + 1.0) / (len(train_target) + 2.0)
    for class_id in range(8):
        values = np.asarray(train_target)[np.asarray(train_class) == class_id]
        priors[class_id] = (
            (float(values.sum()) + 1.0) / (len(values) + 2.0)
            if len(values)
            else global_prior
        )
    return priors[np.asarray(eval_class)], priors.tolist()


def _evaluate_slices(
    target,
    scores,
    class_ids,
    city_ids,
    tile_ids,
    id_to_city,
    tile_keys,
):
    report = {"pooled": {}, "by_city": {}, "by_class": {}, "by_tile": {}}
    for name, values in scores.items():
        report["pooled"][name] = _binary_metrics(target, values)
    for city_id, city in sorted(id_to_city.items()):
        mask = city_ids == city_id
        report["by_city"][city] = {
            name: _binary_metrics(target[mask], values[mask])
            for name, values in scores.items()
        }
    for class_id in range(8):
        mask = class_ids == class_id
        report["by_class"][str(class_id)] = {
            name: _binary_metrics(target[mask], values[mask])
            for name, values in scores.items()
        }
    for tile_id, tile_key in enumerate(tile_keys):
        mask = tile_ids == tile_id
        report["by_tile"][tile_key] = {
            name: _binary_metrics(target[mask], values[mask])
            for name, values in scores.items()
        }
    return report


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite probe report: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("recoverability probe requires CUDA")
    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
    device = torch.device("cuda")
    model, checkpoint = load_frozen_model(
        args.checkpoint,
        "c",
        weights_path,
        device,
    )
    expected = {
        "sha256": EXPECTED_RUN_C_CHECKPOINT_SHA256,
        "epoch": EXPECTED_RUN_C_EPOCH,
        "seed": EXPECTED_RUN_C_SEED,
    }
    observed = {key: checkpoint.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"recoverability checkpoint mismatch: {observed} != {expected}")
    train_dataset = _build_dataset(args, "train")
    val_dataset = _build_dataset(args, "val")
    if len(train_dataset) != EXPECTED_TRAIN_TILES or len(val_dataset) != EXPECTED_VAL_TILES:
        raise RuntimeError("EarthMiss Train/Val manifests changed")
    train_indices = _stratified_train_indices(
        train_dataset,
        args.train_tiles_per_city,
        args.seed,
    )
    val_limit = min(args.val_tiles, len(val_dataset)) if args.val_tiles else len(val_dataset)
    val_indices = tuple(range(val_limit))
    all_cities = tuple(EARTHMISS_CITIES["train"] + EARTHMISS_CITIES["val"])
    city_to_id = {city: index for index, city in enumerate(all_cities)}
    id_to_city = {index: city for city, index in city_to_id.items() if city in EARTHMISS_CITIES["val"]}
    buffers_before = snapshot_batchnorm_buffers(model)
    equivalence = [None]
    with _FRMP5Capture(model) as capture:
        train_data, train_record = _extract_split(
            model,
            capture,
            train_dataset,
            train_indices,
            args,
            device,
            city_to_id,
            equivalence,
        )
        val_data, val_record = _extract_split(
            model,
            capture,
            val_dataset,
            val_indices,
            args,
            device,
            city_to_id,
            equivalence,
        )

    train_x, train_y, train_class, _, _ = train_data
    val_x, val_y, val_class, val_city, val_tile = val_data
    train_x_np = train_x.float().numpy()
    val_x_np = val_x.float().numpy()
    train_y_np = train_y.numpy()
    val_y_np = val_y.numpy()
    train_class_np = train_class.numpy()
    val_class_np = val_class.numpy()
    val_city_np = val_city.numpy()
    val_tile_np = val_tile.numpy()
    main_probe = _fit_probe(train_x_np, train_y_np, args.seed)
    main_scores = main_probe.predict_proba(val_x_np)[:, 1]
    shuffled_y = _within_class_shuffle(
        train_y_np,
        train_class_np,
        stable_seed(args.seed, "within-class-shuffle"),
    )
    shuffled_probe = _fit_probe(train_x_np, shuffled_y, args.seed + 1)
    shuffled_scores = shuffled_probe.predict_proba(val_x_np)[:, 1]
    prior_scores, class_priors = _class_prior_scores(
        train_y_np,
        train_class_np,
        val_class_np,
    )
    scores = {
        "sar_p5_linear": main_scores,
        "within_class_shuffled_target": shuffled_scores,
        "gt_class_conditional_prior": prior_scores,
    }
    evaluation = _evaluate_slices(
        val_y_np,
        scores,
        val_class_np,
        val_city_np,
        val_tile_np,
        id_to_city,
        val_record["tile_keys"],
    )
    pooled = evaluation["pooled"]
    main_ap = pooled["sar_p5_linear"]["average_precision"]
    main_auc = pooled["sar_p5_linear"]["roc_auc"]
    prior_ap = pooled["gt_class_conditional_prior"]["average_precision"]
    shuffled_ap = pooled["within_class_shuffled_target"]["average_precision"]
    city_uplifts = {}
    for city, values in evaluation["by_city"].items():
        left = values["sar_p5_linear"]["average_precision"]
        right = values["gt_class_conditional_prior"]["average_precision"]
        city_uplifts[city] = left - right if left is not None and right is not None else None
    tile_uplifts = []
    for values in evaluation["by_tile"].values():
        left = values["sar_p5_linear"]["average_precision"]
        right = values["gt_class_conditional_prior"]["average_precision"]
        if left is not None and right is not None:
            tile_uplifts.append(left - right)
    gates = {
        "pooled_auroc_at_least_0_60": main_auc is not None and main_auc >= 0.60,
        "ap_uplift_over_class_prior_at_least_0_05": (
            main_ap is not None and prior_ap is not None and main_ap - prior_ap >= 0.05
        ),
        "ap_uplift_over_shuffle_at_least_0_05": (
            main_ap is not None
            and shuffled_ap is not None
            and main_ap - shuffled_ap >= 0.05
        ),
        "at_least_2_of_3_val_cities_above_class_prior": sum(
            value is not None and value > 0.0 for value in city_uplifts.values()
        )
        >= 2,
    }
    predictable = all(gates.values())
    formal = (
        args.train_tiles_per_city == TRAIN_TILES_PER_CITY
        and args.val_tiles == 0
        and args.max_examples == MAX_EXAMPLES_PER_SPLIT
        and args.seed == SEED
    )
    report = {
        "schema": SCHEMA,
        "formal": formal,
        "mm_dino_training_was_performed": False,
        "linear_probe_training_was_performed": True,
        "split": "Train fit / Val evaluate; Test not accessed",
        "git_head": _git_head(),
        "checkpoint": checkpoint,
        "protocol": {
            "feature": "canonical SAR post-FRM P5",
            "cohort": "purity>=0.75 P5 cells where canonical SAR is wrong",
            "target": "paired canonical Full is correct",
            "logit_cell_rule": "exact 32x32 average then argmax",
            "crop_policy": "512x512 non-overlapping diagnostic crops",
            "train_tiles_per_city": args.train_tiles_per_city,
            "maximum_examples_per_split": args.max_examples,
            "probe": "StandardScaler + class-balanced SGD logistic regression",
            "controls": [
                "GT-class-conditional Train prevalence",
                "within-GT-class shuffled Train target",
            ],
            "gate_scope": "linear predictability at post-FRM P5 only",
        },
        "processed": {"train": train_record, "val": val_record},
        "class_conditional_train_priors": class_priors,
        "cached_forward_equivalence": equivalence[0],
        "batchnorm_audit": assert_buffers_unchanged(buffers_before, model),
        "evaluation": evaluation,
        "city_ap_uplift_over_class_prior": city_uplifts,
        "tile_ap_uplift_over_class_prior": {
            "valid_tiles": len(tile_uplifts),
            "positive_fraction": (
                sum(value > 0.0 for value in tile_uplifts) / len(tile_uplifts)
                if tile_uplifts
                else None
            ),
            "mean": float(np.mean(tile_uplifts)) if tile_uplifts else None,
            "median": float(np.median(tile_uplifts)) if tile_uplifts else None,
        },
        "gates": gates,
        "predictable_at_linear_p5_level": predictable,
        "decision": (
            "linear_p5_recoverability_supported"
            if predictable
            else "linear_p5_recoverability_not_supported"
        ),
        "interpretation": (
            "Failure rejects linear recoverability from this SAR P5 feature, "
            "not all nonlinear SAR-predictability. Success still does not prove "
            "that distillation or reconstruction will improve segmentation."
        ),
    }
    write_json_exclusive(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = diagnose(args)
    print(
        {
            "output": args.output,
            "decision": report["decision"],
            "pooled": report["evaluation"]["pooled"],
        }
    )


if __name__ == "__main__":
    main()
