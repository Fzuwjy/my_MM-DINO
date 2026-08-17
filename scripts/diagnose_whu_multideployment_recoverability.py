"""Probe whether each missing WHU endpoint predicts paired Full corrections.

Linear probes are fitted on fixed official-Train crops and evaluated on the
official Test scenes for the preserved historical Run C checkpoint.  This is a
mechanism audit, not MM-DINO training and not a method-selection result.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from models.MMDINO.availability import canonical_availability  # noqa: E402

from scripts.diagnose_earthmiss_scale_transition import (  # noqa: E402
    assert_buffers_unchanged,
    snapshot_batchnorm_buffers,
)
from scripts.earthmiss_causal_diagnostics_common import stable_seed  # noqa: E402
from scripts.earthmiss_scale_transition_common import write_json_exclusive  # noqa: E402
from scripts.whu_multideployment_diagnostics_common import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_RUN_C_CHECKPOINT,
    DEFAULT_WEIGHTS,
    MISSING_ENDPOINTS,
    NUM_CLASSES,
    build_whu_scene_dataset,
    deterministic_coordinate_subset,
    endpoint_recoverability_examples,
    fixed_grid_coordinates,
    load_historical_run_c,
    split_names,
    verify_cached_equivalence_all_endpoints,
)


SCHEMA = "whu_multideployment_recoverability_probe_v1"
DEFAULT_TRAIN_SPLIT = str(REPO_ROOT / "splits" / "whu" / "official_train.txt")
DEFAULT_TEST_SPLIT = str(REPO_ROOT / "splits" / "whu" / "official_test.txt")
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/mm-dino/outputs/whu-multideployment-causal-audit/"
    "historical-run-c-e50-recoverability.json"
)
WINDOW_SIZE = 512
PURITY_THRESHOLD = 0.75
DEFAULT_TRAIN_SCENES = 16
DEFAULT_CROPS_PER_SCENE = 8
DEFAULT_MAX_EXAMPLES = 100_000
SEED = 20260817


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_RUN_C_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-split", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--test-split", default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--train-scenes", type=int, default=DEFAULT_TRAIN_SCENES)
    parser.add_argument(
        "--test-scenes", type=int, default=0,
        help="Zero evaluates all 20 official Test scenes.",
    )
    parser.add_argument(
        "--crops-per-scene", type=int, default=DEFAULT_CROPS_PER_SCENE,
    )
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-cached-equivalence-check", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.train_scenes <= 0 or args.train_scenes > 80:
        raise ValueError("--train-scenes must be in [1,80]")
    if args.test_scenes < 0 or args.test_scenes > 20:
        raise ValueError("--test-scenes must be in [0,20]")
    if args.crops_per_scene <= 0:
        raise ValueError("--crops-per-scene must be positive")
    if args.max_examples < 100 or args.num_workers < 0:
        raise ValueError("invalid max-example/worker count")


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
            raise RuntimeError("FRM P5 capture contract changed")
        return logits, self.value


def _scene_indices(count: int, selected: int, seed: int) -> tuple[int, ...]:
    if selected >= count:
        return tuple(range(count))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randperm(count, generator=generator)[:selected]
    return tuple(int(value) for value in values.sort().values)


def _subsample(values, maximum: int, seed: int):
    if values[0].shape[0] <= maximum:
        return values
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(values[0].shape[0], generator=generator)[:maximum]
    return tuple(value[indices] for value in values)


@torch.inference_mode()
def _extract(
    model,
    capture,
    dataset,
    indices,
    args,
    device,
    split_name,
    equivalence,
):
    data = {
        endpoint: {"features": [], "targets": [], "classes": [], "scenes": []}
        for endpoint in MISSING_ENDPOINTS
    }
    counters = {
        endpoint: {"pure_cells": 0, "endpoint_wrong_cells": 0}
        for endpoint in MISSING_ENDPOINTS
    }
    scene_names = []
    crop_count = 0
    full_availability = canonical_availability("full", batch_size=1, device=device)
    endpoint_availability = {
        endpoint: canonical_availability(endpoint, batch_size=1, device=device)
        for endpoint in MISSING_ENDPOINTS
    }
    for local_scene_id, index in enumerate(
        tqdm(indices, desc=f"whu-recoverability-{split_name}")
    ):
        optical, sar, target = dataset[index]
        target = torch.as_tensor(target)
        scene_names.append(Path(dataset.rgb_files[index]).name)
        coordinates = fixed_grid_coordinates(
            target.shape[-2], target.shape[-1], window_size=WINDOW_SIZE,
        )
        coordinates = deterministic_coordinate_subset(
            coordinates, args.crops_per_scene,
            stable_seed(args.seed, f"{split_name}:{index}"),
        )
        for y1, y2, x1, x2 in coordinates:
            crop_count += 1
            optical_crop = optical[None, :, y1:y2, x1:x2].to(device)
            sar_crop = sar[None, :, y1:y2, x1:x2].to(device)
            target_crop = target[None, y1:y2, x1:x2].to(device)
            backbone_outputs = model.extract_frozen_backbone_outputs(
                optical_crop, sar_crop,
            )
            if equivalence[0] is None and not args.skip_cached_equivalence_check:
                equivalence[0] = verify_cached_equivalence_all_endpoints(
                    model, optical_crop, sar_crop, backbone_outputs,
                )
            full_logits = model.forward_from_backbone_outputs(
                optical_crop, sar_crop, backbone_outputs=backbone_outputs,
                availability=full_availability,
            )
            for endpoint in MISSING_ENDPOINTS:
                endpoint_logits, endpoint_p5 = capture.run(
                    lambda endpoint=endpoint: model.forward_from_backbone_outputs(
                        optical_crop, sar_crop, backbone_outputs=backbone_outputs,
                        availability=endpoint_availability[endpoint],
                    )
                )
                examples = endpoint_recoverability_examples(
                    endpoint_p5, full_logits, endpoint_logits, target_crop,
                    purity_threshold=PURITY_THRESHOLD,
                )
                counters[endpoint]["pure_cells"] += examples.pure_cells
                counters[endpoint]["endpoint_wrong_cells"] += examples.endpoint_wrong_cells
                count = examples.features.shape[0]
                if count:
                    data[endpoint]["features"].append(
                        examples.features.to("cpu", torch.float16)
                    )
                    data[endpoint]["targets"].append(examples.targets.to("cpu", torch.int8))
                    data[endpoint]["classes"].append(examples.class_ids.to("cpu", torch.int8))
                    data[endpoint]["scenes"].append(
                        torch.full((count,), local_scene_id, dtype=torch.int16)
                    )
    joined = {}
    for endpoint, values in data.items():
        if not values["features"]:
            raise RuntimeError(f"no {endpoint} recoverability examples in {split_name}")
        tensors = (
            torch.cat(values["features"]),
            torch.cat(values["targets"]).long(),
            torch.cat(values["classes"]).long(),
            torch.cat(values["scenes"]).long(),
        )
        joined[endpoint] = _subsample(
            tensors, args.max_examples,
            stable_seed(args.seed, f"{split_name}:{endpoint}:subsample"),
        )
        counters[endpoint]["retained_examples"] = int(joined[endpoint][0].shape[0])
        counters[endpoint]["recoverable_prevalence"] = float(
            joined[endpoint][1].float().mean()
        )
    return joined, {
        "scenes": len(indices),
        "scene_names": scene_names,
        "crops": crop_count,
        "endpoints": counters,
    }


def _binary_metrics(target, score):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    result = {
        "n": int(target.size), "positives": int(target.sum()),
        "prevalence": float(target.mean()) if target.size else None,
        "average_precision": None, "roc_auc": None, "brier": None,
    }
    if target.size:
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
        raise RuntimeError("probe Train target must contain both outcomes")
    probe = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss", penalty="l2", alpha=1e-4,
            class_weight="balanced", max_iter=2000, tol=1e-5,
            random_state=seed,
        ),
    )
    probe.fit(features, targets)
    return probe


def _within_class_shuffle(targets, classes, seed):
    result = np.asarray(targets).copy()
    generator = np.random.default_rng(seed)
    for class_id in range(NUM_CLASSES):
        indices = np.flatnonzero(np.asarray(classes) == class_id)
        result[indices] = result[generator.permutation(indices)]
    return result


def _class_prior_scores(train_target, train_class, eval_class):
    priors = np.empty(NUM_CLASSES, dtype=np.float64)
    global_prior = (float(np.sum(train_target)) + 1.0) / (len(train_target) + 2.0)
    for class_id in range(NUM_CLASSES):
        values = np.asarray(train_target)[np.asarray(train_class) == class_id]
        priors[class_id] = (
            (float(values.sum()) + 1.0) / (len(values) + 2.0)
            if len(values) else global_prior
        )
    return priors[np.asarray(eval_class)], priors.tolist()


def _evaluate(target, scores, classes, scenes, scene_names):
    result = {"pooled": {}, "by_scene": {}, "by_class": {}}
    for name, values in scores.items():
        result["pooled"][name] = _binary_metrics(target, values)
    for scene_id, scene_name in enumerate(scene_names):
        mask = scenes == scene_id
        result["by_scene"][scene_name] = {
            name: _binary_metrics(target[mask], values[mask])
            for name, values in scores.items()
        }
    for class_id in range(NUM_CLASSES):
        mask = classes == class_id
        result["by_class"][str(class_id)] = {
            name: _binary_metrics(target[mask], values[mask])
            for name, values in scores.items()
        }
    return result


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite probe report: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("WHU recoverability probe requires CUDA")
    train_names = split_names(args.train_split)
    test_names = split_names(args.test_split)
    if train_names != split_names(DEFAULT_TRAIN_SPLIT):
        raise ValueError("historical probe is bound to official_train.txt")
    if test_names != split_names(DEFAULT_TEST_SPLIT):
        raise ValueError("historical probe is bound to official_test.txt")
    if set(train_names) & set(test_names):
        raise RuntimeError("WHU official Train/Test overlap")
    device = torch.device("cuda")
    model, checkpoint = load_historical_run_c(
        args.checkpoint, args.backbone_weights, device, freeze_model=True,
    )
    train_dataset = build_whu_scene_dataset(args.dataset_root, args.train_split)
    test_dataset = build_whu_scene_dataset(args.dataset_root, args.test_split)
    train_indices = _scene_indices(len(train_dataset), args.train_scenes, args.seed)
    test_count = args.test_scenes or len(test_dataset)
    test_indices = tuple(range(test_count))
    buffers_before = snapshot_batchnorm_buffers(model)
    equivalence = [None]
    with _FRMP5Capture(model) as capture:
        train_data, train_record = _extract(
            model, capture, train_dataset, train_indices, args, device,
            "train", equivalence,
        )
        test_data, test_record = _extract(
            model, capture, test_dataset, test_indices, args, device,
            "test", equivalence,
        )

    endpoint_reports = {}
    for endpoint in MISSING_ENDPOINTS:
        train_x, train_y, train_class, _ = train_data[endpoint]
        test_x, test_y, test_class, test_scene = test_data[endpoint]
        train_x = train_x.float().numpy()
        test_x = test_x.float().numpy()
        train_y = train_y.numpy()
        test_y = test_y.numpy()
        train_class = train_class.numpy()
        test_class = test_class.numpy()
        test_scene = test_scene.numpy()
        probe = _fit_probe(train_x, train_y, stable_seed(args.seed, endpoint))
        main_scores = probe.predict_proba(test_x)[:, 1]
        shuffled_y = _within_class_shuffle(
            train_y, train_class, stable_seed(args.seed, f"{endpoint}:shuffle"),
        )
        shuffled_probe = _fit_probe(
            train_x, shuffled_y, stable_seed(args.seed, f"{endpoint}:shuffle-fit"),
        )
        shuffled_scores = shuffled_probe.predict_proba(test_x)[:, 1]
        prior_scores, priors = _class_prior_scores(train_y, train_class, test_class)
        scores = {
            "endpoint_p5_linear": main_scores,
            "within_class_shuffled_target": shuffled_scores,
            "gt_class_conditional_prior": prior_scores,
        }
        evaluation = _evaluate(
            test_y, scores, test_class, test_scene, test_record["scene_names"],
        )
        pooled = evaluation["pooled"]
        main_ap = pooled["endpoint_p5_linear"]["average_precision"]
        main_auc = pooled["endpoint_p5_linear"]["roc_auc"]
        prior_ap = pooled["gt_class_conditional_prior"]["average_precision"]
        shuffled_ap = pooled["within_class_shuffled_target"]["average_precision"]
        scene_uplifts = {}
        for scene, values in evaluation["by_scene"].items():
            left = values["endpoint_p5_linear"]["average_precision"]
            right = values["gt_class_conditional_prior"]["average_precision"]
            scene_uplifts[scene] = (
                left - right if left is not None and right is not None else None
            )
        gates = {
            "pooled_auroc_at_least_0_60": main_auc is not None and main_auc >= 0.60,
            "ap_uplift_over_class_prior_at_least_0_05": (
                main_ap is not None and prior_ap is not None and main_ap - prior_ap >= 0.05
            ),
            "ap_uplift_over_shuffle_at_least_0_05": (
                main_ap is not None and shuffled_ap is not None
                and main_ap - shuffled_ap >= 0.05
            ),
            "at_least_60pct_test_scenes_above_class_prior": (
                sum(value is not None and value > 0.0 for value in scene_uplifts.values())
                >= int(np.ceil(0.60 * len(scene_uplifts)))
            ),
        }
        endpoint_reports[endpoint] = {
            "class_conditional_train_priors": priors,
            "evaluation": evaluation,
            "scene_ap_uplift_over_class_prior": scene_uplifts,
            "gates": gates,
            "predictable_at_linear_p5_level": all(gates.values()),
        }

    formal = (
        args.train_scenes == DEFAULT_TRAIN_SCENES
        and args.test_scenes == 0
        and args.crops_per_scene == DEFAULT_CROPS_PER_SCENE
        and args.max_examples == DEFAULT_MAX_EXAMPLES
        and args.seed == SEED
    )
    report = {
        "schema": SCHEMA,
        "formal": formal,
        "mm_dino_training_was_performed": False,
        "linear_probe_training_was_performed": True,
        "method_selection_eligible": False,
        "split": "official Train fit / official Test evaluate; development-exposed",
        "git_head": _git_head(),
        "checkpoint": checkpoint,
        "protocol": {
            "features": {
                endpoint: f"canonical {endpoint} post-FRM P5"
                for endpoint in MISSING_ENDPOINTS
            },
            "cohort": "purity>=0.75 P5 cells where target endpoint is wrong",
            "target": "paired canonical Full is correct",
            "crop_policy": "deterministic disjoint 512 crops; border excluded",
            "train_scenes": args.train_scenes,
            "test_scenes": args.test_scenes or 20,
            "crops_per_scene": args.crops_per_scene,
            "probe": "StandardScaler + class-balanced SGD logistic regression",
            "controls": [
                "GT-class-conditional Train prevalence",
                "within-GT-class shuffled Train target",
            ],
            "historical_limitation": (
                "Run C trained Full/SAR only; Optical-only is an unseen deployment state"
            ),
        },
        "processed": {"train": train_record, "test": test_record},
        "cached_forward_equivalence": equivalence[0],
        "batchnorm_audit": assert_buffers_unchanged(buffers_before, model),
        "endpoints": endpoint_reports,
        "interpretation": (
            "Success supports only linear predictability of paired Full correction "
            "at post-FRM P5. Failure does not rule out nonlinear predictability. "
            "Official-Test results cannot select a future innovation."
        ),
    }
    write_json_exclusive(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = diagnose(args)
    print({
        "output": args.output,
        "formal": report["formal"],
        "decisions": {
            endpoint: values["predictable_at_linear_p5_level"]
            for endpoint, values in report["endpoints"].items()
        },
    })


if __name__ == "__main__":
    main()
