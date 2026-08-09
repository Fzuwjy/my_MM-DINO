"""Run the two deployment-facing gates before EarthMiss V3 training.

The diagnostic reads preserved A/B/C checkpoints with raw online BN buffers.
One complete Test pass measures the internal and expert Full-teacher error
complementarity and reuses C-SAR logits for a fixed single-class bias upper
bound.  It never updates model state or writes logits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from datasets import EARTHMISS_CITIES, build_dataset  # noqa: E402
from models.MMDINO.availability import canonical_availability  # noqa: E402
from models.MMDINO.dino_segment import build_model  # noqa: E402
from utils.earthmiss_metrics import EarthMissMetrics  # noqa: E402
from utils.inference import slide_inference  # noqa: E402

from scripts.evaluate_earthmiss_missing_v1 import (  # noqa: E402
    _checkpoint_uses_raw_logits,
)
from scripts.train_earthmiss_missing_v1 import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_WEIGHTS,
)
from scripts.train_earthmiss_missing_v3 import file_sha256  # noqa: E402


SCHEMA = "earthmiss_missing_v3_zero_training_gates_v2"
CLASS_NAMES = (
    "Background",
    "Building",
    "Road",
    "Water",
    "Barren",
    "Forest",
    "Agricultural",
    "Playground",
)
NUM_CLASSES = len(CLASS_NAMES)
BIAS_GRID = tuple(index / 10.0 for index in range(-40, 41))
WINDOW_SIZE = 512
STRIDE = 341
DEFAULT_OUTPUT = (
    "/root/autodl-tmp/mm-dino/outputs/earthmiss-missing-v3-zero-training/"
    "diagnostic.json"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-checkpoint", required=True)
    parser.add_argument("--b-checkpoint", required=True)
    parser.add_argument("--c-checkpoint", required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument(
        "--smoke-tiles",
        type=int,
        default=0,
        help="Non-formal prefix length per split; zero means every tile.",
    )
    return parser.parse_args(argv)


def validate_args(args):
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive")
    if args.smoke_tiles < 0:
        raise ValueError("--smoke-tiles must be non-negative")


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _metrics_with_percent(evaluator):
    result = evaluator.compute()
    result["mIoU_percent"] = result["mIoU"] * 100.0
    result["official_ever_mIoU_percent"] = (
        result["official_ever_mIoU"] * 100.0
    )
    return result


class TeacherPairScope:
    def __init__(self):
        self.student = EarthMissMetrics()
        self.teacher = EarthMissMetrics()
        self.oracle = EarthMissMetrics()
        self.valid = 0
        self.student_errors = 0
        self.correctable = 0
        self.harmful = 0
        self.valid_by_class = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        self.errors_by_class = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        self.correctable_by_class = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        self.harmful_by_class = torch.zeros(NUM_CLASSES, dtype=torch.int64)
        self.confidence_advantage_sum = 0.0

    def update(self, student_logits, teacher_logits, target):
        student_logits = student_logits.detach().to("cpu", torch.float32)
        teacher_logits = teacher_logits.detach().to("cpu", torch.float32)
        target = target.detach().to("cpu", torch.int64)
        student_prediction = student_logits.argmax(dim=1)
        teacher_prediction = teacher_logits.argmax(dim=1)
        valid = (target >= 0) & (target < NUM_CLASSES)
        student_correct = student_prediction == target
        teacher_correct = teacher_prediction == target
        correctable = valid & teacher_correct & ~student_correct
        harmful = valid & student_correct & ~teacher_correct
        student_errors = valid & ~student_correct

        oracle_prediction = student_prediction.clone()
        oracle_prediction[correctable] = teacher_prediction[correctable]
        self.student.update(student_prediction, target)
        self.teacher.update(teacher_prediction, target)
        self.oracle.update(oracle_prediction, target)

        valid_target = target[valid]
        self.valid += int(valid.sum())
        self.student_errors += int(student_errors.sum())
        self.correctable += int(correctable.sum())
        self.harmful += int(harmful.sum())
        self.valid_by_class += torch.bincount(
            valid_target, minlength=NUM_CLASSES
        )
        self.errors_by_class += torch.bincount(
            target[student_errors], minlength=NUM_CLASSES
        )
        self.correctable_by_class += torch.bincount(
            target[correctable], minlength=NUM_CLASSES
        )
        self.harmful_by_class += torch.bincount(
            target[harmful], minlength=NUM_CLASSES
        )

        if correctable.any():
            safe_target = target.clamp(0, NUM_CLASSES - 1).unsqueeze(1)
            student_true_probability = torch.softmax(
                student_logits, dim=1
            ).gather(1, safe_target).squeeze(1)
            teacher_true_probability = torch.softmax(
                teacher_logits, dim=1
            ).gather(1, safe_target).squeeze(1)
            self.confidence_advantage_sum += float(
                (teacher_true_probability - student_true_probability)[
                    correctable
                ].sum()
            )

    def summary(self):
        if self.valid == 0:
            return {
                "valid_pixels": 0,
                "student_error_pixels": 0,
                "teacher_correct_student_wrong_pixels": 0,
                "teacher_wrong_student_correct_pixels": 0,
                "q_abs": None,
                "q_cov": None,
                "mean_teacher_minus_student_true_class_probability_on_correctable": None,
                "student": None,
                "teacher": None,
                "oracle": None,
                "oracle_gain_over_student_pp": None,
                "by_class": [],
            }
        student_metrics = _metrics_with_percent(self.student)
        teacher_metrics = _metrics_with_percent(self.teacher)
        oracle_metrics = _metrics_with_percent(self.oracle)
        class_rows = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            valid = int(self.valid_by_class[class_id])
            errors = int(self.errors_by_class[class_id])
            correctable = int(self.correctable_by_class[class_id])
            harmful = int(self.harmful_by_class[class_id])
            class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "valid_pixels": valid,
                    "student_error_pixels": errors,
                    "teacher_correct_student_wrong_pixels": correctable,
                    "teacher_wrong_student_correct_pixels": harmful,
                    "q_abs": _ratio(correctable, valid),
                    "q_cov": _ratio(correctable, errors),
                }
            )
        return {
            "valid_pixels": self.valid,
            "student_error_pixels": self.student_errors,
            "teacher_correct_student_wrong_pixels": self.correctable,
            "teacher_wrong_student_correct_pixels": self.harmful,
            "q_abs": _ratio(self.correctable, self.valid),
            "q_cov": _ratio(self.correctable, self.student_errors),
            "mean_teacher_minus_student_true_class_probability_on_correctable": (
                _ratio(self.confidence_advantage_sum, self.correctable)
            ),
            "student": student_metrics,
            "teacher": teacher_metrics,
            "oracle": oracle_metrics,
            "oracle_gain_over_student_pp": (
                oracle_metrics["official_ever_mIoU_percent"]
                - student_metrics["official_ever_mIoU_percent"]
            ),
            "by_class": class_rows,
        }


class TeacherPairAccumulator:
    def __init__(self, cities):
        self.pooled = TeacherPairScope()
        self.by_city = {city: TeacherPairScope() for city in cities}

    def update(self, student_logits, teacher_logits, target, city):
        self.pooled.update(student_logits, teacher_logits, target)
        self.by_city[city].update(student_logits, teacher_logits, target)

    def summary(self):
        return {
            "pooled": self.pooled.summary(),
            "by_city": {
                city: scope.summary() for city, scope in self.by_city.items()
            },
        }


def fixed_eight_class_metrics(confusion):
    confusion = torch.as_tensor(confusion, dtype=torch.int64)
    matrix = confusion.to(torch.float64)
    true_count = matrix.sum(dim=1)
    predicted_count = matrix.sum(dim=0)
    true_positive = matrix.diag()
    denominator = true_count + predicted_count - true_positive
    class_iou = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    supported = denominator > 0
    class_iou[supported] = true_positive[supported] / denominator[supported]
    return {
        "mIoU_percent": float(class_iou.mean() * 100.0),
        "class_iou_percent": [float(value * 100.0) for value in class_iou],
        "gt_pixels": [int(value) for value in true_count],
        "predicted_pixels": [int(value) for value in predicted_count],
        "confusion": confusion.tolist(),
    }


class SingleClassBiasAccumulator:
    """Sufficient statistics for exact fixed-grid one-class bias sweeps."""

    def __init__(self, bias_grid=BIAS_GRID):
        self.bias_grid = tuple(float(value) for value in bias_grid)
        if tuple(sorted(self.bias_grid)) != self.bias_grid:
            raise ValueError("bias grid must be sorted")
        if len(set(self.bias_grid)) != len(self.bias_grid):
            raise ValueError("bias grid must contain unique values")
        if 0.0 not in self.bias_grid:
            raise ValueError("bias grid must contain zero")
        self.grid_tensor = torch.tensor(self.bias_grid, dtype=torch.float32)
        self.switch_counts = torch.zeros(
            (
                NUM_CLASSES,
                NUM_CLASSES,
                NUM_CLASSES,
                len(self.bias_grid) + 1,
            ),
            dtype=torch.int64,
        )
        self.baseline = EarthMissMetrics()

    def update(self, logits, target):
        logits = logits.detach().to("cpu", torch.float32)
        target = target.detach().to("cpu", torch.int64)
        if logits.ndim != 4 or logits.shape[0] != 1:
            raise ValueError("bias diagnostics require one tile at a time")
        if logits.shape[1] != NUM_CLASSES:
            raise ValueError("unexpected EarthMiss class count")
        self.baseline.update(logits.argmax(dim=1), target)
        target_2d = target[0]
        valid = (target_2d >= 0) & (target_2d < NUM_CLASSES)
        if not valid.any():
            return
        scores = logits[0, :, valid]
        valid_target = target_2d[valid]
        bins = len(self.bias_grid) + 1
        for class_id in range(NUM_CLASSES):
            other_scores = scores.clone()
            other_scores[class_id] = float("-inf")
            other_value, other_index = other_scores.max(dim=0)
            margin = other_value - scores[class_id]
            lower_bound = torch.bucketize(
                margin, self.grid_tensor, right=False
            )
            upper_bound = torch.bucketize(
                margin, self.grid_tensor, right=True
            )
            # torch.argmax gives an exact tie to the lower class index.
            switch_index = torch.where(
                class_id < other_index, lower_bound, upper_bound
            )
            flat = (
                (valid_target * NUM_CLASSES + other_index) * bins
                + switch_index
            )
            counts = torch.bincount(
                flat, minlength=NUM_CLASSES * NUM_CLASSES * bins
            ).reshape(NUM_CLASSES, NUM_CLASSES, bins)
            self.switch_counts[class_id] += counts

    def confusion_for(self, class_id, bias_index):
        counts = self.switch_counts[class_id]
        switched = counts[..., : bias_index + 1].sum(dim=-1)
        remaining = counts[..., bias_index + 1 :].sum(dim=-1)
        confusion = remaining.clone()
        confusion[:, class_id] += switched.sum(dim=1)
        return confusion

    def assert_zero_matches_baseline(self):
        zero_index = self.bias_grid.index(0.0)
        expected = self.baseline.confusion
        for class_id in range(NUM_CLASSES):
            actual = self.confusion_for(class_id, zero_index)
            if not torch.equal(actual, expected):
                raise RuntimeError(
                    f"bias histogram does not reproduce baseline for class {class_id}"
                )


def _best_index(scores, bias_grid):
    maximum = max(scores)
    candidates = [
        index
        for index, score in enumerate(scores)
        if abs(score - maximum) <= 1e-12
    ]
    return min(candidates, key=lambda index: (abs(bias_grid[index]), bias_grid[index]))


def summarize_bias_sweep(train_accumulator, test_accumulator):
    if train_accumulator.bias_grid != test_accumulator.bias_grid:
        raise ValueError("Train and Test bias grids differ")
    train_accumulator.assert_zero_matches_baseline()
    test_accumulator.assert_zero_matches_baseline()
    grid = train_accumulator.bias_grid
    baseline_train = fixed_eight_class_metrics(train_accumulator.baseline.confusion)
    baseline_test = fixed_eight_class_metrics(test_accumulator.baseline.confusion)
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        train_metrics = [
            fixed_eight_class_metrics(
                train_accumulator.confusion_for(class_id, index)
            )
            for index in range(len(grid))
        ]
        test_metrics = [
            fixed_eight_class_metrics(
                test_accumulator.confusion_for(class_id, index)
            )
            for index in range(len(grid))
        ]
        train_index = _best_index(
            [item["mIoU_percent"] for item in train_metrics], grid
        )
        test_index = _best_index(
            [item["mIoU_percent"] for item in test_metrics], grid
        )
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "train_fitted_bias": grid[train_index],
                "train_at_train_fitted": train_metrics[train_index],
                "test_at_train_fitted": test_metrics[train_index],
                "test_gain_at_train_fitted_pp": (
                    test_metrics[train_index]["mIoU_percent"]
                    - baseline_test["mIoU_percent"]
                ),
                "test_oracle_bias": grid[test_index],
                "test_at_test_oracle": test_metrics[test_index],
                "test_oracle_gain_pp": (
                    test_metrics[test_index]["mIoU_percent"]
                    - baseline_test["mIoU_percent"]
                ),
                "train_boundary_hit": train_index in {0, len(grid) - 1},
                "test_oracle_boundary_hit": test_index in {0, len(grid) - 1},
            }
        )

    train_choice = max(
        rows,
        key=lambda row: (
            row["train_at_train_fitted"]["mIoU_percent"],
            -abs(row["train_fitted_bias"]),
            -row["class_id"],
        ),
    )
    test_oracle_choice = max(
        rows,
        key=lambda row: (
            row["test_at_test_oracle"]["mIoU_percent"],
            -abs(row["test_oracle_bias"]),
            -row["class_id"],
        ),
    )
    return {
        "scope": "one_class_bias_at_a_time",
        "selection_objective": "fixed_8_class_pooled_mIoU",
        "bias_grid": list(grid),
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "per_class": rows,
        "best_train_fitted_single_class": train_choice,
        "best_test_oracle_single_class": test_oracle_choice,
        "test_oracle_is_test_developed_diagnostic_only": True,
    }


def summarize_test_bias_upper_bound(test_accumulator):
    test_accumulator.assert_zero_matches_baseline()
    grid = test_accumulator.bias_grid
    baseline = fixed_eight_class_metrics(test_accumulator.baseline.confusion)
    rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        metrics = [
            fixed_eight_class_metrics(
                test_accumulator.confusion_for(class_id, index)
            )
            for index in range(len(grid))
        ]
        best_index = _best_index(
            [item["mIoU_percent"] for item in metrics], grid
        )
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "test_oracle_bias": grid[best_index],
                "test_at_test_oracle": metrics[best_index],
                "test_oracle_gain_pp": (
                    metrics[best_index]["mIoU_percent"]
                    - baseline["mIoU_percent"]
                ),
                "test_oracle_boundary_hit": best_index in {0, len(grid) - 1},
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["test_at_test_oracle"]["mIoU_percent"],
            -abs(row["test_oracle_bias"]),
            -row["class_id"],
        ),
    )
    return {
        "scope": "one_class_bias_at_a_time_on_complete_test",
        "selection_objective": "fixed_8_class_pooled_mIoU",
        "bias_grid": list(grid),
        "baseline_test": baseline,
        "per_class": rows,
        "best_test_oracle_single_class": best,
        "interpretation": (
            "test-developed calibration upper bound; it decides whether the "
            "observed C-SAR trade-off is explainable by scalar calibration"
        ),
        "independent_test_claim_allowed": False,
    }


def build_loader(args, split):
    dataset = build_dataset(
        "EarthMiss",
        split,
        dataset_root=args.dataset_root,
        window_size=(WINDOW_SIZE, WINDOW_SIZE),
        model_name="DINOv3",
        modality="multi",
        backbone_type="dinov3_vits16",
        apply_train_transform=False if split == "train" else None,
        cache_size=0,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader


def load_frozen_model(checkpoint_path, expected_run, weights_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    run = str(checkpoint.get("run", "")).lower()
    if run != expected_run:
        raise ValueError(
            f"expected Run {expected_run.upper()} checkpoint, got {run!r}"
        )
    expected_selection_state = {
        "a": "sar",
        "b": "full",
        "c": "sar",
    }[expected_run]
    selection_state = checkpoint.get("selection_state")
    if selection_state != expected_selection_state:
        raise ValueError(
            f"Run {expected_run.upper()} diagnostic requires the fixed "
            f"best_{expected_selection_state} checkpoint, got selection_state="
            f"{selection_state!r}"
        )
    raw_logits = _checkpoint_uses_raw_logits(checkpoint)
    model = build_model(
        model_name="DINOv3",
        backbone_weights=str(weights_path),
        backbone_type="dinov3_vits16",
        freeze_backbone=True,
        n_classes=NUM_CLASSES,
        use_lora=False,
        r=3,
        num_modalities=2,
        raw_logits=raw_logits,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    return model, {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "run": run,
        "seed": checkpoint.get("seed"),
        "epoch": checkpoint.get("epoch"),
        "selection_state": selection_state,
        "selection_score": checkpoint.get("selection_score"),
        "raw_logits": raw_logits,
    }


def endpoint_logits(model, rgb, sar, endpoint, batch_size, device):
    availability = canonical_availability(
        endpoint, batch_size=1, device=device
    )
    return slide_inference(
        rgb,
        model,
        n_output_channels=NUM_CLASSES,
        crop_size=(WINDOW_SIZE, WINDOW_SIZE),
        stride=(STRIDE, STRIDE),
        dsm=sar,
        availability=availability,
        batch_size=batch_size,
    )


@torch.inference_mode()
def diagnose_split(args, split, models, device):
    if split != "test":
        raise ValueError("the formal V3 gate is defined on complete Test only")
    dataset, loader = build_loader(args, split)
    cities = tuple(EARTHMISS_CITIES[split])
    internal = TeacherPairAccumulator(cities)
    expert = TeacherPairAccumulator(cities)
    bias = SingleClassBiasAccumulator()
    expected_tiles = len(dataset)
    tile_limit = (
        min(args.smoke_tiles, expected_tiles)
        if args.smoke_tiles
        else expected_tiles
    )
    for sample_index, (rgb, sar, label) in enumerate(
        tqdm(loader, total=tile_limit, desc=f"zero-train {split}")
    ):
        if sample_index >= tile_limit:
            break
        city = dataset.samples[sample_index].city
        rgb = rgb.to(device, non_blocking=True)
        sar = sar.to(device, non_blocking=True)
        c_sar = endpoint_logits(
            models["c"], rgb, sar, "sar", args.inference_batch_size, device
        )
        c_full = endpoint_logits(
            models["c"], rgb, sar, "full", args.inference_batch_size, device
        )
        a_sar = endpoint_logits(
            models["a"], rgb, sar, "sar", args.inference_batch_size, device
        )
        b_full = endpoint_logits(
            models["b"], rgb, sar, "full", args.inference_batch_size, device
        )
        internal.update(c_sar, c_full, label, city)
        expert.update(a_sar, b_full, label, city)
        bias.update(c_sar, label)

    return {
        "expected_tiles": expected_tiles,
        "processed_tiles": tile_limit,
        "cities": list(cities),
        "internal_c_full_to_c_sar": internal.summary(),
        "expert_b_full_to_a_sar": expert.summary(),
    }, bias


def main():
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("zero-training EarthMiss diagnostics require CUDA")
    weights_path = Path(args.backbone_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    models = {}
    checkpoints = {}
    for run, path in (
        ("a", args.a_checkpoint),
        ("b", args.b_checkpoint),
        ("c", args.c_checkpoint),
    ):
        models[run], checkpoints[run] = load_frozen_model(
            path, run, weights_path, device
        )

    test_report, test_bias = diagnose_split(args, "test", models, device)
    formal = args.smoke_tiles == 0
    report = {
        "schema": SCHEMA,
        "formal": formal,
        "training_was_performed": False,
        "protocol": {
            "model_state": "checkpoint_raw_online_bn_eval_frozen",
            "test_transform": "deterministic_full_tile",
            "inference": "fp32_sliding_512_stride341",
            "teacher_pairs": [
                "C-Full_to_C-SAR_internal",
                "B-Full_to_A-SAR_expert_upper_bound",
            ],
            "bias_scan": (
                "complete_Test_C-SAR_single_class_fixed_grid_-4_to_4_step_0.1"
            ),
            "test_usage": "test_developed_diagnostic_not_independent_test",
            "decision_questions": [
                "does_C_Full_correct_C_SAR_across_test_cities_and_classes",
                "does_B_Full_correct_A_SAR_across_test_cities_and_classes",
                "can_single_class_bias_explain_the_C_SAR_tradeoff",
            ],
            "train_census_omitted_reason": (
                "training-set correction prevalence does not establish "
                "learnability and does not decide among the three routes"
            ),
        },
        "checkpoints": checkpoints,
        "splits": {"test": test_report},
        "calibration": summarize_test_bias_upper_bound(test_bias),
        "training_gate": {
            "decision": "pending_manual_review" if formal else "smoke_only",
            "rule": (
                "train only if the exact planned expert teacher corrects "
                "deployed A-SAR across Test cities/classes and scalar bias "
                "does not explain the C-SAR failure mode"
            ),
        },
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "formal": formal,
        "internal_test_q_cov": test_report[
            "internal_c_full_to_c_sar"
        ]["pooled"]["q_cov"],
        "internal_test_oracle_gain_pp": test_report[
            "internal_c_full_to_c_sar"
        ]["pooled"]["oracle_gain_over_student_pp"],
        "expert_test_oracle_gain_pp": test_report[
            "expert_b_full_to_a_sar"
        ]["pooled"]["oracle_gain_over_student_pp"],
        "bias_test_oracle_gain_pp": report["calibration"][
            "best_test_oracle_single_class"
        ]["test_oracle_gain_pp"],
    }, indent=2))


if __name__ == "__main__":
    main()
