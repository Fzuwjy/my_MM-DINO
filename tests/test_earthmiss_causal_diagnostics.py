"""CPU contracts for the EarthMiss failure-directed causal diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SEGMENTATION_ROOT))

from models.MMDINO.Decoder import Decoder  # noqa: E402
from models.MMDINO.sample_adapter import SampleAdapter  # noqa: E402
from scripts.diagnose_earthmiss_oracle_intervention import (  # noqa: E402
    validate_args as validate_oracle_args,
)
from scripts.diagnose_earthmiss_sar_recoverability import (  # noqa: E402
    _binary_metrics,
    _fit_probe,
    _within_class_shuffle,
)
from scripts.diagnose_earthmiss_gradient_conflict import (  # noqa: E402
    BATCH_SIZE,
    _same_city_batches,
    _schedule_audit,
)
from datasets import EARTHMISS_CITIES  # noqa: E402
from scripts.analyze_earthmiss_causal_diagnostics import causal_decision  # noqa: E402
from scripts.earthmiss_causal_diagnostics_common import (  # noqa: E402
    ORACLE_STAGE_ORDER,
    OracleStageIntervention,
    VariantLogitStitcher,
    exact_class_occupancy,
    gradient_pair_statistics,
    parameter_group_manifest,
    recoverability_examples,
    summarize_gradient_records,
)


class TinyReleasedPath(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.adapter = SampleAdapter(
            in_channels=4,
            out_channels=[16, 16, 16, 16],
            num_modalities=2,
        )
        self.decoder = Decoder(
            n_classes=3,
            in_channels=[16, 16, 16, 16],
            out_channels=16,
            num_modalities=2,
            raw_logits=True,
        )

    def decode(self, *features, active_indices):
        slots = self.adapter(
            *features,
            patch_h=4,
            patch_w=4,
            guidance=None,
            modality_indices=active_indices,
        )
        logits = self.decoder(*slots)
        return F.interpolate(logits, size=(64, 64), mode="bilinear")


def _features(value: float) -> tuple[torch.Tensor, ...]:
    base = torch.arange(16 * 4, dtype=torch.float32).reshape(1, 16, 4)
    return tuple(base / 100.0 + value + index for index in range(4))


class OracleInterventionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = TinyReleasedPath().eval()
        self.rgb = _features(0.25)
        self.sar = _features(-0.5)

    def test_all_stage_controls_reproduce_full_exactly(self):
        with torch.inference_mode(), OracleStageIntervention(self.model) as hooks:
            full = hooks.capture_full(
                lambda: self.model.decode(
                    self.rgb,
                    self.sar,
                    active_indices=(0, 1),
                )
            )
            sar = self.model.decode(self.sar, active_indices=(1,))
            self.assertFalse(torch.equal(full, sar))
            for stage in (
                "adapter.all",
                "frm.all",
                "se.all",
                "prn.all",
                "head.logits",
            ):
                actual = hooks.intervene(
                    stage,
                    1.0,
                    lambda: self.model.decode(self.sar, active_indices=(1,)),
                )
                self.assertTrue(torch.equal(actual, full), stage)

    def test_zero_alpha_and_inactive_hooks_preserve_sar(self):
        with torch.inference_mode(), OracleStageIntervention(self.model) as hooks:
            hooks.capture_full(
                lambda: self.model.decode(
                    self.rgb,
                    self.sar,
                    active_indices=(0, 1),
                )
            )
            reference = self.model.decode(self.sar, active_indices=(1,))
            actual = hooks.intervene(
                "frm.P5",
                0.0,
                lambda: self.model.decode(self.sar, active_indices=(1,)),
            )
            after = self.model.decode(self.sar, active_indices=(1,))
        self.assertTrue(torch.equal(actual, reference))
        self.assertTrue(torch.equal(after, reference))

    def test_every_registered_stage_executes(self):
        with torch.inference_mode(), OracleStageIntervention(self.model) as hooks:
            hooks.capture_full(
                lambda: self.model.decode(
                    self.rgb,
                    self.sar,
                    active_indices=(0, 1),
                )
            )
            for stage in ORACLE_STAGE_ORDER:
                value = hooks.intervene(
                    stage,
                    0.5,
                    lambda: self.model.decode(self.sar, active_indices=(1,)),
                )
                self.assertEqual(tuple(value.shape), (1, 3, 64, 64))

    def test_multi_stage_response_curve_is_rejected(self):
        args = SimpleNamespace(
            num_workers=0,
            smoke_tiles=0,
            stages=["frm.P4", "frm.P5"],
            alphas=[0.5, 1.0],
        )
        with self.assertRaisesRegex(ValueError, "multi-stage screen"):
            validate_oracle_args(args)


class StitcherTest(unittest.TestCase):
    def test_overlap_average_is_variant_specific(self):
        stitcher = VariantLogitStitcher(2, 3, 1, ("a", "b"))
        stitcher.add(
            ((0, 2, 0, 2),),
            {
                "a": torch.ones(1, 1, 2, 2),
                "b": torch.full((1, 1, 2, 2), 2.0),
            },
        )
        stitcher.add(
            ((0, 2, 1, 3),),
            {
                "a": torch.full((1, 1, 2, 2), 3.0),
                "b": torch.full((1, 1, 2, 2), 6.0),
            },
        )
        values = stitcher.finalize()
        self.assertTrue(
            torch.equal(values["a"], torch.tensor([[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]]))
        )
        self.assertTrue(
            torch.equal(values["b"], torch.tensor([[[[2.0, 4.0, 6.0], [2.0, 4.0, 6.0]]]]))
        )


class RecoverabilityTest(unittest.TestCase):
    def test_occupancy_preserves_ignore_as_empty_area(self):
        target = torch.tensor([[[0, 0, 1, 1], [0, 8, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]])
        occupancy = exact_class_occupancy(target, (2, 2))
        self.assertEqual(tuple(occupancy.shape), (1, 8, 2, 2))
        self.assertEqual(float(occupancy[0, 0, 0, 0]), 0.75)
        self.assertEqual(float(occupancy[:, :, 0, 0].sum()), 0.75)

    def test_examples_condition_on_sar_wrong_and_full_correct(self):
        feature = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
        target = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]]
        )
        full = torch.zeros(1, 8, 4, 4)
        sar = torch.zeros_like(full)
        sar[:, 7] = 2.0
        full[:, 7] = 1.0
        full[:, 0, :2, :2] = 3.0
        full[:, 2, 2:, :2] = 3.0
        examples = recoverability_examples(feature, full, sar, target)
        self.assertEqual(examples.pure_cells, 4)
        self.assertEqual(examples.sar_wrong_cells, 4)
        self.assertEqual(examples.targets.tolist(), [1, 0, 1, 0])
        self.assertEqual(examples.class_ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(tuple(examples.features.shape), (4, 2))

    def test_within_class_shuffle_never_moves_labels_across_classes(self):
        target = torch.tensor([0, 1, 1, 0, 1, 0]).numpy()
        classes = torch.tensor([2, 2, 2, 3, 3, 3]).numpy()
        shuffled = _within_class_shuffle(target, classes, 7)
        for class_id in (2, 3):
            mask = classes == class_id
            self.assertEqual(sorted(shuffled[mask].tolist()), sorted(target[mask].tolist()))

    def test_fixed_linear_probe_learns_a_separable_signal(self):
        torch.manual_seed(17)
        negative = torch.randn(100, 3) - 2.0
        positive = torch.randn(100, 3) + 2.0
        features = torch.cat((negative, positive)).numpy()
        target = torch.cat((torch.zeros(100), torch.ones(100))).long().numpy()
        probe = _fit_probe(features, target, 11)
        score = probe.predict_proba(features)[:, 1]
        metrics = _binary_metrics(target, score)
        self.assertGreater(metrics["average_precision"], 0.98)
        self.assertGreater(metrics["roc_auc"], 0.98)


class GradientGeometryTest(unittest.TestCase):
    def test_opposite_gradients_are_conflict(self):
        full = {"x": torch.tensor([1.0, 0.0])}
        sar = {"x": torch.tensor([-2.0, 0.0])}
        result = gradient_pair_statistics(full, sar, ("x",))
        self.assertAlmostEqual(result["cosine"], -1.0)
        self.assertTrue(result["conflict"])
        self.assertEqual(result["sar_only"], 0)

    def test_manifest_covers_scale_specific_groups(self):
        manifest = parameter_group_manifest(self.model if hasattr(self, "model") else TinyReleasedPath())
        for name in (
            "adapter.P2",
            "adapter.P5",
            "frm.target.P2",
            "frm.target.P5",
            "sefusion.P2",
            "sefusion.P5",
            "prn.all",
            "head.all",
        ):
            self.assertIn(name, manifest)
            self.assertTrue(manifest[name])

    def test_summary_reports_negative_fraction(self):
        rows = [
            {"g": {"cosine": -0.5, "sar_to_full_norm_ratio": 2.0}},
            {"g": {"cosine": 0.25, "sar_to_full_norm_ratio": 1.0}},
        ]
        summary = summarize_gradient_records(rows, ("g",))
        self.assertEqual(summary["g"]["negative_cosine_fraction"], 0.5)
        self.assertAlmostEqual(summary["g"]["cosine"]["mean"], -0.125)
        self.assertAlmostEqual(summary["g"]["cosine"]["median"], -0.125)


class SameCityScheduleTest(unittest.TestCase):
    @staticmethod
    def _dataset(counts):
        samples = []
        for city in EARTHMISS_CITIES["train"]:
            samples.extend(
                SimpleNamespace(city=city) for _ in range(counts.get(city, BATCH_SIZE))
            )
        return SimpleNamespace(samples=samples)

    def test_small_city_reuses_only_across_unique_batches(self):
        small_city = EARTHMISS_CITIES["train"][-1]
        counts = {
            city: BATCH_SIZE * 2 for city in EARTHMISS_CITIES["train"]
        }
        counts[small_city] = BATCH_SIZE + 3
        dataset = self._dataset(counts)

        first = _same_city_batches(dataset, batches_per_city=2, seed=17)
        second = _same_city_batches(dataset, batches_per_city=2, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2 * len(EARTHMISS_CITIES["train"]))
        for _, indices in first:
            self.assertEqual(len(indices), BATCH_SIZE)
            self.assertEqual(len(set(indices)), BATCH_SIZE)
        audit = _schedule_audit(dataset, first)
        self.assertEqual(audit[small_city]["available_tiles"], BATCH_SIZE + 3)
        self.assertEqual(audit[small_city]["scheduled_examples"], BATCH_SIZE * 2)
        self.assertEqual(audit[small_city]["scheduled_unique_tiles"], BATCH_SIZE + 3)
        self.assertEqual(audit[small_city]["cross_batch_reuses"], BATCH_SIZE - 3)

    def test_city_smaller_than_one_batch_is_rejected(self):
        small_city = EARTHMISS_CITIES["train"][-1]
        counts = {
            city: BATCH_SIZE for city in EARTHMISS_CITIES["train"]
        }
        counts[small_city] = BATCH_SIZE - 1
        with self.assertRaisesRegex(RuntimeError, "within-batch-unique"):
            _same_city_batches(self._dataset(counts), batches_per_city=1, seed=17)


class DecisionTreeTest(unittest.TestCase):
    @staticmethod
    def _reports(*, oracle=True, predictable=True, conflict=True):
        checkpoint = {"sha256": "fixed"}
        oracle_report = {
            "checkpoint": checkpoint,
            "intervention_vs_sar": {
                "frm.P5@alpha=1.00": {"actionable_causal_signal": oracle}
            },
        }
        recoverability_report = {
            "checkpoint": checkpoint,
            "predictable_at_linear_p5_level": predictable,
        }
        cosine = -0.2 if conflict else 0.2
        fraction = 0.75 if conflict else 0.25
        gradient_report = {
            "checkpoint": checkpoint,
            "summary": {
                "train": {
                    "overall": {
                        "adapter.all": {
                            "cosine": {"median": cosine},
                            "negative_cosine_fraction": fraction,
                        },
                        "frm.all": {
                            "cosine": {"median": cosine},
                            "negative_cosine_fraction": fraction,
                        },
                    }
                }
            },
        }
        return oracle_report, recoverability_report, gradient_report

    def test_three_positive_evidence_types_select_protected_compensation(self):
        result = causal_decision(*self._reports())
        self.assertEqual(
            result["decision"],
            "candidate_protected_sar_anchor_with_predictable_compensation",
        )

    def test_unpredictable_oracle_does_not_authorize_reconstruction(self):
        result = causal_decision(*self._reports(predictable=False))
        self.assertEqual(
            result["decision"],
            "protect_sar_anchor_do_not_reconstruct_unpredictable_full_state",
        )

    def test_no_oracle_signal_stops_feature_alignment(self):
        result = causal_decision(
            *self._reports(oracle=False, predictable=False, conflict=False)
        )
        self.assertEqual(
            result["decision"],
            "stop_tested_feature_alignment_no_causal_bottleneck",
        )


if __name__ == "__main__":
    unittest.main()
