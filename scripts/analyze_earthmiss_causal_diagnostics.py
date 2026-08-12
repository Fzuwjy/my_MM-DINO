"""Apply the pre-registered decision tree to three causal diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.earthmiss_scale_transition_common import write_json_exclusive  # noqa: E402
from scripts.earthmiss_causal_diagnostics_common import (  # noqa: E402
    ORACLE_SCREEN_STAGES,
)


SCHEMA = "earthmiss_missing_causal_decision_v1"
ORACLE_SCHEMA = "earthmiss_missing_causal_oracle_v1"
RECOVERABILITY_SCHEMA = "earthmiss_sar_full_recoverability_probe_v1"
GRADIENT_SCHEMA = "earthmiss_missing_gradient_conflict_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--recoverability", required=True)
    parser.add_argument("--gradients", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _load(path: str | Path, schema: str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise ValueError(f"unexpected schema in {path}: {payload.get('schema')!r}")
    if payload.get("formal") is not True:
        raise ValueError(f"decision requires a formal report: {path}")
    split = str(payload.get("split", ""))
    if "Test" not in split or "not accessed" not in split:
        raise ValueError(f"report does not explicitly exclude Test: {path}")
    return payload


def causal_decision(
    oracle: Mapping[str, Any],
    recoverability: Mapping[str, Any],
    gradients: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoints = {
        str(report["checkpoint"]["sha256"])
        for report in (oracle, recoverability, gradients)
    }
    if len(checkpoints) != 1:
        raise ValueError("causal reports use different checkpoints")
    actionable = sorted(
        name
        for name, row in oracle["intervention_vs_sar"].items()
        if row.get("actionable_causal_signal") is True
    )
    predictable = recoverability.get("predictable_at_linear_p5_level") is True
    train_summary = gradients["summary"].get("train", {}).get("overall", {})
    conflict_groups = []
    for group in ("adapter.all", "frm.all"):
        row = train_summary.get(group, {})
        median = row.get("cosine", {}).get("median")
        negative_fraction = row.get("negative_cosine_fraction")
        if (
            median is not None
            and median < 0.0
            and negative_fraction is not None
            and negative_fraction >= 0.5
        ):
            conflict_groups.append(group)
    shared_conflict = bool(conflict_groups)

    if not actionable:
        if shared_conflict:
            decision = "protect_sar_optimization_without_full_feature_alignment"
            rationale = (
                "Shared optimization conflict exists, but tested single-scale "
                "Full tensors lack actionable downstream causal sufficiency."
            )
        else:
            decision = "stop_tested_feature_alignment_no_causal_bottleneck"
            rationale = (
                "No tested single-scale Full intervention improves SAR enough "
                "to justify distillation or reconstruction."
            )
    elif not predictable:
        if shared_conflict:
            decision = "protect_sar_anchor_do_not_reconstruct_unpredictable_full_state"
            rationale = (
                "Full tensors can help downstream, but their correction pattern "
                "is not linearly recoverable from SAR P5 and shared gradients conflict."
            )
        else:
            decision = "stop_direct_privileged_transfer_at_tested_sar_p5"
            rationale = (
                "Oracle usefulness without SAR recoverability does not authorize "
                "missing-feature reconstruction or teacher imitation."
            )
    elif shared_conflict:
        decision = "candidate_protected_sar_anchor_with_predictable_compensation"
        rationale = (
            "Oracle usefulness, SAR recoverability, and shared-gradient conflict "
            "jointly support a protected SAR path plus SAR-predictable compensation."
        )
    else:
        decision = "candidate_transfer_predictor_without_shared_path_rewrite"
        rationale = (
            "Oracle usefulness and SAR recoverability are supported without "
            "major Adapter/FRM gradient conflict; change the transfer predictor/unit first."
        )

    return {
        "schema": SCHEMA,
        "formal": True,
        "training_was_performed": False,
        "test_was_accessed": False,
        "checkpoint_sha256": checkpoints.pop(),
        "evidence": {
            "actionable_oracle_variants": actionable,
            "linear_p5_recoverability_supported": predictable,
            "shared_gradient_conflict_supported": shared_conflict,
            "conflict_groups": conflict_groups,
        },
        "frozen_rules": {
            "oracle": "gain>=0.25 pp and >=2/3 Val cities nonnegative",
            "recoverability": "all four pre-registered probe gates pass",
            "gradient_conflict": (
                "train-BN median cosine<0 and negative fraction>=0.5 in "
                "Adapter or FRM"
            ),
        },
        "decision": decision,
        "rationale": rationale,
        "prohibited_followups": [
            "selecting a route from Test",
            "treating aggregate Full replacement controls as a trainable module",
            "temperature/lambda/scale sweeps on failed V3/FAM/prototype arms",
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite causal decision: {output}")
    oracle = _load(args.oracle, ORACLE_SCHEMA)
    recoverability = _load(args.recoverability, RECOVERABILITY_SCHEMA)
    gradients = _load(args.gradients, GRADIENT_SCHEMA)
    if tuple(oracle.get("protocol", {}).get("stages", ())) != ORACLE_SCREEN_STAGES:
        raise ValueError("causal decision requires the complete oracle screen")
    if oracle.get("protocol", {}).get("alphas") != [1.0]:
        raise ValueError("causal decision requires the alpha=1 oracle screen")
    if set(gradients.get("protocol", {}).get("bn_modes", ())) != {"train", "eval"}:
        raise ValueError("causal decision requires train-BN and eval-BN gradients")
    result = causal_decision(oracle, recoverability, gradients)
    write_json_exclusive(output, result)
    print({"output": str(output), "decision": result["decision"]})


if __name__ == "__main__":
    main()
