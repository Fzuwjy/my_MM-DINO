"""Apply the frozen Val decision rules to completed F2S-CSPT v0 arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_earthmiss_frmp5_prototype_v0 import (
    ELIGIBLE_SELECTION_EPOCHS,
    EPOCHS,
    PROTOCOL_REVISION,
)


SAR_P0_GAIN_PP = 0.50
FULL_GUARD_PP = -0.50
NONNEGATIVE_CITIES = 2
SAR_P1_GAIN_PP = 0.30


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-metrics", required=True)
    parser.add_argument("--p2-metrics", required=True)
    parser.add_argument("--p1-metrics")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _load_arm(metrics_path, expected_arm):
    metrics_path = Path(metrics_path)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    run_path = metrics_path.parent / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"run metadata not found: {run_path}")
    protocol = json.loads(run_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError(f"unexpected protocol for {expected_arm}")
    if protocol.get("arm") != expected_arm or protocol.get("formal") is not True:
        raise ValueError(f"metrics do not describe formal arm {expected_arm}")
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epochs = [record.get("epoch") for record in records]
    if epochs != list(range(1, EPOCHS + 1)):
        raise ValueError(
            f"{expected_arm} must contain exactly epochs 1..{EPOCHS}, got {epochs}"
        )
    eligible = [
        record
        for record in records
        if record.get("epoch") in ELIGIBLE_SELECTION_EPOCHS
        and record.get("eligible_for_selection") is True
        and "validation" in record
    ]
    if len(eligible) != len(ELIGIBLE_SELECTION_EPOCHS):
        raise ValueError(f"{expected_arm} lacks an eligible Val checkpoint")
    selected = max(
        eligible,
        key=lambda record: record["validation"]["sar"]["pooled"]["mIoU"],
    )
    return {
        "arm": expected_arm,
        "path": str(metrics_path),
        "protocol": protocol,
        "records": records,
        "selected": selected,
    }


def _pp(left, right):
    return 100.0 * (float(left) - float(right))


def _selected_summary(arm):
    record = arm["selected"]
    validation = record["validation"]
    return {
        "epoch": record["epoch"],
        "sar_mIoU": validation["sar"]["pooled"]["mIoU"],
        "full_mIoU": validation["full"]["pooled"]["mIoU"],
        "sar_by_city": {
            city: metrics["mIoU"]
            for city, metrics in validation["sar"]["by_city"].items()
        },
        "full_by_city": {
            city: metrics["mIoU"]
            for city, metrics in validation["full"]["by_city"].items()
        },
    }


def _trace_comparison(left, right):
    rows = []
    for left_record, right_record in zip(left["records"], right["records"]):
        left_digest = left_record["train"]["data_trace_sha256"]
        right_digest = right_record["train"]["data_trace_sha256"]
        rows.append(
            {
                "epoch": left_record["epoch"],
                "equal": left_digest == right_digest,
            }
        )
    mismatches = [row["epoch"] for row in rows if not row["equal"]]
    return {
        "all_50_epochs_equal": not mismatches,
        "warmup_epochs_1_5_equal": not any(epoch <= 5 for epoch in mismatches),
        "mismatched_epochs": mismatches,
    }


def analyze(p0, p2, p1=None):
    arms = [p0, p2] + ([p1] if p1 is not None else [])
    seeds = {arm["protocol"]["seed"] for arm in arms}
    initial_states = {
        arm["protocol"].get("initial_model_state_sha256") for arm in arms
    }
    if len(seeds) != 1:
        raise ValueError("matched arms must use the same seed")
    if None in initial_states or len(initial_states) != 1:
        raise ValueError("matched arms do not share one initialized model state")

    p0_summary = _selected_summary(p0)
    p2_summary = _selected_summary(p2)
    trace_p2_p0 = _trace_comparison(p0, p2)
    cities = sorted(set(p0_summary["sar_by_city"]) | set(p2_summary["sar_by_city"]))
    if set(cities) != set(p0_summary["sar_by_city"]) or set(cities) != set(
        p2_summary["sar_by_city"]
    ):
        raise ValueError("P0/P2 Val city sets differ")
    city_deltas = {
        city: _pp(
            p2_summary["sar_by_city"][city],
            p0_summary["sar_by_city"][city],
        )
        for city in cities
    }
    p2_p0 = {
        "sar_mIoU_delta_pp": _pp(p2_summary["sar_mIoU"], p0_summary["sar_mIoU"]),
        "full_mIoU_delta_pp": _pp(
            p2_summary["full_mIoU"], p0_summary["full_mIoU"]
        ),
        "sar_city_delta_pp": city_deltas,
        "nonnegative_cities": sum(value >= 0.0 for value in city_deltas.values()),
        "data_trace": trace_p2_p0,
    }
    p2_p0["gates"] = {
        "same_initial_model": True,
        "all_epoch_data_trace_equal": trace_p2_p0["all_50_epochs_equal"],
        "sar_gain_at_least_0_50pp": p2_p0["sar_mIoU_delta_pp"] >= SAR_P0_GAIN_PP,
        "full_guard_at_least_minus_0_50pp": (
            p2_p0["full_mIoU_delta_pp"] >= FULL_GUARD_PP
        ),
        "at_least_2_of_3_cities_nonnegative": (
            p2_p0["nonnegative_cities"] >= NONNEGATIVE_CITIES
        ),
    }
    p2_p0["passed"] = all(p2_p0["gates"].values())

    p2_p1 = None
    if p1 is not None:
        p1_summary = _selected_summary(p1)
        trace_p2_p1 = _trace_comparison(p1, p2)
        p2_p1 = {
            "sar_mIoU_delta_pp": _pp(
                p2_summary["sar_mIoU"], p1_summary["sar_mIoU"]
            ),
            "data_trace": trace_p2_p1,
            "gates": {
                "same_initial_model": True,
                "all_epoch_data_trace_equal": trace_p2_p1["all_50_epochs_equal"],
                "sar_gain_at_least_0_30pp": (
                    _pp(p2_summary["sar_mIoU"], p1_summary["sar_mIoU"])
                    >= SAR_P1_GAIN_PP
                ),
            },
        }
        p2_p1["passed"] = all(p2_p1["gates"].values())
    if not p2_p0["passed"]:
        decision = "archive_v0_without_followup_sweep"
    elif p1 is None:
        decision = "run_conditional_p1_control"
    elif p2_p1["passed"]:
        decision = "proceed_to_seeds_43_44_and_matched_sar_only_a"
    else:
        decision = "reject_full_state_privilege_attribution"

    selected = {"p0": p0_summary, "p2": p2_summary}
    if p1 is not None:
        selected["p1"] = _selected_summary(p1)
    return {
        "schema": "earthmiss_frmp5_prototype_val_decision_v0",
        "formal": True,
        "split": "Val only; Test not accessed",
        "seed": next(iter(seeds)),
        "selected": selected,
        "p2_vs_p0": p2_p0,
        "p2_vs_p1": p2_p1,
        "decision": decision,
    }


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite decision report: {output}")
    p0 = _load_arm(args.p0_metrics, "p0")
    p2 = _load_arm(args.p2_metrics, "p2")
    p1 = _load_arm(args.p1_metrics, "p1") if args.p1_metrics else None
    report = analyze(p0, p2, p1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
