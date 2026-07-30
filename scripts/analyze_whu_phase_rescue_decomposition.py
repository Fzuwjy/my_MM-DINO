"""Zero-forward D0/D1/D2 decomposition of the frozen Stage-B0 rescue policy.

This analysis does not search a new policy.  D1 is the deterministic projection
of the final exact-cost D2 levels obtained by replacing every K4 cell with K2.
It therefore separates the output consequence of the frozen K1/K2 choices from
the later sparse K2-to-K4 upgrades, subject to the caveats recorded in the
output artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_whu_phase_closure import (
    _git_revision,
    atomic_write_json,
    file_sha256,
    load_stage_a,
    metric_summary,
)
from scripts.phase_closure_common import (
    build_phase_closure_geometry,
    closure_masks_from_levels,
    confusion_for_levels,
    summarize_closure,
    validate_levels,
)


ARTIFACT_TYPE = "whu_phase_rescue_decomposition"
SCHEMA_VERSION = 1
EXPECTED_B0_TYPE = "whu_phase_closure_stage_b0"
EXPECTED_B0_OUTCOME = "PROVISIONAL_GO_B1_KNOWN_B0_GATES_PASS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-b0-json", type=Path, required=True)
    parser.add_argument(
        "--stage-a-json",
        type=Path,
        help=(
            "Optional local Stage-A path. If omitted, use source_stage_a.path "
            "from the B0 artifact. The recorded SHA256 is always enforced."
        ),
    )
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def load_stage_b0(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != EXPECTED_B0_TYPE:
        raise ValueError("input is not a WHU Stage-B0 phase-closure artifact")
    if payload.get("schema_version") != 1:
        raise ValueError("decomposition requires Stage-B0 schema version 1")
    if payload.get("status") != "PASS" or payload.get("scope") != "full-test":
        raise ValueError("decomposition requires a full-test PASS Stage-B0 artifact")
    decision = payload.get("stage_b0_decision", {})
    if (
        decision.get("outcome") != EXPECTED_B0_OUTCOME
        or decision.get("scientific_decision_evaluated") is not True
        or decision.get("known_checks_passed") is not True
        or decision.get("b1_implementation_and_smoke_authorized") is not True
    ):
        raise ValueError("Stage B0 did not authorize the frozen B1 rescue analysis")
    if not isinstance(payload.get("exact_cost_a2_oracle"), Mapping):
        raise ValueError("Stage-B0 artifact lacks the exact-cost A2 rescue oracle")
    return payload


def projected_d1_levels(
    d2_levels: Sequence[int] | np.ndarray,
    geometry: Mapping[str, Any],
) -> np.ndarray:
    """Project the frozen D2 map onto K1/K2 without rerunning an oracle."""

    levels = validate_levels(d2_levels, geometry)
    return np.where(levels >= 2, 2, 1).astype(np.int64, copy=False)


def _per_image_confusions(
    stage_a: Mapping[str, Any], levels: np.ndarray
) -> list[np.ndarray]:
    cells = stage_a["cells"]
    image_ids = np.asarray(
        [int(cell["image_index"]) for cell in cells], dtype=np.int64
    )
    records: list[np.ndarray] = []
    for image in stage_a["images"]:
        image_index = int(image["loader_position"])
        confusion = np.asarray(image["confusion"]["k1"], dtype=np.int64).copy()
        for cell_index in np.flatnonzero(image_ids == image_index):
            cell = cells[int(cell_index)]
            cell_k1 = np.asarray(cell["confusion"]["k1"], dtype=np.int64)
            chosen = np.asarray(
                cell["confusion"][f"k{int(levels[cell_index])}"], dtype=np.int64
            )
            confusion += chosen - cell_k1
        if np.any(confusion < 0):
            raise AssertionError("per-image decomposition produced negative confusion")
        records.append(confusion)
    return records


def _condition_summary(
    *,
    name: str,
    role: str,
    levels: np.ndarray,
    confusion: np.ndarray,
    stage_a: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    class_names = stage_a["class_names"]
    per_image_confusions = _per_image_confusions(stage_a, levels)
    per_image = []
    for image, matrix in zip(stage_a["images"], per_image_confusions, strict=True):
        metrics = metric_summary(matrix, class_names)
        per_image.append(
            {
                "image_index": int(image["loader_position"]),
                "sample_name": str(image["sample_name"]),
                "confusion": metrics["confusion"],
                "errors": metrics["errors"],
                "miou": metrics["miou"],
                "miou_percent": metrics["miou_percent"],
            }
        )
    return {
        "name": name,
        "role": role,
        "levels_by_cell": levels,
        "full_image": metric_summary(confusion, class_names),
        "closure": summarize_closure(levels, geometry),
        "per_image": per_image,
    }


def _class_delta_pp(
    left: Mapping[str, float | None], right: Mapping[str, float | None]
) -> dict[str, float | None]:
    if list(left) != list(right):
        raise ValueError("class-IoU dictionaries use different class order")
    result: dict[str, float | None] = {}
    for name in left:
        lhs, rhs = left[name], right[name]
        result[name] = None if lhs is None or rhs is None else float(rhs - lhs)
    return result


def build_decomposition(
    stage_b0: Mapping[str, Any],
    stage_a: Mapping[str, Any],
) -> dict[str, Any]:
    geometry = build_phase_closure_geometry(stage_a["images"], stage_a["cells"])
    eligible = np.asarray(geometry["eligible"], dtype=bool)
    d0_levels = np.ones(len(eligible), dtype=np.int64)
    d0_levels[eligible] = 2
    d2_levels = validate_levels(
        stage_b0["exact_cost_a2_oracle"]["levels_by_cell"], geometry
    )
    d1_levels = projected_d1_levels(d2_levels, geometry)

    full_k1 = np.asarray(
        stage_a["aggregate"]["endpoints"]["k1"]["full_image"]["confusion"],
        dtype=np.int64,
    )
    confusions = {
        "d0": confusion_for_levels(stage_a["cells"], d0_levels, full_k1),
        "d1": confusion_for_levels(stage_a["cells"], d1_levels, full_k1),
        "d2": confusion_for_levels(stage_a["cells"], d2_levels, full_k1),
    }
    expected_d0 = np.asarray(
        stage_a["aggregate"]["endpoints"]["matched_k2"]["full_image"][
            "confusion"
        ],
        dtype=np.int64,
    )
    expected_d2 = np.asarray(
        stage_b0["exact_cost_a2_oracle"]["full_image"]["confusion"],
        dtype=np.int64,
    )
    if not np.array_equal(confusions["d0"], expected_d0):
        raise AssertionError("D0 no longer reproduces the matched K2x endpoint")
    if not np.array_equal(confusions["d2"], expected_d2):
        raise AssertionError("D2 no longer reproduces the frozen Stage-B0 rescue")

    conditions = {
        "d0_fixed_matched_k2x": _condition_summary(
            name="D0",
            role="uniform matched K2x on every common-support-eligible cell",
            levels=d0_levels,
            confusion=confusions["d0"],
            stage_a=stage_a,
            geometry=geometry,
        ),
        "d1_frozen_k1_k2_projection": _condition_summary(
            name="D1",
            role=(
                "deterministic projection of frozen D2: K4 becomes K2; no "
                "oracle is rerun and all other cell levels are unchanged"
            ),
            levels=d1_levels,
            confusion=confusions["d1"],
            stage_a=stage_a,
            geometry=geometry,
        ),
        "d2_frozen_exact_cost_rescue": _condition_summary(
            name="D2",
            role="unchanged exact-cost A2 rescue from the formal Stage-B0 artifact",
            levels=d2_levels,
            confusion=confusions["d2"],
            stage_a=stage_a,
            geometry=geometry,
        ),
    }
    d0 = conditions["d0_fixed_matched_k2x"]
    d1 = conditions["d1_frozen_k1_k2_projection"]
    d2 = conditions["d2_frozen_exact_cost_rescue"]

    d1_masks = closure_masks_from_levels(d1_levels, geometry)
    d2_masks = closure_masks_from_levels(d2_levels, geometry)
    same_x_closure = d1_masks["x8"] == d2_masks["x8"]
    if not same_x_closure:
        raise AssertionError("D1 and D2 must use exactly the same x8 crop closure")
    if any(d1_masks[name] != tuple(0 for _ in d1_masks[name]) for name in ("y8", "xy8")):
        raise AssertionError("D1 unexpectedly executes y8 or xy8 crops")

    d0_miou = float(d0["full_image"]["miou"])
    d1_miou = float(d1["full_image"]["miou"])
    d2_miou = float(d2["full_image"]["miou"])
    routing_gain = d1_miou - d0_miou
    k4_increment = d2_miou - d1_miou
    total_gain = d2_miou - d0_miou
    if not np.isclose(routing_gain + k4_increment, total_gain, atol=1e-15):
        raise AssertionError("D0/D1/D2 gain decomposition is not additive")

    d0_closure = d0["closure"]
    d2_closure = d2["closure"]
    baseline = int(d0_closure["baseline_k1_crop_forwards"])
    full_k2_plus_d2_yxy_extra = int(
        d0_closure["extra_crop_forwards_by_phase"]["x8"]
        + d2_closure["extra_crop_forwards_by_phase"]["y8"]
        + d2_closure["extra_crop_forwards_by_phase"]["xy8"]
    )
    return {
        "conditions": conditions,
        "decomposition": {
            "d1_minus_d0_pp": float(routing_gain * 100.0),
            "d2_minus_d1_pp": float(k4_increment * 100.0),
            "d2_minus_d0_pp": float(total_gain * 100.0),
            "d1_share_of_total_gain": (
                float(routing_gain / total_gain) if total_gain != 0 else None
            ),
            "d2_increment_share_of_total_gain": (
                float(k4_increment / total_gain) if total_gain != 0 else None
            ),
            "class_d1_minus_d0_iou_pp": _class_delta_pp(
                d0["full_image"]["class_iou_percent"],
                d1["full_image"]["class_iou_percent"],
            ),
            "class_d2_minus_d1_iou_pp": _class_delta_pp(
                d1["full_image"]["class_iou_percent"],
                d2["full_image"]["class_iou_percent"],
            ),
            "d1_and_d2_x8_closure_identical": same_x_closure,
            "full_k2x_plus_d2_yxy_hypothetical_cost": float(
                1.0 + full_k2_plus_d2_yxy_extra / baseline
            ),
        },
        "interpretation": {
            "strongest_supported": (
                "Within one frozen GT-informed D2 policy, about two thirds of "
                "its gain over uniform K2x is already present after projecting "
                "the output choices to K1/K2, while about one third is added by "
                "the 135 sparse K2-to-K4 upgrades."
            ),
            "not_supported": (
                "This is an output-path decomposition of one frozen policy, not "
                "two new oracles, a causal attribution, a deployable router, or "
                "evidence that every cell has both K1 and K2 logits available."
            ),
            "cost_caveat": (
                "D2 remains below 2x partly because it omits 174 x8 crops versus "
                "uniform support-pruned K2x. Paying full K2x first and then D2's "
                "y8/xy8 closure would exceed 2x. Closure spill never authorizes "
                "a cell to use a phase output; the frozen action level does."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_path}")
    stage_b0 = load_stage_b0(args.stage_b0_json)
    source = stage_b0["source_stage_a"]
    stage_a_path = args.stage_a_json or Path(str(source["path"]))
    actual_stage_a_sha = file_sha256(stage_a_path)
    if actual_stage_a_sha != source.get("sha256"):
        raise ValueError("Stage-A SHA256 differs from the source sealed by Stage B0")
    stage_a = load_stage_a(stage_a_path)
    analysis = build_decomposition(stage_b0, stage_a)
    output = {
        "status": "PASS",
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "scope": "full-test",
        "scientific_scope": (
            "Zero-forward output-path decomposition of one frozen GT-informed "
            "Stage-B0 rescue policy; no new policy search and no live sparse model"
        ),
        "source_stage_b0": {
            "path": str(args.stage_b0_json.resolve()),
            "sha256": file_sha256(args.stage_b0_json),
        },
        "source_stage_a": {
            "path": str(stage_a_path.resolve()),
            "sha256": actual_stage_a_sha,
        },
        "reproducibility": {
            "git_revision": _git_revision(),
            "runner_sha256": file_sha256(Path(__file__)),
            "closure_common_sha256": file_sha256(
                REPO_ROOT / "scripts" / "phase_closure_common.py"
            ),
            "numpy": np.__version__,
        },
        **analysis,
    }
    atomic_write_json(args.output_path, output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "output_path": str(args.output_path.resolve()),
                "d1_minus_d0_pp": output["decomposition"]["d1_minus_d0_pp"],
                "d2_minus_d1_pp": output["decomposition"]["d2_minus_d1_pp"],
                "d2_minus_d0_pp": output["decomposition"]["d2_minus_d0_pp"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
