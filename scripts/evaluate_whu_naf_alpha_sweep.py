"""Evaluate residual-amplitude scaling on an existing WHU NAF-P2 checkpoint.

The trained R1 correction is ``ZeroConv(NAF(x) - Bilinear(x))``.  Because the
ZeroConv has no bias, multiplying its saved weight by ``alpha`` is exactly
equivalent to multiplying the complete residual correction by ``alpha``.
This script changes no other checkpoint parameter and performs no training.

Alpha zero is an important control, but it is not R0: it keeps the backbone,
adapter, and decoder after the R1 continuation while suppressing only the NAF
correction.  A separately trained R0 is still required for the paired claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
sys.path.insert(0, str(SEGMENTATION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_whu_naf_e0 import (  # noqa: E402
    build_test_loader,
    build_variant,
    evaluate_variant,
    file_sha256,
)
from scripts.run_whu_naf_short import read_verified_e0  # noqa: E402
from scripts.whu_cache_compat import (  # noqa: E402
    CACHE_CAPACITY,
    install_whu_cache_compat,
)
from scripts.whu_label_dtype_compat import (  # noqa: E402
    install_whu_label_dtype_compat,
)


ZERO_WEIGHT_KEY = "adapter.naf_zero_conv.weight"


def find_epoch_record(path: Path, epoch: int) -> dict:
    matches = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if record.get("epoch") == epoch:
                matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one epoch {epoch} record in {path}, found {len(matches)}"
        )
    return matches[0]


def contains_alpha(values: list[float], target: float) -> bool:
    return any(math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12) for value in values)


def validate_checkpoint(
    payload: dict,
    args: argparse.Namespace,
    baseline_sha: str,
    naf_sha: str,
) -> tuple[int, dict]:
    if not isinstance(payload, dict):
        raise TypeError("R1 checkpoint payload must be a dictionary")
    state = payload.get("model")
    protocol = payload.get("protocol")
    epoch = payload.get("epoch")
    if not isinstance(state, dict) or not isinstance(protocol, dict):
        raise TypeError("R1 checkpoint must contain model and protocol dictionaries")
    if not isinstance(epoch, int) or epoch <= 0:
        raise ValueError(f"Invalid checkpoint epoch: {epoch!r}")
    if protocol.get("variant") != "r1":
        raise ValueError("Alpha sweep requires an R1 checkpoint")
    if protocol.get("baseline_checkpoint_sha256") != baseline_sha:
        raise ValueError("R1 checkpoint baseline provenance does not match")
    if protocol.get("naf_checkpoint_sha256") != naf_sha:
        raise ValueError("R1 checkpoint NAF provenance does not match")
    if protocol.get("naf_backend") != args.naf_backend:
        raise ValueError("R1 checkpoint NAF backend does not match")
    if list(protocol.get("naf_q_tile") or []) != list(args.naf_q_tile or []):
        raise ValueError("R1 checkpoint NAF query tile does not match")
    if list(protocol.get("naf_kv_tile") or []) != list(args.naf_kv_tile or []):
        raise ValueError("R1 checkpoint NAF key/value tile does not match")
    if int(protocol.get("guidance_size", -1)) != args.guidance_size:
        raise ValueError("R1 checkpoint guidance size does not match")
    if ZERO_WEIGHT_KEY not in state:
        raise KeyError(f"R1 checkpoint is missing {ZERO_WEIGHT_KEY}")
    return epoch, protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate alpha-scaled NAF residuals from one trained WHU R1 checkpoint"
    )
    parser.add_argument("--r1-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-eval", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--naf-checkpoint", type=Path, required=True)
    parser.add_argument("--e0-result", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--guidance-size", type=int, default=224)
    parser.add_argument(
        "--naf-backend",
        choices=("cutlass-fna", "flex-fna"),
        default="cutlass-fna",
    )
    parser.add_argument("--naf-q-tile", type=int, nargs=2)
    parser.add_argument("--naf-kv-tile", type=int, nargs=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()

    for path in (
        args.r1_checkpoint,
        args.reference_eval,
        args.baseline_checkpoint,
        args.naf_checkpoint,
        args.e0_result,
    ):
        if not path.is_file():
            parser.error(f"required file does not exist: {path}")
    if args.output_path.exists():
        parser.error(f"refusing to overwrite existing output: {args.output_path}")
    if args.inference_batch_size <= 0 or args.guidance_size <= 0:
        parser.error("inference batch size and guidance size must be positive")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max-images must be positive")
    if any(not math.isfinite(alpha) or alpha < 0 for alpha in args.alphas):
        parser.error("--alphas must be finite and non-negative")
    if len(set(args.alphas)) != len(args.alphas):
        parser.error("--alphas must be unique")
    if not contains_alpha(args.alphas, 0.0) or not contains_alpha(args.alphas, 1.0):
        parser.error("--alphas must include both 0 and 1")
    if (args.naf_q_tile is None) != (args.naf_kv_tile is None):
        parser.error("--naf-q-tile and --naf-kv-tile must be set together")
    if args.naf_q_tile is not None:
        if any(value <= 0 for value in (*args.naf_q_tile, *args.naf_kv_tile)):
            parser.error("NAF tile dimensions must be positive")
        args.naf_q_tile = tuple(args.naf_q_tile)
        args.naf_kv_tile = tuple(args.naf_kv_tile)
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the WHU NAF alpha sweep")

    install_whu_label_dtype_compat()
    install_whu_cache_compat(CACHE_CAPACITY)
    e0, baseline_sha, naf_sha = read_verified_e0(args)
    # The checkpoint is produced by our paired runner. Its recorded
    # ``torch.__version__`` is a TorchVersion (a str subclass), which PyTorch
    # 2.7 intentionally rejects unless it is explicitly allowlisted.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload = torch.load(
            args.r1_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    checkpoint_epoch, protocol = validate_checkpoint(
        payload,
        args,
        baseline_sha,
        naf_sha,
    )
    reference = find_epoch_record(args.reference_eval, checkpoint_epoch)

    model, cfg = build_variant(args, "r1")
    model.load_state_dict(payload["model"], strict=True)
    zero_weight = model.adapter.naf_zero_conv.weight
    original_zero_weight = zero_weight.detach().clone()
    original_zero_norm = float(original_zero_weight.norm())
    if original_zero_norm == 0.0:
        raise ValueError("R1 checkpoint ZeroConv is still zero; there is no residual to sweep")

    loader, full_test_length = build_test_loader(args, cfg["window_size"])
    device = torch.device(args.device)
    results = []
    try:
        for alpha in args.alphas:
            with torch.no_grad():
                zero_weight.copy_(original_zero_weight * alpha)
            evaluation = evaluate_variant(
                model,
                cfg,
                loader,
                device,
                args.inference_batch_size,
            )
            evaluation.update(
                {
                    "alpha": alpha,
                    "gain_over_e0": evaluation["MIoU"] - float(e0["r0"]["MIoU"]),
                    "zero_weight_norm": float(zero_weight.detach().norm()),
                }
            )
            results.append(evaluation)
            print(json.dumps(evaluation, ensure_ascii=False, sort_keys=True))
    finally:
        with torch.no_grad():
            zero_weight.copy_(original_zero_weight)

    alpha_zero = next(result for result in results if contains_alpha([result["alpha"]], 0.0))
    alpha_one = next(result for result in results if contains_alpha([result["alpha"]], 1.0))
    for result in results:
        result["gain_over_alpha_zero"] = result["MIoU"] - alpha_zero["MIoU"]

    alpha_one_reproduced_reference = None
    if args.max_images is None:
        if alpha_one["prediction_sha256"] != reference.get("prediction_sha256"):
            raise AssertionError(
                "Alpha=1 predictions do not reproduce the recorded R1 evaluation"
            )
        for metric_name in ("MIoU", "F1", "Kappa", "Acc"):
            if alpha_one[metric_name] != float(reference[metric_name]):
                raise AssertionError(
                    f"Alpha=1 {metric_name} does not reproduce the recorded R1 evaluation"
                )
        alpha_one_reproduced_reference = True

    best = max(results, key=lambda result: result["MIoU"])
    output = {
        "status": "PASS",
        "scope": "full-test" if args.max_images is None else "subset-smoke",
        "interpretation_warning": (
            "alpha=0 suppresses the NAF correction inside the R1-trained model; "
            "it is not the separately trained R0 control"
        ),
        "r1_checkpoint": str(args.r1_checkpoint.resolve()),
        "r1_checkpoint_sha256": file_sha256(args.r1_checkpoint),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_protocol": protocol,
        "reference_eval": str(args.reference_eval.resolve()),
        "reference_eval_sha256": file_sha256(args.reference_eval),
        "verified_e0_result": str(args.e0_result.resolve()),
        "verified_e0_result_sha256": file_sha256(args.e0_result),
        "e0_miou": float(e0["r0"]["MIoU"]),
        "full_test_length": full_test_length,
        "evaluated_images": len(loader.dataset),
        "alphas": args.alphas,
        "original_zero_weight_norm": original_zero_norm,
        "alpha_one_reproduced_reference": alpha_one_reproduced_reference,
        "best_alpha": best["alpha"],
        "best_miou": best["MIoU"],
        "best_gain_over_e0": best["gain_over_e0"],
        "results": results,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"alpha_sweep_result={args.output_path.resolve()}")
    print("alpha_sweep_status=PASS")


if __name__ == "__main__":
    main()
