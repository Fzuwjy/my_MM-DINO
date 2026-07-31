"""Real-checkpoint smoke for the OPT-only/SAR-only feature-zero protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_ROOT = REPO_ROOT / "tasks" / "segmentation"
for path in (REPO_ROOT, SEGMENTATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.aux_diagnostics_common import FeatureZeroSampleAdapter
from scripts.evaluate_whu_aux_counterfactual import MODEL_PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the two-input feature-zero protocol"
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--model-profile",
        choices=tuple(MODEL_PROFILES),
        default="vitl-lora",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-size", type=int, default=512)
    return parser.parse_args()


def tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}")
    if args.crop_size <= 0 or args.crop_size % 16 != 0:
        raise ValueError("--crop-size must be a positive multiple of 16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from configs import get_cfg
    from datasets import build_dataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    profile = MODEL_PROFILES[args.model_profile]
    cfg = get_cfg(
        "DINOv3",
        "WHU",
        num_modalities=2,
        use_lora=profile["use_lora"],
        r=profile["lora_rank"],
        backbone_type=profile["backbone_type"],
    )
    dataset = build_dataset(
        "WHU",
        "test",
        window_size=cfg["window_size"],
        model_name="DINOv3",
        modality="multi",
        backbone_type=profile["backbone_type"],
    )
    rgb, sar, _ = dataset[0]
    height = min(args.crop_size, int(rgb.shape[-2]))
    width = min(args.crop_size, int(rgb.shape[-1]))
    height -= height % 16
    width -= width % 16
    rgb = rgb[:, :height, :width].unsqueeze(0).cuda()
    sar = sar[:, :height, :width].unsqueeze(0).cuda()
    if not bool(torch.isfinite(rgb).all()) or not bool(torch.isfinite(sar).all()):
        raise AssertionError("Smoke inputs must be finite")

    model = cfg["model"].cuda()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("Expected checkpoint['model']")
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval()

    released_adapter = model.adapter
    released_decoder = model.decoder
    backbone_calls = 0
    decoder_input_counts: list[int] = []
    released_get_intermediate_layers = model.backbone.get_intermediate_layers

    def counted_get_intermediate_layers(input_tensor, *positional, **keywords):
        nonlocal backbone_calls
        backbone_calls += 1
        return released_get_intermediate_layers(
            input_tensor, *positional, **keywords
        )

    model.backbone.get_intermediate_layers = counted_get_intermediate_layers

    def record_decoder_inputs(module, inputs):
        decoder_input_counts.append(len(inputs))

    hook = released_decoder.register_forward_pre_hook(record_decoder_inputs)

    def run(adapter):
        nonlocal backbone_calls
        backbone_calls = 0
        model.adapter = adapter
        with torch.no_grad():
            logits = model(rgb, sar)
        if backbone_calls != 2:
            raise AssertionError(
                f"Condition used {backbone_calls} backbone calls instead of 2"
            )
        if not bool(torch.isfinite(logits).all()):
            raise AssertionError("Condition produced non-finite logits")
        return logits, backbone_calls

    normal_logits, normal_calls = run(released_adapter)
    wrapped_normal_logits, wrapped_normal_calls = run(
        FeatureZeroSampleAdapter(
            released_adapter, zero_modality_index=None
        )
    )
    opt_only_logits, opt_only_calls = run(
        FeatureZeroSampleAdapter(released_adapter, zero_modality_index=1)
    )
    sar_only_logits, sar_only_calls = run(
        FeatureZeroSampleAdapter(released_adapter, zero_modality_index=0)
    )
    hook.remove()

    exact_normal_wrapper = bool(
        torch.equal(normal_logits, wrapped_normal_logits)
    )
    normal_wrapper_difference = (normal_logits - wrapped_normal_logits).abs()
    if not exact_normal_wrapper:
        raise AssertionError(
            "No-mask wrapper changed logits: max_abs="
            f"{normal_wrapper_difference.max().item()}"
        )
    if decoder_input_counts != [2, 2, 2, 2]:
        raise AssertionError(
            f"Decoder input-slot counts changed: {decoder_input_counts}"
        )

    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "model_profile": args.model_profile,
        "crop_shape": list(rgb.shape),
        "strict_checkpoint_load": True,
        "normal_wrapper_exact_logits": exact_normal_wrapper,
        "normal_wrapper_max_abs_difference": float(
            normal_wrapper_difference.max().item()
        ),
        "backbone_calls": {
            "normal": normal_calls,
            "wrapped_normal": wrapped_normal_calls,
            "opt_only": opt_only_calls,
            "sar_only": sar_only_calls,
        },
        "decoder_input_slot_counts": decoder_input_counts,
        "weights_renormalized_after_zero": False,
        "original_weight_denominator_preserved": True,
        "mask_location": "sample-adapter-post-projection-pre-weighted-sum",
        "logits_sha256": {
            "normal": tensor_sha256(normal_logits),
            "wrapped_normal": tensor_sha256(wrapped_normal_logits),
            "opt_only": tensor_sha256(opt_only_logits),
            "sar_only": tensor_sha256(sar_only_logits),
        },
        "opt_only_differs_from_normal": not torch.equal(
            opt_only_logits, normal_logits
        ),
        "sar_only_differs_from_normal": not torch.equal(
            sar_only_logits, normal_logits
        ),
        "opt_only_differs_from_sar_only": not torch.equal(
            opt_only_logits, sar_only_logits
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"protocol_smoke_result={output_path}")


if __name__ == "__main__":
    main()
