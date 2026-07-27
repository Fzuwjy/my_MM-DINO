"""Convert Meta's official Hugging Face DINOv3 ViT-S weights for MM-DINO.

The official Hugging Face repository stores the checkpoint using Transformers
parameter names and safetensors. MM-DINO embeds the original DINOv3 reference
implementation, which expects the reference ``.pth`` parameter names instead.
This script performs only that deterministic name/layout conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from dinov3.hub.backbones import dinov3_vits16


OFFICIAL_REPOSITORY = "facebook/dinov3-vits16-pretrain-lvd1689m"
OUTPUT_FILENAME = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tensor(source, source_key, target, target_key, used):
    if source_key not in source:
        raise KeyError(f"Missing Hugging Face tensor: {source_key}")
    value = source[source_key].detach().cpu()
    expected = target[target_key]
    if value.shape != expected.shape:
        raise ValueError(
            f"Shape mismatch for {source_key} -> {target_key}: "
            f"got {tuple(value.shape)}, expected {tuple(expected.shape)}"
        )
    target[target_key] = value.to(dtype=expected.dtype).contiguous()
    used.add(source_key)


def convert_state_dict(hf_state, reference_state):
    """Map a Transformers DINOv3 ViT-S state dict to reference DINOv3."""
    converted = {key: value.detach().cpu().clone() for key, value in reference_state.items()}
    used = set()

    direct = {
        "embeddings.cls_token": "cls_token",
        "embeddings.register_tokens": "storage_tokens",
        "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
    }
    for source_key, target_key in direct.items():
        _copy_tensor(hf_state, source_key, converted, target_key, used)

    mask_token = hf_state.get("embeddings.mask_token")
    if mask_token is None:
        raise KeyError("Missing Hugging Face tensor: embeddings.mask_token")
    mask_token = mask_token.detach().cpu()
    if mask_token.ndim == 3 and mask_token.shape[0] == 1:
        mask_token = mask_token.squeeze(0)
    if mask_token.shape != converted["mask_token"].shape:
        raise ValueError(
            "Shape mismatch for embeddings.mask_token -> mask_token: "
            f"got {tuple(mask_token.shape)}, expected {tuple(converted['mask_token'].shape)}"
        )
    converted["mask_token"] = mask_token.to(converted["mask_token"].dtype).contiguous()
    used.add("embeddings.mask_token")

    for index in range(12):
        source_prefix = f"model.layer.{index}"
        target_prefix = f"blocks.{index}"
        layer_direct = {
            f"{source_prefix}.norm1.weight": f"{target_prefix}.norm1.weight",
            f"{source_prefix}.norm1.bias": f"{target_prefix}.norm1.bias",
            f"{source_prefix}.attention.o_proj.weight": f"{target_prefix}.attn.proj.weight",
            f"{source_prefix}.attention.o_proj.bias": f"{target_prefix}.attn.proj.bias",
            f"{source_prefix}.layer_scale1.lambda1": f"{target_prefix}.ls1.gamma",
            f"{source_prefix}.norm2.weight": f"{target_prefix}.norm2.weight",
            f"{source_prefix}.norm2.bias": f"{target_prefix}.norm2.bias",
            f"{source_prefix}.mlp.up_proj.weight": f"{target_prefix}.mlp.fc1.weight",
            f"{source_prefix}.mlp.up_proj.bias": f"{target_prefix}.mlp.fc1.bias",
            f"{source_prefix}.mlp.down_proj.weight": f"{target_prefix}.mlp.fc2.weight",
            f"{source_prefix}.mlp.down_proj.bias": f"{target_prefix}.mlp.fc2.bias",
            f"{source_prefix}.layer_scale2.lambda1": f"{target_prefix}.ls2.gamma",
        }
        for source_key, target_key in layer_direct.items():
            _copy_tensor(hf_state, source_key, converted, target_key, used)

        q_weight_key = f"{source_prefix}.attention.q_proj.weight"
        k_weight_key = f"{source_prefix}.attention.k_proj.weight"
        v_weight_key = f"{source_prefix}.attention.v_proj.weight"
        qkv_weight = torch.cat(
            [hf_state[q_weight_key], hf_state[k_weight_key], hf_state[v_weight_key]], dim=0
        ).detach().cpu()
        target_qkv_weight = f"{target_prefix}.attn.qkv.weight"
        if qkv_weight.shape != converted[target_qkv_weight].shape:
            raise ValueError(
                f"Shape mismatch for layer {index} QKV weight: got {tuple(qkv_weight.shape)}, "
                f"expected {tuple(converted[target_qkv_weight].shape)}"
            )
        converted[target_qkv_weight] = qkv_weight.to(
            converted[target_qkv_weight].dtype
        ).contiguous()
        used.update({q_weight_key, k_weight_key, v_weight_key})

        q_bias_key = f"{source_prefix}.attention.q_proj.bias"
        v_bias_key = f"{source_prefix}.attention.v_proj.bias"
        q_bias = hf_state[q_bias_key].detach().cpu()
        v_bias = hf_state[v_bias_key].detach().cpu()
        qkv_bias = torch.cat([q_bias, torch.zeros_like(q_bias), v_bias], dim=0)
        target_qkv_bias = f"{target_prefix}.attn.qkv.bias"
        if qkv_bias.shape != converted[target_qkv_bias].shape:
            raise ValueError(
                f"Shape mismatch for layer {index} QKV bias: got {tuple(qkv_bias.shape)}, "
                f"expected {tuple(converted[target_qkv_bias].shape)}"
            )
        converted[target_qkv_bias] = qkv_bias.to(converted[target_qkv_bias].dtype).contiguous()
        used.update({q_bias_key, v_bias_key})

    unused = sorted(set(hf_state) - used)
    if unused:
        raise ValueError("Unexpected Hugging Face tensors: " + ", ".join(unused))

    return converted


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert official Hugging Face DINOv3 ViT-S safetensors to MM-DINO .pth"
    )
    parser.add_argument("--input", required=True, help="Path to the official model.safetensors")
    parser.add_argument("--output", required=True, help=f"Output path (normally {OUTPUT_FILENAME})")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Official Hugging Face checkpoint not found: {source_path}")
    if source_path.suffix != ".safetensors":
        raise ValueError("The input must be the official model.safetensors file")

    hf_state = load_file(str(source_path), device="cpu")
    reference_model = dinov3_vits16(pretrained=False)
    converted = convert_state_dict(hf_state, reference_model.state_dict())
    reference_model.load_state_dict(converted, strict=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)
    manifest = {
        "source_repository": OFFICIAL_REPOSITORY,
        "source_file": source_path.name,
        "source_sha256": file_sha256(source_path),
        "output_file": output_path.name,
        "output_sha256": file_sha256(output_path),
        "tensor_count": len(converted),
        "strict_load": True,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"converted_checkpoint={output_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    raise SystemExit(main())
