#!/usr/bin/env bash
set -Eeuo pipefail

# Table VI Ours S Multi: only the backbone, its official weight, and output
# namespace differ from the faithful Table III ViT-L non-LoRA launcher.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${MM_DINO_PERSISTENT_ROOT:-/mnt/csip-113/wjy/MM-DINO}"
export MM_DINO_OUTPUT_ROOT="${project_root}/outputs/whu-vits-multi-table6"
export MM_DINO_BACKBONE_FILENAME="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
export MM_DINO_BACKBONE_TYPE="dinov3_vits16"

exec bash "${script_dir}/run_whu_vitl_multi_table3.sh" "$@"
