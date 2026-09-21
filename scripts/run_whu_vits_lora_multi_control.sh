#!/usr/bin/env bash
set -Eeuo pipefail

# Research control: WHU DINOv3-S/16 Multi with the released LoRA adapter
# construction (rank 3). This row is not reported in MM-DINO Table VI.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${MM_DINO_PERSISTENT_ROOT:-/mnt/csip-113/wjy/MM-DINO}"

export MM_DINO_REPO_ROOT="${MM_DINO_REPO_ROOT:-${project_root}/repo-vits-lora-control}"
export MM_DINO_OUTPUT_ROOT="${project_root}/outputs/whu-vits-lora-multi-control"
export MM_DINO_BACKBONE_FILENAME="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
export MM_DINO_BACKBONE_TYPE="dinov3_vits16"
export MM_DINO_USE_LORA=1
export MM_DINO_LORA_RANK=3

exec bash "${script_dir}/run_whu_vitl_multi_table3.sh" "$@"
