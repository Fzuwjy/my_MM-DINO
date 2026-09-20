#!/usr/bin/env bash
set -Eeuo pipefail

# WHU Table VI Ours L-LoRA Multi on a 24 GiB card. This is the documented
# microbatch-4/accumulation-2 compatibility protocol, not released batch-8.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
project_root="${MM_DINO_PERSISTENT_ROOT:-/mnt/csip-113/wjy/MM-DINO}"
output_root="${project_root}/outputs/whu-vitl-lora-multi-table6"
conda_root="${MM_DINO_CONDA_ROOT:-/opt/conda}"
master_port="${MASTER_PORT:-29551}"

case "${1:-}" in
    ""|--preflight-only) ;;
    *) printf 'Usage: %s [--preflight-only]\n' "$0" >&2; exit 2 ;;
esac

export MM_DINO_REPO_ROOT="$repo_root"
export MM_DINO_OUTPUT_ROOT="$output_root"
export MM_DINO_BACKBONE_FILENAME=dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
export MM_DINO_BACKBONE_TYPE=dinov3_vitl16

mkdir -p "$output_root/launcher-logs"
launch_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
launch_log="$output_root/launcher-logs/${launch_stamp}_host-$(hostname)_pid-$$.log"
exec > >(tee -a "$launch_log") 2>&1

printf 'protocol=WHU_VITL_LORA_MULTI_MICROBATCH4_ACCUM2\n'
printf 'fidelity=compatibility_not_exact_released_batch8\n'
printf 'launcher_log=%s\n' "$launch_log"
printf 'repo_root=%s\noutput_root=%s\n' "$repo_root" "$output_root"

# Reuse the sealed dataset, dependency, weight, split, and GPU checks.
bash "$script_dir/run_whu_vitl_multi_table3.sh" --preflight-only
if [[ "${1:-}" == --preflight-only ]]; then
    exit 0
fi

source "$conda_root/etc/profile.d/conda.sh"
conda activate base
export PYTHONPATH="${project_root}/runtime/python-packages${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${project_root}/runtime/torch-home"
export MM_DINO_WHU_CACHE_CAPACITY=32
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=1200
cd "$repo_root"

command=(
    torchrun
    --nnodes=1
    --node_rank=0
    --nproc_per_node=1
    --master_addr=127.0.0.1
    --master_port="$master_port"
    scripts/run_whu_vitl_lora_accumulated.py
    --micro-batch-size 4
    --grad-accum-steps 2
    --num-modalities 2
)
printf 'training_command='
printf ' %q' "${command[@]}"
printf '\n'
set +e
"${command[@]}"
status=$?
set -e
printf 'training_exit_code=%s\n' "$status"
exit "$status"
