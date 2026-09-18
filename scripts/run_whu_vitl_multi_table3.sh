#!/usr/bin/env bash
set -Eeuo pipefail

# Self-contained launcher for Table III DINOv3-L/16 Multi on ephemeral PAI pods.
# Persistent data, weights, runtime wheels, and outputs live on the shared NFS.

PROJECT_ROOT="${MM_DINO_PERSISTENT_ROOT:-/mnt/csip-113/wjy/MM-DINO}"
REPO_ROOT="${MM_DINO_REPO_ROOT:-${PROJECT_ROOT}/repo}"
DATA_ROOT="${PROJECT_ROOT}/datasets/whu-opt-sar"
WEIGHTS_ROOT="${PROJECT_ROOT}/weights"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/whu-vitl-multi-table3"
RUNTIME_ROOT="${PROJECT_ROOT}/runtime"
RUNTIME_DEPS="${RUNTIME_ROOT}/python-packages"
WHEEL_ROOT="${RUNTIME_ROOT}/wheels"
TORCH_HOME_ROOT="${RUNTIME_ROOT}/torch-home"
CONDA_ROOT="${MM_DINO_CONDA_ROOT:-/opt/conda}"
BACKBONE="dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
MASTER_PORT="${MASTER_PORT:-29551}"
REQUIRED_BASE_COMMIT="a1a5c9985dae5e72741dab3d6676b7f5626278f6"

usage() {
    printf 'Usage: %s [--preflight-only]\n' "$0"
}

mode="run"
case "${1:-}" in
    "") ;;
    --preflight-only) mode="preflight" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

fail() {
    printf 'PRECHECK_ERROR: %s\n' "$*" >&2
    exit 20
}

ensure_link() {
    local link_path="$1"
    local target_path="$2"
    local current_path

    [[ -e "$target_path" ]] || fail "link target is missing: $target_path"
    target_path="$(readlink -f "$target_path")"
    mkdir -p "$(dirname "$link_path")"

    if [[ -L "$link_path" ]]; then
        current_path="$(readlink -f "$link_path" || true)"
        if [[ "$current_path" != "$target_path" ]]; then
            ln -sfn "$target_path" "$link_path"
        fi
    elif [[ -e "$link_path" ]]; then
        current_path="$(readlink -f "$link_path")"
        [[ "$current_path" == "$target_path" ]] || \
            fail "refusing to replace non-matching path: $link_path"
    else
        ln -s "$target_path" "$link_path"
    fi

    current_path="$(readlink -f "$link_path")"
    [[ "$current_path" == "$target_path" ]] || \
        fail "link verification failed: $link_path"
}

[[ -d "$REPO_ROOT/.git" ]] || fail "repository is missing: $REPO_ROOT"
[[ -d "$DATA_ROOT" ]] || fail "WHU dataset is missing: $DATA_ROOT"
[[ -s "$WEIGHTS_ROOT/$BACKBONE" ]] || fail "backbone is missing: $WEIGHTS_ROOT/$BACKBONE"
[[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]] || fail "Conda hook is missing"

mkdir -p "$OUTPUT_ROOT/launcher-logs" "$RUNTIME_DEPS" "$WHEEL_ROOT"
launch_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
launch_log="$OUTPUT_ROOT/launcher-logs/${launch_stamp}_host-$(hostname)_pid-$$.log"
exec > >(tee -a "$launch_log") 2>&1

printf 'launcher_log=%s\n' "$launch_log"
printf 'host=%s\n' "$(hostname)"
printf 'repo_root=%s\n' "$REPO_ROOT"
printf 'data_root=%s\n' "$DATA_ROOT"
printf 'weights_root=%s\n' "$WEIGHTS_ROOT"
printf 'output_root=%s\n' "$OUTPUT_ROOT"

# Rebuild author hard-coded paths on every ephemeral pod.
ensure_link "/home/yyyjvm/SS-datasets/whu-opt-sar" "$DATA_ROOT"
mkdir -p /home/yyyjvm/Checkpoints/facebook
ensure_link "/home/yyyjvm/Checkpoints/facebook/$BACKBONE" "$WEIGHTS_ROOT/$BACKBONE"
ensure_link "/home/yyyjvm/SS-projects/dinov3" "$REPO_ROOT"
ensure_link "$REPO_ROOT/tasks/segmentation/logs" "$OUTPUT_ROOT"

# Avoid copying the 1.13 GiB backbone into each pod's system disk.
mkdir -p "$TORCH_HOME_ROOT/hub/checkpoints"
ensure_link "$TORCH_HOME_ROOT/hub/checkpoints/$BACKBONE" "$WEIGHTS_ROOT/$BACKBONE"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate base

export PYTHONPATH="$RUNTIME_DEPS${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="$TORCH_HOME_ROOT"
export MM_DINO_WHU_CACHE_CAPACITY=32
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=1200

if ! python -c 'import imagecodecs' >/dev/null 2>&1; then
    wheel="$(find "$WHEEL_ROOT" -maxdepth 1 -type f -name 'imagecodecs-2025.11.11-*.whl' -print -quit)"
    [[ -n "$wheel" ]] || fail "persistent imagecodecs wheel is missing under $WHEEL_ROOT"
    python -m pip install --no-input --no-deps --target "$RUNTIME_DEPS" "$wheel"
fi

python -c 'import imagecodecs; print(f"imagecodecs={imagecodecs.__version__}")'
python -c 'import torch, torchvision; print(f"torch={torch.__version__}, cuda={torch.version.cuda}, torchvision={torchvision.__version__}")'

cd "$REPO_ROOT"
actual_commit="$(git rev-parse HEAD)"
git merge-base --is-ancestor "$REQUIRED_BASE_COMMIT" "$actual_commit" || \
    fail "repository commit $actual_commit does not contain required base $REQUIRED_BASE_COMMIT"
[[ -z "$(git status --porcelain)" ]] || fail "repository worktree is dirty"

for modality in optical sar lbl; do
    count="$(find "$DATA_ROOT/$modality" -maxdepth 1 -type f -name '*.tif' | wc -l)"
    [[ "$count" -eq 100 ]] || fail "$modality file count is $count, expected 100"
done
cmp -s "$DATA_ROOT/train_list.txt" "$REPO_ROOT/splits/whu/official_train.txt" || \
    fail "train_list.txt differs from the sealed official split"
cmp -s "$DATA_ROOT/test_list.txt" "$REPO_ROOT/splits/whu/official_test.txt" || \
    fail "test_list.txt differs from the sealed official split"

python -c \
    'from pathlib import Path; from skimage.io import imread; p=Path("/home/yyyjvm/SS-datasets/whu-opt-sar/lbl/NH49E001013.tif"); a=imread(p); assert a.shape == (3704, 5556); print(f"whu_decode={a.shape},{a.dtype}")'

gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
[[ -z "$gpu_processes" ]] || fail "GPU is already occupied by process IDs: $gpu_processes"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version \
    --format=csv,noheader
free -h

printf 'repo_commit=%s\n' "$actual_commit"
printf 'whu_cache_capacity=%s\n' "$MM_DINO_WHU_CACHE_CAPACITY"
printf 'torch_home=%s\n' "$TORCH_HOME"
printf 'preflight=PASSED\n'

if [[ "$mode" == "preflight" ]]; then
    exit 0
fi

command=(
    torchrun
    --nnodes=1
    --node_rank=0
    --nproc_per_node=1
    --master_addr=127.0.0.1
    --master_port="$MASTER_PORT"
    scripts/run_seeded_official.py
    --model-name DINOv3
    --dataset-name WHU
    --num-modalities 2
    --backbone-type dinov3_vitl16
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
