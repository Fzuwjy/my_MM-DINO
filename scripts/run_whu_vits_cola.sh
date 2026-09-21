#!/usr/bin/env bash
set -Eeuo pipefail
project_root="${MM_DINO_PERSISTENT_ROOT:-/mnt/csip-113/wjy/MM-DINO}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${MM_DINO_CONDA_ROOT:-/opt/conda}/etc/profile.d/conda.sh"
conda activate base
export PYTHONPATH="${project_root}/runtime/python-packages${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${project_root}/runtime/torch-home"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$repo_root"
# Explicit --mode train is required; no arguments can start a formal run.
exec python -m research.cola.run "$@"
