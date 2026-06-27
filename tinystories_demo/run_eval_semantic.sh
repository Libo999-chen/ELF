#!/usr/bin/env bash
# Run the SM-ELF C1 semantic smoke test on a single GPU.
# Usage: bash tinystories_demo/run_eval_semantic.sh <checkpoint_path> [cuda_device]
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:?usage: run_eval_semantic.sh <checkpoint_path> [cuda_device]}"
CUDA_DEV="${2:-7}"

# CUDA 11.8 ptxas for the old driver (minor-version compat), same as training.
NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
# Force pip CUDA libs (newer cuBLAS) ahead of system CUDA (needed on death).
NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Single device so we coexist with the training job on the other GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_DEV}"
export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/faster3/lc2762/hf_cache}"

exec python3 src/eval_semantic.py \
  --config tinystories_demo/train_tinystories_SM-ELF.yml \
  --checkpoint_path "$CKPT" \
  --num-phi 6 --samples-per-phi 8
