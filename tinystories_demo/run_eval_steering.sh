#!/usr/bin/env bash
# Run the SM-ELF C2 steering eval on a single GPU.
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Single device so we coexist with the training job on the other GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_DEV}"
export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/faster3/lc2762/hf_cache}"

exec python3 src/eval_steering.py \
  --config tinystories_demo/train_tinystories_SM-ELF-M2.yml \
  --checkpoint_path "$CKPT" \
  --label-stories 200 --samples-per-alpha 24
