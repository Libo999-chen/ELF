#!/usr/bin/env bash
# Train SM-ELF (semantic-manifold factorization) on TinyStories (M2).
set -euo pipefail
cd "$(dirname "$0")/.."

# CUDA 11.8 ptxas for the old driver (minor-version compat).
NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
# Allocate GPU memory on demand (don't grab 75% up front) so we coexist with other
# users sharing these GPUs, and avoid first-step OOM.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Exclude GPU6: another user runs there and cuBLAS init fails when it gets tight.
# Train on the 7 free GPUs (global_batch_size must stay divisible by 7).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
# Keep HF cache off the 25GB home quota.
export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/faster3/lc2762/hf_cache}"

exec python3 src/train.py --config tinystories_demo/train_tinystories_SM-ELF-M2.yml "$@"
