#!/usr/bin/env bash
# Generate unconditional TinyStories samples from a trained ELF checkpoint.
# Usage: bash tinystories_demo/run_generate.sh [checkpoint_path]
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="${1:-/mnt/faster3/lc2762/elf_tinystories_output/checkpoint_6250}"

# Same CUDA 11.8 ptxas setup as training (driver 515 / minor-version compat).
NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

exec python3 src/eval.py \
  --config tinystories_demo/train_tinystories_ELF-B.yml \
  --checkpoint_path "$CKPT" \
  --config_override eval_data_path=none \
  --config_override online_eval=false \
  --config_override eval_use_ema=false \
  --config_override num_samples=10 \
  --seed 42
