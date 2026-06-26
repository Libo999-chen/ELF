#!/usr/bin/env bash
# Launch ELF TinyStories training on all local GPUs (single-process JAX pmap data parallelism).
#
# This machine has NVIDIA driver 515 (CUDA 11.7), so we run jax/jaxlib 0.4.25 (cuda11)
# with ptxas 11.8 from the nvidia-cuda-nvcc-cu11 pip package. Minor-version compatibility
# lets the 11.8-compiled cubins load on the 11.7 driver (parallel compilation is disabled).
set -euo pipefail

cd "$(dirname "$0")/.."

# Point XLA at the pip-provided CUDA 11.8 toolkit (ptxas/nvvm) instead of the too-new cu12 one.
NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
export PATH="${NVCC_DIR}/bin:${PATH}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"

# Do NOT launch under torchrun: train.py's distributed init is a no-op, so multi-process
# would run uncoordinated. Single process + jax.pmap uses all visible GPUs.
exec python3 src/train.py --config tinystories_demo/train_tinystories_ELF-B.yml "$@"
