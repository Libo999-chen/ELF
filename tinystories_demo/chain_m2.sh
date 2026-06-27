#!/usr/bin/env bash
# Autonomous chain: wait for M1 training to finish (frees GPUs 0-4) -> train M2
# -> evaluate M2 with C1 (semantic smoke) and C2 (steering). All hands-off.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG="tinystories_demo/logs/chain_m2.log"
M1_PAT="train.py --config tinystories_demo/train_tinystories_SM-ELF.yml"
M2_CFG="tinystories_demo/train_tinystories_SM-ELF-M2.yml"
M2_OUT="/mnt/faster3/lc2762/elf_tinystories_sm_m2_output"
mkdir -p tinystories_demo/logs

# Shared CUDA 11.8 ptxas env for the eval python calls (old driver compat).
setup_env() {
  NVCC_DIR=$(python3 -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))")
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=${NVCC_DIR} ${XLA_FLAGS:-}"
  export PATH="${NVCC_DIR}/bin:${PATH}"
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  NV_DIR=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))")
  export LD_LIBRARY_PATH="$(ls -d ${NV_DIR}/*/lib 2>/dev/null | tr '\n' ':')/usr/local/cuda-11.7/lib64:${LD_LIBRARY_PATH:-}"
  export HF_HOME="${HF_HOME:-/mnt/faster3/lc2762/hf_cache}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/faster3/lc2762/hf_cache}"
}

pick_free_gpu() {  # index of the GPU with the most free memory
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' '
}

echo "[chain] $(date '+%F %T') waiting for M1 training to finish..." | tee -a "$LOG"
while pgrep -f "$M1_PAT" >/dev/null; do sleep 300; done
echo "[chain] $(date '+%F %T') M1 finished. Launching M2 training (GPUs 0-4)." | tee -a "$LOG"

# --- M2 training (foreground; blocks until done) ---
bash tinystories_demo/run_train_sm_m2.sh >> tinystories_demo/logs/train_sm_m2.log 2>&1
echo "[chain] $(date '+%F %T') M2 training exited." | tee -a "$LOG"

# --- find latest M2 checkpoint ---
latest=$(ls -d "$M2_OUT"/checkpoint_* 2>/dev/null | sed 's#.*/checkpoint_##' | grep -E '^[0-9]+$' | sort -n | tail -1)
if [ -z "${latest:-}" ]; then
  echo "[chain] ERROR: no M2 checkpoint found in $M2_OUT; skipping eval." | tee -a "$LOG"; exit 1
fi
CKPT="$M2_OUT/checkpoint_${latest}"
setup_env
GPU=$(pick_free_gpu)
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[chain] $(date '+%F %T') evaluating M2 ${CKPT} on GPU ${GPU}" | tee -a "$LOG"

# --- C1: fixed-phi semantic smoke (M2 config so the manifold encoder is built) ---
echo "==================== C1 @ M2 ${CKPT} ====================" >> tinystories_demo/logs/c1_m2_result.log
python3 src/eval_semantic.py --config "$M2_CFG" --checkpoint_path "$CKPT" \
  --num-phi 8 --samples-per-phi 16 >> tinystories_demo/logs/c1_m2_result.log 2>&1
echo "[chain] $(date '+%F %T') C1 done -> logs/c1_m2_result.log" | tee -a "$LOG"

# --- C2: sentiment steering sweep ---
echo "==================== C2 @ M2 ${CKPT} ====================" >> tinystories_demo/logs/c2_m2_result.log
python3 src/eval_steering.py --config "$M2_CFG" --checkpoint_path "$CKPT" \
  --label-stories 200 --samples-per-alpha 24 --alphas -3,-2,-1,0,1,2,3 \
  >> tinystories_demo/logs/c2_m2_result.log 2>&1
echo "[chain] $(date '+%F %T') C2 done -> logs/c2_m2_result.log" | tee -a "$LOG"
echo "[chain] $(date '+%F %T') ALL DONE." | tee -a "$LOG"
