#!/usr/bin/env bash
# Wait for an SM-ELF checkpoint at >= TARGET_STEP, then auto-run the C1 smoke test
# on the freest non-training GPU. Result (verdict + metrics) lands in C1_LOG.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="/mnt/faster3/lc2762/elf_tinystories_sm_output"
TARGET_STEP="${TARGET_STEP:-15000}"   # epoch 24 (625 steps/epoch)
C1_LOG="tinystories_demo/logs/c1_result.log"
POLL=300                               # seconds between checks
CANDIDATE_GPUS="5 6 7"                 # GPUs NOT used by the 0-4 training job

mkdir -p tinystories_demo/logs
echo "[watch_c1] started; waiting for checkpoint_>=${TARGET_STEP} in ${OUT_DIR}" | tee -a "$C1_LOG"

latest_ckpt_step() {
  local best=-1 step
  for d in "$OUT_DIR"/checkpoint_*; do
    [ -d "$d" ] || continue
    step="${d##*/checkpoint_}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    (( step > best )) && best="$step"
  done
  echo "$best"
}

pick_gpu() {
  # GPU among CANDIDATE_GPUS with most free memory.
  local best_gpu="" best_free=-1 free
  for g in $CANDIDATE_GPUS; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [[ "$free" =~ ^[0-9]+$ ]] || continue
    (( free > best_free )) && { best_free="$free"; best_gpu="$g"; }
  done
  echo "$best_gpu $best_free"
}

while true; do
  step=$(latest_ckpt_step)
  if (( step >= TARGET_STEP )); then
    CKPT="$OUT_DIR/checkpoint_${step}"
    read -r GPU FREE <<<"$(pick_gpu)"
    echo "[watch_c1] $(date '+%F %T') checkpoint_${step} ready; running C1 on GPU ${GPU} (${FREE} MiB free)" | tee -a "$C1_LOG"
    echo "==================== C1 @ checkpoint_${step} ====================" >> "$C1_LOG"
    bash tinystories_demo/run_eval_semantic.sh "$CKPT" "$GPU" >> "$C1_LOG" 2>&1
    echo "[watch_c1] C1 run finished; see ${C1_LOG}" | tee -a "$C1_LOG"
    break
  fi
  echo "[watch_c1] $(date '+%F %T') latest step=${step} (< ${TARGET_STEP}); sleeping ${POLL}s" >> "$C1_LOG"
  sleep "$POLL"
done
