#!/usr/bin/env bash
# Autonomous k-sweep tail: wait for the CURRENT k-training to reach 60 epochs and
# its python proc to exit, run the @60 orthogonalized disentanglement eval on its
# final checkpoint, then (optionally) launch an ENDPOINT k (trains 60ep + auto
# eval@30) and self-spawn a waiter for that endpoint's own @60 eval.
#
# Usage: queue_chain.sh <cur_ckpt_dir> <cur_k> <gpu> [next_k] [final_step]
#   <cur_ckpt_dir>  e.g. /mnt/faster3/lc2762/elf_tinystories_k16
#   <cur_k>         manifold_dim of the current run (for the eval view + proc match)
#   <gpu>           GPU for the @60 eval (default 7)
#   [next_k]        endpoint manifold_dim to launch after (empty = just eval@60, stop)
#   [final_step]    step that marks 60ep (default 37500)
set -uo pipefail
cd "$(dirname "$0")/.."

CUR_DIR="${1:?current ckpt dir}"; CUR_K="${2:?current k}"; GPU="${3:-7}"
NEXT_K="${4:-}"; FINAL="${5:-37500}"
mkdir -p tinystories_demo/logs
LOG="tinystories_demo/logs/queue_k${CUR_K}.log"

latest_step() {
  ls -d "$CUR_DIR"/checkpoint_* 2>/dev/null | sed 's#.*/checkpoint_##' \
    | grep -E '^[0-9]+$' | sort -n | tail -1
}

echo "[queue] $(date '+%F %T') waiting: $CUR_DIR step>=$FINAL (k=$CUR_K), next=${NEXT_K:-none}" | tee -a "$LOG"
# 1) wait for the 60ep checkpoint
while true; do s=$(latest_step); s=${s:-0}; [ "$s" -ge "$FINAL" ] && break; sleep 300; done
# 2) wait for the training proc to actually release the GPUs
while pgrep -af "python3 src/train.py" | grep -q "manifold_dim=${CUR_K}\b"; do sleep 60; done
echo "[queue] $(date '+%F %T') k=$CUR_K finished at step $(latest_step); GPUs free." | tee -a "$LOG"

# 3) @60 eval on the final checkpoint
CK="$CUR_DIR/checkpoint_$(latest_step)"
RES="tinystories_demo/logs/orth_k${CUR_K}_e60.log"
echo "[queue] $(date '+%F %T') eval @60: $CK -> $RES" | tee -a "$LOG"
bash tinystories_demo/run_eval_steering.sh "$CK" "$GPU" \
  tinystories_demo/train_tinystories_SM-ELF-M2.yml \
  --orthogonalize --config_override "manifold_dim=${CUR_K}" >> "$RES" 2>&1
echo "[queue] $(date '+%F %T') @60 eval done for k=$CUR_K." | tee -a "$LOG"

# 4) optional endpoint: launch it (60ep + auto eval@30), then self-spawn a stop-waiter for its @60
if [ -n "$NEXT_K" ]; then
  echo "[queue] $(date '+%F %T') launching endpoint k=$NEXT_K" | tee -a "$LOG"
  bash tinystories_demo/launch_kpoint.sh "$NEXT_K" 18750 "$GPU" >> "$LOG" 2>&1
  sleep 10
  NEXT_DIR="/mnt/faster3/lc2762/elf_tinystories_k${NEXT_K}"
  setsid nohup bash tinystories_demo/queue_chain.sh "$NEXT_DIR" "$NEXT_K" "$GPU" "" "$FINAL" \
    > /dev/null 2>&1 < /dev/null & disown
  echo "[queue] $(date '+%F %T') endpoint k=$NEXT_K launched + @60 waiter spawned." | tee -a "$LOG"
fi
echo "[queue] $(date '+%F %T') DONE (k=$CUR_K)." | tee -a "$LOG"
