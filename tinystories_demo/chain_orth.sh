#!/usr/bin/env bash
# Wait for a k-sweep training to finish, then run the orthogonalized disentanglement
# eval on its final checkpoint. Result (incl. OFF-TARGET LEAKAGE) -> $RESULT.
# Usage: chain_orth.sh <train_pattern> <ckpt_dir> <manifold_dim> <result_log> [gpu]
set -uo pipefail
cd "$(dirname "$0")/.."

TRAIN_PAT="${1:?train pattern}"; CKPT_DIR="${2:?ckpt dir}"; MDIM="${3:?manifold_dim}"
RESULT="${4:?result log}"; GPU="${5:-0}"; TARGET_STEP="${6:-0}"

latest_step() {
  ls -d "$CKPT_DIR"/checkpoint_* 2>/dev/null | sed 's#.*/checkpoint_##' \
    | grep -E '^[0-9]+$' | sort -n | tail -1
}

if [ "${TARGET_STEP}" -gt 0 ]; then
  echo "[orth] $(date '+%F %T') waiting for checkpoint_>=${TARGET_STEP} in ${CKPT_DIR}..." | tee -a "$RESULT"
  while true; do
    s=$(latest_step); s=${s:-0}
    [ "$s" -ge "$TARGET_STEP" ] && break
    sleep 180
  done
  echo "[orth] $(date '+%F %T') checkpoint at step ${s} ready (target ${TARGET_STEP})." | tee -a "$RESULT"
else
  echo "[orth] $(date '+%F %T') waiting for training [$TRAIN_PAT] to finish..." | tee -a "$RESULT"
  # Pattern also appears in this chain's own argv, so filter for the real python line.
  while pgrep -af "$TRAIN_PAT" | grep -q "python3 src/train.py"; do sleep 300; done
  echo "[orth] $(date '+%F %T') training done." | tee -a "$RESULT"
fi

latest=$(latest_step)
if [ -z "${latest:-}" ]; then echo "[orth] no checkpoint in $CKPT_DIR" | tee -a "$RESULT"; exit 1; fi
CKPT="$CKPT_DIR/checkpoint_${latest}"
echo "[orth] $(date '+%F %T') eval ${CKPT} (k=${MDIM}) on GPU ${GPU}" | tee -a "$RESULT"

bash tinystories_demo/run_eval_steering.sh "$CKPT" "$GPU" tinystories_demo/train_tinystories_SM-ELF-M2.yml \
  --orthogonalize --config_override "manifold_dim=${MDIM}" >> "$RESULT" 2>&1
echo "[orth] $(date '+%F %T') DONE -> ${RESULT}" | tee -a "$RESULT"
