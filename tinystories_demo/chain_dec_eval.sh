#!/usr/bin/env bash
# Wait for a decorrelation-regularized training (given lambda) to finish, then
# evaluate its final checkpoint: multi-seed non-saturating leakage (gender
# logit-shift + AUC + on-target control) and superposition interference.
# Robust to grad-accum step numbering by waiting for the train proc to exit.
# Usage: chain_dec_eval.sh <lambda> <ckpt_dir> <gpu>
set -uo pipefail
cd "$(dirname "$0")/.."
LAM="${1:?lambda}"; CKDIR="${2:?ckpt dir}"; GPU="${3:-0}"
RES="tinystories_demo/logs/dec_eval_lam${LAM}.log"
CFG=tinystories_demo/train_tinystories_SM-ELF-M2.yml

echo "[dec] $(date '+%F %T') waiting for decorrelation_weight=${LAM}.0 training to finish..." | tee -a "$RES"
# pattern appears only in the python train argv, not this script's own argv
while pgrep -af "python3 src/train.py" | grep -q "decorrelation_weight=${LAM}\.0"; do sleep 180; done
echo "[dec] $(date '+%F %T') training done." | tee -a "$RES"

latest=$(ls -d "$CKDIR"/checkpoint_* 2>/dev/null | sed 's#.*/checkpoint_##' | grep -E '^[0-9]+$' | sort -n | tail -1)
if [ -z "${latest:-}" ]; then echo "[dec] no checkpoint in $CKDIR" | tee -a "$RES"; exit 1; fi
CK="$CKDIR/checkpoint_${latest}"
echo "[dec] $(date '+%F %T') eval ${CK} (lambda=${LAM}) on GPU ${GPU}" | tee -a "$RES"

echo "===== LEAKAGE (multi-seed, lambda=${LAM}) =====" >> "$RES"
bash tinystories_demo/run_eval_leakage.sh "$CK" "$GPU" "$CFG" \
  --config_override manifold_dim=64 --samples-per-alpha 32 --seeds 5 --label-stories 400 >> "$RES" 2>&1
echo "===== SUPERPOSITION (lambda=${LAM}) =====" >> "$RES"
bash tinystories_demo/run_eval_superposition.sh "$CK" "$GPU" "$CFG" \
  --config_override manifold_dim=64 --label-stories 1500 --seeds 5 >> "$RES" 2>&1
echo "[dec] $(date '+%F %T') DONE lambda=${LAM} -> ${RES}" | tee -a "$RES"
