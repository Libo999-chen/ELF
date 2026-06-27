#!/usr/bin/env bash
# Launch one k-sweep point on the current machine: train a learned ManifoldCode
# at manifold_dim=K (same arch/budget as the others) + an auto-eval chain that
# runs the orthogonalized disentanglement eval at the epoch-N checkpoint.
#
# Usage: bash tinystories_demo/launch_kpoint.sh <K> [target_step] [eval_gpu]
#   <K>           manifold_dim (e.g. 8, 512)
#   [target_step] checkpoint step to eval at (default 18750 = epoch 30); 0 = wait for full training
#   [eval_gpu]    GPU for the eval (default 7, to avoid the 0-4 training GPUs)
set -euo pipefail
cd "$(dirname "$0")/.."

K="${1:?manifold_dim, e.g. 8 or 512}"
TARGET="${2:-18750}"
EVALGPU="${3:-7}"
OUT="/mnt/faster3/lc2762/elf_tinystories_k${K}"
mkdir -p tinystories_demo/logs

: > "tinystories_demo/logs/train_k${K}.log"
setsid nohup bash tinystories_demo/run_train_sm_m2.sh \
  --config_override "manifold_dim=${K}" \
  --config_override "output_dir=${OUT}" \
  > "tinystories_demo/logs/train_k${K}.log" 2>&1 < /dev/null & disown

: > "tinystories_demo/logs/orth_k${K}.log"
setsid nohup bash tinystories_demo/chain_orth.sh \
  "manifold_dim=${K}" "${OUT}" "${K}" \
  "tinystories_demo/logs/orth_k${K}.log" "${EVALGPU}" "${TARGET}" \
  > /dev/null 2>&1 < /dev/null & disown

sleep 4
echo "launched k=${K}: train -> ${OUT}, eval at step>=${TARGET} on GPU ${EVALGPU}"
pgrep -af "manifold_dim=${K}" | grep "python3 src/train.py" | head -1 || echo "(training starting...)"
