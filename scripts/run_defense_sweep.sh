#!/usr/bin/env bash
# Evaluate the attack against every defense, ONE AT A TIME (instruction 5).
set -e
MODEL=${1:-gcope}
GPUS=${2:-0}
for DEF in none prune dominant rigbd tcf; do
  echo "=================== DEFENSE: $DEF ==================="
  python -m src.main --config configs/default.yaml \
    --model "$MODEL" --defense "$DEF" --gpus $GPUS \
    --run-name "sib2_${MODEL}_${DEF}"
done
