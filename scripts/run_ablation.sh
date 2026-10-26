#!/usr/bin/env bash
# Run all ablation studies (instruction 4).
set -e
MODEL=${1:-gcope}
GPUS=${2:-0}
STUDY=${3:-all}   # all | trigger_size | poison_rate | influence_vs_random |
                  # no_spectral | no_orthogonality | clean_loss_term | margin
python -m src.main --config configs/default.yaml \
  --model "$MODEL" --gpus $GPUS --ablation "$STUDY" \
  --run-name "sib2_${MODEL}"
