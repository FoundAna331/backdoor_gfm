#!/usr/bin/env bash
# Single attack run. Select ONE model at a time (instruction 1).
set -e
MODEL=${1:-gcope}        # gcope | samgpt | gft
DEFENSE=${2:-none}       # none | prune | dominant | rigbd | tcf
GPUS=${3:-0}             # e.g. "0" or "0 1 2"
PARALLEL=""
# shard attack targets over all listed GPUs when more than one is given
if [ "$(echo $GPUS | wc -w)" -gt 1 ]; then PARALLEL="--parallel-targets"; fi

python -m src.main \
  --config configs/default.yaml \
  --model "$MODEL" \
  --defense "$DEFENSE" \
  --gpus $GPUS $PARALLEL \
  --run-name "sib2_${MODEL}_${DEFENSE}"
