#!/usr/bin/env bash
# Tiny end-to-end run to verify the install (CPU-friendly).
set -e
python -m src.main --config configs/smoke.yaml
