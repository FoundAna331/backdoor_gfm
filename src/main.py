"""
SIB2-Backdoor — main entry point.

Usage examples:
  python -m src.main --model gcope --defense tcf
  python -m src.main --config configs/model_samgpt.yaml --gpus 0 1
  python -m src.main --model gft --use-node-to-node-influence false --top-k-targets 5
  python -m src.main --ablation all --model gcope

Pipeline per run:
  1. configure GPUs + seed, print all settings (instr. 7, 8)
  2. report dataset availability (instr. 2)
  3. build multi-domain pretraining graph (train datasets) per model's method (instr. 1)
  4. build + pretrain the chosen victim GFM (one model at a time, instr. 1)
  5. run the SIB2 attack on GPU (instr. 3, 6)
  6. evaluate ASR / clean-ACC on each test dataset, optionally under a defense (instr. 5)
"""
from __future__ import annotations

import argparse
import os
import sys

# allow running as `python src/main.py` as well as `python -m src.main`
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import add_cli_args, config_from_args
    from src import pipeline
else:
    from .config import add_cli_args, config_from_args
    from . import pipeline


def build_parser():
    p = argparse.ArgumentParser(
        description="SIB2-Backdoor on graph foundation models (research harness).")
    add_cli_args(p)
    p.add_argument("--ablation", type=str, default=None,
                   help="Run an ablation study instead of a single run. One of: "
                        "all, trigger_size, poison_rate, influence_vs_random, "
                        "no_spectral, no_orthogonality, clean_loss_term, margin.")
    p.add_argument("--pretrain-epochs", type=int, default=None,
                   help="Override model.pretrain_epochs (handy for quick smoke tests).")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.pretrain_epochs is not None:
        cfg.model.pretrain_epochs = args.pretrain_epochs

    if args.ablation:
        from .ablation.runner import run_ablation
        run_ablation(cfg, which=args.ablation)
    else:
        pipeline.run_experiment(cfg)


if __name__ == "__main__":
    main()
