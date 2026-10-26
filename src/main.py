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
