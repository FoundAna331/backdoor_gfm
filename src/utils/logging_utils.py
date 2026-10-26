"""
Experiment-settings printer.

Instruction (7): print *all* experimental settings applied when performing the
attack. We render the fully-resolved config as a readable banner and also mark
the knobs that most affect the three SIB2 mechanisms so they are easy to eyeball
in logs.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from ..config import Config


def _flatten(d: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(d, dict):
        for k, v in d.items():
            out.extend(_flatten(v, f"{prefix}{k}."))
    else:
        out.append((prefix.rstrip("."), d))
    return out


def print_settings(cfg: Config, extra: dict | None = None) -> None:
    line = "=" * 78
    print(line)
    print(" SIB2-Backdoor  —  Spectral-Influence Backdoor w/ Orthogonal Anchoring")
    print(" EXPERIMENTAL SETTINGS (all applied knobs)")
    print(line)

    sections = [
        ("RUNTIME", cfg.runtime),
        ("MODEL (victim GFM)", cfg.model),
        ("DATA", cfg.data),
        ("ATTACK", cfg.attack),
        ("DEFENSE", cfg.defense),
        ("EVAL / FINE-TUNING", cfg.eval),
    ]
    for title, sec in sections:
        print(f"\n[{title}]")
        for key, val in _flatten(asdict(sec)):
            print(f"  {key:<42} = {val}")

    # Highlight the three mechanisms explicitly.
    infl = cfg.attack.influence
    print("\n[MECHANISM SUMMARY]")
    print(f"  (1) Influence target-selection : node_to_node={infl.use_node_to_node_influence}, "
          f"top_k={infl.top_k_targets} (argmax replaced by top-k), "
          f"ihvp={infl.ihvp_method}")
    print(f"  (2) Spectral low-freq anchoring : alpha={cfg.attack.spectral.alpha_spectral}, "
          f"hard_lowpass={cfg.attack.spectral.hard_lowpass_projection}, "
          f"margin_gamma={cfg.attack.trigger.margin_gamma}")
    print(f"  (3) Orthogonal persistence      : mu={cfg.attack.orthogonal.mu_persist}, "
          f"k_subspace={cfg.attack.orthogonal.top_k_subspace}")
    print(f"  clean-loss-preservation term    : {cfg.attack.trigger.use_clean_loss_term} "
          f"(gamma={cfg.attack.trigger.gamma_clean})")

    if extra:
        print("\n[CONTEXT]")
        for k, v in extra.items():
            print(f"  {k:<42} = {v}")
    print(line, flush=True)


def dump_config(cfg: Config, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "resolved_config.json")
    with open(path, "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)
    return path
