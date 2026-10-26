
from __future__ import annotations

import copy
import json
import os
from typing import Dict, List

from ..config import Config


# --------------------------------------------------------------------------- #
#  Variant generators: each returns list[(label, mutator(cfg))]
# --------------------------------------------------------------------------- #
def _trigger_size(cfg):
    sizes = [1, 3, 5, 10]
    return [(f"trigger_size={s}", lambda c, s=s: _set(c, "attack.trigger.trigger_size", s))
            for s in sizes]


def _poison_rate(cfg):
    rates = [0.01, 0.05, 0.10, 0.20]
    return [(f"poison_rate={r}", lambda c, r=r: _set(c, "attack.poison_rate", r))
            for r in rates]


def _influence_vs_random(cfg):
    return [
        ("influence_selection", lambda c: _flag(c, use_influence=True)),
        ("random_selection", lambda c: _flag(c, use_influence=False)),
    ]


def _no_spectral(cfg):
    return [
        ("spectral_on", lambda c: _set_many(c, {
            "attack.spectral.alpha_spectral": cfg.attack.spectral.alpha_spectral or 1.0,
            "attack.spectral.hard_lowpass_projection": True})),
        ("spectral_off(alpha=0)", lambda c: _set_many(c, {
            "attack.spectral.alpha_spectral": 0.0,
            "attack.spectral.hard_lowpass_projection": False})),
    ]


def _no_orthogonality(cfg):
    base_mu = cfg.attack.orthogonal.mu_persist or 1.0
    return [
        ("orthogonality_on", lambda c: _set(c, "attack.orthogonal.mu_persist", base_mu)),
        ("orthogonality_off(mu=0)", lambda c: _set(c, "attack.orthogonal.mu_persist", 0.0)),
    ]


def _clean_loss_term(cfg):
    return [
        ("clean_loss_on", lambda c: _set(c, "attack.trigger.use_clean_loss_term", True)),
        ("clean_loss_off", lambda c: _set(c, "attack.trigger.use_clean_loss_term", False)),
    ]


def _margin(cfg):
    margins = [0.0, 0.02, 0.05, 0.10, 0.20]
    return [(f"margin_gamma={m}", lambda c, m=m: _set(c, "attack.trigger.margin_gamma", m))
            for m in margins]


_STUDIES = {
    "trigger_size": _trigger_size,
    "poison_rate": _poison_rate,
    "influence_vs_random": _influence_vs_random,
    "no_spectral": _no_spectral,
    "no_orthogonality": _no_orthogonality,
    "clean_loss_term": _clean_loss_term,
    "margin": _margin,
}


# --------------------------------------------------------------------------- #
#  Small dotted-path setters
# --------------------------------------------------------------------------- #
def _set(cfg, dotted, value):
    obj = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)
    return cfg


def _set_many(cfg, mapping):
    for k, v in mapping.items():
        _set(cfg, k, v)
    return cfg


def _flag(cfg, use_influence):
    cfg._use_influence_selection = use_influence   # read by run below
    return cfg


# --------------------------------------------------------------------------- #
def run_ablation(base_cfg: Config, which: str = "all"):
    studies = list(_STUDIES) if which == "all" else [which]
    for s in studies:
        if s not in _STUDIES:
            raise ValueError(f"Unknown ablation '{s}'. Options: {sorted(_STUDIES)} or 'all'.")

    # Pretrain ONCE and reuse across all attack-only ablations.
    from .. import pipeline
    from ..utils.device import configure_gpus, primary_device, describe_devices
    from ..utils.seed import set_seed
    from ..utils.logging_utils import print_settings
    from ..data.multi_domain import build_multi_domain
    from ..models.registry import build_model

    configure_gpus(base_cfg.runtime.gpus)
    set_seed(base_cfg.runtime.seed)
    import torch
    device = primary_device()
    print_settings(base_cfg, extra={"mode": f"ABLATION:{which}",
                                    "devices": describe_devices()})

    mdg = build_multi_domain(base_cfg.data.train_datasets, base_cfg.data,
                             base_cfg.model.name)
    model = build_model(base_cfg.model, in_dim=mdg.x.shape[1],
                        num_domains=len(mdg.domain_names))
    model = pipeline.pretrain_encoder(model, mdg, base_cfg, device)
    # Snapshot the CLEAN pretrained weights: pretrain_poison implantation (sib2.py
    # step 7) mutates `model` in place, so without restoring this snapshot before
    # every variant, later variants would start from an already-backdoored encoder
    # (and stack a second implantation on top of it) instead of each variant being
    # an independent attack-only ablation against the same clean pretrained model.
    clean_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    from ..attack.sib2 import run_sib2

    all_results: Dict[str, List[dict]] = {}
    for study in studies:
        print("\n" + "#" * 78)
        print(f"# ABLATION STUDY: {study}")
        print("#" * 78)
        variants = _STUDIES[study](base_cfg)
        study_rows = []
        for label, mutate in variants:
            cfg = copy.deepcopy(base_cfg)
            cfg._use_influence_selection = True
            mutate(cfg)
            use_infl = getattr(cfg, "_use_influence_selection", True)
            print(f"\n--- variant: {label}  (influence_selection={use_infl}) ---")
            mdg.to(device)
            model.load_state_dict(clean_state)        # reset before each variant
            artifacts = run_sib2(model, mdg, cfg, device, verbose=True,
                                 use_influence_selection=use_infl)
            results = pipeline.evaluate(model, cfg, device, artifacts, verbose=True)
            row = {
                "variant": label,
                "r_spec_final": artifacts.r_spec_final,
                "davis_kahan": artifacts.davis_kahan,
                "results": [r.__dict__ for r in results],
            }
            study_rows.append(row)
        all_results[study] = study_rows

    out_dir = f"{base_cfg.runtime.output_dir}/{base_cfg.runtime.run_name}_ablation"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"ablation_{which}.json"), "w") as fh:
        json.dump(all_results, fh, indent=2, default=float)
    print(f"\n[ablation] wrote {os.path.join(out_dir, f'ablation_{which}.json')}")
    return all_results
