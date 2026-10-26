"""
Model selection (instruction 1): choose exactly one victim GFM per run.
"""
from __future__ import annotations

from .gcope import build_gcope
from .samgpt import build_samgpt
from .gft import build_gft

_BUILDERS = {
    "gcope": build_gcope,
    "samgpt": build_samgpt,
    "gft": build_gft,
}


def available_models():
    return sorted(_BUILDERS)


def build_model(cfg_model, in_dim: int, num_domains: int):
    name = cfg_model.name.lower()
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Options: {available_models()}")
    return _BUILDERS[name](cfg_model, in_dim, num_domains)
