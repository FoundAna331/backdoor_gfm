"""Defense selection (instruction 5): one defense at a time."""
from __future__ import annotations

from .none import NoDefense
from .prune import PruneDefense
from .dominant import DominantDefense
from .rigbd import RIGBDDefense
from .tcf import TCFDefense

_DEFENSES = {
    "none": NoDefense,
    "prune": PruneDefense,
    "dominant": DominantDefense,
    "rigbd": RIGBDDefense,
    "tcf": TCFDefense,
}


def available_defenses():
    return sorted(_DEFENSES)


def build_defense(cfg_def):
    name = cfg_def.name.lower()
    if name not in _DEFENSES:
        raise ValueError(f"Unknown defense '{name}'. Options: {available_defenses()}")
    return _DEFENSES[name](cfg_def)
