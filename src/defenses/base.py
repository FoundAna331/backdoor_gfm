"""
Defense interface. A defense takes the (possibly triggered) graph and returns a
purified graph plus a report dict. Defenses are applied ONE AT A TIME
(instruction 5): the runner selects a single defense per evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DefenseReport:
    name: str
    removed_edges: int = 0
    removed_nodes: int = 0
    flagged_nodes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class Defense:
    name = "base"

    def __init__(self, cfg_def):
        self.cfg = cfg_def

    def apply(self, mdg, artifacts=None, model=None):
        """Return (purified_mdg, DefenseReport)."""
        raise NotImplementedError
