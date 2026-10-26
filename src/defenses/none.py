"""No-op defense (baseline: measure raw ASR)."""
from __future__ import annotations

from .base import Defense, DefenseReport


class NoDefense(Defense):
    name = "none"

    def apply(self, mdg, artifacts=None, model=None):
        return mdg, DefenseReport(name="none")
