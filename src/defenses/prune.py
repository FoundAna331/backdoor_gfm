
from __future__ import annotations

from .base import Defense, DefenseReport


class PruneDefense(Defense):
    name = "prune"

    def apply(self, mdg, artifacts=None, model=None):
        import torch
        import torch.nn.functional as F
        import copy

        ei = mdg.edge_index
        x = F.normalize(mdg.x, dim=1)
        sim = (x[ei[0]] * x[ei[1]]).sum(dim=1)
        keep = sim >= self.cfg.prune_threshold
        g = copy.copy(mdg)
        g.edge_index = ei[:, keep]
        return g, DefenseReport(name="prune",
                                removed_edges=int((~keep).sum()),
                                extra={"threshold": self.cfg.prune_threshold})
