
from __future__ import annotations

from .base import Defense, DefenseReport


class RIGBDDefense(Defense):
    name = "rigbd"

    def _embed(self, model, x, edge_index):
        import torch
        if model is not None:
            with torch.no_grad():
                return model.encode(x, edge_index)
        # proxy: one step of mean aggregation (feature propagation)
        from torch_geometric.utils import degree
        row, col = edge_index
        deg = degree(row, x.shape[0]).clamp_min(1).unsqueeze(1)
        agg = torch.zeros_like(x).index_add_(0, row, x[col]) / deg
        return agg

    def apply(self, mdg, artifacts=None, model=None):
        import torch
        from torch_geometric.utils import dropout_edge
        import copy

        n = mdg.x.shape[0]
        base = self._embed(model, mdg.x, mdg.edge_index)
        diffs = torch.zeros(n, device=mdg.x.device)
        for _ in range(self.cfg.rigbd_num_rounds):
            ei_d, _ = dropout_edge(mdg.edge_index, p=self.cfg.rigbd_drop_rate)
            emb = self._embed(model, mdg.x, ei_d)
            diffs += (emb - base).norm(dim=1)
        diffs /= self.cfg.rigbd_num_rounds

        # overly-stable (low-diff) nodes are suspicious
        thr = torch.quantile(diffs, 0.05)
        flagged = diffs < thr
        keep_node = ~flagged
        ei = mdg.edge_index
        keep_edge = keep_node[ei[0]] & keep_node[ei[1]]
        g = copy.copy(mdg)
        g.edge_index = ei[:, keep_edge]
        return g, DefenseReport(name="rigbd",
                                removed_edges=int((~keep_edge).sum()),
                                flagged_nodes=int(flagged.sum()),
                                extra={"drop_rate": self.cfg.rigbd_drop_rate,
                                       "rounds": self.cfg.rigbd_num_rounds})
