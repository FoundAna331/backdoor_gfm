
from __future__ import annotations

from .base import Defense, DefenseReport


def _build_dominant(in_dim, hidden, device):
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GCNConv

    class DOMINANT(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = GCNConv(in_dim, hidden)
            self.enc2 = GCNConv(hidden, hidden)
            self.attr_dec = GCNConv(hidden, in_dim)
            self.act = nn.ReLU()

        def forward(self, x, edge_index):
            h = self.act(self.enc1(x, edge_index))
            z = self.enc2(h, edge_index)
            x_hat = self.attr_dec(z, edge_index)
            return z, x_hat

    return DOMINANT().to(device)


class DominantDefense(Defense):
    name = "dominant"

    def apply(self, mdg, artifacts=None, model=None):
        import torch
        import torch.nn.functional as F
        import copy

        device = mdg.x.device
        net = _build_dominant(mdg.x.shape[1], 64, device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
        ei = mdg.edge_index
        n = mdg.x.shape[0]

        # sampled structural targets for scalable adjacency reconstruction
        for _ in range(self.cfg.dominant_epochs):
            opt.zero_grad()
            z, x_hat = net(mdg.x, ei)
            attr_err = F.mse_loss(x_hat, mdg.x)
            # structure: inner-product adjacency on a sampled edge/non-edge set
            idx = torch.randint(0, ei.shape[1], (min(2048, ei.shape[1]),),
                                device=device)
            pos = (z[ei[0, idx]] * z[ei[1, idx]]).sum(dim=1)
            negs = torch.randint(0, n, (pos.shape[0], 2), device=device)
            neg = (z[negs[:, 0]] * z[negs[:, 1]]).sum(dim=1)
            struct_err = F.binary_cross_entropy_with_logits(
                torch.cat([pos, neg]),
                torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
            loss = self.cfg.dominant_alpha * struct_err + \
                (1 - self.cfg.dominant_alpha) * attr_err
            loss.backward()
            opt.step()

        with torch.no_grad():
            z, x_hat = net(mdg.x, ei)
            attr_score = (mdg.x - x_hat).pow(2).mean(dim=1)
            # structural score: how poorly each node's edges are reconstructed
            deg = torch.zeros(n, device=device).index_add_(
                0, ei[0], torch.ones(ei.shape[1], device=device))
            logits = (z[ei[0]] * z[ei[1]]).sum(dim=1)
            struct_node = torch.zeros(n, device=device).index_add_(
                0, ei[0], F.softplus(-logits))
            struct_node = struct_node / deg.clamp_min(1)
            score = self.cfg.dominant_alpha * struct_node + \
                (1 - self.cfg.dominant_alpha) * attr_score

        thr = torch.quantile(score, 0.95)       # flag top 5% as OOD
        flagged = score > thr
        keep_node = ~flagged
        keep_edge = keep_node[ei[0]] & keep_node[ei[1]]
        g = copy.copy(mdg)
        g.edge_index = ei[:, keep_edge]
        return g, DefenseReport(name="dominant",
                                removed_edges=int((~keep_edge).sum()),
                                flagged_nodes=int(flagged.sum()),
                                extra={"alpha": self.cfg.dominant_alpha})
