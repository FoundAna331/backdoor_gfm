
from __future__ import annotations

from .base import BaseGFM
from .backbones import build_backbone


def build_gcope(cfg_model, in_dim: int, num_domains: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class GCOPE(nn.Module, BaseGFM):
        def __init__(self):
            super().__init__()
            self.embed_dim = cfg_model.hidden_dim
            self.recon_weight = cfg_model.recon_weight
            self.backbone = build_backbone(
                cfg_model.backbone, in_dim, cfg_model.hidden_dim,
                cfg_model.num_layers, cfg_model.dropout)
            # learnable coordinator feature table (one row per domain-coordinator)
            self.coord_table = nn.Parameter(
                torch.zeros(num_domains * cfg_model.num_coordinators_per_domain,
                            in_dim))
            nn.init.xavier_uniform_(self.coord_table)
            # lightweight feature decoder for the reconstruction term
            self.decoder = nn.Linear(cfg_model.hidden_dim, in_dim)

        def _inject_coords(self, x, is_coordinator):
            if is_coordinator is None or is_coordinator.sum() == 0:
                return x
            x = x.clone()
            idx = is_coordinator.nonzero(as_tuple=True)[0]
            k = min(idx.numel(), self.coord_table.shape[0])
            x[idx[:k]] = self.coord_table[:k]
            return x

        def encode(self, x, edge_index, is_coordinator=None, **kw):
            x = self._inject_coords(x, is_coordinator)
            return self.backbone(x, edge_index)

        def pretrain_loss(self, mdg):
            h = self.encode(mdg.x, mdg.edge_index, mdg.is_coordinator)
            # --- contrastive (feature-dropout augmentation; InfoNCE-lite) ---
            x2 = F.dropout(mdg.x, p=0.3, training=True)
            h2 = self.encode(x2, mdg.edge_index, mdg.is_coordinator)
            z1 = F.normalize(h, dim=1)
            z2 = F.normalize(h2, dim=1)
            # sample a subset for the NxN logits to keep it tractable
            n = z1.shape[0]
            m = min(512, n)
            perm = torch.randperm(n, device=z1.device)[:m]
            a, b = z1[perm], z2[perm]
            logits = a @ b.t() / 0.2
            labels = torch.arange(m, device=logits.device)
            contrastive = F.cross_entropy(logits, labels)
            # --- reconstruction ---
            recon = F.mse_loss(self.decoder(h), mdg.x)
            return contrastive + self.recon_weight * recon

    return GCOPE()
