
from __future__ import annotations

from .base import BaseGFM


def build_gft(cfg_model, in_dim: int, num_domains: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from .backbones import build_backbone

    class VectorQuantizer(nn.Module):
        def __init__(self, num_codes, dim, commitment):
            super().__init__()
            self.codebook = nn.Embedding(num_codes, dim)
            self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
            self.commitment = commitment

        def forward(self, z):
            # z: [N, dim]; find nearest code
            d = (z.pow(2).sum(1, keepdim=True)
                 - 2 * z @ self.codebook.weight.t()
                 + self.codebook.weight.pow(2).sum(1))
            idx = d.argmin(dim=1)
            zq = self.codebook(idx)
            codebook_loss = F.mse_loss(zq, z.detach())
            commit_loss = F.mse_loss(z, zq.detach())
            vq_loss = codebook_loss + self.commitment * commit_loss
            # straight-through estimator
            zq_st = z + (zq - z).detach()
            return zq_st, vq_loss, idx

    class GFT(nn.Module, BaseGFM):
        def __init__(self):
            super().__init__()
            self.embed_dim = cfg_model.hidden_dim
            self.backbone = build_backbone(
                cfg_model.backbone, in_dim, cfg_model.hidden_dim,
                cfg_model.num_layers, cfg_model.dropout)
            self.vq = VectorQuantizer(cfg_model.vocab_size, cfg_model.hidden_dim,
                                      cfg_model.vq_commitment)
            self.decoder = nn.Linear(cfg_model.hidden_dim, in_dim)

        def encode(self, x, edge_index, quantize: bool = False, **kw):
            tree_emb = self.backbone(x, edge_index)   # computation-tree embedding
            if quantize:
                zq, _, _ = self.vq(tree_emb)
                return zq
            return tree_emb

        def pretrain_loss(self, mdg):
            tree_emb = self.backbone(mdg.x, mdg.edge_index)
            zq, vq_loss, _ = self.vq(tree_emb)
            recon = F.mse_loss(self.decoder(zq), mdg.x)   # tree reconstruction
            return recon + vq_loss

    return GFT()
