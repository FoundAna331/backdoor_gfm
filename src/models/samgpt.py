"""
SAMGPT victim encoder (TheWebConf'25).

Distinctive mechanism: a set of learnable *structure tokens* per domain, inserted
into *each layer* of the graph encoder, modifying structure-based aggregation to
align structural distributions across domains. Feature distributions are aligned
first (done in data/feature_align.py); structure alignment is these tokens.

We realise "structure tokens modify each layer's aggregation" as a per-domain,
per-layer additive bias applied to node representations at every layer, routed by
each node's `domain_id`. Pretraining uses the same contrastive template as the
other text-free GFMs.

# >>> OFFICIAL-REPO HOOK: replace the additive-token approximation with SAMGPT's
#     exact token-conditioned aggregation and its dual-prompt downstream head.
"""
from __future__ import annotations

from .base import BaseGFM


def build_samgpt(cfg_model, in_dim: int, num_domains: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from .backbones import build_backbone

    class SAMGPT(nn.Module, BaseGFM):
        def __init__(self):
            super().__init__()
            self.embed_dim = cfg_model.hidden_dim
            self.num_layers = cfg_model.num_layers
            self.num_tokens = cfg_model.num_structure_tokens
            self.backbone = build_backbone(
                cfg_model.backbone, in_dim, cfg_model.hidden_dim,
                cfg_model.num_layers, cfg_model.dropout)
            # structure tokens: [num_domains, num_layers, num_tokens, hidden]
            self.structure_tokens = nn.Parameter(
                torch.zeros(num_domains, cfg_model.num_layers,
                            self.num_tokens, cfg_model.hidden_dim))
            nn.init.normal_(self.structure_tokens, std=0.02)
            self.decoder = nn.Linear(cfg_model.hidden_dim, in_dim)

        def _layer_tokens(self, domain_id, num_nodes, device):
            """Per-node, per-layer additive token = mean of its domain's tokens."""
            if domain_id is None:
                return None
            # [num_domains, num_layers, hidden]
            tok = self.structure_tokens.mean(dim=2)
            layer_tokens = []
            for layer in range(self.num_layers):
                # gather per node: [N, hidden]
                per_node = tok[domain_id.clamp(min=0), layer]
                per_node = per_node.to(device)
                layer_tokens.append(per_node)
            return layer_tokens

        def encode(self, x, edge_index, domain_id=None, **kw):
            layer_tokens = self._layer_tokens(domain_id, x.shape[0], x.device)
            return self.backbone(x, edge_index, layer_tokens=layer_tokens)

        def pretrain_loss(self, mdg):
            h = self.encode(mdg.x, mdg.edge_index, mdg.domain_id)
            x2 = F.dropout(mdg.x, p=0.3, training=True)
            h2 = self.encode(x2, mdg.edge_index, mdg.domain_id)
            z1 = F.normalize(h, dim=1)
            z2 = F.normalize(h2, dim=1)
            n = z1.shape[0]
            m = min(512, n)
            perm = torch.randperm(n, device=z1.device)[:m]
            logits = z1[perm] @ z2[perm].t() / 0.2
            labels = torch.arange(m, device=logits.device)
            contrastive = F.cross_entropy(logits, labels)
            recon = F.mse_loss(self.decoder(h), mdg.x)
            return contrastive + 0.2 * recon

    return SAMGPT()
