"""
GNN backbones shared by all three GFMs (GCOPE lists gcn/gat/fagcn/bwgnn as
backbone options; we implement gcn/gat/fagcn).

Each backbone's `forward(x, edge_index, layer_tokens=None)` accepts an optional
list of per-layer additive bias vectors. SAMGPT injects its *structure tokens*
through exactly this hook, so the same backbone serves every model.
"""
from __future__ import annotations


def build_backbone(kind: str, in_dim: int, hidden_dim: int, num_layers: int,
                   dropout: float):
    import torch.nn as nn
    from torch_geometric.nn import GCNConv, GATConv

    kind = kind.lower()

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.num_layers = num_layers
            self.dropout = dropout
            self.convs = nn.ModuleList()
            dims = [in_dim] + [hidden_dim] * num_layers
            for i in range(num_layers):
                if kind == "gcn":
                    self.convs.append(GCNConv(dims[i], dims[i + 1]))
                elif kind == "gat":
                    # single-head to keep output dim = hidden_dim
                    self.convs.append(GATConv(dims[i], dims[i + 1], heads=1))
                elif kind == "fagcn":
                    # FAGCN = frequency-adaptive; approximate with GCN + a learnable
                    # low/high-freq mix gate (keeps the backbone light + dependency-free)
                    self.convs.append(GCNConv(dims[i], dims[i + 1]))
                else:
                    raise ValueError(f"Unknown backbone '{kind}'")
            if kind == "fagcn":
                self.freq_gate = nn.Parameter(__import__("torch").zeros(num_layers))
            self.act = nn.ReLU()
            self.drop = nn.Dropout(dropout)

        def forward(self, x, edge_index, layer_tokens=None):
            import torch
            h = x
            for i, conv in enumerate(self.convs):
                h_in = h
                h = conv(h, edge_index)
                if self.kind == "fagcn":
                    # mix aggregated (low-pass) signal with residual (high-pass-ish)
                    g = torch.sigmoid(self.freq_gate[i])
                    if h_in.shape == h.shape:
                        h = g * h + (1.0 - g) * h_in
                # SAMGPT structure-token injection hook
                if layer_tokens is not None and layer_tokens[i] is not None:
                    h = h + layer_tokens[i]
                if i < self.num_layers - 1:
                    h = self.drop(self.act(h))
            return h

    return _Backbone()
