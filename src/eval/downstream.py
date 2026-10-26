"""
Downstream evaluation head.

The attacker has no downstream labels (threat model, Section 1), but *evaluation*
simulates a victim who fine-tunes/probes the (poisoned) encoder on a downstream
task. We train a simple linear probe on frozen encoder embeddings of the clean
test graph, then measure:
  * Clean ACC  on clean test nodes
  * ASR        on trigger-injected test nodes (prediction == y*)
and, for persistence (Prop 3), the same after a few steps of encoder fine-tuning.
"""
from __future__ import annotations


def train_linear_probe(model, data, device, epochs=200, lr=1e-2, weight_decay=5e-4):
    """Train a linear classifier on frozen embeddings of a single clean graph."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    with torch.no_grad():
        emb = model.encode(data.x, data.edge_index,
                           is_coordinator=getattr(data, "is_coordinator", None),
                           domain_id=getattr(data, "domain_id", None)).detach()
    y = data.y
    mask = y >= 0
    num_classes = int(y[mask].max().item()) + 1
    clf = nn.Linear(emb.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)

    idx = mask.nonzero(as_tuple=True)[0]
    perm = idx[torch.randperm(idx.numel(), device=device)]
    n_train = int(0.6 * perm.numel())
    train_idx = perm[:n_train]

    for _ in range(epochs):
        clf.train(); opt.zero_grad()
        out = clf(emb[train_idx])
        loss = F.cross_entropy(out, y[train_idx])
        loss.backward(); opt.step()
    return clf, num_classes


def predict(model, clf, data, device):
    import torch
    with torch.no_grad():
        emb = model.encode(data.x, data.edge_index,
                           is_coordinator=getattr(data, "is_coordinator", None),
                           domain_id=getattr(data, "domain_id", None))
        return clf(emb).argmax(dim=1)
