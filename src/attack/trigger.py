
from __future__ import annotations

# (dataclass import not needed here)


def build_trigger(feat_dim, trigger_size, device):
    import torch
    import torch.nn as nn

    class Trigger(nn.Module):
        def __init__(self):
            super().__init__()
            # one feature perturbation vector per trigger node
            self.delta = nn.Parameter(torch.zeros(trigger_size, feat_dim,
                                                  device=device))
            nn.init.normal_(self.delta, std=0.01)

        def clamp_norm(self, max_norm):
            with torch.no_grad():
                norm = self.delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
                factor = (norm.clamp_max(max_norm) / norm)
                self.delta.mul_(factor)

    return Trigger()


def greedy_edge_selection(target_node, edge_index, num_nodes, budget,
                          influence_scores=None, exclude_mask=None):
    """Influence-ranked greedy host-edge selection under budget tau (Section 5.4).

    If influence_scores (per-node) is given, connect the trigger to the highest
    scoring neighbours of the target; else connect to the target and random
    1-hop neighbours. Returns an [2, <=budget] edge tensor (trigger-node ids are
    assigned by the caller via inject_trigger)."""
    import torch
    from torch_geometric.utils import k_hop_subgraph as pyg_khop
    subset, _, _, _ = pyg_khop(int(target_node), 1, edge_index,
                               relabel_nodes=False, num_nodes=num_nodes)
    cand = subset[subset != target_node]
    if exclude_mask is not None and cand.numel() > 0:
        # never attach a trigger to a virtual node (e.g. a GCOPE coordinator)
        cand = cand[~exclude_mask[cand]]
    if cand.numel() == 0:
        cand = torch.tensor([int(target_node)], device=edge_index.device)
    if influence_scores is not None and influence_scores.numel() >= num_nodes:
        order = torch.argsort(influence_scores[cand], descending=True)
        cand = cand[order]
    hosts = cand[:max(1, budget)]
    return hosts


def inject_trigger(mdg, target_node, trigger, hosts, device):
    """Append trigger nodes (with features = aligned base + delta) and the
    trigger<->host edges to a *copy* of the multi-domain graph. Returns the new
    graph plus the indices of the injected trigger nodes.

    The base feature for each trigger node is the target node's feature, so the
    learnable `delta` is a perturbation around a realistic feature (keeps ||delta||
    meaningfully small, as the stealth term intends)."""
    import torch
    import copy

    g = copy.copy(mdg)                       # shallow copy; tensors replaced below
    n = mdg.x.shape[0]
    t = trigger.delta.shape[0]
    base = mdg.x[target_node:target_node + 1].expand(t, -1)
    trig_feat = base + trigger.delta
    g.x = torch.cat([mdg.x, trig_feat], dim=0)

    trig_ids = torch.arange(n, n + t, device=device)
    # connect each trigger node to the target and to the selected hosts (both dirs)
    src, dst = [], []
    for ti in trig_ids.tolist():
        targets = torch.cat([torch.tensor([target_node], device=device), hosts])
        for h in targets.tolist():
            src += [ti, h]
            dst += [h, ti]
    new_edges = torch.tensor([src, dst], device=device, dtype=mdg.edge_index.dtype)
    g.edge_index = torch.cat([mdg.edge_index, new_edges], dim=1)

    # extend auxiliary per-node tensors
    if getattr(mdg, "y", None) is not None:
        g.y = torch.cat([mdg.y, torch.full((t,), -1, dtype=mdg.y.dtype,
                                           device=device)])
    if getattr(mdg, "domain_id", None) is not None:
        dom = mdg.domain_id[target_node]
        g.domain_id = torch.cat([mdg.domain_id,
                                 torch.full((t,), int(dom),
                                            dtype=mdg.domain_id.dtype,
                                            device=device)])
    if getattr(mdg, "is_coordinator", None) is not None:
        g.is_coordinator = torch.cat([mdg.is_coordinator,
                                      torch.zeros(t, dtype=torch.bool,
                                                  device=device)])
    return g, trig_ids


# --------------------------------------------------------------------------- #
#  Batched injection (evaluation / fine-tuning)
# --------------------------------------------------------------------------- #
class NeighborIndex:
    """CPU CSR neighbour lookup so host selection for hundreds of nodes does not
    need a k-hop pass over the whole edge list per node."""

    def __init__(self, edge_index, num_nodes, exclude_mask=None):
        import torch
        ei = edge_index.detach().cpu()
        if exclude_mask is not None:
            ex = exclude_mask.detach().cpu()
            ei = ei[:, ~ex[ei[1]]]            # drop edges that point at virtual nodes
        order = torch.argsort(ei[0], stable=True)
        self.col = ei[1][order]
        counts = torch.bincount(ei[0], minlength=num_nodes)
        self.rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        self.rowptr[1:] = torch.cumsum(counts, 0)

    def hosts(self, node, budget):
        """Same rule as greedy_edge_selection(..., influence_scores=None): the first
        `budget` distinct 1-hop neighbours (ascending id), or the node itself."""
        import torch
        nb = self.col[int(self.rowptr[node]):int(self.rowptr[node + 1])]
        nb = torch.unique(nb[nb != node])
        if nb.numel() == 0:
            return torch.tensor([int(node)], dtype=torch.long)
        return nb[:max(1, budget)]


def inject_triggers_batch(g, nodes, deltas, max_edges, nbr, device):
    """Attach one trigger instance to EACH node in `nodes` at once.

    nodes  : LongTensor/list [B]
    deltas : [B, t, F] per-instance trigger feature perturbations (or [t, F],
             broadcast). Trigger node features = target feature + delta.
    Returns (graph copy, trig_ids [B, t]). Instance b owns the t appended nodes
    n + b*t ... n + b*t + t - 1, wired (both directions) to node b and its hosts.
    Instances only interact if their targets lie within the encoder's receptive
    field of each other, so keep batches sparse relative to the graph (the caller
    controls the batch size)."""
    import copy
    import torch
    nodes_t = torch.as_tensor(nodes, dtype=torch.long).view(-1)
    B = nodes_t.numel()
    if deltas.dim() == 2:
        deltas = deltas.unsqueeze(0).expand(B, -1, -1)
    t, F = deltas.shape[1], deltas.shape[2]
    n = g.x.shape[0]

    nodes_dev = nodes_t.to(device)
    base = g.x[nodes_dev].unsqueeze(1)                               # [B,1,F]
    trig_feat = (base + deltas.to(device)).reshape(B * t, F)

    src, dst = [], []
    trig_ids_cpu = n + torch.arange(B * t, dtype=torch.long).view(B, t)
    for b in range(B):
        v = int(nodes_t[b])
        attach = torch.cat([torch.tensor([v], dtype=torch.long),
                            nbr.hosts(v, max_edges)])                 # [h+1]
        ids = trig_ids_cpu[b]                                         # [t]
        s_ = ids.repeat_interleave(attach.numel())
        d_ = attach.repeat(t)
        src += [s_, d_]
        dst += [d_, s_]
    new_edges = torch.stack([torch.cat(src), torch.cat(dst)]).to(
        device=device, dtype=g.edge_index.dtype)

    out = copy.copy(g)
    out.x = torch.cat([g.x, trig_feat.to(g.x.dtype)], dim=0)
    out.edge_index = torch.cat([g.edge_index, new_edges], dim=1)
    if getattr(g, "y", None) is not None:
        out.y = torch.cat([g.y, torch.full((B * t,), -1, dtype=g.y.dtype,
                                           device=device)])
    if getattr(g, "domain_id", None) is not None:
        out.domain_id = torch.cat([g.domain_id,
                                   g.domain_id[nodes_dev].repeat_interleave(t)])
    if getattr(g, "is_coordinator", None) is not None:
        out.is_coordinator = torch.cat([g.is_coordinator,
                                        torch.zeros(B * t, dtype=torch.bool,
                                                    device=device)])
    return out, trig_ids_cpu.to(device)
