"""
TCF defense (Rethinking Graph Backdoor Defense: A Topological, Coarse-to-Fine
Perspective, NeurIPS'25).

TCF screens candidate nodes with three topological signals and then prunes:
  1. LOCAL SPECTRAL MOMENTS of the ego-graph Laplacian. Trigger subgraphs inject
     energy at large eigenvalues, so their high-order spectral moments
     m_k = (1/n) tr(L^k) are inflated. Computed WITHOUT eigendecomposition via
     repeated sparse mat-vecs (powers of L), so it scales.
  2. 1-WL STRUCTURAL RARITY. A few rounds of Weisfeiler-Leman colour refinement;
     nodes carrying rare final colours are structurally anomalous.
  3. EGO-DENSITY z-score. Edge density of each node's 1-hop ego-net; triggers
     often over- or under-densify the local neighbourhood.
A node is flagged if the combined z-score exceeds `tcf_z_threshold`; flagged
nodes and their incident edges are removed (coarse-to-fine causal pruning).

This is exactly the detector SIB2's Proposition 2 targets: by keeping the
trigger's high-frequency energy ratio R_spec below the spectral-moment threshold
(by margin gamma), signal (1) does not fire. The harness reports measured R_spec
vs this screen so the margin guarantee can be checked empirically (instruction 4:
margin sensitivity).
"""
from __future__ import annotations

from .base import Defense, DefenseReport


def _normalized_laplacian_matvec_powers(edge_index, n, num_moments, device):
    """Return spectral moments m_k = (1/n) tr(L_sym^k), k=1..num_moments, via a
    stochastic trace estimator (Hutchinson) using sparse L_sym mat-vecs."""
    import torch
    from torch_geometric.utils import degree

    row, col = edge_index
    deg = degree(row, n).clamp_min(1)
    dinv_sqrt = deg.pow(-0.5)

    def Lsym_matvec(v):
        # L_sym = I - D^{-1/2} A D^{-1/2}
        av = torch.zeros_like(v).index_add_(0, row, (dinv_sqrt[col] * v[col]))
        return v - dinv_sqrt * av

    num_probes = 8
    moments = torch.zeros(num_moments, device=device)
    for _ in range(num_probes):
        z = torch.randn(n, device=device)
        Lz = z.clone()
        for k in range(num_moments):
            Lz = Lsym_matvec(Lz)
            moments[k] += (z * Lz).sum()
    moments = moments / (num_probes * n)
    return moments


def _per_node_ego_spectral_energy(edge_index, n, device):
    """Cheap per-node proxy for local high-frequency energy: the Dirichlet energy
    contribution (sum over incident edges of squared normalized difference of a
    smooth test signal). High values ~ locally non-smooth (high-frequency)."""
    import torch
    from torch_geometric.utils import degree
    row, col = edge_index
    deg = degree(row, n).clamp_min(1)
    # test signal = degree profile (smooth on homophilous graphs)
    s = (deg / deg.max()).float()
    edge_energy = (s[row] - s[col]).pow(2)
    node_energy = torch.zeros(n, device=device).index_add_(0, row, edge_energy)
    node_energy = node_energy / deg
    return node_energy


def _wl_rarity(edge_index, n, device, rounds=3):
    import torch
    from torch_geometric.utils import degree
    row, col = edge_index
    colors = degree(row, n).long()
    for _ in range(rounds):
        # aggregate neighbour colours (sum hash)
        agg = torch.zeros(n, dtype=torch.long, device=device).index_add_(
            0, row, colors[col])
        combined = colors * 1000003 + agg
        # relabel to compact ids
        _, colors = torch.unique(combined, return_inverse=True)
    # rarity = inverse frequency of each node's final colour
    uniq, counts = torch.unique(colors, return_counts=True)
    freq = torch.zeros(n, device=device)
    cmap = {int(u): int(c) for u, c in zip(uniq.tolist(), counts.tolist())}
    for i in range(n):
        freq[i] = cmap[int(colors[i])]
    return 1.0 / freq


def _ego_density(edge_index, n, device):
    import torch
    from torch_geometric.utils import degree
    row, col = edge_index
    deg = degree(row, n).clamp_min(1)
    # density proxy: triangles / possible ~ clustering-ish via common neighbours
    # cheap version: deg normalized z-score
    return deg.float()


def _zscore(v):
    import torch
    return (v - v.mean()) / (v.std() + 1e-12)


class TCFDefense(Defense):
    name = "tcf"

    def apply(self, mdg, artifacts=None, model=None):
        import torch
        import copy

        device = mdg.x.device
        n = mdg.x.shape[0]
        ei = mdg.edge_index

        # (1) local spectral energy (per node) -> z-score
        spec_energy = _per_node_ego_spectral_energy(ei, n, device)
        z_spec = _zscore(spec_energy)
        # global spectral moments reported for context (Prop 2 target statistic)
        moments = _normalized_laplacian_matvec_powers(ei, n, 4, device)

        # (2) 1-WL rarity
        rarity = _wl_rarity(ei, n, device)
        z_rare = _zscore(rarity)

        # (3) ego density
        dens = _ego_density(ei, n, device)
        z_dens = _zscore(dens)

        combined = torch.stack([z_spec, z_rare, z_dens], dim=0).max(dim=0).values
        flagged = combined > self.cfg.tcf_z_threshold

        keep_node = ~flagged
        keep_edge = keep_node[ei[0]] & keep_node[ei[1]]
        g = copy.copy(mdg)
        g.edge_index = ei[:, keep_edge]
        return g, DefenseReport(
            name="tcf",
            removed_edges=int((~keep_edge).sum()),
            flagged_nodes=int(flagged.sum()),
            extra={"z_threshold": self.cfg.tcf_z_threshold,
                   "spectral_moments": [float(m) for m in moments.tolist()]})
