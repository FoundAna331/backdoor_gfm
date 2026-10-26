"""Memory-safe spectral path: sparse Laplacian, virtual-node exclusion, capping,
truncated spectrum. (Needs torch + torch_geometric; skipped otherwise.)"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.attack import spectral as S


def _cfg(**kw):
    base = dict(k_hop=2, high_freq_band_m=0, high_freq_fraction=0.5,
                eig_max_nodes=12000, eig_truncated_k=32, max_subgraph_nodes=0,
                eig_method="auto", eig_tol=1e-4, exclude_virtual_nodes=True)
    base.update(kw)
    return SimpleNamespace(**base)


def _ring_with_chords(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    i = torch.arange(n)
    ring = torch.stack([i, (i + 1) % n])
    ch = torch.randint(0, n, (2, 2 * n), generator=g)
    ei = torch.cat([ring, ch], dim=1)
    return torch.cat([ei, ei.flip(0)], dim=1)          # undirected (with duplicates)


def test_sparse_laplacian_matches_dense_reference():
    n = 30
    ei = _ring_with_chords(n)
    ei = torch.cat([ei, torch.tensor([[3, 3], [3, 3]])], dim=1)   # a self-loop
    A = torch.zeros(n, n)
    for a, b in ei.t().tolist():
        if a != b:
            A[a, b] = A[b, a] = 1.0
    ref = torch.diag(A.sum(1)) - A
    got = S.sparse_laplacian(ei, n).to_dense()
    assert torch.allclose(got, ref)
    assert torch.allclose(got.sum(1), torch.zeros(n), atol=1e-6)    # rows sum to 0


def test_virtual_coordinator_no_longer_swallows_the_graph():
    n = 500
    real = _ring_with_chords(n)
    coord = n                                                    # node id n
    c_edges = torch.stack([torch.cat([torch.arange(n), torch.full((n,), coord)]),
                           torch.cat([torch.full((n,), coord), torch.arange(n)])])
    ei = torch.cat([real, c_edges], dim=1)
    mask = torch.zeros(n + 1, dtype=torch.bool); mask[coord] = True

    sub_full, _, _ = S.k_hop_subgraph(0, ei, n + 1, 2)
    assert sub_full.numel() == n + 1                              # the old failure mode
    sub_real, _, _ = S.k_hop_subgraph(0, S.restrict_edges_to_real_nodes(ei, mask),
                                      n + 1, 2)
    assert coord not in sub_real.tolist()
    assert sub_real.numel() < n // 2


def test_cap_keeps_centre_and_one_hop_first():
    n = 400
    ei = _ring_with_chords(n)
    subset, sub_ei, mapping = S.k_hop_subgraph(5, ei, n, 2)
    center = int(mapping.view(-1)[0])
    cap = 40
    sub2, ei2 = S._cap_subgraph(subset, sub_ei, center, cap, seed=1)
    assert sub2.numel() == cap
    assert 5 in sub2.tolist()
    assert torch.equal(sub2, sub2.sort().values)                 # sorted invariant
    assert int(ei2.max()) < cap
    one_hop = set(ei[1][ei[0] == 5].tolist())
    kept = set(sub2.tolist())
    assert len(one_hop) <= cap - 1 and one_hop <= kept


def test_positions_in_subgraph():
    spec = S.SpectrumCache(eigvecs=torch.eye(4), eigvals=torch.zeros(4),
                           node_index=torch.tensor([3, 8, 11, 20]), num_high=1, n=4)
    pos, valid = S.positions_in_subgraph(spec, [8, 9, 20, 2, 99])
    assert valid.tolist() == [True, False, True, False, False]
    assert pos[valid].tolist() == [1, 3]


def test_truncated_matches_dense_and_upper_bounds_rspec():
    n = 300
    ei = _ring_with_chords(n)
    dense = S.compute_spectrum(0, ei, n, _cfg(), verbose=False)
    trunc = S.compute_spectrum(0, ei, n, _cfg(eig_max_nodes=10, eig_method="scipy"),
                               verbose=False)
    assert dense.L_sp is None and trunc.L_sp is not None
    assert dense.n == trunc.n
    r = trunc.r
    assert torch.allclose(dense.eigvals[:r], trunc.eigvals, atol=1e-3)

    torch.manual_seed(0)
    sig = torch.randn(dense.n, 8)
    rd = S.r_spec_matrix(sig, dense)
    rt = S.r_spec_matrix(sig, trunc)
    assert rt >= rd - 1e-4                                        # conservative bound
    lp = S.lowpass_project_matrix(sig, trunc)
    assert S.r_spec_matrix(lp, trunc) < 1e-4
    lhs, rhs = S.parseval_lhs_rhs(sig[:, 0], dense)
    assert torch.allclose(lhs, rhs, rtol=1e-3, atol=1e-3)
