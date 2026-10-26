"""
Step 2 — Spectral trigger construction / low-frequency anchoring (Section 3).

  * Laplacian eigendecomposition L = U Lambda U^T of the target's k-hop
    neighbourhood (Section 3.2), built SPARSELY on the real (non-virtual) nodes.
    Dense GPU eigh up to `eig_max_nodes`; the neighbourhood is capped at
    `max_subgraph_nodes`; truncated sparse solvers are an opt-in fallback.
    All shown with a PROGRESS BAR (instruction 6).
  * R_spec(delta) = [ sum_{l>n-m} c_l^2 lambda_l ] / [ sum_l c_l^2 lambda_l ],
    c_l = u_l^T delta  (Section 3.2). The differentiable spectral-stealth penalty.
  * low-pass projection onto span(u_1..u_{n-m}) (hard constraint, Section 3.3):
    project delta after each gradient step (spectral PGD).
  * Parseval identity delta^T L delta == sum_l c_l^2 lambda_l (verified in tests).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpectrumCache:
    eigvecs: "object"      # U  [n, r]
    eigvals: "object"      # lambda [r]
    node_index: "object"   # global node ids of the subgraph (order matches U rows)
    num_high: int          # m (size of high-frequency band)
    n: int                 # subgraph size
    L_sp: "object" = None  # sparse Laplacian [n, n] (None => eigvecs span the full spectrum)
    method: str = "dense"  # which solver produced the spectrum (for logging)

    @property
    def r(self) -> int:
        return int(self.eigvecs.shape[1])

    @property
    def keep(self) -> int:
        """# of low-frequency modes that survive the hard low-pass projection:
        min(r, n - m). When only r < n eigenpairs are available this is the
        (stricter) truncated low-pass band."""
        return max(0, min(self.r, self.n - self.num_high))


def k_hop_subgraph(node_idx, edge_index, num_nodes, k):
    import torch
    from torch_geometric.utils import k_hop_subgraph as pyg_khop
    subset, sub_ei, mapping, _ = pyg_khop(
        int(node_idx), k, edge_index, relabel_nodes=True,
        num_nodes=num_nodes, flow="source_to_target")
    # NOTE: `subset` is returned SORTED by global id; positions_in_subgraph()
    # relies on this invariant.
    return subset, sub_ei, mapping


def restrict_edges_to_real_nodes(edge_index, virtual_mask):
    """Drop every edge that touches a virtual node (e.g. a GCOPE coordinator).

    Why this matters: a GCOPE coordinator is wired to *every* real node of its
    domain, so in the full graph the 2-hop neighbourhood of ANY node is the whole
    domain (n = |V|+1, e.g. 169,344 for ogbn-arxiv). That is what made the old code
    ask for a dense 169k x 169k matrix (~107 GiB). The trigger's local spectrum
    should describe the *real* topology around the target, so the spectral step
    works on the graph without virtual nodes.
    """
    if virtual_mask is None or int(virtual_mask.sum()) == 0:
        return edge_index
    keep = ~(virtual_mask[edge_index[0]] | virtual_mask[edge_index[1]])
    return edge_index[:, keep]


def _cap_subgraph(subset, sub_ei, center_pos, max_nodes, seed):
    """Bound the neighbourhood size. Hub nodes can still have huge 2-hop balls even
    without a coordinator. Keep the centre, then as many 1-hop neighbours as fit
    (random if there are too many), then fill with randomly sampled 2-hop nodes.
    Deterministic given `seed`. Returns (sorted subset, relabelled edges)."""
    import torch
    from torch_geometric.utils import subgraph
    n = subset.numel()
    if max_nodes is None or max_nodes <= 0 or n <= max_nodes:
        return subset, sub_ei
    dev = subset.device
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    pri = torch.rand(n, generator=g).to(dev)                 # random tie-break in [0,1)
    nb = sub_ei[1][sub_ei[0] == center_pos]
    one_hop = torch.zeros(n, dtype=torch.bool, device=dev)
    one_hop[nb] = True
    pri = pri + 2.0 * one_hop.float()                        # 1-hop beats 2-hop
    pri[center_pos] = 10.0                                   # centre always kept
    keep_pos = torch.topk(pri, max_nodes).indices.sort().values
    new_ei, _ = subgraph(keep_pos, sub_ei, relabel_nodes=True, num_nodes=n)
    return subset[keep_pos], new_ei


def sparse_laplacian(sub_ei, n, device=None):
    """Combinatorial Laplacian L = D - A as a coalesced *sparse* COO tensor.

    O(E) memory instead of O(n^2). A is the symmetrised 0/1 adjacency with
    self-loops and duplicate edges removed (identical to what the previous dense
    construction produced)."""
    import torch
    device = device if device is not None else sub_ei.device
    r, c = sub_ei[0].to(device), sub_ei[1].to(device)
    nz = r != c
    r, c = r[nz], c[nz]
    lin = torch.cat([r * n + c, c * n + r]).unique()          # symmetrise + dedupe
    rows, cols = lin // n, lin % n
    deg = torch.bincount(rows, minlength=n).to(torch.float32)
    diag = torch.arange(n, device=device)
    idx = torch.stack([torch.cat([rows, diag]), torch.cat([cols, diag])])
    val = torch.cat([-torch.ones(rows.numel(), device=device), deg])
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()


def laplacian_dense(sub_ei, n):
    """Dense Laplacian (small graphs / tests only). Built from the sparse one."""
    return sparse_laplacian(sub_ei, n).to_dense()


def _derive_m(n, cfg_spec):
    if cfg_spec.high_freq_band_m and cfg_spec.high_freq_band_m > 0:
        return min(cfg_spec.high_freq_band_m, n - 1)
    return max(1, int(round(cfg_spec.high_freq_fraction * n)))


# --------------------------------------------------------------------------- #
#  Eigen-solvers
# --------------------------------------------------------------------------- #
_SHIFT = -1e-3     # shift-invert point (L is singular; L + 1e-3 I is SPD)


def smallest_eigs_scipy(L_csc, r, tol=0.0):
    """r smallest eigenpairs of a sparse symmetric PSD matrix via shift-invert
    Lanczos (sparse LU of L - sigma*I). Pure NumPy/SciPy -> unit-testable without
    torch. Returns ascending (evals [r], evecs [n, r]) as float64 arrays."""
    import numpy as np
    from scipy.sparse.linalg import eigsh
    w, v = eigsh(L_csc, k=r, sigma=_SHIFT, which="LM", tol=tol)
    order = np.argsort(w)
    return w[order], v[:, order]


def _eigs_scipy(L_sp, r, tol):
    import numpy as np
    import torch
    import scipy.sparse as sps
    n = L_sp.shape[0]
    idx = L_sp.indices().cpu().numpy()
    val = L_sp.values().cpu().numpy().astype(np.float64)
    L = sps.csc_matrix((val, (idx[0], idx[1])), shape=(n, n))
    w, v = smallest_eigs_scipy(L, r, tol=0.0 if tol is None else tol)
    dev = L_sp.device
    return (torch.from_numpy(w).to(dev, torch.float32),
            torch.from_numpy(np.ascontiguousarray(v)).to(dev, torch.float32))


def _eigs_lobpcg(L_sp, r, tol, niter=300):
    """GPU fallback: sparse LOBPCG for the r smallest eigenpairs."""
    import torch
    n = L_sp.shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    X0 = torch.randn(n, r, generator=g).to(L_sp.device, torch.float32)
    evals, evecs = torch.lobpcg(L_sp, k=r, X=X0, largest=False,
                                niter=niter, tol=tol if tol else None)
    order = torch.argsort(evals)
    return evals[order], evecs[:, order]


def compute_spectrum(node_idx, edge_index, num_nodes, cfg_spec, verbose=True,
                     seed=0):
    """Eigendecompose the k-hop neighbourhood Laplacian (progress bar shown).

    `edge_index` should already be restricted to REAL nodes
    (see restrict_edges_to_real_nodes). Memory is O(E_sub + n*r); no n x n dense
    matrix is ever built unless n <= cfg_spec.eig_max_nodes.

      * n > max_subgraph_nodes        -> neighbourhood is first capped (_cap_subgraph),
                                         so with the defaults n <= 12000 always
      * n <= eig_max_nodes            -> dense eigh ON THE GPU (full spectrum, exact;
                                         n=12000 needs ~0.6 GB for L plus workspace)
      * larger (only if you raise the cap above eig_max_nodes)
                                      -> r = eig_truncated_k smallest eigenpairs via
                                         sparse shift-invert Lanczos (scipy, CPU; can
                                         take minutes for n >~ 20k), falling back to
                                         sparse LOBPCG. The truncated basis is a set of
                                         true, orthonormal eigenvectors, but Lanczos may
                                         skip some copies of highly degenerate
                                         eigenvalues (e.g. many leaves on one hub), so
                                         the "low band" is approximate there; R_spec is
                                         then a conservative upper bound (see
                                         _energy_split).
    """
    import time
    import torch
    from ..utils.progress import ManualBar

    subset, sub_ei, mapping = k_hop_subgraph(node_idx, edge_index, num_nodes,
                                             cfg_spec.k_hop)
    n_full = subset.numel()
    center = int(mapping.view(-1)[0])
    subset, sub_ei = _cap_subgraph(subset, sub_ei, center,
                                   getattr(cfg_spec, "max_subgraph_nodes", 0),
                                   seed=seed + int(node_idx))
    n = subset.numel()
    m = _derive_m(n, cfg_spec)
    L_sp = sparse_laplacian(sub_ei, n)
    n_edges = (L_sp._nnz() - n) // 2
    r = min(cfg_spec.eig_truncated_k, n - 1)

    method = getattr(cfg_spec, "eig_method", "auto")
    if method == "auto":
        method = "dense" if (n <= cfg_spec.eig_max_nodes or n <= 2 * r + 2) else "scipy"
    if method == "dense" and n > 30000:
        raise ValueError(
            f"Refusing dense eigh for n={n} (would need ~{n * n * 4 / 2**30:.0f} GiB). "
            f"Use eig_method=auto/scipy/lobpcg or lower spectral.max_subgraph_nodes.")

    if verbose:
        capped = f" (capped from {n_full})" if n_full != n else ""
        print(f"[spectrum] node={int(node_idx)}  {cfg_spec.k_hop}-hop |V|={n}{capped}  "
              f"|E|={n_edges}  solver={method}", flush=True)

    t0 = time.time()
    tol = getattr(cfg_spec, "eig_tol", 1e-4)
    L_keep = None
    evals = evecs = None
    if method == "dense":
        bar = ManualBar(total=1, desc=f"eigh(L) n={n}", disable=not verbose)
        try:
            evals, evecs = torch.linalg.eigh(L_sp.to_dense())   # ascending, exact
            L_keep = None                                        # full spectrum
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if verbose:
                print(f"[spectrum] GPU OOM in dense eigh (n={n}); "
                      f"falling back to sparse truncated solver", flush=True)
            method = "scipy"
        bar.update(1); bar.close()
    if evals is None:
        bar = ManualBar(total=1, desc=f"{method}(L) n={n} r={r}", disable=not verbose)
        try:
            if method == "lobpcg":
                evals, evecs = _eigs_lobpcg(L_sp, r, tol)
            else:
                evals, evecs = _eigs_scipy(L_sp, r, tol=0.0)
        except Exception as e:                                   # solver fallback chain
            if verbose:
                print(f"[spectrum] {method} failed ({type(e).__name__}: {e}); "
                      f"falling back to sparse LOBPCG", flush=True)
            method = "lobpcg"
            evals, evecs = _eigs_lobpcg(L_sp, r, tol)
        bar.update(1); bar.close()
        L_keep = L_sp                                            # exact total energy
    if verbose:
        print(f"[spectrum] done in {time.time() - t0:.1f}s  r={evecs.shape[1]}  "
              f"lambda_min={float(evals[0]):.3e}  lambda_max_computed={float(evals[-1]):.3e}",
              flush=True)

    return SpectrumCache(eigvecs=evecs, eigvals=evals.clamp_min(0),
                         node_index=subset, num_high=m, n=n,
                         L_sp=L_keep, method=method)


def positions_in_subgraph(spec, global_ids):
    """Rows of U for the given global node ids. Returns (pos, valid) where
    valid[i] is False if that id is not inside the cached subgraph.
    Relies on spec.node_index being sorted ascending."""
    import torch
    ids = torch.as_tensor(global_ids, device=spec.node_index.device).view(-1)
    pos = torch.searchsorted(spec.node_index, ids).clamp_max(spec.n - 1)
    valid = spec.node_index[pos] == ids
    return pos, valid


def spectral_coeffs(delta_sub, spec: SpectrumCache):
    """c_l = u_l^T delta over the subgraph (delta_sub is [n] node-signal)."""
    return spec.eigvecs.t() @ delta_sub            # [r]


def _energy_split(sig2d, spec: SpectrumCache):
    """(total, high) Dirichlet energy per channel for an [n, F] signal.

    Full spectrum (r == n):  total = sum_l c_l^2 lambda_l,  high = last-m band.
    Truncated spectrum:      total = sig^T L sig computed EXACTLY with the sparse
                             Laplacian (Parseval), low = first `keep` modes,
                             high = total - low  (everything not provably low-pass;
                             a conservative upper bound on the true high band)."""
    import torch
    c = spec.eigvecs.t() @ sig2d                         # [r, F]
    energy = c.pow(2) * spec.eigvals.unsqueeze(1)        # [r, F]
    if spec.L_sp is None:
        return energy.sum(dim=0), energy[spec.keep:].sum(dim=0)
    total = (sig2d * torch.sparse.mm(spec.L_sp, sig2d)).sum(dim=0)
    low = energy[:spec.keep].sum(dim=0)
    return total, (total - low).clamp_min(0)


def r_spec(delta_sub, spec: SpectrumCache, eps: float = 1e-12):
    """High-frequency energy ratio (Section 3.2). Differentiable in delta_sub."""
    total, high = _energy_split(delta_sub.unsqueeze(1), spec)
    return (high / (total + eps)).squeeze(0)


def lowpass_project(delta_sub, spec: SpectrumCache):
    """Project onto the low-frequency span (zero out high-frequency coeffs).
    Hard spectral constraint from Prop. 2 / Section 3.3 (spectral PGD step)."""
    return lowpass_project_matrix(delta_sub.unsqueeze(1), spec).squeeze(1)


def build_induced_signal(delta, target_node, hosts, spec: SpectrumCache, device):
    """Map the trigger's per-node feature perturbations onto the subgraph as an
    [n, F] graph-signal matrix.

    Each trigger node's feature row `delta[i]` is written to its attachment point:
    the target node gets the mean trigger row; host nodes that lie inside the
    cached subgraph each receive one trigger row (cycled if there are more hosts
    than trigger nodes). Attachment points outside the subgraph are ignored.
    Vectorised (index_add) — no per-host Python loop / clone."""
    import torch
    n = spec.n
    F = delta.shape[1]
    sig = torch.zeros(n, F, device=device, dtype=delta.dtype)

    tp, tv = positions_in_subgraph(spec, [int(target_node)])
    if bool(tv[0]):
        sig = sig.index_add(0, tp[tv], delta.mean(dim=0, keepdim=True))

    host_t = torch.as_tensor(hosts, device=device).view(-1)
    if host_t.numel() > 0:
        hp, hv = positions_in_subgraph(spec, host_t)
        if bool(hv.any()):
            rows = torch.arange(host_t.numel(), device=device) % delta.shape[0]
            sig = sig.index_add(0, hp[hv], delta[rows[hv]])
    return sig


def r_spec_matrix(signal_mat, spec: SpectrumCache, eps: float = 1e-12):
    """High-frequency energy ratio for an [n, F] matrix of node signals (one graph
    signal per feature channel). Returns the mean ratio over channels — a genuine,
    differentiable function of how the trigger's per-node features vary across the
    subgraph (a constant signal is maximally smooth -> ratio ~0; a sharp, localised
    one -> high ratio)."""
    total, high = _energy_split(signal_mat, spec)
    return (high / (total + eps)).mean()


def lowpass_project_matrix(signal_mat, spec: SpectrumCache):
    """Project each column of an [n, F] signal matrix onto the low-frequency span
    u_1..u_keep (keep = min(r, n-m))."""
    k = spec.keep
    U = spec.eigvecs[:, :k]
    return U @ (U.t() @ signal_mat)                  # [n, F]


def parseval_lhs_rhs(delta_sub, spec: SpectrumCache):
    """Return (delta^T L delta, sum_l c_l^2 lambda_l) for the identity check.
    (For a truncated spectrum rhs <= lhs; equality holds iff r == n.)"""
    import torch
    if spec.L_sp is None:
        L = (spec.eigvecs * spec.eigvals) @ spec.eigvecs.t()
        lhs = delta_sub @ (L @ delta_sub)
    else:
        lhs = delta_sub @ torch.sparse.mm(spec.L_sp, delta_sub.unsqueeze(1)).squeeze(1)
    c = spectral_coeffs(delta_sub, spec)
    rhs = (c.pow(2) * spec.eigvals).sum()
    return lhs, rhs
