"""
Step 3 — Orthogonal-subspace persistence anchoring (Section 4).

  * Pretraining-only representation covariance Sigma = Cov_{z~D_pre} phi(z)
    = V Lambda V^T (Section 4.2); U_k = top-k eigenvectors of Sigma, a label-free
    proxy for the subspace downstream fine-tuning moves within.
  * Persistence penalty L_persist = mu * || U_k^T grad_theta phi(delta) ||^2
    (Section 4.3 / 5.3): pushes the trigger's parameter-gradient footprint into
    the orthogonal complement of U_k.
  * Davis-Kahan sin-theta diagnostic (Section 4.5): bounds U_k vs the true
    downstream subspace; reported, not relied upon.

The progress-barred eigendecomposition for Sigma is small (k x k in embedding
space), so no bar is needed here; the slow eigendecomposition is the graph
Laplacian one in spectral.py.
"""
from __future__ import annotations


def pretraining_covariance_subspace(model, mdg, k, sample_size, device,
                                    verbose=True):
    """Return U_k in R^{H x k}: top-k eigenvectors of the embedding covariance."""
    import torch
    with torch.no_grad():
        emb = model.encode(mdg.x, mdg.edge_index,
                           is_coordinator=getattr(mdg, "is_coordinator", None),
                           domain_id=getattr(mdg, "domain_id", None))
        n = emb.shape[0]
        idx = torch.randperm(n, device=emb.device)[:min(sample_size, n)]
        z = emb[idx]
        z = z - z.mean(dim=0, keepdim=True)
        Sigma = (z.t() @ z) / max(1, z.shape[0] - 1)      # [H, H]
        evals, evecs = torch.linalg.eigh(Sigma)           # ascending
        U_k = evecs[:, -k:]                                # top-k
    if verbose:
        tail = evals[-k:]
        print(f"[orthogonal] Sigma {tuple(Sigma.shape)}  top-{k} eig range "
              f"[{tail.min().item():.3e}, {tail.max().item():.3e}]")
    return U_k.detach(), evals.detach()


def orthogonality_penalty(U_k, grad_phi_delta):
    """|| U_k^T g ||^2 where g = grad_theta phi(delta) projected to embedding space.

    In practice we use the *embedding-space* gradient direction of the triggered
    representation as a tractable stand-in for grad_theta phi(delta) under the
    linearised/NTK view (the two are related by the fixed Jacobian J): penalising
    the embedding-space component inside U_k pushes the trigger's footprint toward
    the orthogonal complement.
    """
    import torch
    proj = U_k.t() @ grad_phi_delta        # [k] or [k, *]
    return proj.pow(2).sum()


def davis_kahan_bound(eigvals_sorted_desc, k, perturbation_norm):
    """|| sin Theta(U_k, U_k_true) || <= pi * ||E|| / delta_gap  (Section 4.5).

    delta_gap = lambda_k - lambda_{k+1} (eigengap at the cut). Returns the bound,
    or inf if the gap collapses.
    """
    import torch
    ev = eigvals_sorted_desc
    if k >= ev.numel():
        return float("inf")
    gap = (ev[k - 1] - ev[k]).item()
    if gap <= 0:
        return float("inf")
    return 3.141592653589793 * perturbation_norm / gap
