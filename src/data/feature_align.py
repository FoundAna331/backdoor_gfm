"""
Feature alignment to a common dimension.

All three target foundation models align per-domain node features to a shared
dimension before combining domains:
  * GCOPE  — projects/ aligns features, then connects domains with coordinators.
  * SAMGPT — "aligns the feature distributions from multiple source domains
              following previous work" before adding structure tokens.
  * GFT    — encodes all domains through a shared encoder over a common input.

We implement the standard SVD projection (used across this line of work): each
domain's feature matrix X_d in R^{N_d x F_d} is mapped to R^{N_d x d} by taking
the top-d right singular vectors of X_d. `pca` additionally mean-centers; `pad`
is a trivial truncate/zero-pad fallback for debugging.
"""
from __future__ import annotations


def align_features(x, target_dim: int, method: str = "svd"):
    """Project a single [N, F] feature matrix to [N, target_dim].

    Returns a new tensor on the same device/dtype as `x`.
    """
    import torch

    n, f = x.shape
    if f == target_dim:
        return x.contiguous()

    if method == "pad":
        if f > target_dim:
            return x[:, :target_dim].contiguous()
        out = x.new_zeros((n, target_dim))
        out[:, :f] = x
        return out

    xc = x
    if method == "pca":
        xc = x - x.mean(dim=0, keepdim=True)

    # Economy SVD: X = U S V^T ; project onto top-k right singular vectors.
    # torch.linalg.svd is used on CPU/GPU; for very wide X we take a thin SVD.
    try:
        U, S, Vh = torch.linalg.svd(xc, full_matrices=False)
    except Exception:
        # numerical fallback via eigendecomposition of the smaller Gram matrix
        gram = xc.t() @ xc
        evals, evecs = torch.linalg.eigh(gram)
        Vh = evecs.flip(dims=[1]).t()          # right singular vectors (desc order)

    k = min(target_dim, Vh.shape[0])
    proj = Vh[:k].t()                      # [F, k]
    out = xc @ proj                        # [N, k]
    if k < target_dim:                     # pad if a domain has < target_dim feats
        pad = out.new_zeros((n, target_dim - k))
        out = torch.cat([out, pad], dim=1)
    return out.contiguous()
