"""
Complete loss functions (paper Section 5).

L_eff     = - sim( phi(G'_v*), e_y* )                                   (5.1)
L_stealth = alpha * R_spec(delta) + beta * ||delta||^2
            + gamma * | L_pre(theta; A+ΔA) - L_pre(theta; A) |          (5.2)
L_persist = mu * || U_k^T grad_theta phi(delta) ||^2                    (5.3)
Joint     = L_eff + L_stealth + L_persist                               (5.4)

Ablation switches (instruction 4) are honoured here:
  * no spectral constraint     -> alpha = 0 (and hard low-pass disabled upstream)
  * no orthogonality term       -> mu = 0
  * clean-loss-preservation off -> gamma term dropped (use_clean_loss_term=False)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossBreakdown:
    total: float
    eff: float
    spec: float
    feat: float
    clean: float
    persist: float
    r_spec_value: float


def effectiveness_loss(triggered_emb_target, target_embedding):
    """- cosine similarity to the attacker's target embedding e_y*."""
    import torch
    import torch.nn.functional as F
    a = F.normalize(triggered_emb_target.view(1, -1), dim=1)
    b = F.normalize(target_embedding.view(1, -1), dim=1)
    return -(a * b).sum()


def feature_penalty(delta):
    return delta.pow(2).sum()


def clean_loss_shift(model, mdg_clean, mdg_triggered):
    """| L_pre(theta; A+ΔA) - L_pre(theta; A) | (Section 5.2, 3rd term)."""
    import torch
    # The clean loss does not depend on the trigger (theta is frozen), so no autograd
    # graph is needed for it — this halves the activation memory of this term.
    with torch.no_grad():
        lc = model.pretrain_loss(mdg_clean)
    lt = model.pretrain_loss(mdg_triggered)
    return (lt - lc).abs()


def compute_joint_loss(*, model, mdg_clean, mdg_triggered, trig_ids,
                       target_node, target_embedding, spec, trigger,
                       hosts, U_k, cfg_attack, device):
    """Assemble the full objective for one optimisation step.

    Returns (scalar loss tensor, LossBreakdown).
    """
    import torch
    from .spectral import r_spec_matrix, build_induced_signal
    from .orthogonal import orthogonality_penalty

    tcfg = cfg_attack.trigger
    scfg = cfg_attack.spectral
    ocfg = cfg_attack.orthogonal

    # --- embeddings on the triggered graph ---
    emb = model.encode(mdg_triggered.x, mdg_triggered.edge_index,
                       is_coordinator=getattr(mdg_triggered, "is_coordinator", None),
                       domain_id=getattr(mdg_triggered, "domain_id", None))
    emb_target = emb[target_node]

    # --- L_eff ---
    L_eff = effectiveness_loss(emb_target, target_embedding)

    # --- spectral penalty R_spec on the trigger signal mapped into the subgraph ---
    # Build an [n, F] node-signal matrix over the cached subgraph: each trigger
    # node's feature-perturbation row is placed at its attachment point (target +
    # hosts). R_spec then measures the high-frequency energy of the signal the
    # trigger actually injects into the local Laplacian eigenbasis, and depends on
    # how the trigger nodes' features VARY across their attachment points — so
    # minimising it smooths the trigger (the §3.1 low-frequency design principle).
    sig_mat = build_induced_signal(trigger.delta, target_node, hosts, spec, device)
    R = r_spec_matrix(sig_mat, spec)
    alpha = scfg.alpha_spectral
    L_spec = alpha * R

    # --- feature magnitude penalty ---
    L_feat = tcfg.beta_feat * feature_penalty(trigger.delta)

    # --- clean-loss-preservation term (ablatable) ---
    if tcfg.use_clean_loss_term:
        L_clean = tcfg.gamma_clean * clean_loss_shift(model, mdg_clean, mdg_triggered)
    else:
        L_clean = torch.zeros((), device=device)

    # --- orthogonal persistence penalty (ablatable via mu) ---
    if ocfg.mu_persist > 0 and U_k is not None:
        # Embedding-space footprint of the triggered target representation.
        # (OOM FIX) This block used to also call
        #     torch.autograd.grad(emb_target.norm(), trigger.delta, create_graph=True)
        # and add `0.0 * g.pow(2).sum()` to the loss. That term contributed exactly
        # zero to both the loss value and its gradient, but create_graph=True forced a
        # double-backward graph through the FULL-graph GCN encoder, which keeps
        # several [num_edges, hidden] buffers alive (13+ GiB each) and caused the
        # CUDA OOM. Removing it leaves the optimisation mathematically unchanged.
        phi_dir = emb_target / (emb_target.norm() + 1e-12)
        L_persist = ocfg.mu_persist * orthogonality_penalty(U_k, phi_dir)
    else:
        L_persist = torch.zeros((), device=device)

    total = L_eff + L_spec + L_feat + L_clean + L_persist

    bd = LossBreakdown(
        total=float(total.detach()),
        eff=float(L_eff.detach()),
        spec=float(L_spec.detach()),
        feat=float(L_feat.detach()),
        clean=float(L_clean.detach()),
        persist=float(L_persist.detach()),
        r_spec_value=float(R.detach()),
    )
    return total, bd
