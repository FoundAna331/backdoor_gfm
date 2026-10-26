
from __future__ import annotations

from dataclasses import dataclass
from typing import List


# --------------------------------------------------------------------------- #
#  Parameter / gradient flattening helpers
# --------------------------------------------------------------------------- #
def _trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _flat_grad(scalar_loss, params, create_graph=False, retain_graph=True):
    import torch
    grads = torch.autograd.grad(scalar_loss, params, create_graph=create_graph,
                                retain_graph=retain_graph, allow_unused=True)
    flat = []
    for g, p in zip(grads, params):
        flat.append((g if g is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(flat)


def _param_groups(model, block_diagonal: bool):
    """Return a list of (name, [params]) blocks for the (block-diagonal) iHVP."""
    if not block_diagonal:
        return [("all", _trainable_params(model))]
    groups = []
    for name, module in model.named_children():
        ps = [p for p in module.parameters() if p.requires_grad]
        if ps:
            groups.append((name, ps))
    # catch params not under a child module
    seen = {id(p) for _, ps in groups for p in ps}
    leftover = [p for p in _trainable_params(model) if id(p) not in seen]
    if leftover:
        groups.append(("_root", leftover))
    return groups


# --------------------------------------------------------------------------- #
#  Fisher-vector product and iHVP
# --------------------------------------------------------------------------- #
class FisherOperator:
    """Matrix-free G_lambda for a block of parameters.

    G u = (1/|B|) sum_{z in B} g_z (g_z^T u)   (empirical Fisher)
    G_lambda u = G u + lambda u

    `per_sample_grads` is a [B, P] tensor of flattened per-sample gradients over a
    reference batch B, pre-computed once and reused across all iHVP solves.
    """

    def __init__(self, per_sample_grads, damping: float):
        self.G = per_sample_grads          # [B, P]
        self.lam = damping
        self.B = per_sample_grads.shape[0]

    def matvec(self, u):
        # (1/B) G^T (G u) + lam u
        Gu = self.G @ u                    # [B]
        GtGu = self.G.t() @ Gu             # [P]
        return GtGu / self.B + self.lam * u

    def spectral_scale(self):
        """Rough upper bound on the largest eigenvalue, for Neumann scaling."""
        import torch
        # ||G^T G||/B <= ||G||_F^2 / B ; add damping.
        return (self.G.pow(2).sum() / self.B + self.lam).item()


def ihvp_neumann(op: "FisherOperator", v, num_terms: int, scale: float):
    """Approximate G_lambda^{-1} v by a scaled Neumann series.

    With A = G_lambda / s (s >= lambda_max), (G_lambda)^{-1} = (1/s) sum_i (I - A)^i.
    r_0 = v/s ; r_{i} = v/s + (I - A) r_{i-1}.
    """
    import torch
    s = max(scale, op.spectral_scale())
    v_s = v / s
    r = v_s.clone()
    for _ in range(num_terms):
        Ar = op.matvec(r) / s
        r = v_s + r - Ar
    return r


def ihvp_lissa(op: "FisherOperator", v, depth: int, samples: int, scale: float):
    """LiSSA stochastic iHVP (here deterministic over the same batch per draw)."""
    import torch
    s = max(scale, op.spectral_scale())
    est = torch.zeros_like(v)
    for _ in range(max(1, samples)):
        r = v.clone()
        for _ in range(depth):
            r = v + r - op.matvec(r) / s
        est = est + r / s
    return est / max(1, samples)


# --------------------------------------------------------------------------- #
#  Per-node losses / embeddings
# --------------------------------------------------------------------------- #
def _checkpointed_encode(model, mdg):
    """model.encode(), wrapped in activation checkpointing.

    use_reentrant=False is required (not just preferred): it's the checkpoint
    implementation that supports a create_graph=True double-backward through
    the checkpointed region, which the propagation-score JVP below needs.
    The model is frozen and in eval() mode during selection (no dropout/BN
    randomness), so re-running the forward inside backward() reproduces
    exactly the same values as a normal, non-checkpointed call -- this is
    free to use anywhere model.encode() was called before, with no change in
    results, only in when/how long its backward buffers are held in memory.
    """
    from torch.utils.checkpoint import checkpoint
    is_coord = getattr(mdg, "is_coordinator", None)
    dom_id = getattr(mdg, "domain_id", None)

    def _fwd(x, edge_index, is_coord, dom_id):
        return model.encode(x, edge_index, is_coordinator=is_coord, domain_id=dom_id)

    return checkpoint(_fwd, mdg.x, mdg.edge_index, is_coord, dom_id,
                      use_reentrant=False)


def default_node_loss_fn(model, mdg):
    """Return a callable node_loss(node_idx) -> scalar self-supervised loss for a
    single node (its local reconstruction error). Used for grad_theta l(z_v)."""
    import torch.nn.functional as F

    def node_loss(node_idx):
        h = _checkpointed_encode(model, mdg)
        # local reconstruction target = the node's own (aligned) features via a
        # linear read-out if the model has a decoder, else embedding norm.
        if hasattr(model, "decoder"):
            recon = model.decoder(h[node_idx:node_idx + 1])
            return F.mse_loss(recon, mdg.x[node_idx:node_idx + 1])
        return h[node_idx].pow(2).mean()

    return node_loss


def embedding_fn(model, mdg):
    def embed():
        return _checkpointed_encode(model, mdg)
    return embed


# --------------------------------------------------------------------------- #
#  Main selector
# --------------------------------------------------------------------------- #
@dataclass
class InfluenceResult:
    target_indices: List[int]
    scores: "object"                 # S(v) for scored candidates
    self_influence: "object"         # |I_lambda(v,v)|
    stealth_eligible: "object"       # boolean mask over scored candidates
    candidate_indices: List[int]


class InfluenceSelector:
    def __init__(self, model, mdg, cfg_infl, device, verbose=True):
        self.model = model
        self.mdg = mdg
        self.cfg = cfg_infl
        self.device = device
        self.verbose = verbose
        self.params_blocks = _param_groups(model, cfg_infl.kfac_block_diagonal)

    # ---- curvature: per-sample grads over the reference batch D ---- #
    def _build_fisher(self, ref_nodes):
        import torch
        from ..utils.progress import pbar
        node_loss = default_node_loss_fn(self.model, self.mdg)
        blocks_grads = {name: [] for name, _ in self.params_blocks}
        for w in pbar(ref_nodes, desc="Fisher per-sample grads"):
            loss = node_loss(int(w))
            for name, ps in self.params_blocks:
                g = _flat_grad(loss, ps, retain_graph=True)
                blocks_grads[name].append(g.detach())
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        ops = {}
        for name, _ in self.params_blocks:
            G = torch.stack(blocks_grads[name], dim=0)   # [B, P_block]
            ops[name] = FisherOperator(G, self.cfg.damping)
        return ops

    def _ihvp(self, ops, vecs_by_block):
        out = {}
        for name, _ in self.params_blocks:
            op = ops[name]
            v = vecs_by_block[name]
            if self.cfg.ihvp_method == "lissa":
                out[name] = ihvp_lissa(op, v, self.cfg.lissa_depth,
                                       self.cfg.lissa_samples, self.cfg.neumann_scale)
            else:
                out[name] = ihvp_neumann(op, v, self.cfg.neumann_terms,
                                         self.cfg.neumann_scale)
        return out

    def _grad_blocks(self, scalar_loss):
        return {name: _flat_grad(scalar_loss, ps, retain_graph=True)
                for name, ps in self.params_blocks}

    def self_influence(self, node_idx, ops):
        """|I_lambda(v,v)| = | g_v^T G_lambda^{-1} g_v |."""
        import torch
        node_loss = default_node_loss_fn(self.model, self.mdg)
        g = self._grad_blocks(node_loss(node_idx))
        a = self._ihvp(ops, g)
        val = sum((g[name] * a[name]).sum() for name, _ in self.params_blocks)
        return val.abs().detach()

    def propagation_score(self, node_idx, ops, ref_nodes):
        """S(v) = E_{w in D} || I(v->w) ||_2 via a JVP along a = -G^{-1} g_v."""
        node_loss = default_node_loss_fn(self.model, self.mdg)
        g = self._grad_blocks(node_loss(node_idx))
        a = self._ihvp(ops, g)
        # map block iHVP vectors back to a per-parameter tangent list for the JVP
        tangents = self._blocks_to_param_tangents(a)
        # JVP of the embedding map along a: row w = J_w a = I(v->w) up to sign.
        jvp_rows = self._jvp_embedding(ref_nodes, tangents)
        I_vw = -jvp_rows                            # minus sign from -G^{-1}
        return I_vw.norm(dim=1).mean().detach()

    def _blocks_to_param_tangents(self, a_blocks):
        """Reshape flat per-block iHVP vectors into a list matching model params."""
        import torch
        params = _trainable_params(self.model)
        # Build name->params map consistent with self.params_blocks order.
        tangents = [torch.zeros_like(p) for p in params]
        pid_to_pos = {id(p): i for i, p in enumerate(params)}
        for name, ps in self.params_blocks:
            vec = a_blocks[name]
            off = 0
            for p in ps:
                numel = p.numel()
                tangents[pid_to_pos[id(p)]] = vec[off:off + numel].view_as(p)
                off += numel
        return tangents

    def _jvp_embedding(self, ref_nodes, tangents):
        """Compute J a for the embedding map restricted to ref_nodes via the
        double-backward trick: J a = d/du [ (grad_theta (emb . u))^T a ] at u=0.
        We instead use the equivalent forward directional derivative through
        torch.autograd.grad with grad_outputs."""
        import torch
        params = _trainable_params(self.model)
        emb = embedding_fn(self.model, self.mdg)()
        ref = emb[ref_nodes]                        # [|D|, H], depends on params
        # u is a dummy with same shape as ref; (ref * u).sum() has grad wrt params
        u = torch.zeros_like(ref, requires_grad=True)
        inner = (ref * u).sum()
        # Some params (decoder / VQ codebook) are used only in pretrain_loss, not in
        # encode(), so they are absent from the embedding graph. allow_unused=True
        # returns None for them; their d(emb)/d(param) is 0, so they contribute
        # nothing to the JVP J a and are simply skipped.
        g_params = torch.autograd.grad(inner, params, create_graph=True,
                                       allow_unused=True)
        # directional derivative: sum_i <g_params_i, tangent_i>  (skip unused params)
        terms = [(gp * t).sum() for gp, t in zip(g_params, tangents)
                 if gp is not None]
        if not terms:
            return torch.zeros_like(ref)
        dir_scalar = sum(terms)
        jvp = torch.autograd.grad(dir_scalar, u, retain_graph=False)[0]  # [|D|, H]
        return jvp

    # ------------------------------------------------------------------ #
    def select(self):
        import torch
        from ..utils.progress import pbar

        mdg = self.mdg
        num_real = getattr(mdg, "num_real_nodes", mdg.x.shape[0])
        # candidate pool = real (non-coordinator) nodes, subsampled for tractability
        is_coord = getattr(mdg, "is_coordinator", None)
        pool = torch.arange(num_real, device=self.device)
        if is_coord is not None:
            pool = pool[~is_coord[:num_real]]
        cand_cap = max(self.cfg.top_k_targets * 10, 200)
        if pool.numel() > cand_cap:
            sel = torch.randperm(pool.numel(), device=self.device)[:cand_cap]
            candidates = pool[sel]
        else:
            candidates = pool
        candidates = candidates.tolist()

        # reference distribution D (proxy downstream queries) — pretraining nodes
        ref_cap = min(self.cfg.reference_sample_size, num_real)
        ref_nodes = torch.randperm(num_real, device=self.device)[:ref_cap].tolist()

        if self.verbose:
            print(f"[influence] candidates={len(candidates)}  |D|={len(ref_nodes)}  "
                  f"node_to_node={self.cfg.use_node_to_node_influence}  "
                  f"ihvp={self.cfg.ihvp_method}  blocks={len(self.params_blocks)}")

        ops = self._build_fisher(ref_nodes)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self_infl = []
        for v in pbar(candidates, desc="self-influence I(v,v)"):
            self_infl.append(self.self_influence(v, ops))
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self_infl = torch.stack(self_infl)

        # stealth-eligible set C: |I(v,v)| <= tau_clean (percentile threshold)
        tau_clean = torch.quantile(self_infl,
                                   self.cfg.self_influence_cap_percentile / 100.0)
        eligible = self_infl <= tau_clean

        # score S(v)
        if self.cfg.use_node_to_node_influence:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            scores = []
            for v in pbar(candidates, desc="propagation score S(v)"):
                scores.append(self.propagation_score(v, ops, ref_nodes))
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            scores = torch.stack(scores)
        else:
            # node-to-node excluded -> rank by self-influence magnitude only
            scores = self_infl.clone()

        # restrict to C, then take top-k (replaces argmax)
        masked = scores.clone()
        masked[~eligible] = float("-inf")
        k = min(self.cfg.top_k_targets, int(eligible.sum().item()) or 1)
        topk = torch.topk(masked, k).indices.tolist()
        target_indices = [candidates[i] for i in topk]

        if self.verbose:
            print(f"[influence] tau_clean(p{self.cfg.self_influence_cap_percentile})"
                  f"={tau_clean.item():.4e}  eligible={int(eligible.sum())}/"
                  f"{len(candidates)}  -> selected top-{k} targets")

        return InfluenceResult(
            target_indices=target_indices,
            scores=scores.detach().cpu(),
            self_influence=self_infl.detach().cpu(),
            stealth_eligible=eligible.detach().cpu(),
            candidate_indices=candidates,
        )


def select_targets_random(mdg, k, device, seed=0):
    """Ablation: no influence-based selection at all (random hosts)."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_real = getattr(mdg, "num_real_nodes", mdg.x.shape[0])
    perm = torch.randperm(num_real, generator=g)[:k]
    return perm.tolist()
