
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImplantReport:
    steps_run: int
    num_poison_nodes: int
    anchor_norm: float
    final_bd_loss: float
    final_keep_loss: float


def _anchor_direction(e_target, U_k, mean_clean_norm, cfg_pp, device):
    """Representation-space point the backdoor should send triggered nodes to."""
    import torch

    if cfg_pp.anchor_mode == "orthogonal" and U_k is not None:
        # Component of e_target orthogonal to the top-k pretraining-covariance
        # subspace (the subspace downstream fine-tuning/probing mostly moves
        # within, per Sec 4). A linear head fit on a few triggered examples
        # separates this direction easily, and gradient updates confined near
        # U_k (as most fine-tuning/probing updates are) barely erode it.
        proj = U_k @ (U_k.t() @ e_target)
        ortho = e_target - proj
        n = ortho.norm()
        direction = ortho / n if n > 1e-8 else torch.nn.functional.normalize(
            e_target, dim=0)
    else:
        direction = torch.nn.functional.normalize(e_target, dim=0)

    return direction * (cfg_pp.anchor_scale * mean_clean_norm)


def implant_backdoor(model, mdg, artifacts, e_target, U_k, cfg, device,
                     verbose=True):
    """Fine-tune `model` in place so triggered pretraining nodes land on a fixed
    anchor while clean embeddings are preserved. Returns an ImplantReport."""
    import torch
    import torch.nn.functional as F

    from .trigger import NeighborIndex, inject_triggers_batch

    cfg_pp = cfg.attack.pretrain_poison
    if not cfg_pp.enabled or not artifacts.triggers:
        return ImplantReport(0, 0, 0.0, float("nan"), float("nan"))

    is_coord = getattr(mdg, "is_coordinator", None)
    real_mask_cpu = ((~is_coord) if is_coord is not None else torch.ones(
        mdg.x.shape[0], dtype=torch.bool)).detach().cpu()
    real_idx_cpu = real_mask_cpu.nonzero(as_tuple=True)[0]
    real_idx = real_idx_cpu.to(device)

    # ---- pick the poisoned pretraining pool: every influence-selected target
    # (these already have optimised triggers and are, by construction, the
    # highest-leverage nodes) plus a random top-up to reach poison_rate ----
    # (selection kept entirely on CPU to avoid device-mismatched indexing)
    g = torch.Generator().manual_seed(cfg.runtime.seed + 7)
    target_set = [int(t) for t in artifacts.target_indices]
    n_pool = max(len(target_set),
                int(round(cfg.attack.poison_rate * real_idx_cpu.numel())))
    if n_pool > len(target_set):
        target_mask = torch.zeros(real_idx_cpu.numel(), dtype=torch.bool)
        tset = torch.as_tensor(target_set, dtype=torch.long)
        target_mask |= torch.isin(real_idx_cpu, tset)
        others = real_idx_cpu[~target_mask]
        perm = others[torch.randperm(others.numel(), generator=g)]
        extra = perm[: n_pool - len(target_set)].tolist()
    else:
        extra = []
    poison_nodes = torch.as_tensor(sorted(set(target_set + extra)),
                                   dtype=torch.long, device=device)

    # ---- mean embedding norm + frozen clean-embedding snapshot (utility term) ----
    model.eval()
    with torch.no_grad():
        clean_emb0 = model.encode(mdg.x, mdg.edge_index,
                                  is_coordinator=is_coord,
                                  domain_id=getattr(mdg, "domain_id", None))
        mean_clean_norm = float(clean_emb0[real_idx].norm(dim=1).mean())

    anchor = _anchor_direction(e_target, U_k, mean_clean_norm, cfg_pp, device
                               ).detach()

    nbr = NeighborIndex(mdg.edge_index, mdg.x.shape[0], is_coord)
    all_deltas = torch.stack([t.delta.detach() for t in artifacts.triggers]
                             ).to(device)
    max_edges = cfg.attack.trigger.max_edges

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg_pp.lr,
                           weight_decay=cfg_pp.weight_decay)

    bs = min(cfg_pp.batch_nodes, poison_nodes.numel())
    last_bd, last_keep = float("nan"), float("nan")
    for step in range(cfg_pp.steps):
        opt.zero_grad()

        # --- backdoor term: triggered poison-pool nodes -> anchor ---
        bperm = poison_nodes[torch.randperm(poison_nodes.numel(),
                                            device=device)[:bs]]
        pos = torch.arange(bperm.numel())
        deltas = all_deltas[(pos % all_deltas.shape[0]).to(all_deltas.device)]
        g_trig, _ = inject_triggers_batch(mdg, bperm, deltas, max_edges, nbr,
                                          device)
        emb_trig = model.encode(g_trig.x, g_trig.edge_index,
                                is_coordinator=getattr(g_trig, "is_coordinator",
                                                       None),
                                domain_id=getattr(g_trig, "domain_id", None))
        emb_at_poison = emb_trig[bperm]
        L_bd = F.mse_loss(emb_at_poison, anchor.unsqueeze(0).expand_as(
            emb_at_poison))

        # --- utility term: clean nodes keep their pre-implant embeddings ---
        kperm = real_idx[torch.randperm(real_idx.numel(),
                                        device=real_idx.device)
                         [:min(cfg_pp.keep_nodes, real_idx.numel())]]
        emb_clean = model.encode(mdg.x, mdg.edge_index,
                                 is_coordinator=is_coord,
                                 domain_id=getattr(mdg, "domain_id", None))
        L_keep = F.mse_loss(emb_clean[kperm], clean_emb0[kperm])

        loss = cfg_pp.bd_weight * L_bd + cfg_pp.keep_weight * L_keep
        if cfg_pp.pretrain_weight > 0:
            loss = loss + cfg_pp.pretrain_weight * model.pretrain_loss(mdg)

        loss.backward()
        opt.step()
        last_bd, last_keep = float(L_bd.detach()), float(L_keep.detach())

        if verbose and (step % max(1, cfg_pp.steps // 5) == 0):
            print(f"[implant] step {step:4d}  L_bd={last_bd:.4f}  "
                 f"L_keep={last_keep:.4f}  poison_nodes={poison_nodes.numel()}",
                 flush=True)
        del g_trig, emb_trig, emb_clean
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model.eval()
    if verbose:
        print(f"[implant] done: {cfg_pp.steps} steps, "
             f"{poison_nodes.numel()} poisoned pretraining nodes, "
             f"anchor ||.||={float(anchor.norm()):.4f} "
             f"(mean clean ||emb||={mean_clean_norm:.4f})", flush=True)
    return ImplantReport(cfg_pp.steps, int(poison_nodes.numel()),
                         float(anchor.norm()), last_bd, last_keep)
