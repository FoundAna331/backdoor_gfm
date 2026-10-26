
from __future__ import annotations

import copy
from typing import Dict


def _encode(model, g):
    return model.encode(g.x, g.edge_index,
                        is_coordinator=getattr(g, "is_coordinator", None),
                        domain_id=getattr(g, "domain_id", None))


def _predict(model, clf, g):
    import torch
    with torch.no_grad():
        return clf(_encode(model, g)).argmax(dim=1)


def make_split(y, train_frac, seed):
    """Random train/test split over labelled real nodes (CPU long tensors)."""
    import torch
    labelled = (y >= 0).nonzero(as_tuple=True)[0].cpu()
    g = torch.Generator().manual_seed(int(seed))
    perm = labelled[torch.randperm(labelled.numel(), generator=g)]
    n_tr = int(train_frac * perm.numel())
    return perm[:n_tr], perm[n_tr:]


def pick_deltas(all_deltas, positions, trigger_index):
    """Trigger perturbations [B, t, F] for the given running positions.
    trigger_index < 0 -> round-robin over all trained triggers."""
    import torch
    K = all_deltas.shape[0]
    if trigger_index is not None and trigger_index >= 0:
        idx = torch.full_like(positions, min(trigger_index, K - 1))
    else:
        idx = positions % K
    return all_deltas[idx.to(all_deltas.device)]


def _fit(model, g, y, train_idx, num_classes, ecfg, device):
    """Train a head (and, in 'full' mode, a copy of the encoder). Returns
    (model_used_for_inference, head)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    train_idx = train_idx.to(device)
    emb_dim = getattr(model, "embed_dim", None)
    if emb_dim is None:
        with torch.no_grad():
            emb_dim = _encode(model, g).shape[1]
    clf = nn.Linear(emb_dim, num_classes).to(device)

    if ecfg.finetune_mode == "full":
        ft = copy.deepcopy(model).to(device)
        ft.train()
        opt = torch.optim.Adam(
            [{"params": [p for p in ft.parameters() if p.requires_grad],
              "lr": ecfg.ft_encoder_lr},
             {"params": clf.parameters(), "lr": ecfg.ft_lr}],
            weight_decay=ecfg.ft_weight_decay)
        for _ in range(ecfg.ft_epochs):
            opt.zero_grad()
            emb = _encode(ft, g)
            loss = F.cross_entropy(clf(emb[train_idx]), y[train_idx])
            loss.backward()
            opt.step()
        ft.eval()
        return ft, clf

    # probe: frozen encoder, embeddings computed once
    model.eval()
    with torch.no_grad():
        emb = _encode(model, g).detach()
    opt = torch.optim.Adam(clf.parameters(), lr=ecfg.ft_lr,
                           weight_decay=ecfg.ft_weight_decay)
    for _ in range(ecfg.ft_epochs):
        opt.zero_grad()
        loss = F.cross_entropy(clf(emb[train_idx]), y[train_idx])
        loss.backward()
        opt.step()
    return model, clf


def _accuracy(preds, y, idx):
    idx = idx.to(preds.device)
    return float((preds[idx] == y[idx]).float().mean())


def _measure_asr(model, clf, tg, asr_nodes, all_deltas, ecfg, max_edges, nbr,
                 target_label, device, defense=None, artifacts=None,
                 defense_model=None):
    """Fraction of `asr_nodes` predicted as `target_label` once their trigger is
    attached. Triggers are injected `asr_batch_size` at a time."""
    import torch
    from ..attack.trigger import inject_triggers_batch
    hits = []
    bs = max(1, ecfg.asr_batch_size)
    for start in range(0, asr_nodes.numel(), bs):
        chunk = asr_nodes[start:start + bs]
        pos = torch.arange(start, start + chunk.numel())
        deltas = pick_deltas(all_deltas, pos, ecfg.trigger_index)
        g_trig, _ = inject_triggers_batch(tg, chunk, deltas, max_edges, nbr, device)
        if defense is not None:
            g_trig, _ = defense.apply(g_trig, artifacts, defense_model)
        preds = _predict(model, clf, g_trig)
        hits.append((preds[chunk.to(device)] == target_label).float())
        del g_trig
    return float(torch.cat(hits).mean()) if hits else float("nan")


def run_once(model, tg, artifacts, cfg, device, seed, defense=None) -> Dict[str, float]:
    """One split of the protocol described in the module docstring."""
    import torch
    from ..attack.trigger import NeighborIndex, inject_triggers_batch

    ecfg = cfg.eval
    y_target = int(cfg.attack.target_label)
    max_edges = cfg.attack.trigger.max_edges
    y = tg.y
    N = tg.x.shape[0]

    num_classes = int(y[y >= 0].max().item()) + 1
    if y_target >= num_classes:
        raise ValueError(f"target_label={y_target} but this dataset only has "
                         f"{num_classes} classes (0..{num_classes - 1}).")
    if not artifacts.triggers:
        raise ValueError("no trained triggers available (attack produced none).")

    train_idx, test_idx = make_split(y, ecfg.train_frac, seed)
    g = torch.Generator().manual_seed(int(seed) + 1)
    y_cpu = y.cpu()
    nbr = NeighborIndex(tg.edge_index, N, getattr(tg, "is_coordinator", None))
    all_deltas = torch.stack([t.delta.detach() for t in artifacts.triggers]).to(device)

    # ---- nodes used for ASR: held-out, true label != y*, same set for both models
    cand = test_idx[y_cpu[test_idx] != y_target]
    asr_nodes = cand[torch.randperm(cand.numel(), generator=g)[:ecfg.max_asr_nodes]]

    out: Dict[str, float] = {
        "num_nodes": N, "num_classes": num_classes,
        "num_train": int(train_idx.numel()), "num_test": int(test_idx.numel()),
        "num_asr_nodes": int(asr_nodes.numel()),
    }

    # ---- 1. CLEAN fine-tune ------------------------------------------------
    m_c, clf_c = _fit(model, tg, y, train_idx, num_classes, ecfg, device)
    out["clean_acc_clean"] = _accuracy(_predict(m_c, clf_c, tg), y, test_idx)
    out["asr_clean_ft"] = _measure_asr(m_c, clf_c, tg, asr_nodes, all_deltas, ecfg,
                                       max_edges, nbr, y_target, device)
    del m_c, clf_c

    # ---- 2. BACKDOORED fine-tune ------------------------------------------
    pool = train_idx[y_cpu[train_idx] != y_target]
    k = min(pool.numel(), max(1, int(round(ecfg.ft_poison_rate * train_idx.numel()))))
    chosen = pool[torch.randperm(pool.numel(), generator=g)[:k]]
    out["num_poisoned_train"] = int(chosen.numel())

    g_p, _ = inject_triggers_batch(
        tg, chosen, pick_deltas(all_deltas, torch.arange(chosen.numel()),
                                ecfg.trigger_index), max_edges, nbr, device)
    y_p = g_p.y.clone()
    y_p[chosen.to(device)] = y_target                       # attacker's label
    m_b, clf_b = _fit(model, g_p, y_p, train_idx, num_classes, ecfg, device)
    del g_p, y_p

    out["clean_acc_backdoored"] = _accuracy(_predict(m_b, clf_b, tg), y, test_idx)
    out["asr"] = _measure_asr(m_b, clf_b, tg, asr_nodes, all_deltas, ecfg,
                              max_edges, nbr, y_target, device)
    if defense is not None and getattr(defense, "name", "none") != "none":
        out["asr_after_defense"] = _measure_asr(
            m_b, clf_b, tg, asr_nodes, all_deltas, ecfg, max_edges, nbr, y_target,
            device, defense=defense, artifacts=artifacts, defense_model=model)
    else:
        out["asr_after_defense"] = out["asr"]
    del m_b, clf_b
    return out
