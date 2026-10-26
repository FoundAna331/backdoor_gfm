
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any

from .influence import InfluenceSelector, InfluenceResult, select_targets_random
from .spectral import (compute_spectrum, build_induced_signal,
                       lowpass_project_matrix, positions_in_subgraph,
                       restrict_edges_to_real_nodes)
from .orthogonal import pretraining_covariance_subspace, davis_kahan_bound
from .trigger import build_trigger, greedy_edge_selection, inject_trigger
from .losses import compute_joint_loss


def _lowpass_correction(trigger, target_node, hosts, spec, device):
    """In-place spectral-PGD step: low-pass the trigger's induced subgraph signal
    and pull each trigger node's features toward the projected (smooth) values at
    its attachment point. Approximate but genuine enforcement of the §3.3 hard
    low-pass constraint at the feature level."""
    import torch
    with torch.no_grad():
        sig = build_induced_signal(trigger.delta, target_node, hosts, spec, device)
        sig_lp = lowpass_project_matrix(sig, spec)      # [n, F]
        host_t = torch.as_tensor(hosts, device=device).view(-1)
        hp, hv = positions_in_subgraph(spec, host_t)
        hp, hv = hp.tolist(), hv.tolist()
        t = trigger.delta.shape[0]
        # pull back each trigger row toward its attachment point's low-pass value
        # (sequential on purpose: rows are revisited when hosts > trigger nodes)
        for i in range(len(hp)):
            if hv[i]:
                row = i % t
                trigger.delta[row].mul_(0.5).add_(0.5 * sig_lp[hp[i]])


@dataclass
class AttackArtifacts:
    target_indices: List[int]
    triggers: list = field(default_factory=list)       # list of Trigger modules
    host_edges: list = field(default_factory=list)
    trig_ids: list = field(default_factory=list)
    U_k: Any = None
    influence: Any = None
    r_spec_final: List[float] = field(default_factory=list)
    davis_kahan: float = float("nan")
    loss_history: List[dict] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    implant_report: Any = None


def _target_embedding(model, mdg, cfg_attack, device):
    """e_y*: mean embedding of nodes with label y* (Scenario 2), or a fixed
    arbitrary cluster centroid (Scenario 1)."""
    import torch
    with torch.no_grad():
        emb = model.encode(mdg.x, mdg.edge_index,
                           is_coordinator=getattr(mdg, "is_coordinator", None),
                           domain_id=getattr(mdg, "domain_id", None))
    if cfg_attack.scenario == 2 and getattr(mdg, "y", None) is not None:
        mask = mdg.y == cfg_attack.target_label
        if mask.sum() > 0:
            return emb[mask].mean(dim=0).detach()
    # Scenario 1: pick a fixed cluster (first k-means-ish centroid = global mean shift)
    return emb.mean(dim=0).detach()


def optimise_target(model, mdg, target, cfg, e_target, U_k, infl_scores_full,
                    real_ei, device, verbose=True, tag="", seed=0,
                    cache=None, cache_name=None):
    """Spectrum + trigger optimisation for ONE target node. Returns a dict of CPU
    tensors / floats so it can be produced in a worker process on another GPU.

    If `cache`/`cache_name` are given, the finished result is written to disk
    immediately (so an interrupted run can resume at the next target)."""
    import torch
    from ..utils.seed import set_seed
    cfg_attack = cfg.attack
    # Per-target seed: the result for a target is the same whether earlier targets
    # were computed in this process, loaded from cache, or run on another GPU.
    set_seed((int(seed) * 1000003 + int(target)) % (2 ** 31 - 1))
    num_nodes = mdg.x.shape[0]

    spec = compute_spectrum(target, real_ei, num_nodes, cfg_attack.spectral,
                            verbose=verbose, seed=seed)

    trigger = build_trigger(mdg.x.shape[1], cfg_attack.trigger.trigger_size, device)
    hosts = greedy_edge_selection(target, mdg.edge_index, num_nodes,
                                  cfg_attack.trigger.max_edges,
                                  influence_scores=infl_scores_full,
                                  exclude_mask=getattr(mdg, "is_coordinator", None))
    opt = torch.optim.Adam(trigger.parameters(), lr=cfg_attack.trigger.opt_lr)

    history, last_bd, trig_ids = [], None, None
    for step in range(cfg_attack.trigger.opt_steps):
        opt.zero_grad()
        mdg_trig, trig_ids = inject_trigger(mdg, target, trigger, hosts, device)
        loss, bd = compute_joint_loss(
            model=model, mdg_clean=mdg, mdg_triggered=mdg_trig,
            trig_ids=trig_ids, target_node=target,
            target_embedding=e_target, spec=spec, trigger=trigger,
            hosts=hosts, U_k=U_k, cfg_attack=cfg_attack, device=device)
        loss.backward()
        opt.step()

        # hard low-pass projection (spectral PGD, Section 3.3)
        if (cfg_attack.spectral.hard_lowpass_projection
                and cfg_attack.spectral.alpha_spectral > 0):
            _lowpass_correction(trigger, target, hosts, spec, device)
        trigger.clamp_norm(cfg_attack.trigger.feat_norm_budget)

        last_bd = bd
        # Drop this step's autograd graph / triggered-graph copy before the next
        # forward so the caching allocator can reuse the big [E,H] buffers instead of
        # growing (full-graph GNN passes are the dominant memory cost here).
        del loss, mdg_trig
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if verbose and (step % max(1, cfg_attack.trigger.opt_steps // 5) == 0):
            print(f"{tag}  step {step:4d}  total={bd.total:.4f}  eff={bd.eff:.4f}  "
                  f"R_spec={bd.r_spec_value:.4f}  persist={bd.persist:.4f}  "
                  f"clean={bd.clean:.4f}", flush=True)
        history.append({"target": target, "step": step, **bd.__dict__})

    r_final = last_bd.r_spec_value if last_bd is not None else None
    if verbose and last_bd is not None:
        print(f"{tag}  final R_spec={r_final:.4f} "
              f"(target margin gamma={cfg_attack.trigger.margin_gamma})", flush=True)
    del spec
    result = {
        "target": int(target),
        "delta": trigger.delta.detach().cpu(),
        "hosts": hosts.detach().cpu(),
        "trig_ids": trig_ids.detach().cpu() if trig_ids is not None else None,
        "r_spec_final": r_final,
        "loss_history": history,
    }
    if cache is not None and cache_name:
        if cache.save("attack", cache_name, result) and verbose:
            print(f"{tag}  [cache] saved result for target {target}", flush=True)
    return result


def _run_all_targets(model, mdg, cfg, targets, e_target, U_k, infl_scores_full,
                     device, verbose, cache=None, cache_names=None):
    """Sequential by default; sharded over all visible GPUs when
    runtime.parallel_targets is set (targets are independent).

    Targets whose finished result is already in the cache are loaded instead of
    recomputed; every newly finished target is cached straight away."""
    import torch
    cfg_spec = cfg.attack.spectral
    cache_names = cache_names or {}

    def cached(t):
        name = cache_names.get(int(t))
        if cache is None or not name:
            return None
        return cache.load("attack", name)

    use_parallel = (getattr(cfg.runtime, "parallel_targets", False)
                    and torch.cuda.is_available()
                    and torch.cuda.device_count() > 1 and len(targets) > 1)
    par = {}
    if use_parallel:
        todo = [t for t in targets
                if not (cache is not None and cache_names.get(int(t))
                        and cache.has("attack", cache_names[int(t)]))]
        if len(todo) > 1:
            from .parallel import run_targets_parallel
            res_list = run_targets_parallel(model, mdg, cfg, todo, e_target, U_k,
                                            infl_scores_full, verbose,
                                            cache=cache, cache_names=cache_names)
            par = {int(r["target"]): r for r in res_list}
        # (0 or 1 target left to compute -> handled by the sequential loop below)

    exclude = getattr(mdg, "is_coordinator", None) if cfg_spec.exclude_virtual_nodes else None
    real_ei = None                      # built lazily: not needed if all cached
    out = []
    for ti, target in enumerate(targets):
        if int(target) in par:
            out.append(par[int(target)])
            continue
        hit = cached(target)
        if hit is not None:
            if verbose:
                print(f"\n[sib2] === target {ti+1}/{len(targets)}  node={target} "
                      f"=== loaded finished result from cache (skipping "
                      f"optimisation)", flush=True)
            out.append(hit)
            continue
        if real_ei is None:
            real_ei = restrict_edges_to_real_nodes(mdg.edge_index, exclude)
        if verbose:
            print(f"\n[sib2] === target {ti+1}/{len(targets)}  node={target} ===",
                  flush=True)
        out.append(optimise_target(model, mdg, target, cfg, e_target, U_k,
                                   infl_scores_full, real_ei, device,
                                   verbose=verbose, seed=cfg.runtime.seed,
                                   cache=cache,
                                   cache_name=cache_names.get(int(target))))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def run_sib2(model, mdg, cfg, device, verbose=True,
             use_influence_selection=True):
    """Execute the full attack. `use_influence_selection=False` is the ablation
    'no influence-based selection at all' (random hosts)."""
    import torch

    cfg_attack = cfg.attack
    settings = {
        "model": cfg.model.name,
        "backbone": cfg.model.backbone,
        "scenario": cfg_attack.scenario,
        "target_label": cfg_attack.target_label,
        "poison_rate": cfg_attack.poison_rate,
        "trigger_size": cfg_attack.trigger.trigger_size,
        "max_edges": cfg_attack.trigger.max_edges,
        "alpha_spectral": cfg_attack.spectral.alpha_spectral,
        "mu_persist": cfg_attack.orthogonal.mu_persist,
        "use_clean_loss_term": cfg_attack.trigger.use_clean_loss_term,
        "use_node_to_node_influence": cfg_attack.influence.use_node_to_node_influence,
        "top_k_targets": cfg_attack.influence.top_k_targets,
        "use_influence_selection": use_influence_selection,
        "margin_gamma": cfg_attack.trigger.margin_gamma,
    }

    # ---- saved-work cache (pretrained weights are handled in pipeline.py) ----
    from ..utils.seed import set_seed
    from ..utils.cache import (DiskCache, fingerprint_model, fingerprint_graph,
                               hash_obj)
    cache = DiskCache.from_cfg(cfg)
    base_key = None
    if cache.enabled:
        base_key = hash_obj("attack-base", fingerprint_model(model),
                            fingerprint_graph(mdg), cfg.runtime.seed)

    # ---- 2+3. target/host selection ----
    # Each stage re-seeds, so its result does not depend on whether an earlier
    # stage was computed or loaded from the cache.
    set_seed(cfg.runtime.seed)
    influence_result = None
    if use_influence_selection:
        infl_name = (f"influence_{hash_obj('influence', base_key, cfg_attack.influence)}"
                     if cache.enabled else None)
        payload = cache.load("attack", infl_name) if infl_name else None
        if payload is not None:
            influence_result = InfluenceResult(**payload)
            if verbose:
                print(f"[cache] loaded influence-based target selection "
                      f"({len(influence_result.target_indices)} targets) from "
                      f"{cache.path('attack', infl_name)}", flush=True)
        else:
            selector = InfluenceSelector(model, mdg, cfg_attack.influence, device,
                                         verbose=verbose)
            influence_result = selector.select()
            if infl_name:
                cache.save("attack", infl_name, dict(vars(influence_result)))
        targets = influence_result.target_indices
    else:
        targets = select_targets_random(mdg, cfg_attack.influence.top_k_targets,
                                        device, seed=cfg.runtime.seed)
        if verbose:
            print(f"[sib2] ABLATION: random host selection -> {len(targets)} targets")

    # ---- 4. orthogonality proxy subspace U_k ----
    set_seed(cfg.runtime.seed + 1)
    U_k, sigma_eigs = (None, None)
    if cfg_attack.orthogonal.mu_persist > 0:
        U_k, sigma_eigs = pretraining_covariance_subspace(
            model, mdg, cfg_attack.orthogonal.top_k_subspace,
            cfg_attack.orthogonal.cov_sample_size, device, verbose=verbose)

    e_target = _target_embedding(model, mdg, cfg_attack, device)

    art = AttackArtifacts(target_indices=targets, U_k=U_k,
                          influence=influence_result, settings=settings)

    num_nodes = mdg.x.shape[0]
    infl_scores_full = None
    if influence_result is not None:
        # dense per-node self-influence vector for edge ranking (optional)
        infl_scores_full = torch.zeros(num_nodes)
        for idx, v in enumerate(influence_result.candidate_indices):
            infl_scores_full[v] = influence_result.self_influence[idx]
        infl_scores_full = infl_scores_full.to(device)

    # ---- 5+6. per-target spectrum + joint optimisation ----
    # The local Laplacian is built on REAL nodes only (see spectral.py): a GCOPE
    # coordinator touches every node, which used to make the "2-hop neighbourhood"
    # the entire graph (-> a dense n x n matrix, ~107 GiB for ogbn-arxiv).
    # A target's result depends on the model weights, the graph, the seed, the
    # WHOLE attack config and whether targets came from influence selection; the
    # key covers all of these, so any changed knob is a cache miss.
    cache_names = {}
    if cache.enabled:
        for t in targets:
            cache_names[int(t)] = "target_" + hash_obj(
                "target", base_key, cfg_attack, bool(use_influence_selection), int(t))
    results = _run_all_targets(model, mdg, cfg, targets, e_target, U_k,
                               infl_scores_full, device, verbose,
                               cache=cache, cache_names=cache_names)
    for res in results:
        trigger = build_trigger(mdg.x.shape[1], cfg_attack.trigger.trigger_size,
                                device)
        with torch.no_grad():
            trigger.delta.copy_(res["delta"].to(device))
        art.triggers.append(trigger)
        art.host_edges.append(res["hosts"])
        art.trig_ids.append(res["trig_ids"])
        art.loss_history.extend(res["loss_history"])
        if res["r_spec_final"] is not None:
            art.r_spec_final.append(res["r_spec_final"])

    # ---- Davis-Kahan diagnostic (Section 4.5) ----
    if sigma_eigs is not None:
        ev_desc = sigma_eigs.flip(0)
        art.davis_kahan = davis_kahan_bound(
            ev_desc, cfg_attack.orthogonal.top_k_subspace, perturbation_norm=1.0)
        if verbose:
            print(f"\n[sib2] Davis-Kahan sin-theta bound (||E||=1 proxy) = "
                  f"{art.davis_kahan:.4e}")

    # ---- 7. implant the optimised triggers into the PRETRAINED encoder ----
    # (Sec 6.7). Without this the attack only ever measured transfer of a
    # trigger optimised against frozen weights; this is the step that actually
    # backdoors the encoder the downstream fine-tuning protocol then probes.
    if cfg_attack.pretrain_poison.enabled and art.triggers:
        from .implant import implant_backdoor
        if verbose:
            print(f"\n[sib2] implanting backdoor into pretrained encoder "
                  f"({cfg_attack.pretrain_poison.steps} steps) ...", flush=True)
        art.implant_report = implant_backdoor(
            model, mdg, art, e_target, U_k, cfg, device, verbose=verbose)

    return art
