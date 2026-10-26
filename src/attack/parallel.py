"""
Optional multi-GPU execution of the per-target attack loop.

Targets are independent (each one gets its own spectrum + trigger optimisation),
so they shard cleanly across GPUs: one worker process per visible GPU, each with
its own copy of the (frozen) victim model and the graph. Nothing is split *inside*
a target — the memory fix for the spectral step is in spectral.py; this only buys
throughput (3 GPUs -> roughly 3x faster over the top-k targets).

Enable with `--parallel-targets` (or runtime.parallel_targets: true) and
`--gpus 0 1 2`. Requires >1 visible GPU; otherwise the sequential path is used.

Workers are started with the 'spawn' method, so everything handed to them must be
picklable. The victim models are classes defined inside builder functions (not
picklable by name), so we ship `state_dict()` + the config and rebuild the model
inside each worker.
"""
from __future__ import annotations

import traceback


def _worker(rank, cfg, state_dict, mdg_cpu, targets, e_target, U_k,
            infl_scores_full, verbose, out_q, cache=None, cache_names=None):
    try:
        import torch
        from ..models.registry import build_model
        from ..utils.seed import set_seed
        from .sib2 import optimise_target
        from .spectral import restrict_edges_to_real_nodes

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        set_seed(cfg.runtime.seed + rank)

        model = build_model(cfg.model, in_dim=mdg_cpu.x.shape[1],
                            num_domains=len(mdg_cpu.domain_names))
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        mdg = mdg_cpu.to(device)
        e_target = e_target.to(device)
        U_k = U_k.to(device) if U_k is not None else None
        infl = infl_scores_full.to(device) if infl_scores_full is not None else None

        exclude = (mdg.is_coordinator if cfg.attack.spectral.exclude_virtual_nodes
                   else None)
        real_ei = restrict_edges_to_real_nodes(mdg.edge_index, exclude)

        results = []
        for k, target in enumerate(targets):
            tag = f"[gpu{rank}]"
            print(f"\n{tag} === target {k+1}/{len(targets)}  node={target} ===",
                  flush=True)
            # progress bars / step logs only from worker 0 to keep logs readable
            res = optimise_target(model, mdg, target, cfg, e_target, U_k, infl,
                                  real_ei, device, verbose=(verbose and rank == 0),
                                  tag=tag, seed=cfg.runtime.seed,
                                  cache=cache,
                                  cache_name=(cache_names or {}).get(int(target)))
            print(f"{tag} finished node={target}  "
                  f"R_spec={res['r_spec_final']}", flush=True)
            results.append(res)
            torch.cuda.empty_cache()
        out_q.put(("ok", rank, results))
    except Exception:
        out_q.put(("error", rank, traceback.format_exc()))


def run_targets_parallel(model, mdg, cfg, targets, e_target, U_k,
                         infl_scores_full, verbose, cache=None, cache_names=None):
    """Shard `targets` round-robin over all visible GPUs. Returns the per-target
    result dicts in the ORIGINAL target order."""
    import torch
    import torch.multiprocessing as mp

    ndev = torch.cuda.device_count()
    shards = [targets[i::ndev] for i in range(ndev)]
    active = [(i, sh) for i, sh in enumerate(shards) if sh]
    print(f"[sib2] parallel attack: {len(targets)} targets over {len(active)} GPU(s): "
          + ", ".join(f"cuda:{i}->{len(sh)}" for i, sh in active), flush=True)

    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    mdg_cpu = mdg.__class__(**{**mdg.__dict__})                # new container, same tensors
    mdg_cpu.to("cpu")                                          # rebinds ITS attrs only
    e_cpu = e_target.detach().cpu()
    U_cpu = U_k.detach().cpu() if U_k is not None else None
    infl_cpu = (infl_scores_full.detach().cpu()
                if infl_scores_full is not None else None)

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = []
    for rank, sh in active:
        p = ctx.Process(target=_worker,
                        args=(rank, cfg, state_dict, mdg_cpu, sh, e_cpu, U_cpu,
                              infl_cpu, verbose, out_q, cache, cache_names))
        p.start()
        procs.append(p)

    by_target, errors = {}, []
    for _ in procs:
        status, rank, payload = out_q.get()          # blocks until a worker reports
        if status == "ok":
            for res in payload:
                by_target[res["target"]] = res
        else:
            errors.append(f"[gpu{rank}] worker failed:\n{payload}")
    for p in procs:
        p.join()
    if errors:
        raise RuntimeError("\n".join(errors))
    return [by_target[int(t)] for t in targets]
