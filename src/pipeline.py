
from __future__ import annotations

from typing import List

from .config import Config
from .utils.seed import set_seed
from .utils.device import (configure_gpus, primary_device, describe_devices)
from .utils.logging_utils import print_settings, dump_config
from .data.registry import availability_report
from .data.multi_domain import build_multi_domain
from .models.registry import build_model


def _pretrain_cache_name(cfg, mdg):
    """Key for the pretrained-weights checkpoint. Depends ONLY on what influences
    pretraining (model config, the train-data settings, the seed and the actual
    training graph) -- never on the attack/defense/eval settings -- so one
    checkpoint serves every attack variant that shares them."""
    from .utils.cache import hash_obj, fingerprint_graph
    d = cfg.data
    data_part = {k: getattr(d, k) for k in ("train_datasets", "align_dim",
                                            "align_method", "undirected",
                                            "add_self_loops")}
    key = hash_obj("pretrain", cfg.model, data_part, cfg.runtime.seed,
                   fingerprint_graph(mdg))
    return f"{cfg.model.name}_{cfg.model.backbone}_{key}"


def pretrain_encoder(model, mdg, cfg, device, verbose=True):
    import torch
    from .utils.cache import DiskCache
    model = model.to(device)
    mdg.to(device)

    # ---- reuse previously pretrained weights when available ----
    cache = DiskCache.from_cfg(cfg)
    cname = _pretrain_cache_name(cfg, mdg) if cache.enabled else None
    if cname is not None:
        payload = cache.load("pretrain", cname)
        if payload is not None:
            sd = payload.get("state_dict", {})
            ref = model.state_dict()
            compatible = (set(sd) == set(ref)
                          and all(tuple(sd[k].shape) == tuple(ref[k].shape)
                                  for k in ref))
            if compatible:
                model.load_state_dict(sd)
                model.eval()
                print(f"[cache] loaded pretrained {cfg.model.name}/"
                      f"{cfg.model.backbone} weights from "
                      f"{cache.path('pretrain', cname)} -- skipping "
                      f"{cfg.model.pretrain_epochs} pretraining epochs", flush=True)
                return model
            print("[cache] saved weights do not match this model; retraining",
                  flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.pretrain_lr,
                           weight_decay=cfg.model.weight_decay)
    from .utils.progress import pbar
    model.train()
    it = pbar(range(cfg.model.pretrain_epochs), desc=f"pretrain[{cfg.model.name}]")
    for ep in it:
        opt.zero_grad()
        loss = model.pretrain_loss(mdg)
        loss.backward()
        opt.step()
        if verbose and ep % max(1, cfg.model.pretrain_epochs // 5) == 0:
            print(f"  pretrain epoch {ep:4d}  loss={float(loss):.4f}")
    model.eval()

    if cname is not None:
        ok = cache.save("pretrain", cname, {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "meta": {"model": cfg.model.name, "backbone": cfg.model.backbone,
                     "epochs": cfg.model.pretrain_epochs,
                     "seed": cfg.runtime.seed},
        })
        if ok:
            print(f"[cache] saved pretrained weights -> "
                  f"{cache.path('pretrain', cname)}", flush=True)
    return model


def _load_test_graph(name, cfg, model_name):
    """A single test dataset as its own (single-domain) multi-domain graph so the
    encoder can embed it (coordinators/structure tokens handle the domain tag)."""
    from .data.multi_domain import build_multi_domain
    return build_multi_domain([name], cfg.data, model_name)


_METRIC_KEYS = ["asr", "clean_acc_backdoored", "clean_acc_clean", "asr_clean_ft",
                "asr_after_defense"]


def _fmt(v):
    return "  nan " if v != v else f"{v:.4f}"


def _print_dataset_block(res, cfg, artifacts):
    ecfg = cfg.eval
    tri = ("round-robin x%d" % len(artifacts.triggers) if ecfg.trigger_index < 0
           else f"#{ecfg.trigger_index}")
    sd = lambda k: (f" ±{res.std[k]:.4f}" if res.std.get(k) is not None
                    and res.num_runs > 1 else "")
    bar = "-" * 78
    print(f"\n{bar}\n[eval] {res.dataset}   (fine-tune={res.finetune_mode}, "
          f"runs={res.num_runs})\n{bar}")
    print(f"  nodes={res.num_nodes}  classes={res.num_classes}  train={res.num_train}  "
          f"test={res.num_test}")
    print(f"  backdoored fine-tune: {res.num_poisoned_train} poisoned train nodes "
          f"({100 * ecfg.ft_poison_rate:.1f}% of train), target class y*="
          f"{cfg.attack.target_label}, trigger={tri}")
    print(f"  ASR nodes (held-out, label != y*): {res.num_asr_nodes}")
    print(f"  >> Attack success rate (ASR)                : {_fmt(res.asr)}{sd('asr')}")
    print(f"  >> Clean ACC, backdoored fine-tune          : "
          f"{_fmt(res.clean_acc_backdoored)}{sd('clean_acc_backdoored')}")
    print(f"  >> Clean ACC, clean fine-tune               : "
          f"{_fmt(res.clean_acc_clean)}{sd('clean_acc_clean')}")
    print(f"     clean-accuracy drop (clean - backdoored)  : {_fmt(res.cad)}")
    print(f"     ASR, clean fine-tune (no downstream poison): "
          f"{_fmt(res.asr_clean_ft)}{sd('asr_clean_ft')}")
    print(f"     ASR after defense '{cfg.defense.name}'"
          f"{' ' * max(0, 21 - len(cfg.defense.name))}: "
          f"{_fmt(res.asr_after_defense)}{sd('asr_after_defense')}", flush=True)


def _print_summary_table(results, cfg):
    print("\n" + "=" * 100)
    print(f" SUMMARY  (model={cfg.model.name}/{cfg.model.backbone}, "
          f"fine-tune={cfg.eval.finetune_mode}, defense={cfg.defense.name})")
    print("=" * 100)
    hdr = (f"{'dataset':<12}{'status':<9}{'ASR':>8}{'CleanAcc(bd)':>14}"
           f"{'CleanAcc(clean)':>17}{'CAD':>8}{'ASR(cleanFT)':>14}"
           f"{'ASR@def':>9}{'R_spec':>9}")
    print(hdr)
    print("-" * 100)
    for r in results:
        print(f"{r.dataset:<12}{r.status:<9}{_fmt(r.asr):>8}"
              f"{_fmt(r.clean_acc_backdoored):>14}{_fmt(r.clean_acc_clean):>17}"
              f"{_fmt(r.cad):>8}{_fmt(r.asr_clean_ft):>14}"
              f"{_fmt(r.asr_after_defense):>9}{_fmt(r.r_spec_mean):>9}")
    print("=" * 100, flush=True)


def _write_dataset_json(out_dir, res):
    """Written as soon as a dataset finishes, so a later crash loses nothing."""
    import json, os
    d = os.path.join(out_dir, "datasets")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{res.dataset}.json"), "w") as fh:
        json.dump(res.__dict__, fh, indent=2, default=float)


def evaluate(model, cfg, device, artifacts, verbose=True, out_dir=None):
    """Per-test-dataset fine-tuning evaluation (see eval/finetune.py).

    For every dataset in cfg.data.test_datasets: fine-tune a CLEAN model and a
    BACKDOORED model (trigger-poisoned fine-tuning data), then report/record
    ASR, clean accuracy of the backdoored model and clean accuracy of the clean
    model. Failures are recorded per dataset (status/error) and never abort the
    remaining datasets."""
    import traceback
    import numpy as np
    import torch
    from .eval.finetune import run_once
    from .eval.metrics import EvalResult
    from .defenses.registry import build_defense

    defense = build_defense(cfg.defense)
    ecfg = cfg.eval
    results: List[EvalResult] = []
    r_spec_mean = (float(np.mean(artifacts.r_spec_final))
                   if artifacts.r_spec_final else float("nan"))

    for name in cfg.data.test_datasets:
        res = EvalResult(dataset=name, finetune_mode=ecfg.finetune_mode,
                         num_runs=ecfg.num_runs, r_spec_mean=r_spec_mean)
        tg = None
        try:
            tg = _load_test_graph(name, cfg, cfg.model.name).to(device)
        except FileNotFoundError as e:
            res.status, res.error = "skipped", str(e)
        except Exception as e:
            res.status, res.error = "skipped", f"load error: {e}"

        if tg is not None:
            try:
                runs = []
                for run in range(max(1, ecfg.num_runs)):
                    if verbose:
                        print(f"[eval] {name}: run {run + 1}/{ecfg.num_runs} "
                              f"(fine-tune={ecfg.finetune_mode}) ...", flush=True)
                    runs.append(run_once(model, tg, artifacts, cfg, device,
                                         seed=cfg.runtime.seed + run,
                                         defense=defense))
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                res.runs = runs
                for k in _METRIC_KEYS:
                    vals = np.array([r[k] for r in runs], dtype=float)
                    setattr(res, k, float(np.nanmean(vals)))
                    res.std[k] = float(np.nanstd(vals)) if len(vals) > 1 else 0.0
                for k in ("num_nodes", "num_classes", "num_train", "num_test",
                          "num_poisoned_train", "num_asr_nodes"):
                    setattr(res, k, int(runs[0][k]))
                res.clean_acc = res.clean_acc_clean              # back-compat alias
                res.cad = res.clean_acc_clean - res.clean_acc_backdoored
            except Exception as e:
                res.status = "error"
                res.error = f"{type(e).__name__}: {e}"
                if verbose:
                    print(traceback.format_exc(), flush=True)
            finally:
                del tg
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        results.append(res)
        if verbose:
            if res.status == "ok":
                _print_dataset_block(res, cfg, artifacts)
            else:
                print(f"\n[eval] {name:<12} {res.status.upper()}: {res.error}",
                      flush=True)
        if out_dir:
            _write_dataset_json(out_dir, res)

    if verbose:
        _print_summary_table(results, cfg)
    return results


def run_experiment(cfg: Config, verbose=True):
    # ---- 1. runtime ----
    gpu_str = configure_gpus(cfg.runtime.gpus)
    set_seed(cfg.runtime.seed)

    # torch imported only now (after CUDA_VISIBLE_DEVICES is set)
    import torch
    device = primary_device()

    print_settings(cfg, extra={
        "device_config": gpu_str,
        "devices": describe_devices(),
        "torch": torch.__version__,
    })
    out_dir = f"{cfg.runtime.output_dir}/{cfg.runtime.run_name}"
    dump_config(cfg, out_dir)

    # ---- 2. dataset availability ----
    all_names = list(dict.fromkeys(cfg.data.train_datasets + cfg.data.test_datasets))
    print(availability_report(all_names))

    # ---- 3. multi-domain pretraining graph ----
    if verbose:
        print(f"\n[data] building multi-domain train graph from "
              f"{cfg.data.train_datasets} via {cfg.model.name} method ...")
    mdg = build_multi_domain(cfg.data.train_datasets, cfg.data, cfg.model.name)
    num_domains = len(mdg.domain_names)
    if verbose:
        print(f"[data] combined: N={mdg.x.shape[0]}  F={mdg.x.shape[1]}  "
              f"E={mdg.edge_index.shape[1]}  domains={num_domains}  "
              f"coordinators={int(mdg.is_coordinator.sum())}")

    # ---- 4. build + pretrain victim GFM ----
    model = build_model(cfg.model, in_dim=mdg.x.shape[1], num_domains=num_domains)
    model = pretrain_encoder(model, mdg, cfg, device, verbose=verbose)

    # ---- 5. attack (on GPU) ----
    from .attack.sib2 import run_sib2
    if verbose:
        print(f"\n[attack] launching SIB2 on {describe_devices()} ...")
    artifacts = run_sib2(model, mdg, cfg, device, verbose=verbose,
                         use_influence_selection=True)

    # ---- 6. evaluation (+ defense) ----
    if verbose:
        print(f"\n[eval] evaluating on test datasets under defense="
              f"'{cfg.defense.name}' ...")
    results = evaluate(model, cfg, device, artifacts, verbose=verbose,
                       out_dir=out_dir)

    _save_results(cfg, out_dir, artifacts, results)
    return model, artifacts, results


def _save_results(cfg, out_dir, artifacts, results):
    import csv, json, os
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "settings": artifacts.settings,
        "target_indices": artifacts.target_indices,
        "r_spec_final": artifacts.r_spec_final,
        "davis_kahan": artifacts.davis_kahan,
        "eval_config": cfg.to_dict()["eval"],
        "defense": cfg.defense.name,
        "results": [r.__dict__ for r in results],
    }
    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    cols = ["dataset", "status", "asr", "clean_acc_backdoored", "clean_acc_clean",
            "cad", "asr_clean_ft", "asr_after_defense", "r_spec_mean",
            "num_nodes", "num_classes", "num_train", "num_test",
            "num_poisoned_train", "num_asr_nodes", "finetune_mode", "num_runs", "error"]
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            w.writerow([getattr(r, c) for c in cols])
    print(f"\n[done] wrote {path}\n[done] wrote {csv_path}\n"
          f"[done] per-dataset files in {os.path.join(out_dir, 'datasets')}")
