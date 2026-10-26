# SIB²-Backdoor — research implementation

Reference implementation of **SIB²-Backdoor** (Spectral-Influence Backdoor with
Orthogonal-Subspace Anchoring), the *proposed method only* from the accompanying
document, as a benchmark harness against graph foundation models (GFMs).

The three mechanisms from the paper:

1. **Influence target/host selection** (§2) — damped Fisher/Gauss-Newton
   curvature, inverse-HVP via Neumann/LiSSA, self-influence stealth cap, and a
   global influence-propagation score `S(v)`. *Top-k selection* (not argmax) and
   a *flag to include/exclude node-to-node `I(v→w)` influence*.
2. **Spectral low-frequency anchoring** (§3) — Laplacian eigendecomposition of
   the target's k-hop neighbourhood (with a progress bar), the high-frequency
   energy ratio `R_spec`, a hard low-pass projection (spectral PGD), and the
   Proposition-2 margin.
3. **Orthogonal-subspace persistence** (§4) — top-k pretraining-covariance
   subspace `U_k`, the orthogonality penalty, and the Davis–Kahan diagnostic.

> ⚠️ **Responsible use.** This is adversarial-ML *research* code for studying
> backdoor robustness of graph foundation models and for stress-testing the
> published defenses it evaluates against (Prune, DOMINANT, RIGBD, TCF). Run it
> only on public benchmark datasets and models you are authorized to test. Do not
> poison or distribute third-party models. The point of an attack implementation
> here is the same as in the defense papers it cites: measure, then defend.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # install the torch / PyG build matching your CUDA
```

## Quick start

```bash
# tiny CPU smoke test (verifies the pipeline end-to-end)
bash scripts/smoke_test.sh

# full run: one model at a time (instruction 1)
bash scripts/run_attack.sh gcope none "0"      # model, defense, GPUs
bash scripts/run_attack.sh samgpt tcf "0 1"
bash scripts/run_attack.sh gft rigbd "0"

# every defense, one at a time (instruction 5)
bash scripts/run_defense_sweep.sh gcope "0"

# ablations (instruction 4)
bash scripts/run_ablation.sh gcope "0" all
```

Direct CLI:

```bash
python -m src.main --model gft --defense tcf --gpus 0 1 \
  --use-node-to-node-influence false --top-k-targets 5 \
  --set attack.poison_rate=0.1 attack.spectral.alpha_spectral=2.0
```

## Saved weights & resumable runs (cache)

Pretraining and the attack are expensive, so finished work is cached on disk
(`./cache` by default; see `src/utils/cache.py`):

| What | When it is reused |
|---|---|
| Pretrained victim weights | any run with the same model/backbone settings, train data and seed -- attack, defense and ablation settings do **not** matter, because pretraining happens before the attack |
| Influence-based target selection | same weights, graph, seed and `attack.influence.*` |
| Each attacked target (trigger, host edges, loss history) | same weights, graph, seed and the *entire* `attack.*` config. Saved the moment a target finishes, so a crashed/interrupted run resumes at the next target |

Every entry is keyed by content hashes (model-weight and graph fingerprints plus
the config), so changing a setting is simply a cache miss -- it can never silently
reuse a stale result. Defense sweeps (`run_defense_sweep.sh`) therefore pretrain and
attack once and only repeat evaluation.

```bash
python -m src.main ... --no-cache            # ignore and don't write the cache
python -m src.main ... --cache-dir /scratch/sib2_cache
rm -rf cache                                 # start fresh
```

Each attack stage and each target re-seeds deterministically, so results are the
same whether earlier stages ran or were loaded from the cache.

## Memory model & multi-GPU

**Spectral step (was the OOM).** For each target the attack eigendecomposes the
Laplacian of its k-hop neighbourhood. Two things used to blow this up on GCOPE:
(1) the coordinator node is wired to *every* real node, so the "2-hop
neighbourhood" of any target was the whole graph (n = 169,344 on ogbn-arxiv ->
a dense n x n matrix = 107 GiB); (2) the Laplacian was always built dense.
Now: the local Laplacian is built **sparsely, on real nodes only**
(`attack.spectral.exclude_virtual_nodes`), the ball is capped at
`max_subgraph_nodes` (hubs get a deterministic 1-hop-first sample), and anything
up to `eig_max_nodes` (12,000) uses an exact dense `eigh` **on the GPU**
(~1 GB). Peak memory is therefore independent of dataset size. Only if you raise
the cap above `eig_max_nodes` does the truncated sparse solver kick in
(`eig_method: auto|dense|scipy|lobpcg`; CPU shift-invert Lanczos can take minutes
for n >~ 20k).

**Multiple GPUs.** One GPU is enough now. To also gain throughput, shard the
(independent) attack targets over all GPUs:

```bash
python -m src.main --model gcope --gpus 0 1 2 --parallel-targets
```

Each GPU gets a copy of the frozen victim model and handles every 3rd target.
Evaluation datasets run sequentially.

## Backdoor implantation into the pretrained encoder (Sec 6.7)

Steps 1-6 of the pipeline (influence selection, spectral anchoring,
orthogonal-subspace persistence, joint-loss PGD) only ever *optimised the
trigger* against a **frozen** encoder — the encoder's own weights were never
touched, so every downstream number measured how well a frozen-encoder
trigger happens to transfer, not an actually backdoored encoder. `src/attack/
implant.py` (`attack.pretrain_poison` in the config) closes that gap: after
the triggers are trained, it fine-tunes the pretrained encoder itself so that

* triggered pretraining nodes (the influence-selected targets, topped up to
  `attack.poison_rate` of real pretraining nodes) are pulled toward a fixed
  representation-space anchor (`L_bd`), and
* clean embeddings are distilled against a frozen pre-implant snapshot so
  clean accuracy is preserved (`L_keep`).

`attack.pretrain_poison.anchor_mode: orthogonal` (default) places that anchor
along the component of e_y* that is orthogonal to the top-k pretraining-
covariance subspace `U_k` already computed for the persistence term (Sec 4):
a direction clean data barely uses, so a downstream linear probe separates it
from a handful of poisoned fine-tuning examples, and encoder/head updates
that mostly move within `U_k` (ordinary fine-tuning) don't erode it. Knobs:
`attack.pretrain_poison.{enabled,steps,lr,batch_nodes,bd_weight,keep_weight,
keep_nodes,anchor_mode,anchor_scale}`. This step is not disk-cached (it's
cheap relative to the attack/influence stages); it reruns from the cached
clean pretrained weights every time `run_sib2` is called, including once per
defense in `run_defense_sweep.sh` and once per ablation *variant* (the
ablation runner resets the encoder to its clean snapshot before each variant
so implantations never stack across variants).

## Per-dataset fine-tuning results (printed + recorded)

For every dataset in `data.test_datasets` the harness fine-tunes a **clean**
model and a **backdoored** model (trigger attached to `eval.ft_poison_rate` of
the training nodes, relabelled to `attack.target_label`) and reports

| metric | meaning |
|---|---|
| `asr` | triggered held-out nodes (true label != y*) predicted as y*, backdoored model |
| `clean_acc_backdoored` | clean held-out accuracy of the backdoored model |
| `clean_acc_clean` | clean held-out accuracy of the clean model |
| `cad` | `clean_acc_clean - clean_acc_backdoored` |
| `asr_clean_ft` | ASR of the clean model (trigger transfer through the encoder alone) |
| `asr_after_defense` | `asr` after `--defense` purifies the triggered graph |

Outputs under `runs/<run_name>/`: `datasets/<name>.json` (written the moment a
dataset finishes), `metrics.csv`, `results.json`; a summary table is printed at
the end. Knobs: `--finetune-mode {probe,full}` (frozen encoder + linear head vs.
full fine-tuning), `--set eval.num_runs=5` (mean/std over random splits),
`eval.ft_poison_rate`, `eval.max_asr_nodes`, `eval.asr_batch_size`,
`eval.trigger_index`.

## Models (instruction 1)

Select one per run via `--model {gcope,samgpt,gft}`. Each wraps the model's
*original* cross-domain combination method:

| Model  | Cross-domain unification                   | Official repo |
|--------|--------------------------------------------|---------------|
| GCOPE  | virtual **coordinator** nodes + inter-dataset edges | https://github.com/cshhzhao/GCOPE |
| SAMGPT | per-domain **structure tokens** in each GNN layer   | TheWebConf'25 (smufang) |
| GFT    | shared **tree vocabulary** (VQ) + tree reconstruction | https://github.com/Zehong-Wang/GFT |

These are faithful, self-contained re-implementations of each model's
distinctive mechanism on a shared backbone — enough to run the attack
end-to-end. For paper-exact numbers, drop the official repo code behind the same
interface at the points marked `# >>> OFFICIAL-REPO HOOK`.

## Datasets (instruction 2)

Default: **train/pretrain on OGB-ArXiv**, **test on the rest**. Availability is
printed at startup.

| Dataset           | Status | Loader |
|-------------------|--------|--------|
| Cora, CiteSeer    | AUTO   | PyG `Planetoid` |
| OGB-ArXiv         | AUTO   | `ogb` `PygNodePropPredDataset` |
| Amazon Computers  | AUTO   | PyG `Amazon` |
| Amazon Photos     | AUTO   | PyG `Amazon` |
| Reddit            | AUTO   | PyG `Reddit2` |
| Flickr            | AUTO   | PyG `Flickr` |
| Facebook          | AUTO   | PyG `FacebookPagePage` |
| Elliptic (base)   | AUTO   | PyG `EllipticBitcoinDataset` |
| **Elliptic++**    | MANUAL | not in PyG — place `datasets/manual/elliptic++.pt` |
| **Twitter**       | MANUAL | no canonical PyG node dataset — place `datasets/manual/twitter.pt` |

For the two MANUAL datasets, save a `torch_geometric.data.Data(x, edge_index, y)`
at the path shown; the loader picks it up automatically.

## Defenses (instruction 5)

`--defense {none,prune,dominant,rigbd,tcf}`, applied one at a time. `tcf` is the
spectral defense Proposition 2 targets; the harness reports the trigger's
measured `R_spec` alongside ASR-after-defense.

## Ablations (instruction 4)

`--ablation {trigger_size,poison_rate,influence_vs_random,no_spectral,
no_orthogonality,clean_loss_term,margin,all}` (shares one pretrained encoder
across variants).

## Repo layout

```
configs/      YAML configs (default + per-model + smoke)
scripts/      run_attack / run_defense_sweep / run_ablation / smoke_test
src/
  config.py           dataclass config + YAML + CLI merge
  main.py             entry point
  pipeline.py         data -> pretrain -> attack -> defense -> eval
  utils/              device (multi-GPU), seed, progress bar, settings printer
  data/               registry (availability), loaders, feature align, multi-domain
  models/             base, backbones, gcope, samgpt, gft, registry
  attack/             influence, spectral, orthogonal, trigger, losses, sib2, parallel
  defenses/           none, prune, dominant, rigbd, tcf, registry
  eval/               fine-tuning protocol (finetune.py), probe, metrics
  ablation/           runner
tests/        core-math verifier (NumPy) + pytest suite
```

## Verify the core math (no torch needed)

```bash
python tests/verify_core_math.py       # Parseval, low-pass, orthogonality, Neumann iHVP
pytest -q                              # torch/PyG mirrors + influence shape tests
```

## Paper → code map

| Paper | Code |
|-------|------|
| §2.2 `G_λ`, `I(v→w)`         | `attack/influence.py` (`FisherOperator`, `propagation_score`) |
| §2.3 self-influence cap `C`  | `attack/influence.py` (`self_influence`, `select`) |
| §2.4 `S(v)` + §2.5 Prop 1    | `attack/influence.py` (`InfluenceSelector`) |
| §2.6 Neumann/LiSSA, K-FAC    | `attack/influence.py` (`ihvp_*`, `_param_groups`) |
| §3.2 `R_spec`, Parseval      | `attack/spectral.py` |
| §3.3/Prop 2 low-pass PGD     | `attack/spectral.py` (`lowpass_project`), `attack/sib2.py` |
| §4.2 `U_k`, §4.3 L_persist   | `attack/orthogonal.py`, `attack/losses.py` |
| §4.5 Davis–Kahan            | `attack/orthogonal.py` (`davis_kahan_bound`) |
| §5 full loss                 | `attack/losses.py` (`compute_joint_loss`) |
| §6 algorithm                 | `attack/sib2.py` (`run_sib2`) |
| §9 metrics/defenses          | `eval/`, `defenses/` |
```
