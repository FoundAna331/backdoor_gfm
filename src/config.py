"""
Configuration system for SIB2-Backdoor.

A single dataclass tree holds every knob. Precedence (low -> high):
    dataclass defaults  <  YAML file  <  command-line overrides

This keeps experiments reproducible: the fully-resolved config is printed at
the start of every run (see utils.logging_utils.print_settings) and dumped to
the run directory.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# --------------------------------------------------------------------------- #
#  Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    # By default: pretrain on OGB-ArXiv, evaluate/transfer on everything else.
    train_datasets: List[str] = field(default_factory=lambda: ["ogbn-arxiv", "reddit", "computers"])
    test_datasets: List[str] = field(default_factory=lambda: [
        "cora", "citeseer", "photo", "flickr",
    ])
    root: str = "./datasets"
    # Common feature dimension that all domains are projected to (SVD align).
    # All three foundation models align features before combining domains.
    align_dim: int = 100
    align_method: str = "svd"           # {svd, pca, pad}
    undirected: bool = True
    add_self_loops: bool = False


@dataclass
class ModelConfig:
    name: str = "gcope"                 # {gcope, samgpt, gft}
    backbone: str = "gcn"               # {gcn, gat, fagcn}
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.5
    # GCOPE
    num_coordinators_per_domain: int = 1
    recon_weight: float = 0.2
    # SAMGPT
    num_structure_tokens: int = 4
    # GFT
    vocab_size: int = 128
    vq_commitment: float = 0.25
    # pretraining
    pretrain_epochs: int = 100
    pretrain_lr: float = 1e-4
    weight_decay: float = 5e-4


@dataclass
class InfluenceConfig:
    # Instruction (3): top-k instead of argmax, and a flag to include/exclude
    # node-to-node (v -> w) influence entirely.
    use_node_to_node_influence: bool = True   # False => fall back to self-influence-only scoring
    top_k_targets: int = 10                   # top-k selection replaces argmax
    damping: float = 0.01                     # lambda in G_lambda = G + lambda*I
    ihvp_method: str = "neumann"              # {neumann, lissa}
    neumann_terms: int = 50
    neumann_scale: float = 1.0
    lissa_depth: int = 50
    lissa_samples: int = 1
    self_influence_cap_percentile: float = 50.0   # tau_clean as a percentile of |I(v,v)|
    reference_sample_size: int = 256          # |D| proxy nodes for S(v)
    kfac_block_diagonal: bool = True          # per-layer K-FAC approximation


@dataclass
class SpectralConfig:
    k_hop: int = 2                            # receptive field of the encoder
    # low-pass: keep the lowest (n - m) frequencies; m = high-freq band size.
    high_freq_band_m: int = 0                 # 0 => derived from high_freq_fraction
    high_freq_fraction: float = 0.5           # fraction of spectrum treated as "high"
    alpha_spectral: float = 1.0               # weight of R_spec penalty
    hard_lowpass_projection: bool = True      # project delta onto low-freq span each step
    eig_max_nodes: int = 12000                # dense GPU eigh up to this many nodes
    eig_truncated_k: int = 512                # #eigs for truncated (sparse) path
    max_subgraph_nodes: int = 12000           # cap on the k-hop ball (0 = no cap); hubs get
                                              # a deterministic 1-hop-first random sample
    eig_method: str = "auto"                  # {auto, dense, scipy, lobpcg}
    eig_tol: float = 1e-4                     # LOBPCG tolerance (fallback only)
    exclude_virtual_nodes: bool = True        # ignore GCOPE coordinators in the local
                                              # Laplacian (they touch every node)


@dataclass
class OrthogonalConfig:
    top_k_subspace: int = 64                  # k for U_k (top-k pretraining-cov eigvecs)
    mu_persist: float = 1.0                   # weight of orthogonality penalty
    cov_sample_size: int = 2048               # #nodes to estimate Sigma


@dataclass
class TriggerConfig:
    trigger_size: int = 3                     # #trigger nodes (Gfm-Ba uses 3)
    max_edges: int = 6                        # |Delta A| budget tau
    feat_norm_budget: float = 1.0             # max ||delta||
    beta_feat: float = 0.1                    # weight of ||delta||^2
    gamma_clean: float = 1.0                  # weight of clean-loss-preservation term
    use_clean_loss_term: bool = True          # ablation switch
    margin_gamma: float = 0.05                # Proposition 2 margin (tau_def - gamma)
    opt_steps: int = 200
    opt_lr: float = 1e-2


@dataclass
class PretrainPoisonConfig:
    """Backdoor implantation into the PRETRAINED encoder (paper Sec 6.7).

    The trained triggers are attached to a `attack.poison_rate` fraction of the
    pretraining nodes and the victim encoder is trained so that every triggered
    node is embedded at a fixed attacker-chosen anchor, while embeddings of clean
    nodes are distilled to stay where they were (utility preservation). The
    downstream victim then fine-tunes / probes THIS encoder.
    """
    enabled: bool = True
    steps: int = 150                          # implantation optimisation steps
    lr: float = 1e-3                          # (pretrain_lr=1e-4 is too small to implant quickly)
    weight_decay: float = 0.0
    batch_nodes: int = 2048                   # max triggered nodes per step (also capped by
                                              # poison_rate * #real pretraining nodes)
    bd_weight: float = 1.0                    # triggered embedding -> anchor
    keep_weight: float = 1.0                  # clean embeddings -> original clean embeddings
    keep_nodes: int = 8192                    # clean nodes sampled per step for the keep term
    pretrain_weight: float = 0.0              # optional original self-supervised objective
    # Where triggered nodes are sent in embedding space.
    #   target_mean : mean embedding of label-y* pretraining nodes (paper's e_y*).
    #                 Sits INSIDE the clean cloud, so a linear head can hardly carve
    #                 it out of the surrounding classes downstream.
    #   orthogonal  : e_y* shifted along a direction orthogonal to the top-k
    #                 pretraining-covariance subspace U_k (Sec 4): clean data barely
    #                 uses that direction, so it is linearly separable and is not
    #                 overwritten by fine-tuning that moves inside U_k.
    anchor_mode: str = "orthogonal"
    anchor_scale: float = 3.0                 # shift length, in units of mean clean ||emb||


@dataclass
class AttackConfig:
    target_label: int = 0                     # y* (Scenario 2, target-controlled)
    scenario: int = 2                         # {1: uncontrolled, 2: controlled}
    poison_rate: float = 0.05                 # fraction of pretraining nodes poisoned (see pretrain_poison)
    influence: InfluenceConfig = field(default_factory=InfluenceConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    orthogonal: OrthogonalConfig = field(default_factory=OrthogonalConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    pretrain_poison: PretrainPoisonConfig = field(default_factory=PretrainPoisonConfig)


@dataclass
class DefenseConfig:
    name: str = "none"                        # {none, prune, dominant, rigbd, tcf}
    prune_threshold: float = 0.1              # cosine-sim edge prune cutoff
    rigbd_drop_rate: float = 0.2
    rigbd_num_rounds: int = 20
    dominant_alpha: float = 0.5               # struct vs attr reconstruction balance
    dominant_epochs: int = 100
    tcf_z_threshold: float = 2.0              # ego-density / spectral-moment z cutoff


@dataclass
class EvalConfig:
    """Downstream fine-tuning + metric protocol, run once per test dataset.

    For every test dataset three models/measurements are produced:
      * CLEAN fine-tune      : fine-tuned on the clean downstream training split.
      * BACKDOORED fine-tune : fine-tuned on the same split but with the trigger
                               attached to `ft_poison_rate` of the training nodes,
                               which are relabelled to the attacker's target class.
    and three headline metrics are reported:
      * asr                   : attack success rate of the backdoored fine-tune
                                (triggered test nodes whose true label != y*,
                                 fraction predicted y*)
      * clean_acc_backdoored  : accuracy on CLEAN test nodes of the backdoored model
      * clean_acc_clean       : accuracy on CLEAN test nodes of the clean model
    (plus asr_clean_ft = trigger transfer through the pretrained encoder alone,
     with NO downstream poisoning, and asr_after_defense.)
    """
    finetune_mode: str = "probe"         # {probe: frozen encoder + linear head,
                                         #  full: fine-tune encoder + head}
    ft_epochs: int = 200
    ft_lr: float = 1e-2                  # head lr (probe) / head lr (full)
    ft_encoder_lr: float = 1e-3          # encoder lr (full mode only)
    ft_weight_decay: float = 5e-4
    ft_poison_rate: float = 0.10         # fraction of TRAIN nodes triggered+relabelled
    train_frac: float = 0.6              # rest is the held-out test split
    max_asr_nodes: int = 200             # triggered test nodes used to measure ASR
    asr_batch_size: int = 50             # trigger instances injected per forward pass
                                         # (1 = fully isolated, slowest)
    trigger_index: int = -1              # -1: round-robin over all trained triggers;
                                         # >=0: always use that trigger
    num_runs: int = 1                    # repeated random splits -> mean/std


@dataclass
class RuntimeConfig:
    gpus: List[int] = field(default_factory=lambda: [0])   # multi-GPU selection
    seed: int = 42
    output_dir: str = "./runs"
    run_name: str = "sib2_run"
    num_workers: int = 2
    verbose: bool = True
    parallel_targets: bool = False         # shard attack targets over all visible GPUs
    use_cache: bool = True                 # reuse saved pretrained weights / finished
                                           # attack targets (see src/utils/cache.py)
    cache_dir: str = "./cache"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # --------------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Loading / merging
# --------------------------------------------------------------------------- #
def _update_dataclass(obj: Any, updates: Dict[str, Any]) -> Any:
    """Recursively overlay a (possibly nested) dict onto a dataclass instance."""
    if not dataclasses.is_dataclass(obj):
        return updates
    field_types = {f.name: f.type for f in dataclasses.fields(obj)}
    for key, val in updates.items():
        if key not in field_types:
            raise KeyError(f"Unknown config key: {key!r} on {type(obj).__name__}")
        cur = getattr(obj, key)
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            _update_dataclass(cur, val)
        else:
            setattr(obj, key, val)
    return obj


def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed; `pip install pyyaml`.")
    with open(path, "r") as fh:
        return yaml.safe_load(fh) or {}


def build_config(yaml_path: Optional[str] = None,
                 overrides: Optional[Dict[str, Any]] = None) -> Config:
    cfg = Config()
    if yaml_path:
        _update_dataclass(cfg, load_yaml(yaml_path))
    if overrides:
        _update_dataclass(cfg, overrides)
    return cfg


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """A curated set of the most-used knobs as flat CLI flags.

    Anything not exposed here can still be set through --config yaml or
    --set dotted.key=value.
    """
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file.")
    parser.add_argument("--set", nargs="*", default=[],
                        help="Ad-hoc overrides, e.g. --set attack.poison_rate=0.1 model.name=gft")

    # common shortcuts
    parser.add_argument("--model", type=str, choices=["gcope", "samgpt", "gft"])
    parser.add_argument("--backbone", type=str, choices=["gcn", "gat", "fagcn"])
    parser.add_argument("--train-datasets", nargs="+")
    parser.add_argument("--test-datasets", nargs="+")
    parser.add_argument("--defense", type=str,
                        choices=["none", "prune", "dominant", "rigbd", "tcf"])
    parser.add_argument("--gpus", nargs="+", type=int,
                        help="GPU ids, e.g. --gpus 0 1 2")
    parser.add_argument("--parallel-targets", action="store_true", default=None,
                        help="shard attack targets over all visible GPUs")
    parser.add_argument("--finetune-mode", type=str, choices=["probe", "full"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-cache", action="store_true", default=None,
                        help="do not read or write saved weights / attack results")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="where pretrained weights / attack results are cached")
    parser.add_argument("--run-name", type=str)

    # attack shortcuts
    parser.add_argument("--poison-rate", type=float)
    parser.add_argument("--trigger-size", type=int)
    parser.add_argument("--top-k-targets", type=int)
    parser.add_argument("--use-node-to-node-influence", type=lambda s: s.lower() == "true",
                        help="true/false: include node-to-node (v->w) influence")
    parser.add_argument("--target-label", type=int)
    parser.add_argument("--scenario", type=int, choices=[1, 2])
    return parser


def _coerce(value: str) -> Any:
    """Turn a CLI string into bool/int/float/list when it looks like one."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if "," in value:
        return [_coerce(v) for v in value.split(",")]
    return value


def _nest(dotted: str, value: Any) -> Dict[str, Any]:
    keys = dotted.split(".")
    out: Dict[str, Any] = {}
    cur = out
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
    return out


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def config_from_args(args: argparse.Namespace) -> Config:
    overrides: Dict[str, Any] = {}

    # flat shortcuts -> dotted paths
    shortcut_map = {
        "model": "model.name",
        "backbone": "model.backbone",
        "train_datasets": "data.train_datasets",
        "test_datasets": "data.test_datasets",
        "defense": "defense.name",
        "gpus": "runtime.gpus",
        "seed": "runtime.seed",
        "parallel_targets": "runtime.parallel_targets",
        "finetune_mode": "eval.finetune_mode",
        "run_name": "runtime.run_name",
        "cache_dir": "runtime.cache_dir",
        "poison_rate": "attack.poison_rate",
        "trigger_size": "attack.trigger.trigger_size",
        "top_k_targets": "attack.influence.top_k_targets",
        "use_node_to_node_influence": "attack.influence.use_node_to_node_influence",
        "target_label": "attack.target_label",
        "scenario": "attack.scenario",
    }
    for attr, dotted in shortcut_map.items():
        val = getattr(args, attr, None)
        if val is not None:
            _deep_merge(overrides, _nest(dotted, val))

    if getattr(args, "no_cache", None):
        _deep_merge(overrides, _nest("runtime.use_cache", False))

    # generic --set key=value
    for item in getattr(args, "set", []) or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        _deep_merge(overrides, _nest(key, _coerce(raw)))

    return build_config(getattr(args, "config", None), overrides)
