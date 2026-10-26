"""
Evaluation metrics (paper Section 9.1):
  ASR   — fraction of triggered nodes predicted as y*
  ACC   — clean accuracy of the (backdoored) model
  CAD   — clean-accuracy drop vs a clean model
  ASR drop after fine-tuning — persistence (Prop 3)
  measured R_spec — trigger high-frequency energy ratio vs defense threshold (Prop 2)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EvalResult:
    """One row per test (fine-tuning) dataset.

    Headline metrics (see config.EvalConfig):
      asr                  attack success rate of the BACKDOORED fine-tune
      clean_acc_backdoored accuracy on clean test nodes, backdoored fine-tune
      clean_acc_clean      accuracy on clean test nodes, clean fine-tune
    `clean_acc` is kept as an alias of clean_acc_clean for older scripts.
    """
    dataset: str
    status: str = "ok"                 # ok | skipped | error
    error: str = ""
    asr: float = float("nan")
    clean_acc_backdoored: float = float("nan")
    clean_acc_clean: float = float("nan")
    asr_clean_ft: float = float("nan")   # trigger transfer w/o downstream poisoning
    clean_acc: float = float("nan")      # alias of clean_acc_clean
    asr_after_defense: float = float("nan")
    num_nodes: int = 0
    num_classes: int = 0
    num_train: int = 0
    num_test: int = 0
    num_poisoned_train: int = 0
    num_asr_nodes: int = 0
    finetune_mode: str = ""
    num_runs: int = 1
    std: Dict[str, float] = field(default_factory=dict)   # per-metric std over runs
    runs: list = field(default_factory=list)              # per-run raw metrics
    cad: float = float("nan")
    asr_before_ft: float = float("nan")
    asr_after_ft: float = float("nan")
    asr_drop_ft: float = float("nan")
    r_spec_mean: float = float("nan")
    extra: Dict[str, float] = field(default_factory=dict)


def clean_accuracy(preds, y):
    import torch
    mask = y >= 0
    if mask.sum() == 0:
        return float("nan")
    return float((preds[mask] == y[mask]).float().mean())


def attack_success_rate(preds_on_triggered, target_label, trigger_node_ids):
    """ASR = fraction of triggered target nodes predicted as y*."""
    import torch
    if len(trigger_node_ids) == 0:
        return float("nan")
    idx = torch.as_tensor(list(trigger_node_ids), device=preds_on_triggered.device)
    return float((preds_on_triggered[idx] == target_label).float().mean())


def clean_accuracy_drop(clean_model_acc, backdoored_model_acc):
    return float(clean_model_acc - backdoored_model_acc)


def asr_drop_after_finetuning(asr_before, asr_after):
    return float(asr_before - asr_after)
