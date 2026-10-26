"""
Base interface every victim GFM encoder implements, so the attack code is
model-agnostic. The SIB2 attack only needs:

    encode(x, edge_index, **kw) -> node embeddings [N, H]
    pretrain_loss(mdg)          -> scalar self-supervised loss  (for influence +
                                   clean-loss-preservation term)
    embed_dim                    -> H

Each concrete model (GCOPE / SAMGPT / GFT) wires in its own distinctive
mechanism (coordinators / structure tokens / tree vocabulary) but exposes this
same surface.

IMPORTANT (faithfulness note): these are *faithful, self-contained research
re-implementations* of each model's distinctive cross-domain mechanism on a
shared GNN backbone — enough to run the SIB2 attack end-to-end on the available
datasets. For exact reproduction of published numbers, drop the official repo
code behind this same interface:
    GCOPE  : https://github.com/cshhzhao/GCOPE
    SAMGPT : https://smufang.github.io/ (TheWebConf'25)   [structure tokens]
    GFT    : https://github.com/Zehong-Wang/GFT           [tree vocabulary / VQ]
Integration points are marked with `# >>> OFFICIAL-REPO HOOK`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _nn():
    import torch.nn as nn
    return nn


class BaseGFM:
    """Mixin documenting the required surface (concrete models subclass nn.Module
    and this)."""

    embed_dim: int

    def encode(self, x, edge_index, **kwargs):
        raise NotImplementedError

    def pretrain_loss(self, mdg) -> "torch.Tensor":
        raise NotImplementedError
