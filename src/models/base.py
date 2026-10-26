
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
