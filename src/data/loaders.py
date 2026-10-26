"""
Concrete dataset loaders. Each returns a single normalized PyG `Data` with:
    x          [N, F]   float
    edge_index [2, E]   long (undirected if cfg.data.undirected)
    y          [N]      long  (class ids, -1 for unlabeled)
    domain     str      the dataset key (used by the multi-domain combiner)

Loaders import torch / torch_geometric lazily so that unit tests covering the
pure-math parts of the attack can run without the heavy deep-learning stack.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .registry import entry, DatasetEntry

if TYPE_CHECKING:  # pragma: no cover
    from torch_geometric.data import Data


def _to_undirected_selfloop(data, undirected: bool, self_loops: bool):
    import torch
    from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops
    ei = data.edge_index
    if undirected:
        ei = to_undirected(ei, num_nodes=data.num_nodes)
    if self_loops:
        ei, _ = remove_self_loops(ei)
        ei, _ = add_self_loops(ei, num_nodes=data.num_nodes)
    data.edge_index = ei
    return data


def _normalize(data, key: str, undirected: bool, self_loops: bool):
    import torch
    # squeeze OGB-style [N,1] labels
    if data.y is not None and data.y.dim() > 1:
        data.y = data.y.view(-1)
    if data.y is None:
        data.y = torch.full((data.num_nodes,), -1, dtype=torch.long)
    data.x = data.x.float()
    data = _to_undirected_selfloop(data, undirected, self_loops)
    data.domain = key
    return data


def _load_manual(e: DatasetEntry, root: str):
    import torch
    path = os.path.join(root, "manual", e.manual_filename or f"{e.key}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{e.key}] requires a manual download. Expected a saved "
            f"torch_geometric.data.Data at:\n    {path}\n"
            f"Create it with: torch.save(Data(x=..., edge_index=..., y=...), '{path}')"
        )
    obj = torch.load(path, weights_only=False)
    if not hasattr(obj, "edge_index"):
        raise ValueError(f"[{e.key}] file at {path} is not a PyG Data object.")
    return obj


def load_one(name: str, root: str, undirected: bool = True,
             self_loops: bool = False):
    """Load a single dataset by (possibly aliased) name -> normalized Data."""
    e = entry(name)
    os.makedirs(root, exist_ok=True)

    if e.loader == "planetoid":
        from torch_geometric.datasets import Planetoid
        ds = Planetoid(root=os.path.join(root, "Planetoid"), name=e.arg)
        data = ds[0]

    elif e.loader == "ogb":
        from ogb.nodeproppred import PygNodePropPredDataset
        ds = PygNodePropPredDataset(name=e.arg, root=os.path.join(root, "OGB"))
        data = ds[0]

    elif e.loader == "amazon":
        from torch_geometric.datasets import Amazon
        ds = Amazon(root=os.path.join(root, "Amazon"), name=e.arg)
        data = ds[0]

    elif e.loader == "reddit2":
        from torch_geometric.datasets import Reddit2
        ds = Reddit2(root=os.path.join(root, "Reddit2"))
        data = ds[0]

    elif e.loader == "flickr":
        from torch_geometric.datasets import Flickr
        ds = Flickr(root=os.path.join(root, "Flickr"))
        data = ds[0]

    elif e.loader == "facebook":
        from torch_geometric.datasets import FacebookPagePage
        ds = FacebookPagePage(root=os.path.join(root, "FacebookPagePage"))
        data = ds[0]

    elif e.loader == "elliptic":
        from torch_geometric.datasets import EllipticBitcoinDataset
        ds = EllipticBitcoinDataset(root=os.path.join(root, "Elliptic"))
        data = ds[0]

    elif e.loader == "manual":
        data = _load_manual(e, root)

    else:  # pragma: no cover
        raise ValueError(f"Unhandled loader '{e.loader}' for {e.key}")

    return _normalize(data, e.key, undirected, self_loops)
