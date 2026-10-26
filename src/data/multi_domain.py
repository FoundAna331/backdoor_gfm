"""
Combine multiple domains into one pretraining object, per each model's *original*
method (instruction 1).

Shared first step (all three): align every domain's features to `align_dim`
(feature_align.align_features), then build a batched multi-domain graph.

Model-specific combination:
  * GCOPE : add `num_coordinators_per_domain` virtual "coordinator" nodes per
            domain; connect every real node of a domain to that domain's
            coordinator(s), and connect all coordinators to each other. This is
            GCOPE's "unify isolated source datasets into one large-scale dataset
            with inter-dataset connections". Coordinator node features are
            learnable (initialised to the domain feature mean here; the learnable
            table lives in models/gcope.py).
  * SAMGPT: keep domains as disjoint components (block-diagonal), and attach a
            `domain_id` per node. Structure alignment happens *inside* the encoder
            via per-domain structure tokens (models/samgpt.py), not by adding
            edges. We therefore only batch + tag domains here.
  * GFT   : same disjoint batched graph with `domain_id`; GFT's cross-domain
            unification is the shared tree-vocabulary / VQ in the encoder
            (models/gft.py), so no extra edges are added at the data level.

Output: a container with
    x           [Ntot, align_dim]
    edge_index  [2, Etot]
    y           [Ntot]
    domain_id   [Ntot]            integer domain index
    domain_names list[str]
    is_coordinator [Ntot] bool    (GCOPE only; all False otherwise)
    node_offsets list[int]        start index of each domain's real nodes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .feature_align import align_features
from .loaders import load_one


@dataclass
class MultiDomainGraph:
    x: "object"
    edge_index: "object"
    y: "object"
    domain_id: "object"
    domain_names: List[str]
    is_coordinator: "object"
    node_offsets: List[int] = field(default_factory=list)
    num_real_nodes: int = 0

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.y = self.y.to(device)
        self.domain_id = self.domain_id.to(device)
        self.is_coordinator = self.is_coordinator.to(device)
        return self


def _batch_domains(datas, align_dim: int, align_method: str):
    """Block-diagonal batch of several domains with aligned features."""
    import torch

    xs, eis, ys, dids = [], [], [], []
    offsets, names = [], []
    offset = 0
    for d_idx, data in enumerate(datas):
        x = align_features(data.x, align_dim, align_method)
        n = x.shape[0]
        xs.append(x)
        eis.append(data.edge_index + offset)
        ys.append(data.y)
        dids.append(torch.full((n,), d_idx, dtype=torch.long))
        offsets.append(offset)
        names.append(getattr(data, "domain", f"domain{d_idx}"))
        offset += n

    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1)
    y = torch.cat(ys, dim=0)
    domain_id = torch.cat(dids, dim=0)
    return x, edge_index, y, domain_id, offsets, names, offset


def _add_gcope_coordinators(x, edge_index, domain_id, offsets, names,
                            num_real, coords_per_domain: int):
    """GCOPE virtual coordinators + inter-dataset connections."""
    import torch

    num_domains = len(names)
    coord_feats = []
    coord_edges = []
    coord_ids = []
    coord_index = num_real
    coord_global_ids = []

    for d_idx in range(num_domains):
        start = offsets[d_idx]
        end = offsets[d_idx + 1] if d_idx + 1 < len(offsets) else num_real
        node_ids = torch.arange(start, end)
        dom_mean = x[start:end].mean(dim=0, keepdim=True)
        for _ in range(coords_per_domain):
            coord_feats.append(dom_mean.clone())
            # connect coordinator <-> every real node in its domain (both dirs)
            c = coord_index
            src = torch.cat([node_ids, torch.full_like(node_ids, c)])
            dst = torch.cat([torch.full_like(node_ids, c), node_ids])
            coord_edges.append(torch.stack([src, dst], dim=0))
            coord_ids.append(d_idx)
            coord_global_ids.append(c)
            coord_index += 1

    # fully connect coordinators to each other (inter-dataset backbone)
    if len(coord_global_ids) > 1:
        cg = torch.tensor(coord_global_ids)
        src = cg.repeat_interleave(len(cg))
        dst = cg.repeat(len(cg))
        mask = src != dst
        coord_edges.append(torch.stack([src[mask], dst[mask]], dim=0))

    x = torch.cat([x] + coord_feats, dim=0)
    extra_ids = torch.tensor(coord_ids, dtype=torch.long)
    domain_id = torch.cat([domain_id, extra_ids], dim=0)
    edge_index = torch.cat([edge_index] + coord_edges, dim=1)

    total = x.shape[0]
    is_coord = torch.zeros(total, dtype=torch.bool)
    is_coord[num_real:] = True
    return x, edge_index, domain_id, is_coord


def build_multi_domain(dataset_names, cfg_data, model_name: str) -> MultiDomainGraph:
    import torch

    datas = [load_one(n, cfg_data.root, cfg_data.undirected,
                      cfg_data.add_self_loops) for n in dataset_names]

    x, edge_index, y, domain_id, offsets, names, num_real = _batch_domains(
        datas, cfg_data.align_dim, cfg_data.align_method)

    is_coord = torch.zeros(num_real, dtype=torch.bool)

    if model_name == "gcope":
        x, edge_index, domain_id, is_coord = _add_gcope_coordinators(
            x, edge_index, domain_id, offsets, names, num_real,
            coords_per_domain=1)  # overridden by model cfg at build time if needed
        # pad y for coordinator nodes with -1 (no label)
        pad = torch.full((x.shape[0] - y.shape[0],), -1, dtype=torch.long)
        y = torch.cat([y, pad], dim=0)

    return MultiDomainGraph(
        x=x, edge_index=edge_index, y=y, domain_id=domain_id,
        domain_names=names, is_coordinator=is_coord,
        node_offsets=offsets, num_real_nodes=num_real,
    )
