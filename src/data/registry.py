
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DatasetEntry:
    key: str
    loader: str                 # {"planetoid","ogb","amazon","reddit2","flickr",
                                #  "facebook","elliptic","manual"}
    arg: Optional[str] = None   # sub-name passed to the loader
    auto: bool = True
    note: str = ""
    manual_filename: Optional[str] = None


# Canonical keys and a few friendly aliases.
_ALIASES: Dict[str, str] = {
    "arxiv": "ogbn-arxiv",
    "ogb-arxiv": "ogbn-arxiv",
    "ogbarxiv": "ogbn-arxiv",
    "amazon-computers": "computers",
    "amazon-photo": "photo",
    "amazon-photos": "photo",
    "photos": "photo",
    "fb": "facebook",
    "elliptic-bitcoin": "elliptic",
    "ellipticpp": "elliptic++",
    "elliptic-plus-plus": "elliptic++",
}

REGISTRY: Dict[str, DatasetEntry] = {
    "cora":       DatasetEntry("cora", "planetoid", "Cora"),
    "citeseer":   DatasetEntry("citeseer", "planetoid", "CiteSeer"),
    "pubmed":     DatasetEntry("pubmed", "planetoid", "PubMed"),  # handy extra
    "ogbn-arxiv": DatasetEntry("ogbn-arxiv", "ogb", "ogbn-arxiv"),
    "computers":  DatasetEntry("computers", "amazon", "Computers"),
    "photo":      DatasetEntry("photo", "amazon", "Photo"),
    "reddit":     DatasetEntry("reddit", "reddit2", "Reddit2",
                               note="Reddit2 (sparser variant; large)."),
    "flickr":     DatasetEntry("flickr", "flickr", "Flickr"),
    "facebook":   DatasetEntry("facebook", "facebook", "FacebookPagePage",
                               note="PyG FacebookPagePage (page-page network)."),
    "elliptic":   DatasetEntry("elliptic", "elliptic", "EllipticBitcoin",
                               note="Base Elliptic (binary licit/illicit)."),
    "elliptic++": DatasetEntry("elliptic++", "manual", auto=False,
                               note="Elliptic++ extended graph not in PyG.",
                               manual_filename="elliptic++.pt"),
    "twitter":    DatasetEntry("twitter", "manual", auto=False,
                               note="No canonical PyG Twitter node dataset.",
                               manual_filename="twitter.pt"),
}


def canonical(name: str) -> str:
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Known: {sorted(REGISTRY)} "
            f"(aliases: {sorted(_ALIASES)})"
        )
    return key


def entry(name: str) -> DatasetEntry:
    return REGISTRY[canonical(name)]


def availability_report(names) -> str:
    """Human-readable table of which requested datasets auto-load vs need manual
    download. Printed once at startup (instruction 2)."""
    rows = ["Dataset availability:"]
    manual = []
    for n in names:
        try:
            e = entry(n)
        except KeyError as ex:
            rows.append(f"  - {n:<14} UNKNOWN  ({ex})")
            continue
        tag = "AUTO " if e.auto else "MANUAL"
        rows.append(f"  - {e.key:<14} {tag}  {e.note}")
        if not e.auto:
            manual.append(e)
    if manual:
        rows.append("")
        rows.append("  >>> The following need a manual download; place a saved "
                    "torch_geometric.data.Data")
        rows.append("      object at <data.root>/manual/<filename>:")
        for e in manual:
            rows.append(f"        {e.key:<14} -> manual/{e.manual_filename}")
    return "\n".join(rows)
