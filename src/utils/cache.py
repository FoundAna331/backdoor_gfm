"""
Disk cache for the expensive, *deterministic* stages of a run.

What is cached (and why it cannot change the attack's outcome)
--------------------------------------------------------------
kind="pretrain"  the pretrained victim-GFM weights (state_dict).
                 Pretraining happens BEFORE the attack and never sees the attack
                 config, so one checkpoint is valid for every attack variant,
                 defense, and ablation that shares the same model/data/seed.
kind="attack"    (a) the influence-based target selection and
                 (b) the finished result of each attacked target (optimised
                     trigger features, host edges, loss history).
                 Each entry is keyed on EVERYTHING it depends on (model-weight
                 fingerprint, graph fingerprint, seed, the relevant config, the
                 target id), so a changed setting simply misses the cache and is
                 recomputed. Per-target entries are written the moment a target
                 finishes, which lets an interrupted / crashed run resume.

Everything is keyed by content hashes; nothing is ever matched by file name alone.
Cache problems (unwritable dir, corrupt file, version mismatch) never abort a run:
they degrade to "recompute".

Disable with `--no-cache` (or runtime.use_cache=false). Delete the cache directory
(default ./cache) to start fresh. Bump CACHE_VERSION when code semantics change.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

CACHE_VERSION = "v1"


# --------------------------------------------------------------------------- #
#  hashing helpers (torch-free)
# --------------------------------------------------------------------------- #
def _jsonable(o):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    return str(o)


def hash_obj(*parts, n: int = 16) -> str:
    """Stable short hash of arbitrary (JSON-able / dataclass) parts."""
    blob = json.dumps(list(parts), sort_keys=True, default=_jsonable)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:n]


def fingerprint_model(model) -> str:
    """Hash of every parameter/buffer value. Identical weights -> identical hash,
    regardless of whether they were just trained or loaded from disk."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        t = v.detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        try:
            h.update(t.numpy().tobytes())
        except Exception:                      # e.g. bfloat16 has no numpy dtype
            h.update(t.float().numpy().tobytes())
    return h.hexdigest()[:16]


def fingerprint_graph(mdg) -> str:
    """Cheap content fingerprint of the multi-domain graph (shape + checksums)."""
    import torch
    with torch.no_grad():
        parts = [list(mdg.x.shape), list(mdg.edge_index.shape),
                 [str(d) for d in getattr(mdg, "domain_names", [])],
                 round(float(mdg.x.double().sum()), 4),
                 round(float(mdg.x.double().abs().sum()), 4),
                 int(mdg.edge_index.sum())]
        y = getattr(mdg, "y", None)
        if y is not None:
            parts.append(int(y.sum()))
    return hash_obj(*parts)


# --------------------------------------------------------------------------- #
#  the cache itself
# --------------------------------------------------------------------------- #
class DiskCache:
    """Tiny picklable key->payload store under `root/<kind>/<name>.pt`."""

    def __init__(self, root: str = "./cache", enabled: bool = True):
        self.root = root
        self.enabled = bool(enabled)

    @classmethod
    def from_cfg(cls, cfg) -> "DiskCache":
        rt = cfg.runtime
        return cls(getattr(rt, "cache_dir", "./cache"),
                   getattr(rt, "use_cache", True))

    def path(self, kind: str, name: str) -> str:
        return os.path.join(self.root, kind, f"{name}.pt")

    def has(self, kind: str, name: str) -> bool:
        return self.enabled and os.path.isfile(self.path(kind, name))

    def load(self, kind: str, name: str):
        """Return the stored payload, or None on a miss / unreadable entry."""
        if not self.has(kind, name):
            return None
        p = self.path(kind, name)
        try:
            import torch
            obj = torch.load(p, map_location="cpu", weights_only=False)
            if (isinstance(obj, dict) and obj.get("cache_version") == CACHE_VERSION
                    and "payload" in obj):
                return obj["payload"]
            print(f"[cache] ignoring {p} (different cache version)", flush=True)
        except Exception as e:                  # corrupt / truncated / incompatible
            print(f"[cache] could not read {p} ({type(e).__name__}: {e}); "
                  f"recomputing", flush=True)
        return None

    def save(self, kind: str, name: str, payload) -> bool:
        """Atomically write `payload`. Returns False (and warns) on failure."""
        if not self.enabled:
            return False
        p = self.path(kind, name)
        tmp = f"{p}.tmp{os.getpid()}"
        try:
            import torch
            os.makedirs(os.path.dirname(p), exist_ok=True)
            torch.save({"cache_version": CACHE_VERSION, "payload": payload}, tmp)
            os.replace(tmp, p)                  # atomic: never a half-written file
            return True
        except Exception as e:
            print(f"[cache] could not write {p} ({type(e).__name__}: {e}); "
                  f"continuing without caching", flush=True)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return False
