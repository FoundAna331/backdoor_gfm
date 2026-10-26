"""Cache layer: hashing, atomic save/load, and failure handling."""
import os

import pytest

torch = pytest.importorskip("torch")

from src.config import build_config
from src.utils.cache import DiskCache, hash_obj


def test_hash_is_stable_and_sensitive():
    cfg = build_config()
    assert hash_obj("a", cfg.model) == hash_obj("a", cfg.model)
    other = build_config(overrides={"model": {"hidden_dim": 64}})
    assert hash_obj(cfg.model) != hash_obj(other.model)


def test_roundtrip_and_failure_modes(tmp_path):
    c = DiskCache(str(tmp_path))
    assert c.load("k", "n") is None
    assert c.save("k", "n", {"w": torch.arange(4)})
    assert torch.equal(c.load("k", "n")["w"], torch.arange(4))
    assert not [f for f in os.listdir(tmp_path / "k") if ".tmp" in f]

    (tmp_path / "k" / "bad.pt").write_bytes(b"garbage")
    assert c.load("k", "bad") is None                      # corrupt -> miss
    torch.save({"cache_version": "old", "payload": 1}, str(tmp_path / "k" / "old.pt"))
    assert c.load("k", "old") is None                      # version mismatch -> miss

    off = DiskCache(str(tmp_path), enabled=False)
    assert off.load("k", "n") is None and off.save("k", "x", 1) is False
