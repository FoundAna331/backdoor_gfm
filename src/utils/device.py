"""
Device management.

Instruction (8): run the attack on GPU, with options to select multiple GPUs.

Strategy:
  * `configure_gpus([0,1])` sets CUDA_VISIBLE_DEVICES before heavy CUDA init so
    the process only ever sees the requested cards.
  * `primary_device()` returns the device the trigger optimisation / influence
    computation runs on (cuda:0 within the visible set, else CPU).
  * `maybe_data_parallel(module)` wraps an nn.Module in DataParallel when more
    than one GPU is visible, so pretraining / covariance passes use all of them.

The eigendecomposition and iHVP solves are kept on a single device on purpose:
they are latency-bound linear-algebra routines, not throughput-bound, and
sharding them across cards only adds transfer overhead.
"""
from __future__ import annotations

import os
from typing import List, Optional


def configure_gpus(gpu_ids: List[int]) -> str:
    """Pin the process to the requested GPUs. Returns a human-readable string.

    Must be called *before* the first CUDA allocation to take effect.
    """
    # The caching allocator can fragment badly on long runs that mix many
    # small allocations (per-node Fisher grads) with a few very large ones
    # (the full-graph GNN forward/backward) -- exactly this workload. Expandable
    # segments let the allocator grow a reservation instead of hunting for a
    # single contiguous free block, which is what PyTorch itself suggests in
    # the OOM message this is fixing. Only set if the user hasn't already
    # chosen a value.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        return f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return "CUDA_VISIBLE_DEVICES=<none> (CPU)"


def _torch():
    import torch
    return torch


def cuda_available() -> bool:
    try:
        return _torch().cuda.is_available()
    except Exception:
        return False


def visible_device_count() -> int:
    try:
        return _torch().cuda.device_count()
    except Exception:
        return 0


def primary_device(prefer_gpu: bool = True):
    torch = _torch()
    if prefer_gpu and cuda_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def maybe_data_parallel(module, gpu_ids: Optional[List[int]] = None):
    """Wrap in DataParallel if >1 visible GPU. `gpu_ids` here are *logical*
    ids within the visible set (0..n-1), not the physical ones."""
    torch = _torch()
    if cuda_available() and visible_device_count() > 1:
        dev_ids = gpu_ids if gpu_ids else list(range(visible_device_count()))
        return torch.nn.DataParallel(module, device_ids=dev_ids)
    return module


def describe_devices() -> str:
    if not cuda_available():
        return "CPU only"
    torch = _torch()
    n = visible_device_count()
    names = []
    for i in range(n):
        try:
            names.append(f"cuda:{i} ({torch.cuda.get_device_name(i)})")
        except Exception:
            names.append(f"cuda:{i}")
    return f"{n} visible GPU(s): " + ", ".join(names)
