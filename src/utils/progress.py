
from __future__ import annotations

from typing import Iterable, Optional

try:
    from tqdm import tqdm  # type: ignore
    _HAS_TQDM = True
except Exception:  # pragma: no cover
    _HAS_TQDM = False


def pbar(iterable: Iterable, desc: str = "", total: Optional[int] = None,
         disable: bool = False):
    """Return a progress-bar-wrapped iterator (or the raw iterable)."""
    if _HAS_TQDM and not disable:
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
    return iterable


class ManualBar:
    """Explicit update()-driven bar for loops that aren't a single iterable
    (e.g. block-wise truncated eigendecomposition)."""

    def __init__(self, total: int, desc: str = "", disable: bool = False):
        self.total = total
        self.disable = disable or not _HAS_TQDM
        self._bar = tqdm(total=total, desc=desc, dynamic_ncols=True) \
            if (_HAS_TQDM and not disable) else None
        self._n = 0

    def update(self, n: int = 1) -> None:
        self._n += n
        if self._bar is not None:
            self._bar.update(n)
        elif not self.disable:
            pass  # silent

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
