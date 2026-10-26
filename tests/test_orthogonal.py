"""Torch mirror of the orthogonal-anchoring bound (Section 4.4)."""
import pytest

torch = pytest.importorskip("torch")

from src.attack.orthogonal import orthogonality_penalty, davis_kahan_bound


def test_orthogonal_overlap_near_zero():
    H, k = 64, 8
    Q, _ = torch.linalg.qr(torch.randn(H, H))
    U_k = Q[:, :k]
    g = torch.randn(H)
    g_orth = g - U_k @ (U_k.t() @ g)
    pen_orth = orthogonality_penalty(U_k, g_orth)
    pen_rand = orthogonality_penalty(U_k, g / g.norm())
    assert pen_orth < 1e-8
    assert pen_rand > pen_orth


def test_davis_kahan_monotone_in_gap():
    ev = torch.tensor([10.0, 8.0, 6.0, 1.0, 0.5])
    small_gap = davis_kahan_bound(ev, k=3, perturbation_norm=1.0)  # gap=6-1=5
    ev2 = torch.tensor([10.0, 8.0, 6.0, 5.9, 0.5])
    big_bound = davis_kahan_bound(ev2, k=3, perturbation_norm=1.0)  # gap=6-5.9=0.1
    assert big_bound > small_gap
