"""Torch mirror of the Parseval identity + low-pass projection (Section 3.2)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.attack.spectral import SpectrumCache, r_spec, lowpass_project, parseval_lhs_rhs


def _random_spectrum(n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, n, generator=g) > 0.7).float()
    A = torch.triu(A, 1); A = A + A.t()
    L = torch.diag(A.sum(1)) - A
    evals, evecs = torch.linalg.eigh(L)
    return SpectrumCache(eigvecs=evecs, eigvals=evals.clamp_min(0),
                         node_index=torch.arange(n), num_high=n // 2, n=n), L


def test_parseval_identity():
    spec, L = _random_spectrum()
    delta = torch.randn(spec.n)
    lhs, rhs = parseval_lhs_rhs(delta, spec)
    assert torch.allclose(lhs, rhs, atol=1e-4), (lhs.item(), rhs.item())


def test_lowpass_kills_high_freq():
    spec, _ = _random_spectrum()
    delta = torch.randn(spec.n)
    before = r_spec(delta, spec)
    after = r_spec(lowpass_project(delta, spec), spec)
    assert after <= before + 1e-6
    assert after < 1e-5
