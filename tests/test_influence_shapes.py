"""Shape / smoke tests for the influence machinery (Section 2)."""
import pytest

torch = pytest.importorskip("torch")

from src.attack.influence import FisherOperator, ihvp_neumann, ihvp_lissa


def test_fisher_matvec_and_ihvp():
    B, P = 16, 20
    G = torch.randn(B, P)
    op = FisherOperator(G, damping=0.1)
    v = torch.randn(P)
    Gv = op.matvec(v)
    assert Gv.shape == (P,)
    # iHVP should approximately invert: G_lam (G_lam^{-1} v) ~ v
    inv = ihvp_neumann(op, v, num_terms=300, scale=1.0)
    recon = op.matvec(inv)
    rel = (recon - v).norm() / v.norm()
    assert rel < 0.1, rel.item()


def test_lissa_runs():
    op = FisherOperator(torch.randn(8, 10), damping=0.2)
    v = torch.randn(10)
    out = ihvp_lissa(op, v, depth=50, samples=2, scale=1.0)
    assert out.shape == (10,)
