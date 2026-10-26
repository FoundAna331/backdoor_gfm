"""
Standalone NumPy verification of the three mathematical claims SIB2 rests on.
Runs WITHOUT torch/PyG so the core math can be checked anywhere.

  (A) Parseval / Rayleigh identity (paper Section 3.2):
        delta^T L delta == sum_l c_l^2 lambda_l,  c_l = u_l^T delta
  (B) Low-pass projection kills high-frequency energy -> R_spec drops to ~0.
  (C) Orthogonality bound (Section 4.4):
        a trigger direction built orthogonal to the top-k covariance subspace
        has ~0 overlap with an update confined to that subspace, vs O(1) for a
        random direction.
  (D) Damped Neumann series recovers G_lambda^{-1} v (Section 2.6).

Exit code 0 iff all checks pass.
"""
import numpy as np


def check_parseval(seed=0, n=40, tol=1e-9):
    rng = np.random.default_rng(seed)
    # random symmetric graph -> Laplacian
    A = rng.integers(0, 2, size=(n, n)).astype(float)
    A = np.triu(A, 1); A = A + A.T
    L = np.diag(A.sum(1)) - A
    evals, evecs = np.linalg.eigh(L)
    delta = rng.standard_normal(n)
    lhs = delta @ (L @ delta)
    c = evecs.T @ delta
    rhs = np.sum(c**2 * evals)
    err = abs(lhs - rhs) / (abs(lhs) + 1e-12)
    ok = err < tol
    print(f"(A) Parseval identity:   lhs={lhs:.6f} rhs={rhs:.6f} rel_err={err:.2e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok, (evals, evecs, delta)


def check_lowpass(evals, evecs, delta, high_frac=0.5):
    n = len(evals)
    m = max(1, int(round(high_frac * n)))

    def r_spec(sig):
        c = evecs.T @ sig
        energy = c**2 * evals
        return energy[-m:].sum() / (energy.sum() + 1e-12)

    def lowpass(sig):
        c = evecs.T @ sig
        c[-m:] = 0.0
        return evecs @ c

    before = r_spec(delta)
    after = r_spec(lowpass(delta))
    ok = after <= before and after < 1e-8
    print(f"(B) Low-pass projection: R_spec before={before:.4f} after={after:.2e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_orthogonality(seed=1, H=64, k=8, tol=1e-10):
    rng = np.random.default_rng(seed)
    # random orthonormal basis; U_k = top-k directions
    Q, _ = np.linalg.qr(rng.standard_normal((H, H)))
    U_k = Q[:, :k]                              # [H, k]
    # fine-tuning update confined to span(U_k)
    coeffs = rng.standard_normal(k)
    dtheta = U_k @ coeffs                       # in span(U_k)

    # orthogonal trigger direction: project a random vector off U_k
    g = rng.standard_normal(H)
    g_orth = g - U_k @ (U_k.T @ g)
    g_orth /= np.linalg.norm(g_orth)
    overlap_orth = abs(g_orth @ dtheta)

    # non-orthogonal random direction
    g_rand = g / np.linalg.norm(g)
    overlap_rand = abs(g_rand @ dtheta)

    ok = overlap_orth < tol and overlap_rand > 1e-3
    print(f"(C) Orthogonal anchoring: overlap(orth)={overlap_orth:.2e}  "
          f"overlap(random)={overlap_rand:.4f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_neumann(seed=2, P=50, damping=0.5, terms=400, tol=1e-3):
    rng = np.random.default_rng(seed)
    # empirical Fisher G = (1/B) sum g g^T  (PSD), G_lambda = G + lambda I
    B = 80
    Gmat = rng.standard_normal((B, P))
    G = Gmat.T @ Gmat / B
    G_lam = G + damping * np.eye(P)
    v = rng.standard_normal(P)
    exact = np.linalg.solve(G_lam, v)

    # scaled Neumann: (1/s) sum_i (I - G_lam/s)^i v
    s = np.linalg.norm(G_lam, 2) * 1.01
    r = v / s
    vs = v / s
    for _ in range(terms):
        r = vs + r - (G_lam @ r) / s
    err = np.linalg.norm(r - exact) / (np.linalg.norm(exact) + 1e-12)
    ok = err < tol
    print(f"(D) Neumann iHVP:        rel_err vs exact solve={err:.2e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_truncated_energy(seed=3, n=300, r=60, tol=1e-8):
    """(E) Truncated-spectrum R_spec (the large-graph path).

    With only the r smallest eigenpairs, the TOTAL energy is still exact via
    Parseval (delta^T L delta, a sparse mat-vec), and
        high = total - (energy of the first `keep` computed modes)
    is an upper bound on the true high-band energy that equals it when r == n.
    Also checks the shift-invert Lanczos solver against dense eigh on a connected
    graph, and that low-pass projection leaves ~0 residual high energy."""
    import scipy.sparse as sps
    from scipy.sparse.linalg import eigsh
    rng = np.random.default_rng(seed)
    # connected graph: ring + random chords
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    chords = rng.integers(0, n, size=(2 * n, 2))
    for a, b in chords:
        if a != b:
            A[a, b] = A[b, a] = 1
    L = np.diag(A.sum(1)) - A
    wd, vd = np.linalg.eigh(L)
    w, v = eigsh(sps.csc_matrix(L), k=r, sigma=-1e-3, which="LM")
    o = np.argsort(w); w, v = w[o], v[:, o]
    eig_err = np.abs(w - wd[:r]).max()

    m = n // 2
    keep = min(r, n - m)
    x = rng.standard_normal(n)
    total = x @ L @ x
    exact_total = ((vd.T @ x) ** 2 * wd).sum()
    high_trunc = max(total - (((v.T @ x) ** 2) * w)[:keep].sum(), 0.0)
    high_exact = (((vd.T @ x) ** 2) * wd)[n - m:].sum()
    xp = v[:, :keep] @ (v[:, :keep].T @ x)
    resid = max(xp @ L @ xp - (((v.T @ xp) ** 2) * w)[:keep].sum(), 0) / (xp @ L @ xp)
    ok = (eig_err < 1e-6 and abs(total - exact_total) / total < tol
          and high_trunc >= high_exact - 1e-9 and resid < 1e-9)
    print(f"(E) Truncated spectrum:  eig_err={eig_err:.1e}  Parseval total ok  "
          f"R_spec_trunc={high_trunc / total:.3f} >= exact={high_exact / total:.3f}  "
          f"lowpass_resid={resid:.1e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 66)
    print("SIB2 core-math verification (NumPy, no torch required)")
    print("=" * 66)
    okA, (evals, evecs, delta) = check_parseval()
    okB = check_lowpass(evals, evecs, delta)
    okC = check_orthogonality()
    okD = check_neumann()
    okE = check_truncated_energy()
    all_ok = all([okA, okB, okC, okD, okE])
    print("=" * 66)
    print("RESULT:", "ALL PASS" if all_ok else "SOME FAILED")
    print("=" * 66)
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
