"""Tests for the direct difference covariance Omega (Step 2, rows T1-T4)."""
import numpy as np

from rankci.core.bandwidth import andrews_bandwidth
from rankci.core.pairwise import nw_se
from rankci.core.omega import (
    cov_via_omega,
    pair_index,
    stacked_nw_balanced,
    _difference_series,
    _cross_sum,
)


# ── panel generators ─────────────────────────────────────────────────────────

def balanced_panel(n=300, p=4, seed=0):
    """AR(1) idiosyncratic losses + a common shock (cancels in differences)."""
    rng = np.random.default_rng(seed)
    common = np.zeros(n)
    for t in range(1, n):
        common[t] = 0.5 * common[t - 1] + rng.standard_normal()
    X = np.empty((n, p))
    for j in range(p):
        e = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = e[0]
        rho = 0.3 + 0.1 * j
        for t in range(1, n):
            x[t] = rho * x[t - 1] + e[t]
        X[:, j] = 1.0 + x + common          # common shock shared across j
    return X


def unbalanced_panel(n=300, p=4, seed=0, drop_frac=0.15):
    X = balanced_panel(n, p, seed)
    rng = np.random.default_rng(seed + 100)
    mask = rng.random((n, p)) < drop_frac
    X[mask] = np.nan
    return X


# ── T1: diagonal-consistency ─────────────────────────────────────────────────

def test_diagonal_equals_nw_se_balanced():
    """Balanced panel: Omega_aa == nw_se(d^a)^2 exactly (same lag convention)."""
    X = balanced_panel()
    pairs = pair_index(X.shape[1])
    _, diag = cov_via_omega(X, return_diagnostics=True)
    Omega_raw = diag["Omega_raw"]
    for a, (j, k) in enumerate(pairs):
        d = X[:, j] - X[:, k]
        _, se = nw_se(d)                    # Andrews bandwidth, Bartlett
        assert np.isclose(Omega_raw[a, a], se**2, rtol=1e-10, atol=1e-14), (a, j, k)


def _ref_diag_realtime(d_full, L):
    """Independent real-time univariate Bartlett LRV / n_a for a gapped series."""
    finite = np.isfinite(d_full)
    na = int(finite.sum())
    dt = d_full - np.nanmean(d_full)
    s = float(np.nansum(dt * dt))
    for h in range(1, L + 1):
        w = 1.0 - h / (L + 1)
        s += 2.0 * w * float(np.nansum(dt[h:] * dt[:-h]))
    return s / na**2


def test_diagonal_realtime_reference_unbalanced():
    """Unbalanced: Omega_aa matches an independent real-time LRV reference."""
    X = unbalanced_panel()
    pairs = pair_index(X.shape[1])
    Dtil, L, n_a = _difference_series(X, pairs)
    _, diag = cov_via_omega(X, return_diagnostics=True)
    Omega_raw = diag["Omega_raw"]
    for a, (j, k) in enumerate(pairs):
        d_full = X[:, j] - X[:, k]
        ref = _ref_diag_realtime(d_full, int(L[a]))
        assert np.isclose(Omega_raw[a, a], ref, rtol=1e-10, atol=1e-14), (a, j, k)


# ── T2: balanced-nesting ─────────────────────────────────────────────────────

def test_nesting_common_bandwidth():
    """Balanced + common L: cov_via_omega raw == single-bandwidth stacked NW."""
    X = balanced_panel()
    for L0 in (0, 3, 8):
        _, diag = cov_via_omega(X, L=L0, return_diagnostics=True)
        ref = stacked_nw_balanced(X, L0)
        assert np.allclose(diag["Omega_raw"], ref, rtol=1e-10, atol=1e-14), L0


# ── T3: calibration across the (omega) seam ──────────────────────────────────

def test_calibration_diag_is_se_squared():
    """se_q used to studentize the omega path == sqrt(diag(Omega_psd))."""
    from rankci.core.covariance import studentized_null_draws
    X = unbalanced_panel()
    d = studentized_null_draws(X, method="omega", B=10, seed=0)
    se_q, cov = d["se_q"], d["cov"]
    assert np.allclose(se_q**2, np.diag(cov), rtol=1e-12, atol=1e-14)


# ── T4: PSD ──────────────────────────────────────────────────────────────────

def test_common_bandwidth_balanced_is_psd():
    """PSD-for-free holds for the COMMON-bandwidth balanced estimator only."""
    Xb = balanced_panel()
    for L0 in (3, 5, 8):
        _, diag = cov_via_omega(Xb, L=L0, return_diagnostics=True)
        assert diag["min_eig_raw"] >= -1e-12, L0
        assert diag["frob_gap"] < 1e-10, L0


def test_per_series_bandwidth_balanced_may_project():
    """With per-series Andrews bandwidths, heterogeneous persistence breaks the
    single-quadratic-form structure, so even a balanced panel can need a (small)
    projection. The projected matrix is still PSD, and the gap is small relative
    to the matrix norm. (This qualifies the 'PSD for free' claim in the tex.)"""
    Xb = balanced_panel()
    Ob, diag = cov_via_omega(Xb, return_diagnostics=True)
    assert np.linalg.eigvalsh(Ob).min() >= -1e-12
    rel_gap = diag["frob_gap"] / np.linalg.norm(diag["Omega_raw"], ord="fro")
    assert rel_gap < 0.05, rel_gap


def test_unbalanced_projection_psd():
    Xu = unbalanced_panel()
    Ou, diag_u = cov_via_omega(Xu, return_diagnostics=True)
    assert np.linalg.eigvalsh(Ou).min() >= -1e-12
    assert diag_u["frob_gap"] >= 0.0


def test_cross_sum_counts_joint_support():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.0, np.nan, 3.0, 1.0])
    # h=0: t in {0,3} -> a[0]*b[0] + a[3]*b[3] = 1*1 + 4*1 = 5
    assert _cross_sum(a, b, 0) == 5.0
    # h=1: sum_t a[t]*b[t-1]; t=1 (2*1) and t=3 (4*3) are jointly observed -> 14
    assert _cross_sum(a, b, 1) == 14.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
