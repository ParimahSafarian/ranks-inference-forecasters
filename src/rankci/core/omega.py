"""
Approach 2 — the direct difference covariance Omega on an unbalanced panel.

We estimate the covariance of the q = C(p, 2) pairwise difference-mean
estimators directly, rather than reconstructing a p-dimensional level
covariance and differencing it (Approach 1, ``cov_theta_pairwise``). This is
the *pairwise-of-pairs* (route (i)) estimator of ``Reports/omega_unbalanced.tex``.

For unordered pairs a = (j, k), j < k, with difference series
d^a_t = X_{t,j} - X_{t,k} on support T_a = T_j ∩ T_k:

    Omega_ab = (1 / n_ab^2) [ g0 + sum_{h=1}^{L_ab} w_h (g^{ab}_h + g^{ba}_h) ],
    g^{ab}_h = sum_{t : t, t-h both in-support} dtil^a_t dtil^b_{t-h},
    w_h      = 1 - h / (L_ab + 1),          (Bartlett)

where n_ab = |T_a ∩ T_b| is the *pair-of-pairs overlap* (the single common
divisor applied across all lags — the normalization that keeps the balanced
estimator PSD), and dtil^a is d^a demeaned on its own support.

Design points (all from the LaTeX, or flagged in STEP2_DESIGN.md):
  * Bandwidth per series by the Andrews plug-in; off-diagonal L_ab = min(L_a, L_b).
  * n_ab == 0 (two pairs never co-observed) -> entry 0; nearest-PSD absorbs it.
  * Assembled matrix is symmetric but generally not PSD on an unbalanced panel;
    repaired by the Frobenius-nearest PSD projection (Higham eigen-clip).

Two properties tie it back to the pipeline and are covered by the tests:
  * Diagonal-consistency: Omega_aa equals the univariate Bartlett long-run
    variance of d^a divided by n_a — i.e. the serial-dependence SE squared.
  * Nesting: on a balanced panel each entry equals the (a, b) entry of the
    stacked multivariate Newey--West estimator S_hat / n.
"""
import numpy as np

from .bandwidth import andrews_bandwidth
from .pairwise import _nearest_psd


# ── Pair indexing ────────────────────────────────────────────────────────────

def pair_index(p: int) -> list[tuple[int, int]]:
    """Ordered list of unordered pairs (j, k), j < k. Index a = position."""
    return [(j, k) for j in range(p) for k in range(j + 1, p)]


# ── Difference-series construction ───────────────────────────────────────────

def _difference_series(
    X: np.ndarray,
    pairs: list[tuple[int, int]],
    L_fixed: int | None = None,
):
    """
    Build, for each pair a = (j, k), the full-length demeaned difference series
    dtil^a (NaN off support) and its bandwidth L_a.

    Parameters
    ----------
    L_fixed : if given, use this common bandwidth for every series (capped at
              n_a - 1); otherwise each series gets its own Andrews plug-in.

    Returns
    -------
    Dtil : (q, n) array, row a = demeaned d^a with NaN off T_a.
    L    : (q,) int array of per-series bandwidths.
    n_a  : (q,) int array of per-series support sizes.
    """
    n, _ = X.shape
    q = len(pairs)
    Dtil = np.full((q, n), np.nan)
    L = np.zeros(q, dtype=int)
    n_a = np.zeros(q, dtype=int)
    for a, (j, k) in enumerate(pairs):
        d = X[:, j] - X[:, k]                       # NaN where either missing
        mask = np.isfinite(d)
        na = int(mask.sum())
        n_a[a] = na
        if na == 0:
            continue
        if L_fixed is None:
            # Bandwidth on the compressed overlap series, matching nw_se.
            L[a] = andrews_bandwidth(d[mask])
        else:
            L[a] = min(int(L_fixed), na - 1)
        Dtil[a, mask] = d[mask] - d[mask].mean()    # demean on own support
    return Dtil, L, n_a


def _cross_sum(a_full: np.ndarray, b_full: np.ndarray, h: int) -> float:
    """
    sum_t a_full[t] * b_full[t - h] over t where both are finite (h >= 0).

    NaN entries (off-support) drop out via nansum, so only jointly observed
    (t, t - h) pairs contribute — exactly the paired index set T_ab(h).
    """
    if h == 0:
        prod = a_full * b_full
    else:
        prod = a_full[h:] * b_full[:-h]
    return float(np.nansum(prod))


# ── The estimator ────────────────────────────────────────────────────────────

def cov_via_omega(
    X: np.ndarray,
    min_overlap: int = 2,
    L: int | None = None,
    return_diagnostics: bool = False,
):
    """
    Direct difference covariance Omega (q x q), pairwise-of-pairs, PSD-projected.

    Parameters
    ----------
    X                  : (n, p) array of losses, may contain NaN.
    min_overlap        : entries with pair-of-pairs overlap n_ab < min_overlap
                         are set to 0 (unidentified); the diagonal always uses
                         the full support.
    L                  : optional common bandwidth override for every difference
                         series (mainly for testing / the nesting reduction).
                         None -> per-series Andrews plug-in.
    return_diagnostics : if True, also return a dict with the raw (pre-PSD)
                         matrix, the Frobenius projection gap, the smallest
                         pre-projection eigenvalue, per-pair bandwidths, and the
                         pair-of-pairs overlap matrix.

    Returns
    -------
    Omega_psd : (q, q) PSD covariance of the q difference-mean estimators,
                ordered by ``pair_index(p)``.
    (diagnostics dict, if requested)
    """
    X = np.asarray(X, dtype=float)
    _, p = X.shape
    pairs = pair_index(p)
    q = len(pairs)

    Dtil, L_per, n_a = _difference_series(X, pairs, L_fixed=L)

    Omega = np.zeros((q, q))
    n_ab_mat = np.zeros((q, q), dtype=int)
    finite = np.isfinite(Dtil)                        # (q, n) support masks

    for a in range(q):
        if n_a[a] == 0:
            continue
        for b in range(a, q):
            if n_a[b] == 0:
                continue
            overlap = finite[a] & finite[b]
            n_ab = int(overlap.sum())
            n_ab_mat[a, b] = n_ab_mat[b, a] = n_ab
            # Diagonal always uses full support; off-diagonals require min_overlap.
            if a != b and n_ab < max(min_overlap, 1):
                continue
            if n_ab == 0:
                continue

            L_ab = min(int(L_per[a]), int(L_per[b]), n_ab - 1)
            L_ab = max(L_ab, 0)

            s = _cross_sum(Dtil[a], Dtil[b], 0)       # gamma_0 (unnormalized)
            for h in range(1, L_ab + 1):
                w = 1.0 - h / (L_ab + 1)
                g_ab = _cross_sum(Dtil[a], Dtil[b], h)  # gamma^{ab}_h
                g_ba = _cross_sum(Dtil[b], Dtil[a], h)  # gamma^{ba}_h
                s += w * (g_ab + g_ba)

            val = s / (n_ab ** 2)                      # single common divisor
            Omega[a, b] = Omega[b, a] = val

    Omega = (Omega + Omega.T) / 2.0
    Omega_psd = _nearest_psd(Omega)

    if not return_diagnostics:
        return Omega_psd

    eigvals = np.linalg.eigvalsh(Omega)
    diagnostics = {
        "Omega_raw": Omega,
        "frob_gap": float(np.linalg.norm(Omega - Omega_psd, ord="fro")),
        "min_eig_raw": float(eigvals.min()),
        "L_per_pair": L_per,
        "n_overlap": n_ab_mat,
        "pairs": pairs,
    }
    return Omega_psd, diagnostics


# ── Reference: stacked multivariate Newey--West (balanced only) ───────────────

def stacked_nw_balanced(X: np.ndarray, L: int) -> np.ndarray:
    """
    Stacked multivariate Newey--West estimator S_hat / n on a *balanced* panel
    (no NaN), with a single common Bartlett bandwidth ``L``:

        S_hat   = Gamma_0 + sum_{h=1}^{L} w_h (Gamma_h + Gamma_h^T),
        Gamma_h = (1/n) sum_t dtil_t dtil_{t-h}^T,   w_h = 1 - h/(L+1).

    This is the standard single-bandwidth multivariate NW (PSD by construction,
    Bartlett in Andrews' class K_2). Reference for the nesting test: with a
    common bandwidth, ``cov_via_omega(X, L=L)`` must reproduce ``S_hat / n``
    entrywise on a balanced panel. Raises if X contains NaN.
    """
    X = np.asarray(X, dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("stacked_nw_balanced requires a fully observed panel.")
    n, p = X.shape
    pairs = pair_index(p)

    D = np.column_stack([X[:, j] - X[:, k] for (j, k) in pairs])   # (n, q)
    Dtil = D - D.mean(axis=0, keepdims=True)
    L = min(int(L), n - 1)

    S = Dtil.T @ Dtil / n                            # Gamma_0
    for h in range(1, L + 1):
        w = 1.0 - h / (L + 1)
        Gamma_h = Dtil[h:].T @ Dtil[:-h] / n         # (q, q)
        S += w * (Gamma_h + Gamma_h.T)
    return S / n
