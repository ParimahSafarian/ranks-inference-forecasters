"""
Shared pairwise utilities for rank confidence intervals.

- Pairwise mean differences and standard errors (IID and Newey-West HAC)
- Rank CI construction from pairwise CIs or rejection sets
"""
import numpy as np


# ── Newey-West HAC SE ────────────────────────────────────────────────────────

def nw_se(d: np.ndarray, L: int | None = None) -> float:
    """
    Newey-West HAC standard error for the mean of 1-D time series d.

    Parameters
    ----------
    d : 1-D array of pairwise differences (no NaNs).
    L : bandwidth (number of lags).  If None, uses the automatic rule
        L = floor(4 * (n/100)^{2/9}).
    """
    n = len(d)
    if L is None:
        L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    L = min(L, n - 1)

    dc = d - d.mean()

    V = np.dot(dc, dc) / n                          # lag-0
    for tau in range(1, L + 1):
        w = 1.0 - tau / (L + 1)                     # Bartlett kernel
        V += 2.0 * w * np.dot(dc[tau:], dc[:-tau]) / n

    return np.sqrt(max(V, 0.0) / n)


# ── Pairwise statistics ──────────────────────────────────────────────────────

def compute_pairwise(
    X: np.ndarray,
    se_method: str = "nw",
    L: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pairwise mean differences and standard errors, handling NaN (unbalanced panels).

    For each pair (j, k), uses only periods where both j and k are observed.

    Parameters
    ----------
    X         : (n, p) array of observations, may contain NaN.
    se_method : "nw" for Newey-West HAC, "iid" for plain std/sqrt(n).
    L         : NW bandwidth (ignored if se_method="iid").

    Returns
    -------
    delta_hat : (p, p) mean differences.
    se        : (p, p) standard errors.
    n_pairs   : (p, p) int array of shared observation counts.
    """
    p = X.shape[1]
    se_mat    = np.full((p, p), np.nan)
    delta_hat = np.full((p, p), np.nan)
    n_pairs   = np.zeros((p, p), dtype=int)

    for j in range(p):
        for k in range(p):
            if j == k:
                continue
            mask = ~np.isnan(X[:, j]) & ~np.isnan(X[:, k])
            n_jk = mask.sum()
            if n_jk < 2:
                continue

            d = X[mask, j] - X[mask, k]
            delta_hat[j, k] = d.mean()
            n_pairs[j, k]   = n_jk

            if se_method == "nw":
                se_mat[j, k] = nw_se(d, L)
            else:
                se_mat[j, k] = d.std(ddof=1) / np.sqrt(n_jk)

    return delta_hat, se_mat, n_pairs


# ── Rank CIs from pairwise CIs ──────────────────────────────────────────────

def rank_ci_from_pairwise_ci(pairwise_ci: np.ndarray, p: int) -> np.ndarray:
    """
    Convert (p, p, 2) pairwise confidence intervals to (p, 2) rank CIs.

    CI(j,k) = [lower, upper] for delta_{j,k} = theta_j - theta_k.
      - upper < 0  =>  k is better than j   (count toward n_minus)
      - lower > 0  =>  j is better than k   (count toward n_plus)
    """
    rank_ci = np.empty((p, 2), dtype=int)
    for j in range(p):
        n_minus = n_plus = 0
        for k in range(p):
            if j == k:
                continue
            ljk, ujk = pairwise_ci[j, k]
            if np.isnan(ljk) or np.isnan(ujk):
                continue
            if ujk < 0:
                n_minus += 1
            elif ljk > 0:
                n_plus += 1
        rank_ci[j] = [n_minus + 1, p - n_plus]
    return rank_ci


def rank_ci_from_rejections(rejected: set, p: int) -> np.ndarray:
    """
    Convert a set of rejected pairs {(j,k)} to (p, 2) rank CIs.

    (j, k) in rejected  =>  theta_j > theta_k  (j confirmed worse than k).
    """
    rank_ci = np.empty((p, 2), dtype=int)
    for j in range(p):
        rej_minus = sum(1 for k in range(p) if k != j and (k, j) in rejected)
        rej_plus  = sum(1 for k in range(p) if k != j and (j, k) in rejected)
        rank_ci[j] = [rej_minus + 1, p - rej_plus]
    return rank_ci
