"""
Shared pairwise utilities for rank confidence intervals.

- Pairwise mean differences and standard errors (IID and Newey-West HAC)
- Rank CI construction from pairwise CIs or rejection sets
"""
import numpy as np


# ── Newey-West HAC SE ────────────────────────────────────────────────────────

def nw_se(
    d: np.ndarray,
    L: int | None = None,
    winsor_pct: float | None = None,
) -> tuple[float, float]:
    """
    Newey-West HAC standard error for the mean of 1-D time series d.

    Parameters
    ----------
    d          : 1-D array of pairwise differences (no NaNs).
    L          : bandwidth. If None, uses the automatic rule
                 L = floor(4 * (n/100)^{2/9}).
    winsor_pct : if set (e.g. 95), symmetrically winsorize d at the
                 (100 - winsor_pct, winsor_pct) percentiles before
                 computing the mean and SE. None disables winsorization.

    Returns
    -------
    (mean, se) of the (possibly winsorized) series.
    """
    n = len(d)
    if L is None:
        L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    L = min(L, n - 1)

    if winsor_pct is not None:
        lo = np.percentile(d, 100 - winsor_pct)
        hi = np.percentile(d, winsor_pct)
        d = np.clip(d, lo, hi)

    mean = d.mean()
    dc = d - mean

    V = np.dot(dc, dc) / n                          # lag-0
    for tau in range(1, L + 1):
        w = 1.0 - tau / (L + 1)                     # Bartlett kernel
        V += 2.0 * w * np.dot(dc[tau:], dc[:-tau]) / n

    return mean, np.sqrt(max(V, 0.0) / n)


# ── Pairwise statistics ──────────────────────────────────────────────────────

def compute_pairwise(
    X: np.ndarray,
    se_method: str = "nw",
    L: int | None = None,
    winsor_pct: float | None = None,
    min_overlap: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pairwise mean differences and standard errors, handling NaN (unbalanced panels).

    For each pair (j, k), uses only periods where both j and k are observed.

    Parameters
    ----------
    X          : (n, p) array of observations, may contain NaN.
    se_method  : "nw" for Newey-West HAC, "iid" for plain std/sqrt(n).
    L          : NW bandwidth (ignored if se_method="iid").
    winsor_pct : if set (e.g. 95), symmetrically winsorize each pairwise
                 difference series at (100-pct, pct) percentiles before
                 computing the mean and SE. Only used when se_method="nw".
    min_overlap: minimum number of shared observations per pair (default 2).

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
            n_pairs[j, k] = n_jk
            if n_jk < max(min_overlap, 2):
                continue

            d = X[mask, j] - X[mask, k]

            if se_method == "nw":
                delta_hat[j, k], se_mat[j, k] = nw_se(d, L, winsor_pct)
            else:
                delta_hat[j, k] = d.mean()
                se_mat[j, k] = d.std(ddof=1) / np.sqrt(n_jk)

    return delta_hat, se_mat, n_pairs


# ── PSD projection and pairwise Cov(theta_hat) ──────────────────────────────

def _nearest_psd(A: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD matrix via eigenvalue clipping."""
    A = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    out = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return (out + out.T) / 2


def cov_theta_pairwise(
    X: np.ndarray,
    min_overlap: int = 2,
    se_method: str = "iid",
    L: int | None = None,
    se_pair: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate Cov(theta_hat) consistent with the chosen pairwise SE.

    Built via the Schoenberg / classical-MDS construction: treat
    `se_pair[j, k]^2` as squared distances and double-center,

        Sigma_hat = -1/2 * J @ S2 @ J,    J = I - 11^T / p, S2_jk = se_pair[j,k]^2,

    which by construction satisfies

        Var(theta_hat_j - theta_hat_k) = Sigma_jj + Sigma_kk - 2*Sigma_jk
                                       = se_pair[j, k]^2

    for every pair, while keeping the matrix PSD whenever the SE matrix is
    Euclidean-embeddable (and otherwise nearest-PSD-projected). This makes
    the simulation null Z ~ N(0, Sigma_hat) yield pairwise variances that
    EXACTLY match the studentization denominator — fixing the HAC/IID
    mismatch that would otherwise inflate simulation critical values.

    Parameters
    ----------
    X           : (n, p) data matrix, may contain NaN. Used only to determine p
                  and to compute se_pair internally when not provided.
    min_overlap : minimum shared observations per pair (used by internal
                  compute_pairwise call).
    se_method   : "iid" or "nw" — used only when se_pair is None.
    L           : NW bandwidth (None = auto). Used only when se_method="nw"
                  and se_pair is None.
    se_pair     : optional (p, p) pre-computed pairwise SE matrix. If None,
                  computed via `compute_pairwise(X, se_method=se_method,
                  L=L, min_overlap=min_overlap)`.

    Returns
    -------
    Sigma_hat : (p, p) PSD covariance matrix of theta_hat. Rank-deficient by
                construction (1^T Sigma_hat = 0); only pairwise contrasts are
                identified, which is all the simulation needs.
    """
    p = X.shape[1]

    if se_pair is None:
        _, se_pair, _ = compute_pairwise(
            X, se_method=se_method, L=L, min_overlap=min_overlap,
        )

    # Squared SE matrix with NaN/diagonal set to 0
    S2 = np.where(np.isfinite(se_pair), se_pair ** 2, 0.0)
    np.fill_diagonal(S2, 0.0)

    # Classical MDS / Schoenberg double-centering
    J = np.eye(p) - np.ones((p, p)) / p
    Sigma = -0.5 * (J @ S2 @ J)

    return _nearest_psd(Sigma)


# ── Rank CIs from pairwise CIs ──────────────────────────────────────────────

def rank_ci_from_pairwise_ci(pairwise_ci: np.ndarray, p: int) -> np.ndarray:
    """
    Convert (p, p, 2) pairwise confidence intervals to (p, 2) rank CIs.

    Ranks are in ASCENDING order of theta (rank 1 = smallest theta = best for MSE).

    CI(j,k) = [lower, upper] for delta_{j,k} = theta_j - theta_k.
      - lower > 0  =>  theta_j > theta_k confirmed  =>  k BETTER than j
      - upper < 0  =>  theta_j < theta_k confirmed  =>  k WORSE  than j

    Then  rank_ci[j] = [(# better than j) + 1, p - (# worse than j)].
    """
    rank_ci = np.empty((p, 2), dtype=int)
    for j in range(p):
        n_better = n_worse = 0
        for k in range(p):
            if j == k:
                continue
            ljk, ujk = pairwise_ci[j, k]
            if np.isnan(ljk) or np.isnan(ujk):
                continue
            if ljk > 0:
                n_better += 1
            elif ujk < 0:
                n_worse += 1
        rank_ci[j] = [n_better + 1, p - n_worse]
    return rank_ci


def rank_ci_from_rejections(rejected: set, p: int) -> np.ndarray:
    """
    Convert a set of rejected pairs {(j,k)} to (p, 2) rank CIs.

    Ranks are in ASCENDING order of theta (rank 1 = smallest theta = best for MSE).

    (j, k) in rejected  =>  theta_j > theta_k confirmed  =>  k BETTER than j
                                                              j WORSE  than k

    For forecaster j:
      n_better = |{k : (j, k) in rejected}|   (k confirmed better than j)
      n_worse  = |{k : (k, j) in rejected}|   (k confirmed worse  than j)
    """
    rank_ci = np.empty((p, 2), dtype=int)
    for j in range(p):
        n_better = sum(1 for k in range(p) if k != j and (j, k) in rejected)
        n_worse  = sum(1 for k in range(p) if k != j and (k, j) in rejected)
        rank_ci[j] = [n_better + 1, p - n_worse]
    return rank_ci
