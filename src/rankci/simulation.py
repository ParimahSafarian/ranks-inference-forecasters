"""
Simulation-based (Gaussian) rank confidence intervals.

Two variants:
  - rank_confidence_intervals_simulation:          complete-cases (no NaN)
  - rank_confidence_intervals_simulation_pairwise: unbalanced panels (NaN OK)

Both use the asymptotic normal approximation with an estimated covariance:
draw Z ~ N(0, Sigma_hat) B times, compute the supremum of standardized
pairwise differences, and take its (1-alpha) quantile as the critical value.

Rank convention: rank 1 = smallest theta (best for MSE-style losses).
"""
import numpy as np

from .pairwise import (
    compute_pairwise,
    cov_theta_pairwise,
    rank_ci_from_pairwise_ci,
)


def rank_confidence_intervals_simulation(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 20000,
    seed: int | None = None,
) -> dict:
    """
    Simulation-based rank CIs via the multivariate normal approximation.

    Complete-cases version: requires a fully observed (n, p) matrix.

    Returns
    -------
    dict with keys: theta_hat, Sigma_hat, pairwise_ci, rank_ci, critical_value.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n, p).")
    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least 2 rows.")

    theta_hat = X.mean(axis=0)
    Sigma_hat = np.cov(X, rowvar=False, ddof=1) / n

    delta_hat = theta_hat[:, None] - theta_hat[None, :]

    var_pair = (
        np.diag(Sigma_hat)[:, None]
        + np.diag(Sigma_hat)[None, :]
        - 2 * Sigma_hat
    )
    var_pair = np.maximum(var_pair, 0.0)
    se_pair = np.sqrt(var_pair)
    np.fill_diagonal(se_pair, np.nan)

    Z = rng.multivariate_normal(np.zeros(p), Sigma_hat, size=B)

    T = np.empty(B)
    off_diag = ~np.eye(p, dtype=bool)
    for b in range(B):
        D = Z[b][:, None] - Z[b][None, :]
        std_D = D / se_pair
        T[b] = np.nanmax(np.abs(std_D[off_diag]))

    critical_value = float(np.quantile(T, 1 - alpha))

    lower = delta_hat - critical_value * se_pair
    upper = delta_hat + critical_value * se_pair
    pairwise_ci = np.stack([lower, upper], axis=-1)

    rank_ci = rank_ci_from_pairwise_ci(pairwise_ci, p)

    return {
        "theta_hat": theta_hat,
        "Sigma_hat": Sigma_hat,
        "pairwise_ci": pairwise_ci,
        "rank_ci": rank_ci,
        "critical_value": critical_value,
    }


def rank_confidence_intervals_simulation_pairwise(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 20000,
    seed: int | None = None,
    use_hac: bool = False,
    min_overlap: int = 2,
) -> dict:
    """
    Simulation-based rank CIs via the multivariate normal approximation,
    pairwise-complete version (handles NaN entries in X).

    Each pairwise quantity (delta_hat, se) is computed on overlapping
    observations only; Sigma_hat is estimated entry-by-entry from overlaps
    and projected to the nearest PSD matrix.

    Parameters
    ----------
    X           : (n, p) array, may contain NaN.
    alpha       : miscoverage level (default 0.05 for 95% CIs).
    B           : number of simulation draws.
    seed        : random seed.
    use_hac     : if True, use Newey-West HAC SEs for each pair; else IID.
    min_overlap : minimum number of shared observations required per pair.

    Returns
    -------
    dict with keys:
        theta_hat      : (p,) column means (using all available obs per column)
        Sigma_hat      : (p, p) estimated covariance of theta_hat (PSD-projected)
        pairwise_ci    : (p, p, 2) simultaneous pairwise CIs
        rank_ci        : (p, 2) rank confidence intervals (rank 1 = smallest theta)
        critical_value : simulated critical value
        n_overlap      : (p, p) number of overlapping observations per pair
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n, p).")
    p = X.shape[1]

    theta_hat = np.nanmean(X, axis=0)

    se_method = "nw" if use_hac else "iid"
    delta_hat, se_pair, n_overlap = compute_pairwise(
        X, se_method=se_method, min_overlap=min_overlap,
    )
    Sigma_hat = cov_theta_pairwise(X, min_overlap=min_overlap)

    Z = rng.multivariate_normal(np.zeros(p), Sigma_hat, size=B)

    valid = np.isfinite(se_pair) & (se_pair > 0)
    T = np.empty(B)
    for b in range(B):
        D = Z[b][:, None] - Z[b][None, :]
        std_D = np.full((p, p), np.nan)
        std_D[valid] = D[valid] / se_pair[valid]
        T[b] = np.nanmax(np.abs(std_D))

    critical_value = float(np.quantile(T, 1 - alpha))

    lower = delta_hat - critical_value * se_pair
    upper = delta_hat + critical_value * se_pair
    pairwise_ci = np.stack([lower, upper], axis=-1)

    rank_ci = rank_ci_from_pairwise_ci(pairwise_ci, p)

    return {
        "theta_hat": theta_hat,
        "Sigma_hat": Sigma_hat,
        "pairwise_ci": pairwise_ci,
        "rank_ci": rank_ci,
        "critical_value": critical_value,
        "n_overlap": n_overlap,
    }
