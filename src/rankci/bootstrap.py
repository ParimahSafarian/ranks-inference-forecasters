"""
Simultaneous bootstrap rank confidence intervals.

Complete-cases version: requires a fully observed (n, p) matrix.
Uses IID standard errors (no HAC).
"""
import numpy as np

from .pairwise import rank_ci_from_pairwise_ci


def rank_confidence_intervals_bootstrap(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 2000,
    seed: int | None = None,
) -> dict:
    """
    Bootstrap rank CIs — simultaneous two-sided CIs for all pairwise differences.

    Rows are i.i.d. joint observations; columns are populations.
    Resamples whole rows to preserve cross-column dependence.

    Returns
    -------
    dict with keys: theta_hat, pairwise_ci, rank_ci, critical_value.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n, p).")
    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least 2 rows.")

    theta_hat = X.mean(axis=0)
    delta_hat = theta_hat[:, None] - theta_hat[None, :]

    D = X[:, :, None] - X[:, None, :]
    se = D.std(axis=0, ddof=1) / np.sqrt(n)
    np.fill_diagonal(se, np.nan)

    # Bootstrap max statistics
    T_boot = np.empty(B)
    off_diag = ~np.eye(p, dtype=bool)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]

        theta_b = Xb.mean(axis=0)
        delta_b = theta_b[:, None] - theta_b[None, :]

        Db = Xb[:, :, None] - Xb[:, None, :]
        se_b = Db.std(axis=0, ddof=1) / np.sqrt(n)
        np.fill_diagonal(se_b, np.nan)

        Zb = (delta_b - delta_hat) / se_b
        T_boot[b] = np.nanmax(np.abs(Zb[off_diag]))

    critical_value = np.quantile(T_boot, 1 - alpha)

    lower = delta_hat - critical_value * se
    upper = delta_hat + critical_value * se
    pairwise_ci = np.stack([lower, upper], axis=-1)

    rank_ci = rank_ci_from_pairwise_ci(pairwise_ci, p)

    return {
        "theta_hat": theta_hat,
        "pairwise_ci": pairwise_ci,
        "rank_ci": rank_ci,
        "critical_value": critical_value,
    }
