"""
Simulation-based (Gaussian) rank confidence intervals.

Complete-cases version: requires a fully observed (n, p) matrix.
Uses the asymptotic normal approximation with estimated covariance.
"""
import numpy as np

from .pairwise import rank_ci_from_pairwise_ci


def rank_confidence_intervals_simulation(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 20000,
    seed: int | None = None,
) -> dict:
    """
    Simulation-based rank CIs via the multivariate normal approximation.

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

    # Simulate Z ~ N(0, Sigma_hat)
    Z = rng.multivariate_normal(np.zeros(p), Sigma_hat, size=B)

    T = np.empty(B)
    off_diag = ~np.eye(p, dtype=bool)
    for b in range(B):
        D = Z[b][:, None] - Z[b][None, :]
        std_D = D / se_pair
        T[b] = np.nanmax(np.abs(std_D[off_diag]))

    critical_value = np.quantile(T, 1 - alpha)

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
