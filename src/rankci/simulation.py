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

"""
Simulation-based (Gaussian) rank confidence intervals — pairwise-complete version.

Handles unbalanced panels with NaN entries. Each pairwise quantity (delta, se)
is computed on overlapping observations only. The joint covariance matrix Sigma_hat
is estimated pairwise and projected to the nearest PSD matrix if needed.
"""
import numpy as np


def _nearest_psd(A: np.ndarray) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest positive semi-definite matrix.
    Uses eigenvalue clipping: negative eigenvalues are set to zero.
    """
    A = (A + A.T) / 2  # ensure symmetry
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    out = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return (out + out.T) / 2


def _newey_west_var(d: np.ndarray) -> float:
    """
    Newey-West HAC long-run variance estimate for a 1-D series d.
    Bartlett kernel, automatic bandwidth: L = min(floor(4*(n/100)^{2/9}), n-1).

    Returns the estimated long-run variance V_hat (not divided by n).
    """
    n = len(d)
    d_demean = d - d.mean()
    L = min(int(np.floor(4 * (n / 100) ** (2 / 9))), n - 1)
    L = max(L, 0)

    # lag-0 autocovariance
    gamma0 = np.dot(d_demean, d_demean) / n

    V = gamma0
    for tau in range(1, L + 1):
        weight = 1 - tau / (L + 1)  # Bartlett weight
        gamma_tau = np.dot(d_demean[tau:], d_demean[:-tau]) / n
        V += 2 * weight * gamma_tau

    return max(V, 0.0)


def rank_ci_from_pairwise_ci(pairwise_ci: np.ndarray, p: int) -> np.ndarray:
    """N-/N+ counting rule: convert pairwise CIs to rank CIs."""
    rank_ci = np.empty((p, 2), dtype=int)
    for j in range(p):
        n_minus = 0
        n_plus = 0
        for k in range(p):
            if j == k:
                continue
            ljk, ujk = pairwise_ci[j, k]
            # skip pairs with NaN (insufficient overlap)
            if np.isnan(ljk) or np.isnan(ujk):
                continue
            if ujk < 0:
                n_minus += 1
            elif ljk > 0:
                n_plus += 1
        rank_ci[j] = [n_minus + 1, p - n_plus]
    return rank_ci


def rank_confidence_intervals_simulation_pairwise(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 20000,
    seed: int | None = None,
    use_hac: bool = False,
    min_overlap: int = 2,
) -> dict:
    """
    Simulation-based rank CIs via multivariate normal approximation.
    Pairwise-complete version: handles NaN entries in X.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, p). May contain NaN.
    alpha : float
        Miscoverage level (default 0.05 for 95% CIs).
    B : int
        Number of simulation draws.
    seed : int or None
        Random seed.
    use_hac : bool
        If True, use Newey-West HAC standard errors for each pair.
        If False, use i.i.d. standard errors (std / sqrt(n_jk)).
    min_overlap : int
        Minimum number of shared observations required per pair.

    Returns
    -------
    dict with keys:
        theta_hat      : (p,) column means (using all available obs per column)
        Sigma_hat      : (p, p) estimated covariance of theta_hat (PSD-projected)
        pairwise_ci    : (p, p, 2) simultaneous pairwise CIs
        rank_ci        : (p, 2) rank confidence intervals
        critical_value : simulated critical value
        n_overlap      : (p, p) number of overlapping observations per pair
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n, p).")
    n, p = X.shape

    # ── Step 1: Estimate theta_hat (column means, ignoring NaN) ──────────
    theta_hat = np.nanmean(X, axis=0)  # shape (p,)

    # ── Step 2: Pairwise delta_hat and se_pair ───────────────────────────
    # Each pair uses only overlapping observations (matching thesis eq. 11)
    delta_hat = np.full((p, p), np.nan)
    se_pair = np.full((p, p), np.nan)
    n_overlap = np.zeros((p, p), dtype=int)

    for j in range(p):
        for k in range(p):
            if j == k:
                continue
            mask = np.isfinite(X[:, j]) & np.isfinite(X[:, k])
            njk = mask.sum()
            n_overlap[j, k] = njk

            if njk < min_overlap:
                continue

            d_jk = X[mask, j] - X[mask, k]
            delta_hat[j, k] = d_jk.mean()

            if use_hac:
                V_hat = _newey_west_var(d_jk)
                se_pair[j, k] = np.sqrt(max(V_hat, 0.0) / njk)
            else:
                se_pair[j, k] = d_jk.std(ddof=1) / np.sqrt(njk)

    # ── Step 3: Build Sigma_hat (covariance of theta_hat) pairwise ───────
    #
    # Under i.i.d. rows:
    #   Cov(theta_hat_j, theta_hat_k) = (n_jk / (n_j * n_k)) * sigma_jk
    #
    # where sigma_jk = Cov(X_j, X_k), estimated from overlapping obs.
    #
    # For the diagonal:
    #   Var(theta_hat_j) = Var(X_j) / n_j
    #
    Sigma_hat = np.zeros((p, p))

    for j in range(p):
        mask_j = np.isfinite(X[:, j])
        nj = mask_j.sum()
        if nj < 2:
            continue
        Sigma_hat[j, j] = np.var(X[mask_j, j], ddof=1) / nj

        for k in range(j + 1, p):
            mask_jk = np.isfinite(X[:, j]) & np.isfinite(X[:, k])
            njk = mask_jk.sum()
            if njk < min_overlap:
                continue

            nk = np.isfinite(X[:, k]).sum()

            # Estimate population covariance from overlapping obs
            sigma_jk = np.cov(X[mask_jk, j], X[mask_jk, k], ddof=1)[0, 1]

            # Covariance of the means: only overlapping obs contribute
            Sigma_hat[j, k] = (njk / (nj * nk)) * sigma_jk
            Sigma_hat[k, j] = Sigma_hat[j, k]

    # ── Step 4: Project Sigma_hat to nearest PSD ─────────────────────────
    Sigma_hat = _nearest_psd(Sigma_hat)

    # ── Step 5: Simulate Z ~ N(0, Sigma_hat) ────────────────────────────
    Z = rng.multivariate_normal(np.zeros(p), Sigma_hat, size=B)

    # ── Step 6: Max-statistic over valid pairs ───────────────────────────
    valid = np.isfinite(se_pair) & (se_pair > 0)
    T = np.empty(B)

    for b in range(B):
        D = Z[b][:, None] - Z[b][None, :]
        std_D = np.full((p, p), np.nan)
        std_D[valid] = D[valid] / se_pair[valid]
        T[b] = np.nanmax(np.abs(std_D))

    critical_value = float(np.quantile(T, 1 - alpha))

    # ── Step 7: Simultaneous pairwise CIs ────────────────────────────────
    lower = delta_hat - critical_value * se_pair
    upper = delta_hat + critical_value * se_pair
    pairwise_ci = np.stack([lower, upper], axis=-1)

    # ── Step 8: Rank CIs via N-/N+ counting ──────────────────────────────
    rank_ci = rank_ci_from_pairwise_ci(pairwise_ci, p)

    return {
        "theta_hat": theta_hat,
        "Sigma_hat": Sigma_hat,
        "pairwise_ci": pairwise_ci,
        "rank_ci": rank_ci,
        "critical_value": critical_value,
        "n_overlap": n_overlap,
    }


# ─── Example / smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(123)

    # Generate complete data, then punch some NaN holes
    n, p = 200, 5
    true_means = np.array([1.0, 0.9, 0.88, 0.7, 0.5])
    cov = np.array([
        [1.0, 0.4, 0.3, 0.2, 0.1],
        [0.4, 1.0, 0.4, 0.2, 0.1],
        [0.3, 0.4, 1.0, 0.3, 0.1],
        [0.2, 0.2, 0.3, 1.0, 0.2],
        [0.1, 0.1, 0.1, 0.2, 1.0],
    ])
    X = rng.multivariate_normal(mean=true_means, cov=cov, size=n)

    # Introduce ~15% missing at random
    miss = rng.random(X.shape) < 0.15
    X[miss] = np.nan
    print(f"Missing rate: {np.isnan(X).mean():.1%}")

    out = rank_confidence_intervals_simulation_pairwise(
        X, alpha=0.05, B=10000, seed=42, use_hac=False,
    )

    print("\nEstimated means:")
    print(np.round(out["theta_hat"], 3))

    print("\nRank confidence intervals [lower, upper]:")
    for j, ci in enumerate(out["rank_ci"], start=1):
        print(f"  Population {j}: {ci.tolist()}")

    print(f"\nSimulated critical value: {out['critical_value']:.3f}")

    print("\nOverlap matrix (n_jk):")
    print(out["n_overlap"])