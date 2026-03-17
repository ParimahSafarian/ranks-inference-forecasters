import numpy as np


def rank_confidence_intervals_simulation(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 20000,
    seed: int | None = None,
):
    """
    Simulation-based rank confidence intervals for the case:
        X_i = (X_{i1}, ..., X_{ip}), i = 1,...,n

    Minimal version:
    - feature = column mean
    - asymptotic normal approximation
    - two-sided simultaneous CIs for all pairwise differences
    - rank CIs via the N^- / N^+ counting rule

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, p), rows are joint observations.
    alpha : float
        Miscoverage level, e.g. 0.05 for 95% CIs.
    B : int
        Number of simulation draws.
    seed : int | None
        Random seed.

    Returns
    -------
    dict with:
        theta_hat      : shape (p,)
        Sigma_hat      : estimated covariance of theta_hat, shape (p, p)
        pairwise_ci    : simultaneous CIs for all pairwise differences, shape (p, p, 2)
        rank_ci        : rank confidence intervals, shape (p, 2)
        critical_value : simulated max-stat critical value
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n, p).")

    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least 2 rows.")

    # 1) Estimate theta: here, column means
    theta_hat = X.mean(axis=0)  # shape (p,)

    # 2) Estimate covariance of theta_hat
    # If theta_hat is the sample mean vector, then:
    #   Cov(theta_hat) ≈ sample_cov(X) / n
    Sigma_hat = np.cov(X, rowvar=False, ddof=1) / n  # shape (p, p)

    # 3) Pairwise differences and pairwise SEs
    delta_hat = theta_hat[:, None] - theta_hat[None, :]  # shape (p, p)

    # Var(theta_j - theta_k) = Var(theta_j) + Var(theta_k) - 2 Cov(theta_j, theta_k)
    var_pair = (
        np.diag(Sigma_hat)[:, None]
        + np.diag(Sigma_hat)[None, :]
        - 2 * Sigma_hat
    )
    var_pair = np.maximum(var_pair, 0.0)  # numerical safety
    se_pair = np.sqrt(var_pair)
    np.fill_diagonal(se_pair, np.nan)  # diagonal not used

    # 4) Simulate Z ~ N(0, Sigma_hat)
    Z = rng.multivariate_normal(
        mean=np.zeros(p),
        cov=Sigma_hat,
        size=B,  # shape (B, p)
    )

    # 5) For each simulated draw, convert to pairwise standardized differences
    #    D_b[j,k] = (Z_b[j] - Z_b[k]) / se_pair[j,k]
    #    Then take max absolute value over all off-diagonal pairs.
    T = np.empty(B)
    off_diag = ~np.eye(p, dtype=bool)

    for b in range(B):
        D = Z[b][:, None] - Z[b][None, :]      # shape (p, p)
        std_D = D / se_pair                    # shape (p, p)
        T[b] = np.nanmax(np.abs(std_D[off_diag]))

    critical_value = np.quantile(T, 1 - alpha)

    # 6) Simultaneous pairwise CIs
    lower = delta_hat - critical_value * se_pair
    upper = delta_hat + critical_value * se_pair
    pairwise_ci = np.stack([lower, upper], axis=-1)  # shape (p, p, 2)

    # 7) Convert pairwise CIs into rank CIs
    # Nminus_j = number of k with CI(j,k) entirely below 0  -> k definitely better than j
    # Nplus_j  = number of k with CI(j,k) entirely above 0  -> j definitely better than k
    rank_ci = np.empty((p, 2), dtype=int)

    for j in range(p):
        n_minus = 0
        n_plus = 0
        for k in range(p):
            if j == k:
                continue

            ljk, ujk = pairwise_ci[j, k]

            if ujk < 0:
                n_minus += 1
            elif ljk > 0:
                n_plus += 1

        rank_lower = n_minus + 1
        rank_upper = p - n_plus
        rank_ci[j] = [rank_lower, rank_upper]

    return {
        "theta_hat": theta_hat,
        "Sigma_hat": Sigma_hat,
        "pairwise_ci": pairwise_ci,
        "rank_ci": rank_ci,
        "critical_value": critical_value,
    }


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    rng = np.random.default_rng(123)

    # Toy data: rows are joint observations, columns are populations
    n, p = 300, 5
    true_means = np.array([1.0, 0.9, 0.88, 0.7, 0.5])

    # Correlated data across populations
    cov = np.array([
        [1.0, 0.4, 0.3, 0.2, 0.1],
        [0.4, 1.0, 0.4, 0.2, 0.1],
        [0.3, 0.4, 1.0, 0.3, 0.1],
        [0.2, 0.2, 0.3, 1.0, 0.2],
        [0.1, 0.1, 0.1, 0.2, 1.0],
    ])

    X = rng.multivariate_normal(mean=true_means, cov=cov, size=n)

    out = rank_confidence_intervals_simulation(
        X,
        alpha=0.05,
        B=10000,
        seed=42,
    )

    print("Estimated means:")
    print(np.round(out["theta_hat"], 3))
    print()

    print("Rank confidence intervals [lower, upper]:")
    for j, ci in enumerate(out["rank_ci"], start=1):
        print(f"Population {j}: {ci.tolist()}")

    print()
    print("Simulated critical value:", round(out["critical_value"], 3))