import numpy as np


def rank_confidence_intervals_bootstrap(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 2000,
    seed: int | None = None,
):
    """
    Bootstrap rank confidence intervals for the case where each row is one joint observation:
        X_i = (X_{i1}, ..., X_{ip})

    Assumptions:
    - Feature of interest is the population mean of each column.
    - Rows are i.i.d.
    - Bootstrap resamples rows, preserving dependence across columns.

    Returns
    -------
    result : dict with keys
        theta_hat : original column means, shape (p,)
        pairwise_ci : simultaneous two-sided CIs for all pairwise differences, shape (p, p, 2)
        rank_ci : rank confidence intervals for each population, shape (p, 2)
        critical_value : bootstrap critical value for the max statistic
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n, p).")

    n, p = X.shape
    if n < 2:
        raise ValueError("Need at least 2 rows.")

    # 1) Original estimates: column means
    theta_hat = X.mean(axis=0)  # shape (p,)

    # 2) Pairwise estimated differences: delta_hat[j, k] = theta_hat[j] - theta_hat[k]
    delta_hat = theta_hat[:, None] - theta_hat[None, :]  # shape (p, p)

    # 3) Standard errors for pairwise mean differences
    #    For means, theta_hat[j] - theta_hat[k] is the mean of (X[:, j] - X[:, k])
    D = X[:, :, None] - X[:, None, :]  # shape (n, p, p)
    se = D.std(axis=0, ddof=1) / np.sqrt(n)  # shape (p, p)

    # Avoid division by zero on diagonal; diagonal is never used anyway
    np.fill_diagonal(se, np.nan)

    # 4) Bootstrap max statistics for symmetric simultaneous CIs
    T_boot = np.empty(B)
    off_diag = ~np.eye(p, dtype=bool)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx, :]

        theta_b = Xb.mean(axis=0)
        delta_b = theta_b[:, None] - theta_b[None, :]

        Db = Xb[:, :, None] - Xb[:, None, :]
        se_b = Db.std(axis=0, ddof=1) / np.sqrt(n)
        np.fill_diagonal(se_b, np.nan)

        # Standardized bootstrap pairwise errors
        Zb = (delta_b - delta_hat) / se_b
        T_boot[b] = np.nanmax(np.abs(Zb[off_diag]))

    critical_value = np.quantile(T_boot, 1 - alpha)

    # 5) Simultaneous pairwise confidence intervals
    lower = delta_hat - critical_value * se
    upper = delta_hat + critical_value * se
    pairwise_ci = np.stack([lower, upper], axis=-1)  # shape (p, p, 2)

    # 6) Convert pairwise CIs to rank CIs
    # Nminus_j = number of k such that CI(j,k) lies entirely below 0 -> k definitely better than j
    # Nplus_j  = number of k such that CI(j,k) lies entirely above 0 -> j definitely better than k
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
        "pairwise_ci": pairwise_ci,
        "rank_ci": rank_ci,
        "critical_value": critical_value,
    }


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Toy data: n rows = joint observations, p columns = populations
    n, p = 200, 5
    true_means = np.array([0.8, 0.7, 0.68, 0.55, 0.45])

    # Correlated columns to mimic the "joint observation" setup
    cov = np.array([
        [1.0, 0.4, 0.3, 0.2, 0.1],
        [0.4, 1.0, 0.4, 0.2, 0.1],
        [0.3, 0.4, 1.0, 0.3, 0.1],
        [0.2, 0.2, 0.3, 1.0, 0.2],
        [0.1, 0.1, 0.1, 0.2, 1.0],
    ])

    X = rng.multivariate_normal(mean=true_means, cov=cov, size=n)

    out = rank_confidence_intervals_bootstrap(X, alpha=0.05, B=1000, seed=1)

    print("Estimated means:")
    print(np.round(out["theta_hat"], 3))
    print()

    print("Rank confidence intervals [lower, upper]:")
    for j, ci in enumerate(out["rank_ci"], start=1):
        print(f"Population {j}: {ci.tolist()}")

    print()
    print("Bootstrap critical value:", round(out["critical_value"], 3))