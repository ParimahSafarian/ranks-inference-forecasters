"""
Stepwise simulation-based rank confidence intervals (Algorithm 3.2, Mogstad et al. 2024).

Uses the multivariate normal approximation instead of bootstrap resampling
to compute the critical value at each step.

Three variants:
  - rank_ci_stepwise_simulation:           complete-cases, IID SE
  - rank_ci_stepwise_simulation_pairwise:  unbalanced panel, optional NW-HAC SE
  - rank_ci_marginal_simulation_pairwise:  per-forecaster marginal CIs

Rank convention: rank 1 = smallest theta (best for MSE-style losses).
"""
import numpy as np

from .pairwise import (
    compute_pairwise,
    cov_theta_pairwise,
    rank_ci_from_rejections,
)


# ─── Shared simulation critical value ────────────────────────────────────────

def _simulation_cv(
    Sigma_hat: np.ndarray,
    se: np.ndarray,
    active_pairs,
    alpha: float,
    B: int,
    rng: np.random.Generator,
) -> float:
    """
    One-sided simulation critical value over active pairs.

    Draws Z ~ N(0, Sigma_hat) and computes
        T_b = max_{(j,k) in active}  (Z_j - Z_k) / se[j, k]
    then returns the (1-alpha) empirical quantile of {T_b}.

    Pairs with NaN or non-positive se are silently skipped.
    """
    p = Sigma_hat.shape[0]
    valid_pairs = [
        (j, k) for j, k in active_pairs
        if np.isfinite(se[j, k]) and se[j, k] > 0
    ]
    if not valid_pairs:
        return float("inf")

    Z = rng.multivariate_normal(np.zeros(p), Sigma_hat, size=B)
    T = np.empty(B)
    for b in range(B):
        T[b] = max(
            (Z[b, j] - Z[b, k]) / se[j, k]
            for j, k in valid_pairs
        )
    return float(np.quantile(T, 1 - alpha))


# ─── Complete-cases stepwise simulation ──────────────────────────────────────

def rank_ci_stepwise_simulation(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 20000,
    seed: int | None = None,
) -> dict:
    """
    Stepwise rank CIs — complete cases, IID standard errors, simulation-based.

    Parameters
    ----------
    X     : (n, p) array, no NaN.
    alpha : miscoverage level.
    B     : number of Gaussian simulation draws per stepwise iteration.
    seed  : random seed.

    Returns
    -------
    dict with: theta_hat, rank_ci, Sigma_hat.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = X.mean(axis=0)
    Sigma_hat = np.cov(X, rowvar=False, ddof=1) / n
    delta_hat = theta_hat[:, None] - theta_hat[None, :]

    D = X[:, :, None] - X[:, None, :]
    se = D.std(axis=0, ddof=1) / np.sqrt(n)
    np.fill_diagonal(se, np.nan)

    active = {(j, k) for j in range(p) for k in range(p) if j != k}
    rejected = set()

    while active:
        cv = _simulation_cv(Sigma_hat, se, list(active), alpha, B, rng)
        new_rejections = {
            (j, k) for (j, k) in active
            if delta_hat[j, k] - cv * se[j, k] > 0
        }
        if not new_rejections:
            break
        rejected |= new_rejections
        active -= new_rejections

    return {
        "theta_hat": theta_hat,
        "rank_ci": rank_ci_from_rejections(rejected, p),
        "Sigma_hat": Sigma_hat,
    }


# ─── Pairwise stepwise simulation (unbalanced panel) ─────────────────────────

def rank_ci_stepwise_simulation_pairwise(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 20000,
    seed: int | None = None,
    use_hac: bool = False,
    min_overlap: int = 2,
    verbose: bool = True,
) -> dict:
    """
    Stepwise rank CIs for unbalanced panels — simulation-based.

    Replaces the bootstrap resampling in rank_ci_stepwise_pairwise with
    draws from N(0, Sigma_hat), where Sigma_hat is estimated pairwise
    and projected to PSD.

    Parameters
    ----------
    X           : (n, p) array, may contain NaN.
    alpha       : miscoverage level (default 0.05).
    B           : simulation draws per stepwise iteration.
    seed        : random seed.
    use_hac     : if True, use Newey-West HAC SEs; else IID.
    min_overlap : minimum shared observations per pair.
    verbose     : print diagnostic summary.

    Returns
    -------
    dict with: theta_hat, rank_ci, Sigma_hat, n_overlap.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = np.nanmean(X, axis=0)
    se_method = "nw" if use_hac else "iid"
    delta_hat, se, n_overlap = compute_pairwise(
        X, se_method=se_method, min_overlap=min_overlap,
    )
    Sigma_hat = cov_theta_pairwise(X, min_overlap=min_overlap)

    if verbose:
        valid_n = n_overlap[n_overlap > 0]
        print("=== Pairwise shared observations ===")
        print(f"  Min: {valid_n.min()}, Mean: {valid_n.mean():.1f}, Max: {valid_n.max()}")
        with np.errstate(invalid="ignore"):
            t_stats = delta_hat / se
        vals = t_stats[np.isfinite(t_stats)]
        print(f"  Max t-stat: {vals.max():.3f}")

    active = {
        (j, k) for j in range(p) for k in range(p)
        if j != k and np.isfinite(se[j, k])
    }
    rejected = set()
    step = 0

    while active:
        step += 1
        cv = _simulation_cv(Sigma_hat, se, list(active), alpha, B, rng)
        new_rejections = {
            (j, k) for (j, k) in active
            if np.isfinite(delta_hat[j, k])
            and delta_hat[j, k] - cv * se[j, k] > 0
        }
        if verbose and new_rejections:
            print(f"  Step {step}: cv = {cv:.3f}, rejected {len(new_rejections)} pairs")

        if not new_rejections:
            if verbose:
                print(f"  Step {step}: cv = {cv:.3f}, no rejections — done.")
            break
        rejected |= new_rejections
        active -= new_rejections

    return {
        "theta_hat": theta_hat,
        "rank_ci": rank_ci_from_rejections(rejected, p),
        "Sigma_hat": Sigma_hat,
        "n_overlap": n_overlap,
    }


# ─── Marginal (per-forecaster) simulation CIs ────────────────────────────────

def rank_ci_marginal_simulation_pairwise(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 20000,
    seed: int | None = None,
    use_hac: bool = False,
    min_overlap: int = 2,
) -> dict:
    """
    Marginal (per-forecaster) rank CIs — simulation-based, pairwise data.

    For each forecaster j, the critical value is computed from the 2(p-1)
    one-sided test statistics involving j only. Tighter than the simultaneous
    procedure but does NOT control joint coverage.

    Rank convention: rank 1 = smallest theta.

    Returns
    -------
    dict with: theta_hat, rank_ci, critical_values, n_overlap.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = np.nanmean(X, axis=0)
    se_method = "nw" if use_hac else "iid"
    delta_hat, se, n_overlap = compute_pairwise(
        X, se_method=se_method, min_overlap=min_overlap,
    )
    Sigma_hat = cov_theta_pairwise(X, min_overlap=min_overlap)

    rank_ci = np.empty((p, 2), dtype=int)
    cvs = np.empty(p)

    for j in range(p):
        # All pairs involving j, both directions
        pairs_j = [
            (a, c) for a, c in
            ([(j, k) for k in range(p) if k != j]
             + [(k, j) for k in range(p) if k != j])
            if np.isfinite(se[a, c])
        ]
        cv_j = _simulation_cv(Sigma_hat, se, pairs_j, alpha, B, rng)
        cvs[j] = cv_j

        # (j, k): theta_j > theta_k confirmed → k is smaller → k BETTER than j
        n_better = sum(
            1 for k in range(p) if k != j
            and np.isfinite(se[j, k])
            and (delta_hat[j, k] - cv_j * se[j, k]) > 0
        )
        # (k, j): theta_k > theta_j confirmed → k is larger → k WORSE than j
        n_worse = sum(
            1 for k in range(p) if k != j
            and np.isfinite(se[k, j])
            and (delta_hat[k, j] - cv_j * se[k, j]) > 0
        )
        rank_ci[j] = [n_better + 1, p - n_worse]

    return {
        "theta_hat": theta_hat,
        "rank_ci": rank_ci,
        "critical_values": cvs,
        "n_overlap": n_overlap,
    }
