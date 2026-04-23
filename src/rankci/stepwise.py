"""
Stepwise bootstrap rank confidence intervals (Algorithm 3.2, Mogstad et al. 2024).

Two variants:
  - rank_ci_stepwise:          complete-cases, IID SE
  - rank_ci_stepwise_pairwise: unbalanced panel, NW-HAC SE, pairwise resampling
"""
import numpy as np

from .pairwise import (
    compute_pairwise,
    rank_ci_from_rejections,
)


# ── Complete-cases stepwise ──────────────────────────────────────────────────

def _compute_se_complete(X: np.ndarray) -> np.ndarray:
    """IID pairwise SE for a complete (no NaN) matrix."""
    n = X.shape[0]
    D = X[:, :, None] - X[:, None, :]
    se = D.std(axis=0, ddof=1) / np.sqrt(n)
    np.fill_diagonal(se, np.nan)
    return se


def _bootstrap_cv_complete(X, theta_hat, se, active_pairs, alpha, B, rng):
    """One-sided bootstrap critical value — complete-cases."""
    n = X.shape[0]
    T = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        theta_b = X[idx].mean(axis=0)
        T[b] = max(
            ((theta_b[k] - theta_b[l]) - (theta_hat[k] - theta_hat[l])) / se[k, l]
            for k, l in active_pairs
        )
    return float(np.quantile(T, 1 - alpha))


def rank_ci_stepwise(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 5000,
    seed: int | None = None,
) -> dict:
    """Stepwise rank CIs — complete cases, IID standard errors."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = X.mean(axis=0)
    se = _compute_se_complete(X)
    delta_hat = theta_hat[:, None] - theta_hat[None, :]

    active = {(k, l) for k in range(p) for l in range(p) if k != l}
    rejected = set()

    while active:
        cv = _bootstrap_cv_complete(X, theta_hat, se, list(active), alpha, B, rng)
        new_rejections = {
            (k, l) for (k, l) in active
            if delta_hat[k, l] - cv * se[k, l] > 0
        }
        if not new_rejections:
            break
        rejected |= new_rejections
        active -= new_rejections

    return {
        "theta_hat": theta_hat,
        "rank_ci": rank_ci_from_rejections(rejected, p),
    }


# ── Pairwise stepwise (unbalanced panel, NW-HAC) ────────────────────────────

def _bootstrap_cv_pairwise(X, delta_hat, se, active_pairs, alpha, B, rng):
    """Bootstrap critical value with pairwise resampling for unbalanced panels."""
    T = np.empty(B)
    for b in range(B):
        max_stat = -np.inf
        for j, k in active_pairs:
            mask = ~np.isnan(X[:, j]) & ~np.isnan(X[:, k])
            n_jk = mask.sum()
            if n_jk < 2:
                continue
            X_pair = X[mask][:, [j, k]]
            idx = rng.integers(0, n_jk, size=n_jk)
            diff_b = X_pair[idx, 0] - X_pair[idx, 1]
            stat = (diff_b.mean() - delta_hat[j, k]) / se[j, k]
            if stat > max_stat:
                max_stat = stat
        T[b] = max_stat
    return float(np.quantile(T, 1 - alpha))


def rank_ci_stepwise_pairwise(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 5000,
    seed: int | None = None,
    se_method: str = "nw",
    verbose: bool = True,
) -> dict:
    """
    Stepwise rank CIs for unbalanced panels.

    Uses pairwise complete observations and (by default) Newey-West HAC SEs.

    Parameters
    ----------
    X         : (n, p) array, may contain NaN.
    se_method : "nw" for Newey-West HAC, "iid" for plain SE.
    verbose   : print diagnostic summary.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = np.nanmean(X, axis=0)
    delta_hat, se, n_pairs = compute_pairwise(X, se_method=se_method)

    if verbose:
        valid = n_pairs[n_pairs > 0]
        print("=== Pairwise shared observations ===")
        print(f"  Min: {valid.min()}, Mean: {valid.mean():.1f}, Max: {valid.max()}")
        print(f"  Pairs with < 20 shared obs: {(valid < 20).sum()}")

        with np.errstate(invalid="ignore"):
            t_stats = delta_hat / se
        vals = t_stats[~np.isnan(t_stats)]
        print(f"\n=== Test statistics (delta_hat / se) ===")
        print(f"  Max: {vals.max():.4f}, Pairs with t > 1.96: {(vals > 1.96).sum()}")

    active = {
        (j, k) for j in range(p) for k in range(p)
        if j != k and not np.isnan(se[j, k])
    }
    rejected = set()

    while active:
        cv = _bootstrap_cv_pairwise(X, delta_hat, se, list(active), alpha, B, rng)
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
        "n_pairs": n_pairs,
    }
