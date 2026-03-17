import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _compute_se(X: np.ndarray) -> np.ndarray:
    """se[j,k] = std(X[:,j] - X[:,k]) / sqrt(n). Diagonal is nan."""
    n = X.shape[0]
    D = X[:, :, None] - X[:, None, :]
    se = D.std(axis=0, ddof=1) / np.sqrt(n)
    np.fill_diagonal(se, np.nan)
    return se


def _bootstrap_cv(
    X, theta_hat, se, active_pairs, alpha, B, rng
) -> float:
    """One-sided bootstrap critical value over active pairs only."""
    n = X.shape[0]
    T = np.empty(B)
    for b in range(B):
        idx     = rng.integers(0, n, size=n)
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
    """Stepwise simultaneous rank CIs — Algorithm 3.2, Mogstad et al. (2024)."""
    rng = np.random.default_rng(seed)
    X   = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = X.mean(axis=0)
    se        = _compute_se(X)
    delta_hat = theta_hat[:, None] - theta_hat[None, :]

    active   = {(k, l) for k in range(p) for l in range(p) if k != l}
    rejected = set()

    while active:
        cv = _bootstrap_cv(X, theta_hat, se, list(active), alpha, B, rng)

        new_rejections = {
            (k, l) for (k, l) in active
            if delta_hat[k, l] - cv * se[k, l] > 0
        }

        if not new_rejections:
            break

        rejected |= new_rejections
        active   -= new_rejections

    rank_ci = np.empty((p, 2), dtype=int)
    for j in range(p):
        rej_minus = sum(1 for k in range(p) if k != j and (k, j) in rejected)
        rej_plus  = sum(1 for k in range(p) if k != j and (j, k) in rejected)
        rank_ci[j] = [rej_minus + 1, p - rej_plus]

    return {"theta_hat": theta_hat, "rank_ci": rank_ci}

