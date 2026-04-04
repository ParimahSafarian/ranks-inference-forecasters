import numpy as np
import pandas as pd


def _compute_se_pairwise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pairwise se[j,k] = std(X[:,j] - X[:,k], over periods where both observed) / sqrt(n_jk).
    Also returns:
      - delta_hat[j,k]: pairwise mean difference (theta_j - theta_k)
      - n_pairs[j,k]:   number of periods used for each pair
    Diagonal is nan.
    """
    p = X.shape[1]
    se        = np.full((p, p), np.nan)
    delta_hat = np.full((p, p), np.nan)
    n_pairs   = np.zeros((p, p), dtype=int)

    for j in range(p):
        for k in range(p):
            if j == k:
                continue
            mask = ~np.isnan(X[:, j]) & ~np.isnan(X[:, k])  # periods both observed
            n_jk = mask.sum()
            if n_jk < 2:
                continue  # not enough data for this pair
            diff = X[mask, j] - X[mask, k]
            delta_hat[j, k] = diff.mean()
            se[j, k]        = diff.std(ddof=1) / np.sqrt(n_jk)
            n_pairs[j, k]   = n_jk

    return delta_hat, se, n_pairs

def _nw_se(d: np.ndarray, L: int | None = None) -> float:
    """
    Newey-West HAC standard error for the mean of 1D time series d.
    
    Parameters
    ----------
    d : np.ndarray
        1D array of pairwise differences (already masked for non-nan).
    L : int | None
        Bandwidth (number of lags). If None, uses automatic rule:
        L = floor(4 * (n/100)^(2/9))  — standard in the HAC literature.
    
    Returns
    -------
    float : HAC standard error of the mean d̄.
    """
    n = len(d)
    if L is None:
        L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    L = min(L, n - 1)                        # can never exceed n-1

    d_centered = d - d.mean()

    # Lag-0 autocovariance (plain variance)
    V = np.dot(d_centered, d_centered) / n

    # Add Bartlett-weighted autocovariances for lags 1..L
    for tau in range(1, L + 1):
        weight    = 1.0 - tau / (L + 1)      # Bartlett kernel — guarantees V > 0
        gamma_tau = np.dot(d_centered[tau:], d_centered[:-tau]) / n
        V        += 2.0 * weight * gamma_tau

    return np.sqrt(max(V, 0.0) / n)          # max guards against numerical negatives


def _compute_se_nw(X: np.ndarray, L: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pairwise Newey-West SE and mean difference matrix.

    For each pair (j,k):
      - finds periods where both j and k are observed
      - computes d_t = X[t,j] - X[t,k] over those periods
      - estimates delta_hat[j,k] = mean(d)
      - estimates se[j,k]        = NW HAC se of mean(d)

    Parameters
    ----------
    X : np.ndarray of shape (n, p)
        Panel of squared errors. May contain NaNs.
    L : int | None
        NW bandwidth. If None, automatic per pair based on n_jk.

    Returns
    -------
    delta_hat : (p, p) array of mean differences
    se        : (p, p) array of NW standard errors
    n_pairs   : (p, p) int array of shared observations per pair
    """
    p       = X.shape[1]
    se        = np.full((p, p), np.nan)
    delta_hat = np.full((p, p), np.nan)
    n_pairs   = np.zeros((p, p), dtype=int)

    for j in range(p):
        for k in range(p):
            if j == k:
                continue
            mask = ~np.isnan(X[:, j]) & ~np.isnan(X[:, k])
            n_jk = mask.sum()
            if n_jk < 2:
                continue

            d               = X[mask, j] - X[mask, k]
            delta_hat[j, k] = d.mean()
            se[j, k]        = _nw_se(d, L)   # ← NW instead of std/sqrt(n)
            n_pairs[j, k]   = n_jk

    return delta_hat, se, n_pairs

def _bootstrap_cv_pairwise(
    X, delta_hat, se, active_pairs, alpha, B, rng
) -> float:
    """
    Bootstrap critical value using pairwise resampling.
    For each pair (j,k), resample only from periods where both are observed.
    """
    p = X.shape[1]
    T = np.empty(B)

    for b in range(B):
        max_stat = -np.inf
        for j, k in active_pairs:
            mask = ~np.isnan(X[:, j]) & ~np.isnan(X[:, k])
            n_jk = mask.sum()
            if n_jk < 2:
                continue
            X_pair = X[mask][:, [j, k]]              # (n_jk, 2) — only complete periods for this pair
            idx    = rng.integers(0, n_jk, size=n_jk) # resample within that pair's periods
            diff_b = X_pair[idx, 0] - X_pair[idx, 1]
            delta_b_jk = diff_b.mean()
            stat = (delta_b_jk - delta_hat[j, k]) / se[j, k]
            if stat > max_stat:
                max_stat = stat
        T[b] = max_stat

    return float(np.quantile(T, 1 - alpha))


def rank_ci_stepwise_pairwise(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 5000,
    seed: int | None = None,
) -> dict:
    """Stepwise simultaneous rank CIs with pairwise missing-data handling."""
    rng = np.random.default_rng(seed)
    X   = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat             = np.nanmean(X, axis=0)          # each forecaster's marginal mean MSE
    delta_hat, se, n_pairs = _compute_se_nw(X, L=None) #_compute_se_pairwise(X) # before nw

    #####################################
    # ── Diagnostic 1: pairwise overlap ───────────────────────────────────────────
    valid_pairs = n_pairs[n_pairs > 0]
    print("=== Pairwise shared observations ===")
    print(f"Min  : {valid_pairs.min()}")
    print(f"Mean : {valid_pairs.mean():.1f}")
    print(f"Max  : {valid_pairs.max()}")
    print(f"Pairs with < 10 shared obs: {(valid_pairs < 10).sum()}")
    print(f"Pairs with < 20 shared obs: {(valid_pairs < 20).sum()}")

    # ── Diagnostic 2: test statistics vs critical value ──────────────────────────
    with np.errstate(invalid='ignore'):
        t_stats = delta_hat / se                      # signed t-statistics
    np.fill_diagonal(t_stats, np.nan)

    print("\n=== Test statistics (delta_hat / se) ===")
    vals = t_stats.flatten()
    vals = vals[~np.isnan(vals)]
    print(f"Min  : {vals.min():.4f}")
    print(f"Max  : {vals.max():.4f}")
    print(f"Mean : {vals.mean():.4f}")
    print(f"Pairs with t > 1.96 : {(vals > 1.96).sum()}")
    print(f"Pairs with t > 1.0  : {(vals > 1.0).sum()}")

    # ── Diagnostic 3: bootstrap critical value ───────────────────────────────────
    rng_diag = np.random.default_rng(42)
    active_diag = [(j,k) for j in range(p) for k in range(p)
                if j != k and not np.isnan(se[j,k])]
    cv_diag = _bootstrap_cv_pairwise(X, delta_hat, se, active_diag,
                                    alpha=0.05, B=500, rng=rng_diag)
    print(f"\n=== Bootstrap critical value (B=500) ===")
    print(f"CV : {cv_diag:.4f}")
    print(f"Max test stat : {vals.max():.4f}")
    print(f"Gap (CV - max_stat) : {cv_diag - vals.max():.4f}")
    #####################################


    active   = {(j, k) for j in range(p) for k in range(p)
                if j != k and not np.isnan(se[j, k])}      # only pairs with enough shared data
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
        active   -= new_rejections

    rank_ci = np.empty((p, 2), dtype=int)
    for j in range(p):
        rej_minus = sum(1 for k in range(p) if k != j and (k, j) in rejected)
        rej_plus  = sum(1 for k in range(p) if k != j and (j, k) in rejected)
        rank_ci[j] = [rej_minus + 1, p - rej_plus]

    return {
        "theta_hat": theta_hat,
        "rank_ci":   rank_ci,
        "n_pairs":   n_pairs,    # useful diagnostic: how many periods back each comparison
    }