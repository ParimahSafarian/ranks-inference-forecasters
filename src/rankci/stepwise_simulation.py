"""
Stepwise simulation-based rank confidence intervals (Algorithm 3.2, Mogstad et al. 2024).

Uses the multivariate normal approximation instead of bootstrap resampling
to compute the critical value at each step.

Two variants:
  - rank_ci_stepwise_simulation:          complete-cases, IID SE
  - rank_ci_stepwise_simulation_pairwise: unbalanced panel, optional NW-HAC SE
"""
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _nearest_psd(A: np.ndarray) -> np.ndarray:
    """Project symmetric matrix to nearest positive semi-definite (eigenvalue clipping)."""
    A = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    out = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return (out + out.T) / 2


def _newey_west_var(d: np.ndarray) -> float:
    """Newey-West HAC long-run variance (Bartlett kernel, automatic bandwidth)."""
    n = len(d)
    d_demean = d - d.mean()
    L = min(int(np.floor(4 * (n / 100) ** (2 / 9))), n - 1)
    L = max(L, 0)
    gamma0 = np.dot(d_demean, d_demean) / n
    V = gamma0
    for tau in range(1, L + 1):
        weight = 1 - tau / (L + 1)
        gamma_tau = np.dot(d_demean[tau:], d_demean[:-tau]) / n
        V += 2 * weight * gamma_tau
    return max(V, 0.0)


def _rank_ci_from_rejections(rejected: set, p: int) -> np.ndarray:
    """N-/N+ counting rule from a set of rejected (j, k) pairs."""
    rank_ci = np.empty((p, 2), dtype=int)
    for j in range(p):
        n_better = sum(1 for k in range(p) if k != j and (k, j) in rejected)
        n_worse  = sum(1 for k in range(p) if k != j and (j, k) in rejected)
        rank_ci[j] = [n_better + 1, p - n_worse]
    return rank_ci


# ═══════════════════════════════════════════════════════════════════════════════
# Complete-cases stepwise simulation
# ═══════════════════════════════════════════════════════════════════════════════

def _simulation_cv_complete(
    Sigma_hat, se, active_pairs, alpha, B, rng
) -> float:
    """
    One-sided simulation critical value over active pairs only.

    Instead of resampling rows, draws Z ~ N(0, Sigma_hat) and computes
    T_b = max_{(j,k) in active} (Z_j - Z_k) / se[j,k].
    """
    p = Sigma_hat.shape[0]
    Z = rng.multivariate_normal(np.zeros(p), Sigma_hat, size=B)  # (B, p)
    T = np.empty(B)
    for b in range(B):
        T[b] = max(
            (Z[b, j] - Z[b, k]) / se[j, k]
            for j, k in active_pairs
        )
    return float(np.quantile(T, 1 - alpha))


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
    X : (n, p) array, no NaN.
    alpha : miscoverage level.
    B : number of Gaussian simulation draws per stepwise iteration.
    seed : random seed.

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

    # IID pairwise SE
    D = X[:, :, None] - X[:, None, :]
    se = D.std(axis=0, ddof=1) / np.sqrt(n)
    np.fill_diagonal(se, np.nan)

    active = {(j, k) for j in range(p) for k in range(p) if j != k}
    rejected = set()

    while active:
        cv = _simulation_cv_complete(
            Sigma_hat, se, list(active), alpha, B, rng,
        )
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
        "rank_ci": _rank_ci_from_rejections(rejected, p),
        "Sigma_hat": Sigma_hat,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pairwise stepwise simulation (unbalanced panel)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_pairwise_stats(X, use_hac, min_overlap):
    """Compute delta_hat, se, n_overlap from pairwise-complete observations."""
    n, p = X.shape
    delta_hat = np.full((p, p), np.nan)
    se = np.full((p, p), np.nan)
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
                V = _newey_west_var(d_jk)
                se[j, k] = np.sqrt(max(V, 0.0) / njk)
            else:
                se[j, k] = d_jk.std(ddof=1) / np.sqrt(njk)
    return delta_hat, se, n_overlap


def _build_Sigma_pairwise(X, min_overlap):
    """Estimate Cov(theta_hat) entry-by-entry from overlapping observations."""
    n, p = X.shape
    Sigma = np.zeros((p, p))

    for j in range(p):
        mask_j = np.isfinite(X[:, j])
        nj = mask_j.sum()
        if nj < 2:
            continue
        Sigma[j, j] = np.var(X[mask_j, j], ddof=1) / nj

        for k in range(j + 1, p):
            mask_jk = np.isfinite(X[:, j]) & np.isfinite(X[:, k])
            njk = mask_jk.sum()
            if njk < min_overlap:
                continue
            nk = np.isfinite(X[:, k]).sum()
            sigma_jk = np.cov(X[mask_jk, j], X[mask_jk, k], ddof=1)[0, 1]
            Sigma[j, k] = (njk / (nj * nk)) * sigma_jk
            Sigma[k, j] = Sigma[j, k]

    return _nearest_psd(Sigma)


def _simulation_cv_pairwise(
    Sigma_hat, se, active_pairs, alpha, B, rng
) -> float:
    """
    One-sided simulation critical value over active pairs for pairwise data.

    Draws Z ~ N(0, Sigma_hat), computes T_b = max_{(j,k) active} (Z_j - Z_k) / se[j,k].
    Skips pairs with NaN standard errors.
    """
    p = Sigma_hat.shape[0]
    Z = rng.multivariate_normal(np.zeros(p), Sigma_hat, size=B)
    T = np.empty(B)
    # Pre-filter to valid active pairs
    valid_pairs = [(j, k) for j, k in active_pairs if np.isfinite(se[j, k]) and se[j, k] > 0]
    if not valid_pairs:
        return np.inf

    for b in range(B):
        T[b] = max(
            (Z[b, j] - Z[b, k]) / se[j, k]
            for j, k in valid_pairs
        )
    return float(np.quantile(T, 1 - alpha))


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
    X          : (n, p) array, may contain NaN.
    alpha      : miscoverage level (default 0.05).
    B          : simulation draws per stepwise iteration.
    seed       : random seed.
    use_hac    : if True, use Newey-West HAC SEs; else IID.
    min_overlap: minimum shared observations per pair.
    verbose    : print diagnostic summary.

    Returns
    -------
    dict with: theta_hat, rank_ci, Sigma_hat, n_overlap.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = np.nanmean(X, axis=0)
    delta_hat, se, n_overlap = _build_pairwise_stats(X, use_hac, min_overlap)
    Sigma_hat = _build_Sigma_pairwise(X, min_overlap)

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
        cv = _simulation_cv_pairwise(
            Sigma_hat, se, list(active), alpha, B, rng,
        )
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
        "rank_ci": _rank_ci_from_rejections(rejected, p),
        "Sigma_hat": Sigma_hat,
        "n_overlap": n_overlap,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Marginal (per-forecaster) simulation CIs
# ═══════════════════════════════════════════════════════════════════════════════

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

    Returns
    -------
    dict with: theta_hat, rank_ci, critical_values, n_overlap.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = np.nanmean(X, axis=0)
    delta_hat, se, n_overlap = _build_pairwise_stats(X, use_hac, min_overlap)
    Sigma_hat = _build_Sigma_pairwise(X, min_overlap)

    rank_ci = np.empty((p, 2), dtype=int)
    cvs = np.empty(p)

    for j in range(p):
        # All pairs involving j (both directions)
        pairs_j = [
            (a, c) for a, c in
            ([(j, k) for k in range(p) if k != j]
             + [(k, j) for k in range(p) if k != j])
            if np.isfinite(se[a, c])
        ]
        cv_j = _simulation_cv_pairwise(
            Sigma_hat, se, pairs_j, alpha, B, rng,
        )
        cvs[j] = cv_j

        # N⁺: number of k that j definitively beats (θ_j > θ_k)
        n_plus = sum(
            1 for k in range(p) if k != j
            and np.isfinite(se[j, k])
            and (delta_hat[j, k] - cv_j * se[j, k]) > 0
        )
        # N⁻: number of k that definitively beat j (θ_k > θ_j)
        n_minus = sum(
            1 for k in range(p) if k != j
            and np.isfinite(se[k, j])
            and (delta_hat[k, j] - cv_j * se[k, j]) > 0
        )
        # Rank 1 = highest theta; rank p = lowest
        rank_ci[j] = [n_minus + 1, p - n_plus]

    return {
        "theta_hat": theta_hat,
        "rank_ci": rank_ci,
        "critical_values": cvs,
        "n_overlap": n_overlap,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(123)
    n, p = 300, 5
    true_means = np.array([1.0, 0.9, 0.88, 0.7, 0.5])
    cov = np.array([
        [1.0, 0.4, 0.3, 0.2, 0.1],
        [0.4, 1.0, 0.4, 0.2, 0.1],
        [0.3, 0.4, 1.0, 0.3, 0.1],
        [0.2, 0.2, 0.3, 1.0, 0.2],
        [0.1, 0.1, 0.1, 0.2, 1.0],
    ])
    X = rng.multivariate_normal(mean=true_means, cov=cov, size=n)

    # ── Test 1: Complete-case stepwise simulation ──
    print("=" * 60)
    print("COMPLETE-CASE STEPWISE SIMULATION")
    print("=" * 60)
    out1 = rank_ci_stepwise_simulation(X, alpha=0.05, B=20000, seed=42)
    print(f"theta_hat: {np.round(out1['theta_hat'], 3)}")
    for j, ci in enumerate(out1["rank_ci"]):
        print(f"  Pop {j+1}: {ci.tolist()}")

    # ── Test 2: Pairwise stepwise simulation (with NaN) ──
    X_miss = X.copy()
    miss_rates = [0.05, 0.10, 0.20, 0.15, 0.30]
    for j, rate in enumerate(miss_rates):
        idx = rng.choice(n, size=int(n * rate), replace=False)
        X_miss[idx, j] = np.nan
    print(f"\nMissing rate: {np.isnan(X_miss).mean():.1%}")

    print("\n" + "=" * 60)
    print("PAIRWISE STEPWISE SIMULATION")
    print("=" * 60)
    out2 = rank_ci_stepwise_simulation_pairwise(
        X_miss, alpha=0.05, B=20000, seed=42, use_hac=False,
    )
    print(f"\ntheta_hat: {np.round(out2['theta_hat'], 3)}")
    for j, ci in enumerate(out2["rank_ci"]):
        print(f"  Pop {j+1}: {ci.tolist()}")

    # ── Test 3: Marginal simulation ──
    print("\n" + "=" * 60)
    print("MARGINAL (PER-FORECASTER) SIMULATION")
    print("=" * 60)
    out3 = rank_ci_marginal_simulation_pairwise(
        X_miss, alpha=0.05, B=20000, seed=42, use_hac=False,
    )
    for j, ci in enumerate(out3["rank_ci"]):
        print(f"  Pop {j+1}: {ci.tolist()}  (cv = {out3['critical_values'][j]:.3f})")