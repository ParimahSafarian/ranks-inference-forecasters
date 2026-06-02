"""
Confidence sets for the τ-best populations (Algorithm 3.3, Mogstad et al. 2024).

Implements Section 3.4 of the paper:  given τ ∈ {1, …, p}, construct a set
J^{τ-best}_n that contains ALL truly top-τ populations with probability ≥ 1 − α.

Three entry points:
  - tau_best_from_rank_ci :           naive projection from joint rank CIs (eq. 27)
  - tau_best_pairwise :               bootstrap, unbalanced panel (Algorithm 3.3)
  - tau_best_simulation_pairwise :    simulation variant of Algorithm 3.3

Rank convention (matching the rest of rankci):
  rank 1 = smallest theta = best for MSE-type losses.
  "τ-best" = the τ populations with the SMALLEST θ.

Per Remark 3.11 the paper's procedure (higher θ = better) is applied to −θ,
which flips the test statistic direction:

    T_{n,j} = min_{K ∈ K}  max_{k ∈ J \\ K}  (θ̂_j − θ̂_k) / se_{j,k}

Large T_{n,j}  →  even after exempting τ−1 populations, someone is still
                  substantially better than j  →  reject H_j  →  exclude j.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .pairwise import compute_pairwise, cov_theta_pairwise


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Naive projection from joint rank CIs  (equation 27)
# ═══════════════════════════════════════════════════════════════════════════════

def tau_best_from_rank_ci(
    rank_ci: np.ndarray,
    tau: int,
) -> np.ndarray:
    """
    Naive τ-best set by projecting from simultaneous rank CIs.

    J^{τ-best}_n = {j : τ ∈ R^{joint}_{n,j}}.

    Parameters
    ----------
    rank_ci : (p, 2) array of [lower, upper] rank confidence intervals.
    tau     : the τ in "τ-best" (e.g. τ=1 for the single best).

    Returns
    -------
    1-D boolean array of length p.  True = included in the τ-best set.
    """
    return (rank_ci[:, 0] <= tau) & (tau <= rank_ci[:, 1])


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Direct bootstrap τ-best  (Algorithm 3.3, pairwise resampling)
# ═══════════════════════════════════════════════════════════════════════════════

def _test_stat_j(j: int, delta_hat: np.ndarray, se: np.ndarray,
                 p: int, K_sets: list[tuple[int, ...]]) -> float:
    """
    Compute T_{n,j} = min_K max_{k ∈ J\\K} delta_hat[j,k] / se[j,k].

    In our convention (lower θ = better), delta_hat[j,k] = θ̂_j − θ̂_k.
    A large positive value means k has much lower MSE than j.
    """
    best = np.inf
    for K in K_sets:
        K_set = set(K)
        worst_in_complement = -np.inf
        for k in range(p):
            if k == j or k in K_set:
                continue
            if np.isnan(se[j, k]) or se[j, k] <= 0:
                continue
            t = delta_hat[j, k] / se[j, k]
            if t > worst_in_complement:
                worst_in_complement = t
        if worst_in_complement < best:
            best = worst_in_complement
    return best


def _bootstrap_cv_tau_best(
    X: np.ndarray,
    delta_hat: np.ndarray,
    se: np.ndarray,
    I_set: list[int],
    K_sets: list[tuple[int, ...]],
    alpha: float,
    B: int,
    rng: np.random.Generator,
) -> float:
    """
    Bootstrap critical value for Algorithm 3.3 (pairwise resampling).

    ĉ_n(1−α, I) = max_{K ∈ K}  Quantile_{1−α}({T*_{b,I,K}})

    where T*_{b,I,K} = max_{j ∈ I}  max_{k ∈ J\\K}
                       (d̄*_{b,j,k} − Δ̂_{j,k}) / se_{j,k}

    The bootstrap resamples each pair (j,k) independently from its
    overlap set, keeping se fixed at the original estimate.
    """
    p = X.shape[1]

    # ── Collect all pairs we might need ──────────────────────────────────
    I_s = set(I_set)
    all_K_members = set()
    for K in K_sets:
        all_K_members.update(K)
    # Pairs (j, k) with j ∈ I and k ∈ J (could appear in some J\K)
    needed_pairs = set()
    for j in I_set:
        for k in range(p):
            if k == j:
                continue
            if np.isfinite(se[j, k]) and se[j, k] > 0:
                needed_pairs.add((j, k))

    # ── Pre-compute overlap data for each needed pair ────────────────────
    pair_data = {}
    for j, k in needed_pairs:
        mask = ~np.isnan(X[:, j]) & ~np.isnan(X[:, k])
        n_jk = mask.sum()
        if n_jk < 2:
            continue
        pair_data[(j, k)] = X[mask][:, [j, k]]  # shape (n_jk, 2)

    # ── Bootstrap loop ───────────────────────────────────────────────────
    # For each b, compute the centered bootstrap stat for all needed pairs,
    # then for each K compute T*_{b,I,K}.
    n_K = len(K_sets)
    T_K = np.full((B, n_K), -np.inf)

    for b in range(B):
        # Resample each pair independently
        boot_stat = {}
        for (j, k), Xpair in pair_data.items():
            n_jk = Xpair.shape[0]
            idx = rng.integers(0, n_jk, size=n_jk)
            d_bar_b = (Xpair[idx, 0] - Xpair[idx, 1]).mean()
            boot_stat[(j, k)] = (d_bar_b - delta_hat[j, k]) / se[j, k]

        for ki, K in enumerate(K_sets):
            K_set = set(K)
            max_val = -np.inf
            for j in I_set:
                for k in range(p):
                    if k == j or k in K_set:
                        continue
                    s = boot_stat.get((j, k), None)
                    if s is not None and s > max_val:
                        max_val = s
            T_K[b, ki] = max_val

    # Critical value = max over K of the (1-α) quantile
    quantiles = np.quantile(T_K, 1 - alpha, axis=0)  # shape (n_K,)
    return float(np.max(quantiles))


def tau_best_pairwise(
    X: np.ndarray,
    tau: int = 1,
    alpha: float = 0.05,
    B: int = 5000,
    seed: int | None = None,
    se_method: str = "nw",
    L: int | None = None,
    winsor_pct: float | None = None,
    verbose: bool = True,
) -> dict:
    """
    Confidence set for the τ-best populations (Algorithm 3.3).

    Bootstrap variant with pairwise resampling for unbalanced panels.

    Parameters
    ----------
    X          : (n, p) array of observations (e.g. squared errors), may
                 contain NaN for missing forecaster-quarter combinations.
    tau        : number of "best" populations to identify (default 1).
    alpha      : miscoverage level (default 0.05).
    B          : bootstrap replications (default 5000).
    seed       : random seed for reproducibility.
    se_method  : "nw" for Newey-West HAC, "iid" for plain SE.
    L          : NW bandwidth (None = automatic rule).
    winsor_pct : if set, symmetrically winsorize pairwise differences.
    verbose    : print diagnostic information.

    Returns
    -------
    dict with keys:
        tau_best_set : 1-D boolean array, True = included in τ-best set.
        tau          : the τ used.
        theta_hat    : (p,) estimated features (MSE).
        test_stats   : (p,) test statistic T_{n,j} for each j.
        rejected     : 1-D boolean array, True = rejected (excluded).
        n_in_set     : number of populations in the confidence set.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if tau < 1 or tau > p:
        raise ValueError(f"tau must be in {{1, …, {p}}}, got {tau}.")

    theta_hat = np.nanmean(X, axis=0)
    delta_hat, se, n_pairs = compute_pairwise(
        X, se_method=se_method, L=L, winsor_pct=winsor_pct,
    )

    # ── Enumerate K = {K ⊂ J : |K| = τ−1} ──────────────────────────────
    if tau == 1:
        K_sets = [()]  # K = {∅}
    else:
        K_sets = list(combinations(range(p), tau - 1))

    if verbose:
        print(f"=== τ-best procedure (τ={tau}, α={alpha}) ===")
        print(f"  Populations: {p}")
        print(f"  |K| = C({p},{tau-1}) = {len(K_sets)}")
        valid = n_pairs[n_pairs > 0]
        print(f"  Pairwise overlaps: min={valid.min()}, "
              f"mean={valid.mean():.1f}, max={valid.max()}")

    # ── Compute test statistics T_{n,j} for all j ───────────────────────
    T_n = np.full(p, np.nan)
    for j in range(p):
        T_n[j] = _test_stat_j(j, delta_hat, se, p, K_sets)

    if verbose:
        print(f"  Test statistics: min={np.nanmin(T_n):.3f}, "
              f"max={np.nanmax(T_n):.3f}")

    # ── Stepwise testing (Algorithm 3.3) ─────────────────────────────────
    I_set = list(range(p))  # all candidates
    rejected = np.zeros(p, dtype=bool)
    step = 0

    while I_set:
        step += 1
        cv = _bootstrap_cv_tau_best(
            X, delta_hat, se, I_set, K_sets, alpha, B, rng,
        )

        new_rejections = [j for j in I_set if T_n[j] > cv]

        if verbose:
            print(f"  Step {step}: cv={cv:.3f}, "
                  f"|I|={len(I_set)}, rejected={len(new_rejections)}")

        if not new_rejections:
            break

        for j in new_rejections:
            rejected[j] = True
        I_set = [j for j in I_set if not rejected[j]]

    tau_best_set = ~rejected

    if verbose:
        print(f"  Result: {tau_best_set.sum()} populations in τ-best set")

    return {
        "tau_best_set": tau_best_set,
        "tau": tau,
        "theta_hat": theta_hat,
        "test_stats": T_n,
        "rejected": rejected,
        "n_in_set": int(tau_best_set.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Direct simulation τ-best  (Algorithm 3.3, Gaussian approximation)
# ═══════════════════════════════════════════════════════════════════════════════

def _simulation_cv_tau_best(
    Sigma_hat: np.ndarray,
    se: np.ndarray,
    I_set: list[int],
    K_sets: list[tuple[int, ...]],
    alpha: float,
    B: int,
    rng: np.random.Generator,
) -> float:
    """
    Simulation-based critical value for Algorithm 3.3.

    Draws Z ~ N(0, Σ̂) and computes:
        T_{b,I,K} = max_{j ∈ I}  max_{k ∈ J\\K}  (Z_j − Z_k) / se_{j,k}
    then ĉ(1−α, I) = max_K Quantile_{1−α}({T_{b,I,K}}).
    """
    p = Sigma_hat.shape[0]
    n_K = len(K_sets)

    Z = rng.multivariate_normal(np.zeros(p), Sigma_hat, size=B)

    T_K = np.full((B, n_K), -np.inf)

    for b in range(B):
        z = Z[b]
        for ki, K in enumerate(K_sets):
            K_set = set(K)
            max_val = -np.inf
            for j in I_set:
                for k in range(p):
                    if k == j or k in K_set:
                        continue
                    if not np.isfinite(se[j, k]) or se[j, k] <= 0:
                        continue
                    t = (z[j] - z[k]) / se[j, k]
                    if t > max_val:
                        max_val = t
            T_K[b, ki] = max_val

    quantiles = np.quantile(T_K, 1 - alpha, axis=0)
    return float(np.max(quantiles))


def tau_best_simulation_pairwise(
    X: np.ndarray,
    tau: int = 1,
    alpha: float = 0.05,
    B: int = 20000,
    seed: int | None = None,
    se_method: str = "nw",
    L: int | None = None,
    winsor_pct: float | None = None,
    min_overlap: int = 2,
    verbose: bool = True,
) -> dict:
    """
    Confidence set for the τ-best populations — simulation variant.

    Replaces bootstrap resampling with draws from N(0, Σ̂), where Σ̂ is
    the pairwise covariance of θ̂ (PSD-projected via Schoenberg construction).

    Parameters
    ----------
    X           : (n, p) array, may contain NaN.
    tau         : number of "best" populations to identify.
    alpha       : miscoverage level.
    B           : number of Gaussian draws per stepwise iteration.
    seed        : random seed.
    se_method   : "nw" or "iid".
    L           : NW bandwidth (None = auto).
    winsor_pct  : winsorization percentile for pairwise differences.
    min_overlap : minimum shared observations per pair.
    verbose     : print diagnostics.

    Returns
    -------
    dict with keys: tau_best_set, tau, theta_hat, test_stats,
                    rejected, n_in_set, Sigma_hat.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if tau < 1 or tau > p:
        raise ValueError(f"tau must be in {{1, …, {p}}}, got {tau}.")

    theta_hat = np.nanmean(X, axis=0)
    delta_hat, se, n_pairs = compute_pairwise(
        X, se_method=se_method, L=L, winsor_pct=winsor_pct,
        min_overlap=min_overlap,
    )
    Sigma_hat = cov_theta_pairwise(
        X, min_overlap=min_overlap, se_method=se_method, se_pair=se,
    )

    # ── Enumerate K ──────────────────────────────────────────────────────
    if tau == 1:
        K_sets = [()]
    else:
        K_sets = list(combinations(range(p), tau - 1))

    if verbose:
        print(f"=== τ-best procedure — simulation (τ={tau}, α={alpha}) ===")
        print(f"  Populations: {p}, |K| = C({p},{tau-1}) = {len(K_sets)}")

    # ── Test statistics ──────────────────────────────────────────────────
    T_n = np.full(p, np.nan)
    for j in range(p):
        T_n[j] = _test_stat_j(j, delta_hat, se, p, K_sets)

    if verbose:
        print(f"  Test statistics: min={np.nanmin(T_n):.3f}, "
              f"max={np.nanmax(T_n):.3f}")

    # ── Stepwise testing ─────────────────────────────────────────────────
    I_set = list(range(p))
    rejected = np.zeros(p, dtype=bool)
    step = 0

    while I_set:
        step += 1
        cv = _simulation_cv_tau_best(
            Sigma_hat, se, I_set, K_sets, alpha, B, rng,
        )

        new_rejections = [j for j in I_set if T_n[j] > cv]

        if verbose:
            print(f"  Step {step}: cv={cv:.3f}, "
                  f"|I|={len(I_set)}, rejected={len(new_rejections)}")

        if not new_rejections:
            break

        for j in new_rejections:
            rejected[j] = True
        I_set = [j for j in I_set if not rejected[j]]

    tau_best_set = ~rejected

    if verbose:
        print(f"  Result: {tau_best_set.sum()} populations in τ-best set")

    return {
        "tau_best_set": tau_best_set,
        "tau": tau,
        "theta_hat": theta_hat,
        "test_stats": T_n,
        "rejected": rejected,
        "n_in_set": int(tau_best_set.sum()),
        "Sigma_hat": Sigma_hat,
    }
