"""
Confidence sets for the τ-best populations (Algorithm 3.3, Mogstad et al. 2024).

Implements Section 3.4 of the paper:  given τ ∈ {1, …, p}, construct a set
J^{τ-best}_n that contains ALL truly top-τ populations with probability ≥ 1 − α.

Three entry points:
  - tau_best_from_rank_ci :           naive projection from joint rank CIs (eq. 27)
  - tau_best_pairwise :               joint circular block bootstrap, unbalanced panel (Algorithm 3.3)
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

from .block_bootstrap import block_bootstrap_draws, default_block_length
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

    J^{τ-best}_n = {j : L_j ≤ τ},   where R^{joint}_{n,j} = [L_j, U_j].

    On the event that every R^{joint}_{n,j} covers the true rank r_j, each
    truly top-τ population (r_j ≤ τ) has L_j ≤ r_j ≤ τ, so the set inherits
    the joint coverage.  Units with U_j < τ are certainly top-τ and must be
    kept: the rule {j : τ ∈ R^{joint}_{n,j}}, as eq. 27 is printed, drops
    them and loses coverage for τ ≥ 2 (for τ = 1 the two rules coincide).

    Parameters
    ----------
    rank_ci : (p, 2) array of [lower, upper] rank confidence intervals.
    tau     : the τ in "τ-best" (e.g. τ=1 for the single best).

    Returns
    -------
    1-D boolean array of length p.  True = included in the τ-best set.
    """
    return rank_ci[:, 0] <= tau


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Direct bootstrap τ-best  (Algorithm 3.3, joint circular block bootstrap)
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


def _tau_best_cv(
    T: np.ndarray,
    col: dict,
    valid: np.ndarray,
    I_set: list[int],
    K_sets: list[tuple[int, ...]],
    alpha: float,
) -> float:
    """
    Critical value for Algorithm 3.3 from draws taken once.

    ĉ_n(1−α, I) = max_{K ∈ K}  Quantile_{1−α}({T*_{b,I,K}}),
    T*_{b,I,K}  = max_{j ∈ I}  max_{k ∈ J\\K}  T*_{b,j,k},

    where T has one column per unordered pair (j, k), j < k, mapped by ``col``,
    and the ordered statistic (k, j) is the negated column. Only pairs with
    ``valid[j, k]`` enter; a NaN draw does not enter the maximum.
    """
    p = valid.shape[0]
    best = -np.inf
    for K in K_sets:
        K_set = set(K)
        cols = [
            T[:, col[(j, k)]] if j < k else -T[:, col[(k, j)]]
            for j in I_set for k in range(p)
            if k != j and k not in K_set and valid[j, k]
        ]
        if not cols:
            continue
        M = np.column_stack(cols)
        M = np.where(np.isnan(M), -np.inf, M)
        best = max(best, float(np.quantile(M.max(axis=1), 1 - alpha)))
    return best


def _pair_columns(p: int):
    """Unordered pairs (j, k), j < k, and the map (j, k) -> column."""
    pairs = [(j, k) for j in range(p) for k in range(j + 1, p)]
    return pairs, {pr: a for a, pr in enumerate(pairs)}


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
    block_length: int | None = None,
) -> dict:
    """
    Confidence set for the τ-best populations (Algorithm 3.3).

    Bootstrap variant for unbalanced panels; critical values come from a
    joint circular block bootstrap of whole time rows, drawn once and reused
    in every step.

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
    block_length : block length. None uses ``default_block_length(X, L)``
                 (= L + 1 if L is given, else median Andrews bandwidth + 1).

    Returns
    -------
    dict with keys:
        tau_best_set : 1-D boolean array, True = included in τ-best set.
        tau          : the τ used.
        theta_hat    : (p,) estimated features (MSE).
        test_stats   : (p,) test statistic T_{n,j} for each j.
        rejected     : 1-D boolean array, True = rejected (excluded).
        n_in_set     : number of populations in the confidence set.
        block_length : block length used.
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
    if block_length is None:
        block_length = default_block_length(X, L=L)

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
        print(f"  Block length: {block_length}")

    # ── Compute test statistics T_{n,j} for all j ───────────────────────
    T_n = np.full(p, np.nan)
    for j in range(p):
        T_n[j] = _test_stat_j(j, delta_hat, se, p, K_sets)

    if verbose:
        print(f"  Test statistics: min={np.nanmin(T_n):.3f}, "
              f"max={np.nanmax(T_n):.3f}")

    # ── Bootstrap draws, taken once for every pair ───────────────────────
    pairs, col = _pair_columns(p)
    P = np.asarray(pairs, dtype=int).reshape(-1, 2)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(se) & (se > 0) & (n_pairs >= 2)
    T = block_bootstrap_draws(
        X, P, delta_hat[P[:, 0], P[:, 1]], se[P[:, 0], P[:, 1]], B, block_length, rng,
    )

    # ── Stepwise testing (Algorithm 3.3) ─────────────────────────────────
    I_set = list(range(p))  # all candidates
    rejected = np.zeros(p, dtype=bool)
    step = 0

    while I_set:
        step += 1
        cv = _tau_best_cv(T, col, valid, I_set, K_sets, alpha)

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
        "block_length": int(block_length),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Direct simulation τ-best  (Algorithm 3.3, Gaussian approximation)
# ═══════════════════════════════════════════════════════════════════════════════

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
    B           : number of Gaussian draws, taken once and reused in every step.
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

    # ── Gaussian draws, taken once: T_{b,j,k} = (Z_j − Z_k) / se_{j,k} ───
    pairs, col = _pair_columns(p)
    J = np.array([j for j, _ in pairs], dtype=int)
    K = np.array([k for _, k in pairs], dtype=int)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = np.isfinite(se) & (se > 0)
        Z = rng.multivariate_normal(np.zeros(p), Sigma_hat, size=B)
        T = (Z[:, J] - Z[:, K]) / se[J, K]

    # ── Stepwise testing ─────────────────────────────────────────────────
    I_set = list(range(p))
    rejected = np.zeros(p, dtype=bool)
    step = 0

    while I_set:
        step += 1
        cv = _tau_best_cv(T, col, valid, I_set, K_sets, alpha)

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
