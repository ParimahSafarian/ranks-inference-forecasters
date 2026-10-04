"""
Stepwise bootstrap rank confidence intervals (Algorithm 3.2, Mogstad et al. 2024).

Two variants:
  - rank_ci_stepwise:          complete-cases, IID SE
  - rank_ci_stepwise_pairwise: unbalanced panel, NW-HAC SE

Both draw their critical values from a joint moving block bootstrap of whole
time rows (see :mod:`rankci.core.block_bootstrap`), which preserves the serial
and cross-sectional dependence of the panel.
"""
import numpy as np

from .block_bootstrap import default_block_length, mbb_indices, pairwise_means
from .pairwise import (
    compute_pairwise,
    rank_ci_from_rejections,
)


# ── Complete-cases stepwise ──────────────────────────────────────────────────

def _compute_se_complete(X: np.ndarray) -> np.ndarray:
    """IID pairwise SE for a complete (no NaN) matrix.
    
    se[j, k] = sd(X[:, j] - X[:, k]) / sqrt(n).
    Diagonal is NaN.
    """
    n = X.shape[0]
    D = X[:, :, None] - X[:, None, :]          # shape (n, p, p)
    se = D.std(axis=0, ddof=1) / np.sqrt(n)    # shape (p, p)
    np.fill_diagonal(se, np.nan)
    return se




def _bootstrap_cv_complete(X, theta_hat, active_pairs, alpha, B, rng,
                           block_length):
    """One-sided bootstrap critical value — complete-cases.
    
    Implements the fully studentized bootstrap of Mogstad et al. (2024) eq. (6)
    with P replaced by the moving-block-bootstrap distribution: in each
    replication b, whole time rows are resampled in contiguous blocks of length
    ``block_length`` (joint across columns), and both the pairwise mean
    differences AND the pairwise standard errors are recomputed on the
    resampled data.
    
    The bootstrap test statistic for pair (k, l) in replication b is
    
        T*_{b, k, l} = ((theta*_b[k] - theta*_b[l]) - (theta_hat[k] - theta_hat[l]))
                       / se*_b[k, l]
    
    and T_b = max over active pairs of T*_{b, k, l}. The critical value is the
    (1 - alpha) empirical quantile of {T_b}_{b=1}^B.
    """
    n = X.shape[0]
    T = np.empty(B)
    for b in range(B):
        idx     = mbb_indices(n, block_length, rng)
        Xb      = X[idx]
        theta_b = Xb.mean(axis=0)
        se_b    = _compute_se_complete(Xb)   # recomputed on the bootstrap sample
        T[b] = max(
            ((theta_b[k] - theta_b[l]) - (theta_hat[k] - theta_hat[l])) / se_b[k, l]
            for k, l in active_pairs
        )
    return float(np.quantile(T, 1 - alpha))


def rank_ci_stepwise(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 5000,
    seed: int | None = None,
    block_length: int | None = None,
) -> dict:
    """Stepwise rank CIs — complete cases, IID standard errors.

    block_length : MBB block length. None uses ``default_block_length(X)``.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = X.mean(axis=0)
    se = _compute_se_complete(X)
    delta_hat = theta_hat[:, None] - theta_hat[None, :]

    if block_length is None:
        block_length = default_block_length(X)

    active = {(k, l) for k in range(p) for l in range(p) if k != l}
    rejected = set()

    while active:
        cv = _bootstrap_cv_complete(X, theta_hat, list(active), alpha, B, rng,
                                    block_length)
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
        "block_length": int(block_length),
    }


# ── Pairwise stepwise (unbalanced panel, NW-HAC) ────────────────────────────

def _bootstrap_cv_pairwise(X, delta_hat, se, active_pairs, alpha, B, rng,
                           block_length):
    """Joint moving-block-bootstrap critical value for unbalanced panels.

    In each replication one set of row indices is drawn for the whole panel
    (contiguous blocks of ``block_length`` rows), so every pair's difference
    mean is recomputed on the *same* resampled rows — preserving serial
    dependence (within blocks) and cross-pair dependence (shared rows). The
    pairwise mean uses the rows where both members are observed in the
    resample; the NW standard error stays fixed at the original estimate.

        T_b = max_{(j,k) active} (d̄*_{b,j,k} − Δ̂_{j,k}) / se_{j,k}
    """
    n = X.shape[0]
    pairs = np.asarray(active_pairs, dtype=int).reshape(-1, 2)
    if pairs.shape[0] == 0:
        return float("inf")
    center = delta_hat[pairs[:, 0], pairs[:, 1]]
    scale  = se[pairs[:, 0], pairs[:, 1]]

    T = np.empty(B)
    for b in range(B):
        idx  = mbb_indices(n, block_length, rng)
        stat = (pairwise_means(X[idx], pairs) - center) / scale
        stat = stat[np.isfinite(stat)]
        T[b] = stat.max() if stat.size else -np.inf
    return float(np.quantile(T, 1 - alpha))


def rank_ci_stepwise_pairwise(
    X: np.ndarray,
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
    Stepwise rank CIs for unbalanced panels.

    Uses pairwise complete observations and (by default) Newey-West HAC SEs.
    Critical values come from a joint moving block bootstrap of whole rows.

    Parameters
    ----------
    X            : (n, p) array, may contain NaN.
    se_method    : "nw" for Newey-West HAC, "iid" for plain SE.
    L            : NW bandwidth. None uses the automatic rule
                   L = floor(4 * (n/100)^{2/9}). Ignored if se_method="iid".
    winsor_pct   : if set (e.g. 95), symmetrically winsorize each pairwise
                   difference series before computing the SE.
    verbose      : print diagnostic summary.
    block_length : MBB block length. None uses ``default_block_length(X, L)``
                   (= L + 1 if L is given, else median Andrews bandwidth + 1).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = np.nanmean(X, axis=0)
    delta_hat, se, n_pairs = compute_pairwise(
        X, se_method=se_method, L=L, winsor_pct=winsor_pct,
    )
    if block_length is None:
        block_length = default_block_length(X, L=L)

    if verbose:
        valid = n_pairs[n_pairs > 0]
        print("=== Pairwise shared observations ===")
        print(f"  Min: {valid.min()}, Mean: {valid.mean():.1f}, Max: {valid.max()}")
        print(f"  Pairs with < 20 shared obs: {(valid < 20).sum()}")
        print(f"  MBB block length: {block_length}")

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
        cv = _bootstrap_cv_pairwise(X, delta_hat, se, list(active), alpha, B, rng,
                                    block_length)
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
        "rejected": rejected,
        "block_length": int(block_length),
    }


# ── Marginal (per-forecaster) CIs ────────────────────────────────────────────


"""
fix: how se is computed in the bootstrap samples? 
currently fixed across samples, but should it be 
recomputed on each bootstrap sample? 
"""

def rank_ci_marginal_pairwise(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 5000,
    seed: int | None = None,
    se_method: str = "nw",
    L: int | None = None,
    winsor_pct: float | None = None,
    block_length: int | None = None,
) -> dict:
    """
    Marginal (per-forecaster) rank CIs for unbalanced panels.

    For each forecaster j, P(rank_j ∈ CI_j) ≥ 1 - α holds *marginally* —
    the joint coverage across forecasters is NOT controlled.  Each j gets
    its own bootstrap critical value, computed from the 2(p-1) one-sided
    test statistics involving j.  Tighter than the simultaneous procedure.

    Parameters
    ----------
    X          : (n, p) array, may contain NaN.
    se_method  : "nw" for Newey-West HAC, "iid" for plain SE.
    L          : NW bandwidth. None uses the automatic rule
                 L = floor(4 * (n/100)^{2/9}). Ignored if se_method="iid".
    winsor_pct : if set, symmetrically winsorize each pairwise difference
                 series before computing the SE.
    block_length : MBB block length. None uses ``default_block_length(X, L)``.

    Returns
    -------
    dict with keys: theta_hat, rank_ci, n_pairs, critical_values (one per j),
    block_length.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = np.nanmean(X, axis=0)
    delta_hat, se, n_pairs = compute_pairwise(
        X, se_method=se_method, L=L, winsor_pct=winsor_pct,
    )
    if block_length is None:
        block_length = default_block_length(X, L=L)

    rank_ci = np.empty((p, 2), dtype=int)
    cvs = np.empty(p)

    for j in range(p):
        # All pairs involving j (both directions): 2(p-1) one-sided tests
        pairs_j = [
            (a, c)
            for a, c in (
                [(j, k) for k in range(p) if k != j]
                + [(k, j) for k in range(p) if k != j]
            )
            if not np.isnan(se[a, c])
        ]

        cv_j = _bootstrap_cv_pairwise(X, delta_hat, se, pairs_j, alpha, B, rng,
                                      block_length)
        cvs[j] = cv_j

        # (j, k): theta_j > theta_k confirmed → k smaller → k BETTER than j
        n_better = sum(
            1 for k in range(p) if k != j
            and not np.isnan(se[j, k])
            and (delta_hat[j, k] - cv_j * se[j, k]) > 0
        )
        # (k, j): theta_k > theta_j confirmed → k larger → k WORSE than j
        n_worse = sum(
            1 for k in range(p) if k != j
            and not np.isnan(se[k, j])
            and (delta_hat[k, j] - cv_j * se[k, j]) > 0
        )
        rank_ci[j] = [n_better + 1, p - n_worse]

    return {
        "theta_hat": theta_hat,
        "rank_ci": rank_ci,
        "n_pairs": n_pairs,
        "critical_values": cvs,
        "block_length": int(block_length),
    }
