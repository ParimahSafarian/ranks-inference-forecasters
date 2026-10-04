"""
Stepwise bootstrap rank confidence intervals (Algorithm 3.2, Mogstad et al. 2024).

Two variants:
  - rank_ci_stepwise:          complete-cases, IID SE
  - rank_ci_stepwise_pairwise: unbalanced panel, NW-HAC SE

Both draw their critical values from a joint circular block bootstrap of whole
time rows (see :mod:`rankci.core.block_bootstrap`), which preserves the serial
and cross-sectional dependence of the panel. The B resamples are drawn once and
the Romano--Wolf stepdown of :func:`rankci.core.covariance.stepdown_rank_ci`
restricts them to the active pairs in every round, as on the simulation route.
"""
import numpy as np

from .block_bootstrap import block_bootstrap_draws, default_block_length, mbb_indices
from .covariance import stepdown_rank_ci
from .pairwise import compute_pairwise


def _pair_index(p: int):
    """Unordered pairs (j, k), j < k, and the map (j, k) -> column."""
    pairs = [(j, k) for j in range(p) for k in range(j + 1, p)]
    return pairs, {pr: a for a, pr in enumerate(pairs)}


def _signed_column(T: np.ndarray, col: dict, u: int, v: int) -> np.ndarray:
    """Draws for the ordered hypothesis (u, v) from the unordered-pair columns."""
    return T[:, col[(u, v)]] if u < v else -T[:, col[(v, u)]]


def _max_quantile(columns: list, alpha: float) -> float:
    """(1 - alpha) quantile of the row-wise maximum; NaN draws do not enter it."""
    if not columns:
        return float("inf")
    M = np.column_stack(columns)
    M = np.where(np.isnan(M), -np.inf, M)
    return float(np.quantile(M.max(axis=1), 1 - alpha))


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




def _bootstrap_draws_complete(X, theta_hat, pairs, B, rng, block_length):
    """Fully studentized bootstrap draws — complete-cases.

    Implements the fully studentized bootstrap of Mogstad et al. (2024) eq. (6)
    with P replaced by the block-bootstrap distribution: in each replication b,
    whole time rows are resampled in circular blocks of length
    ``block_length`` (joint across columns), and both the pairwise mean
    differences AND the pairwise standard errors are recomputed on the
    resampled data. For unordered pair a = (k, l), k < l,

        T*[b, a] = ((theta*_b[k] - theta*_b[l]) - (theta_hat[k] - theta_hat[l]))
                   / se*_b[k, l],

    and the reverse orientation (l, k) is -T*[b, a]. Returns the (B, q) array.
    """
    n = X.shape[0]
    J = np.array([k for k, _ in pairs])
    K = np.array([l for _, l in pairs])
    center = theta_hat[J] - theta_hat[K]
    T = np.empty((B, len(pairs)))
    with np.errstate(invalid="ignore", divide="ignore"):
        for b in range(B):
            idx     = mbb_indices(n, block_length, rng)
            Xb      = X[idx]
            theta_b = Xb.mean(axis=0)
            se_b    = _compute_se_complete(Xb)   # recomputed on the bootstrap sample
            T[b] = ((theta_b[J] - theta_b[K]) - center) / se_b[J, K]
    return T


def rank_ci_stepwise(
    X: np.ndarray,
    alpha: float = 0.05,
    B: int = 5000,
    seed: int | None = None,
    block_length: int | None = None,
) -> dict:
    """Stepwise rank CIs — complete cases, IID standard errors.

    block_length : block length. None uses ``default_block_length(X)``.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    theta_hat = X.mean(axis=0)
    se = _compute_se_complete(X)

    if block_length is None:
        block_length = default_block_length(X)

    pairs, _ = _pair_index(p)
    draws = {
        "pairs": pairs,
        "delta_q": np.array([theta_hat[k] - theta_hat[l] for k, l in pairs]),
        "se_q": np.array([se[k, l] for k, l in pairs]),
        "T_draws": _bootstrap_draws_complete(X, theta_hat, pairs, B, rng, block_length),
    }
    out = stepdown_rank_ci(draws, p, alpha=alpha)

    return {
        "theta_hat": theta_hat,
        "rank_ci": out["rank_ci"],
        "rejected": out["rejected"],
        "n_steps": out["n_steps"],
        "block_length": int(block_length),
    }


# ── Pairwise stepwise (unbalanced panel, NW-HAC) ────────────────────────────

def _bootstrap_draws_pairwise(X, delta_hat, se, pairs, B, rng, block_length):
    """Joint block-bootstrap draws for unbalanced panels, one column per pair.

    In each replication one set of row indices is drawn for the whole panel
    (circular blocks of ``block_length`` rows), so every pair's difference mean
    is recomputed on the *same* resampled rows — preserving serial dependence
    (within blocks) and cross-pair dependence (shared rows). The pairwise mean
    uses the rows where both members are observed in the resample; the NW
    standard error stays fixed at the original estimate:

        T*[b, a] = (d̄*_{b,j,k} − Δ̂_{j,k}) / se_{j,k},    a = (j, k), j < k.
    """
    P = np.asarray(pairs, dtype=int).reshape(-1, 2)
    return block_bootstrap_draws(
        X, P, delta_hat[P[:, 0], P[:, 1]], se[P[:, 0], P[:, 1]], B, block_length, rng,
    )


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
    Critical values come from a joint circular block bootstrap of whole rows;
    the B resamples are drawn once and reused in every stepdown round.

    Parameters
    ----------
    X            : (n, p) array, may contain NaN.
    se_method    : "nw" for Newey-West HAC, "iid" for plain SE.
    L            : NW bandwidth. None uses the Andrews (1991) plug-in for each
                   pair. Ignored if se_method="iid".
    winsor_pct   : if set (e.g. 95), symmetrically winsorize each pairwise
                   difference series before computing the SE.
    verbose      : print diagnostic summary.
    block_length : block length. None uses ``default_block_length(X, L)``
                   (= L + 1 if L is given, else median Andrews bandwidth + 1).

    Returns
    -------
    dict with keys: theta_hat, rank_ci, n_pairs, rejected, n_steps,
    critical_values (one per stepdown round), block_length.
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
        print(f"  Block length: {block_length}")

        with np.errstate(invalid="ignore"):
            t_stats = delta_hat / se
        vals = t_stats[~np.isnan(t_stats)]
        print(f"\n=== Test statistics (delta_hat / se) ===")
        print(f"  Max: {vals.max():.4f}, Pairs with t > 1.96: {(vals > 1.96).sum()}")

    pairs, _ = _pair_index(p)
    draws = {
        "pairs": pairs,
        "delta_q": np.array([delta_hat[j, k] for j, k in pairs]),
        "se_q": np.array([se[j, k] for j, k in pairs]),
        "T_draws": _bootstrap_draws_pairwise(X, delta_hat, se, pairs, B, rng, block_length),
    }
    out = stepdown_rank_ci(draws, p, alpha=alpha)

    return {
        "theta_hat": theta_hat,
        "rank_ci": out["rank_ci"],
        "n_pairs": n_pairs,
        "rejected": out["rejected"],
        "n_steps": out["n_steps"],
        "critical_values": out["critical_values"],
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
    test statistics involving j; all of them come from one set of block
    resamples.  Tighter than the simultaneous procedure.

    Parameters
    ----------
    X          : (n, p) array, may contain NaN.
    se_method  : "nw" for Newey-West HAC, "iid" for plain SE.
    L          : NW bandwidth. None uses the Andrews (1991) plug-in for each
                 pair. Ignored if se_method="iid".
    winsor_pct : if set, symmetrically winsorize each pairwise difference
                 series before computing the SE.
    block_length : block length. None uses ``default_block_length(X, L)``.

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

    pairs, col = _pair_index(p)
    T = _bootstrap_draws_pairwise(X, delta_hat, se, pairs, B, rng, block_length)

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

        cv_j = _max_quantile([_signed_column(T, col, a, c) for a, c in pairs_j], alpha)
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
