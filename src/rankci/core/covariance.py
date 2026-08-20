"""
The covariance seam.

Approaches 1 (MDS level covariance) and 2 (direct difference covariance Omega)
live in different spaces: MDS returns a p x p covariance you draw Z in R^p from
and then difference; Omega returns a q x q covariance you draw D in R^q from
directly. To keep one downstream stepdown for both, the seam does NOT return a
raw covariance — it returns **simulated studentized difference draws**, a
(B, q) array whose columns are the unordered pairs a = (j, k), j < k. The
covariance method is a runtime knob; everything downstream is identical.

    method="mds":   Z ~ N(0, Sigma_pxp);  D[:,a] = Z_j - Z_k;  T[:,a] = D[:,a]/se_a
    method="omega": D ~ N(0, Omega_qxq) drawn directly;         T[:,a] = D[:,a]/se_a

Both studentize by the pair's NW-HAC SE ``se_a``. For "omega" that SE is
sqrt(Omega_aa) (calibration is automatic and exact for the projected matrix);
for "mds" it is the univariate NW SE the MDS matrix is calibrated to. On a
balanced/contiguous panel the two SE conventions coincide.
"""
import numpy as np

from .pairwise import compute_pairwise, cov_theta_pairwise, rank_ci_from_rejections
from .omega import cov_via_omega, pair_index


# ── The seam: draw + studentize ──────────────────────────────────────────────

def studentized_null_draws(
    X: np.ndarray,
    method: str = "mds",
    B: int = 20000,
    seed: int | None = None,
    min_overlap: int = 2,
    winsor_pct: float | None = None,
) -> dict:
    """
    Draw ``B`` studentized difference vectors under the Gaussian null.

    Parameters
    ----------
    X           : (n, p) array, may contain NaN.
    method      : "mds" (Approach 1) or "omega" (Approach 2).
    B           : number of Gaussian draws.
    seed        : RNG seed.
    min_overlap : minimum shared observations per pair.
    winsor_pct  : optional winsorization for the NW-HAC SEs.

    Returns
    -------
    dict with keys:
        pairs      : list of (j, k), j < k, indexing the q columns.
        theta_hat  : (p,) column means.
        delta_q    : (q,) observed difference means d_hat_a = mean(X_j - X_k).
        se_q       : (q,) studentization SEs.
        T_draws    : (B, q) studentized null draws (NaN columns for invalid pairs).
        cov        : the covariance actually drawn from (p x p for mds, q x q for omega).
        method     : echoed.
    """
    if method not in ("mds", "omega"):
        raise ValueError(f"method must be 'mds' or 'omega', got {method!r}.")

    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    pairs = pair_index(p)
    q = len(pairs)

    theta_hat = np.nanmean(X, axis=0)

    # Observed difference means and univariate NW SEs (shared by both methods).
    delta_mat, se_mat, _ = compute_pairwise(
        X, se_method="nw", winsor_pct=winsor_pct, min_overlap=min_overlap,
    )
    delta_q = np.array([delta_mat[j, k] for (j, k) in pairs])

    if method == "mds":
        cov = cov_theta_pairwise(X, min_overlap=min_overlap,
                                 se_method="nw", se_pair=se_mat)     # p x p
        se_q = np.array([se_mat[j, k] for (j, k) in pairs])
        Z = rng.multivariate_normal(np.zeros(p), cov, size=B)        # (B, p)
        D = np.column_stack([Z[:, j] - Z[:, k] for (j, k) in pairs])  # (B, q)
    else:  # omega
        cov = cov_via_omega(X, min_overlap=min_overlap)             # q x q
        se_q = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        D = rng.multivariate_normal(np.zeros(q), cov, size=B)        # (B, q)

    valid = np.isfinite(se_q) & (se_q > 0)
    T = np.full((B, q), np.nan)
    T[:, valid] = D[:, valid] / se_q[valid]

    return {
        "pairs": pairs,
        "theta_hat": theta_hat,
        "delta_q": delta_q,
        "se_q": se_q,
        "T_draws": T,
        "cov": cov,
        "method": method,
    }


# ── Shared stepdown over the studentized draws ───────────────────────────────

def stepdown_rank_ci(
    draws: dict,
    p: int,
    alpha: float = 0.05,
    verbose: bool = False,
) -> dict:
    """
    Romano--Wolf stepdown on the studentized draws produced by
    :func:`studentized_null_draws`. Draws once; restricts the active set each
    round. Identical for the MDS and Omega paths.

    For an ordered pair (u, v) mapping to unordered column a with sign s
    (s = +1 if u < v, else -1):
        observed statistic   S_obs(u,v) = s * delta_q[a] / se_q[a]
        null draw column      s * T_draws[:, a]
    Reject (u, v) — i.e. confirm theta_u > theta_v, so v is BETTER than u —
    when S_obs(u,v) exceeds the (1-alpha) quantile of the active-set max.
    """
    pairs = draws["pairs"]
    delta_q = draws["delta_q"]
    se_q = draws["se_q"]
    T = draws["T_draws"]

    # Build the two oriented hypotheses per valid unordered pair.
    # entry: (u, v) -> (col a, sign s, observed studentized stat, signed draws)
    oriented = {}
    for a, (j, k) in enumerate(pairs):
        s_a = se_q[a]
        if not np.isfinite(s_a) or s_a <= 0 or not np.isfinite(delta_q[a]):
            continue
        stat = delta_q[a] / s_a
        col = T[:, a]
        oriented[(j, k)] = (a, +1.0, stat, col)          # theta_j > theta_k ?
        oriented[(k, j)] = (a, -1.0, -stat, -col)        # theta_k > theta_j ?

    active = set(oriented.keys())
    rejected: set[tuple[int, int]] = set()
    step = 0

    while active:
        step += 1
        signed_cols = np.column_stack([oriented[uv][3] for uv in active])  # (B, |active|)
        row_max = signed_cols.max(axis=1)
        cv = float(np.quantile(row_max, 1 - alpha))

        new_rej = {uv for uv in active if oriented[uv][2] > cv}
        if verbose:
            print(f"  step {step}: |active|={len(active)}, cv={cv:.3f}, "
                  f"rejected={len(new_rej)}")
        if not new_rej:
            break
        rejected |= new_rej
        active -= new_rej

    return {
        "rank_ci": rank_ci_from_rejections(rejected, p),
        "rejected": rejected,
        "n_steps": step,
    }
