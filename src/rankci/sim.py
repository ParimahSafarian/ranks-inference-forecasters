"""
Monte Carlo data-generating process and coverage-study driver (thesis Section C).

Validates the covariance layer feeding MRSW on synthetic panels where the truth
(the theta ordering) is known. The DGP is the additive common-shock model the
theory assumes,

    X_{t,j} = theta_j + c_t + u_{t,j},

so the common shock c_t cancels in the differences d^{jk}_t = X_{t,j} - X_{t,k}
(the object the covariance estimator targets) but inflates level variances.

  - u_{t,j} : idiosyncratic AR(1), coef rho, stationary variance sigma_u^2 (serial dep).
  - c_t     : common shock, AR(1) coef rho_c, variance sigma_c^2 (cross-sectional dep).
  - theta_j : known means fixing the true rank (rank 1 = smallest theta).
  - imbalance: staggered entry/exit; 0 = balanced, up to nearly all-missing.

The head-to-head experiment runs ``covariance="mds"`` and ``"omega"`` on the SAME
panels and reports coverage, CI width, false-separation, and the PSD-projection
distance for each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .core.stepwise import rank_ci_stepwise_pairwise
from .core.stepwise_simulation import rank_ci_stepwise_simulation_pairwise


# ── Data-generating process ──────────────────────────────────────────────────

def _draw_innov(shape, tail_df, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance innovations: standard normal, or Student-t (heavy-tailed)
    rescaled to unit variance when ``tail_df`` is set (df > 2)."""
    if tail_df is None:
        return rng.standard_normal(shape)
    t = rng.standard_t(tail_df, size=shape)
    return t / np.sqrt(tail_df / (tail_df - 2.0))     # var(t_df) = df/(df-2)


def _ar1_panel(T: int, k: int, rho: float, sigma: float,
               rng: np.random.Generator, tail_df: float | None = None) -> np.ndarray:
    """(T, k) AR(1) columns with stationary variance sigma^2 and coef rho.

    ``tail_df`` (Student-t degrees of freedom) makes the innovations heavy-tailed
    while keeping unit-variance scaling — this is what pushes the loss-difference
    SE matrix away from Euclidean-embeddability and stresses the MDS route.
    """
    if sigma == 0.0:
        return np.zeros((T, k))
    innov_sd = sigma * np.sqrt(max(1.0 - rho**2, 0.0)) if abs(rho) < 1 else 0.0
    e = _draw_innov((T, k), tail_df, rng)
    y = np.empty((T, k))
    y[0] = e[0] * sigma                       # stationary initial draw
    for t in range(1, T):
        y[t] = rho * y[t - 1] + e[t] * innov_sd
    return y


def _volatility_path(T: int, n_episodes: int, vol_scale: float,
                     vol_len: int | None, rng: np.random.Generator) -> np.ndarray:
    """Length-T volatility multiplier: 1.0 baseline with ``n_episodes`` localized
    windows of height ``vol_scale`` (crisis-quarter non-stationarity). Placed at
    random starts; combined with staggered coverage this is what drives the
    pairwise SE matrix non-Euclidean and reveals the direct-Omega advantage."""
    if vol_len is None:
        vol_len = max(int(round(0.04 * T)), 3)
    w = np.ones(T)
    for _ in range(n_episodes):
        s = int(rng.integers(0, max(T - vol_len, 1)))
        w[s:s + vol_len] = vol_scale
    return w


def _apply_stagger(X: np.ndarray, imbalance: float, min_active: int) -> np.ndarray:
    """Staggered entry/exit: each forecaster active on an equal-length window of
    length (1-imbalance)*T, with start positions spread evenly across the sample."""
    T, p = X.shape
    L_active = int(round((1.0 - imbalance) * T))
    L_active = min(max(L_active, min_active), T)
    if p == 1 or L_active >= T:
        return X
    starts = [int(round(j / (p - 1) * (T - L_active))) for j in range(p)]
    Xm = X.copy()
    for j, s in enumerate(starts):
        Xm[:s, j] = np.nan
        Xm[s + L_active:, j] = np.nan
    return Xm


def simulate_panel(
    theta,
    T: int,
    rho: float = 0.3,
    sigma_u: float = 1.0,
    sigma_c: float = 1.0,
    rho_c: float = 0.0,
    imbalance: float = 0.0,
    tail_df: float | None = None,
    n_vol_episodes: int = 0,
    vol_scale: float = 1.0,
    vol_len: int | None = None,
    min_active: int = 10,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate one loss panel X_{t,j} = theta_j + c_t + w_t * u_{t,j}.

    Non-stationarity knobs (both keep the common shock additive, so it still
    cancels in differences):
      - ``tail_df`` : Student-t df for heavy-tailed idiosyncratic innovations.
      - ``n_vol_episodes`` / ``vol_scale`` / ``vol_len`` : localized volatility
        episodes (w_t), i.e. crisis quarters. Combined with ``imbalance`` these
        drive the pairwise SE matrix non-Euclidean and reveal the direct-Omega
        calibration advantage; without them MDS and Omega are ~equivalent.

    Returns
    -------
    X : (T, p) array, NaN where a forecaster is inactive (staggered entry/exit).
    """
    theta = np.asarray(theta, dtype=float)
    p = theta.size
    rng = rng if rng is not None else np.random.default_rng(seed)

    u = _ar1_panel(T, p, rho, sigma_u, rng, tail_df=tail_df)  # idiosyncratic
    if n_vol_episodes > 0 and vol_scale != 1.0:
        u = u * _volatility_path(T, n_vol_episodes, vol_scale, vol_len, rng)[:, None]
    c = _ar1_panel(T, 1, rho_c, sigma_c, rng)[:, 0]           # common shock
    X = theta[None, :] + c[:, None] + u

    if imbalance > 0.0:
        X = _apply_stagger(X, imbalance, min_active)
    return X


def true_ranks(theta) -> np.ndarray:
    """True ranks, rank 1 = smallest theta (best for MSE-style losses)."""
    theta = np.asarray(theta, dtype=float)
    order = np.argsort(theta, kind="stable")
    ranks = np.empty(theta.size, dtype=int)
    ranks[order] = np.arange(1, theta.size + 1)
    return ranks


# ── Coverage-study driver ────────────────────────────────────────────────────

def _accumulate(acc: dict, out: dict, tr: np.ndarray, distinct: bool,
                tied_pairs: list, p: int, has_proj: bool) -> None:
    ci = out["rank_ci"]
    if distinct:
        covered = np.array([ci[j, 0] <= tr[j] <= ci[j, 1] for j in range(p)], float)
        acc["cov_joint"] += float(covered.all())
        acc["cov_marg"] += covered
    acc["width"].append(float(np.mean(ci[:, 1] - ci[:, 0])))
    if tied_pairs:
        rej = out.get("rejected", set())
        for (j, k) in tied_pairs:
            acc["false"].append(1.0 if ((j, k) in rej or (k, j) in rej) else 0.0)
    if has_proj and out.get("proj_gap") is not None:
        acc["proj"].append(float(out["proj_gap"]))
    acc["steps"].append(float(out.get("n_steps", np.nan)))


def coverage_study(
    designs: list[dict],
    R: int = 500,
    B: int = 2000,
    alpha: float = 0.05,
    base_seed: int = 0,
    methods: tuple[str, ...] = ("mds", "omega"),
    include_bootstrap: bool = False,
    min_overlap: int = 2,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Run the coverage study over a list of design dicts.

    Each design is a dict with keys: ``name``, ``theta`` (array-like), ``T``, and
    optionally ``rho``, ``sigma_u``, ``sigma_c``, ``rho_c``, ``imbalance``.

    The same simulated panel is fed to every method (paired comparison). Returns
    one tidy row per (design, method) with: joint_coverage, marg_coverage_mean,
    marg_coverage_min, mean_width, false_sep_rate, proj_gap_mean, n_steps_mean.
    Coverage is reported only for distinct-theta designs; false-separation only
    for designs containing tied theta.
    """
    labels = list(methods) + (["bootstrap"] if include_bootstrap else [])
    rows = []

    for di, design in enumerate(designs):
        theta = np.asarray(design["theta"], dtype=float)
        p = theta.size
        T = int(design["T"])
        tr = true_ranks(theta)
        distinct = np.unique(theta).size == p
        tied_pairs = [(j, k) for j in range(p) for k in range(j + 1, p)
                      if theta[j] == theta[k]]

        acc = {m: {"cov_joint": 0.0, "cov_marg": np.zeros(p),
                   "width": [], "false": [], "proj": [], "steps": []}
               for m in labels}

        for r in range(R):
            data_seed = base_seed + di * 1_000_000 + r
            draw_seed = data_seed + 500_000_000        # decouple draw RNG from data
            X = simulate_panel(
                theta=theta, T=T,
                rho=design.get("rho", 0.3), sigma_u=design.get("sigma_u", 1.0),
                sigma_c=design.get("sigma_c", 1.0), rho_c=design.get("rho_c", 0.0),
                imbalance=design.get("imbalance", 0.0),
                tail_df=design.get("tail_df", None),
                n_vol_episodes=design.get("n_vol_episodes", 0),
                vol_scale=design.get("vol_scale", 1.0),
                vol_len=design.get("vol_len", None), seed=data_seed,
            )
            for m in methods:
                out = rank_ci_stepwise_simulation_pairwise(
                    X, alpha=alpha, B=B, seed=draw_seed, covariance=m,
                    min_overlap=min_overlap, verbose=False,
                )
                _accumulate(acc[m], out, tr, distinct, tied_pairs, p, has_proj=True)
            if include_bootstrap:
                outb = rank_ci_stepwise_pairwise(
                    X, alpha=alpha, B=B, seed=draw_seed, verbose=False,
                )
                _accumulate(acc["bootstrap"], outb, tr, distinct, tied_pairs, p,
                            has_proj=False)

        if progress:
            print(f"[{di+1}/{len(designs)}] {design.get('name','')} done "
                  f"(R={R}, p={p}, T={T})")

        meta = {k: design.get(k) for k in
                ("name", "T", "rho", "sigma_c", "rho_c", "imbalance", "tail_df",
                 "n_vol_episodes", "vol_scale")}
        meta["p"] = p
        for m in labels:
            a = acc[m]
            rows.append({
                **meta, "method": m,
                "joint_coverage": a["cov_joint"] / R if distinct else np.nan,
                "marg_coverage_mean": float((a["cov_marg"] / R).mean()) if distinct else np.nan,
                "marg_coverage_min": float((a["cov_marg"] / R).min()) if distinct else np.nan,
                "mean_width": float(np.mean(a["width"])),
                "false_sep_rate": float(np.mean(a["false"])) if tied_pairs else np.nan,
                "proj_gap_mean": float(np.mean(a["proj"])) if a["proj"] else np.nan,
                "n_steps_mean": float(np.mean(a["steps"])),
                "R": R, "B": B,
            })

    return pd.DataFrame(rows)
