"""
M-competition application: rank standard forecasting methods by per-series accuracy.

Step 4 "sharp" application. Each of the M3 series is forecast by several standard
methods; the per-series sMAPE forms a ``(series x method)`` loss panel fed to the
same rank-CI pipeline used for the forecaster panels. With thousands of series the
ranking is sharp, and — because the methods are highly correlated across series (a
hard series hurts everyone) — the paired *differences* have small variance, so the
method separates methods that look tied marginally.

Structure mirrors GJP / macro: each series' difficulty is a shock shared by every
method → it cancels in the sMAPE differences (cross-sectional common shock). Series
are unordered, so serial dependence ~ 0 (this isolates the cross-sectional / Omega
side). Panel is essentially balanced (every method scores every series).

Forecasting methods (statsmodels) are imported lazily, so importing this module does
not require statsmodels/datasetsforecast unless you actually build a panel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# horizon and seasonal period per M3 frequency group
HORIZON = {"Yearly": 6, "Quarterly": 8, "Monthly": 18, "Other": 8}
SEASON = {"Yearly": 1, "Quarterly": 4, "Monthly": 12, "Other": 1}


# ── data ─────────────────────────────────────────────────────────────────────

def load_m3(group: str = "Monthly", directory: str = "data/mcomp") -> pd.DataFrame:
    """Load one M3 frequency group as a long DataFrame (unique_id, ds, y).
    The last ``HORIZON[group]`` points of each series are the test window."""
    from datasetsforecast.m3 import M3
    df, *_ = M3.load(directory=directory, group=group)
    return df[["unique_id", "ds", "y"]].copy()


# ── scores ───────────────────────────────────────────────────────────────────

def smape(f, a) -> float:
    f, a = np.asarray(f, float), np.asarray(a, float)
    denom = np.abs(f) + np.abs(a)
    return float(np.mean(np.where(denom == 0, 0.0, 200 * np.abs(f - a) / denom)))


def mase(f, a, train, m) -> float:
    f, a, train = np.asarray(f, float), np.asarray(a, float), np.asarray(train, float)
    scale = np.mean(np.abs(train[m:] - train[:-m])) if len(train) > m else np.nan
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return float(np.mean(np.abs(f - a)) / scale)


# ── forecasting methods: (train, H, m) -> forecast of length H ───────────────

def _seasonal_indices(y, m):
    """Multiplicative seasonal indices by calendar position (ratio to level)."""
    pos = np.arange(len(y)) % m
    idx = np.array([np.mean(y[pos == p]) for p in range(m)])
    idx = idx / idx.mean()
    idx[idx <= 0] = 1.0
    return idx


def _is_seasonal(y, m):
    """M4-style seasonality test: |r_m| beyond the 90% white-noise band."""
    if m <= 1 or len(y) < 3 * m:
        return False
    yc = y - y.mean()
    r = np.array([np.dot(yc[k:], yc[:-k]) / np.dot(yc, yc) for k in range(1, m + 1)])
    band = 1.645 * np.sqrt((1 + 2 * np.sum(r[:-1] ** 2)) / len(y))
    return abs(r[-1]) > band


def m_naive(train, H, m):
    return np.repeat(train[-1], H)


def m_snaive(train, H, m):
    if m <= 1:
        return m_naive(train, H, m)
    return np.array([train[-m + (i % m)] for i in range(H)])


def m_naive2(train, H, m):
    """Seasonally-adjusted naive (the M-competition benchmark)."""
    if m > 1 and _is_seasonal(train, m):
        s = _seasonal_indices(train, m)
        pos = np.arange(len(train)) % m
        deseason = train / s[pos]
        fut = (np.arange(len(train), len(train) + H)) % m
        return deseason[-1] * s[fut]
    return m_naive(train, H, m)


def m_ets(train, H, m, damped=False):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    seasonal = "add" if (m > 1 and len(train) >= 2 * m) else None
    try:
        fit = ExponentialSmoothing(
            train, trend="add", damped_trend=damped,
            seasonal=seasonal, seasonal_periods=(m if seasonal else None),
            initialization_method="estimated",
        ).fit()
        return np.asarray(fit.forecast(H))
    except Exception:
        # fall back to non-seasonal, then to naive
        try:
            fit = ExponentialSmoothing(train, trend="add", damped_trend=damped,
                                       initialization_method="estimated").fit()
            return np.asarray(fit.forecast(H))
        except Exception:
            return m_naive(train, H, m)


def m_damped(train, H, m):
    return m_ets(train, H, m, damped=True)


def m_theta(train, H, m):
    from statsmodels.tsa.forecasting.theta import ThetaModel
    try:
        fit = ThetaModel(np.asarray(train, float), period=(m if m > 1 else 1),
                         deseasonalize=(m > 1)).fit()
        return np.asarray(fit.forecast(H))
    except Exception:
        return m_naive(train, H, m)


METHODS = {
    "naive": m_naive,
    "naive2": m_naive2,
    "snaive": m_snaive,
    "ets": m_ets,
    "damped": m_damped,
    "theta": m_theta,
}
# a combination of the serious methods (combinations are strong in the M-competitions)
_COMB_PARTS = ("ets", "damped", "theta")


# ── panel construction ───────────────────────────────────────────────────────

def build_panel(df, group, methods=None, metric="smape", include_comb=True):
    """
    Build the ``(series x method)`` loss panel from a long DataFrame
    ``(unique_id, ds, y)``. Data-source-agnostic (used by :func:`method_panel`
    and directly by tests).
    """
    if methods is None:
        methods = list(METHODS)
    H, m = HORIZON[group], SEASON[group]
    score_fn = smape if metric == "smape" else mase

    rows = []
    for uid, g in df.groupby("unique_id", sort=False):
        y = g["y"].to_numpy(float)
        tr, te = y[:-H], y[-H:]
        fcs = {}
        for name in methods:
            try:
                f = np.asarray(METHODS[name](tr, H, m), float)
            except Exception:
                f = np.full(H, np.nan)
            fcs[name] = f
        if include_comb:
            parts = [fcs[p] for p in _COMB_PARTS if p in fcs]
            if parts:
                fcs["comb"] = np.mean(parts, axis=0)
        rec = {"series": uid}
        for name, f in fcs.items():
            rec[name] = (score_fn(f, te, tr, m) if metric == "mase"
                         else score_fn(f, te))
        rows.append(rec)
    return pd.DataFrame(rows).set_index("series")


def method_panel(
    group: str = "Monthly",
    methods=None,
    metric: str = "smape",
    directory: str = "data/mcomp",
    cache: bool = True,
    include_comb: bool = True,
):
    """
    Build the ``(series x method)`` loss panel for one M3 group (loads M3, then
    :func:`build_panel`). Caches to ``<directory>/panel_<group>_<metric>.csv``
    (delete the file to rebuild).
    """
    cache_path = Path(directory) / f"panel_{group}_{metric}.csv"
    if cache and cache_path.exists():
        return pd.read_csv(cache_path, index_col=0)

    df = load_m3(group, directory=directory)
    panel = build_panel(df, group, methods=methods, metric=metric,
                        include_comb=include_comb)
    if cache:
        Path(directory).mkdir(parents=True, exist_ok=True)
        panel.to_csv(cache_path)
    return panel
