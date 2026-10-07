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

A second source, :func:`load_m3_official`, gives the forecasts the 24 M3
competitors actually submitted (CRAN package Mcomp); :func:`official_panel` turns
one indicator of it into a loss panel. The thesis application uses the Paris
temperature record from this source.

Forecasting methods (statsmodels) and the .rda reader (rdata) are imported lazily,
so importing this module does not require them unless you actually build a panel.
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


# ── official M3 submissions (the competitors' own forecasts) ─────────────────
#
# The forecasts actually submitted by the 24 M3 methods, with the held-out
# actuals and a description of every series, as distributed in the CRAN package
# Mcomp. Version 2.8 is pinned and checksum-verified so the numbers cannot drift
# (CRAN moves a release to Archive/ once superseded, so both URLs are tried). The
# .rda files are read in pure Python with ``rdata`` (application-only dep).

MCOMP_URLS = (
    "https://cran.r-project.org/src/contrib/Mcomp_2.8.tar.gz",
    "https://cran.r-project.org/src/contrib/Archive/Mcomp/Mcomp_2.8.tar.gz",
)
MCOMP_SHA256 = "c0e873054f91b4345f963d5e03df067dfcd512fe55d4be31d0c476d84676e5f3"
OFFICIAL_ID_COLS = ["series", "description", "type", "period", "horizon", "actual"]


def _fetch_mcomp(directory) -> Path:
    """Download the pinned Mcomp tarball once; return the dir holding the .rda files."""
    import hashlib
    import tarfile
    import urllib.error
    import urllib.request
    out = Path(directory) / "Mcomp_2.8"
    if (out / "M3.rda").exists() and (out / "M3Forecast.rda").exists():
        return out
    tgz = Path(directory) / "Mcomp_2.8.tar.gz"
    if not tgz.exists():
        Path(directory).mkdir(parents=True, exist_ok=True)
        for url in MCOMP_URLS:
            try:
                urllib.request.urlretrieve(url, tgz)
                break
            except urllib.error.HTTPError:
                continue
        else:
            raise RuntimeError(f"could not download Mcomp 2.8 from {MCOMP_URLS}")
    digest = hashlib.sha256(tgz.read_bytes()).hexdigest()
    if digest != MCOMP_SHA256:
        tgz.unlink()
        raise RuntimeError(f"Mcomp tarball checksum mismatch ({digest}); deleted it")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz) as tf:
        for name in ("M3.rda", "M3Forecast.rda"):
            member = tf.getmember(f"Mcomp/data/{name}")
            (out / name).write_bytes(tf.extractfile(member).read())
    return out


def load_m3_official(directory: str = "data/mcomp", cache: bool = True) -> pd.DataFrame:
    """
    Long DataFrame of the official M3 forecasts: one row per (series, horizon)
    with ``OFFICIAL_ID_COLS`` plus one column per method (24). A method that did
    not forecast a series (AAM1/AAM2 skipped the yearly group) is NaN there.
    Caches to ``<directory>/m3_official_long.csv``.
    """
    cache_path = Path(directory) / "m3_official_long.csv"
    if cache and cache_path.exists():
        return pd.read_csv(cache_path)

    import warnings
    import rdata
    src = _fetch_mcomp(directory)
    with warnings.catch_warnings():               # Mdata/Mcomp classes -> plain dicts
        warnings.simplefilter("ignore")
        m3 = rdata.read_rda(src / "M3.rda", default_encoding="latin1")["M3"]
        fc = rdata.read_rda(src / "M3Forecast.rda")["M3Forecast"]

    methods = [str(m) for m in fc]
    blocks = []
    for sn in m3:
        s = m3[sn]
        actual = np.asarray(s["xx"], float).ravel()
        h = actual.size
        block = pd.DataFrame({
            "series": str(sn),
            "description": str(s["description"][0]),
            "type": str(s["type"][0]),
            "period": str(s["period"][0]),
            "horizon": np.arange(1, h + 1),
            "actual": actual,
        })
        for m in methods:
            # rows are keyed by series id; AAM1/AAM2 lack some series entirely
            f = fc[m]
            block[m] = (np.asarray(f.loc[str(sn)].iloc[:h], float)
                        if str(sn) in f.index else np.nan)
        blocks.append(block)
    df = pd.concat(blocks, ignore_index=True)
    if cache:
        Path(directory).mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


def official_panel_from_long(df: pd.DataFrame, description: str) -> pd.DataFrame:
    """
    ``(series, horizon) x method`` sMAPE panel for every series with the given
    description. Rows keep the source order (for a record split into consecutive
    blocks, that is calendar time); methods with any missing forecast are dropped.
    """
    g = df[df["description"] == description]
    if g.empty:
        raise ValueError(f"no M3 series with description {description!r}")
    methods = [c for c in df.columns
               if c not in OFFICIAL_ID_COLS and g[c].notna().all()]
    a = g["actual"].to_numpy(float)[:, None]
    F = g[methods].to_numpy(float)
    denom = np.abs(F) + np.abs(a)
    loss = np.where(denom == 0, 0.0, 200 * np.abs(F - a) / denom)
    index = pd.MultiIndex.from_arrays([g["series"], g["horizon"]],
                                      names=["series", "horizon"])
    return pd.DataFrame(loss, index=index, columns=methods)


def official_panel(description: str = "Average temperature in Paris",
                   directory: str = "data/mcomp") -> pd.DataFrame:
    """sMAPE panel of the official M3 submissions on one indicator (see
    :func:`official_panel_from_long`). The default is the Paris monthly mean
    temperature record: 17 consecutive 8-year blocks, 1857-1992, 18 held-out
    months each -> a 306 x 24 panel."""
    return official_panel_from_long(load_m3_official(directory), description)
