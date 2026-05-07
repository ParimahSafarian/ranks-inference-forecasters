"""
Forecasting models to compare against the SPF panel.

Each model implements the signature

    forecast(history: np.ndarray, h: int) -> float

where `history` is the 1-D real-time series available at the survey quarter
and `h` is the forecast horizon (in quarters from the last observation).

The driver functions handle vintage selection so every model "sees" only the
data the SPF forecaster would have seen in survey quarter S — the vintage
column published in S, restricted to rows ≤ S-1.
"""
import numpy as np
import pandas as pd

from .data import HORIZON_OFFSETS, get_advance_estimate


# ── Vintage helpers ──────────────────────────────────────────────────────────

def vintage_col_at_survey(survey_year: int, survey_quarter: int, prefix: str) -> str:
    """Column name of the vintage published in survey quarter S."""
    return f"{prefix}{str(survey_year)[2:]}Q{survey_quarter}"


def history_at_survey(rtdsm: pd.DataFrame, survey_year: int, survey_quarter: int) -> pd.Series | None:
    """
    Return the real-time series available at survey quarter S, indexed by
    target quarter (YEAR, QUARTER), through S-1.

    Uses `rtdsm.attrs['prefix']` to find the vintage column. Returns None if
    the vintage column is missing.
    """
    prefix = rtdsm.attrs.get("prefix", "NOUTPUT")
    col = vintage_col_at_survey(survey_year, survey_quarter, prefix)
    if col not in rtdsm.columns:
        return None

    cutoff = (survey_year, survey_quarter - 1) if survey_quarter > 1 else (survey_year - 1, 4)
    series = rtdsm[col].dropna().sort_index()
    return series.loc[:cutoff]


# ── Models ──────────────────────────────────────────────────────────────────

def forecast_naive(history: np.ndarray, h: int = 1) -> float:
    """Random-walk forecast: ŷ_{T+h} = y_T. h is ignored."""
    if len(history) == 0:
        return float("nan")
    return float(history[-1])


def _fit_ar1(y: np.ndarray) -> tuple[float, float]:
    """OLS fit of y_t = c + φ y_{t-1} + ε_t. Returns (c, φ)."""
    A = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    coef, *_ = np.linalg.lstsq(A, y[1:], rcond=None)
    return float(coef[0]), float(coef[1])


def forecast_ar1(
    history: np.ndarray,
    h: int = 1,
    transform: str = "levels",
) -> float:
    """
    AR(1) forecast iterated h steps ahead.

    Parameters
    ----------
    history   : 1-D array of historical values.
    h         : number of steps ahead from the last observation.
    transform : "levels"   — fit AR(1) on raw values
                "diff"     — fit AR(1) on first differences, integrate back
                "log_diff" — fit AR(1) on log first differences (growth rates)
    """
    y = np.asarray(history, dtype=float)
    if len(y) < 3:
        return float("nan")

    if transform == "levels":
        c, phi = _fit_ar1(y)
        yhat = y[-1]
        for _ in range(h):
            yhat = c + phi * yhat
        return float(yhat)

    if transform == "log_diff":
        if (y <= 0).any():
            return float("nan")
        y = np.log(y)

    if transform in ("diff", "log_diff"):
        d = np.diff(y)
        if len(d) < 3:
            return float("nan")
        c, phi = _fit_ar1(d)
        # Iterate h forecasts of the difference series
        d_path = []
        d_curr = d[-1]
        for _ in range(h):
            d_curr = c + phi * d_curr
            d_path.append(d_curr)
        yhat = y[-1] + sum(d_path)
        return float(np.exp(yhat) if transform == "log_diff" else yhat)

    raise ValueError(f"transform must be 'levels', 'diff', or 'log_diff', got {transform!r}")


# ── Drivers ─────────────────────────────────────────────────────────────────

def model_forecast_series(
    rtdsm: pd.DataFrame,
    model_fn,
    target_quarters,
    spf_horizon: int = 3,
) -> pd.Series:
    """
    Generate model forecasts for each target quarter, using only data the
    SPF forecaster at survey S would have seen.

    Parameters
    ----------
    rtdsm         : RTDSM vintage matrix with `attrs['prefix']` set.
    model_fn      : callable (history: np.ndarray, h: int) -> float.
                    Use functools.partial to bind extra kwargs.
    target_quarters : iterable of (year, quarter) tuples.
    spf_horizon   : SPF horizon convention. 3 = one-step-ahead (NGDP3).
                    Maps to model horizon h = HORIZON_OFFSETS[spf_horizon] + 1
                    (forecast steps from history through S-1 to target T).

    Returns
    -------
    Series of forecasts indexed by (YEAR, QUARTER).
    """
    offset = HORIZON_OFFSETS[spf_horizon]
    h = offset + 1

    forecasts = {}
    for (T_y, T_q) in target_quarters:
        S_total = (T_y - 1) * 4 + (T_q - 1) - offset
        S_y = (S_total // 4) + 1
        S_q = (S_total % 4) + 1

        history = history_at_survey(rtdsm, S_y, S_q)
        if history is None or len(history) < 3:
            forecasts[(T_y, T_q)] = np.nan
            continue

        try:
            forecasts[(T_y, T_q)] = float(model_fn(history.values, h=h))
        except Exception:
            forecasts[(T_y, T_q)] = np.nan

    s = pd.Series(forecasts)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["YEAR", "QUARTER"])
    return s.sort_index()


def model_error_panel(
    rtdsm: pd.DataFrame,
    models: dict,
    target_quarters,
    spf_horizon: int = 3,
    metric: str = "squared",
) -> pd.DataFrame:
    """
    Run several models against the same target quarters and return a wide
    panel of errors, ready to concatenate with the SPF panel.

    Parameters
    ----------
    models : dict mapping model name → callable (history, h) → float.
             Use functools.partial to bind extra kwargs (transform, etc.).
    metric : "squared" → (forecast - actual)²
             "absolute" → |forecast - actual|

    Returns
    -------
    Wide DataFrame indexed by (YEAR, QUARTER) with one column per model,
    values are the chosen metric of (forecast - actual).
    """
    target_quarters = list(target_quarters)

    actuals = pd.Series(
        {(y, q): get_advance_estimate(y, q, rtdsm) for (y, q) in target_quarters},
    )
    actuals.index = pd.MultiIndex.from_tuples(actuals.index, names=["YEAR", "QUARTER"])
    actuals = actuals.sort_index()

    panel = {}
    for name, fn in models.items():
        forecasts = model_forecast_series(rtdsm, fn, target_quarters, spf_horizon=spf_horizon)
        err = forecasts - actuals
        if metric == "squared":
            panel[name] = err ** 2
        elif metric == "absolute":
            panel[name] = err.abs()
        else:
            raise ValueError(f"metric must be 'squared' or 'absolute', got {metric!r}")

    return pd.DataFrame(panel)
