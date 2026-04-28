"""
Data loading and error computation for SPF / RTDSM analysis.

- load_spf:                  Load SPF microdata (NGDP sheet).
- load_rtdsm:                Load and parse the NOUTPUT vintage matrix.
- advance_vintage_col:       Map a target quarter to its advance-estimate column name.
- get_advance_estimate:      Look up a single advance estimate.
- compute_errors:            Compute forecast errors for all horizons, return a long DataFrame.
- compute_squared_error_panel: Convenience wrapper — compute errors, square them,
                               filter to one horizon, and pivot to wide format.
"""
import numpy as np
import pandas as pd


# ── SPF microdata ────────────────────────────────────────────────────────────

def load_spf(path: str, sheet: str = "NGDP") -> pd.DataFrame:
    """Load the SPF microdata for a given variable (default: NGDP)."""
    return pd.read_excel(path, sheet_name=sheet)


# ── RTDSM vintage matrix ────────────────────────────────────────────────────

def load_rtdsm(path: str) -> pd.DataFrame:
    """
    Load the NOUTPUT vintage matrix and index by (YEAR, QUARTER).

    The DATE column (e.g. "1947:Q1") is parsed into integer YEAR and QUARTER.
    """
    noutput = pd.read_excel(path)
    noutput[["YEAR", "QUARTER"]] = (
        noutput["DATE"]
        .str.extract(r"(\d{4}):Q(\d)")
        .astype(int)
    )
    noutput = noutput.set_index(["YEAR", "QUARTER"]).drop(columns=["DATE"], errors="ignore")
    return noutput


def advance_vintage_col(target_year: int, target_quarter: int) -> str:
    """
    Return the RTDSM column name for the advance estimate of a target quarter.

    The advance estimate is published in the following quarter's vintage:
        Q1 → NOUTPUTYYQ2, Q2 → NOUTPUTYYQ3, Q3 → NOUTPUTYYQ4,
        Q4 → NOUTPUT(YY+1)Q1.
    """
    adv_quarter = target_quarter + 1
    adv_year = target_year
    if adv_quarter > 4:
        adv_quarter = 1
        adv_year += 1
    return f"NOUTPUT{str(adv_year)[2:]}Q{adv_quarter}"


def get_advance_estimate(
    target_year: int,
    target_quarter: int,
    noutput: pd.DataFrame,
) -> float:
    """
    Look up the advance estimate for a single target quarter.

    Returns np.nan if the quarter or vintage column is missing.
    """
    col = advance_vintage_col(target_year, target_quarter)
    try:
        val = noutput.loc[(target_year, target_quarter), col]
        return val if not pd.isna(val) else np.nan
    except KeyError:
        return np.nan


# ── Forecast error computation ───────────────────────────────────────────────

HORIZON_OFFSETS = {
    "NGDP1": -1,   # t-1  (previous quarter, near-historical)
    "NGDP2":  0,   # t+0  (nowcast)
    "NGDP3":  1,   # t+1  (one-quarter-ahead)
    "NGDP4":  2,   # t+2
    "NGDP5":  3,   # t+3
    "NGDP6":  4,   # t+4
}


def compute_errors(
    df: pd.DataFrame,
    noutput: pd.DataFrame,
    horizons: dict[str, int] | None = None,
) -> pd.DataFrame:
    """
    Compute forecast errors for each horizon: error = forecast - advance estimate.

    Parameters
    ----------
    df      : SPF microdata with columns YEAR, QUARTER, ID, INDUSTRY, NGDP1..NGDP6.
    noutput : RTDSM vintage matrix indexed by (YEAR, QUARTER).
    horizons: dict mapping column name to quarter offset.
              Defaults to HORIZON_OFFSETS (NGDP1..NGDP6).

    Returns
    -------
    DataFrame with columns: YEAR, QUARTER, ID, INDUSTRY, error_NGDP1..error_NGDP6.
    """
    if horizons is None:
        horizons = HORIZON_OFFSETS

    errors = df[["YEAR", "QUARTER", "ID", "INDUSTRY"] + list(horizons.keys())].copy()

    for col, offset in horizons.items():
        total = (errors["YEAR"] - 1) * 4 + (errors["QUARTER"] - 1) + offset
        t_year = ((total // 4) + 1).astype(int)
        t_qtr = ((total % 4) + 1).astype(int)

        gdp_actual = np.array([
            get_advance_estimate(y, q, noutput)
            for y, q in zip(t_year, t_qtr)
        ])

        errors[f"error_{col}"] = errors[col].values - gdp_actual
        errors.drop(columns=[col], inplace=True)

    return errors


def compute_squared_error_panel(
    df: pd.DataFrame,
    noutput: pd.DataFrame,
    horizon: str = "NGDP3",
) -> pd.DataFrame:
    """
    End-to-end: compute errors → square → pivot to wide (rows=quarters, cols=forecaster IDs).

    Parameters
    ----------
    df      : SPF microdata.
    noutput : RTDSM vintage matrix.
    horizon : which horizon column to keep (default "NGDP3").

    Returns
    -------
    Wide DataFrame indexed by (YEAR, QUARTER) with one column per forecaster ID,
    values are squared forecast errors.
    """
    errors = compute_errors(df, noutput)
    error_col = f"error_{horizon}"

    panel = errors[["YEAR", "QUARTER", "ID", error_col]].copy()
    panel["squared_error"] = panel[error_col] ** 2

    wide = panel.pivot_table(
        index=["YEAR", "QUARTER"],
        columns="ID",
        values="squared_error",
    )
    return wide


def winsorize_panel(
    X: np.ndarray,
    upper_pct: float = 95,
) -> np.ndarray:
    """
    Winsorize each column of X at its upper percentile (NaNs preserved).

    For squared errors, only the upper tail needs clipping (lower bound is 0).

    Parameters
    ----------
    X         : (n, p) array, may contain NaN.
    upper_pct : percentile cap (e.g. 95 = clip top 5%).

    Returns
    -------
    Winsorized copy of X (same shape, NaNs preserved).
    """
    Xw = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            continue
        cap = np.percentile(col[valid], upper_pct)
        Xw[valid, j] = np.clip(col[valid], None, cap)
    return Xw


def select_top_forecasters(
    X_wide: pd.DataFrame,
    N: int,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    Select the top-N forecasters by observation count (with at least min_obs).

    Parameters
    ----------
    X_wide  : Wide panel (rows=quarters, cols=forecaster IDs).
    N       : Number of forecasters to keep.
    min_obs : Minimum observations required to be considered.

    Returns
    -------
    Filtered wide DataFrame with N columns (forecasters).
    """
    obs_counts = X_wide.notna().sum()
    eligible = obs_counts[obs_counts >= min_obs]
    top_ids = eligible.nlargest(N).index
    return X_wide[top_ids]
