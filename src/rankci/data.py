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

def load_rtdsm(
    path: str,
    prefix: str = "NOUTPUT",
    freq: str = "quarterly",
) -> pd.DataFrame:
    """
    Load a Philly Fed real-time vintage matrix, indexed by (YEAR, QUARTER).

    Parameters
    ----------
    path   : path to the Excel file.
    prefix : column-name prefix (e.g. "NOUTPUT" for GDP, "RUC" for unemployment).
             Stored on the returned DataFrame as `.attrs['prefix']` so the
             rest of the pipeline can find it.
    freq   : "quarterly" — DATE like "1947:Q1", one row per quarter.
             "monthly"   — DATE like "1947:01", averaged to quarterly.
    """
    df = pd.read_excel(path)

    if freq == "quarterly":
        df[["YEAR", "QUARTER"]] = (
            df["DATE"].str.extract(r"(\d{4}):Q(\d)").astype(int)
        )
        df = df.set_index(["YEAR", "QUARTER"]).drop(columns=["DATE"], errors="ignore")
    elif freq == "monthly":
        df[["YEAR", "MONTH"]] = (
            df["DATE"].str.extract(r"(\d{4}):(\d{2})").astype(int)
        )
        df["QUARTER"] = ((df["MONTH"] - 1) // 3) + 1
        df = (
            df.drop(columns=["DATE", "MONTH"])
              .groupby(["YEAR", "QUARTER"]).mean()
        )
    else:
        raise ValueError(f"freq must be 'quarterly' or 'monthly', got {freq!r}")

    df.attrs["prefix"] = prefix
    return df


def advance_vintage_col(
    target_year: int,
    target_quarter: int,
    prefix: str = "NOUTPUT",
) -> str:
    """
    Return the RTDSM column name for the advance estimate of a target quarter.

    The advance estimate is published in the following quarter's vintage:
        Q1 → {prefix}YYQ2, Q2 → {prefix}YYQ3, Q3 → {prefix}YYQ4,
        Q4 → {prefix}(YY+1)Q1.
    """
    adv_quarter = target_quarter + 1
    adv_year = target_year
    if adv_quarter > 4:
        adv_quarter = 1
        adv_year += 1
    return f"{prefix}{str(adv_year)[2:]}Q{adv_quarter}"


def get_advance_estimate(
    target_year: int,
    target_quarter: int,
    rtdsm: pd.DataFrame,
) -> float:
    """
    Look up the advance estimate for a single target quarter.

    Uses `rtdsm.attrs['prefix']` to build the column name; defaults to "NOUTPUT".
    Returns np.nan if the quarter or vintage column is missing.
    """
    prefix = rtdsm.attrs.get("prefix", "NOUTPUT")
    col = advance_vintage_col(target_year, target_quarter, prefix)
    try:
        val = rtdsm.loc[(target_year, target_quarter), col]
        return val if not pd.isna(val) else np.nan
    except KeyError:
        return np.nan


# ── Forecast error computation ───────────────────────────────────────────────

# Horizon offsets (in quarters) applied to the survey quarter to get the target.
# Same convention across SPF indicators: 1 = previous (history), 2 = nowcast, 3..6 = ahead.
HORIZON_OFFSETS = {1: -1, 2: 0, 3: 1, 4: 2, 5: 3, 6: 4}


def compute_errors(
    df: pd.DataFrame,
    rtdsm: pd.DataFrame,
    indicator: str = "NGDP",
) -> pd.DataFrame:
    """
    Compute forecast errors for each horizon: error = forecast - advance estimate.

    Parameters
    ----------
    df        : SPF microdata with columns YEAR, QUARTER, ID, INDUSTRY,
                {indicator}1..{indicator}6.
    rtdsm     : RTDSM vintage matrix indexed by (YEAR, QUARTER).
    indicator : SPF column prefix, e.g. "NGDP" or "UNEMP".
    """
    horizon_cols = [f"{indicator}{h}" for h in HORIZON_OFFSETS]
    errors = df[["YEAR", "QUARTER", "ID", "INDUSTRY"] + horizon_cols].copy()

    for h, offset in HORIZON_OFFSETS.items():
        col = f"{indicator}{h}"
        total = (errors["YEAR"] - 1) * 4 + (errors["QUARTER"] - 1) + offset
        t_year = ((total // 4) + 1).astype(int)
        t_qtr = ((total % 4) + 1).astype(int)

        actual = np.array([
            get_advance_estimate(y, q, rtdsm)
            for y, q in zip(t_year, t_qtr)
        ])

        errors[f"error_{col}"] = errors[col].values - actual
        errors.drop(columns=[col], inplace=True)

    return errors


def compute_squared_error_panel(
    df: pd.DataFrame,
    rtdsm: pd.DataFrame,
    indicator: str = "NGDP",
    horizon: int = 3,
) -> pd.DataFrame:
    """
    End-to-end: compute errors → square → pivot to wide (rows=quarters, cols=forecaster IDs).

    Parameters
    ----------
    df        : SPF microdata.
    rtdsm     : RTDSM vintage matrix.
    indicator : SPF column prefix, e.g. "NGDP" or "UNEMP".
    horizon   : which horizon (1..6) to keep. Default 3 = one-quarter-ahead.
    """
    errors = compute_errors(df, rtdsm, indicator=indicator)
    error_col = f"error_{indicator}{horizon}"

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
