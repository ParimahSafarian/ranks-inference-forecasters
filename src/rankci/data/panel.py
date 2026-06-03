"""Source-agnostic panel helpers.

Anything that operates on a wide (quarters × forecaster_id) panel after the
errors have been computed lives here, regardless of where the errors came
from (Philly Fed RTDSM, ECB SPF, simulated).
"""
import numpy as np
import pandas as pd


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
    X_wide  : Wide panel (rows=target periods, cols=forecaster IDs).
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
