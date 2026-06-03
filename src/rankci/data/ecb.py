"""
ECB SPF individual-forecast loader.

The ECB Survey of Professional Forecasters publishes one CSV per survey
vintage (release quarter), named ``YYYYQq.csv``. Each CSV contains four
indicator blocks separated by blank rows:

  1. ``INFLATION EXPECTATIONS; YEAR-ON-YEAR CHANGE IN HICP``
  2. ``CORE INFLATION EXPECTATIONS; YEAR-ON-YEAR CHANGE IN CORE``  (later years)
  3. ``GROWTH EXPECTATIONS; YEAR-ON-YEAR CHANGE IN REAL GDP``
  4. ``EXPECTED UNEMPLOYMENT RATE; PERCENTAGE OF LABOUR FORCE``

Each block starts with the section title row, then a header row
``TARGET_PERIOD, FCT_SOURCE, POINT, <bin-edges>...``, then forecaster rows.
The ``ASSUMPTIONS`` block at the bottom is metadata, not forecasts.

TARGET_PERIOD encodes the forecast horizon and comes in three shapes:

- ``YYYY``       — calendar-year average rate for that year
- ``YYYY<Mon>``  — rolling 12-month rate as of the named month of YYYY
                   (e.g. ``2022Sep``, ``1999Nov``). Mapped to the calendar
                   quarter containing that month for horizon alignment.
- ``YYYYQq``     — quarterly rate referring to that specific quarter

This module returns a *long* DataFrame keyed by
``(survey_year, survey_quarter, indicator, target_period, forecaster_id)``.
Pivoting to the wide ``(target × forecaster)`` panel that the rank-CI engines
consume requires realized values; that step lives in
``compute_error_panel`` once a realization series is supplied.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ── Indicator labels ────────────────────────────────────────────────────────

INDICATORS: dict[str, str] = {
    "HICP":       "INFLATION EXPECTATIONS",
    "CORE_HICP":  "CORE INFLATION EXPECTATIONS",
    "RGDP":       "GROWTH EXPECTATIONS",
    "UNEMP":      "EXPECTED UNEMPLOYMENT RATE",
}
_HEADER_TO_INDICATOR = {v: k for k, v in INDICATORS.items()}

_TARGET_RE_YEAR  = re.compile(r"^\d{4}$")
_TARGET_RE_MONTH = re.compile(r"^(\d{4})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$")
_TARGET_RE_QTR   = re.compile(r"^(\d{4})Q([1-4])$")

_MONTH_TO_QUARTER = {
    "Jan": 1, "Feb": 1, "Mar": 1,
    "Apr": 2, "May": 2, "Jun": 2,
    "Jul": 3, "Aug": 3, "Sep": 3,
    "Oct": 4, "Nov": 4, "Dec": 4,
}

_SURVEY_FILE_RE  = re.compile(r"^(\d{4})Q([1-4])\.csv$", re.IGNORECASE)


# ── Survey-quarter parsing ──────────────────────────────────────────────────

def parse_survey_quarter(path: str | Path) -> tuple[int, int]:
    """Extract (survey_year, survey_quarter) from a ``YYYYQq.csv`` filename."""
    name = Path(path).name
    m = _SURVEY_FILE_RE.match(name)
    if not m:
        raise ValueError(f"Filename {name!r} does not match YYYYQq.csv")
    return int(m.group(1)), int(m.group(2))


# ── Target-period parsing ───────────────────────────────────────────────────

def classify_target(target: str) -> dict:
    """
    Classify a TARGET_PERIOD string.

    Returns a dict with keys:
      ``kind``        : ``"year"`` | ``"month"`` | ``"quarter"`` | ``"unknown"``
      ``target_year`` : int or None
      ``target_qtr``  : int or None (for ``"month"`` and ``"quarter"`` kinds)
      ``target_month``: int or None (1-12, only for ``"month"`` kind)
    """
    s = str(target).strip()
    if _TARGET_RE_YEAR.match(s):
        return {"kind": "year", "target_year": int(s),
                "target_qtr": None, "target_month": None}
    m = _TARGET_RE_MONTH.match(s)
    if m:
        year = int(m.group(1))
        mon = m.group(2)
        return {"kind": "month", "target_year": year,
                "target_qtr": _MONTH_TO_QUARTER[mon],
                "target_month": list(_MONTH_TO_QUARTER).index(mon) + 1}
    m = _TARGET_RE_QTR.match(s)
    if m:
        return {"kind": "quarter",
                "target_year": int(m.group(1)),
                "target_qtr": int(m.group(2)),
                "target_month": None}
    return {"kind": "unknown", "target_year": None,
            "target_qtr": None, "target_month": None}


# ── Single-CSV parser ───────────────────────────────────────────────────────

def _is_section_header(cells: list[str]) -> str | None:
    """If ``cells[0]`` starts with one of the known section titles, return the
    indicator key (HICP/CORE_HICP/RGDP/UNEMP). Otherwise None."""
    if not cells:
        return None
    first = cells[0].strip()
    for prefix, key in _HEADER_TO_INDICATOR.items():
        if first.startswith(prefix):
            return key
    return None


def _coerce_float(x):
    if x is None or x == "":
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_ecb_csv(path: str | Path) -> pd.DataFrame:
    """
    Parse one ECB SPF survey CSV into a long DataFrame.

    Returns
    -------
    DataFrame with columns:
      survey_year, survey_quarter, indicator, target_period,
      target_kind, target_year, target_qtr, forecaster_id, point
    """
    path = Path(path)
    survey_year, survey_quarter = parse_survey_quarter(path)
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False,
                      engine="python", on_bad_lines="skip")
    rows: list[dict] = []

    current_indicator: str | None = None
    in_section = False
    saw_column_header = False

    for _, line in raw.iterrows():
        cells = [str(c) if c is not None else "" for c in line.tolist()]
        cells = [c.strip() for c in cells]
        first = cells[0] if cells else ""

        # End-of-data section
        if first.startswith("ASSUMPTIONS"):
            current_indicator = None
            in_section = False
            continue

        indicator_hit = _is_section_header(cells)
        if indicator_hit is not None:
            current_indicator = indicator_hit
            in_section = True
            saw_column_header = False
            continue

        if not in_section or current_indicator is None:
            continue

        if not saw_column_header:
            # The first non-blank row inside a section is the column header
            if first.upper().startswith("TARGET_PERIOD"):
                saw_column_header = True
            continue

        if first == "" and (len(cells) < 2 or cells[1] == ""):
            # blank separator row
            continue

        # Data row: TARGET_PERIOD, FCT_SOURCE, POINT, <bin probs...>
        target = first
        fid_raw = cells[1] if len(cells) > 1 else ""
        point_raw = cells[2] if len(cells) > 2 else ""
        if target == "" or fid_raw == "":
            continue
        try:
            fid = int(fid_raw)
        except ValueError:
            continue

        cls = classify_target(target)
        rows.append({
            "survey_year":    survey_year,
            "survey_quarter": survey_quarter,
            "indicator":      current_indicator,
            "target_period":  target,
            "target_kind":    cls["kind"],
            "target_year":    cls["target_year"],
            "target_qtr":     cls["target_qtr"],
            "target_month":   cls["target_month"],
            "forecaster_id":  fid,
            "point":          _coerce_float(point_raw),
        })

    return pd.DataFrame(rows)


# ── Multi-CSV loader ────────────────────────────────────────────────────────

def list_survey_csvs(directory: str | Path) -> list[Path]:
    """Return all ``YYYYQq.csv`` paths under *directory*, sorted chronologically."""
    directory = Path(directory)
    files = [p for p in directory.iterdir()
             if p.is_file() and _SURVEY_FILE_RE.match(p.name)]
    files.sort(key=lambda p: parse_survey_quarter(p))
    return files


def load_ecb_spf(
    directory: str | Path,
    indicators: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Load all ECB SPF survey CSVs under *directory* into one long DataFrame.

    Parameters
    ----------
    directory  : folder containing ``YYYYQq.csv`` files.
    indicators : subset of ``INDICATORS`` keys to keep. Default: all four.

    Returns
    -------
    Long DataFrame as in :func:`load_ecb_csv`, concatenated across surveys.
    """
    if indicators is not None:
        indicators = set(indicators)
        for k in indicators:
            if k not in INDICATORS:
                raise ValueError(
                    f"Unknown indicator {k!r}. Choose from {list(INDICATORS)}.",
                )
    frames = []
    for p in list_survey_csvs(directory):
        df = load_ecb_csv(p)
        if indicators is not None and not df.empty:
            df = df[df["indicator"].isin(indicators)]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(
            columns=["survey_year", "survey_quarter", "indicator",
                     "target_period", "target_kind", "target_year",
                     "target_qtr", "target_month", "forecaster_id", "point"],
        )
    return pd.concat(frames, ignore_index=True)


# ── Horizon assignment ─────────────────────────────────────────────────────

def horizon_in_quarters(survey_year: int, survey_quarter: int,
                        target_year: int, target_qtr: int | None,
                        target_kind: str) -> int | None:
    """
    Horizon, measured in quarters from the survey to the target.

    For yearly targets we anchor at Q4 (end-of-year, matching typical
    realization timing for annual averages). For monthly rolling targets
    (``YYYYMon``) and explicit quarterly targets we use the calendar
    quarter of the named month / quarter. Returns ``None`` for unknown kinds.
    """
    if target_year is None:
        return None
    if target_kind == "year":
        tq = 4
    elif target_kind in ("month", "quarter"):
        tq = target_qtr
    else:
        return None
    if tq is None:
        return None
    return (target_year - survey_year) * 4 + (tq - survey_quarter)


def add_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``horizon_q`` column (quarters survey→target) to a long panel."""
    out = df.copy()
    out["horizon_q"] = [
        horizon_in_quarters(
            r.survey_year, r.survey_quarter,
            r.target_year, r.target_qtr, r.target_kind,
        )
        for r in out.itertuples(index=False)
    ]
    return out


# ── Pivot to wide error panel ──────────────────────────────────────────────

def error_panel(
    long_df: pd.DataFrame,
    realized: pd.Series,
    indicator: str,
    target_kind: str,
    horizon_q: int | None = None,
    metric: str = "squared",
) -> pd.DataFrame:
    """
    Pivot point forecasts into a wide ``(target_period × forecaster_id)`` panel
    of losses against *realized* values.

    Parameters
    ----------
    long_df     : output of :func:`load_ecb_spf` (or filtered subset).
    realized    : Series indexed by ``target_period`` (string) with the same
                  scale as the forecasts (e.g. percent, percentage points).
    indicator   : indicator key to keep (HICP / CORE_HICP / RGDP / UNEMP).
    target_kind : ``"year"`` | ``"dec"`` | ``"quarter"`` to keep.
    horizon_q   : if set, also restrict to rows whose computed horizon matches.
    metric      : ``"squared"`` or ``"absolute"``.

    Returns
    -------
    Wide DataFrame indexed by ``target_period`` (string), one column per
    forecaster_id, values are the chosen loss.
    """
    df = long_df[(long_df["indicator"] == indicator)
                 & (long_df["target_kind"] == target_kind)
                 & long_df["point"].notna()].copy()

    if horizon_q is not None:
        if "horizon_q" not in df.columns:
            df = add_horizon(df)
        df = df[df["horizon_q"] == horizon_q]

    if df.empty:
        return pd.DataFrame()

    realized_aligned = df["target_period"].map(realized)
    err = df["point"] - realized_aligned
    if metric == "squared":
        df["loss"] = err ** 2
    elif metric == "absolute":
        df["loss"] = err.abs()
    else:
        raise ValueError(f"metric must be 'squared' or 'absolute', got {metric!r}")

    df = df.dropna(subset=["loss"])
    return df.pivot_table(index="target_period",
                          columns="forecaster_id",
                          values="loss",
                          aggfunc="mean")
