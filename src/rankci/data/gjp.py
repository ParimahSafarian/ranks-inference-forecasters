"""
Good Judgment Project (GJP) loader and Brier-score panel.

Step 4 informative application: rank individual GJP forecasters by Brier score
through the same rank-CI pipeline used for macro forecasters. Public data
(Harvard Dataverse doi:10.7910/DVN/BPCDH5, CC0) lives in ``data/gjp/``:
``survey_fcasts.yrN.csv`` (individual daily forecasts, long format — one row per
answer option) and ``ifps.csv`` (question metadata + outcomes).

The panel is ``(question x forecaster)`` per-question Brier scores:
  - rows = questions, ordered by close date (the "time" axis for the HAC layer),
  - each question's realized outcome is shared by all forecasters who answered it
    -> a question-level common shock that cancels in Brier differences,
  - forecasters answer different question subsets -> an unbalanced panel.

Brier is the canonical multi-category tournament score, daily-averaged over the
life of each question with the last forecast carried forward (0 = best, 2 = worst).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_FC_COLS = ["ifp_id", "user_id", "team", "ctt", "fcast_type",
            "answer_option", "value", "fcast_date", "timestamp", "year"]


# ── loaders ──────────────────────────────────────────────────────────────────

def load_gjp_ifps(path: str | Path) -> pd.DataFrame:
    """Load ``ifps.csv``; keep Closed questions with a resolved outcome. Indexed
    by ``ifp_id`` with columns ``outcome``, ``date_start``, ``date_closed``,
    ``n_opts`` (dates parsed to Timestamps; M/D/YY format)."""
    ifp = pd.read_csv(path, encoding="latin-1",
                      usecols=["ifp_id", "q_status", "date_start", "date_closed",
                               "outcome", "n_opts"])
    ifp = ifp[ifp["q_status"].str.lower() == "closed"]
    ifp = ifp[ifp["outcome"].notna()].copy()
    for c in ("date_start", "date_closed"):
        ifp[c] = pd.to_datetime(ifp[c], format="%m/%d/%y", errors="coerce")
    ifp = ifp[ifp["date_closed"].notna()]
    return ifp.set_index("ifp_id")


def load_gjp_forecasts(paths) -> pd.DataFrame:
    """Load one or more ``survey_fcasts.yrN.csv`` files (long format), concatenated.
    ``fcast_date`` is parsed to Timestamp."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for p in paths:
        df = pd.read_csv(p, encoding="latin-1",
                         usecols=lambda c: c in _FC_COLS)
        frames.append(df)
    fc = pd.concat(frames, ignore_index=True)
    fc["fcast_date"] = pd.to_datetime(fc["fcast_date"], errors="coerce")
    return fc


# ── daily-averaged Brier for one (forecaster, question) ──────────────────────

def _brier_daily_avg(sub: pd.DataFrame, outcome: str, close: pd.Timestamp) -> float:
    """
    Canonical GJP score for one forecaster on one question: expand submissions to
    a daily grid (first forecast .. close), carry the last forecast forward, and
    average the daily multi-category Brier. Computed exactly by time-weighting the
    piecewise-constant daily Brier over the segments between updates.
    """
    sub = sub[sub["fcast_type"] != 4]                     # drop withdrawals
    if sub.empty:
        return np.nan
    # option-vector per date = the last submission that day
    piv = (sub.sort_values("timestamp")
              .pivot_table(index="fcast_date", columns="answer_option",
                           values="value", aggfunc="last")
              .sort_index())
    if outcome not in piv.columns:
        piv[outcome] = 0.0                               # never assigned -> 0 mass
    piv = piv.fillna(0.0)

    ind = (piv.columns.to_numpy() == outcome).astype(float)      # outcome indicator
    brier_per_date = ((piv.to_numpy() - ind[None, :]) ** 2).sum(axis=1)

    dates = list(piv.index)
    close = pd.Timestamp(close)
    if close < dates[0]:
        return np.nan
    bounds = dates + [close + pd.Timedelta(days=1)]              # inclusive of close
    lengths = np.array([(bounds[k + 1] - bounds[k]).days
                        for k in range(len(dates))], dtype=float)
    total = lengths.sum()
    if total <= 0:
        return np.nan
    return float((brier_per_date * lengths).sum() / total)


# ── panel construction ───────────────────────────────────────────────────────

def brier_panel(
    fcasts: pd.DataFrame,
    ifps: pd.DataFrame,
    condition: str = "individual",
    min_questions: int = 40,
    top_n: int | None = None,
    years=None,
):
    """
    Build the ``(question x forecaster)`` daily-averaged Brier panel.

    Parameters
    ----------
    condition     : "individual" keeps only non-team forecasters (team is NaN);
                    "all" keeps everyone (teams included).
    min_questions : keep forecasters answering >= this many closed questions.
    top_n         : if set, keep only the top_n most active of those.
    years         : optional int or list to restrict to survey year(s).

    Returns
    -------
    panel : DataFrame indexed by ifp_id (rows ordered by close date), columns =
            user_id, values = per-question Brier (NaN where not answered).
    info  : dict with users, questions, and per-user question counts.
    """
    fc = fcasts
    if years is not None:
        fc = fc[fc["year"].isin(np.atleast_1d(years))]
    if condition == "individual":
        fc = fc[fc["team"].isna()]
    elif condition != "all":
        raise ValueError("condition must be 'individual' or 'all'.")

    fc = fc[fc["ifp_id"].isin(ifps.index) & (fc["fcast_type"] != 4)]

    qcount = fc.groupby("user_id")["ifp_id"].nunique()
    users = qcount[qcount >= min_questions]
    if top_n is not None:
        users = users.nlargest(top_n)
    users = users.index
    fc = fc[fc["user_id"].isin(users)]

    rows = []
    for (uid, ifp_id), sub in fc.groupby(["user_id", "ifp_id"], sort=False):
        rec = ifps.loc[ifp_id]
        score = _brier_daily_avg(sub, rec["outcome"], rec["date_closed"])
        if not np.isnan(score):
            rows.append((ifp_id, uid, score))

    long = pd.DataFrame(rows, columns=["ifp_id", "user_id", "brier"])
    panel = long.pivot(index="ifp_id", columns="user_id", values="brier")

    order = ifps.loc[panel.index, "date_closed"].sort_values().index
    panel = panel.loc[order]

    info = {
        "n_questions": panel.shape[0],
        "n_forecasters": panel.shape[1],
        "questions_per_user": panel.notna().sum(axis=0),
        "users_per_question": panel.notna().sum(axis=1),
    }
    return panel, info
