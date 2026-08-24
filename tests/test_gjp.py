"""Tests for the GJP loader and Brier-score panel (Step 4, informative application)."""
import numpy as np
import pandas as pd

from rankci.data.gjp import _brier_daily_avg, brier_panel


def _sub(rows):
    """rows: list of (fcast_date, option, value, fcast_type)."""
    df = pd.DataFrame(rows, columns=["fcast_date", "answer_option", "value", "fcast_type"])
    df["fcast_date"] = pd.to_datetime(df["fcast_date"])
    df["timestamp"] = df["fcast_date"].astype(str)
    return df


# ── daily-averaged Brier ─────────────────────────────────────────────────────

def test_binary_single_forecast():
    # 0.75 on the outcome -> (0.75-1)^2 + (0.25-0)^2 = 0.125 (paper's worked example)
    sub = _sub([("2011-09-01", "a", 0.75, 0), ("2011-09-01", "b", 0.25, 0)])
    got = _brier_daily_avg(sub, outcome="a", close=pd.Timestamp("2011-09-05"))
    assert np.isclose(got, 0.125)


def test_carry_forward_daily_average():
    # day0 (a=.5,b=.5)->Brier .5 held days 0-1 (2d); day2 (a=.9,b=.1)->Brier .02 held days 2-3 (2d)
    sub = _sub([("2011-09-01", "a", 0.5, 0), ("2011-09-01", "b", 0.5, 0),
                ("2011-09-03", "a", 0.9, 1), ("2011-09-03", "b", 0.1, 1)])
    got = _brier_daily_avg(sub, outcome="a", close=pd.Timestamp("2011-09-04"))
    assert np.isclose(got, (0.5 * 2 + 0.02 * 2) / 4)   # = 0.26


def test_multinomial_brier():
    # a,b,c = .2,.5,.3, outcome b -> .2^2 + (.5-1)^2 + .3^2 = 0.38
    sub = _sub([("2011-09-01", "a", 0.2, 0), ("2011-09-01", "b", 0.5, 0),
                ("2011-09-01", "c", 0.3, 0)])
    got = _brier_daily_avg(sub, outcome="b", close=pd.Timestamp("2011-09-01"))
    assert np.isclose(got, 0.38)


def test_withdrawal_dropped():
    # a withdrawal row (fcast_type 4) is ignored; score = the real forecast's Brier
    sub = _sub([("2011-09-01", "a", 0.6, 0), ("2011-09-01", "b", 0.4, 0),
                ("2011-09-02", "a", 0.0, 4), ("2011-09-02", "b", 0.0, 4)])
    got = _brier_daily_avg(sub, outcome="a", close=pd.Timestamp("2011-09-01"))
    assert np.isclose(got, (0.6 - 1) ** 2 + 0.4 ** 2)   # 0.32


def test_outcome_option_never_assigned():
    # forecaster only ever lists a,b but outcome is 'c' -> c treated as 0 mass
    sub = _sub([("2011-09-01", "a", 0.5, 0), ("2011-09-01", "b", 0.5, 0)])
    got = _brier_daily_avg(sub, outcome="c", close=pd.Timestamp("2011-09-01"))
    assert np.isclose(got, 0.5 ** 2 + 0.5 ** 2 + (0 - 1) ** 2)   # 1.5


# ── panel construction ───────────────────────────────────────────────────────

def _mini_dataset():
    ifps = pd.DataFrame({
        "ifp_id": ["Q1", "Q2"],
        "q_status": ["closed", "closed"],
        "date_start": pd.to_datetime(["2011-09-01", "2011-09-01"]),
        "date_closed": pd.to_datetime(["2011-09-10", "2011-09-05"]),  # Q2 closes first
        "outcome": ["a", "b"],
        "n_opts": [2, 2],
    }).set_index("ifp_id")

    recs = []
    def add(ifp, uid, team, opt, val, date="2011-09-01"):
        recs.append(dict(ifp_id=ifp, user_id=uid, team=team, ctt="1a", fcast_type=0,
                         answer_option=opt, value=val, fcast_date=date,
                         timestamp=date, year=1))
    # user 1 (individual) answers both; user 2 (individual) answers Q1 only;
    # user 3 is on a team -> excluded by condition="individual"
    for opt, v in [("a", 0.8), ("b", 0.2)]: add("Q1", 1, np.nan, opt, v)
    for opt, v in [("a", 0.3), ("b", 0.7)]: add("Q2", 1, np.nan, opt, v)
    for opt, v in [("a", 0.5), ("b", 0.5)]: add("Q1", 2, np.nan, opt, v)
    for opt, v in [("a", 0.9), ("b", 0.1)]: add("Q1", 3, 7.0, opt, v)
    fc = pd.DataFrame(recs)
    fc["fcast_date"] = pd.to_datetime(fc["fcast_date"])
    return fc, ifps


def test_panel_shape_and_order_and_filter():
    fc, ifps = _mini_dataset()
    panel, info = brier_panel(fc, ifps, condition="individual", min_questions=1)
    # team user 3 excluded; users 1 and 2 kept
    assert set(panel.columns) == {1, 2}
    # questions ordered by close date: Q2 (09-05) before Q1 (09-10)
    assert list(panel.index) == ["Q2", "Q1"]
    # user 2 answered only Q1 -> NaN on Q2 (imbalance)
    assert np.isnan(panel.loc["Q2", 2])
    # user 1 on Q1: outcome a, forecast a=0.8 -> (0.8-1)^2+(0.2)^2 = 0.08
    assert np.isclose(panel.loc["Q1", 1], 0.08)


def test_min_questions_filter():
    fc, ifps = _mini_dataset()
    panel, _ = brier_panel(fc, ifps, condition="individual", min_questions=2)
    assert list(panel.columns) == [1]           # only user 1 answered >= 2 questions


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
