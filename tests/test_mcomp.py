"""Tests for the M-competition panels: our own method panel and the official M3
submissions (Paris temperature, the thesis application)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rankci.data.mcomp import (
    smape, mase, m_naive, m_snaive, m_naive2, m_ets, m_theta,
    build_panel, HORIZON, SEASON,
    load_m3_official, official_panel_from_long,
)


# ── scores ───────────────────────────────────────────────────────────────────

def test_smape_perfect_and_known():
    assert smape([1, 2, 3], [1, 2, 3]) == 0.0
    # 200*|4-6|/(4+6) = 40 for a single point
    assert np.isclose(smape([4.0], [6.0]), 40.0)


def test_mase_scaling():
    train = np.array([1.0, 2, 3, 4, 5])          # seasonal-naive(m=1) diffs all = 1
    # forecast off by 2 everywhere -> MASE = 2 / 1 = 2
    assert np.isclose(mase([10, 11], [12, 13], train, m=1), 2.0)


# ── benchmark methods ────────────────────────────────────────────────────────

def test_naive_and_snaive():
    tr = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])    # m=4
    assert list(m_naive(tr, 3, 4)) == [8, 8, 8]
    # seasonal-naive repeats the last season [5,6,7,8]
    assert list(m_snaive(tr, 6, 4)) == [5, 6, 7, 8, 5, 6]


def test_naive2_falls_back_when_nonseasonal():
    tr = np.arange(1.0, 30.0)                      # trend, not seasonal at m=4
    # not seasonal -> naive2 == naive1
    assert np.allclose(m_naive2(tr, 4, 4), m_naive(tr, 4, 4))


def test_naive2_seasonal_adjustment():
    # clear multiplicative seasonality: level 10 with pattern [1,2,1,0.5-ish]... use additive-ish
    m = 4
    base = np.tile([10, 20, 30, 40], 10).astype(float)   # strong seasonal, no trend
    fc = m_naive2(base, m, m)
    # naive2 should roughly reproduce the seasonal pattern, not a flat line
    assert fc.std() > 5


# ── statsmodels methods run and return finite forecasts ──────────────────────

def test_ets_and_theta_smoke():
    rng = np.random.default_rng(0)
    t = np.arange(60)
    y = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.standard_normal(60)
    for fn in (m_ets, m_theta):
        f = fn(y, 18, 12)
        assert len(f) == 18 and np.isfinite(f).all()


# ── panel construction ───────────────────────────────────────────────────────

def _synth_df(n_series=4, T=60, group="Monthly", seed=0):
    rng = np.random.default_rng(seed)
    m = SEASON[group]
    frames = []
    for i in range(n_series):
        t = np.arange(T)
        y = 100 + 0.3 * t + (8 * np.sin(2 * np.pi * t / max(m, 1)) if m > 1 else 0) \
            + rng.standard_normal(T) * (1 + i)
        frames.append(pd.DataFrame({"unique_id": f"S{i}", "ds": t, "y": y}))
    return pd.concat(frames, ignore_index=True)


def test_build_panel_shape_and_comb():
    df = _synth_df(n_series=5, group="Monthly")
    panel = build_panel(df, "Monthly", methods=["naive2", "ets", "theta"],
                        include_comb=True)
    assert panel.shape[0] == 5
    assert set(panel.columns) == {"naive2", "ets", "theta", "comb"}
    assert (panel.values >= 0).all() and np.isfinite(panel.values).all()


def test_build_panel_no_comb():
    df = _synth_df(n_series=3, group="Quarterly")
    panel = build_panel(df, "Quarterly", methods=["naive", "snaive"], include_comb=False)
    assert list(panel.columns) == ["naive", "snaive"]


# ── official M3 submissions (thesis application: Paris temperature) ──────────

FIXTURE = Path(__file__).parent / "fixtures" / "m3_paris_official.csv"
PARIS = "Average temperature in Paris"


def test_official_panel_paris_shape_and_order():
    long = pd.read_csv(FIXTURE)
    P = official_panel_from_long(long, PARIS)
    assert P.shape == (306, 24)                    # 17 blocks x 18 months, 24 methods
    series = P.index.get_level_values("series")
    assert series[0] == "N2784" and series[-1] == "N2800"   # calendar order kept
    assert list(P.index.get_level_values("horizon")[:18]) == list(range(1, 19))
    assert (P.values >= 0).all() and np.isfinite(P.values).all()


def test_official_panel_paris_known_smapes():
    # mean sMAPEs quoted in the thesis (Table tab:mcomp)
    theta = official_panel_from_long(pd.read_csv(FIXTURE), PARIS).mean()
    assert np.isclose(theta["B-J auto"], 10.10, atol=0.005)
    assert np.isclose(theta["THETA"], 10.50, atol=0.005)
    assert np.isclose(theta["Flors-Pearc1"], 37.80, atol=0.005)


def test_official_panel_drops_incomplete_methods():
    long = pd.read_csv(FIXTURE)
    long.loc[long.index[0], "AAM1"] = np.nan
    assert "AAM1" not in official_panel_from_long(long, PARIS).columns


def test_official_panel_unknown_description():
    with pytest.raises(ValueError):
        official_panel_from_long(pd.read_csv(FIXTURE), "no such series")


@pytest.mark.skipif(not os.environ.get("RANKCI_NETWORK"),
                    reason="downloads Mcomp from CRAN; set RANKCI_NETWORK=1")
def test_load_m3_official_from_cran(tmp_path):
    df = load_m3_official(directory=str(tmp_path))
    assert df.shape == (37014, 30) and df["series"].nunique() == 3003
    fixture = pd.read_csv(FIXTURE)
    paris = df[df["description"] == PARIS].reset_index(drop=True)
    pd.testing.assert_frame_equal(paris, fixture, check_dtype=False)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
