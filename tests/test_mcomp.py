"""Tests for the M-competition method panel (Step 4, sharp application)."""
import numpy as np
import pandas as pd

from rankci.data.mcomp import (
    smape, mase, m_naive, m_snaive, m_naive2, m_ets, m_theta,
    build_panel, HORIZON, SEASON,
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
