"""Tests for the Monte Carlo DGP and coverage driver (Step 3)."""
import numpy as np

from rankci.sim import (
    simulate_panel, true_ranks, coverage_study, _ar1_panel, _apply_stagger,
)


# ── DGP ──────────────────────────────────────────────────────────────────────

def test_shape_and_means():
    theta = np.array([0.0, 1.0, 2.0, 3.0])
    X = simulate_panel(theta, T=20000, rho=0.3, sigma_u=1.0, sigma_c=2.0, seed=0)
    assert X.shape == (20000, 4)
    # column means recover theta (common shock has mean ~0 over long T)
    assert np.allclose(np.nanmean(X, axis=0), theta, atol=0.1)


def test_common_shock_cancels_in_differences():
    """The additive common shock must vanish from every difference series, so a
    huge sigma_c inflates level variances but NOT difference variances."""
    theta = np.zeros(3)
    small_c = simulate_panel(theta, T=5000, rho=0.0, sigma_u=1.0, sigma_c=0.0, seed=1)
    big_c   = simulate_panel(theta, T=5000, rho=0.0, sigma_u=1.0, sigma_c=10.0, seed=1)
    # level variance explodes with the common shock ...
    assert np.var(big_c[:, 0]) > 20 * np.var(small_c[:, 0])
    # ... but the difference variance is essentially unchanged.
    d_small = small_c[:, 0] - small_c[:, 1]
    d_big   = big_c[:, 0] - big_c[:, 1]
    assert np.isclose(np.var(d_small), np.var(d_big), rtol=1e-6)


def test_ar1_stationary_variance():
    rng = np.random.default_rng(2)
    y = _ar1_panel(200000, 1, rho=0.6, sigma=2.0, rng=rng)[:, 0]
    assert np.isclose(np.var(y), 4.0, rtol=0.05)                 # stationary var = sigma^2
    assert np.isclose(np.corrcoef(y[1:], y[:-1])[0, 1], 0.6, atol=0.05)


def test_stagger_imbalance():
    X = simulate_panel(np.zeros(6), T=100, imbalance=0.4, seed=3)
    miss = np.isnan(X).mean(axis=0)
    assert np.all(miss > 0)                                      # every col loses obs
    assert np.all(np.isfinite(X).sum(axis=0) >= 10)             # min_active respected
    # balanced -> no missing
    Xb = simulate_panel(np.zeros(6), T=100, imbalance=0.0, seed=3)
    assert np.isfinite(Xb).all()


def test_true_ranks():
    assert list(true_ranks([3.0, 1.0, 2.0])) == [3, 1, 2]        # rank 1 = smallest


def test_volatility_episodes_raise_local_variance():
    """Volatility episodes inflate variance in a few windows but not globally,
    and (being idiosyncratic) they enlarge difference variance too."""
    theta = np.zeros(3)
    calm  = simulate_panel(theta, T=2000, rho=0.0, n_vol_episodes=0, seed=4)
    crisis = simulate_panel(theta, T=2000, rho=0.0, n_vol_episodes=6,
                            vol_scale=10.0, vol_len=20, seed=4)
    # difference variance is larger with episodes (idiosyncratic vol, not common)
    assert np.var(crisis[:, 0] - crisis[:, 1]) > 2 * np.var(calm[:, 0] - calm[:, 1])


# ── Driver ───────────────────────────────────────────────────────────────────

def test_coverage_study_shapes_and_columns():
    designs = [
        {"name": "grad", "theta": [0.0, 1.0, 2.0], "T": 120, "imbalance": 0.0},
        {"name": "ties", "theta": [0.0, 0.0, 1.0], "T": 120, "imbalance": 0.0},
    ]
    df = coverage_study(designs, R=15, B=400, alpha=0.1, base_seed=0,
                        methods=("mds", "omega"), include_bootstrap=True)
    assert len(df) == 2 * 3                                      # 2 designs x 3 methods
    for col in ("joint_coverage", "mean_width", "false_sep_rate", "proj_gap_mean"):
        assert col in df.columns
    # distinct-theta design has coverage but NaN false-sep; tie design the reverse
    grad = df[df.name == "grad"]
    ties = df[df.name == "ties"]
    assert grad["joint_coverage"].notna().all()
    assert grad["false_sep_rate"].isna().all()
    assert ties["false_sep_rate"].notna().all()
    assert ties["joint_coverage"].isna().all()
    # bootstrap has no covariance projection; omega does
    assert np.isnan(df[df.method == "bootstrap"]["proj_gap_mean"]).all()
    assert df[df.method == "omega"]["proj_gap_mean"].notna().all()


def test_coverage_near_nominal_easy_balanced():
    """On an easy balanced design the simultaneous procedure must not under-cover
    (>= nominal, up to Monte Carlo error) for both covariance routes."""
    designs = [{"name": "easy", "theta": [0.0, 1.5, 3.0, 4.5], "T": 150,
                "rho": 0.2, "sigma_c": 1.0, "imbalance": 0.0}]
    df = coverage_study(designs, R=120, B=800, alpha=0.1, base_seed=7,
                        methods=("mds", "omega"))
    for m in ("mds", "omega"):
        cov = df[df.method == m]["joint_coverage"].iloc[0]
        assert cov >= 0.85, (m, cov)                            # nominal 0.90, MC slack


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
