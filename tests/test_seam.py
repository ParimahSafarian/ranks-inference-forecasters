"""Tests for the covariance seam + shared stepdown (Step 2, rows T6-T7)."""
import numpy as np

from rankci.core.covariance import studentized_null_draws, stepdown_rank_ci
from rankci.core.omega import pair_index
from rankci import rank_ci_stepwise_simulation_pairwise


def balanced_iid(n=200, p=5, seed=0):
    """iid losses, no skill spread -> both methods should give trivial [1,p]."""
    rng = np.random.default_rng(seed)
    return 1.0 + rng.standard_normal((n, p)) ** 2


def separated_panel(n=400, p=4, seed=0):
    """Clear mean spread so the procedure actually separates some ranks."""
    rng = np.random.default_rng(seed)
    means = np.array([0.0, 1.0, 2.0, 3.0])[:p]
    return means + rng.standard_normal((n, p)) * 0.5


# ── T6: seam shape / validity ────────────────────────────────────────────────

def test_draw_shapes_both_methods():
    X = balanced_iid()
    p = X.shape[1]
    q = len(pair_index(p))
    for method in ("mds", "omega"):
        d = studentized_null_draws(X, method=method, B=1234, seed=1)
        assert d["T_draws"].shape == (1234, q)
        assert d["delta_q"].shape == (q,)
        assert d["se_q"].shape == (q,)
        assert np.isfinite(d["T_draws"]).all()


def test_trivial_interval_on_iid():
    X = balanced_iid()
    p = X.shape[1]
    for method in ("mds", "omega"):
        out = rank_ci_stepwise_simulation_pairwise(
            X, alpha=0.05, B=3000, seed=7, covariance=method, verbose=False,
        )
        assert out["rank_ci"].tolist() == [[1, p]] * p, (method, out["rank_ci"])


def test_methods_agree_on_balanced_separated():
    """On a balanced, well-separated panel the two covariance routes should
    give the same rank CIs (they estimate the same object)."""
    X = separated_panel()
    p = X.shape[1]
    outs = {}
    for method in ("mds", "omega"):
        outs[method] = rank_ci_stepwise_simulation_pairwise(
            X, alpha=0.10, B=8000, seed=11, covariance=method, verbose=False,
        )["rank_ci"]
    assert np.array_equal(outs["mds"], outs["omega"]), outs


# ── T7: stepdown monotonicity ────────────────────────────────────────────────

def test_rank_ci_well_formed():
    X = separated_panel()
    p = X.shape[1]
    out = rank_ci_stepwise_simulation_pairwise(
        X, alpha=0.10, B=4000, seed=3, covariance="omega", verbose=False,
    )
    ci = out["rank_ci"]
    assert (ci[:, 0] >= 1).all() and (ci[:, 1] <= p).all()
    assert (ci[:, 0] <= ci[:, 1]).all()


def test_lower_alpha_wider_intervals():
    """Smaller alpha (higher coverage) -> weakly wider intervals."""
    X = separated_panel()
    wide = rank_ci_stepwise_simulation_pairwise(
        X, alpha=0.01, B=6000, seed=5, covariance="omega", verbose=False,
    )["rank_ci"]
    tight = rank_ci_stepwise_simulation_pairwise(
        X, alpha=0.20, B=6000, seed=5, covariance="omega", verbose=False,
    )["rank_ci"]
    w_wide = (wide[:, 1] - wide[:, 0]).sum()
    w_tight = (tight[:, 1] - tight[:, 0]).sum()
    assert w_wide >= w_tight


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
