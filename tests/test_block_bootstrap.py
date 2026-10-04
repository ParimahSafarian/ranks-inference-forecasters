"""Joint block bootstrap: index construction, block-length rule, engines."""
import numpy as np
import pytest

from rankci import (
    compute_pairwise,
    default_block_length,
    mbb_indices,
    rank_ci_marginal_pairwise,
    rank_ci_stepwise,
    rank_ci_stepwise_pairwise,
    rank_confidence_intervals_bootstrap,
    tau_best_pairwise,
)
from rankci.core.bandwidth import andrews_bandwidth
from rankci.core.block_bootstrap import block_bootstrap_draws, pairwise_means
from rankci.core.covariance import stepdown_rank_ci
from rankci.core.pairwise import nw_se
from rankci.sim import simulate_panel


def test_mbb_indices_are_contiguous_blocks():
    rng = np.random.default_rng(0)
    n, l = 23, 5
    for circular in (True, False):
        idx = mbb_indices(n, l, rng, circular=circular)
        assert idx.shape == (n,)
        assert idx.min() >= 0 and idx.max() < n
        # every full block is a run of consecutive rows (wrapping around if circular)
        for start in range(0, n - l + 1, l):
            steps = np.diff(idx[start:start + l])
            assert np.all(steps % n == 1) if circular else np.all(steps == 1)


def test_circular_blocks_draw_every_row_equally_often():
    rng = np.random.default_rng(4)
    n, l, B = 23, 5, 4000
    counts = {c: np.zeros(n) for c in (True, False)}
    for c in counts:
        for _ in range(B):
            counts[c] += np.bincount(mbb_indices(n, l, rng, circular=c), minlength=n)
        counts[c] /= B                        # mean number of copies of each row
    assert np.all(np.abs(counts[True] - 1.0) < 0.1)
    # the non-circular version under-samples the first and last rows
    assert counts[False][0] < 0.5 and counts[False][-1] < 0.5


def test_block_bootstrap_variance_matches_newey_west():
    # With l = L + 1 the block-bootstrap variance of a mean is the Bartlett
    # (Newey-West) long-run variance with bandwidth L, divided by n.
    rng = np.random.default_rng(0)
    n, rho = 400, 0.5
    e = rng.normal(size=n + 100)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, x.size):
        x[t] = rho * x[t - 1] + e[t]
    x = x[100:]
    L = andrews_bandwidth(x)
    means = np.array([x[mbb_indices(n, L + 1, rng)].mean() for _ in range(4000)])
    assert means.var() / nw_se(x, L=L)[1] ** 2 == pytest.approx(1.0, abs=0.12)
    assert abs(means.mean() - x.mean()) < 4 * means.std() / np.sqrt(means.size)


def test_stepwise_bootstrap_draws_once():
    X = simulate_panel(theta=np.linspace(0, 1.2, 5), T=150, rho=0.5,
                       imbalance=0.3, seed=9)
    p = X.shape[1]
    out = rank_ci_stepwise_pairwise(X, alpha=0.1, B=400, seed=3, verbose=False)
    # one set of draws, restricted each round: critical values never increase
    assert np.all(np.diff(out["critical_values"]) <= 0)
    # and the engine equals the shared stepdown applied to one array of draws
    delta, se, _ = compute_pairwise(X, se_method="nw")
    pairs = [(j, k) for j in range(p) for k in range(j + 1, p)]
    P = np.array(pairs)
    T = block_bootstrap_draws(X, P, delta[P[:, 0], P[:, 1]], se[P[:, 0], P[:, 1]],
                              400, out["block_length"], np.random.default_rng(3))
    manual = stepdown_rank_ci({"pairs": pairs, "delta_q": delta[P[:, 0], P[:, 1]],
                               "se_q": se[P[:, 0], P[:, 1]], "T_draws": T}, p, alpha=0.1)
    assert np.array_equal(out["rank_ci"], manual["rank_ci"])


def test_mbb_block_length_one_is_iid_rows():
    rng = np.random.default_rng(1)
    idx = mbb_indices(50, 1, rng)
    assert idx.shape == (50,)
    assert len(np.unique(idx)) < 50          # with replacement


def test_mbb_indices_rejects_bad_block_length():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        mbb_indices(10, 0, rng)
    with pytest.raises(ValueError):
        mbb_indices(10, 11, rng)


def test_default_block_length_uses_L_plus_one():
    X = np.random.default_rng(0).normal(size=(40, 3))
    assert default_block_length(X, L=4) == 5
    assert default_block_length(X, L=100) == 40   # capped at n
    assert 1 <= default_block_length(X) <= 40


def test_default_block_length_grows_with_persistence():
    white = simulate_panel(theta=np.zeros(4), T=400, rho=0.0, rho_c=0.0, seed=0)
    persistent = simulate_panel(theta=np.zeros(4), T=400, rho=0.9, rho_c=0.9, seed=0)
    assert default_block_length(persistent) > default_block_length(white)


def test_pairwise_means_handles_missing_overlap():
    Xb = np.array([[1.0, 2.0, np.nan],
                   [3.0, 5.0, np.nan],
                   [np.nan, 1.0, 4.0]])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    m = pairwise_means(Xb, pairs)
    assert m[0] == pytest.approx(-1.5)
    assert np.isnan(m[1])                     # columns 0 and 2 never co-observed
    assert m[2] == pytest.approx(-3.0)


def test_engines_run_on_unbalanced_panel_and_report_block_length():
    X = simulate_panel(theta=np.linspace(0, 1, 5), T=120, rho=0.5,
                       imbalance=0.3, seed=3)
    out = rank_ci_stepwise_pairwise(X, alpha=0.1, B=200, seed=1, verbose=False)
    assert out["rank_ci"].shape == (5, 2)
    assert out["block_length"] >= 1
    assert np.all(out["rank_ci"][:, 0] <= out["rank_ci"][:, 1])

    out_m = rank_ci_marginal_pairwise(X, alpha=0.1, B=200, seed=1)
    assert out_m["rank_ci"].shape == (5, 2)
    assert out_m["block_length"] == out["block_length"]

    out_t = tau_best_pairwise(X, tau=2, alpha=0.1, B=200, seed=1, verbose=False)
    assert out_t["tau_best_set"].dtype == bool
    assert out_t["block_length"] == out["block_length"]


def test_complete_case_engines_accept_block_length():
    X = simulate_panel(theta=np.linspace(0, 1, 4), T=100, rho=0.4, seed=5)
    out = rank_ci_stepwise(X, alpha=0.1, B=200, seed=2, block_length=4)
    assert out["block_length"] == 4
    out2 = rank_confidence_intervals_bootstrap(X, alpha=0.1, B=200, seed=2,
                                               block_length=4)
    assert out2["block_length"] == 4
    assert out2["rank_ci"].shape == (4, 2)


def test_seed_reproducible():
    X = simulate_panel(theta=np.linspace(0, 1, 4), T=80, rho=0.3, seed=7)
    a = rank_ci_stepwise_pairwise(X, B=300, seed=11, verbose=False)
    b = rank_ci_stepwise_pairwise(X, B=300, seed=11, verbose=False)
    assert np.array_equal(a["rank_ci"], b["rank_ci"])
