"""Joint moving block bootstrap: index construction, block-length rule, engines."""
import numpy as np
import pytest

from rankci import (
    default_block_length,
    mbb_indices,
    rank_ci_marginal_pairwise,
    rank_ci_stepwise,
    rank_ci_stepwise_pairwise,
    rank_confidence_intervals_bootstrap,
    tau_best_pairwise,
)
from rankci.core.block_bootstrap import pairwise_means
from rankci.sim import simulate_panel


def test_mbb_indices_are_contiguous_blocks():
    rng = np.random.default_rng(0)
    n, l = 23, 5
    idx = mbb_indices(n, l, rng)
    assert idx.shape == (n,)
    assert idx.min() >= 0 and idx.max() < n
    # every full block is a run of consecutive integers
    for start in range(0, n - l + 1, l):
        blk = idx[start:start + l]
        assert np.all(np.diff(blk) == 1)


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
