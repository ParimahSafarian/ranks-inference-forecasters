"""Tests for the naive tau-best projection from joint rank sets (thesis sec:taubest)."""
import numpy as np

from rankci import tau_best_from_rank_ci

# Joint rank sets [N+ + 1, p - N-] when unit 0 is confirmed better than everyone,
# unit 4 worse than everyone, and units 1-3 cannot be separated from each other.
# They cover the true ranks 1..5.
RANK_CI = np.array([[1, 1], [2, 4], [2, 4], [2, 4], [5, 5]])
TRUE_RANK = np.arange(1, 6)


def test_pinned_best_in_tau2_set():
    """A unit with rank set [1,1] is certainly top-2, so the tau=2 set keeps it."""
    s = tau_best_from_rank_ci(RANK_CI, 2)
    assert s[0]
    assert s.tolist() == [True, True, True, True, False]


def test_tau1_unchanged():
    """For tau=1 the projection {L_j <= 1} equals the old rule {1 in [L_j, U_j]}."""
    p = 5
    rank_ci = np.array([[lo, hi] for lo in range(1, p + 1) for hi in range(lo, p + 1)])
    old = (rank_ci[:, 0] <= 1) & (1 <= rank_ci[:, 1])
    assert np.array_equal(tau_best_from_rank_ci(rank_ci, 1), old)
    assert tau_best_from_rank_ci(RANK_CI, 1).tolist() == [True, False, False, False, False]


def test_contains_true_top_tau():
    """When every rank set covers the true rank, each tau-best set contains all
    units with true rank <= tau."""
    for tau in range(1, 6):
        s = tau_best_from_rank_ci(RANK_CI, tau)
        assert s[TRUE_RANK <= tau].all(), tau
