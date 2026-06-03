"""Source-agnostic rank-inference engines.

These modules operate on a plain (n, p) numpy panel and do not know where the
data came from.
"""
from .bootstrap import rank_confidence_intervals_bootstrap
from .simulation import (
    rank_confidence_intervals_simulation,
    rank_confidence_intervals_simulation_pairwise,
)
from .stepwise import (
    rank_ci_stepwise,
    rank_ci_stepwise_pairwise,
    rank_ci_marginal_pairwise,
)
from .stepwise_simulation import (
    rank_ci_stepwise_simulation,
    rank_ci_stepwise_simulation_pairwise,
    rank_ci_marginal_simulation_pairwise,
)
from .pairwise import compute_pairwise, nw_se, cov_theta_pairwise
from .tau_best import (
    tau_best_from_rank_ci,
    tau_best_pairwise,
    tau_best_simulation_pairwise,
)

__all__ = [
    "rank_confidence_intervals_bootstrap",
    "rank_confidence_intervals_simulation",
    "rank_confidence_intervals_simulation_pairwise",
    "rank_ci_stepwise",
    "rank_ci_stepwise_pairwise",
    "rank_ci_marginal_pairwise",
    "rank_ci_stepwise_simulation",
    "rank_ci_stepwise_simulation_pairwise",
    "rank_ci_marginal_simulation_pairwise",
    "compute_pairwise",
    "nw_se",
    "cov_theta_pairwise",
    "tau_best_from_rank_ci",
    "tau_best_pairwise",
    "tau_best_simulation_pairwise",
]
