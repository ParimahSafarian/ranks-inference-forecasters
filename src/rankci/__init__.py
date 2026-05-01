"""rankci — Inference for Rankings (Mogstad, Romano, Shaikh, Wilhelm 2024)."""

from .bootstrap import rank_confidence_intervals_bootstrap
from .simulation import rank_confidence_intervals_simulation
from .stepwise import (
    rank_ci_stepwise,
    rank_ci_stepwise_pairwise,
    rank_ci_marginal_pairwise,
)
from .pairwise import compute_pairwise, nw_se
from .data import (
    load_spf,
    load_rtdsm,
    compute_errors,
    compute_squared_error_panel,
    select_top_forecasters,
    winsorize_panel,
)

__all__ = [
    "rank_confidence_intervals_bootstrap",
    "rank_confidence_intervals_simulation",
    "rank_ci_stepwise",
    "rank_ci_stepwise_pairwise",
    "rank_ci_marginal_pairwise",
    "compute_pairwise",
    "nw_se",
    "load_spf",
    "load_rtdsm",
    "compute_errors",
    "compute_squared_error_panel",
    "select_top_forecasters",
    "winsorize_panel",
]
