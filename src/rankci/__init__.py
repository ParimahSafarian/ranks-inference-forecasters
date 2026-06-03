"""rankci — Inference for rankings, applied to professional forecasters.

Package layout
--------------
- ``rankci.core``   : source-agnostic inference engines (pairwise SE, bootstrap,
                      simulation, stepwise, tau-best). Operate on a plain
                      ``(n, p)`` numpy panel.
- ``rankci.data``   : data loaders. One submodule per source:
                          * ``rankci.data.philly`` — Philadelphia Fed SPF + RTDSM
                          * ``rankci.data.ecb``    — ECB SPF individual forecasts
                          * ``rankci.data.panel``  — shared helpers
- ``rankci.models`` : baseline forecasting models (source-specific because each
                      needs the right vintage discipline).

Importing names directly from ``rankci`` re-exports the core engines and the
shared panel helpers; source-specific functions must be imported from their
respective submodules (``from rankci.data.philly import load_spf`` etc.). This
keeps source-of-truth obvious in the notebooks.
"""
# Re-export source-agnostic engines and shared panel helpers
from .core import (  # noqa: F401
    rank_confidence_intervals_bootstrap,
    rank_confidence_intervals_simulation,
    rank_confidence_intervals_simulation_pairwise,
    rank_ci_stepwise,
    rank_ci_stepwise_pairwise,
    rank_ci_marginal_pairwise,
    rank_ci_stepwise_simulation,
    rank_ci_stepwise_simulation_pairwise,
    rank_ci_marginal_simulation_pairwise,
    compute_pairwise,
    nw_se,
    cov_theta_pairwise,
    tau_best_from_rank_ci,
    tau_best_pairwise,
    tau_best_simulation_pairwise,
)
from .data import select_top_forecasters, winsorize_panel  # noqa: F401

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
    "select_top_forecasters",
    "winsorize_panel",
]
