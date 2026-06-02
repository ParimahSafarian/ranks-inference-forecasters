"""rankci — Inference for Rankings (Mogstad, Romano, Shaikh, Wilhelm 2024)."""

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
from .data import (
    load_spf,
    load_rtdsm,
    compute_errors,
    compute_error_panel,
    select_top_forecasters,
    winsorize_panel,
)
from .models import (
    forecast_naive,
    forecast_ar1,
    forecast_ar,
    forecast_rw_drift,
    forecast_ma4,
    forecast_historical_mean,
    model_forecast_series,
    model_error_panel,
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
    "load_spf",
    "load_rtdsm",
    "compute_errors",
    "compute_error_panel",
    "select_top_forecasters",
    "winsorize_panel",
    "forecast_naive",
    "forecast_ar1",
    "forecast_ar",
    "forecast_rw_drift",
    "forecast_ma4",
    "forecast_historical_mean",
    "model_forecast_series",
    "model_error_panel",
]


from .tau_best import (
    tau_best_from_rank_ci,
    tau_best_pairwise,
    tau_best_simulation_pairwise,
)