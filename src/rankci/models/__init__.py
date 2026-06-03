"""Baseline forecasting models.

Source-specific because each model needs the appropriate vintage discipline.
``philly`` uses Philly Fed RTDSM vintages.
"""
from .philly import (
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
    "forecast_naive",
    "forecast_ar1",
    "forecast_ar",
    "forecast_rw_drift",
    "forecast_ma4",
    "forecast_historical_mean",
    "model_forecast_series",
    "model_error_panel",
]
