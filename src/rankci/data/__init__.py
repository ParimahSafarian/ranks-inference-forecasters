"""Source-specific data loaders.

Each submodule produces a wide DataFrame indexed by target period with one
column per forecaster ID, holding the chosen loss metric. The rank-CI engines
in ``rankci.core`` consume that panel without knowing the source.

Submodules
----------
- :mod:`rankci.data.philly` : Philadelphia Fed SPF + Real-Time Data Set.
- :mod:`rankci.data.ecb`    : ECB SPF individual point + density forecasts.
- :mod:`rankci.data.panel`  : Shared helpers (select_top_forecasters,
                              winsorize_panel).
"""
from .panel import select_top_forecasters, winsorize_panel
from . import philly, ecb  # noqa: F401

__all__ = ["select_top_forecasters", "winsorize_panel", "philly", "ecb"]
