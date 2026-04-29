# Rank Confidence Intervals for Forecaster Evaluation

Master's thesis project applying the rank confidence interval methodology of
Mogstad, Romano, Shaikh & Wilhelm (2024) to the Survey of Professional
Forecasters (SPF).

The goal is to rank professional forecasters by mean squared error and
construct simultaneous confidence sets for those ranks, instead of relying
on point estimates alone.

## What's here

- `src/rankci/` — installable Python package with three CI methods
  (simultaneous bootstrap, Gaussian simulation, stepwise bootstrap),
  pairwise NW-HAC standard errors with optional winsorization, and helpers
  for loading SPF microdata and Philly Fed RTDSM vintage matrices.
- `notebooks/`
  - `01_toydataset.ipynb` — synthetic forecasters with known sigmas, used
    to check that the procedure recovers the true ranking.
  - `NGDP_CI.ipynb` — main analysis on nominal GDP forecasts, realizations
    from `NOUTPUTQvQd.xlsx` (advance estimates).
  - `UNEMP_CI.ipynb` — same pipeline applied to unemployment-rate
    forecasts, realizations from `rucQvMd.xlsx`.
  - `EDA.ipynb` — exploratory look at panel structure and participation.
- `data/` — SPF microdata and the RTDSM vintage matrices used as
  realizations.

## Setup

```
pip install -e .
```

Then open any notebook in `notebooks/`.

## Reference

Mogstad, M., Romano, J. P., Shaikh, A., & Wilhelm, D. (2024).
*Inference for Ranks with Applications to Mobility Across Neighbourhoods
and Academic Achievement Across Countries.* Review of Economic Studies,
91(1), 476–518.
