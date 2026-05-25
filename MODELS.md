# Forecasting models

Reference for the time-series models in [src/rankci/models.py](src/rankci/models.py) and the ranking results in [notebooks/MODEL_COMPARISON.ipynb](notebooks/MODEL_COMPARISON.ipynb).

The models exist for one purpose: provide simple, transparent baselines that an SPF panelist could (in principle) beat. Each is wrapped so that the same `model_error_panel` driver evaluates it under exactly the vintage discipline an SPF respondent faced.

## Real-time evaluation contract

Every model takes the signature `forecast(history, h) -> float`, where:

- `history` is the 1-D real-time series available **at survey quarter S**, drawn from the RTDSM vintage column published in `S` and truncated to observations through `S − 1`. So if a respondent submitted on, say, February 15, 1996 (= 1996:Q1 survey), the model sees the same `NOUTPUT96Q1` column they would have read off the BEA release that month — no future revisions, no later vintages.
- `h` is the number of steps from the last available observation to the target. For SPF horizon `H` (where `H=3` is one-quarter-ahead by Philly Fed convention), `h = HORIZON_OFFSETS[H] + 1`.

The driver function [model_forecast_series](src/rankci/models.py) handles the vintage lookup and history truncation; you only write the pure forecasting logic.

The realized side for the loss `(forecast − actual)²` uses the **advance estimate** of the target quarter (the first BEA print after `T`), via [get_advance_estimate](src/rankci/data.py). This matches the convention used for SPF errors throughout the package.

## The models

Notation: history is `y_1, y_2, ..., y_T`, the forecast horizon is `h` steps after `T`, and `Δy_t = y_t − y_{t−1}`.

### `forecast_naive` — random walk

$$
\hat y_{T+h} = y_T
$$

The simplest possible benchmark. Implements the null hypothesis "tomorrow looks like today." Ignores `h`. Cost: zero parameters, zero fitting.

### `forecast_rw_drift` — random walk with drift

$$
\hat y_{T+h} = y_T + h \cdot \bar{\Delta y},
\qquad
\bar{\Delta y} = \frac{1}{T-1}\sum_{t=2}^{T}\Delta y_t = \frac{y_T - y_1}{T - 1}
$$

A naive baseline that also extracts the unconditional average change per step. For a trending series like nominal GDP, this is meaningfully better than `naive` because it stops predicting zero growth.

### `forecast_ma4` — last 4-quarter average

$$
\hat y_{T+h} = \frac{1}{4}\sum_{i=0}^{3} y_{T-i}
$$

Mean-reverts toward the *recent* level. Sensible for a stationary or near-stationary series (e.g. unemployment rate); pathological for an exponentially-trending one (it always under-predicts the next observation).

### `forecast_historical_mean` — cumulative mean

$$
\hat y_{T+h} = \frac{1}{T}\sum_{t=1}^{T} y_t
$$

Predicts the level using the entire historical mean. The most aggressive form of mean reversion. For a trending series this is *very* bad — the mean of an exponentially growing path lags the current level by orders of magnitude. Included as a worst-case baseline.

### `forecast_ar1` — AR(1) with optional transform

Three variants, selected by `transform`:

**`transform="levels"`** — fit OLS on the raw series:

$$
y_t = c + \phi\, y_{t-1} + \varepsilon_t
$$

then iterate $\hat y_{T+h} = c + \phi\, \hat y_{T+h-1}$ for `h` steps from $\hat y_T = y_T$.

**`transform="diff"`** — fit AR(1) on first differences $d_t = \Delta y_t$:

$$
d_t = c + \phi\, d_{t-1} + \varepsilon_t,
\qquad
\hat y_{T+h} = y_T + \sum_{i=1}^{h} \hat d_{T+i}.
$$

**`transform="log_diff"`** — fit AR(1) on log-growth $g_t = \log y_t - \log y_{t-1}$:

$$
g_t = c + \phi\, g_{t-1} + \varepsilon_t,
\qquad
\hat y_{T+h} = \exp\!\left(\log y_T + \sum_{i=1}^{h} \hat g_{T+i}\right).
$$

For NGDP (exponentially trending and bounded below by zero), `log_diff` is the natural choice. For unemployment-rate-style data, `levels` makes sense. `diff` is the linear analog of `log_diff`.

### `forecast_ar` — generalized AR(p)

$$
y_t = c + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \varepsilon_t
$$

with the same three `transform` options as `forecast_ar1`. Coefficients are fit by OLS via `_fit_ar(y, p)`; the iterated forecast carries a length-`p` rolling state.

For `lags=1` this reduces exactly to `forecast_ar1`. Used in the model comparison as `ar2_log_diff` with `lags=2, transform="log_diff"`.

### Fitting and numerical guards

All AR fits use `numpy.linalg.lstsq` on the stacked design matrix `[1, y_{t-1}, ..., y_{t-p}]`. Each model returns `NaN` if:

- the history has fewer observations than the model requires (`< lags + 2` for AR(p), `< 2` for `rw_drift`, `< 4` for `ma4`),
- the `log_diff` transform sees a non-positive value,
- the model raises any other exception during `model_forecast_series`.

`NaN` rows are dropped after the panel is built, so models with longer warm-ups simply contribute fewer target quarters.

## How the models are wired into the rank-CI pipeline

The notebook builds the panel via:

```python
models = {
    "naive":            forecast_naive,
    "rw_drift":         forecast_rw_drift,
    "ma4":              forecast_ma4,
    "historical_mean":  forecast_historical_mean,
    "ar1_levels":       partial(forecast_ar1, transform="levels"),
    "ar1_log_diff":     partial(forecast_ar1, transform="log_diff"),
    "ar2_log_diff":     partial(forecast_ar, lags=2, transform="log_diff"),
}

X_models = model_error_panel(rtdsm, models, target_quarters, spf_horizon=HORIZON, metric=METRIC)
```

`model_error_panel` calls `model_forecast_series` once per model, computes per-quarter squared (or absolute) errors against the advance estimate, and returns a wide DataFrame indexed by `(YEAR, QUARTER)`. The resulting `X_models` is then handed to `rank_ci_stepwise_pairwise` and `rank_ci_marginal_pairwise` from the `rankci` package.

The evaluation grid is taken from `select_top_forecasters(..., N=8).index` — i.e. the target quarters where the SPF top-8 panel had coverage — so the comparison window matches the SPF notebooks. The SPF columns themselves are *not* included in the ranking; only the seven models compete.

## Ranking results

Run on **NGDP**, horizon `H=3` (one-quarter-ahead), squared-error loss, `α = 0.2`, `B = 5000`, NW-HAC pairwise standard errors. The evaluation window is 224 target quarters from `(1968, 4)` to `(2025, 1)`. CIs are computed with both the **stepwise (simultaneous)** procedure and the **marginal (per-model)** procedure.

| Rank | model            | MSE        | RMSE     | CI_step | CI_marg | cv_j |
|-----:|------------------|-----------:|---------:|:-------:|:-------:|-----:|
| 1    | `ar1_levels`     | 87,873     | 296.4    | [1, 3]  | [1, 3]  | 1.42 |
| 2    | `ar2_log_diff`   | 120,514    | 347.2    | [1, 6]  | [1, 5]  | 2.11 |
| 3    | `ar1_log_diff`   | 122,081    | 349.4    | [1, 6]  | [1, 5]  | 2.09 |
| 4    | `rw_drift`       | 134,142    | 366.3    | [2, 4]  | [2, 4]  | 1.41 |
| 5    | `naive`          | 183,451    | 428.3    | [3, 5]  | [3, 5]  | 1.36 |
| 6    | `ma4`            | 389,640    | 624.2    | [4, 6]  | [6, 6]  | 1.20 |
| 7    | `historical_mean`| 81,695,360 | 9,038.6  | [7, 7]  | [7, 7]  | 0.96 |

Point ranks are by mean squared error (smaller is better). The interval `[L, U]` means we cannot reject — at the chosen `α` — any rank in `[L, U]` for that model.

### What the CIs tell us

- **`historical_mean` is decisively last** under both procedures (`[7, 7]`). For a strongly trending series, the cumulative mean is hopeless — the RMSE of ~9,000 is roughly 25× the next-worst model, and *every* pair `(historical_mean, k)` rejects in the stepwise procedure.
- **`ar1_levels` is confirmed to be in the top 3 simultaneously** (`[1, 3]`). It cannot be statistically distinguished from `ar2_log_diff` or `ar1_log_diff`, but it *can* be distinguished from `rw_drift` and worse models. This is somewhat surprising given that log-diff is the textbook recommendation for NGDP — the AR(1) on levels apparently picks up enough of the trend through its intercept to stay competitive at one-step-ahead.
- **`naive` lands solidly mid-pack** at `[3, 5]`. Confirmed worse than `ar1_levels` (`naive` is excluded from the top 2), confirmed better than `ma4` and `historical_mean`.
- **`ma4` is pinned to rank 6** under the marginal procedure but only constrained to `[4, 6]` simultaneously. The marginal procedure rejects more pairs because it doesn't pay the family-wise multiplicity cost — at the cost of no joint coverage guarantee.
- **The two log-diff AR models are statistically indistinguishable from each other** (overlapping `[1, 6]` and `[1, 5]` intervals). Adding the second lag changed the MSE by ~1.5%, well inside the noise band.

The marginal CIs are narrower than the stepwise CIs everywhere except where they coincide — exactly the trade-off described in the package docs. Marginal controls per-model coverage only; stepwise controls coverage jointly across all models.

### Reproducing

Open [MODEL_COMPARISON.ipynb](notebooks/MODEL_COMPARISON.ipynb), keep `INDICATOR="NGDP"` and the defaults, run all cells. Switching to UNEMP (uncomment the alternative `INDICATOR` block at the top) re-runs the same pipeline on the unemployment series. To add a model, append one entry to the `models` dict — the rest of the panel construction and ranking picks it up automatically.
