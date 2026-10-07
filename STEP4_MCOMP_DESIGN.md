# Step 4 · Sharp application — Design Spec: ranking forecasting methods on the M-competition

**Status:** proposed, awaiting approval. Data path verified hands-on (M3 loads;
methods→sMAPE→panel runs). No production code yet.
**Goal:** rank standard forecasting **methods** by per-series accuracy over the
M-competition series, through the existing rank-CI pipeline — a **sharp, informative**
ranking (huge n) to complete the arc:
**macro (null) → GJP (partial) → M-competition (sharp).**

**Why it's sharp and non-obvious.** The serious methods are *close* (on 40 monthly
series: ETS 8.77 vs Theta 8.88 sMAPE — Theta *winning* M3 was a genuine surprise), so
"is the winner really #1?" is a real, unanswered question. With ~1,400+ series the SEs
shrink; and because the methods are **highly correlated across series** (a hard series
hurts everyone), the **paired difference** has far smaller variance than the marginals —
so the method separates methods that look tied marginally. This is the core value of
difference-based rank inference, on display.

**Method fit (identical structure to GJP/macro):** each series' difficulty is a shock
shared by all methods → **cancels in the sMAPE differences** (cross-sectional common
shock). Series are unordered → serial dependence ≈ 0 (Andrews L≈0), so this application
isolates the cross-sectional / Ω side. Panel = `(series × method)`.

---

## 1. Data (public, auto-downloaded, small)

`datasetsforecast` (pip) pulls the **M3** series (also M4 available). Groups and
horizons: Yearly (645 series, h=6), Quarterly (756, h=8), **Monthly (1,428, h=18)**,
Other (174, h=8) — 3,003 total. Long format `(unique_id, ds, y)`; the last `h` points
are the test window. ~0.3 MB/group, cached in `data/mcomp/` (git-ignored).

## 2. The loss

For each (series, method): fit the method on the training window, forecast `h` ahead,
score against the held-out test window with **sMAPE** (`mean 200·|F−A|/(|F|+|A|)`),
which is scale-free and comparable across series/frequencies. One number per
(series, method). (MASE offered as a secondary metric.)

## 3. The methods (serious contenders + standard benchmarks — no strawmen)

- **Benchmarks:** Naive2 (seasonally-adjusted naive — the M-comp reference), Seasonal-Naive.
- **Serious:** ETS (additive), Damped-trend ETS, **Theta** (M3 winner), AutoARIMA
  (simple order search or a fixed (p,d,q)(P,D,Q)), and optionally a simple combination.
All are legitimate; the informative question is the ordering *among the serious ones*.
Implemented via `statsmodels` (ExponentialSmoothing, ThetaModel, ARIMA) — lightweight.

## 4. The panel → existing pipeline

Wide `(series × method)` sMAPE matrix → `rank_ci_stepwise_pairwise` /
`..._simulation_pairwise(covariance=…)` / `..._marginal_pairwise`. Rows = series
(the observation axis; unordered), columns = methods. Essentially balanced (every
method scores every series) → MDS ≈ Ω expected (the balanced regime). **No pipeline
changes.**

## 5. Module layout

```
src/rankci/data/mcomp.py   NEW  load_m3(group); the method wrappers (naive2, snaive,
                                ets, damped, theta, arima); smape/mase;
                                method_panel(group, methods) -> wide (series x method) sMAPE
tests/test_mcomp.py        NEW  sMAPE/MASE correctness; method wrappers on synthetic
                                series; panel shape/ordering
notebooks/mcomp/MCOMP_CI.ipynb  NEW  the sharp ranking (bootstrap + mds + omega) +
                                the three-regime contrast (macro / GJP / M-comp)
```

## 6. Decisions (my defaults — flag any to change)

1. **Headline group = Monthly** (1,428 series — the largest single-frequency panel);
   report Quarterly/Yearly and the pooled 3,003-series panel as robustness.
   (sMAPE is scale-free, so pooling across frequencies is valid.)
2. **Metric = sMAPE**; MASE secondary.
3. **Methods** = the 6 above; the ordering among {ETS, Damped, Theta, ARIMA} is the
   non-obvious result.
4. **Competition = M3** first (fast, classic); M4 (100k series) as an extension.
5. **Dependencies:** `statsmodels` + `datasetsforecast` are *application-only* deps
   (in the venv, noted in the notebook) — NOT added to the core `rankci` package deps.
6. Runtime: ~2–4 min to build the Monthly panel (6 methods × 1,428 series); cache it.
