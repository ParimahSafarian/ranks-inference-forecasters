# Step 4 · Informative application — Design Spec: ranking Good Judgment Project forecasters

**Status:** proposed, awaiting approval + a data-download go-ahead. No code yet.
(Supersedes the shelved SPF-density App 2, `STEP4_APP2_DESIGN.md`.)
**Goal:** rank individual GJP forecasters by **Brier score** through the *existing*
rank-CI pipeline (bootstrap + MDS + Ω), producing an **informative** ranking that
separates a top tier — the counterpoint to the deliberately null-ish macro results.
**Harmony = same method, two regimes.** One pipeline: on macro consensus it honestly
reports wide overlapping ranks; on GJP it confidently separates skill, and we
validate that the separated top coincides with the designated **superforecasters**.

Why the fit is exact: each question's realized outcome is shared by every forecaster
who answered it → a **question-level common shock** that cancels in Brier
*differences*. This is the additive-common-shock model, and it is precisely the
"standardize Brier within question" step of the GJP papers, done properly through the
difference covariance. Forecasters answer different question subsets → a heavily
**unbalanced** panel → the regime where Ω separates from MDS.

---

## 1. Data (public, must be downloaded)

Harvard Dataverse, **DOI `10.7910/DVN/BPCDH5`**, **CC0 license** (unrestricted), no
account needed. Files needed:
- `survey_fcasts.yr1..4.tab` — individual daily forecasts (long: one row per answer
  option). **~29 / 41 / 71 / 166 MB** (≈307 MB total).
- `ifps.csv` — question metadata + **outcomes** (~1 MB).
Ignore the `pm_*` prediction-market files and `all_individual_differences.*` (optional
covariates). **This is a ~300 MB download → needs your explicit go-ahead** (§7).

**Forecast columns:** `ifp_id`, `user_id`, `fcast_date`, `timestamp`, `answer_option`
(a–e), `value` (prob 0–1), `fcast_type` (0 new/1 update/2 affirm/4 withdraw),
condition fields (`ctt`/`cond`/`training`/`team`), `year`.
**IFP columns:** `ifp_id`, `outcome` (letter that occurred), `q_status` (keep Closed,
drop Voided), `date_start`, `date_closed`, `q_type`, `q_text`.

## 2. The loss: canonical daily-averaged Brier

Multi-category Brier, **0 (best) – 2 (worst)**:
```
Brier_day = Σ_options ( value_option − 1{option == outcome} )²
```
Per (forecaster, question): expand submissions to a **daily grid** from the
forecaster's first forecast to `date_closed`, **carry the last submission forward**
(the GJP convention), compute daily Brier, **average over the active days**. One Brier
per (forecaster × question). Binary = 2 options, multinomial = 3–5 (sum over all).

## 3. The panel → existing pipeline

Wide **`(question × forecaster)`** matrix of per-question Brier scores:
- **Rows = questions**, ordered by `date_closed` → the "time" index for the HAC layer.
- **Columns = forecasters.** Unbalanced (NaN where a forecaster didn't answer).
Hand it to `rank_ci_stepwise_pairwise` / `..._simulation_pairwise(covariance=…)` /
`..._marginal_pairwise`. **No pipeline changes.**

Note on dependence emphasis: here the **cross-sectional** common shock (question
outcome) is the dominant structure; **serial** dependence across consecutive
questions is likely mild (the Andrews bandwidth adapts, often small L). So GJP mainly
exercises the Ω / difference-covariance side — complementary to macro, which had both.

## 4. Sample construction (decisions in §6)

1. **Condition:** independent individuals only (exclude teams and the `pm_*` markets)
   so scores are comparable. Keep the **superforecaster** designation aside as an
   external **validation label**, not a filter.
2. **Questions:** `q_status == Closed`, drop Voided; pool all four years; binary +
   multinomial both (multi-category Brier handles both).
3. **Forecasters:** keep those with ≥ `min_questions` answered (dense overlap for
   pairwise SEs) — analogous to `select_top_forecasters`. Rank the top-N most active.

## 5. Module layout

```
src/rankci/data/gjp.py   NEW  load_gjp_forecasts(dir); load_gjp_ifps(dir);
                              brier_panel(..., condition, min_questions) -> wide (question x forecaster)
                              + daily carry-forward Brier; a download helper (run only on approval)
tests/test_gjp.py        NEW  Brier correctness (binary + multinomial), carry-forward
                              daily-average, panel shape/imbalance — on tiny synthetic frames
notebooks/gjp/GJP_CI.ipynb  NEW  ranking (bootstrap + mds + omega) + superforecaster-recovery
                              check + explicit two-regime contrast with the macro panels
```

## 6. Decisions (my defaults — flag any to change)

1. **Score = canonical daily-averaged multi-category Brier** (carry-forward). Offer a
   lighter "average over submission events" variant for speed/robustness checks.
2. **Condition = independent individuals**, superforecaster flag used only to validate.
3. **Pool all 4 years**; questions ordered by close date as the time axis.
4. **`min_questions` ≈ 40** and **top-N ≈ 30–50** forecasters for the headline panel
   (enough overlap to separate; tune once data is in).
5. **Withdrawals** (`fcast_type==4`): drop the row; carry the last non-withdraw forecast
   to close (simplification; note it).
6. **Validation:** report whether the rank-CI top tier matches designated
   superforecasters (Yr2–4) — the informative-regime payoff.

## 7. Data-download step (needs approval)

I will not download without your OK. On approval I'll pull the 4 `survey_fcasts.yrN.tab`
+ `ifps.csv` (~300 MB, CC0) from Dataverse into `data/gjp/` (git-ignored like other raw
data). Alternatively you download them and drop them in `data/gjp/`. Could also start
with **yr1+yr2 only (~70 MB)** for a faster first pass if you prefer.
