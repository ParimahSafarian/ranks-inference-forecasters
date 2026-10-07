# Step 3 — Design Spec: Monte Carlo coverage study (thesis Section C)

**Status:** IMPLEMENTED (2026-08-11). `src/rankci/sim.py` + `tests/test_sim.py`
(8 tests) + rewritten `notebooks/philly/01_toydataset.ipynb` (21 cells, executed
at R=500/B=2000, 0 errors). Suite: 27 passing.

**Key finding — the approved DGP was extended.** On clean/stationary panels MDS and
Omega are *equivalent* (MDS marginally sharper) — this confirms the theory, but does
NOT show the contribution. Seven DGP variants showed the Omega advantage is a
**calibration** property that appears only when the pairwise SE matrix is driven
non-Euclidean, which needs **non-stationarity (volatility episodes) + imbalance**.
Added a `n_vol_episodes`/`vol_scale` knob to `simulate_panel`. The notebook shows the
two-regime story (equivalence → crossover) with the MDS projection gap as the
x-axis, plus the real-NGDP calibration diagnostic (MDS draw-SD 1→12, cv 24; Omega
draw-SD ~1, cv 2.7). See memory `omega-advantage-is-calibration`.

**Structure (per user):** the MDS study (Part 1) and Omega study (Part 2) are each
self-contained/skippable; Part 3 is the head-to-head. `coverage_study(methods=...)`
enables this. Bootstrap was dropped from the sweep (its per-resample cv is far too
slow for a coverage study; it remains available for single runs).

---

## Original spec (as approved)

**Goal:** upgrade `notebooks/philly/01_toydataset.ipynb` from a point check to a
coverage study on synthetic panels where the TRUTH (θ ordering) is known.
**The contribution shows here:** run `covariance="mds"` and `"omega"` head-to-head
on the SAME simulated panels; expect equal coverage, look for where Ω wins on
width / avoids PSD-projection damage, and where imbalance hurts each.
**Framing:** we validate the covariance estimator feeding MRSW; MRSW is unchanged.

---

## 1. The data-generating process (the crux)

The theory assumes the common shock is **additive at the loss level** so it cancels
in differences. So the DGP is

    X_{t,j} = θ_j + c_t + u_{t,j},     t = 1..T,  j = 1..p

- `u_{t,j}` — idiosyncratic, AR(1): `u_{t,j} = ρ u_{t-1,j} + ε`, `ε ~ N(0, σ_u²(1-ρ²))`
  so the stationary idiosyncratic variance is `σ_u²` regardless of ρ (serial dependence).
- `c_t` — common shock, AR(1) with coef `ρ_c`, variance `σ_c²` (**swept** — this is the
  cross-sectional dependence). Additive ⇒ `d^{jk}_t = (θ_j-θ_k) + (u_{t,j}-u_{t,k})`,
  the common shock cancels exactly. This is the object the covariance layer models.
- `θ_j` — known means fixing the true rank (rank 1 = smallest θ). Two families:
    * **gradient** `θ_j = (j-1)/(p-1) · spread` — evenly separated (coverage + width/power).
    * **clustered** — groups with equal θ (truly-tied pairs, for false-separation).
- **Imbalance** — staggered entry/exit: forecaster j observed on `[entry_j, exit_j]`,
  others NaN. Swept from balanced (0) to heavily staggered.

Rationale for additive (not squared-error) losses: the whole "common shock cancels in
differences" result requires additivity; squaring `(idio+c)²` reintroduces the common
term via cross-products and breaks the model the estimator is built on. Coverage of rank
inference is sign/scale-agnostic, so Gaussian-ish losses are fine for validation.

---

## 2. Estimand & metrics (per design point, averaged over R reps)

True rank of j = position of θ_j in ascending order. For each method m ∈ {mds, omega}
(same panel fed to both — **paired** comparison):

| Metric | Definition | Target |
|---|---|---|
| **Joint coverage** | mean over reps of `1{ true_rank_j ∈ R_{n,j}  ∀ j }` | ≥ 1−α |
| Marginal coverage | per-j `1{true_rank_j ∈ R_{n,j}}`; report mean & min over j | ≥ 1−α |
| Mean CI width | mean over j,reps of `U_j − L_j` (lower = more informative, *given* coverage) | — |
| False-separation | over truly-tied pairs (θ_j=θ_k): rate the procedure separates them | ≤ α |
| **Projection distance** | `‖Σ − Σ₊‖_F` (mds) / `‖Ω − Ω₊‖_F` (omega), mean over reps | diagnostic |

Headline is **joint** (simultaneous) coverage — that is what the stepwise procedure
controls. Bootstrap (`rank_ci_stepwise_pairwise`) is included as a third reference
column (resampling gold standard, no covariance) but is not the head-to-head.

---

## 3. Sweeps (one-at-a-time around a baseline)

Baseline: `p=5, T=200, ρ=0.3, σ_c²=1, balanced, gradient θ (spread=2·σ_u/√T scale)`.

- serial dependence   `ρ ∈ {0, 0.3, 0.6, 0.9}`
- common-shock var    `σ_c² ∈ {0, 1, 4, 16}`
- imbalance           `∈ {0, 0.1, 0.25, 0.4}`   ← **the money plot** (where Ω should win)
- size                `p ∈ {5, 8}`, `T ∈ {100, 200, 400}`
- ties                one clustered-θ design for false-separation

Expectation: balanced panels → mds ≈ omega (the two coincide given consistent inputs).
Imbalance → MDS's double-centering + projection degrades; Ω stays calibrated. The NGDP
result (Ω matched bootstrap, MDS collapsed to [1,p]) is the unbalanced-panel preview.

---

## 4. Module layout

```
src/rankci/sim.py          NEW  simulate_panel(...) DGP; true_ranks(theta);
                                coverage_study(...) driver -> tidy DataFrame
src/rankci/core/pairwise.py EDIT cov_theta_pairwise gains return_diagnostics
                                (raw Σ, ‖Σ−Σ₊‖_F, min eig) to mirror cov_via_omega
tests/test_sim.py          NEW  DGP means/ranks correct; common shock cancels in
                                differences; nominal coverage ~achieved on an easy
                                balanced design; driver output shapes
notebooks/philly/01_toydataset.ipynb  REWRITE  Section C: keep a short point-check
                                sanity, then DGP -> sweeps -> head-to-head tables + plots
```

`coverage_study` returns one row per (design point, method) with all metrics above, so the
notebook is mostly plotting. Reproducible via a base seed; each rep gets a distinct data seed.

---

## 5. Compute budget (needs a decision — it sets notebook runtime)

Cost ≈ R reps × (#grid points) × 2 methods × stepdown(B draws). Pure-Python stepdown +
`cov_via_omega` (O(q²·L·T)) dominate. Rough per-rep cost ~20–80 ms (p=5) / ~3–5× (p=8).

| Mode | R (reps) | B (draws) | ~Grid | Rough wall-clock | Use |
|---|---|---|---|---|---|
| pilot | 200 | 1000 | reduced | ~1–3 min | iterate on plots |
| **standard** | 500 | 2000 | full | ~8–15 min | thesis numbers (recommended) |
| thorough | 2000 | 5000 | full | ~1 hr+ | final camera-ready |

The notebook exposes `MODE` at the top; I'll default to **standard** and make pilot a
one-line switch. Monte Carlo SE on a 95% coverage estimate at R=500 is ≈ 1%, at R=2000 ≈ 0.5%.

---

## 6. Open decisions (my defaults — flag any to change)

1. **DGP = additive common-shock** (§1), not squared errors — required for the model
   the estimator targets. State this explicitly in the write-up.
2. **False-separation via truly-tied θ configs** (needs a clustered design in the grid).
3. **Coverage headline = joint/simultaneous**; marginal reported alongside.
4. **Head-to-head = mds vs omega** on the stepwise-simulation procedure; **bootstrap as a
   reference column**, not part of the contrast.
5. **Keep it in `01_toydataset.ipynb`** (rewrite) rather than a new notebook, per the plan.
```
