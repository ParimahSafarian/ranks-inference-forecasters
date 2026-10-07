# Step 2 — Design Spec: covariance seam + Andrews bandwidth

**Status:** IMPLEMENTED (2026-08-11). All 19 tests pass. See "Findings" below —
one result qualifies a claim in `omega_unbalanced.tex`.
**Scope:** Step 2 only (refactor + Andrews plug-in). Steps 3 (Monte Carlo) and 4
(application) are out of scope here.
**Framing:** we estimate a covariance for serially + cross-sectionally dependent
panel data and feed it to MRSW unchanged. Approach 1 (MDS) and Approach 2 (Ω)
are two *covariance constructions* selected by a runtime flag; MRSW downstream is
identical for both.

Locked decisions (from review):
- **Hard replace** the bandwidth: delete the `L = floor(4 (n/100)^{2/9})` rule.
  Andrews (1991) plug-in becomes the only automatic rule. Explicit integer `L`
  override is retained (needed for tests). This changes **every** existing
  result — bootstrap *and* simulation, Philly *and* ECB — because they all call
  `compute_pairwise(..., L=None)`. Intended.
- **The seam returns simulated studentized difference draws** (`B × q`), never a
  raw covariance. MDS lives in R^p (p×p level cov), Ω lives in R^q (q×q
  difference cov); pushing draw+studentize behind the seam is what lets one
  stepdown consume both.

---

## 1. Module layout

```
src/rankci/core/
  bandwidth.py      NEW  andrews_alpha1(), andrews_bandwidth()
  omega.py          NEW  cov_via_omega()  (+ _omega_entry, _stacked_nw reference)
  covariance.py     NEW  simulate_studentized_diffs(method=...)  ← THE SEAM
                         pair_index(p)  (unordered-pair <-> (j,k) maps)
  pairwise.py       EDIT nw_se(): Andrews default; delete T^{2/9}. Keep _nearest_psd,
                         cov_theta_pairwise (MDS, untouched — Step-3 benchmark).
  stepwise_simulation.py  EDIT route the pairwise stepdown through the seam;
                         draw once, restrict active columns per round.
  simulation.py     EDIT single-step pairwise variant uses the seam (1 round).
  __init__.py       EDIT export new public names.
```

Approach 2 is physically isolated in `omega.py` + `bandwidth.py`, so it can be
lifted out without touching the MDS path.

---

## 2. Andrews bandwidth (`bandwidth.py`)

Per **difference** series `d` (length `n` on its own support `T_a`):

1. Fit AR(1) by OLS of `d_t` on `d_{t-1}`, using only adjacent in-support pairs
   (`t, t-1` both observed). Get `rho`.
2. Clamp `rho` to `[-0.97, 0.97]` (guards the ρ→±1 blow-ups).
3. Andrews eq. (6.4), univariate case (σ⁴ cancels for a single series):
   `alpha1 = 4*rho**2 / ((1-rho)**2 * (1+rho)**2)`
4. `L = 1.1447 * (alpha1 * n) ** (1/3)`  (Bartlett rate T^{1/3}, Andrews Thm.1/eq.6.2).
5. Return `L_int = min(int(floor(L)), n-1)`.

```python
def andrews_alpha1(d: np.ndarray) -> float: ...
def andrews_bandwidth(d: np.ndarray) -> int: ...   # the L_a used everywhere
```

`nw_se(d, L=None, winsor_pct=None)`: when `L is None`, `L = andrews_bandwidth(d)`.
The old auto formula is removed. Bartlett kernel and the rest of `nw_se` unchanged.

Edge cases: `n < 3` → `L = 0` (falls back to lag-0 / iid-like). White noise →
`rho≈0` → `alpha1≈0` → `L≈0`.

---

## 3. Ω estimator (`omega.py`)

`cov_via_omega(X, min_overlap=2, return_diagnostics=False) -> Omega_psd (q×q)`

Unordered pairs `a=(j,k)`, `j<k`, indexed `0..q-1`, `q = p(p-1)/2`.
For each `a`: build full-length `d^a` (NaN off `T_a`), `L_a = andrews_bandwidth(d^a)`,
demean on `T_a` → `dtil^a`.

Entry `(a,b)`, with `a=(j,k)`, `b=(l,m)`:
```
Tab   = { t : d^a_t and d^b_t both observed }      # overlap, n_ab = |Tab|
if n_ab == 0:  Omega[a,b] = 0                        # unidentified → 0, PSD fixes
L_ab  = min(L_a, L_b, n_ab - 1)
g0    = sum_{t in Tab}            dtil^a_t * dtil^b_t
gh    = sum_{t: t,t-h in-support} dtil^a_t * dtil^b_{t-h}     # gamma^{ab}_h
gh_rev= sum_{t: t,t-h in-support} dtil^b_t * dtil^a_{t-h}     # gamma^{ba}_h
Omega[a,b] = (1/n_ab) * ( g0/n_ab + sum_{h=1..L_ab} w_h*(gh+gh_rev)/n_ab )
             with w_h = 1 - h/(L_ab+1)
```
i.e. **single common divisor `n_ab`** applied across all lags (not per-lag counts).
Symmetrize, then `_nearest_psd` (Higham eigen-clip). Diagnostics (optional):
`frob_gap = ||Omega - Omega_psd||_F`, `min_eig_before`.

Diagonal `a=b`: `Tab=T_a`, `n_ab=n_a`, `L_aa=L_a` → reduces to the univariate
Bartlett LRV / n_a = `nw_se(d^a, L_a)[1]**2`. This is asserted by test.

Reference `_stacked_nw(X) -> q×q` (balanced only) for the nesting test.

Complexity `O(q² · L̄ · n̄)`; fine for `p ≲ 12` (`q ≤ 66`). Flagged, not optimized.

---

## 4. The seam (`covariance.py`)

```python
def simulate_studentized_diffs(
    X, *, method, se_q, delta_q, pairs, alpha_inputs..., B, seed, min_overlap
) -> np.ndarray            # shape (B, q), columns = unordered pairs a
```
- `pairs`: list of `(j,k)` with `j<k`; `pair_index(p)` builds it + the reverse map.
- `se_q[a]`: NW-HAC SE of pair `a` (the studentization denominator, **same for
  both methods**). For omega, `se_q[a] == sqrt(Omega[a,a])` exactly (calibration).
- `method="mds"`: `Sigma = cov_theta_pairwise(...)` (p×p); draw `Z ~ N(0,Sigma)`
  (B×p); `D[:,a] = Z[:,j]-Z[:,k]`; `T[:,a] = D[:,a]/se_q[a]`.
- `method="omega"`: `Omega = cov_via_omega(...)` (q×q); draw `D ~ N(0,Omega)`
  (B×q) directly; `T[:,a] = D[:,a]/se_q[a]`.

**Stepdown consumes `T` identically** (draw once, reuse across rounds):
```
ordered pair (j,k), j<k -> column a, sign +1 ;  (k,j) -> column a, sign -1
active ordered set A_r each round r:
   cv_r = quantile_{1-alpha}( max_{(u,v) in A_r} sign_{uv} * T[:, a(u,v)] )
   reject (u,v) in A_r with  delta_hat[u,v] - cv_r * se_a > 0
   shrink; repeat until no rejection
```
This replaces the current re-draw-per-round in `_simulation_cv`. Same draws across
rounds is both faster and the correct monotone stepdown. `delta_hat[u,v]` for the
reverse direction is `-delta_q[a]`.

Runtime knob `covariance ∈ {"mds","omega"}` is threaded through
`rank_ci_stepwise_simulation_pairwise(..., covariance="mds")` and the single-step
variant, sitting next to indicator / metric / bandwidth.

`tau_best` simulation omega support: **out of scope** for Step 2 (different
statistic). The `cov_via_omega` draw is reusable there later.

---

## 5. What changes / breaks

- All simulation + bootstrap results shift (new bandwidth). `tau_best_report.tex`
  and `ecb_data_adoption.tex` numbers become stale — re-run in Step 4.
- `nw_se(..., L=None)` semantics change (Andrews, not T^{2/9}).
- Simulation stepdown RNG stream changes (draw-once), so seeded outputs won't
  byte-match old runs; validity is asserted instead of exact reproduction.
- Bootstrap path (`stepwise.py`, `tau_best.py` bootstrap) untouched except via the
  shared bandwidth.
- **API break:** `rank_ci_stepwise_simulation_pairwise` dropped `use_hac`, added
  `covariance ∈ {"mds","omega"}` (+ `winsor_pct`). It now always uses NW-HAC SEs.
  `NGDP_CI.ipynb` calls it with `use_hac=True` in 3 cells → map to
  `covariance="mds"` when re-running in Step 4. Other variants
  (`..._simulation_pairwise` single-step, `..._marginal_simulation_pairwise`)
  keep `use_hac`, unchanged.

---

## Findings (this qualifies the tex)

**"PSD for free on balanced panels" holds only under a COMMON bandwidth.**
The tex prescribes per-series Andrews bandwidths with `L_ab = min(L_a, L_b)`
(bandwidth paragraph), but also claims the balanced case nests the single-
bandwidth stacked NW `Ŝ/n` and is PSD-for-free (nesting subsection + remark).
These are jointly true only when all `L_a` are equal. Empirically, on a balanced
panel with heterogeneous persistence (`L = [7,7,7,10,10,10]`), the raw Ω has
`min_eig = -3.5e-5` and needs a real projection; with a common bandwidth,
`min_eig ≈ -1e-18` (exactly PSD). So the remark's "PSD for free is a
balanced-panel claim" should be weakened to "balanced-panel, **common-bandwidth**
claim" — or, with per-series bandwidths, the projection can be non-inert even on
balanced data. Does not affect correctness (projection handles it); it is a
wording/scope fix for the write-up, and the nesting test is written with a
common `L` to reflect this.

---

## 6. Test plan (`tests/`, currently empty)

| # | Test | Assertion |
|---|------|-----------|
| T1 | diagonal-consistency | `cov_via_omega(X)[a,a] == nw_se(d^a, L_a)[1]**2` (balanced & unbalanced), tol 1e-10 |
| T2 | balanced-nesting | on fully-observed `X`, `cov_via_omega == _stacked_nw(X)` entrywise, tol 1e-10 |
| T3 | calibration across seam | `se_q[a]**2 == Omega[a,a]` for omega path |
| T4 | PSD | `min_eig(Omega_psd) >= -1e-12`; `frob_gap == 0` on a balanced panel |
| T5 | Andrews bandwidth | white noise → `L≈0`; AR(1) ρ=0.5, n=500 → `L` matches hand calc |
| T6 | seam shape/validity | both methods return `(B,q)`; on balanced iid data both give rank CI `[1,p]` |
| T7 | stepdown monotonicity | active set is non-increasing; cv non-increasing across rounds |

---

## 7. Open design points (my defaults, flag any you'd change)

1. **Unidentified cross-entry** (`n_ab == 0`): set `Omega[a,b]=0`, let PSD projection
   absorb it. Alternative (shrink toward diagonal) not proposed.
2. **ρ clamp** at ±0.97 and **L cap** at `n_a-1`. Prevents blow-ups; standard.
3. **Andrews AR(1) on gapped support**: OLS on adjacent in-support pairs only
   (not interpolated). Simple, avoids inventing observations.
4. **Explicit `L`** still overrides Andrews (kept for tests/experiments); only the
   *automatic* rule is hard-replaced.
```
