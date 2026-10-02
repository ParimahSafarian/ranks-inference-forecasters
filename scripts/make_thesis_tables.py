"""
Regenerate the thesis result tables that no notebook writes, into thesis/tables/.

The thesis/ folder holds only the document; all code that produces its tables
lives here or in the notebooks.

Run from anywhere:  <repo>/.venv/bin/python scripts/make_thesis_tables.py [--out DIR]
Tables: spf_indicators, three_regimes.
(spf_dep_* come from notebooks/02_primary_analysis.ipynb; mcomp from
notebooks/mcomp/MCOMP_CI.ipynb section 5; mc_coverage and mc_crossover from
notebooks/01_toydataset.ipynb; ngdp from notebooks/philly/NGDP_CI.ipynb.)
"""
import argparse
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]      # repo root
DATA = ROOT / "data"
OUT = ROOT / "thesis" / "tables"

SEED = 42


# ── panels ───────────────────────────────────────────────────────────────────

def spf_panel(indicator, N=8):
    """Top-N (by coverage, >= 20 obs) one-quarter-ahead squared-error panel."""
    from rankci import select_top_forecasters
    from rankci.data.philly import load_spf, load_rtdsm, compute_error_panel
    rtdsm_file, prefix, freq = {
        "NGDP": ("NOUTPUTQvQd.xlsx", "NOUTPUT", "quarterly"),
        "RGDP": ("ROUTPUTQvQd.xlsx", "ROUTPUT", "quarterly"),
        "UNEMP": ("rucQvMd.xlsx", "RUC", "monthly"),
    }[indicator]
    spf = load_spf(DATA / "philly" / "SPFmicrodata.xlsx", sheet=indicator)
    rt = load_rtdsm(DATA / "philly" / rtdsm_file, prefix=prefix, freq=freq)
    wide = compute_error_panel(spf, rt, indicator=indicator, horizon=3, metric="squared")
    return select_top_forecasters(wide, N=N, min_obs=20).values


def gjp_panel():
    """Top-20 individual GJP forecasters, years 1-2, daily-averaged Brier."""
    from rankci.data.gjp import load_gjp_ifps, load_gjp_forecasts, brier_panel
    ifps = load_gjp_ifps(DATA / "gjp" / "ifps.csv")
    fc = load_gjp_forecasts([DATA / "gjp" / "survey_fcasts.yr1.csv",
                             DATA / "gjp" / "survey_fcasts.yr2.csv"])
    panel, _ = brier_panel(fc, ifps, condition="individual",
                           min_questions=60, top_n=20, years=[1, 2])
    return panel.values


def paris_panel():
    """The 24 official M3 submissions on the Paris temperature record (306 x 24)."""
    from rankci.data.mcomp import official_panel
    return official_panel("Average temperature in Paris",
                          directory=str(DATA / "mcomp")).values


# ── helpers ──────────────────────────────────────────────────────────────────

def _separated(ci, p):
    """Units whose rank set is non-trivial (not [1, p])."""
    return (ci[:, 0] > 1) | (ci[:, 1] < p)


def _write(out_dir, name, tex):
    path = Path(out_dir) / f"{name}.tex"
    path.write_text(tex)
    print("wrote", path)


# ── tables ───────────────────────────────────────────────────────────────────

def tab_spf_indicators(out_dir):
    from rankci import compute_pairwise, rank_ci_stepwise_pairwise
    labels = {"NGDP": "NGDP (nominal output)", "RGDP": "RGDP (real output)",
              "UNEMP": "UNEMP (unemployment rate)"}
    rows = []
    for ind, label in labels.items():
        X = spf_panel(ind)
        n, p = X.shape
        delta, se, _ = compute_pairwise(X, se_method="nw")
        with np.errstate(invalid="ignore", divide="ignore"):
            max_t = float(np.nanmax(np.abs(delta / se)))
        ci = rank_ci_stepwise_pairwise(X, alpha=0.2, B=5000, seed=SEED,
                                       verbose=False)["rank_ci"]
        lo, hi = ci[int(np.argmin(np.nanmean(X, axis=0)))]
        rows.append(f"    {label:<26} & {p} & {n} & {max_t:.2f} & "
                    f"{int(_separated(ci, p).sum())} & $[{lo},{hi}]$\\\\")
    tex = r"""\begin{table}[htb]
  \centering
  \caption[SPF rank inference across indicators]{U.S.\ SPF rank inference across
  indicators (top-8 forecasters by coverage, one-quarter-ahead squared error,
  $\alpha=0.2$, bootstrap stepwise). The best forecaster carries the trivial $[1,8]$
  everywhere: even the unemployment panel, with the largest maximum pairwise
  $t$-statistic, cannot name a best forecaster. ``Separated'' counts forecasters whose
  rank set is non-trivial.}
  \label{tab:spf-indicators}
  \small
  \begin{tabular}{lccccc}
    \toprule
    indicator & $p$ & quarters & $\max|t|$ & separated & best-forecaster CI\\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    _write(out_dir, "spf_indicators", tex)


def tab_three_regimes(out_dir):
    from rankci import rank_ci_stepwise_simulation_pairwise as sim
    panels = [("macro (NGDP)", spf_panel("NGDP"), "null"),
              ("GJP (geopolitical)", gjp_panel(), "partial"),
              ("M3 (Paris temperature)", paris_panel(), "informative")]
    rows = []
    for name, X, regime in panels:
        n, p = X.shape
        ci = sim(X, alpha=0.1, B=20000, seed=SEED, covariance="omega",
                 verbose=False)["rank_ci"]
        width = (ci[:, 1] - ci[:, 0]).mean() / (p - 1)
        sep = 100 * _separated(ci, p).mean()
        exact = int((ci[:, 0] == ci[:, 1]).sum())
        rows.append(f"    {name:<22} & {p} & {n} & {width:.2f} & {sep:.0f}\\% & "
                    f"{exact} \\ (\\emph{{{regime}}})\\\\")
    tex = r"""\begin{table}[htb]
  \centering
  \caption[The three regimes]{The three applications under the same procedure (direct
  construction, $\alpha=0.1$ for all three; the GJP section uses $\alpha=0.2$).
  Normalized width is the mean rank-CI width divided by $p-1$; ``separated'' is the
  fraction of units with a non-trivial set; ``exact'' counts units pinned to a single
  rank.}
  \label{tab:threeregimes}
  \small
  \begin{tabular}{lccccl}
    \toprule
    dataset & $p$ & obs & norm.\ width & separated & exact ranks\\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    _write(out_dir, "three_regimes", tex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=str(OUT), help="output directory (default: thesis/tables)")
    out_dir = ap.parse_args().out
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tab_spf_indicators(out_dir)
    tab_three_regimes(out_dir)
    print("done ->", out_dir)
