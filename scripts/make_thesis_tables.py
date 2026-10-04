"""
Regenerate the thesis result tables that no notebook writes, into thesis/tables/,
and print the application numbers the text quotes without a table.

The thesis/ folder holds only the document; all code that produces its tables
lives here or in the notebooks. Every rank set comes from the simulation route
(rank_ci_stepwise_simulation_pairwise); the block bootstrap appears only as a
benchmark column/sentence.

Run from anywhere:  <repo>/.venv/bin/python scripts/make_thesis_tables.py [--out DIR] [--skip-mc]
Tables: spf_indicators, ngdp, three_regimes; the Monte Carlo tables mc_coverage,
mc_crossover and the Appendix B (app:additional) tables app_mc_coverage, app_mc_width,
app_mc_crossover, all from one run (a few minutes; --skip-mc leaves them alone); and
app_spf_sets, app_ecb_sets, app_gjp, app_m3_blocks, app_m3_robust.
Printed: the NGDP tau-best sets (sec:apps-spf), the euro-area results (sec:apps-ecb), and
the block-bootstrap benchmark on the application panels (par:mbb; the bootstrap
sentences of Chapter 8).
(spf_dep_* come from notebooks/02_primary_analysis.ipynb; mcomp from
notebooks/mcomp/MCOMP_CI.ipynb. The Monte Carlo designs and seeds are those
of notebooks/01_toydataset.ipynb.)
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

def spf_frame(indicator, N=8):
    """Top-N (by coverage, >= 20 obs) one-quarter-ahead squared-error panel, with IDs."""
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
    return select_top_forecasters(wide, N=N, min_obs=20)


def spf_panel(indicator, N=8):
    """Top-N (by coverage, >= 20 obs) one-quarter-ahead squared-error panel."""
    return spf_frame(indicator, N).values


def ecb_panel(indicator, N=8):
    """Euro-area SPF top-N (by coverage, >= 15 obs) annual-target squared-error panel."""
    return ecb_frame(indicator, N).values


def ecb_frame(indicator, N=8):
    """Euro-area SPF top-N (by coverage, >= 15 obs) annual-target squared-error panel, with IDs."""
    from rankci import select_top_forecasters
    from rankci.data.ecb import (load_ecb_spf, add_horizon, error_panel,
                                 load_hicp_realized, hicp_realized_by_target_period,
                                 load_rgdp_index_sdmx, rgdp_yoy_from_index,
                                 rgdp_realized_by_target_period)
    ecb = DATA / "ecb"
    if indicator == "HICP":
        realized = hicp_realized_by_target_period(load_hicp_realized(
            str(ecb / "prc_hicp_manr__custom_21709551_spreadsheet.xlsx")))
    else:
        index = load_rgdp_index_sdmx(str(ecb / "namq_10_gdp_EA_RGDP.xml"), expected_dims={
            "unit": "CLV_I20", "na_item": "B1GQ", "s_adj": "SCA", "geo": "EA"})
        realized = rgdp_realized_by_target_period(rgdp_yoy_from_index(index),
                                                  annual_method="mean")
    spf = add_horizon(load_ecb_spf(str(ecb / "individual_forecasts"), indicators=[indicator]))
    wide = error_panel(spf, realized, indicator=indicator, target_kind="year",
                       horizon_q=3, metric="squared")
    return select_top_forecasters(wide, N=N, min_obs=15)


def gjp_panel():
    """Top-20 individual GJP forecasters, years 1-2, daily-averaged Brier."""
    return gjp_frame().values


def gjp_frame():
    """Top-20 individual GJP forecasters, years 1-2, daily-averaged Brier, with IDs;
    rows are the questions ordered by close date."""
    from rankci.data.gjp import load_gjp_ifps, load_gjp_forecasts, brier_panel
    ifps = load_gjp_ifps(DATA / "gjp" / "ifps.csv")
    fc = load_gjp_forecasts([DATA / "gjp" / "survey_fcasts.yr1.csv",
                             DATA / "gjp" / "survey_fcasts.yr2.csv"])
    panel, _ = brier_panel(fc, ifps, condition="individual",
                           min_questions=60, top_n=20, years=[1, 2])
    return panel


def paris_panel():
    """The 24 official M3 submissions on the Paris temperature record (306 x 24)."""
    from rankci.data.mcomp import official_panel
    return official_panel("Average temperature in Paris",
                          directory=str(DATA / "mcomp")).values


# ── helpers ──────────────────────────────────────────────────────────────────

def _separated(ci, p):
    """Units whose rank set is non-trivial (not [1, p])."""
    return (ci[:, 0] > 1) | (ci[:, 1] < p)


def _max_t(X):
    """Largest |studentized contrast| over all pairs, NW-HAC standard errors."""
    from rankci import compute_pairwise
    delta, se, _ = compute_pairwise(X, se_method="nw")
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmax(np.abs(delta / se)))


def _write(out_dir, name, tex):
    path = Path(out_dir) / f"{name}.tex"
    path.write_text(tex)
    print("wrote", path)


# ── tables ───────────────────────────────────────────────────────────────────

def tab_spf_indicators(out_dir):
    from rankci import rank_ci_stepwise_simulation_pairwise as sim
    labels = {"NGDP": "NGDP (nominal output)", "RGDP": "RGDP (real output)",
              "UNEMP": "UNEMP (unemployment rate)"}
    rows = []
    for ind, label in labels.items():
        X = spf_panel(ind)
        n, p = X.shape
        ci = sim(X, alpha=0.2, B=5000, seed=SEED, covariance="omega",
                 verbose=False)["rank_ci"]
        lo, hi = ci[int(np.argmin(np.nanmean(X, axis=0)))]
        rows.append(f"    {label:<26} & {p} & {n} & {_max_t(X):.2f} & "
                    f"{int(_separated(ci, p).sum())} & $[{lo},{hi}]$\\\\")
    tex = r"""\begin{table}[htb]
  \centering
  \caption[SPF rank inference across indicators]{U.S.\ SPF rank inference across
  indicators (top-8 forecasters by coverage, one-quarter-ahead squared error, direct
  construction, $\alpha=0.2$, $B=5000$). No indicator names a best forecaster.
  ``Separated'' counts forecasters whose rank set is non-trivial; the last column is the
  rank set of the forecaster with the smallest mean loss.}
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


def tab_ngdp(out_dir):
    from rankci import rank_ci_stepwise_simulation_pairwise as sim, tau_best_from_rank_ci
    P = spf_frame("NGDP", N=5)
    X, ids = P.values, [int(i) for i in P.columns]
    ci = {cov: sim(X, alpha=0.2, B=5000, seed=SEED, covariance=cov,
                   verbose=False)["rank_ci"] for cov in ("mds", "omega")}
    theta = np.nanmean(X, axis=0)
    rows = []
    for j in np.argsort(theta):
        mse = f"{theta[j]:,.0f}".replace(",", "{,}")
        rows.append(f"    {ids[j]:<3} & {mse:<9} & $[{ci['mds'][j, 0]},{ci['mds'][j, 1]}]$ & "
                    f"$[{ci['omega'][j, 0]},{ci['omega'][j, 1]}]$\\\\")
    tex = r"""\begin{table}[htb]
  \centering
  \caption[NGDP rank confidence sets]{NGDP, top-5 SPF forecasters, one-quarter-ahead
  squared error ($\alpha=0.2$, $B=5000$). The MDS simulation loses calibration and
  reports the trivial $[1,5]$ throughout; the direct construction separates three
  forecasters.}
  \label{tab:ngdp}
  \small
  \begin{tabular}{lrcc}
    \toprule
    ID & mean MSE & CI (\textsc{mds}) & CI ($\OmegaDirect$)\\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    _write(out_dir, "ngdp", tex)
    # tau-best sets of sec:apps-spf: projections of the direct construction's rank sets
    for tau in (1, 2, 3):
        members = [ids[j] for j in np.where(tau_best_from_rank_ci(ci["omega"], tau))[0]]
        print(f"  NGDP top-5, tau={tau}-best set (direct, projected): {members}")


def ecb_numbers():
    """Euro-area SPF (sec:apps-ecb): max |t| and the rank sets under MDS and direct."""
    from rankci import rank_ci_stepwise_simulation_pairwise as sim
    for ind in ("HICP", "RGDP"):
        X = ecb_panel(ind)
        n, p = X.shape
        print(f"  euro-area {ind}: p={p}, years={n}, max|t|={_max_t(X):.2f}")
        for cov in ("mds", "omega"):
            ci = sim(X, alpha=0.2, B=5000, seed=SEED, covariance=cov,
                     verbose=False)["rank_ci"]
            sets = " ".join(f"[{lo},{hi}]" for lo, hi in ci)
            print(f"    {cov:<5} separated={int(_separated(ci, p).sum())}: {sets}")


def block_bootstrap_numbers(B=5000):
    """Block-bootstrap benchmark (par:mbb) on the application panels: the bootstrap
    sentences of Chapter 8. Each panel is compared with the direct-construction rank
    sets at the level and B of the result it checks; the bootstrap uses B=5000."""
    from rankci import rank_ci_stepwise_pairwise as boot
    from rankci import rank_ci_stepwise_simulation_pairwise as sim
    from rankci.data.mcomp import official_panel
    panels = [("SPF NGDP top-8", spf_frame("NGDP"), 0.2, 5000),
              ("SPF RGDP top-8", spf_frame("RGDP"), 0.2, 5000),
              ("SPF UNEMP top-8", spf_frame("UNEMP"), 0.2, 5000),
              ("euro-area HICP", ecb_frame("HICP"), 0.2, 5000),
              ("euro-area RGDP", ecb_frame("RGDP"), 0.2, 5000),
              ("GJP top-20", gjp_frame(), 0.2, 5000),
              ("M3 Paris", official_panel(PARIS, directory=str(DATA / "mcomp")), 0.1, 20000)]
    for name, F, alpha, B_direct in panels:
        X, ids = F.values, list(F.columns)
        p = X.shape[1]
        direct = sim(X, alpha=alpha, B=B_direct, seed=SEED, covariance="omega",
                     verbose=False)["rank_ci"]
        out = boot(X, alpha=alpha, B=B, seed=SEED, verbose=False)
        cbb = out["rank_ci"]
        width = lambda ci: (ci[:, 1] - ci[:, 0]).mean() / (p - 1)
        changed = [j for j in np.argsort(np.nanmean(X, axis=0))
                   if not np.array_equal(cbb[j], direct[j])]
        wider = sum(cbb[j, 0] <= direct[j, 0] and cbb[j, 1] >= direct[j, 1] for j in changed)
        print(f"  block bootstrap, {name} (alpha={alpha}, block length "
              f"{out['block_length']}): separated {int(_separated(cbb, p).sum())} vs "
              f"{int(_separated(direct, p).sum())} direct; {len(changed)} of {p} sets "
              f"differ ({wider} wider), max shift {int(np.abs(cbb - direct).max())}; "
              f"norm. width {width(cbb):.2f} vs {width(direct):.2f}")
        for j in changed:
            print(f"      {ids[j]}: direct [{direct[j, 0]},{direct[j, 1]}] -> "
                  f"bootstrap [{cbb[j, 0]},{cbb[j, 1]}]")


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


# ── Appendix B (app:additional) ──────────────────────────────────────────────

# Monte Carlo grid of notebooks/01_toydataset.ipynb. The list order fixes the seeds
# (replication r of design i uses seed 10^6 i + r), so keep it in sync with the notebook.
MC_BASE = dict(theta=list(np.linspace(0, 1.6, 5)), T=200, rho=0.3, sigma_c=1.0,
               imbalance=0.0)
MC_DESIGNS = [
    {**MC_BASE, "name": "baseline"},
    {**MC_BASE, "name": "rho=0.0", "rho": 0.0},
    {**MC_BASE, "name": "rho=0.6", "rho": 0.6},
    {**MC_BASE, "name": "rho=0.9", "rho": 0.9},
    {**MC_BASE, "name": "sigc=0", "sigma_c": 0.0},
    {**MC_BASE, "name": "sigc=4", "sigma_c": 4.0},
    {**MC_BASE, "name": "imbal=0.2", "imbalance": 0.2},
    {**MC_BASE, "name": "imbal=0.4", "imbalance": 0.4},
    {**MC_BASE, "name": "nonstat", "n_vol_episodes": 5, "vol_scale": 12.0},
    {**MC_BASE, "name": "realistic", "imbalance": 0.4, "n_vol_episodes": 5,
     "vol_scale": 12.0},
    {"theta": [0.0, 0.0, 0.8, 0.8, 1.6], "T": 200, "name": "ties"},
]
MC_LABELS = {"baseline": "baseline", "rho=0.0": r"$\rho=0$", "rho=0.6": r"$\rho=0.6$",
             "rho=0.9": r"$\rho=0.9$", "sigc=0": r"$\sigma_c=0$", "sigc=4": r"$\sigma_c=4$",
             "imbal=0.2": r"imbalance $0.2$", "imbal=0.4": r"imbalance $0.4$",
             "nonstat": "non-stationary", "realistic": "realistic", "ties": "ties"}
MC_VOL_GRID = [1, 3, 6, 10, 16, 24]

# official M3 column -> (name in the thesis, class of tab:mcomp)
M3_NAMES = {
    "B-J auto": ("B-J automatic", "A"), "ForecastPro": ("ForecastPro", "E"),
    "ForcX": ("ForecastX", "E"), "DAMPEN": ("Dampen", "T"),
    "COMB S-H-D": ("Comb S-H-D", "T"), "THETA": ("Theta", "D"), "WINTER": ("Winter", "T"),
    "HOLT": ("Holt", "T"), "SINGLE": ("Single", "N"), "RBF": ("RBF", "E"),
    "SMARTFCS": ("SmartFcs", "E"), "AutoBox1": ("AutoBox1", "A"),
    "AutoBox2": ("AutoBox2", "A"), "AutoBox3": ("AutoBox3", "A"), "NAIVE2": ("Naive2", "N"),
    "AAM1": ("AAM1", "A"), "AAM2": ("AAM2", "A"), "THETAsm": ("Theta-sm", "T"),
    "Auto-ANN": ("Automat ANN", "NN"), "PP-Autocast": ("PP-Autocast", "T"),
    "ARARMA": ("ARARMA", "A"), "Flors-Pearc1": ("Flores-Pearce1", "E"),
    "Flors-Pearc2": ("Flores-Pearce2", "E"), "ROBUST-Trend": ("Robust-Trend", "T"),
}
PARIS = "Average temperature in Paris"


def _set(ci_row):
    return f"$[{ci_row[0]},{ci_row[1]}]$"


def _num(x):
    """Mean loss: thousands separator for large values, three decimals for small."""
    if x >= 1000:
        return f"{x:,.0f}".replace(",", "{,}")
    return f"{x:.0f}" if x >= 100 else f"{x:.3f}"


def _table(label, short, caption, colspec, header, body):
    return (r"""\begin{table}[htb]
  \centering
  \caption[""" + short + "]{" + caption + r"""}
  \label{""" + label + r"""}
  \small
  \begin{tabular}{""" + colspec + r"""}
    \toprule
""" + header + r"""
    \midrule
""" + "\n".join(body) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")


def _rank_sets(X, alpha, B, covs=("omega", "mds")):
    from rankci import rank_ci_stepwise_simulation_pairwise as sim
    return {c: sim(X, alpha=alpha, B=B, seed=SEED, covariance=c, verbose=False)["rank_ci"]
            for c in covs}


def m3_official_long():
    from rankci.data.mcomp import load_m3_official
    return load_m3_official(str(DATA / "mcomp"))


def tab_app_mc(out_dir, R=500, B=2000, alpha=0.05):
    """Full Monte Carlo grid (all 11 designs, MDS / direct / block bootstrap on the same
    panels) and the crossover sweep with both projection distances and coverage."""
    from rankci.sim import coverage_study
    res = coverage_study(MC_DESIGNS, R=R, B=B, alpha=alpha, methods=("mds", "omega"),
                         include_bootstrap=True, progress=True)
    get = lambda name, m, col: res.loc[(res["name"] == name) & (res["method"] == m),
                                       col].item()
    routes = ("mds", "omega", "bootstrap")
    cov_rows, width_rows = [], []
    for d in MC_DESIGNS:
        name = d["name"]
        lab = MC_LABELS[name]
        if name != "ties":
            cov_rows.append(f"    {lab} & " + " & ".join(
                f"{get(name, m, c):.3f}" for c in ("joint_coverage", "marg_coverage_min")
                for m in routes) + r"\\")
        width_rows.append(f"    {lab} & " + " & ".join(
            [f"{get(name, m, 'mean_width'):.3f}" for m in routes]
            + [f"{get(name, m, 'proj_gap_mean'):.3f}" for m in ("mds", "omega")]
            + [f"{get(name, m, 'n_steps_mean'):.2f}" for m in ("mds", "omega")]) + r"\\")
    cov_rows.append(r"    \midrule" + "\n    ties: false separation & " + " & ".join(
        f"{get('ties', m, 'false_sep_rate'):.3f}" for m in routes) + r" & & & \\")
    three = r"\textsc{mds} & $\OmegaDirect$ & \textsc{cbb}"
    _write(out_dir, "app_mc_coverage", _table(
        "tab:app-mc-coverage", "Monte Carlo coverage, full grid",
        r"Joint coverage and the smallest marginal coverage over the five forecasters, "
        r"for every design of the Monte Carlo grid ($R=500$, $B=2000$, nominal level "
        r"$0.95$; identical panels for the three routes). The last row is the share of "
        r"replications in which a tied pair of the tie design is confidently ordered.",
        "lcccccc",
        r"    & \multicolumn{3}{c}{joint coverage} & \multicolumn{3}{c}{min.\ marginal coverage}\\"
        "\n" r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
        f"    design & {three} & {three}\\\\", cov_rows))
    _write(out_dir, "app_mc_width", _table(
        "tab:app-mc-width", "Monte Carlo width and projection distance, full grid",
        r"Mean rank-set width, mean projection distance $\Frob{A-A^{+}}$ of the matrix "
        r"each simulation route draws from, and mean number of stepdown rounds, for every "
        r"design of the Monte Carlo grid ($R=500$, $B=2000$, $\alpha=0.05$).",
        "lccccccc",
        r"    & \multicolumn{3}{c}{mean width} & \multicolumn{2}{c}{projection distance}"
        r" & \multicolumn{2}{c}{stepdown rounds}\\" "\n"
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n"
        f"    design & {three} & \\textsc{{mds}} & $\\OmegaDirect$ & \\textsc{{mds}} & $\\OmegaDirect$\\\\",
        width_rows))

    # Table 7.1 (tab:mc-coverage): the main-text subset of the same run
    main_rows = []
    for name in ("baseline", "rho=0.6", "rho=0.9", "sigc=0", "sigc=4", "imbal=0.4",
                 "nonstat", "realistic"):
        bold = (lambda s: rf"\textbf{{{s}}}") if name == "realistic" else (lambda s: s)
        cells = ([f"{get(name, m, 'joint_coverage'):.3f}" for m in routes]
                 + [bold(f"{get(name, m, 'mean_width'):.3f}") for m in ("mds", "omega")]
                 + [f"{get(name, 'bootstrap', 'mean_width'):.3f}"]
                 + [f"{get(name, m, 'proj_gap_mean'):.3f}" for m in ("mds", "omega")])
        main_rows.append(f"    {bold(MC_LABELS[name])} & " + " & ".join(cells) + r"\\")
    _write(out_dir, "mc_coverage", _table(
        "tab:mc-coverage", "Monte Carlo coverage and width",
        r"Joint coverage, mean rank-CI width, and" "\n  "
        r"mean projection gap, MDS vs.\ direct-$\OmegaDirect$, with the joint circular block" "\n  "
        r"bootstrap (\textsc{cbb}, \S\ref{par:mbb}) as a covariance-free benchmark ($R=500$," "\n  "
        r"$B=2000$, $\alpha=0.05$, identical panels). Coverage is nominal for all three; on" "\n  "
        r"clean panels they agree and the MDS gap is $\approx 0$; the gap and the width" "\n  "
        r"ordering flip only in the \emph{realistic} (non-stationary $+$ unbalanced) design," "\n  "
        r"where the bootstrap sides with $\OmegaDirect$.",
        "lcccccccc",
        r"    & \multicolumn{3}{c}{joint coverage} & \multicolumn{3}{c}{mean width}" "\n"
        r"    & \multicolumn{2}{c}{projection gap}\\" "\n"
        r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}" "\n"
        r"    design & \textsc{mds} & $\OmegaDirect$ & \textsc{cbb} & \textsc{mds} & $\OmegaDirect$" "\n"
        r"    & \textsc{cbb} & \textsc{mds} & $\OmegaDirect$\\", main_rows))

    sweep = [{**MC_BASE, "name": f"vol{v}", "imbalance": 0.4, "n_vol_episodes": 5,
              "vol_scale": float(v)} for v in MC_VOL_GRID]
    rs = coverage_study(sweep, R=R, B=B, alpha=alpha, methods=("mds", "omega"),
                        progress=True)
    rows = []
    for v in MC_VOL_GRID:
        s = rs[rs["name"] == f"vol{v}"].set_index("method")
        rows.append(f"    {v} & " + " & ".join(
            f"{s.loc[m, c]:.3f}" for c in ("joint_coverage", "mean_width", "proj_gap_mean")
            for m in ("mds", "omega")) + r"\\")
    two = r"\textsc{mds} & $\OmegaDirect$"
    _write(out_dir, "app_mc_crossover", _table(
        "tab:app-mc-crossover", "Crossover sweep, full results",
        r"The crossover sweep of Table~\ref{tab:mc-crossover} with the coverage and the "
        r"projection distance of both routes (imbalance $0.4$, five volatility episodes "
        r"of growing scale, $R=500$, $B=2000$, $\alpha=0.05$).",
        "ccccccc",
        r"    & \multicolumn{2}{c}{joint coverage} & \multicolumn{2}{c}{mean width}"
        r" & \multicolumn{2}{c}{projection distance}\\" "\n"
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
        f"    \\texttt{{vol\\_scale}} & {two} & {two} & {two}\\\\", rows))

    # Table 7.2 (tab:mc-crossover): widths and the MDS gap of the same sweep;
    # bold marks the direct construction once it is the narrower one
    at = lambda v, m, c: rs.loc[(rs["name"] == f"vol{v}") & (rs["method"] == m), c].item()
    w_om = [f"{at(v, 'omega', 'mean_width'):.2f}" for v in MC_VOL_GRID]
    w_om = [rf"\textbf{{{w}}}" if at(v, "omega", "mean_width") < at(v, "mds", "mean_width")
            else w for v, w in zip(MC_VOL_GRID, w_om)]
    _write(out_dir, "mc_crossover", r"""\begin{table}[htb]
  \centering
  \caption[Width crossover]{Width crossover under growing non-stationarity (imbalance
  fixed at $0.4$; larger \texttt{vol\_scale} $=$ bigger crisis episodes). As the MDS
  projection gap grows, its rank sets overtake the direct construction's.}
  \label{tab:mc-crossover}
  \small
  \begin{tabular}{lccccccc}
    \toprule
    \texttt{vol\_scale} & """ + " & ".join(map(str, MC_VOL_GRID)) + r"""\\
    \midrule
    \textsc{mds} width          & """ + " & ".join(
        f"{at(v, 'mds', 'mean_width'):.2f}" for v in MC_VOL_GRID) + r"""\\
    $\OmegaDirect$ width        & """ + " & ".join(w_om) + r"""\\
    \textsc{mds} projection gap & """ + " & ".join(
        f"{at(v, 'mds', 'proj_gap_mean'):.2f}" for v in MC_VOL_GRID) + r"""\\
    \bottomrule
  \end{tabular}
\end{table}
""")


def _macro_block(title, F, alpha=0.2, B=5000):
    """Rows of one macro panel: ID, observations, mean loss, direct and MDS rank sets."""
    X = F.values
    ci = _rank_sets(X, alpha, B)
    theta, nobs = np.nanmean(X, axis=0), np.isfinite(X).sum(axis=0)
    rows = [r"    \addlinespace" + "\n" + f"    \\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}}\\\\"]
    for j in np.argsort(theta):
        rows.append(f"    {int(F.columns[j])} & {nobs[j]} & {_num(theta[j])} & "
                    f"{_set(ci['omega'][j])} & {_set(ci['mds'][j])}\\\\")
    return rows


def tab_app_macro(out_dir):
    """Every rank set behind tab:spf-indicators and sec:apps-ecb."""
    header = r"    ID & obs. & mean loss & $\OmegaDirect$ & \textsc{mds}\\"
    rows = []
    for ind, title in (("NGDP", "nominal output (NGDP)"), ("RGDP", "real output (RGDP)"),
                       ("UNEMP", "unemployment rate (UNEMP)")):
        F = spf_frame(ind)
        rows += _macro_block(f"{title}, {F.shape[0]} quarters", F)
    rows[0] = rows[0].split("\n", 1)[1]                     # no space above the first block
    _write(out_dir, "app_spf_sets", _table(
        "tab:app-spf-sets", "U.S. SPF: all rank sets",
        r"Rank confidence sets of the eight best-covered U.S.\ SPF forecasters, the full "
        r"sets behind Table~\ref{tab:spf-indicators} (one-quarter-ahead squared error, "
        r"$\alpha=0.2$, $B=5000$). Forecasters are sorted by mean loss; ``obs.''\ is the "
        r"number of quarters in which a forecaster reported. The loss is in squared "
        r"billions of dollars for NGDP and RGDP and in squared percentage points for UNEMP.",
        "rrrcc", header, rows))
    rows = []
    for ind, title in (("HICP", "HICP inflation"), ("RGDP", "real GDP growth")):
        F = ecb_frame(ind)
        rows += _macro_block(f"{title}, target years {F.index.min()}--{F.index.max()}", F)
    rows[0] = rows[0].split("\n", 1)[1]
    _write(out_dir, "app_ecb_sets", _table(
        "tab:app-ecb-sets", "Euro-area SPF: all rank sets",
        r"Rank confidence sets of the eight best-covered euro-area SPF forecasters "
        r"(\S\ref{sec:apps-ecb}; current-year forecast from the first-quarter round, "
        r"squared error in percentage points, $\alpha=0.2$, $B=5000$). Forecasters are "
        r"sorted by mean loss; ``obs.''\ is the number of target years covered.",
        "rrrcc", header, rows))


def tab_app_gjp(out_dir):
    """All twenty GJP rank sets, with the split-half ranks of sec:apps-gjp."""
    P = gjp_frame()
    X = P.values
    ci = _rank_sets(X, 0.2, 5000)
    theta, nobs = np.nanmean(X, axis=0), np.isfinite(X).sum(axis=0)
    halves = [np.nanmean(P.iloc[k::2].values, axis=0) for k in (0, 1)]
    half_rank = [np.argsort(np.argsort(h)) + 1 for h in halves]
    rows = []
    for r, j in enumerate(np.argsort(theta), start=1):
        rows.append(f"    {r} & {int(P.columns[j])} & {nobs[j]} & {theta[j]:.3f} & "
                    f"{_set(ci['omega'][j])} & {_set(ci['mds'][j])} & "
                    f"{half_rank[0][j]} & {half_rank[1][j]}\\\\")
    rho = np.corrcoef(*half_rank)[0, 1]
    _write(out_dir, "app_gjp", _table(
        "tab:app-gjp", "GJP: all rank sets",
        r"Rank confidence sets of the twenty most active GJP forecasters, the sets "
        r"plotted in Figure~\ref{fig:gjp-caterpillar} (" + f"{X.shape[0]}" + r" questions, "
        r"daily-averaged Brier score, $\alpha=0.2$, $B=5000$). ``Questions'' is the number "
        r"of questions a forecaster answered. The last two columns rank the forecasters by "
        r"mean Brier score on the odd- and even-numbered questions in order of closing "
        r"date; their Spearman correlation is " + f"{rho:.2f}" + ".",
        "rrrrccrr",
        r"    & & & & & & \multicolumn{2}{c}{split-half rank}\\"
        "\n" r"    \cmidrule(lr){7-8}" "\n"
        r"    rank & forecaster & questions & mean Brier & $\OmegaDirect$ & \textsc{mds}"
        r" & odd & even\\", rows))


def tab_app_m3(out_dir):
    """M3 Paris record: block-level sMAPEs and robustness of the rank sets."""
    from rankci import rank_ci_stepwise_simulation_pairwise as sim
    from rankci.data.mcomp import official_panel_from_long
    long = m3_official_long()
    P = official_panel_from_long(long, PARIS)
    X, methods = P.values, list(P.columns)
    theta = X.mean(axis=0)
    order = np.argsort(theta)

    # block-level sMAPE; * marks a block in which the method's forecast is constant
    per_block = P.groupby(level="series", sort=False).mean()
    paris = long[long["description"] == PARIS]
    flat = paris.groupby("series", sort=False)[methods].std() == 0
    show = ["B-J auto", "THETA", "PP-Autocast", "ROBUST-Trend", "Flors-Pearc1",
            "Flors-Pearc2"]
    two_line = {"B-J auto": r"B-J\\automatic", "THETA": "Theta",
                "PP-Autocast": r"PP-\\Autocast", "ROBUST-Trend": r"Robust-\\Trend",
                "Flors-Pearc1": r"Flores-\\Pearce1", "Flors-Pearc2": r"Flores-\\Pearce2"}
    rows = []
    for i, s in enumerate(per_block.index):
        start = 1857 + 8 * i                                  # consecutive 8-year blocks
        cells = [f"{per_block.loc[s, m]:.1f}" + (r"$^{\ast}$" if flat.loc[s, m] else "")
                 for m in show]
        rows.append(f"    {start}--{start + 7} & " + " & ".join(cells) + r"\\")
    rows.append(r"    \midrule" + "\n    mean & " + " & ".join(
        f"{theta[methods.index(m)]:.1f}" for m in show) + r"\\")
    _write(out_dir, "app_m3_blocks", _table(
        "tab:app-m3-blocks", "M3 Paris record: sMAPE by block",
        r"sMAPE of six M3 submissions in each of the seventeen eight-year blocks of the "
        r"Paris temperature record (mean over the 18 held-out months). A star marks a "
        r"block in which the method forecast the same value for all 18 months. The last "
        r"row is the mean sMAPE of Table~\ref{tab:mcomp}.",
        "lcccccc",
        "    block & " + " & ".join(rf"\shortstack{{{two_line[m]}}}" for m in show)
        + r"\\", rows))

    # robustness: alpha = 0.05, and one observation per block
    run = lambda Z, a: sim(Z, alpha=a, B=20000, seed=SEED, covariance="omega",
                           verbose=False)["rank_ci"]
    base, tight, blocks = run(X, 0.1), run(X, 0.05), run(per_block.values, 0.1)
    rows = [f"    {M3_NAMES[methods[j]][0]} & {theta[j]:.2f} & {_set(base[j])} & "
            f"{_set(tight[j])} & {_set(blocks[j])}\\\\" for j in order]
    _write(out_dir, "app_m3_robust", _table(
        "tab:app-m3-robust", "M3 Paris record: robustness of the rank sets",
        r"Rank confidence sets of the 24 M3 submissions on the Paris record under the "
        r"direct construction ($B=20000$): the sets of Table~\ref{tab:mcomp} "
        r"($\alpha=0.1$, $n=306$), the same panel at $\alpha=0.05$, and the panel of "
        r"block averages ($\alpha=0.1$, $n=17$), in which each block's loss is averaged "
        r"over its 18 held-out months.",
        "lrccc",
        r"    method & mean sMAPE & $\alpha=0.1$ & $\alpha=0.05$ & block averages\\", rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", default=str(OUT), help="output directory (default: thesis/tables)")
    ap.add_argument("--skip-mc", action="store_true",
                    help="skip the Monte Carlo tables of Appendix B (the slow part)")
    args = ap.parse_args()
    out_dir = args.out
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tab_spf_indicators(out_dir)
    tab_ngdp(out_dir)
    tab_three_regimes(out_dir)
    ecb_numbers()
    block_bootstrap_numbers()
    tab_app_macro(out_dir)
    tab_app_gjp(out_dir)
    tab_app_m3(out_dir)
    if not args.skip_mc:
        tab_app_mc(out_dir)
    print("done ->", out_dir)
