"""
Regenerate the thesis figures into thesis/figures/*.pdf.

The thesis/ folder holds only the document (LaTeX sources and the rendered
figures); all code that produces its figures lives here.

Run from anywhere:  <repo>/.venv/bin/python scripts/make_thesis_figures.py
Figures: mc_crossover, gjp_caterpillar, mcomp_caterpillar.
(spf_dep_* come from notebooks/02_primary_analysis.ipynb.)
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]      # repo root
DATA = ROOT / "data"
OUT = ROOT / "thesis" / "figures"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 120})


def fig_mc_crossover():
    from rankci.sim import coverage_study
    vol_grid = [1, 3, 6, 10, 16, 24]
    base = dict(theta=list(np.linspace(0, 1.6, 5)), T=200, rho=0.3, sigma_c=1.0)
    designs = [{**base, "name": f"vol{v}", "imbalance": 0.4,
                "n_vol_episodes": 5, "vol_scale": float(v)} for v in vol_grid]
    res = coverage_study(designs, R=500, B=2000, alpha=0.05, base_seed=0,
                         methods=("mds", "omega"))
    sm = res[res.method == "mds"].reset_index(drop=True)
    so = res[res.method == "omega"].reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax[0].plot(vol_grid, sm["mean_width"], "o-", color="C3", label="MDS")
    ax[0].plot(vol_grid, so["mean_width"], "s-", color="C0", label=r"$\Omega$")
    ax[0].set_xlabel("crisis magnitude (vol\\_scale)"); ax[0].set_ylabel("mean rank-CI width")
    ax[0].set_title("width vs.\\ misspecification"); ax[0].legend(frameon=False)
    ax[1].plot(vol_grid, sm["proj_gap_mean"], "o-", color="C3", label="MDS")
    ax[1].plot(vol_grid, so["proj_gap_mean"], "s-", color="C0", label=r"$\Omega$")
    ax[1].set_yscale("log"); ax[1].set_xlabel("crisis magnitude (vol\\_scale)")
    ax[1].set_ylabel("projection gap"); ax[1].set_title("PSD projection distance")
    ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "mc_crossover.pdf", bbox_inches="tight")
    plt.close(fig); print("wrote mc_crossover.pdf")


def _caterpillar(theta, rank_ci, labels, title, fname, figsize=(5.2, 4.2)):
    order = np.argsort(theta)
    ci = rank_ci[order]
    yy = np.arange(len(theta))
    fig, ax = plt.subplots(figsize=figsize)
    ax.hlines(yy, ci[:, 0], ci[:, 1], color="C0", lw=3, alpha=0.55)
    ax.plot(np.arange(1, len(theta) + 1), yy, "o", color="black", ms=4)
    ax.set_yticks(yy); ax.set_yticklabels([labels[i] for i in order], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("rank (1 = best)")
    ax.set_xticks(range(1, len(theta) + 1)); ax.set_title(title)
    ax.tick_params(axis="x", labelsize=7 if len(theta) > 12 else 10)
    fig.tight_layout(); fig.savefig(OUT / fname, bbox_inches="tight")
    plt.close(fig); print("wrote", fname)


def fig_gjp_caterpillar():
    from rankci.data.gjp import load_gjp_ifps, load_gjp_forecasts, brier_panel
    from rankci import rank_ci_stepwise_simulation_pairwise as sim
    ifps = load_gjp_ifps(DATA / "gjp" / "ifps.csv")
    fc = load_gjp_forecasts([DATA / "gjp" / "survey_fcasts.yr1.csv",
                             DATA / "gjp" / "survey_fcasts.yr2.csv"])
    panel, _ = brier_panel(fc, ifps, condition="individual",
                           min_questions=60, top_n=20, years=[1, 2])
    X = panel.values
    out = sim(X, alpha=0.2, B=5000, seed=42, covariance="omega", verbose=False)
    _caterpillar(X.mean(0), out["rank_ci"], [f"user {u}" for u in panel.columns],
                 "GJP forecasters (212 questions)", "gjp_caterpillar.pdf")


def fig_mcomp_caterpillar():
    from rankci.data.mcomp import official_panel
    from rankci import rank_ci_stepwise_simulation_pairwise as sim
    panel = official_panel("Average temperature in Paris", directory=str(DATA / "mcomp"))
    X = panel.values
    out = sim(X, alpha=0.1, B=20000, seed=42, covariance="omega", verbose=False)
    _caterpillar(X.mean(0), out["rank_ci"], list(panel.columns),
                 "M3 methods, Paris monthly temperature (17 blocks x 18 months)",
                 "mcomp_caterpillar.pdf", figsize=(6.0, 4.6))


if __name__ == "__main__":
    fig_mc_crossover()
    fig_gjp_caterpillar()
    fig_mcomp_caterpillar()
    print("done ->", OUT)
