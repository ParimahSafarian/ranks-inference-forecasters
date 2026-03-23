# Rank Confidence Intervals for Forecaster Evaluation
### An Empirical Application of Mogstad, Romano, Shaikh & Wilhelm (2024)

This repository implements the rank confidence interval methodology from Mogstad et al. (2024, *Review of Economic Studies*) and applies it to the **Survey of Professional Forecasters (SPF)** dataset to rank professional forecasters by their mean squared prediction error for nominal GDP (NGDP) three-quarter-ahead forecasts.

---

## Overview

A common problem in empirical economics is that rankings based on point estimates do not convey the statistical uncertainty around those ranks. This project implements the framework of Mogstad et al. (2024), which constructs valid confidence sets for ranks by inverting simultaneous pairwise hypothesis tests — rather than naively ranking point estimates.

Three variants of the procedure are implemented: a **simultaneous bootstrap**, an **asymptotic simulation**, and a **stepwise bootstrap** refinement.

---

## Repository Structure

```
├── data/
│   └── SPFmicrodata.xlsx        # Survey of Professional Forecasters panel data
├── notebooks/
│   ├── EDA.ipynb                # Exploratory data analysis: forecaster participation, panel structure
│   ├── 01_Analysis.ipynb        # Main analysis: all three CI methods and comparisons
│   ├── Bootstrap.py             # Simultaneous bootstrap rank CI (Algorithm 3.1)
│   ├── Simulation.py            # Asymptotic simulation rank CI
│   └── Stepwise_bootstrap.py   # Stepwise bootstrap rank CI (Algorithm 3.2)
└── README.md
```

---

## Data

**Source:** Survey of Professional Forecasters (SPF), Federal Reserve Bank of Philadelphia.

- 462 unique forecasters observed over quarterly periods
- Forecast variable: **NGDP3** — nominal GDP, three-quarter-ahead forecast
- Forecaster quality is measured by **mean squared error (MSE)** relative to realised values
- Due to unbalanced panel participation, a complete-case analysis requires restricting to forecasters who overlap in the same time periods. After scanning group sizes, **the top 8 most frequently participating forecasters** are retained, yielding a balanced panel of **22 quarterly observations**

---

## Methods

All three methods share the same core logic: construct simultaneous confidence intervals for all pairwise mean differences, then convert these to rank confidence intervals using the N⁻/N⁺ counting rule:

- **rank lower bound** = (number of forecasters conclusively better than j) + 1  
- **rank upper bound** = p − (number of forecasters conclusively worse than j)

### 1. Simultaneous Bootstrap (`Bootstrap.py`)

Resamples rows of the data matrix jointly (preserving cross-sectional dependence) across `B = 20,000` bootstrap draws. A single critical value is derived from the maximum studentised pairwise statistic. All pairwise confidence intervals are constructed at this common critical value.

### 2. Asymptotic Simulation (`Simulation.py`)

Replaces bootstrap resampling with draws from a fitted multivariate normal distribution `N(0, Σ̂)`, where `Σ̂` is the estimated covariance of the sample mean vector divided by `n`. This is the asymptotic analogue of the bootstrap. Results are near-identical to the bootstrap, which serves as a useful consistency check.

### 3. Stepwise Bootstrap (`Stepwise_bootstrap.py`)

Implements Algorithm 3.2 of Mogstad et al. (2024). At each iteration, pairwise comparisons that can be conclusively rejected are removed from the active set, and the bootstrap critical value is recomputed over the remaining (unrejected) pairs only. This yields a strictly smaller critical value at each step, producing **tighter rank intervals** without inflating the error rate.

---

## Results

All results are at the **95% confidence level** (`α = 0.05`). Forecasters are ranked from highest MSE (rank 1 = worst) to lowest MSE (rank 8 = best). Bootstrap draws: `B = 20,000` (simultaneous and simulation) and `B = 5,000` (stepwise).

### Simultaneous Bootstrap vs. Asymptotic Simulation

| Est. Rank | Forecaster ID | Mean MSE     | CI Bootstrap | CI Simulation |
|:---------:|:-------------:|:------------:|:------------:|:-------------:|
| 1         | 65            | 331,218.5    | [1, 7]       | [1, 3]        |
| 2         | 426           | 319,657.3    | [1, 7]       | [1, 5]        |
| 3         | 411           | 310,621.2    | [1, 7]       | [1, 6]        |
| 4         | 433           | 303,701.3    | [1, 8]       | [2, 8]        |
| 5         | 421           | 293,552.7    | [1, 8]       | [2, 8]        |
| 6         | 428           | 291,917.6    | [1, 8]       | [3, 8]        |
| 7         | 84            | 289,922.3    | [1, 8]       | [4, 8]        |
| 8         | 40            | 282,969.8    | [4, 8]       | [4, 8]        |

Bootstrap critical value: **4.066** | Simulation critical value: **2.991**

### Simultaneous Bootstrap vs. Stepwise Bootstrap

| Est. Rank | Forecaster ID | Mean MSE     | CI Bootstrap | CI Stepwise  |
|:---------:|:-------------:|:------------:|:------------:|:------------:|
| 1         | 65            | 331,218.5    | [1, 7]       | [1, 3]       |
| 2         | 426           | 319,657.3    | [1, 7]       | [1, 5]       |
| 3         | 411           | 310,621.2    | [1, 7]       | [1, 6]       |
| 4         | 433           | 303,701.3    | [1, 8]       | [2, 7]       |
| 5         | 421           | 293,552.7    | [1, 8]       | [2, 8]       |
| 6         | 428           | 291,917.6    | [1, 8]       | [3, 8]       |
| 7         | 84            | 289,922.3    | [1, 8]       | [4, 8]       |
| 8         | 40            | 282,969.8    | [4, 8]       | [5, 8]       |

### Key Takeaways

- The **simultaneous bootstrap** produces wide intervals (e.g., [1, 7] or [1, 8] for most middle-ranked forecasters), reflecting the limited sample size of 22 periods. It is the most conservative of the three approaches.
- The **asymptotic simulation** yields somewhat tighter intervals, particularly at the extremes, but relies on the normality approximation being accurate in small samples.
- The **stepwise bootstrap** produces the tightest intervals across all methods. For example, the top-ranked forecaster's interval shrinks from [1, 7] to [1, 3], and the bottom-ranked from [4, 8] to [5, 8]. This demonstrates the practical value of the stepwise refinement even in small samples.
- Despite large differences in point estimates (MSE ranging from ~283,000 to ~331,000), the **high degree of overlap in the confidence intervals** reflects genuine statistical uncertainty — the sample of 22 periods is insufficient to precisely resolve the middle ranks.

---

## Reference

Mogstad, M., Romano, J. P., Shaikh, A., & Wilhelm, D. (2024). Inference for Ranks with Applications to Mobility Across Neighbourhoods and Academic Achievement Across Countries. *Review of Economic Studies*, 91(1), 476–518.

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
openpyxl
```
