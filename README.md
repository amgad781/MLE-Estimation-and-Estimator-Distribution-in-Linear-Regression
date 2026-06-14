# MLE Estimation and Estimator Distribution in Linear Regression

**Course:** CIE 457 — Statistical Inference and Data Analysis  
**Institution:** University of Science and Technology, Zewail City  
**Semester:** Spring 2026  
**Author:** Ahmed Amgad (202200393)

---

## Overview

This project investigates Maximum Likelihood Estimation (MLE) in linear regression, covering both the theoretical derivation of estimators and their empirical behaviour under repeated sampling. The work is structured around five core tasks and an optional bonus section, progressing from first-principles likelihood derivations through Monte Carlo simulation to a real-data application on U.S. wage data.

The central question is: given that the OLS estimator is a random variable, what does its distribution look like — and how does it change under different noise structures, sample sizes, and feature correlation regimes?

---

## Repository Structure

```
.
├── CIE457_MLE_Notebook.ipynb          # Full analysis notebook (Tasks 1–5 + Bonus)
├── Statistical_Inference_MLE_Project_Report.pdf   # Formal written report
└── README.md
```

The notebook and report are designed to be read together. Every equation in the report is cross-referenced to the corresponding function or code block in the notebook.

---

## Project Structure

### Task 1 — Likelihood and the Linear Model

Derives the conditional likelihood for the linear model `y = Xβ + ε` under two noise assumptions:

- **Homoscedastic noise** `ε ~ N(0, σ²I)`: maximising the Gaussian log-likelihood reduces to minimising the squared-error term, which yields the familiar OLS closed-form solution `β_OLS = (XᵀX)⁻¹Xᵀy`. This establishes OLS as the MLE under constant variance.
- **Heteroscedastic noise** `ε ~ N(0, Σ)`: the log-likelihood involves the full covariance matrix, and maximisation produces the Weighted Least Squares estimator `β_WLS = (XᵀΣ⁻¹X)⁻¹XᵀΣ⁻¹y`, which down-weights unreliable observations proportionally to their variance.

The derivations proceed from the factored joint pdf `f(x,y) = f(y|x) · f(x)`, retaining only the conditional likelihood since the marginal distribution of `x` carries no information about `β`.

---

### Task 2 — The Estimator as a Random Variable

Establishes the full statistical characterisation of `β_hat`:

| Property | Result |
|---|---|
| Expansion | `β_hat = β + (XᵀX)⁻¹Xᵀε` |
| Mean | `E[β_hat] = β` (unbiased) |
| Covariance | `Cov(β_hat) = σ²(XᵀX)⁻¹` |
| Distribution | `β_hat ~ N(β, σ²(XᵀX)⁻¹)` |

Key insight: `β_hat` is random because it is a linear function of the noise `ε`, not because the true parameter `β` varies. The spread of `β_hat` across repeated samples is entirely governed by `σ²(XᵀX)⁻¹` — larger noise or a poorly conditioned design matrix both inflate estimator variance.

---

### Task 3 — Estimator Distribution and Simulation

Empirically validates the theoretical distribution using **B = 2,000 Monte Carlo replications** with N = 500 observations, β = (2.0, 5.0, −3.0)ᵀ, and σ² = 4. The design matrix is held fixed across replications; only the noise is resampled.

Three experiments are run:

**Homoscedastic, independent features**  
Empirical means land within 0.03% of the true values. Empirical variances match the theoretical `σ²(XᵀX)⁻¹` entries to within Monte Carlo noise. The histogram shape is indistinguishable from the theoretical Gaussian.

**Correlated features (r = 0.784)**  
Feature correlation inflates `(XᵀX)⁻¹`, increasing slope variances by 2.60× and 3.11× while leaving the intercept variance essentially unchanged (1.00×). The estimator remains unbiased; only its efficiency is reduced.

| Parameter | Var (Independent) | Var (Correlated) | Inflation |
|---|---|---|---|
| β_0 | 0.008031 | 0.008004 | 1.00× |
| β_1 | 0.000244 | 0.000633 | 2.60× |
| β_2 | 0.000309 | 0.000961 | 3.11× |

**Heteroscedastic noise (σ²_i = 1 + 0.5 x²_i,1), OLS vs. WLS**  
Both estimators remain unbiased under heteroscedastic noise. WLS achieves 1.4–3.3× lower variance by correctly weighting each observation by the inverse of its noise variance. OLS pays a provable variance penalty for ignoring the noise structure.

| Parameter | Var(OLS) | Var(WLS) | Efficiency Ratio |
|---|---|---|---|
| β_0 | 0.03365 | 0.01046 | 3.22× |
| β_1 | 0.00178 | 0.00129 | 1.38× |
| β_2 | 0.00134 | 0.00041 | 3.28× |

---

### Task 4 — Inference: Confidence and Prediction Intervals

Compares 95% confidence intervals (CI) and prediction intervals (PI) across sample sizes N = 500 and N = 30.

The structural difference between the two interval types:

```
CI:  x*ᵀβ_hat  ±  z_{0.975} · sqrt( σ_hat² · x*ᵀ(XᵀX)⁻¹x* )
PI:  x*ᵀβ_hat  ±  z_{0.975} · sqrt( σ_hat² · (1 + x*ᵀ(XᵀX)⁻¹x*) )
```

The `+1` inside the PI square root represents the irreducible observation noise. As a result:

- The **CI** targets the true mean `E[y | x*]` and shrinks to zero as N grows.
- The **PI** targets a new observation `y* = x*ᵀβ + ε*` and converges to a noise floor of `2 × z_{0.975} × σ ≈ 7.84` regardless of sample size.

At N = 500 the PI width for β_1 is 8.015; at N = 30 it is 7.660 — a change of less than 5%. The corresponding CI widths are 0.063 and 0.322, a 5× increase.

---

### Task 5 — Real Data Application: Wage Dataset

**Dataset:** Wooldridge `wage1` — 526 U.S. workers (1976 Current Population Survey).  
**Model:** `wage_i = β_0 + β_1·educ_i + β_2·exper_i + β_3·tenure_i + ε_i`

The dataset is preferred over single-predictor alternatives because heteroscedasticity is a genuine empirical finding rather than an imposed condition, and because the OLS vs. WLS comparison carries a substantive economic interpretation.

**Heteroscedasticity detection:** Both the Breusch–Pagan test (LM = 43.10, p < 0.001) and White's test (LM = 63.96, p < 0.001) reject homoscedasticity. The OLS residual-vs-fitted plot shows a clear funnel pattern — spread growing with fitted wage — confirming that higher-educated workers enter more variable occupations.

**OLS vs. WLS coefficients:**

| Parameter | OLS Coef | OLS SE | WLS Coef | WLS SE |
|---|---|---|---|---|
| Intercept | −2.8727 | 0.7290 | 0.4363 | 0.1909 |
| Education | 0.5990 | 0.0513 | 0.3125 | 0.0146 |
| Experience | 0.0223 | 0.0121 | 0.0381 | 0.0049 |
| Tenure | 0.1693 | 0.0216 | 0.1216 | 0.0094 |

The education coefficient falls from 0.599 (OLS) to 0.313 (WLS) — nearly half. OLS assigns equal weight to every worker, including high-earning professionals with the most education and the most variable wages. WLS down-weights these noisy observations, shifting influence to the lower-variance majority. The WLS estimate is the more reliable measure of the return to an additional year of education for the average worker.

---

### Bonus — Fisher Information, CRLB, and Sufficient Statistics

**Fisher Information Matrix**

The Fisher information measures the sharpness of the likelihood peak around the true `β`:

- Homoscedastic: `I(β) = (1/σ²) · XᵀX`
- Heteroscedastic: `I(β) = XᵀΣ⁻¹X`

**Cramér–Rao Lower Bound**

The CRLB states that no unbiased estimator can achieve a covariance matrix smaller (in the positive semi-definite sense) than `I(β)⁻¹`. Both OLS (under homoscedastic noise) and WLS (under heteroscedastic noise) achieve this bound exactly, confirming they are Minimum Variance Unbiased Estimators (MVUE).

OLS under heteroscedastic noise exceeds the CRLB by 1.4–3.4×, precisely quantifying the efficiency loss from noise-structure misspecification.

**Sufficient Statistics and Rao–Blackwell**

By the Fisher–Neyman Factorisation Theorem, the homoscedastic likelihood factors through `T(y) = (Xᵀy, ‖y‖²)`. Since `β_OLS = (XᵀX)⁻¹Xᵀy` is a one-to-one function of `Xᵀy`, it is itself sufficient for `β`. By the Lehmann–Scheffé theorem, a complete sufficient statistic that produces an unbiased estimator is the unique MVUE — no other unbiased estimator, linear or nonlinear, can achieve lower variance under Gaussian noise.

---

## Dependencies

```
numpy
scipy
matplotlib
statsmodels
wooldridge
```

Install all dependencies with:

```bash
pip install numpy scipy matplotlib statsmodels wooldridge
```

The notebook was developed and tested on Python 3.10. No GPU is required.

---

## How to Run

1. Clone or download the repository.
2. Install dependencies (see above).
3. Open `CIE457_MLE_Notebook.ipynb` in Jupyter Lab or Jupyter Notebook.
4. Run all cells in order. The notebook is self-contained; all data is loaded programmatically via the `wooldridge` package.

All random operations use a fixed seed (`np.random.default_rng(42)`) for full reproducibility.

---

## Key Concepts Reference

| Term | Definition |
|---|---|
| MLE | Maximum Likelihood Estimation — finds the parameter value that maximises the probability of the observed data |
| OLS | Ordinary Least Squares — MLE under homoscedastic Gaussian noise |
| WLS | Weighted Least Squares — MLE under heteroscedastic Gaussian noise |
| MVUE | Minimum Variance Unbiased Estimator — the most efficient unbiased estimator possible |
| CRLB | Cramér–Rao Lower Bound — the theoretical floor on estimator variance |
| Heteroscedasticity | Non-constant error variance across observations |
| Multicollinearity | Linear dependence among predictors, inflating estimator variance |
| Fisher Information | Measure of the amount of information the data carries about a parameter |
| Sufficient Statistic | A statistic that retains all information in the data relevant to estimating a parameter |

---

## Author

| Name | Student ID |
|---|---|
| Mohammed Ali | 202200594 |
