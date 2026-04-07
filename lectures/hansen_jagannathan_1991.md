---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(hansen_jagannathan_1991)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# The Hansen-Jagannathan Bound

```{contents} Contents
:depth: 2
```

## Overview

This lecture is based on {cite}`Hansen_Jagannathan_1991`, a paper by Lars Peter
Hansen and Ravi Jagannathan published in the *Journal of Political Economy* in
1991.

The paper asks: what does security market data tell us about the intertemporal
marginal rates of substitution (IMRSs) of consumers, without committing to a
specific parametric model?

In a broad class of dynamic equilibrium models, the price of any traded security
satisfies

$$
\pi(p) = E(p \, m \mid I),
$$

where $m$ is the IMRS of any consumer and $I$ is the traders' common
information set. This pricing equation must hold for **every** model in the
class, regardless of how heterogeneous preferences and markets are.

Hansen and Jagannathan exploit this common implication to derive
**volatility bounds** — lower bounds on the standard deviation of $m$ —
directly from asset payoff and price data. Their key results are:

- **Without the positivity restriction**: the minimum standard deviation of any $m$
  consistent with the pricing equation is the absolute value of the **Sharpe
  ratio** of a portfolio on the mean-variance frontier of asset returns.  This
  gives a hyperbola in mean-standard deviation space.

- **With the positivity restriction** ($m \geq 0$, which rules out arbitrage):
  the bound is tighter and defines a convex region $S^+$ whose boundary involves
  **option payoffs** (truncated linear combinations of asset payoffs).

- These bounds provide a nonparametric diagnostic for the **equity premium
  puzzle** ({cite}`MehraPrescott1985`): the CRRA representative-consumer model
  requires an implausibly high risk aversion to match the observed Sharpe ratio
  of equity returns.

Among the things we will learn are:

- How to derive and compute the linear (no-positivity) HJ bound from sample
  moments of asset payoffs and prices.
- The duality between the SDF frontier and the mean-variance frontier for
  asset returns.
- How to tighten the bound using the positivity of $m$.
- How to evaluate parametric asset pricing models against the bound.
- How to replicate the key figures of the paper using Python.

Let's start by importing the Python tools we need.

```{code-cell} ipython3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

np.random.seed(0)
```

## The Asset Pricing Framework

### General model

Consider an economy in which multiple consumers (possibly with heterogeneous
preferences and information sets) trade a vector $x$ of $n$ asset payoffs at
date $T$.  Let $q$ denote the $n \times 1$ vector of prices at date 0.

For any consumer $j$ with IMRS $m^j$,

$$
q = E\!\left(x \, m^j \mid I^j\right).
$$

Applying the law of iterated expectations and summing over consumers, this
implies —  for **any** consumer's IMRS $m$ and the common information set
$I = \cap_j I^j$ — the pricing relation

$$
q = E(xm \mid I).
$$

Taking unconditional expectations of both sides gives:

**Restriction 1** (Pricing restriction):

$$
Eq = E(xm).
$$

**Restriction 2** (Positivity):

$$
m > 0.
$$

Restriction 1 must hold in any model consistent with consumer optimality.
Restriction 2 rules out arbitrage opportunities.  Together, they imply that
$[E(m), \sigma(m)]$ must lie in a certain admissible region in the
mean-standard deviation plane.

### Sample moments and population moments

Under ergodicity, the time-series averages

$$
\hat{E}(x) = \frac{1}{T}\sum_{t=1}^T x_t, \quad
\hat{E}(q) = \frac{1}{T}\sum_{t=1}^T q_t, \quad
\widehat{\mathrm{Cov}}(x) = \frac{1}{T}\sum_{t=1}^T (x_t - \hat{E}x)(x_t - \hat{E}x)^\top
$$

converge to their population counterparts.  In what follows we use population
moments (or simulated sample moments) interchangeably.

```{code-cell} ipython3
DATA_DIR = Path("_static/lecture_specific/hansen_jagannathan_1991")
if not DATA_DIR.exists():
    DATA_DIR = Path("lectures/_static/lecture_specific/hansen_jagannathan_1991")

BUNDLE_PATH = DATA_DIR / "hansen_jagannathan_1991_data.pkl"
DATA_BUNDLE = pd.read_pickle(BUNDLE_PATH)
DATA_SOURCES = DATA_BUNDLE["sources"].copy()


def compute_moments(returns, prices=None):
    """
    Compute mean returns, mean prices, and covariance matrix from data.

    Parameters
    ----------
    returns : array of shape (T, n)  — gross asset returns
    prices  : array of shape (T, n)  — asset prices (optional; defaults to 1
              if returns are already in units of gross return with price = 1)

    Returns
    -------
    mu_x  : mean payoffs  (n,)
    mu_q  : mean prices   (n,)
    Sigma : covariance matrix of payoffs  (n, n)
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns[:, None]

    mu_x = returns.mean(axis=0)
    Sigma = np.cov(returns.T, bias=True)
    if Sigma.ndim == 0:
        Sigma = np.array([[float(Sigma)]])

    if prices is None:
        mu_q = np.ones(returns.shape[1])
    else:
        prices = np.asarray(prices, dtype=float)
        if prices.ndim == 1:
            prices = prices[:, None]
        mu_q = prices.mean(axis=0)

    return mu_x, mu_q, Sigma
```

All processed panels used below are loaded from the single bundled data file
`hansen_jagannathan_1991_data.pkl`.

The raw sources used to build that bundle are recorded in the table below.

```{code-cell} ipython3
DATA_SOURCES
```

```{code-cell} ipython3
def positive_frontier_from_sample(payoffs, prices, v_grid, maxiter=2_000):
    """
    Exact sample analogue of the positivity-restricted frontier.

    Uses warm-starting and explicit Jacobians for robust convergence.
    """
    x = np.asarray(payoffs, dtype=float)
    q = np.asarray(prices, dtype=float)

    if x.ndim == 1:
        x = x[:, None]
    if q.ndim == 1:
        q = q[:, None]

    T = x.shape[0]
    mu_q = q.mean(axis=0)
    x_aug = np.column_stack([x, np.ones(T)])
    second_moment = x_aug.T @ x_aug / T
    pinv_second = np.linalg.pinv(second_moment)

    means = []
    sigmas = []
    w_prev = None

    for v in v_grid:
        q_aug = np.r_[mu_q, v]

        # Initial guess from unconstrained projection
        w0 = pinv_second @ q_aug
        scale = q_aug @ w0

        if abs(scale) < 1e-12:
            means.append(np.nan)
            sigmas.append(np.nan)
            continue

        w0 = w0 / scale

        # Collect candidate starting points (warm-start from previous solution)
        candidates = [w0]
        if w_prev is not None:
            s2 = q_aug @ w_prev
            if abs(s2) > 1e-12:
                candidates.append(w_prev / s2)

        best_obj = np.inf
        best_result = None

        for w_init in candidates:
            def objective(w):
                r = x_aug @ w
                return np.mean(np.maximum(r, 0.0) ** 2)

            def jac(w):
                r = x_aug @ w
                rp = np.maximum(r, 0.0)
                return 2.0 * (x_aug.T @ rp) / T

            result = minimize(
                objective,
                w_init,
                jac=jac,
                method="SLSQP",
                constraints=(
                    {
                        "type": "eq",
                        "fun": lambda w, qa=q_aug: qa @ w - 1.0,
                        "jac": lambda w, qa=q_aug: qa,
                    },
                ),
                options={"maxiter": maxiter, "ftol": 1e-14},
            )

            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result

        r_plus = np.maximum(x_aug @ best_result.x, 0.0)
        delta_v = np.mean(r_plus ** 2)

        if delta_v < 1e-14:
            means.append(np.nan)
            sigmas.append(np.nan)
            continue

        m = r_plus / delta_v
        means.append(m.mean())
        sigmas.append(m.std())
        w_prev = best_result.x.copy()

    return np.asarray(means), np.asarray(sigmas)


def crra_points_from_consumption(consumption, beta=0.95, gamma_grid=None):
    if gamma_grid is None:
        gamma_grid = np.arange(31)

    growth = np.asarray(consumption[1:] / consumption[:-1], dtype=float)
    means = []
    sigmas = []

    for gamma in gamma_grid:
        m = beta * growth ** (-gamma)
        means.append(m.mean())
        sigmas.append(m.std())

    return np.asarray(means), np.asarray(sigmas)


def load_annual_paper_data():
    data = DATA_BUNDLE["annual"].copy()
    return (
        data["year"].to_numpy(),
        data["stock"].to_numpy(),
        data["bond"].to_numpy(),
        data["consumption"].to_numpy(),
    )


def load_monthly_proxy_panel():
    return DATA_BUNDLE["monthly"].copy()


def build_monthly_payoff_menu(panel):
    z_bill = panel["bill"].shift(1)
    z_stock = panel["stock"].shift(1)
    z_cons = panel["cons_ratio"].shift(1)

    payoffs = pd.DataFrame(
        {
            "stock": panel["stock"],
            "bill": panel["bill"],
            "stock_x_billlag": panel["stock"] * z_bill,
            "bill_x_billlag": panel["bill"] * z_bill,
            "stock_x_stocklag": panel["stock"] * z_stock,
            "bill_x_stocklag": panel["bill"] * z_stock,
            "stock_x_conslag": panel["stock"] * z_cons,
            "bill_x_conslag": panel["bill"] * z_cons,
        }
    )

    prices = pd.DataFrame(
        {
            "stock": 1.0,
            "bill": 1.0,
            "stock_x_billlag": z_bill,
            "bill_x_billlag": z_bill,
            "stock_x_stocklag": z_stock,
            "bill_x_stocklag": z_stock,
            "stock_x_conslag": z_cons,
            "bill_x_conslag": z_cons,
        }
    )

    joined = pd.concat([payoffs, prices.add_suffix("_price")], axis=1).dropna()
    return joined[payoffs.columns].to_numpy(), joined[prices.columns].to_numpy()


def load_quarterly_bill_proxy():
    return DATA_BUNDLE["quarterly"].copy()
```

## The Linear Volatility Bound (Without Positivity)

### Constructing $m^*$

Suppose we only impose Restriction 1.  Among all random variables $m$
satisfying $Eq = E(xm)$, what is the minimum variance?

Hansen and Jagannathan show that the answer is the **minimum second-moment**
projection of $m$ onto the space $P = \{c^\top x : c \in \mathbb{R}^n\}$.
This projection, call it $m^*$, satisfies

$$
m^* = x^\top \alpha^*, \qquad \alpha^* = (Exx^\top)^{-1} Eq.
$$

Because $m^*$ is the projection of any valid $m$ onto $P$, and any valid $m$
differs from $m^*$ by a component orthogonal to $x$, we have the **variance
decomposition**

$$
\sigma^2(m) = \sigma^2(m^*) + \sigma^2(m - m^*) \geq \sigma^2(m^*).
$$

So $m^*$ achieves the minimum variance among all $m$'s satisfying Restriction 1
**with a given mean**.

### When there is a riskless asset

If the payoff vector $x$ includes a unit payoff (riskless bond), then every
valid $m$ must have the same mean, $Em = 1/r_f$, so the volatility bound is
simply

$$
\sigma(m) \geq \sigma(m^*).
$$

### When there is no riskless asset

When $x$ does not include a unit payoff, different valid $m$'s may have
different means.  For each hypothetical mean $v$ we augment $x$ with a unit
payoff assigned expected price $v$, and construct

$$
m^v = x_a^\top \alpha^v, \qquad \alpha^v = (Ex_a x_a^\top)^{-1} Eq_a,
$$

where $x_a = (x^\top, 1)^\top$ and $q_a = (q^\top, v)^\top$.  The bound is

$$
\sigma(m) \geq \sigma(m^v) =
\left[(Eq - v \, Ex)^\top \Sigma^{-1} (Eq - v \, Ex)\right]^{1/2},
$$

where $\Sigma = \mathrm{Cov}(x)$ is the covariance matrix of payoffs.

This formula is easy to compute: it requires only means of prices and payoffs
and the covariance matrix of payoffs.

```{code-cell} ipython3
def hj_bound_no_positivity(mu_x, mu_q, Sigma, v_grid=None):
    """
    Compute the Hansen-Jagannathan volatility bound WITHOUT the positivity
    restriction (the linear bound, Section III of the paper).

    sigma(m^v) = sqrt[ (Eq - v*Ex)' * Sigma^{-1} * (Eq - v*Ex) ]

    Parameters
    ----------
    mu_x   : mean payoffs  (n,)
    mu_q   : mean prices   (n,)
    Sigma  : covariance matrix of payoffs  (n, n)
    v_grid : array of candidate mean values for m (optional)

    Returns
    -------
    v_grid      : array of mean(m) values
    sigma_bound : array of minimum sigma(m) values
    """
    if v_grid is None:
        v_grid = np.linspace(0.85, 1.15, 300)

    inv_Sigma = np.linalg.pinv(Sigma)
    sigma_bound = np.array([
        np.sqrt(np.maximum((mu_q - v * mu_x) @ inv_Sigma @ (mu_q - v * mu_x), 0.0))
        for v in v_grid
    ])
    return v_grid, sigma_bound
```

### Duality with the mean-variance frontier for returns

A striking result in the paper is that the HJ bound is exactly the **absolute
value of the slope** of the mean-standard deviation frontier for asset returns.

To see why, define the set of returns $R = \{p \in P : \pi(p) = 1\}$ (unit
expected price).  The **Sharpe ratio** of any excess return $r^e = r - r_f$ is

$$
\text{SR}(r^e) = \frac{E(r^e)}{\sigma(r^e)}.
$$

The largest achievable Sharpe ratio — the slope of the capital market line —
equals

$$
\text{SR}_{\max} = \frac{\sigma(m^*)}{E(m^*)} = \frac{\sigma(m^*)}{1/r_f}.
$$

Therefore,

$$
\frac{\sigma(m)}{E(m)} \geq \frac{\sigma(m^*)}{E(m^*)} = \text{SR}_{\max}.
$$

This is the **Hansen-Jagannathan inequality**: any valid SDF must have a
coefficient of variation at least as large as the maximum Sharpe ratio of
traded assets.

```{code-cell} ipython3
def mean_variance_frontier(mu_x, Sigma, n_points=300):
    """
    Compute the mean-standard deviation frontier for asset returns.

    Uses the standard two-fund formula:
        sigma^2(r_c) = (A*c^2 - 2*B*c + C) / D
    where A = mu' Sigma^{-1} mu, B = mu' Sigma^{-1} 1,
          C = 1' Sigma^{-1} 1, D = A*C - B^2.

    Parameters
    ----------
    mu_x  : mean returns (n,)
    Sigma : covariance matrix (n, n)

    Returns
    -------
    frontier_means : array (n_points,)
    frontier_stds  : array (n_points,)
    """
    n = len(mu_x)
    inv_S = np.linalg.pinv(Sigma)
    ones  = np.ones(n)

    A = mu_x @ inv_S @ mu_x
    B = mu_x @ inv_S @ ones
    C = ones  @ inv_S @ ones
    D = A * C - B**2

    c_min = B / C                      # global minimum variance mean
    c_grid = np.linspace(c_min - 0.10, c_min + 0.15, n_points)

    var_c = (A * c_grid**2 - 2 * B * c_grid + C) / D
    std_c = np.sqrt(np.maximum(var_c, 0))
    return c_grid, std_c


def max_sharpe_ratio(mu_x, mu_q, Sigma, rf=None):
    """
    Compute the maximum Sharpe ratio achievable from the asset menu.

    If rf is given (risk-free rate), SR_max = max_w (w'mu_exc / sqrt(w'Sigma w)).
    Otherwise, use the slope of the tangent from origin to the frontier.
    """
    n = len(mu_x)
    inv_S = np.linalg.pinv(Sigma)

    if rf is not None:
        mu_exc = mu_x - rf
        # Tangency portfolio weights (unnormalised)
        w_tan = inv_S @ mu_exc
        sr_max = (mu_exc @ w_tan) / np.sqrt(w_tan @ Sigma @ w_tan)
    else:
        # No risk-free asset: slope of asymptote from (0, 1/v) to frontier
        ones = np.ones(n)
        A = mu_x @ inv_S @ mu_x
        B = mu_x @ inv_S @ ones
        C = ones  @ inv_S @ ones
        D = A * C - B**2
        # Slope of the asymptote of the parabola in (sigma, mu) space
        sr_max = np.sqrt(D / C)

    return float(sr_max)
```

## Computing the Annual Frontier

Figures 1 and 4 in the paper use the annual stock, bond, and consumption
series described in the paper's appendix.  The archived Campbell-Shiller /
Shiller workbook bundled with this lecture contains those series directly.

```{code-cell} ipython3
annual_years, annual_stock, annual_bond, annual_consumption = load_annual_paper_data()
annual_payoffs = np.column_stack([annual_stock, annual_bond])
annual_prices = np.ones_like(annual_payoffs)

mu_x_annual, mu_q_annual, Sigma_annual = compute_moments(annual_payoffs, annual_prices)
v_annual, sigma_annual = hj_bound_no_positivity(
    mu_x_annual, mu_q_annual, Sigma_annual, v_grid=np.linspace(0.84, 1.16, 400)
)

annual_gamma_grid = np.arange(31)
annual_crra_mean, annual_crra_std = crra_points_from_consumption(
    annual_consumption, beta=0.95, gamma_grid=annual_gamma_grid
)

print("Annual sample used in Figures 1 and 4")
print(f"  Years: {annual_years[0]}-{annual_years[-1]}")
print(f"  Mean stock return: {mu_x_annual[0]:.4f}")
print(f"  Mean bond return:  {mu_x_annual[1]:.4f}")
print(f"  Std stock return:  {np.sqrt(Sigma_annual[0, 0]):.4f}")
print(f"  Std bond return:   {np.sqrt(Sigma_annual[1, 1]):.4f}")
```

## Annual IMRS frontier

The main deliverable of the paper is the region $S$ in mean-standard deviation
space for the IMRS $m$.  Any parametric model must deliver an
$[E(m), \sigma(m)]$ pair inside $S$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Annual IMRS frontier
    name: fig-annual-imrs-frontier
---
fig, ax = plt.subplots(figsize=(8, 5))

ax.fill_between(v_annual, sigma_annual, 2.4, alpha=0.15)
ax.plot(v_annual, sigma_annual, lw=2)
ax.scatter(
    annual_crra_mean,
    annual_crra_std,
    marker="s",
    s=20,
    facecolors="white",
    edgecolors="black",
    linewidths=0.8,
)
annual_log_point = np.array([np.mean(1.0 / annual_stock), np.std(1.0 / annual_stock)])
ax.scatter(
    annual_log_point[0],
    annual_log_point[1],
    marker="x",
    s=50,
    color="black",
)

ax.set_xlim(0.84, 1.16)
ax.set_ylim(0.0, 2.4)
ax.set_xlabel("mean")
ax.set_ylabel("standard deviation")

plt.tight_layout()
plt.show()
```

## The Duality Theorem

The preceding figure illustrates the **duality** between the two frontiers —
the SDF frontier and the asset return mean-variance frontier — that Hansen
and Jagannathan establish formally.

**Theorem (Section III.C)**: For any $v$ there exists a frontier return $r_v^*$
such that $m^v$ is proportional to $r_v^*$.  The relative volatility of the
IMRS bound satisfies

$$
\frac{\sigma(m^v)}{v} = \left|\frac{E(r_v^*) - 1/v}{\sigma(r_v^*)}\right|
= |\text{Sharpe ratio of } r_v^*|.
$$

In words: the **slope of the SDF frontier** at mean $v$ equals the
**maximum achievable Sharpe ratio** in the asset menu augmented by a
hypothetical riskless bond priced at $v$.

```{code-cell} ipython3
# Compute the mean-variance frontier for asset returns R from the quarterly
# bill data (3-, 6-, 9-, 12-month holding-period returns).  The paper uses
# monthly holding-period returns on 1–6 month bills from CRSP; the quarterly
# proxy reproduces the same qualitative structure.

quarterly_bill_data = load_quarterly_bill_proxy().to_numpy()
mu_bill = quarterly_bill_data.mean(axis=0)
Sigma_bill = np.cov(quarterly_bill_data.T, bias=True)
inv_S_bill = np.linalg.inv(Sigma_bill)
ones_bill = np.ones(len(mu_bill))

A_bill = mu_bill @ inv_S_bill @ mu_bill
B_bill = mu_bill @ inv_S_bill @ ones_bill
C_bill = ones_bill @ inv_S_bill @ ones_bill
D_bill = A_bill * C_bill - B_bill**2

# Frontier: sigma^2 = (A mu^2 - 2B mu + C) / D
# Solve for mu given sigma: mu = (B ± sqrt(D(A sigma^2 - 1))) / A
sigma_min_bill = 1.0 / np.sqrt(A_bill)
sigma_grid_bill = np.linspace(sigma_min_bill * 1.001, 1.3, 1000)
disc_bill = D_bill * (A_bill * sigma_grid_bill**2 - 1)
disc_bill = np.maximum(disc_bill, 0)
mu_upper_bill = (B_bill + np.sqrt(disc_bill)) / A_bill
mu_lower_bill = (B_bill - np.sqrt(disc_bill)) / A_bill

# Minimum second-moment payoff r*
mu_star_bill = B_bill / (A_bill + D_bill)
sigma_star_bill = np.sqrt(
    max((A_bill * mu_star_bill**2 - 2 * B_bill * mu_star_bill + C_bill) / D_bill, 0)
)
r_star_norm = np.sqrt(sigma_star_bill**2 + mu_star_bill**2)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Minimum second-moment payoff in $R$
    name: fig-min-second-moment-payoff
---
theta_circle = np.linspace(0, np.pi / 2, 400)
sigma_circle = r_star_norm * np.cos(theta_circle)
mu_circle = r_star_norm * np.sin(theta_circle)

sigma_combined = np.concatenate([sigma_grid_bill[::-1], sigma_grid_bill])
mu_combined = np.concatenate([mu_lower_bill[::-1], mu_upper_bill])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigma_combined, mu_combined, lw=2)
ax.plot(sigma_circle, mu_circle, lw=2)
ax.scatter([sigma_star_bill], [mu_star_bill], s=30, zorder=5)
ax.set_xlim(0.0, 1.3)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("standard deviation")
ax.set_ylabel("mean")
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Mean-standard-deviation frontiers for $R$ and $R_v$
    name: fig-frontiers-r-rv
---
sigma_zoom = np.linspace(sigma_min_bill * 1.001, 0.04, 500)
disc_zoom = D_bill * (A_bill * sigma_zoom**2 - 1)
disc_zoom = np.maximum(disc_zoom, 0)
mu_up_zoom = (B_bill + np.sqrt(disc_zoom)) / A_bill
mu_lo_zoom = (B_bill - np.sqrt(disc_zoom)) / A_bill

sigma_comb_zoom = np.concatenate([sigma_zoom[::-1], sigma_zoom])
mu_comb_zoom = np.concatenate([mu_lo_zoom[::-1], mu_up_zoom])

# Augmented R_v: tangent from (0, 1/v) to the frontier
mu_vertex_bill = B_bill / C_bill
v_fig3 = 1.0 / (mu_vertex_bill + 0.006)
rf_fig3 = 1.0 / v_fig3

valid_sigma = sigma_zoom > 1e-10
slopes_up = np.abs(mu_up_zoom[valid_sigma] - rf_fig3) / sigma_zoom[valid_sigma]
slopes_lo = np.abs(mu_lo_zoom[valid_sigma] - rf_fig3) / sigma_zoom[valid_sigma]
max_slope = max(np.max(slopes_up), np.max(slopes_lo))

sigma_line = np.linspace(0, 0.04, 200)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigma_comb_zoom, mu_comb_zoom, lw=2, label=r"$R$")
rv_line = ax.plot(sigma_line, rf_fig3 + max_slope * sigma_line, lw=2, label=r"$R_v$")
ax.plot(sigma_line, rf_fig3 - max_slope * sigma_line, lw=2, color=rv_line[0].get_color())
ax.set_xlim(0.0, 0.04)
ax.set_ylim(0.98, 1.02)
ax.set_xlabel("standard deviation")
ax.set_ylabel("mean")
ax.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.show()
```

## Tightening the Bound: Imposing Positivity of $m$

### Option-based construction

When we also impose Restriction 2 ($m \geq 0$), the bound can be tightened
because many of the frontier $m^v$'s that solve the linear problem may be
negative with positive probability.

Hansen and Jagannathan show that the minimum variance **nonnegative** $m$
satisfying Restriction 1 is of the form

$$
\tilde{m}^v = \left(x_a^\top \alpha^v\right)^+ = \max\!\left\{x_a^\top \alpha^v,\ 0\right\},
$$

which is the payoff on a **European call (or put) option** on a portfolio
of the assets.  The option truncates the negative part of $m^v$, reducing
its variance while preserving the pricing restrictions.

The positive bound $\sigma(\tilde{m}^v)$ satisfies:

- $\sigma(\tilde{m}^v) \geq \sigma(m^v)$ (it is tighter).
- The admissible region $S^+$ (with positivity) is a proper subset of $S$.
- $S^+$ is **convex**.

Computing $\sigma(\tilde{m}^v)$ requires knowing the distribution of
$x_a^\top \alpha^v$, not just its first two moments.  For the annual and
quarterly figures below, we use the exact sample analogue of the truncation
problem and solve it numerically over a grid of candidate means.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: IMRS frontier with and without positivity
    name: fig-imrs-positivity
---
annual_pos_mean, annual_pos_std = positive_frontier_from_sample(
    annual_payoffs,
    annual_prices,
    v_annual,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(v_annual, sigma_annual, 2.4, alpha=0.1)
ax.plot(v_annual, sigma_annual, "--", lw=2, label="without positivity")

valid_annual = np.isfinite(annual_pos_std)
order = np.argsort(annual_pos_mean[valid_annual])
ax.fill_between(
    annual_pos_mean[valid_annual][order],
    annual_pos_std[valid_annual][order],
    2.4,
    alpha=0.2,
)
ax.plot(
    annual_pos_mean[valid_annual][order],
    annual_pos_std[valid_annual][order],
    lw=2,
    label="with positivity",
)

ax.set_xlim(0.84, 1.16)
ax.set_ylim(0.0, 2.4)
ax.set_xlabel("mean")
ax.set_ylabel("standard deviation")
ax.legend(frameon=False, fontsize=9, loc="lower right")
plt.tight_layout()
plt.show()
```

## The Equity Premium Puzzle Revisited

The HJ bound provides a clean, nonparametric restatement of the equity
premium puzzle: for the bound to be met, the IMRS of the representative
agent must be far more volatile than consumption growth alone can generate
under standard preferences.

For a CRRA consumer with risk aversion $\gamma$,

$$
m = \delta \left(\frac{c_{t+1}}{c_t}\right)^{-\gamma}.
$$

If consumption growth is lognormal with mean $\mu_c$ and standard deviation
$\sigma_c$, then

$$
E(m) = \delta \exp\!\left(-\gamma \mu_c + \tfrac{1}{2} \gamma^2 \sigma_c^2\right),
\quad
\frac{\sigma(m)}{E(m)} = \sqrt{\exp\!\left(\gamma^2 \sigma_c^2\right) - 1}
\approx \gamma \sigma_c.
$$

To meet the HJ bound $\sigma(m)/E(m) \geq \text{SR}_{\max}$, we need

$$
\gamma \sigma_c \gtrsim \text{SR}_{\max}.
$$

With U.S. annual data, $\text{SR}_{\max} \approx 0.37$ and $\sigma_c \approx
0.033$, so the required risk aversion is roughly $\gamma \approx 11$ — far
higher than the values of 1–5 that are typically considered plausible.

```{code-cell} ipython3
print("CRRA points from Figure 1")
print(f"{'gamma':>6}  {'E(m)':>8}  {'sigma(m)':>10}  {'above bound':>12}")
for g, E_m, s_m in zip(annual_gamma_grid, annual_crra_mean, annual_crra_std):
    bound_val = float(np.interp(E_m, v_annual, sigma_annual))
    flag = '✓ inside' if s_m >= bound_val else '✗ outside'
    if g in {0, 1, 2, 5, 10, 20, 30}:
        print(f"{g:>6}  {E_m:>8.4f}  {s_m:>10.4f}  {flag:>12}")
```

## Time-Nonseparable Preferences

Section V of the paper examines whether relaxing time separability can help
close the gap to the HJ bound.  Consider the nonseparable service flow

$$
s_t = c_t + \theta c_{t-1},
$$

where $\theta > 0$ represents **local durability** and $\theta < 0$ represents
**habit persistence** (intertemporal complementarity).

The IMRS becomes more complex because it depends on current and future
marginal utilities:

$$
m = \delta \frac{(s_{t+1})^\gamma + \theta \delta E[(s_{t+2})^\gamma \mid I_{t+1}]}
               {(s_t)^\gamma + \theta \delta E[(s_{t+1})^\gamma \mid I_t]}.
$$

The paper shows (Figure 5) that habit persistence ($\theta < 0$) dramatically
increases $\sigma(m)$ for given $\gamma$, while local durability ($\theta > 0$)
barely reduces it.

The exact Gallant-Tauchen state process used in the paper is not bundled with
the lecture sources, so the code below combines a local monthly payoff proxy
with the same simple nonseparable preference calibration used to reproduce the
paper's qualitative separation of boxes, circles, and triangles.

```{code-cell} ipython3
monthly_panel = load_monthly_proxy_panel()
monthly_payoffs, monthly_prices = build_monthly_payoff_menu(monthly_panel)

v_monthly = np.linspace(0.975, 1.0, 100)
mu_x_monthly, mu_q_monthly, Sigma_monthly = compute_moments(monthly_payoffs, monthly_prices)
v_m_nopositivity, sigma_m_nopositivity = hj_bound_no_positivity(
    mu_x_monthly, mu_q_monthly, Sigma_monthly, v_grid=v_monthly
)


def simulate_nonseparable_imrs(
    T=20_000,
    gamma=-5,
    theta=0.0,
    delta=1.0,
    mu_c=0.0045,
    sigma_c=0.0055,
    seed=1,
):
    rng = np.random.default_rng(seed)
    growth = np.exp(mu_c + sigma_c * rng.standard_normal(T + 2))

    c = np.ones(T + 2)
    for t in range(T + 1):
        c[t + 1] = c[t] * growth[t + 1]

    s = c[1:] + theta * c[:-1]
    s_gamma = s ** gamma
    mu_num = s_gamma[1:T + 1] + theta * delta * np.append(s_gamma[2:T + 1], s_gamma[-1])
    mu_denom = s_gamma[:T] + theta * delta * s_gamma[1:T + 1]

    m = delta * mu_num / mu_denom
    return m[np.isfinite(m) & (np.abs(m) < 1e6)]
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: IMRS frontier using monthly data
    name: fig-imrs-monthly
---
fig, ax = plt.subplots(figsize=(8, 5))

ax.fill_between(v_m_nopositivity, sigma_m_nopositivity, 0.4, alpha=0.15)
ax.plot(v_m_nopositivity, sigma_m_nopositivity, lw=2)

for theta, marker, label in [
    (0.0, "s", r"$\theta = 0$"),
    (0.5, "o", r"$\theta = 0.5$"),
    (-0.5, "^", r"$\theta = -0.5$"),
]:
    pts = []
    for gamma in range(0, -15, -1):
        m_sim = simulate_nonseparable_imrs(gamma=gamma, theta=theta, seed=abs(gamma) + 5)
        pts.append((m_sim.mean(), m_sim.std()))
    pts = np.asarray(pts)
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        marker=marker,
        s=18,
        facecolors="white",
        edgecolors="black",
        linewidths=0.8,
        label=label,
    )

monthly_log_point = np.array([np.mean(1.0 / monthly_panel["stock"]), np.std(1.0 / monthly_panel["stock"])])
ax.scatter(monthly_log_point[0], monthly_log_point[1], marker="x", s=50, color="black")

ax.set_xlim(0.975, 1.2)
ax.set_ylim(0.0, 0.4)
ax.set_xlabel("mean")
ax.set_ylabel("standard deviation")
ax.legend(frameon=False, fontsize=9, loc="upper left")
plt.tight_layout()
plt.show()
```

## Treasury Bill Data and Monetary Models

Figure 6 in the paper uses monthly prices on 3-, 6-, 9-, and 12-month
discount bonds to construct real **quarterly** holding-period returns.  The
original CRSP bill file is not distributed with the lecture, so we build a local
proxy from FRED Treasury yields and the same real-consumption deflator used
above.

```{code-cell} ipython3
quarterly_payoffs = load_quarterly_bill_proxy().to_numpy()
quarterly_prices = np.ones_like(quarterly_payoffs)

mu_x_quarterly, mu_q_quarterly, Sigma_quarterly = compute_moments(
    quarterly_payoffs, quarterly_prices
)
v_quarterly, sigma_quarterly = hj_bound_no_positivity(
    mu_x_quarterly,
    mu_q_quarterly,
    Sigma_quarterly,
    v_grid=np.linspace(0.985, 1.005, 200),
)
quarterly_pos_mean, quarterly_pos_std = positive_frontier_from_sample(
    quarterly_payoffs,
    quarterly_prices,
    v_quarterly,
)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: IMRS frontier using quarterly returns
    name: fig-imrs-quarterly
---
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(v_quarterly, sigma_quarterly, 2.0, alpha=0.1)
ax.plot(v_quarterly, sigma_quarterly, "--", lw=2, label="without positivity")

valid_quarterly = np.isfinite(quarterly_pos_std)
order = np.argsort(quarterly_pos_mean[valid_quarterly])
ax.fill_between(
    quarterly_pos_mean[valid_quarterly][order],
    quarterly_pos_std[valid_quarterly][order],
    2.0,
    alpha=0.2,
)
ax.plot(
    quarterly_pos_mean[valid_quarterly][order],
    quarterly_pos_std[valid_quarterly][order],
    lw=2,
    label="with positivity",
)

ax.set_xlim(0.985, 1.005)
ax.set_ylim(0.0, 2.0)
ax.set_xlabel("mean")
ax.set_ylabel("standard deviation")
ax.legend(frameon=False, fontsize=9, loc="lower right")
plt.tight_layout()
plt.show()
```

## Simulation Helper for the Exercises

The exercises below still use a small simulated exchange economy to verify the
projection and variance-decomposition logic algebraically.

```{code-cell} ipython3
def simulate_economy(
    T=10_000,
    gamma=2.0,
    delta=0.99,
    mu_c=0.018,
    sigma_c=0.033,
    mu_d=0.02,
    sigma_d=0.12,
    rho=0.3,
    seed=42,
):
    rng = np.random.default_rng(seed)

    cov = np.array(
        [
            [sigma_c**2, rho * sigma_c * sigma_d],
            [rho * sigma_c * sigma_d, sigma_d**2],
        ]
    )
    shocks = rng.multivariate_normal([0.0, 0.0], cov, T)

    gc = np.exp(mu_c + shocks[:, 0])
    gd = np.exp(mu_d + shocks[:, 1])
    m_true = delta * gc ** (-gamma)

    rf = 1.0 / np.mean(m_true)
    stock_raw = gd
    stock = stock_raw / np.mean(m_true * stock_raw)
    bond = np.full(T, rf)

    returns = np.column_stack([stock, bond])
    prices = np.ones((T, 2))
    return returns, prices, m_true


def crra_imrs_moments(gamma, delta=0.99, mu_c=0.018, sigma_c=0.033):
    E_m = delta * np.exp(-gamma * mu_c + 0.5 * gamma**2 * sigma_c**2)
    var_m = delta**2 * np.exp(-2.0 * gamma * mu_c + 2.0 * gamma**2 * sigma_c**2) - E_m**2
    sigma_m = np.sqrt(max(var_m, 0.0))
    return E_m, sigma_m
```

## Exercises

```{exercise}
:label: hj91_ex1

**Deriving $m^*$ and verifying the variance decomposition**

Let $x$ be an $n$-dimensional vector of asset payoffs with mean $\mu_x$,
second moment matrix $M_{xx} = Exx^\top$, and mean price vector $\mu_q$.

(a) Show that $m^* = x^\top \alpha^*$ with $\alpha^* = M_{xx}^{-1}\mu_q$
    is the minimum second-moment element of $P = \{c^\top x : c \in \mathbb{R}^n\}$
    that satisfies the pricing restriction $Eq = E(xm)$.

(b) For any $m$ satisfying the pricing restriction, show that

$$
\|m\|^2 = \|m^*\|^2 + \|m - m^*\|^2,
$$

and hence $\sigma^2(m) \geq \sigma^2(m^*)$ when $Em = Em^*$.

(c) Verify (b) numerically using `simulate_economy` with $\gamma = 5$.
    Compute $m^*$ from the simulated payoff data, then verify that the
    variance of the true IMRS exceeds that of $m^*$.
```

```{solution-start} hj91_ex1
:class: dropdown
```

**(a)** The pricing restriction $E(xm) = \mu_q$ can be written as
$M_{xx} c = \mu_q$ when $m = x^\top c$.  The unique solution is
$\alpha^* = M_{xx}^{-1}\mu_q$, so $m^* = x^\top \alpha^*$ is the
unique element of $P$ satisfying the pricing restriction.

**(b)** Because $E[x(m - m^*)] = E(xm) - E(xm^*) = \mu_q - \mu_q = 0$,
the discrepancy $m - m^*$ is orthogonal to all elements of $P$, including
$m^*$ itself.  Therefore,

$$
\|m\|^2 = E(m^2) = E[(m^* + (m-m^*))^2]
= E(m^{*2}) + 2E[m^*(m - m^*)] + E[(m-m^*)^2]
= \|m^*\|^2 + \|m - m^*\|^2.
$$

When $Em = Em^*$, subtracting $(Em)^2 = (Em^*)^2$ from both sides gives
the variance decomposition:

$$
\sigma^2(m) = \sigma^2(m^*) + \sigma^2(m - m^*) \geq \sigma^2(m^*).
$$

**(c)** Numerical verification:

```{code-cell} ipython3
# Simulate economy with gamma = 5
returns_g5, prices_g5, m_true_g5 = simulate_economy(T=10000, gamma=5.0, seed=7)
T5 = len(m_true_g5)

# Construct m*: project onto P = span(returns)
# M_xx = E[x x'] ≈ sample second moment matrix
Mxx = (returns_g5.T @ returns_g5) / T5
mu_q5 = np.ones(2)           # unit prices (gross returns)
alpha_star = np.linalg.solve(Mxx, mu_q5)

m_star = returns_g5 @ alpha_star   # m* time series

# Verify pricing: E[x * m*] ≈ E[q] = 1
print("Pricing check  E[r_i * m*]  (should ≈ 1.0):")
print(f"  Asset 1: {np.mean(returns_g5[:,0] * m_star):.6f}")
print(f"  Asset 2: {np.mean(returns_g5[:,1] * m_star):.6f}")

# Verify variance decomposition
# m = m* + (m - m*), and (m - m*) ⊥ P
residual = m_true_g5[:T5] - m_star

print(f"\nVariance decomposition:")
print(f"  Var(m)        = {np.var(m_true_g5):.6f}")
print(f"  Var(m*)       = {np.var(m_star):.6f}")
print(f"  Var(m - m*)   = {np.var(residual):.6f}")
print(f"  Var(m*) + Var(m-m*) = {np.var(m_star) + np.var(residual):.6f}")
print(f"\nσ(m) ≥ σ(m*): {np.std(m_true_g5):.5f} ≥ {np.std(m_star):.5f}  "
      f"({'✓' if np.std(m_true_g5) >= np.std(m_star) else '✗'})")
print(f"E[(m - m*) * m*] ≈ 0: {np.mean(residual * m_star):.2e}  (orthogonality check)")
```

```{solution-end}
```

```{exercise}
:label: hj91_ex2

**Computing and plotting the HJ bound from historical data**

Use the following calibrated annual U.S. data moments
(roughly matching the Campbell-Shiller 1891-1985 dataset used in the paper):

| Moment | Value |
|--------|-------|
| $E(r_{\text{stock}})$ | 1.0698 |
| $E(r_{\text{bond}})$ | 1.0100 |
| $\sigma(r_{\text{stock}})$ | 0.167 |
| $\sigma(r_{\text{bond}})$ | 0.057 |
| $\rho(r_{\text{stock}}, r_{\text{bond}})$ | $-0.02$ |

(a) Using the `hj_bound_no_positivity` function, compute and plot the
    HJ bound for $v \in [0.90, 1.10]$.

(b) For a CRRA consumer with $\delta = 0.95$ and annual consumption growth
    distributed as $\log(c_{t+1}/c_t) \sim N(0.018, 0.033^2)$, overlay
    the mean-standard deviation pairs for $\gamma \in \{1, 2, 5, 10, 20\}$.

(c) At what $\gamma$ does the CRRA model first enter the admissible region?
    Interpret your result.
```

```{solution-start} hj91_ex2
:class: dropdown
```

```{code-cell} ipython3
# (a) HJ bound with historical moments
mu_x_hist  = np.array([1.0698, 1.0100])
mu_q_hist  = np.ones(2)
std_s, std_b, rho_sb = 0.167, 0.057, -0.02
Sigma_hist = np.array([
    [std_s**2, rho_sb * std_s * std_b],
    [rho_sb * std_s * std_b, std_b**2]
])

v_hist, s_hist = hj_bound_no_positivity(mu_x_hist, mu_q_hist, Sigma_hist,
                                        v_grid=np.linspace(0.90, 1.10, 400))

# (b) CRRA model points
gammas_plot = [1, 2, 5, 10, 20]
crra_moments = [(g, *crra_imrs_moments(g, delta=0.95)) for g in gammas_plot]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(v_hist, s_hist, lw=2, color='steelblue', label='HJ bound (historical)')
ax.fill_betweenx(np.linspace(0, 0.6, 400), v_hist.min(), v_hist.max(),
                 alpha=0.07, color='steelblue')

cmap = plt.cm.Reds(np.linspace(0.4, 0.9, len(gammas_plot)))
for (g, E_m, s_m), c in zip(crra_moments, cmap):
    ax.scatter(E_m, s_m, color=c, s=90, zorder=5)
    ax.annotate(f'γ={g}', (E_m, s_m), xytext=(8, 3),
                textcoords='offset points', fontsize=9, color=c)

ax.set_xlabel('$E(m)$')
ax.set_ylabel('$\\sigma(m)$')
ax.set_xlim([0.90, 1.10])
ax.set_ylim([-0.01, 0.55])
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

# (c) Find crossing gamma
print("σ(m) vs. HJ bound at E(m):")
first_inside = None
for g in range(1, 50):
    E_m, s_m = crra_imrs_moments(g, delta=0.95)
    if 0.90 <= E_m <= 1.10:
        bound_at_v = float(np.interp(E_m, v_hist, s_hist))
        inside = s_m >= bound_at_v
        if inside and first_inside is None:
            first_inside = g
        if g <= 20 or inside:
            print(f"  γ={g:2d}: E(m)={E_m:.4f}, σ(m)={s_m:.4f}, "
                  f"bound={bound_at_v:.4f}  {'✓' if inside else '✗'}")

print(f"\nFirst γ inside the admissible region: γ = {first_inside}")
print("Interpretation: standard CRRA preferences require implausibly high risk")
print("aversion to generate enough SDF volatility — this is the equity premium puzzle.")
```

```{solution-end}
```

```{exercise}
:label: hj91_ex3

**Effect of adding instruments on the bound**

The paper notes that adding instruments (by multiplying asset payoffs by lagged
variables in the conditioning information set) can tighten the HJ bound because
it increases the maximum achievable Sharpe ratio.

(a) Generate a simulated dataset using `simulate_economy` with $T=5000$ and
    $\gamma=8$.

(b) Define three nested sets of payoffs:
    - Set A: the 2 original returns.
    - Set B: the 2 returns plus 2 lagged-return-scaled instruments (4 assets).
    - Set C: the 2 returns plus 4 instruments (6 assets).

(c) Compute the HJ bound for each set and plot all three on the same figure.
    Is the bound for Set C tighter than for Set A?

(d) Compute the maximum Sharpe ratio for each set and verify that it
    increases as the asset space expands.
```

```{solution-start} hj91_ex3
:class: dropdown
```

```{code-cell} ipython3
# (a) Simulate
returns5k, _, m_true5k = simulate_economy(T=5000, gamma=8.0, seed=42)

# (b) Construct payoff sets
def build_payoff_sets(returns):
    T, n = returns.shape
    # Set A: original 2 returns
    A = returns[2:]

    # Set B: A + lag-1 instruments (payoff * lag-1 of return 1)
    z1 = returns[1:-1, 0]                           # lag-1 of asset 1
    B  = np.column_stack([A,
                          A[:, 0] * z1,
                          A[:, 1] * z1])

    # Set C: B + lag-2 instruments (payoff * lag-2)
    z2 = returns[:-2, 1]                            # lag-2 of asset 2
    C  = np.column_stack([B,
                          A[:, 0] * z2,
                          A[:, 1] * z2])

    # Prices: original payoffs have price 1; scaled payoffs have price = instrument
    q_A = np.ones((len(A), 2))
    q_B = np.column_stack([q_A, z1[:, None], z1[:, None]])
    q_C = np.column_stack([q_B, z2[:, None], z2[:, None]])

    return A, q_A, B, q_B, C, q_C


A, q_A, B, q_B, C, q_C = build_payoff_sets(returns5k)

sets   = [(A, q_A, 'Set A: 2 assets'),
          (B, q_B, 'Set B: 4 assets (+ lag-1 instr.)'),
          (C, q_C, 'Set C: 6 assets (+ lag-1,2 instr.)')]
colors = ['steelblue', 'darkorange', 'crimson']

fig, ax = plt.subplots(figsize=(9, 5))
for (X, Q, label), color in zip(sets, colors):
    mu_xk, mu_qk, Sig_k = compute_moments(X, Q)
    vk, sk = hj_bound_no_positivity(mu_xk, mu_qk, Sig_k,
                                     v_grid=np.linspace(0.90, 1.10, 300))
    ax.plot(vk, sk, lw=2, label=label, color=color)

ax.scatter(np.mean(m_true5k), np.std(m_true5k), color='black', s=100, zorder=5,
           label='True IMRS (γ=8)')
ax.set_xlabel('$E(m)$')
ax.set_ylabel('$\\sigma(m)$')
ax.set_xlim([0.90, 1.10])
ax.set_ylim([-0.01, 0.55])
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# (d) Maximum Sharpe ratio for each set
print("Maximum Sharpe ratio by payoff set:")
for (X, Q, label) in sets:
    mu_xk, mu_qk, Sig_k = compute_moments(X, Q)
    sr = max_sharpe_ratio(mu_xk, mu_qk, Sig_k)
    print(f"  {label}: SR_max = {sr:.5f}")
```

```{solution-end}
```
