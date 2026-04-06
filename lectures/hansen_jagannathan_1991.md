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
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.stats import norm
import pandas as pd

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
    T, n = returns.shape
    mu_x  = returns.mean(axis=0)
    Sigma = np.cov(returns.T)
    if prices is None:
        # returns have price 1 by construction
        mu_q = np.ones(n)
    else:
        mu_q = prices.mean(axis=0)
    return mu_x, mu_q, Sigma
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

    inv_Sigma = np.linalg.inv(Sigma)
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
    inv_S = np.linalg.inv(Sigma)
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
    inv_S = np.linalg.inv(Sigma)

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

## Computing the Bound with Simulated Data

To illustrate the theory we simulate an economy with a CRRA representative
consumer and two assets: a stock and a bond.

```{code-cell} ipython3
def simulate_economy(T=10000, gamma=2.0, delta=0.99, mu_c=0.018, sigma_c=0.033,
                     mu_d=0.02, sigma_d=0.12, rho=0.3, seed=42):
    """
    Simulate a Lucas (1978) exchange economy with log-normal consumption and
    dividend growth.

    Parameters
    ----------
    T       : number of periods
    gamma   : CRRA coefficient
    delta   : time discount factor
    mu_c    : mean log consumption growth
    sigma_c : std of log consumption growth
    mu_d    : mean log dividend growth
    sigma_d : std of log dividend growth
    rho     : correlation between consumption and dividend shocks

    Returns
    -------
    returns  : (T, 2) array — [stock return, bond return]
    prices   : (T, 2) array — prices (≡ 1 for returns)
    m_true   : (T,)   array — true IMRS
    """
    rng = np.random.default_rng(seed)

    # Correlated shocks
    cov = np.array([[sigma_c**2, rho * sigma_c * sigma_d],
                    [rho * sigma_c * sigma_d, sigma_d**2]])
    shocks = rng.multivariate_normal([0, 0], cov, T)

    # Consumption and dividend growth
    gc = np.exp(mu_c + shocks[:, 0])   # gross consumption growth
    gd = np.exp(mu_d + shocks[:, 1])   # gross dividend growth

    # IMRS: m = delta * (c_t+1/c_t)^(-gamma)
    m_true = delta * gc ** (-gamma)

    # Risk-free gross return (from the bond Euler equation): 1/E[m]
    rf = 1.0 / np.mean(m_true)

    # Stock return: proportional to dividend growth scaled to have E[m*R] = 1
    # In a simple calibration, R_stock ~ gd / price_to_div
    # We back out a price so that E[m * R_stock] = 1 in population
    # Use sample-consistent construction
    R_stock_raw = gd
    scale = np.mean(m_true * R_stock_raw)   # should equal 1 if correctly priced
    R_stock = R_stock_raw / scale            # re-scale to satisfy pricing

    # Bond return: constant rf
    R_bond = np.full(T, rf)

    returns = np.column_stack([R_stock, R_bond])
    prices  = np.ones((T, 2))
    return returns, prices, m_true


returns, prices, m_true = simulate_economy(T=10000, gamma=2.0)
mu_x, mu_q, Sigma = compute_moments(returns, prices)

print("Asset moments (simulated economy, gamma=2):")
print(f"  Mean stock return:  {mu_x[0]:.5f}")
print(f"  Mean bond return:   {mu_x[1]:.5f}")
print(f"  Std stock return:   {np.sqrt(Sigma[0,0]):.5f}")
print(f"  Std bond return:    {np.sqrt(Sigma[1,1]):.6f}")
print(f"\nTrue IMRS:")
print(f"  E(m):      {np.mean(m_true):.5f}")
print(f"  sigma(m):  {np.std(m_true):.5f}")
```

## Plotting the HJ Bound and the SDF Frontier

The main deliverable of the paper is the region $S$ in mean-standard deviation
space for the IMRS $m$.  Any parametric model must deliver an
$[E(m), \sigma(m)]$ pair inside $S$.

```{code-cell} ipython3
# Compute the linear HJ bound
v_grid, sigma_bound = hj_bound_no_positivity(mu_x, mu_q, Sigma)

# Compute the mean-variance frontier for asset returns
frontier_means, frontier_stds = mean_variance_frontier(mu_x, Sigma)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# — Left panel: HJ bound (SDF frontier) —
ax = axes[0]
ax.plot(v_grid, sigma_bound, lw=2, color='steelblue', label='HJ bound (no positivity)')
ax.fill_betweenx(sigma_bound, v_grid, v_grid[-1],
                 alpha=0.12, color='steelblue', label='Admissible region $S$')

# Mark the true IMRS
ax.scatter(np.mean(m_true), np.std(m_true), color='red', s=80, zorder=5,
           label=f'True IMRS (γ=2): ({np.mean(m_true):.3f}, {np.std(m_true):.3f})')

ax.set_xlabel('Mean of IMRS  $E(m)$')
ax.set_ylabel('Std of IMRS  $\\sigma(m)$')
ax.set_title('HJ Volatility Bound\n(linear, no positivity)')
ax.legend(fontsize=9)
ax.set_xlim([0.88, 1.12])
ax.set_ylim([-0.005, 0.25])

# — Right panel: asset returns mean-variance frontier —
ax = axes[1]
ax.plot(frontier_stds, frontier_means, lw=2, color='darkorange',
        label='MV frontier for returns')
ax.scatter(np.sqrt(np.diag(Sigma)), mu_x, s=80, color='green', zorder=5,
           label='Individual assets')
ax.set_xlabel('Std of return  $\\sigma(r)$')
ax.set_ylabel('Mean of return  $E(r)$')
ax.set_title('Mean-Variance Frontier for Asset Returns\n(dual of the HJ bound)')
ax.legend(fontsize=9)

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
def sharpe_ratio_grid(mu_x, mu_q, Sigma, v_grid):
    """
    For each candidate risk-free price v, compute the maximum Sharpe ratio
    of excess returns over the hypothetical riskless rate 1/v.

    Returns the ratio sigma(m^v)/v, which should equal max SR.
    """
    n = len(mu_x)
    inv_S = np.linalg.inv(Sigma)

    sr_max = []
    for v in v_grid:
        rf = 1.0 / v                  # hypothetical risk-free return
        mu_exc = mu_x - rf            # excess returns
        w_tan  = inv_S @ mu_exc       # unnormalized tangency weights
        denom  = np.sqrt(w_tan @ Sigma @ w_tan)
        if denom < 1e-12:
            sr_max.append(0.0)
        else:
            sr_max.append(float((mu_exc @ w_tan) / denom))

    return np.array(sr_max)


v_grid, sigma_bound = hj_bound_no_positivity(mu_x, mu_q, Sigma)
sr_ratios = sharpe_ratio_grid(mu_x, mu_q, Sigma, v_grid)
# HJ bound / v should equal max SR
hj_ratio  = sigma_bound / v_grid

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(v_grid, hj_ratio,  lw=2, label=r'$\sigma(m^v)/v$  (HJ bound / mean)')
ax.plot(v_grid, sr_ratios, lw=2, linestyle='--', label='Max Sharpe ratio at $r_f = 1/v$')
ax.set_xlabel('$E(m) = v$')
ax.set_ylabel('Ratio')
ax.set_title('Duality: HJ Bound/Mean = Maximum Sharpe Ratio')
ax.legend()
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
$x_a^\top \alpha^v$, not just its first two moments.  Under normality (which
we use below as an approximation), we can compute it analytically.

```{code-cell} ipython3
def phi(z):
    """Standard normal CDF."""
    return norm.cdf(z)

def phi_pdf(z):
    """Standard normal PDF."""
    return norm.pdf(z)


def hj_bound_with_positivity(mu_x, mu_q, Sigma, v_grid=None):
    """
    Compute the Hansen-Jagannathan volatility bound WITH the positivity
    restriction, assuming m^v = x_a' alpha^v is approximately normal.

    For a normal random variable Y ~ N(mu_Y, sigma_Y^2):
        E[max(Y,0)] = mu_Y * Phi(mu_Y/sigma_Y) + sigma_Y * phi(mu_Y/sigma_Y)
        E[max(Y,0)^2] = (mu_Y^2 + sigma_Y^2) * Phi(mu_Y/sigma_Y)
                       + mu_Y * sigma_Y * phi(mu_Y/sigma_Y)
    where Phi = standard normal CDF, phi = standard normal PDF.

    Parameters
    ----------
    mu_x   : mean payoffs  (n,)
    mu_q   : mean prices   (n,)
    Sigma  : covariance matrix of payoffs  (n, n)
    v_grid : array of candidate means for m

    Returns
    -------
    v_plus  : array of E[tilde_m^v] values (may differ from v_grid)
    s_plus  : array of sigma[tilde_m^v] values
    """
    if v_grid is None:
        v_grid = np.linspace(0.85, 1.15, 300)

    n = len(mu_x)
    inv_S = np.linalg.inv(Sigma)

    means_out  = []
    sigmas_out = []

    for v in v_grid:
        # Augmented system
        mu_xa = np.append(mu_x, 1.0)
        mu_qa = np.append(mu_q, v)

        # Second moment matrix of x_a (use E[x_a x_a'] = Sigma_a + mu_xa mu_xa')
        Sigma_a = np.zeros((n+1, n+1))
        Sigma_a[:n, :n] = Sigma
        # x_a = (x', 1)', so Cov(x_a) has zero last row/col (unit payoff has zero var)
        Exa_xa = Sigma_a + np.outer(mu_xa, mu_xa)

        try:
            alpha = np.linalg.solve(Exa_xa, mu_qa)
        except np.linalg.LinAlgError:
            continue

        # m^v = x_a . alpha is approximately normal with:
        mu_mv    = float(mu_xa @ alpha)          # = v by construction
        var_mv   = float(alpha @ Sigma_a @ alpha)
        sigma_mv = np.sqrt(max(var_mv, 0.0))

        if sigma_mv < 1e-12:
            means_out.append(mu_mv)
            sigmas_out.append(0.0)
            continue

        z = mu_mv / sigma_mv
        # E[max(m^v, 0)]
        mean_plus  = mu_mv * phi(z) + sigma_mv * phi_pdf(z)
        # E[max(m^v, 0)^2]
        e2_plus    = (mu_mv**2 + sigma_mv**2) * phi(z) + mu_mv * sigma_mv * phi_pdf(z)
        sigma_plus = np.sqrt(max(e2_plus - mean_plus**2, 0.0))

        means_out.append(mean_plus)
        sigmas_out.append(sigma_plus)

    return np.array(means_out), np.array(sigmas_out)


v_plus, s_plus = hj_bound_with_positivity(mu_x, mu_q, Sigma)
v_lin,  s_lin  = hj_bound_no_positivity(mu_x, mu_q, Sigma)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(v_lin,  s_lin,  lw=2, linestyle='--', color='steelblue',
        label='Linear bound (no positivity)')
ax.plot(v_plus, s_plus, lw=2, color='navy',
        label='Positive bound (with positivity)')
ax.fill_between(v_plus, s_plus, s_plus.max() * 1.2,
                alpha=0.15, color='navy', label='Admissible region $S^+$')

ax.scatter(np.mean(m_true), np.std(m_true), color='red', s=100, zorder=6,
           label=f'True IMRS (γ=2)')

ax.set_xlabel('Mean of IMRS  $E(m)$')
ax.set_ylabel('Std of IMRS  $\\sigma(m)$')
ax.set_title('HJ Bound: With and Without Positivity')
ax.set_xlim([0.88, 1.12])
ax.set_ylim([-0.005, 0.25])
ax.legend(fontsize=9)
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
def crra_imrs_moments(gamma, delta=0.99, mu_c=0.018, sigma_c=0.033):
    """
    Compute E(m) and sigma(m) for a CRRA representative consumer with
    log-normal consumption growth.
    """
    # m = delta * exp(-gamma * log(c_{t+1}/c_t))
    # log(c_{t+1}/c_t) ~ N(mu_c, sigma_c^2)
    E_m     = delta * np.exp(-gamma * mu_c + 0.5 * gamma**2 * sigma_c**2)
    Var_m   = (delta**2 * np.exp(-2*gamma*mu_c + 2*gamma**2*sigma_c**2)
               - E_m**2)
    sigma_m = np.sqrt(max(Var_m, 0))
    return E_m, sigma_m


# HJ bound using calibrated U.S. annual data (Campbell-Shiller 1891-1985)
# We use two assets: S&P 500 and short bond
# Calibrated moments from the paper (approximately)
mu_sp500  = 1.0698   # mean annual gross stock return
mu_bond   = 1.0100   # mean annual gross bond return
std_sp500 = 0.167    # std of annual stock return
std_bond  = 0.057    # std of annual bond return
rho_sb    = -0.02    # correlation stock-bond

Sigma_cal = np.array([
    [std_sp500**2, rho_sb * std_sp500 * std_bond],
    [rho_sb * std_sp500 * std_bond, std_bond**2]
])
mu_x_cal  = np.array([mu_sp500, mu_bond])
mu_q_cal  = np.ones(2)   # unit-price returns

v_lin_cal, s_lin_cal = hj_bound_no_positivity(mu_x_cal, mu_q_cal, Sigma_cal,
                                               v_grid=np.linspace(0.88, 1.12, 300))

# CRRA model points for different gamma values
gammas = [0, 1, 2, 5, 10, 20, 30]
crra_pts = [crra_imrs_moments(g) for g in gammas]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(v_lin_cal, s_lin_cal, lw=2, color='steelblue',
        label='HJ bound (U.S. annual data, calibrated)')
ax.fill_betweenx(np.linspace(0, 0.6, 300),
                 0.88, 1.12, alpha=0.06, color='steelblue')

colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(gammas)))
for (E_m, s_m), g, c in zip(crra_pts, gammas, colors):
    ax.scatter(E_m, s_m, color=c, s=90, zorder=5)
    ax.annotate(f'γ={g}', (E_m, s_m), textcoords='offset points',
                xytext=(6, 3), fontsize=8, color=c)

ax.set_xlabel('$E(m)$', fontsize=12)
ax.set_ylabel('$\\sigma(m)$', fontsize=12)
ax.set_title('HJ Bound and the Equity Premium Puzzle\n'
             'CRRA model: most γ values are in the inadmissible region', fontsize=11)
ax.set_xlim([0.88, 1.12])
ax.set_ylim([-0.01, 0.55])
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

print("CRRA model moments vs. HJ bound:")
print(f"{'gamma':>6}  {'E(m)':>8}  {'sigma(m)':>10}  {'sigma/E':>8}  {'above bound':>12}")
for (E_m, s_m), g in zip(crra_pts, gammas):
    bound_val = float(np.interp(E_m, v_lin_cal, s_lin_cal))
    flag = '✓ inside' if s_m >= bound_val else '✗ outside'
    print(f"{g:>6}  {E_m:>8.4f}  {s_m:>10.4f}  {s_m/E_m:>8.4f}  {flag:>12}")
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

```{code-cell} ipython3
def simulate_nonseparable_imrs(T=20000, gamma=-5, theta=0.0, delta=1.0,
                                mu_c=0.002, sigma_c=0.0055, seed=1):
    """
    Simulate the IMRS for a time-nonseparable consumer with service flow
    s_t = c_t + theta * c_{t-1}  (monthly calibration).

    Uses a simple AR(1) approximation for log consumption growth.
    """
    rng = np.random.default_rng(seed)
    gc = np.exp(mu_c + sigma_c * rng.standard_normal(T + 2))  # gross c growth

    # Build levels: c_t+1 = c_t * gc_t+1,  start at 1
    c = np.ones(T + 2)
    for t in range(T + 1):
        c[t+1] = c[t] * gc[t+1]

    # Service flow
    s = c[1:] + theta * c[:-1]   # length T+1

    # Marginal utility of consumption good:
    # mu_t = (s_t)^gamma + theta * delta * E[(s_{t+1})^gamma | I_t]
    # For simplicity, approximate E[(s_{t+1})^gamma | It] ≈ (s_{t+1})^gamma
    # (true for a deterministic approximation or in simulation)
    s_gamma = s ** gamma           # length T+1

    # IMRS: ratio of scaled marginal utilities at t+1 and t
    mu_num   = s_gamma[1:]   + theta * delta * s_gamma[2:] if T + 2 > len(s_gamma) \
               else s_gamma[1:T+1] + theta * delta * np.append(s_gamma[2:T+1], s_gamma[-1])
    mu_denom = s_gamma[:T]  + theta * delta * s_gamma[1:T+1]

    m = delta * mu_num / mu_denom
    return m[~np.isnan(m) & (np.abs(m) < 1e6)]


fig, ax = plt.subplots(figsize=(9, 5))

# Use calibrated annual HJ bound
ax.plot(v_lin_cal, s_lin_cal, lw=2.5, color='steelblue',
        label='HJ bound (annual data)')

for theta, marker, color, label in [
    (0.0,  'o', 'black',  'θ=0 (time-separable)'),
    (0.5,  's', 'green',  'θ=+0.5 (durability)'),
    (-0.5, '^', 'crimson','θ=−0.5 (habit)'),
]:
    pts = []
    for g in [0, -1, -2, -5, -8, -10, -14]:
        m_sim = simulate_nonseparable_imrs(gamma=g, theta=theta)
        if len(m_sim) < 100:
            continue
        Em = float(np.mean(m_sim))
        sm = float(np.std(m_sim))
        if 0.88 <= Em <= 1.12 and sm < 0.55:
            pts.append((Em, sm))
    if pts:
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], marker=marker, color=color,
                lw=1.2, ms=6, label=label)

ax.set_xlabel('$E(m)$', fontsize=12)
ax.set_ylabel('$\\sigma(m)$', fontsize=12)
ax.set_title('Nonseparable Preferences and the HJ Bound\n'
             '(monthly calibration, varying γ and θ)', fontsize=11)
ax.set_xlim([0.88, 1.12])
ax.set_ylim([-0.01, 0.55])
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
```

## How Many Assets Does the Bound Need?

One practical question the paper addresses (Section III.C) via the duality
result is: should we use all available assets to compute the bound, or can we
reduce dimensionality using **factor analysis**?

The answer:
- If asset pricing can be approximated by a **factor model** — payoffs have a
  factor structure and the pricing relation holds exactly for the factors —
  then the bound computed from the factors equals that from the full portfolio
  space.
- If factor pricing holds only approximately, information is lost by dimension
  reduction and the bound weakens.

We illustrate this below by comparing bounds computed from 2 assets vs. 8
instruments as in the paper (scaled returns using lagged instruments).

```{code-cell} ipython3
def add_instruments(returns, n_lags=2):
    """
    Expand the payoff space by scaling original returns by lagged values,
    mimicking the 8-asset construction in the paper.

    Returns array of shape (T - n_lags, k) where k = n_assets * (1 + n_lags).
    """
    T, n = returns.shape
    payoff_list = [returns[n_lags:]]
    for lag in range(1, n_lags + 1):
        for j in range(n):
            payoff_list.append(
                (returns[n_lags:, j] * returns[n_lags - lag : T - lag, j])[:, None]
            )
    return np.hstack(payoff_list)


# Simulate 2-asset economy
returns2, prices2, m_true2 = simulate_economy(T=10000, gamma=10.0, seed=99)

# Compute moments for 2 assets and 6 instruments (= 8-asset version)
returns8 = add_instruments(returns2, n_lags=3)
mu_q8    = np.hstack([np.ones(2), returns2[2:, 0], returns2[2:, 0],
                       returns2[3:, 1], returns2[3:, 1],
                       returns2[1:(len(returns8)+1), 0],
                       returns2[2:(len(returns8)+2), 1]])[:returns8.shape[1]]

mu_x2_, mu_q2_, Sigma2_ = compute_moments(returns2, prices2)
mu_x8_, mu_q8_, Sigma8_ = compute_moments(returns8)

v2, s2 = hj_bound_no_positivity(mu_x2_, mu_q2_, Sigma2_)
v8, s8 = hj_bound_no_positivity(mu_x8_, mu_q8_, Sigma8_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(v2, s2, lw=2, label='2 assets (no instruments)', color='steelblue')
ax.plot(v8, s8, lw=2, linestyle='--', label='8 assets (with instruments)',
        color='darkorange')
ax.scatter(np.mean(m_true2), np.std(m_true2), color='red', s=100, zorder=5,
           label='True IMRS (γ=10)')
ax.set_xlabel('$E(m)$')
ax.set_ylabel('$\\sigma(m)$')
ax.set_title('More Assets = Tighter Bound\n(instruments expand the SDF frontier)')
ax.set_xlim([0.88, 1.12])
ax.set_ylim([-0.01, 0.6])
ax.legend()
plt.tight_layout()
plt.show()
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
ax.plot(v_hist, s_hist, lw=2.5, color='steelblue', label='HJ bound (historical)')
ax.fill_betweenx(np.linspace(0, 0.6, 400), v_hist.min(), v_hist.max(),
                 alpha=0.07, color='steelblue')

cmap = plt.cm.Reds(np.linspace(0.4, 0.9, len(gammas_plot)))
for (g, E_m, s_m), c in zip(crra_moments, cmap):
    ax.scatter(E_m, s_m, color=c, s=90, zorder=5)
    ax.annotate(f'γ={g}', (E_m, s_m), xytext=(8, 3),
                textcoords='offset points', fontsize=9, color=c)

ax.set_xlabel('$E(m)$', fontsize=12)
ax.set_ylabel('$\\sigma(m)$', fontsize=12)
ax.set_title('HJ Bound vs. CRRA Model\n'
             'Annual U.S. data · δ=0.95 · varying γ', fontsize=11)
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
ax.set_title('Tighter HJ Bounds with More Assets/Instruments')
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
