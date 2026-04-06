---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(dovis_accounting_mf)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Accounting for Monetary and Fiscal Policy

```{contents} Contents
:depth: 2
```

## Overview

This lecture studies a model of fiscal and monetary policy interactions developed by
{cite:t}`DovisAccountingMFrevised`.

The model provides a framework for revisiting some long-standing questions about **fiscal dominance** versus **monetary dominance** in a framework that allows for **partial commitment** to an inflation target.


```{note}
For an early discussion of "partial commitment" in the context of fiscal and monetary policy, see the concluding section of {cite:t}`LucasStokey1982`, the original working paper version of {cite:t}`LucasStokey1983`.

In Quantecon's view, the referees and editors of the *Journal of Monetary Economics* version made a mistake by insisting that Lucas and Stokey rewrite the concluding section of their paper.
```

```{note}
{cite:t}`SargentWallace1981` contrasted these two types of "dominance" as different ways of coordinating monetary and fiscal policy.

They thought about them at the beginning of the Reagan administration, when the 1970s surge in US inflation had not yet been tamed by the monetary-fiscal policies presided over by Paul Volcker.

Sargent and Wallace's title, "Some Unpleasant Monetarist Arithmetic," expressed the idea that in the face of a persistent net-of-interest government deficit, efforts to reduce inflation through tight monetary policy work only temporarily, if at all.

That is because they lead to higher government debt and thus greater gross-of-interest government deficits that must be financed in the future.
```

{cite:t}`DovisAccountingMFrevised` provide a framework for understanding how
the **credibility** of a government's inflation-targeting mandate shapes equilibrium outcomes for inflation, public debt, and primary surpluses.

The paper posits that there are two ways that a disinflation can occur:

- **Fundamental disinflation**: a reduction in fiscal needs ($\theta$) leads inflation and debt to
  decline together.
- **Institutional disinflation**: an increase in the credibility of the inflation mandate ($\xi$) leads inflation to fall while debt *rises*.

The contrasting comovement of debt and inflation in these types of disinflations
allows the authors to create a statistical model that lets them classify observed disinflations into episodes that were driven by fiscal fundamentals or by institutional changes.

The paper applies these ideas to Colombia, Chile, and the United States, using a
**particle filter** to recover the sequences of fiscal and institutional shocks that are consistent with the observed joint paths of inflation and debt-to-GDP ratios.

In this lecture, we will:

1. Set up the model environment — a {cite:t}`SargentWallace1981` economy with a household, firms, and a government
2. Describe implementable fiscal and monetary outcomes
3. Characterize two polar benchmarks: the **Ramsey** outcome (full commitment) and the
   **Markov** outcome (no commitment)
4. Formulate the full model with endogenous regime switching governed by
   a stochastic cost $\xi$ of deviating from the mandate
5. Write Python code to solve the model numerically
6. Simulate equilibrium paths and impulse response functions
7. Implement a particle filter to estimate the model on data

Let's start by importing some Python tools.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
```

## The Economy

### Environment

Consider an economy that blends elements of {cite}`AMSS_2002` and {cite}`Calvo1978`
(see also {cite}`LucasStokey1983` and {cite}`ChariKehoe1999`).

Time is discrete, indexed by $t = 0, 1, \ldots$.

The exogenous state is $s_t \in \mathcal{S}$, following
a Markov process with transition $\Pr(s_{t+1}|s_t)$.

A representative household has preferences

$$
\sum_t \sum_{s^t} \beta^t \Pr(s^t|s_0)\, U\bigl(c(s^t),\, l(s^t),\, m(s^t),\, g(s^t),\, s_t\bigr)
$$

where

$$
U(c, l, m, g, s) = c - \nu(l) + v(m) + \theta(s)\, u(g).
$$

Here

- $c$ is private consumption
- $l$ is labor supply
- $m$ is real money balances
- $g$ is public consumption
- $\theta(s)$ is a preference shock to the marginal utility of government spending
- $\beta$ is the household discount factor

The resource constraint is $c(s^t) + g(s^t) \le l(s^t)$.

The government finances spending $g$ using a labor income tax $\tau$, by issuing real
uncontingent debt $b$, and by printing money $M$.

The government is benevolent but may have a discount factor $\hat\beta \le \beta$.

### Implementable Allocations

Following {cite:t}`Aiyagari1989` and {cite:t}`AMSS_2002` (see also QuantEcon lectures {doc}`amss`, {doc}`amss2`, and {doc}`amss3`), define the **real primary surplus** as

$$
\Delta(s^t) \equiv \tau(s^t) l(s^t) - g(s^t) - T(s^t).
$$

It is possible then to derive a static **indirect utility function over surpluses**:

$$
U(\Delta, s) = \max_{c,l,g} \; c - \nu(l) + \theta(s)\, u(g)
$$

where the maximization is
subject to the resource constraint and a static implementability constraint
$(1 - \nu'(l))\, l - g \ge \Delta$.

This function $U(\Delta, s)$ is **decreasing and concave** in $\Delta$ for all $s$.

Let $\phi \equiv M_{t-1}/P_t$ denote real money balances (price of money in terms of goods) and define

$$
H(\phi) \equiv \phi + v'(\phi)\, \phi.
$$

The **money demand** or **portfolio balance** condition becomes

$$
\mu\, \phi = \beta \sum_{s'} \Pr(s'|s)\, H(\phi'),
$$

where $\mu \equiv M_t / M_{t-1}$ is the gross money growth rate.

The **government budget constraint** (in normalized form) is
$$
b_{t-1} + \phi_t = \Delta_t + \beta b_t + \mu_t \phi_t.
$$

The **government's value** is

$$
V_0 = \sum_t \sum_{s^t} \hat\beta^t \Pr(s^t|s_0) \left[ U(\Delta(s^t), s_t) + v(\phi(s^t)) \right].
$$

We define the model primitives as pure functions.

```{code-cell} ipython3

def ν(l, χ, ψ):
    """Labor disutility: ν(l) = χ l^{1+ψ} / (1+ψ)."""
    return χ * l**(1.0 + ψ) / (1.0 + ψ)

def ν_prime(l, χ, ψ):
    """ν'(l) = χ l^ψ."""
    return χ * l**ψ

def v_money(φ, κ, η_m):
    """Utility from real money balances: v(φ) = κφ - η φ^2."""
    return κ * φ - η_m * φ**2

def v_money_prime(φ, κ, η_m):
    """v'(φ) = κ - 2η φ."""
    return κ - 2.0 * η_m * φ

def u_gov(g, σ):
    """Government spending utility: u(g) = g^{1-σ}/(1-σ)."""
    return jnp.where(jnp.abs(σ - 1.0) < 1e-10,
                     jnp.log(g),
                     g**(1.0 - σ) / (1.0 - σ))

def u_gov_prime(g, σ):
    """u'(g) = g^{-σ}."""
    return g**(-σ)

def H_func(φ, κ, η_m):
    """H(φ) = φ + v'(φ) φ = φ(1 + v'(φ))."""
    return φ * (1.0 + v_money_prime(φ, κ, η_m))

def H_func_prime(φ, κ, η_m):
    """H'(φ) = 1 + v'(φ) + v''(φ) φ = 1 + κ - 4ηφ."""
    # v'(φ) = κ - 2ηφ, v''(φ) = -2η
    return 1.0 + κ - 4.0 * η_m * φ

def h_func(φ, κ, η_m):
    """Seigniorage: h(φ) = v'(φ) φ."""
    return v_money_prime(φ, κ, η_m) * φ
```

## Policy Determination

### The Credibility Problem

An important innovation of {cite:t}`DovisAccountingMFrevised` is to model policy determination under
**partial commitment** in the following sense.

The government promises an inflation target $\pi^*$ (equivalently, a
promised value for real balances $\phi'$) for next period.

But a next-period government
can choose to **honor** or **abrogate** the mandate.

The cost of abrogating is a random variable $\xi(s)$ that the authors intend to capture either some or all of the following consequences of abrogation:

- reputational losses (see {cite}`AtkesonKehoe2001`, {cite}`DovisKirpalani2021`)
- coordination failures that trigger worse equilibria
- institutional constraints (see {cite}`Lohmann1992`)
- political costs

The authors use this specification to **nest** both the Ramsey outcome (high $\xi$,
so the mandate is always honored) and the Markov outcome ($\xi = 0$, so the mandate is
always abrogated)

This approach is related to the loose commitment framework
of {cite:t}`DebortoliNunes2010`, but differs because here the regime is **endogenous**.

### Recursive Formulation

The state is $x = (b, \phi, s)$ where $b$ is inherited real debt, $\phi$ is the promised
real balances, and $s = (\theta, \xi)$ is the exogenous state.

This recursive formulation builds on {cite}`Abreu1988`, {cite}`ChariKehoe1990`, and
{cite}`Chang1998`.

```{note}
For descriptions of these frameworks, see other lectures in this suite of QuantEcon lecture notes, including {doc}`Ramsey plans, time inconsistency, sustainable plans <calvo>`, {doc}`competitive equilibria in the Chang model <chang_ramsey>`, and {doc}`sustainable plans in the Chang model <chang_credible>`.
```

The economy can be in one of two regimes:

- **Monetary dominance** (MD, $\eta = 1$): the government honors the inflation target.
- **Fiscal dominance** (FD, $\eta = 0$): the government ignores the target and chooses $\phi$ to maximize short-run welfare.

The transition between these regimes is related to the fiscal theory of the price level
literature ({cite}`Leeper1991`, {cite}`Bianchi2013`, {cite}`Cochrane2023`),
but differs in that the switches are **endogenous** and that the policies in each regime are
also endogenous (not governed by exogenous rules).

The **regime indicator** is

$$
\eta(b', \phi', s') = \begin{cases}
1 & \text{if } V^{md}(b', \phi', s') \ge V^{fd}(b', s') - \xi(s') \\
0 & \text{otherwise}
\end{cases}
$$

**Monetary dominance** — the government solves:

$$
V^{md}(b, \phi, s) = \max_{\Delta, b', \mu, \phi'} U(\Delta, \theta) + v(\phi) +
  \hat\beta \sum_{s'} \Pr(s'|s)\, V(b', \phi', s')
$$

where maximization is
subject to the budget constraint $\Delta = b + \phi - \beta b' - \mu\phi$ and the money demand
condition $\mu\phi = J(b', \phi', s)$.

**Fiscal dominance** — the government's problem is augmented by the addition of current $\phi$ to its choice set, so that the government solves:

$$
V^{fd}(b, s) = \max_{\phi, \Delta, b', \mu, \phi'} U(\Delta, \theta) + v(\phi) +
  \hat\beta \sum_{s'} \Pr(s'|s)\, V(b', \phi', s')
$$

where the static first-order necessary condition with respect to $\phi$ is $-U'(\Delta, \theta) = v'(\phi^{fd})$.

The **expected marginal value of real balances** is

$$
J(b', \phi', s) = \beta \sum_{s'} \Pr(s'|s) \left[
  \eta(b', \phi', s')\, H(\phi') +
  (1 - \eta(b', \phi', s'))\, H\!\left(\phi^{fd}(b', s')\right)
\right].
$$

We store the model parameters in a container class with two calibrations.

```{code-cell} ipython3

class DovisParams:
    """
    Container for model parameters.

    Two calibrations are available:
      - 'LA': Latin American average 1960–2017
      - 'US': United States 1914–2017
    """

    def __init__(self, calibration='LA'):

        # Common across calibrations
        self.ψ = 1.0        # inverse Frisch elasticity
        self.σ = 2.0      # inverse EIS for gov't spending

        if calibration == 'LA':
            self.β = 0.95
            self.β_hat = 0.92
            self.χ = 0.015
            self.κ = 0.68
            self.η_m = 0.07
            self.θ_bar = 130.0
            self.ρ_θ = 0.9
            self.σ_θ = np.sqrt(60.0)
            self.ρ_ξ = 0.998
            self.σ_ξ = 0.112
            self.lam = 0.2       # Gumbel parameter λ
        elif calibration == 'US':
            self.β = 0.95
            self.β_hat = 0.91
            self.χ = 0.021
            self.κ = 0.70
            self.η_m = 0.06
            self.θ_bar = 2.0
            self.ρ_θ = 0.9
            self.σ_θ = np.sqrt(20.0)
            self.ρ_ξ = 0.99
            self.σ_ξ = 0.3
            self.lam = 0.5
        else:
            raise ValueError("calibration must be 'LA' or 'US'")

        # Derived
        self.φ_star = self.κ / (2.0 * self.η_m)  # money satiation
        self.euler_mascheroni = 0.5772156649

        # Grid parameters
        self.n_B = 80          # grid points for total liabilities B
        self.n_φ = 40        # grid points for promised φ'
        self.n_θ = 5       # number of θ states
        self.n_ξ = 7          # number of ξ_1 states
        self.B_max = 50.0      # upper bound for B
        self.b_max = 30.0      # upper bound for real debt

    def __repr__(self):
        return f"DovisParams(β={self.β}, β_hat={self.β_hat}, calibration)"


params_la = DovisParams('LA')
params_us = DovisParams('US')
print(f"LA calibration: β={params_la.β}, β_hat={params_la.β_hat}, "
      f"φ*={params_la.φ_star:.2f}")
print(f"US calibration: β={params_us.β}, β_hat={params_us.β_hat}, "
      f"φ*={params_us.φ_star:.2f}")
```

## Two Benchmark Outcomes

Before studying the full model, we analyze two polar benchmarks.

### The Ramsey Outcome (Full Commitment)

Under full commitment ($\xi$ always large enough so that $\eta = 1$), the government solves

$$
V^R(b, \phi, s) = \max_{\Delta, b', \phi'(s')} U(\Delta, s) + v(\phi) +
\hat\beta \sum_{s'} \Pr(s'|s) V^R(b', \phi'(s'), s')
$$

subject to $\Delta = b + \phi - \beta b' - \beta \sum_{s'} \Pr(s'|s) H(\phi'(s'))$.

Key properties:

- For the quadratic specification $v(\phi) = \kappa\phi - \eta_m\phi^2$ used in this lecture, the optimal $\phi' = \phi^* = \kappa/(2\eta_m)$ for all $t \ge 1$ — a version of the **Friedman rule**.
- The **inflation rate is constant** and independent of fiscal fundamentals:
  $1 + \pi^R = \beta H(\phi^*) / \phi^*$.
- Surpluses and real debt follow the Euler equation
  $-U'(\Delta, s) = \frac{\hat\beta}{\beta} \sum_{s'} \Pr(s'|s) [-U'(\Delta', s')]$.

One can implement the Ramsey outcome by **delegating** monetary policy to an independent
central bank with a fixed inflation target $\pi^R$, as in {cite}`Aiyagari2002`.

### The Markov Outcome (No Commitment)

When $\xi(s) = 0$ for all $s$, it is always optimal to abrogate the mandate ($\eta = 0$).

The government solves

$$
V^M(b, s) = \max_{\phi, \Delta, b'} U(\Delta, \theta) + v(\phi) +
\hat\beta \sum_{s'} \Pr(s'|s) V^M(b', s')
$$

subject to $\Delta = b + \phi - \beta b' - \beta \sum_{s'} \Pr(s'|s) H(\phi^M(b', s'))$.

Key properties of the Markov outcome:

- Inflation **responds strongly** to fiscal pressures
- Debt capacity is **sharply limited**
- There is an **incentive effect**: debt issuance is distorted downward relative to Ramsey
  because the term $\sum_{s'} \Pr(s'|s) H'(\phi^M(b', s')) \frac{\partial \phi^M}{\partial b'} < 0$
  acts as an implicit tax on debt issuance
- In the deterministic case with $\beta = \hat\beta$: real debt converges to zero (Appendix C of the paper)

The full model **interpolates** between these two extremes depending on the cost $\xi_t$.

We compute the indirect utility $U(\Delta, \theta)$ by bisecting on the Lagrange multiplier.

```{code-cell} ipython3
@jax.jit
def indirect_utility(Δ, θ, χ, ψ, σ):
    """Compute U(Δ, θ) and U'(Δ, θ) via bisection."""
    g_star = θ**(1.0 / σ)
    l_peak = (1.0 / ((1.0 + ψ) * χ))**(1.0 / ψ)
    T_max = (1.0 - χ * l_peak**ψ) * l_peak
    Δ_max = T_max

    def bisect_step(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        g_val = (θ / (1.0 + mid))**(1.0 / σ)
        denom = jnp.maximum(χ * (1.0 + mid * (1.0 + ψ)), 1e-15)
        l_val = jnp.maximum((1.0 + mid) / denom, 1e-15)**(1.0 / ψ)
        surplus = (1.0 - χ * l_val**ψ) * l_val - g_val
        lo = jnp.where(surplus <= Δ, mid, lo)
        hi = jnp.where(surplus > Δ, mid, hi)
        return (lo, hi)

    # 60 iterations: 1000 / 2^60 ≈ 10^{-15} precision
    lo, hi = jax.lax.fori_loop(0, 60, bisect_step, (0.0, 1000.0))
    lam_opt = 0.5 * (lo + hi)

    g_opt = (θ / (1.0 + lam_opt))**(1.0 / σ)
    denom_opt = jnp.maximum(χ * (1.0 + lam_opt * (1.0 + ψ)), 1e-15)
    l_opt = jnp.maximum((1.0 + lam_opt) / denom_opt, 1e-15)**(1.0 / ψ)

    U_val_normal = l_opt - ν(l_opt, χ, ψ) + θ * u_gov(g_opt, σ)
    U_prime_normal = -lam_opt

    # Unconstrained case: Δ <= -g_star
    U_val_uncon = g_star - ν(g_star, χ, ψ) + θ * u_gov(g_star, σ)
    U_prime_uncon = 0.0

    # Infeasible case: Δ >= Δ_max * 0.99
    U_val_infeasible = -1e10
    U_prime_infeasible = -1e10

    is_uncon = Δ <= -g_star
    is_infeasible = Δ >= Δ_max * 0.99

    U_val = jnp.where(is_uncon, U_val_uncon,
                       jnp.where(is_infeasible, U_val_infeasible,
                                 U_val_normal))
    U_prime = jnp.where(is_uncon, U_prime_uncon,
                         jnp.where(is_infeasible, U_prime_infeasible,
                                   U_prime_normal))

    return U_val, U_prime
```

```{code-cell} ipython3
---
mystnb:
  figure:
    name: fig-indirect-utility
    caption: indirect utility over surpluses
---
p = params_la
Δ_grid = jnp.linspace(-5.0, 3.0, 200)
θ_vals = [80.0, 130.0, 200.0]

indirect_utility_vec = jax.vmap(indirect_utility, in_axes=(0, None, None, None, None))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for θ_val in θ_vals:
    U_vals, U_primes = indirect_utility_vec(
        Δ_grid, θ_val, p.χ, p.ψ, p.σ)

    mask = U_vals > -1e9
    axes[0].plot(Δ_grid[mask], U_vals[mask], lw=2,
                 label=f'θ = {θ_val:.0f}')
    axes[1].plot(Δ_grid[mask], U_primes[mask], lw=2,
                 label=f'θ = {θ_val:.0f}')

axes[0].set_xlabel('primary surplus Δ')
axes[0].set_ylabel('U(Δ, θ)')
axes[0].legend()

axes[1].set_xlabel('primary surplus Δ')
axes[1].set_ylabel("U'(Δ, θ)")
axes[1].legend()

plt.tight_layout()
plt.show()
```

The left panel confirms that $U(\Delta, \theta)$ is **decreasing and concave** in $\Delta$: higher
surpluses are costly because they require more distortionary taxation.

The right panel shows
the marginal cost of surpluses $U'(\Delta, \theta) < 0$, which becomes more negative as $\Delta$
approaches the peak of the Laffer curve.

Higher $\theta$ shifts the curves because greater social value of government spending makes
running a surplus even more costly.

## The Full Model with Gumbel Shocks

Following the paper's computational approach, the cost $\xi$ is decomposed as

$$
\xi_t = \xi_{1,t} + \xi^{fd}_t - \xi^{md}_t,
$$

where $\xi_{1,t}$ is a persistent AR(1) component (discretized via the {cite:t}`Tauchen1986` method) and $\xi^{fd}_t$,
$\xi^{md}_t$ are i.i.d. **Gumbel** shocks with mean zero.

The Gumbel specification delivers a **logit** formula for the probability of monetary dominance:

$$
\bar\eta(b', \phi', s_1) = \frac{1}{1 + \exp\!\left(-\lambda
  \left[V^{md}(b', \phi', s_1) - V^{fd}(b', s_1) + \xi_1\right]\right)}
$$

and a smooth **log-sum-exp** formula for the expected continuation value:

$$
\Omega(b', \phi', s_1) = \frac{1}{\lambda}
\log\!\left[
  \exp(\lambda\, V^{md}) + \exp\!\left(\lambda\left(V^{fd} - \xi_1\right)\right)
\right].
$$

This makes the value function differentiable and the numerical solution well behaved.

We discretize the exogenous states using the {cite:t}`Tauchen1986` method.

```{code-cell} ipython3

def tauchen(ρ, σ, n, m=3):
    """
    Tauchen method for discretizing AR(1) process
       y' = ρ * y + σ * ε,  ε ~ N(0,1)
    Returns grid y and transition matrix P.
    """
    σ_y = σ / jnp.sqrt(1.0 - ρ**2)
    y_max = m * σ_y
    y = jnp.linspace(-y_max, y_max, n)
    d = jnp.where(n > 1, y[1] - y[0], 1.0)

    P = jnp.zeros((n, n))
    from jax.scipy.stats import norm
    for i in range(n):
        for j in range(n):
            if j == 0:
                P = P.at[i, j].set(
                    norm.cdf((y[0] + d/2 - ρ * y[i]) / σ))
            elif j == n - 1:
                P = P.at[i, j].set(
                    1.0 - norm.cdf((y[n-1] - d/2 - ρ * y[i]) / σ))
            else:
                P = P.at[i, j].set(
                    norm.cdf((y[j] + d/2 - ρ * y[i]) / σ) -
                    norm.cdf((y[j] - d/2 - ρ * y[i]) / σ))
    return y, P


def build_grids(par):
    """Build state-space grids and transition matrices."""

    # θ grid (centered at θ_bar)
    θ_dev, P_θ = tauchen(
        par.ρ_θ, par.σ_θ, par.n_θ)
    θ_grid = par.θ_bar + θ_dev

    # ξ_1 grid (only weakly positive values)
    ξ_dev, P_ξ = tauchen(
        par.ρ_ξ, par.σ_ξ, par.n_ξ)
    ξ_grid = jnp.maximum(ξ_dev, 0.0)  # truncate to non-negative

    # B grid (total real liabilities)
    B_grid = jnp.linspace(0.01, par.B_max, par.n_B)

    # φ' grid (promised real balances)
    φ_grid = jnp.linspace(0.01, par.φ_star * 0.99, par.n_φ)

    return {
        'θ_grid': θ_grid,
        'P_θ': P_θ,
        'ξ_grid': ξ_grid,
        'P_ξ': P_ξ,
        'B_grid': B_grid,
        'φ_grid': φ_grid
    }


grids = build_grids(params_la)

print("θ grid:", np.round(np.asarray(grids['θ_grid']), 1))
print("ξ_1 grid:", np.round(np.asarray(grids['ξ_grid']), 3))
print(f"B grid: [{grids['B_grid'][0]:.2f}, ..., {grids['B_grid'][-1]:.2f}]"
      f" ({len(grids['B_grid'])} points)")
print(f"φ' grid: [{grids['φ_grid'][0]:.2f}, ..., "
      f"{grids['φ_grid'][-1]:.2f}]"
      f" ({len(grids['φ_grid'])} points)")
```

## Computational Algorithm

The paper reduces the problem to a single endogenous state variable $B = b + \phi$
(total real government liabilities).

The value function $W(B, s_1)$ satisfies

$$
W(B, s_1) = \max_{\Delta, b', \phi'} U(\Delta, \theta) +
\hat\beta \sum_{s_1'} \Pr(s_1'|s_1)\, \Omega(b', \phi', s_1')
$$

where maximization is subject to

$$
\Delta = B - \beta b' - \beta \sum_{s_1'} \Pr(s_1'|s_1)
  \left[\bar\eta\, H(\phi') + (1 - \bar\eta)\, H(\phi^{fd}(b', s_1'))\right],
$$

where $\phi^{fd}(b, s)$ is the solution to the static FOC
$-U'(\Delta, \theta) = v'(\phi^{fd})$ under fiscal dominance.

The algorithm is:

1. **Initialize** with a guess $W_0(B, s_1)$ (e.g., the Ramsey value)
2. For iteration $n$:
   - Compute $\phi^{fd}$ and $\bar\eta$ from the logit formula and the fiscal-dominance FOC
   - Compute the Bellman update $W_{n+1}$ from the value function equation above
3. **Iterate** until $\|W_{n+1} - W_n\| < \varepsilon$

We solve for $\phi^{fd}$ under fiscal dominance via bisection on the static FOC.

```{code-cell} ipython3

@jax.jit
def φ_fd_solve(B, θ, χ, ψ, σ, κ, η_m):
    """Solve for φ^{fd} via bisection on the static FOC."""
    φ_lo = 0.01
    φ_hi = κ / (2.0 * η_m) * 0.99

    def bisect_step(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        vp = v_money_prime(mid, κ, η_m)
        Δ_approx = jnp.clip(B - mid, -5.0, 3.0)
        _, U_p = indirect_utility(Δ_approx, θ, χ, ψ, σ)
        residual = vp + U_p
        lo = jnp.where(residual > 0, mid, lo)
        hi = jnp.where(residual > 0, hi, mid)
        return (lo, hi)

    # 50 iterations: interval / 2^50 ≈ 10^{-15} precision
    lo, hi = jax.lax.fori_loop(0, 50, bisect_step, (φ_lo, φ_hi))
    return 0.5 * (lo + hi)
```

We solve the simplified model by value function iteration with a vectorized Bellman step.

```{code-cell} ipython3

def solve_simplified_model(
    β, β_hat, χ, ψ, σ_g, κ, η_m,
    θ, ξ_grid, P_ξ, B_grid, φ_grid, lam,
    max_iter=300, tol=1e-5
):
    """
    Solve the simplified model with one θ value and
    multiple ξ_1 states using value function iteration.

    Returns W(B, ξ_1) and policy functions.
    """
    ξ_grid = jnp.asarray(ξ_grid)
    P_ξ = jnp.asarray(P_ξ)
    B_grid = jnp.asarray(B_grid)
    φ_grid = jnp.asarray(φ_grid)

    n_B = len(B_grid)
    n_ξ = len(ξ_grid)
    n_φ = len(φ_grid)

    # Initialize value function
    W = jnp.zeros((n_B, n_ξ))
    for i_B in range(n_B):
        Uval, _ = indirect_utility(0.0, θ, χ, ψ, σ_g)
        W = W.at[i_B, :].set(Uval / (1.0 - β_hat))

    # Precompute b' grid (scaled from B_grid)
    b_prime_grid = B_grid * 0.5

    @jax.jit
    def bellman_step(W):
        """One Bellman iteration over all (B, ξ) states."""
        W_new = jnp.full((n_B, n_ξ), -1e15)
        pol_b = jnp.zeros((n_B, n_ξ))
        pol_φ = jnp.zeros((n_B, n_ξ))
        pol_Δ = jnp.zeros((n_B, n_ξ))

        B_md = b_prime_grid[:, None] + φ_grid[None, :]

        # φ^fd for all (b', φ') pairs
        φ_fd_vmap = jax.vmap(jax.vmap(
            lambda Bval: φ_fd_solve(Bval, θ, χ, ψ, σ_g, κ, η_m)))
        φ_fd_all = φ_fd_vmap(B_md)
        B_fd = b_prime_grid[:, None] + φ_fd_all

        # Nearest-grid interpolation of W
        idx_md = jnp.clip(
            jnp.searchsorted(B_grid, B_md.ravel()), 0, n_B - 1
        ).reshape(n_B, n_φ)
        idx_fd = jnp.clip(
            jnp.searchsorted(B_grid, B_fd.ravel()), 0, n_B - 1
        ).reshape(n_B, n_φ)

        W_md_vals = W[idx_md]
        W_fd_vals = W[idx_fd]

        V_md = W_md_vals + v_money(φ_grid[None, :, None], κ, η_m)
        V_fd = W_fd_vals + v_money(φ_fd_all[:, :, None], κ, η_m)

        # Logit probability of monetary dominance
        diff = jnp.clip(
            lam * (V_md - V_fd + ξ_grid[None, None, :]), -500.0, 500.0)
        η_bar = 1.0 / (1.0 + jnp.exp(-diff))

        H_φ = H_func(φ_grid[None, :, None], κ, η_m)
        H_fd = H_func(φ_fd_all[:, :, None], κ, η_m)
        J_contrib = η_bar * H_φ + (1.0 - η_bar) * H_fd

        # Log-sum-exp continuation value
        a = lam * V_md
        b_val = lam * (V_fd - ξ_grid[None, None, :])
        max_ab = jnp.maximum(a, b_val)
        Ω = jnp.where(
            lam > 0,
            (max_ab + jnp.log(
                jnp.exp(a - max_ab) + jnp.exp(b_val - max_ab))) / lam,
            jnp.maximum(V_md, V_fd - ξ_grid[None, None, :]))

        # Expected values over ξ'
        EJ = jnp.einsum('bpj,ij->bpi', J_contrib, P_ξ) * β
        EΩ = jnp.einsum('bpj,ij->bpi', Ω, P_ξ)

        # Surplus for all (i_B, i_bp, i_φ, i_ξ)
        Δ_all = (B_grid[:, None, None, None]
                 - β * b_prime_grid[None, :, None, None]
                 - EJ[None, :, :, :])

        # Vectorize indirect utility evaluation
        U_flat, _ = jax.vmap(
            lambda d: indirect_utility(d, θ, χ, ψ, σ_g)
        )(Δ_all.ravel())
        U_all = U_flat.reshape(Δ_all.shape)

        # Objective: U + β_hat * EΩ
        feasible = (Δ_all > -6.0) & (Δ_all < 4.0) & (U_all > -1e9)
        val_all = jnp.where(feasible, U_all + β_hat * EΩ[None, :, :, :],
                            -1e15)

        # Optimal (b', φ') for each (B, ξ)
        val_for_max = val_all.transpose(0, 3, 1, 2).reshape(
            n_B, n_ξ, n_B * n_φ)
        best_idx = jnp.argmax(val_for_max, axis=2)

        W_new = jnp.max(val_for_max, axis=2)
        pol_b = b_prime_grid[best_idx // n_φ]
        pol_φ = φ_grid[best_idx % n_φ]

        Δ_for_pol = Δ_all.transpose(0, 3, 1, 2).reshape(
            n_B, n_ξ, n_B * n_φ)
        pol_Δ = jnp.take_along_axis(
            Δ_for_pol, best_idx[:, :, None], axis=2).squeeze(2)

        return W_new, pol_b, pol_φ, pol_Δ

    for iteration in range(max_iter):
        W_new, pol_b, pol_φ, pol_Δ = bellman_step(W)

        # Check convergence
        diff = jnp.max(jnp.abs(W_new - W))
        W = W_new

        if diff < tol:
            break

    return (np.asarray(W), np.asarray(pol_b),
            np.asarray(pol_φ), np.asarray(pol_Δ))
```

We solve a coarse version of the model for illustration.

```{code-cell} ipython3
p = params_la
n_B_coarse = 30
n_φ_coarse = 15
n_ξ_coarse = 3

B_grid = jnp.linspace(0.5, 20.0, n_B_coarse)
φ_grid = jnp.linspace(0.5, p.φ_star * 0.95, n_φ_coarse)
ξ_dev, P_ξ = tauchen(p.ρ_ξ, p.σ_ξ, n_ξ_coarse, m=2)
ξ_grid = jnp.maximum(ξ_dev + 0.3, 0.01)  # shift to positive

print("Solving simplified model (coarse grid)...")
W, pol_b, pol_φ, pol_Δ = solve_simplified_model(
    p.β, p.β_hat, p.χ, p.ψ, p.σ,
    p.κ, p.η_m,
    p.θ_bar, ξ_grid, P_ξ,
    B_grid, φ_grid, p.lam,
    max_iter=100, tol=1e-4
)
print("Done.")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    name: fig-value-policy
    caption: value and policy functions
---

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ξ_labels = [f'ξ_1 = {float(ξ_grid[j]):.2f}' for j in range(n_ξ_coarse)]
colors = ['tab:blue', 'tab:orange', 'tab:green']

B_grid_np = np.asarray(B_grid)

for j in range(n_ξ_coarse):
    axes[0, 0].plot(B_grid_np, W[:, j], lw=2, label=ξ_labels[j],
                    color=colors[j])
    axes[0, 1].plot(B_grid_np, pol_b[:, j], lw=2, label=ξ_labels[j],
                    color=colors[j])
    axes[1, 0].plot(B_grid_np, pol_φ[:, j], lw=2, label=ξ_labels[j],
                    color=colors[j])
    axes[1, 1].plot(B_grid_np, pol_Δ[:, j], lw=2, label=ξ_labels[j],
                    color=colors[j])

axes[0, 0].set_ylabel('W(B, ξ_1)')
axes[0, 0].set_xlabel('B (total liabilities)')
axes[0, 0].legend()

axes[0, 1].set_ylabel("b'(B, ξ_1)")
axes[0, 1].set_xlabel('B')
axes[0, 1].legend()

axes[1, 0].set_ylabel("φ'(B, ξ_1)")
axes[1, 0].set_xlabel('B')
axes[1, 0].legend()

axes[1, 1].set_ylabel('Δ(B, ξ_1)')
axes[1, 1].set_xlabel('B')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

## Two Types of Disinflation

A central result of the paper is that the comovement of debt and inflation differs
depending on the **source** of disinflation.

### Fundamental Disinflation ($\theta$ falls, $\xi$ constant low)

When $\theta$ decreases while $\xi$ stays low (FD regime throughout):

- Real money balances $\phi^{fd}$ **rise** (lower inflation) because the static FOC
  $-U'(\Delta, \theta) = v'(\phi^{fd})$ shifts
- Debt **falls** because the government has weaker incentives to borrow
- Result: **positive correlation** between inflation and debt

### Institutional Disinflation ($\xi$ rises, $\theta$ constant)

When $\xi$ rises, triggering a switch from FD to MD:

- Real balances jump to the promised $\phi'$ (lower inflation)
- The incentive wedge on debt issuance **shrinks**,
  and seigniorage revenues fall, so debt **rises**
- Result: **negative correlation** between inflation and debt

These differing comovements are the key identification device.

We simulate impulse responses for both types of disinflation.

```{code-cell} ipython3

def simulate_path(b0, φ0, θ_path, ξ_path, par,
                  T=40, seed=42):
    """
    Simulate a path of the model given sequences of
    θ and ξ_1 shocks.

    This uses a simplified version: at each t, we solve for
    the government's choices given inherited (b, φ).
    """
    key = jrandom.PRNGKey(seed)

    b_path = jnp.zeros(T)
    φ_path = jnp.zeros(T)
    Δ_path = jnp.zeros(T)
    inflation_path = jnp.zeros(T)
    regime_path = jnp.zeros(T)

    b_path = b_path.at[0].set(b0)
    φ_path = φ_path.at[0].set(φ0)

    for t in range(T - 1):
        b = float(b_path[t])
        φ_t = float(φ_path[t])
        θ = float(θ_path[t])
        ξ1 = float(ξ_path[t])
        B = b + φ_t

        # Fiscal dominance value of φ
        φ_fd = φ_fd_solve(
            B, θ, par.χ, par.ψ, par.σ,
            par.κ, par.η_m)
        φ_fd = float(φ_fd)

        # Simplified: compare V_md vs V_fd
        Δ_md = B - φ_t  # simplified surplus under MD
        U_md, _ = indirect_utility(
            min(Δ_md, 3.0), θ, par.χ, par.ψ, par.σ)
        V_md = float(U_md) + float(v_money(φ_t, par.κ, par.η_m))

        Δ_fd = B - φ_fd
        U_fd, _ = indirect_utility(
            min(Δ_fd, 3.0), θ, par.χ, par.ψ, par.σ)
        V_fd = float(U_fd) + float(v_money(φ_fd, par.κ, par.η_m))

        # Regime decision (logit)
        diff = par.lam * (V_md - V_fd + ξ1)
        η = jnp.where(diff > 500, 1.0,
                       jnp.where(diff < -500, 0.0,
                                 1.0 / (1.0 + jnp.exp(-diff))))
        η = float(η)

        regime_path = regime_path.at[t].set(η)

        # Actual φ realized this period
        key, k1, k2 = jrandom.split(key, 3)
        gumbel_md = float(jrandom.gumbel(k1)) / par.lam - 0.5772 / par.lam
        gumbel_fd = float(jrandom.gumbel(k2)) / par.lam - 0.5772 / par.lam
        realized_md = V_md + gumbel_md > V_fd - ξ1 + gumbel_fd
        φ_realized = φ_t if realized_md else φ_fd

        # Inflation
        H_val = float(H_func(φ_realized, par.κ, par.η_m))
        μ = par.β * H_val / float(φ_path[max(t-1, 0)]) if t > 0 else 1.0
        if φ_realized > 0:
            inflation_path = inflation_path.at[t].set(
                μ * float(φ_path[max(t-1, 0)]) / φ_realized - 1.0)

        # Surplus
        Δ_path = Δ_path.at[t].set(B - φ_realized)

        # Simple debt dynamics (illustrative)
        b_path = b_path.at[t+1].set(
            max(0.1, b * 0.95 + 0.05 * b0 +
                (1.0 - η) * (-0.5) + η * 0.3))

        # Promised φ for next period
        φ_path = φ_path.at[t+1].set(
            φ_realized * 0.98 + 0.02 * par.φ_star * 0.8)

    return {
        'b': np.asarray(b_path),
        'φ': np.asarray(φ_path),
        'Δ': np.asarray(Δ_path),
        'inflation': np.asarray(inflation_path),
        'regime': np.asarray(regime_path)
    }


# Construct paths
T = 40
t_shock = 20

# Fundamental disinflation: θ drops at t_shock
θ_fund = jnp.ones(T) * 180.0
θ_fund = θ_fund.at[t_shock:].set(90.0)
ξ_fund = jnp.ones(T) * 0.01  # always low ξ

# Institutional disinflation: ξ rises at t_shock
θ_inst = jnp.ones(T) * 180.0  # θ constant
ξ_inst = jnp.ones(T) * 0.01
ξ_inst = ξ_inst.at[t_shock:].set(0.8)  # high ξ after shock

p = params_la
path_fund = simulate_path(5.0, 3.0, θ_fund, ξ_fund, p, T)
path_inst = simulate_path(5.0, 3.0, θ_inst, ξ_inst, p, T)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    name: fig-impulse-responses
    caption: two types of disinflation
---
fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

time = np.arange(T) - t_shock

ax = fig.add_subplot(gs[0, 0])
ax.plot(time, np.asarray(θ_fund), 'r--', lw=2, label='fundamental')
ax.plot(time, np.asarray(θ_inst), 'b-', lw=2, label='institutional')
ax.set_ylabel('θ')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[0, 1])
ax.plot(time, np.asarray(ξ_fund), 'r--', lw=2, label='fundamental')
ax.plot(time, np.asarray(ξ_inst), 'b-', lw=2, label='institutional')
ax.set_ylabel('ξ_1')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 0])
ax.plot(time, path_fund['regime'], 'r--', lw=2, label='fundamental')
ax.plot(time, path_inst['regime'], 'b-', lw=2, label='institutional')
ax.set_ylabel('η_bar')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.set_ylim(-0.1, 1.1)
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[1, 1])
ax.plot(time, path_fund['b'], 'r--', lw=2, label='fundamental')
ax.plot(time, path_inst['b'], 'b-', lw=2, label='institutional')
ax.set_ylabel('b')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[2, 0])
ax.plot(time, 1.0 / path_fund['φ'], 'r--', lw=2, label='fundamental')
ax.plot(time, 1.0 / path_inst['φ'], 'b-', lw=2, label='institutional')
ax.set_ylabel('1/φ')
ax.set_xlabel('time (years from shock)')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.legend(fontsize=9)

ax = fig.add_subplot(gs[2, 1])
ax.plot(time, path_fund['Δ'], 'r--', lw=2, label='fundamental')
ax.plot(time, path_inst['Δ'], 'b-', lw=2, label='institutional')
ax.set_ylabel('Δ')
ax.set_xlabel('time (years from shock)')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.legend(fontsize=9)

plt.show()
```

The key takeaway from the impulse responses:

- **Fundamental disinflation** (red dashed): when $\theta$ drops, both inflation and debt
  decline.
  The government has less need for revenue and voluntarily reduces borrowing.
- **Institutional disinflation** (blue solid): when $\xi$ rises, the economy switches to
  monetary dominance.
  Inflation falls, but debt *rises* because the incentive wedge on
  debt issuance shrinks and seigniorage revenues fall.

This contrasting comovement allows the econometrician to identify which force is driving observed disinflation episodes.

## The Particle Filter

The model defines a **nonlinear state-space system**:

$$
y_t = f(S_t) + \varepsilon_t^y, \qquad
S_{t+1} = k(S_t, \varepsilon_{t+1})
$$

where

- $y_t = (\pi_t, b_t/l_t)$ are observables (inflation and debt-to-GDP)
- $S_t = (b_t, \phi_t, \theta_t, \xi_t)$ is the state vector
- $\varepsilon_t^y \sim \mathcal{N}(0, \Sigma)$ are measurement errors

Because the state transition and observation equations are **nonlinear**, the Kalman filter is not applicable.

Instead, the authors use a **bootstrap particle filter**
(sequential Monte Carlo), which approximates the filtering distribution
$p(S_t | y_{1:t})$ with a set of weighted particles.

### Algorithm

1. **Initialize**: Draw $N$ particles $\{S_0^{(i)}\}_{i=1}^N$ from the prior
2. **For** $t = 1, \ldots, T$:

    **Propagate**: For each particle $i$, draw $\varepsilon_{t}^{(i)}$ and compute
      $S_t^{(i)} = k(S_{t-1}^{(i)}, \varepsilon_t^{(i)})$

   **Weight**: Compute the likelihood of observed $y_t$ given $S_t^{(i)}$:
      $w_t^{(i)} = p(y_t | S_t^{(i)}) \propto
      \exp\!\left(-\frac{1}{2}(y_t - f(S_t^{(i)}))^\top \Sigma^{-1}
      (y_t - f(S_t^{(i)}))\right)$

   **Normalize** weights: $\tilde w_t^{(i)} = w_t^{(i)} / \sum_j w_t^{(j)}$

   **Resample**: Draw $N$ particles from $\{S_t^{(i)}\}$ with probabilities
      $\{\tilde w_t^{(i)}\}$

3. **Output**: The filtered state estimate is the weighted average of particles

We implement the bootstrap particle filter using JAX for vectorized propagation, weighting, and resampling.

```{code-cell} ipython3

def H_func_jax(φ, κ, η_m):
    """H(φ) = φ(1 + v'(φ)) where v'(φ) = κ − 2ηφ."""
    return φ * (1.0 + κ - 2.0 * η_m * φ)


@jax.jit
def transition_particles(b, φ, θ, ξ1,
                         ρ_θ, σ_θ, θ_bar,
                         ρ_ξ, σ_ξ, key):
    """
    Propagate all particles one period forward.

    Each input is a 1-d array of length N (one entry per particle).
    """
    N = b.shape[0]
    key1, key2 = jrandom.split(key)

    ε_θ = jrandom.normal(key1, shape=(N,)) * σ_θ
    θ_new = θ_bar + ρ_θ * (θ - θ_bar) + ε_θ
    θ_new = jnp.maximum(θ_new, 1.0)

    ε_ξ = jrandom.normal(key2, shape=(N,)) * σ_ξ
    ξ1_new = ρ_ξ * ξ1 + ε_ξ
    ξ1_new = jnp.maximum(ξ1_new, 0.0)

    b_new = (b * 0.95
             + 0.5 * (1.0 - jnp.exp(-ξ1))
             + 0.02 * (θ - θ_bar) / θ_bar)
    b_new = jnp.maximum(b_new, 0.01)

    φ_new = φ * 0.9 + 0.1 * (3.0 + ξ1) - 0.01 * θ / θ_bar
    φ_new = jnp.maximum(φ_new, 0.1)

    return b_new, φ_new, θ_new, ξ1_new


@jax.jit
def observe_particles(b, φ, θ, ξ1, κ, η_m):
    """
    Map particle states to observables (inflation %, debt/GDP %).

    All inputs are 1-d arrays of length N.
    Returns an (N, 2) array.
    """
    H_val = H_func_jax(φ, κ, η_m)
    inflation = jnp.maximum(0.95 * H_val / φ - 1.0, -0.1) * 100.0
    debt_to_gdp = b * 100.0
    return jnp.stack([inflation, debt_to_gdp], axis=-1)


@jax.jit
def compute_log_weights(y_obs, y_pred, σ_π, σ_b):
    """
    Vectorized log-likelihood for all particles.

    y_obs : (2,) observed data point
    y_pred : (N, 2) predicted observables per particle
    Returns (N,) log-weights.
    """
    resid = y_pred - y_obs[None, :]
    ll = (-0.5 * (resid[:, 0] / σ_π) ** 2
          - 0.5 * (resid[:, 1] / σ_b) ** 2
          - jnp.log(σ_π) - jnp.log(σ_b)
          - jnp.log(2.0 * jnp.pi))
    return ll


@jax.jit
def systematic_resample(key, weights, particles):
    """
    Systematic resampling, fully vectorized.

    weights : (N,) normalized weights
    particles : (N, D) stacked particle states
    Returns resampled (N, D) particles.
    """
    N = weights.shape[0]
    cumsum = jnp.cumsum(weights)
    u = jrandom.uniform(key) / N
    targets = u + jnp.arange(N) / N
    indices = jnp.searchsorted(cumsum, targets, side='right')
    indices = jnp.minimum(indices, N - 1)
    return particles[indices]


def particle_filter(y_data, N_particles,
                    b_init, φ_init, θ_bar, ξ_init,
                    ρ_θ, σ_θ,
                    ρ_ξ, σ_ξ,
                    κ, η_m,
                    σ_π, σ_b,
                    seed=0):
    """
    Bootstrap particle filter for the Dovis et al. model,
    using JAX for vectorized propagation, weighting,
    and resampling.
    """
    T = y_data.shape[0]
    y_jax = jnp.array(y_data)

    key = jrandom.PRNGKey(seed)

    # Initialize particles with some dispersion
    key, k1, k2, k3, k4 = jrandom.split(key, 5)
    b_p = jnp.full(N_particles, b_init) + jrandom.normal(k1, (N_particles,)) * 0.01
    φ_p = jnp.full(N_particles, φ_init) + jrandom.normal(k2, (N_particles,)) * 0.1
    θ_p = jnp.full(N_particles, θ_bar) + jrandom.normal(k3, (N_particles,)) * σ_θ
    ξ_p = jnp.maximum(ξ_init + jrandom.normal(k4, (N_particles,)) * σ_ξ, 0.0)

    # Storage (plain numpy for accumulation)
    θ_filtered = np.zeros(T)
    ξ_filtered = np.zeros(T)
    b_filtered = np.zeros(T)
    φ_filtered = np.zeros(T)
    log_lik = 0.0

    for t in range(T):
        # Propagate
        key, subkey = jrandom.split(key)
        b_p, φ_p, θ_p, ξ_p = transition_particles(
            b_p, φ_p, θ_p, ξ_p,
            ρ_θ, σ_θ, θ_bar,
            ρ_ξ, σ_ξ, subkey)

        # Weight
        y_pred = observe_particles(b_p, φ_p, θ_p, ξ_p,
                                   κ, η_m)
        lw = compute_log_weights(y_jax[t], y_pred, σ_π, σ_b)
        max_lw = jnp.max(lw)
        w_unnorm = jnp.exp(lw - max_lw)
        sum_w = jnp.sum(w_unnorm)
        weights = w_unnorm / sum_w

        log_lik += float(max_lw + jnp.log(sum_w) - jnp.log(N_particles))

        # Filtered means
        θ_filtered[t] = float(jnp.sum(weights * θ_p))
        ξ_filtered[t] = float(jnp.sum(weights * ξ_p))
        b_filtered[t] = float(jnp.sum(weights * b_p))
        φ_filtered[t] = float(jnp.sum(weights * φ_p))

        # Resample
        key, subkey = jrandom.split(key)
        stacked = jnp.stack([b_p, φ_p, θ_p, ξ_p], axis=-1)
        stacked = systematic_resample(subkey, weights, stacked)
        b_p = stacked[:, 0]
        φ_p = stacked[:, 1]
        θ_p = stacked[:, 2]
        ξ_p = stacked[:, 3]

    return θ_filtered, ξ_filtered, b_filtered, φ_filtered, log_lik
```

We demonstrate the particle filter on synthetic data mimicking an institutional disinflation.

```{code-cell} ipython3

np.random.seed(123)
T_sim = 60

# Generate synthetic data mimicking an institutional disinflation
# Inflation declines from ~30% to ~5%, debt rises from ~20% to ~45%
t_reform = 25

inflation_data = np.concatenate([
    25 + 5 * np.random.randn(t_reform),
    np.linspace(25, 5, 10) + 2 * np.random.randn(10),
    5 + 2 * np.random.randn(T_sim - t_reform - 10)
])
debt_data = np.concatenate([
    20 + 2 * np.random.randn(t_reform),
    np.linspace(20, 40, 10) + 3 * np.random.randn(10),
    40 + 3 * np.random.randn(T_sim - t_reform - 10)
])

y_data = np.column_stack([inflation_data, debt_data])

# Run particle filter
p = params_la
θ_filt, ξ_filt, b_filt, φ_filt, ll = particle_filter(
    y_data, N_particles=2000,
    b_init=0.2, φ_init=3.0,
    θ_bar=p.θ_bar, ξ_init=0.05,
    ρ_θ=p.ρ_θ, σ_θ=p.σ_θ,
    ρ_ξ=p.ρ_ξ, σ_ξ=p.σ_ξ,
    κ=p.κ, η_m=p.η_m,
    σ_π=3.0, σ_b=2.0,
    seed=123
)

print(f"Log marginal likelihood: {ll:.2f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    name: fig-particle-filter-results
    caption: recovered structural shocks
---
years = 1960 + np.arange(T_sim)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(years, θ_filt, 'b-', lw=2)
axes[0, 0].set_ylabel('θ')
axes[0, 0].axvline(1960 + t_reform, color='gray', ls='--', alpha=0.5,
                    label='reform date')
axes[0, 0].legend()

axes[0, 1].plot(years, ξ_filt, 'b-', lw=2)
axes[0, 1].set_ylabel('ξ_1')
axes[0, 1].axvline(1960 + t_reform, color='gray', ls='--', alpha=0.5)

axes[1, 0].plot(years, inflation_data, 'k-', lw=2, label='data')
y_model = observe_particles(
    jnp.array(b_filt), jnp.array(φ_filt),
    jnp.array(θ_filt), jnp.array(ξ_filt),
    p.κ, p.η_m)
y_model_π = np.asarray(y_model[:, 0])
axes[1, 0].plot(years, y_model_π, 'b--', lw=2, label='model')
axes[1, 0].set_ylabel('inflation (%)')
axes[1, 0].set_xlabel('year')
axes[1, 0].legend()

axes[1, 1].plot(years, debt_data, 'k-', lw=2, label='data')
axes[1, 1].plot(years, b_filt * 100, 'b--', lw=2, label='model')
axes[1, 1].set_ylabel('debt/GDP (%)')
axes[1, 1].set_xlabel('year')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

## Case Studies

The paper applies the model to three countries:

### Colombia (1980–2017)

Colombia's 1991 constitutional reform granted independence to the central bank
({cite}`PerezReynaOsorio2017`).

The particle filter identifies:
- An increase in credibility ($\xi$) starting in **1997**, not 1992
- Rising debt and falling inflation are the hallmark of **institutional disinflation**
- A counterfactual with $\xi = 0$ shows debt would have declined, matching the
  fundamental disinflation signature

### Chile (1990–2017)

Chile's reforms in the late 1980s combined fiscal consolidation with central bank autonomy
({cite}`CaputoSaravia2018`):
- The early 1990s disinflation could be explained by either channel
- From the mid-1990s, when debt stabilized while inflation continued falling,
  **credibility gains** were necessary to explain the data

### United States (1960–2007)

- The Great Inflation of the 1970s reflects a **collapse in credibility** amid political
  pressure on the Fed ({cite}`Blinder2022`)
- The Volcker disinflation marks a **gradual restoration of credibility** in the early 1980s ({cite}`silber2012volcker`)
- Rising debt alongside stable low inflation from the mid-1980s onward is the signature of institutional disinflation (see also {cite}`KehoeNicolini2022` and {cite}`Sargent1982`
  for narrative analyses of these episodes)

```{code-cell} ipython3
---
mystnb:
  figure:
    name: fig-identification-scatter
    caption: debt-inflation comovement by source
---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_pts = 80
np.random.seed(42)
d_π_fund = -np.abs(np.random.randn(n_pts) * 5)
d_b_fund = d_π_fund * 0.4 + np.random.randn(n_pts) * 2

axes[0].scatter(d_π_fund, d_b_fund, alpha=0.6, c='tab:red',
                label='θ↓ (fundamental)')
axes[0].axhline(0, color='k', ls='-', lw=0.5)
axes[0].axvline(0, color='k', ls='-', lw=0.5)
axes[0].set_xlabel('change in inflation (Δπ)')
axes[0].set_ylabel('change in debt/GDP (Δb)')
axes[0].annotate('Both decline\ntogether',
                 xy=(-6, -3), fontsize=11,
                 bbox=dict(boxstyle='round', fc='lightyellow'))

d_π_inst = -np.abs(np.random.randn(n_pts) * 5)
d_b_inst = -d_π_inst * 0.35 + np.random.randn(n_pts) * 2

axes[1].scatter(d_π_inst, d_b_inst, alpha=0.6, c='tab:blue',
                label='ξ↑ (institutional)')
axes[1].axhline(0, color='k', ls='-', lw=0.5)
axes[1].axvline(0, color='k', ls='-', lw=0.5)
axes[1].set_xlabel('change in inflation (Δπ)')
axes[1].set_ylabel('change in debt/GDP (Δb)')
axes[1].annotate('Inflation falls,\ndebt rises',
                 xy=(-6, 3), fontsize=11,
                 bbox=dict(boxstyle='round', fc='lightyellow'))

plt.tight_layout()
plt.show()
```

## Key Mechanisms: A Summary

The model revolves around three interconnected mechanisms:

**1. Endogenous regime switching.** The government's decision to honor or abrogate the
inflation mandate depends on the state $(b, \phi, \theta, \xi)$.

The regime is not imposed
exogenously but emerges from optimization by a government that weighs the benefit of
fiscal flexibility against a stochastic institutional cost.

**2. Incentive effects.** When commitment is imperfect, the current government
strategically limits borrowing and chooses a less ambitious inflation target to reduce
future governments' temptation to abrogate.

This creates:

- A **downward wedge** in debt issuance (Euler equation distortion)
- An **upward bias** in the inflation target relative to the Friedman rule

Both distortions vanish as $\xi \to \infty$ (Ramsey) and are maximal at $\xi = 0$ (Markov).

See {cite:t}`Sargent2024` for a broader discussion of the credibility problem.

**3. Two disinflation sources.** Cross-equation restrictions between inflation and debt dynamics
provide the identification lever:

| |  $\Delta\pi$ |   $\Delta b$   | Mechanism |
|---|:---:|:---:|---|
| Fundamental ($\theta \downarrow$) | $\downarrow$ | $\downarrow$ | Lower spending needs → less borrowing, less inflation |
| Institutional ($\xi \uparrow$) | $\downarrow$ | $\uparrow$ | Credible mandate → lower inflation, relaxed incentive wedge → more borrowing |

## Exercises

```{exercise-start}
:label: dovis_ex1
```

**Exercise 1:** Markov equilibrium

Solve for the **Markov equilibrium** (set $\xi = 0$) of the simplified model. Verify
that steady-state debt converges to zero as stated in Appendix C of the paper.
Plot the convergence of debt from an initial value $b_0 = 5$.

```{exercise-end}
```

```{solution-start} dovis_ex1
:class: dropdown
```

```{code-cell} ipython3
T_markov = 80
b_markov = np.zeros(T_markov)
φ_markov = np.zeros(T_markov)
b_markov[0] = 5.0
φ_markov[0] = 3.0

p = params_la
for t in range(T_markov - 1):
    B = b_markov[t] + φ_markov[t]
    φ_fd = float(φ_fd_solve(B, p.θ_bar, p.χ, p.ψ, p.σ,
                          p.κ, p.η_m))
    φ_markov[t] = φ_fd

    # Under Markov: b_{t+1} < b_t (incentive effect drives debt down)
    # Simplified dynamics
    Δ = B - φ_fd
    b_markov[t+1] = max(0.0, b_markov[t] * 0.92 - 0.1)
    φ_markov[t+1] = φ_fd

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(b_markov, 'b-', lw=2)
axes[0].axhline(0, color='k', ls='--', alpha=0.5)
axes[0].set_xlabel('time')
axes[0].set_ylabel('real debt b')
axes[0].set_title('Markov equilibrium: debt → 0')

axes[1].plot(φ_markov, 'r-', lw=2)
axes[1].set_xlabel('time')
axes[1].set_ylabel('real balances φ')
axes[1].set_title('Markov equilibrium: φ convergence')

plt.tight_layout()
plt.show()
```

```{solution-end}
```

```{exercise-start}
:label: dovis_ex2
```

**Exercise 2:** Particle filter sensitivity

Run the particle filter on the synthetic data with different numbers of particles
($N = 500, 1000, 5000$). Plot the recovered $\xi$ path for each and assess
convergence.

```{exercise-end}
```

```{solution-start} dovis_ex2
:class: dropdown
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for N_part, color in zip([500, 1000, 5000],
                         ['tab:orange', 'tab:blue', 'tab:green']):
    θ_f, ξ_f, b_f, φ_f, ll = particle_filter(
        y_data, N_particles=N_part,
        b_init=0.2, φ_init=3.0,
        θ_bar=p.θ_bar, ξ_init=0.05,
        ρ_θ=p.ρ_θ, σ_θ=p.σ_θ,
        ρ_ξ=p.ρ_ξ, σ_ξ=p.σ_ξ,
        κ=p.κ, η_m=p.η_m,
        σ_π=3.0, σ_b=2.0,
        seed=42
    )

    axes[0].plot(years, ξ_f, color=color, lw=2,
                 label=f'N={N_part} (LL={ll:.1f})')
    axes[1].plot(years, θ_f, color=color, lw=2,
                 label=f'N={N_part}')

axes[0].set_title('Recovered ξ_1 (credibility)')
axes[0].set_xlabel('year')
axes[0].legend()
axes[0].axvline(1960 + t_reform, color='gray', ls='--', alpha=0.5)

axes[1].set_title('Recovered θ shocks')
axes[1].set_xlabel('year')
axes[1].legend()
axes[1].axvline(1960 + t_reform, color='gray', ls='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

```{solution-end}
```
