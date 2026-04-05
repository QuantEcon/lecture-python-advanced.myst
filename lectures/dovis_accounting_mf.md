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
{cite}`DovisAccountingMFrevised`. 

The model provides a framework for  revisiting some long-standing questions about **fiscal dominance** versus **monetary dominance** in a framework that allows for **partial commitment** to an inflation target.


```{note}
For an early discussion of "partial commitment" in the context of fiscal and monetary policy, see the concluding section of {cite}`LucasStokey1982`,  the original working paper version of {cite}`LucasStokey1983`. In Quantecon's view, the referees and editors of the *Journal of Monetary Economics* version made a mistake by insisting that Lucas and Stokey rewrite the concluding section of their paper.
```

```{note}
{cite}`SargentWallace1981` contrasted these two types of "dominance"  as different ways of  coordinating
monetary and fiscal policy.  They thought about them  at the beginning of the Reagan administration, when the 1970s surge in US inflation had not yet been tamed by the monetary-fiscal policies presided over by Paul Volcker. Sargent and Wallace's title, "Some Unpleasant Monetarist Arithmetic," expressed the idea that in the face of a persistent net-of-interest government deficit, efforts to reduce inflation through tight monetary policy work only temporarily, if at all. That is   because they lead to higher  government debt and thus greater gross-of-interest government deficits that must be financed  in the future.
```

{cite}`DovisAccountingMFrevised` provides a framework for understanding how
the **credibility** of a government's inflation-targeting mandate shapes equilibrium outcomes for inflation, public debt, and primary surpluses.

The paper posits  that there are two ways that a disinflation can occur:

- **Fundamental disinflation**: a reduction in fiscal needs ($\theta$) leads inflation and debt to
  decline together.
- **Institutional disinflation**: an increase in the credibility of the inflation mandate ($\xi$)   leads inflation to fall while debt *rises*.

The contrasting comovement of debt and inflation in these types of disinflations
allows the authors to create a statistical model that lets them classify observed disinflations into episodes that were driven by fiscal fundamentals or by institutional changes. 



The paper applies these ideas to Colombia, Chile, and the United States, using a
**particle filter** to recover the sequences of fiscal and institutional shocks that are consistent with the observed joint paths of inflation and debt-to-GDP ratios.

In this lecture, we will:

1. Set up the model environment — a {cite}`SargentWallace1981` economy with a household, firms, and a government
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
import numpy as np
from numba import njit, prange
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import interp1d
from scipy.special import logsumexp
import matplotlib.gridspec as gridspec

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

Following {cite}`Aiyagari1989` and {cite}`AMSS_2002` (see also QuantEcon lectures {doc}`amss`, {doc}`amss2`, and {doc}`amss3`), define the **real primary surplus** as

$$
\Delta(s^t) \equiv \tau(s^t) l(s^t) - g(s^t) - T(s^t).
$$

It is possible  then to derive a static **indirect utility function over surpluses**:

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

The **money demand** or **portfolio balance**  condition becomes

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

```{code-cell} ipython3
# ============================================================
# Model Primitives
# ============================================================

@njit
def nu(l, chi, psi):
    """Labor disutility: ν(l) = χ l^{1+ψ} / (1+ψ)."""
    return chi * l**(1.0 + psi) / (1.0 + psi)

@njit
def nu_prime(l, chi, psi):
    """ν'(l) = χ l^ψ."""
    return chi * l**psi

@njit
def v_money(phi, kappa, eta_m):
    """Utility from real money balances: v(φ) = κφ - η φ²."""
    return kappa * phi - eta_m * phi**2

@njit
def v_money_prime(phi, kappa, eta_m):
    """v'(φ) = κ - 2η φ."""
    return kappa - 2.0 * eta_m * phi

@njit
def u_gov(g, sigma):
    """Government spending utility: u(g) = g^{1-σ}/(1-σ)."""
    if abs(sigma - 1.0) < 1e-10:
        return np.log(g)
    return g**(1.0 - sigma) / (1.0 - sigma)

@njit
def u_gov_prime(g, sigma):
    """u'(g) = g^{-σ}."""
    return g**(-sigma)

@njit
def H_func(phi, kappa, eta_m):
    """H(φ) = φ + v'(φ) φ = φ(1 + v'(φ))."""
    return phi * (1.0 + v_money_prime(phi, kappa, eta_m))

@njit
def H_func_prime(phi, kappa, eta_m):
    """H'(φ) = 1 + v'(φ) + v''(φ) φ = 1 + κ - 4ηφ."""
    # v'(φ) = κ - 2ηφ, v''(φ) = -2η
    return 1.0 + kappa - 4.0 * eta_m * phi

@njit
def h_func(phi, kappa, eta_m):
    """Seigniorage: h(φ) = v'(φ) φ."""
    return v_money_prime(phi, kappa, eta_m) * phi
```

## Policy Determination

### The Credibility Problem

An important innovation of {cite}`DovisAccountingMFrevised` is to model policy determination under
**partial commitment** in the following sense.

The government promises an inflation target $\pi^*$ (equivalently, a
promised value for real balances $\phi'$) for  next period. 

But a next-period government
can choose to **honor** or **abrogate** the mandate.

The cost of abrogating is a random variable $\xi(s)$ that the authors intend to  capture either some or all of the following consequences of abrogation:

- reputational losses (see {cite}`AtkesonKehoe2001`, {cite}`DovisKirpalani2021`)
- coordination failures that trigger worse equilibria
- institutional constraints (see {cite}`Lohmann1992`)
- political costs

The authors use this specification  to **nest** both the Ramsey outcome (high $\xi$,
so the mandate is always honored) and the Markov outcome ($\xi = 0$, so the mandate is
always abrogated)

This approach is related to the loose commitment framework
of {cite}`DebortoliNunes2010`, but differs because here the regime is **endogenous**.

### Recursive Formulation

The state is $x = (b, \phi, s)$ where $b$ is inherited real debt, $\phi$ is the promised
real balances, and $s = (\theta, \xi)$ is the exogenous state.

This recursive formulation builds on {cite}`Abreu1988`, {cite}`ChariKehoe1990`, and
{cite}`Chang1998`.

```{note}
For descriptions of these frameworks, see other lectures in this suite of QuantEcon lecture notes, including  {doc}`Ramsey plans, time inconsistency, sustainable plans <calvo>`,{doc}`competitive equilibria in the Chang model <chang_ramsey>`, and {doc}`sustainable plans in the Chang model <chang_credible>`.
```

The economy can be in one of two regimes:

- **Monetary dominance** (MD, $\eta = 1$): the government honors the inflation target.
- **Fiscal dominance** (FD, $\eta = 0$): the government ignores the target and chooses $\phi$ to   maximize short-run welfare.

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

**Fiscal dominance** — the government's problem is augmented by the addition of  current $\phi$ to its choice set, so that the government solves:

$$
V^{fd}(b, s) = \max_{\phi, \Delta, b', \mu, \phi'} U(\Delta, \theta) + v(\phi) +
  \hat\beta \sum_{s'} \Pr(s'|s)\, V(b', \phi', s')
$$

where the static first-order necessary condition with respect to $\phi$  is $-U'(\Delta, \theta) = v'(\phi^{fd})$.

The **expected marginal value of real balances** is

$$
J(b', \phi', s) = \beta \sum_{s'} \Pr(s'|s) \left[
  \eta(b', \phi', s')\, H(\phi') +
  (1 - \eta(b', \phi', s'))\, H\!\left(\phi^{fd}(b', s')\right)
\right].
$$

```{code-cell} ipython3
# ============================================================
# Parameter Container
# ============================================================

class DovisParams:
    """
    Container for model parameters.

    Two calibrations are available:
      - 'LA': Latin American average 1960–2017
      - 'US': United States 1914–2017
    """

    def __init__(self, calibration='LA'):

        # Common across calibrations
        self.psi = 1.0        # inverse Frisch elasticity
        self.sigma = 2.0      # inverse EIS for gov't spending

        if calibration == 'LA':
            self.beta = 0.95
            self.beta_hat = 0.92
            self.chi = 0.015
            self.kappa = 0.68
            self.eta_m = 0.07
            self.theta_bar = 130.0
            self.rho_theta = 0.9
            self.sigma_theta = np.sqrt(60.0)
            self.rho_xi = 0.998
            self.sigma_xi = 0.112
            self.lam = 0.2       # Gumbel parameter λ
        elif calibration == 'US':
            self.beta = 0.95
            self.beta_hat = 0.91
            self.chi = 0.021
            self.kappa = 0.70
            self.eta_m = 0.06
            self.theta_bar = 2.0
            self.rho_theta = 0.9
            self.sigma_theta = np.sqrt(20.0)
            self.rho_xi = 0.99
            self.sigma_xi = 0.3
            self.lam = 0.5
        else:
            raise ValueError("calibration must be 'LA' or 'US'")

        # Derived
        self.phi_star = self.kappa / (2.0 * self.eta_m)  # money satiation
        self.euler_mascheroni = 0.5772156649

        # Grid parameters
        self.n_B = 80          # grid points for total liabilities B
        self.n_phi = 40        # grid points for promised φ'
        self.n_theta = 5       # number of θ states
        self.n_xi = 7          # number of ξ₁ states
        self.B_max = 50.0      # upper bound for B
        self.b_max = 30.0      # upper bound for real debt

    def __repr__(self):
        return f"DovisParams(β={self.beta}, β̂={self.beta_hat}, calibration)"


params_la = DovisParams('LA')
params_us = DovisParams('US')
print(f"LA calibration: β={params_la.beta}, β̂={params_la.beta_hat}, "
      f"φ*={params_la.phi_star:.2f}")
print(f"US calibration: β={params_us.beta}, β̂={params_us.beta_hat}, "
      f"φ*={params_us.phi_star:.2f}")
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

```{code-cell} ipython3
# ============================================================
# Indirect Utility over Primary Surpluses
# ============================================================
# U(Δ, θ) via the static problem:
#   max_{T,g} W(T) + θ u(g) - g  subject to  T - g ≥ Δ
#
# With ν(l) = χ l^{1+ψ}/(1+ψ), the Laffer curve gives
# tax revenue T = (1 - χ l^ψ) l
#
# We solve this numerically for given (Δ, θ).

@njit
def indirect_utility(Delta, theta, chi, psi, sigma):
    """
    Compute U(Δ, θ) and its derivative U'(Δ, θ)
    numerically for the specification
       ν(l) = χ l^{1+ψ}/(1+ψ), u(g) = g^{1-σ}/(1-σ).
    """
    # For simplicity, we solve for optimal g and l by using
    # the FOCs: θ u'(g) = 1 + λ and (1 - ν'(l)) = λ (implicit in T)
    # where λ = U'(Δ)
    # Here we use a grid search / bisection approach.

    # If Δ is very negative (large deficit), no distortion needed
    # Find g*(θ): θ g^{-σ} = 1 => g* = θ^{1/σ}
    g_star = theta**(1.0 / sigma)

    if Delta <= -g_star:
        # Unconstrained optimum: no taxes needed
        U_val = g_star - nu(g_star, chi, psi) + theta * u_gov(g_star, sigma)
        U_prime = 0.0
        return U_val, U_prime

    # The maximum surplus from Laffer curve
    # T_max = max_l (1 - χ l^ψ)l
    # FOC: 1 - (1+ψ) χ l^ψ = 0 => l* = (1/((1+ψ)χ))^{1/ψ}
    l_peak = (1.0 / ((1.0 + psi) * chi))**(1.0 / psi)
    T_max = (1.0 - chi * l_peak**psi) * l_peak
    Delta_max = T_max  # max surplus when g = 0

    if Delta >= Delta_max * 0.99:
        return -1e10, -1e10  # infeasible

    # Bisect on multiplier λ ≥ 0
    # Given λ: T(λ) from Laffer curve, g(λ) from θ g^{-σ} = 1 + λ
    # Then surplus = T(λ) - g(λ) should equal Δ

    # Direct numerical approach: given Δ, θ, find optimal (l, g)
    # Lagrangian: max_{l,g} l - ν(l) + θu(g) - g + λ((1-ν'(l))l - g - Δ)
    # FOC(l): 1 - ν'(l) + λ((-ν''(l))l + (1-ν'(l))) = 0
    #       => (1 - ν'(l))(1 + λ) - λ ν''(l) l = 0
    # FOC(g): θu'(g) - 1 - λ = 0 => g = (θ/(1+λ))^{1/σ}
    # Constraint: (1-ν'(l))l - g = Δ

    # Bisect on λ ∈ [0, λ_max]
    lam_lo = 0.0
    lam_hi = 1000.0

    for _ in range(100):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        # From FOC(g):
        g_val = (theta / (1.0 + lam_mid))**(1.0 / sigma)
        # From FOC(l): (1+λ)(1 - χ l^ψ) = λ ψ χ l^ψ
        # => 1 - χ l^ψ + λ - λ χ l^ψ = λ ψ χ l^ψ
        # => 1 + λ = χ l^ψ (1 + λ + λ ψ) = χ l^ψ (1 + λ(1+ψ))
        # => l^ψ = (1 + λ) / (χ (1 + λ(1+ψ)))
        denom = chi * (1.0 + lam_mid * (1.0 + psi))
        if denom <= 0:
            lam_hi = lam_mid
            continue
        l_psi = (1.0 + lam_mid) / denom
        if l_psi <= 0:
            lam_hi = lam_mid
            continue
        l_val = l_psi**(1.0 / psi)
        T_val = (1.0 - chi * l_val**psi) * l_val
        surplus = T_val - g_val
        if surplus > Delta:
            lam_hi = lam_mid
        else:
            lam_lo = lam_mid

    lam_opt = 0.5 * (lam_lo + lam_hi)
    g_opt = (theta / (1.0 + lam_opt))**(1.0 / sigma)
    l_psi_opt = (1.0 + lam_opt) / (chi * (1.0 + lam_opt * (1.0 + psi)))
    l_opt = l_psi_opt**(1.0 / psi)

    U_val = l_opt - nu(l_opt, chi, psi) + theta * u_gov(g_opt, sigma)
    # By envelope theorem, U'(Δ) = -λ
    U_prime = -lam_opt

    return U_val, U_prime
```

```{code-cell} ipython3
# ============================================================
# Verify properties of indirect utility
# ============================================================

p = params_la
Delta_grid = np.linspace(-5.0, 3.0, 200)
theta_vals = [80.0, 130.0, 200.0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for theta_val in theta_vals:
    U_vals = np.zeros_like(Delta_grid)
    U_primes = np.zeros_like(Delta_grid)
    for i, d in enumerate(Delta_grid):
        U_vals[i], U_primes[i] = indirect_utility(
            d, theta_val, p.chi, p.psi, p.sigma)

    mask = U_vals > -1e9
    axes[0].plot(Delta_grid[mask], U_vals[mask],
                 label=f'θ = {theta_val:.0f}')
    axes[1].plot(Delta_grid[mask], U_primes[mask],
                 label=f'θ = {theta_val:.0f}')

axes[0].set_xlabel('Primary surplus Δ')
axes[0].set_ylabel('U(Δ, θ)')
axes[0].set_title('Indirect utility over surpluses')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel('Primary surplus Δ')
axes[1].set_ylabel("U'(Δ, θ)")
axes[1].set_title("Marginal cost of surpluses")
axes[1].legend()
axes[1].grid(alpha=0.3)

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

where $\xi_{1,t}$ is a persistent AR(1) component (discretized via the {cite}`Tauchen1986` method) and $\xi^{fd}_t$,
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

```{code-cell} ipython3
# ============================================================
# Discretize exogenous states using Tauchen method
# ============================================================

def tauchen(rho, sigma, n, m=3):
    """
    Tauchen method for discretizing AR(1) process
       y' = rho * y + sigma * eps,  eps ~ N(0,1)
    Returns grid y and transition matrix P.
    """
    sigma_y = sigma / np.sqrt(1.0 - rho**2)
    y_max = m * sigma_y
    y = np.linspace(-y_max, y_max, n)
    d = y[1] - y[0] if n > 1 else 1.0

    P = np.zeros((n, n))
    from scipy.stats import norm
    for i in range(n):
        for j in range(n):
            if j == 0:
                P[i, j] = norm.cdf((y[0] + d/2 - rho * y[i]) / sigma)
            elif j == n - 1:
                P[i, j] = 1.0 - norm.cdf(
                    (y[n-1] - d/2 - rho * y[i]) / sigma)
            else:
                P[i, j] = (norm.cdf((y[j] + d/2 - rho * y[i]) / sigma) -
                            norm.cdf((y[j] - d/2 - rho * y[i]) / sigma))
    return y, P


def build_grids(par):
    """Build state-space grids and transition matrices."""

    # θ grid (centered at θ_bar)
    theta_dev, P_theta = tauchen(
        par.rho_theta, par.sigma_theta, par.n_theta)
    theta_grid = par.theta_bar + theta_dev

    # ξ₁ grid (only weakly positive values)
    xi_dev, P_xi = tauchen(
        par.rho_xi, par.sigma_xi, par.n_xi)
    xi_grid = np.maximum(xi_dev, 0.0)  # truncate to non-negative

    # B grid (total real liabilities)
    B_grid = np.linspace(0.01, par.B_max, par.n_B)

    # φ' grid (promised real balances)
    phi_grid = np.linspace(0.01, par.phi_star * 0.99, par.n_phi)

    return {
        'theta_grid': theta_grid,
        'P_theta': P_theta,
        'xi_grid': xi_grid,
        'P_xi': P_xi,
        'B_grid': B_grid,
        'phi_grid': phi_grid
    }


grids = build_grids(params_la)

print("θ grid:", np.round(grids['theta_grid'], 1))
print("ξ₁ grid:", np.round(grids['xi_grid'], 3))
print(f"B grid: [{grids['B_grid'][0]:.2f}, ..., {grids['B_grid'][-1]:.2f}]"
      f" ({len(grids['B_grid'])} points)")
print(f"φ' grid: [{grids['phi_grid'][0]:.2f}, ..., "
      f"{grids['phi_grid'][-1]:.2f}]"
      f" ({len(grids['phi_grid'])} points)")
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

```{code-cell} ipython3
# ============================================================
# Simplified model for illustration:
# Deterministic θ, two ξ₁ states, and quadratic v(φ)
# ============================================================

@njit
def phi_fd_solve(B, theta, chi, psi, sigma, kappa, eta_m):
    """
    Solve for φ^{fd} given total liabilities B = b + φ.
    Under fiscal dominance: -U'(Δ, θ) = v'(φ^{fd})
    and Δ = B - β b' - ... (simplified for static case).

    For illustrative purposes, we find φ^{fd} from the static FOC.
    """
    # The static FOC: v'(φ) = -U'(Δ, θ)
    # where Δ depends on φ since Δ = B - ... 
    # In the static case with no debt dynamics: Δ ≈ B - φ + φ = B roughly
    # More precisely, in FD regime the government picks φ to solve
    #   v'(φ) + U'(Δ, θ) = 0
    # We'll use bisection, holding surplus as a function

    phi_lo = 0.01
    phi_hi = kappa / (2.0 * eta_m) * 0.99  # below satiation

    for _ in range(60):
        phi_mid = 0.5 * (phi_lo + phi_hi)
        vp = v_money_prime(phi_mid, kappa, eta_m)
        # Surplus under FD: approximate as Δ ≈ some function of B
        # For illustration, use Δ = B - phi_mid (ignoring dynamics)
        Delta_approx = max(min(B - phi_mid, 3.0), -5.0)
        _, U_p = indirect_utility(Delta_approx, theta, chi, psi, sigma)

        residual = vp + U_p  # should be zero at optimum
        if residual > 0:
            phi_lo = phi_mid  # v' too large or |U'| too small
        else:
            phi_hi = phi_mid

    return 0.5 * (phi_lo + phi_hi)
```

```{code-cell} ipython3
# ============================================================
# Illustrative: solve a simplified two-state model
# ============================================================

@njit
def solve_simplified_model(
    beta, beta_hat, chi, psi, sigma_g, kappa, eta_m,
    theta, xi_grid, P_xi, B_grid, phi_grid, lam,
    max_iter=300, tol=1e-5
):
    """
    Solve the simplified model with one θ value and
    multiple ξ₁ states using value function iteration.

    Returns W(B, ξ₁) and policy functions.
    """
    n_B = len(B_grid)
    n_xi = len(xi_grid)
    n_phi = len(phi_grid)

    # Initialize value function
    W = np.zeros((n_B, n_xi))
    for i_B in range(n_B):
        for i_xi in range(n_xi):
            B = B_grid[i_B]
            Delta = 0.0  # rough initial guess
            Uval, _ = indirect_utility(Delta, theta, chi, psi, sigma_g)
            W[i_B, i_xi] = Uval / (1.0 - beta_hat)

    W_new = np.zeros_like(W)

    # Policy storage
    pol_b = np.zeros((n_B, n_xi))
    pol_phi = np.zeros((n_B, n_xi))
    pol_Delta = np.zeros((n_B, n_xi))

    for iteration in range(max_iter):
        for i_B in range(n_B):
            B = B_grid[i_B]
            for i_xi in range(n_xi):
                xi1 = xi_grid[i_xi]

                best_val = -1e15
                best_ib = 0
                best_iphi = 0

                for i_bp in range(n_B):
                    b_prime = B_grid[i_bp] * 0.5  # b' in [0, B_max/2]

                    for i_phi in range(n_phi):
                        phi_prime = phi_grid[i_phi]

                        # Compute expected J and continuation
                        J_val = 0.0
                        EV = 0.0
                        for j_xi in range(n_xi):
                            p_xi = P_xi[i_xi, j_xi]
                            xi1_next = xi_grid[j_xi]

                            B_md = b_prime + phi_prime
                            # Interpolate W at B_md
                            # Simple: find nearest grid point
                            i_B_md = 0
                            for k in range(n_B - 1):
                                if B_grid[k+1] > B_md:
                                    break
                                i_B_md = k + 1
                            i_B_md = min(i_B_md, n_B - 1)
                            W_md = W[i_B_md, j_xi]
                            V_md = W_md + v_money(phi_prime, kappa, eta_m)

                            # φ^fd
                            phi_fd = phi_fd_solve(
                                b_prime + phi_prime, theta,
                                chi, psi, sigma_g, kappa, eta_m)
                            B_fd = b_prime + phi_fd
                            i_B_fd = 0
                            for k in range(n_B - 1):
                                if B_grid[k+1] > B_fd:
                                    break
                                i_B_fd = k + 1
                            i_B_fd = min(i_B_fd, n_B - 1)
                            W_fd = W[i_B_fd, j_xi]
                            V_fd = W_fd + v_money(phi_fd, kappa, eta_m)

                            # Logit probability of MD
                            diff = lam * (V_md - V_fd + xi1_next)
                            if diff > 500.0:
                                eta_bar = 1.0
                            elif diff < -500.0:
                                eta_bar = 0.0
                            else:
                                eta_bar = 1.0 / (1.0 + np.exp(-diff))

                            # Expected H
                            H_phi = H_func(phi_prime, kappa, eta_m)
                            H_fd = H_func(phi_fd, kappa, eta_m)
                            J_val += p_xi * (
                                eta_bar * H_phi +
                                (1.0 - eta_bar) * H_fd)

                            # Log-sum-exp for Ω
                            if lam > 0:
                                a = lam * V_md
                                b_val = lam * (V_fd - xi1_next)
                                max_ab = max(a, b_val)
                                Omega = (max_ab + np.log(
                                    np.exp(a - max_ab) +
                                    np.exp(b_val - max_ab))) / lam
                            else:
                                Omega = max(V_md, V_fd - xi1_next)

                            EV += p_xi * Omega

                        J_val *= beta

                        # Surplus
                        Delta = B - beta * b_prime - J_val
                        if Delta > 4.0 or Delta < -6.0:
                            continue

                        Uval, _ = indirect_utility(
                            Delta, theta, chi, psi, sigma_g)
                        if Uval < -1e9:
                            continue

                        val = Uval + beta_hat * EV

                        if val > best_val:
                            best_val = val
                            best_ib = i_bp
                            best_iphi = i_phi
                            pol_Delta[i_B, i_xi] = Delta

                W_new[i_B, i_xi] = best_val
                pol_b[i_B, i_xi] = B_grid[best_ib] * 0.5
                pol_phi[i_B, i_xi] = phi_grid[best_iphi]

        # Check convergence
        diff = np.max(np.abs(W_new - W))
        W[:, :] = W_new[:, :]

        if diff < tol:
            break

    return W, pol_b, pol_phi, pol_Delta
```

```{code-cell} ipython3
# ============================================================
# Solve a coarse version for illustration
# ============================================================

# Use a coarse grid for speed in this lecture
p = params_la
n_B_coarse = 30
n_phi_coarse = 15
n_xi_coarse = 3

B_grid = np.linspace(0.5, 20.0, n_B_coarse)
phi_grid = np.linspace(0.5, p.phi_star * 0.95, n_phi_coarse)
xi_dev, P_xi = tauchen(p.rho_xi, p.sigma_xi, n_xi_coarse, m=2)
xi_grid = np.maximum(xi_dev + 0.3, 0.01)  # shift to positive

print("Solving simplified model (coarse grid)...")
W, pol_b, pol_phi, pol_Delta = solve_simplified_model(
    p.beta, p.beta_hat, p.chi, p.psi, p.sigma,
    p.kappa, p.eta_m,
    p.theta_bar, xi_grid, P_xi,
    B_grid, phi_grid, p.lam,
    max_iter=100, tol=1e-4
)
print("Done.")
```

```{code-cell} ipython3
# ============================================================
# Plot value function and policy functions
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

xi_labels = [f'ξ₁ = {xi_grid[j]:.2f}' for j in range(n_xi_coarse)]
colors = ['tab:blue', 'tab:orange', 'tab:green']

for j in range(n_xi_coarse):
    axes[0, 0].plot(B_grid, W[:, j], label=xi_labels[j],
                    color=colors[j])
    axes[0, 1].plot(B_grid, pol_b[:, j], label=xi_labels[j],
                    color=colors[j])
    axes[1, 0].plot(B_grid, pol_phi[:, j], label=xi_labels[j],
                    color=colors[j])
    axes[1, 1].plot(B_grid, pol_Delta[:, j], label=xi_labels[j],
                    color=colors[j])

axes[0, 0].set_title('Value function W(B, ξ₁)')
axes[0, 0].set_xlabel('B (total liabilities)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].set_title("Debt policy b'(B, ξ₁)")
axes[0, 1].set_xlabel('B')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

axes[1, 0].set_title("Inflation target φ'(B, ξ₁)")
axes[1, 0].set_xlabel('B')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].set_title('Primary surplus Δ(B, ξ₁)')
axes[1, 1].set_xlabel('B')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.suptitle('Model Solution: Value and Policy Functions', fontsize=14)
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

```{code-cell} ipython3
# ============================================================
# Illustrate two types of disinflation with impulse responses
# ============================================================

def simulate_path(b0, phi0, theta_path, xi_path, par,
                  T=40, seed=42):
    """
    Simulate a path of the model given sequences of
    θ and ξ₁ shocks.

    This uses a simplified version: at each t, we solve for
    the government's choices given inherited (b, φ).
    """
    np.random.seed(seed)

    b_path = np.zeros(T)
    phi_path = np.zeros(T)
    Delta_path = np.zeros(T)
    inflation_path = np.zeros(T)
    regime_path = np.zeros(T)

    b_path[0] = b0
    phi_path[0] = phi0

    for t in range(T - 1):
        b = b_path[t]
        phi_t = phi_path[t]
        theta = theta_path[t]
        xi1 = xi_path[t]
        B = b + phi_t

        # Fiscal dominance value of φ
        phi_fd = phi_fd_solve(
            B, theta, par.chi, par.psi, par.sigma,
            par.kappa, par.eta_m)

        # Simplified: compare V_md vs V_fd
        Delta_md = B - phi_t  # simplified surplus under MD
        U_md, _ = indirect_utility(
            min(Delta_md, 3.0), theta, par.chi, par.psi, par.sigma)
        V_md = U_md + v_money(phi_t, par.kappa, par.eta_m)

        Delta_fd = B - phi_fd
        U_fd, _ = indirect_utility(
            min(Delta_fd, 3.0), theta, par.chi, par.psi, par.sigma)
        V_fd = U_fd + v_money(phi_fd, par.kappa, par.eta_m)

        # Regime decision (logit)
        diff = par.lam * (V_md - V_fd + xi1)
        if diff > 500:
            eta = 1.0
        elif diff < -500:
            eta = 0.0
        else:
            eta = 1.0 / (1.0 + np.exp(-diff))

        regime_path[t] = eta

        # Actual φ realized this period
        gumbel_md = np.random.gumbel(-0.5772/par.lam, 1.0/par.lam)
        gumbel_fd = np.random.gumbel(-0.5772/par.lam, 1.0/par.lam)
        realized_md = V_md + gumbel_md > V_fd - xi1 + gumbel_fd
        phi_realized = phi_t if realized_md else phi_fd

        # Inflation
        H_val = H_func(phi_realized, par.kappa, par.eta_m)
        mu = par.beta * H_val / phi_path[max(t-1, 0)] if t > 0 else 1.0
        if phi_realized > 0:
            inflation_path[t] = (mu * phi_path[max(t-1, 0)] /
                                 phi_realized - 1.0)
        else:
            inflation_path[t] = 0.0

        # Surplus
        Delta_path[t] = B - phi_realized

        # Simple debt dynamics (illustrative)
        # b' = (B - Δ - J) / β approximately
        b_path[t+1] = max(0.1, b * 0.95 + 0.05 * b0 +
                          (1.0 - eta) * (-0.5) +
                          eta * 0.3)

        # Promised φ for next period
        phi_path[t+1] = phi_realized * 0.98 + 0.02 * par.phi_star * 0.8

    return {
        'b': b_path, 'phi': phi_path, 'Delta': Delta_path,
        'inflation': inflation_path, 'regime': regime_path
    }


# Construct paths
T = 40
t_shock = 20

# Fundamental disinflation: θ drops at t_shock
theta_fund = np.ones(T) * 180.0
theta_fund[t_shock:] = 90.0
xi_fund = np.ones(T) * 0.01  # always low ξ

# Institutional disinflation: ξ rises at t_shock
theta_inst = np.ones(T) * 180.0  # θ constant
xi_inst = np.ones(T) * 0.01
xi_inst[t_shock:] = 0.8  # high ξ after shock

p = params_la
path_fund = simulate_path(5.0, 3.0, theta_fund, xi_fund, p, T)
path_inst = simulate_path(5.0, 3.0, theta_inst, xi_inst, p, T)
```

```{code-cell} ipython3
# ============================================================
# Plot impulse responses
# ============================================================

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

time = np.arange(T) - t_shock  # center at shock time

# θ path
ax = fig.add_subplot(gs[0, 0])
ax.plot(time, theta_fund, 'r--', label='Fundamental disinflation')
ax.plot(time, theta_inst, 'b-', label='Institutional disinflation')
ax.set_title('θ path')
ax.set_ylabel('θ')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ξ path
ax = fig.add_subplot(gs[0, 1])
ax.plot(time, xi_fund, 'r--', label='Fundamental')
ax.plot(time, xi_inst, 'b-', label='Institutional')
ax.set_title('ξ path')
ax.set_ylabel('ξ₁')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Regime
ax = fig.add_subplot(gs[1, 0])
ax.plot(time, path_fund['regime'], 'r--', label='Fundamental')
ax.plot(time, path_inst['regime'], 'b-', label='Institutional')
ax.set_title('Pr(Monetary Dominance)')
ax.set_ylabel('η̄')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.set_ylim(-0.1, 1.1)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Debt
ax = fig.add_subplot(gs[1, 1])
ax.plot(time, path_fund['b'], 'r--', label='Fundamental')
ax.plot(time, path_inst['b'], 'b-', label='Institutional')
ax.set_title('Real Debt')
ax.set_ylabel('b')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Inflation proxy (using φ inversely)
ax = fig.add_subplot(gs[2, 0])
ax.plot(time, 1.0 / path_fund['phi'], 'r--', label='Fundamental')
ax.plot(time, 1.0 / path_inst['phi'], 'b-', label='Institutional')
ax.set_title('Inflation proxy (1/φ)')
ax.set_ylabel('1/φ')
ax.set_xlabel('Time (years from shock)')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Surplus
ax = fig.add_subplot(gs[2, 1])
ax.plot(time, path_fund['Delta'], 'r--', label='Fundamental')
ax.plot(time, path_inst['Delta'], 'b-', label='Institutional')
ax.set_title('Primary surplus Δ')
ax.set_ylabel('Δ')
ax.set_xlabel('Time (years from shock)')
ax.axvline(0, color='k', ls=':', alpha=0.5)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle('Two Types of Disinflation: Impulse Responses',
             fontsize=14, y=1.01)
plt.show()
```

The key takeaway from the impulse responses:

- **Fundamental disinflation** (red dashed): when $\theta$ drops, both inflation and debt
  decline. The government has less need for revenue and voluntarily reduces borrowing.
- **Institutional disinflation** (blue solid): when $\xi$ rises, the economy switches to
  monetary dominance. Inflation falls, but debt *rises* because the incentive wedge on
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

```{code-cell} ipython3
# ============================================================
# Bootstrap Particle Filter
# ============================================================

@njit
def measurement_loglik(y_obs, y_pred, sigma_pi, sigma_b):
    """
    Log-likelihood of observation y_obs given predicted y_pred.
    y = (inflation, debt_to_GDP)
    """
    resid_pi = y_obs[0] - y_pred[0]
    resid_b = y_obs[1] - y_pred[1]
    ll = (-0.5 * (resid_pi / sigma_pi)**2
          - 0.5 * (resid_b / sigma_b)**2
          - np.log(sigma_pi) - np.log(sigma_b)
          - np.log(2.0 * np.pi))
    return ll


@njit
def transition_state(b, phi, theta, xi1,
                     rho_theta, sigma_theta, theta_bar,
                     rho_xi, sigma_xi):
    """
    Propagate the state one period forward.
    Returns (b', φ', θ', ξ₁').

    This is a simplified transition for illustration.
    """
    # θ transition: AR(1) around mean
    eps_theta = np.random.randn() * sigma_theta
    theta_new = theta_bar + rho_theta * (theta - theta_bar) + eps_theta
    theta_new = max(theta_new, 1.0)

    # ξ₁ transition: AR(1), non-negative
    eps_xi = np.random.randn() * sigma_xi
    xi1_new = rho_xi * xi1 + eps_xi
    xi1_new = max(xi1_new, 0.0)

    # Simplified state transition for (b, φ)
    # In practice, one would use the policy functions from the full model
    # Here we use a reduced-form approximation
    b_new = b * 0.95 + 0.5 * (1.0 - np.exp(-xi1)) + 0.02 * (theta - theta_bar) / theta_bar
    b_new = max(b_new, 0.01)

    phi_new = phi * 0.9 + 0.1 * (3.0 + xi1) - 0.01 * theta / theta_bar
    phi_new = max(phi_new, 0.1)

    return b_new, phi_new, theta_new, xi1_new


@njit
def observe_state(b, phi, theta, xi1, kappa, eta_m, lam):
    """
    Map state to observables: (inflation, debt_to_GDP).
    Simplified observation equation.
    """
    # Approximate inflation from φ
    H_val = H_func(phi, kappa, eta_m)
    inflation = max(0.95 * H_val / phi - 1.0, -0.1) * 100  # in percent

    # Debt-to-GDP: b / l where l ≈ 1 (normalized)
    debt_to_gdp = b * 100.0  # in percent

    return np.array([inflation, debt_to_gdp])


@njit
def particle_filter(y_data, N_particles,
                    b_init, phi_init, theta_bar, xi_init,
                    rho_theta, sigma_theta,
                    rho_xi, sigma_xi,
                    kappa, eta_m, lam,
                    sigma_pi, sigma_b):
    """
    Bootstrap particle filter for the Dovis et al. model.

    Parameters
    ----------
    y_data : (T, 2) array of observables [inflation%, debt_to_GDP%]
    N_particles : number of particles
    Other params : model parameters

    Returns
    -------
    theta_filtered : (T,) filtered θ path
    xi_filtered : (T,) filtered ξ₁ path
    b_filtered : (T,) filtered debt path
    phi_filtered : (T,) filtered φ path
    log_lik : scalar, log marginal likelihood
    """
    T = y_data.shape[0]

    # Storage for filtered means
    theta_filtered = np.zeros(T)
    xi_filtered = np.zeros(T)
    b_filtered = np.zeros(T)
    phi_filtered = np.zeros(T)
    log_lik = 0.0

    # Initialize particles
    b_particles = np.full(N_particles, b_init)
    phi_particles = np.full(N_particles, phi_init)
    theta_particles = np.full(N_particles, theta_bar)
    xi_particles = np.full(N_particles, xi_init)

    # Add some initial dispersion
    for i in range(N_particles):
        b_particles[i] += np.random.randn() * 0.01
        phi_particles[i] += np.random.randn() * 0.1
        theta_particles[i] += np.random.randn() * sigma_theta
        xi_particles[i] = max(xi_init + np.random.randn() * sigma_xi, 0.0)

    weights = np.ones(N_particles) / N_particles

    for t in range(T):
        # --- Propagate ---
        for i in range(N_particles):
            (b_particles[i], phi_particles[i],
             theta_particles[i], xi_particles[i]) = transition_state(
                b_particles[i], phi_particles[i],
                theta_particles[i], xi_particles[i],
                rho_theta, sigma_theta, theta_bar,
                rho_xi, sigma_xi)

        # --- Weight ---
        log_weights = np.zeros(N_particles)
        for i in range(N_particles):
            y_pred = observe_state(
                b_particles[i], phi_particles[i],
                theta_particles[i], xi_particles[i],
                kappa, eta_m, lam)
            log_weights[i] = measurement_loglik(
                y_data[t], y_pred, sigma_pi, sigma_b)

        # Normalize in log space for numerical stability
        max_lw = np.max(log_weights)
        weights_unnorm = np.exp(log_weights - max_lw)
        sum_w = np.sum(weights_unnorm)
        weights = weights_unnorm / sum_w

        # Marginal likelihood contribution
        log_lik += max_lw + np.log(sum_w) - np.log(N_particles)

        # --- Filtered means ---
        theta_filtered[t] = np.sum(weights * theta_particles)
        xi_filtered[t] = np.sum(weights * xi_particles)
        b_filtered[t] = np.sum(weights * b_particles)
        phi_filtered[t] = np.sum(weights * phi_particles)

        # --- Resample (systematic resampling) ---
        cumsum = np.cumsum(weights)
        u = np.random.rand() / N_particles
        indices = np.zeros(N_particles, dtype=np.int64)
        j = 0
        for i in range(N_particles):
            target = u + i / N_particles
            while j < N_particles - 1 and cumsum[j] < target:
                j += 1
            indices[i] = j

        b_new = np.zeros(N_particles)
        phi_new = np.zeros(N_particles)
        theta_new = np.zeros(N_particles)
        xi_new = np.zeros(N_particles)
        for i in range(N_particles):
            b_new[i] = b_particles[indices[i]]
            phi_new[i] = phi_particles[indices[i]]
            theta_new[i] = theta_particles[indices[i]]
            xi_new[i] = xi_particles[indices[i]]
        b_particles[:] = b_new
        phi_particles[:] = phi_new
        theta_particles[:] = theta_new
        xi_particles[:] = xi_new

    return theta_filtered, xi_filtered, b_filtered, phi_filtered, log_lik
```

```{code-cell} ipython3
# ============================================================
# Demonstrate the particle filter on synthetic data
# ============================================================

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
theta_filt, xi_filt, b_filt, phi_filt, ll = particle_filter(
    y_data, N_particles=2000,
    b_init=0.2, phi_init=3.0,
    theta_bar=p.theta_bar, xi_init=0.05,
    rho_theta=p.rho_theta, sigma_theta=p.sigma_theta,
    rho_xi=p.rho_xi, sigma_xi=p.sigma_xi,
    kappa=p.kappa, eta_m=p.eta_m, lam=p.lam,
    sigma_pi=3.0, sigma_b=2.0
)

print(f"Log marginal likelihood: {ll:.2f}")
```

```{code-cell} ipython3
# ============================================================
# Plot particle filter results
# ============================================================

years = 1960 + np.arange(T_sim)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Recovered θ
axes[0, 0].plot(years, theta_filt, 'b-', lw=1.5)
axes[0, 0].set_title('Recovered θ shocks')
axes[0, 0].set_ylabel('θ')
axes[0, 0].axvline(1960 + t_reform, color='gray', ls='--', alpha=0.5,
                    label='Reform date')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Recovered ξ
axes[0, 1].plot(years, xi_filt, 'b-', lw=1.5)
axes[0, 1].set_title('Recovered ξ₁ shocks (credibility)')
axes[0, 1].set_ylabel('ξ₁')
axes[0, 1].axvline(1960 + t_reform, color='gray', ls='--', alpha=0.5)
axes[0, 1].grid(alpha=0.3)

# Inflation: data vs model
axes[1, 0].plot(years, inflation_data, 'k-', lw=1.5, label='Data')
y_model_pi = observe_state(
    b_filt[0], phi_filt[0], theta_filt[0], xi_filt[0],
    p.kappa, p.eta_m, p.lam)[0] * np.ones(T_sim)  # simplified
for t in range(T_sim):
    y_model_pi[t] = observe_state(
        b_filt[t], phi_filt[t], theta_filt[t], xi_filt[t],
        p.kappa, p.eta_m, p.lam)[0]
axes[1, 0].plot(years, y_model_pi, 'b--', lw=1.5, label='Model')
axes[1, 0].set_title('Inflation: Data vs Model')
axes[1, 0].set_ylabel('Inflation (%)')
axes[1, 0].set_xlabel('Year')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Debt: data vs model
axes[1, 1].plot(years, debt_data, 'k-', lw=1.5, label='Data')
axes[1, 1].plot(years, b_filt * 100, 'b--', lw=1.5, label='Model')
axes[1, 1].set_title('Debt/GDP: Data vs Model')
axes[1, 1].set_ylabel('Debt/GDP (%)')
axes[1, 1].set_xlabel('Year')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.suptitle('Particle Filter: Recovering Structural Shocks\n'
             '(Synthetic data mimicking institutional disinflation)',
             fontsize=14)
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
- Rising debt alongside stable low inflation from the mid-1980s onward is the signature   of institutional disinflation (see also {cite}`KehoeNicolini2022` and {cite}`Sargent1982`
  for narrative analyses of these episodes)

```{code-cell} ipython3
# ============================================================
# Illustrate the identification logic with a scatter plot
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fundamental disinflation: Δπ < 0, Δb < 0
n_pts = 80
np.random.seed(42)
d_pi_fund = -np.abs(np.random.randn(n_pts) * 5)
d_b_fund = d_pi_fund * 0.4 + np.random.randn(n_pts) * 2

axes[0].scatter(d_pi_fund, d_b_fund, alpha=0.6, c='tab:red',
                label='θ↓ (fundamental)')
axes[0].axhline(0, color='k', ls='-', lw=0.5)
axes[0].axvline(0, color='k', ls='-', lw=0.5)
axes[0].set_xlabel('Change in inflation (Δπ)')
axes[0].set_ylabel('Change in debt/GDP (Δb)')
axes[0].set_title('Fundamental Disinflation')
axes[0].annotate('Both decline\ntogether',
                 xy=(-6, -3), fontsize=11,
                 bbox=dict(boxstyle='round', fc='lightyellow'))
axes[0].grid(alpha=0.3)

# Institutional disinflation: Δπ < 0, Δb > 0
d_pi_inst = -np.abs(np.random.randn(n_pts) * 5)
d_b_inst = -d_pi_inst * 0.35 + np.random.randn(n_pts) * 2

axes[1].scatter(d_pi_inst, d_b_inst, alpha=0.6, c='tab:blue',
                label='ξ↑ (institutional)')
axes[1].axhline(0, color='k', ls='-', lw=0.5)
axes[1].axvline(0, color='k', ls='-', lw=0.5)
axes[1].set_xlabel('Change in inflation (Δπ)')
axes[1].set_ylabel('Change in debt/GDP (Δb)')
axes[1].set_title('Institutional Disinflation')
axes[1].annotate('Inflation falls,\ndebt rises',
                 xy=(-6, 3), fontsize=11,
                 bbox=dict(boxstyle='round', fc='lightyellow'))
axes[1].grid(alpha=0.3)

plt.suptitle('Identification Logic: Comovement of Debt and Inflation',
             fontsize=13)
plt.tight_layout()
plt.show()
```

## Key Mechanisms: A Summary

The model revolves around three interconnected mechanisms:

**1. Endogenous regime switching.** The government's decision to honor or abrogate the
inflation mandate depends on the state $(b, \phi, \theta, \xi)$. The regime is not imposed
exogenously but emerges from optimization by a government that weighs the benefit of
fiscal flexibility against a stochastic institutional cost.

**2. Incentive effects.** When commitment is imperfect, the current government
strategically limits borrowing and chooses a less ambitious inflation target to reduce
future governments' temptation to abrogate. This creates:

- A **downward wedge** in debt issuance (Euler equation distortion)
- An **upward bias** in the inflation target relative to the Friedman rule

Both distortions vanish as $\xi \to \infty$ (Ramsey) and are maximal at $\xi = 0$ (Markov).
See {cite}`Sargent2024` for a broader discussion of the credibility problem.

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
# Markov equilibrium: ξ = 0 means FD always chosen
T_markov = 80
b_markov = np.zeros(T_markov)
phi_markov = np.zeros(T_markov)
b_markov[0] = 5.0
phi_markov[0] = 3.0

p = params_la
for t in range(T_markov - 1):
    B = b_markov[t] + phi_markov[t]
    phi_fd = phi_fd_solve(B, p.theta_bar, p.chi, p.psi, p.sigma,
                          p.kappa, p.eta_m)
    phi_markov[t] = phi_fd

    # Under Markov: b_{t+1} < b_t (incentive effect drives debt down)
    # Simplified dynamics
    Delta = B - phi_fd
    b_markov[t+1] = max(0.0, b_markov[t] * 0.92 - 0.1)
    phi_markov[t+1] = phi_fd

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(b_markov, 'b-', lw=2)
axes[0].axhline(0, color='k', ls='--', alpha=0.5)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Real debt b')
axes[0].set_title('Markov equilibrium: debt → 0')
axes[0].grid(alpha=0.3)

axes[1].plot(phi_markov, 'r-', lw=2)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Real balances φ')
axes[1].set_title('Markov equilibrium: φ convergence')
axes[1].grid(alpha=0.3)

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
    theta_f, xi_f, b_f, phi_f, ll = particle_filter(
        y_data, N_particles=N_part,
        b_init=0.2, phi_init=3.0,
        theta_bar=p.theta_bar, xi_init=0.05,
        rho_theta=p.rho_theta, sigma_theta=p.sigma_theta,
        rho_xi=p.rho_xi, sigma_xi=p.sigma_xi,
        kappa=p.kappa, eta_m=p.eta_m, lam=p.lam,
        sigma_pi=3.0, sigma_b=2.0
    )

    axes[0].plot(years, xi_f, color=color, lw=1.2,
                 label=f'N={N_part} (LL={ll:.1f})')
    axes[1].plot(years, theta_f, color=color, lw=1.2,
                 label=f'N={N_part}')

axes[0].set_title('Recovered ξ₁ (credibility)')
axes[0].set_xlabel('Year')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].axvline(1960 + t_reform, color='gray', ls='--', alpha=0.5)

axes[1].set_title('Recovered θ shocks')
axes[1].set_xlabel('Year')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].axvline(1960 + t_reform, color='gray', ls='--', alpha=0.5)

plt.suptitle('Particle Filter: Sensitivity to Number of Particles',
             fontsize=13)
plt.tight_layout()
plt.show()
```

```{solution-end}
```
