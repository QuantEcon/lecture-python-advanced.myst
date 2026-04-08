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

This lecture studies a model of fiscal and monetary policy interactions developed by {cite}`DovisAccountingMFrevised`.

The model provides a framework for  revisiting some long-standing questions about **fiscal dominance** versus **monetary dominance** in a framework that allows for **partial commitment** to an inflation target.


```{note}
For an early discussion of "partial commitment" in the context of fiscal and monetary policy, see the concluding section of {cite}`LucasStokey1982`,  the original working paper version of {cite}`LucasStokey1983`. 

In Quantecon's view, the referees and editors of the *Journal of Monetary Economics* version made a mistake by insisting that Lucas and Stokey rewrite the concluding section of their paper.
```

```{note}
{cite}`SargentWallace1981` contrasted "fiscal dominance" and "monetary dominance"  as different ways of  coordinating
monetary and fiscal policy.  

They thought about them  at the beginning of the Reagan administration, when the 1970s surge in US inflation had not yet been tamed by the monetary-fiscal policies presided over by Paul Volcker. 

Sargent and Wallace's title, "Some Unpleasant Monetarist Arithmetic," expressed the idea that in the face of a persistent net-of-interest government deficit, efforts to reduce inflation through tight monetary policy work only temporarily, if at all. 

That is   because they lead to higher  government debt and thus greater gross-of-interest government deficits that must be financed  in the future.
```

A benevolent but unable-to-commit government has incentives to delegate monetary policy to a central bank with a narrow inflation-targeting mandate, but ex-post may choose to abrogate the mandate to generate seigniorage revenues.

The decision to abrogate depends on two state variables: the level of fiscal fundamentals (outstanding public debt and the marginal utility of government spending) and a stochastic institutional cost that captures the legal, reputational, and political hurdles associated with overriding the mandate.

Joint movements in these states lead the economy to transit endogenously between two regimes: a **monetary-dominant** regime, where the inflation target is honored, and a **fiscal-dominant** regime, where it is not.

The model nests the Ramsey allocation (obtained when the cost of deviation is prohibitively large) and the Markov equilibrium (obtained when that cost is zero).

The paper distinguishes two ways that a disinflation can occur:

- **Fundamental disinflation**: a reduction in fiscal needs ($\theta$) leads inflation and debt to
  decline together.
- **Institutional disinflation**: an increase in the credibility of the inflation mandate ($\xi$) leads inflation to fall while debt *rises*.

The contrasting comovement of debt and inflation in these types of disinflations
allows the authors to create a statistical model that lets them classify observed disinflations into episodes that were driven by fiscal fundamentals or by institutional changes. 

The paper applies these ideas to Colombia and Chile, using a
**particle filter** to recover the sequences of fiscal and institutional shocks that are consistent with the observed joint paths of inflation and debt-to-GDP ratios.

In this lecture, we will:

1. Set up the model environment in a {cite:t}`SargentWallace1981` economy with a household and a government
2. Describe implementable fiscal and monetary outcomes
3. Characterize two polar benchmarks: the **Ramsey** outcome (full commitment) and the
   **Markov** outcome (no commitment)
4. Formulate a partial-commitment model with endogenous regime switching governed by
   a stochastic cost $\xi$ of deviating from the mandate
5. Write Python code to solve the model numerically
6. Simulate the two types of disinflation (fundamental and institutional)
7. Implement an illustrative particle filter on synthetic data
8. Summarize the paper's case studies of Colombia and Chile

We use JAX to vectorize the key computations and accelerate value function iteration and particle filtering.

```{admonition} GPU
:class: warning

This lecture was built using a machine with JAX installed and access to a GPU.

To run this lecture on [Google Colab](https://colab.research.google.com/), click on the "rocket" icon at the top of the page, select "Colab", and set the runtime environment to include a GPU.

To run this lecture on your own machine, you need to install [Google JAX](https://github.com/google/jax).
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install jax
```


```{code-cell} ipython3
import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from collections import namedtuple
from functools import partial
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

jax.config.update("jax_enable_x64", True)
```

## The economy

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
- $\nu(l)$ is a strictly increasing and convex labor disutility function
- $v(m)$ and $u(g)$ are strictly increasing and concave
- $\theta(s)$ is a preference shock to the marginal utility of government spending
- $\beta$ is the household discount factor

The linear production technology is operated by competitive firms, and the resource constraint is $c(s^t) + g(s^t) \leq l(s^t)$.

The government finances spending $g$ with linear taxes on labor income, by issuing real uncontingent debt $b$, and by printing money injected into the economy via open market operations.

The government is benevolent but may have a different discount factor $\hat\beta \leq \beta$.

### Implementable allocations

Following the insight in {cite:t}`Aiyagari1989` and {cite:t}`AMSS_2002` (see also QuantEcon lectures {doc}`amss`, {doc}`amss2`, and {doc}`amss3`), we define the **real primary surplus** as

$$
\Delta(s^t) \equiv \tau(s^t) l(s^t) - g(s^t).
$$

We can then define the static **indirect utility function over surpluses** as

$$
U(\Delta, s) = \max_{c,\,l,\,g} \; c - \nu(l) + \theta(s)\, u(g)
$$

subject to the resource constraint $c + g \leq l$ and the static implementability constraint
$(1 - \nu'(l))\, l - g \geq \Delta$.

This function is well-defined for all surplus values below the maximal surplus implied by the static Laffer curve, $\Delta \leq \bar\Delta \equiv \max_l (1 - \nu'(l))\,l$.

The indirect utility function $U(\Delta, s)$ is *decreasing and concave* in $\Delta$ for all $s$, and the marginal disutility of primary surplus is increasing and is affected by the fundamental shocks.

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

We now define the model primitives as Python functions.

The labor disutility is $\nu(l) = \chi\, l^{1+\psi}/(1+\psi)$.

Money utility is $v(\phi) = \kappa\phi - \eta_m\phi^2$.

Government spending utility is $u(g) = g^{1-\sigma}/(1-\sigma)$.

The function $H(\phi) = \phi\,(1 + v'(\phi))$ appears in the money demand condition, and $h(\phi) = v'(\phi)\,\phi$ captures seigniorage.

```{code-cell} ipython3
def ν(l, χ, ψ):
    return χ * l**(1.0 + ψ) / (1.0 + ψ)

def ν_prime(l, χ, ψ):
    return χ * l**ψ

def v_money(φ, κ, η_m):
    return κ * φ - η_m * φ**2

def v_money_prime(φ, κ, η_m):
    return κ - 2.0 * η_m * φ

def u_gov(g, σ):
    return jnp.where(jnp.abs(σ - 1.0) < 1e-10,
                     jnp.log(g),
                     g**(1.0 - σ) / (1.0 - σ))

def u_gov_prime(g, σ):
    return g**(-σ)

def H_func(φ, κ, η_m):
    return φ * (1.0 + v_money_prime(φ, κ, η_m))

def H_func_prime(φ, κ, η_m):
    return 1.0 + κ - 4.0 * η_m * φ

def h_func(φ, κ, η_m):
    return v_money_prime(φ, κ, η_m) * φ
```

## Policy determination

### The credibility problem

An important innovation of {cite}`DovisAccountingMFrevised` is to model policy determination under
**partial commitment** in the following sense.

The government promises an inflation target $\pi^*$ (equivalently, a
promised value for real balances $\phi'$) for  next period. 

But a next-period government
can choose to **honor** or **abrogate** the mandate.

The cost of abrogating is a random variable $\xi$ that captures the legal, reputational, and political hurdles associated with overriding the mandate:

- reputational losses (see {cite}`AtkesonKehoe2001`, {cite}`DovisKirpalani2021`)
- coordination failures that lead to inferior equilibria
- institutional constraints and political costs faced by policymakers

This specification *nests* both the Ramsey outcome (when $\xi$ is always large enough so that the mandate is always honored) and the Markov outcome (when $\xi = 0$ so the mandate is always abrogated).

The approach is related to the loose commitment framework of {cite}`DebortoliNunes2010`, but differs in that the regime is *endogenous*.

### Recursive formulation

The state is $x = (b, \phi, s)$ where $b$ is inherited real debt, $\phi$ is the promised real balances, and $s = (\theta, \xi)$ is the exogenous state.

This recursive formulation builds on {cite}`Abreu1988`, {cite}`ChariKehoe1990`, and {cite}`Chang1998`.

```{note}
For descriptions of these frameworks, see other lectures in this suite of QuantEcon lecture notes, including  {doc}`Ramsey plans, time inconsistency, sustainable plans <calvo>`,{doc}`competitive equilibria in the Chang model <chang_ramsey>`, and {doc}`sustainable plans in the Chang model <chang_credible>`.
```

The economy can be in one of two regimes:

- **Monetary dominance** (MD, $\eta = 1$): the government honors the inflation target.
- **Fiscal dominance** (FD, $\eta = 0$): the government ignores the target and chooses $\phi$ to   maximize short-run welfare.

The transition between these regimes is related to the influential work of {cite}`Leeper1991`, {cite}`Bianchi2013`, and {cite}`BianchiIlut2017`, but differs on two fronts: first, the switches from one regime to the other are **endogenous**, and second, the policy chosen in each regime is also endogenous rather than governed by exogenous monetary and fiscal rules.

The **regime indicator** is

$$
\eta(b', \phi', s') = \begin{cases}
1 & \text{if } V^{md}(b', \phi', s') \geq V^{fd}(b', s') - \xi(s') \\
0 & \text{otherwise}
\end{cases}
$$

**Monetary dominance** -- the government solves:

$$
V^{md}(b, \phi, s) = \max_{\Delta, b', \mu, \phi'} U(\Delta, \theta) + v(\phi) +
  \hat\beta \sum_{s'} \Pr(s'|s)\, V(b', \phi', s')
$$

where maximization is subject to the budget constraint $\Delta = b + \phi - \beta b' - \mu\phi$ and the money demand condition $\mu\phi = J(b', \phi', s)$.

**Fiscal dominance** -- the government's problem adds current $\phi$ to its choice set:

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

We collect all model parameters into a named tuple, calibrated to an average of Colombia and Chile 1960–2017 following Table 1 of {cite:t}`DovisAccountingMFrevised`.

```{code-cell} ipython3
DovisParams = namedtuple('DovisParams', [
    'β', 'β_hat', 'χ', 'κ', 'η_m', 'ψ', 'σ',
    'θ_bar', 'ρ_θ', 'σ_θ', 'ρ_ξ', 'σ_ξ', 'λ',
    'φ_star', 'n_B', 'n_φ', 'n_θ', 'n_ξ', 'B_max'
])


def create_params():
    """Create a DovisParams named tuple."""
    ψ, σ = 1.0, 2.0
    β, β_hat = 0.95, 0.92
    χ, κ, η_m = 0.015, 0.70, 0.06
    θ_bar = 130.0
    # Table 1 reports *unconditional* variances; convert to
    # innovation std devs for Tauchen: σ_ε = sqrt(σ^2_y * (1 - ρ^2))
    ρ_θ = 0.8
    σ_θ = float(np.sqrt(15.0 * (1.0 - 0.8**2)))
    ρ_ξ = 0.99
    σ_ξ = float(np.sqrt(0.05 * (1.0 - 0.99**2)))
    λ = 20.0

    φ_star = κ / (2.0 * η_m)

    return DovisParams(
        β=β, β_hat=β_hat, χ=χ, κ=κ, η_m=η_m, ψ=ψ, σ=σ,
        θ_bar=θ_bar, ρ_θ=ρ_θ, σ_θ=σ_θ, ρ_ξ=ρ_ξ, σ_ξ=σ_ξ, λ=λ,
        φ_star=φ_star, n_B=80, n_φ=40, n_θ=5, n_ξ=7, B_max=50.0
    )


params_la = create_params()
```

## Two benchmark outcomes

Before turning to the full model, it is useful to analyze two polar benchmarks.

### The Ramsey outcome (full commitment)

Under full commitment ($\xi$ always large enough so that $\eta = 1$), the government solves

$$
V^R(b, \phi, s) = \max_{\Delta, b', \phi'(s')} U(\Delta, s) + v(\phi) +
\hat\beta \sum_{s'} \Pr(s'|s) V^R(b', \phi'(s'), s')
$$

subject to $\Delta = b + \phi - \beta b' - \beta \sum_{s'} \Pr(s'|s) H(\phi'(s'))$.

```{note}
The choice of $\phi'(s')$ here is allowed to be *state-contingent* -- the Ramsey planner can promise different real balances in different future states.

The partial-commitment model studied below instead restricts the promise to a single $\phi'$ that does not vary with $s'$.

We present the Ramsey problem in its more general form as a benchmark; the restriction to a non-contingent promise is what makes abrogation tempting and gives rise to the credibility problem.
```

Under the Ramsey outcome, there is a trade-off between following the Friedman rule and making real debt state contingent.

If the volatility of the marginal value of government expenditures is sufficiently small, the benefits of making the real debt state contingent are small relative to the cost of anticipated inflation and it is optimal to set $\phi(s') = \phi^*$ for all $s'$ next period.

Key properties:

- For the quadratic specification $v(\phi) = \kappa\phi - \eta_m\phi^2$, the satiation point is $\phi^* = \kappa/(2\eta_m)$.
- Under the conditions of Proposition 3 of the paper, the Ramsey outcome has a fixed inflation level
  $1 + \pi^R = \beta H(\phi^*) / \phi^*$ that does not depend on fiscal fundamentals -- the level of debt and $\theta$.
- Surpluses and real debt follow the Euler equation
  $-U'(\Delta, s) = \frac{\hat\beta}{\beta} \sum_{s'} \Pr(s'|s) [-U'(\Delta', s')]$.

One way to implement the Ramsey outcome is to delegate monetary policy to an independent central bank with a mandate to target an inflation rate of $\pi_R$, while fiscal policy is determined by the treasury, which solves a problem similar to that in the real economy studied by {cite}`AMSS_2002`, taking as given a constant flow of seigniorage revenues (which may be negative).

### The Markov outcome (no commitment)

We now turn to the polar opposite case in which the government has no way to commit to inflation and consider a Markov equilibrium outcome.

This is a special case of our environment with $\xi(s) = 0$ for all $s$ in which case the fiscal dominant regime is always optimal.

Consequently, we can drop $\phi'$ as a choice since it has no effect on the value.

The problem reduces to

$$
V^M(b, s) = \max_{\phi, \Delta, b'} U(\Delta, \theta) + v(\phi) +
\hat\beta \sum_{s'} \Pr(s'|s) V^M(b', s')
$$

subject to $\Delta = b + \phi - \beta b' - \beta \sum_{s'} \Pr(s'|s) H(\phi^M(b', s'))$.

Key properties of the Markov outcome:

- The static optimality condition $-U'(\Delta, \theta) = v'(\phi^{fd})$ equates the marginal benefit of real balances to the marginal cost of the primary surplus, so the model predicts a higher price level (lower real balances) when the marginal cost of the surplus is high.
- Inflation *responds strongly* to fiscal pressures -- it is high on average, volatile, and closely related to fiscal considerations.
- Debt capacity is *sharply limited* -- the term $\frac{\partial J(b', \phi', s)}{\partial b'}/\beta$ is negative and effectively acts as a tax on debt issuance, leading to lower debt levels relative to the Ramsey outcome.
- In the deterministic case with $\beta = \hat\beta$: real debt converges to zero while the Ramsey outcome sustains positive debt levels (Appendix B of the paper).

The full model *interpolates* between these two extremes depending on the cost $\xi_t$.

We compute the indirect utility $U(\Delta, \theta)$ from the static problem

$$
\max_{l,\,g}\; l - g - \nu(l) + \theta\,u(g) \quad \text{s.t.}\quad (1 - \nu'(l))\,l - g \geq \Delta.
$$

With $\nu(l) = \chi\,l^{1+\psi}/(1+\psi)$, the Laffer curve gives tax revenue $T(l) = (1 - \chi\,l^\psi)\,l$.

The first-order conditions are $\theta\,u'(g) = 1 + \lambda$ and $(1 - \nu'(l))(1+\lambda) = \lambda\,\nu''(l)\,l$, where $\lambda$ is the multiplier on the surplus constraint.

We bisect on $\lambda \geq 0$ to find the optimal $(l, g)$ for given $(\Delta, \theta)$.

By the envelope theorem, $U'(\Delta) = -\lambda$.

The following code implements this procedure, returning both $U(\Delta, \theta)$ and $U'(\Delta, \theta)$.

```{code-cell} ipython3
@jit
def indirect_utility(Δ, θ, χ, ψ, σ):
    """Compute U(Δ, θ) and U'(Δ, θ) by bisection on λ."""
    g_star = θ**(1.0 / σ)
    l_star = (1.0 / χ)**(1.0 / ψ)

    l_peak = (1.0 / ((1.0 + ψ) * χ))**(1.0 / ψ)
    T_max = (1.0 - χ * l_peak**ψ) * l_peak

    def bisect_body(_, bounds):
        λ_lo, λ_hi = bounds
        λ_mid = 0.5 * (λ_lo + λ_hi)
        g_val = (θ / (1.0 + λ_mid))**(1.0 / σ)
        denom = jnp.maximum(χ * (1.0 + λ_mid * (1.0 + ψ)), 1e-15)
        l_ψ = jnp.maximum((1.0 + λ_mid) / denom, 1e-15)
        l_val = l_ψ**(1.0 / ψ)
        T_val = (1.0 - χ * l_val**ψ) * l_val
        surplus = T_val - g_val
        new_lo = jnp.where(surplus <= Δ, λ_mid, λ_lo)
        new_hi = jnp.where(surplus > Δ, λ_mid, λ_hi)
        return (new_lo, new_hi)

    λ_lo, λ_hi = lax.fori_loop(0, 100, bisect_body, (0.0, 1000.0))
    λ_opt = 0.5 * (λ_lo + λ_hi)
    g_opt = (θ / (1.0 + λ_opt))**(1.0 / σ)
    denom_opt = jnp.maximum(χ * (1.0 + λ_opt * (1.0 + ψ)), 1e-15)
    l_opt = jnp.maximum((1.0 + λ_opt) / denom_opt, 1e-15)**(1.0 / ψ)

    U_bisect = l_opt - g_opt - ν(l_opt, χ, ψ) + θ * u_gov(g_opt, σ)
    U_uncon = l_star - ν(l_star, χ, ψ) - g_star + θ * u_gov(g_star, σ)

    unconstrained = Δ <= -g_star
    infeasible = Δ >= T_max * 0.99

    U_val = jnp.where(unconstrained, U_uncon,
                      jnp.where(infeasible, -1e10, U_bisect))
    U_prime = jnp.where(unconstrained, 0.0,
                        jnp.where(infeasible, -1e10, -λ_opt))
    return U_val, U_prime
```

Let's plot $U(\Delta, \theta)$ and $U'(\Delta, \theta)$ for a range of $\Delta$ values and three different $\theta$ values

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Indirect utility and surplus costs
    name: fig-indirect-utility
---
p = params_la
Δ_grid = jnp.linspace(-5.0, 3.0, 200)
θ_vals = [80.0, 130.0, 200.0]

iu_vec = vmap(indirect_utility, in_axes=(0, None, None, None, None))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for θ_val in θ_vals:
    U_vals, U_primes = iu_vec(Δ_grid, θ_val, p.χ, p.ψ, p.σ)

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

The left panel confirms that $U(\Delta, \theta)$ is *decreasing and concave* in $\Delta$: higher surpluses are costly because they require more distortionary taxation.

The right panel shows the marginal cost of surpluses $U'(\Delta, \theta) < 0$, which becomes more negative as $\Delta$ approaches the peak of the Laffer curve.

Higher $\theta$ shifts the curves because greater social value of government spending makes running a surplus even more costly.

## The full model with Gumbel shocks

Following the paper's computational approach, the cost $\xi$ is decomposed as

$$
\xi_t = \xi_{1,t} + \xi^{fd}_t - \xi^{md}_t,
$$

where $\xi_{1,t}$ is a persistent AR(1) component (discretized via the {cite:t}`Tauchen1986` method) and $\xi^{fd}_t$, $\xi^{md}_t$ are i.i.d. **Gumbel** shocks with mean zero.

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

We discretize the AR(1) processes for $\theta$ and $\xi_1$ using the {cite:t}`Tauchen1986` method, then build grids for total liabilities $B$ and promised real balances $\phi'$.

```{code-cell} ipython3
def tauchen(ρ, σ, n, m=3):
    """Discretize AR(1) process via Tauchen's method."""
    σ_y = σ / np.sqrt(1.0 - ρ**2)
    y_max = m * σ_y
    y = np.linspace(-y_max, y_max, n)
    d = y[1] - y[0] if n > 1 else 1.0

    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j == 0:
                P[i, j] = norm.cdf((y[0] + d/2 - ρ * y[i]) / σ)
            elif j == n - 1:
                P[i, j] = 1.0 - norm.cdf(
                    (y[n-1] - d/2 - ρ * y[i]) / σ)
            else:
                P[i, j] = (norm.cdf((y[j] + d/2 - ρ * y[i]) / σ) -
                            norm.cdf((y[j] - d/2 - ρ * y[i]) / σ))
    return y, P


def build_grids(par):
    """Build state-space grids and transition matrices."""
    θ_dev, P_θ = tauchen(par.ρ_θ, par.σ_θ, par.n_θ)
    θ_grid = par.θ_bar + θ_dev

    ξ_dev, P_ξ = tauchen(par.ρ_ξ, par.σ_ξ, par.n_ξ)
    ξ_grid = ξ_dev

    B_grid = np.linspace(0.01, par.B_max, par.n_B)
    φ_grid = np.linspace(0.01, par.φ_star * 0.99, par.n_φ)

    return {
        'θ_grid': θ_grid, 'P_θ': P_θ,
        'ξ_grid': ξ_grid, 'P_ξ': P_ξ,
        'B_grid': B_grid, 'φ_grid': φ_grid
    }


grids = build_grids(params_la)
```

## Computational algorithm

Because the budget constraint depends on $b$ and $\phi$ only through their sum, the problem can be written in terms of a single endogenous state variable $B = b + \phi$ (total real government liabilities).

The value function $W(B, s_1)$ then satisfies

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

We now set up a simplified version of the model with deterministic $\theta$, multiple $\xi_1$ states, and the quadratic money-utility specification.

The static FOC $v'(\phi^{fd}) + U'(\Delta, \theta) = 0$ is a useful benchmark for how fiscal dominance trades off inflation and surplus costs.

The following function computes that static benchmark by bisection, approximating the surplus as $\Delta \approx B - \phi$.

```{code-cell} ipython3
@jit
def φ_fd_solve(B, θ, χ, ψ, σ, κ, η_m):
    """Solve for φ^fd from the static FOC v'(φ) = -U'(Δ, θ)."""
    φ_hi_init = κ / (2.0 * η_m) * 0.99

    def bisect_body(_, bounds):
        φ_lo, φ_hi = bounds
        φ_mid = 0.5 * (φ_lo + φ_hi)
        vp = v_money_prime(φ_mid, κ, η_m)
        Δ_approx = jnp.clip(B - φ_mid, -12.0, 8.0)
        _, U_p = indirect_utility(Δ_approx, θ, χ, ψ, σ)
        residual = vp + U_p
        new_lo = jnp.where(residual > 0, φ_mid, φ_lo)
        new_hi = jnp.where(residual <= 0, φ_mid, φ_hi)
        return (new_lo, new_hi)

    φ_lo, φ_hi = lax.fori_loop(0, 60, bisect_body, (0.01, φ_hi_init))
    return 0.5 * (φ_lo + φ_hi)
```

Appendix C of the paper instead defines fiscal dominance recursively by

$$
V^{fd}(b', \xi') = \max_{\phi} \left[ W(b' + \phi, \xi') + v(\phi) \right].
$$

Accordingly, the Bellman solver below recovers $\phi^{fd}$ by a grid search over $\phi$ using the continuation value $W$.

It also linearly interpolates $W(B, \xi)$ over the liabilities grid rather than snapping to the nearest grid point.

This avoids the step-function artifacts that otherwise produce flat or jagged policy plots.

The inner expectation over $\xi'$ is a matrix multiply against $P_\xi$ via `jnp.einsum`, the search over $(b', \phi')$ is a vectorized `argmax`, and the VFI loop uses `lax.scan`.

```{code-cell} ipython3
def interp_B_values(B_points, B_grid, values):
    """Linearly interpolate values(B, ξ) over B for each ξ state."""
    flat = jnp.ravel(B_points)
    interp_cols = vmap(
        lambda col: jnp.interp(flat, B_grid, col),
        in_axes=1, out_axes=0)(values)
    return jnp.moveaxis(
        interp_cols.reshape(values.shape[1], *B_points.shape), 0, -1)


def fd_from_continuation(W, B_grid, b_prime_grid, φ_grid, κ, η_m):
    """Recover V^fd and φ^fd from max_φ [W(b' + φ, ξ) + v(φ)]."""

    # Candidate total liabilities B = b' + φ for each (b', φ) pair
    B_choices = b_prime_grid[:, None] + φ_grid[None, :]

    # Interpolate W at each candidate B
    W_choices = interp_B_values(B_choices, B_grid, W)

    # Add money utility to get total value for each φ choice
    V_choices = W_choices + v_money(φ_grid, κ, η_m)[None, :, None]

    # Pick the φ that maximizes value for each (b', ξ)
    best_idx = jnp.argmax(V_choices, axis=1)
    idx = best_idx[:, None, :]

    φ_choices = jnp.broadcast_to(φ_grid[None, :, None], V_choices.shape)
    φ_fd = jnp.take_along_axis(φ_choices, idx, axis=1).squeeze(1)
    V_fd = jnp.take_along_axis(V_choices, idx, axis=1).squeeze(1)
    H_fd = H_func(φ_fd, κ, η_m)

    return B_choices, W_choices, V_choices, V_fd, φ_fd, H_fd


@partial(jit, static_argnums=(14,))
def solve_model(β, β_hat, χ, ψ, σ, κ, η_m, θ,
                ξ_grid, P_ξ, B_grid, φ_grid, λ,
                U_interp, n_iter):
    """Vectorized value function iteration."""
    Δ_fine, U_fine = U_interp
    n_B = B_grid.shape[0]
    n_φ = φ_grid.shape[0]
    n_ξ = ξ_grid.shape[0]

    # Debt choices are a fraction of the B grid
    b_prime_grid = B_grid * 0.5
    n_bp = n_B

    # Precompute money utility and H on the φ grid
    v_φ = v_money(φ_grid, κ, η_m)
    H_φ = H_func(φ_grid, κ, η_m)

    # Initialize W with the perpetuity value of zero-surplus utility
    U0, _ = indirect_utility(0.0, θ, χ, ψ, σ)
    W0 = jnp.full((n_B, n_ξ), U0 / (1.0 - β_hat))

    def bellman_step(W, _):
        # Recover V^fd and φ^fd from continuation value
        B_md, W_md, V_md, V_fd, φ_fd_arr, H_fd_arr = fd_from_continuation(
            W, B_grid, b_prime_grid, φ_grid, κ, η_m)

        # Logit regime probability η_bar (monetary dominance)
        η_bar = jax.nn.sigmoid(
            λ * (V_md - V_fd[:, None, :] + ξ_grid[None, None, :]))

        # Expected H combining MD and FD, then J (money demand)
        H_comb = (η_bar * H_φ[None, :, None]
                  + (1 - η_bar) * H_fd_arr[:, None, :])
        J = β * jnp.einsum('abj,kj->abk', H_comb, P_ξ)

        # Log-sum-exp continuation value Ω and its expectation
        log_md = λ * V_md
        log_fd = λ * (V_fd[:, None, :] - ξ_grid[None, None, :])
        Ω = jnp.logaddexp(log_md, log_fd) / λ
        EV = jnp.einsum('abj,kj->abk', Ω, P_ξ)

        # Implied surplus from the budget constraint
        Δ = (B_grid[None, None, :, None]
             - β * b_prime_grid[:, None, None, None]
             - J[:, :, None, :])

        # Evaluate indirect utility, penalizing infeasible surpluses
        U_all = jnp.interp(Δ.ravel(), Δ_fine, U_fine).reshape(Δ.shape)
        feasible = (Δ > -12.0) & (Δ < 8.0)
        U_all = jnp.where(feasible, U_all, -1e15)

        # Maximize over (b', φ') pairs
        val = U_all + β_hat * EV[:, :, None, :]
        val_flat = val.reshape(n_bp * n_φ, n_B, n_ξ)
        W_new = jnp.max(val_flat, axis=0)
        return W_new, None

    W, _ = lax.scan(bellman_step, W0, None, length=n_iter)

    # Extract policy functions from the converged W

    # Recover V^fd and φ^fd from continuation value
    B_md, W_md, V_md, V_fd, φ_fd_arr, H_fd_arr = fd_from_continuation(
        W, B_grid, b_prime_grid, φ_grid, κ, η_m)

    # Logit regime probability η_bar
    η_bar = jax.nn.sigmoid(
        λ * (V_md - V_fd[:, None, :] + ξ_grid[None, None, :]))

    # Expected H and money demand J
    H_comb = (η_bar * H_φ[None, :, None]
              + (1 - η_bar) * H_fd_arr[:, None, :])
    J = β * jnp.einsum('abj,kj->abk', H_comb, P_ξ)

    # Log-sum-exp continuation value and expectation
    log_md = λ * V_md
    log_fd = λ * (V_fd[:, None, :] - ξ_grid[None, None, :])
    Ω = jnp.logaddexp(log_md, log_fd) / λ
    EV = jnp.einsum('abj,kj->abk', Ω, P_ξ)

    # Expected regime indicator and FD real balances
    η_current = jnp.einsum('abj,kj->abk', η_bar, P_ξ)
    φ_fd_current = jnp.einsum('aj,kj->ak', φ_fd_arr, P_ξ)

    # Implied surplus from budget constraint
    Δ = (B_grid[None, None, :, None]
         - β * b_prime_grid[:, None, None, None]
         - J[:, :, None, :])

    # Indirect utility at each candidate surplus
    U_all = jnp.interp(Δ.ravel(), Δ_fine, U_fine).reshape(Δ.shape)
    U_all = jnp.where((Δ > -12.0) & (Δ < 8.0), U_all, -1e15)

    # Total value and optimal (b', φ') for each (B, ξ)
    val = U_all + β_hat * EV[:, :, None, :]
    val_flat = val.reshape(n_bp * n_φ, n_B, n_ξ)
    best_idx = jnp.argmax(val_flat, axis=0)

    # Extract policies at the optimum
    pol_b = b_prime_grid[best_idx // n_φ]
    pol_φ_out = φ_grid[best_idx % n_φ]

    Δ_flat = Δ.reshape(n_bp * n_φ, n_B, n_ξ)
    pol_Δ = jnp.take_along_axis(Δ_flat, best_idx[None], axis=0).squeeze(0)

    η_flat = η_current.reshape(n_bp * n_φ, n_ξ)
    pol_η = jnp.take_along_axis(η_flat, best_idx, axis=0)

    J_flat = J.reshape(n_bp * n_φ, n_ξ)
    pol_J = jnp.take_along_axis(J_flat, best_idx, axis=0)

    φ_fd_flat = jnp.repeat(
        φ_fd_current[:, None, :], n_φ, axis=1
    ).reshape(n_bp * n_φ, n_ξ)
    pol_φ_fd = jnp.take_along_axis(φ_fd_flat, best_idx, axis=0)

    return W, pol_b, pol_φ_out, pol_Δ, pol_η, pol_J, pol_φ_fd


def solve_policy_cache(θ_nodes, p, ξ_grid, P_ξ, B_grid, φ_grid, λ,
                       Δ_fine, n_iter=100):
    """Solve the model for several θ values and store policy arrays."""
    out = {k: [] for k in ['W', 'b', 'φ', 'Δ', 'η', 'J', 'φ_fd']}
    θ_nodes = np.asarray(θ_nodes, dtype=float)

    for θ_val in θ_nodes:
        U_fine, _ = vmap(
            lambda d: indirect_utility(d, θ_val, p.χ, p.ψ, p.σ))(Δ_fine)
        U_interp = (Δ_fine, U_fine)

        results = solve_model(
            p.β, p.β_hat, p.χ, p.ψ, p.σ,
            p.κ, p.η_m, θ_val,
            ξ_grid, P_ξ, B_grid, φ_grid, λ,
            U_interp, n_iter
        )

        names = ['W', 'b', 'φ', 'Δ', 'η', 'J', 'φ_fd']
        for name, arr in zip(names, results):
            out[name].append(np.asarray(arr))

    return {
        'θ_nodes': θ_nodes,
        'ξ_grid': np.asarray(ξ_grid),
        'B_grid': np.asarray(B_grid),
        **{k: np.stack(v) for k, v in out.items()}
    }
```

The first call triggers JIT compilation, which takes a moment.

```{code-cell} ipython3
p = params_la
n_B_coarse = 120
n_φ_coarse = 120
n_ξ_coarse = 9

B_grid = jnp.linspace(0.1, 25.0, n_B_coarse)
φ_grid = jnp.linspace(0.5, p.φ_star * 0.99, n_φ_coarse)
ξ_dev, P_ξ = tauchen(p.ρ_ξ, p.σ_ξ, n_ξ_coarse, m=3)
ξ_grid = jnp.array(np.maximum(ξ_dev, 0.0))
P_ξ = jnp.array(P_ξ)

Δ_fine = jnp.linspace(-12.0, 8.0, 800)
U_fine, _ = vmap(lambda d: indirect_utility(d, p.θ_bar, p.χ, p.ψ, p.σ))(Δ_fine)
U_interp = (Δ_fine, U_fine)

W, pol_b, pol_φ, pol_Δ, pol_η, pol_J, pol_φ_fd = solve_model(
    p.β, p.β_hat, p.χ, p.ψ, p.σ,
    p.κ, p.η_m, p.θ_bar,
    ξ_grid, P_ξ, B_grid, φ_grid, p.λ,
    U_interp, 200
)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Value and policy functions
    name: fig-value-policy
---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ξ_plot_idx = [0, n_ξ_coarse // 2, n_ξ_coarse - 1]
ξ_labels = [f'ξ_1 = {ξ_grid[j]:.2f}' for j in ξ_plot_idx]
colors = ['tab:blue', 'tab:orange', 'tab:green']

for k, j in enumerate(ξ_plot_idx):
    feasible = np.asarray(W[:, j]) > -1e12

    axes[0, 0].plot(B_grid[feasible], 
                    W[:, j][feasible], lw=2, label=ξ_labels[k],
                    color=colors[k])
    axes[0, 1].plot(B_grid[feasible], 
                    pol_b[:, j][feasible], lw=2, label=ξ_labels[k],
                    color=colors[k])
    axes[1, 0].plot(B_grid[feasible], 
                    pol_φ[:, j][feasible], lw=2, label=ξ_labels[k],
                    color=colors[k])
    axes[1, 1].plot(B_grid[feasible], 
                    pol_Δ[:, j][feasible], lw=2, label=ξ_labels[k],
                    color=colors[k])

axes[0, 0].set_xlabel('B (total liabilities)')
axes[0, 0].set_ylabel('W(B, ξ_1)')
axes[0, 0].legend()

axes[0, 1].set_xlabel('B')
axes[0, 1].set_ylabel("b'(B, ξ_1)")
axes[0, 1].legend()

axes[1, 0].set_xlabel('B')
axes[1, 0].set_ylabel("φ'(B, ξ_1)")
axes[1, 0].legend()

axes[1, 1].set_xlabel('B')
axes[1, 1].set_ylabel('Δ(B, ξ_1)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

## Two types of disinflation

A central result of the model is that inflation can decline for two distinct reasons, each with different implications for the dynamics of public debt.

We refer to a reduction in the marginal value of government spending $\theta$ as **fundamental disinflation** and to an increase in the (expected) cost of deviating from the promised inflation $\xi$ as **institutional disinflation**.

### Fundamental disinflation ($\theta$ falls, $\xi$ fixed)

Consider a path in which the realization of $\xi_t$ is low enough so that it is always optimal to be in the fiscal dominant regime.

Along this path, the value of real money balances (and inflation) is determined by the static condition $-U'(\Delta, \theta) = v'(\phi^{fd})$ and is therefore closely tied to fiscal considerations, as in a Markov equilibrium.

When $\theta$ falls from $\theta_H$ to $\theta_L$, the reduction in $\theta$ shifts the policy function $\phi^{fd}(b, \theta)$ upward: for any level of real debt, the government finds it optimal to choose a higher value for real balances, reflecting the lower marginal value of relaxing its budget constraint when government spending is less valuable.

The optimal policy for debt issuance shifts downward because the government now has stronger precautionary saving motives and therefore chooses to reduce its debt issuance.

As a result, a decline in $\theta$ while keeping $\xi$ at a low level leads to an increase in real money balances (lower inflation) and a decrease in real debt -- a **positive correlation** between inflation and debt.

### Institutional disinflation ($\xi$ rises, $\theta$ constant)

Now consider the effects of an increase in the cost of deviating from the promised inflation target.

When $\xi$ rises from $\xi_L$ to $\xi_H$ (high enough to make it optimal to switch to the monetary dominant regime), the realized $\phi$ now equals the promised value of real balances, which is higher than the statically optimal level $\phi^{fd}$.

Critically, if the process for $\xi$ is persistent, an increase in current $\xi$ implies an increase in the expected value for $\xi'$, so the government now has lower incentives to reduce the amount of debt it issues because the wedge in the Euler equation is smaller in absolute value.

As the government shifts to the monetary-dominant regime, the present value of seigniorage revenues falls, and the government must finance the inherited real liabilities with a higher present value of surpluses -- since the government is impatient, these higher surpluses are back-loaded, also resulting in an increase in the level of debt issued.

Thus inflation and debt move in **opposite** directions -- the signature of institutional disinflation.

We now illustrate the two types of disinflation by simulating impulse responses.

Both experiments treat the shock as an **MIT shock**: the change is permanent and unanticipated, so the agent does not anticipate the shock before it occurs.

Under fiscal dominance, current real balances $\phi_t$ come from the static FOC $v'(\phi) = -U'(\Delta, \theta)$.
Debt evolves as $b' = (\hat\beta/\beta)\,b$ (Appendix B of the paper).
The surplus is recovered from the budget constraint $\Delta = b + \phi - \beta b' - \beta H(\phi')$.
Realized inflation follows from $1 + \pi_t = \beta\,H(\phi_t)/\phi_t$.

For the institutional disinflation, we use the solved VFI policy functions to capture regime-switching dynamics.

```{code-cell} ipython3
def solve_φ_fd_continuous(b, θ, p):
    """Solve the FD static FOC v'(φ) = -U'(Δ,θ) with continuous bisection."""
    φ_lo, φ_hi = 0.1, p.φ_star * 0.999
    for _ in range(100):
        φ_mid = 0.5 * (φ_lo + φ_hi)
        Δ_mid = b + φ_mid - p.β * float(H_func(φ_mid, p.κ, p.η_m))
        _, U_p = indirect_utility(Δ_mid, θ, p.χ, p.ψ, p.σ)
        if float(v_money_prime(φ_mid, p.κ, p.η_m)) > float(-U_p):
            φ_lo = φ_mid
        else:
            φ_hi = φ_mid
    return 0.5 * (φ_lo + φ_hi)


def simulate_fd_mit(b0, θ_pre, θ_post, p, T, t_shock):
    """Simulate FD dynamics with an unanticipated permanent θ shock.

    Pre-shock the agent expects θ_pre forever.
    At t_shock, θ permanently drops to θ_post.
    """
    out = {k: np.zeros(T) for k in
           ['b', 'φ', 'φ_prime', 'Δ', 'π', 'η']}
    b = float(b0)

    for t in range(T):
        θ_t = θ_pre if t < t_shock else θ_post
        θ_expect = θ_t 

        # Current φ from the FD static FOC
        φ_t = solve_φ_fd_continuous(max(b, 0.001), θ_t, p)

        # Debt evolves
        b_next = max((p.β_hat / p.β) * b, 0.001)

        # Next-period φ from the expected FOC
        φ_next = solve_φ_fd_continuous(
            max(b_next, 0.001), θ_expect, p)
        H_next = float(H_func(φ_next, p.κ, p.η_m))

        # Surplus from the budget constraint
        Δ_t = b + φ_t - p.β * b_next - p.β * H_next

        # Realized inflation: 1 + π = β H(φ) / φ
        H_t = float(H_func(φ_t, p.κ, p.η_m))
        π_t = (p.β * H_t / max(φ_t, 1e-6) - 1.0) * 100.0

        out['b'][t] = b
        out['φ'][t] = φ_t
        out['φ_prime'][t] = φ_next
        out['Δ'][t] = Δ_t
        out['π'][t] = π_t
        out['η'][t] = 0.0  # always FD

        b = b_next

    return out


θ_high = 200.0
θ_irf = np.array([p.θ_bar, θ_high])
cache = solve_policy_cache(
    θ_irf, p, ξ_grid, P_ξ, B_grid, φ_grid, p.λ, Δ_fine, n_iter=200
)


def simulate_irf_inst(b0, ξ_idx_path, cache, p):
    """Simulate institutional disinflation using VFI policies.

    θ stays at θ_bar throughout; ξ jumps from low to high,
    causing a regime shift from FD to MD.
    """
    T = len(ξ_idx_path)
    θ_n = cache['θ_nodes']
    B_g = cache['B_grid']
    θ_t = p.θ_bar

    out = {k: np.zeros(T) for k in
           ['b', 'φ', 'φ_prime', 'Δ', 'π', 'η']}
    b = float(b0)
    φ_promise = None

    for t in range(T):
        ξi = int(ξ_idx_path[t])

        # Realized φ: honor the promise under MD, use static FOC under FD
        if (φ_promise is not None and t > 0
                and out['η'][t - 1] > 0.5):
            φ_t = φ_promise                             # MD regime
        else:
            φ_t = solve_φ_fd_continuous(b, θ_t, p)      # FD regime

        # Total liabilities B = b + φ
        B = np.clip(b + φ_t, B_g[0], B_g[-1])

        # Look up policy functions from the VFI cache at (B, ξ)
        idx = 0  # θ_bar is the first node
        b_prime = float(np.interp(B, B_g, cache['b'][idx, :, ξi]))
        φ_prime = float(np.interp(B, B_g, cache['φ'][idx, :, ξi]))
        Δ_t = float(np.interp(B, B_g, cache['Δ'][idx, :, ξi]))
        η_t = float(np.interp(B, B_g, cache['η'][idx, :, ξi]))

        # Realized inflation: 1 + π = β H(φ) / φ
        H_t = float(H_func(φ_t, p.κ, p.η_m))
        π_t = (p.β * H_t / max(φ_t, 1e-6) - 1.0) * 100.0

        # Store results
        out['b'][t] = b
        out['φ'][t] = φ_t
        out['φ_prime'][t] = φ_prime
        out['Δ'][t] = Δ_t
        out['π'][t] = π_t
        out['η'][t] = η_t

        b = b_prime
        φ_promise = φ_prime

    return out


T, t_shock = 60, 10
p = params_la

# Fundamental: start near FD steady state (b to 0 under FD)
irf_fund = simulate_fd_mit(0.3, θ_high, p.θ_bar, p, T, t_shock)

# Institutional: start near FD steady state, ξ jumps
ξ_inst = np.where(
    np.arange(T) < t_shock, 0, n_ξ_coarse - 1).astype(int)
irf_inst = simulate_irf_inst(0.3, ξ_inst, cache, p)

time = np.arange(T) - t_shock

θ_fund = np.where(np.arange(T) < t_shock, θ_high, p.θ_bar)
θ_inst = np.full(T, p.θ_bar)
```

```{code-cell} ipython3
def plot_irf(irf, θ_path, ξ_path, time, title):
    """Plot a 4x2 IRF figure matching the paper's layout."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    fig.suptitle(title, fontsize=14, y=1.01)
    kw = dict(lw=2, color='tab:blue')
    vkw = dict(color='k', ls=':', alpha=0.4)

    axes[0, 0].plot(time, θ_path, **kw)
    axes[0, 0].set_title(r'$\theta$'); axes[0, 0].axvline(0, **vkw)
    axes[0, 1].plot(time, ξ_path, **kw)
    axes[0, 1].set_title(r'$\xi$'); axes[0, 1].axvline(0, **vkw)
    axes[1, 0].plot(time, irf['η'], **kw)
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_title('Regime (0 = FD)'); axes[1, 0].axvline(0, **vkw)
    axes[1, 1].plot(time, irf['b'], **kw)
    axes[1, 1].set_title('Debt'); axes[1, 1].axvline(0, **vkw)
    axes[2, 0].plot(time, irf['Δ'], **kw)
    axes[2, 0].set_title('Surplus'); axes[2, 0].axvline(0, **vkw)
    axes[2, 1].plot(time, irf['π'], **kw)
    axes[2, 1].set_title('Inflation Rate (%)'); axes[2, 1].axvline(0, **vkw)
    axes[3, 0].plot(time, irf['φ'], **kw)
    axes[3, 0].set_title(r'Current $\phi$')
    axes[3, 0].set_xlabel('time'); axes[3, 0].axvline(0, **vkw)
    axes[3, 1].plot(time, irf['φ_prime'], **kw)
    axes[3, 1].set_title(r"Promised $\phi'$")
    axes[3, 1].set_xlabel('time'); axes[3, 1].axvline(0, **vkw)
    plt.tight_layout()
    return fig
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Fundamental disinflation
    name: fig-fundamental
---
fig = plot_irf(irf_fund, θ_fund, np.zeros(T), time,
               'Fundamental disinflation')
plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Institutional disinflation
    name: fig-institutional
---
fig = plot_irf(irf_inst, θ_inst, np.asarray(ξ_grid)[ξ_inst], time,
               'Institutional disinflation')
plt.show()
```

**Fundamental disinflation** ({numref}`fig-fundamental`): a permanent drop in $\theta$ from 200 to $\bar\theta = 130$ reduces fiscal pressure.

The regime stays in FD throughout ($\eta \approx 0$).

Following the shock, the debt-to-GDP ratio declines, while the inflation rate initially drops sharply in the period when $\theta$ falls, reflecting the lower marginal cost of generating primary surpluses.

Inflation then continues to decline gradually, tracking the falling path of debt, which further reduces the marginal cost of surpluses.

Inflation and debt move *together* downward -- the signature of fundamental disinflation.

**Institutional disinflation** ({numref}`fig-institutional`): a permanent rise in $\xi$ switches the regime from FD ($\eta \approx 0$) to MD ($\eta \approx 1$).

The inflation rate instead initially jumps downward at $t_0$ as real balances move toward the promised value $\phi' \approx \phi^*$.

The path of real government debt is increasing because the switch to the monetary-dominant regime allows for greater debt issuances and seigniorage revenues fall, requiring the government to finance inherited liabilities with higher (back-loaded) surpluses.

Inflation and debt move in *opposite* directions -- the signature of institutional disinflation.

## The particle filter

The empirical strategy centers on a **nonlinear state-space system** estimated with a bootstrap particle filter.

To keep the lecture computationally light, we illustrate the filtering algorithm on a reduced-form nonlinear state-space system:

$$
y_t = f(S_t) + \varepsilon_t^y, \qquad
S_{t+1} = k(S_t, \varepsilon_{t+1})
$$

where

- $y_t = (\pi_t, 100 b_t)$ are observables (inflation and debt in percent of GDP)
- $S_t = (b_t, \phi_t, \theta_t, \xi_t)$ is the state vector
- $\varepsilon_t^y \sim \mathcal{N}(0, \Sigma)$ are measurement errors

Because the state transition and observation equations are *nonlinear*, the Kalman filter is not applicable.

A **bootstrap particle filter** (sequential Monte Carlo) approximates the filtering distribution $p(S_t | y_{1:t})$ with a set of weighted particles.

We mirror that approach here, but with simplified transition and observation equations.

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

The JAX implementation below vectorizes over particles with `vmap` and loops over time with `lax.scan`.

Propagation and weighting are fully parallel across particles.

Resampling uses `jnp.searchsorted` on the cumulative weight array.

```{code-cell} ipython3
@partial(jit, static_argnums=(2,))
def particle_filter(y_data, key, N_particles,
                    b_init, φ_init, θ_bar, ξ_init,
                    ρ_θ, σ_θ, ρ_ξ, σ_ξ,
                    κ, η_m, λ, σ_π, σ_b):
    """Bootstrap particle filter returning filtered paths and log-likelihood."""

    φ_star = κ / (2.0 * η_m)

    # Particles: [b, φ, θ, ξ_1, φ_old]
    key, *ks = jax.random.split(key, 5)
    φ_init_particles = φ_init + 0.2 * jax.random.normal(ks[1], (N_particles,))
    particles = jnp.column_stack([
        b_init + 0.02 * jax.random.normal(ks[0], (N_particles,)),
        φ_init_particles,
        θ_bar + σ_θ * jax.random.normal(ks[2], (N_particles,)),
        ξ_init + 0.1 * jax.random.normal(ks[3], (N_particles,)),
        φ_init_particles
    ])

    def propagate_one(particle, pk):
        b, φ, θ, ξ1, _ = particle
        k1, k2 = jax.random.split(pk)

        θ_new = jnp.maximum(
            θ_bar + ρ_θ * (θ - θ_bar) + σ_θ * jax.random.normal(k1), 1.0)
        ξ_new = ρ_ξ * ξ1 + σ_ξ * jax.random.normal(k2)

        # Regime probability: calibrate V_gap so low ξ to FD, high ξ to MD
        V_gap = -0.15
        η = jax.nn.sigmoid(10.0 * λ * (V_gap + ξ1))

        # φ dynamics from the static FOC
        φ_fd = jnp.clip(6.0 - 0.025 * θ, 1.0, φ_star * 0.95)
        φ_md = φ_star * 0.92
        φ_target = η * φ_md + (1 - η) * φ_fd
        φ_new = jnp.clip(0.5 * φ + 0.5 * φ_target, 0.5, φ_star * 0.99)

        # Debt dynamics: FD to low debt, MD to higher debt
        b_fd_ss = jnp.clip(0.18 - 0.0005 * (θ - θ_bar), 0.10, 0.25)
        b_md_ss = jnp.clip(b_fd_ss + 0.25, 0.25, 0.50)
        b_target = η * b_md_ss + (1 - η) * b_fd_ss
        b_new = jnp.maximum(0.01, 0.90 * b + 0.10 * b_target)

        return jnp.array([b_new, φ_new, θ_new, ξ_new, φ])

    def observe_one(particle):
        """Map state to observables: π = β*H(φ)/φ - 1, debt/GDP."""
        b, φ, θ, ξ1, φ_old = particle
        β = 0.95
        H_val = H_func(φ, κ, η_m)
        inflation = (β * H_val / jnp.maximum(φ, 0.1) - 1.0) * 100.0
        debt_to_gdp = b * 100.0
        return jnp.array([inflation, debt_to_gdp])

    σ_vec = jnp.array([σ_π, σ_b])

    def pf_step(carry, inputs):
        particles, log_lik = carry
        y_t, step_key = inputs
        k_prop, k_resamp = jax.random.split(step_key)

        # Propagate each particle forward one period
        prop_keys = jax.random.split(k_prop, N_particles)
        particles = vmap(propagate_one)(particles, prop_keys)

        # Map particles to observables and compute log-likelihood weights
        y_preds = vmap(observe_one)(particles)
        resid = y_t[None, :] - y_preds
        log_w = (-0.5 * jnp.sum((resid / σ_vec)**2, axis=1)
                 - jnp.sum(jnp.log(σ_vec)) - jnp.log(2 * jnp.pi))

        # Normalize weights (log-sum-exp trick for numerical stability)
        max_lw = jnp.max(log_w)
        w_unnorm = jnp.exp(log_w - max_lw)
        sum_w = jnp.sum(w_unnorm)
        weights = w_unnorm / sum_w

        # Accumulate log-likelihood
        log_lik += max_lw + jnp.log(sum_w) - jnp.log(N_particles)

        # Weighted average gives the filtered state estimate
        filtered = jnp.sum(weights[:, None] * particles, axis=0)

        # Resampling
        cumsum = jnp.cumsum(weights)
        u = jax.random.uniform(k_resamp) / N_particles
        targets = u + jnp.arange(N_particles) / N_particles
        indices = jnp.clip(jnp.searchsorted(cumsum, targets),
                           0, N_particles - 1)
        particles = particles[indices]

        return (particles, log_lik), filtered

    step_keys = jax.random.split(key, y_data.shape[0])
    (_, total_ll), filtered_all = lax.scan(
        pf_step, (particles, 0.0), (y_data, step_keys))

    return (filtered_all[:, 2], filtered_all[:, 3],
            filtered_all[:, 0], filtered_all[:, 1], total_ll)
```

We demonstrate the particle filter on synthetic data that mimics an institutional disinflation: inflation declines from roughly 30% to 5% while debt rises from 20% to 45% of GDP.

```{code-cell} ipython3
rng = np.random.default_rng(0)
T_sim = 60
t_reform = 25

inflation_data = np.concatenate([
    25 + 5 * rng.standard_normal(t_reform),
    np.linspace(25, 5, 10) + 2 * rng.standard_normal(10),
    5 + 2 * rng.standard_normal(T_sim - t_reform - 10)
])
debt_data = np.concatenate([
    20 + 2 * rng.standard_normal(t_reform),
    np.linspace(20, 40, 10) + 3 * rng.standard_normal(10),
    40 + 3 * rng.standard_normal(T_sim - t_reform - 10)
])

y_data = jnp.column_stack([inflation_data, debt_data])

p = params_la
pf_key = jax.random.PRNGKey(123)

θ_filt, ξ_filt, b_filt, φ_filt, ll = particle_filter(
    y_data, pf_key, 5000,
    b_init=0.22, φ_init=2.5,
    θ_bar=p.θ_bar, ξ_init=-0.3,
    ρ_θ=p.ρ_θ, σ_θ=p.σ_θ,
    ρ_ξ=p.ρ_ξ, σ_ξ=p.σ_ξ,
    κ=p.κ, η_m=p.η_m, λ=p.λ,
    σ_π=2.0, σ_b=2.0
)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Particle filter on synthetic data
    name: fig-particle-filter
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
H_filt = H_func(φ_filt, p.κ, p.η_m)
y_model_π = (p.β * H_filt / jnp.maximum(φ_filt, 0.1) - 1.0) * 100
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

## Case studies

{cite:t}`DovisAccountingMFrevised` apply the model to two prominent disinflation episodes in Latin America.

We do not reproduce those estimates here, but the points below summarize the main empirical findings.

### Colombia (1980–2017)

In 1991 Colombia instituted a new constitution that granted substantial independence to its central bank, Banco de la República, explicitly mandating price stability as its primary objective and significantly insulating monetary policy from political influence ({cite}`PerezReynaOsorio2017`).

In 2001 Colombia adopted an explicit inflation targeting regime with a long-term inflation goal of 3%.

Prior to the 1991 reform, the central bank lacked autonomy, often making monetary policy susceptible to government pressures, and as a result Colombia suffered from persistent high inflation despite the relatively low level of debt.

The particle filter identifies an increase in the cost of deviating from the inflation target ($\xi$) starting in **1997**, not 1992 -- the first year after the reform.

This may be driven by the fact that it took time for the government to convince private agents that the new institutional arrangement was credible and not just a cosmetic adjustment.

The model accounts for the reduction in inflation in the 1990s with an increase in the cost of deviating from the inflation target in 1997, resulting in a persistent shift to a monetary-dominant regime from 1997 onward.

The observed increase in the debt-to-GDP ratio from 1994 to 2002 is driven by the switch to a monetary-dominant regime that allows for greater debt issuances and by higher-than-average realizations of $\theta_t$.

A counterfactual with $\xi_t = 0$ throughout shows that, without a credible constitutional reform, debt would have similarly increased driven by the high realizations of $\theta_t$, but inflation would have remained constant or even risen during the latter half of the decade.

This result underscores the crucial role credible institutional reforms played in simultaneously achieving higher debt levels and declining inflation in Colombia during this period.

### Chile (1990–2017)

Beginning in the late 1980s, Chile enacted a variety of fiscal and monetary reforms ({cite}`CaputoSaravia2018`).

It tightened public finances and, for roughly three decades, consistently posted budget surpluses.

On the monetary front, a 1989 constitutional law granted the Central Bank of Chile full autonomy and the country moved to an explicit inflation regime targeting soon after.

In contrast to Colombia, both inflation and the debt-to-GDP ratio declined over this period.

During the first half of the 1990s, the drop in inflation can be replicated either by a fall in fiscal needs or by a rise in the penalty for deviating from the inflation target -- each channel on its own is sufficient to match the joint movements in inflation and debt.

The distinction between them becomes critical in the second half of the decade: inflation keeps falling while debt-to-GDP merely flattens out.

Replicating this pattern requires credibility shocks -- an isolated increase in fiscal needs would stabilize the debt ratio but, counterfactually, would drive inflation back up.

The contrasting experiences of Colombia and Chile illuminate the two disinflation channels implied by the model: in Colombia the data can only be reconciled with a credibility gain (positive $\xi_t$ shocks), whereas in Chile the early-1990s disinflation could be matched either way, yet the continued decline in inflation once debt-to-GDP leveled off required additional credibility gains.

## Key Mechanisms: A Summary

The model revolves around three interconnected mechanisms.

*1. Endogenous regime switching.*

Whether the government honors or abrogates its inflation mandate depends on the state $(b, \phi, \theta, \xi)$.

The regime emerges from optimization -- the government weighs the benefit of fiscal flexibility against a stochastic institutional cost -- rather than from an exogenous rule.

Moreover, movements in inflation are the direct consequence of deliberate policy choices as in {cite}`SargentWallace1981`, and not the result of an equilibrium selection mechanism.

*2. Incentive effects on debt and inflation targets.*

Under imperfect commitment, the current government strategically limits borrowing and chooses a less ambitious inflation target to reduce future governments' temptation to abrogate.

These incentive effects create a *downward wedge* in debt issuance relative to the Ramsey Euler equation and an *upward bias* in the inflation target relative to the Friedman rule.

Both distortions vanish as $\xi \to \infty$ (Ramsey) and are maximal at $\xi = 0$ (Markov).

The incentive to limit indebtedness becomes stronger as the probability of switching to the fiscal dominant regime increases.

See {cite:t}`Ljungqvist2012`, chapter 23, for a broader discussion of the credibility problem.

*3. Two disinflation sources with distinct debt dynamics.*

Fundamental disinflations generate a positive correlation between inflation and the level of government debt, whereas institutional disinflations produce a negative correlation between the two.

This contrasting behavior allows the authors to use the dynamics of debt and inflation to identify the contribution of institutional changes to inflation dynamics.

| |  $\Delta\pi$ |   $\Delta b$   | Mechanism |
|---|:---:|:---:|---|
| Fundamental ($\theta \downarrow$) | $\downarrow$ | $\downarrow$ | Lower spending needs $\to$ less borrowing, less inflation |
| Institutional ($\xi \uparrow$) | $\downarrow$ | $\uparrow$ | Credible mandate $\to$ lower inflation, relaxed incentive wedge $\to$ more borrowing |

The credibility of the monetary regime is a necessary condition for supporting high levels of public debt with low levels of inflation.

## Exercises

```{exercise-start}
:label: dovis_ex1
```

For the middle $\xi_1$ state on the coarse grid, compute

$$
\phi^{fd}(b') = \arg\max_{\phi} \left[ W(b' + \phi, \xi_1) + v(\phi) \right]
$$

using the solved value function.

Plot $\phi^{fd}(b')$ and the associated fiscal-dominance value $V^{fd}(b', \xi_1)$.

```{exercise-end}
```

```{solution-start} dovis_ex1
:class: dropdown
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Fiscal-dominance policy from continuation values
    name: fig-fd-policy
---
p = params_la
b_prime_grid = B_grid * 0.5
ξ_mid = n_ξ_coarse // 2

_, _, _, V_fd_mid, φ_fd_mid, _ = fd_from_continuation(
    W[:, ξ_mid:ξ_mid + 1], B_grid, b_prime_grid, φ_grid, p.κ, p.η_m)

φ_fd_mid = np.asarray(φ_fd_mid[:, 0])
V_fd_mid = np.asarray(V_fd_mid[:, 0])
feasible_fd = V_fd_mid > -1e12

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(b_prime_grid[feasible_fd], φ_fd_mid[feasible_fd], 'b-', lw=2)
axes[0].set_xlabel("next-period debt $b'$")
axes[0].set_ylabel(r"$\phi^{fd}(b', \xi_1)$")

axes[1].plot(b_prime_grid[feasible_fd], V_fd_mid[feasible_fd], 'r-', lw=2)
axes[1].set_xlabel("next-period debt $b'$")
axes[1].set_ylabel(r"$V^{fd}(b', \xi_1)$")

plt.tight_layout()
plt.show()
```

```{solution-end}
```

```{exercise-start}
:label: dovis_ex2
```

Run the particle filter on the synthetic data with different numbers of particles ($N = 500, 2000, 10000$).

Plot the recovered $\xi$ path for each and assess convergence.

```{exercise-end}
```

```{solution-start} dovis_ex2
:class: dropdown
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Particle filter sensitivity
    name: fig-pf-sensitivity
---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (N_part, color) in enumerate(zip(
        [500, 2000, 10000], ['tab:orange', 'tab:blue', 'tab:green'])):
    pf_k = jax.random.PRNGKey(100 + i)
    θ_f, ξ_f, b_f, φ_f, ll = particle_filter(
        y_data, pf_k, N_part,
        b_init=0.20, φ_init=3.0,
        θ_bar=p.θ_bar, ξ_init=-0.3,
        ρ_θ=p.ρ_θ, σ_θ=p.σ_θ,
        ρ_ξ=p.ρ_ξ, σ_ξ=p.σ_ξ,
        κ=p.κ, η_m=p.η_m, λ=p.λ,
        σ_π=2.5, σ_b=3.0
    )

    axes[0].plot(years, ξ_f, color=color, lw=2,
                 label=f'N={N_part} (LL={ll:.1f})')
    axes[1].plot(years, θ_f, color=color, lw=2,
                 label=f'N={N_part}')

axes[0].set_ylabel('ξ_1')
axes[0].set_xlabel('year')
axes[0].legend()
axes[0].axvline(1960 + t_reform, color='gray', ls='--', alpha=0.5)

axes[1].set_ylabel('θ')
axes[1].set_xlabel('year')
axes[1].legend()
axes[1].axvline(1960 + t_reform, color='gray', ls='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

```{solution-end}
```
