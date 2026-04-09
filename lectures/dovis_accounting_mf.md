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

This lecture studies a model of fiscal and monetary policy interactions developed by {cite:t}`DovisAccountingMFrevised`.

The model provides a framework for  revisiting some long-standing questions about **fiscal dominance** versus **monetary dominance** in a framework that allows for **partial commitment** to an inflation target.


```{note}
For an early discussion of "partial commitment" in the context of fiscal and monetary policy, see the concluding section of {cite:t}`LucasStokey1982`,  the original working paper version of {cite:t}`LucasStokey1983`. 

In Quantecon's view, the referees and editors of the *Journal of Monetary Economics* version made a mistake by insisting that Lucas and Stokey rewrite the concluding section of their paper.
```

```{note}
{cite:t}`SargentWallace1981` contrasted "fiscal dominance" and "monetary dominance"  as different ways of  coordinating
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
from typing import NamedTuple
from functools import partial
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

jax.config.update("jax_enable_x64", True)
```

## The economy

### Environment

Consider an economy that blends elements of {cite:t}`AMSS_2002` and {cite:t}`Calvo1978`
(see also {cite:t}`LucasStokey1983` and {cite:t}`ChariKehoe1999`).

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

An important innovation of {cite:t}`DovisAccountingMFrevised` is to model policy determination under
**partial commitment** in the following sense.

The government promises an inflation target $\pi^*$ (equivalently, a
promised value for real balances $\phi'$) for  next period. 

But a next-period government
can choose to **honor** or **abrogate** the mandate.

The cost of abrogating is a random variable $\xi$ that captures the legal, reputational, and political hurdles associated with overriding the mandate:

- reputational losses (see {cite:t}`AtkesonKehoe2001`, {cite:t}`DovisKirpalani2021`)
- coordination failures that lead to inferior equilibria
- institutional constraints and political costs faced by policymakers

This specification *nests* both the Ramsey outcome (when $\xi$ is always large enough so that the mandate is always honored) and the Markov outcome (when $\xi = 0$ so the mandate is always abrogated).

The approach is related to the loose commitment framework of {cite:t}`DebortoliNunes2010`, but differs in that the regime is *endogenous*.

### Recursive formulation

The state is $x = (b, \phi, s)$ where $b$ is inherited real debt, $\phi$ is the promised real balances, and $s = (\theta, \xi)$ is the exogenous state.

This recursive formulation builds on {cite:t}`Abreu1988`, {cite:t}`ChariKehoe1990`, and {cite:t}`Chang1998`.

```{note}
For descriptions of these frameworks, see other lectures in this suite of QuantEcon lecture notes, including  {doc}`Ramsey plans, time inconsistency, sustainable plans <calvo>`,{doc}`competitive equilibria in the Chang model <chang_ramsey>`, and {doc}`sustainable plans in the Chang model <chang_credible>`.
```

The economy can be in one of two regimes:

- **Monetary dominance** (MD, $\eta = 1$): the government honors the inflation target.
- **Fiscal dominance** (FD, $\eta = 0$): the government ignores the target and chooses $\phi$ to   maximize short-run welfare.

The transition between these regimes is related to the influential work of {cite:t}`Leeper1991`, {cite:t}`Bianchi2013`, and {cite:t}`BianchiIlut2017`, but differs on two fronts: first, the switches from one regime to the other are **endogenous**, and second, the policy chosen in each regime is also endogenous rather than governed by exogenous monetary and fiscal rules.

The **regime indicator** is

$$
\eta(b', \phi', s') = \begin{cases}
1 & \text{if } V^{md}(b', \phi', s') \geq V^{fd}(b', s') - \xi(s') \\
0 & \text{otherwise}
\end{cases}
$$

The inflation target summarized by a promised $\phi$ is satisfied if and only if

$$
\xi \geq \xi^* = V^{fd}(b, s) - V^{md}(b, \phi, s)
  = \max_{\phi_{fd}} V^{md}(b, \phi_{fd}, s) - V^{md}(b, \phi, s).
$$

Deviating from the target allows the government to attain the maximum utility possible net of the cost $\xi$. A cost $\xi$ greater than the cutoff $\xi^*$ is required for the target to be sustained. More ambitious inflation targets (closer to the Ramsey value $\phi^*$) are harder to achieve because the cutoff $\xi^*$ is larger. The target is also easier to achieve when the marginal utility of government expenditure $\theta$ is lower, because positive surpluses are less valuable. {numref}`fig-credibility-targets` below illustrates this logic.

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
class DovisParams(NamedTuple):
    β: float              # Household discount factor
    β_hat: float          # Government discount factor
    χ: float              # Labor disutility parameter
    κ: float              # Money-demand parameter
    η_m: float            # Money-demand parameter
    ψ: float              # Inverse Frisch elasticity
    σ: float              # Inverse EIS for gov't expenditures
    θ_bar: float          # Mean θ
    ρ_θ: float            # Persistence of θ
    σ_θ: float            # Innovation std dev of θ
    α_l: float            # Prob of ξ₁ resetting to 0
    α_ξ: float            # Persistence of ξ₁
    ξ_bar: float          # Upper bound of ξ₁
    λ: float              # Gumbel scale parameter
    φ_star: float         # Satiation point κ/(2η_m)


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
    α_l = 0.005
    α_ξ = 0.99
    ξ_bar = 0.5
    λ = 20.0
    φ_star = κ / (2.0 * η_m)

    return DovisParams(
        β=β, β_hat=β_hat, χ=χ, κ=κ, η_m=η_m, ψ=ψ, σ=σ,
        θ_bar=θ_bar, ρ_θ=ρ_θ, σ_θ=σ_θ,
        α_l=α_l, α_ξ=α_ξ, ξ_bar=ξ_bar, λ=λ,
        φ_star=φ_star
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

One way to implement the Ramsey outcome is to delegate monetary policy to an independent central bank with a mandate to target an inflation rate of $\pi_R$, while fiscal policy is determined by the treasury, which solves a problem similar to that in the real economy studied by {cite:t}`AMSS_2002`, taking as given a constant flow of seigniorage revenues (which may be negative).

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
Δ_grid = jnp.linspace(-5.0, 3.0, 300)
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

### Credibility of inflation targets

We can now illustrate the credibility condition from the recursive formulation.
For a given level of inherited debt $b$ and optimal continuation choices, the value of entering the period with real balances $\phi$ is approximately

$$
V^{md}(\phi;\, b,\, \theta) \;\approx\; U(\phi + D,\, \theta) \;+\; v(\phi),
$$

where $D = b - \beta b' - \text{seigniorage}$ collects the non-$\phi$ terms in the surplus $\Delta$. This is hump-shaped in $\phi$: low $\phi$ sacrifices money utility while high $\phi$ forces a large, costly surplus.

Under fiscal dominance the government picks $\phi^{fd}$ to maximize this expression, so $V^{fd} = \max_\phi V^{md}(\phi)$.
The cutoff cost $\xi^* = V^{fd} - V^{md}(\phi')$ is the temptation to deviate from the promised $\phi'$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: "Credibility of inflation targets"
    name: fig-credibility-targets
---
p = params_la
iu_vec = vmap(indirect_utility, in_axes=(0, None, None, None, None))

# φ grid — from near zero up to just below φ* = κ/(2η_m)
phi_max = p.κ / (2.0 * p.η_m) * 0.99
phi_grid = jnp.linspace(0.05, phi_max, 500)

# Surplus: Δ = φ + D where D = b − βb' − seigniorage
# (higher φ ⟹ larger surplus ⟹ more costly)
D = 1.0
Delta_grid = phi_grid + D

# Two θ values: baseline (high fiscal pressure) and low θ_L
θ_high = 150.0
θ_low  = 100.0

U_high, _ = iu_vec(Delta_grid, θ_high, p.χ, p.ψ, p.σ)
U_low, _  = iu_vec(Delta_grid, θ_low,  p.χ, p.ψ, p.σ)
v_phi = v_money(phi_grid, p.κ, p.η_m)

V_blue_raw = np.array(U_high + v_phi)   # baseline θ (high)
V_red_raw  = np.array(U_low  + v_phi)   # θ_L (low)
phi_np = np.array(phi_grid)

mask_b = V_blue_raw > -1e5
mask_r = V_red_raw  > -1e5

# Shift curves so peaks are at comparable heights
# (the absolute level difference is driven by θ·u(g), not the φ tradeoff)
V_blue = V_blue_raw - V_blue_raw[mask_b].min()
V_red  = V_red_raw  - V_red_raw[mask_r].min()
V_blue += V_red.max() - V_blue.max()   # align peaks

# Find peaks (φ^fd for each θ)
idx_peak_b = np.argmax(V_blue[mask_b])
idx_peak_r = np.argmax(V_red[mask_r])
phi_fd_b = phi_np[mask_b][idx_peak_b]
phi_fd_r = phi_np[mask_r][idx_peak_r]
V_peak_b = V_blue[mask_b][idx_peak_b]
V_peak_r = V_red[mask_r][idx_peak_r]

# Promise point φ' — to the right of both peaks
phi_prime = 0.5 * (phi_fd_r + phi_max)
phi_star  = phi_prime + 0.4

# Values at the promise
V_at_b = float(np.interp(phi_prime, phi_np[mask_b], V_blue[mask_b]))
V_at_r = float(np.interp(phi_prime, phi_np[mask_r], V_red[mask_r]))

# ---- Plot ----
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(phi_np[mask_b], V_blue[mask_b], 'b-', lw=2.5)
ax.plot(phi_np[mask_r], V_red[mask_r],  'r-', lw=2.5)

# Vertical dashed lines
for xv in [phi_fd_b, phi_fd_r, phi_prime, phi_star]:
    ax.axvline(xv, ls='--', color='gray', alpha=0.35, lw=0.7)

# Horizontal dashed lines at peak levels
ax.hlines(V_peak_b, phi_fd_b, phi_prime,
          ls='--', color='blue', alpha=0.35, lw=0.7)
ax.hlines(V_peak_r, phi_fd_r, phi_prime,
          ls='--', color='red', alpha=0.35, lw=0.7)

# Horizontal dashed lines at promise levels
ax.hlines(V_at_b, phi_fd_b, phi_prime,
          ls='--', color='blue', alpha=0.25, lw=0.7)
ax.hlines(V_at_r, phi_fd_r, phi_prime,
          ls='--', color='red', alpha=0.25, lw=0.7)

# ξ* double arrows — both measured at φ'
# Blue ξ* (large gap): arrow just LEFT of φ', label further left
x_xi_b = phi_fd_b
ax.annotate('', xy=(x_xi_b, V_peak_b),
            xytext=(x_xi_b, V_at_b),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax.text(x_xi_b - 0.25, 0.5 * (V_peak_b + V_at_b),
        r'$\xi^*$', fontsize=14, color='blue',
        ha='right', va='center')

# Red ξ̂* (small gap): arrow just RIGHT of φ', label further right
x_xi_r = phi_fd_r
ax.annotate('', xy=(x_xi_r, V_peak_r),
            xytext=(x_xi_r, V_at_r),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax.text(x_xi_r + 0.25, 0.5 * (V_peak_r + V_at_r),
        r'$\hat{\xi}^*$', fontsize=14, color='red',
        ha='left', va='center')

# Curve labels — place in the right half where curves separate
x_lab_r = phi_prime + 0.9
x_lab_b = phi_np[mask_b][-1] * 0.92
ax.text(x_lab_r,
        float(np.interp(x_lab_r, phi_np[mask_r], V_red[mask_r])),
        r'$V_{md}(\phi,\,\theta_L)$', fontsize=13, color='red',
        va='bottom', ha='left')
ax.text(x_lab_b,
        float(np.interp(x_lab_b, phi_np[mask_b], V_blue[mask_b])),
        r'$V_{md}(\phi)$', fontsize=13, color='blue',
        va='top', ha='left')

# x-axis labels
trans = ax.get_xaxis_transform()
for xv, lab in [(phi_fd_b, r'$\phi_{fd}$'),
                (phi_fd_r, r'$\hat\phi_{fd}$'),
                (phi_prime, r"$\phi'$"),
                (phi_star, r'$\phi^*$')]:
    ax.text(xv, -0.06, lab, transform=trans,
            fontsize=12, ha='center', va='top', clip_on=False)

ax.set_ylabel(r'$V$', fontsize=14)
ax.set_xlabel(r'$\phi$', fontsize=14)
ax.tick_params(labelbottom=False, labelleft=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
```

The blue curve plots $V^{md}(\phi) = U(\phi + D,\, \theta) + v(\phi)$ for baseline fiscal pressure $\theta$, and the red curve for a lower value $\theta_L$. 

Each curve peaks at the corresponding $\phi^{fd}$, the level of real balances chosen under fiscal dominance. The gap $\xi^*$ between the peak (the fiscal-dominance value $V^{fd}$) and the value at the promised $\phi'$ is the minimum institutional cost needed to sustain the inflation target. 

With lower fiscal pressure (red), the gap shrinks: the target becomes easier to sustain.

## The full model with Gumbel shocks

Following the paper's computational approach, the cost $\xi$ is decomposed as

$$
\xi_t = \xi_{1,t} + \xi^{fd}_t - \xi^{md}_t,
$$

where $\xi_{1,t}$ is a persistent component and $\xi^{fd}_t$, $\xi^{md}_t$ are i.i.d. **Gumbel** shocks with mean zero.

The persistent component $\xi_{1,t}$ follows a Markov chain on $[0, \bar\xi]$ with the following transition probabilities:

$$
\Pr(\xi_1' = 0 \mid \xi_1) = \alpha_l, \qquad
\Pr(\xi_1' = \xi_1 \mid \xi_1) = \alpha, \qquad
\Pr(\xi_1' \sim \text{Uniform}[0, \bar\xi]) = 1 - \alpha_l - \alpha.
$$

The parameter $\alpha$ controls persistence, $\alpha_l$ is the probability of resetting to zero (making deviation costless), and $\bar\xi$ is a large upper bound.

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

We discretize $\xi_1$ via the paper's Markov chain on $[0, \bar\xi]$, then build grids for total liabilities $B$, debt $b'$, and promised real balances $\phi'$.

```{code-cell} ipython3
def build_ξ_grid(n_ξ, α_l, α_ξ, ξ_bar):
    """Build ξ_1 grid on [0, ξ_bar] and transition matrix.

    Transition rules (Section 6.1 of the paper):
      Pr(ξ_1' = 0  | ξ_1) = α_l,
      Pr(ξ_1' = ξ_1 | ξ_1) = α_ξ,
      with prob 1 - α_l - α_ξ, draw ξ_1' ~ Uniform{grid points}.
    """
    ξ_grid = np.linspace(0.0, ξ_bar, n_ξ)
    p_unif = (1.0 - α_l - α_ξ) / n_ξ
    P_ξ = np.full((n_ξ, n_ξ), p_unif)
    P_ξ[:, 0] += α_l          # reset to 0
    for i in range(n_ξ):
        P_ξ[i, i] += α_ξ      # stay at current value
    return ξ_grid, P_ξ
```

## Computational algorithm

Because the budget constraint depends on $b$ and $\phi$ only through their sum, the problem can be written in terms of a single endogenous state variable $B = b + \phi$ (total real government liabilities), as described in Appendix C of {cite:t}`DovisAccountingMFrevised`.

We define a **reduced continuation value** $W(B, s_1)$ that strips the current-period money utility $v(\phi)$ out of the recursive problem.  

The full value of entering a period with state $(b, \phi, s_1)$ is recovered as $v(\phi) + W(b + \phi, s_1)$ under monetary dominance, and the fiscal-dominance value is $V^{fd}(b', s_1') = \max_{\phi}\left[W(b' + \phi, s_1') + v(\phi)\right]$.

The function $W(B, s_1)$ satisfies

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

1. *Initialize* with a guess $W_0(B, s_1)$ (e.g., the Ramsey value)
2. For iteration $n$:
   - Compute $\phi^{fd}$ and $\bar\eta$ from the logit formula and the fiscal-dominance FOC
   - Compute the Bellman update $W_{n+1}$ from the value function equation above
3. *Iterate* until $\|W_{n+1} - W_n\| < \varepsilon$

We now set up a simplified version of the model with deterministic $\theta$, multiple $\xi_1$ states, and the quadratic money-utility specification.

Appendix C of the paper defines fiscal dominance recursively by

$$
V^{fd}(b', \xi') = \max_{\phi} \left[ W(b' + \phi, \xi') + v(\phi) \right].
$$

The Bellman solver below recovers $\phi^{fd}$ by a grid search over $\phi$ using the continuation value $W$, and linearly interpolates $W(B, \xi)$ over the liabilities grid via `jnp.interp`.

The inner expectation over $\xi'$ is a matrix multiply against $P_\xi$ via `jnp.einsum`, the search over $(b', \phi')$ is a vectorized `argmax`, and the VFI loop uses `lax.scan`.

The discrete grid causes the `argmax` to oscillate between adjacent grid points, preventing exact convergence. To produce smooth policy functions we average the last 50 VFI iterates, which cancels the oscillation while preserving the correct level of the value function.

We pack the grids, transition matrix, and precomputed indirect utility into a `DovisModel` named tuple.

```{code-cell} ipython3
class DovisModel(NamedTuple):
    """All grids and precomputed objects needed by the Bellman operator."""
    β: float
    β_hat: float
    κ: float
    η_m: float
    λ: float
    φ_star: float
    B_grid: jnp.ndarray     # (n_B,)
    b_grid: jnp.ndarray     # (n_b,)
    φ_grid: jnp.ndarray     # (n_φ,)
    ξ_grid: jnp.ndarray     # (n_ξ,)
    P_ξ: jnp.ndarray        # (n_ξ, n_ξ)
    Δ_fine: jnp.ndarray     # fine surplus grid
    U_fine: jnp.ndarray     # U(Δ, θ) on fine grid


def create_dovis_model(p, θ=None,
                       n_B=200, n_b=200, n_φ=150, n_ξ=9,
                       B_max=50.0):
    """Create a DovisModel from DovisParams *p*."""
    if θ is None:
        θ = p.θ_bar
    φ_lo, φ_hi = 0.5, p.φ_star * 0.99
    B_grid = jnp.linspace(0.1, B_max, n_B)
    b_bar = max(B_max - φ_hi, 0.1)
    b_grid = jnp.linspace(0.1, b_bar, n_b)
    φ_grid = jnp.linspace(φ_lo, φ_hi, n_φ)
    ξ_np, P_ξ_np = build_ξ_grid(n_ξ, p.α_l, p.α_ξ, p.ξ_bar)
    Δ_fine = jnp.linspace(-55.0, 20.0, 3000)
    U_fine, _ = vmap(lambda d: indirect_utility(d, θ, p.χ, p.ψ, p.σ))(Δ_fine)
    return DovisModel(
        β=p.β, β_hat=p.β_hat, κ=p.κ, η_m=p.η_m,
        λ=p.λ, φ_star=p.φ_star,
        B_grid=B_grid, b_grid=b_grid, φ_grid=φ_grid,
        ξ_grid=jnp.array(ξ_np), P_ξ=jnp.array(P_ξ_np),
        Δ_fine=Δ_fine, U_fine=U_fine
    )
```

We now define the **Bellman operator** `T(W, model)`.

Given the current guess $W$, it:

1. Computes $V^{fd}(b', \xi')$ and $\phi^{fd}(b', \xi')$ by maximizing $W(b'+\phi, \xi') + v(\phi)$ over the $\phi$ grid.
2. Computes $V^{md}(b', \phi', \xi') = W(b'+\phi', \xi') + v(\phi')$.
3. Evaluates the logit regime probability $\bar\eta$ and log-sum-exp continuation $\Omega$.
4. Takes expectations over $\xi_1'$ to get $EV$ and money-demand $J$.
5. For each state $(B, \xi_1)$, evaluates the budget constraint $\Delta = B - \beta b' - J$ and the indirect utility $U(\Delta)$ at every candidate $(b', \phi')$ pair, then takes the `argmax`.

```{code-cell} ipython3
def compute_continuation(W, model):
    """Compute continuation-value objects from W.

    Returns EV, J (both shape (n_b, n_φ, n_ξ)), and
    V_fd, H_fd (both shape (n_b, n_ξ)).
    """
    β, κ, η_m, lam = model.β, model.κ, model.η_m, model.λ
    B_grid, b_grid, φ_grid, ξ_grid, P_ξ = (
        model.B_grid, model.b_grid, model.φ_grid,
        model.ξ_grid, model.P_ξ)
    n_b, n_φ = len(b_grid), len(φ_grid)

    # B_candidate[i, k] = b'[i] + φ[k]
    B_cand = b_grid[:, None] + φ_grid[None, :]   # (n_b, n_φ)

    # Interpolate W at all candidate B values for each ξ' column
    def interp_col(W_col):
        return jnp.interp(B_cand.ravel(), B_grid, W_col).reshape(n_b, n_φ)
    W_at = vmap(interp_col, in_axes=1, out_axes=2)(W)  # (n_b, n_φ, n_ξ)

    v_φ = v_money(φ_grid, κ, η_m)          # (n_φ,)
    total = W_at + v_φ[None, :, None]       # (n_b, n_φ, n_ξ)

    # V^fd = max_φ total,  V^md = total (for each φ')
    V_fd = jnp.max(total, axis=1)           # (n_b, n_ξ)
    φ_fd = φ_grid[jnp.argmax(total, axis=1)]
    H_fd = H_func(φ_fd, κ, η_m)
    V_md = total                             # (n_b, n_φ, n_ξ)

    # Logit η̄ and log-sum-exp Ω
    z = lam * (V_md - V_fd[:, None, :] + ξ_grid[None, None, :])
    η_bar = jax.nn.sigmoid(z)
    Ω = jnp.logaddexp(
        lam * V_md,
        lam * (V_fd[:, None, :] - ξ_grid[None, None, :])
    ) / lam

    # Expectations over ξ₁'
    EV = jnp.einsum('abj,kj->abk', Ω, P_ξ)
    H_φ = H_func(φ_grid, κ, η_m)
    H_comb = η_bar * H_φ[None, :, None] + (1.0 - η_bar) * H_fd[:, None, :]
    J = β * jnp.einsum('abj,kj->abk', H_comb, P_ξ)

    return EV, J, V_fd, H_fd


def T(W, model):
    """Bellman operator."""
    β, β_hat = model.β, model.β_hat
    B_grid, b_grid = model.B_grid, model.b_grid
    Δ_fine, U_fine = model.Δ_fine, model.U_fine
    n_B, n_b, n_φ, n_ξ = (
        len(B_grid), len(b_grid),
        len(model.φ_grid), len(model.ξ_grid))

    EV, J, _, _ = compute_continuation(W, model)

    # Budget constraint: Δ[b', φ', B, ξ₁]
    Δ = (B_grid[None, None, :, None]
         - β * b_grid[:, None, None, None]
         - J[:, :, None, :])

    U_all = jnp.interp(Δ.ravel(), Δ_fine, U_fine).reshape(Δ.shape)
    feasible = (Δ > Δ_fine[0]) & (Δ < Δ_fine[-1])
    U_all = jnp.where(feasible, U_all, -1e15)

    val = U_all + β_hat * EV[:, :, None, :]
    return jnp.max(val.reshape(n_b * n_φ, n_B, n_ξ), axis=0)


def get_greedy(W, model):
    """Greedy policy: flat index into (b', φ') grid."""
    β, β_hat = model.β, model.β_hat
    B_grid, b_grid = model.B_grid, model.b_grid
    Δ_fine, U_fine = model.Δ_fine, model.U_fine
    n_B, n_b, n_φ, n_ξ = (
        len(B_grid), len(b_grid),
        len(model.φ_grid), len(model.ξ_grid))

    EV, J, _, _ = compute_continuation(W, model)

    Δ = (B_grid[None, None, :, None]
         - β * b_grid[:, None, None, None]
         - J[:, :, None, :])
    U_all = jnp.interp(Δ.ravel(), Δ_fine, U_fine).reshape(Δ.shape)
    feasible = (Δ > Δ_fine[0]) & (Δ < Δ_fine[-1])
    U_all = jnp.where(feasible, U_all, -1e15)

    val = U_all + β_hat * EV[:, :, None, :]
    return jnp.argmax(val.reshape(n_b * n_φ, n_B, n_ξ), axis=0)
```

### Solving the model

We solve via value function iteration using `lax.scan`.
After a burn-in phase the Bellman operator has settled near the fixed point;
we then collect 50 additional iterates and average them to cancel discrete-grid oscillation.

```{code-cell} ipython3
def solve_dovis(model, p, n_iter=300, n_avg=50, verbose=True):
    """Solve via VFI, averaging the last n_avg iterates."""
    n_B = model.B_grid.shape[0]
    n_ξ = model.ξ_grid.shape[0]
    U0, _ = indirect_utility(0.0, p.θ_bar, p.χ, p.ψ, p.σ)
    W_init = jnp.full((n_B, n_ξ), U0 / (1.0 - model.β_hat))
    n_burn = n_iter - n_avg

    @jit
    def run_vfi(W_init):
        def burn_step(W, _):
            return T(W, model), None
        W_burned, _ = lax.scan(burn_step, W_init, None, length=n_burn)
        def avg_step(W, _):
            W_new = T(W, model)
            return W_new, W_new
        _, W_stack = lax.scan(avg_step, W_burned, None, length=n_avg)
        return jnp.mean(W_stack, axis=0)

    if verbose:
        print(f"Starting VFI ({n_burn} burn-in + {n_avg} averaging)...")
    from time import time
    t0 = time()
    W = run_vfi(W_init)
    W.block_until_ready()
    W_check = jit(T)(W, model)
    W_check.block_until_ready()
    err = float(jnp.max(jnp.abs(W_check - W)))
    if verbose:
        print(f"  Done in {time()-t0:.1f}s, Bellman residual = {err:.4f}")
    return W


def extract_policies(W, model):
    """Extract (b', φ', Δ) from converged W."""
    n_φ = len(model.φ_grid)
    σ = jit(get_greedy)(W, model)
    pol_b = model.b_grid[σ // n_φ]
    pol_φ = model.φ_grid[σ % n_φ]
    EV, J, _, _ = compute_continuation(W, model)
    n_b = len(model.b_grid)
    J_flat = J.reshape(n_b * n_φ, len(model.ξ_grid))
    J_σ = jnp.take_along_axis(J_flat, σ, axis=0)
    pol_Δ = model.B_grid[:, None] - model.β * pol_b - J_σ
    return pol_b, pol_φ, pol_Δ
```

The first call triggers JIT compilation, which takes a moment.

```{code-cell} ipython3
p = params_la
model = create_dovis_model(p)
W = solve_dovis(model, p)
pol_b, pol_φ, pol_Δ = extract_policies(W, model)
B_grid = model.B_grid
ξ_grid = model.ξ_grid
n_ξ_coarse = len(ξ_grid)
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

To line the simulations up with Figures 3 and 4 of the paper, we treat the two experiments differently.

We also initialize the two IRFs from different parts of the state space: a **high-debt** fiscal-dominant state for the fundamental disinflation and a **low-debt** fiscal-dominant state for the institutional disinflation. This mirrors the paper's figures, which are designed to highlight the contrasting debt dynamics rather than to start from the same debt level.

For the **fundamental disinflation**, we keep the *realized* path in fiscal dominance throughout, matching the paper's description of a low-$\xi$ path. We recover current real balances from the continuation-value object

$$
V^{fd}(b, s_1) = \max_{\phi}\left[W(b + \phi, s_1) + v(\phi)\right]
$$

rather than from a static approximation, and then feed the resulting realized liabilities $B_t = b_t + \phi_t$ into the low-$\xi$ policy rules for next-period debt and promised real balances.

For the **institutional disinflation**, we evaluate the current target-honoring probability using the inherited promise $\phi_t$ and the current continuation values, then let the realized regime switch endogenously once the promise becomes sufficiently credible.

```{code-cell} ipython3
def solve_policy_cache(θ_nodes, p, n_iter=300, n_avg=50,
                       n_B=200, n_b=200, n_φ=150, n_ξ=9,
                       B_max=50.0, verbose=False):
    """Solve the model for several θ values and store policy arrays."""
    results = {'θ_nodes': np.asarray(θ_nodes, dtype=float)}

    W_list, b_list, φ_list, Δ_list = [], [], [], []
    for θ_val in θ_nodes:
        if verbose:
            print(f'  θ = {θ_val:.1f} ...', end=' ', flush=True)
        m = create_dovis_model(p, θ=θ_val,
                               n_B=n_B, n_b=n_b, n_φ=n_φ,
                               n_ξ=n_ξ, B_max=B_max)
        W = solve_dovis(m, p, n_iter=n_iter, n_avg=n_avg, verbose=False)
        pol_b, pol_φ, pol_Δ = extract_policies(W, m)
        W_list.append(np.asarray(W))
        b_list.append(np.asarray(pol_b))
        φ_list.append(np.asarray(pol_φ))
        Δ_list.append(np.asarray(pol_Δ))
        if verbose:
            print('done.')

    results['W'] = np.stack(W_list)      # (n_θ, n_B, n_ξ)
    results['b'] = np.stack(b_list)      # (n_θ, n_B, n_ξ)
    results['φ'] = np.stack(φ_list)
    results['Δ'] = np.stack(Δ_list)

    # Store grids from the last model (all share the same grids)
    results['B_grid'] = np.asarray(m.B_grid)
    results['ξ_grid'] = np.asarray(m.ξ_grid)
    results['b_grid'] = np.asarray(m.b_grid)
    results['φ_grid'] = np.asarray(m.φ_grid)

    # Compute V^fd and φ^fd for each (θ, b', ξ')
    V_fd_list, φ_fd_list = [], []
    B_g = results['B_grid']
    b_g = results['b_grid']
    φ_g = results['φ_grid']
    for θi in range(len(θ_nodes)):
        W_θ = results['W'][θi]   # (n_B, n_ξ)
        n_ξ_val = W_θ.shape[1]
        V_fd = np.empty((len(b_g), n_ξ_val))
        φ_fd = np.empty((len(b_g), n_ξ_val))
        for j in range(n_ξ_val):
            for i, bp in enumerate(b_g):
                B_cand = bp + φ_g
                W_interp = np.interp(B_cand, B_g, W_θ[:, j])
                v_vals = p.κ * φ_g - p.η_m * φ_g**2
                total = W_interp + v_vals
                best = np.argmax(total)
                V_fd[i, j] = total[best]
                φ_fd[i, j] = float(φ_g[best])
        V_fd_list.append(V_fd)
        φ_fd_list.append(φ_fd)

    results['V_fd'] = np.stack(V_fd_list)  # (n_θ, n_b, n_ξ)
    results['φ_fd'] = np.stack(φ_fd_list)
    return results


def interp_1d(x, grid, values):
    """Linear interpolation with clipping."""
    x = float(np.clip(x, grid[0], grid[-1]))
    return float(np.interp(x, grid, values))


def current_eta_prob(b, φ_promise, θi, ξi, cache, p):
    """Probability that today's inherited target is honored."""
    B_g = cache['B_grid']
    ξ_val = cache['ξ_grid'][ξi]

    B_md = float(np.clip(b + φ_promise, B_g[0], B_g[-1]))
    V_md = interp_1d(B_md, B_g, cache['W'][θi, :, ξi])
    V_md += float(v_money(φ_promise, p.κ, p.η_m))

    b_g = cache['b_grid']
    V_fd = interp_1d(b, b_g, cache['V_fd'][θi, :, ξi])

    z = p.λ * (V_md - V_fd + ξ_val)
    η_prob = 1.0 / (1.0 + np.exp(-z))
    return η_prob, V_md, V_fd


def interp_fd(cache, θi, ξi, b):
    """Interpolate FD value and policy at inherited debt b."""
    b_g = cache['b_grid']
    φ_fd = interp_1d(b, b_g, cache['φ_fd'][θi, :, ξi])
    V_fd = interp_1d(b, b_g, cache['V_fd'][θi, :, ξi])
    return φ_fd, V_fd


def find_fd_steady_state(θ_idx, ξ_idx, cache, p):
    """Steady state for a path that remains in fiscal dominance."""
    B_g = cache['B_grid']
    b = 5.0
    φ_pr = p.φ_star * 0.6
    for _ in range(500):
        φ_fd, _ = interp_fd(cache, θ_idx, ξ_idx, b)
        B = float(np.clip(b + φ_fd, B_g[0], B_g[-1]))
        b_next = interp_1d(B, B_g, cache['b'][θ_idx, :, ξ_idx])
        φ_next = interp_1d(B, B_g, cache['φ'][θ_idx, :, ξ_idx])
        if abs(b_next - b) + abs(φ_next - φ_pr) < 1e-8:
            break
        b = 0.5 * b + 0.5 * b_next
        φ_pr = 0.5 * φ_pr + 0.5 * φ_next
    return b, φ_pr


def static_allocation(Δ, θ, χ, ψ, σ):
    """Recover equilibrium labor l and government spending g from surplus Δ."""
    g_star = θ ** (1.0 / σ)
    l_peak = (1.0 / ((1.0 + ψ) * χ)) ** (1.0 / ψ)
    T_max = (1.0 - χ * l_peak ** ψ) * l_peak

    if Δ <= -g_star:
        return (1.0 / χ) ** (1.0 / ψ), g_star, 0.0
    if Δ >= 0.999 * T_max:
        return l_peak, 1e-8, 1000.0

    lo, hi = 0.0, 1000.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        g_val = (θ / (1.0 + mid)) ** (1.0 / σ)
        denom = max(χ * (1.0 + mid * (1.0 + ψ)), 1e-15)
        l_val = max((1.0 + mid) / denom, 1e-15) ** (1.0 / ψ)
        if (1.0 - χ * l_val ** ψ) * l_val - g_val <= Δ:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    g_opt = (θ / (1.0 + lam)) ** (1.0 / σ)
    denom = max(χ * (1.0 + lam * (1.0 + ψ)), 1e-15)
    l_opt = max((1.0 + lam) / denom, 1e-15) ** (1.0 / ψ)
    return l_opt, g_opt, lam


def simulate_fundamental_irf(
        b0, φ_promise0, θ_idx_path, ξ_idx, cache, p, t_shock):
    """Simulate a fundamental disinflation (FD throughout)."""
    T = len(θ_idx_path)
    B_g = cache['B_grid']
    θ_nodes = cache['θ_nodes']
    ξi = int(ξ_idx)

    out = {k: np.zeros(T) for k in
           ['b', 'φ', 'φ_prime', 'Δ', 'π', 'η', 'η_prob', 'regime',
            'debt_gdp', 'surplus_gdp']}

    θi_pre = int(θ_idx_path[0])
    η_pre, _, _ = current_eta_prob(b0, φ_promise0, θi_pre, ξi, cache, p)
    φ_pre, _ = interp_fd(cache, θi_pre, ξi, b0)
    B_pre = float(np.clip(b0 + φ_pre, B_g[0], B_g[-1]))
    b_next_pre = interp_1d(B_pre, B_g, cache['b'][θi_pre, :, ξi])
    φ_next_pre = interp_1d(B_pre, B_g, cache['φ'][θi_pre, :, ξi])
    Δ_pre = interp_1d(B_pre, B_g, cache['Δ'][θi_pre, :, ξi])
    J_prev = B_pre - Δ_pre - p.β * b_next_pre
    π_pre = (J_prev / max(φ_pre, 1e-12) - 1.0) * 100.0
    l_pre, _, _ = static_allocation(Δ_pre, float(θ_nodes[θi_pre]), p.χ, p.ψ, p.σ)

    for k, v in [('b', b0), ('φ', φ_pre), ('φ_prime', φ_promise0),
                 ('Δ', Δ_pre), ('π', π_pre), ('η', η_pre),
                 ('η_prob', η_pre), ('regime', 0.0),
                 ('debt_gdp', 100*b0/max(l_pre,1e-12)),
                 ('surplus_gdp', 100*Δ_pre/max(l_pre,1e-12))]:
        out[k][:t_shock] = v

    b, φ_promise = float(b0), float(φ_promise0)
    for t in range(t_shock, T):
        θi = int(θ_idx_path[t])
        η_prob, _, _ = current_eta_prob(b, φ_promise, θi, ξi, cache, p)
        φ_t, _ = interp_fd(cache, θi, ξi, b)
        B = float(np.clip(b + φ_t, B_g[0], B_g[-1]))
        π_t = (J_prev / max(φ_t, 1e-12) - 1.0) * 100.0
        b_prime = interp_1d(B, B_g, cache['b'][θi, :, ξi])
        φ_prime = interp_1d(B, B_g, cache['φ'][θi, :, ξi])
        Δ_t = interp_1d(B, B_g, cache['Δ'][θi, :, ξi])
        l_t, _, _ = static_allocation(Δ_t, float(θ_nodes[θi]), p.χ, p.ψ, p.σ)

        out['b'][t] = b; out['φ'][t] = φ_t
        out['φ_prime'][t] = φ_prime; out['Δ'][t] = Δ_t
        out['π'][t] = π_t; out['η'][t] = 0.0
        out['η_prob'][t] = η_prob; out['regime'][t] = 0.0
        out['debt_gdp'][t] = 100*b/max(l_t,1e-12)
        out['surplus_gdp'][t] = 100*Δ_t/max(l_t,1e-12)

        J_prev = B - Δ_t - p.β * b_prime
        b, φ_promise = float(b_prime), float(φ_prime)
    return out


def simulate_institutional_irf(
        b0, φ_promise0, θ_idx_path, ξ_idx_path, cache, p, t_shock):
    """Simulate an institutional disinflation (endogenous regime switch)."""
    T = len(ξ_idx_path)
    B_g = cache['B_grid']
    θ_nodes = cache['θ_nodes']

    out = {k: np.zeros(T) for k in
           ['b', 'φ', 'φ_prime', 'Δ', 'π', 'η', 'η_prob', 'regime',
            'debt_gdp', 'surplus_gdp']}

    θi_pre, ξi_pre = int(θ_idx_path[0]), int(ξ_idx_path[0])
    η_pre, _, _ = current_eta_prob(b0, φ_promise0, θi_pre, ξi_pre, cache, p)
    φ_pre, _ = interp_fd(cache, θi_pre, ξi_pre, b0)
    B_pre = float(np.clip(b0 + φ_pre, B_g[0], B_g[-1]))
    b_next_pre = interp_1d(B_pre, B_g, cache['b'][θi_pre, :, ξi_pre])
    Δ_pre = interp_1d(B_pre, B_g, cache['Δ'][θi_pre, :, ξi_pre])
    J_prev = B_pre - Δ_pre - p.β * b_next_pre
    π_pre = (J_prev / max(φ_pre, 1e-12) - 1.0) * 100.0
    l_pre, _, _ = static_allocation(Δ_pre, float(θ_nodes[θi_pre]), p.χ, p.ψ, p.σ)

    for k, v in [('b', b0), ('φ', φ_pre), ('φ_prime', φ_promise0),
                 ('Δ', Δ_pre), ('π', π_pre), ('η', η_pre),
                 ('η_prob', η_pre), ('regime', 0.0),
                 ('debt_gdp', 100*b0/max(l_pre,1e-12)),
                 ('surplus_gdp', 100*Δ_pre/max(l_pre,1e-12))]:
        out[k][:t_shock] = v

    b, φ_promise = float(b0), float(φ_promise0)
    for t in range(t_shock, T):
        θi, ξi = int(θ_idx_path[t]), int(ξ_idx_path[t])
        η_t, _, _ = current_eta_prob(b, φ_promise, θi, ξi, cache, p)
        regime_t = float(η_t >= 0.5)
        φ_fd_t, _ = interp_fd(cache, θi, ξi, b)
        φ_t = float(φ_promise if regime_t else φ_fd_t)
        B = float(np.clip(b + φ_t, B_g[0], B_g[-1]))
        π_t = (J_prev / max(φ_t, 1e-12) - 1.0) * 100.0
        b_prime = interp_1d(B, B_g, cache['b'][θi, :, ξi])
        φ_prime = interp_1d(B, B_g, cache['φ'][θi, :, ξi])
        Δ_t = interp_1d(B, B_g, cache['Δ'][θi, :, ξi])
        l_t, _, _ = static_allocation(Δ_t, float(θ_nodes[θi]), p.χ, p.ψ, p.σ)

        out['b'][t] = b; out['φ'][t] = φ_t
        out['φ_prime'][t] = φ_prime; out['Δ'][t] = Δ_t
        out['π'][t] = π_t; out['η'][t] = η_t
        out['η_prob'][t] = η_t; out['regime'][t] = regime_t
        out['debt_gdp'][t] = 100*b/max(l_t,1e-12)
        out['surplus_gdp'][t] = 100*Δ_t/max(l_t,1e-12)

        J_prev = B - Δ_t - p.β * b_prime
        b, φ_promise = float(b_prime), float(φ_prime)
    return out
```

We solve the model for three values of $\theta$ needed by the two IRF experiments, then run the simulations.

```{code-cell} ipython3
θ_fund_L, θ_fund_H = 12.0, 22.0
θ_inst_val = p.θ_bar  # 130
θ_irf = np.array([θ_fund_L, θ_fund_H, θ_inst_val])

cache = solve_policy_cache(θ_irf, p, verbose=True)

T, t_shock = 60, 10
ξ_pre, ξ_post = 0, n_ξ_coarse // 2

# Fundamental disinflation: FD steady state at θ_H (idx 1), ξ=0
b0_fund, φ0_fund = find_fd_steady_state(1, 0, cache, p)
θ_fund_idx = np.where(np.arange(T) < t_shock, 1, 0).astype(int)
irf_fund = simulate_fundamental_irf(
    b0_fund, φ0_fund, θ_fund_idx, 0, cache, p, t_shock
)

# Institutional disinflation: FD steady state at θ_inst (idx 2), ξ=0
b0_inst, φ0_inst = find_fd_steady_state(2, ξ_pre, cache, p)
θ_inst_idx = np.full(T, 2, dtype=int)
ξ_inst_idx = np.where(
    np.arange(T) < t_shock, ξ_pre, ξ_post).astype(int)
irf_inst = simulate_institutional_irf(
    b0_inst, φ0_inst, θ_inst_idx, ξ_inst_idx, cache, p, t_shock
)

print(f"Fundamental initial state: b0 = {b0_fund:.4f}, φ0' = {φ0_fund:.4f}")
print(f"Institutional initial state: b0 = {b0_inst:.4f}, φ0' = {φ0_inst:.4f}")

time = np.arange(T) - t_shock
θ_fund = np.where(np.arange(T) < t_shock, θ_fund_H, θ_fund_L)
θ_inst = np.full(T, θ_inst_val)
ξ_inst = np.asarray(cache['ξ_grid'])[ξ_inst_idx]
```


```{code-cell} ipython3
def plot_irf(irf, θ_path, ξ_path, time, title):
    """Plot IRF figure."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=14, y=1.01)
    kw = dict(lw=2, color='tab:blue')
    vkw = dict(color='k', ls=':', alpha=0.4)

    axes[0, 0].plot(time, θ_path, **kw)
    axes[0, 0].set_title(r'$\theta$'); axes[0, 0].axvline(0, **vkw)
    axes[0, 1].plot(time, ξ_path, **kw)
    axes[0, 1].set_title(r'$\xi$'); axes[0, 1].axvline(0, **vkw)
    axes[1, 0].plot(time, irf['regime'], **kw, label='regime')
    axes[1, 0].plot(time, irf['η_prob'], lw=2, color='tab:blue',
                    ls='--', label='Pr target met')
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set_title('Regime (0 = FD) and Pr target met (dashed)')
    axes[1, 0].legend()
    axes[1, 0].axvline(0, **vkw)
    axes[1, 1].plot(time, irf['π'], **kw)
    axes[1, 1].set_title('Inflation Rate (%)'); axes[2, 1].axvline(0, **vkw)
    axes[1, 1].axvline(0, **vkw)
    axes[2, 0].plot(time, irf['φ'], **kw)
    axes[2, 0].set_title(r'Current $\phi$')
    axes[2, 0].set_xlabel('time'); axes[2, 0].axvline(0, **vkw)
    axes[2, 1].plot(time, irf['φ_prime'], **kw)
    axes[2, 1].set_title(r"Promised $\phi'$")
    axes[2, 1].set_xlabel('time'); axes[2, 1].axvline(0, **vkw)
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

**Fundamental disinflation** ({numref}`fig-fundamental`): a permanent drop in $\theta$ from $\theta_H$ to $\theta_L$ reduces fiscal pressure.

The realized regime remains fiscal dominant throughout (solid line), while the target-honoring probability (dashed line) stays well below the institutional case.

Following the shock, the debt-to-GDP ratio declines, while the inflation rate initially drops sharply in the period when $\theta$ falls, reflecting the lower marginal cost of generating primary surpluses.

Inflation then continues to decline gradually, tracking the falling path of debt, which further reduces the marginal cost of surpluses.

Inflation and debt move *together* downward -- the signature of fundamental disinflation.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Institutional disinflation
    name: fig-institutional
---
fig = plot_irf(irf_inst, θ_inst, ξ_inst, time,
               'Institutional disinflation')
plt.show()
```

**Institutional disinflation** ({numref}`fig-institutional`): a permanent rise in $\xi$ pushes the economy from fiscal dominance toward monetary dominance.

The realized regime switches to monetary dominance and the target-honoring probability jumps close to one.

The inflation rate initially jumps downward at $t_0$ as real balances move toward the promised value $\phi' \approx \phi^*$.

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

1. *Initialize*: Draw $N$ particles $\{S_0^{(i)}\}_{i=1}^N$ from the prior
2. *For* $t = 1, \ldots, T$:
   
    *Propagate*: For each particle $i$, draw $\varepsilon_{t}^{(i)}$ and compute
      $S_t^{(i)} = k(S_{t-1}^{(i)}, \varepsilon_t^{(i)})$

   *Weight*: Compute the likelihood of observed $y_t$ given $S_t^{(i)}$:
      $w_t^{(i)} = p(y_t | S_t^{(i)}) \propto
      \exp\!\left(-\frac{1}{2}(y_t - f(S_t^{(i)}))^\top \Sigma^{-1}
      (y_t - f(S_t^{(i)}))\right)$

   *Normalize* weights: $\tilde w_t^{(i)} = w_t^{(i)} / \sum_j w_t^{(j)}$

   *Resample*: Draw $N$ particles from $\{S_t^{(i)}\}$ with probabilities
      $\{\tilde w_t^{(i)}\}$

3. *Output*: The filtered state estimate is the weighted average of particles

The JAX implementation below vectorizes over particles with `vmap` and loops over time with `lax.scan`.

Propagation and weighting are fully parallel across particles.

Resampling uses `jnp.searchsorted` on the cumulative weight array.

```{code-cell} ipython3
@partial(jit, static_argnums=(2,))
def particle_filter(y_data, key, N_particles,
                    b_init, φ_init, θ_bar, ξ_init,
                    ρ_θ, σ_θ, α_l, α_ξ, ξ_bar,
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
        jnp.clip(ξ_init + 0.1 * jax.random.normal(ks[3], (N_particles,)),
                 0.0, ξ_bar),
        φ_init_particles
    ])

    def propagate_one(particle, pk):
        b, φ, θ, ξ1, _ = particle
        k1, k2, k3 = jax.random.split(pk, 3)

        θ_new = jnp.maximum(
            θ_bar + ρ_θ * (θ - θ_bar) + σ_θ * jax.random.normal(k1), 1.0)

        # ξ_1 Markov chain: reset to 0 with prob α_l,
        # stay with prob α_ξ, uniform draw otherwise.
        u = jax.random.uniform(k2)
        ξ_uniform = jax.random.uniform(k3) * ξ_bar
        ξ_new = jnp.where(u < α_l, 0.0,
                    jnp.where(u < α_l + α_ξ, ξ1, ξ_uniform))

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
    θ_bar=p.θ_bar, ξ_init=0.1,
    ρ_θ=p.ρ_θ, σ_θ=p.σ_θ,
    α_l=p.α_l, α_ξ=p.α_ξ, ξ_bar=p.ξ_bar,
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

Moreover, movements in inflation are the direct consequence of deliberate policy choices as in {cite:t}`SargentWallace1981`, and not the result of an equilibrium selection mechanism.

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

The following code plots the fiscal-dominance policy $\phi^{fd}(b')$ and the associated value $V^{fd}(b', \xi_1)$ from continuation values.

```{code-cell} ipython3
# Use the cache from the institutional disinflation (θ = θ_bar)
θi_inst = 2   # index for θ_bar in cache
ξ_mid = n_ξ_coarse // 2
b_g = cache['b_grid']

φ_fd_mid = cache['φ_fd'][θi_inst, :, ξ_mid]
V_fd_mid = cache['V_fd'][θi_inst, :, ξ_mid]
feasible_fd = V_fd_mid > -1e12

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(b_g[feasible_fd], φ_fd_mid[feasible_fd], 'b-', lw=2)
axes[0].set_xlabel("next-period debt $b'$")
axes[0].set_ylabel(r"$\phi^{fd}(b', \xi_1)$")

axes[1].plot(b_g[feasible_fd], V_fd_mid[feasible_fd], 'r-', lw=2)
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

The following code runs the particle filter with different numbers of particles and plots the recovered paths to assess convergence.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (N_part, color) in enumerate(zip(
        [500, 2000, 10000], ['tab:orange', 'tab:blue', 'tab:green'])):
    pf_k = jax.random.PRNGKey(100 + i)
    θ_f, ξ_f, b_f, φ_f, ll = particle_filter(
        y_data, pf_k, N_part,
        b_init=0.20, φ_init=3.0,
        θ_bar=p.θ_bar, ξ_init=0.1,
        ρ_θ=p.ρ_θ, σ_θ=p.σ_θ,
        α_l=p.α_l, α_ξ=p.α_ξ, ξ_bar=p.ξ_bar,
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
