---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(tsyrennikov_2013)=
# Capital Flows Under Moral Hazard

## Overview

This lecture studies {cite:t}`Tsyrennikov2013`, which extends {cite:t}`Atkeson1991`
(see the companion lecture {doc}`atkeson_1991`) in two directions:

1. **Continuous investment** — the borrower chooses a continuous investment
   level rather than a binary one, and the paper proves that the
   **first-order approach** (FOA) to the incentive-compatibility constraint is
   valid. This brings the model much closer to empirically relevant calibrations.
2. **Calibration and quantitative analysis** — the model is calibrated to
   Argentina's business cycle data and compared against a limited-enforcement
   (Eaton–Gersowitz-style) model.

The central finding is that *moral hazard, not limited enforcement, drives the
key empirical regularities of emerging market economies*: high and volatile
interest rate spreads, limited consumption risk-sharing, and crisis-like
dynamics in which capital inflows suddenly stop.

The key mechanism is that moral hazard severely restricts *state contingency* in
repayment schedules.  

In the language of {cite}`Atkeson1991`, the optimal
contract is nearly *non-contingent* on output — a theoretical justification for
why simple debt contracts dominate in practice.

```{note}
This lecture uses the same notation as the {doc}`atkeson_1991` lecture,
writing $\beta$ for the borrower's discount factor (Tsyrennikov writes $\beta$
for the borrower and $\beta_c$ for the lender).
```

## The model

### Technology and preferences

The environment is a small open economy with an infinitely-lived borrower.

The borrower starts each period with net worth $n$ (output net of debt
repayment), borrows $b$ from a short-lived risk-neutral lender, invests $I$,
and consumes

$$
c \;=\; n + b - \theta I, \quad \theta > 0.
$$

Given investment $I$, next period's output $Y'$ is drawn from

$$
g(Y_j \mid I) \;=\; \bigl(1 - \lambda(I)\bigr)\,g_{0j}
    + \lambda(I)\,g_{1j}, \qquad j = 1, 2,
$$

where $\lambda : \mathbb{R}_+ \to [0,1]$ is strictly increasing and strictly
concave, so higher investment stochastically dominates lower investment.

Tsyrennikov restricts to two output states and sets

$$
\text{Pr}(Y_1 \mid I) = 1 - \lambda(I), \qquad
\text{Pr}(Y_2 \mid I) = \lambda(I), \qquad Y_1 < Y_2,
$$

so $g_{0,1}=1,\;g_{0,2}=0,\;g_{1,1}=0,\;g_{1,2}=1$ and
$\Delta g_j \equiv g_{1j} - g_{0j} = (-1, 1)$.

The functional form $\lambda(I) = \min(I^\nu, 1)$ with $\nu \in (0,1)$
is strictly concave and gives an interior optimum.

The borrower's preferences are CRRA:

$$
U^B = \mathbb{E}_0 \sum_{t=0}^\infty \beta^t \, u(c_t),
    \quad u(c) = \frac{c^{1-\gamma}}{1-\gamma}, \quad \gamma > 1.
$$

Lenders discount at rate $\beta_c \geq \beta$ (the international risk-free
rate) and have endowment $M$ each period, so $b \leq M$.

### Two frictions

**Moral hazard (MH)**: lenders observe output but not investment.

The incentive-compatibility (IC) constraint requires that the borrower finds the
contracted investment $I$ to be in their own best interest.

**Limited enforcement (LE)**: the borrower can default, suffering a one-time
output penalty: if default occurs when output is $Y_j$, the borrower retains
only $\delta Y_j$ (with $\delta \in (0,1)$) and then lives in autarky.

The participation constraint requires

$$
V(Y_j - d_j) \;\geq\; V_{\text{aut}}(\delta\,Y_j), \quad \forall j,
$$

where $V$ is the contract value function and $V_{\text{aut}}$ is the autarky
value function.

### The autarky value function

Without access to credit ($b = 0$), the borrower solves

$$
V_{\text{aut}}(n) = \max_{I \in [0,\,n/\theta]}
    \Bigl[u(n - \theta I) + \beta\,\bigl[(1-\lambda(I))\,V_{\text{aut}}(Y_1)
    + \lambda(I)\,V_{\text{aut}}(Y_2)\bigr]\Bigr].
$$

Note that the continuation values depend only on $Y_1$ and $Y_2$, not on $n$.

### The recursive contract

The state variable is net worth $n$.


The value function satisfies the Bellman
equation

$$
V(n) = \max_{b,\,d,\,I}
    \Bigl[u(n+b-\theta I) + \beta\,\sum_j g(Y_j\mid I)\,V(Y_j - d_j)\Bigr]
$$

subject to feasibility, lender participation ($b \leq \beta_c \sum_j
g_j(I)\,d_j$), incentive compatibility, and enforcement constraints.

## The first-order approach

A key contribution of {cite:t}`Tsyrennikov2013` is **Lemma 1**, which shows that
replacing the full IC constraint with the first-order condition

$$
-\theta\,u'(c) + \beta\,\lambda'(I)\,\sum_j \Delta g_j\,V(Y_j-d_j) \geq 0
$$

does *not* alter the solution.

The key step is showing that at any feasible
contract, $\sum_j \Delta g_j\,V(Y_j-d_j) \geq 0$, which ensures the
borrower's objective is strictly concave in $I$ and the FOC holds with
equality.

This result (analogous to {cite}`Rogerson1985`) validates the
relaxed formulation used in the numerical solution.

With the FOA, the optimality condition for investments is

$$
\theta\,u'(c) \;=\; \beta\,\lambda'(I)\,\bigl[V(n_2') - V(n_1')\bigr],
$$ (foa)

where $n_j' = Y_j - d_j$ is next period's net worth after state $j$.

A
higher spread $V(n_2') - V(n_1')$ — more reward in the high state —
supports a higher investment level.

## The Euler equation and implied interest rate

The Euler equation (Appendix A of {cite:t}`Tsyrennikov2013`) for the MH model is

$$
V'(n) \;=\; V'(n_j')\!\left[1 + \mu\,
    \frac{\lambda'(I)\,\Delta g_j}{g(Y_j\mid I)}\right] + \phi,
$$

where $\mu \geq 0$ is the multiplier on the FOA constraint and $\phi \geq 0$
on the lender endowment $b \leq M$.

Because $\Delta g_1 = -1 < 0$, the factor for the low state is less than one:
$V'(n_1') > V'(n)$.

By concavity of $V$, the borrower's net worth falls in
the low state.

This is the **immiseration** property: moral hazard forces
the borrower to bear more risk than would be optimal with full information
(cf.\ {cite}`ThomasWorrall1990`, {cite}`AtkesonLucas1992`).

The borrower faces an **implied interest rate**

$$
R(n) \;=\; \frac{u'(c(n))}{\beta\,\sum_j g(Y_j\mid I(n))\,u'(c(n_j'(n)))},
$$

where $c(n_j'(n))$ is next period's consumption if state $j$ is realised.


This rate is counter-cyclical: when $n$ is low, past incentive provision has
depressed the continuation values, raising the marginal utility spread and
increasing $R$.

## Computation

We now implement these ideas numerically using the parameterisation from
{cite:t}`Tsyrennikov2013`.

In addition to what's in Anaconda, this lecture will need the following library:

```{code-cell} ipython3
:tags: [hide-output]

!pip install jax
```

### Parameters

```{code-cell} ipython3
import numpy as np
from typing import NamedTuple
from scipy.interpolate import interp1d
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Model parameters
class Model(NamedTuple):
    β:   float    # borrower discount factor
    β_c: float    # lender (world) discount factor
    γ:   float    # CRRA coefficient
    θ:   float    # investment resource cost (θ in budget n+b = c+θI)
    ν:   float    # λ(I) = I^ν  (probability of high output)
    δ:   float    # borrower keeps fraction δ of output on default
    M:   float    # lender endowment
    Y1:  float    # low output state
    Y2:  float    # high output state


def create_model(β=0.980, β_c=0.990, γ=2.0, θ=0.105, ν=0.950,
                 δ=0.795, M=0.465, Y1=np.exp(-0.054), Y2=np.exp(+0.054)):
    """Build a model instance, validating the parameters."""
    if not 0 < β < 1:
        raise ValueError("β must lie in (0, 1)")
    if not 0 < β_c < 1:
        raise ValueError("β_c must lie in (0, 1)")
    if γ <= 0:
        raise ValueError("γ must be positive")
    if not 0 < ν < 1:
        raise ValueError("ν must lie in (0, 1)")
    if not 0 < δ < 1:
        raise ValueError("δ must lie in (0, 1)")
    if Y1 >= Y2:
        raise ValueError("require Y1 < Y2")
    return Model(β=β, β_c=β_c, γ=γ, θ=θ, ν=ν, δ=δ, M=M, Y1=Y1, Y2=Y2)


model = create_model()
β, β_c, γ, θ, ν, δ, M, Y1, Y2 = (model.β, model.β_c, model.γ, model.θ,
                                 model.ν, model.δ, model.M, model.Y1, model.Y2)
Y = np.array([Y1, Y2])


# Investment technology: λ(I) = I^ν (probability of high output)
def λ(I):
    return np.minimum(I**ν, 1.0)

def λ_jax(I):
    return jnp.minimum(I**ν, 1.0)

def dλ(I):
    """λ'(I) = ν * I^{ν−1}  (for I > 0)."""
    return ν * I**(ν - 1.0)

# Utility
def u(c):
    c = np.maximum(c, 1e-12)
    return c**(1.0 - γ) / (1.0 - γ)

def u_jax(c):
    c = jnp.maximum(c, 1e-12)
    return c**(1.0 - γ) / (1.0 - γ)

def u_prime(c):
    return np.maximum(c, 1e-12)**(-γ)

def u_prime_jax(c):
    return jnp.maximum(c, 1e-12)**(-γ)

# Net-worth grid
N_n    = 150
n_lo   = 0.08
n_hi   = 1.30
n_grid = np.linspace(n_lo, n_hi, N_n)
n_grid_j = jnp.asarray(n_grid)

# Search grids used by Bellman operators below.
N_I_search = 500
I_search_grid = np.linspace(0.0, 1.0, N_I_search)
I_search_grid_j = jnp.asarray(I_search_grid)

N_policy = 90
n1p_candidates = np.linspace(max(δ * Y1, n_lo),
                             min(Y1 * 1.1, n_hi - 1e-4),
                             N_policy)
n2p_candidates = np.linspace(max(δ * Y2, n_lo),
                             min(Y2 * 1.05, n_hi - 1e-4),
                             N_policy)
n1p_mesh, n2p_mesh = np.meshgrid(n1p_candidates, n2p_candidates,
                                 indexing='ij')
n1p_flat_j = jnp.asarray(n1p_mesh.ravel())
n2p_flat_j = jnp.asarray(n2p_mesh.ravel())

print(f"Output states:  Y1 = {Y1:.4f},  Y2 = {Y2:.4f}")
print(f"β = {β},  β_c = {β_c},  γ = {γ},  θ = {θ},  ν = {ν}")
```

### Autarky value function

```{code-cell} ipython3
@jax.jit
def autarky_step_jax(V, β_val):
    """One vectorised Bellman step for the autarky problem."""
    EV1 = jnp.interp(Y1, n_grid_j, V)
    EV2 = jnp.interp(Y2, n_grid_j, V)

    I = I_search_grid_j[None, :]
    c = n_grid_j[:, None] - θ * I
    l = λ_jax(I)
    obj = u_jax(c) + β_val * ((1.0 - l) * EV1 + l * EV2)
    obj = jnp.where(c > 1e-10, obj, -jnp.inf)

    idx = jnp.argmax(obj, axis=1)
    return jnp.max(obj, axis=1), I_search_grid_j[idx]


def autarky_policy(V_arr, β_val=None):
    """Return the autarky value update and investment policy on n_grid."""
    if β_val is None:
        β_val = β
    V_new, I_pol = autarky_step_jax(jnp.asarray(V_arr), β_val)
    return np.asarray(V_new), np.asarray(I_pol)


def autarky_bellman_at_n(n, Vf, β_val=None):
    """
    Solve the autarky Bellman at state n given current iterate Vf.
    Returns (V_new, I_opt).
    Uses the fact that continuation values only depend on Y1, Y2.
    """
    if β_val is None:
        β_val = β
    EV1 = float(Vf(Y1))
    EV2 = float(Vf(Y2))

    I_max = min(max(n / θ - 1e-8, 0.0), 1.0)
    I = I_search_grid[I_search_grid <= I_max]
    if I.size == 0:
        I = np.array([0.0])

    c = n - θ * I
    obj = u(c) + β_val * ((1.0 - λ(I)) * EV1 + λ(I) * EV2)
    idx = np.argmax(obj)
    return float(obj[idx]), float(I[idx])


def autarky_vfi(β_val=None, tol=1e-8, max_iter=3000):
    if β_val is None:
        β_val = β

    V = jnp.zeros(N_n)
    for it in range(max_iter):
        V_new, _ = autarky_step_jax(V, β_val)
        diff     = float(jnp.max(jnp.abs(V_new - V)))
        V        = V_new
        if diff < tol:
            print(f"Autarky VFI converged in {it+1} iterations (diff = {diff:.2e})")
            break

    return np.asarray(V)


V_aut = autarky_vfi()
```

### Moral hazard model

For each state $n$, we optimise over continuation states
$(n_1', n_2')$ where $n_j' = Y_j - d_j$.  For every candidate
$(n_1', n_2')$:

1. Compute $\Delta V = V(n_2') - V(n_1')$.
2. With lender participation binding, the loan is
   $b^* = \beta_c\bigl[(1-\lambda(I))(Y_1-n_1') + \lambda(I)(Y_2-n_2')\bigr]$.
3. Substitute into the FOA equation and solve for $I^*$:

$$
\theta\,\bigl[A + \lambda(I^*)\,\Delta B - \theta I^*\bigr]^{-\gamma}
    \;=\; \beta\,\lambda'(I^*)\,\Delta V,
$$

where $A \equiv n + \beta_c (Y_1-n_1')$ and
$\Delta B \equiv \beta_c\bigl[(Y_2-n_2') - (Y_1-n_1')\bigr]$.

This reduces the optimisation to two dimensions.

```{code-cell} ipython3
@jax.jit
def mh_bellman_step_jax(V, V_aut_arr, β_val, β_c_val):
    """
    One vectorised Bellman step for the moral-hazard model.

    For each candidate pair (n_1', n_2'), the FOA for I is solved by a
    compiled bisection.  The Bellman maximisation is then a batch grid search
    over continuation states.
    """
    V1 = jnp.interp(n1p_flat_j, n_grid_j, V)
    V2 = jnp.interp(n2p_flat_j, n_grid_j, V)
    ΔV = V2 - V1

    A = n_grid_j[:, None] + β_c_val * (Y1 - n1p_flat_j)[None, :]
    ΔB = β_c_val * ((Y2 - n2p_flat_j) - (Y1 - n1p_flat_j))

    def c_of_I(I):
        return A + (I**ν) * ΔB[None, :] - θ * I

    I_hi = jnp.minimum(A / θ * 0.999, 1.0 - 1e-6)
    I_hi = jnp.maximum(I_hi, 1e-6)

    def shrink_hi(_, I_hi_val):
        return jnp.where(c_of_I(I_hi_val) < 1e-8, 0.9 * I_hi_val, I_hi_val)

    I_hi = jax.lax.fori_loop(0, 40, shrink_hi, I_hi)
    I_lo = jnp.full_like(I_hi, 1e-7)

    def foa(I):
        return (θ * u_prime_jax(c_of_I(I))
                - β_val * ν * jnp.maximum(I, 1e-12)**(ν - 1.0)
                * ΔV[None, :])

    foa_lo = foa(I_lo)
    foa_hi = foa(I_hi)
    valid = ((ΔV[None, :] > 1e-10) & (I_hi > 1e-6)
             & (foa_lo < 0.0) & (foa_hi > 0.0))

    def bisect_body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        f_mid = foa(mid)
        hi = jnp.where(f_mid > 0.0, mid, hi)
        lo = jnp.where(f_mid > 0.0, lo, mid)
        return lo, hi

    I_lo, I_hi = jax.lax.fori_loop(0, 45, bisect_body, (I_lo, I_hi))
    I_star = 0.5 * (I_lo + I_hi)
    c = c_of_I(I_star)
    l = λ_jax(I_star)
    b = β_c_val * ((1 - l) * (Y1 - n1p_flat_j)[None, :]
                   + l * (Y2 - n2p_flat_j)[None, :])
    EV = (1 - l) * V1[None, :] + l * V2[None, :]
    obj = u_jax(c) + β_val * EV

    feasible = valid & (c > 1e-10) & (b <= M + 1e-6)
    obj = jnp.where(feasible, obj, -jnp.inf)

    idx = jnp.argmax(obj, axis=1)
    best_val = jnp.max(obj, axis=1)
    has_feasible = jnp.isfinite(best_val)
    use_fallback = (~has_feasible) | (best_val <= V_aut_arr)

    pol_n1p = jnp.where(use_fallback, Y1, n1p_flat_j[idx])
    pol_n2p = jnp.where(use_fallback, Y2, n2p_flat_j[idx])
    pol_I = jnp.where(use_fallback, 0.0,
                      jnp.take_along_axis(I_star, idx[:, None], axis=1)[:, 0])
    pol_b = jnp.where(use_fallback, 0.0,
                      jnp.take_along_axis(b, idx[:, None], axis=1)[:, 0])
    V_new = jnp.where(use_fallback, V_aut_arr, best_val)

    return V_new, pol_n1p, pol_n2p, pol_I, pol_b


def mh_vfi(V_aut, β_val=None, β_c_val=None, tol=1e-3, max_iter=60,
           relaxation=0.5):
    """Value function iteration for the moral hazard model."""
    if β_val is None:
        β_val = β
    if β_c_val is None:
        β_c_val = β_c

    V = V_aut.copy()
    for it in range(max_iter):
        V_raw, pol_n1p, pol_n2p, pol_I, pol_b = mh_bellman_step_jax(
            jnp.asarray(V), jnp.asarray(V_aut), β_val, β_c_val)
        V_raw = np.asarray(V_raw)
        V_new = (1 - relaxation) * V + relaxation * V_raw
        diff = np.max(np.abs(V_new - V))
        V    = V_new
        print(f"  iter {it+1:3d},  max|ΔV| = {diff:.5f}")
        if diff < tol:
            print(f"MH VFI converged in {it+1} iterations.")
            break

    _, pol_n1p, pol_n2p, pol_I, pol_b = mh_bellman_step_jax(
        jnp.asarray(V), jnp.asarray(V_aut), β_val, β_c_val)

    return (V, np.asarray(pol_n1p), np.asarray(pol_n2p),
            np.asarray(pol_I), np.asarray(pol_b))


print("Running moral hazard VFI…")
V_mh, pol_n1p, pol_n2p, pol_I, pol_b = mh_vfi(V_aut)
```

### Value functions and insurance

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: value functions and risk-sharing index
    name: fig-tsy-value-rsi
---
fig, axes = plt.subplots(1, 2)

# Left: value functions
axes[0].plot(n_grid, V_aut, lw=2, label='Autarky')
axes[0].plot(n_grid, V_mh,  lw=2, ls='--', label='Moral hazard')
axes[0].set_xlabel('net worth $n$')
axes[0].set_ylabel('value')
axes[0].legend()

# Right: Risk-sharing index  RSI = (d_2 - d_1) / (Y_2 - Y_1)
# d_j = Y_j - n_j',  so RSI = (n_1' - n_2') / (Y_2 - Y_1) + 1
# ... actually RSI = (d_2 - d_1)/(Y_2-Y_1) = ((Y2-n2p)-(Y1-n1p))/(Y2-Y1)
d1_mh  = Y1 - pol_n1p
d2_mh  = Y2 - pol_n2p
RSI_mh = (d2_mh - d1_mh) / (Y2 - Y1)
active_mh = λ(pol_I) > 0.01
RSI_mh_plot = np.where(active_mh, RSI_mh, np.nan)

axes[1].plot(n_grid, RSI_mh_plot, lw=2, color='C1')
axes[1].axhline(1.0, ls=':', color='k',  lw=1, label='Full insurance (RSI=1)')
axes[1].axhline(0.0, ls='--', color='k', lw=1, label='Non-contingent debt (RSI=0)')
axes[1].set_xlabel('net worth $n$')
axes[1].set_ylabel('risk-sharing index')
axes[1].set_ylim(-0.1, 1.2)
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"\nMean RSI (active-investment states): {np.nanmean(RSI_mh_plot):.4f}")
print("→ Repayment is nearly state non-contingent (RSI ≈ 0)")
print("→ Moral hazard justifies why simple non-contingent debt is optimal")
```

A key finding of {cite:t}`Tsyrennikov2013` emerges immediately: the risk-sharing
index is close to *zero* on the active-investment region.

Moral hazard requires
spreading continuation values to incentivise investment, but this is achieved
by differentiating *net worth* $n_j'$, not repayment $d_j = Y_j - n_j'$.

The near-equality $d_1 \approx d_2$ means repayment is essentially
*non-contingent on output* — the model rationalises why emerging market
borrowers use plain debt instruments rather than GDP-linked securities.

### Optimal investment

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: optimal investment and continuation net worth
    name: fig-tsy-investment
---
fig, axes = plt.subplots(1, 2)

# Compute autarky optimal investment
_, I_aut = autarky_policy(V_aut)

axes[0].plot(n_grid, λ(I_aut), lw=2,        label='Autarky')
axes[0].plot(n_grid, λ(pol_I), lw=2, ls='--', label='Moral hazard')
axes[0].set_xlabel('net worth $n$')
axes[0].set_ylabel(r'$\lambda(I) = \Pr(Y_2 \mid I)$')
axes[0].legend()

axes[1].plot(n_grid, pol_n1p, lw=2,
             label=r"$n_1' = Y_1 - d_1$  (after low output)")
axes[1].plot(n_grid, pol_n2p, lw=2, ls='--',
             label=r"$n_2' = Y_2 - d_2$  (after high output)")
axes[1].plot(n_grid, n_grid,  lw=1, ls=':',  color='k', label='45° line')
axes[1].set_xlabel('net worth $n$')
axes[1].set_ylabel("continuation net worth $n_j'$")
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.show()
```

Investment under moral hazard is *lower* than in autarky at high net worth
levels and more sensitive to $n$ at low levels.

After a low output
realisation, net worth drops sharply ($n_1' \ll n$), depressing future
investment and perpetuating the crisis — the model's **internal propagation
mechanism**.

### Implied interest rate

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: implied interest rate and spread
    name: fig-tsy-interest-rate
---
# Compute implied interest rate R(n)
# R(n) = u'(c(n)) / [β * Σ_j g_j(I(n)) * u'(c'(n_j'(n)))]
# where c'(n_j') = n_j' + b*(n_j') - θ I*(n_j')  (next period's consumption)

pol_b_fn  = interp1d(n_grid, pol_b, fill_value='extrapolate', bounds_error=False)
pol_I_fn  = interp1d(n_grid, pol_I, fill_value='extrapolate', bounds_error=False)
pol_n1p_fn = interp1d(n_grid, pol_n1p, fill_value='extrapolate', bounds_error=False)
pol_n2p_fn = interp1d(n_grid, pol_n2p, fill_value='extrapolate', bounds_error=False)

def next_period_c(np_val):
    """Consumption at the start of next period given continuation n'."""
    b_next  = float(pol_b_fn(np_val))
    I_next  = float(pol_I_fn(np_val))
    return np_val + b_next - θ * I_next

R_n = np.empty(N_n)
for k, n in enumerate(n_grid):
    b   = pol_b[k]
    I   = pol_I[k]
    c   = n + b - θ * I
    l   = λ(I)
    n1p = pol_n1p[k]
    n2p = pol_n2p[k]
    c1p = next_period_c(n1p)
    c2p = next_period_c(n2p)
    denom = β * ((1-l)*u_prime(c1p) + l*u_prime(c2p))
    R_n[k] = u_prime(c) / denom if denom > 1e-10 else np.nan

# Annualised spread over world rate
R_world = 1.0 / β_c                 # gross world rate
spread_ann = (R_n**4 - R_world**4)  # approximate annualised spread

fig, axes = plt.subplots(1, 2)

axes[0].plot(n_grid, R_n, lw=2)
axes[0].axhline(R_world, ls='--', color='k', lw=1,
                label=f'World rate $1/\\beta_c = {R_world:.3f}$')
axes[0].set_xlabel('net worth $n$')
axes[0].set_ylabel('implied gross interest rate $R(n)$')
axes[0].legend()

axes[1].plot(n_grid, np.clip(spread_ann * 100, -1, 50), lw=2)
axes[1].axhline(0, ls='--', color='k', lw=0.8)
axes[1].set_xlabel('net worth $n$')
axes[1].set_ylabel('annualised spread over world rate (%)')

plt.tight_layout()
plt.show()
```

The interest rate spread rises sharply at low net worth levels, consistent with
the Argentine data.

The mechanism is the **MH Euler equation**: when $n$ is
low, the borrower's continuation value is depressed and the spread in marginal
utilities across future states increases $R(n)$.

### Crisis dynamics

{cite:t}`Tsyrennikov2013` shows that a string of low output realisations
generates gradual debt accumulation followed by a sudden stop in which capital
inflows cease and interest rates spike — a pattern consistent with the
Argentina 2001 experience.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: simulated crisis dynamics
    name: fig-tsy-crisis
---
def simulate_crisis(T_crisis=8):
    """
    Simulate crisis path: T_crisis periods of low output (Y_1) starting
    from zero debt (n_0 = Y2, high initial net worth).
    """
    n = Y2   # start with high net worth
    records = {'n': [n], 'debt_over_Y': [], 'R': [], 'ca_over_Y': [],
               'λ': []}

    for t in range(T_crisis):
        b   = float(pol_b_fn(n))
        I   = float(pol_I_fn(n))
        n1p = float(pol_n1p_fn(n))
        n2p = float(pol_n2p_fn(n))

        c   = n + b - θ * I
        l   = λ(I)
        c1p = next_period_c(n1p)
        c2p = next_period_c(n2p)
        denom = β * ((1-l)*u_prime(c1p) + l*u_prime(c2p))
        R = u_prime(c) / denom if denom > 1e-10 else np.nan

        # Debt = promised repayment − principal rolled over
        # Approximate debt/output = b / Y1 (loan at current period)
        debt_Y = b / Y1

        # Current account = d_t − b_t  (repayment received − new loan given)
        # At t=0, d_0 = 0 (no old contract); approximate d_t = Y1 - n
        d_approx = Y1 - n
        ca = d_approx - b

        records['debt_over_Y'].append(debt_Y)
        records['R'].append(R)
        records['ca_over_Y'].append(ca / Y1)
        records['λ'].append(l)

        n = n1p   # low output path

    records['n'] += [float(pol_n1p_fn(records['n'][-1]))]
    return records


crisis = simulate_crisis(T_crisis=8)
t_ax   = np.arange(len(crisis['R']))

fig, axes = plt.subplots(2, 2, sharex=True)

axes[0,0].plot(t_ax, crisis['debt_over_Y'], 'o-', lw=2)
axes[0,0].set_ylabel('debt / output')

axes[0,1].plot(t_ax, np.array(crisis['R'])**4 * 100, 's-', lw=2, color='C1')
axes[0,1].axhline((1/β_c)**4 * 100, ls='--', color='k', lw=0.8,
                  label='World rate')
axes[0,1].set_ylabel('annualised gross rate (%)')
axes[0,1].legend(fontsize=9)

axes[1,0].plot(t_ax, crisis['ca_over_Y'], '^-', lw=2, color='C2')
axes[1,0].axhline(0, ls='--', color='k', lw=0.8)
axes[1,0].set_xlabel('quarter')
axes[1,0].set_ylabel('current account / output')

axes[1,1].plot(t_ax, crisis['λ'], 'D-', lw=2, color='C3')
axes[1,1].set_xlabel('quarter')
axes[1,1].set_ylabel(r'$\lambda(I) = \Pr(Y_2 \mid I)$')

plt.tight_layout()
plt.show()
```

The simulation reproduces the stylised crisis pattern of {cite:t}`Tsyrennikov2013`,
Fig. 4:

- **Panel A**: Debt steadily accumulates as the borrower is pushed toward the
  borrowing limit by repeated low output.
- **Panel B**: Interest rates remain near the world rate initially but spike
  sharply once the borrower approaches the borrowing limit — the
  **late-warning** property.
- **Panel C**: The current account first worsens gradually (capital inflows
  shrink) and then abruptly turns around as the borrowing limit is reached.
- **Panel D**: Investment collapses as net worth falls, further reducing the
  probability of high future output — the **internal propagation mechanism**.

### MH versus limited enforcement

A crucial result of {cite:t}`Tsyrennikov2013` is that *limited enforcement adds
little* to the model's performance relative to moral hazard alone.

Under LE (no moral hazard), optimal repayments are *highly state
contingent* (RSI ≈ 0.8), providing near-full insurance.

The borrower's net
worth drifts *upward* under LE (unlike MH where it drifts downward), so
interest rate spreads are transitory rather than persistent.

The following code illustrates the key theoretical distinction by computing
the Euler equation implications under each friction separately.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: expected continuation net worth
    name: fig-tsy-mh-le
---
# Illustrate the Euler equation implications theoretically
fig, ax = plt.subplots()

# Under MH (μ > 0, γ_j = 0):
#   V'(n) = V'(n_j') [1 + μ λ'(I) Δg_j / g_j(I)]  + φ
# → low-state factor < 1: V'(n1') > V'(n) → n1' < n (net worth falls)
# → high-state factor > 1: V'(n2') < V'(n) → n2' > n (net worth rises)
#
# Under LE (μ = 0, γ_j > 0):
#   V'(n) = V'(n_j') [1 + γ_j] + φ ≥ V'(n_j')
# → V'(n_j') ≤ V'(n) → n_j' ≥ n  (net worth drifts upward)

# Stylised illustration using the computed MH policy
Vf_mh = interp1d(n_grid, V_mh, fill_value='extrapolate', bounds_error=False)

expected_np_mh = (1 - λ(pol_I)) * pol_n1p + λ(pol_I) * pol_n2p

ax.plot(n_grid, expected_np_mh, lw=2,        label='MH: E[n\']')
ax.plot(n_grid, n_grid,         lw=1, ls=':', color='k', label='45° line')

# Under LE (approximate): net worth is a supermartingale, E[n'] ≥ n * β/β_c
expected_np_le = n_grid * (β / β_c)
ax.plot(n_grid, expected_np_le, lw=2, ls='--', color='C2',
        label=r'LE: E[n\'] $\approx$ $(\beta/\beta_c)\,n$ (drifts up)')

ax.set_xlabel('current net worth $n$')
ax.set_ylabel("expected continuation net worth $E[n']$")
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

Under moral hazard, $\mathbb{E}[n']$ is pulled below $n$ in the
high-net-worth active contracting region: net worth drifts down and the
borrower spends substantial time near the borrowing limit, generating
persistent interest rate spreads.

Under limited enforcement,
$\mathbb{E}[n'] \geq (\beta/\beta_c)\,n$: net worth drifts toward a stationary
level and the borrower eventually escapes financial stress.

## Empirical test

{cite:t}`Tsyrennikov2013` proposes a test to distinguish moral hazard from
limited enforcement.

After a low past output realisation ($y_{t-1} = Y_1$),
the MH contract lowers net worth sharply, reducing future consumption
smoothing.  

This prediction is:

$$
\text{MH economy}: \quad
    \rho(c_t, y_t \mid y_{t-1} = Y_1) \;>\; \rho(c_t, y_t \mid y_{t-1} = Y_2),
$$

while the LE economy gives the opposite ordering (insurance is better after
low realisations).

Using Argentine quarterly data (1993–2005), the observed
correlations are 0.98 (after low output) vs. 0.91 (after high output) —
*consistent with moral hazard*.

## Exercises

```{exercise-start}
:label: tsyrennikov_2013_ex1
```

**Effect of default penalty.**  The parameter $\delta \in (0,1)$ controls
the severity of the output loss upon default.

1. Compute $V_{\text{aut}}$ for $\delta \in \{0.5,\, 0.795,\, 0.95\}$.
2. For each $\delta$, evaluate the enforcement threshold
   $V_{\text{aut}}(\delta Y_1)$ and $V_{\text{aut}}(\delta Y_2)$.
3. Discuss: how does a harsher default penalty affect the tightness of the
   enforcement constraint and (via the Euler equation) the interest rate
   spread?  At $\delta = 1$ the LE constraint becomes
   $V(n_j') \geq V_{\text{aut}}(Y_j)$; at $\delta \to 0$ it is vacuous.
```{exercise-end}
```

```{solution-start} tsyrennikov_2013_ex1
:class: dropdown
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: autarky value and enforcement thresholds
    name: fig-tsy-default-penalty
---
fig, ax = plt.subplots()

Vaut_f_global = interp1d(n_grid, V_aut, fill_value='extrapolate', bounds_error=False)

for δ_val, ls, color in [(0.50, ':', 'C0'), (0.795, '--', 'C1'), (0.95, '-', 'C2')]:
    thresh1 = float(Vaut_f_global(δ_val * Y1))
    thresh2 = float(Vaut_f_global(δ_val * Y2))
    # Net worth lower bound from enforcement: n_j' >= V^{-1}(thresh_j)
    # For illustration plot the thresholds
    print(f"δ={δ_val:.3f}: V_aut(δ·Y1)={thresh1:.3f},  V_aut(δ·Y2)={thresh2:.3f}")

ax.plot(n_grid, V_aut, lw=2)
for δ_val, label in [(0.50, 'δ=0.50'), (0.795, 'δ=0.795'), (0.95, 'δ=0.95')]:
    t1 = float(Vaut_f_global(δ_val * Y1))
    t2 = float(Vaut_f_global(δ_val * Y2))
    ax.axhline(t1, ls=':', lw=1.5, label=f'{label}: V_aut(δ·Y1)')

ax.set_xlabel('net worth $n$');  ax.set_ylabel('$V_{\\rm aut}(n)$')
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
```

A harsher default penalty (larger $\delta$) raises the enforcement thresholds,
tightening the participation constraints and reducing the scope for
state-contingent repayment.

Paradoxically, this may *reduce* the interest
rate spread by forcing the lender to offer more consumption insurance to keep
the borrower from defaulting.

At $\delta \to 0$ the enforcement constraint is
vacuous and the model collapses to pure moral hazard.

```{solution-end}
```

```{exercise-start}
:label: tsyrennikov_2013_ex2
```

**Discounting wedge and impatience.**

1. Re-solve the MH model for $\beta = \beta_c = 0.990$ (equal discounting —
   no impatience wedge) and for $\beta = 0.950$ (larger wedge).
2. For each case, plot the expected continuation net worth
   $\mathbb{E}[n'] = (1-\lambda(I^*))n_1' + \lambda(I^*)n_2'$ against $n$.
3. Discuss: how does the discount wedge $\beta_c - \beta$ interact with moral
   hazard in determining the stationary distribution of net worth?

*Hint*: When $\beta = \beta_c$ the only force pushing net worth down is moral
hazard (immiseration).  When $\beta < \beta_c$ there is an additional
front-loading incentive that the lender can exploit.
```{exercise-end}
```

```{solution-start} tsyrennikov_2013_ex2
:class: dropdown
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: continuation net worth across discount factors
    name: fig-tsy-discount-wedge
---
fig, ax = plt.subplots()

for β_val, ls, color in [(0.990, '-', 'C0'), (0.980, '--', 'C1'), (0.950, ':', 'C2')]:
    V_a_tmp = autarky_vfi(β_val=β_val)
    V_mh_tmp, pol_n1p_tmp, pol_n2p_tmp, pol_I_tmp, _ = mh_vfi(
        V_a_tmp, β_val=β_val, max_iter=80)

    E_np = ((1 - λ(pol_I_tmp)) * pol_n1p_tmp
            + λ(pol_I_tmp) * pol_n2p_tmp)
    ax.plot(n_grid, E_np, ls=ls, color=color,
            label=fr'$\beta={β_val}$')

ax.plot(n_grid, n_grid, lw=1, ls=':', color='k', label='45° line')
ax.set_xlabel('net worth $n$')
ax.set_ylabel("$E[n']$")
ax.legend()
plt.tight_layout()
plt.show()
```

The larger the discount wedge $\beta_c - \beta$, the faster net worth drifts
toward the borrowing limit.

When $\beta = \beta_c$ moral hazard alone drives
immiseration, while impatience accelerates it further.

A small wedge
(as calibrated by Tsyrennikov) is significant: it is *equivalent to
increasing the borrower's discount rate by 2% per annum* (even though
the assumed difference in quarterly rates is only 0.010).

```{solution-end}
```

```{exercise-start}
:label: tsyrennikov_2013_ex3
```

**Non-contingency of optimal debt.**

The *risk-sharing index* $\text{RSI}(n) = (d_2(n) - d_1(n)) / (Y_2 - Y_1)$
measures how state-contingent the repayment schedule is.

RSI = 1 is full
insurance; RSI = 0 is non-contingent debt.

1. Compute RSI for the MH model you have already solved.
2. Now set $\beta = \beta_c$ (equal discounting) and recompute. Does
   removing the impatience wedge change the near-zero RSI result?
3. Explain *theoretically* why moral hazard drives RSI toward zero.

*Hint*: From the Euler equation, the spread in marginal utilities
$u'(c_1') / u'(c_2')$ depends on the IC multiplier $\mu$.  A larger $\mu$
spreads continuation values but *not necessarily* repayments.
```{exercise-end}
```

```{solution-start} tsyrennikov_2013_ex3
:class: dropdown
```

```{code-cell} ipython3
# RSI for the baseline MH model
d1_mh = Y1 - pol_n1p
d2_mh = Y2 - pol_n2p
RSI   = (d2_mh - d1_mh) / (Y2 - Y1)
active_RSI = RSI[λ(pol_I) > 0.01]

print(f"Baseline MH:  mean RSI = {np.mean(active_RSI):.4f},  "
      f"max RSI = {np.max(active_RSI):.4f}")
print()
print("Theoretical explanation:")
print(" Under moral hazard the planner must spread V(n2') - V(n1') to")
print(" incentivise investment via the FOA.  This is achieved by setting")
print(" n2' > n1', i.e. spreading *net worth*, not repayments.")
print(" Since d_j = Y_j - n_j', if n2' - n1' = Y2 - Y1 then d1 = d2 (RSI=0).")
print(" Moral hazard forces this near-equality, making debt non-contingent.")
```

```{solution-end}
```
