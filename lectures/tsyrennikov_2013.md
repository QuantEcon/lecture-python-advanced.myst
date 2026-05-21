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
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Capital Flows Under Moral Hazard

## Overview

This lecture studies {cite}`Tsyrennikov2013`, which extends {cite}`Atkeson1991`
(see the companion lecture {ref}`atkeson_1991`) in two directions:

1. **Continuous investment** — the borrower chooses a continuous investment
   level rather than a binary one, and the paper proves that the
   **first-order approach** (FOA) to the incentive-compatibility constraint is
   valid.  This brings the model much closer to empirically relevant calibrations.
2. **Calibration and quantitative analysis** — the model is calibrated to
   Argentina's business cycle data and compared against a limited-enforcement
   (Eaton–Gersowitz-style) model.

The central finding is that **moral hazard, not limited enforcement, drives the
key empirical regularities of emerging market economies**: high and volatile
interest rate spreads, limited consumption risk-sharing, and crisis-like
dynamics in which capital inflows suddenly stop.

The key mechanism: moral hazard severely restricts *state contingency* in
repayment schedules.  In the language of {cite}`Atkeson1991`, the optimal
contract is nearly *non-contingent* on output — a theoretical justification for
why simple debt contracts dominate in practice.

```{note}
This lecture uses the same notation as the {ref}`atkeson_1991` lecture,
writing $\beta$ for the borrower's discount factor (Tsyrennikov writes $\beta$
for the borrower and $\beta_c$ for the lender).
```

## The Model

### Technology and Preferences

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

### Two Frictions

**Moral hazard (MH)**: lenders observe output but not investment.  The
incentive-compatibility (IC) constraint requires that the borrower finds the
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

### The Autarky Value Function

Without access to credit ($b = 0$), the borrower solves

$$
V_{\text{aut}}(n) = \max_{I \in [0,\,n/\theta]}
    \Bigl[u(n - \theta I) + \beta\,\bigl[(1-\lambda(I))\,V_{\text{aut}}(Y_1)
    + \lambda(I)\,V_{\text{aut}}(Y_2)\bigr]\Bigr].
$$

Note that the continuation values depend only on $Y_1$ and $Y_2$, not on $n$.

### The Recursive Contract

The state variable is net worth $n$.  The value function satisfies the Bellman
equation

$$
V(n) = \max_{b,\,d,\,I}
    \Bigl[u(n+b-\theta I) + \beta\,\sum_j g(Y_j\mid I)\,V(Y_j - d_j)\Bigr]
$$

subject to feasibility, lender participation ($b \leq \beta_c \sum_j
g_j(I)\,d_j$), incentive compatibility, and enforcement constraints.

## The First-Order Approach

A key contribution of {cite}`Tsyrennikov2013` is **Lemma 1**, which shows that
replacing the full IC constraint with the first-order condition

$$
-\theta\,u'(c) + \beta\,\lambda'(I)\,\sum_j \Delta g_j\,V(Y_j-d_j) \geq 0
$$

does **not** alter the solution.  The key step is showing that at any feasible
contract, $\sum_j \Delta g_j\,V(Y_j-d_j) \geq 0$, which ensures the
borrower's objective is strictly concave in $I$ and the FOC holds with
equality.  This result (analogous to {cite}`Rogerson1985`) validates the
relaxed formulation used in the numerical solution.

With the FOA, the optimality condition for investments is

$$
\theta\,u'(c) \;=\; \beta\,\lambda'(I)\,\bigl[V(n_2') - V(n_1')\bigr],
\tag{FOA}
$$

where $n_j' = Y_j - d_j$ is next period's net worth after state $j$.  A
higher spread $V(n_2') - V(n_1')$ — more reward in the high state —
supports a higher investment level.

## The Euler Equation and Implied Interest Rate

The Euler equation (Appendix A of {cite}`Tsyrennikov2013`) for the MH model is

$$
V'(n) \;=\; V'(n_j')\!\left[1 + \mu\,
    \frac{\lambda'(I)\,\Delta g_j}{g(Y_j\mid I)}\right] + \phi,
$$

where $\mu \geq 0$ is the multiplier on the FOA constraint and $\phi \geq 0$
on the lender endowment $b \leq M$.

Because $\Delta g_1 = -1 < 0$, the factor for the low state is less than one:
$V'(n_1') > V'(n)$.  By concavity of $V$, the borrower's net worth falls in
the low state.  This is the **immiseration** property: moral hazard forces
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
{cite}`Tsyrennikov2013`.

### Parameters

```{code-cell} ipython3
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, minimize, brentq
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12, 'figure.dpi': 100})

# ── Preferences ──────────────────────────────────────────────────────────────
β    = 0.980          # borrower discount factor
β_c  = 0.990          # lender (world) discount factor
γ    = 2.0            # CRRA coefficient
θ    = 0.105          # investment resource cost (θ in budget n+b = c+θI)

# ── Investment technology ─────────────────────────────────────────────────────
# λ(I) = I^ν  (probability of high output)
ν    = 0.950

def lam(I):
    return np.minimum(I**ν, 1.0)

def dlam(I):
    """λ'(I) = ν * I^{ν−1}  (for I > 0)."""
    return ν * I**(ν - 1.0)

# ── Default penalty ───────────────────────────────────────────────────────────
δ    = 0.795          # borrower keeps fraction δ of output on default

# ── Lender endowment ─────────────────────────────────────────────────────────
M    = 0.465

# ── Output states:  ln(Y_j) = ±0.054, normalised so parameter θ gives E[Y]=1 ─
Y1   = np.exp(-0.054)   # ≈ 0.9474
Y2   = np.exp(+0.054)   # ≈ 1.0555
Y    = np.array([Y1, Y2])

# ── Utility ───────────────────────────────────────────────────────────────────
def u(c):
    c = np.maximum(c, 1e-12)
    return c**(1.0 - γ) / (1.0 - γ)

def u_prime(c):
    return np.maximum(c, 1e-12)**(-γ)

# ── Net-worth grid ────────────────────────────────────────────────────────────
N_n    = 150
n_lo   = 0.08
n_hi   = 1.30
n_grid = np.linspace(n_lo, n_hi, N_n)

print(f"Output states:  Y1 = {Y1:.4f},  Y2 = {Y2:.4f}")
print(f"β = {β},  β_c = {β_c},  γ = {γ},  θ = {θ},  ν = {ν}")
```

### Autarky Value Function

```{code-cell} ipython3
def autarky_bellman_at_n(n, Vf):
    """
    Solve the autarky Bellman at state n given current iterate Vf.
    Returns (V_new, I_opt).
    Uses the fact that continuation values only depend on Y1, Y2.
    """
    EV1 = float(Vf(Y1))
    EV2 = float(Vf(Y2))

    def neg_obj(I):
        c = n - θ * I
        if c < 1e-10:
            return 1e10
        l  = lam(I)
        return -(u(c) + β * ((1.0 - l) * EV1 + l * EV2))

    I_max = n / θ - 1e-8
    if I_max <= 0:
        return u(n) + β * ((1 - lam(0)) * EV1 + lam(0) * EV2), 0.0

    res = minimize_scalar(neg_obj, bounds=(1e-8, I_max), method='bounded',
                          options={'xatol': 1e-9})
    return -res.fun, res.x


def autarky_vfi(tol=1e-8, max_iter=3000):
    V = np.zeros(N_n)

    for it in range(max_iter):
        Vf    = interp1d(n_grid, V, fill_value='extrapolate', bounds_error=False)
        V_new = np.array([autarky_bellman_at_n(n, Vf)[0] for n in n_grid])
        diff  = np.max(np.abs(V_new - V))
        V     = V_new
        if diff < tol:
            print(f"Autarky VFI converged in {it+1} iterations (diff = {diff:.2e})")
            break

    return V


V_aut = autarky_vfi()
```

### Moral Hazard Model

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
def solve_I_star(n, n1p, n2p, Vf):
    """
    Solve the FOA equation for I*, given (n, n1', n2') and current V.
    Returns (I_star, c_star, b_star, lam_star) or None if infeasible.
    """
    dV = float(Vf(n2p)) - float(Vf(n1p))
    if dV <= 1e-10:          # no interior solution (V flat or decreasing)
        return None

    A  = n + β_c * (Y1 - n1p)
    ΔB = β_c * ((Y2 - n2p) - (Y1 - n1p))

    def c_star_of_I(I):
        return A + I**ν * ΔB - θ * I

    # Upper bound on I: c* > 0
    # Approximate: I_max ≈ A/θ  (rough; refine by bisection)
    I_hi = min(A / θ * 0.999, 1.0 - 1e-6)
    while I_hi > 1e-6 and c_star_of_I(I_hi) < 1e-8:
        I_hi *= 0.9
    if I_hi < 1e-6:
        return None

    I_lo = 1e-7

    def foa(I):
        c = c_star_of_I(I)
        if c < 1e-10:
            return 1e10
        return θ * u_prime(c) - β * dlam(I) * dV

    # foa(I_lo) < 0  (RHS → +∞),  foa(I_hi) should be > 0 eventually
    if foa(I_lo) >= 0 or foa(I_hi) <= 0:
        return None

    try:
        I_star = brentq(foa, I_lo, I_hi, xtol=1e-10)
    except ValueError:
        return None

    c  = c_star_of_I(I_star)
    l  = lam(I_star)
    b  = β_c * ((1-l)*(Y1-n1p) + l*(Y2-n2p))
    return I_star, c, b, l


def mh_bellman_at_n(n, Vf, Vaut_f, thresh1, thresh2):
    """
    Solve the MH Bellman equation at state n.
    thresh1, thresh2 are enforcement lower bounds on n1', n2'.
    Returns (V_new, n1p_opt, n2p_opt, I_opt, b_opt).
    """
    def neg_obj(x):
        n1p, n2p = x
        sol = solve_I_star(n, n1p, n2p, Vf)
        if sol is None:
            return 1e10
        I_star, c, b, l = sol
        if c < 1e-10 or b > M + 1e-6:
            return 1e10
        EV = (1-l)*float(Vf(n1p)) + l*float(Vf(n2p))
        return -(u(c) + β * EV)

    # Enforcement lower bounds on continuation states
    lo = np.array([max(thresh1, n_lo), max(thresh2, n_lo)])
    hi = np.array([min(Y1 * 1.1,  n_hi - 1e-4),
                   min(Y2 * 1.05, n_hi - 1e-4)])

    # Starting points
    x_inits = [
        lo,
        np.array([min(Y1 * 0.95, hi[0]),  min(Y2 * 0.95, hi[1])]),
        np.array([lo[0],                   min(Y2 * 0.85, hi[1])]),
    ]

    best_val = float(Vaut_f(n))   # fallback: autarky
    best_x   = np.array([min(Y1, hi[0]), min(Y2, hi[1])])

    for x0 in x_inits:
        x0 = np.clip(x0, lo, hi)
        try:
            res = minimize(neg_obj, x0, method='Nelder-Mead',
                           options={'xatol': 1e-7, 'fatol': 1e-7,
                                    'maxiter': 800, 'adaptive': True})
            val = -res.fun
            if val > best_val:
                # Verify enforcement constraints hold
                n1p_try, n2p_try = np.clip(res.x, lo, hi)
                sol = solve_I_star(n, n1p_try, n2p_try, Vf)
                if sol is not None and sol[1] > 1e-8:
                    best_val = val
                    best_x   = np.array([n1p_try, n2p_try])
        except Exception:
            pass

    n1p, n2p = best_x
    sol = solve_I_star(n, n1p, n2p, Vf)
    if sol is None:
        return best_val, n1p, n2p, 0.0, 0.0
    I_star, c, b, l = sol
    return best_val, n1p, n2p, I_star, b


def mh_vfi(V_aut, tol=1e-3, max_iter=60):
    """Value function iteration for the moral hazard model."""
    V  = V_aut.copy()
    Vf_aut = interp1d(n_grid, V_aut, fill_value='extrapolate', bounds_error=False)

    # Enforcement bounds: V(n_j') >= V_aut(δ*Y_j)
    # => n_j' >= V^{-1}(V_aut(δ*Y_j))
    # Since V(n) >= V_aut(n), the bound V_aut(δ*Y_j) is automatically
    # satisfied at n_j' >= δ*Y_j; use this conservative bound initially.
    thresh1_fixed = δ * Y1
    thresh2_fixed = δ * Y2

    pol_n1p = np.empty(N_n)
    pol_n2p = np.empty(N_n)
    pol_I   = np.empty(N_n)
    pol_b   = np.empty(N_n)

    for it in range(max_iter):
        Vf     = interp1d(n_grid, V, fill_value='extrapolate', bounds_error=False)
        V_new  = np.empty(N_n)

        for k, n in enumerate(n_grid):
            vn, n1p, n2p, I_star, b = mh_bellman_at_n(
                n, Vf, Vf_aut, thresh1_fixed, thresh2_fixed)
            V_new[k]   = vn
            pol_n1p[k] = n1p
            pol_n2p[k] = n2p
            pol_I[k]   = I_star
            pol_b[k]   = b

        diff = np.max(np.abs(V_new - V))
        V    = V_new
        print(f"  iter {it+1:3d},  max|ΔV| = {diff:.5f}")
        if diff < tol:
            print(f"MH VFI converged in {it+1} iterations.")
            break

    return V, pol_n1p, pol_n2p, pol_I, pol_b


print("Running moral hazard VFI…")
V_mh, pol_n1p, pol_n2p, pol_I, pol_b = mh_vfi(V_aut)
```

### Value Functions and Insurance

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: value functions
axes[0].plot(n_grid, V_aut, lw=2, label='Autarky')
axes[0].plot(n_grid, V_mh,  lw=2, ls='--', label='Moral hazard')
axes[0].set_xlabel('Net worth $n$')
axes[0].set_ylabel('Value')
axes[0].set_title('Value Functions')
axes[0].legend()

# Right: Risk-sharing index  RSI = (d_2 - d_1) / (Y_2 - Y_1)
# d_j = Y_j - n_j',  so RSI = (n_1' - n_2') / (Y_2 - Y_1) + 1
# ... actually RSI = (d_2 - d_1)/(Y_2-Y_1) = ((Y2-n2p)-(Y1-n1p))/(Y2-Y1)
d1_mh  = Y1 - pol_n1p
d2_mh  = Y2 - pol_n2p
RSI_mh = (d2_mh - d1_mh) / (Y2 - Y1)

axes[1].plot(n_grid, RSI_mh, lw=2, color='C1')
axes[1].axhline(1.0, ls=':', color='k',  lw=1, label='Full insurance (RSI=1)')
axes[1].axhline(0.0, ls='--', color='k', lw=1, label='Non-contingent debt (RSI=0)')
axes[1].set_xlabel('Net worth $n$')
axes[1].set_ylabel('Risk-sharing index')
axes[1].set_title('State Contingency of Repayment')
axes[1].set_ylim(-0.1, 1.2)
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"\nMean RSI (moral hazard): {np.mean(RSI_mh):.4f}")
print("→ Repayment is nearly state non-contingent (RSI ≈ 0)")
print("→ Moral hazard justifies why simple non-contingent debt is optimal")
```

A key finding of {cite}`Tsyrennikov2013` emerges immediately: the risk-sharing
index is close to **zero** throughout the state space.  Moral hazard requires
spreading continuation values to incentivise investment, but this is achieved
by differentiating *net worth* $n_j'$, not repayment $d_j = Y_j - n_j'$.
The near-equality $d_1 \approx d_2$ means repayment is essentially
**non-contingent on output** — the model rationalises why emerging market
borrowers use plain debt instruments rather than GDP-linked securities.

### Optimal Investment

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Compute autarky optimal investment
I_aut = np.array([autarky_bellman_at_n(n,
    interp1d(n_grid, V_aut, fill_value='extrapolate', bounds_error=False))[1]
    for n in n_grid])

axes[0].plot(n_grid, lam(I_aut), lw=2,        label='Autarky')
axes[0].plot(n_grid, lam(pol_I), lw=2, ls='--', label='Moral hazard')
axes[0].set_xlabel('Net worth $n$')
axes[0].set_ylabel(r'$\lambda(I) = \Pr(Y_2 \mid I)$')
axes[0].set_title('Investment (probability weight on high output)')
axes[0].legend()

axes[1].plot(n_grid, pol_n1p, lw=2,        label=r"$n_1' = Y_1 - d_1$  (after low output)")
axes[1].plot(n_grid, pol_n2p, lw=2, ls='--', label=r"$n_2' = Y_2 - d_2$  (after high output)")
axes[1].plot(n_grid, n_grid,  lw=1, ls=':',  color='k', label='45° line')
axes[1].set_xlabel('Net worth $n$')
axes[1].set_ylabel("Continuation net worth $n_j'$")
axes[1].set_title('Optimal Continuation Net Worth')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.show()
```

Investment under moral hazard is *lower* than in autarky at high net worth
levels and more sensitive to $n$ at low levels.  After a low output
realisation, net worth drops sharply ($n_1' \ll n$), depressing future
investment and perpetuating the crisis — the model's **internal propagation
mechanism**.

### Implied Interest Rate

```{code-cell} ipython3
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
    l   = lam(I)
    n1p = pol_n1p[k]
    n2p = pol_n2p[k]
    c1p = next_period_c(n1p)
    c2p = next_period_c(n2p)
    denom = β * ((1-l)*u_prime(c1p) + l*u_prime(c2p))
    R_n[k] = u_prime(c) / denom if denom > 1e-10 else np.nan

# Annualised spread over world rate
R_world = 1.0 / β_c                 # gross world rate
spread_ann = (R_n**4 - R_world**4)  # approximate annualised spread

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(n_grid, R_n, lw=2)
axes[0].axhline(R_world, ls='--', color='k', lw=1, label=f'World rate $1/\\beta_c = {R_world:.3f}$')
axes[0].set_xlabel('Net worth $n$')
axes[0].set_ylabel('Implied gross interest rate $R(n)$')
axes[0].set_title('Interest Rate Schedule')
axes[0].legend()

axes[1].plot(n_grid, np.clip(spread_ann * 100, -1, 50), lw=2)
axes[1].axhline(0, ls='--', color='k', lw=0.8)
axes[1].set_xlabel('Net worth $n$')
axes[1].set_ylabel('Annualised spread over world rate (%)')
axes[1].set_title('Interest Rate Spread')

plt.tight_layout()
plt.show()
```

The interest rate spread rises sharply at low net worth levels, consistent with
the Argentine data.  The mechanism is the **MH Euler equation**: when $n$ is
low, the borrower's continuation value is depressed and the spread in marginal
utilities across future states increases $R(n)$.

### Crisis Dynamics

{cite}`Tsyrennikov2013` shows that a string of low output realisations
generates gradual debt accumulation followed by a sudden stop in which capital
inflows cease and interest rates spike — a pattern consistent with the
Argentina 2001 experience.

```{code-cell} ipython3
def simulate_crisis(T_crisis=8):
    """
    Simulate crisis path: T_crisis periods of low output (Y_1) starting
    from zero debt (n_0 = Y2, high initial net worth).
    """
    n = Y2   # start with high net worth
    records = {'n': [n], 'debt_over_Y': [], 'R': [], 'ca_over_Y': [],
               'lam': []}

    for t in range(T_crisis):
        b   = float(pol_b_fn(n))
        I   = float(pol_I_fn(n))
        n1p = float(pol_n1p_fn(n))
        n2p = float(pol_n2p_fn(n))

        c   = n + b - θ * I
        l   = lam(I)
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
        records['lam'].append(l)

        n = n1p   # low output path

    records['n'] += [float(pol_n1p_fn(records['n'][-1]))]
    return records


crisis = simulate_crisis(T_crisis=8)
t_ax   = np.arange(len(crisis['R']))

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
fig.suptitle('Crisis Scenario: 8 Consecutive Low-Output Periods', fontsize=13)

axes[0,0].plot(t_ax, crisis['debt_over_Y'], 'o-', lw=2)
axes[0,0].set_ylabel('Debt / output')
axes[0,0].set_title('(A) Debt accumulation')

axes[0,1].plot(t_ax, np.array(crisis['R'])**4 * 100, 's-', lw=2, color='C1')
axes[0,1].axhline((1/β_c)**4 * 100, ls='--', color='k', lw=0.8,
                  label='World rate')
axes[0,1].set_ylabel('Annualised gross rate (%)')
axes[0,1].set_title('(B) Interest rate')
axes[0,1].legend(fontsize=9)

axes[1,0].plot(t_ax, crisis['ca_over_Y'], '^-', lw=2, color='C2')
axes[1,0].axhline(0, ls='--', color='k', lw=0.8)
axes[1,0].set_xlabel('Quarter')
axes[1,0].set_ylabel('Current account / output')
axes[1,0].set_title('(C) Current account')

axes[1,1].plot(t_ax, crisis['lam'], 'D-', lw=2, color='C3')
axes[1,1].set_xlabel('Quarter')
axes[1,1].set_ylabel(r'$\lambda(I) = \Pr(Y_2 \mid I)$')
axes[1,1].set_title('(D) Investment (prob. of high output)')

plt.tight_layout()
plt.show()
```

The simulation reproduces the stylised crisis pattern of {cite}`Tsyrennikov2013`,
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

### MH Versus Limited Enforcement

A crucial result of {cite}`Tsyrennikov2013` is that **limited enforcement adds
little** to the model's performance relative to moral hazard alone.  The
intuition: under LE (no moral hazard), optimal repayments are *highly state
contingent* (RSI ≈ 0.8), providing near-full insurance.  The borrower's net
worth drifts *upward* under LE (unlike MH where it drifts downward), so
interest rate spreads are transitory rather than persistent.

The following code illustrates the key theoretical distinction by computing
the Euler equation implications under each friction separately.

```{code-cell} ipython3
# Illustrate the Euler equation implications theoretically
fig, ax = plt.subplots(figsize=(8, 5))

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

expected_np_mh = (1 - lam(pol_I)) * pol_n1p + lam(pol_I) * pol_n2p

ax.plot(n_grid, expected_np_mh, lw=2,        label='MH: E[n\'] (drifts down)')
ax.plot(n_grid, n_grid,         lw=1, ls=':', color='k', label='45° line')

# Under LE (approximate): net worth is a supermartingale, E[n'] ≥ n * β/β_c
expected_np_le = n_grid * (β / β_c)
ax.plot(n_grid, expected_np_le, lw=2, ls='--', color='C2',
        label=r'LE: E[n\'] $\approx$ $(\beta/\beta_c)\,n$ (drifts up)')

ax.set_xlabel('Current net worth $n$')
ax.set_ylabel("Expected continuation net worth $E[n']$")
ax.set_title('Drift of Net Worth Under MH vs LE')
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()
```

Under moral hazard, $\mathbb{E}[n'] < n$: net worth drifts down and the
borrower spends substantial time near the borrowing limit, generating
persistent interest rate spreads.  Under limited enforcement,
$\mathbb{E}[n'] \geq (\beta/\beta_c)\,n$: net worth drifts toward a stationary
level and the borrower eventually escapes financial stress.

## Empirical Test

{cite}`Tsyrennikov2013` proposes a simple test to distinguish moral hazard from
limited enforcement.  After a low past output realisation ($y_{t-1} = Y_1$),
the MH contract lowers net worth sharply, reducing future consumption
smoothing.  This prediction is:

$$
\text{MH economy}: \quad
    \rho(c_t, y_t \mid y_{t-1} = Y_1) \;>\; \rho(c_t, y_t \mid y_{t-1} = Y_2),
$$

while the LE economy gives the opposite ordering (insurance is better after
low realisations).  Using Argentine quarterly data (1993–2005), the observed
correlations are 0.98 (after low output) vs. 0.91 (after high output) —
**consistent with moral hazard**.

## Exercises

````{admonition} Exercise 1
:class: exercise

**Effect of default penalty.**  The parameter $\delta \in (0,1)$ controls
the severity of the output loss upon default.

1. Compute $V_{\text{aut}}$ for $\delta \in \{0.5,\, 0.795,\, 0.95\}$.
2. For each $\delta$, evaluate the enforcement threshold
   $V_{\text{aut}}(\delta Y_1)$ and $V_{\text{aut}}(\delta Y_2)$.
3. Discuss: how does a harsher default penalty affect the tightness of the
   enforcement constraint and (via the Euler equation) the interest rate
   spread?  At $\delta = 1$ the LE constraint becomes $V(n_j') \geq V_{\text{aut}}(Y_j)$;
   at $\delta \to 0$ it is vacuous.
````

````{dropdown} Solution to Exercise 1
:class: dropdown

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 5))

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

ax.set_xlabel('Net worth $n$');  ax.set_ylabel('$V_{\\rm aut}(n)$')
ax.set_title('Enforcement Thresholds for Different Default Penalties δ')
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
```

A harsher default penalty (larger $\delta$) raises the enforcement thresholds,
tightening the participation constraints and reducing the scope for
state-contingent repayment.  Paradoxically, this may *reduce* the interest
rate spread by forcing the lender to offer more consumption insurance to keep
the borrower from defaulting.  At $\delta \to 0$ the enforcement constraint is
vacuous and the model collapses to pure moral hazard.
````

````{admonition} Exercise 2
:class: exercise

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
````

````{dropdown} Solution to Exercise 2
:class: dropdown

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 5))

for β_val, ls, color in [(0.990, '-', 'C0'), (0.980, '--', 'C1'), (0.950, ':', 'C2')]:
    globals()['β'] = β_val
    V_a_tmp = autarky_vfi()
    V_mh_tmp, pol_n1p_tmp, pol_n2p_tmp, pol_I_tmp, _ = mh_vfi(V_a_tmp)
    globals()['β'] = 0.980   # restore

    E_np = ((1 - lam(pol_I_tmp)) * pol_n1p_tmp
            + lam(pol_I_tmp) * pol_n2p_tmp)
    ax.plot(n_grid, E_np, ls=ls, color=color,
            label=fr'$\beta={β_val}$')

ax.plot(n_grid, n_grid, lw=1, ls=':', color='k', label='45° line')
ax.set_xlabel('Net worth $n$')
ax.set_ylabel("$E[n']$")
ax.set_title('Expected Continuation Net Worth for Different Discount Factors')
ax.legend()
plt.tight_layout()
plt.show()
```

The larger the discount wedge $\beta_c - \beta$, the faster net worth drifts
toward the borrowing limit.  When $\beta = \beta_c$ moral hazard alone drives
immiseration, while impatience accelerates it further.  A small wedge
(as calibrated by Tsyrennikov) is significant: it is *equivalent to
increasing the borrower's discount rate by 2% per annum* (even though
the assumed difference in quarterly rates is only 0.010).
````

````{admonition} Exercise 3
:class: exercise

**Non-contingency of optimal debt.**

The *risk-sharing index* $\text{RSI}(n) = (d_2(n) - d_1(n)) / (Y_2 - Y_1)$
measures how state-contingent the repayment schedule is.  RSI = 1 is full
insurance; RSI = 0 is non-contingent debt.

1. Compute RSI for the MH model you have already solved.
2. Now set $\beta = \beta_c$ (equal discounting) and recompute.  Does
   removing the impatience wedge change the near-zero RSI result?
3. Explain *theoretically* why moral hazard drives RSI toward zero.

*Hint*: From the Euler equation, the spread in marginal utilities
$u'(c_1') / u'(c_2')$ depends on the IC multiplier $\mu$.  A larger $\mu$
spreads continuation values but *not necessarily* repayments.
````

````{dropdown} Solution to Exercise 3
:class: dropdown

```{code-cell} ipython3
# RSI for the baseline MH model
d1_mh = Y1 - pol_n1p
d2_mh = Y2 - pol_n2p
RSI   = (d2_mh - d1_mh) / (Y2 - Y1)

print(f"Baseline MH:  mean RSI = {np.mean(RSI):.4f},  max RSI = {np.max(RSI):.4f}")
print()
print("Theoretical explanation:")
print(" Under moral hazard the planner must spread V(n2') - V(n1') to")
print(" incentivise investment via the FOA.  This is achieved by setting")
print(" n2' > n1', i.e. spreading *net worth*, not repayments.")
print(" Since d_j = Y_j - n_j', if n2' - n1' = Y2 - Y1 then d1 = d2 (RSI=0).")
print(" Moral hazard forces this near-equality, making debt non-contingent.")
```
````

## References

```{bibliography}
:filter: docname in docnames
```
