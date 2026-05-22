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

(atkeson_1991)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Lending with Moral Hazard and Risk of Repudiation

## Overview

This lecture studies {cite}`Atkeson1991`, which examines the **constrained
optimal pattern of capital flows** between an international lender and a
sovereign borrower subject to two frictions:

1. **Moral hazard** — lenders cannot observe whether the borrower invests or
   simply consumes loan proceeds.
2. **Risk of repudiation** — as a sovereign, the borrower can unilaterally
   renounce its debt at any time.

A  central result is tha,  under an optimal contract, a ''sudden stop'' or ''debt-crisis'' emerges in  which  a borrowing
country must **export capital** after suffering an adverse output shock.  

Outflows of capital after bad output realizations are the
mechanism that incentivizes investment.

The model extends recursive techniques from {cite}`APS1986`, {cite}`APS1990`,
and {cite}`Spear_Srivastava_87` to an environment with a **physical state
variable** that changes across periods.

```{note}
Atkeson (1991) uses $\delta$ for the discount factor.  We follow the
QuantEcon convention and write $\beta$ throughout.
```

## The Environment

### Technology

Time is discrete, $t = 0, 1, 2, \ldots$.

In each period the borrower chooses
investment $I_t \geq 0$.

Given investment $I_t$, next period's output
$Y_{t+1}$ is drawn from the conditional distribution

$$
g(Y';\,I) \;=\; \lambda(I)\,g_0(Y') + \bigl[1 - \lambda(I)\bigr]\,g_1(Y'),
$$

where $Y' \in \mathcal{Y} = \{Y_1, \ldots, Y_N\}$ with $Y_N \geq \cdots \geq
Y_1 > 0$.

The weight $\lambda : [0,I_{\max}] \to [0,1]$ is strictly
increasing, so higher investment shifts the distribution toward higher outputs.

The ratio $g_0(Y_i')/g_1(Y_i')$ is assumed to be **monotone increasing in
$i$** (monotone likelihood ratio property).

This means low output is a
relatively strong signal that the borrower invested little.

### Agents and Preferences

**The borrower** is an infinitely-lived, risk-averse agent with normalised
discounted utility

$$
U^B(\sigma) \;=\; (1 - \beta)\,\mathbb{E}_0^{\sigma}
    \sum_{t=0}^{\infty} \beta^t \, u(c_t),
$$

where $u$ is strictly concave with $u'(0) = +\infty$.

**Lenders** are a sequence of short-lived, risk-neutral agents, one born each
period.  The lender born at $t$ extends loan $b_t$ when young and collects
state-contingent repayment $d_{t+1}(Y_{t+1})$ when old.  A lender's
participation (zero-profit) constraint is

$$
-b_t + \beta \sum_{Y'} d_{t+1}(Y')\,g(Y';\,I_t) \;\geq\; 0.
$$

### State Variable and Feasibility

Define

$$
Q_t \;\equiv\; Y_t - d_t(Y_t)
$$

as **output net of repayment** — the resources available to the borrower
after settling the old lender's claim.

An allocation
$\sigma = \{c_t(Q^t),\,I_t(Q^t),\,b_t(Q^t),\,d_{t+1}(Y_{t+1};Q^t)\}$
is **feasible** if for all $t$ and histories:

$$
c_t + I_t - b_t \;\leq\; Q_t, \quad c_t,\, I_t \geq 0,
    \quad b_t,\; -d_{t+1}(Y') \leq M,
$$

where $M$ is the lender's endowment per period.

### Autarky Value

The value the borrower can attain without credit access satisfies

$$
U^B_{\text{aut}}(Z) \;=\; \max_{I \in [0, Z]}
    \Bigl[(1-\beta)\,u(Z - I) + \beta \sum_{Y'} U^B_{\text{aut}}(Y')\,g(Y';\,I)\Bigr].
$$

## Two Impediments to Contracting

### Moral Hazard

Because lenders can observe only $Y_t$, not the borrower's investment $I_t$,
any feasible allocation must be **incentive compatible**: the borrower prefers
the prescribed $(c_t, I_t)$ to any alternative consumption-investment plan
given the loan and repayment schedule.

### Risk of Repudiation

If the borrower repudiates its debt after $Y_{t+1}$ is realized, future
lenders refuse credit and the borrower is confined to autarky.

An allocation is **immune from repudiation** if, for every output realization $Y_{t+1}$,

$$
U^B\bigl(\sigma\,\big|\,_{Q^t;\,Y_{t+1}}\bigr)
    \;\geq\; U^B_{\text{aut}}(Y_{t+1}).
$$

The continuation value of the contract — evaluated at the post-repayment state
$Q_{t+1} = Y_{t+1} - d_{t+1}(Y_{t+1})$ — must weakly exceed what the
borrower would obtain by repudiating and retaining all of $Y_{t+1}$.

## The Constrained Pareto Problem

An allocation is **constrained Pareto optimal** if it maximises the borrower's
payoff $U^B(\sigma)$ subject to:

1. Feasibility
2. Individual rationality (lenders earn at least zero)
3. Immunity from repudiation
4. Incentive compatibility

## Recursive Formulation

### Self-Generation and Factorization

Let $V(Q)$ be the set of payoffs the borrower can achieve from allocations
satisfying constraints (1)–(4) when the state is $Q$.

Atkeson adapts the
**self-generation** and **factorization** results of {cite}`APS1990` to this
setting with a physical state variable.

Define a pair $(A, U)$ of current
controls and a continuation value function to be *admissible with respect to*
$W$ at $Q$ if it satisfies one-period versions of constraints (1)–(4) and
$U(Q') \in W(Q')$ for all $Q'$.

Let $B(W)(Q)$ be the set of payoffs
generated by admissible pairs.

Then:

- **Proposition 1 (Self-generation):** If $W$ is self-generating
  ($W(Q) \subseteq B(W)(Q)$ for all $Q$), then $B(W)(Q) \subseteq V(Q)$.
- **Proposition 2 (Factorization):** $V(Q) \subseteq B(V)(Q)$ for all $Q$.

These propositions imply $V = B(V)$, characterising the utility possibility
correspondence as the fixed point of $B$.

### Program P*

**Proposition 5** ({cite}`Atkeson1991`): The value function $\bar V(Q)$ of the
constrained Pareto optimal contract satisfies the functional equation

$$
\bar{V}(Q) \;=\; \max_{c,\,I,\,b,\,d'(\cdot)}
    \;(1-\beta)\,u(c) + \beta \sum_{Y'} \bar{V}\!\bigl(Y' - d'(Y')\bigr)\,g(Y';\,I)
$$

subject to feasibility, lender participation, no-repudiation, and incentive
compatibility.

Moreover, the optimal *continuation* value function equals
$\bar{V}$ itself.

This mirrors Bellman's principle: the **continuation of the optimal contract
is itself optimal** at the updated state.

### Capital Outflows After Low Output

The first-order condition of the Lagrangian for Program P* with respect to the
continuation value $U_d(Y_i')$ reveals that the no-repudiation Lagrange
multiplier $\mu_3(Y_i') > 0$ whenever

$$
1 + \mu_4 \,\frac{g_1(Y_i';\,I)}{g(Y_i';\,I)} < 0.
$$

The ratio $g_1(Y_i';I)/g(Y_i';I)$ measures the likelihood that output $Y_i'$
signals *low* investment.

By the monotone likelihood ratio property, this
ratio is largest for the **lowest output states**.

When $\mu_3(Y_i') > 0$,
the no-repudiation constraint binds: $\bar{V}(Y_i' - d'(Y_i')) =
U^B_{\text{aut}}(Y_i')$.

Repayment $d'(Y_i')$ is then at its maximum and the
new loan available at the continuation state is limited, producing a net
**capital outflow**:

$$
\underbrace{d'(Y_i')}_{\text{repayment to old lender}}
    \;>\; \underbrace{b'\!\bigl(Q'\bigr)}_{\text{new loan from young lender}}.
$$

## Computation

We illustrate these results with a binary-investment, two-output version of
the model.

### Setup

```{code-cell} ipython3
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12, 'figure.dpi': 100})

# ── Discount factor ──────────────────────────────────────────────────────────
β = 0.9

# ── Binary investment (high I_h or zero) ─────────────────────────────────────
I_h = 0.2          # resource cost of high investment

# ── Output states ─────────────────────────────────────────────────────────────
Y_L, Y_H = 0.5, 1.0
Y = np.array([Y_L, Y_H])

# ── Output distributions ──────────────────────────────────────────────────────
# g_h[j] = Pr(Y[j] | invest I_h),  g_l[j] = Pr(Y[j] | invest 0)
g_h = np.array([0.25, 0.75])   # high investment → likely high output
g_l = np.array([0.75, 0.25])   # low  investment → likely low  output

# Monotone likelihood ratio: g_l[0]/g_h[0] = 3 > 1 > g_l[1]/g_h[1] = 1/3
# → Y_L strongly signals low investment; Y_H signals high investment
Δg = g_h - g_l                  # [−0.5,  0.5]

# ── Lender endowment (large: b, −d ≤ M) ──────────────────────────────────────
M = 10.0

# ── State grid: Q = Y − d (resources after repaying old debt) ─────────────────
N_Q   = 200
Q_MIN = 0.02
Q_MAX = 1.8
Q_grid = np.linspace(Q_MIN, Q_MAX, N_Q)

# ── Utility ────────────────────────────────────────────────────────────────────
def u(c):
    return np.log(np.maximum(c, 1e-12))

print(f"Likelihood ratios g_l / g_h : {g_l / g_h}")
print(f"Y_L signals low investment with ratio {g_l[0]/g_h[0]:.1f}x")
```

### Autarky Value Function

In autarky the borrower has no access to credit.

Starting each period with
resources $Q$, the borrower solves

$$
U_{\text{aut}}(Q) =
    \max_{I \in \{0,\,I_h\}}
    \Bigl[(1-\beta)\,u(Q - I) + \beta
    \bigl[g(I)_L\,U_{\text{aut}}(Y_L) + g(I)_H\,U_{\text{aut}}(Y_H)\bigr]\Bigr].
$$

Note that the continuation values depend only on $Y_L$ and $Y_H$, not on the
current $Q$, because next period's state is simply the realised output.

```{code-cell} ipython3
def autarky_vfi(tol=1e-8, max_iter=3000):
    """Value function iteration for the autarky problem (binary investment)."""
    V = np.zeros(N_Q)

    for it in range(max_iter):
        Vf    = interp1d(Q_grid, V, fill_value='extrapolate', bounds_error=False)
        EV_h  = float(g_h @ Vf(Y))   # E[V(Y') | invest I_h]
        EV_l  = float(g_l @ Vf(Y))   # E[V(Y') | invest 0 ]

        V_new = np.empty(N_Q)
        for k, Q in enumerate(Q_grid):
            val_h = (1-β)*u(Q - I_h) + β*EV_h  if Q > I_h + 1e-10 else -np.inf
            val_l = (1-β)*u(Q)       + β*EV_l
            V_new[k] = max(val_h, val_l)

        diff = np.max(np.abs(V_new - V))
        V    = V_new
        if diff < tol:
            print(f"Autarky VFI converged in {it+1} iterations (diff={diff:.2e})")
            break

    return V

V_aut = autarky_vfi()
```

### Constrained Pareto Optimal Contract

We solve Program P* iteratively.

At each state $Q$, the planner chooses
continuation states $(Q'_L, Q'_H)$ — equivalently, state-contingent
repayments $d_j = Y_j - Q'_j$ — to maximise the borrower's payoff.

Taking lender participation **binding** (Proposition 5 implies this is without
loss of generality), the loan is determined by

$$
b^* \;=\; \beta\bigl[g_{h,L}(Y_L - Q'_L) + g_{h,H}(Y_H - Q'_H)\bigr],
$$

and current consumption is $c^* = Q + b^* - I_h$.

The optimisation reduces
to a two-dimensional problem in $(Q'_L, Q'_H)$:

$$
\max_{Q'_L,\,Q'_H}
    (1-\beta)\,u(c^*) + \beta\bigl[V(Q'_L)\,g_{h,L} + V(Q'_H)\,g_{h,H}\bigr]
$$

subject to:

- **(NR)** $V(Q'_j) \geq U_{\text{aut}}(Y_j)$, i.e. $Q'_j \geq Q^*_j$
- **(IC)** $\beta\,\Delta g \cdot V(Q') \geq (1-\beta)\,[u(c^*+I_h) - u(c^*)]$
- **(F)** $c^* \geq 0$

where $\Delta g_j = g_{h,j} - g_{l,j}$ and $Q^*_j = V^{-1}(U_{\text{aut}}(Y_j))$.

```{code-cell} ipython3
def find_Qmin(V_arr, v_thresh):
    """Return min Q on grid with V(Q) >= v_thresh (no-repudiation lower bound)."""
    if v_thresh <= V_arr[0]:
        return float(Q_MIN)
    if v_thresh >= V_arr[-1]:
        return float(Q_MAX)
    # Use searchsorted on a monotone version of V
    V_mono = np.maximum.accumulate(V_arr)   # enforce monotone for inversion
    idx    = np.searchsorted(V_mono, v_thresh)
    idx    = np.clip(idx, 1, N_Q - 1)
    denom  = V_mono[idx] - V_mono[idx-1]
    if abs(denom) < 1e-14:
        return float(Q_grid[idx-1])
    t = (v_thresh - V_mono[idx-1]) / denom
    return float(Q_grid[idx-1] + t * (Q_grid[idx] - Q_grid[idx-1]))


def pareto_bellman(V, V_aut_arr):
    """
    One application of the constrained Pareto Bellman operator.
    Returns updated V, optimal loan policy pol_b, and continuation
    states pol_Qp[:,0] (after Y_L) and pol_Qp[:,1] (after Y_H).
    """
    Vf      = interp1d(Q_grid, V,         fill_value='extrapolate',
                       bounds_error=False)
    Vaut_f  = interp1d(Q_grid, V_aut_arr, fill_value='extrapolate',
                       bounds_error=False)

    # No-repudiation lower bounds on Q'_j
    Vaut_Y  = np.array([float(Vaut_f(yj)) for yj in Y])
    Qp_min  = np.array([find_Qmin(V, v)   for v  in Vaut_Y])
    Qp_min  = np.clip(Qp_min, Q_MIN, Q_MAX - 1e-4)

    V_new   = np.empty(N_Q)
    pol_b   = np.empty(N_Q)
    pol_Qp  = np.empty((N_Q, 2))

    for k, Q in enumerate(Q_grid):

        def bc(Qp):
            """Compute loan b* and consumption c* from continuation states."""
            b = β * (g_h[0]*(Y_L - Qp[0]) + g_h[1]*(Y_H - Qp[1]))
            c = Q + b - I_h
            return b, c

        def neg_obj(Qp):
            b, c = bc(Qp)
            if c < 1e-10:
                return 1e10
            VQp = np.array([float(Vf(Qp[0])), float(Vf(Qp[1]))])
            return -((1-β)*u(c) + β*(VQp @ g_h))

        def ic_slack(Qp):
            """IC: β Δg·V(Q') − (1−β)[u(c+I_h)−u(c)] ≥ 0."""
            b, c = bc(Qp)
            if c < 1e-10:
                return -1e10
            VQp = np.array([float(Vf(Qp[0])), float(Vf(Qp[1]))])
            return β*(Δg @ VQp) - (1-β)*(u(c + I_h) - u(c))

        def feas_slack(Qp):
            """Feasibility: c* ≥ 0."""
            _, c = bc(Qp)
            return c

        constraints = [
            {'type': 'ineq', 'fun': ic_slack},
            {'type': 'ineq', 'fun': feas_slack},
        ]
        bounds = [(Qp_min[0], Q_MAX), (Qp_min[1], Q_MAX)]

        # Candidate starting points
        x_inits = [
            np.array([Qp_min[0],              Qp_min[1]]),
            np.array([min(Y_L * 0.9, Q_MAX),  min(Y_H * 0.9, Q_MAX)]),
            np.array([Qp_min[0],              min(Y_H * 0.8, Q_MAX)]),
        ]

        best_val = float(Vaut_f(Q))   # fallback: autarky payoff
        best_Qp  = np.array([min(max(Y_L, Qp_min[0]), Q_MAX),
                              min(max(Y_H, Qp_min[1]), Q_MAX)])

        for x0 in x_inits:
            x0 = np.clip(x0, [Qp_min[0], Qp_min[1]],
                         [Q_MAX - 1e-5, Q_MAX - 1e-5])
            try:
                res = minimize(
                    neg_obj, x0, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'ftol': 1e-10, 'maxiter': 400})
                if (res.success or res.status in (0, 9)):
                    val = -res.fun
                    if (val > best_val
                            and ic_slack(res.x)   >= -1e-5
                            and feas_slack(res.x) >= -1e-5):
                        best_val = val
                        best_Qp  = res.x
            except Exception:
                pass

        V_new[k]  = best_val
        pol_Qp[k] = best_Qp
        b_opt, _  = bc(best_Qp)
        pol_b[k]  = b_opt

    return V_new, pol_b, pol_Qp


def pareto_vfi(V_aut_arr, tol=1e-3, max_iter=60):
    """Value function iteration for Program P*."""
    V = V_aut_arr.copy()

    for it in range(max_iter):
        V_new, pol_b, pol_Qp = pareto_bellman(V, V_aut_arr)
        diff = np.max(np.abs(V_new - V))
        V    = V_new
        print(f"  iter {it+1:3d},  max|ΔV| = {diff:.5f}")
        if diff < tol:
            print(f"Pareto VFI converged in {it+1} iterations.")
            break

    return V, pol_b, pol_Qp


print("Running constrained Pareto VFI …")
V_pareto, pol_b, pol_Qp = pareto_vfi(V_aut)
```

### Value Functions

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(Q_grid, V_aut,    lw=2,       label=r'Autarky  $U_{\rm aut}(Q)$')
ax.plot(Q_grid, V_pareto, lw=2, ls='--',
        label=r'Optimal contract  $\bar{V}(Q)$')

ax.set_xlabel(r'State $Q$ (output net of repayment)')
ax.set_ylabel('Normalised utility')
ax.set_title('Value Functions')
ax.legend()
plt.tight_layout()
plt.show()
```

The optimal contract strictly dominates autarky, $\bar{V}(Q) > U_{\text{aut}}(Q)$,
because access to credit allows the borrower to share risk with lenders and
smooth consumption across output realisations.

### Optimal Continuation States and the No-Repudiation Constraint

```{code-cell} ipython3
# Compute no-repudiation floors
Vaut_at_Y = np.array([float(interp1d(Q_grid, V_aut,
                fill_value='extrapolate', bounds_error=False)(yj)) for yj in Y])
Qp_min_L  = find_Qmin(V_pareto, Vaut_at_Y[0])
Qp_min_H  = find_Qmin(V_pareto, Vaut_at_Y[1])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Q'_L (continuation state after low output)
axes[0].plot(Q_grid, pol_Qp[:, 0], lw=2, label=r"$Q'_L = Y_L - d_L$")
axes[0].axhline(Qp_min_L, ls='--', color='C3',
                label=fr"NR floor $Q^*_L \approx {Qp_min_L:.3f}$")
axes[0].set_xlabel(r'State $Q$')
axes[0].set_ylabel(r"$Q'_L$")
axes[0].set_title(r'Continuation state after low output $Y_L$')
axes[0].legend()

# Right: Q'_H (continuation state after high output)
axes[1].plot(Q_grid, pol_Qp[:, 1], lw=2, color='C1',
             label=r"$Q'_H = Y_H - d_H$")
axes[1].axhline(Qp_min_H, ls='--', color='C3',
                label=fr"NR floor $Q^*_H \approx {Qp_min_H:.3f}$")
axes[1].set_xlabel(r'State $Q$')
axes[1].set_ylabel(r"$Q'_H$")
axes[1].set_title(r'Continuation state after high output $Y_H$')
axes[1].legend()

plt.tight_layout()
plt.show()
```

After a **low-output** realisation, the continuation state $Q'_L$ is pinned at the no-repudiation floor $Q^*_L$.  This means the repayment $d_L = Y_L - Q'_L$ is as large as the repudiation constraint allows.  After a **high-output** realisation, $Q'_H > Q^*_H$: the constraint is slack and the borrower retains more resources, rewarding the high investment that produced good output.

### Optimal Loan and Net Capital Flows

```{code-cell} ipython3
# Repayments at the two output states as functions of current state Q
d_L_policy = Y_L - pol_Qp[:, 0]   # d_L(Q) = Y_L − Q'_L(Q)
d_H_policy = Y_H - pol_Qp[:, 1]   # d_H(Q) = Y_H − Q'_H(Q)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(Q_grid, pol_b,       lw=2, label='Loan $b^*(Q)$')
axes[0].plot(Q_grid, d_L_policy,  lw=2, ls='--', label=r'Repayment $d_L$')
axes[0].plot(Q_grid, d_H_policy,  lw=2, ls=':',  label=r'Repayment $d_H$')
axes[0].axhline(0, color='k', lw=0.6, ls=':')
axes[0].set_xlabel(r'State $Q$')
axes[0].set_title('Loan and Repayment Policies')
axes[0].legend()

# Net capital outflow at continuation state
pol_b_fn   = interp1d(Q_grid, pol_b, fill_value='extrapolate', bounds_error=False)
net_out_L  = d_L_policy - pol_b_fn(pol_Qp[:, 0])
net_out_H  = d_H_policy - pol_b_fn(pol_Qp[:, 1])

axes[1].plot(Q_grid, net_out_L, lw=2,        label=r'After $Y_L$ (low output)')
axes[1].plot(Q_grid, net_out_H, lw=2, ls='--', label=r'After $Y_H$ (high output)')
axes[1].axhline(0, color='k', lw=0.8, ls=':')
axes[1].set_xlabel(r'State $Q$')
axes[1].set_ylabel(r'Net outflow  $d(Y') - b'(Q')$')
axes[1].set_title('Capital Flows in the Optimal Contract')
axes[1].legend()

plt.tight_layout()
plt.show()
```

After a low-output realisation, $d_L > b'(Q'_L)$: the net capital flow is an
**outflow**.  After a high-output realisation, the borrower typically receives
a net capital inflow.  This is the numerical counterpart of Proposition 7 in
{cite}`Atkeson1991`.

### Simulation

```{code-cell} ipython3
def simulate_contract(V_pareto, pol_b, pol_Qp, T=150, seed=0):
    """
    Simulate the constrained optimal contract.
    At each period the borrower invests I_h and output is drawn from g_h.
    Returns time series for Q, Y, consumption c, loan b, repayment d,
    and net capital outflow.
    """
    rng = np.random.default_rng(seed)

    Qp_fn = [interp1d(Q_grid, pol_Qp[:, j],
                      fill_value='extrapolate', bounds_error=False)
             for j in range(2)]
    b_fn  = interp1d(Q_grid, pol_b, fill_value='extrapolate', bounds_error=False)

    Q = float(np.median(Q_grid))   # start at median state

    out = {'Q': [], 'Y': [], 'c': [], 'b': [], 'd': [], 'net_out': []}

    for _ in range(T):
        b  = float(b_fn(Q))
        c  = Q + b - I_h
        c  = max(c, 1e-10)

        j  = int(rng.choice(2, p=g_h))    # draw next output index
        Yp = Y[j]
        Qp = float(Qp_fn[j](Q))           # next state

        d       = Yp - Qp                  # repayment at start of next period
        b_next  = float(b_fn(Qp))
        net_out = d - b_next               # net capital outflow at next period

        out['Q'].append(Q); out['Y'].append(Yp); out['c'].append(c)
        out['b'].append(b); out['d'].append(d);  out['net_out'].append(net_out)

        Q = Qp

    return {k: np.array(v) for k, v in out.items()}


sim = simulate_contract(V_pareto, pol_b, pol_Qp, T=150)
t   = np.arange(len(sim['Q']))

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(t, sim['Y'], alpha=0.6, label='Output $Y_{t+1}$')
axes[0].plot(t, sim['c'], lw=1.8,    label='Consumption $c_t$')
axes[0].set_ylabel('Level')
axes[0].legend(ncol=2, loc='upper right')
axes[0].set_title('Simulation of the Constrained Optimal Contract')

axes[1].plot(t, sim['d'], lw=1.8,          label='Repayment $d_t$')
axes[1].plot(t, sim['b'], lw=1.8, ls='--', label='New loan $b_t$')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_ylabel('Level')
axes[1].legend(ncol=2)

colors = ['#d73027' if x > 0 else '#4575b4' for x in sim['net_out']]
axes[2].bar(t, sim['net_out'], color=colors, label='Net capital outflow')
axes[2].axhline(0, color='k', lw=0.6)
axes[2].set_xlabel('Period $t$')
axes[2].set_ylabel('Net outflow')
axes[2].legend()

plt.tight_layout()
plt.show()

# Tabulate statistics
low_out_frac = np.mean(sim['net_out'] > 0)
print(f"\nFraction of periods with capital outflow:  {low_out_frac:.2%}")
print(f"Fraction of low-output periods:            "
      f"{np.mean(sim['Y'] == Y_L):.2%}")
```

Red bars (capital outflows) in the bottom panel systematically coincide with
periods in which low output $Y_L$ is realised, confirming the key prediction
of the model.

## Exercises

````{admonition} Exercise 1
:class: exercise

**Patience and the severity of debt crises.**

Redo the analysis with $\beta = 0.8$ and $\beta = 0.95$ (keep all other
parameters fixed).

1. For each value of $\beta$, compute the autarky and optimal contract value
   functions.
2. Compute the no-repudiation lower bounds $Q^*_L$ and $Q^*_H$.
3. Plot $Q'_L(Q)$ for the three values of $\beta$ on a single figure.
4. Discuss: how does the borrower's patience affect how tightly the
   no-repudiation constraint binds after low output?
````

````{dropdown} Solution to Exercise 1
:class: dropdown

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 5))

for β_val, ls, color in [(0.8, '-', 'C0'), (0.9, '--', 'C1'), (0.95, ':', 'C2')]:
    β_orig = β
    import builtins
    # Temporarily override β in module scope
    globals()['β'] = β_val
    globals()['Δg'] = g_h - g_l   # unchanged but recomputed for clarity

    V_a   = autarky_vfi()
    V_p, _, pQp = pareto_vfi(V_a)

    globals()['β'] = β_orig
    globals()['Δg'] = g_h - g_l

    Vaut_fn_tmp = interp1d(Q_grid, V_a, fill_value='extrapolate',
                           bounds_error=False)
    Vaut_Y_tmp  = np.array([float(Vaut_fn_tmp(yj)) for yj in Y])
    Qmin_L_tmp  = find_Qmin(V_p, Vaut_Y_tmp[0])

    ax.plot(Q_grid, pQp[:, 0], ls=ls, color=color,
            label=fr'$\beta = {β_val}$  (NR floor $\approx {Qmin_L_tmp:.3f}$)')

ax.set_xlabel(r'State $Q$')
ax.set_ylabel(r"$Q'_L$  (continuation state after low output)")
ax.set_title('Effect of Patience on No-Repudiation Constraint')
ax.legend()
plt.tight_layout()
plt.show()
```

More patient borrowers ($\beta$ closer to 1) value the continuation of the
contract more highly, which relaxes the no-repudiation constraint: the
no-repudiation floor $Q^*_L$ falls and the capital outflow after low output is
less severe.  Impatient borrowers more readily prefer autarky, tightening the
constraint and worsening debt-crisis dynamics.
````

````{admonition} Exercise 2
:class: exercise

**Signal quality and capital flows.**

Replace the output distribution with the more symmetric values
$g_h = (0.40, 0.60)$ and $g_l = (0.60, 0.40)$, so that output is a
weaker signal of investment.

1. Recompute the autarky and optimal contract value functions.
2. Plot the net capital outflow curves $d(Y_j) - b'(Q'_j)$ as a function
   of $Q$ for both the baseline and the weak-signal specification.
3. Explain intuitively why weaker signal quality changes the capital flow
   pattern.
````

````{dropdown} Solution to Exercise 2
:class: dropdown

```{code-cell} ipython3
g_h_base, g_l_base = g_h.copy(), g_l.copy()
Δg_base = Δg.copy()

# Weak-signal specification
globals()['g_h'] = np.array([0.40, 0.60])
globals()['g_l'] = np.array([0.60, 0.40])
globals()['Δg']  = g_h - g_l

print("Weak-signal likelihood ratios g_l/g_h:", g_l / g_h)

V_aut_ws           = autarky_vfi()
V_par_ws, pb_ws, pQp_ws = pareto_vfi(V_aut_ws)

pb_fn_ws = interp1d(Q_grid, pb_ws, fill_value='extrapolate', bounds_error=False)
net_L_ws = (Y_L - pQp_ws[:, 0]) - pb_fn_ws(pQp_ws[:, 0])
net_H_ws = (Y_H - pQp_ws[:, 1]) - pb_fn_ws(pQp_ws[:, 1])

# Restore baseline
globals()['g_h'] = g_h_base
globals()['g_l'] = g_l_base
globals()['Δg']  = Δg_base

pb_fn_bl = interp1d(Q_grid, pol_b, fill_value='extrapolate', bounds_error=False)
net_L_bl = (Y_L - pol_Qp[:, 0]) - pb_fn_bl(pol_Qp[:, 0])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Q_grid, net_L_bl, lw=2,          label=r'After $Y_L$, baseline (strong signal)')
ax.plot(Q_grid, net_L_ws, lw=2, ls='--', label=r'After $Y_L$, weak signal')
ax.axhline(0, color='k', lw=0.8, ls=':')
ax.set_xlabel(r'State $Q$')
ax.set_ylabel('Net capital outflow')
ax.set_title('Capital Flows: Baseline vs Weak Signal')
ax.legend()
plt.tight_layout()
plt.show()
```

With a weaker signal ($g_l/g_h$ closer to 1), low output is less informative
about past investment.  The moral hazard problem is milder, incentive
constraints are easier to satisfy, and the no-repudiation constraint binds less
tightly.  Capital outflows after bad output realisations are smaller in
magnitude.
````

````{admonition} Exercise 3
:class: exercise

**Debt forgiveness and welfare.**

A debt relief programme can be modelled as an exogenous upward shift in the
no-repudiation threshold: suppose the borrower's outside option improves to
$\tilde{U}_{\text{aut}}(Y_j) = U_{\text{aut}}(Y_j) + \varepsilon$ for a small
$\varepsilon > 0$.

1. For $\varepsilon \in \{0, 0.05, 0.10\}$, compute the constrained optimal
   value function under the tightened repudiation constraint.
2. Plot $\bar{V}(Q)$ for each $\varepsilon$.
3. Discuss: when is debt forgiveness welfare improving for the borrower?
   What is the cost to lenders?

*Hint:* implement the shift by adding $\varepsilon$ to `Vaut_at_Y` inside
`pareto_bellman`.
````

````{dropdown} Solution to Exercise 3
:class: dropdown

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 5))

for eps, ls, color in [(0.0, '-', 'C0'), (0.05, '--', 'C1'), (0.10, ':', 'C2')]:

    def pareto_bellman_shifted(V, V_aut_arr, epsilon=eps):
        """Same as pareto_bellman but with tightened NR threshold."""
        Vf     = interp1d(Q_grid, V,         fill_value='extrapolate',
                          bounds_error=False)
        Vaut_f = interp1d(Q_grid, V_aut_arr, fill_value='extrapolate',
                          bounds_error=False)

        Vaut_Y = np.array([float(Vaut_f(yj)) for yj in Y]) + epsilon
        Qp_min = np.array([find_Qmin(V, v) for v in Vaut_Y])
        Qp_min = np.clip(Qp_min, Q_MIN, Q_MAX - 1e-4)

        V_new  = np.empty(N_Q)
        pol_b_ = np.empty(N_Q)
        pol_Qp_= np.empty((N_Q, 2))

        for k, Q in enumerate(Q_grid):
            def bc(Qp):
                b = β * (g_h[0]*(Y_L-Qp[0]) + g_h[1]*(Y_H-Qp[1]))
                return b, Q + b - I_h

            def neg_obj(Qp):
                b, c = bc(Qp)
                if c < 1e-10: return 1e10
                VQp = np.array([float(Vf(Qp[0])), float(Vf(Qp[1]))])
                return -((1-β)*u(c) + β*(VQp @ g_h))

            def ic_slack(Qp):
                b, c = bc(Qp)
                if c < 1e-10: return -1e10
                VQp = np.array([float(Vf(Qp[0])), float(Vf(Qp[1]))])
                return β*(Δg @ VQp) - (1-β)*(u(c+I_h) - u(c))

            def feas_slack(Qp):
                return bc(Qp)[1]

            constr = [{'type': 'ineq', 'fun': ic_slack},
                      {'type': 'ineq', 'fun': feas_slack}]
            bounds = [(Qp_min[0], Q_MAX), (Qp_min[1], Q_MAX)]

            best_val = float(Vaut_f(Q)) - epsilon   # rough fallback
            best_Qp  = np.clip([max(Qp_min[0], Y_L), max(Qp_min[1], Y_H)],
                               [Qp_min[0], Qp_min[1]], [Q_MAX]*2)

            for x0 in [np.array([Qp_min[0], Qp_min[1]]),
                       np.array([min(Y_L, Q_MAX), min(Y_H*0.9, Q_MAX)])]:
                x0 = np.clip(x0, [Qp_min[0], Qp_min[1]], [Q_MAX-1e-5]*2)
                try:
                    res = minimize(neg_obj, x0, method='SLSQP',
                                   bounds=bounds, constraints=constr,
                                   options={'ftol': 1e-10, 'maxiter': 300})
                    if (res.success or res.status in (0, 9)):
                        val = -res.fun
                        if val > best_val and ic_slack(res.x) >= -1e-5:
                            best_val = val
                            best_Qp  = res.x
                except Exception:
                    pass

            V_new[k]   = best_val
            pol_Qp_[k] = best_Qp
            pol_b_[k]  = bc(best_Qp)[0]

        return V_new, pol_b_, pol_Qp_

    V_eps = V_aut.copy()
    for it in range(50):
        V_new_eps, _, _ = pareto_bellman_shifted(V_eps, V_aut, epsilon=eps)
        diff = np.max(np.abs(V_new_eps - V_eps))
        V_eps = V_new_eps
        if diff < 1e-3:
            break

    ax.plot(Q_grid, V_eps, ls=ls, color=color,
            label=fr'$\varepsilon = {eps}$')

ax.plot(Q_grid, V_aut, lw=1, color='k', ls=':', label='Autarky')
ax.set_xlabel(r'State $Q$')
ax.set_ylabel(r'$\bar{V}(Q)$')
ax.set_title('Effect of Tighter Repudiation Constraint on Welfare')
ax.legend()
plt.tight_layout()
plt.show()
```

Tightening the no-repudiation threshold ($\varepsilon > 0$) shrinks the set of
feasible contracts, reducing $\bar{V}(Q)$.
Debt forgiveness improves the
borrower's outside option but makes lenders less willing to extend credit
(smaller loans at higher cost), leaving the borrower worse off in equilibrium.
This illustrates the {cite}`BulowRogoff1989b` result that debt forgiveness
need not benefit the borrowing country.
````

