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

(gorman_heterogeneous_households)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

```{index} single: python
```

# Gorman Heterogeneous Households and Limited Markets

This lecture implements the Gorman heterogeneous-household economy in Chapter 12 of {cite:t}`HS2013` using the `quantecon.DLE` class.

It complements [](hs_recursive_models), [](growth_in_dles), and [](irfs_in_hall_model) by focusing on how to recover household allocations and portfolios from an aggregate DLE solution.

The headline result is that a complete-markets allocation can be implemented with a mutual fund and a one-period bond when Gorman aggregation holds.

This lecture does three things.

- It solves the aggregate planner problem with `DLE`.
- It computes household weights and deviation terms.
- It simulates a limited-markets portfolio strategy that replicates the Arrow–Debreu allocation.

In addition to what's in Anaconda, this lecture uses the quantecon library.

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade quantecon
```

We make the following imports.

```{code-cell} ipython3
import numpy as np
from scipy.linalg import solve_discrete_are
from quantecon import DLE
from quantecon._lqcontrol import LQ
import matplotlib.pyplot as plt
```

## Overview

Gorman aggregation lets us solve heterogeneous-household economies in two steps.

First, solve a representative-agent linear-quadratic planning problem to obtain aggregate allocations, shadow prices, and the closed-loop law of motion.

Second, recover household-level allocations via a wealth-share rule with a household-specific deviation term.

The idea behind Gorman aggregation is beautifully simple yet powerful.

In a general heterogeneous-consumer economy, the set of Pareto optimal allocations depends on how utility is distributed across consumers. 

If we change the aggregate endowment, the new utility possibility frontier may cross the old one, meaning there is no way to rank aggregate allocations independently of distribution. 

Gorman's conditions eliminate this problem: they ensure that all consumers have parallel Engel curves, so the coefficient on wealth in the demand function is the same for everyone. 

This lets us determine aggregate allocations and prices before worrying about how to divide consumption among households.

With the help of this powerful result, we proceed in three steps in this lecture:

1. Solve the planner's problem and compute selection matrices that map the aggregate state into allocations and prices.
2. Compute household-specific policies and the Gorman sharing rule.
3. Implement the same Arrow-Debreu allocation using only a mutual fund (aggregate stock) and a one-period bond.

We then simulate examples with two and many households.

### Notation

Time is discrete, $t = 0,1,2,\dots$.

Households are indexed by $j$ and we use $N$ for the number of simulated households.

#### Exogenous state

The exogenous state $z_t$ follows a first-order vector autoregression

$$
z_{t+1} = A_{22} z_t + C_2 w_{t+1},
$$

where $A_{22}$ governs persistence and $C_2$ maps i.i.d. shocks $w_{t+1}$ into the state.

The vector $z_t$ typically contains three types of components.

1. Constant: The first element is set to 1 and remains constant.

2. Aggregate shocks: Components with persistent dynamics (nonzero entries in $A_{22}$) that affect all households. 

    - In the examples in {ref}`gorman_twohh`, an AR(2) process drives aggregate endowment fluctuations.

3. Idiosyncratic shocks: Components with transitory or zero persistence that enter individual endowments with loadings summing to zero across households. 

    - These generate cross-sectional heterogeneity while preserving aggregate resource constraints.

The selection matrices $U_b$ and $U_d$ pick out which components of $z_t$ affect 
household preferences (bliss points) and endowments.

#### Aggregate planner state

The aggregate planner state stacks lagged endogenous stocks and current exogenous variables:

$$
x_t = [h_{t-1}^\top, k_{t-1}^\top, z_t^\top]^\top.
$$

Here $h_{t-1}$ is the lagged household durable stock (habits or durables affecting utility), $k_{t-1}$ is lagged physical capital, and $z_t$ is the current exogenous state. 

Together, $x_t$ contains everything the planner needs to make decisions at time $t$.

#### Aggregates

Aggregates are economy-wide totals summed across households: consumption $c_t$, investment $i_t$, capital $k_t$, household stock $h_t$, service flow $s_t$, intermediate good $g_t$, bliss $b_t$, and endowment $d_t$.

#### Gorman weight

The Gorman weight $\mu_j := u_j/u_a$ is household $j$'s wealth share, where $u_j$ is household $j$'s marginal utility of wealth and $u_a$ is the aggregate. 

The weights sum to one: $\sum_j \mu_j = 1$.

Under the Gorman sharing rule, household consumption is

$$
c_{jt} = \mu_j c_t + \tilde{\chi}_{jt}.
$$

The proportional term $\mu_j c_t$ is household $j$'s share of aggregate consumption.

The deviation term $\tilde{\chi}_{jt}$ captures how household $j$'s consumption differs from its proportional share due to preference and endowment heterogeneity. 

These deviations sum to zero across households: $\sum_j \tilde{\chi}_{jt} = 0$.

#### Technologies

$$
\Phi_c c_t + \Phi_g g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t, \quad
k_t = \Delta_k k_{t-1} + \Theta_k i_t,
$$

and

$$
h_t = \Delta_h h_{t-1} + \Theta_h c_t, \quad
s_t = \Lambda h_{t-1} + \Pi_h c_t, \quad
b_t = U_b z_t, \quad
d_t = U_d z_t.
$$

Selection matrices such as $S_c, S_k, \ldots$ satisfy $c_t = S_c x_t$.

Shadow-price mappings $M_c, M_k, \ldots$ are used to value streams and recover equilibrium prices.

### The individual household problem

Consider an economy with $J$ consumers indexed by $j = 1, 2, \ldots, J$.

Consumers differ in preferences and endowments but share a common information set.

Consumer $j$ maximizes

$$
-\frac{1}{2} \mathbb{E}_0 \sum_{t=0}^\infty \beta^t \left[(s_{jt} - b_{jt})^\top (s_{jt} - b_{jt}) + \ell_{jt}^\top \ell_{jt}\right]
$$ (eq:hh_objective)

subject to the household service technology

$$
\begin{aligned}
s_{jt} &= \Lambda h_{j,t-1} + \Pi_h c_{jt}, \\
h_{jt} &= \Delta_h h_{j,t-1} + \Theta_h c_{jt},
\end{aligned}
$$ (eq:hh_service_tech)

and the intertemporal budget constraint

$$
\mathbb{E}_0 \sum_{t=0}^\infty \beta^t p_{0t} \cdot c_{jt}
= \mathbb{E}_0 \sum_{t=0}^\infty \beta^t (w_{t0} \ell_{jt} + \alpha_{0t} \cdot d_{jt}) + v_0 \cdot k_{j,-1},
$$ (eq:hh_budget)

where $b_{jt} = U_b^j z_t$ is household $j$'s preference shock, $d_{jt} = U_d^j z_t$ is household $j$'s endowment stream, $h_{j,-1}$ and $k_{j,-1}$ are given initial household stocks, and $\ell_{jt}$ is household labor supply.

Prices $(p_{0t}, w_{t0}, \alpha_{0t}, v_0)$ are determined in equilibrium.

Three key features structure the example in this lecture:

1. All households share the same technology matrices $(\Lambda, \Pi_h, \Delta_h, \Theta_h)$.

2. Heterogeneity enters only through household-specific preference and endowment parameters $(U_b^j, U_d^j, h_{j,-1}, k_{j,-1})$.

3. All households observe the same aggregate information $\mathcal{J}_t = [w_t, x_0]$.

These restrictions enable Gorman aggregation by ensuring that household demands are affine in wealth.

### From individual problems to the aggregate problem

The Gorman aggregation conditions establish a powerful equivalence: the competitive equilibrium allocation solves a social planner's problem with a specific set of Pareto weights.

We can solve for aggregate quantities by maximizing a weighted sum of household utilities rather than solving each household's problem separately.

Equilibrium prices emerge naturally as the shadow prices from the planner's problem.

When preferences satisfy Gorman's restrictions, the equilibrium allocation solves a planning problem with Pareto weights given by the household wealth multipliers $\mu_{0j}^w$.

The problem decomposes into two steps:

1. In the aggregate step, we solve the planner's problem with a representative consumer by summing across all households.

2. In the allocation step, we use the sharing rule to distribute aggregate consumption to individual households based on their wealth shares and preference shocks.

### The aggregate planning problem

Summing the household budget constraints across $j = 1,\ldots,J$ gives

$$
\mathbb{E}_0 \sum_{t=0}^\infty \beta^t p_{0t} \cdot \left(\sum_j c_{jt}\right)
= \mathbb{E}_0 \sum_{t=0}^\infty \beta^t \left(w_{t0} \sum_j \ell_{jt} + \alpha_{0t} \cdot \sum_j d_{jt}\right) + v_0 \cdot \sum_j k_{j,-1}.
$$ (eq:agg_budget_sum)

In equilibrium, aggregate consumption $c_t = \sum_j c_{jt}$ must satisfy the economy's resource constraint

$$
\Phi_c c_t + \Phi_g g_t + \Phi_i i_t = \Gamma k_{t-1} + d_t,
$$ (eq:agg_resource_constraint)

where aggregate endowment is $d_t = \sum_j d_{jt} = U_d z_t$ with $U_d = \sum_j U_d^j$.

We construct the representative consumer's preferences by summing

$$
h_{-1} = \sum_j h_{j,-1}, \quad
k_{-1} = \sum_j k_{j,-1}, \quad
U_b = \sum_j U_b^j, \quad
U_d = \sum_j U_d^j.
$$ (eq:agg_preference_aggregates)

With these aggregates, the planner maximizes

$$
-\frac{1}{2} \mathbb{E}_0 \sum_{t=0}^\infty \beta^t
\left[(s_t - b_t)^\top(s_t - b_t) + g_t^\top g_t\right]
$$ (eq:planner_objective)

subject to technology constraints

$$
\begin{aligned}
\Phi_c c_t + \Phi_g g_t + \Phi_i i_t &= \Gamma k_{t-1} + d_t, \\
k_t &= \Delta_k k_{t-1} + \Theta_k i_t, \\
h_t &= \Delta_h h_{t-1} + \Theta_h c_t, \\
s_t &= \Lambda h_{t-1} + \Pi_h c_t,
\end{aligned}
$$ (eq:planner_constraints)

and exogenous processes

$$
z_{t+1} = A_{22} z_t + C_2 w_{t+1}, \quad
b_t = U_b z_t, \quad
d_t = U_d z_t.
$$ (eq:exogenous_process)

Solving this aggregate planning problem yields aggregate allocations $(c_t, i_t, k_t, h_t, s_t, g_t)$ as functions of the aggregate state $x_t$.

The first-order conditions give shadow prices $(M^c_t, M^k_t, M^h_t, M^s_t)$ associated with each constraint.

These shadow prices correspond to competitive equilibrium prices.

## Helper routines

We repeatedly use matrix sums of the form $\sum_{t \ge 0} A_1^t B_1 B_2^\top (A_2^\top)^t$.

The following helper computes this sum with a doubling algorithm.

```{code-cell} ipython3
def doublej2(A1, B1, A2, B2, tol=1e-15, max_iter=10_000):
    r"""
    Compute V = Σ_{t=0}^∞ A1^t B1 B2' (A2')^t via a doubling algorithm.
    """
    A1 = np.asarray(A1, dtype=float)
    A2 = np.asarray(A2, dtype=float)
    B1 = np.asarray(B1, dtype=float)
    B2 = np.asarray(B2, dtype=float)

    α1, α2 = A1.copy(), A2.copy()
    V = B1 @ B2.T
    diff, it = np.inf, 0

    while diff > tol and it < max_iter:
        α1_next = α1 @ α1
        α2_next = α2 @ α2
        V_next = V + α1 @ V @ α2.T

        diff = np.max(np.abs(V_next - V))
        α1, α2, V = α1_next, α2_next, V_next
        it += 1

    return V
```

## Household allocations

Under the Gorman sharing rule, household $j$'s consumption is

$$
c_{jt} = \mu_j c_t + \tilde{\chi}_{jt}, \qquad \mu_j := u_j/u_a.
$$ (eq:sharing_rule)

The proportional term $\mu_j c_t$ is household $j$'s share of aggregate consumption.

The deviation term $\tilde{\chi}_{jt}$ captures deviations driven by preference and endowment heterogeneity.

### Algorithm for computing $\mu_j$ and $\tilde{\chi}_{jt}$

Computing household allocations requires three steps.

**Step 1: Solve the household deviation problem.**

The "deviation" consumption $\tilde{\chi}_{jt}$ satisfies the inverse canonical representation

$$
\begin{aligned}
\tilde{\chi}_{jt} &= -\Pi_h^{-1} \Lambda \tilde{\eta}_{j,t-1} + \Pi_h^{-1} \tilde{b}_{jt}, \\
\tilde{\eta}_{jt} &= (\Delta_h - \Theta_h \Pi_h^{-1} \Lambda) \tilde{\eta}_{j,t-1} + \Theta_h \Pi_h^{-1} \tilde{b}_{jt},
\end{aligned}
$$

where $\tilde{b}_{jt} = b_{jt} - \mu_j b_t$ is the deviation of household $j$'s bliss point from its share of the aggregate, and $\tilde{\eta}_{j,-1} = h_{j,-1} - \mu_j h_{-1}$.

This recursion comes from inverting the household service technology. 

When preferences include durables or habits ($\Lambda \neq 0$), the deviation consumption depends on the lagged deviation state $\tilde{\eta}_{j,t-1}$.

The code solves this as a linear-quadratic control problem using a scaling trick: multiplying the transition matrices by $\sqrt{\beta}$ converts the discounted problem into an undiscounted one that can be solved with a standard discrete algebraic Riccati equation.

**Step 2: Compute present values.**

The Gorman weight $\mu_j$ is determined by the household's budget constraint. 

We compute four present values:

- $W_d$: present value of household $j$'s endowment stream $\{d_{jt}\}$
- $W_k$: value of household $j$'s initial capital $k_{j,-1}$
- $W_{c1}$: present value of the "unit" consumption stream (what it costs to consume $c_t$)
- $W_{c2}$: present value of the deviation consumption stream $\{\tilde{\chi}_{jt}\}$

Each present value is computed using `doublej2` to sum infinite series of the form $\sum_{t \ge 0} \beta^t M_t S_t'$, where $M_t$ is a shadow price and $S_t$ is a selection matrix.

**Step 3: Solve for the Gorman weight.**

The budget constraint pins down $\mu_j$:

$$
\mu_j = \frac{W_k + W_d - W_{c2}}{W_{c1} - W_g}.
$$

The numerator is household $j$'s wealth (initial capital plus present value of endowments minus the cost of the deviation consumption stream). 

The denominator is the net cost of consuming one unit of aggregate consumption (consumption value minus intermediate good value).

**Step 4: Build selection matrices.**

Finally, the code constructs selection matrices $S_{ci}, S_{hi}, S_{si}$ that map the augmented state $X_t = [h_{j,t-1}^\top, x_t^\top]^\top$ into household $j$'s allocations:

$$
c_{jt} = S_{ci} X_t, \quad h_{jt} = S_{hi} X_t, \quad s_{jt} = S_{si} X_t.
$$

The augmented state includes household $j$'s own lagged durable stock $h_{j,t-1}$ because the deviation term $\tilde{\chi}_{jt}$ depends on it through the inverse canonical representation.

```{code-cell} ipython3
def heter(
    Λ, Θ_h, Δ_h, Π_h, β,
    Θ_k, Δ_k, A_22, C_2,
    U_bi, U_di,
    A0, C, M_d, M_g, M_k, Γ, M_c,
    x0, h0i, k0i,
    S_h, S_k, S_i, S_g, S_d, S_b, S_c, S_s,
    M_h, M_s, M_i,
    tol=1e-15,
):
    """
    Compute household i selection matrices, the Gorman weight μ_i,
    and valuation objects.
    """
    # Dimensions
    n_s, n_h = np.asarray(Λ).shape
    _, n_c = np.asarray(Θ_h).shape
    n_z, n_w = np.asarray(C_2).shape
    n_k, _ = np.asarray(Θ_k).shape
    n_d, _ = np.asarray(U_di).shape
    n_x = n_h + n_k + n_z

    β = float(np.asarray(β).squeeze())

    # Household deviation problem (Chapter 3 scaling trick)
    A = np.asarray(Δ_h, dtype=float) * np.sqrt(β)
    B = np.asarray(Θ_h, dtype=float) * np.sqrt(β)

    Λ = np.asarray(Λ, dtype=float)
    Π_h = np.asarray(Π_h, dtype=float)
    U_bi = np.asarray(U_bi, dtype=float)

    R_hh = Λ.T @ Λ + tol * np.eye(n_h)
    Q_hh = Π_h.T @ Π_h
    S_hh = Π_h.T @ Λ

    if np.linalg.matrix_rank(Q_hh) < max(Q_hh.shape):
        raise ValueError("The Π_h'Π_h block is singular.")

    Q_inv = np.linalg.inv(Q_hh)

    # Eliminate cross term
    A = A - B @ Q_inv @ S_hh
    R_hh = R_hh - S_hh.T @ Q_inv @ S_hh

    V11 = solve_discrete_are(A, B, R_hh, Q_hh)
    Λ_s = np.linalg.solve(Q_hh + B.T @ V11 @ B, B.T @ V11 @ A)

    A0_dev = A - B @ Λ_s
    Λ_s = Λ_s + Q_inv @ S_hh

    # Feedforward for U_bi z_t
    A12 = -B @ Q_inv @ Π_h.T @ U_bi
    R12 = Λ.T @ U_bi - Λ.T @ Π_h @ Q_inv @ Π_h.T @ U_bi + A0_dev.T @ V11 @ A12

    V12 = doublej2(A0_dev.T, R12, A_22, np.eye(n_z))
    Q_opt = np.linalg.inv(Q_hh + B.T @ V11 @ B)
    U_b_s = Q_opt @ B.T @ (V12 @ A_22 + V11 @ A12) + Q_inv @ Π_h.T @ U_bi

    # Undo √β scaling
    A0_dev = A0_dev / np.sqrt(β)

    # Long-run covariance under aggregate law of motion
    V_mat = doublej2(β * A0, C, A0, C) * β / (1 - β)

    # Present value of household endowment stream
    U_di = np.asarray(U_di, dtype=float)
    S_di = np.hstack((np.zeros((n_d, n_h + n_k)), U_di))
    W_d = doublej2(β * A0.T, M_d.T, A0.T, S_di.T)
    W_d = x0.T @ W_d @ x0 + np.trace(M_d @ V_mat @ S_di.T)

    # Present value of intermediate good
    W_g = doublej2(β * A0.T, M_g.T, A0.T, S_g.T)
    W_g = x0.T @ W_g @ x0 + np.trace(M_g @ V_mat @ S_g.T)

    # Shadow price object for durables
    M_hs = β * doublej2(β * A0_dev.T, Λ_s.T, A0.T, M_c.T) @ A0
    S_c1 = Θ_h.T @ M_hs - M_c

    # Augmented system for consumption valuation
    A_s = np.block([[A0_dev, Θ_h @ S_c1], [np.zeros((n_x, n_h)), A0]])
    C_s = np.vstack((np.zeros((n_h, n_w)), C))

    M_cs = np.hstack((np.zeros((M_c.shape[0], n_h)), M_c))
    S_cs = np.hstack((-Λ_s, S_c1))

    x0s = np.vstack((np.zeros((n_h, 1)), x0))

    W_c1 = doublej2(β * A_s.T, M_cs.T, A_s.T, S_cs.T)
    V_s = doublej2(β * A_s, C_s, A_s, C_s) * β / (1 - β)
    W_c1 = float((x0s.T @ W_c1 @ x0s + np.trace(M_cs @ V_s @ S_cs.T)).squeeze())

    S_c2 = np.hstack((np.zeros((M_c.shape[0], n_h + n_k)), U_b_s))
    A_s2 = np.block([[A0_dev, Θ_h @ S_c2], [np.zeros((n_x, n_h)), A0]])
    S_cs2 = np.hstack((-Λ_s, S_c2))
    x0s2 = np.vstack((h0i, x0))

    W_c2 = doublej2(β * A_s2.T, M_cs.T, A_s2.T, S_cs2.T)
    V_s2 = doublej2(β * A_s2, C_s, A_s2, C_s) * β / (1 - β)
    W_c2 = float((x0s2.T @ W_c2 @ x0s2 + np.trace(M_cs @ V_s2 @ S_cs2.T)).squeeze())

    # Present value of initial capital
    W_k = float((k0i.T @ (
        np.asarray(Δ_k).T @ M_k +
        np.asarray(Γ).T @ M_d) @ x0).squeeze())

    μ = float(((W_k + W_d - W_c2) / (W_c1 - W_g)).squeeze())

    # Household selection matrices on augmented state X_t = [h^i_{t-1}, x_t]
    S_cs = μ * S_c1 + S_c2
    A0_i = np.block([[A0_dev, Θ_h @ S_cs], [np.zeros((n_x, n_h)), A0]])
    S_ci = np.hstack((-Λ_s, S_cs))
    S_hi = A0_i[:n_h, :n_h + n_x]
    S_si = np.hstack((Λ, np.zeros((n_s, n_x)))) + Π_h @ S_ci
    S_bi = np.hstack((np.zeros((n_s, 2 * n_h + n_k)), U_bi))
    S_di = np.hstack((np.zeros((n_d, 2 * n_h + n_k)), U_di))

    # Embed aggregate selection and pricing objects on X_t
    S_ha = np.hstack((np.zeros((n_h, n_h)), S_h))
    S_ka = np.hstack((np.zeros((n_k, n_h)), S_k))
    S_ia = np.hstack((np.zeros((S_i.shape[0], n_h)), S_i))
    S_ga = np.hstack((np.zeros((S_g.shape[0], n_h)), S_g))
    S_da = np.hstack((np.zeros((n_d, n_h)), S_d))
    S_ba = np.hstack((np.zeros((n_s, n_h)), S_b))
    S_ca = np.hstack((np.zeros((S_c.shape[0], n_h)), S_c))
    S_sa = np.hstack((np.zeros((n_s, n_h)), S_s))

    M_ha = np.hstack((np.zeros((M_h.shape[0], n_h)), M_h))
    M_ka = np.hstack((np.zeros((M_k.shape[0], n_h)), M_k))
    M_sa = np.hstack((np.zeros((M_s.shape[0], n_h)), M_s))
    M_ga = np.hstack((np.zeros((M_g.shape[0], n_h)), M_g))
    M_ia = np.hstack((np.zeros((M_i.shape[0], n_h)), M_i))
    M_da = np.hstack((np.zeros((M_d.shape[0], n_h)), M_d))
    M_ca = M_cs

    return {
        "μ": μ,
        "S_ci": S_ci,
        "S_hi": S_hi,
        "S_si": S_si,
        "S_bi": S_bi,
        "S_di": S_di,
        "S_ha": S_ha,
        "S_ka": S_ka,
        "S_ia": S_ia,
        "S_ga": S_ga,
        "S_da": S_da,
        "S_ba": S_ba,
        "S_ca": S_ca,
        "S_sa": S_sa,
        "M_ha": M_ha,
        "M_ka": M_ka,
        "M_sa": M_sa,
        "M_ga": M_ga,
        "M_ia": M_ia,
        "M_da": M_da,
        "M_ca": M_ca,
        "A0_i": A0_i,
        "C_s": C_s,
        "x0s": np.vstack((h0i, x0)),
    }
```

The book sets the intermediate-good multiplier to $M_g = S_g$, so we pass `econ.Sg` for `M_g`.

## Risk sharing implications

The Gorman sharing rule has strong implications for risk sharing across households.

Because the weight $\mu_j$ is time-invariant and determined at date zero, the deviation process $\{c_{jt} - \tilde{\chi}_{jt}\}$ exhibits perfect risk pooling: all households share aggregate consumption risk in fixed proportions.

Non-separabilities in preferences (over time or across goods) affect only the baseline process $\{\tilde{\chi}_{jt}\}$ and the calculation of the risk-sharing coefficient $\mu_j$. They do not break the proportional sharing of aggregate risk.

In the special case where the preference shock processes $\{b_{jt}\}$ are deterministic (known at time zero), individual consumption is perfectly correlated with aggregate consumption conditional on initial information. 

The figures we will be showing below confirm this: household consumption paths track aggregate consumption closely.

## Limited markets

This section implements a key result: the Arrow-Debreu allocation can be replicated using only a mutual fund and a one-period bond, rather than a complete set of contingent claims.

### The trading mechanism

Consider an economy where each household $j$ initially owns claims to its own endowment stream $\{d_{jt}\}$. Instead of trading in a complete set of Arrow-Debreu markets, we open:

1. A stock market with securities paying dividends $\{d_{jt}\}$ for each endowment process
2. A market for one-period riskless bonds

At time zero, household $j$ executes the following trades:

1. **Sell individual claims**: Household $j$ sells all shares of its own endowment security
2. **Buy the mutual fund**: Purchase $\mu_j$ shares of all securities (equivalently, $\mu_j$ shares of a mutual fund holding the aggregate endowment)
3. **Adjust bond position**: Take position $\hat{k}_{j0}$ in the one-period bond

After this initial rebalancing, household $j$ holds the portfolio forever.

### Why $\mu_j$ shares?

The portfolio weight $\mu_j$ is not arbitrary — it is the unique weight that allows the limited-markets portfolio to replicate the Arrow-Debreu consumption allocation.

**The requirement.** Under Gorman aggregation, household $j$'s equilibrium consumption satisfies

$$
c_{jt} = \mu_j c_t + \tilde{\chi}_{jt}.
$$

We need a portfolio strategy that delivers exactly this consumption stream.

**The mutual fund.** The mutual fund holds claims to all individual endowment streams. Total dividends paid by the fund each period are

$$
\sum_{i=1}^J d_{it} = d_t,
$$

the aggregate endowment. If household $j$ holds fraction $\theta_j$ of this fund, it receives $\theta_j d_t$ in dividends.

**Matching the proportional term.** The proportional part of consumption $\mu_j c_t$ must be financed by the mutual fund and capital holdings. Since the aggregate resource constraint is

$$
c_t + i_t = (\delta_k + \gamma_1) k_{t-1} + d_t,
$$

holding $\theta_j$ shares of aggregate output (capital income plus endowments) delivers $\theta_j [(\delta_k + \gamma_1)k_{t-1} + d_t]$. For this to finance $\mu_j c_t$ plus reinvestment $\mu_j k_t$, we need $\theta_j = \mu_j$.

**The deviation term.** The remaining term $\tilde{\chi}_{jt}$ is financed by adjusting the bond position according to the recursion derived below.

**Conclusion.** Setting $\theta_j = \mu_j$ ensures that the mutual fund finances exactly the proportional share $\mu_j c_t$, while the bond handles the deviation $\tilde{\chi}_{jt}$. This transforms heterogeneous endowment risk into proportional shares of aggregate risk.

### Derivation of the bond recursion

We now derive the law of motion for the bond position $\hat{k}_{jt}$.

**Step 1: Write the budget constraint.**

Household $j$'s time-$t$ resources equal uses:

$$
\underbrace{\mu_j [(\delta_k + \gamma_1) k_{t-1} + d_t]}_{\text{mutual fund income}} + \underbrace{R \hat{k}_{j,t-1}}_{\text{bond return}} = \underbrace{c_{jt}}_{\text{consumption}} + \underbrace{\mu_j k_t}_{\text{new fund shares}} + \underbrace{\hat{k}_{jt}}_{\text{new bonds}}
$$

where $R := \delta_k + \gamma_1$ is the gross return.

**Step 2: Substitute the sharing rule.**

Replace $c_{jt}$ with $\mu_j c_t + \tilde{\chi}_{jt}$:

$$
\mu_j [(\delta_k + \gamma_1) k_{t-1} + d_t] + R \hat{k}_{j,t-1} = \mu_j c_t + \tilde{\chi}_{jt} + \mu_j k_t + \hat{k}_{jt}
$$

**Step 3: Use the aggregate resource constraint.**

The aggregate economy satisfies $c_t + k_t = (\delta_k + \gamma_1) k_{t-1} + d_t$, so:

$$
(\delta_k + \gamma_1) k_{t-1} + d_t = c_t + k_t
$$

Substituting into the left-hand side:

$$
\mu_j (c_t + k_t) + R \hat{k}_{j,t-1} = \mu_j c_t + \tilde{\chi}_{jt} + \mu_j k_t + \hat{k}_{jt}
$$

**Step 4: Simplify.**

The $\mu_j c_t$ and $\mu_j k_t$ terms cancel:

$$
R \hat{k}_{j,t-1} = \tilde{\chi}_{jt} + \hat{k}_{jt}
$$

Rearranging gives the bond recursion:

$$
\hat{k}_{jt} = R \hat{k}_{j,t-1} - \tilde{\chi}_{jt}.
$$ (eq:bond-recursion)

This says that the bond position grows at rate $R$ but is drawn down by the deviation consumption $\tilde{\chi}_{jt}$. When $\tilde{\chi}_{jt} > 0$ (household $j$ consumes more than its share), it finances this by running down its bond holdings.

### Initial bond position

To find $\hat{k}_{j0}$, we solve the recursion {eq}`eq:bond-recursion` forward.

**Step 1: Iterate forward.**

From $\hat{k}_{jt} = R \hat{k}_{j,t-1} - \tilde{\chi}_{jt}$, we get:

$$
\begin{aligned}
\hat{k}_{j1} &= R \hat{k}_{j0} - \tilde{\chi}_{j1} \\
\hat{k}_{j2} &= R \hat{k}_{j1} - \tilde{\chi}_{j2} = R^2 \hat{k}_{j0} - R\tilde{\chi}_{j1} - \tilde{\chi}_{j2} \\
&\vdots \\
\hat{k}_{jT} &= R^T \hat{k}_{j0} - \sum_{t=1}^T R^{T-t} \tilde{\chi}_{jt}
\end{aligned}
$$

**Step 2: Apply transversality.**

For the budget constraint to hold with equality, we need $\lim_{T \to \infty} R^{-T} \hat{k}_{jT} = 0$. Dividing by $R^T$:

$$
R^{-T} \hat{k}_{jT} = \hat{k}_{j0} - \sum_{t=1}^T R^{-t} \tilde{\chi}_{jt}
$$

Taking $T \to \infty$ and using transversality:

$$
0 = \hat{k}_{j0} - \sum_{t=1}^\infty R^{-t} \tilde{\chi}_{jt}
$$

**Step 3: Solve for initial position.**

$$
\hat{k}_{j0} = \sum_{t=1}^\infty R^{-t} \tilde{\chi}_{jt}.
$$

This is the present value of future deviation consumption. Since $\tilde{\chi}_{jt}$ depends only on the deterministic deviation between household $j$'s preference shocks and its share of aggregate preference shocks, this sum is in the time-zero information set and can be computed at date zero.

The household's total asset position is $a_{jt} = \mu_j k_t + \hat{k}_{jt}$.

```{code-cell} ipython3
def compute_household_paths(econ, U_b_list, U_d_list, x0, x_path, γ_1, Λ, h0i=None, k0i=None):
    """
    Compute household allocations and limited-markets portfolios
    along a fixed aggregate path.
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1, 1)
    x_path = np.asarray(x_path, dtype=float)

    Θ_h = np.atleast_2d(econ.thetah)
    Δ_h = np.atleast_2d(econ.deltah)
    Π_h = np.atleast_2d(econ.pih)
    Λ = np.atleast_2d(Λ)

    n_h = Θ_h.shape[0]
    n_k = np.atleast_2d(econ.thetak).shape[0]

    z_path = x_path[n_h + n_k:, :]

    c = econ.Sc @ x_path
    k = econ.Sk @ x_path
    b = econ.Sb @ x_path
    d = econ.Sd @ x_path

    δ_k = float(np.asarray(econ.deltak).squeeze())
    R = δ_k + float(γ_1)

    Π_inv = np.linalg.inv(Π_h)
    A_h = Δ_h - Θ_h @ Π_inv @ Λ
    B_h = Θ_h @ Π_inv

    if h0i is None:
        h0i = np.zeros((n_h, 1))
    if k0i is None:
        k0i = np.zeros((n_k, 1))

    N = len(U_b_list)
    _, T = c.shape

    μ = np.empty(N)
    χ_tilde = np.zeros((N, T))
    c_j = np.zeros((N, T))
    d_j = np.zeros((N, T))
    d_share = np.zeros((N, T))  # Dividend income under limited markets
    k_share = np.zeros((N, T))
    k_hat = np.zeros((N, T))
    a_total = np.zeros((N, T))

    for j in range(N):
        U_bj = np.asarray(U_b_list[j], dtype=float)
        U_dj = np.asarray(U_d_list[j], dtype=float)

        res = heter(
            econ.llambda,
            econ.thetah,
            econ.deltah,
            econ.pih,
            econ.beta,
            econ.thetak,
            econ.deltak,
            econ.a22,
            econ.c2,
            U_bj,
            U_dj,
            econ.A0,
            econ.C,
            econ.Md,
            econ.Sg,
            econ.Mk,
            econ.gamma,
            econ.Mc,
            x0,
            h0i,
            k0i,
            econ.Sh,
            econ.Sk,
            econ.Si,
            econ.Sg,
            econ.Sd,
            econ.Sb,
            econ.Sc,
            econ.Ss,
            econ.Mh,
            econ.Ms,
            econ.Mi,
        )
        μ[j] = res["μ"]

        b_tilde = U_bj @ z_path - μ[j] * b

        η = np.zeros((n_h, T + 1))
        η[:, 0] = np.asarray(h0i).reshape(-1)
        for t in range(1, T):
            χ_tilde[j, t] = (-Π_inv @ Λ @ η[:, t - 1] + Π_inv @ b_tilde[:, t]).squeeze()
            η[:, t] = (A_h @ η[:, t - 1] + B_h @ b_tilde[:, t]).squeeze()

        c_j[j] = (μ[j] * c[0] + χ_tilde[j]).squeeze()
        # Original endowment claim (before trading)
        d_j[j] = (U_dj @ z_path)[0, :]
        # Dividend income under limited markets: μ_j × aggregate endowment
        d_share[j] = (μ[j] * d[0]).squeeze()
        # Capital share in mutual fund
        k_share[j] = (μ[j] * k[0]).squeeze()

        if abs(R - 1.0) >= 1e-14:
            k_hat[j, -1] = χ_tilde[j, -1] / (R - 1.0)
            for t in range(T - 1, 0, -1):
                k_hat[j, t - 1] = (k_hat[j, t] + χ_tilde[j, t]) / R

        a_total[j] = k_share[j] + k_hat[j]

    return {
        "μ": μ,
        "χ_tilde": χ_tilde,
        "c": c,
        "k": k,
        "d": d,
        "c_j": c_j,
        "d_j": d_j,
        "d_share": d_share,  # μ_j × d_t: dividend income under limited markets
        "k_share": k_share,
        "k_hat": k_hat,
        "a_total": a_total,
        "x_path": x_path,
        "z_path": z_path,
        "R": R,
    }
```

```{code-cell} ipython3
def simulate_state_path(A0, C, x0, ts_length, rng=None):
    """
    Simulate x_{t+1} = A0 x_t + C w_{t+1} with iid standard normal shocks.
    """
    rng = np.random.default_rng(rng)
    x0 = np.asarray(x0, dtype=float).reshape(-1, 1)
    n_x = x0.shape[0]
    n_w = np.asarray(C).shape[1]

    x = np.empty((n_x, ts_length))
    w = np.empty((n_w, ts_length))
    x[:, 0] = x0[:, 0]
    w[:, 0] = 0.0

    for t in range(ts_length - 1):
        w[:, t + 1] = rng.standard_normal(n_w)
        x[:, t + 1] = (A0 @ x[:, t] + C @ w[:, t + 1]).reshape(-1)

    return x, w
```

(gorman_twohh)=
## Example: two-household economy

We reproduce the two-household Hall-style calibration from Chapter 12.6 of {cite:t}`HS2013`.

```{code-cell} ipython3
ϕ_1 = 1e-5
γ_1 = 0.1
δ_k = 0.95
β = 1.0 / (γ_1 + δ_k)

θ_k = 1.0
δ_h = 0.2
θ_h = 0.1
Λ = 0.0
Π_h = 1.0

Φ_c = np.array([[1.0], [0.0]])
Φ_g = np.array([[0.0], [1.0]])
Φ_i = np.array([[1.0], [-ϕ_1]])
Γ = np.array([[γ_1], [0.0]])

A_22 = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.2, -0.22, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
], dtype=float)

C_2 = np.array([
    [0.0, 0.0],
    [0.0, 0.25],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
], dtype=float)

U_b1 = np.array([[15.0, 0.0, 0.0, 0.0, 0.0]])
U_b2 = np.array([[15.0, 0.0, 0.0, 0.0, 0.0]])
U_b = U_b1 + U_b2

U_d1 = np.array([[4.0, 0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
U_d2 = np.array([[3.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
U_d = U_d1 + U_d2

info = (A_22, C_2, U_b, U_d)
tech = (Φ_c, Φ_g, Φ_i, Γ, np.array([[δ_k]]), np.array([[θ_k]]))
pref = (np.array([[β]]),
        np.array([[Λ]]),
        np.array([[Π_h]]),
        np.array([[δ_h]]),
        np.array([[θ_h]]))

econ = DLE(info, tech, pref)
```

We simulate the closed-loop DLE state and compute household paths.

```{code-cell} ipython3
x0 = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]).T
ts_length = 2_000

x_path, _ = simulate_state_path(econ.A0, econ.C, x0, ts_length, rng=1234)

paths = compute_household_paths(
    econ=econ,
    U_b_list=[U_b1, U_b2],
    U_d_list=[U_d1, U_d2],
    x0=x0,
    x_path=x_path,
    γ_1=γ_1,
    Λ=Λ,
)

paths["μ"]
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: consumption paths
    name: fig-gorman-consumption
---
T_plot = 250
fig, ax = plt.subplots()
ax.plot(paths["c"][0, :T_plot], lw=2, label="aggregate")
ax.plot(paths["c_j"][0, :T_plot], lw=2, label="household 1")
ax.plot(paths["c_j"][1, :T_plot], lw=2, label="household 2")
ax.set_xlabel("time")
ax.set_ylabel("consumption")
ax.legend()
plt.show()
```

The next figure plots the limited-markets bond adjustment and confirms that the adjustments sum to approximately zero.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: bond adjustment positions
    name: fig-gorman-bond-adjustment
---
fig, ax = plt.subplots()
ax.plot(paths["k_hat"][0, :T_plot], lw=2, label="household 1")
ax.plot(paths["k_hat"][1, :T_plot], lw=2, label="household 2")
ax.plot(paths["k_hat"][:, :T_plot].sum(axis=0), lw=2, label="sum")
ax.axhline(0.0, color="k", lw=1, alpha=0.5)
ax.set_xlabel("time")
ax.set_ylabel("bond position")
ax.legend()
plt.show()
```

```{code-cell} ipython3
np.max(np.abs(paths["k_hat"].sum(axis=0)))
```


## Many-Household Gorman Economies

This section presents a scalable Gorman economy specification that accommodates many households with heterogeneous endowments and preferences.

### Specification

The exogenous state vector is

$$
z_t = \begin{bmatrix} 1 \\ d_{a,t} \\ d_{a,t-1} \\ w_{2,t} \\ \vdots \\ w_{n,t} \\ w_{n+1,t} \\ \vdots \\ w_{2n,t} \end{bmatrix},
$$

where the first three components track a constant and the "ghost" aggregate endowment process, and the remaining components are i.i.d. shocks for individual endowments ($w_{2,t}, \ldots, w_{n,t}$) and preference shocks ($w_{n+1,t}, \ldots, w_{2n,t}$).

The aggregate endowment follows an AR(2) process:

$$
d_{a,t} = \rho_1 d_{a,t-1} + \rho_2 d_{a,t-2} + \sigma_a w_{1,t},
$$

where we can set the $\rho_j$'s to capture persistent aggregate fluctuations.

Individual endowments are:

$$
d_{jt} = \alpha_j + \phi_j d_{a,t} + \sigma_j w_{j,t}, \quad j = 2, \ldots, n,
$$

and for household 1 (which absorbs the negative of all idiosyncratic shocks to ensure aggregation):

$$
d_{1t} = \alpha_1 + \phi_1 d_{a,t} - \sum_{j=2}^n \sigma_j w_{j,t}.
$$

This construction ensures that

$$
\sum_{j=1}^n d_{j,t} = \sum_{j=1}^n \alpha_j + \left(\sum_{j=1}^n \phi_j\right) d_{a,t}.
$$

Imposing $\sum_{j=1}^n \phi_j = 1$ gives

$$
\sum_{j=1}^n d_{j,t} = \sum_{j=1}^n \alpha_j + d_{a,t}.
$$

Preference shocks are muted to simplify initial experiments:

$$
b_{jt} = \bar{b} + \gamma_j w_{n+j,t}, \quad j = 1, \ldots, n,
$$

where the $\gamma_j$'s are small.

```{code-cell} ipython3
def build_reverse_engineered_gorman_extended(
    n,
    rho1, rho2, sigma_a,
    alphas, phis, sigmas,
    b_bar, gammas,
    rho_idio=0.0,
    rho_pref=0.0,
    n_absorb=None,
):
    """
    Extended version that includes idiosyncratic shocks as state variables,
    allowing the full heterogeneous dynamics to be captured.

    The state vector is:
    z_t = [1, d_{a,t}, d_{a,t-1}, eta_{n_absorb+1,t}, ..., eta_{n,t}, xi_{1,t}, ..., xi_{n,t}]

    The first n_absorb households absorb the negative sum of all idiosyncratic shocks
    to ensure shocks sum to zero (Gorman requirement):
        sum_{j=1}^{n_absorb} (-1/n_absorb * sum_{k>n_absorb} eta_k) + sum_{k>n_absorb} eta_k = 0

    Each household k > n_absorb has its own idiosyncratic endowment shock eta_{k,t}
    following an AR(1):

        eta_{k,t+1} = rho_idio[k] * eta_{k,t} + sigma_k * w_{k,t+1}

    with rho_idio = 0 recovering i.i.d. shocks (AR(0)).

    Preference shocks xi_{j,t} are included for each household.

    Parameters
    ----------
    n_absorb : int, optional
        Number of households that absorb the idiosyncratic shocks.
        If None, defaults to max(1, n // 10) (10% of households, at least 1).
    """
    alphas = np.asarray(alphas).reshape(-1)
    phis = np.asarray(phis).reshape(-1)
    sigmas = np.asarray(sigmas).reshape(-1)
    gammas = np.asarray(gammas).reshape(-1)

    assert len(alphas) == len(phis) == len(sigmas) == len(gammas) == n

    # Default: 10% of households absorb shocks (at least 1)
    if n_absorb is None:
        n_absorb = max(1, n // 10)
    n_absorb = int(n_absorb)
    if n_absorb < 1 or n_absorb >= n:
        raise ValueError(f"n_absorb must be in [1, n-1], got {n_absorb} with n={n}")

    # State vector: z_t = [1, d_{a,t}, d_{a,t-1}, eta_{n_absorb+1,t}, ..., eta_{n,t}, xi_{1,t}, ..., xi_{n,t}]
    # where eta_{j,t} are idiosyncratic endowment shocks (j=n_absorb+1..n)
    # and xi_{j,t} are preference shocks (j=1..n)
    # First n_absorb households absorb -1/n_absorb * sum(eta_j) to ensure shocks sum to zero
    # Dimension: 3 + (n - n_absorb) + n = 2n + 3 - n_absorb

    n_idio = n - n_absorb       # eta_{n_absorb+1}, ..., eta_n
    n_pref = n                  # xi_1, ..., xi_n
    nz = 3 + n_idio + n_pref
    nw = 1 + n_idio + n_pref    # aggregate + idio + pref shocks

    # Persistence parameters (scalars broadcast to vectors)
    rho_idio = np.asarray(rho_idio, dtype=float)
    if rho_idio.ndim == 0:
        rho_idio = np.full(n_idio, float(rho_idio))
    if rho_idio.shape != (n_idio,):
        raise ValueError(f"rho_idio must be scalar or shape ({n_idio},), got {rho_idio.shape}")

    rho_pref = np.asarray(rho_pref, dtype=float)
    if rho_pref.ndim == 0:
        rho_pref = np.full(n_pref, float(rho_pref))
    if rho_pref.shape != (n_pref,):
        raise ValueError(f"rho_pref must be scalar or shape ({n_pref},), got {rho_pref.shape}")

    # A22: transition matrix
    A22 = np.zeros((nz, nz))
    A22[0, 0] = 1.0           # constant
    A22[1, 1] = rho1          # d_{a,t} AR(2) first coef
    A22[1, 2] = rho2          # d_{a,t} AR(2) second coef
    A22[2, 1] = 1.0           # lag transition
    for j in range(n_idio):
        A22[3 + j, 3 + j] = rho_idio[j]
    for j in range(n_pref):
        A22[3 + n_idio + j, 3 + n_idio + j] = rho_pref[j]

    # C2: shock loading
    C2 = np.zeros((nz, nw))
    C2[1, 0] = sigma_a                      # aggregate shock -> d_{a,t}
    for j in range(n_idio):
        # Map to households n_absorb+1, ..., n (indices n_absorb, ..., n-1 in 0-based)
        C2[3 + j, 1 + j] = sigmas[n_absorb + j]
    for j in range(n_pref):
        C2[3 + n_idio + j, 1 + n_idio + j] = gammas[j]  # gamma_j -> xi_j

    # Ud_per_house: endowment loading
    # First n_absorb households: d_{jt} = alpha_j + phi_j * d_{a,t} - (1/n_absorb) * sum_{k>n_absorb} eta_{k,t}
    # Remaining households k > n_absorb: d_{kt} = alpha_k + phi_k * d_{a,t} + eta_{k,t}
    Ud_per_house = []
    for j in range(n):
        block = np.zeros((2, nz))
        block[0, 0] = alphas[j]    # constant
        block[0, 1] = phis[j]      # loading on d_{a,t}

        if j < n_absorb:
            # Absorbing households: load -1/n_absorb on each idiosyncratic shock
            for k in range(n_idio):
                block[0, 3 + k] = -1.0 / n_absorb
        else:
            # Non-absorbing household j loads on its own shock eta_j
            # Household j (j >= n_absorb) corresponds to eta_{j+1} at position 3 + (j - n_absorb)
            block[0, 3 + (j - n_absorb)] = 1.0

        Ud_per_house.append(block)

    Ud = sum(Ud_per_house)

    # Ub_per_house: bliss loading
    # b_{jt} = b_bar + xi_{j,t}
    Ub_per_house = []
    for j in range(n):
        row = np.zeros((1, nz))
        row[0, 0] = b_bar                    # constant bliss
        row[0, 3 + n_idio + j] = 1.0         # loading on xi_j
        Ub_per_house.append(row)

    Ub = sum(Ub_per_house)

    # Initial state
    x0 = np.zeros((nz, 1))
    x0[0, 0] = 1.0  # constant = 1

    return A22, C2, Ub, Ud, Ub_per_house, Ud_per_house, x0
```

### Many-Household Example

We now instantiate a 100-household economy using the reverse-engineered Gorman specification.

```{code-cell} ipython3
np.random.seed(42)
N = 100

# AR(2) parameters for aggregate endowment
# rho1 = 1.3435
# rho2 = -0.9025
rho1 = 0.95
rho2 = 0.0

sigma_a = 0.5

# Individual endowment parameters
# alphas: intercepts (drawn randomly for each household)
alphas = np.random.uniform(3.0, 5.0, N)

# phis: loadings on aggregate endowment (must sum to 1)
phis_raw = np.random.uniform(0.5, 1.5, N)
phis = phis_raw / np.sum(phis_raw)  # normalize to sum to 1

wealth_rank_proxy = np.argsort(np.argsort(alphas))          # 0 = lowest alpha, N-1 = highest
wealth_pct_proxy = (wealth_rank_proxy + 0.5) / N            # in (0, 1)
poorness = 1.0 - wealth_pct_proxy                           # 1 = poorest, 0 = richest

# Number of households that absorb idiosyncratic shocks (Gorman requirement)
# Must be defined BEFORE creating rho_idio array
n_absorb = 50

# sigmas: idiosyncratic endowment shock scales
# Each household has a sigma, but only households n_absorb+1..N have their own eta state
sigma_idio_min, sigma_idio_max = 0.2, 5.0
sigmas = sigma_idio_min + (sigma_idio_max - sigma_idio_min) * (poorness ** 2.0)

# rho_idio: persistence for eta_{n_absorb+1..N}
# Only households n_absorb+1 through N have independent idiosyncratic shocks
rho_idio_min, rho_idio_max = 0.0, 0.98
rho_idio = rho_idio_min + (rho_idio_max - rho_idio_min) * (poorness[n_absorb:] ** 1.0)

# Preference parameters
b_bar = 5.0
# Idiosyncratic preference shocks (turn on to generate household-specific consumption movements)
enable_pref_shocks = False  # ENABLE to see time-varying consumption heterogeneity
pref_shock_scale = 0.5     # Increase scale to see clearer effects
pref_shock_persistence = 0.7

if enable_pref_shocks:
    gammas_pref = pref_shock_scale * np.ones(N)
    rho_pref = pref_shock_persistence
else:
    gammas_pref = np.zeros(N)   # mute preference shocks; shock endowment only
    rho_pref = 0.0              # keep preference states deterministic at 0

# Burn in for the aggregate and individual series
burn_in = 200

# Build the economy
A22, C2, Ub, Ud, Ub_list, Ud_list, x0 = build_reverse_engineered_gorman_extended(
    n=N,
    rho1=rho1, rho2=rho2, sigma_a=sigma_a,
    alphas=alphas, phis=phis, sigmas=sigmas,
    b_bar=b_bar, gammas=gammas_pref,
    rho_idio=rho_idio, rho_pref=rho_pref,
    n_absorb=n_absorb,
)

print(f"State dimension nz = {A22.shape[0]}")
print(f"Shock dimension nw = {C2.shape[1]}")
print(f"sum(phis) = {np.sum(phis):.6f} (should be 1.0)")
```

```{code-cell} ipython3
# Initialize economy via DLE for the reverse-engineered Gorman case.
info_ar1 = (A22, C2, Ub, Ud)
pref_ar1 = (np.array([[beta]]),
            np.array([[lam]]),
            np.array([[pih]]),
            np.array([[deltah]]),
            np.array([[thetah]]))
tech_ar1 = (phic, phig, phii, gamma,
            np.array([[deltak]]),
            np.array([[thetak]]))

paths, econ = solve_model(info_ar1, tech_ar1, pref_ar1,
                    Ub_list, Ud_list, gamma1, lam, z0=x0)
```

The following figure shows household-level consumption, endowments, and bond positions.

```{code-cell} ipython3
T = 50

plt.figure(figsize=(14, 8))

# Individual consumption paths
plt.subplot(2, 2, 1)
plt.title('Consumption (N households)')
plt.plot(paths["c_j"][:, 1:T].T)

# Individual dividends
plt.subplot(2, 2, 2)
plt.title('Dividends (N households)')
plt.plot(paths["d_j"][:, 1:T].T)

# Bond purchases k̂_jt
plt.subplot(2, 2, 3)
plt.title(r'Bond purchases $\hat{k}_{jt}$')
plt.plot(paths["khat_j"][:, :T].T)

# Total bonds
plt.subplot(2, 2, 4)
plt.title('Total Bonds $b_{jt}$')
plt.plot(paths["bonds_j"][:, :T].T)

plt.tight_layout()
plt.show()
```

## State-Space Representation

This section connects the Gorman economy to its underlying linear state-space representation.

### Closed-Loop Dynamics

The aggregate planning problem has a linear-quadratic structure. Following Chapter 5 of {cite:t}`HS2013`, the state and control are

$$
x_t =
\begin{bmatrix}
h_{t-1} \\
k_{t-1} \\
z_t
\end{bmatrix},
\qquad
u_t := i_t,
\qquad
z_{t+1} = A_{22} z_t + C_2 w_{t+1},
$$

where $h_{t-1}$ is the household service stock, $k_{t-1}$ is capital, and $z_t$ stacks exogenous variables.

The controlled transition is

$$
x_{t+1} = A x_t + B u_t + C w_{t+1}.
$$

Solving the LQ problem yields optimal feedback $u_t = -F x_t$, giving the closed-loop law of motion

$$
x_{t+1} = A_0 x_t + C w_{t+1}, \qquad A_0 := A - BF.
$$

### Selection Matrices

All equilibrium quantities are linear functions of the state. The `quantecon.DLE` class provides selection matrices $S_\bullet$ such that

$$
c_t = S_c x_t, \quad i_t = S_i x_t, \quad k_t = S_k x_t, \quad d_t = S_d x_t, \quad \text{etc.}
$$

Stacking these gives a measurement equation $y_t = G x_t$ where $G$ collects the relevant selection matrices.

```{code-cell} ipython3
# Closed-loop state space matrices
A0 = econ.A0
C = econ.C

print(f"State dimension: {A0.shape[0]}")
print(f"Shock dimension: {C.shape[1]}")

# Dominant eigenvalues of the closed-loop dynamics
eigvals = np.linalg.eigvals(A0)
print(f"Largest eigenvalues: {np.sort(np.abs(eigvals))[-3:][::-1]}")
```

The following figure shows selected components of the aggregate state $x_t$.

```{code-cell} ipython3
xp = paths["xp"]
T_plot = 250

plt.figure(figsize=(10, 4))
plt.plot(xp[1, :T_plot], label=r"$k_{t-1}$ (capital)")
plt.plot(xp[3, :T_plot], label=r"$d_{a,t}$ (aggregate endowment)")
plt.xlabel("t")
plt.legend()
plt.tight_layout()
plt.show()
```


## Redistribution via Pareto Weight Reallocation

This section develops a framework for analyzing the distributional effects of tax-and-transfer schemes within the Gorman aggregation model. The key insight is that we can reinterpret the competitive equilibrium consumption allocation in terms of Pareto weights, then consider alternative weight distributions that implement redistribution while preserving aggregate consumption and capital accumulation.

### Competitive Equilibrium and Pareto Weights

Start with the competitive equilibrium consumption allocation described by equation (12.4.4) on page 264 of Hansen and Sargent (2013):

$$
c_{jt} - \chi_{jt} = (u_j/u_a) (c_{at} - \chi_{at}), 
$$

where:
- The $a$ subscript pertains to the representative agent
- $c_{at}$ and $\chi_{at}$ are computed from the representative agent problem
- The $j$ subscript pertains to the $j$th household
- $c_{jt}$ and $\chi_{jt}$ are computed via the Gorman aggregation framework (the `heter` function above)

Recall from {eq}`eq:sharing_rule` that the Gorman sharing rule gives $c_{jt} = \mu_j c_t + \tilde{\chi}_{jt}$ where $\mu_j := u_j/u_a$ is household $j$'s wealth share. This $\mu_j$ directly corresponds to the implicit Pareto weight attached to individual $j$ in the competitive equilibrium.

Define the Pareto weight vector:

$$
\lambda_j := \frac{u_j}{u_a} = \mu_j,
$$

where the Pareto weights are determined by the initial distribution of wealth (the assignment of $\{k_{j,-1}\}_j$ and the endowment processes $\{d_{jt}\}_j$ across households $j \in \{1, \ldots, J\}$).

A fundamental property of these weights is that they sum to unity:

$$
\sum_{j=1}^J \lambda_j = \sum_{j=1}^J \frac{u_j}{u_a} = 1.
$$

This follows from the construction of the representative agent, whose utility weight $u_a$ is the sum of individual household utility weights (see equation {eq}`eq:agg_preference_aggregates`).

Using the Pareto weight notation, we can rewrite equation (1) as:

$$
c_{jt} - \chi_{jt} = \lambda_j (c_{at} - \chi_{at}).
$$

This expression clarifies that household $j$'s consumption net of the deviation term is proportional to aggregate consumption net of its deviation, with the proportion determined by the Pareto weight $\lambda_j$.

### Redistribution Proposal

Consider an efficient tax-and-transfer scheme that leaves aggregate consumption and capital accumulation unaltered but redistributes incomes in a way consistent with a new set of Pareto weights $\{\lambda_j^*\}_{j=1}^J$ satisfying:

$$
\lambda_j^* \geq 0 \quad \forall j, \qquad \sum_{j=1}^J \lambda_j^* = 1. 
$$

The associated competitive equilibrium consumption allocation under the new Pareto weights is:

$$
c_{jt} - \chi_{jt} = \lambda_j^* (c_{at} - \chi_{at}).
$$

Since the aggregate allocation $(c_{at}, k_{at}, \chi_{at})$ is unchanged, this redistribution preserves efficiency while reallocating consumption across households according to the new weights.

### Post-Tax-and-Transfer Consumption and Income

In what follows we treat the model’s household consumption panel $\{c_{jt}\}$ as our “CEX-like” consumption measure.

For income, recall from Section 12.6 (equation {eq}`eq:lm_budget`) that in the limited-markets implementation, household $j$'s budget satisfies:

$$
c_{jt} - \chi_{jt} + \mu_j \chi_{at} + \mu_j k_t = \mu_j[(\delta_k + \gamma) k_{t-1} + d_t].
$$

In this Hall-style calibration, a share-based “income” concept like $\lambda_j^*\big[(\delta_k + \gamma - 1)k_{t-1}+d_t\big]$ tends to move almost one-for-one with $\lambda_j^* c_t$ because aggregate investment is tiny and the resource constraint implies $c_t \approx (\delta_k + \gamma - 1)k_{t-1}+d_t$.

To get an income panel with idiosyncratic risk (closer to what’s used empirically), we instead construct household private income from (i) household endowments and (ii) net asset income from the limited-markets asset position. Let $R := \delta_k + \gamma$ be the gross return and let $a_{j,t}$ denote household $j$’s total asset position (in code: `paths["bonds_j"]`). Define

$$
y^{pre}_{j,t} := d_{j,t} + (R-1)a_{j,t-1}.
$$

We then define disposable (post-tax-and-transfer) income via a balanced-budget tax-and-transfer rule that shrinks idiosyncratic variation by a factor $(1-\tau)$ while holding aggregate income fixed:

$$
y^{post}_{j,t} := (1-\tau) y^{pre}_{j,t} + \tau\,\bar y^{pre}_t, \qquad \bar y^{pre}_t := \frac{1}{J}\sum_{j=1}^J y^{pre}_{j,t}.
$$

Here $\tau \in [0,1]$ indexes redistribution/insurance intensity: $\tau=0$ is no redistribution, while $\tau=1$ fully pools income cross-sectionally each date.

### Pre-Tax Private Income

We compare consumption to $y^{pre}_{j,t}$ and $y^{post}_{j,t}$ using standard “consumption insurance” metrics (pass-through of idiosyncratic income shocks to idiosyncratic consumption) and percentile plots.


The difference between equations (6)-(7) and equations (8)-(9) captures the redistributive impact of the tax-and-transfer system, measured by the change from $\lambda_j$ to $\lambda_j^*$.

To implement a more egalitarian distribution, we construct new Pareto weights $\{\lambda_j^*\}_{j=1}^J$ that:

1. Lower the weights for low-$j$ types (wealthier households in our ordering)
2. Increase the weights for high-$j$ types (less wealthy households)
3. Leave middle-$j$ types relatively unaffected
4. Preserve the constraint $\sum_{j=1}^J \lambda_j^* = 1$

We implement this using a smooth transformation. Let $\{\lambda_j\}_{j=1}^J$ denote the original competitive equilibrium Pareto weights (sorted in descending order). Define the redistribution function:

$$
f(j; J) = \frac{j-1}{J-1}, \qquad
g(j; J) = 2\left|f(j; J) - \frac{1}{2}\right|, \qquad
\tau(j; J, \alpha, \beta) = \alpha \cdot \left[g(j; J)\right]^\beta,
$$

where:
- $j \in \{1, \ldots, J\}$ is the household index
- $\alpha > 0$ controls the overall magnitude of redistribution (with $\tau$ maximized at the extremes)
- $\beta > 1$ controls the progressivity (higher $\beta$ concentrates redistribution more strongly in the tails)

The redistributed Pareto weights are:

$$
\tilde{\lambda}_j = \lambda_j + \tau(j; J, \alpha, \beta) \cdot (\bar{\lambda} - \lambda_j), 
$$

where $\bar{\lambda} = J^{-1}$ is the equal-weight benchmark.

To ensure the weights sum to unity, we normalize:

$$
\lambda_j^* = \frac{\tilde{\lambda}_j}{\sum_{k=1}^J \tilde{\lambda}_k}.
$$

```{code-cell} ipython3
def create_redistributed_weights(lambda_orig, alpha=0.5, beta=2.0):
    """Create more egalitarian Pareto weights via smooth redistribution."""
    lambda_orig = np.asarray(lambda_orig, dtype=float)
    J = len(lambda_orig)
    if J == 0:
        raise ValueError("lambda_orig must be non-empty")
    if J == 1:
        return np.array([1.0])
    lambda_bar = 1.0 / J
    j_indices = np.arange(J)
    f_j = j_indices / (J - 1)
    dist_from_median = np.abs(f_j - 0.5) / 0.5  # in [0, 1], smallest near the median
    tau_j = np.clip(alpha * (dist_from_median ** beta), 0.0, 1.0)
    lambda_tilde = lambda_orig + tau_j * (lambda_bar - lambda_orig)
    lambda_star = lambda_tilde / lambda_tilde.sum()
    return lambda_star
```

```{code-cell} ipython3
paths["u_ratio"]
```

```{code-cell} ipython3
mu_values = paths["u_ratio"]
idx_sorted = np.argsort(-mu_values)
lambda_orig_sorted = mu_values[idx_sorted]

red_alpha = 0.8
red_beta = 0.0

lambda_star_sorted = create_redistributed_weights(lambda_orig_sorted, alpha=red_alpha, beta=red_beta)

# Map redistributed weights back to the original household ordering
lambda_star = np.empty_like(mu_values, dtype=float)
lambda_star[idx_sorted] = lambda_star_sorted

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_plot = len(lambda_orig_sorted)
axes[0].plot(lambda_orig_sorted[:n_plot], 'o-', label=r'Original $\lambda_j$', alpha=0.7)
axes[0].plot(lambda_star_sorted[:n_plot], 's-', label=r'Redistributed $\lambda_j^*$', alpha=0.7)
axes[0].axhline(1.0 / len(lambda_orig_sorted), color='k', linestyle='--',
                label=f'Equal weight (1/{len(lambda_orig_sorted)})', alpha=0.5)
axes[0].set_xlabel(r'Household index $j$ (sorted by $\lambda$)')
axes[0].set_ylabel('Pareto weight')
axes[0].legend()

delta_lambda_sorted = lambda_star_sorted - lambda_orig_sorted
axes[1].bar(range(n_plot), delta_lambda_sorted[:n_plot], alpha=0.7,
            color=['g' if x > 0 else 'r' for x in delta_lambda_sorted[:n_plot]])
axes[1].axhline(0, color='k', linestyle='-', linewidth=0.8)
axes[1].set_xlabel(r'Household index $j$ (sorted by $\lambda$)')
axes[1].set_ylabel(r'$\lambda_j^* - \lambda_j$')

plt.tight_layout()
plt.show()
```

Let $R = \delta_k + \gamma_1$. For a given Pareto-weight vector $\omega$ (pre: $\omega=\mu$; post: $\omega=\lambda^*$), define household assets
$$
a_{j,t} \equiv \omega_j k_t + \hat{k}_{j,t}.
$$

The income measure we use is
$$
y_{j,t}(\omega)
= \omega_j d_t + (R-1)a_{j,t-1}
= \omega_j d_t + (R-1)\big(\omega_j k_{t-1} + \hat{k}_{j,t-1}\big).
$$

So
$$
y^{pre}_{j,t} = y_{j,t}(\mu),
\qquad
y^{post}_{j,t} = y_{j,t}(\lambda^*).
$$

```{code-cell} ipython3
mu_values = np.asarray(paths["u_ratio"]).reshape(-1)
t0 = burn_in

c_hh = paths["c_j"][:, t0:]
d_hh = paths["d_j"][:, t0:]
a_hh = paths["bonds_j"][:, t0:]
khat_hh = paths["khat_j"][:, t0:]

chi_hh = paths["chi_tilde"][:, t0:]
c_share = mu_values[:, None] * paths["c"][0, t0:]

print(f"max|chi_tilde| over panel = {float(np.max(np.abs(chi_hh))):.6g}")
print(f"std(mu*c_agg) = {float(np.std(c_share)):.6g}, std(chi_tilde) = {float(np.std(chi_hh)):.6g}")
resid = c_hh - c_share
var_c = float(np.var(c_hh))
var_resid = float(np.var(resid))
print(f"Variance share explained by mu*c_agg (pooled): {1.0 - (var_resid / var_c):.6f}")

R = float(deltak + gamma1)

# Aggregate quantities
c_agg = paths["c"][0, t0:]   # aggregate consumption
d_agg = paths["d"][0, t0:]   # aggregate endowment
k_agg = paths["k"][0, t0:]   # aggregate capital

# Pre-redistribution: income and consumption based on original Pareto weights μ_j
# c_jt = μ_j * c_t + χ̃_jt  (Gorman sharing rule)
# y_pre = μ_j * d_t + (R-1) * a_lag  (mutual fund income), with a_{j,t} = μ_j k_t + k̂_{j,t}
a_lag = np.concatenate([a_hh[:, [0]], a_hh[:, :-1]], axis=1)
k_lag = np.concatenate([k_agg[[0]], k_agg[:-1]], axis=0)[None, :]  # (1, T)
khat_lag = np.concatenate([khat_hh[:, [0]], khat_hh[:, :-1]], axis=1)

# Consistency check: total assets a_{j,t-1} = μ_j k_{t-1} + k̂_{j,t-1}
a_lag_check = mu_values[:, None] * k_lag + khat_lag
print(f"max|a_lag - (mu*k_lag + khat_lag)|: {float(np.max(np.abs(a_lag - a_lag_check))):.2e}")

# Post-redistribution: use redistributed Pareto weights λ_j^*
# Sort original weights, redistribute, then map back to original household ordering
idx_sorted = np.argsort(-mu_values)
lambda_orig_sorted = mu_values[idx_sorted]

# alpha controls redistribution strength beta controls shape
lambda_star_sorted = create_redistributed_weights(lambda_orig_sorted, alpha=red_alpha, beta=red_beta)
lambda_star = np.empty_like(mu_values, dtype=float)
lambda_star[idx_sorted] = lambda_star_sorted


# Pre/post allocations implied by weights (hold aggregate paths fixed)
ubs_for_alloc = Ub_list if "Ub_list" in globals() else ubs
h0i_alloc = np.array([[0.0]])

pre = allocation_from_weights(paths, econ, ubs_for_alloc, mu_values, gamma1, lam, h0i_alloc)
post = allocation_from_weights(paths, econ, ubs_for_alloc, lambda_star, gamma1, lam, h0i_alloc)

c_pre = pre["c"][:, t0:]
y_pre = pre["y_net"][:, t0:]

c_post = post["c"][:, t0:]
y_post = post["y_net"][:, t0:]

# Balanced transfer (flow): implements the new allocation with aggregate feasibility
T_flow = c_post - c_pre

# Budget constraint check: savings = income - consumption
# s_t = y_t - c_t should equal change in assets: a_t - a_{t-1}/R (approx)
savings_pre = y_pre - c_pre
savings_post = y_post - c_post

def _pct(panel, ps=(90, 50, 10)):
    return np.percentile(panel, ps, axis=0)


# Cross-sectional percentiles over time (drop the first period + leave a tail unplotted)
head_drop = 1
tail_drop = 50
T_avail = c_pre.shape[1]
start = min(max(head_drop, 0), T_avail)
end = max(start, T_avail - tail_drop)

T_ts = min(500, end - start)
t = t0 + start + np.arange(T_ts)

y_pre_pct = _pct(y_pre[:, start:start + T_ts], ps=(90, 50, 10))
y_post_pct = _pct(y_post[:, start:start + T_ts], ps=(90, 50, 10))
c_pre_pct = _pct(c_pre[:, start:start + T_ts], ps=(90, 50, 10))
c_post_pct = _pct(c_post[:, start:start + T_ts], ps=(90, 50, 10))

fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
titles = [
    r"Pre-tax income $y^{pre}$",
    rf"Post-tax income $y^{{post}}$",
    rf"Consumption $c^{{post}}$"
]
panels = [y_pre_pct, y_post_pct, c_post_pct]

for idx in range(3):
    ax[idx].set_title(titles[idx])
    ax[idx].plot(t, panels[idx][0], label="p90")
    ax[idx].plot(t, panels[idx][1], label="p50")
    ax[idx].plot(t, panels[idx][2], label="p10")
    ax[idx].set_xlabel("t")

plt.legend()

plt.tight_layout()
plt.show()
```