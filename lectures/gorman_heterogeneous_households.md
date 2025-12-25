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

This lecture implements the Gorman heterogeneous-household economy in Chapter 12 of {cite:t}`HansenSargent2013` using the `quantecon.DLE` class.

It complements {doc}`hs_recursive_models`, {doc}`growth_in_dles`, and {doc}`irfs_in_hall_model` by focusing on how to recover household allocations and portfolios from an aggregate DLE solution.

The headline result is that a complete-markets allocation can be implemented with a mutual fund and a one-period bond when Gorman aggregation holds.

In addition to what's in Anaconda, this lecture uses the `quantecon` library.

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

Gorman aggregation lets us solve heterogeneous-household economies in two steps: solve a representative-agent linear-quadratic planning problem for aggregates, then recover household allocations via a wealth-share rule with household-specific deviation terms.

The key insight is that Gorman conditions ensure all consumers have parallel Engel curves, so aggregate allocations and prices can be determined independently of distribution.

This eliminates the standard problem where the utility possibility frontier shifts with endowment changes, making it impossible to rank allocations without specifying distributional weights.

```{note}
Computing a competitive equilibrium typically entails solving a fixed point problem.

{cite:t}`negishi1960welfare` described a social welfare function that is maximized, subject to resource and technological constraints, by a competitive equilibrium allocation.

For Negishi, that social welfare function is a "linear combination of the individual utility functions of consumers, with the weights in the combination in inverse proportion to the marginal utilities of income."

Because Negishi's weights depend on the allocation through the marginal utilities of income, computing a competitive equilibrium via constrained maximization of a Negishi-style welfare function requires finding a fixed point in the weights.

When they apply, the beauty of Gorman's aggregation conditions is that time series aggregates and market prices can be computed *without* resorting to Negishi's fixed point approach.
```

With the help of this powerful result, we proceed in three steps in this lecture:

1. Solve the planner's problem and compute selection matrices that map the aggregate state into allocations and prices.
2. Compute household-specific policies and the Gorman sharing rule.
3. Implement the same Arrow-Debreu allocation using only a mutual fund (aggregate stock) and a one-period bond.

We then simulate examples with two and many households.

### Gorman aggregation in a static economy

To see where the sharing rule comes from, start with a static economy with $n$ goods, price vector $p$, and consumers $j = 1, \ldots, J$.

Let $c^a$ denote the aggregate amount of consumption to be allocated among consumers.

Associated with $c^a$ is an Edgeworth box and a set of Pareto optimal allocations.

From the Pareto optimal allocations, one can construct utility allocation surfaces that describe the frontier of alternative feasible utility assignments to individual consumers.

Imagine moving from the aggregate vector $c^a$ to some other vector $\tilde{c}^a$ and hence to a new Edgeworth box.

If neither the original box nor the new box contains the other, then it is possible that the utility allocation surfaces for the two boxes may *cross*, in which case there exists no ordering of aggregate consumption that is independent of the utility weights assigned to individual consumers.

Here is an example showing when aggregation fails.

Consider a two-person, two-good, pure-exchange economy where agent A has utility function $U^A = X_A^{1/3} Y_A^{2/3}$, while consumer B has utility function $U^B = X_B^{2/3} Y_B^{1/3}$.

With aggregate endowment pairs $E = (8,3)$ and $E = (3,8)$, the utility possibility frontiers cross, indicating that these two aggregate endowment vectors cannot be ranked in a way that ignores how utility is distributed between consumers A and B.

Furthermore, for a given endowment, the slope of the consumers' indifference curves at the tangencies that determine the contract curve varies as one moves along the contract curve.

This means that for a given aggregate endowment, the competitive equilibrium price depends on the allocation between consumers A and B.

It follows that for this economy, one cannot determine equilibrium prices independently of the equilibrium allocation.

{cite:t}`gorman1953community` described restrictions on preferences under which it *is* possible to obtain a community preference ordering.

Whenever Gorman's conditions are satisfied, there occur substantial simplifications in solving multiple-consumer optimal resource allocation problems: in intertemporal contexts, it becomes possible first to determine the optimal allocation of aggregate resources over time, and then allocate aggregate consumption among consumers by assigning utility levels to each person.

Gorman's aggregation conditions amount to assuming each consumer's compensated demand can be written as an affine function of a *scalar* utility index:

$$
c^j = \psi_j(p) + u^j \psi_c(p),
$$

where $\psi_c(p)$ is common across consumers and $\psi_j(p)$ is consumer-specific.

Aggregating over consumers gives a representative-consumer demand system

$$
c^a = \psi_a(p) + u^a \psi_c(p),
\qquad
u^a = \sum_{j=1}^J u^j,
\qquad
\psi_a(p) = \sum_{j=1}^J \psi_j(p).
$$

The baseline functions $\psi_j$ and the common function $\psi_c$ are the derivatives of concave functions that are positively homogeneous of degree 1.

Hence these functions are homogeneous of degree zero in prices, so indifference curve slopes depend only on price *ratios*.

This homogeneity allows us to invert $c^a = \psi_a(p) + u^a \psi_c(p)$ for the gradient vector $p$ (determined up to scale) and then allocate utility among consumers while respecting adding-up constraints.

A decentralized version of this analysis proceeds as follows.

Let $W^j$ denote the wealth of consumer $j$ and $W^a$ denote aggregate wealth.
Then $W^j$ should satisfy

$$
W^j = p \cdot c^j = p \cdot \psi_j(p) + u^j p \cdot \psi_c(p).
$$

Solving for $u^j$ gives

$$
u^j = \frac{W^j - p \cdot \psi_j(p)}{p \cdot \psi_c(p)}.
$$

Hence, the Engel curve for consumer $j$ is

$$
c^j = \psi_j(p) - \frac{p \cdot \psi_j(p)}{p \cdot \psi_c(p)} \psi_c(p) + W^j \frac{\psi_c(p)}{p \cdot \psi_c(p)}.
$$

Notice that the coefficient on $W^j$ is the same for all $j$ since $\psi_c(p)/(p \cdot \psi_c(p))$ is a function only of the price vector $p$.

The individual allocations can be determined from the Engel curves by substituting for $p$ the gradient vector obtained from the representative consumer's optimal allocation problem.

In the quadratic specifications used in this lecture (and in {cite}`HansenSargent2013`), the baseline components are degenerate in the sense that $\psi_j(p) = \chi^j$ is independent of $p$, where $\chi^j$ is a consumer-specific bliss point represented by a vector with the same dimension as $c^j$.

In that case, the static sharing rule reduces to

$$
c^j - \chi^j = \frac{u^j}{u^a}\,(c^a - \chi^a),
$$

so that there is a common scale factor $(u^j/u^a)$ across all goods for person $j$.

Hence the fraction of total utility assigned to consumer $j$ determines his fraction of the vector $(c^a - \chi^a)$.

This is exactly the form we use below, except that goods are indexed by both dates and states.

## Set up

Time is discrete, $t = 0,1,2,\dots$.

Households are indexed by $j = 1, \ldots, J$, where $J$ denotes the total number of households in the economy.

### Exogenous state

The exogenous state $z_t$ follows a first-order vector autoregression

$$
z_{t+1} = A_{22} z_t + C_2 w_{t+1},
$$

where $A_{22}$ governs persistence and $C_2$ maps i.i.d. shocks $w_{t+1}$ into the state.

The vector $z_t$ typically contains three types of components.

1. Constant: The first element is set to 1 and remains constant.

2. Aggregate shocks: Components with persistent dynamics that affect all households. 

    - In {ref}`gorman_twohh`, an AR(2) process drives aggregate endowment fluctuations.

3. Idiosyncratic shocks: Components that enter individual endowments with loadings summing to zero across households. 

    - These generate cross-sectional heterogeneity while preserving aggregate resource constraints.

The selection matrices $U_b$ and $U_d$ pick out which components of $z_t$ affect 
household preferences (bliss points) and endowments.

### Aggregate planner state

The aggregate planner state stacks lagged endogenous stocks and current exogenous variables:

$$
x_t = [h_{t-1}^\top, k_{t-1}^\top, z_t^\top]^\top.
$$

Here $h_{t-1}$ is the lagged household durable stock (habits or durables affecting utility), $k_{t-1}$ is lagged physical capital, and $z_t$ is the current exogenous state. 

Together, $x_t$ contains everything the planner needs to make decisions at time $t$. 

Aggregates are economy-wide totals summed across households: consumption $c_t$, investment $i_t$, capital $k_t$, household stock $h_t$, service flow $s_t$, intermediate good $g_t$, bliss $b_t$, and endowment $d_t$.

### Gorman weight

Let $\mu_{0j}^w$ denote household $j$'s *time-zero marginal utility of wealth*, the Lagrange multiplier on its intertemporal budget constraint. 

Define the aggregate multiplier as

$$
\mu_{0a}^w := \sum_{j=1}^J \mu_{0j}^w.
$$

The ratio $\mu_{0j}^w / \mu_{0a}^w$ serves as household $j$'s Gorman weight. 

For notational convenience, we write

$$
\mu_j := \frac{\mu_{0j}^w}{\mu_{0a}^w},
$$

so that $\sum_j \mu_j = 1$.

The sharing rule can be written either in the "baseline" form,

$$
c_{jt} - \chi_{jt} = \mu_j (c_t - \chi_t),
$$

or, equivalently, in the deviation form used in this lecture,

$$
c_{jt} = \mu_j c_t + \tilde{\chi}_{jt}, \qquad \tilde{\chi}_{jt} := \chi_{jt} - \mu_j \chi_t.
$$ (eq:sharing_rule)

Here $\mu_j c_t$ is household $j$'s proportional share and $\tilde{\chi}_{jt}$ captures deviations due to preference heterogeneity and initial conditions.

These deviations sum to zero: $\sum_j \tilde{\chi}_{jt} = 0$.

### Technologies

The economy's resource constraint is

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

Selection matrices such as $S_{(q)}$ map the aggregate state $x_t$ into aggregate quantities such as $q_t = S_{(q)} x_t$
for $q = c, i, k, h, s, g, b, d$.

Shadow-price mappings $M_c, M_k, \ldots$ are used to value streams and recover equilibrium prices.

### The individual household problem

Recall that we operate in an economy with $J$ consumers indexed by $j = 1, 2, \ldots, J$.

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
= \mathbb{E}_0 \sum_{t=0}^\infty \beta^t (w_{0t} \ell_{jt} + p_{0t} \cdot d_{jt}) + v_0 \cdot k_{j,-1},
$$ (eq:hh_budget)

where $b_{jt} = U_b^j z_t$ is household $j$'s preference shock, $d_{jt} = U_d^j z_t$ is household $j$'s endowment stream, $h_{j,-1}$ and $k_{j,-1}$ are given initial household stocks, and $\ell_{jt}$ is household labor supply.

The prices are:
- $p_{0t}$: the time-0 Arrow-Debreu price vector for date-$t$ consumption goods (across states), so $p_{0t} \cdot c_{jt}$ is the time-0 value of date-$t$ consumption
- $w_{0t}$: the time-0 value of date-$t$ labor
- $v_0$: the time-0 value of initial capital

These prices are determined in equilibrium.

This specification confines heterogeneity among consumers to four sources:

1. Differences in the preference processes $\{b_{jt}\}$, represented by different selections of $U_b^j$.

2. Differences in the endowment processes $\{d_{jt}\}$, represented by different selections of $U_d^j$.

3. Differences in initial household capital $h_{j,-1}$.

4. Differences in initial physical capital $k_{j,-1}$.

The matrices $\Lambda, \Pi_h, \Delta_h, \Theta_h$ do *not* depend on $j$.

Because the technology matrices are common across households, every household's demand system has the same functional form. The four sources of heterogeneity listed above determine each household's effective wealth, which in turn determines the Lagrange multiplier $\mu_j$ on the budget constraint. 

This multiplier appears as the household's share of aggregate consumption in the sharing rule {eq}`eq:sharing_rule`.

All households observe the same aggregate information $\mathcal{J}_t = [w_t, x_0]$.

These restrictions enable Gorman aggregation by ensuring that household demands are affine in wealth.

#### From individual problems to the aggregate problem

Under Gorman's restrictions, the competitive equilibrium allocation solves a social planner's problem with Pareto weights $\mu_j = \mu_{0j}^w / \mu_{0a}^w$.

Equilibrium prices emerge as the shadow prices from this problem.

Next, we formulate the aggregate planning problem.

### The aggregate planning problem

Summing the household budget constraints across $j = 1,\ldots,J$ gives

$$
\mathbb{E}_0 \sum_{t=0}^\infty \beta^t p_{0t} \cdot \left(\sum_j c_{jt}\right)
= \mathbb{E}_0 \sum_{t=0}^\infty \beta^t \left(w_{0t} \sum_j \ell_{jt} + p_{0t} \cdot \sum_j d_{jt}\right) + v_0 \cdot \sum_j k_{j,-1}.
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

where $g_t = \sum_j \ell_{jt}$ is the aggregate "intermediate good" that represents total labor supply in the DLE formulation.

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

## Household allocations

A direct way to compute allocations to individuals would be to solve the problem each household faces in competitive equilibrium at equilibrium prices.

For a fixed Lagrange multiplier $\mu_{0j}^w$ on the household's budget constraint, the household's problem can be expressed as an optimal linear regulator with a state vector augmented to reflect aggregate state variables determining scaled Arrow-Debreu prices.

One could compute individual allocations by iterating to find the multiplier $\mu_{0j}^w$ that satisfies the budget constraint.

But there is a more elegant approach using Gorman's allocation rules.

### Allocation rules

The allocation rule for household $j$'s labor (the "intermediate good" $g_{jt} = \ell_{jt}$) is

$$
\ell_{jt} = \mu_j \, \ell_t,
$$ (eq:labor_allocation)

where $\ell_t = \sum_i \ell_{it}$ is aggregate labor and $\mu_j = \mu_{0j}^w / \mu_{0a}^w$ is the Gorman weight.

The first-order conditions for consumption services give

$$
s_{jt} - b_{jt} = \mu_{0j}^w \rho_t^0,
$$ (eq:foc_services)

where $\rho_t^0$ is the shadow price of services from the aggregate problem.

Since the aggregate first-order condition is $s_t - b_t = \mu_{0a}^w \rho_t^0$, we can eliminate the price $\rho_t^0$ to obtain the service allocation rule in deviation form:

$$
s_{jt} - b_{jt} = \mu_j (s_t - b_t),
$$ (eq:service_allocation)

or equivalently $\tilde{s}_{jt} = \tilde{b}_{jt}$, where $\tilde{y}_{jt} := y_{jt} - \mu_j y_t$ for any variable $y$.

The beauty of this representation is that it does not involve prices directly.

### Deviation consumption

Given the sharing rule {eq}`eq:sharing_rule` introduced above, we now solve for household $j$'s deviation term $\tilde{\chi}_{jt}$ as a function of its deviation state $\tilde{\eta}_{jt} = h_{j,t-1} - \mu_j h_{t-1}$. 

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

Step two is to compute the Gorman weight $\mu_j$ using household $j$'s budget constraint.

We compute five present values:

- $W_d$: present value of household $j$'s endowment stream $\{d_{jt}\}$
- $W_k$: value of household $j$'s initial capital $k_{j,-1}$
- $W_{c1}$: present value of the "unit" consumption stream (what it costs to consume $c_t$)
- $W_{c2}$: present value of the deviation consumption stream $\{\tilde{\chi}_{jt}\}$
- $W_g$: present value of the intermediate good stream $\{g_t\}$

Each present value is computed using `doublej2` below to sum infinite series.

The budget constraint requires that the present value of consumption equals wealth. 

Substituting $c_{jt} = \mu_j c_t + \tilde{\chi}_{jt}$ gives

$$
\mu_j W_{c1} + W_{c2} = W_k + W_d + \mu_j W_g,
$$

where $\mu_j W_g$ is household $j$'s share of intermediate good value. Solving for $\mu_j$:

$$
\mu_j = \frac{W_k + W_d - W_{c2}}{W_{c1} - W_g}.
$$

The numerator is household $j$'s wealth (initial capital plus present value of endowments minus the cost of the deviation consumption stream). 

The denominator is the net cost of consuming one unit of aggregate consumption (consumption value minus intermediate good value).

Finally, the code constructs selection matrices $S_{ci}, S_{hi}, S_{si}$ that map the augmented state $X_t = [h_{j,t-1}^\top, x_t^\top]^\top$ into household $j$'s allocations:

$$
c_{jt} = S_{ci} X_t, \quad h_{jt} = S_{hi} X_t, \quad s_{jt} = S_{si} X_t.
$$

The augmented state includes household $j$'s own lagged durable stock $h_{j,t-1}$ because the deviation term $\tilde{\chi}_{jt}$ depends on it through the inverse canonical representation.

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

    ## Household deviation problem 

    # scaling trick in Chapter 3 of Hansen and Sargent (2013)
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

    # Undo sqrt β scaling
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
    W_c1 = float(
        (x0s.T @ W_c1 @ x0s + np.trace(M_cs @ V_s @ S_cs.T)).squeeze())

    S_c2 = np.hstack((np.zeros((M_c.shape[0], n_h + n_k)), U_b_s))
    A_s2 = np.block([[A0_dev, Θ_h @ S_c2], [np.zeros((n_x, n_h)), A0]])
    S_cs2 = np.hstack((-Λ_s, S_c2))
    x0s2 = np.vstack((h0i, x0))

    W_c2 = doublej2(β * A_s2.T, M_cs.T, A_s2.T, S_cs2.T)
    V_s2 = doublej2(β * A_s2, C_s, A_s2, C_s) * β / (1 - β)
    W_c2 = float(
        (x0s2.T @ W_c2 @ x0s2 + np.trace(M_cs @ V_s2 @ S_cs2.T)).squeeze())

    # Present value of initial capital
    W_k = float((k0i.T @ (
        np.asarray(Δ_k).T @ M_k +
        np.asarray(Γ).T @ M_d) @ x0).squeeze())

    ## Compute the Gorman weight
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

## Limited markets

This section implements a key result: the Arrow-Debreu allocation can be replicated using only a mutual fund and a one-period bond, rather than a complete set of contingent claims.

### The trading mechanism

Consider again an economy where each household $j$ initially owns claims to its own endowment stream $\{d_{jt}\}$. 

Instead of trading in a complete set of Arrow-Debreu markets, we open:

1. A stock market with securities paying dividends $\{d_{jt}\}$ for each endowment process
2. A market for one-period riskless bonds

At time zero, household $j$ executes the following trades:

1. Household $j$ sells all shares of its own endowment security.
2. Household $j$ purchases $\mu_j$ shares of all securities (equivalently, $\mu_j$ shares of a mutual fund holding the aggregate endowment).
3. Household $j$ takes position $\hat{k}_{j0}$ in the one-period bond.

After this initial rebalancing, household $j$ maintains a constant risky portfolio share $\mu_j$ forever, while using the one-period bond for dynamic rebalancing. 

The bond position $\hat{k}_{jt}$ evolves each period according to the recursion derived below.


The portfolio weight $\mu_j$ is not arbitrary: it is the unique weight that allows the limited-markets portfolio to replicate the Arrow-Debreu consumption allocation.

Under Gorman aggregation, household $j$'s equilibrium consumption satisfies

$$
c_{jt} = \mu_j c_t + \tilde{\chi}_{jt}.
$$

We need a portfolio strategy that delivers exactly this consumption stream. 

The mutual fund holds claims to all individual endowment streams. 

Total dividends paid by the fund each period are

$$
\sum_{i=1}^J d_{it} = d_t,
$$

which is the aggregate endowment. 

If household $j$ holds fraction $\theta_j$ of this fund, it receives $\theta_j d_t$ in dividends.

The proportional part of consumption $\mu_j c_t$ must be financed by the mutual fund and capital holdings. 

Using our calibration ($\Phi_c = \Phi_i = \theta_k = 1$, $\Gamma = \gamma_1$), the resource constraint becomes $c_t + i_t = \gamma_1 k_{t-1} + d_t$ and capital accumulation is $k_t = \delta_k k_{t-1} + i_t$. 

Substituting $i_t = k_t - \delta_k k_{t-1}$ into the resource constraint gives:

$$
c_t + k_t = (\delta_k + \gamma_1) k_{t-1} + d_t.
$$

Holding $\theta_j$ shares of aggregate wealth (capital plus claims to endowments) delivers $\theta_j [(\delta_k + \gamma_1)k_{t-1} + d_t] = \theta_j (c_t + k_t)$.

For this to finance $\mu_j c_t$ plus reinvestment $\mu_j k_t$, we need $\theta_j = \mu_j$. 

The remaining term $\tilde{\chi}_{jt}$ is financed by adjusting the bond position according to the recursion derived below.

Setting $\theta_j = \mu_j$ thus ensures that the mutual fund finances exactly the proportional share $\mu_j c_t$, while the bond handles the deviation $\tilde{\chi}_{jt}$. 

This transforms heterogeneous endowment risk into proportional shares of aggregate risk.

We now derive the law of motion for the bond position $\hat{k}_{jt}$.

First, write the budget constraint. Household $j$'s time-$t$ resources equal uses:

$$
\underbrace{\mu_j [(\delta_k + \gamma_1) k_{t-1} + d_t]}_{\text{mutual fund income}} + \underbrace{R \hat{k}_{j,t-1}}_{\text{bond return}} = \underbrace{c_{jt}}_{\text{consumption}} + \underbrace{\mu_j k_t}_{\text{new fund shares}} + \underbrace{\hat{k}_{jt}}_{\text{new bonds}}
$$

where $R := \delta_k + \gamma_1$ is the gross return.

```{note}
The constant gross return $R = \delta_k + \gamma_1$ arises from our specific calibration ($\Phi_c = \Phi_i = \Theta_k = 1$, $\Gamma = \gamma_1$).
```

Substituting the sharing rule by replacing $c_{jt}$ with $\mu_j c_t + \tilde{\chi}_{jt}$ gives:

$$
\mu_j [(\delta_k + \gamma_1) k_{t-1} + d_t] + R \hat{k}_{j,t-1} = \mu_j c_t + \tilde{\chi}_{jt} + \mu_j k_t + \hat{k}_{jt}
$$

The aggregate economy satisfies $c_t + k_t = (\delta_k + \gamma_1) k_{t-1} + d_t$, so:

$$
(\delta_k + \gamma_1) k_{t-1} + d_t = c_t + k_t
$$

Substituting into the left-hand side:

$$
\mu_j (c_t + k_t) + R \hat{k}_{j,t-1} = \mu_j c_t + \tilde{\chi}_{jt} + \mu_j k_t + \hat{k}_{jt}
$$

The $\mu_j c_t$ and $\mu_j k_t$ terms cancel:

$$
R \hat{k}_{j,t-1} = \tilde{\chi}_{jt} + \hat{k}_{jt}
$$

Rearranging gives the bond recursion:

$$
\hat{k}_{jt} = R \hat{k}_{j,t-1} - \tilde{\chi}_{jt}.
$$ (eq:bond-recursion)

This says that the bond position grows at rate $R$ but is drawn down by the deviation consumption $\tilde{\chi}_{jt}$. 

When $\tilde{\chi}_{jt} > 0$ (household $j$ consumes more than its share), it finances this by running down its bond holdings.

The household's total asset position is hence $a_{jt} = \mu_j k_t + \hat{k}_{jt}$.


### Initial bond position

To find $\hat{k}_{j0}$, we solve the recursion {eq}`eq:bond-recursion` forward.

Iterating forward from $\hat{k}_{jt} = R \hat{k}_{j,t-1} - \tilde{\chi}_{jt}$, we get:

$$
\begin{aligned}
\hat{k}_{j1} &= R \hat{k}_{j0} - \tilde{\chi}_{j1} \\
\hat{k}_{j2} &= R \hat{k}_{j1} - \tilde{\chi}_{j2} = R^2 \hat{k}_{j0} - R\tilde{\chi}_{j1} - \tilde{\chi}_{j2} \\
&\vdots \\
\hat{k}_{jT} &= R^T \hat{k}_{j0} - \sum_{t=1}^T R^{T-t} \tilde{\chi}_{jt}
\end{aligned}
$$

For the budget constraint to hold with equality, we apply transversality and require $\lim_{T \to \infty} R^{-T} \hat{k}_{jT} = 0$. Dividing by $R^T$:

$$
R^{-T} \hat{k}_{jT} = \hat{k}_{j0} - \sum_{t=1}^T R^{-t} \tilde{\chi}_{jt}
$$

Taking $T \to \infty$ and using transversality:

$$
0 = \hat{k}_{j0} - \sum_{t=1}^\infty R^{-t} \tilde{\chi}_{jt}
$$

Solving for the initial position gives:

$$
\hat{k}_{j0} = \sum_{t=1}^\infty R^{-t} \tilde{\chi}_{jt}.
$$

This is the present value of future deviation consumption. 

Since $\tilde{\chi}_{jt}$ depends only on the deterministic deviation between household $j$'s preference shocks and its share of aggregate preference shocks, this sum is in the time-zero information set and can be computed at date zero.

```{note} Connection to Rubinstein's two-fund theorem

This construction displays a multiperiod counterpart to an aggregation result for security markets derived by {cite:t}`rubinstein1974aggregation`.

In a two-period model, Rubinstein provided sufficient conditions on preferences of consumers and asset market payoffs for the implementation of an Arrow-Debreu contingent claims allocation with incomplete security markets.

In Rubinstein's implementation, all consumers hold the same portfolio of risky assets.

In our construction, consumers also hold the same portfolio of risky assets, and portfolio weights do not vary over time.

All changes over time in portfolio composition take place through transactions in the bond market.
```

Below is the code that computes household allocations and limited-markets portfolios along a fixed aggregate path according to the mechanism described above

```{code-cell} ipython3
def compute_household_paths(econ, U_b_list, U_d_list, x0, x_path, γ_1, Λ, h0i=None, k0i=None):
    """
    Compute household allocations and limited-markets portfolios
    along a fixed aggregate path.
    """

    # Organize inputs
    x0 = np.asarray(x0, dtype=float).reshape(-1, 1)
    x_path = np.asarray(x_path, dtype=float)

    Θ_h = np.atleast_2d(econ.thetah)
    Δ_h = np.atleast_2d(econ.deltah)
    Π_h = np.atleast_2d(econ.pih)
    Λ = np.atleast_2d(Λ)

    n_h = Θ_h.shape[0]
    n_k = np.atleast_2d(econ.thetak).shape[0]

    z_path = x_path[n_h + n_k:, :]

    
    # Select consumption, capital, bonds, and dividends
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

    # Prepare data containers
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

    # For each household, compute Gorman weight and paths
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
            # The book sets the intermediate-good multiplier to $M_g = S_g$
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
        # At t=0, assume η_{-1} = 0
        χ_tilde[j, 0] = (Π_inv @ b_tilde[:, 0]).squeeze()
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

        # Solve for bond path by backward iteration from terminal condition.
        if abs(R - 1.0) >= 1e-14:
            k_hat[j, -1] = χ_tilde[j, -1] / (R - 1.0)
            for t in range(T - 1, 0, -1):
                k_hat[j, t - 1] = (k_hat[j, t] + χ_tilde[j, t]) / R

        a_total[j] = k_share[j] + k_hat[j]

    # Validate that Gorman weights sum to 1 (required for sharing rule consistency)
    μ_sum = np.sum(μ)
    if abs(μ_sum - 1.0) > 1e-6:
        import warnings
        warnings.warn(f"Gorman weights μ sum to {μ_sum:.6f}, not 1.0. "
                      "This may indicate calibration issues.")

    return {
        "μ": μ,
        "χ_tilde": χ_tilde,
        "c": c,
        "k": k,
        "d": d,
        "c_j": c_j,
        "d_j": d_j,
        "d_share": d_share,
        "k_share": k_share,
        "k_hat": k_hat,
        "a_total": a_total,
        "x_path": x_path,
        "z_path": z_path,
        "R": R,
    }
```

The next function collects everything to solve the planner's problem and compute household paths 
into one function

```{code-cell} ipython3
def solve_model(info, tech, pref, U_b_list, U_d_list, γ_1, Λ, z0, ts_length=2000):
    """
    Solve the representative-agent DLE problem and compute household paths.
    """
    econ = DLE(info, tech, pref)

    # Build full initial state x0 = [h_{-1}, k_{-1}, z0]
    z0 = np.asarray(z0, dtype=float).reshape(-1, 1)
    n_h = np.atleast_2d(econ.thetah).shape[0]
    n_k = np.atleast_2d(econ.thetak).shape[0]
    x0_full = np.vstack([np.zeros((n_h, 1)), np.zeros((n_k, 1)), z0])

    # Solve LQ problem and simulate paths
    lq = LQ(econ.Q, econ.R, econ.A, econ.B,
            econ.C, N=econ.W, beta=econ.beta)
    x_path, _, _ = lq.compute_sequence(x0_full, ts_length=ts_length)

    paths = compute_household_paths(
        econ=econ,
        U_b_list=U_b_list,
        U_d_list=U_d_list,
        x0=x0_full,
        x_path=x_path,
        γ_1=γ_1,
        Λ=Λ,
    )
    return paths, econ
```

(gorman_twohh)=
## Example: two-household economy

We reproduce the two-household example from Chapter 12.6 of {cite:t}`HansenSargent2013`.

There are two households, each with preferences

$$
-\frac{1}{2} \mathbb{E} \sum_{t=0}^{\infty} \beta^t \left[(c^i_t - b^i_t)^2 + \ell^{i\,2}_t\right] \mid \mathcal{J}_0, \quad i = 1, 2.
$$

We set $b^i_t = 15$ for $i = 1, 2$, so the aggregate preference shock is $b_t = \sum_i b^i_t = 30$. The endowment processes are

$$
d^1_t = 4 + 0.2\, w^1_t, \qquad
d^2_t = 3 + \tilde{d}^2_t,
$$

where $w^1_t$ is Gaussian white noise with variance $(0.2)^2$, and $\tilde{d}^2_t$ follows

$$
\tilde{d}^2_t = 1.2\, \tilde{d}^2_{t-1} - 0.22\, \tilde{d}^2_{t-2} + 0.25\, w^2_t,
$$

with $w^2_t$ Gaussian white noise with variance $(0.25)^2$.

```{code-cell} ipython3
ϕ_1 = 1e-5
γ_1 = 0.1
δ_k = 0.95
β = 1.0 / (γ_1 + δ_k)

θ_k = 1.0
δ_h = 0.0
θ_h = 0.0
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

# Solve LQ problem and simulate paths
lq = LQ(econ.Q, econ.R, econ.A, econ.B,
        econ.C, N=econ.W, beta=econ.beta)
x_path, _, _ = lq.compute_sequence(x0, ts_length=ts_length)

paths = compute_household_paths(
    econ=econ,
    U_b_list=[U_b1, U_b2],
    U_d_list=[U_d1, U_d2],
    x0=x0,
    x_path=x_path,
    γ_1=γ_1,
    Λ=Λ,
)
```

The next figure plots the consumption paths for the aggregate economy and the two households

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: consumption paths
    name: fig-gorman-consumption
---
T_plot = 250
fig, ax = plt.subplots()
ax.plot(paths["c"][0, 5:T_plot], lw=2, label="aggregate")
ax.plot(paths["c_j"][0, 5:T_plot], lw=2, label="household 1")
ax.plot(paths["c_j"][1, 5:T_plot], lw=2, label="household 2")
ax.set_xlabel("time")
ax.set_ylabel("consumption")
ax.legend()
plt.show()
```

### Risk sharing

The Gorman sharing rule has strong implications for risk sharing across households.

Because the coefficient $\mu_j = \mu_{0j}^w / \mu_{0a}^w$ is invariant over time and across goods, allocation rule {eq}`eq:sharing_rule` implies a form of risk pooling in the deviation process $\{c_{jt} - \chi_{jt}\}$.

All households share aggregate consumption risk in fixed proportions determined at date zero.

Non-separabilities in preferences (either over time or across goods) affect only the construction of the baseline process $\{\chi_{jt}\}$ and the calculation of the risk-sharing coefficient $\mu_j$ implied by the distribution of wealth.

They do not break the proportional sharing of aggregate risk.

In the special case where the preference shock processes $\{b_{jt}\}$ are deterministic in the sense that they reside in the information set $\mathcal{J}_0$, individual consumption goods will be perfectly correlated with their aggregate counterparts (conditioned on $\mathcal{J}_0$).

The next figure plots the limited-markets bond adjustment and confirms that the adjustments sum to approximately zero.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: bond adjustment positions
    name: fig-gorman-bond-adjustment
---
fig, ax = plt.subplots()
ax.plot(paths["k_hat"][0, 5:T_plot], lw=2, label="household 1")
ax.plot(paths["k_hat"][1, 5:T_plot], lw=2, label="household 2")
ax.plot(paths["k_hat"][:, 5:T_plot].sum(axis=0), lw=2, label="sum")
ax.axhline(0.0, color="k", lw=1, alpha=0.5)
ax.set_xlabel("time")
ax.set_ylabel("bond position")
ax.legend()
plt.show()
```

The final check is to verify that the bond positions sum to zero at all times, confirming that the limited-markets implementation is self-financing

```{code-cell} ipython3
np.max(np.abs(paths["k_hat"].sum(axis=0)))
```

## Example: many-household Gorman economies

We now extend the two-household example to an economy with many households.

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

Let $n_a$ denote the number of "absorbing" households. For households $j > n_a$, individual endowments are

$$
d_{jt} = \alpha_j + \phi_j d_{a,t} + \sigma_j w_{j,t}, \quad j = n_a + 1, \ldots, n.
$$

The first $n_a$ households absorb the negative of all idiosyncratic shocks to ensure aggregation:

$$
d_{jt} = \alpha_j + \phi_j d_{a,t} - \frac{1}{n_a} \sum_{k=n_a+1}^n \sigma_k w_{k,t}, \quad j = 1, \ldots, n_a.
$$

This construction ensures that

$$
\sum_{j=1}^n d_{j,t} = \sum_{j=1}^n \alpha_j + \left(\sum_{j=1}^n \phi_j\right) d_{a,t}.
$$

Imposing $\sum_{j=1}^n \phi_j = 1$ gives

$$
\sum_{j=1}^n d_{j,t} = \sum_{j=1}^n \alpha_j + d_{a,t}.
$$

Preference shocks are muted to simplify experiments:

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
    to ensure shocks sum to zero:
        sum_{j=1}^{n_absorb} (-1/n_absorb * sum_{k>n_absorb} eta_k) + sum_{k>n_absorb} eta_k = 0

    Each household k > n_absorb has its own idiosyncratic endowment shock eta_{k,t}
    following an AR(1):

        eta_{k,t+1} = rho_idio[k] * eta_{k,t} + sigma_k * w_{k,t+1}

    with rho_idio = 0 recovering i.i.d. shocks (AR(0)).

    Preference shocks xi_{j,t} are included for each household but 
    can be set to zero if desired.
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

    # State vector: 
    # z_t = [1, d_{a,t}, d_{a,t-1}, eta_{n_absorb+1,t}, ..., eta_{n,t}, xi_{1,t}, ..., xi_{n,t}]
    # where eta_{j,t} are idiosyncratic endowment shocks (j=n_absorb+1..n)
    # and xi_{j,t} are preference shocks (j=1..n)
    # First n_absorb households absorb -1/n_absorb * sum(eta_j) to ensure shocks sum to zero
    # Dimension: 3 + (n - n_absorb) + n = 2n + 3 - n_absorb

    n_idio = n - n_absorb       # eta_{n_absorb+1}, ..., eta_n
    n_pref = n                  # xi_1, ..., xi_n
    nz = 3 + n_idio + n_pref
    nw = 1 + n_idio + n_pref    # aggregate + idio + pref shocks

    # Persistence parameters
    rho_idio = np.asarray(rho_idio, dtype=float)
    if rho_idio.ndim == 0:
        rho_idio = np.full(n_idio, float(rho_idio))
    if rho_idio.shape != (n_idio,):
        raise ValueError(
        f"rho_idio must be scalar or shape ({n_idio},), got {rho_idio.shape}")

    rho_pref = np.asarray(rho_pref, dtype=float)
    if rho_pref.ndim == 0:
        rho_pref = np.full(n_pref, float(rho_pref))
    if rho_pref.shape != (n_pref,):
        raise ValueError(
        f"rho_pref must be scalar or shape ({n_pref},), got {rho_pref.shape}")

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
    C2[1, 0] = sigma_a  
    for j in range(n_idio):
        # Map to households n_absorb+1, ..., n 
        C2[3 + j, 1 + j] = sigmas[n_absorb + j]
    for j in range(n_pref):
        C2[3 + n_idio + j, 1 + n_idio + j] = gammas[j]  # gamma_j -> xi_j

    # Ud_per_house: endowment loading
    # First n_absorb households: 
    # d_{jt} = alpha_j + phi_j * d_{a,t} - (1/n_absorb) * sum_{k>n_absorb} eta_{k,t}
    # Remaining households k > n_absorb: 
    # d_{kt} = alpha_k + phi_k * d_{a,t} + eta_{k,t}
    Ud_per_house = []
    for j in range(n):
        block = np.zeros((2, nz))
        block[0, 0] = alphas[j]    # constant
        block[0, 1] = phis[j]      # loading on d_{a,t}

        if j < n_absorb:
            for k in range(n_idio):
                block[0, 3 + k] = -1.0 / n_absorb
        else:
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

### 100-household reverse engineered economy

We now instantiate a 100-household economy using the reverse-engineered Gorman specification.

We use the same technology and preference parameters as the two-household example.

```{code-cell} ipython3
# Technology and preference parameters
ϕ_1 = 1e-5
γ_1 = 0.1
δ_k = 0.95
β = 1.0 / (γ_1 + δ_k)

θ_k = 1.0
δ_h = 0.0
θ_h = 0.0
Λ = 0.0
Π_h = 1.0

Φ_c = np.array([[1.0], [0.0]])
Φ_g = np.array([[0.0], [1.0]])
Φ_i = np.array([[1.0], [-ϕ_1]])
Γ = np.array([[γ_1], [0.0]])
```

We set household-specific parameters below and impose $\sum_j \phi_j = 1$.

```{code-cell} ipython3
np.random.seed(42)
N = 100

# Aggregate endowment process parameters
ρ1 = 0.95
ρ2 = 0.0
σ_a = 0.5


# Mean endowments α_j and aggregate exposure φ_j
αs = np.random.uniform(3.0, 5.0, N)
φs_raw = np.random.uniform(0.5, 1.5, N)
φs = φs_raw / np.sum(φs_raw)  # normalize so Σ φ_j = 1

# Rank households by mean endowment to assign idiosyncratic risk
wealth_rank_proxy = np.argsort(np.argsort(αs))
wealth_pct_proxy = (wealth_rank_proxy + 0.5) / N
poorness = 1.0 - wealth_pct_proxy

# First n_absorb households absorb idiosyncratic shocks
n_absorb = 50

# Poorer households face larger, more persistent idiosyncratic shocks
σ_idio_min, σ_idio_max = 0.2, 5.0
σs = σ_idio_min + (σ_idio_max - σ_idio_min) * (poorness ** 2.0)
ρ_idio_min, ρ_idio_max = 0.0, 0.98
ρ_idio = ρ_idio_min + (ρ_idio_max - ρ_idio_min) * (poorness[n_absorb:] ** 1.0)

# Preference shocks (disabled by default)
b_bar = 5.0
enable_pref_shocks = False
pref_shock_scale = 0.5
pref_shock_persistence = 0.7

if enable_pref_shocks:
    γs_pref = pref_shock_scale * np.ones(N)
    ρ_pref = pref_shock_persistence
else:
    γs_pref = np.zeros(N)
    ρ_pref = 0.0

t0 = 200

A22, C2, Ub, Ud, Ub_list, Ud_list, x0 = build_reverse_engineered_gorman_extended(
    n=N,
    rho1=ρ1, rho2=ρ2, sigma_a=σ_a,
    alphas=αs, phis=φs, sigmas=σs,
    b_bar=b_bar, gammas=γs_pref,
    rho_idio=ρ_idio, rho_pref=ρ_pref,
    n_absorb=n_absorb,
)

print(f"State dimension nz = {A22.shape[0]}")
print(f"Shock dimension nw = {C2.shape[1]}")
print(f"sum(φs) = {np.sum(φs):.6f} (should be 1.0)")
```

```{code-cell} ipython3
info_ar1 = (A22, C2, Ub, Ud)
pref_ar1 = (np.array([[β]]),
            np.array([[Λ]]),
            np.array([[Π_h]]),
            np.array([[δ_h]]),
            np.array([[θ_h]]))
tech_ar1 = (Φ_c, Φ_g, Φ_i, Γ,
            np.array([[δ_k]]),
            np.array([[θ_k]]))

paths, econ = solve_model(info_ar1, tech_ar1, pref_ar1,
                    Ub_list, Ud_list, γ_1, Λ, z0=x0)
```

```{code-cell} ipython3
print(f"State dimension: {A22.shape[0]}, shock dimension: {C2.shape[1]}")
print(f"Aggregate: ρ₁={ρ1:.3f}, ρ₂={ρ2:.3f}, σₐ={σ_a:.2f}")
print(f"Endowments: α ∈ [{np.min(αs):.2f}, {np.max(αs):.2f}], Σφ={np.sum(φs):.6f}")
```

The next plots show household consumption and dividend paths after discarding the initial burn-in period.

```{code-cell} ipython3
T_plot = 50

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(paths["c_j"][:, t0:t0+T_plot].T)
axes[0].set_xlabel("time")
axes[0].set_ylabel("consumption")

axes[1].plot(paths["d_share"][:, t0:t0+T_plot].T)
axes[1].set_xlabel("time")
axes[1].set_ylabel("dividends")

plt.tight_layout()
plt.show()
```

## State-space representation

### Closed-loop state-space system

The DLE framework represents the economy as a linear state-space system. After solving the optimal control problem and substituting the policy rule, we obtain a closed-loop system:

$$
x_{t+1} = A_0 x_t + C w_{t+1},
$$

where the aggregate state vector is

$$
x_t =
\begin{bmatrix}
h_{t-1}\\
k_{t-1}\\
z_t
\end{bmatrix},
$$

with $h_{t-1}$ the household service stock, $k_{t-1}$ the capital stock, and $z_t$ the exogenous state (constant, aggregate endowment states, and idiosyncratic shock states).

Any equilibrium quantity is a linear function of the state. 

The `quantecon.DLE` module provides selection matrices $S_\bullet$ such that:

$$
c_t = S_c x_t,\quad g_t = S_g x_t,\quad i_t = S_i x_t,\quad h_t = S_h x_t,\quad k_t = S_k x_t,\quad d_t = S_d x_t,\quad b_t = S_b x_t,\quad s_t = S_s x_t.
$$

We can stack these to form a measurement matrix:

$$
G =
\begin{bmatrix}
S_c\\ S_g\\ S_i\\ S_h\\ S_k\\ S_d\\ S_b\\ S_s
\end{bmatrix},
\qquad\text{giving}\qquad
y_t = G x_t.
$$

```{code-cell} ipython3
A0 = econ.A0
C = econ.C

G = np.vstack([
    econ.Sc,
    econ.Sg,
    econ.Si,
    econ.Sh,
    econ.Sk,
    econ.Sd,
    econ.Sb,
    econ.Ss,
])

# Extract dimensions for proper slicing
n_h = np.atleast_2d(econ.thetah).shape[0]
n_k = np.atleast_2d(econ.thetak).shape[0]
n_endo = n_h + n_k  # endogenous state dimension

print(f"Shapes: A0 {A0.shape}, C {C.shape}, G {G.shape}")
print(f"max |A0[{n_endo}:,{n_endo}:] - A22| = {np.max(np.abs(A0[n_endo:, n_endo:] - A22)):.2e}")
```

With the state space representation, we can compute impulse responses to show how shocks propagate through the economy. 

To trace the impulse response to shock $j$, we set `shock_idx=j` which selects column $j$ of the loading matrix $C$. 

This creates an initial state $x_0 = C e_j \sigma$ where $e_j$ is the $j$-th standard basis vector and $\sigma$ is the shock size. 

We then iterate forward with $x_{t+1} = A_0 x_t$ and compute observables via $y_t = G x_t$.

```{code-cell} ipython3
def compute_irf(A0, C, G, shock_idx, T=50, shock_size=1.0):
    """Compute impulse response to shock shock_idx with given shock size."""
    n_x = A0.shape[0]
    n_w = C.shape[1]

    x_path = np.zeros((n_x, T))
    y_path = np.zeros((G.shape[0], T))

    w0 = np.zeros(n_w)
    w0[shock_idx] = shock_size
    x_path[:, 0] = C @ w0
    y_path[:, 0] = G @ x_path[:, 0]

    for t in range(1, T):
        x_path[:, t] = A0 @ x_path[:, t-1]
        y_path[:, t] = G @ x_path[:, t]

    return x_path, y_path

n_h = np.atleast_2d(econ.thetah).shape[0]
n_k = np.atleast_2d(econ.thetak).shape[0]

idx_da = n_h + n_k + 1
T_irf = 50
shock_size = 1.0
irf_x, irf_y = compute_irf(A0, C, G, shock_idx=0, T=T_irf, shock_size=shock_size)
idx_k = 4


da_irf = irf_x[idx_da, :]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))


axes[0].plot(irf_y[idx_k, :], lw=2)
axes[0].set_xlabel('time')
axes[0].set_ylabel('capital')

axes[1].plot(da_irf, lw=2)
axes[1].set_xlabel('time')
axes[1].set_ylabel(r'$d_{a,t}$ (aggregate endowment shock)')


plt.tight_layout()
plt.show()
```

The aggregate endowment shock $d_{a,t}$ follows an AR(1) process with $\rho_1 = 0.95$, so after the initial impact it decays monotonically toward zero as $0.95^t$. 

Capital, by contrast, is an endogenous stock that accumulates when the planner smooths consumption. 

The positive endowment shock increases resources temporarily, but Hall-style preferences imply the planner saves part of the windfall rather than consuming it immediately. 

This causes capital to rise as $d_{a,t}$ falls — this is permanent income logic at work.

Next, we examine if the household consumption and endowment paths generated by the simulation obey the Gorman sharing rule

```{code-cell} ipython3
# Now stack the household consumption panel with household endowments and rerun DMD.
c_j_t0 = paths["c_j"][..., t0:]
d_j_t0 = paths["d_share"][..., t0:]

c_panel = np.asarray(c_j_t0 - c_j_t0.mean(axis=0))  # (N, T - t0)
d_panel = np.asarray(d_j_t0 - d_j_t0.mean(axis=0))  # (N, T - t0)
assert c_panel.shape == d_panel.shape

# Plot aggregate endowment and individual household endowments
d_agg = paths["d"][0, t0:]
d_households = d_j_t0

T_plot = min(500, d_agg.shape[0])
time_idx = np.arange(T_plot)
n_to_plot = 1


fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Top panel: Aggregate endowment
axes[0].plot(time_idx, d_agg[:T_plot], linewidth=2.5, color='C0', label='Aggregate endowment $d_t$')
axes[0].set_ylabel('Endowment')
axes[0].set_title('Aggregate Endowment (contains ghost AR(2) process)')
axes[0].legend()

# Also plot the mean across households
d_mean = d_households[:, :T_plot].mean(axis=0)
axes[1].plot(time_idx, d_mean, linewidth=2.5, color='black', linestyle='--',
             label=f'Mean across {d_households.shape[0]} households', alpha=0.8)

axes[1].set_xlabel('Time (after burn-in)')
axes[1].set_ylabel('Endowment')
axes[1].set_title(f'Average of Individual Household Endowments')
axes[1].legend(loc='upper right', ncol=2)

plt.tight_layout()
plt.show()
```

Indeed, the average of individual household endowments tracks the aggregate endowment process, confirming that the construction of the idiosyncratic shocks and absorbing households works as intended.


## Redistribution via Pareto weight reallocation

This section analyzes tax-and-transfer schemes by reinterpreting competitive equilibrium allocations in terms of Pareto weights, then considering alternative weight distributions that redistribute consumption while preserving aggregate dynamics.

### Redistribution via Pareto weights

The sharing rule {eq}`eq:sharing_rule` can be written as $c_{jt} - \chi_{jt} = \mu_j (c_t - \chi_t)$, where $\mu_j$ is household $j$'s wealth share. 

Define the Pareto weight $\lambda_j := \mu_j$, with $\sum_{j=1}^J \lambda_j = 1$.

Consider redistributing consumption by choosing new weights $\{\lambda_j^*\}$ satisfying $\lambda_j^* \geq 0$ and $\sum_j \lambda_j^* = 1$. 

The new allocation $c_{jt} - \chi_{jt} = \lambda_j^* (c_t - \chi_t)$ preserves aggregate dynamics $(c_t, k_t)$ while reallocating consumption across households.

### Post-tax-and-transfer consumption and income

Let $R := \delta_k + \gamma_1$. 

For a given Pareto-weight vector $\omega$, 
household $j$'s assets are $a_{j,t} \equiv \omega_j k_t + \hat{k}_{j,t}$ and income is

$$
y_{j,t}(\omega) = \omega_j d_t + (R-1) a_{j,t-1}.
$$

In this section, we compare pre- and post-redistribution consumption and income.

Let $\mu := \{\mu_j\}_{j=1}^J$ denote the original competitive equilibrium Pareto weights, and let $\lambda^* := \{\lambda_j^*\}_{j=1}^J$ denote the redistributed weights.

We write $y^{pre}_{j,t} = y_{j,t}(\mu)$ and $y^{post}_{j,t} = y_{j,t}(\lambda^*)$, then compare consumption to income using percentile plots.

To implement a redistribution from $\mu$ to $\lambda^*$, we construct new Pareto weights $\{\lambda_j^*\}_{j=1}^J$ that:

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
def create_redistributed_weights(λ_orig, α=0.5, β=2.0):
    """Create more egalitarian Pareto weights via smooth redistribution."""
    λ_orig = np.asarray(λ_orig, dtype=float)
    J = len(λ_orig)
    if J == 0:
        raise ValueError("λ_orig must be non-empty")
    if J == 1:
        return np.array([1.0])
    λ_bar = 1.0 / J
    j_indices = np.arange(J)
    f_j = j_indices / (J - 1)
    dist_from_median = np.abs(f_j - 0.5) / 0.5  # in [0, 1], smallest near the median
    τ_j = np.clip(α * (dist_from_median ** β), 0.0, 1.0)
    λ_tilde = λ_orig + τ_j * (λ_bar - λ_orig)
    λ_star = λ_tilde / λ_tilde.sum()
    return λ_star
```

Let's plot the effect of redistribution on the Pareto weights in our 100-household economy

```{code-cell} ipython3
μ_values = paths["μ"]
idx_sorted = np.argsort(-μ_values)
λ_orig_sorted = μ_values[idx_sorted]

red_α = 0.8
red_β = 0.0

λ_star_sorted = create_redistributed_weights(λ_orig_sorted, α=red_α, β=red_β)

# Map redistributed weights back to the original household ordering
λ_star = np.empty_like(μ_values, dtype=float)
λ_star[idx_sorted] = λ_star_sorted

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n_plot = len(λ_orig_sorted)
axes[0].plot(λ_orig_sorted[:n_plot], 'o-', label=r'original $\lambda_j$', alpha=0.7, lw=2)
axes[0].plot(λ_star_sorted[:n_plot], 's-', label=r'redistributed $\lambda_j^*$', alpha=0.7, lw=2)
axes[0].axhline(1.0 / len(λ_orig_sorted), color='k', linestyle='--',
                label=f'equal weight (1/{len(λ_orig_sorted)})', alpha=0.5, lw=2)
axes[0].set_xlabel(r'household index $j$ (sorted by $\lambda$)')
axes[0].set_ylabel('Pareto weight')
axes[0].legend()

Δλ_sorted = λ_star_sorted - λ_orig_sorted
axes[1].bar(range(n_plot), Δλ_sorted[:n_plot], alpha=0.7,
            color=['g' if x > 0 else 'r' for x in Δλ_sorted[:n_plot]])
axes[1].axhline(0, color='k', linestyle='-', lw=2)
axes[1].set_xlabel(r'household index $j$ (sorted by $\lambda$)')
axes[1].set_ylabel(r'$\lambda_j^* - \lambda_j$')

plt.tight_layout()
plt.show()
```

Once we have the redistributed Pareto weights, we can compute the new household consumption and income paths.

The function below computes household consumption and income under given Pareto weights

```{code-cell} ipython3
def allocation_from_weights(paths, econ, U_b_list, weights, γ_1, Λ, h0i=None):
    """
    Compute household consumption and income under given Pareto weights.
    """

    weights = np.asarray(weights).reshape(-1)
    N = len(weights)

    # Extract aggregate paths
    x_path = paths["x_path"]
    c_agg = paths["c"]
    d_agg = paths["d"]
    k_agg = paths["k"]
    z_path = paths["z_path"]

    _, T = c_agg.shape

    # Get parameters
    Θ_h = np.atleast_2d(econ.thetah)
    Δ_h = np.atleast_2d(econ.deltah)
    Π_h = np.atleast_2d(econ.pih)
    Λ = np.atleast_2d(Λ)
    n_h = Θ_h.shape[0]

    δ_k = float(np.asarray(econ.deltak).squeeze())
    R = δ_k + float(γ_1)

    Π_inv = np.linalg.inv(Π_h)
    A_h = Δ_h - Θ_h @ Π_inv @ Λ
    B_h = Θ_h @ Π_inv

    if h0i is None:
        h0i = np.zeros((n_h, 1))

    # Compute household allocations
    c_j = np.zeros((N, T))
    χ_tilde = np.zeros((N, T))
    k_hat = np.zeros((N, T))

    for j in range(N):
        U_bj = np.asarray(U_b_list[j], dtype=float)
        b_agg = econ.Sb @ x_path
        b_tilde = U_bj @ z_path - weights[j] * b_agg

        η = np.zeros((n_h, T + 1))
        η[:, 0] = np.asarray(h0i).reshape(-1)
        # At t=0, assume η_{-1} = 0 
        χ_tilde[j, 0] = (Π_inv @ b_tilde[:, 0]).squeeze()
        for t in range(1, T):
            χ_tilde[j, t] = (-Π_inv @ Λ @ η[:, t - 1] + Π_inv @ b_tilde[:, t]).squeeze()
            η[:, t] = (A_h @ η[:, t - 1] + B_h @ b_tilde[:, t]).squeeze()

        c_j[j] = (weights[j] * c_agg[0] + χ_tilde[j]).squeeze()

        # Solve for bond path by backward iteration from terminal condition.
        if abs(R - 1.0) >= 1e-14:
            k_hat[j, -1] = χ_tilde[j, -1] / (R - 1.0)
            for t in range(T - 1, 0, -1):
                k_hat[j, t - 1] = (k_hat[j, t] + χ_tilde[j, t]) / R

    # Compute income
    k_share = weights[:, None] * k_agg[0, :]
    a_total = k_share + k_hat

    # Lagged assets
    a_lag = np.concatenate([a_total[:, [0]], a_total[:, :-1]], axis=1)

    # Net income: dividend share + asset return
    y_net = weights[:, None] * d_agg[0, :] + (R - 1) * a_lag

    return {"c": c_j, "y_net": y_net, "χ_tilde": χ_tilde, "k_hat": k_hat, "a_total": a_total}
```

```{code-cell} ipython3
μ_values = np.asarray(paths["μ"]).reshape(-1)

R = float(δ_k + γ_1)

# Redistribution parameters (same as visualization above)
red_α = 0.8
red_β = 0.0

idx_sorted = np.argsort(-μ_values)
λ_orig_sorted = μ_values[idx_sorted]

λ_star_sorted = create_redistributed_weights(λ_orig_sorted, α=red_α, β=red_β)
λ_star = np.empty_like(μ_values, dtype=float)
λ_star[idx_sorted] = λ_star_sorted

print(f"Weight redistribution:")
print(f"  std(μ): {np.std(μ_values):.4f} → std(λ*): {np.std(λ_star):.4f}")
print(f"  p90/p10: {np.percentile(μ_values, 90)/np.percentile(μ_values, 10):.2f} → {np.percentile(λ_star, 90)/np.percentile(λ_star, 10):.2f}")

h0i_alloc = np.array([[0.0]])

pre = allocation_from_weights(paths, econ, Ub_list, μ_values, γ_1, Λ, h0i_alloc)
post = allocation_from_weights(paths, econ, Ub_list, λ_star, γ_1, Λ, h0i_alloc)

c_pre = pre["c"][:, t0:]
y_pre = pre["y_net"][:, t0:]
c_post = post["c"][:, t0:]
y_post = post["y_net"][:, t0:]

def _pct(panel, ps=(90, 50, 10)):
    return np.percentile(panel, ps, axis=0)

T_ts = min(500, c_pre.shape[1] - 50)
t = t0 + np.arange(T_ts)

y_pre_pct = _pct(y_pre[:, :T_ts])
y_post_pct = _pct(y_post[:, :T_ts])
c_pre_pct = _pct(c_pre[:, :T_ts])
c_post_pct = _pct(c_post[:, :T_ts])

print(f"\nRedistribution effects:")
print(f"  income dispersion: {np.std(y_pre):.3f} → {np.std(y_post):.3f}")
print(f"  consumption dispersion: {np.std(c_pre):.3f} → {np.std(c_post):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

ax[0].plot(t, y_pre_pct[0], label="p90", lw=2)
ax[0].plot(t, y_pre_pct[1], label="p50", lw=2)
ax[0].plot(t, y_pre_pct[2], label="p10", lw=2)
ax[0].set_xlabel("time")
ax[0].set_ylabel(r"pre-tax income $y^{pre}$")
ax[0].legend()

ax[1].plot(t, y_post_pct[0], label="p90", lw=2)
ax[1].plot(t, y_post_pct[1], label="p50", lw=2)
ax[1].plot(t, y_post_pct[2], label="p10", lw=2)
ax[1].set_xlabel("time")
ax[1].set_ylabel(r"post-tax income $y^{post}$")
ax[1].legend()

ax[2].plot(t, c_post_pct[0], label="p90", lw=2)
ax[2].plot(t, c_post_pct[1], label="p50", lw=2)
ax[2].plot(t, c_post_pct[2], label="p10", lw=2)
ax[2].set_xlabel("time")
ax[2].set_ylabel(r"consumption $c^{post}$")
ax[2].legend()

plt.tight_layout()
plt.show()
```
