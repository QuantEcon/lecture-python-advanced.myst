---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Repeated Moral Hazard

## Overview

This lecture computes the information-constrained optima studied by
{cite:t}`Phelan_Townsend_91`.

Their paper studies a continuum-agent economy with unobserved effort.

The planner chooses lotteries over individual histories, subject to
promise-keeping and incentive-compatibility constraints, and maximizes
discounted social surplus.

The key recursive idea comes from {cite:t}`Spear_Srivastava_87`: an
agent's promised continuation utility is a sufficient state variable.

Phelan and Townsend combine that idea with lotteries, finite grids, and
linear programming to compute full-information, static
unobserved-action, and repeated unobserved-action allocations.

We proceed as follows.

*  We review the promised-utility recursion of
   {cite:t}`Spear_Srivastava_87`.
*  We formulate the Phelan-Townsend lottery problem and its finite-grid
   linear-programming approximation.
*  We solve the *static* version of the economy and
   replicate Figures 1--4 of {cite:t}`Phelan_Townsend_91`.
*  We solve the *repeated* economy and replicate
   Figures 5--12 of {cite:t}`Phelan_Townsend_91`.


## Promised-utility Recursion

{cite:t}`Spear_Srivastava_87` showed how to write an infinitely repeated,
discounted principal-agent problem recursively.

*  A **principal** owns a technology that produces output $q_t$ at
   time $t$ according to a conditional distribution $F(q_t | a_t)$
   that depends on the **effort** $a_t$ chosen by an **agent**.
*  The principal does *not* observe $a_t$.
*  The principal *does* observe $q_t$ at the end of period $t$ and
   remembers the full history $\{q_s\}_{s=0}^t$.
*  The principal is risk-neutral and has access to a loan market with
   gross risk-free interest rate $\beta^{-1}$.
*  The agent has preferences over random consumption streams ordered
   by $E_0 \sum_{t=0}^\infty \beta^t u(c_t, a_t)$, where $u$ is
   increasing in $c$ and decreasing in $a$.

A **contract** recommends effort before output is realized and then
assigns consumption and continuation promises as functions of observed
output histories.

The principal designs the contract to maximize expected discounted
surplus $E_0 \sum_{t=0}^\infty \beta^t \{q_t - c_t\}$.

Let $w$ denote the discounted expected continuation utility that the
principal has promised to the agent at the start of a period.

The promised utility $w$ summarizes payoff-relevant history.  

Given
$w$, the principal chooses a recommended action $a(w)$, an
output-contingent consumption rule $c(w,q)$, and an output-contingent
next-period promise $\tilde w(w,q)$ subject to

$$
w = \int \bigl\{ u[c(w,q),\, a(w)] + \beta\,\tilde{w}(w,q)
    \bigr\}\, dF[q \mid a(w)]
$$ (eq:eq1)

and, for all alternative actions $\hat a \in A$,

$$
\int \bigl\{ u[c(w,q),\, a(w)] + \beta\,\tilde{w}(w,q) \bigr\}\,
dF[q\mid a(w)]
\;\geq\;
\int \bigl\{ u[c(w,q),\, \hat a] + \beta\,\tilde{w}(w,q) \bigr\}\,
dF[q\mid\hat a].
$$ (eq:eq2)

Equation {eq}`eq:eq1` is the **promise-keeping** constraint: the
contract must deliver the promised continuation utility $w$.

Equation {eq}`eq:eq2` is the **incentive-compatibility** constraint:
the agent must prefer the recommended action $a(w)$ over any
deviation $\hat a$.

The principal's value function $v(w)$ -- the maximum expected
discounted surplus attainable when the agent has been promised $w$ --
satisfies the Bellman equation

$$
v(w) = \max_{a,\,c,\,\tilde{w}}\
\int \bigl\{q - c(w,q) + \beta\, v[\tilde{w}(w,q)]\bigr\}\,
dF[q\mid a(w)]
$$ (eq:eq3)

subject to the promise-keeping constraint {eq}`eq:eq1` and the
incentive-compatibility constraint {eq}`eq:eq2`.


## Phelan and Townsend (1991): Lotteries and Linear Programming

A technical difficulty in problems like {eq}`eq:eq3` is that
incentive constraints can make deterministic contract problems
non-convex.

{cite}`Phelan_Townsend_91` instead formulate the planning problem in
terms of **lotteries** over actions, outputs, consumptions, and
continuation utilities.

At the aggregate level these probabilities
are also population fractions, so individual randomization creates no
aggregate uncertainty in their continuum-agent economy.

For computation, all relevant sets are restricted to finite grids.

On
those grids the Bellman step is a **linear program**.

*Setup.* Let $P(q | a)$ be a family of discrete conditional
probability distributions over finite sets $Q$ (outputs) and $A$
(actions).

Let $C$ and $W'$ be finite grids for current consumption and next-period
promised utility.

For each current promise $w$, the planner chooses a joint probability
$\Pi^w(a, q, c, w')$ subject to:

$$
\sum_{c \in C}\sum_{w' \in W'} \Pi^w(\bar a, \bar q, c, w') =
P(\bar q \mid \bar a)\,
\sum_{q \in Q}\sum_{c \in C}\sum_{w' \in W'}
    \Pi^w(\bar a, q, c, w'),
\quad \forall\, \bar a,\, \bar q
$$ (eq:town1a)

$$
\Pi^w(a, q, c, w') \geq 0,
\qquad
\sum_{a \in A}\sum_{q \in Q}\sum_{c \in C}\sum_{w' \in W'}
    \Pi^w(a, q, c, w') = 1.
$$ (eq:town1b)

Equation {eq}`eq:town1a` says that conditional on action $\bar a$,
the marginal distribution of output follows $P(q \mid \bar a)$.

Equations {eq}`eq:town1b` require the choice variables to be proper
probabilities.

The **promise-keeping** constraint is

$$
w = \sum_{A \times Q \times C \times W'}
\bigl\{u(c, a) + \beta w'\bigr\}\,\Pi^w(a, q, c, w').
$$ (eq:eq1prime)

The **incentive-compatibility** constraint, for each pair
$(a, \hat a)$, is

$$
\sum_{Q \times C \times W'} \bigl\{u(c,a) + \beta w'\bigr\}\,
\Pi^w(a, q, c, w')
\;\geq\;
\sum_{Q \times C \times W'} \bigl\{u(c,\hat a) + \beta w'\bigr\}\,
\frac{P(q\mid\hat a)}{P(q\mid a)}\,\Pi^w(a, q, c, w').
$$ (eq:eq2prime)

The ratio $P(q\mid\hat a)/P(q\mid a)$ is the likelihood ratio that
updates the probability of outcome $q$ when the agent deviates from
the recommended action $a$ to $\hat a$.

*Bellman operator as a linear program.* The principal's value
function satisfies

$$
v(w) = \max_{\Pi}\
\sum_{A \times Q \times C \times W'}
\bigl\{(q - c) + \beta\, v(w')\bigr\}\,\Pi^w(a, q, c, w'),
$$ (eq:bell2)

where the maximization is over probabilities $\Pi$ satisfying
{eq}`eq:town1a`, {eq}`eq:town1b`, {eq}`eq:eq1prime`, and
{eq}`eq:eq2prime`.

This is a **linear program**: the objective and all constraints are
linear in the decision variables $\Pi^w$.

Because $v(w')$ on the right side of {eq}`eq:bell2` is treated as a
*fixed* vector from the previous iteration, the Bellman operator
itself is a linear program.

Phelan and Townsend solve one LP for each grid point $w \in W$ and
iterate on the surplus function.

Their Theorem 4 gives the
contraction result that justifies this iteration for the
infinite-horizon problem.

## Implementation

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
!pip install cvxpy
```

We import some Python packages.

```{code-cell} ipython3
import numpy as np
import cvxpy as cp
from time import time
import gc
import matplotlib.pyplot as plt
from warnings import filterwarnings
```

## The Static Economy

This section replicates Sections II and III of
{cite}`Phelan_Townsend_91`.

Section II studies the full-information benchmark.

Section III adds
unobserved actions and the resulting incentive constraints.

### Setting

The **full-information problem** (FIP) maximizes expected surplus
subject only to feasibility and promise keeping:

$$
\max_{\Pi^w}\; \sum_{A \times Q \times C} (q - c)\,\Pi^w(a, q, c)
$$

subject to

$$
\begin{aligned}
\text{C1:}&\quad
  w = \sum_{A \times Q \times C} U(a,c)\,\Pi^w(a,q,c) \\[4pt]
\text{C2:}&\quad
  \sum_C \Pi^w(\bar a, \bar q, c)
  = P(\bar q \mid \bar a)
    \sum_{Q \times C} \Pi^w(\bar a, q, c),
  \quad \forall\, \bar a,\, \bar q \\[4pt]
\text{C3:}&\quad
  \sum_{A \times Q \times C} \Pi^w(a,q,c) = 1,
  \quad \Pi^w(a,q,c) \geq 0
\end{aligned}
$$

The **unobserved-action problem** adds incentive compatibility.  For
each recommended action $a$ and each possible deviation $\hat a$, the
utility from obeying must be at least as large as the utility from
deviating while preserving the same output-contingent consumption rule:

$$
\text{C4:}\quad
\sum_{Q \times C} U(a,c)\,\Pi^w(a,q,c)
\;\geq\;
\sum_{Q \times C} U(\hat a, c)\,
\frac{P(q\mid\hat a)}{P(q\mid a)}\,\Pi^w(a,q,c),
\quad \forall\, a,\, \hat a \in A.
$$

### Parameterisation

Following {cite}`Phelan_Townsend_91`, we use the period utility
function

$$
U(a, c) = 2\sqrt{c} + 2\sqrt{1-a}
$$

with discrete grids

| Variable | Values |
| :------- | :----- |
| $a \in A$ | $\{0,\; 0.2,\; 0.4,\; 0.6\}$ |
| $q \in Q$ | $\{1,\; 2\}$ |
| $c \in C$ | $81$ equally spaced points on $[0,\, 2.25]$ |

and conditional output probabilities

|   $a$   | $P(q=1)$ | $P(q=2)$ |
| :-----: | :------: | :------: |
|   0     |   0.9    |   0.1    |
|  0.2    |   0.6    |   0.4    |
|  0.4    |   0.4    |   0.6    |
|  0.6    |   0.25   |   0.75   |

These are the parameter values used to construct Figures 1--8 in the
paper.

The static grid of promised utility values below spans the
interval $[1,5]$, matching the horizontal scale in Figures 1--4.

```{code-cell} ipython3
def u(a, c):
    return c**0.5 / 0.5 + (1 - a)**0.5 / 0.5

A = np.array([0, 0.2, 0.4, 0.6])
Q = np.array([1, 2])
C = np.linspace(0, 2.25, 81)
P = np.array([[0.9, 0.1],
              [0.6, 0.4],
              [0.4, 0.6],
              [0.25, 0.75]])
```

### Solving the Static Problem

The function `solve_static_problem` solves both the FIP and the
unobserved-action problem for an array of promised utility values $w$.

It implements constraints C1--C3 for the full information case and
adds C4 for the unobserved-action case.

```{code-cell} ipython3
# Define the function that solves the static problem
def solve_static_problem(W=None,
                         u=None,
                         A=None,
                         Q=None,
                         C=None,
                         P=None,
                         problem_type=None):
    '''
    Function: Solve the static problem
    
    Parameters
    ----------
    W: 1-D array
        The expected utility.
    u: function
        The utility function in terms of actions and consumptions.
    A: 1-D array
        The finite set of possible actions.
    Q: 1-D array
        The finite set of possible outputs.
    C: 1-D array
        The finite set of possible consumptions.
    P: 2-D array
        The probability matrix of outputs given an action.
    problem_type: str, "full information" or "unobserved-actions"
        The problem type, i.e. the full information problem or the unobserved-action problem.
        
    Returns
    -------
    s_W: 1-D array
        The optimal values of surplus for each w in w_vec.
    Pi: 4-D array
        The probability of (a, q, c) given w.
    '''
    
    # Define parameter
    n_A, n_Q, n_C = len(A), len(Q), len(C)
    A_ind, Q_ind, C_ind = range(n_A), range(n_Q), range(n_C)
    
    Phi = np.array([[q-c for c in C] for q in Q])
    U = np.array([[u(a, c) for c in C] for a in A])
    
    w = cp.Parameter()
        
    # Define variable Pi_x
    Pi_list = list(np.zeros(n_A))
    
    for a_ind in A_ind:
        Pi_list[a_ind] = cp.Variable((n_Q, n_C))

    # Define objective function
    obj_expr = cp.sum([cp.sum(cp.multiply(Pi_list[a_ind], Phi))
                       for a_ind in A_ind])
    obj = cp.Maximize(obj_expr)
    
    # Define constraints
    C1 = [cp.sum([cp.sum([cp.sum(cp.multiply(Pi_list[a_ind][q_ind, :],
                                             U[a_ind, :])) for a_ind in A_ind]) 
                  for q_ind in Q_ind]) == w]
    C2 = [(cp.sum(Pi_list[a_ind], axis=1)[q_ind] == P[a_ind, q_ind] * cp.sum(Pi_list[a_ind])) 
          for a_ind in A_ind for q_ind in Q_ind]
    C3 = [cp.sum([cp.sum(Pi_list[a_ind]) for a_ind in A_ind]) == 1] + \
            [(Pi_list[a_ind] >= 0) for a_ind in A_ind]
        
    problem_type = problem_type.lower()
    if problem_type == "full information":
        constraints = C1 + C2 + C3
    else:
        C4 = [(cp.sum([cp.sum(cp.multiply(Pi_list[a_ind][q_ind, :], U[a_ind, :]))
                       for q_ind in Q_ind]) >= 
               cp.sum([cp.sum(cp.multiply(Pi_list[a_ind][q_ind, :],
                                          U[a_ind_hat, :])) * P[a_ind_hat, q_ind]/P[a_ind, q_ind] 
                       for q_ind in Q_ind])) 
              for a_ind in A_ind for a_ind_hat in A_ind]
        constraints = C1 + C2 + C3 + C4

    # Create the problem
    problem = cp.Problem(obj, constraints)
    
    # Initialize output variables
    s_W = np.zeros(len(W))
    Pi = np.zeros((len(W), len(A), len(Q), len(C)))
    
    # Solve the problem
    for i in range(len(W)):
        w.value = W[i]
        problem.solve(solver=cp.HIGHS)
        s_W[i] = obj_expr.value
        for a_ind in A_ind:
            Pi[i, a_ind, :, :] = Pi_list[a_ind].value
    
    return s_W, Pi
```

### Figures 1-4

```{note}
Phelan and Townsend report solutions computed with standard revised
simplex methods.  We use HiGHS through CVXPY.  At degenerate utility
grid points, a different LP solver can select a different optimal
lottery, so some consumption schedules can differ slightly even when
the surplus function is unchanged.
```

```{code-cell} ipython3
W_static = np.linspace(1, 5, 100)
filterwarnings("ignore")

s_W_full, Pi_full = solve_static_problem(W_static, u, A, Q, C, P,
                                          "full information")
s_W_unobs, Pi_unobs = solve_static_problem(W_static, u, A, Q, C, P,
                                            "unobserved-actions")
```

```{code-cell} ipython3
# Figure 1 – Surplus functions
plt.figure(figsize=(6.5, 6.5))
plt.plot(W_static, s_W_full)
plt.plot(W_static, s_W_unobs)
plt.hlines(0, 1.0, 5.0, linestyle="dashed")
plt.xlabel("w")
plt.ylabel("s(w)")
plt.xlim([1.0, 5.0])
plt.ylim([-1.5, 2.0])
plt.title("Figure 1\n Optimized surplus function", y=-0.2)
plt.text(2.5, 1.6, "Full Information", size=15)
plt.text(1.5, 0.8, "Unobserved Action", size=15)
plt.show()
```

```{code-cell} ipython3
# Figure 2 – Expected effort
Ea_full  = np.einsum('a,waqc->w', A, Pi_full)
Ea_unobs = np.einsum('a,waqc->w', A, Pi_unobs)

plt.figure(figsize=(6.5, 6.5))
plt.plot(W_static, Ea_full)
plt.plot(W_static, Ea_unobs)
plt.xlabel("w")
plt.ylabel("E{a(w)}")
plt.xlim([1.0, 5.0])
plt.ylim([0.0, 0.8])
plt.title("Figure 2\n Actions", y=-0.2)
plt.text(2.3, 0.65, "Full Information", size=15)
plt.text(2.6, 0.15, "Unobserved Action", size=15)
plt.show()
```

```{code-cell} ipython3
# Figure 3 – Unobserved-action consumption schedule
Pi_sum_unobs = Pi_unobs.sum(axis=-1)  # shape (W, A, Q)
Ec_unobs = (np.einsum('c,waqc->waq', C, Pi_unobs)
            / np.where(Pi_sum_unobs > 1e-12, Pi_sum_unobs, 1.0))

l, m = len(A), len(Q)
X, Y = range(l), range(m)

plt.figure(figsize=(6.5, 6.5))
for x in X:
    for y in Y:
        plt.plot(W_static, Ec_unobs[:, x, y])
plt.xlabel("w")
plt.ylabel("E(c) given a, q, w")
plt.xlim([1.0, 5.0])
plt.ylim([0.0, 2.25])
plt.title("Figure 3\n Unobserved Action Consumption", y=-0.3)
plt.annotate("a=.4, q=2", xy=(2.5, 0.5), xytext=(1.3, 0.7),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=2", xy=(3.7, 1.5), xytext=(2.2, 1.65),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)", xy=(4.8, 2.05), xytext=(3.0, 2.15),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2,\nq=2", xy=(2.0, 0.15), xytext=(1.3, 0.24),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.4, q=1", xy=(3.0, 0.10), xytext=(3.6, 0.2),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2,\nq=1", xy=(4.0, 0.9), xytext=(4.3, 0.75),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)\na=.2, q=1", xy=(2.1, 0),
             xytext=(1.8, -0.3), arrowprops={"arrowstyle":"-"})
plt.annotate(r"$\{$", fontsize=25, xy=(2.1, 0), xytext=(1.6, -0.3))
plt.annotate(r"$\}$", fontsize=25, xy=(2.1, 0), xytext=(2.5, -0.3))
plt.show()
```

```{code-cell} ipython3
# Figure 4 – Full-information consumption schedule
Pi_sum_full = Pi_full.sum(axis=-1)  # shape (W, A, Q)
Ec_full = (np.einsum('c,waqc->waq', C, Pi_full)
           / np.where(Pi_sum_full > 1e-12, Pi_sum_full, 1.0))

plt.figure(figsize=(6.5, 6.5))
for x in X:
    for y in Y:
        plt.plot(W_static, Ec_full[:, x, y])
plt.xlabel("w")
plt.ylabel("E{c(w)}")
plt.xlim([1.0, 5.0])
plt.ylim([0.0, 2.25])
plt.title("Figure 4\n Full Information Consumption", y=-0.2)
plt.show()
```

## The Repeated Economy

We now move from the one-period economy to the finite- and
infinite-horizon economies studied in Section IV of
{cite}`Phelan_Townsend_91`.

The planner maximizes discounted social surplus.

As in the paper, this
can be interpreted as allowing society to borrow and lend at the constant
gross interest rate $\beta^{-1}$, so that discounted surplus is the
right feasibility criterion.

### Formulation

The recursive repeated problem chooses today's action, output,
consumption, and next-period promised utility.

The surplus function
$s(w)$ satisfies

$$
s(w) = \max_{\Pi^w}\;
\sum_{A \times Q \times C \times W'}
\bigl\{(q - c) + \beta\, s(w')\bigr\}\,\Pi^w(a, q, c, w')
$$

subject to, for all $(\bar a, \bar q)$,

$$
\begin{aligned}
\text{C5:}&\quad
  w = \sum_{A \times Q \times C \times W'}
      \bigl\{U(a,c) + \beta w'\bigr\}\,\Pi^w(a,q,c,w') \\[4pt]
\text{C6:}&\quad
  \sum_{C \times W'} \Pi^w(\bar a, \bar q, c, w')
  = P(\bar q\mid\bar a)
    \sum_{Q \times C \times W'} \Pi^w(\bar a, q, c, w') \\[4pt]
\text{C7:}&\quad
  \sum_{A \times Q \times C \times W'} \Pi^w(a,q,c,w') = 1,
  \quad \Pi^w(a,q,c,w') \geq 0 \\[4pt]
\text{C8:}&\quad
  \sum_{Q \times C \times W'}
    \bigl\{U(a,c)+\beta w'\bigr\}\Pi^w(a,q,c,w')
  \;\geq\;
  \sum_{Q \times C \times W'}
    \bigl\{U(\hat a,c)+\beta w'\bigr\}
    \frac{P(q\mid\hat a)}{P(q\mid a)}\Pi^w(a,q,c,w'),
  \quad \forall\, a,\, \hat a.
\end{aligned}
$$

Constraints C5--C8 are the dynamic analogues of C1--C4.

* Constraint C5 keeps the promise $w$.

* Constraint C6 maintains the action-output law of motion.

* Constraint C7 is a probability constraint.
  
* Constraint C8 is incentive compatibility.

For a finite horizon, the one-period surplus function is used to solve
the two-period problem, the two-period surplus function is used to solve
the three-period problem, and so on.

For the infinite horizon, we
iterate on the Bellman operator until the surplus function is stable.

At each iteration, a separate LP is solved for each grid point
$w \in W$.

### The Two-Step Factored Algorithm

Solving the full LP over $(a,q,c,w')$ at each grid point is
computationally demanding.

Section VI of {cite}`Phelan_Townsend_91` proposes a factored
algorithm that splits each period into two sub-steps, exploiting the
additive separability of the utility function

$$
U(a, c) = 2\sqrt{1-a} + 2\sqrt{c}.
$$

*Step 1* (action and output, before consumption is assigned).

Let $w^m$ be the **intermediate** promised utility after the output
is observed but before consumption is allocated.

Thus $w^m$ includes
the utility from current consumption and the discounted next-period
promise, but not the current effort utility.

Solve

$$
\begin{aligned}
\max_{\Pi^w}\; &
\sum_{A \times Q \times W^m}
  \bigl\{q + s^m(w^m)\bigr\}\,\Pi^w(a, q, w^m) \\[4pt]
\text{C5:}&\quad
  w = \sum_{A \times Q \times W^m}
      \bigl\{2\sqrt{1-a} + w^m\bigr\}\,\Pi^w(a, q, w^m) \\[4pt]
\text{C6:}&\quad
  \sum_{W^m} \Pi^w(\bar a, \bar q, w^m)
  = P(\bar q\mid\bar a)
    \sum_{Q \times W^m} \Pi^w(\bar a, q, w^m) \\[4pt]
\text{C7:}&\quad
  \Pi^w(a,q,w^m) \geq 0,\quad
  \sum_{A \times Q \times W^m} \Pi^w(a,q,w^m) = 1 \\[4pt]
\text{C8:}&\quad
  \sum_{Q \times W^m}
    \bigl\{2\sqrt{1-a} + w^m\bigr\}\Pi^w(a,q,w^m)
  \;\geq\;
  \sum_{Q \times W^m}
    \bigl\{2\sqrt{1-\hat a} + w^m\bigr\}
    \frac{P(q\mid\hat a)}{P(q\mid a)}\Pi^w(a,q,w^m),
  \quad \forall\, a,\, \hat a.
\end{aligned}
$$

*Step 2* (consumption allocation).
Given $w^m$, solve

$$
\begin{aligned}
\max_{\Pi^{w^m}}\; &
\sum_{C \times W'}
  \bigl\{\beta s(w') - c\bigr\}\,\Pi^{w^m}(c, w') \\[4pt]
\text{C5:}&\quad
  w^m = \sum_{C \times W'}
         \bigl\{2\sqrt{c} + \beta w'\bigr\}\,\Pi^{w^m}(c, w') \\[4pt]
\text{C7:}&\quad
  \Pi^{w^m}(c,w') \geq 0,\quad
  \sum_{C \times W'} \Pi^{w^m}(c,w') = 1.
\end{aligned}
$$

Step 2 is solved first computationally, for all $w^m \in W^m$, to
obtain $s^m(w^m)$.

Step 1 then uses this intermediate surplus function
as input.

This factored algorithm significantly reduces computation time because
each sub-LP is smaller than the original joint LP.

The two-step formulation is an approximation when $W^m$ is
discretized: as the number of grid points $N_m$ grows, it converges
to the exact solution.

### Functions

The function `solve_repeated_problem_2` implements one Bellman
iteration using the two-step algorithm.

The function `solve_multi_period_economy_2` iterates to convergence
(or for a fixed number of periods $T$).

```{code-cell} ipython3
# Define the function that solves the dynamic problem at one iteration
def solve_repeated_problem_2(W=None,
                             W_m=None,
                             A=None,
                             Q=None,
                             C=None,
                             W_prime=None,
                             s_W_prime=None,
                             P=None,
                             problem_type=None,
                             β=0.8):
    '''
    Function: Solve the dynamic problem at one iteration
    
    Parameters
    ----------
    W: 1-D array
        The expected utility.
    A: 1-D array
        The finite set of possible actions.
    Q: 1-D array
        The finite set of possible outputs.
    C: 1-D array
        The finite set of possible consumptions.
    W_prime: 1-D array
        The finite set of possible w_prime.
    s_W_prime: 1-D array
        The finite set of optimal values of surplus of w_prime.
    P: 2-D array
        The probability matrix of outputs given an action.
    problem_type: str, "full information" or "unobserved-actions"
        The problem type, i.e. the full information problem or the unobserved-action problem.
    β: float, optional
        The discount factor. The value is 0.8 by default.
        
    Returns
    -------
    s_W: 1-D array
        The optimal values of surplus for each w in w_vec.
    Pi_W_s1: 4-D array
        The probability of (a, q, w_m) given w.
    Pi_W_m_s2: 3-D array
        The probability of (c, w_prime) given w_m.
    '''
    
    n_A, n_Q, n_C, n_W = len(A), len(Q), len(C), len(W) 
    n_W_m, n_W_prime = len(W_m), len(W_prime)
    A_ind, Q_ind, C_ind = range(n_A), range(n_Q), range(n_C)
    W_ind, W_m_ind, W_prime_ind = range(n_W), range(n_W_m), range(n_W_prime)
    
    # Problem of step 2
    
    # Define parameters
    Phi_s2 = np.array([[β * s_w_prime - c for s_w_prime in s_W_prime] for c in C])
    U_disc_s2 = np.array([[2 * c**0.5 + β * w_prime
                           for w_prime in W_prime] for c in C])
    
    w_m_para = cp.Parameter()
    
    
    # Define variables
    Pi_w_m = cp.Variable((n_C, n_W_prime))

    # Define the objective function
    obj_expr_s2 = cp.sum(cp.multiply(Phi_s2, Pi_w_m))
    obj_s2 = cp.Maximize(obj_expr_s2)
    
    # Define constraints
    C5_s2 = [cp.sum(cp.multiply(U_disc_s2, Pi_w_m)) == w_m_para]
    C7_s2 = [cp.sum(Pi_w_m) == 1] + [Pi_w_m >= 0]
    
    # Create the problem of step 2
    problem_s2 = cp.Problem(obj_s2, C5_s2 + C7_s2)
    
    # Solve the problem of step 2
    s_W_m = np.zeros(n_W_m)
    Pi_W_m_s2 = np.zeros((n_W_m, n_C, n_W_prime))
    for w_m, w_m_ind in zip(W_m, W_m_ind):
        w_m_para.value = w_m
        problem_s2.solve(solver = cp.HIGHS)
        s_W_m[w_m_ind] = obj_expr_s2.value
        Pi_W_m_s2[w_m_ind, :, :] = Pi_w_m.value
    
    # Problem of step 1
    
    # Define parameters
    Phi_s1 = np.array([[(q+s_w_m) for s_w_m in s_W_m] for q in Q])
    U_disc_s1 = np.array([[[2 * (1 - a)**0.5 + w_m
                            for w_m in W_m] for q in Q] for a in A])
    U_disc_hat_s1 = np.array([[[[(2 * (1 - A[a_hat_ind])**0.5 + W_m[w_m_ind]) *\
                                 P[a_hat_ind, q_ind]/P[a_ind, q_ind] 
                                 for w_m_ind in W_m_ind] for q_ind in Q_ind]
                               for a_ind in A_ind] 
                              for a_hat_ind in A_ind])
    
    w_para = cp.Parameter()
    
    # Define variables
    Pi_w_list = list(np.zeros(n_A))
    for a_ind in A_ind:
        Pi_w_list[a_ind] = cp.Variable((n_Q, n_W_m))
    
    # Define the objective function
    obj_expr_s1 = cp.sum([cp.sum(cp.multiply(Phi_s1, Pi_w_list[a_ind]))
                          for a_ind in A_ind])
    obj_s1 = cp.Maximize(obj_expr_s1)
                                       
    # Define constraints
    C5_s1 = [cp.sum([cp.sum(cp.multiply(U_disc_s1[a_ind, :, :],
                                        Pi_w_list[a_ind]))
                     for a_ind in A_ind]) == w_para]
    C6_s1 = [(cp.sum(Pi_w_list[a_ind][q_ind, :]) == P[a_ind, q_ind] *\
              cp.sum(Pi_w_list[a_ind])) 
             for q_ind in Q_ind for a_ind in A_ind]
    C7_s1 = [cp.sum([cp.sum(Pi_w_list[a_ind]) for a_ind in A_ind]) == 1]
    C7_s1 = C7_s1 + [(Pi_w_list[a_ind] >= 0) for a_ind in A_ind]
    
    problem_type = problem_type.lower()
    if problem_type == "full information":
        constraints_s1 = C5_s1 + C6_s1 + C7_s1
    else:
        C8_s1 = [(cp.sum(cp.multiply(U_disc_s1[a_ind, :, :],
                                     Pi_w_list[a_ind])) >= 
                 cp.sum(cp.multiply(U_disc_hat_s1[a_hat_ind, a_ind, :, :],
                                    Pi_w_list[a_ind]))) 
                 for a_ind in A_ind for a_hat_ind in A_ind] 
        constraints_s1 = C5_s1 + C6_s1 + C7_s1 + C8_s1
    
    # Create the problem of step 1
    problem_s1 = cp.Problem(obj_s1, constraints_s1)
    
    # Solve the problem of step 1
    s_W = np.zeros(n_W)
    Pi_W_s1 = np.zeros((n_W, n_A, n_Q, n_W_m))
    for w, w_ind in zip(W, W_ind):
        w_para.value = w
        problem_s1.solve(solver = cp.HIGHS)
        s_W[w_ind] = obj_expr_s1.value
        for a_ind in A_ind:
            Pi_W_s1[w_ind, a_ind, :, :] = Pi_w_list[a_ind].value                
    return s_W, Pi_W_s1, Pi_W_m_s2
```

```{code-cell} ipython3
# Define the function that solves the infinite-period or finite-period economy
def solve_multi_period_economy_2(A=None,
                                 Q=None,
                                 C=None,
                                 P=None,
                                 problem_type=None,
                                 T=None,
                                 β=0.8,
                                 N=100,
                                 N_m=100,
                                 s_W_0=None,
                                 tol=1e-8,
                                 verbose=False):
    '''
    Function: Solve the multi-period problem, either infinite-period or finite-period
    
    Parameters
    ----------
    A: 1-D array
        The finite set of possible actions.
    Q: 1-D array
        The finite set of possible outputs.
    C: 1-D array
        The finite set of possible consumptions.
    P: 2-D array
        The probability matrix of outputs given an action.
    problem_type: str, "full information" or "unobserved-actions"
        The problem type, i.e. the full information problem or the unobserved-action problem.
    T: int, optional
        The number of periods. If T is None, the algorithm solves the infinite-period economy. If T is some
        integer, the algorithm solves the T-period economy. By default, T is None.
    β: float, optional
        The discount factor in (0,1). The value is 0.8 by default.
    N: int, optional
        The length of discretized parameter space W.
    N_m: int, optional
        The length of discretized parameter space W_m.
    s_W_0: 1-D array, optional
        The initial guess for s_W with a length of N.
    tol: float, optional
        The precision of convergence.
    verbose: bool, optional
        If True, print progress at each iteration. The default is False.

    Returns
    -------
    s_W: 1-D array
        The optimal values of convergent surplus for each w in w_vec.
    Pi_W_s1: 4-D array
        The probability of (a, q, w_m) given w.
    Pi_W_m_s2: 3-D array
        The probability of (c, w_prime) given w_m.
    '''
    
    if β >= 1 or β <= 0:
        raise ValueError('β must lie in (0, 1)')
        
    # Define the function u[a,c]
    def u(a, c):
        return c**0.5/0.5 + (1-a)**0.5/0.5
        
    if T is None:
        # Discretize the parameter space W and W_m
        problem_type = problem_type.lower()
        if problem_type == "full information":
            w_l = u(A.max(), C.min())/(1 - β)
            w_u = u(A.min(), C.max())/(1 - β)
        else:
            w_l = u(A.min(), C.min())/(1 - β)
            w_u = u(A.min(), C.max())/(1 - β)
        W = np.linspace(w_l, w_u, N)
        
        W_m_l = β * w_l + 2 * C.min()**0.5
        W_m_u = β * w_u + 2 * C.max()**0.5
        W_m = np.linspace(W_m_l, W_m_u, N_m)

        # Assign initial value for s_W
        if s_W_0 is not None:
            s_W_prime = s_W_0
        else:
            s_W_prime = np.zeros(N)

        # Iterate
        optimal = False
        iteration = 1
        while not optimal:
            if verbose:
                print('Iteration %i in process' % iteration)
            start_time = time()
            s_W, Pi_W_s1, Pi_W_m_s2 = solve_repeated_problem_2(W=W, W_m=W_m,
                                                               A=A, Q=Q,
                                                               C=C, W_prime=W,
                                                               s_W_prime=s_W_prime,
                                                               P=P,
                                                               problem_type=problem_type,
                                                               β=β)
            end_time = time()
            if verbose:
                print('Iteration %i finished in:' % iteration,
                      round(end_time - start_time, 3), 's')
                print('---------')
            
            if np.max(np.abs(s_W - s_W_prime)) <= tol:
                optimal = True
            else:
                s_W_prime = s_W
                
            iteration += 1
    
    if T is not None:
        # Discretize the parameter space W
        W_mat = np.zeros((T, N))
        
        problem_type = problem_type.lower()
        if problem_type == "full information":
            w_l = u(A.max(), C.min())
            w_u = u(A.min(), C.max())
        else:
            w_l = u(A.min(), C.min())
            w_u = u(A.min(), C.max())
        W_mat = np.cumsum(np.logspace(0, T-1, T, base=β).reshape(T, 1) *\
                          np.linspace(w_l, w_u, N).reshape(1, N), 
                          axis=0)
        
        # Solve the 1-period economy
        if verbose:
            print('Solving the 1-period economy')
            print('-------')
        s_W, Pi = solve_static_problem(W=W_mat[0, :], u=u,
                                       A=A, Q=Q, C=C, P=P,
                                       problem_type=problem_type)

        if T != 1:
            for t in range(2, T+1):
                if verbose:
                    print('Solving the %i-period economy' % t)
                    print('-------')
                s_W_prime = np.copy(s_W)
                W_m_l = β*W_mat[t-2,:].min() + 2*C.min()**0.5
                W_m_u = β*W_mat[t-2,:].max() + 2*C.max()**0.5
                W_m = np.linspace(W_m_l, W_m_u, N_m)
                s_W, Pi_W_s1, Pi_W_m_s2 = solve_repeated_problem_2(W=W_mat[t-1,:],
                                                                   W_m=W_m, A=A,
                                                                   Q=Q, C=C, 
                                                                   W_prime=W_mat[t-2,:],
                                                                   s_W_prime=s_W_prime,
                                                                   P=P, 
                                                                   problem_type=problem_type,
                                                                   β=β)
    return s_W, Pi_W_s1, Pi_W_m_s2
```

### Improved Solver: Pre-built Problems with Anderson Acceleration

The original solver rebuilds all CVXPY problem objects on every
Bellman iteration, which causes memory to accumulate when many
iterations are needed -- a serious issue for $\beta$ close to 1.

The function `solve_multi_period_economy_vfi` fixes this by building
the two sub-problems *once* with CVXPY `Parameter` objects for the
components that change between iterations ($\Phi^{s2}$ and $\Phi^{s1}$).

It also applies *Anderson acceleration* with a history of $m$
recent iterates to speed up convergence, and accepts a `max_iter`
cap to prevent infinite loops.

```{code-cell} ipython3
def solve_multi_period_economy_vfi(A=None,
                                   Q=None,
                                   C=None,
                                   P=None,
                                   problem_type=None,
                                   β=0.95,
                                   N=50,
                                   N_m=50,
                                   s_W_0=None,
                                   tol=1e-4,
                                   max_iter=300,
                                   m_anderson=5,
                                   verbose=True):
    """
    Infinite-horizon VFI using the two-step factored algorithm.

    Improvements over solve_multi_period_economy_2:
      * CVXPY problems are built once with Parameter objects --
        no memory leak across iterations.
      * Anderson acceleration (window m_anderson) reduces
        the number of Bellman iterations needed.
      * max_iter cap prevents unbounded runtime.

    Returns
    -------
    s_W       : 1-D array, converged surplus function on W
    Pi_W_s1   : 4-D array, Pi(a, q, w_m | w)
    Pi_W_m_s2 : 3-D array, Pi(c, w' | w_m)
    W         : 1-D array, the utility grid used
    """
    if β >= 1 or β <= 0:
        raise ValueError('β must lie in (0, 1)')

    def u_fn(a, c):
        return c**0.5 / 0.5 + (1 - a)**0.5 / 0.5

    problem_type = problem_type.lower()
    if problem_type == "full information":
        w_l = u_fn(A.max(), C.min()) / (1 - β)
        w_u = u_fn(A.min(), C.max()) / (1 - β)
    else:
        w_l = u_fn(A.min(), C.min()) / (1 - β)
        w_u = u_fn(A.min(), C.max()) / (1 - β)

    W    = np.linspace(w_l, w_u, N)
    W_m  = np.linspace(β * w_l + 2 * C.min()**0.5,
                        β * w_u + 2 * C.max()**0.5, N_m)

    n_A, n_Q, n_C = len(A), len(Q), len(C)
    A_ind, Q_ind  = range(n_A), range(n_Q)

    # Fixed arrays (depend only on grids and P, not on s_W)
    U_disc_s2 = np.array([[2 * c**0.5 + β * wp
                            for wp in W] for c in C])         # (n_C, N)
    U_disc_s1 = np.array([[[2 * (1 - a)**0.5 + wm
                             for wm in W_m] for q in Q]
                           for a in A])                        # (n_A, n_Q, N_m)
    U_disc_hat_s1 = np.array([[[[
        (2 * (1 - A[ah])**0.5 + W_m[wmi]) * P[ah, qi] / P[ai, qi]
        for wmi in range(N_m)] for qi in Q_ind]
        for ai in A_ind] for ah in A_ind])                     # (n_A, n_A, n_Q, N_m)

    # ----------------------------------------------------------
    # Step-2 CVXPY problem  (built once)
    # ----------------------------------------------------------
    Phi_s2_param = cp.Parameter((n_C, N))   # updated each outer iteration
    w_m_para     = cp.Parameter()
    Pi_w_m       = cp.Variable((n_C, N))

    obj_expr_s2  = cp.sum(cp.multiply(Phi_s2_param, Pi_w_m))
    C5_s2 = [cp.sum(cp.multiply(U_disc_s2, Pi_w_m)) == w_m_para]
    C7_s2 = [cp.sum(Pi_w_m) == 1, Pi_w_m >= 0]
    problem_s2   = cp.Problem(cp.Maximize(obj_expr_s2), C5_s2 + C7_s2)

    # ----------------------------------------------------------
    # Step-1 CVXPY problem  (built once)
    # ----------------------------------------------------------
    Phi_s1_param = cp.Parameter((n_Q, N_m))  # updated after step 2
    w_para       = cp.Parameter()
    Pi_w_list    = [cp.Variable((n_Q, N_m)) for _ in A_ind]

    obj_expr_s1  = cp.sum([cp.sum(cp.multiply(Phi_s1_param, Pi_w_list[ai]))
                            for ai in A_ind])
    C5_s1 = [cp.sum([cp.sum(cp.multiply(U_disc_s1[ai], Pi_w_list[ai]))
                     for ai in A_ind]) == w_para]
    C6_s1 = [(cp.sum(Pi_w_list[ai][qi, :]) ==
               P[ai, qi] * cp.sum(Pi_w_list[ai]))
              for qi in Q_ind for ai in A_ind]
    C7_s1 = ([cp.sum([cp.sum(Pi_w_list[ai]) for ai in A_ind]) == 1] +
              [Pi_w_list[ai] >= 0 for ai in A_ind])

    if problem_type == "full information":
        constraints_s1 = C5_s1 + C6_s1 + C7_s1
    else:
        C8_s1 = [(cp.sum(cp.multiply(U_disc_s1[ai], Pi_w_list[ai])) >=
                  cp.sum(cp.multiply(U_disc_hat_s1[ah, ai], Pi_w_list[ai])))
                 for ai in A_ind for ah in A_ind]
        constraints_s1 = C5_s1 + C6_s1 + C7_s1 + C8_s1

    problem_s1 = cp.Problem(cp.Maximize(obj_expr_s1), constraints_s1)

    # ----------------------------------------------------------
    # Initialise
    # ----------------------------------------------------------
    s_W_prime = (np.array(s_W_0, dtype=float)
                 if s_W_0 is not None else np.zeros(N))

    hist_x, hist_fx = [], []
    err = np.inf

    # ----------------------------------------------------------
    # Main iteration loop
    # ----------------------------------------------------------
    for iteration in range(1, max_iter + 1):
        t0 = time()

        # --- Step 2: solve for s_W_m ---
        Phi_s2_param.value = np.array([[β * sv - c
                                         for sv in s_W_prime] for c in C])
        s_W_m     = np.zeros(N_m)
        Pi_W_m_s2 = np.zeros((N_m, n_C, N))
        for i, wm in enumerate(W_m):
            w_m_para.value = wm
            problem_s2.solve(solver=cp.HIGHS, warm_start=True)
            if obj_expr_s2.value is not None:
                s_W_m[i]       = obj_expr_s2.value
                Pi_W_m_s2[i]   = Pi_w_m.value

        # --- Step 1: solve for s_W ---
        Phi_s1_param.value = np.array([[(q + swm)
                                         for swm in s_W_m] for q in Q])
        s_W     = np.zeros(N)
        Pi_W_s1 = np.zeros((N, n_A, n_Q, N_m))
        for i, w in enumerate(W):
            w_para.value = w
            problem_s1.solve(solver=cp.HIGHS, warm_start=True)
            if obj_expr_s1.value is not None:
                s_W[i] = obj_expr_s1.value
                for ai in A_ind:
                    if Pi_w_list[ai].value is not None:
                        Pi_W_s1[i, ai] = Pi_w_list[ai].value

        t1 = time()
        err = np.max(np.abs(s_W - s_W_prime))
        if verbose:
            print(f"Iter {iteration:3d}: max|ΔsW| = {err:.2e}  ({t1-t0:.1f}s)")

        if err <= tol:
            if verbose:
                print(f"Converged in {iteration} iterations.")
            break

        # --- Anderson acceleration ---
        hist_x.append(s_W_prime.copy())
        hist_fx.append(s_W.copy())
        mk = min(len(hist_x), m_anderson)
        if len(hist_x) > m_anderson:
            hist_x.pop(0)
            hist_fx.pop(0)

        if mk >= 2:
            X   = np.column_stack(hist_x[-mk:])
            FX  = np.column_stack(hist_fx[-mk:])
            F   = FX - X
            FtF = F.T @ F
            reg = max(1e-10 * np.trace(FtF) / mk, 1e-14)
            ones = np.ones(mk)
            try:
                theta      = np.linalg.solve(FtF + reg * np.eye(mk), ones)
                theta     /= ones @ theta
                s_candidate = FX @ theta
                s_next = (s_candidate
                          if np.all(np.isfinite(s_candidate))
                          else s_W)
            except np.linalg.LinAlgError:
                s_next = s_W
        else:
            s_next = s_W

        s_W_prime = s_next
        gc.collect()

    else:
        print(f"Warning: did not converge after {max_iter} iterations. "
              f"Final max|ΔsW| = {err:.2e}")

    return s_W, Pi_W_s1, Pi_W_m_s2, W
```

### Numerical Results

We use the same parameters as for the static economy, plus a
discount factor $\beta = 0.8$ and grids of $N = N_m = 100$ points.

*Initial values.*
We initialise the value function iteration with the one-period
(static) solution, scaled to discounted-sum units.

```{code-cell} ipython3
β = 0.8
N = 100
N_m = 100

W_l = u(A.min(), C.min()) / (1 - β)
W_u = u(A.min(), C.max()) / (1 - β)
W   = np.linspace(W_l, W_u, N)

W_m_l = β * W_l + 2 * C.min()**0.5
W_m_u = β * W_u + 2 * C.max()**0.5
W_m   = np.linspace(W_m_l, W_m_u, N_m)

in_time = time()
s_W_0, Pi_0 = solve_static_problem(W * (1 - β), u,
                                    A, Q, C, P,
                                    "unobserved-actions")
out_time = time()
print("Time(s):", round(out_time - in_time, 3))
```

*Finite-period economy ($T = 3$).*

```{code-cell} ipython3
in_time = time()
s_W_T, Pi_W_s1_T, Pi_W_m_s2_T = solve_multi_period_economy_2(
    A, Q, C, P, "unobserved-actions",
    T=3, N=N, N_m=N_m)
out_time = time()
print("Time(s):", round(out_time - in_time, 3))
```

```{code-cell} ipython3
T = 3
w_l_T = u(A.min(), C.min())
w_u_T = u(A.min(), C.max())
W_mat = np.cumsum(
    np.logspace(0, T - 1, T, base=β).reshape(T, 1) *
    np.linspace(w_l_T, w_u_T, N).reshape(1, N),
    axis=0)
W_T = W_mat[2, :]

plt.figure(figsize=(6.5, 6.5))
plt.plot(W_T, s_W_T, "k-.")
plt.text(8, 3, "3-Period Unobserved Action", size=12)
plt.title("Figure\n Optimized surplus function", y=-0.2)
plt.show()
```

*Infinite-period economy.*

```{code-cell} ipython3
:tags: [hide-output]

in_time = time()
s_W, Pi_W_s1, Pi_W_m_s2 = solve_multi_period_economy_2(
    A, Q, C, P, "unobserved-actions",
    N=N, N_m=N_m,
    s_W_0=s_W_0 / (1 - β),
    tol=1e-8)
out_time = time()
print("Time(s):", round(out_time - in_time, 3))
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
plt.plot(W, s_W, "k-.")
plt.xlim([5.0, 25.0])
plt.ylim([-7.5, 10.0])
plt.xlabel("w")
plt.ylabel("s(w)")
plt.title("Figure\n Optimized surplus function", y=-0.2)
plt.text(15, 6.5, "Infinity Unobserved Action", size=12)
plt.show()
```

### Recovering $\Pi^w(a, q, c, w')$

The two-step algorithm returns
$\Pi^w(a, q, w^m)$ and $\Pi^{w^m}(c, w')$ separately.

We recover the full joint distribution by using

$$
\Pi^w(a,q,c,w')
= \sum_{w^m}
  \Pi^w(a,q,w^m)\,\Pi^{w^m}(c,w').
$$

```{code-cell} ipython3
n_A, n_Q, n_C, n_W, n_W_prime = 4, 2, 81, N, N
A_ind, Q_ind, C_ind = range(n_A), range(n_Q), range(n_C)
W_ind, W_prime_ind  = range(n_W), range(n_W_prime)

Pi = np.array([[[[[
    Pi_W_s1[w_ind, a_ind, q_ind, :] @
    Pi_W_m_s2[:, c_ind, w_prime_ind]
    for w_prime_ind in W_prime_ind]
    for c_ind in C_ind]
    for q_ind in Q_ind]
    for a_ind in A_ind]
    for w_ind in W_ind])
```

#### Figure 5

```{code-cell} ipython3
# Solve the static full information
W_full = np.linspace(5, 25, N)

in_time = time()
s_W_1, Pi_1 = solve_static_problem(W_full*(1-β), u, A,
                                   Q, C, P, "full information")
out_time = time()

print("Time(s):", round(out_time - in_time, 3))
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
plt.plot(W, s_W, "k-.")
plt.plot(W, s_W_0/(1 - β), "yellow")
plt.plot(W_full, s_W_1/(1 - β), "red")
plt.xlim([5.0, 25.0])
plt.ylim([-7.5, 10.0])
plt.hlines(0, 5.0, 25.0, linestyle="dashed")
plt.xlabel("w")
plt.ylabel("s(w)")
plt.title("Figure 5\n Optimized surplus function", y=-0.2)
plt.text(5.4, -2.0, "Full Information (top)", size=12)
plt.text(5.4, -3.0, "T = infinity Unobserved Action (middle)", size=12)
plt.text(5.4, -4.0, "T = 1 Unobserved Action", size=12)
plt.show()
```

Figure 5 compares three surplus functions.

The full-information frontier is highest.

The infinite-horizon unobserved-action frontier is
below it because incentive constraints are added, but it lies above the
frontier obtained by repeating the one-period unobserved-action contract.

The difference between the two unobserved-action curves is the gain from
history dependence.

#### Figure 6

```{code-cell} ipython3
# Calculate expected efforts
# T=1 Unobserved Action
X, Y = list(range(len(A))), list(range(len(Q)))
Z, N = list(range(len(C))), list(range(len(W)))
Ea_1 = np.array([np.sum([A[x]*Pi_0[i,x,:,:] for x in X]) for i in N])

# T=infinity unobserved Action
Ea_inf = np.array([np.sum([A[x]*Pi[i,x,:,:,:] for x in X]) for i in N])

# Plot expected efforts
plt.figure(figsize=(6.5, 6.5))
plt.plot(W, Ea_1)
plt.plot(W, Ea_inf)
plt.xlabel("w")
plt.ylabel("E{a(w)}")
plt.xlim([5.0, 25.0])
plt.ylim([0.0, 0.8])
plt.title("Figure 6\n Actions", y=-0.2)
plt.text(14, 0.60, "T = infinity Unobserved Action (top)", size=10)
plt.text(14, 0.55, "T = 1 Unobserved Action (bottom)", size=10)
plt.show()
```

History dependence also raises effort relative to repeated one-period
contracts.

Near the lower utility bound, incentive compatibility forces
low effort, but away from that bound continuation promises help provide
incentives without relying only on current consumption.

#### Figure 7

```{code-cell} ipython3
def ex_con(Pi, A, Q, C, W, type="infinity"):
    X, Y = list(range(len(A))), list(range(len(Q)))
    Z, N = list(range(len(C))), list(range(len(W)))
    Ec = np.zeros((len(N), len(X), len(Y)))
    for i in N:
        for x in X:
            for y in Y:
                if type == "infinity":
                    total_prob = np.sum(Pi[i,x,y,:,:])
                    if total_prob <= 1e-9:
                        Ec[i,x,y] = float("-inf")
                    else:
                        Ec[i,x,y] = np.sum([np.sum(C[z] * Pi[i, x, y, z, :])
                                            for z in Z])/total_prob
                elif type == "one":
                    total_prob = np.sum(Pi[i,x,y,:])
                    if total_prob <= 1e-9:
                        Ec[i,x,y] = float("-inf")
                    else:
                        Ec[i,x,y] = np.sum([C[z] * Pi[i, x, y, :]
                                            for z in Z])/total_prob                   
    return Ec
```

```{code-cell} ipython3
Ec_inf = ex_con(Pi, A, Q, C, W)

# Plot expected consumption
plt.figure(figsize=(10.5, 10.5))
for x in X:
    for y in Y:
        plt.plot(W, Ec_inf[:, x, y])
plt.xlabel("w")
plt.ylabel("E(c) given a, q, w")
plt.xlim([5.0, 25.0])
plt.ylim([0.0, 2.25])
plt.title("Figure 7\n Unobserved Action Consumption", y=-0.3)
plt.annotate("a=.4, q=2", xy=(13.5, 0.5), xytext=(10.5, 0.7),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=2", xy=(20.0, 1.3), xytext=(15.5, 1.65),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)", xy=(24, 2.15), xytext=(15.0, 2.15),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=2", xy=(10.1, 0.01), xytext=(7.5, 0.03),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.4, q=2", xy=(10.5, 0.10), xytext=(7.5, 0.15),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.6, q=2", xy=(11.5, 0.25), xytext=(8.5, 0.30),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.6, q=1", xy=(12.5, 0.05), xytext=(14.5, 0.10),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.4, q=1", xy=(15.0, 0.35), xytext=(18, 0.2),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=1", xy=(20.0, 1.1), xytext=(21.5, 0.75),
             arrowprops={"arrowstyle":"-"})
plt.annotate("", xy=(10.0, 0), xytext=(11.5, -0.1),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)\na={.2,.4}, q=1", fontsize=15, xy=(10.5, 0),
             xytext=(5.5, -0.3))
plt.annotate(r"$\{$",fontsize=35, xy=(10.5, 0), xytext=(4.5, -0.3))
plt.annotate(r"$\}$",fontsize=35, xy=(10.5, 0), xytext=(9.5, -0.3))
plt.show()
```

Figure 7 shows how dynamic contracts smooth current consumption relative
to the static unobserved-action economy.

Output still affects rewards,
but a large part of the reward and punishment is shifted into future
promised utility.

#### Figure 8

```{code-cell} ipython3
def ex_ut(Pi, A, Q, C, W):
    X, Y = list(range(len(A))), list(range(len(Q)))
    Z, N = list(range(len(C))), list(range(len(W)))
    Ew = np.zeros((len(N),len(X),len(Y)))
    for i in N:
        for x in X:
            for y in Y:
                total_prob = np.sum(Pi[i, x, y, :, :])
                if total_prob <= 1e-9:
                    Ew[i,x,y] = float("-inf")
                else:
                    Ew[i,x,y] = np.sum([np.sum(W[w] * Pi[i, x, y, :, w])
                                        for w in N])/total_prob
    return Ew
```

```{code-cell} ipython3
Ew_inf = ex_ut(Pi, A, Q, C, W)


# Plot expected consumption
plt.figure(figsize=(7.5, 7.5))
marker = [["o","v"],[">","<"],["x","1"],["2","3"]]
for x in X:
    for y in Y:
        plt.plot(W, Ew_inf[:,x,y],marker=marker[x][y])
plt.plot(W,W,"k-.")
plt.xlabel("w")
plt.ylabel("E(w') given a, q, w")
plt.xlim([10.0, 25.0])
plt.ylim([10.0, 25.0])
plt.title("Figure 8\n Future Utility", y=-0.2)
plt.annotate("a=.4, q=2", xy=(14.0, 15.0), xytext=(10.5, 17.0),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=2", xy=(19.5, 20.0), xytext=(15.0, 23.0),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)", xy=(24.5, 24.5), xytext=(18.0, 24.5),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=2", xy=(10.0, 10.7), xytext=(7.5, 10.7),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.4, q=2", xy=(10.3, 11.2), xytext=(7.5, 11.2),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.6, q=2", xy=(11.5, 12.2), xytext=(10.1, 14.0),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.6, q=1", xy=(11.7, 10.7), xytext=(13.5, 10.7),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.4, q=1", xy=(15.0, 14.0), xytext=(16.5, 12.5),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=.2, q=1", xy=(20.0, 19.5), xytext=(21.0, 18.0),
             arrowprops={"arrowstyle":"-"})
plt.annotate("", xy=(10.1, 10.1), xytext=(12.0, 9.2),
             arrowprops={"arrowstyle":"-"})
plt.annotate("a=0, q=(1,2)\na={.2,.4}, q=1", xy=(10.1, 10.1), xytext=(9.5, 8.5))
plt.annotate(r"$\{$",fontsize=25, xy=(10.1, 10.1), xytext=(8.5, 8.5))
plt.annotate(r"$\}$",fontsize=25, xy=(10.1, 10.1), xytext=(12.5, 8.5))
plt.show()
```

Figure 8 displays the expected next-period promise conditional on
current $w$, recommended action $a$, and realized output $q$.

High
output generally raises continuation utility and low output lowers it.

At the endpoints of the feasible promise set, the transition stays on the
45-degree line because only the corresponding extreme plan can deliver
that endpoint.

For figures 9--12, {cite}`Phelan_Townsend_91` used $\beta = 0.95$.

They report that Figures 5--8 are easier to read at $\beta = 0.8$, while
the simulated individual paths and distributions in Figures 9--12 are
more informative at the higher discount factor.

We now use `solve_multi_period_economy_vfi` -- which builds the CVXPY
problems once and applies Anderson acceleration -- to solve the
infinite-horizon economy at $\beta = 0.95$ with a grid of $N = N_m = 50$
points.

Starting from the static solution rescaled to discounted-sum units, the
iteration converges to tolerance $10^{-4}$.

```{code-cell} ipython3
β_95  = 0.95
N_95  = 50
N_m95 = 50

# Initial value: static solution rescaled to infinite-horizon units
w_l_95 = u(A.min(), C.min()) / (1 - β_95)
w_u_95 = u(A.min(), C.max()) / (1 - β_95)
W_95   = np.linspace(w_l_95, w_u_95, N_95)

s_W_0_95, _ = solve_static_problem(W_95 * (1 - β_95), u, A, Q, C, P,
                                    "unobserved-actions")

in_time = time()
s_W_new, Pi_W_s1_new, Pi_W_m_s2_new, W_new = solve_multi_period_economy_vfi(
    A, Q, C, P, "unobserved-actions",
    β=β_95, N=N_95, N_m=N_m95,
    s_W_0=s_W_0_95 / (1 - β_95),
    tol=1e-4, max_iter=300, m_anderson=5,
    verbose=True)
out_time = time()
print("Time(s):", round(out_time - in_time, 3))
```

```{code-cell} ipython3
# Recover full joint distribution Pi(a, q, c, w' | w)
N_new = len(W_new)
A_ind_new  = range(n_A)
Q_ind_new  = range(n_Q)
C_ind_new  = range(n_C)
W_ind_new  = range(N_new)
Wp_ind_new = range(N_new)

Pi_new = np.array([[[[[
    Pi_W_s1_new[wi, ai, qi, :] @ Pi_W_m_s2_new[:, ci, wpi]
    for wpi in Wp_ind_new]
    for ci  in C_ind_new]
    for qi  in Q_ind_new]
    for ai  in A_ind_new]
    for wi  in W_ind_new])
```

```{code-cell} ipython3
Ew_beta = ex_ut(Pi_new, A, Q, C, W_new)
```

```{code-cell} ipython3
def simulation(W, C, s_W, T, Pi, Ew, seed=12345):
    # initial w such that s(w)=0
    w_index = np.argwhere(np.abs(s_W) == np.min(np.abs(s_W)))[0][0]
    w0 = W[w_index]
    date = np.arange(T)
    
    # set seed for random number
    np.random.seed(seed)
    randn = np.random.rand(T, 8)
    
    w_index1, w_index2 = w_index, w_index
    w_series = w0*np.ones(T+1)
    c_series = np.zeros(T)
    Pi_c = list(np.zeros(T))
    Pi_w = list(np.zeros(T))
    
    for i in range(T):
        
        w_index_temp1 = w_index1
        
        Pi_temp_a = Pi[w_index_temp1, :, :, :, :].sum(
                                            axis=1).sum(
                                            axis=1).sum(
                                            axis=1)
        Pi_temp_a_cum = np.cumsum(Pi_temp_a / np.sum(Pi_temp_a))
        a_index = np.sum(randn[i, 0] >= Pi_temp_a_cum)
        Pi_temp_q = Pi[w_index_temp1, a_index, :, :, :].sum(
                                            axis=1).sum(
                                            axis=1)
        Pi_temp_q_cum = np.cumsum(Pi_temp_q / np.sum(Pi_temp_q))
        q_index = np.sum(randn[i, 1] >= Pi_temp_q_cum)
        
        Pi_temp_w = Pi[w_index_temp1, a_index, q_index, :, :].sum(
                                                            axis=0)
        Pi_temp_w_cum = np.cumsum(Pi_temp_w/np.sum(Pi_temp_w))
        w_index1 = np.sum(randn[i, 2] >= Pi_temp_w_cum)
        
        # simulation for consumption as well as its distribution
        Pi_c[i] = Pi[w_index_temp1, a_index, q_index, :, w_index1]
        Pi_c[i] /= np.sum(Pi_c[i])
        Pi_temp_c_cum = np.cumsum(Pi_c[i])
        c_index = np.sum(randn[i, 3] >= Pi_temp_c_cum)
        c_series[i] = C[c_index]
        
        # simulation for expected utility
        w_series[i+1] = Ew[w_index_temp1, a_index, q_index]
        
        # simulation for distribution over future utility
        Pi_temp_a = Pi[w_index2, :, :, :, :].sum(axis=1).sum(
                                                axis=1).sum(axis=1)
        Pi_temp_a_cum = np.cumsum(Pi_temp_a / np.sum(Pi_temp_a))
        a_index = np.sum(randn[i, 4] >= Pi_temp_a_cum)
        Pi_temp_q = Pi[w_index2, a_index, :, :, :].sum(
                                                axis=1).sum(axis=1)
        Pi_temp_q_cum = np.cumsum(Pi_temp_q / np.sum(Pi_temp_q))
        q_index = np.sum(randn[i, 5] >= Pi_temp_q_cum)
        Pi_temp_c = Pi[w_index2, a_index, q_index, :, :].sum(axis=1)
        Pi_temp_c_cum = np.cumsum(Pi_temp_c/np.sum(Pi_temp_c))
        c_index = np.sum(randn[i, 6] >= Pi_temp_c_cum)
        Pi_w[i] = Pi[w_index2, a_index, q_index, c_index, :]
        Pi_w[i] /= np.sum(Pi_w[i])
        w_index2 = np.sum(randn[i,7] >= np.cumsum(Pi_w[i]))
    
    return c_series, w_series, Pi_w, Pi_c
```

```{code-cell} ipython3
c_series = np.zeros((80, 4))
w_series = np.zeros((81, 4))
for i in range(4):
    c_series[:, i], w_series[:, i], _, _ = simulation(
        W_new, C, s_W_new, 80, Pi_new, Ew_beta, seed=(12345 + i))
```

The simulations start from the grid point at which surplus is closest to
zero.

This corresponds to the ex ante symmetric, or "fair", allocation
in the paper: it is the highest common promised utility that can be
assigned while keeping discounted social surplus nonnegative.

#### Figure 9

```{code-cell} ipython3
# Plot consumption simulation
date_c = np.arange(80) + 1
plt.figure(figsize=(6.5, 6.5))
plt.plot(date_c, c_series[:, 0])
plt.plot(date_c, c_series[:, 1])
plt.plot(date_c, c_series[:, 2])
plt.plot(date_c, c_series[:, 3])
plt.xlabel("date")
plt.ylabel("consumption")
plt.xlim([0, 80])
plt.ylim([0.00, 2.25])
plt.title("Figure 9\n Individual Consumptions ($\\beta=0.95$)", y=-0.2)
plt.show()
```

#### Figure 10

```{code-cell} ipython3
# Plot expected utility simulation
date_w = np.arange(81)
plt.figure(figsize=(6.5, 6.5))
plt.plot(date_w, w_series[:, 0])
plt.plot(date_w, w_series[:, 1])
plt.plot(date_w, w_series[:, 2])
plt.plot(date_w, w_series[:, 3])
plt.xlabel("date")
plt.ylabel("expected utility")
plt.title("Figure 10\n Individual Utilities ($\\beta=0.95$)", y=-0.2)
plt.show()
```

```{code-cell} ipython3
%time _, _, Pi_w, Pi_c = simulation(W_new, C, s_W_new, 80, Pi_new, Ew_beta)
```

#### Figure 11

```{code-cell} ipython3
# Plotting distribution for consumption
%matplotlib inline

date_mat_c = np.reshape(np.arange(80) + 1, (80, 1)) * \
             np.ones((1, len(C)))
c_mat = np.ones((80, 1)) @ np.reshape(C, (1, len(C)))

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(projection='3d')
plt.title("Figure 11 \n Consumptions over time ($\\beta=0.95$)", y=-0.3)
plt.xlabel('date')
plt.ylabel('consumption')
ax.set_zlabel('percentage')

surf = ax.plot_surface(date_mat_c, c_mat, np.array(Pi_c),
                       cmap='viridis')
plt.show()
```

#### Figure 12

```{code-cell} ipython3
# Plotting distribution for future utilities
%matplotlib inline
date_mat_w = np.reshape(np.arange(80) + 1, (80, 1)) * \
                np.ones((1, len(W_new)))
W_mat_12 = np.ones((80, 1)) @ np.reshape(W_new, (1, len(W_new)))

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(projection='3d')
plt.title("Figure 12 \n Utilities over time ($\\beta=0.95$)", y=-0.3)
plt.xlabel('date')
plt.ylabel('w')
ax.set_zlabel('percentage')

surf = ax.plot_surface(date_mat_w, W_mat_12, np.array(Pi_w),
                       cmap='viridis')
plt.show()
```

## Concluding Remarks

### Economics

*Moral hazard and the cost of private information.*

When the principal cannot observe the agent's effort, the optimal contract
must balance two competing objectives: *insurance* (smoothing the agent's
consumption across output realizations) and *incentives* (rewarding high
output to make effort attractive).

The unobserved-action surplus function in Figure 1 lies everywhere below
the full-information frontier, and the gap between them measures the
surplus cost of unobserved effort.

*Dynamic contracts and promised utility.*

The recursive formulation of {cite}`Spear_Srivastava_87` compresses all
payoff-relevant history into a single scalar state: the discounted
expected continuation utility $w$ that the principal has promised the
agent.

By tracking $w$ rather than the full history of outputs, the dynamic
contracting problem becomes tractable.

Figures 7--8 show that under the optimal infinite-horizon contract the
principal rewards high output by granting the agent a higher continuation
utility and punishes low output by lowering it.

Continuation promises
therefore substitute partly for large contemporaneous consumption
spreads.

*Diversity over time.*

Figures 9--12 illustrate the paper's central computational message:
starting from a common initial promise, dynamic incentives generate
non-trivial individual histories and cross-sectional dispersion in
consumption and promised utility.

With the finite grids used here, the endpoints of the promise set are
absorbing.

The simulations should therefore be read as finite-grid
illustrations of how history dependence spreads the distribution over
time, not as a separate theorem about the limiting distribution.

### Technical Tricks

*Lotteries and convexification.*

Incentive constraints can render the set of feasible contracts non-convex,
making standard optimization techniques unreliable.

{cite}`Phelan_Townsend_91` circumvented this by allowing the planner to
choose a joint *lottery* $\Pi(a, q, c, w')$ over actions, outputs,
consumptions, and continuation values.

Because any mixture of feasible lotteries is itself feasible, the
constraint set becomes convex, and global optima are well-defined.

*Linear programming.*

With finite grids, the convexified Bellman equation is a linear program:
the objective $(q - c + \beta v(w'))$ and every constraint are linear in
$\Pi$.

Treating $v(w')$ as a fixed vector from the previous iteration, value
function iteration reduces to solving one LP per grid point per
iteration -- a task handled efficiently by modern LP solvers such as
HiGHS.

*Dynamic programming.*

The promised-utility state variable $w$ makes the problem recursive.

At each iteration the Bellman operator maps a surplus function $v$ to an
updated surplus function $Tv$; repeated application converges to the
infinite-horizon fixed point.

The implementation initializes the iteration from the scaled static
solution, which is a useful numerical starting point.

*Two-step factored algorithm.*

The additive separability $U(a,c) = 2\sqrt{1-a} + 2\sqrt{c}$ allows the
four-dimensional LP to be split into two smaller sub-problems.

*Step 2* allocates consumption given an intermediate promised utility
$w^m$; *Step 1* assigns actions, outputs, and intermediate continuation
utilities given $w$.

Because each sub-LP has far fewer decision variables than the full joint
LP, computation is substantially faster and the approach scales to finer
grids.

*Dynamic programming squared.*

This lecture is closely related to what Lars Ljungqvist and Thomas
Sargent call *dynamic programming squared* in
{cite}`Ljungqvist2012`.

The phrase refers to recursive problems in which one continuation object
is carried as a state variable inside another recursive problem.

Here the surplus function $s(w)$ -- the solution to the principal's
outer dynamic program -- has the agent's continuation utility $w$ as its
state variable, while feasible movements in $w$ are governed by
promise-keeping and incentive constraints.

The same architecture reappears throughout this lecture series.

In {doc}`Stackelberg plans <dyn_stack>` the Stackelberg leader's
value function takes the followers' competitive-equilibrium value
function as an argument.

In {doc}`Optimal Taxation with State-Contingent Debt <opt_tax_recur>`,
a Ramsey planner's outer Bellman equation uses the household's
marginal utility of wealth $x$ -- itself defined by an inner
implementability constraint -- as its state variable.

In the {doc}`Calvo model <calvo>` and the two Chang lectures
({doc}`Ramsey plans <chang_ramsey>` and
{doc}`credible policies <chang_credible>`), a government's value
function takes the private sector's continuation value $\theta$
as an argument, with $\theta$ governed by its own Bellman equation.

In {doc}`Unemployment Insurance <un_insure>` the planner's
contract-design problem embeds the worker's continuation utility
as the state variable in an outer surplus-maximization program,
producing a closely related nested recursive structure.

In all of these settings, the inner dynamic program defines a
state variable -- a promised utility, a marginal value, or a
continuation value -- that restricts what the outer dynamic
program can promise or deliver. 



## Exercises

````{admonition} Exercise 1
:class: exercise

Using the surplus arrays `s_W_full` and `s_W_unobs` computed in the
static section, define the **agency cost function**

$$
\delta(w) = s^{FI}(w) - s^{UA}(w), \quad w \in W_{static}.
$$

1. Plot $\delta(w)$ over $W_{static} = [1, 5]$.
2. Report the value $\hat{w}$ at which $\delta$ is largest.
3. Explain intuitively why agency costs are highest at that level of
   promised utility.
````

````{dropdown} Solution to Exercise 1
:class: dropdown

```{code-cell} ipython3
delta_W = s_W_full - s_W_unobs

plt.figure(figsize=(6.5, 6.5))
plt.plot(W_static, delta_W)
plt.xlabel("w")
plt.ylabel(r"$\delta(w) = s^{FI}(w) - s^{UA}(w)$")
plt.xlim([1.0, 5.0])
plt.ylim(bottom=0.0)
plt.title("Agency Cost in the Static Model", y=-0.2)
plt.show()

w_hat = W_static[np.argmax(delta_W)]
print(f"Largest agency cost at w = {w_hat:.3f},  δ = {delta_W.max():.4f}")
```

Agency costs are highest near intermediate levels of promised utility
because at those values the principal most values inducing high effort
(output is valuable) while the agent still requires meaningful
consumption-state variation to be incentivized.

At low $w$ the agent is near subsistence and effort is low anyway;
at high $w$ the agent is nearly fully insured and the marginal incentive
cost of each additional unit of effort is small.
````

````{admonition} Exercise 2
:class: exercise

The output probability matrix $P$ governs how informative output is about
effort.
Define a **flatter** probability matrix

$$
P_{flat} = \begin{pmatrix}
0.70 & 0.30 \\
0.55 & 0.45 \\
0.45 & 0.55 \\
0.30 & 0.70
\end{pmatrix}
$$

in which output is less informative about effort than in the baseline $P$.

1. Re-solve the static unobserved-action problem with $P_{flat}$ and the
   same grid $W_{static}$.
2. On a single figure with two panels, compare the surplus functions
   $s^{UA}(w)$ and expected effort levels $E\{a(w)\}$ under $P$ and
   $P_{flat}$.
3. Explain the economic intuition for any differences you find.
````

````{dropdown} Solution to Exercise 2
:class: dropdown

```{code-cell} ipython3
P_flat = np.array([[0.70, 0.30],
                   [0.55, 0.45],
                   [0.45, 0.55],
                   [0.30, 0.70]])

s_W_flat, Pi_flat = solve_static_problem(W_static, u, A, Q, C, P_flat,
                                          "unobserved-actions")
Ea_flat = np.einsum('a,waqc->w', A, Pi_flat)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

axes[0].plot(W_static, s_W_unobs, label="Baseline $P$")
axes[0].plot(W_static, s_W_flat,  label="Flat $P$")
axes[0].hlines(0, 1.0, 5.0, linestyle="dashed")
axes[0].set_xlabel("w")
axes[0].set_ylabel("s(w)")
axes[0].set_xlim([1.0, 5.0])
axes[0].set_title("Surplus Function", y=-0.2)
axes[0].legend()

axes[1].plot(W_static, Ea_unobs, label="Baseline $P$")
axes[1].plot(W_static, Ea_flat,  label="Flat $P$")
axes[1].set_xlabel("w")
axes[1].set_ylabel(r"$E\{a(w)\}$")
axes[1].set_xlim([1.0, 5.0])
axes[1].set_ylim([0.0, 0.8])
axes[1].set_title("Expected Effort", y=-0.2)
axes[1].legend()

plt.tight_layout()
plt.show()
```

With $P_{flat}$ output carries less statistical information about effort:
the likelihood ratio $P(q \mid \hat{a}) / P(q \mid a)$ is closer to 1
for all deviations $\hat{a} \neq a$.

The incentive-compatibility constraint {eq}`eq:eq2prime` therefore becomes
harder to satisfy: large consumption rewards for high output must be
offered to deter deviations, crowding out insurance.

As a result the principal extracts less surplus and induces less effort
than under the baseline $P$ -- the surplus function shifts down and
expected effort falls.
````
