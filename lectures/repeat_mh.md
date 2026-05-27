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

# Repeated moral hazard

## Overview

This lecture computes information-constrained optima in the
Phelan-Townsend repeated moral-hazard environment
{cite}`Phelan_Townsend_91`.

The environment is a continuum-agent economy with unobserved effort.

The planner chooses lotteries over individual histories, subject to
promise-keeping and incentive-compatibility constraints, and maximizes
discounted social surplus.

The key recursive idea comes from {cite:t}`Spear_Srivastava_87`: an
agent's promised continuation utility is a sufficient state variable.

Phelan and Townsend combine that idea with lotteries, finite grids, and
linear programming to compute full-information, static
unobserved-action, and repeated unobserved-action allocations.

The lecture proceeds from the recursive formulation to the computational
implementation.

*  We review the promised-utility recursion of
   {cite:t}`Spear_Srivastava_87`.
*  We formulate the Phelan-Townsend lottery problem and its finite-grid
   linear-programming approximation.
*  We use the static economy to isolate the surplus cost of hidden
   effort and the role of output-contingent consumption.
*  We use the repeated economy to show how continuation promises become
   an additional incentive instrument and generate dispersion over time.


## Promised-utility recursion

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

The principal's value function $v(w)$ is the maximum expected
discounted surplus attainable when the agent has been promised $w$.

It satisfies the Bellman equation

$$
v(w) = \max_{a,\,c,\,\tilde{w}}\
\int \bigl\{q - c(w,q) + \beta\, v[\tilde{w}(w,q)]\bigr\}\,
dF[q\mid a(w)]
$$ (eq:eq3)

subject to the promise-keeping constraint {eq}`eq:eq1` and the
incentive-compatibility constraint {eq}`eq:eq2`.


## Lotteries and linear programming

A technical difficulty in problems like {eq}`eq:eq3` is that
incentive constraints can make deterministic contract problems
non-convex.

```{prf:example} A non-convex deterministic contract set
:label: repeat_mh_nonconvex_example
:class: dropdown

Consider a one-period version of the problem with two outputs,
$q_H$ and $q_L$, and two actions, high effort $H$ and low effort $L$.
Suppose that

$$
P(q_H \mid H)=3/4,
\qquad
P(q_H \mid L)=1/4,
$$

and that the agent's utility is

$$
u(c,H)=\sqrt c - 1/2,
\qquad
u(c,L)=\sqrt c.
$$

A deterministic contract that recommends high effort pays $c_H$ after
$q_H$ and $c_L$ after $q_L$.
Incentive compatibility requires

$$
\frac34 \sqrt{c_H}+\frac14\sqrt{c_L}-\frac12
\geq
\frac14 \sqrt{c_H}+\frac34\sqrt{c_L},
$$

or

$$
\sqrt{c_H}-\sqrt{c_L}\geq 1.
$$

The two contracts $(c_H,c_L)=(1,0)$ and $(c_H,c_L)=(9,4)$ both satisfy
this constraint.
But their midpoint, $(c_H,c_L)=(5,2)$, violates it because

$$
\sqrt 5-\sqrt 2 \approx 0.82 < 1.
$$

Thus the set of deterministic contracts satisfying incentive
compatibility is not convex.
```

The Phelan-Townsend approach formulates the planning problem in
terms of **lotteries** over actions, outputs, consumptions, and
continuation utilities.

At the aggregate level these probabilities
are also population fractions, so individual randomization creates no
aggregate uncertainty in a continuum-agent economy.

For computation, we "grid" the relevant sets of
possible utilities, allowing only finitely many points.

With finite sets, or finite approximations to sets, $A$, $Q$, and $C$,
the planner's problem becomes a finite-dimensional optimization problem
with linear constraints.

Each stage of the computation therefore amounts to solving a finite
**linear programming** problem.

We begin with the finite objects in the planning problem.

Let $P(q | a)$ be a family of discrete conditional probability
distributions over finite sets $Q$ (outputs) and $A$ (actions).

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

The corresponding Bellman operator is also a linear program.
The principal's value function satisfies

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

The algorithm solves one LP for each grid point $w \in W$ and iterates
on the surplus function.

Their Theorem 4 gives the
contraction result that justifies this iteration for the
infinite-horizon problem.

## Implementation

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install cvxpy
```

We import some Python packages.

```{code-cell} ipython3
import numpy as np
import cvxpy as cp
from time import time
import gc
import matplotlib.pyplot as plt
```

## The static economy

A one-period economy is the cleanest place to isolate the
informational friction.

This isolates the static informational friction before the dynamic
promised-utility channel is introduced.

We first compute the full-information benchmark, where effort can be
controlled directly.

We then make effort private information and add incentive compatibility.

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

The baseline utility specification is

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

These parameter values define the baseline numerical economy for the
static comparisons and the first dynamic calculations.

The static grid of promised utility values below spans the
interval $[1,5]$, covering the promise range emphasized in the
one-period analysis.

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

### Solving the static problem

The function `solve_static_problem` solves one LP for each promised
utility value $w$.

The code keeps the notation close to the mathematical problem:
`π[a_i][q_i, c_i]` is the lottery probability
$\Pi^w(a_i,q_i,c_i)$, `Φ[q_i, c_i]` is output net of consumption,
and `U[a_i, c_i]` is period utility.

For the full-information problem we impose C1--C3.

For the unobserved-action problem we add C4.

```{code-cell} ipython3
def solve_static_problem(W, u, A, Q, C, P, problem_type):
    """
    Solve the static Phelan-Townsend LP on a grid of promises W.

    Returns
    -------
    s_W : ndarray
        Optimal surplus at each w in W.
    π_W : ndarray
        Lottery probabilities π_W[w_i, a_i, q_i, c_i].
    """
    n_a, n_q, n_c = len(A), len(Q), len(C)
    A_i, Q_i = range(n_a), range(n_q)

    Φ = np.array([[q - c for c in C] for q in Q])
    U = np.array([[u(a, c) for c in C] for a in A])

    w = cp.Parameter()
    π = [cp.Variable((n_q, n_c), nonneg=True) for _ in A_i]

    surplus = cp.sum([
        cp.sum(cp.multiply(Φ, π[a_i]))
        for a_i in A_i
    ])

    promise = [
        cp.sum([
            cp.sum(cp.multiply(U[a_i], π[a_i][q_i, :]))
            for a_i in A_i
            for q_i in Q_i
        ]) == w
    ]

    output_law = [
        cp.sum(π[a_i][q_i, :]) == P[a_i, q_i] * cp.sum(π[a_i])
        for a_i in A_i
        for q_i in Q_i
    ]

    probability = [cp.sum([cp.sum(π[a_i]) for a_i in A_i]) == 1]

    constraints = promise + output_law + probability

    if problem_type.lower() != "full information":
        incentives = []
        for a_i in A_i:
            for a_hat_i in A_i:
                obey = cp.sum([
                    cp.sum(cp.multiply(U[a_i], π[a_i][q_i, :]))
                    for q_i in Q_i
                ])
                deviate = cp.sum([
                    cp.sum(cp.multiply(U[a_hat_i], π[a_i][q_i, :]))
                    * P[a_hat_i, q_i] / P[a_i, q_i]
                    for q_i in Q_i
                ])
                incentives.append(obey >= deviate)
        constraints += incentives

    problem = cp.Problem(cp.Maximize(surplus), constraints)

    s_W = np.full(len(W), np.nan)
    π_W = np.full((len(W), n_a, n_q, n_c), np.nan)
    for w_i, w_value in enumerate(W):
        w.value = w_value
        problem.solve(solver=cp.HIGHS)
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            s_W[w_i] = surplus.value
            for a_i in A_i:
                π_W[w_i, a_i] = π[a_i].value

    return s_W, π_W
```

### Static allocations

```{note}
The original calculations used standard revised simplex methods.

We use HiGHS through CVXPY.

At degenerate utility grid points, different LP solvers can select
different optimal lotteries.

Some consumption schedules can therefore differ slightly even when the
surplus function is unchanged.
```

```{code-cell} ipython3
W_static = np.linspace(1, 5, 100)

s_W_full, π_full = solve_static_problem(W_static, u, A, Q, C, P,
                                          "full information")
s_W_unobs, π_unobs = solve_static_problem(W_static, u, A, Q, C, P,
                                            "unobserved-actions")
```

The arrays returned by `solve_static_problem` have a direct economic
interpretation.

`s_W_full` and `s_W_unobs` are the optimized surplus frontiers.

`π_full` and `π_unobs` store the optimal lotteries over
$(a,q,c)$ at each promised utility.

Grid points outside a problem's feasible promise set are recorded as
`nan`, so Matplotlib leaves those parts of the graph blank.

```{code-cell} ipython3
def expected_consumption_static(π_W, C):
    π0 = np.nan_to_num(π_W, nan=0.0)
    mass = π0.sum(axis=-1)
    numerator = np.einsum('c,waqc->waq', C, π0)
    Ec = np.full(mass.shape, np.nan)
    np.divide(numerator, mass, out=Ec, where=mass > 1e-10)
    return Ec
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
plt.plot(W_static, s_W_full, label="Full information")
plt.plot(W_static, s_W_unobs, label="Hidden effort")
plt.hlines(0, 1.0, 5.0, linestyle="dashed")
plt.xlabel("w")
plt.ylabel("s(w)")
plt.xlim([1.0, 5.0])
plt.ylim([-1.5, 2.0])
plt.title("Surplus frontiers", y=-0.2)
plt.legend()
plt.show()
```

The full-information frontier is higher because the planner can choose
effort directly.

The unobserved-action frontier lies below it because effort must be
induced with state-contingent rewards.

The gap is the agency cost of private effort.

```{code-cell} ipython3
Ea_full  = np.einsum('a,waqc->w', A, π_full)
Ea_unobs = np.einsum('a,waqc->w', A, π_unobs)

plt.figure(figsize=(6.5, 6.5))
plt.plot(W_static, Ea_full, label="Full information")
plt.plot(W_static, Ea_unobs, label="Hidden effort")
plt.xlabel("w")
plt.ylabel("E{a(w)}")
plt.xlim([1.0, 5.0])
plt.ylim([0.0, 0.8])
plt.title("Expected effort", y=-0.2)
plt.legend()
plt.show()
```

Here the code integrates the action grid against the lottery
probabilities, producing $E\{a(w)\}$.

Under full information, effort is chosen to maximize surplus at each
promise.

With unobserved action, expected effort is lower where incentives are
costly to provide.

```{code-cell} ipython3
Ec_unobs = expected_consumption_static(π_unobs, C)

fig, axes = plt.subplots(1, len(Q), figsize=(11, 4), sharey=True)
for q_i, ax in enumerate(axes):
    for a_i, a in enumerate(A):
        ax.plot(W_static, Ec_unobs[:, a_i, q_i], label=f"a={a:g}")
    ax.set_title(f"q={Q[q_i]:g}")
    ax.set_xlabel("w")
    ax.set_xlim([1.0, 5.0])
    ax.set_ylim([0.0, 2.25])
axes[0].set_ylabel("E(c | w, a, q)")
axes[-1].legend(title="Action", loc="lower right")
fig.suptitle("Consumption when effort is hidden")
fig.tight_layout()
plt.show()
```

This cell conditions on the recommended action and realized output, then
computes expected consumption.

The unobserved-action schedule uses current consumption as an incentive
device: high-output histories tend to receive higher consumption, while
low-output histories receive less.

```{code-cell} ipython3
Ec_full = expected_consumption_static(π_full, C)

fig, axes = plt.subplots(1, len(Q), figsize=(11, 4), sharey=True)
for q_i, ax in enumerate(axes):
    for a_i, a in enumerate(A):
        ax.plot(W_static, Ec_full[:, a_i, q_i], label=f"a={a:g}")
    ax.set_title(f"q={Q[q_i]:g}")
    ax.set_xlabel("w")
    ax.set_xlim([1.0, 5.0])
    ax.set_ylim([0.0, 2.25])
axes[0].set_ylabel("E(c | w, a, q)")
axes[-1].legend(title="Action", loc="lower right")
fig.suptitle("Consumption under full information")
fig.tight_layout()
plt.show()
```

With full information, output does not need to carry incentive rewards.

Consumption therefore depends primarily on the promise $w$ rather than
on output.

## The repeated economy

We now move from the one-period economy to finite- and infinite-horizon
contracts.

The planner maximizes discounted social surplus.

This can be interpreted as allowing society to borrow and lend at the constant
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

### The two-step factored algorithm

Solving the full LP over $(a,q,c,w')$ at each grid point is
computationally demanding.

We use a factored algorithm that splits each period into two sub-steps.

The split exploits the additive separability of the utility function

$$
U(a, c) = 2\sqrt{1-a} + 2\sqrt{c}.
$$

#### Step 1: action and output

Before consumption is assigned, the planner chooses the action, output,
and intermediate promised utility.

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

#### Step 2: consumption allocation

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

The function `solve_repeated_problem_2` implements one Bellman step using
the two-step algorithm.

The variables in the code follow the two sub-problems above.

`π_w_m` is the lottery over $(c,w')$ conditional on $w^m$, while
`π_w` is the lottery over $(a,q,w^m)$ conditional on $w$.

The function `solve_multi_period_economy_2` then repeats this Bellman
step to convergence, or works backward for a fixed number of periods
$T$.

```{code-cell} ipython3
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
    One Bellman update using the two-step algorithm.

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
    π_W_s1: 4-D array
        The probability of (a, q, w_m) given w.
    π_W_m_s2: 3-D array
        The probability of (c, w_prime) given w_m.
    '''
    
    n_A, n_Q, n_C, n_W = len(A), len(Q), len(C), len(W) 
    n_W_m, n_W_prime = len(W_m), len(W_prime)
    A_ind, Q_ind, C_ind = range(n_A), range(n_Q), range(n_C)
    W_ind, W_m_ind, W_prime_ind = range(n_W), range(n_W_m), range(n_W_prime)
    
    # Step 2
    Phi_s2 = np.array([[β * s_w_prime - c for s_w_prime in s_W_prime] for c in C])
    U_disc_s2 = np.array([[2 * c**0.5 + β * w_prime
                           for w_prime in W_prime] for c in C])
    
    w_m_para = cp.Parameter()
    π_w_m = cp.Variable((n_C, n_W_prime))

    obj_expr_s2 = cp.sum(cp.multiply(Phi_s2, π_w_m))
    obj_s2 = cp.Maximize(obj_expr_s2)
    
    C5_s2 = [cp.sum(cp.multiply(U_disc_s2, π_w_m)) == w_m_para]
    C7_s2 = [cp.sum(π_w_m) == 1] + [π_w_m >= 0]
    
    problem_s2 = cp.Problem(obj_s2, C5_s2 + C7_s2)
    
    s_W_m = np.zeros(n_W_m)
    π_W_m_s2 = np.zeros((n_W_m, n_C, n_W_prime))
    for w_m, w_m_ind in zip(W_m, W_m_ind):
        w_m_para.value = w_m
        problem_s2.solve(solver = cp.HIGHS)
        if problem_s2.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Step 2 LP failed at w_m={w_m}: "
                               f"{problem_s2.status}")
        s_W_m[w_m_ind] = obj_expr_s2.value
        π_W_m_s2[w_m_ind, :, :] = π_w_m.value
    
    # Step 1
    Phi_s1 = np.array([[(q+s_w_m) for s_w_m in s_W_m] for q in Q])
    U_disc_s1 = np.array([[[2 * (1 - a)**0.5 + w_m
                            for w_m in W_m] for q in Q] for a in A])
    U_disc_hat_s1 = np.array([[[[(2 * (1 - A[a_hat_ind])**0.5 + W_m[w_m_ind]) *\
                                 P[a_hat_ind, q_ind]/P[a_ind, q_ind] 
                                 for w_m_ind in W_m_ind] for q_ind in Q_ind]
                               for a_ind in A_ind] 
                              for a_hat_ind in A_ind])
    
    w_para = cp.Parameter()

    π_w = list(np.zeros(n_A))
    for a_ind in A_ind:
        π_w[a_ind] = cp.Variable((n_Q, n_W_m))
    
    obj_expr_s1 = cp.sum([cp.sum(cp.multiply(Phi_s1, π_w[a_ind]))
                          for a_ind in A_ind])
    obj_s1 = cp.Maximize(obj_expr_s1)

    C5_s1 = [cp.sum([cp.sum(cp.multiply(U_disc_s1[a_ind, :, :],
                                        π_w[a_ind]))
                     for a_ind in A_ind]) == w_para]
    C6_s1 = [(cp.sum(π_w[a_ind][q_ind, :]) == P[a_ind, q_ind] *\
              cp.sum(π_w[a_ind]))
             for q_ind in Q_ind for a_ind in A_ind]
    C7_s1 = [cp.sum([cp.sum(π_w[a_ind]) for a_ind in A_ind]) == 1]
    C7_s1 = C7_s1 + [(π_w[a_ind] >= 0) for a_ind in A_ind]

    problem_type = problem_type.lower()
    if problem_type == "full information":
        constraints_s1 = C5_s1 + C6_s1 + C7_s1
    else:
        C8_s1 = [(cp.sum(cp.multiply(U_disc_s1[a_ind, :, :],
                                     π_w[a_ind])) >=
                 cp.sum(cp.multiply(U_disc_hat_s1[a_hat_ind, a_ind, :, :],
                                    π_w[a_ind])))
                 for a_ind in A_ind for a_hat_ind in A_ind]
        constraints_s1 = C5_s1 + C6_s1 + C7_s1 + C8_s1

    problem_s1 = cp.Problem(obj_s1, constraints_s1)

    s_W = np.zeros(n_W)
    π_W_s1 = np.zeros((n_W, n_A, n_Q, n_W_m))
    for w, w_ind in zip(W, W_ind):
        w_para.value = w
        problem_s1.solve(solver = cp.HIGHS)
        if problem_s1.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Step 1 LP failed at w={w}: "
                               f"{problem_s1.status}")
        s_W[w_ind] = obj_expr_s1.value
        for a_ind in A_ind:
            π_W_s1[w_ind, a_ind, :, :] = π_w[a_ind].value
    return s_W, π_W_s1, π_W_m_s2
```

```{code-cell} ipython3
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
    Solve the finite- or infinite-horizon economy.
    
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
    π_W_s1: 4-D array
        The probability of (a, q, w_m) given w.
    π_W_m_s2: 3-D array
        The probability of (c, w_prime) given w_m.
    '''
    
    if β >= 1 or β <= 0:
        raise ValueError('β must lie in (0, 1)')
        
    def u(a, c):
        return c**0.5/0.5 + (1-a)**0.5/0.5
        
    if T is None:
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

        if s_W_0 is not None:
            s_W_prime = s_W_0
        else:
            s_W_prime = np.zeros(N)

        optimal = False
        iteration = 1
        while not optimal:
            if verbose:
                print('Iteration %i in process' % iteration)
            start_time = time()
            s_W, π_W_s1, π_W_m_s2 = solve_repeated_problem_2(W=W, W_m=W_m,
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
                print('---')
            
            if np.max(np.abs(s_W - s_W_prime)) <= tol:
                optimal = True
            else:
                s_W_prime = s_W
                
            iteration += 1
    
    if T is not None:
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
        
        if verbose:
            print('Solving the 1-period economy')
            print('---')
        s_W, π = solve_static_problem(W=W_mat[0, :], u=u,
                                       A=A, Q=Q, C=C, P=P,
                                       problem_type=problem_type)

        if T != 1:
            for t in range(2, T+1):
                if verbose:
                    print('Solving the %i-period economy' % t)
                    print('---')
                s_W_prime = np.copy(s_W)
                W_m_l = β*W_mat[t-2,:].min() + 2*C.min()**0.5
                W_m_u = β*W_mat[t-2,:].max() + 2*C.max()**0.5
                W_m = np.linspace(W_m_l, W_m_u, N_m)
                s_W, π_W_s1, π_W_m_s2 = solve_repeated_problem_2(W=W_mat[t-1,:],
                                                                   W_m=W_m, A=A,
                                                                   Q=Q, C=C, 
                                                                   W_prime=W_mat[t-2,:],
                                                                   s_W_prime=s_W_prime,
                                                                   P=P, 
                                                                   problem_type=problem_type,
                                                                   β=β)
    return s_W, π_W_s1, π_W_m_s2
```

### Improved solver: pre-built problems with Anderson acceleration

The original solver rebuilds all CVXPY problem objects on every
Bellman iteration, which causes memory to accumulate when many
iterations are needed.

This is a serious issue for $\beta$ close to 1.

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
      * CVXPY problems are built once with Parameter objects,
        so there is no memory leak across iterations.
      * Anderson acceleration (window m_anderson) reduces
        the number of Bellman iterations needed.
      * max_iter cap prevents unbounded runtime.

    Returns
    -------
    s_W       : 1-D array, converged surplus function on W
    π_W_s1   : 4-D array, π(a, q, w_m | w)
    π_W_m_s2 : 3-D array, π(c, w' | w_m)
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

    # Terms that do not change across iterations.
    U_disc_s2 = np.array([[2 * c**0.5 + β * wp
                            for wp in W] for c in C])
    U_disc_s1 = np.array([[[2 * (1 - a)**0.5 + wm
                             for wm in W_m] for q in Q]
                           for a in A])
    U_disc_hat_s1 = np.array([[[[
        (2 * (1 - A[ah])**0.5 + W_m[wmi]) * P[ah, qi] / P[ai, qi]
        for wmi in range(N_m)] for qi in Q_ind]
        for ai in A_ind] for ah in A_ind])

    # Step 2 problem.
    Phi_s2_param = cp.Parameter((n_C, N))
    w_m_para     = cp.Parameter()
    π_w_m        = cp.Variable((n_C, N))

    obj_expr_s2  = cp.sum(cp.multiply(Phi_s2_param, π_w_m))
    C5_s2 = [cp.sum(cp.multiply(U_disc_s2, π_w_m)) == w_m_para]
    C7_s2 = [cp.sum(π_w_m) == 1, π_w_m >= 0]
    problem_s2   = cp.Problem(cp.Maximize(obj_expr_s2), C5_s2 + C7_s2)

    # Step 1 problem.
    Phi_s1_param = cp.Parameter((n_Q, N_m))
    w_para       = cp.Parameter()
    π_w          = [cp.Variable((n_Q, N_m)) for _ in A_ind]

    obj_expr_s1  = cp.sum([cp.sum(cp.multiply(Phi_s1_param, π_w[ai]))
                            for ai in A_ind])
    C5_s1 = [cp.sum([cp.sum(cp.multiply(U_disc_s1[ai], π_w[ai]))
                     for ai in A_ind]) == w_para]
    C6_s1 = [(cp.sum(π_w[ai][qi, :]) ==
               P[ai, qi] * cp.sum(π_w[ai]))
              for qi in Q_ind for ai in A_ind]
    C7_s1 = ([cp.sum([cp.sum(π_w[ai]) for ai in A_ind]) == 1] +
              [π_w[ai] >= 0 for ai in A_ind])

    if problem_type == "full information":
        constraints_s1 = C5_s1 + C6_s1 + C7_s1
    else:
        C8_s1 = [(cp.sum(cp.multiply(U_disc_s1[ai], π_w[ai])) >=
                  cp.sum(cp.multiply(U_disc_hat_s1[ah, ai], π_w[ai])))
                 for ai in A_ind for ah in A_ind]
        constraints_s1 = C5_s1 + C6_s1 + C7_s1 + C8_s1

    problem_s1 = cp.Problem(cp.Maximize(obj_expr_s1), constraints_s1)

    s_W_prime = (np.array(s_W_0, dtype=float)
                 if s_W_0 is not None else np.zeros(N))

    hist_x, hist_fx = [], []
    err = np.inf

    for iteration in range(1, max_iter + 1):
        t0 = time()

        # Step 2.
        Phi_s2_param.value = np.array([[β * sv - c
                                         for sv in s_W_prime] for c in C])
        s_W_m     = np.zeros(N_m)
        π_W_m_s2 = np.zeros((N_m, n_C, N))
        for i, wm in enumerate(W_m):
            w_m_para.value = wm
            problem_s2.solve(solver=cp.HIGHS, warm_start=True)
            if problem_s2.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError(f"Step 2 LP failed at w_m={wm}: "
                                   f"{problem_s2.status}")
            s_W_m[i]       = obj_expr_s2.value
            π_W_m_s2[i]    = π_w_m.value

        # Step 1.
        Phi_s1_param.value = np.array([[(q + swm)
                                         for swm in s_W_m] for q in Q])
        s_W     = np.zeros(N)
        π_W_s1 = np.zeros((N, n_A, n_Q, N_m))
        for i, w in enumerate(W):
            w_para.value = w
            problem_s1.solve(solver=cp.HIGHS, warm_start=True)
            if problem_s1.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError(f"Step 1 LP failed at w={w}: "
                                   f"{problem_s1.status}")
            s_W[i] = obj_expr_s1.value
            for ai in A_ind:
                π_W_s1[i, ai] = π_w[ai].value

        t1 = time()
        err = np.max(np.abs(s_W - s_W_prime))
        if verbose:
            print(f"Iter {iteration:3d}: max|ΔsW| = {err:.2e}  ({t1-t0:.1f}s)")

        if err <= tol:
            if verbose:
                print(f"Converged in {iteration} iterations.")
            break

        # Anderson acceleration.
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

    return s_W, π_W_s1, π_W_m_s2, W
```

### Dynamic allocations

We use the same parameters as for the static economy, plus a
discount factor $\beta = 0.8$ and grids of $N = N_m = 100$ points.

#### Initial values

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
s_W_0, π_0 = solve_static_problem(W * (1 - β), u,
                                    A, Q, C, P,
                                    "unobserved-actions")
out_time = time()
print("Time(s):", round(out_time - in_time, 3))
```

#### Finite-period economy

```{code-cell} ipython3
in_time = time()
s_W_T, π_W_s1_T, π_W_m_s2_T = solve_multi_period_economy_2(
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
plt.plot(W_T, s_W_T, "k-.", label="Three-period hidden effort")
plt.title("Finite-horizon surplus frontier", y=-0.2)
plt.legend()
plt.show()
```

This finite-horizon computation is a useful check on the recursion.

The three-period surplus function already has the shape of the
infinite-horizon frontier, but it is still affected by the approaching
terminal date because continuation promises have value for only a few
periods.

#### Infinite-period economy

```{code-cell} ipython3
:tags: [hide-output]

in_time = time()
s_W, π_W_s1, π_W_m_s2 = solve_multi_period_economy_2(
    A, Q, C, P, "unobserved-actions",
    N=N, N_m=N_m,
    s_W_0=s_W_0 / (1 - β),
    tol=1e-8)
out_time = time()
print("Time(s):", round(out_time - in_time, 3))
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
plt.plot(W, s_W, "k-.", label="Infinite-horizon hidden effort")
plt.xlim([5.0, 25.0])
plt.ylim([-7.5, 10.0])
plt.xlabel("w")
plt.ylabel("s(w)")
plt.title("Infinite-horizon surplus frontier", y=-0.2)
plt.legend()
plt.show()
```

The infinite-horizon solution removes the terminal-date effect.

At each promised utility, the surplus function is the fixed point of the
Bellman operator: the current lottery and the continuation promise are
jointly chosen so that tomorrow's promise is priced by the same surplus
function plotted here.

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
π = np.einsum("waqm,mcx->waqcx", π_W_s1, π_W_m_s2)
```

The `einsum` line is just the law of total probability.

It sums over the intermediate promise $w^m$ and reconstructs the full
lottery over $(a,q,c,w')$.

#### Surplus and history dependence

```{code-cell} ipython3
W_full = np.linspace(5, 25, N)

in_time = time()
s_W_1, π_1 = solve_static_problem(W_full*(1-β), u, A,
                                   Q, C, P, "full information")
out_time = time()

print("Time(s):", round(out_time - in_time, 3))
```

```{code-cell} ipython3
plt.figure(figsize=(6.5, 6.5))
plt.plot(W_full, s_W_1/(1 - β), label="Full information")
plt.plot(W, s_W, "k-.", label="Repeated hidden effort")
plt.plot(W, s_W_0/(1 - β), label="Static hidden effort")
plt.xlim([5.0, 25.0])
plt.ylim([-7.5, 10.0])
plt.hlines(0, 5.0, 25.0, linestyle="dashed")
plt.xlabel("w")
plt.ylabel("s(w)")
plt.title("History dependence and surplus", y=-0.2)
plt.legend()
plt.show()
```

This comparison separates two forces.

The full-information frontier is highest because effort can be controlled
directly.

The infinite-horizon hidden-effort frontier is below it because
incentive constraints remain, but it lies above the frontier obtained by
repeating the one-period hidden-effort contract.

The difference between the two unobserved-action curves is the gain from
history dependence.

#### Effort and history dependence

```{code-cell} ipython3
X, Y = list(range(len(A))), list(range(len(Q)))
Ea_1 = np.einsum('a,waqc->w', A, π_0)
Ea_inf = np.einsum('a,waqcx->w', A, π)

plt.figure(figsize=(6.5, 6.5))
plt.plot(W, Ea_inf, label="Repeated hidden effort")
plt.plot(W, Ea_1, label="Static hidden effort")
plt.xlabel("w")
plt.ylabel("E{a(w)}")
plt.xlim([5.0, 25.0])
plt.ylim([0.0, 0.8])
plt.title("Effort with and without history dependence", y=-0.2)
plt.legend()
plt.show()
```

History dependence also raises effort relative to repeated one-period
contracts.

Near the lower utility bound, incentive compatibility forces
low effort, but away from that bound continuation promises help provide
incentives without relying only on current consumption.

#### Current consumption

The full lottery $\pi(w,a,q,c,w')$ is high-dimensional.

To read it, we summarize it by conditional means.

The next helper computes $E[c \mid w,a,q]$, first summing over
continuation promises and then normalizing by the probability of the
conditioning event.

```{code-cell} ipython3
def expected_consumption(π, C):
    """
    E[c | w, a, q] from either π(w,a,q,c) or π(w,a,q,c,w').
    """
    if π.ndim == 4:
        mass = π.sum(axis=3)
        total = np.einsum("c,waqc->waq", C, π)
    else:
        mass = π.sum(axis=(3, 4))
        total = np.einsum("c,waqcx->waq", C, π)

    return np.divide(total, mass,
                     out=np.full_like(total, np.nan, dtype=float),
                     where=mass > 1e-12)
```

```{code-cell} ipython3
Ec_inf = expected_consumption(π, C)

fig, axes = plt.subplots(1, len(Q), figsize=(11, 4), sharey=True)
for q_i, ax in enumerate(axes):
    for a_i, a in enumerate(A):
        ax.plot(W, Ec_inf[:, a_i, q_i], label=f"a={a:g}")
    ax.set_title(f"q={Q[q_i]:g}")
    ax.set_xlabel("w")
    ax.set_xlim([5.0, 25.0])
    ax.set_ylim([0.0, 2.25])
axes[0].set_ylabel("E(c | w, a, q)")
axes[-1].legend(title="Action", loc="lower right")
fig.suptitle("Current consumption in the repeated contract")
fig.tight_layout()
plt.show()
```

The repeated contract smooths current consumption relative to the static
hidden-effort economy.

Output still affects rewards,
but a large part of the reward and punishment is shifted into future
promised utility.

#### Continuation promises

The parallel statistic for the dynamic margin is
$E[w' \mid w,a,q]$.

This is the object that reveals how the contract uses future utility as a
reward or punishment.

```{code-cell} ipython3
def expected_promise(π, W):
    """
    E[w' | w, a, q] from π(w,a,q,c,w').
    """
    mass = π.sum(axis=(3, 4))
    total = np.einsum("x,waqcx->waq", W, π)
    return np.divide(total, mass,
                     out=np.full_like(total, np.nan, dtype=float),
                     where=mass > 1e-12)
```

```{code-cell} ipython3
Ew_inf = expected_promise(π, W)


fig, axes = plt.subplots(1, len(Q), figsize=(11, 4), sharey=True)
for q_i, ax in enumerate(axes):
    for a_i, a in enumerate(A):
        ax.plot(W, Ew_inf[:, a_i, q_i], label=f"a={a:g}")
    ax.plot(W, W, "k-.", label="45-degree line")
    ax.set_title(f"q={Q[q_i]:g}")
    ax.set_xlabel("w")
    ax.set_xlim([10.0, 25.0])
    ax.set_ylim([10.0, 25.0])
axes[0].set_ylabel("E(w' | w, a, q)")
axes[-1].legend(title="Action", loc="lower right")
fig.suptitle("Continuation promises")
fig.tight_layout()
plt.show()
```

This plot displays the expected next-period promise conditional on
current $w$, recommended action $a$, and realized output $q$.

High
output generally raises continuation utility and low output lowers it.

At the endpoints of the feasible promise set, the transition stays on the
45-degree line because only the corresponding extreme plan can deliver
that endpoint.

For the simulations, we use a higher discount factor, $\beta = 0.95$.

The higher discount factor makes promised utility a stronger incentive
instrument and makes the evolution of individual histories easier to
see.

We now use `solve_multi_period_economy_vfi`, which builds the CVXPY
problems once and applies Anderson acceleration, to solve the
infinite-horizon economy at $\beta = 0.95$ with a grid of $N = N_m = 50$
points.

Starting from the static solution rescaled to discounted-sum units, the
iteration converges to tolerance $10^{-4}$.

```{code-cell} ipython3
β_95  = 0.95
N_95  = 50
N_m95 = 50

w_l_95 = u(A.min(), C.min()) / (1 - β_95)
w_u_95 = u(A.min(), C.max()) / (1 - β_95)
W_95   = np.linspace(w_l_95, w_u_95, N_95)

s_W_0_95, _ = solve_static_problem(W_95 * (1 - β_95), u, A, Q, C, P,
                                    "unobserved-actions")

in_time = time()
s_W_new, π_W_s1_new, π_W_m_s2_new, W_new = solve_multi_period_economy_vfi(
    A, Q, C, P, "unobserved-actions",
    β=β_95, N=N_95, N_m=N_m95,
    s_W_0=s_W_0_95 / (1 - β_95),
    tol=1e-4, max_iter=300, m_anderson=5,
    verbose=True)
out_time = time()
print("Time(s):", round(out_time - in_time, 3))
```

```{code-cell} ipython3
π_new = np.einsum("waqm,mcx->waqcx", π_W_s1_new, π_W_m_s2_new)
```

For the simulation, a state is a current promise grid point.

Given that state, the code draws an action, then output, then a pair
$(c,w')$ from the joint lottery.

The next period's state is the realized $w'$.

```{code-cell} ipython3
def draw_from(probabilities, rng):
    probabilities = np.maximum(np.asarray(probabilities, dtype=float), 0.0)
    total = probabilities.sum()
    if total <= 1e-12:
        return rng.integers(len(probabilities))
    probabilities = probabilities / total
    return min(np.searchsorted(np.cumsum(probabilities), rng.random()),
               len(probabilities) - 1)


def simulation(W, C, s_W, T, π, seed=12345):
    w_index = np.nanargmin(np.abs(s_W))
    rng = np.random.default_rng(seed)
    
    w_series = np.empty(T + 1)
    c_series = np.empty(T)
    w_series[0] = W[w_index]
    
    for i in range(T):
        joint = np.maximum(π[w_index], 0.0)
        
        a_index = draw_from(joint.sum(axis=(1, 2, 3)), rng)
        q_index = draw_from(joint[a_index].sum(axis=(1, 2)), rng)
        
        cw_prob = joint[a_index, q_index]
        cw_index = draw_from(cw_prob.ravel(), rng)
        c_index, w_next_index = np.unravel_index(cw_index, cw_prob.shape)
        
        c_series[i] = C[c_index]
        w_index = w_next_index
        w_series[i + 1] = W[w_index]
    
    return c_series, w_series
```

```{code-cell} ipython3
c_series = np.zeros((80, 4))
w_series = np.zeros((81, 4))
for i in range(4):
    c_series[:, i], w_series[:, i] = simulation(
        W_new, C, s_W_new, 80, π_new, seed=(12345 + i))
```

The simulations start from the grid point at which surplus is closest to
zero.

This corresponds to the ex ante symmetric, or "fair", allocation.

It is the highest common promised utility that can be assigned while
keeping discounted social surplus nonnegative.

#### Simulated consumption histories

```{code-cell} ipython3
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
plt.title("Individual consumption histories ($\\beta=0.95$)", y=-0.2)
plt.show()
```

The four consumption paths differ even though all agents begin with the
same promised utility.

Different output histories move agents to different continuation
promises, so the contract gradually creates heterogeneous consumption
histories.

#### Simulated promised utilities

```{code-cell} ipython3
date_w = np.arange(81)
plt.figure(figsize=(6.5, 6.5))
plt.plot(date_w, w_series[:, 0])
plt.plot(date_w, w_series[:, 1])
plt.plot(date_w, w_series[:, 2])
plt.plot(date_w, w_series[:, 3])
plt.xlabel("date")
plt.ylabel("expected utility")
plt.ylim([40.0, 100.0])
plt.title("Individual promised utilities ($\\beta=0.95$)", y=-0.2)
plt.show()
```

The promised-utility paths show the state variable moving directly.

High-output histories tend to move the agent upward, while low-output
histories move the agent downward.

This is the dynamic incentive mechanism in the model.

```{code-cell} ipython3
def population_distributions(W, C, s_W, T, π):
    w_index = np.nanargmin(np.abs(s_W))
    μ = np.zeros(len(W))
    μ[w_index] = 1.0

    π_pos = np.maximum(π, 0.0)
    row_sums = π_pos.sum(axis=(1, 2, 3, 4), keepdims=True)
    π_pos = np.divide(π_pos, row_sums,
                       out=np.zeros_like(π_pos),
                       where=row_sums > 1e-12)

    π_c = np.zeros((T, len(C)))
    π_w = np.zeros((T, len(W)))

    for t in range(T):
        joint = np.tensordot(μ, π_pos, axes=(0, 0))
        π_c[t] = joint.sum(axis=(0, 1, 3))
        μ = joint.sum(axis=(0, 1, 2))
        μ = μ / μ.sum()
        π_w[t] = μ

    return π_c, π_w


π_c, π_w = population_distributions(W_new, C, s_W_new, 80, π_new)
```

The distribution calculation above keeps the whole population rather
than drawing individual sample paths.

Starting from a point mass over $w$, it applies the optimal lottery each
period and records the implied marginal distributions of consumption and
promised utility.

#### Cross-sectional consumption distributions

```{code-cell} ipython3
%matplotlib inline

date_mat_c = np.reshape(np.arange(80) + 1, (80, 1)) * \
             np.ones((1, len(C)))
c_mat = np.ones((80, 1)) @ np.reshape(C, (1, len(C)))

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(projection='3d')
plt.title("Consumption distribution over time ($\\beta=0.95$)", y=-0.3)
plt.xlabel('date')
plt.ylabel('consumption')
ax.set_zlabel('percentage')

ax.set_zlim(0.0, 1.0)
ax.view_init(elev=25, azim=-65)
wire = ax.plot_wireframe(date_mat_c, c_mat, π_c,
                         rstride=1, cstride=2,
                         color="black", linewidth=0.35)
plt.show()
```

The consumption distribution spreads out over time because histories
receive different rewards and punishments.

On the finite grid, some mass eventually reaches the edges of the
feasible promise set.

#### Cross-sectional promise distributions

```{code-cell} ipython3
%matplotlib inline
date_mat_w = np.reshape(np.arange(80) + 1, (80, 1)) * \
                np.ones((1, len(W_new)))
W_mat_12 = np.ones((80, 1)) @ np.reshape(W_new, (1, len(W_new)))

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(projection='3d')
plt.title("Promise distribution over time ($\\beta=0.95$)", y=-0.3)
plt.xlabel('date')
plt.ylabel('w')
ax.set_zlabel('percentage')

ax.set_zlim(0.0, 1.0)
ax.view_init(elev=25, azim=-65)
wire = ax.plot_wireframe(date_mat_w, W_mat_12, π_w,
                         rstride=1, cstride=2,
                         color="black", linewidth=0.35)
plt.show()
```

The promise distribution is the deeper state-space picture behind the
consumption distribution.

It shows how repeated incentives convert a common initial promise into a
distribution of continuation utilities.

## Concluding remarks

### Economics

#### Moral hazard and the cost of private information

When the principal cannot observe the agent's effort, the optimal contract
must balance two competing objectives: *insurance* (smoothing the agent's
consumption across output realizations) and *incentives* (rewarding high
output to make effort attractive).

The hidden-effort surplus frontier lies below the full-information
frontier, and the gap between them measures the surplus cost of
unobserved effort.

#### Dynamic contracts and promised utility

The recursive formulation of {cite}`Spear_Srivastava_87` compresses all
payoff-relevant history into a single scalar state: the discounted
expected continuation utility $w$ that the principal has promised the
agent.

By tracking $w$ rather than the full history of outputs, the dynamic
contracting problem becomes tractable.

In the optimal infinite-horizon contract, the principal rewards high
output by granting the agent a higher continuation utility and punishes
low output by lowering it.

Continuation promises
therefore substitute partly for large contemporaneous consumption
spreads.

#### Diversity over time

The simulations illustrate the central computational message:
starting from a common initial promise, dynamic incentives generate
non-trivial individual histories and cross-sectional dispersion in
consumption and promised utility.

With the finite grids used here, the endpoints of the promise set are
absorbing.

The simulations should therefore be read as finite-grid
illustrations of how history dependence spreads the distribution over
time, not as a separate theorem about the limiting distribution.

### Technical tricks

#### Lotteries and convexification

Incentive constraints can render the set of feasible contracts non-convex,
making standard optimization techniques unreliable.

{cite}`Phelan_Townsend_91` circumvented this by allowing the planner to
choose a joint *lottery* $\Pi(a, q, c, w')$ over actions, outputs,
consumptions, and continuation values.

Because any mixture of feasible lotteries is itself feasible, the
constraint set becomes convex, and global optima are well-defined.

#### Linear programming

With finite grids, the convexified Bellman equation is a linear program:
the objective $(q - c + \beta v(w'))$ and every constraint are linear in
$\Pi$.

Treating $v(w')$ as a fixed vector from the previous iteration, value
function iteration reduces to solving one LP per grid point per
iteration, a task handled efficiently by modern LP solvers such as
HiGHS.

#### Dynamic programming

The promised-utility state variable $w$ makes the problem recursive.

At each iteration the Bellman operator maps a surplus function $v$ to an
updated surplus function $Tv$; repeated application converges to the
infinite-horizon fixed point.

The implementation initializes the iteration from the scaled static
solution, which is a useful numerical starting point.

#### Two-step factored algorithm

The additive separability $U(a,c) = 2\sqrt{1-a} + 2\sqrt{c}$ allows the
four-dimensional LP to be split into two smaller sub-problems.

Step 2 allocates consumption given an intermediate promised utility
$w^m$; Step 1 assigns actions, outputs, and intermediate continuation
utilities given $w$.

Because each sub-LP has far fewer decision variables than the full joint
LP, computation is substantially faster and the approach scales to finer
grids.

#### Dynamic programming squared

This lecture is closely related to what Lars Ljungqvist and Thomas
Sargent call *dynamic programming squared* in
{cite}`Ljungqvist2012`.

The phrase refers to recursive problems in which one continuation object
is carried as a state variable inside another recursive problem.

Here the surplus function $s(w)$, the solution to the principal's
outer dynamic program, has the agent's continuation utility $w$ as its
state variable, while feasible movements in $w$ are governed by
promise-keeping and incentive constraints.

The same architecture reappears throughout this lecture series.

In {doc}`Stackelberg plans <dyn_stack>` the Stackelberg leader's
value function takes the followers' competitive-equilibrium value
function as an argument.

In {doc}`Optimal Taxation with State-Contingent Debt <opt_tax_recur>`,
a Ramsey planner's outer Bellman equation uses the household's
marginal utility of wealth $x$, itself defined by an inner
implementability constraint, as its state variable.

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
state variable (a promised utility, a marginal value, or a
continuation value) that restricts what the outer dynamic
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

```{solution-start} repeat_mh_ex1
:class: dropdown
```

```{code-cell} ipython3
delta_W = s_W_full - s_W_unobs

plt.figure(figsize=(6.5, 6.5))
plt.plot(W_static, delta_W)
plt.xlabel("w")
plt.ylabel(r"$\delta(w) = s^{FI}(w) - s^{UA}(w)$")
plt.xlim([1.0, 5.0])
plt.ylim(bottom=0.0)
plt.title("Agency cost in the static model", y=-0.2)
plt.show()

max_i = np.nanargmax(delta_W)
w_hat = W_static[max_i]
print(f"Largest agency cost at w = {w_hat:.3f},  δ = {delta_W[max_i]:.4f}")
```

Agency costs are highest near intermediate levels of promised utility
because at those values the principal most values inducing high effort
(output is valuable) while the agent still requires meaningful
consumption-state variation to be incentivized.

At low $w$ the agent is near subsistence and effort is low anyway;
at high $w$ the agent is nearly fully insured and the marginal incentive
cost of each additional unit of effort is small.

```{solution-end}
```

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

```{solution-start} repeat_mh_ex2
:class: dropdown
```

```{code-cell} ipython3
P_flat = np.array([[0.70, 0.30],
                   [0.55, 0.45],
                   [0.45, 0.55],
                   [0.30, 0.70]])

s_W_flat, π_flat = solve_static_problem(W_static, u, A, Q, C, P_flat,
                                          "unobserved-actions")
Ea_flat = np.einsum('a,waqc->w', A, π_flat)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

axes[0].plot(W_static, s_W_unobs, label="Baseline $P$")
axes[0].plot(W_static, s_W_flat,  label="Flat $P$")
axes[0].hlines(0, 1.0, 5.0, linestyle="dashed")
axes[0].set_xlabel("w")
axes[0].set_ylabel("s(w)")
axes[0].set_xlim([1.0, 5.0])
axes[0].set_title("Surplus function", y=-0.2)
axes[0].legend()

axes[1].plot(W_static, Ea_unobs, label="Baseline $P$")
axes[1].plot(W_static, Ea_flat,  label="Flat $P$")
axes[1].set_xlabel("w")
axes[1].set_ylabel(r"$E\{a(w)\}$")
axes[1].set_xlim([1.0, 5.0])
axes[1].set_ylim([0.0, 0.8])
axes[1].set_title("Expected effort", y=-0.2)
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
than under the baseline $P$: the surplus function shifts down and
expected effort falls.

```{solution-end}
```
