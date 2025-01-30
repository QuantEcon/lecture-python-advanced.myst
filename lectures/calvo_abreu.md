---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(calvo_abreu)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Sustainable Plans for a Calvo Model

```{index} single: Models; Additive functionals
```



## Overview


This is a sequel to this  quantecon lecture {doc}`calvo`.

That lecture studied a linear-quadratic version of a model that Guillermo Calvo {cite}`Calvo1978` used to study  the **time inconsistency** of the optimal government plan that emerges when  a **Stackelberg** government  (a.k.a. a **Ramsey planner**) at time $0$ once and for all chooses  a sequence $\vec \mu = \{\mu_t\}_{t=0}^\infty$ of gross rates of growth in the supply of money.

A consequence of that choice is a (rational expectations equilibrium)  sequence $\vec \theta = \{\theta_t\}_{t=0}^\infty$  of gross rates of increase in the price level that  we call inflation rates. 

{cite}`Calvo1978` showed that  a  Ramsey plan  would
not emerge from  alternative timing protocols and associated supplementary assumptions about what government authorities who set $\mu_t$ at time $t$ believe how future government authorities who set $\mu_{t+j}$ for $j >0$ will respond to their decisions.  

In this lecture, we explore another set of assumptions about what government authorities who set $\mu_t$ at time $t$ believe how future government authorities who set $\mu_{t+j}$ for $j >0$ will respond to their decisions.


We shall assume that there is  sequence of separate policymakers;  a time $t$ policymaker chooses  only $\mu_t$,  but now  believes that its choice of $\mu_t$  shapes the representative agent's beliefs about  future rates of money creation and inflation, and through them, future government actions.

This timing protocol and belief structure leads to a  model of a  **credible government policy**, also known as a **sustainable plan**.

In  quantecon lecture {doc}`calvo`  we used  ideas from  papers by Cagan {cite}`Cagan`, Calvo {cite}`Calvo1978`, and  Chang {cite}`chang1998credible`.

In addition to those ideas,  we'll also use ideas from Abreu {cite}`Abreu`, Stokey {cite}`stokey1989reputation`, {cite}`Stokey1991`, and 
Chari and Kehoe {cite}`chari1990sustainable` to study outcomes under 
our    timing protocol.




## Model Components

We'll start with a brief review of the setup.  

There is no uncertainty.

Let

- $p_t$ be the log of the price level
- $m_t$ be the log of nominal money balances
- $\theta_t = p_{t+1} - p_t$ be the net rate of inflation between $t$ and $t+1$
- $\mu_t = m_{t+1} - m_t$ be the net rate of growth of nominal balances

The demand for real balances is governed by a perfect foresight
version of a Cagan {cite}`Cagan` demand function for real balances:

```{math}
:label: eq_old1_new

m_t - p_t = -\alpha(p_{t+1} - p_t) \: , \: \alpha > 0
```

for $t \geq 0$.

Equation {eq}`eq_old1_new` asserts that the demand for real balances is inversely
related to the public's expected rate of inflation, which  equals
the actual rate of inflation because there is no uncertainty here.

(When there is no uncertainty, an assumption of **rational expectations** that becomes equivalent to  **perfect foresight**).


Subtracting the demand function {eq}`eq_old1_new` at time $t$ from the demand
function at $t+1$ gives:

$$
\mu_t - \theta_t = -\alpha \theta_{t+1} + \alpha \theta_t
$$

or

```{math}
:label: eq_old2_new

\theta_t = \frac{\alpha}{1+\alpha} \theta_{t+1} + \frac{1}{1+\alpha} \mu_t
```

Because $\alpha > 0$,  $0 < \frac{\alpha}{1+\alpha} < 1$.

**Definition:** For  scalar $b_t$, let $L^2$ be the space of sequences
$\{b_t\}_{t=0}^\infty$ satisfying

$$
\sum_{t=0}^\infty  b_t^2 < +\infty
$$

We say that a sequence that belongs to $L^2$   is **square summable**.

When we assume that the sequence $\vec \mu = \{\mu_t\}_{t=0}^\infty$ is square summable and we require that the sequence $\vec \theta = \{\theta_t\}_{t=0}^\infty$ is square summable,
the linear difference equation {eq}`eq_old2_new` can be solved forward to get:

```{math}
:label: eq_old3_new

\theta_t = \frac{1}{1+\alpha} \sum_{j=0}^\infty \left(\frac{\alpha}{1+\alpha}\right)^j \mu_{t+j}
```

**Insight:** In the spirit of Chang {cite}`chang1998credible`,  equations {eq}`eq_old1_new` and {eq}`eq_old3_new` show that $\theta_t$ intermediates
how choices of $\mu_{t+j}, \ j=0, 1, \ldots$ impinge on time $t$
real balances $m_t - p_t = -\alpha \theta_t$.

An equivalence class of continuation money growth sequences $\{\mu_{t+j}\}_{j=0}^\infty$ deliver the same $\theta_t$.

That future rates of money creation influence earlier rates of inflation
makes  timing protocols matter for modeling optimal government policies.

Quantecon lecture {doc}`calvo` used this  insight to  simplify   analysis of alternative  government policy problems.


## Another  Timing Protocol

The Quantecon lecture {doc}`calvo`  considered three  models of government policy making that  differ in

- *what* a  policymaker chooses, either a sequence
  $\vec \mu$ or just   $\mu_t$ in a single period $t$.
- *when* a  policymaker chooses, either once and for all at time $0$, or at some time or times  $t \geq 0$.
- what a policymaker *assumes* about how its choice of $\mu_t$
  affects the representative  agent's expectations about 
  inflation rates.

In this lecture,  there is a sequence of policymakers, each of whom
sets $\mu_t$ at  $t$ only.

To set the stage, recall that in a **Markov perfect equilibrium** 

- a time $t$  policymaker cares only about $v_t$ and  ignores  effects that its choice of $\mu_t$ has on $v_s$ at  dates $s = 0, 1, \ldots, t-1$.

In particular, in a Markov perfect equilibrium, there is a sequence indexed by $t =0, 1, 2, \ldots$ of separate policymakers;  a time $t$ policymaker chooses $\mu_t$ only and forecasts that future government decisions are unaffected by its choice.

By way of contrast, in this lecture there is  sequence of distinct policymakers;  a time $t$ policymaker chooses  only $\mu_t$,  but now  believes that its choice of $\mu_t$  shapes the representative agent's beliefs about  future rates of money creation and inflation, and through them, future government actions.

This timing protocol and belief structure leads to a  model of a  **credible government policy** also known as a **sustainable plan**

The relationship between  outcomes under a  (Ramsey) timing protocol and the  timing protocol and belief structure in this lecture  is the subject of a literature on **sustainable** or **credible** public policies  created by Abreu {cite}`Abreu`,  {cite}`chari1990sustainable`
{cite}`stokey1989reputation`, and Stokey {cite}`Stokey1991`. 


They discovered conditions under which a Ramsey plan can be rescued from the complaint that it is not credible.

They  accomplished this by expanding the
description of a plan to include expectations about *adverse consequences* of deviating from
it that can serve to deter deviations.

In this version of our model 

- the government does not set $\{\mu_t\}_{t=0}^\infty$  once and
  for all at $t=0$
- instead it sets $\mu_t$ at time $t$ 
- the representative agent's forecasts of
  $\{\mu_{t+j+1}, \theta_{t+j+1}\}_{j=0}^\infty$ respond to
  whether the government at $t$ *confirms* or *disappoints*
  its forecasts of $\mu_t$ brought into period $t$ from
  period $t-1$.
- the government at each time $t$ understands how the representative agent's
  forecasts will respond to its choice of $\mu_t$.
- at each $t$, the government chooses $\mu_t$ to maximize
  a continuation discounted utility.

### Government Decisions

$\vec \mu$ is chosen by a sequence of government
decision makers, one for each $t \geq 0$.

The time $t$ decision maker chooses $\mu_t$.

We assume the following within-period and between-period timing protocol
for each $t \geq 0$:

- at time $t-1$, private agents expect  that the government will set
  $\mu_t = \tilde \mu_t$, and more generally that it will set
  $\mu_{t+j} = \tilde \mu_{t+j}$ for all $j \geq 0$.
- The forecasts $\{\tilde \mu_{t+j}\}_{j \geq 0}$ determine a
  $\theta_t = \tilde \theta_t$ and an associated log
  of real balances $m_t - p_t = -\alpha\tilde \theta_t$ at
  $t$.
- Given those expectations and an associated $\theta_t = \tilde \theta_t$, at
  $t$ a government is free to set $\mu_t \in {\bf R}$.
- If the government at $t$ *confirms* the representative agent's
  expectations by setting $\mu_t = \tilde \mu_t$ at time
  $t$, private agents expect the continuation government policy
  $\{\tilde \mu_{t+j+1}\}_{j=0}^\infty$ and therefore bring
  expectation $\tilde \theta_{t+1}$ into period $t+1$.
- If the government at $t$ *disappoints* private agents by setting
  $\mu_t \neq \tilde \mu_t$, private agents expect
  $\{\mu^A_j\}_{j=0}^\infty$ as the
  continuation policy for $t+1$, i.e.,
  $\{\mu_{t+j+1}\} = \{\mu_j^A \}_{j=0}^\infty$ and therefore
  expect an associated $\theta_0^A$ for $t+1$. Here $\vec \mu^A = \{\mu_j^A \}_{j=0}^\infty$ is
  an alternative government plan to be described below.

### Temptation to Deviate from Plan

The government's one-period return function $s(\theta,\mu)$
described in equation {eq}`eq_old6` in quantecon lecture {cite}`Calvo1978`  has the property that for all
$\theta$

$$
s(\theta, 0) \geq  s(\theta, \mu) \quad
$$

This inequality implies that whenever the policy calls for the
government to set $\mu \neq 0$, the government could raise its
one-period payoff by setting $\mu =0$.

Disappointing private sector expectations in that way would increase the
government's *current* payoff but would have adverse consequences for
*subsequent* government payoffs because the private sector would alter
its expectations about future settings of $\mu$.

The *temporary* gain constitutes the government's temptation to
deviate from a plan.

If the government at $t$ is to resist the temptation to raise its
current payoff, it is only because it forecasts adverse  consequences that
its setting of $\mu_t$ would bring for continuation  government payoffs via  alterations  in the private sector's expectations.

## Sustainable or Credible Plan

We call a plan $\vec \mu$ **sustainable** or **credible** if at
each $t \geq 0$ the government chooses to confirm private
agents' prior expectation of its setting for $\mu_t$.

The government will choose to confirm prior expectations only if the
long-term *loss* from disappointing private sector expectations --
coming from the government's understanding of the way the private sector
adjusts its  expectations in response to having its prior
expectations at $t$ disappointed -- outweigh the short-term
*gain* from disappointing those expectations.

The theory of sustainable or credible plans assumes throughout that private sector
expectations about what future governments will do are based on the
assumption that governments at times $t \geq 0$ always act to
maximize the continuation discounted utilities that describe those
governments' purposes.

This aspect of the theory means that credible plans always come in *pairs*:

- a credible (continuation) plan to be followed if the government at
  $t$ *confirms* private sector expectations
- a credible plan to be followed if the government at $t$
  *disappoints* private sector expectations

That credible plans come in pairs threaten to bring an explosion of plans to keep track of

* each credible plan itself consists of two credible plans
* therefore, the number of plans underlying one plan is unbounded

But Dilip Abreu showed how to render manageable the number of plans that must be kept track of.

The key is an  object called a **self-enforcing** plan.

We'll proceed to compute one.

In addition to what's in Anaconda, this lecture will use  the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade quantecon
```

We'll start with some imports:

```{code-cell} ipython3
import numpy as np
from quantecon import LQ
import matplotlib.pyplot as plt
import pandas as pd
```

### Abreu's Self-Enforcing Plan

A plan $\vec \mu^A$ (here the superscipt $A$ is for Abreu) is said to be **self-enforcing** if

- the consequence of disappointing the representative agent's expectations at time
  $j$ is to *restart*  plan $\vec \mu^A$  at time $j+1$
- the consequence of restarting the plan is sufficiently adverse that it forever deters all
  deviations from the plan

More precisely, a government plan $\vec \mu^A$ with equilibrium inflation sequence $\vec \theta^A$ is
**self-enforcing** if

```{math}
:label: eq_old10

\begin{aligned}
v_j^A & = s(\theta^A_j, \mu^A_j) + \beta v_{j+1}^A \\
& \geq s(\theta^A_j, 0 ) + \beta v_0^A \equiv v_j^{A,D}, \quad j \geq 0
\end{aligned}
```

(Here it is useful to recall that setting $\mu=0$ is the maximizing choice for the government's one-period return function)

The first line tells the consequences of confirming the representative agent's
expectations by following the plan, while the second line tells the consequences of
disappointing the representative agent's expectations by deviating from the plan.

A consequence of the inequality stated in the  definition is that a self-enforcing plan is
credible.

Self-enforcing plans can be used to construct other credible plans, including ones with better values.

Thus, where $\vec v^A$ is the value associated with a self-enforcing plan $\vec \mu^A$,
a sufficient condition for another plan $\vec \mu$ associated with inflation $\vec \theta$ and value $\vec v$  to be **credible** is that

```{math}
:label: eq_old100a

\begin{aligned}
v_j & = s( \theta_j, \mu_j) + \beta  v_{j+1} \\
& \geq  s( \theta_j, 0) + \beta v_0^A \quad \forall j \geq 0
\end{aligned}
```

For this condition to be satisfied it is necessary and sufficient that

$$
s( \theta_j, 0) - s( \theta_j, \mu_j)  <  \beta ( v_{j+1} - v_0^A )
$$

The left side of the above inequality is the government's *gain* from deviating from the plan, while the right side is the government's *loss* from deviating
from the plan.

A government never wants to deviate from a credible plan.

Abreu taught us that  key step in constructing a credible plan is first constructing a
self-enforcing plan that has a low time $0$ value.

The idea is to use the self-enforcing plan as a continuation plan whenever
the government's choice at time $t$ fails to confirm private
agents' expectation.

We shall use a construction featured in Abreu ({cite}`Abreu`) to construct a
self-enforcing plan with low time $0$ value.

### Abreu's Carrot-Stick Plan

{cite}`Abreu` invented a way to create a self-enforcing plan with a low
initial value.

Imitating his idea, we can construct a self-enforcing plan
$\vec \mu$ with a low time $0$ value to the government by
insisting that future government decision makers set $\mu_t$ to a value yielding low
one-period utilities to the household for a long time, after which
government  decisions thereafter  yield high one-period utilities.

- Low one-period utilities early are a *stick*
- High one-period utilities later are a *carrot*

Consider a candidate plan $\vec \mu^A$ that sets
$\mu_t^A = \bar \mu$ (a high positive
number) for $T_A$ periods, and then reverts to the Ramsey plan.

Denote this sequence by $\{\mu_t^A\}_{t=0}^\infty$.

The sequence of inflation rates implied by this plan,
$\{\theta_t^A\}_{t=0}^\infty$, can be calculated using:

$$
\theta_t^A = \frac{1}{1+\alpha} \sum_{j=0}^{\infty} \left(\frac{\alpha}{1+\alpha}\right)^j \mu^A_{t+j}
$$

The value of $\{\theta_t^A,\mu_t^A \}_{t=0}^\infty$ at time $0$ is

$$
v^A_0 =  \sum_{t=0}^{T_A-1} \beta^t s(\theta_t^A,\mu_t^A) +\beta^{T_A} J(\theta^R_0)
$$

For an appropriate $T_A$, this plan can be verified to be self-enforcing and therefore credible.



From quantecon lecture  {doc}`calvo`, we'll again bring in the Python  class ChangLQ that constructs equilibria under timing protocols studied in that lecture. 

```{code-cell} ipython3
class ChangLQ:
    """
    Class to solve LQ Chang model
    """
    def __init__(self, β, c, α=1, u0=1, u1=0.5, u2=3, T=1000, θ_n=200):
        # Record parameters
        self.α, self.u0, self.u1, self.u2 = α, u0, u1, u2
        self.β, self.c, self.T, self.θ_n = β, c, T, θ_n

        self.setup_LQ_matrices()
        self.solve_LQ_problem()
        self.compute_policy_functions()
        self.simulate_ramsey_plan()
        self.compute_θ_range()
        self.compute_value_and_policy()

    def setup_LQ_matrices(self):
        # LQ Matrices
        self.R = -np.array([[self.u0, -self.u1 * self.α / 2],
                            [-self.u1 * self.α / 2, 
                             -self.u2 * self.α**2 / 2]])
        self.Q = -np.array([[-self.c / 2]])
        self.A = np.array([[1, 0], [0, (1 + self.α) / self.α]])
        self.B = np.array([[0], [-1 / self.α]])

    def solve_LQ_problem(self):
        # Solve LQ Problem (Subproblem 1)
        lq = LQ(self.Q, self.R, self.A, self.B, beta=self.β)
        self.P, self.F, self.d = lq.stationary_values()

        # Compute g0, g1, and g2 (41.16)
        self.g0, self.g1, self.g2 = [-self.P[0, 0], 
                                     -2 * self.P[1, 0], -self.P[1, 1]]
        
        # Compute b0 and b1 (41.17)
        [[self.b0, self.b1]] = self.F

        # Compute d0 and d1 (41.18)
        self.cl_mat = (self.A - self.B @ self.F)  # Closed loop matrix
        [[self.d0, self.d1]] = self.cl_mat[1:]

        # Solve Subproblem 2
        self.θ_R = -self.P[0, 1] / self.P[1, 1]
        
        # Find the bliss level of θ
        self.θ_B = -self.u1 / (self.u2 * self.α)

    def compute_policy_functions(self):
        # Solve the Markov Perfect Equilibrium
        self.μ_MPE = -self.u1 / ((1 + self.α) / self.α * self.c 
                                 + self.α / (1 + self.α)
                                 * self.u2 + self.α**2 
                                 / (1 + self.α) * self.u2)
        self.θ_MPE = self.μ_MPE
        self.μ_CR = -self.α * self.u1 / (self.u2 * self.α**2 + self.c)
        self.θ_CR = self.μ_CR

        # Calculate value under MPE and CR economy
        self.J_θ = lambda θ_array: - np.array([1, θ_array]) \
                                   @ self.P @ np.array([1, θ_array]).T
        self.V_θ = lambda θ: (self.u0 + self.u1 * (-self.α * θ)
                              - self.u2 / 2 * (-self.α * θ)**2 
                              - self.c / 2 * θ**2) / (1 - self.β)
        
        self.J_MPE = self.V_θ(self.μ_MPE)
        self.J_CR = self.V_θ(self.μ_CR)

    def simulate_ramsey_plan(self):
        # Simulate Ramsey plan for large number of periods
        θ_series = np.vstack((np.ones((1, self.T)), np.zeros((1, self.T))))
        μ_series = np.zeros(self.T)
        J_series = np.zeros(self.T)
        θ_series[1, 0] = self.θ_R
        [μ_series[0]] = -self.F.dot(θ_series[:, 0])
        J_series[0] = self.J_θ(θ_series[1, 0])

        for i in range(1, self.T):
            θ_series[:, i] = self.cl_mat @ θ_series[:, i-1]
            [μ_series[i]] = -self.F @ θ_series[:, i]
            J_series[i] = self.J_θ(θ_series[1, i])

        self.J_series = J_series
        self.μ_series = μ_series
        self.θ_series = θ_series

    def compute_θ_range(self):
        # Find the range of θ in Ramsey plan
        θ_LB = min(min(self.θ_series[1, :]), self.θ_B)
        θ_UB = max(max(self.θ_series[1, :]), self.θ_MPE)
        θ_range = θ_UB - θ_LB
        self.θ_LB = θ_LB - 0.05 * θ_range
        self.θ_UB = θ_UB + 0.05 * θ_range
        self.θ_range = θ_range

    def compute_value_and_policy(self):        
        # Create the θ_space
        self.θ_space = np.linspace(self.θ_LB, self.θ_UB, 200)
        
        # Find value function and policy functions over range of θ
        self.J_space = np.array([self.J_θ(θ) for θ in self.θ_space])
        self.μ_space = -self.F @ np.vstack((np.ones(200), self.θ_space))
        x_prime = self.cl_mat @ np.vstack((np.ones(200), self.θ_space))
        self.θ_prime = x_prime[1, :]
        self.CR_space = np.array([self.V_θ(θ) for θ in self.θ_space])
        
        self.μ_space = self.μ_space[0, :]
        
        # Calculate J_range, J_LB, and J_UB
        self.J_range = np.ptp(self.J_space)
        self.J_LB = np.min(self.J_space) - 0.05 * self.J_range
        self.J_UB = np.max(self.J_space) + 0.05 * self.J_range
```

Let's create an instance of ChangLQ with the following parameters:

```{code-cell} ipython3
clq = ChangLQ(β=0.85, c=2)
```

### Example of Self-Enforcing Plan

The following example implements an Abreu stick-and-carrot plan.

The government sets $\mu_t^A = 0.1$ for $t=0, 1, \ldots, 9$
and then starts the **Ramsey plan**.

We have computed outcomes for this plan.

For this plan, we plot the $\theta^A$, $\mu^A$ sequences as
well as the implied $v^A$ sequence.

Notice that because the government sets money supply growth high for 10
periods, inflation starts high.

Inflation gradually slowly declines  because people  expect the government to lower the money growth rate after period
$10$.

From the 10th period onwards, the inflation rate $\theta^A_t$
associated with this **Abreu plan** starts the Ramsey plan from its
beginning, i.e., $\theta^A_{t+10} =\theta^R_t \ \ \forall t \geq 0$.

```{code-cell} ipython3
:tags: [hide-input]

def abreu_plan(clq, T=1000, T_A=10, μ_bar=0.1, T_Plot=20):
    """
    Compute and plot the Abreu plan for the given ChangLQ instance.

    Here clq is an instance of ChangLQ.
    """
    # Append Ramsey μ series to stick μ series
    clq.μ_A = np.append(np.full(T_A, μ_bar), clq.μ_series[:-T_A])

    # Calculate implied stick θ series
    discount = (clq.α / (1 + clq.α)) ** np.arange(T)
    clq.θ_A = np.array([
        1 / (clq.α + 1) * np.sum(clq.μ_A[t:] * discount[:T - t])
        for t in range(T)
    ])

    # Calculate utility of stick plan
    U_A = clq.β ** np.arange(T) * (
        clq.u0 + clq.u1 * (-clq.θ_A) - clq.u2 / 2 
        * (-clq.θ_A) ** 2 - clq.c * clq.μ_A ** 2
    )

    clq.V_A = np.array([np.sum(U_A[t:] / clq.β ** t) for t in range(T)])

    # Make sure Abreu plan is self-enforcing
    clq.V_dev = np.array([
        (clq.u0 + clq.u1 * (-clq.θ_A[t]) - clq.u2 / 2 
         * (-clq.θ_A[t]) ** 2) + clq.β * clq.V_A[0]
        for t in range(T_Plot)
    ])

    # Plot the results
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    axes[2].plot(clq.V_dev[:T_Plot], label="$V^{A, D}_t$", c="orange")

    plots = [clq.θ_A, clq.μ_A, clq.V_A]
    labels = [r"$\theta_t^A$", r"$\mu_t^A$", r"$V^A_t$"]

    for plot, ax, label in zip(plots, axes, labels):
        ax.plot(plot[:T_Plot], label=label)
        ax.set(xlabel="$t$", ylabel=label)
        ax.legend()

    plt.tight_layout()
    plt.show()

abreu_plan(clq)
```

To confirm that the plan $\vec \mu^A$ is **self-enforcing**,  we
plot an object that we call $V_t^{A,D}$, defined in the key inequality in the second line of equation {eq}`eq_old10` above.

$V_t^{A,D}$ is the value at $t$ of deviating from the
self-enforcing plan $\vec \mu^A$ by setting $\mu_t = 0$ and
then restarting the plan at $v^A_0$ at $t+1$:

$$
v_t^{A,D} = s( \theta_j, 0) + \beta v_0^A
$$

In the above graph  $v_t^A > v_t^{A,D}$, which confirms that $\vec \mu^A$ is a self-enforcing plan.

We can also verify the inequalities required for $\vec \mu^A$ to
be self-confirming numerically as follows

```{code-cell} ipython3
np.all(clq.V_A[0:20] > clq.V_dev[0:20])
```

Given that plan $\vec \mu^A$ is self-enforcing, we can check that
the Ramsey plan $\vec \mu^R$ is credible by verifying that:

$$
v^R_t \geq s(\theta^R_t,0) + \beta v^A_0 , \quad \forall t \geq 0
$$

```{code-cell} ipython3
def check_ramsey(clq, T=1000):
    # Make sure Ramsey plan is sustainable
    R_dev = np.zeros(T)
    for t in range(T):
        R_dev[t] = (clq.u0 + clq.u1 * (-clq.θ_series[1, t])
                    - clq.u2 / 2 * (-clq.θ_series[1, t])**2) \
                    + clq.β * clq.V_A[0]

    return np.all(clq.J_series > R_dev)

check_ramsey(clq)
```

### Recursive Representation of a Sustainable Plan

We can represent a sustainable plan recursively by taking the
continuation value $v_t$ as a state variable.

We form the following 3-tuple of functions:

```{math}
:label: eq_old11

\begin{aligned}
\hat \mu_t & = \nu_\mu(v_t) \\
\theta_t & = \nu_\theta(v_t) \\
v_{t+1} & = \nu_v(v_t, \mu_t )
\end{aligned}
```

In addition to these equations, we need an initial value $v_0$ to
characterize a sustainable plan.

The first equation of {eq}`eq_old11` tells the recommended value of
$\hat \mu_t$ as a function of the promised value $v_t$.

The second equation of {eq}`eq_old11`  tells the inflation rate as a function of
$v_t$.

The third equation of {eq}`eq_old11`  updates the continuation value in a way that
depends on whether the government at $t$ confirms the representative agent's
expectations by setting $\mu_t$ equal to the recommended value
$\hat \mu_t$, or whether it disappoints those expectations.

## Whose  Plan is It?

A credible government plan $\vec \mu$ plays multiple roles.

* It is a sequence of actions chosen by the government.
* It is a sequence of the representative agent's forecasts of government actions.

Thus, $\vec \mu$ is both a government policy and a collection of the representative agent's forecasts of  government policy.

Does the government *choose*  policy actions or does it simply *confirm* prior private sector forecasts of those actions?

An argument in favor of the *government chooses* interpretation comes from noting that the theory of credible plans builds in a theory that the government each period chooses
the action that it wants.

An argument in favor of the *simply confirm* interpretation is gathered from staring at the key inequality {eq}`eq_old100a` that defines a credible policy.



We have also computed **credible plans** for a government or sequence
of governments that choose sequentially.

These include

- a **self-enforcing** plan that gives a low initial value $v_0$.
- a better plan -- possibly one that attains values associated with
  Ramsey plan -- that is not self-enforcing.


