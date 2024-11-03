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

(calvo)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Time Inconsistency of Ramsey Plans

```{index} single: Models; Additive functionals
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade quantecon
```

## Overview

This lecture describes several  linear-quadratic versions of a model that Guillermo Calvo {cite}`Calvo1978` used to illustrate the **time inconsistency** of optimal government
plans.

Like Chang {cite}`chang1998credible`, we use these models as  laboratories in which to explore consequences of  timing protocols for government decision making.

The models focus attention on intertemporal tradeoffs between

- welfare benefits that anticipations of future  deflation generate  by decreasing  costs of holding real money balances and thereby increasing a representative agent's *liquidity*, as measured by his or her holdings of real money balances, and
- costs associated with the  distorting taxes that the government must levy in order to acquire the paper money that it will  destroy  in order to generate anticipated deflation

The models feature

- rational expectations
- several explicit timing protocols
- costly government actions at all dates $t \geq 1$ that increase household utilities at dates before $t$
- sets of Bellman equations, one set for each timing protocol
  
   - for example, in a timing protocol used to pose a **Ramsey plan**, a government chooses an infinite sequence of money supply growth rates once and for all at time $0$.
   
   - in this timing protocol, there are  two value functions and associated Bellman equations, one that expresses a representative  private expectation of future inflation as a function of current and future government actions, another that describes  the value function of a  Ramsey planner

   - in other timing protocols, other Bellman equations and associated  value functions will appear

A theme of this lecture is that  timing protocols affect  outcomes.

We'll use ideas from  papers by Cagan {cite}`Cagan`, Calvo {cite}`Calvo1978`, and  Chang {cite}`chang1998credible` as
well as from chapter 19 of {cite}`Ljungqvist2012`.

In addition, we'll use ideas from linear-quadratic dynamic programming
described in  [Linear Quadratic Control](https://python-intro.quantecon.org/lqcontrol.html) as applied to Ramsey problems in {doc}`Stackelberg problems <dyn_stack>`.

We  specify model fundamentals  in  ways that allow us to use
linear-quadratic discounted dynamic programming to compute an optimal government
plan under each of our timing protocols. 

A sister lecture {doc}`calvo_machine_learn` studies some of the same models but does not use dynamic programming.

Instead it uses a  **machine learning** approach that does not explicitly recognize the recursive structure structure of the Ramsey problem that Chang {cite}`chang1998credible` saw and that we exploit in this lecture.

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
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from IPython.display import display, Math
```

## Model Components

There is no uncertainty.

Let:

- $p_t$ be the log of the price level
- $m_t$ be the log of nominal money balances
- $\theta_t = p_{t+1} - p_t$ be the net rate of inflation between $t$ and $t+1$
- $\mu_t = m_{t+1} - m_t$ be the net rate of growth of nominal balances

The demand for real balances is governed by a perfect foresight
version of a Cagan {cite}`Cagan` demand function for real balances:

```{math}
:label: eq_old1

m_t - p_t = -\alpha(p_{t+1} - p_t) \: , \: \alpha > 0
```

for $t \geq 0$.

Equation {eq}`eq_old1` asserts that the demand for real balances is inversely
related to the public's expected rate of inflation, which  equals
the actual rate of inflation because there is no uncertainty here.

(When there is no uncertainty, an assumption of **rational expectations** that becomes equivalent to  **perfect foresight**).

(See {cite}`Sargent77hyper` for a rational expectations version of the model when there is uncertainty.)

Subtracting the demand function {eq}`eq_old1` at time $t$ from the demand
function at $t+1$ gives:

$$
\mu_t - \theta_t = -\alpha \theta_{t+1} + \alpha \theta_t
$$

or

```{math}
:label: eq_old2

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
the linear difference equation {eq}`eq_old2` can be solved forward to get:

```{math}
:label: eq_old3

\theta_t = \frac{1}{1+\alpha} \sum_{j=0}^\infty \left(\frac{\alpha}{1+\alpha}\right)^j \mu_{t+j}
```

**Insight:** In the spirit of Chang {cite}`chang1998credible`,  equations {eq}`eq_old1` and {eq}`eq_old3` show that $\theta_t$ intermediates
how choices of $\mu_{t+j}, \ j=0, 1, \ldots$ impinge on time $t$
real balances $m_t - p_t = -\alpha \theta_t$.

An equivalence class of continuation money growth sequences $\{\mu_{t+j}\}_{j=0}^\infty$ deliver the same $\theta_t$.

We shall use this insight to help us simplify our analysis of alternative  government policy problems.

That future rates of money creation influence earlier rates of inflation
makes  timing protocols matter for modeling optimal government policies.

When $\vec \theta = \{\theta_t\}_{t=0}^\infty$ is square summable, we can  represent restriction {eq}`eq_old3`  as

$$
\begin{bmatrix}
  1 \\
  \theta_{t+1}
\end{bmatrix} =
\begin{bmatrix}
  1 & 0 \\
  0 & \frac{1+\alpha}{\alpha}
\end{bmatrix}
\begin{bmatrix}
  1 \\
  \theta_{t}
\end{bmatrix}  +
\begin{bmatrix}
  0 \\
  -\frac{1}{\alpha}
\end{bmatrix}
\mu_t
$$ (eq_old4a)

or

```{math}
:label: eq_old4

x_{t+1} = A x_t + B \mu_t
```

Even though $\theta_0$ is to be determined by our  model and so is not an initial condition,
as it ordinarily would be in the state-space model described in our lecture on  [Linear Quadratic Control](https://python-intro.quantecon.org/lqcontrol.html), we nevertheless write the model in the state-space form {eq}`eq_old4`.

 

We use  form {eq}`eq_old4` because we want to apply an approach described in  our lecture on {doc}`Stackelberg problems <dyn_stack>`.

Notice that $\frac{1+\alpha}{\alpha} > 1$ is an eigenvalue of transition matrix $A$ that threatens to destabilize the state-space system. 

The Ramsey planner will design   a decision rule for $\mu_t$ that  stabilizes  the system. 

The  government  values  a representative household's utility of real balances at time $t$ according to the utility function

```{math}
:label: eq_old5

U(m_t - p_t) = u_0 + u_1 (m_t - p_t) - \frac{u_2}{2} (m_t - p_t)^2, \quad u_0 > 0, u_1 > 0, u_2 > 0
```

The money demand function {eq}`eq_old1` and the utility function {eq}`eq_old5` imply that 

$$
U(-\alpha \theta_t) = u_0 + u_1 (-\alpha \theta_t) -\frac{u_2}{2}(-\alpha \theta_t)^2 . 
$$ (eq_old5a)

The ``bliss level`` of real balances is  $\frac{u_1}{u_2}$ and the inflation rate that attains
it is $-\frac{u_1}{u_2 \alpha}$.

(TO TOM: the first sentece in the next section is very similar to the sentence above.)

## Friedman's Optimal Rate of Deflation

According to {eq}`eq_old5a`, the "bliss level" of real balances is  $\frac{u_1}{u_2}$ and the inflation rate that attains it is


$$
\theta_t = \theta^* = -\frac{u_1}{u_2 \alpha}
$$ (eq:Friedmantheta)

Milton Friedman recommended that  the government  withdraw and destroy money at a rate 
that implies an inflation rate given by  {eq}`eq:Friedmantheta`.

In our setting, that could be accomplished by setting 


$$
\mu_t = \mu^* = \theta^* , t \geq 0
$$ (eq:Friedmanrule)

where $\theta^*$ is given by equation {eq}`eq:Friedmantheta`.

To deduce this recommendation, Milton Friedman assumed that the taxes that government must impose in order to acquire money at rate $\mu_t$ do not distort economic decisions.

  - for example, perhaps the government can impose lump sum taxes that distort no decisions by private agents

## Calvo's Distortion 

The starting point of Calvo {cite}`Calvo1978` and  Chang {cite}`chang1998credible`
is that such lump sum taxes are not available.

Instead, the government acquires money by levying taxes that distort decisions and thereby impose costs on the representative consumer.

In the models of  Calvo {cite}`Calvo1978` and  Chang {cite}`chang1998credible`, the government takes those costs tax-distortion costs into account.

It balances the costs of imposing the distorting taxes needed to acquire the money that it destroys in order to generate deflation against the benefits that expected deflation generates by raising the representative households' holdings of real balances.  

Let's see how the government does that in our version of the models of  Calvo {cite}`Calvo1978` and  Chang {cite}`chang1998credible`. 


Via equation {eq}`eq_old3`, a government plan
$\vec \mu = \{\mu_t \}_{t=0}^\infty$ leads to a
sequence of inflation outcomes
$\vec \theta = \{ \theta_t \}_{t=0}^\infty$.

We assume that the government incurs  social costs $\frac{c}{2} \mu_t^2$ at
$t$ when it  changes the stock of nominal money
balances at rate $\mu_t$.

Therefore, the one-period welfare function of a benevolent government
is:

```{math}
:label: eq_old6

-s(\theta_t, \mu_t) \equiv - r(x_t,\mu_t) = \begin{bmatrix} 1 \\ \theta_t \end{bmatrix}' \begin{bmatrix} u_0 & -\frac{u_1 \alpha}{2} \\ -\frac{u_1 \alpha}{2} & -\frac{u_2 \alpha^2}{2} \end{bmatrix} \begin{bmatrix} 1 \\ \theta_t \end{bmatrix} - \frac{c}{2} \mu_t^2 =  - x_t'Rx_t - Q \mu_t^2
```

The  government's time $0$ value is 

```{math}
:label: eq_old7

v_0 = - \sum_{t=0}^\infty \beta^t r(x_t,\mu_t) = - \sum_{t=0}^\infty \beta^t s(\theta_t,\mu_t)
```

where $\beta \in (0,1)$ is a discount factor. 

The government's time $t$ continuation value $v_t$ is 

$$
v_t = - \sum_{j=0}^\infty \beta^j s(\theta_{t+j}, \mu_{t+j}) .
$$

We can represent  dependence of  $v_0$ on $(\vec \theta, \vec \mu)$ recursively via the  difference equation

```{math}
:label: eq_old8

v_t = - s(\theta_t, \mu_t) + \beta v_{t+1}
```

It is useful to evaluate {eq}`eq_old8` under a time-invariant money growth rate $\mu_t = \bar \mu$
that according to equation {eq}`eq_old3` would bring forth a constant inflation rate equal to $\bar \mu$.  

Under that policy,

$$
v_t = V(\bar \mu) = - \frac{s(\bar \mu, \bar \mu)}{1-\beta} 
$$ (eq:barvdef)

for all $t \geq 0$.

Values of $V(\bar \mu)$ computed according to formula {eq}`eq:barvdef` for three different  values of $\bar \mu$ will play important roles below.

* $V(\mu^{MP})$ is the value of attained by the government in a **Markov perfect equilibrium** 
* $V(\mu^{CR})$ is the value of attained by the government in a **constrained to constant $\mu$ equilibrium**
* $V(\mu^R_\infty)$ is the limiting value attained by a continuation Ramsey planner under a Ramsey plan.
    * We shall see that $V(\mu^R_\infty)$ is a worst continuation value attained along a Ramsey plan  

## Structure

The following structure is induced by a representative  agent's
behavior as summarized by the demand function for money {eq}`eq_old1` that leads to equation {eq}`eq_old3`, which  tells how future settings of $\mu$ affect the current value of $\theta$.

Equation {eq}`eq_old3` maps a **policy** sequence of money growth rates
$\vec \mu =\{\mu_t\}_{t=0}^\infty \in L^2$  into an inflation sequence
$\vec \theta = \{\theta_t\}_{t=0}^\infty \in L^2$.

These, in turn, induce a discounted value to a government sequence
$\vec v = \{v_t\}_{t=0}^\infty \in L^2$ that satisfies the
recursion

$$
v_t = - s(\theta_t,\mu_t) + \beta v_{t+1}
$$ (eq_new100)

where we have called $s(\theta_t, \mu_t) = r(x_t, \mu_t)$, as
in {eq}`eq_old7`.

Thus,  a triple of sequences
$(\vec \mu, \vec \theta, \vec v)$ depends on  a
sequence $\vec \mu \in L^2$.

At this point $\vec \mu \in L^2$ is an arbitrary exogenous policy.

A theory of government
decisions will  make $\vec \mu$ endogenous, i.e., a theoretical *output* instead of an *input*.


### Intertemporal Aspects 

Criterion function {eq}`eq_old7` and the constraint system {eq}`eq_old4` exhibit the following
structure:

- Setting the money growth rate $\mu_t \neq 0$ imposes costs
  $\frac{c}{2} \mu_t^2$ at time $t$ and at no other times;
  but
- The money growth rate $\mu_t$ affects the government's  one-period utilities at all dates
  $s = 0, 1, \ldots, t$.


This structure  sets the stage for the emergence of a time-inconsistent
optimal government plan  under a **Ramsey**   timing protocol
 
  * it is  also called a **Stackelberg** timing protocol.

We'll  study outcomes under a Ramsey timing protocol.

We'll also study outcomes under other timing protocols.

## Three Timing Protocols

We consider three  models of government policy making that  differ in

- *what* a  policymaker chooses, either a sequence
  $\vec \mu$ or just   $\mu_t$ in a single period $t$.
- *when* a  policymaker chooses, either once and for all at time $0$, or at some time or times  $t \geq 0$.
- what a policymaker *assumes* about how its choice of $\mu_t$
  affects the representative  agent's expectations about earlier and later
  inflation rates.

In two of our models, a single policymaker  chooses a sequence
$\{\mu_t\}_{t=0}^\infty$ once and for all, knowing  how
$\mu_t$ affects household one-period utilities at dates $s = 0, 1, \ldots, t-1$

- these two models  thus employ a  **Ramsey** or **Stackelberg** timing protocol.

In a third  model, there is a sequence of policymakers, each of whom
sets $\mu_t$ at one $t$ only.

- a time $t$  policymaker cares only about $v_t$ and  ignores  effects that its choice of $\mu_t$ has on $v_s$ at  dates $s = 0, 1, \ldots, t-1$.

The three models differ with respect to timing protocols, constraints on
government choices, and government policymakers' beliefs about how their
decisions affect the representative agent's beliefs about future government
decisions.

The models are distinguished by their having  either

- A single Ramsey planner that chooses a sequence
  $\{\mu_t\}_{t=0}^\infty$ once and for all at time $0$; or
- A single Ramsey planner that  chooses a sequence
  $\{\mu_t\}_{t=0}^\infty$ once and for all at time $0$
  subject to the constraint that $\mu_t = \mu$ for all
  $t \geq 0$; or
- A sequence indexed by $t =0, 1, 2, \ldots$ of separate policymakers 
    - a time $t$ policymaker chooses $\mu_t$ only and forecasts that future government decisions are unaffected by its choice.


The first model describes a **Ramsey plan** chosen by a **Ramsey planner**

The second model describes a **Ramsey plan** chosen by a *Ramsey planner constrained to choose a time-invariant $\mu_t$*

The third model describes a **Markov perfect equilibrium**


```{note}
 In the  quantecon lecture {doc}`calvo_abreu`, we'll study outcomes under another timing protocol in where there is a sequence of separate policymakers and  a time $t$ policymaker chooses  only $\mu_t$ but believes that its choice of $\mu_t$  shapes the representative agent's beliefs about  future rates of money creation and inflation, and through them, future government actions.
 This is a model of  a **credible government policy** also known as a **sustainable plan**.
The relationship between  outcomes in  the first (Ramsey) timing protocol and the {doc}`calvo_abreu` timing protocol and belief structure is the subject of a literature on **sustainable** or **credible** public policies (Chari and Kehoe {cite}`chari1990sustainable`
{cite}`stokey1989reputation`, and Stokey {cite}`Stokey1991`). 
```


## Note on Dynamic Programming Squared

We'll begin with the timing protocol associated with a Ramsey plan and deploy 
an application of what we  nickname **dynamic programming squared**.

The nickname refers to the feature that a value satisfying one Bellman equation appears as an argument in a second Bellman equation.

Thus, our models have involved two Bellman equations:

- equation {eq}`eq_old1` expresses how $\theta_t$ depends on $\mu_t$
  and $\theta_{t+1}$
- equation {eq}`eq_old4` expresses how value $v_t$ depends on
  $(\mu_t, \theta_t)$ and $v_{t+1}$

A value $\theta$ from one Bellman equation appears as an argument of a second Bellman equation for another value $v$.

## A Ramsey Planner

Here  we consider a Ramsey planner that  chooses
$\{\mu_t, \theta_t\}_{t=0}^\infty$ to maximize {eq}`eq_old7`
subject to the law of motion {eq}`eq_old4`.

We can split this problem into two stages, as in {doc}`Stackelberg problems <dyn_stack>` and  {cite}`Ljungqvist2012` Chapter 19.

In the first stage, we take the initial inflation rate $\theta_0$ as given
and solve what looks like an ordinary  LQ discounted dynamic programming problem.

In the second stage, we choose an optimal  initial inflation rate $\theta_0$.

Define a feasible set of
$(\overrightarrow x_1, \overrightarrow \mu_0)$ sequences, both of which must belong to $L^2$:

$$
\Omega(x_0) = \left \lbrace ( \overrightarrow x_1, \overrightarrow \mu_0) : x_{t+1}
= A x_t + B \mu_t \: , \: \forall t \geq 0; (\vec x_1, \vec \mu_0) \in L^2 \times L^2 \right \rbrace
$$

### Subproblem 1

The value function

$$
J(x_0) = \max_{(\overrightarrow x_1, \overrightarrow \mu_0) \in \Omega(x_0)}
- \sum_{t=0}^\infty \beta^t r(x_t,\mu_t)
$$ (eq:subprob1LQ)

satisfies the Bellman equation

$$
J(x) = \max_{\mu,x'}\{-r(x,\mu) + \beta J(x')\}
$$

subject to:

$$
x' = Ax + B\mu
$$

As in {doc}`Stackelberg problems <dyn_stack>`, we can map this problem into a linear-quadratic control problem and deduce an optimal value function $J(x)$.

Guessing that $J(x) = - x'Px$ and substituting into the Bellman
equation gives rise to the algebraic matrix Riccati equation:

$$
P = R + \beta A'PA - \beta^2 A'PB(Q + \beta B'PB)^{-1} B'PA
$$

and an optimal decision rule

$$
\mu_t = - F x_t
$$

where

$$
F = \beta (Q + \beta B'PB)^{-1} B'PA
$$ (eq:formulaF)

The QuantEcon [LQ](https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/lqcontrol.py) class solves for $F$ and $P$ given inputs
$Q, R, A, B$, and $\beta$.


The value function for a (continuation) Ramsey planner is

$$ v_t = - \begin{bmatrix} 1 & \theta_t \end{bmatrix} \begin{bmatrix} P_{11} & P_{12} \cr P_{21} & P_{22} \end{bmatrix} \begin{bmatrix} 1 \cr \theta_t \end{bmatrix}
$$

or

$$
v_t = - P_{11} - 2 P_{21}\theta_t - P_{22}\theta_t^2
$$

or

$$ 
v_t = g_0 + g_1 \theta_t + g_2 \theta_t^2
$$ (eq:continuationvfn)

where

$$
g_0 = - P_{11}, \quad g_1 = - 2 P_{21}, \quad g_2 =  - P_{22}
$$


The Ramsey plan for setting $\mu_t$ is

$$ 
\mu_t = - \begin{bmatrix} F_1 & F_2 \end{bmatrix} \begin{bmatrix} 1 \cr \theta_t \end{bmatrix}
$$

or 

$$ 
\mu_t = b_0 + b_1 \theta_t 
$$ (eq:muRamseyrule)

where $b_0 = -F_1, b_1 = - F_2$ and  $F$ satisfies equation {eq}`eq:formulaF`,

The Ramsey planner's  decision rule for updating $\theta_{t+1}$ is

$$
\theta_{t+1} = d_0 + d_1 \theta_t
$$ (eq:thetaRamseyrule)

where $\begin{bmatrix} d_0 & d_1 \end{bmatrix}$ is the second row of 
the closed-loop matrix $A - BF$ for computed in subproblem 1 above.

The linear quadratic control problem {eq}`eq:subprob1LQ`  satisfies regularity conditions that
guarantee that $A - BF $ is a stable matrix (i.e., its maximum eigenvalue is strictly less than
$1$ in absolute value).

Consequently, we are assured that

$$ 
| d_1 | < 1 ,
$$ (eq:stabilityd1)

a stability condition that will play an important role.

It remains for us to describe how the Ramsey planner sets $\theta_0$.  

Subproblem 2 does that.


### Subproblem 2

The value of the Ramsey problem is

$$
V^R = \max_{\theta} J(\theta)
$$

where $V^R$ is the maximum value of $v_0$ defined in equation {eq}`eq_old7`.

We have taken the liberty of abusing notation slightly by writing $J(x)$ as $J(\theta)$

  * notice that $x = \begin{bmatrix} 1 \cr \theta \end{bmatrix}$, so $\theta$ is the only component of $x$ that can possibly vary

Value function $J(\theta_0)$ satisfies

$$
 J(\theta_0) = -\begin{bmatrix} 1 & \theta_0 \end{bmatrix} \begin{bmatrix} P_{11} & P_{12} \\ P_{21} & P_{22} \end{bmatrix} \begin{bmatrix} 1 \\ \theta_0 \end{bmatrix} = -P_{11} - 2 P_{21} \theta_0 - P_{22} \theta_0^2
$$

Maximizing $J(\theta_0)$  with respect to $\theta_0$ yields the FOC:

$$
- 2 P_{21} - 2 P_{22} \theta_0 =0
$$

which implies

$$
\theta_0 = \theta_0^R = - \frac{P_{21}}{P_{22}}
$$

## Representation of Ramsey Plan

The preceding calculations indicate that we can represent a Ramsey plan
$\vec \mu$ recursively with the following system created in the spirit of Chang {cite}`chang1998credible`:

```{math}
:label: eq_old9

\begin{aligned}
\theta_0 & = \theta_0^R \\
\mu_t &  = b_0 + b_1 \theta_t \\
v_t & = g_0 +g_1\theta_t + g_2 \theta_t^2 \\
\theta_{t+1} & = d_0 + d_1 \theta_t , \quad  d_0 >0, d_1 \in (0,1) \\
\end{aligned}
```

where $b_0, b_1, g_0, g_1, g_2$ are positive parameters that we shall compute with Python code below.

From condition {eq}`eq:stabilityd1`, we know that $|d_1| < 1$. 

To interpret system {eq}`eq_old9`, think of the  sequence
$\{\theta_t\}_{t=0}^\infty$ as a sequence of
synthetic **promised inflation rates**.

For some purposes, we can think of these promised inflation rates  just as computational devices for
generating a sequence $\vec\mu$ of money growth rates that when  substituted into equation {eq}`eq_old3` generate  *actual* rates of inflation.

It can be verified that if we substitute a plan
$\vec \mu = \{\mu_t\}_{t=0}^\infty$ that satisfies these equations
into equation {eq}`eq_old3`, we obtain the same sequence $\vec \theta$
generated by the system {eq}`eq_old9`.

(Here an application of the Big $K$, little $k$ trick is again at work.)

Thus,  within the  Ramsey plan,  **promised
inflation** equals **actual inflation**.


System {eq}`eq_old9` implies that under the Ramsey plan

$$
 \theta_t = d_0 \left(\frac{1 - d_1^t}{1 - d_1} \right)  + d_1^t \theta_0^R ,
$$ (eq:thetatimeinconsist)

Because $d_1 \in (0,1)$, it follows  from {eq}`eq:thetatimeinconsist` that  as $t \to \infty$ $\theta_t^R $ converges to

$$
\lim_{t \rightarrow +\infty} \theta_t^R =  \theta_\infty^R = \frac{d_0}{1 - d_1}.  
$$ (eq:thetaasymptotic)

Furthermore, we shall see that  $\theta_t^R$ converges to $\theta_\infty^R$ from above.   

Meanwhile,  $\mu_t$ varies over time according to 

$$
 \mu_t = b_0 + b_1 d_0 \left(\frac{1 - d_1^t}{1 - d_1} \right)  + b_1 d_1^t \theta_0^R.
$$ (eq:mutimeinconsist)

 
Variation of  $ \vec \mu^R, \vec \theta^R, \vec v^R $ over time  are  symptoms of time inconsistency.

- The Ramsey planner reaps immediate benefits from promising lower
  inflation later to be achieved by costly distorting taxes.
- These benefits are intermediated by reductions in expected inflation
  that precede the  reductions in money creation rates that rationalize them, as indicated by
  equation {eq}`eq_old3`.



## Multiple roles of $\theta_t$

The inflation rate $\theta_t$ plays three roles simultaneously:

- In equation {eq}`eq_old3`, $\theta_t$ is the actual rate of inflation
  between $t$ and $t+1$.
- In equation  {eq}`eq_old2` and {eq}`eq_old3`, $\theta_t$ is also the public's
  expected rate of inflation between $t$ and $t+1$.
- In system {eq}`eq_old9`, $\theta_t$ is a promised rate of inflation
  chosen by the Ramsey planner at time $0$.

That the same variable $\theta_t$ takes on these multiple roles brings insights about 
  commitment and forward guidance, about whether the government  follows  or   leads the market, and
about dynamic or time inconsistency.

## Time inconsistency

As discussed in {doc}`Stackelberg problems <dyn_stack>` and {doc}`Optimal taxation with state-contingent debt <opt_tax_recur>`, a continuation Ramsey plan is not a Ramsey plan.

This is a concise way of characterizing the time inconsistency of a Ramsey plan.

The time inconsistency of a Ramsey plan has motivated other models of government decision making
that, relative to a Ramsey plan,  alter either

- the timing protocol and/or
- assumptions about how government decision makers think their decisions affect the representative agent's beliefs about future government decisions





## Constrained-to-Constant-Growth-Rate Ramsey Plan

We now describe a  model in which we restrict the Ramsey planner's choice set.

Instead of choosing a sequence of money growth rates $\vec \mu \in {\bf L}^2$, we restrict the 
government to choose a time-invariant money growth rate $\bar \mu$. 

We created this version of the model  to highlight an aspect of a Ramsey plan associated with its time inconsistency, namely, the feature that optimal settings of the  policy instrument vary over time.

Thus, instead of allowing the government at time $0$ to choose a different $\mu_t$ for each $t \geq 0$, we now assume that a  government at time $0$ once and for all  chooses a *constant* sequence $\mu_t = \bar \mu$ for all $t \geq 0$.

We assume that the government knows the perfect foresight outcome implied by equation {eq}`eq_old2` that $\theta_t = \bar  \mu$ when $\mu_t = \bar \mu$ for all $t \geq 0$.

The government chooses $\bar \mu$  to maximize


$$
V^{CR}(\bar \mu) = V(\bar \mu)
$$

where $V(\bar \mu)$ is defined in equation {eq}`eq:barvdef`.

We can express $V^{CR}(\bar \mu)$ as


$$
V^{CR} (\bar \mu) = (1-\beta)^{-1} \left[ U (-\alpha \bar \mu) - \frac{c}{2} (\bar \mu)^2 \right]
$$ (eq:vcrformula20)

With the quadratic form {eq}`eq_old5` for the utility function $U$, the
maximizing $\bar \mu$ is

$$
\mu^{CR} = - \frac{\alpha u_1}{\alpha^2 u_2 + c }
$$ (eq:muRamseyconstrained)

The optimal value attained by a *constrained to constant $\mu$* Ramsey planner is

$$
V^{CR}(\mu^{CR}) = v^{CR} = (1-\beta)^{-1} \left[ U (-\alpha \mu^{CR}) - \frac{c}{2} (\mu^{CR})^2 \right]
$$ (eq:vcrformula)


**Remark:** We have  introduced the constrained-to-constant $\mu$
government in order eventually to highlight the   time-variation of
$\mu_t$   that is a telltale sign of a Ramsey plan's  **time inconsistency**.

## Markov Perfect Governments

We now describe yet another timing protocol.

In this one, there is a sequence of government policymakers.

A time $t$ government chooses $\mu_t$ and expects all future governments to set
$\mu_{t+j} = \bar \mu$.

This assumption mirrors an assumption made   in this QuantEcon lecture:  [Markov Perfect Equilibrium](https://python-intro.quantecon.org/markov_perf.html).

When it sets $\mu_t$, the  government   at $t$ believes that $\bar \mu$ is
unaffected by its choice of $\mu_t$.

According to equation {eq}`eq_old3`,  the time $t$ rate of inflation is then 

$$
\theta_t =  \frac{1}{1+\alpha} \mu_t + \frac{\alpha}{1+\alpha} \bar \mu, 
$$ (eq_Markov2)

which expresses inflation $\theta_t$ as a geometric weighted average of the money growth today
$\mu_t$ and money growth from tomorrow onward $\bar \mu$.

Given $\bar \mu$, the time $t$ government  chooses $\mu_t$ to
maximize:

$$
Q(\mu_t, \bar \mu) = U(-\alpha \theta_t) - \frac{c}{2} \mu_t^2 + \beta V(\bar \mu)
$$ (eq_Markov3)

where $V(\bar \mu)$ is given by formula  {eq}`eq:barvdef` for  the time $0$ value $v_0$ of
recursion {eq}`eq_old8` under a money supply growth rate that is forever constant
at $\bar \mu$. 

Substituting  {eq}`eq_Markov2` into {eq}`eq_Markov3` and expanding gives:

$$ 
\begin{aligned}
Q(\mu_t, \bar \mu) & = u_0 + u_1\left(-\frac{\alpha^2}{1+\alpha} \bar \mu - \frac{\alpha}{1+\alpha} \mu_t\right) - \frac{u_2}{2}\left(-\frac{\alpha^2}{1+\alpha} \bar \mu - \frac{\alpha}{1+\alpha} \mu_t\right)^2   \\ 
& \quad \quad \quad - \frac{c}{2} \mu_t^2 + \beta V(\bar \mu)
\end{aligned}
$$ (eq:Vmutemp)

The first-order necessary condition for maximing $Q(\mu_t, \bar \mu)$ with respect to $\mu_t$ is:

$$
- \frac{\alpha}{1+\alpha} u_1 - u_2(-\frac{\alpha^2}{1+\alpha} \bar \mu - \frac{\alpha}{1+\alpha} \mu_t)(- \frac{\alpha}{1+\alpha}) - c \mu_t = 0
$$

Rearranging we get the time $t$ government's best response map

$$
\mu_t = f(\bar \mu)
$$

where

$$
f(\bar \mu)= \frac{- u_1}{\frac{1+\alpha}{\alpha}c + \frac{\alpha}{1+\alpha}u_2} - \frac{\alpha^2 u_2}{\left[ \frac{1+\alpha}{\alpha}c + \frac{\alpha}{1+\alpha} u_2 \right] (1+\alpha)}\bar \mu
$$

A **Markov Perfect Equilibrium** (MPE) outcome $ \mu^{MPE}$ is a fixed point of the best response map:

$$
\mu^{MPE} = f(\mu^{MPE})
$$

Calculating $\mu^{MPE}$, we find

$$
\mu^{MPE} = \frac{-u_1}{\frac{1+\alpha}{\alpha}c + \frac{\alpha}{1+\alpha} u_2 + \frac{\alpha^2}{1+\alpha} u_2}
$$

This can be simplified to

$$
\mu^{MPE}  = - \frac{\alpha u_1}{\alpha^2 u_2 + (1+\alpha)c} .
$$ (eq:Markovperfectmu)

The value of a Markov perfect equilibrium is 

$$
V^{MPE} = -\frac{s(\mu^{MPE}, \mu^{MPE})}{1-\beta}
$$ (eq:VMPE)

or 

$$
V^{MPE} = V(\mu^{MPE})
$$

where $V(\cdot)$ is given by formula {eq}`eq:barvdef`.


Under the  Markov perfect timing protocol 

 * a government takes $\bar \mu$ as given when it chooses $\mu_t$
 * we equate $\mu_t = \mu$ only *after* we have computed a time $t$ government's first-order condition for $\mu_t$.

(compute_lq)=
## Outcomes under Three Timing Protocols

We  want to compare outcome sequences  $\{ \theta_t,\mu_t \}$ under three timing protocols associated with 

  * a standard Ramsey plan with its time varying $\{ \theta_t,\mu_t \}$ sequences 
  * a Markov perfect equilibrium 
  * our nonstandard  Ramsey plan in which the planner is restricted to choose a time-invariant  $\mu_t = \mu$ for all $t \geq 0$.

We have computed closed form formulas for several of these outcomes, which we find it convenient to repeat here.

In particular, the constrained to constant inflation Ramsey inflation outcome is $\mu^{CR}$,
which according to equation {eq}`eq:muRamseyconstrained` is

$$
\theta^{CR} = - \frac{\alpha u_1}{\alpha^2 u_2 + c }
$$ 

Equation {eq}`eq:Markovperfectmu` implies that the Markov perfect constant inflation rate is 

$$
\theta^{MPE}  = - \frac{\alpha u_1}{\alpha^2 u_2 + (1+\alpha)c}
$$ 

According to equation {eq}`eq:Friedmantheta`, the bliss level of inflation that we associated with a Friedman rule is

$$
 \theta^* = -\frac{u_1}{u_2 \alpha}
$$ 

**Proposition 1:** When $c=0$,  $\theta^{MPE} = \theta^{CR} = \theta^*$ and 
$\theta_0^R = \theta_\infty^R$. 

The first two equalities follow from the preceding three equations. 

We'll illustrate  the third equality that equates $\theta_0^R$ to $ \theta_\infty^R$ with some quantitative examples below.

Proposition 1 draws attention to how   a positive tax distortion parameter $c$ alters  the  optimal rate of deflation that Milton Friedman financed  by imposing a lump sum tax.  

We'll compute 

 *   $(\vec \theta^R, \vec \mu^R)$:  ordinary time-varying Ramsey sequences
 *   $(\theta^{MPE} = \mu^{MPE})$:  Markov perfect equilibrium (MPE) fixed values
 *   $(\theta^{CR} = \mu^{CR})$:  fixed values associated with a constrained to time-invariant $\mu$ Ramsey plan
 *   $\theta^*$:   bliss level of inflation prescribed by a Friedman rule

We will create a class ChangLQ that solves the models and stores their values

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

The following code  plots value functions for a continuation Ramsey
planner.

```{code-cell} ipython3
:tags: [hide-input]

def compute_θs(clq):
    """
    Method to compute θ and assign corresponding labels and colors 

    Here clq is an instance of ChangLQ
    """
    θ_points = [clq.θ_B, clq.θ_series[1, -1], 
                clq.θ_CR, clq.θ_space[np.argmax(clq.J_space)], 
                clq.θ_MPE]
    labels = [r"$\theta^*$", r"$\theta_\infty^R$", 
              r"$\theta^{CR}$", r"$\theta_0^R$", 
              r"$\theta^{MPE}$"]
    θ_colors = ['r', 'C5', 'g', 'C0', 'orange']

    return θ_points, labels, θ_colors

def plot_policy_functions(clq):
    """
    Method to plot the policy functions over the relevant range of θ

    Here clq is an instance of ChangLQ
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    θ_points, labels, θ_colors = compute_θs(clq)

    # Plot θ' function
    ax.plot(clq.θ_space, clq.θ_prime, 
            label=r"$\theta'(\theta)$", 
            lw=2, alpha=0.5, color='blue')
    ax.plot(clq.θ_space, clq.θ_space, 'k--', lw=2, alpha=0.7)  # Identity line
    
    # Plot μ function
    ax.plot(clq.θ_space, clq.μ_space, lw=2, 
            label=r"$\mu(\theta)$", 
            color='green', alpha=0.5)

    # Plot labels and points for μ function
    μ_min, μ_max = min(clq.μ_space), max(clq.μ_space)
    μ_range = μ_max - μ_min
    offset = 0.02 * μ_range
    for θ, label in zip(θ_points, labels):
        ax.scatter(θ, μ_min - offset, 60, color='black', marker='v')
        ax.annotate(label, xy=(θ, μ_min - offset),
                    xytext=(θ - 0.012 * clq.θ_range, μ_min + 0.01 * μ_range),
                    fontsize=14)

    # Set labels and limits
    ax.set_xlabel(r"$\theta$", fontsize=18)
    ax.set_xlim([clq.θ_LB, clq.θ_UB])
    ax.set_ylim([min(clq.θ_LB, μ_min) - 0.05 * clq.θ_range, 
                 max(clq.θ_UB, μ_max) + 0.05 * clq.θ_range])
    
    # Add legend
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.show()

plot_policy_functions(clq)
```

The dotted line in the above graph is the 45-degree line.

The blue line shows  the choice of $\theta_{t+1} = \theta'$ chosen by a
continuation Ramsey planner who inherits $\theta_t = \theta$.

The green line shows a  continuation Ramsey planner's choice of
$\mu_t = \mu$ as a function of an inherited $\theta_t = \theta$.

Dynamics under the Ramsey plan are confined to $\theta \in \left[ \theta_\infty^R, \theta_0^R \right]$.  

The blue and green lines intersect each other and the 45-degree line at $\theta =\theta_{\infty}^R$.

Notice that for  $\theta \in \left(\theta_\infty^R, \theta_0^R \right]$

  * $\theta' < \theta$  because the blue line is below the 45-degree line
  * $\mu > \theta $ because the green line is above the 45-degree line

It follows that under the Ramsey plan  $\{\theta_t\}$ and $\{\mu_t\}$ both converge monotonically from above to $\theta_\infty^R$. 


The next code  plots the Ramsey planner's value function $J(\theta)$, which we know is maximized at   $\theta^R_0$, the promised inflation that the Ramsey planner  sets
at time $t=0$.

The figure also plots the limiting value $\theta_\infty^R$ to which  the promised  inflation rate $\theta_t$ converges under the Ramsey plan.

In addition, the figure indicates an MPE inflation rate $\theta^{MPE}$, $\theta^{CR}$, and a bliss inflation $\theta^*$.

```{code-cell} ipython3
:tags: [hide-input]

def plot_value_function(clq):
    """
    Method to plot the value function over the relevant range of θ

    Here clq is an instance of ChangLQ
    """
    fig, ax = plt.subplots()

    ax.set_xlim([clq.θ_LB, clq.θ_UB])
    ax.set_ylim([clq.J_LB, clq.J_UB])

    # Plot value function
    ax.plot(clq.θ_space, clq.J_space, lw=2)
    plt.xlabel(r"$\theta$", fontsize=18)
    plt.ylabel(r"$J(\theta)$", fontsize=18)

    θ_points, labels, _ = compute_θs(clq)

    # Add points for θs
    for θ, label in zip(θ_points, labels):
        ax.scatter(θ, clq.J_LB + 0.02 * clq.J_range, 
                   60, color='black', marker='v')
        ax.annotate(label,
                    xy=(θ, clq.J_LB + 0.01 * clq.J_range),
                    xytext=(θ - 0.01 * clq.θ_range, 
                            clq.J_LB + 0.08 * clq.J_range),
                    fontsize=14)
    plt.tight_layout()
    plt.show()
    
plot_value_function(clq)
```

In the above graph, notice that $\theta^* < \theta_\infty^R < \theta^{CR} < \theta_0^R < \theta^{MPE} .$

In some subsequent calculations, we'll use our Python code to study how gaps between
these outcome vary depending on parameters such as the cost parameter $c$ and the discount factor $\beta$. 

## Ramsey Planner's Value Function 

The next code  plots the Ramsey Planner's value function $J(\theta)$  as well as the value function
of a constrained  Ramsey planner who  must choose a constant
$\mu$.

A time-invariant $\mu$ implies a time-invariant $\theta$, we take the liberty of
labeling this value function $V^{CR}(\theta)$.   

We'll use the code to plot $J(\theta)$ and $V^{CR}(\theta)$ for several values of the discount factor $\beta$ and  the cost of $\mu_t^2$ parameter $c$.

In all of the graphs below, we disarm the Proposition 1 equivalence results by setting $c >0$.

The graphs reveal interesting relationships among $\theta$'s associated with various timing protocols:

 *  $\theta_0^R < \theta^{MPE} $: the initial Ramsey inflation rate exceeds the MPE inflation rate 
 *  $\theta_\infty^R < \theta^{CR} <\theta_0^R$: the initial Ramsey deflation rate, and the associated tax distortion cost $c \mu_0^2$ is less than the limiting Ramsey inflation rate $\theta_\infty^R$ and the associated tax distortion cost $\mu_\infty^2$  
 *  $\theta^* < \theta^R_\infty$: the limiting Ramsey inflation rate exceeds the bliss level of inflation
 *  $J(\theta) \geq V^{CR}(\theta)$
 *  $J(\theta_\infty^R) = V^{CR}(\theta_\infty^R)$

Before doing anything else, let's write code to verify our claim that
$J(\theta_\infty^R) = V^{CR}(\theta_\infty^R)$.

Here is the code.

```{code-cell} ipython3
θ_inf = clq.θ_series[1, -1]
np.allclose(clq.J_θ(θ_inf),
            clq.V_θ(θ_inf))
```

So our claim that $J(\theta_\infty^R) = V^{CR}(\theta_\infty^R)$ is verified numerically.

Since  $J(\theta_\infty^R) = V^{CR}(\theta_\infty^R)$ occurs at a tangency point at which
$J(\theta)$ is increasing in $\theta$, it follows that

$$
V(\theta_\infty^R) \leq J(\theta^{CR})
$$ (eq:comparison2)

with strict inequality when $c > 0$.  

Thus, the limiting continuation value of continuation Ramsey planners is worse that the 
constant value attained by a constrained-to-constant $\mu_t$ Ramsey planner.

Now let's write some code to  plot outcomes under our three timing protocols.

Then we'll use the code to explore how key parameters affect outcomes.

```{code-cell} ipython3
:tags: [hide-input]

def compare_ramsey_CR(clq, ax):
    """
    Method to compare values of Ramsey and Constrained Ramsey (CR)

    Here clq is an instance of ChangLQ
    """
    
    # Calculate CR space range and bounds
    min_CR, max_CR = min(clq.CR_space), max(clq.CR_space)
    range_CR = max_CR - min_CR
    l_CR, u_CR = min_CR - 0.05 * range_CR, max_CR + 0.05 * range_CR
    
    # Set axis limits
    ax.set_xlim([clq.θ_LB, clq.θ_UB])
    ax.set_ylim([l_CR, u_CR])

    # Plot J(θ) and v^CR(θ)
    J_line, = ax.plot(clq.θ_space, clq.J_space, lw=2, label=r"$J(\theta)$")
    CR_line, = ax.plot(clq.θ_space, clq.CR_space, lw=2, label=r"$V^{CR}(\theta)$")

    # Mark key points
    θ_points, labels, θ_colors = compute_θs(clq)
    markers = [ax.scatter(θ, l_CR + 0.02 * range_CR, 60, 
                          marker='v', label=label, color=color)
               for θ, label, color in zip(θ_points, labels, θ_colors)]
    
    # Plot lines at \theta_\infty^R, \theta^{CR}, and \theta^{MPE}
    for i in [1, 2, -1]:
        ax.axvline(θ_points[i], ymin=0.05, lw=2, 
                   linestyle='--', color=θ_colors[i])
        ax.axhline(y=clq.V_θ(θ_points[i]), linestyle='dotted', 
                   lw=1.5, color=θ_colors[i], alpha=0.7)

    v_CR = clq.V_θ(θ_points[2])
    vcr_line = ax.axhline(y=v_CR, linestyle='--', lw=1.5, 
                          color='black', alpha=0.7, label=r"$v^{CR}$")

    return [J_line, CR_line, vcr_line], markers

def plt_clqs(clqs, axes):
    """
    A helper function to plot two separate legends on top and bottom

    Here clqs is a list of ChangLQ instances 
    axes is a list of Matplotlib axes
    """
    line_handles, scatter_handles = {}, {}

    for ax, clq in zip(axes, clqs):
        lines, markers = compare_ramsey_CR(clq, ax)
        ax.set_title(fr'$\beta$={clq.β}, $c$={clq.c}')
        ax.tick_params(axis='x', rotation=45)

        line_handles.update({line.get_label(): line for line in lines})
        scatter_handles.update({marker.get_label(): marker for marker in markers})

    # Collect handles and labels
    line_handles = list(line_handles.values())
    scatter_handles = list(scatter_handles.values())

    # Create legends
    fig = plt.gcf()
    fig.legend(handles=line_handles, 
               labels=[line.get_label() for line in line_handles],
               loc='upper center', ncol=4, 
               bbox_to_anchor=(0.5, 1.1), prop={'size': 12})
    fig.legend(handles=scatter_handles, 
               labels=[marker.get_label() for marker in scatter_handles],
               loc='lower center', ncol=5, 
               bbox_to_anchor=(0.5, -0.1), prop={'size': 12})
    
    plt.tight_layout()
    plt.show()

def generate_table(clqs, dig=3):
    """
    A function to generate a table of θ values and display it using LaTeX

    Here clqs is a list of ChangLQ instances and 
    dig is the number of digits to round to
    """

    # Collect data
    label_maps = {rf'$\beta={clq.β}, c={clq.c}$': 
                  [f'{round(val, dig):.3f}' for val in compute_θs(clq)[0]] for clq in clqs}
    labels = compute_θs(clqs[0])[1]
    data_frame = pd.DataFrame(label_maps, index=labels)

    # Generate table
    columns = ' & '.join([f'\\text{{{col}}}' for col in data_frame.columns])
    rows = ' \\\\\n'.join(
        [' & '.join([f'\\text{{{label}}}'] + [f'{val}' for val in row]) 
         for label, row in zip(data_frame.index, data_frame.values)])

    latex_code = rf"""
    \begin{{array}}{{{'c' * (len(data_frame.columns) + 1)}}}
    & {columns} \\
    \hline
    {rows}
    \end{{array}}
    """
    
    display(Math(latex_code))
```

```{code-cell} ipython3
# Compare different β values
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
β_values = [0.7, 0.8, 0.99]

clqs = [ChangLQ(β=β, c=2) for β in β_values]
plt_clqs(clqs, axes)
```

```{code-cell} ipython3
generate_table(clqs, dig=3)
```

The above graphs and table convey many useful things.

The horizontal dotted lines indicate values 
 $V(\mu_\infty^R), V(\mu^{CR}), V(\mu^{MPE}) $ of time-invariant money
growth rates $\mu_\infty^R, \mu^{CR}$ and $\mu^{MPE}$, respectfully. 

Notice how $J(\theta)$ and $V^{CR}(\theta)$ are tangent and increasing at
 $\theta = \theta_\infty^R$, which implies that $\theta^{CR} > \theta_\infty^R$
 and $J(\theta^{CR}) > J(\theta_\infty^R)$. 

 Notice how changes in $\beta$ alter $\theta_\infty^R$
 and $\theta_0^R$ but neither $\theta^*, \theta^{CR}$, nor $\theta^{MPE}$, in accord with formulas
 {eq}`eq:Friedmantheta`,  {eq}`eq:muRamseyconstrained`, and {eq}`eq:Markovperfectmu`, which imply that

$$
\begin{aligned}
\theta^{CR} & = - \frac{\alpha u_1}{\alpha^2 u_2 + c } \\
\theta^{MPE} &  = - \frac{\alpha u_1}{\alpha^2 u_2 + (1+\alpha)c} \\
\theta^{MPE} & = - \frac{\alpha u_1}{\alpha^2 u_2 + (1+\alpha)c}
\end{aligned}
$$

(TO TOM: $\theta^{MPE}$ is repeated in the above equations. Should one of them be $\theta^*$?)


 But let's see what happens when we change $c$.

```{code-cell} ipython3
# Increase c to 100
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
c_values = [1, 10, 100]

clqs = [ChangLQ(β=0.85, c=c) for c in c_values]
plt_clqs(clqs, axes)
```

```{code-cell} ipython3
generate_table(clqs, dig=4)
```

The above table and figures show how 
  changes in $c$ alter $\theta_\infty^R$
 and $\theta_0^R$ as well as  $\theta^{CR}$ and  $\theta^{MPE}$, but not
 $\theta^*$, again  in accord with formulas
 {eq}`eq:Friedmantheta`,  {eq}`eq:muRamseyconstrained`, and {eq}`eq:Markovperfectmu`. 

Notice that as $c $ gets larger and larger,   $\theta_\infty^R, \theta_0^R$ 
and $\theta^{CR}$ all converge to $\theta^{MPE}$. 

Now let's watch what happens when we drive $c$ toward zero.

```{code-cell} ipython3
# Decrease c towards 0
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
c_limits = [1, 0.1, 0.01]

clqs = [ChangLQ(β=0.85, c=c) for c in c_limits]
plt_clqs(clqs, axes)
```

The above graphs indicate that as $c$ approaches zero, $\theta_\infty^R, \theta_0^R, \theta^{CR}$,
and $\theta^{MPE}$ all approach $\theta^*$.

This makes sense, because it was by adding costs of distorting taxes that Calvo {cite}`Calvo1978` drove a wedge between Friedman's optimal deflation rate and the inflation rates chosen by a Ramsey planner. 

The following code  plots sequences  $\vec \mu$ and $\vec \theta$ prescribed by a Ramsey plan as well as the constant levels $\mu^{CR}$ and $\mu^{MPE}$.

The following graphs report values for the value function parameters $g_0, g_1, g_2$,
and the Ramsey policy function parameters $b_0, b_1, d_0, d_1$ associated with the indicated
parameter pair $\beta, c$.  

We'll vary $\beta$ while keeping a small $c$.

After that we'll study consequences of raising $c$. 

We'll watch how the decay rate $d_1$ governing the dynamics of $\theta_t^R$ is affected by alterations in the parameters $\beta, c$.

```{code-cell} ipython3
:tags: [hide-input]

def plot_ramsey_MPE(clq, T=15):
    """
    Method to plot Ramsey plan against Markov Perfect Equilibrium

    Here clq is an instance of ChangLQ
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plots = [clq.θ_series[1, 0:T], clq.μ_series[0:T]]
    MPEs = [clq.θ_MPE, clq.μ_MPE]
    labels = [r"\theta", r"\mu"]

    for ax, plot, MPE, label in zip(axes, plots, MPEs, labels):
        ax.plot(plot, label=fr"${label}^R$")
        ax.hlines(MPE, 0, T-1, colors='orange', label=fr"${label}^{{MPE}}$")
        if label == r"\theta":
            ax.hlines(clq.θ_B, 0, T-1, colors='r', label=r"$\theta^*$")
        ax.hlines(clq.μ_CR, 0, T-1, colors='g', label=fr"${label}^{{CR}}$")
        ax.set_xlabel(r"$t$", fontsize=14)
        ax.set_ylabel(fr"${label}_t$", fontsize=16)
        ax.legend(loc='upper right')

    fig.suptitle(fr'$\beta$={clq.β}, $c$={clq.c}', fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()

def generate_param_table(clq):
    """
    Method to generate a table of parameters

    Here clq is an instance of ChangLQ
    """

    # Collect data
    param_names = [r'$g_0$', r'$g_1$', r'$g_2$', 
                   r'$b_0$', r'$b_1$', r'$d_0$', r'$d_1$']
    params = [clq.g0, clq.g1, clq.g2, 
              clq.b0, clq.b1, clq.d0, clq.d1]

    # Generate table
    label = rf'$\beta={clq.β}, c={clq.c}$'
    data_frame = pd.DataFrame({label: params}, index=param_names).round(2).T

    columns = ' & '.join([f'\\text{{{col}}}' for col in data_frame.columns])
    rows = ' \\\\\n'.join([' & '.join([f'\\text{{{index}}}'] + [
        f'{val}' for val in row]) for index, row in data_frame.iterrows()])
    
    latex_code = rf"""
    \begin{{array}}{{{'c' * (len(data_frame.columns) + 1)}}}
    & {columns} \\
    \hline
    {rows}
    \end{{array}}
    """
    
    display(Math(latex_code))
```

```{code-cell} ipython3
for β in β_values:
    clq = ChangLQ(β=β, c=2)
    generate_param_table(clq)
    plot_ramsey_MPE(clq)
```

Notice how $d_1$ changes as we raise the discount factor parameter $\beta$.

Now let's study how increasing $c$ affects $\vec \theta, \vec \mu$ outcomes.

```{code-cell} ipython3
# Increase c to 100
for c in c_values:
    clq = ChangLQ(β=0.85, c=c)
    generate_param_table(clq)
    plot_ramsey_MPE(clq)
```

Evidently, increasing $c$ causes the decay factor $d_1$ to increase.


Next, let's look at consequences of increasing the demand for real balances parameter
$\alpha$ from its default value  $\alpha=1$ to $\alpha=4$.

```{code-cell} ipython3
# Increase c to 100
for c in [10, 100]:
    clq = ChangLQ(α=4, β=0.85, c=c)
    generate_param_table(clq)
    plot_ramsey_MPE(clq)
```

The above panels for an $\alpha = 4$ setting indicate that $\alpha$ and $c$ affect outcomes 
in interesting ways. 

We leave it to the reader to explore consequences of other constellations of parameter values.

### Time Inconsistency of Ramsey Plan

The variation over time in $\vec \mu$ chosen by the Ramsey planner
is a symptom of time inconsistency.

- The Ramsey planner reaps immediate benefits from promising lower
  inflation later to be achieved by costly distorting taxes.
- These benefits are intermediated by reductions in expected inflation
  that precede the  reductions in money creation rates that rationalize them, as indicated by
  equation {eq}`eq_old3`.
- A government authority offered the opportunity to ignore effects on
  past utilities and to reoptimize at date $t \geq 1$ would, if allowed, want
  to deviate from a Ramsey plan.

```{note}
A constrained-to-constant-$\mu$  Ramsey plan  is  time consistent by construction. So is a Markov perfect plan.
```

### Implausibility of Ramsey Plan 

In settings in which governments actually choose sequentially, many economists
regard a time inconsistent plan as implausible because of the incentives to
deviate that are presented  along the plan.

(TO TOM: In our meeting, you suggested that we can improve the sentence above.)

A way to state  this reaction   is to say that a Ramsey plan is not credible because there are persistent incentives for policymakers to deviate from it.

For that reason, the Markov perfect equilibrium concept attracts many
economists.

* A Markov perfect equilibrium plan is constructed to insure that government policymakers who choose sequentially do not want to deviate from it.

The *no incentive to deviate from the plan* property is what makes the Markov perfect equilibrium concept attractive.


## Comparison of Equilibrium Values

We have computed plans for

- an ordinary (unrestricted) Ramsey planner who chooses a sequence
  $\{\mu_t\}_{t=0}^\infty$ at time $0$
- a Ramsey planner restricted to choose a constant $\mu$ for all
  $t \geq 0$
- a Markov perfect sequence of governments

Below we compare equilibrium time zero values for these three.

We confirm that the value delivered by the unrestricted Ramsey planner
exceeds the value delivered by the restricted Ramsey planner which in
turn exceeds the value delivered by the Markov perfect sequence of
governments.

```{code-cell} ipython3
clq.J_series[0]
```

```{code-cell} ipython3
clq.J_CR
```

```{code-cell} ipython3
clq.J_MPE
```

## Digression on Timeless Perspective

Our calculations have confirmed that  $ \vec \mu^R, \vec \theta^R, \vec v^R $ are each monotone sequences that are bounded below and converge from above  to limiting values.  

Some authors are fond of focusing only on these limiting values.

They justify that by saying that they are taking a **timeless perspective** that ignores  the transient movements in $ \vec \mu^R, \vec \theta^R, \vec v^R $ that are destined  eventually to fade away as $\theta_t$ described by Ramsey plan system {eq}`eq_old9` converges from above.  

   * the timeless perspective pretends that  Ramsey plan was actually solved long ago, and that we are stuck with it.  



### Ramsey Plan Strikes Back

Research by Abreu {cite}`Abreu`,  Chari and Kehoe {cite}`chari1990sustainable`
{cite}`stokey1989reputation`, and Stokey {cite}`Stokey1991` discovered conditions under which a Ramsey plan can be rescued from the complaint that it is not credible.

They  accomplished this by expanding the
description of a plan to include expectations about *adverse consequences* of deviating from
it that can serve to deter deviations.

We turn to such theories in this  quantecon lecture {doc}`calvo_abreu`.
