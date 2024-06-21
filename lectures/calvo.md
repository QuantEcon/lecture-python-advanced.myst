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

# Ramsey Plans, Time Inconsistency, Sustainable Plans

```{index} single: Models; Additive functionals
```

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade quantecon
```

## Overview

This lecture describes several  linear-quadratic versions of a model that Guillermo Calvo {cite}`Calvo1978` used to illustrate the **time inconsistency** of optimal government
plans.

Like Chang {cite}`chang1998credible`, we use the models as a laboratory in which to explore consequences of  timing protocols for government decision making.

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

We'll use ideas from  papers by Cagan {cite}`Cagan`, Calvo {cite}`Calvo1978`, Stokey {cite}`stokey1989reputation`, {cite}`Stokey1991`,
Chari and Kehoe {cite}`chari1990sustainable`, Chang {cite}`chang1998credible`, and Abreu {cite}`Abreu` as
well as from chapter 19 of {cite}`Ljungqvist2012`.

In addition, we'll use ideas from linear-quadratic dynamic programming
described in  [Linear Quadratic Control](https://python-intro.quantecon.org/lqcontrol.html) as applied to Ramsey problems in {doc}`Stackelberg problems <dyn_stack>`.

We  specify model fundamentals  in  ways that allow us to use
linear-quadratic discounted dynamic programming to compute an optimal government
plan under each of our timing protocols. 


We'll start with some imports:

```{code-cell} ipython3
import numpy as np
from quantecon import LQ
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.ticker import FormatStrFormatter
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

(When there is no uncertainty, an assumption of **rational expectations** implies **perfect foresight**).

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

We shall use this insight to help us simplify our  analsis of alternative  government policy problems.

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

We assume that a government  believes that a representative household's utility of real balances at time $t$ is:

```{math}
:label: eq_old5

U(m_t - p_t) = a_0 + a_1 (m_t - p_t) - \frac{a_2}{2} (m_t - p_t)^2, \quad a_0 > 0, a_1 > 0, a_2 > 0
```

The "bliss level" of real balances is then $\frac{a_1}{a_2}$.

## Friedman's Optimal Rate of Deflation

The money demand function {eq}`eq_old1` % and the utility function {eq}`eq_old5`
imply that inflation rate $\theta_t$ that maximizes {eq}`eq_old5` is 

$$
\theta_t = \theta^* = -\frac{a_1}{a_2 \alpha}
$$ (eq:Friedmantheta)

According to Milton Friedman, the government should withdraw and destroy money at a rate 
that implies an inflation rate given by  {eq}`eq:Friedmantheta`.

In our setting, that could be accomplished by setting 


$$
\mu_t = \mu^* = \theta^* , t \geq 0
$$ (eq:Friedmanrule)

where $\theta^*$ is given by equation {eq}`eq:Friedmantheta`.

To deduce this recommendation, Milton Friedman assumed that the taxes that government must impose in order acquire money at rate $\mu_t$ do not distort economic decisions.

  - for example, the government imposes lump sum taxes that distort no decisions by private agents

## Calvo's Perturbing of Friedman's Optimal Quantity of Money

The starting point of Calvo {cite}`Calvo1978` and  Chang {cite}`chang1998credible`
is that such lump sum taxes are not available.

Instead, the government acquires money by levying taxes that distort decisions and thereby impose costs on the representative consumer.

In the models of  Calvo {cite}`Calvo1978` and  Chang {cite}`chang1998credible`, the government takes those costs tax-distortion costs into account.

It balances the costs of imposing the distorting taxes needed to acquire the money that it destroys in order to generate deflation against the benefits that expected deflation generates by raising the representative households holdings of real balances.  

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

-s(\theta_t, \mu_t) \equiv - r(x_t,\mu_t) = \begin{bmatrix} 1 \\ \theta_t \end{bmatrix}' \begin{bmatrix} a_0 & -\frac{a_1 \alpha}{2} \\ -\frac{a_1 \alpha}{2} & -\frac{a_2 \alpha^2}{2} \end{bmatrix} \begin{bmatrix} 1 \\ \theta_t \end{bmatrix} - \frac{c}{2} \mu_t^2 =  - x_t'Rx_t - Q \mu_t^2
```

The  government's time $0$ value is 

```{math}
:label: eq_old7

v_0 = - \sum_{t=0}^\infty \beta^t r(x_t,\mu_t) = - \sum_{t=0}^\infty \beta^t s(\theta_t,\mu_t)
```

The government's time $t$ continuation value $v_t$ is 

$$
v_t = - \sum_{j=0}^\infty \beta^j s(\theta_{t+j}, \mu_{t+j}) .
$$

We can represent the dependence of  $v_0$ on $(\vec \theta, \vec \mu)$ recursively via the  difference equation

```{math}
:label: eq_old8

v_t = - s(\theta_t, \mu_t) + \beta v_{t+1}
```

It is useful to evaluate {eq}`eq_old8` under a time invariant money growth rate $\mu_t = \bar \mu$
that according to equation {eq}`eq_old3` would bring forth a constant inflation rate equal to $\bar \mu$.  

Under that policy,

$$
v_t = v^{\bar \mu} \equiv  - \frac{s(\bar \mu, \bar \mu)}{1-\beta} 
$$ (eq:barvdef)

for all $t \geq 0$. 


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
$$

where we have called $s(\theta_t, \mu_t) = r(x_t, \mu_t)$, as
in {eq}`eq_old7`.

Thus,  a triple of sequences
$(\vec \mu, \vec \theta, \vec v)$ depends on  a
sequence $\vec \mu \in L^2$.

At this point $\vec \mu \in L^2$ is an arbitrary exogenous policy.

A theory of government
decisions will  make $\vec \mu$ endogenous, i.e., a theoretical **output** instead of an **input**.


## Intertemporal Structure 

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

## Four Timing Protocols

We consider four models of government policy making that  differ in

- **what** a  policymaker is allowed to choose, either a sequence
  $\vec \mu$ or just   $\mu_t$ in a single period $t$.
- **when** a  policymaker chooses, either once and for all at time $0$, or at some time or times  $t \geq 0$.
- what a policymaker **assumes** about how its choice of $\mu_t$
  affects the representative  agent's expectations about earlier and later
  inflation rates.

In two of our models, a single policymaker  chooses a sequence
$\{\mu_t\}_{t=0}^\infty$ once and for all, knowing  how
$\mu_t$ affects household one-period utilities at dates $s = 0, 1, \ldots, t-1$

- these two models  thus employ a  **Ramsey** or **Stackelberg** timing protocol.

In two other models, there is a sequence of policymakers, each of whom
sets $\mu_t$ at one $t$ only.

- Each time $t$  policymaker ignores  effects that its choice of $\mu_t$ has on household one-period utilities at dates $s = 0, 1, \ldots, t-1$.

The four models differ with respect to timing protocols, constraints on
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
    - a time $t$ policymaker chooses $\mu_t$ only and forecasts that future government decisions are unaffected by its choice; or
- A sequence of separate policymakers  in which 
    - a time $t$ policymaker chooses  only $\mu_t$ but believes that its choice of $\mu_t$  shapes the representative agent's beliefs about  future rates of money creation and inflation, and through them, future government actions.

The relationship between  outcomes in  the first (Ramsey) timing protocol and the fourth timing protocol and belief structure is the subject of a literature on **sustainable** or **credible** public policies (Chari and Kehoe {cite}`chari1990sustainable`
{cite}`stokey1989reputation`, and Stokey {cite}`Stokey1991`). 

We'll discuss that topic later in this lecture.

We'll begin with the timing protocol associated with a Ramsey plan.

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
$$

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

It remains for the Ramsey planner to set $\theta_0$.  

Subproblem 2 does that.


### Subproblem 2



The value of the Ramsey problem is

$$
V = \max_{x_0} J(x_0)
$$

where $V$ is the maximum value of $v_0$ defined in equation {eq}`eq_old7`.

The value function

$$
J(x_0) = -\begin{bmatrix} 1 & \theta_0 \end{bmatrix} \begin{bmatrix} P_{11} & P_{12} \\ P_{21} & P_{22} \end{bmatrix} \begin{bmatrix} 1 \\ \theta_0 \end{bmatrix} = -P_{11} - 2 P_{21} \theta_0 - P_{22} \theta_0^2
$$

Maximizing $J(x_0)$  with respect to $\theta_0$ yields the FOC:

$$
- 2 P_{21} - 2 P_{22} \theta_0 =0
$$

which implies

$$
\theta_0 = \theta_0^R = - \frac{P_{21}}{P_{22}}
$$




### Representation of Ramsey Plan

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



To interpret system {eq}`eq_old9`, think of the  sequence
$\{\theta_t\}_{t=0}^\infty$ as a sequence of
synthetic **promised inflation rates**.

For some purposes, we can think of these promised inflation rates  just as computational devices for
generating a sequence $\vec\mu$ of money growth rates that when  substituted into equation {eq}`eq_old3` generate  **actual** rates of inflation.

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

while $\mu_t$ varies over time according to 

$$
 \mu_t = b_0 + b_1 d_0 \left(\frac{1 - d_1^t}{1 - d_1} \right)  + b_1 d_1^t \theta_0^R.
$$ (eq:mutimeinconsist)

 
Variation of  $ \vec \mu^R, \vec \theta^R, \vec v^R $ over time  are  symptoms of time inconsistency.

As our subsequent calculations will verify, $ \vec \mu^R, \vec \theta^R, \vec v^R $ are each monotone sequences that are bounded below and converge to limiting values.  

Some authors are fond of focusing only on these limiting values.

They justify that by saying that they are taking a **timeless perspective** that ignores that the
Ramsey planner ignores the transient movements in $ \vec \mu^R, \vec \theta^R, \vec v^R $ that will eventually fade away as $\theta_t$ described by Ramsey plan system {eq}`eq_old9` eventually fade away.  

   * the timeless perspective proceeds as if the Ramsey plan was actually solved long ago, and that we are stuck with it now.  

The Ramsey planner reaps immediate benefits from promising lower in the future   inflation  by   later imposing costly distortions. 

These benefits are intermediated by reductions in expected inflation
that precede  reductions in  money creation rates that foreshadow  them, as indicated by equation {eq}`eq_old3`.  

A government decision maker offered an opportunity to ignore effects on
  past utilities and to re-optimize at date $ t \geq 1 $ would want to deviate from a Ramsey plan.




### Multiple roles of $\theta_t$

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

### Time Inconsistency

As discussed in {doc}`Stackelberg problems <dyn_stack>` and {doc}`Optimal taxation with state-contingent debt <opt_tax_recur>`, a continuation Ramsey plan is not a Ramsey plan.

This is a concise way of characterizing the time inconsistency of a Ramsey plan.

The time inconsistency of a Ramsey plan has motivated other models of government decision making
that, relative to a Ramsey plan,  alter either

- the timing protocol and/or
- assumptions about how government decision makers think their decisions affect the representative agent's beliefs about future government decisions

## A Constrained-to-a-Constant-Growth-Rate Ramsey Government

We now consider a different model of optimal government behavior.

We created this version of the model  to highlight an aspect of a Ramsey plan associated with its time inconsistency, namely, the feature that optimal settings of the  policy instrument vary over time.

Instead of allowing the government to choose different settings of its instrument at different moments, we now assume that
at time $0$, a  government at time $0$ once and for all  chooses a **constant** sequence
$\mu_t = \check \mu$ for all $t \geq 0$.

We  assume that the government knows the perfect foresight outcome implied by equation {eq}`eq_old2` that $\theta_t = \check \mu$ when there is  a constant
$\mu$ for all $t \geq 0$.

The government chooses $\mu$  to maximize

$$
U(-\alpha \check \mu) - \frac{c}{2} \check \mu^2
$$



With the quadratic form {eq}`eq_old5` for the utility function $U$, the
maximizing $\bar \mu$ is

$$
\check \mu = - \frac{\alpha a_1}{\alpha^2 a_2 + c }
$$ (eq:muRamseyconstrained)

The value function of a **constrained to constant $\mu$** Ramsey planner is

$$
\check V \equiv (1-\beta)^{-1} \left[ U (-\alpha \check \mu) - \frac{c}{2} \check \mu^2 \right]
$$ (eq:vcheckformula)


**Summary:** We have  introduced the constrained-to-a-constant $\mu$
government in order to highlight  time-variation of
$\mu_t$ as a telltale sign of  time inconsistency of a Ramsey plan.

## Markov Perfect Governments

We now  alter the timing protocol by assuming  a sequence of
government policymakers.

A time $t$ government chooses $\mu_t$ and expects all future governments to set
$\mu_{t+j} = \bar \mu$.

This assumption mirrors an assumption made in a different setting  in this QuantEcon lecture:  [Markov Perfect Equilibrium](https://python-intro.quantecon.org/markov_perf.html).

When it sets $\mu_t$ at time $t$, the  government   at $t$ believes that $\bar \mu$ is
unaffected by its choice of $\mu_t$.

The time $t$ rate of inflation is then:

$$
\theta_t = \frac{\alpha}{1+\alpha} \bar \mu + \frac{1}{1+\alpha} \mu_t
$$

The time $t$ government policymaker then chooses $\mu_t$ to
maximize:

$$
W = U(-\alpha \theta_t) - \frac{c}{2} \mu_t^2 + \beta V(\bar \mu)
$$

where $V(\bar \mu)$ is the time $0$ value $v_0$ of
recursion {eq}`eq_old8` under a money supply growth rate that is forever constant
at $\bar \mu$.

Substituting for $U$ and $\theta_t$ gives:

$$ 
\begin{aligned}
V(\mu_t) & = a_0 + a_1\left(-\frac{\alpha^2}{1+\alpha} \bar \mu - \frac{\alpha}{1+\alpha} \mu_t\right) - \frac{a_2}{2}\left(-\frac{\alpha^2}{1+\alpha} \bar \mu - \frac{\alpha}{1+\alpha} \mu_t\right)^2 - \frac{c}{2} \mu_t^2  \\ 
& \quad \quad \quad + \beta V(\bar \mu)
\end{aligned}
$$ (eq:Vmutemp)

The first-order necessary condition for $\mu_t$ is then:

$$
- \frac{\alpha}{1+\alpha} a_1 - a_2(-\frac{\alpha^2}{1+\alpha} \bar \mu - \frac{\alpha}{1+\alpha} \mu_t)(- \frac{\alpha}{1+\alpha}) - c \mu_t = 0
$$

Rearranging we get:

$$
\mu_t = \frac{- a_1}{\frac{1+\alpha}{\alpha}c + \frac{\alpha}{1+\alpha}a_2} - \frac{\alpha^2 a_2}{\left[ \frac{1+\alpha}{\alpha}c + \frac{\alpha}{1+\alpha} a_2 \right] (1+\alpha)}\bar \mu
$$

A **Markov Perfect Equilibrium** (MPE) outcome sets
$\mu_t = \bar \mu$:

$$
\mu_t = \bar \mu = \frac{-a_1}{\frac{1+\alpha}{\alpha}c + \frac{\alpha}{1+\alpha} a_2 + \frac{\alpha^2}{1+\alpha} a_2}
$$

This can be simplified to:

$$
\mu^{MPE} \equiv \bar \mu = - \frac{\alpha a_1}{\alpha^2 a_2 + (1+\alpha)c}
$$ (eq:Markovperfectmu)

and the value of a Markov perfect equilibrium is 

$$
V^{MPE} = V(\mu_t) \vert_{\mu_t = \bar \mu}
$$ (eq:VMPE)

where $V(\mu_t)$ satisfies equation {eq}`eq:Vmutemp`.

Under the  Markov perfect timing protocol 

 * a government takes $\bar \mu$ as given when it chooses $\mu_t$
 * we equate $\mu_t = \mu$ only **after** we have computed a time $t$ government's first-order condition for $\mu_t$.

## Outcomes under Three Timing Protocols

We  want to compare outcome sequences  $\{ \theta_t,\mu_t \}$ under three timing protocols associated with 

  * a standard Ramsey plan with its time varying $\{ \theta_t,\mu_t \}$ sequences 
  * a Markov perfect equilibrium 
  * our nonstandard  Ramsey plan in which the planner is restricted to choose $\mu_t = \check\mu$
for all $t \geq 0$.

We have computed closed form formulas for several of these outcomes, which we find it convenient to repeat here.

In particular, the constrained to constant inflation Ramsey inflation outcome is $\check \mu$,
which according to equation {eq}`eq:muRamseyconstrained` is

$$
\check \theta = - \frac{\alpha a_1}{\alpha^2 a_2 + c }
$$ 

Equation {eq}`eq:Markovperfectmu` implies that the Markov perfect constant inflation rate is 

$$
\theta^{MPE}  = - \frac{\alpha a_1}{\alpha^2 a_2 + (1+\alpha)c}
$$ 

According to equation {eq}`eq:Friedmantheta`, the bliss level of inflation that we associated with a Friedman rule is

$$
 \theta^* = -\frac{a_1}{a_2 \alpha}
$$ 


**Proposition 1:** When $c=0$,  $\theta^{MPE} = \check \theta = \theta^*$ and 
$\theta_0^R = \theta_\infty^R$. 

The first two equalities follow from the preceding three equations. We'll illustrate the assertion in  the third equality that equates $\theta_0^R$ to $ \theta_\infty^R$ with some quantitative examples below.

Proposition 1 draws attention to how   a positive tax distortion parameter $c$ alters  the prescription for an optimal rate of deflation that Milton Friedman financed  by imposing a lump sum tax.  

We'll compute and display

 *   $(\vec \theta^R, \vec \mu^R)$: the ordinary time-varying Ramsey sequences
 *   $(\theta^{MPE}, \mu^{MPE})$: the MPE fixed values
 *   $(\check \theta, \check \mu)$: the fixed values associate with  our nonstandard time-invariant 
values Ramsey plan
 *   $\theta^*$: the  bliss level of inflation prescribed by a Friedman rule




We will create a class ChangLQ that solves the models and stores their values

```{code-cell} ipython3
class ChangLQ:
    """
    Class to solve LQ Chang model
    """
    def __init__(self, β, c, α=1, α0=1, α1=0.5, α2=3, T=1000, θ_n=200):

        # Record parameters
        self.α, self.α0, self.α1, self.α2 = α, α0, α1, α2
        self.β, self.c, self.T, self.θ_n = β, c, T, θ_n

        # Solve the Ramsey Problem #

        # LQ Matrices
        R = -np.array([[α0,            -α1 * α / 2],
                       [-α1 * α/2, -α2 * α**2 / 2]])
        Q = -np.array([[-c / 2]])
        A = np.array([[1, 0], [0, (1 + α) / α]])
        B = np.array([[0], [-1 / α]])

        # Solve LQ Problem (Subproblem 1)
        lq = LQ(Q, R, A, B, beta=self.β)
        self.P, self.F, self.d = lq.stationary_values()

        # Solve Subproblem 2
        self.θ_R = -self.P[0, 1] / self.P[1, 1]

        # Find bliss level of θ
        self.θ_B = - α1 / (α2* α)

        # Solve the Markov Perfect Equilibrium
        self.μ_MPE = -α1 / ((1 + α) / α * c + α / (1 + α)
                      * α2 + α**2 / (1 + α) * α2)
        self.θ_MPE = self.μ_MPE
        self.μ_check = -α * α1 / (α2 * α**2 + c)
        self.θ_check = self.μ_check

        # Calculate value under MPE and Check economy
        self.J_MPE  = (α0 + α1 * (-α * self.μ_MPE) - α2 / 2
                      * (-α * self.μ_MPE)**2 - c/2 * self.μ_MPE**2) / (1 - self.β)
        self.J_check = (α0 + α1 * (-α * self.μ_check) - α2/2
                        * (-α * self.μ_check)**2 - c / 2 * self.μ_check**2) \
                        / (1 - self.β)

        # Simulate Ramsey plan for large number of periods
        θ_series = np.vstack((np.ones((1, T)), np.zeros((1, T))))
        μ_series = np.zeros(T)
        J_series = np.zeros(T)
        θ_series[1, 0] = self.θ_R
        [μ_series[0]] = -self.F.dot(θ_series[:, 0])
        J_series[0] = -θ_series[:, 0] @ self.P @ θ_series[:, 0].T
        for i in range(1, T):
            θ_series[:, i] = (A - B @ self.F) @ θ_series[:, i-1]
            [μ_series[i]] = -self.F @ θ_series[:, i]
            J_series[i] = -θ_series[:, i] @ self.P @ θ_series[:, i].T

        self.J_series = J_series
        self.μ_series = μ_series
        self.θ_series = θ_series

        # Find the range of θ in Ramsey plan
        θ_LB = min(θ_series[1, :])
        θ_LB = min(θ_LB, self.θ_B)
        θ_UB = max(θ_series[1, :])
        θ_UB = max(θ_UB, self.θ_MPE)
        θ_range = θ_UB - θ_LB
        self.θ_LB = θ_LB - 0.05 * θ_range
        self.θ_UB = θ_UB + 0.05 * θ_range
        self.θ_range = θ_range

        # Find value function and policy functions over range of θ
        θ_space = np.linspace(self.θ_LB, self.θ_UB, 200)
        J_space = np.zeros(200)
        check_space = np.zeros(200)
        μ_space = np.zeros(200)
        θ_prime = np.zeros(200)

        self.J_θ = lambda θ: - np.array((1, θ)) \
                    @ self.P @ np.array((1, θ)).T
        self.V_θ = lambda θ: (α0 + α1 * (-α * θ)
                    - α2/2 * (-α * θ)**2 - c/2 * θ**2) \
                    / (1 - self.β)
        
        for i in range(200):
            J_space[i] = self.J_θ(θ_space[i])
            [μ_space[i]] = - self.F @ np.array((1, θ_space[i]))
            x_prime = (A - B @ self.F) @ np.array((1, θ_space[i]))
            θ_prime[i] = x_prime[1]
            check_space[i] = self.V_θ(θ_space[i])

        J_LB = min(J_space)
        J_UB = max(J_space)
        J_range = J_UB - J_LB
        self.J_LB = J_LB - 0.05 * J_range
        self.J_UB = J_UB + 0.05 * J_range
        self.J_range = J_range
        self.J_space = J_space
        self.θ_space = θ_space
        self.μ_space = μ_space
        self.θ_prime = θ_prime
        self.check_space = check_space
```

Let's create an instance of ChangLQ with the following parameters:

```{code-cell} ipython3
clq = ChangLQ(β=0.85, c=2)
```

The following code  plots the Ramsey planner's value function $J(\theta)$, which we know is maximized at   $\theta^R_0$, the promised inflation planner that the Ramsey planner chooses to set
at time $t=0$.

The figure also plots the limiting value $\theta_\infty^R$ to which  the promised  inflation rate $\theta_t$ converges under the Ramsey plan.

In addition, the figure indicates  an MPE inflation rate $\check \theta$ and a bliss inflation $\theta^*$.

```{code-cell} ipython3
def compute_θs(clq):
    """
    Method to compute θ and assign corresponding labels and colors 

    Here clq is an instance of ChangLQ
    """
    θ_points = [clq.θ_B, clq.θ_series[1, -1], 
                clq.θ_check, clq.θ_space[np.argmax(clq.J_space)], 
                clq.θ_MPE]
    labels = [r"$\theta^*$", r"$\theta_\infty^R$", 
              r"$\theta^\check$", r"$\theta_0^R$", 
              r"$\theta^{MPE}$"]
    θ_colors = ['r', 'C5', 'g', 'C0', 'orange']

    return θ_points, labels, θ_colors


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

    t1 = clq.θ_space[np.argmax(clq.J_space)]
    tR = clq.θ_series[1, -1]
    θ_points = [t1, tR, clq.θ_B, clq.θ_MPE, clq.θ_check]
    labels = [r"$\theta_0^R$", r"$\theta_\infty^R$",
              r"$\theta^*$", r"$\theta^{MPE}$",
              r"$\theta^\check$"]

    θ_points, labels, _ = compute_θs(clq)
    
    # Add points for θs
    for θ, label in zip(θ_points, labels):
        ax.scatter(θ, clq.J_LB + 0.02 * clq.J_range, 60, 'black', 'v')
        ax.annotate(label,
                    xy=(θ, clq.J_LB + 0.01 * clq.J_range),
                    xytext=(θ - 0.01 * clq.θ_range,
                    clq.J_LB + 0.08 * clq.J_range),
                    fontsize=14)
    plt.tight_layout()
    plt.show()

plot_value_function(clq)
```

The next code  plots the Ramsey Planner's value function $J(\theta)$  as well the value function
of a constrained  Ramsey planner who  must choose a constant
$\mu$.

A time-invariant $\mu$ implies a time-invariant $\theta$, we take the liberty of
labeling this value function $\check V(\theta)$.   

We'll use the code to plot $J(\theta)$ and $\check V(\theta)$ for several values of the discount factor $\beta$ and  the cost of $\mu_t^2$ parameter $c$.

In all of the graphs below, we disarm the Proposition 1 equivalence results by setting $c >0$.

The graphs reveal interesting relationships among $\theta$'s associated with various timing protocols:

 *  $\theta_0^R < \theta^{MPE} $: the initial Ramsey inflation rate exceeds the MPE inflation rate 
 *  $\theta_\infty^R < \check \theta <\theta_0^R$: the initial Ramsey deflation rate, and the associated tax distortion cost $c \mu_0^2$ is less than the limiting Ramsey inflation rate $\theta_\infty^R$ and the associated tax distortion cost $\mu_\infty^2$  
 *  $\theta^* < \theta^R_\infty$: the limiting Ramsey inflation rate exceeds the bliss level of inflation
 *  $J(\theta) \geq \check V(\theta)$
 *  $J(\theta_\infty^R) = \check V(\theta_\infty^R)$

Before doing anything else, let's write code to verify our claim that
$J(\theta_\infty^R) = \check V(\theta_\infty^R)$.

Here is the code.


```{code-cell} ipython3
θ_inf = clq.θ_series[1, -1]
np.allclose(clq.J_θ(θ_inf),
            clq.V_θ(θ_inf))
```
So our claim is verified numerically.

Since  $J(\theta_\infty^R) = \check V(\theta_\infty^R)$ occurs at a tangency point at which
$J(\theta)$ is increasing in $\theta$, it follows that

$$
V(\theta_\infty^R) \leq J(\check \theta)
$$ (eq:comparison2)

with strict inequality when $c > 0$.  

Thus, the limiting continuation value of continuation Ramsey planners is worse that the 
constant value attained by a constrained-to-constant $\mu_t$ Ramsey planner.

Now let's write some code to generate and plot outcomes under our three timing protocols.


```{code-cell} ipython3
def compare_ramsey_check(clq, ax):
    """
    Method to compare values of Ramsey and Check

    Here clq is an instance of ChangLQ
    """
    # Calculate check space range and bounds
    check_min, check_max = min(clq.check_space), max(clq.check_space)
    check_range = check_max - check_min
    check_LB, check_UB = check_min - 0.05 * check_range, check_max + 0.05 * check_range

    # Set axis limits
    ax.set_xlim([clq.θ_LB, clq.θ_UB])
    ax.set_ylim([check_LB, check_UB])

    # Plot J(θ) and V^check(θ)
    J_line, = ax.plot(clq.θ_space, clq.J_space, 
                      lw=2, label=r"$J(\theta)$")
    check_line, = ax.plot(clq.θ_space, clq.check_space, 
                          lw=2, label=r"$V^\check(\theta)$")

    # Mark key points
    θ_points, labels, θ_colors = compute_θs(clq)
    
    markers = [ax.scatter(θ, check_LB + 0.02 * check_range, 
                          60, marker='v', label=label, color=color)
               for θ, label, color in zip(θ_points, labels, θ_colors)]

    return J_line, check_line, markers

def plt_clqs(clqs, axes):
    line_handles, scatter_handles = {}, {}

    for ax, clq in zip(axes, clqs):
        J_line, check_line, markers = compare_ramsey_check(clq, ax)
        ax.set_title(fr'$\beta$={clq.β}, $c$={clq.c}')
        ax.tick_params(axis='x', rotation=45)

        line_handles[J_line.get_label()] = J_line
        line_handles[check_line.get_label()] = check_line
        for marker in markers:
            scatter_handles[marker.get_label()] = marker

    # Consolidate handles and labels
    line_handles = list(line_handles.values())
    scatter_handles = list(scatter_handles.values())
    line_labels = [line.get_label() for line in line_handles]
    scatter_labels = [marker.get_label() for marker in scatter_handles]

    # Create legends
    fig = plt.gcf()
    fig.legend(handles=line_handles, labels=line_labels, 
               loc='upper center', ncol=2, 
               bbox_to_anchor=(0.5, 1.1), prop={'size': 12})
    fig.legend(handles=scatter_handles, labels=scatter_labels, 
               loc='lower center', ncol=5, 
               bbox_to_anchor=(0.5, -0.1), prop={'size': 12})
    
    plt.tight_layout()
    plt.show()
```

```{code-cell} ipython3
# Compare different β values
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
β_values = [0.7, 0.85, 0.9]

clqs = [ChangLQ(β=β, c=2) for β in β_values]
plt_clqs(clqs, axes)
```

```{code-cell} ipython3
# Compare different c values
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
c_values = [1, 4, 8]

clqs = [ChangLQ(β=0.85, c=c) for c in c_values]
plt_clqs(clqs, axes)
```

The next code generates  figures that plot  policy functions for a continuation Ramsey
planner.

The left figure shows the choice of $\theta'$ chosen by a
continuation Ramsey planner who inherits $\theta$.

The right figure plots a continuation Ramsey planner's choice of
$\mu$ as a function of an inherited $\theta$.

**Request for Humphrey** Please consolidate the right figure into the left figure.  We'll just plot
$\theta_{t+1}$ as function of  $\theta_t$, $\mu_t$ as a function of $\theta_t$, and the 45degree line
all on the same graph. Something pretty will happen.  We'll have to relabel the legends so that it is clear what is being plotted. Can you please do this?

When we are done with this beautiful new graph, I'll want to move it forward in the lecture. It should go before some of the above graphs -- it sets the stage for  the dynamics that are played out in those graphs.  

Thanks!

**end of request for Humphrey**

```{code-cell} ipython3
def plot_policy_functions(clq):
    """
    Method to plot the policy functions over the relevant range of θ

    Here clq is an instance of ChangLQ
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    θ_points, labels, θ_colors = compute_θs(clq)

    ax = axes[0]
    ax.set_ylim([clq.θ_LB, clq.θ_UB])
    ax.plot(clq.θ_space, clq.θ_prime,
            label=r"$\theta'(\theta)$", lw=2,
            alpha=0.7)
    x = np.linspace(clq.θ_LB, clq.θ_UB, 5)
    ax.plot(x, x, 'k--', lw=2, alpha=0.7)
    ax.set_ylabel(r"$\theta'$", fontsize=18)

    for θ, label in zip(θ_points, labels):
        ax.scatter(θ, clq.θ_LB + 0.02 * clq.θ_range, 60, 'k', 'v')
        ax.annotate(label,
                    xy=(θ, clq.θ_LB + 0.01 * clq.θ_range),
                    xytext=(θ - 0.012 * clq.θ_range,
                            clq.θ_LB + 0.08 * clq.θ_range),
                    fontsize=13)

    ax = axes[1]
    μ_min = min(clq.μ_space)
    μ_max = max(clq.μ_space)
    μ_range = μ_max - μ_min
    ax.set_ylim([μ_min - 0.05 * μ_range, μ_max + 0.05 * μ_range])
    ax.plot(clq.θ_space, clq.μ_space, lw=2)
    ax.set_ylabel(r"$\mu(\theta)$", fontsize=18)

    for ax in axes:
        ax.set_xlabel(r"$\theta$", fontsize=18)
        ax.set_xlim([clq.θ_LB, clq.θ_UB])

    for θ, label in zip(θ_points, labels):
        ax.scatter(θ, μ_min - 0.03 * μ_range, 60, 'black', 'v')
        ax.annotate(label, xy=(θ, μ_min - 0.03 * μ_range),
                    xytext=(θ - 0.012 * clq.θ_range,
                            μ_min + 0.03 * μ_range),
                    fontsize=13)
    plt.tight_layout()
    plt.show()

plot_policy_functions(clq)
```

The following code generates a figure that plots sequences of $\mu$ and $\theta$
in the Ramsey plan and compares these to the constant levels in a MPE
and in a Ramsey plan with a government restricted to set $\mu_t$
to a constant for all $t$.

```{code-cell} ipython3
def plot_ramsey_MPE(clq, T=15):
    """
    Method to plot Ramsey plan against Markov Perfect Equilibrium

    Here clq is an instance of ChangLQ
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plots = [clq.θ_series[1, 0:T], clq.μ_series[0:T]]
    MPEs = [clq.θ_MPE, clq.μ_MPE]
    labels = [r"\theta", r"\mu"]

    axes[0].hlines(clq.θ_B, 0, T-1, 'r', label=r"$\theta^*$")

    for ax, plot, MPE, label in zip(axes, plots, MPEs, labels):
        ax.plot(plot, label=r"$" + label + "^R$")
        ax.hlines(MPE, 0, T-1, 'orange', label=r"$" + label + "^{MPE}$")
        ax.hlines(clq.μ_check, 0, T, 'g', label=r"$" + label + "^\check$")
        ax.set_xlabel(r"$t$", fontsize=14)
        ax.set_ylabel(r"$" + label + "_t$", fontsize=16)
        ax.legend(loc='upper right')
    fig.suptitle(fr'$\beta$={clq.β}, $c$={clq.c}', fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make space for the suptitle
    plt.show()
```

```{code-cell} ipython3
# Compare different β values
β_values = [0.8, 0.85, 0.9]

for β in β_values:
    clq = ChangLQ(β=β, c=2)
    plot_ramsey_MPE(clq)
```

```{code-cell} ipython3
# Compare different c values
c_values = [1, 4, 8]

for c in c_values:
    clq = ChangLQ(β=0.85, c=c)
    plot_ramsey_MPE(clq)
```


```{code-cell} ipython3
def compute_v(clq, θs):
    """
    Compute v_t and v_check for given θ values.

    Here clq is an instance of ChangLQ.
    """
    # Compute v values for corresponding θ
    v_t = -clq.P[0, 0] - 2 * clq.P[1, 0] * θs - clq.P[1, 1] * θs**2
    
    # Define the utility function
    U = lambda x: clq.α0 + clq.α1 * x - (clq.α2 / 2) * x**2
    
    # Compute v_check
    v_check = 1 / (1 - clq.β) * (U(-clq.α * clq.μ_check) 
                                 - (clq.c / 2) * clq.μ_check**2)

    return v_t, v_check

def plot_J(clq, ax, add_legend=False):
    """
    Plot v(θ) and v^check with θ markers.
    
    Here clq is an instance of ChangLQ.
    """
    # Compute J(θ) and v_check
    Jθ, v_check = compute_v(clq, clq.θ_space)
    
    # Plot J(θ)
    v_line, = ax.plot(clq.θ_space, Jθ, lw=2, 
                      label=r"$J(\theta)$", alpha=0.7)
    
    # Plot v^check as a horizontal line
    check_line = ax.axhline(y=v_check, linestyle='--', 
                            color='black', 
                            alpha=0.5, label=r"$v^\check$")
    
    # Add markers
    θ_points, labels, θ_colors = compute_θs(clq)
    
    markers = []
    for θ, label, color in zip(θ_points, labels, θ_colors):
        closest_index = np.argmin(np.abs(clq.θ_space - θ))
        marker = ax.scatter(clq.θ_space[closest_index], 
                            Jθ[closest_index], 
                            60, marker='v', 
                            label=label, color=color)
        markers.append(marker)
    
    # Set axis labels and title
    ax.set_xlabel(r"$\theta$", fontsize=18)
    ax.set_title(fr'$\beta$={clq.β}, $c$={clq.c}')
    
    # Format y-axis
    ax.tick_params(axis='x', rotation=45)
    
    if add_legend:
        handles = [v_line, check_line] + markers
        labels = [handle.get_label() for handle in handles]
        ax.legend(handles, labels, loc='upper center', ncol=7, 
               bbox_to_anchor=(1.7, 1.2), prop={'size': 14})
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for i, β in enumerate(β_values):
    clq = ChangLQ(β=β, c=2)
    plot_J(clq, axes[i], add_legend=(i==0))
plt.show()
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(16,6))
for i, c in enumerate(c_values):
    clq = ChangLQ(β=0.85, c=c)
    plot_J(clq, axes[i], add_legend=(i==0))
plt.show()
```

```{code-cell} ipython3
def plot_vt(clq, T, ax):
    """
    Plot v_t and v^check with θ markers 
    """
    
    # Define θ_ts and compute v_t
    θ_ts = clq.θ_series[1, 0:T]

    v_t, v_check = compute_v(clq, θ_ts)

    # Generate plots
    ax.plot(v_t, lw=2, label=r"$v_t$")
    ax.axhline(y=v_check, linestyle='--', 
               color='black', alpha=0.5,
               label=r"$v^\check$")
    ax.set_xlabel(r"$t$", fontsize=18)
    ax.set_title(fr'$\beta$={clq.β}, $c$={clq.c}')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=14)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, β in enumerate(β_values):
    clq = ChangLQ(β=β, c=2)
    plot_vt(clq, 10, axes[i])
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, c in enumerate(c_values):
    clq = ChangLQ(β=0.85, c=c)
    plot_vt(clq, 10, axes[i])
```

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

**Note:** A modified Ramsey plan constructed under the restriction that
$\mu_t$ must be constant over time is time consistent (see
$\check \mu$ and $\check \theta$ in the above graphs).

### Meaning of Time Inconsistency

In settings in which governments actually choose sequentially, many economists
regard a time inconsistent plan as implausible because of the incentives to
deviate that are presented  along the plan.

A way to summarize this *defect* in a Ramsey plan is to say that it
is not credible because there  endure incentives for policymakers
to deviate from it.

For that reason, the Markov perfect equilibrium concept attracts many
economists.

* A Markov perfect equilibrium plan is constructed to insure that government policymakers who choose sequentially do not want to deviate from it.

The *no incentive to deviate from the plan* property is what makes the Markov perfect equilibrium concept attractive.

### Ramsey Plans Strike Back

Research by Abreu {cite}`Abreu`,  Chari and Kehoe {cite}`chari1990sustainable`
{cite}`stokey1989reputation`, and Stokey {cite}`Stokey1991` discovered conditions under which a Ramsey plan can be rescued from the complaint that it is not credible.

They  accomplished this by expanding the
description of a plan to include expectations about **adverse consequences** of deviating from
it that can serve to deter deviations.

We turn to such theories of **sustainable plans** next.

## A Fourth Model of Government Decision Making

This is a model in which

- the government chooses $\{\mu_t\}_{t=0}^\infty$ not once and
  for all at $t=0$ but chooses to set $\mu_t$ at time $t$, not before.
- the representative agent's forecasts of
  $\{\mu_{t+j+1}, \theta_{t+j+1}\}_{j=0}^\infty$ respond to
  whether the government at $t$ **confirms** or **disappoints**
  their forecasts of $\mu_t$ brought into period $t$ from
  period $t-1$.
- the government at each time $t$ understands how the representative agent's
  forecasts will respond to its choice of $\mu_t$.
- at each $t$, the government chooses $\mu_t$ to maximize
  a continuation discounted utility.

### A Theory of Government Decision Making

$\vec \mu$ is chosen by a sequence of government
decision makers, one for each $t \geq 0$.

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
- If the government at $t$ **confirms** the representative agent's
  expectations by setting $\mu_t = \tilde \mu_t$ at time
  $t$, private agents expect the continuation government policy
  $\{\tilde \mu_{t+j+1}\}_{j=0}^\infty$ and therefore bring
  expectation $\tilde \theta_{t+1}$ into period $t+1$.
- If the government at $t$ **disappoints** private agents by setting
  $\mu_t \neq \tilde \mu_t$, private agents expect
  $\{\mu^A_j\}_{j=0}^\infty$ as the
  continuation policy for $t+1$, i.e.,
  $\{\mu_{t+j+1}\} = \{\mu_j^A \}_{j=0}^\infty$ and therefore
  expect an associated $\theta_0^A$ for $t+1$. Here $\vec \mu^A = \{\mu_j^A \}_{j=0}^\infty$ is
  an alternative government plan to be described below.

### Temptation to Deviate from Plan

The government's one-period return function $s(\theta,\mu)$
described in equation {eq}`eq_old6` above has the property that for all
$\theta$

$$
- s(\theta, 0 ) \geq  - s(\theta, \mu) \quad
$$

This inequality implies that whenever the policy calls for the
government to set $\mu \neq 0$, the government could raise its
one-period payoff by setting $\mu =0$.

Disappointing private sector expectations in that way would increase the
government's **current** payoff but would have adverse consequences for
**subsequent** government payoffs because the private sector would alter
its expectations about future settings of $\mu$.

The **temporary** gain constitutes the government's temptation to
deviate from a plan.

If the government at $t$ is to resist the temptation to raise its
current payoff, it is only because it forecasts adverse  consequences that
its setting of $\mu_t$ would bring for continuation  government payoffs via  alterations  in the private sector's expectations.

## Sustainable or Credible Plan

We call a plan $\vec \mu$ **sustainable** or **credible** if at
each $t \geq 0$ the government chooses to confirm private
agents' prior expectation of its setting for $\mu_t$.

The government will choose to confirm prior expectations only if the
long-term **loss** from disappointing private sector expectations --
coming from the government's understanding of the way the private sector
adjusts its  expectations in response to having its prior
expectations at $t$ disappointed -- outweigh the short-term
**gain** from disappointing those expectations.

The theory of sustainable or credible plans assumes throughout that private sector
expectations about what future governments will do are based on the
assumption that governments at times $t \geq 0$ always act to
maximize the continuation discounted utilities that describe those
governments' purposes.

This aspect of the theory means that credible plans always come in **pairs**:

- a credible (continuation) plan to be followed if the government at
  $t$ **confirms** private sector expectations
- a credible plan to be followed if the government at $t$
  **disappoints** private sector expectations

That credible plans come in pairs threaten to bring an explosion of plans to keep track of

* each credible plan itself consists of two credible plans
* therefore, the number of plans underlying one plan is unbounded

But Dilip Abreu showed how to render manageable the number of plans that must be kept track of.

The key is an  object called a **self-enforcing** plan.

### Abreu's Self-Enforcing Plan

A plan $\vec \mu^A$ (here the superscipt $A$ is for Abreu) is said to be **self-enforcing** if

- the consequence of disappointing the representative agent's expectations at time
  $j$ is to **restart**  plan $\vec \mu^A$  at time $j+1$
- the consequence of restarting the plan is sufficiently adverse that it forever deters all
  deviations from the plan

More precisely, a government plan $\vec \mu^A$ with equilibrium inflation sequence $\vec \theta^A$ is
**self-enforcing** if

```{math}
:label: eq_old10

\begin{aligned}
v_j^A & = - s(\theta^A_j, \mu^A_j) + \beta v_{j+1}^A \\
& \geq - s(\theta^A_j, 0 ) + \beta v_0^A \equiv v_j^{A,D}, \quad j \geq 0
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
v_j & = - s( \theta_j, \mu_j) + \beta  v_{j+1} \\
& \geq  -s( \theta_j, 0) + \beta v_0^A \quad \forall j \geq 0
\end{aligned}
```

For this condition to be satisfied it is necessary and sufficient that

$$
-s( \theta_j, 0) - ( - s( \theta_j, \mu_j) )  <  \beta ( v_{j+1} - v_0^A )
$$

The left side of the above inequality is the government's **gain** from deviating from the plan, while the right side is the government's **loss** from deviating
from the plan.

A government never wants to deviate from a credible plan.

Abreu taught us that  key step in constructing a credible plan is first constructing a
self-enforcing plan that has a low time $0$ value.

The idea is to use the self-enforcing plan as a continuation plan whenever
the government's choice at time $t$ fails to confirm private
agents' expectation.

We shall use a construction featured in Abreu ({cite}`Abreu`) to construct a
self-enforcing plan with low time $0$ value.

### Abreu Carrot-Stick Plan

Abreu ({cite}`Abreu`) invented a way to create a self-enforcing plan with a low
initial value.

Imitating his idea, we can construct a self-enforcing plan
$\vec \mu$ with a low time $0$ value to the government by
insisting that future government decision makers set $\mu_t$ to a value yielding low
one-period utilities to the household for a long time, after which
government  decisions thereafter  yield high one-period utilities.

- Low one-period utilities early are a **stick**
- High one-period utilities later are a **carrot**

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
v^A_0 =  - \sum_{t=0}^{T_A-1} \beta^t s(\theta_t^A,\mu_t^A) +\beta^{T_A} J(\theta^R_0)
$$

For an appropriate $T_A$, this plan can be verified to be self-enforcing and therefore credible.

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
def abreu_plan(clq, T=1000, T_A=10, μ_bar=0.1, T_Plot=20):

    # Append Ramsey μ series to stick μ series
    clq.μ_A = np.append(np.full(T_A, μ_bar), clq.μ_series[:-T_A])

    # Calculate implied stick θ series
    clq.θ_A = np.zeros(T)
    discount = np.zeros(T)
    for t in range(T):
        discount[t] = (clq.α / (1 + clq.α))**t
    for t in range(T):
        length = clq.μ_A[t:].shape[0]
        clq.θ_A[t] = 1 / (clq.α + 1) * sum(clq.μ_A[t:] * discount[0:length])

    # Calculate utility of stick plan
    U_A = np.zeros(T)
    for t in range(T):
        U_A[t] = clq.β**t \
                 * (clq.α0 + clq.α1 * (-clq.θ_A[t])
                 - clq.α2 / 2 * (-clq.θ_A[t])**2 
                 - clq.c * clq.μ_A[t]**2)

    clq.V_A = np.zeros(T)
    for t in range(T):
        clq.V_A[t] = sum(U_A[t:] / clq.β**t)

    # Make sure Abreu plan is self-enforcing
    clq.V_dev = np.zeros(T_Plot)
    for t in range(T_Plot):
        clq.V_dev[t] = (clq.α0 + clq.α1 * (-clq.θ_A[t])
                        - clq.α2 / 2 * (-clq.θ_A[t])**2) \
                        + clq.β * clq.V_A[0]

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    axes[2].plot(clq.V_dev[0:T_Plot], label="$V^{A, D}_t$", c="orange")

    plots = [clq.θ_A, clq.μ_A, clq.V_A]
    labels = [r"$\theta_t^A$", r"$\mu_t^A$", r"$V^A_t$"]

    for plot, ax, label in zip(plots, axes, labels):
        ax.plot(plot[0:T_Plot], label=label)
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
v_t^{A,D} = -s( \theta_j, 0) + \beta v_0^A
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
v^R_t \geq - s(\theta^R_t,0) + \beta v^A_0 , \quad \forall t \geq 0
$$

```{code-cell} ipython3
def check_ramsey(clq, T=1000):
    # Make sure Ramsey plan is sustainable
    R_dev = np.zeros(T)
    for t in range(T):
        R_dev[t] = (clq.α0 + clq.α1 * (-clq.θ_series[1, t])
                    - clq.α2 / 2 * (-clq.θ_series[1, t])**2) \
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

## Whose Credible Plan is it?

A credible government plan $\vec \mu$ plays multiple roles.

* It is a sequence of actions chosen by the government.
* It is a sequence of the representative agent's forecasts of government actions.

Thus, $\vec \mu$ is both a government policy and a collection of the representative agent's forecasts of  government policy.

Does the government *choose*  policy actions or does it simply *confirm* prior private sector forecasts of those actions?

An argument in favor of the *government chooses* interpretation comes from noting that the theory of credible plans builds in a theory that the government each period chooses
the action that it wants.

An argument in favor of the *simply confirm* interpretation is gathered from staring at the key inequality {eq}`eq_old100a` that defines a credible policy.

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
clq.J_check
```

```{code-cell} ipython3
clq.J_MPE
```

We have also computed **credible plans** for a government or sequence
of governments that choose sequentially.

These include

- a **self-enforcing** plan that gives a low initial value $v_0$.
- a better plan -- possibly one that attains values associated with
  Ramsey plan -- that is not self-enforcing.

## Note on Dynamic Programming Squared

The theory deployed  in this lecture is an application of what we  nickname **dynamic programming squared**.

The nickname refers to the feature that a value satisfying one Bellman equation appears as an argument in a second Bellman equation.

Thus, our models have involved two Bellman equations:

- equation {eq}`eq_old1` expresses how $\theta_t$ depends on $\mu_t$
  and $\theta_{t+1}$
- equation {eq}`eq_old4` expresses how value $v_t$ depends on
  $(\mu_t, \theta_t)$ and $v_{t+1}$

A value $\theta$ from one Bellman equation appears as an argument of a second Bellman equation for another value $v$.
