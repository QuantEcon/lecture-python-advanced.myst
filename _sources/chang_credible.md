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

(chang_credible)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Credible Government Policies in a Model of Chang

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install polytope
```

## Overview

Some of the material in this lecture and {doc}`competitive equilibria in the Chang model <chang_ramsey>`
can be viewed as more sophisticated and complete treatments of the topics discussed in
{doc}`Ramsey plans, time inconsistency, sustainable plans <calvo>`.

This lecture assumes almost  the same economic environment analyzed in
{doc}`competitive equilibria in the Chang model <chang_ramsey>`.

The only change  --  and it is a substantial one -- is the timing protocol for making government decisions.

In  {doc}`competitive equilibria in the Chang model <chang_ramsey>`, a *Ramsey planner*
chose a comprehensive government policy once-and-for-all at time $0$.

Now in this lecture, there is no time $0$ Ramsey planner.

Instead there is a sequence of government decision-makers, one for each $t$.

The time $t$ government decision-maker choose time $t$ government
actions after forecasting what future governments will do.

We use the notion of a *sustainable plan* proposed in {cite}`chari1990sustainable`,
also referred to as a *credible public policy* in {cite}`stokey1989reputation`.

Technically, this lecture starts where lecture
{doc}`competitive equilibria in the Chang model <chang_ramsey>` on Ramsey plans
within the Chang {cite}`chang1998credible` model stopped.

That lecture presents recursive representations of  *competitive equilibria* and a *Ramsey plan* for a
version of a model of Calvo {cite}`Calvo1978` that Chang used to analyze and illustrate these concepts.

We used two operators to characterize competitive equilibria and a Ramsey plan,
respectively.

In this lecture, we define a *credible public policy* or *sustainable plan*.

Starting from a large enough initial set $Z_0$, we use iterations on
Chang's set-to-set  operator $\tilde D(Z)$ to
compute a set of values associated with sustainable plans.

Chang's operator $\tilde D(Z)$ is closely connected with the operator
$D(Z)$ introduced in lecture {doc}`competitive equilibria in the Chang model <chang_ramsey>`.

* $\tilde D(Z)$ incorporates all of the restrictions imposed in
  constructing the operator $D(Z)$, but $\ldots$.
* It adds some additional restrictions
    * these additional restrictions incorporate the idea that a plan must be *sustainable*.
    * *sustainable* means that the government wants to implement it at all times after all histories.

Let's start with some standard imports:

```{code-cell} ipython
import numpy as np
import polytope
import matplotlib.pyplot as plt
```

## The Setting

We begin by reviewing the set up deployed in  {doc}`competitive equilibria in the Chang model <chang_ramsey>`.

Chang's  model, adopted from Calvo, is designed to focus on the intertemporal trade-offs between
the welfare benefits of deflation and the welfare costs associated with
the high tax collections required to retire money at a rate that
delivers deflation.

A benevolent time $0$ government can promote
utility generating increases in real balances only by imposing an
infinite sequence of sufficiently large distorting tax collections.

To promote the welfare increasing effects of high real balances, the
government wants to induce  *gradual deflation*.

We start by reviewing notation.

For a sequence of scalars
$\vec z \equiv \{z_t\}_{t=0}^\infty$, let
$\vec z^t = (z_0,  \ldots , z_t)$,
$\vec z_t = (z_t, z_{t+1}, \ldots )$.

An infinitely lived
representative agent and an infinitely lived government exist at dates
$t = 0, 1, \ldots$.

The objects in play are

* an initial quantity $M_{-1}$ of nominal money holdings
* a sequence of inverse money growth rates $\vec h$ and an associated sequence of nominal money holdings $\vec M$
* a sequence of values of money $\vec q$
* a sequence of real money holdings $\vec m$
* a sequence of total tax collections $\vec x$
* a sequence of per capita rates of consumption $\vec c$
* a sequence of per capita incomes $\vec y$

A benevolent government chooses sequences
$(\vec M, \vec h, \vec x)$ subject to a sequence of budget
constraints and other constraints imposed by competitive equilibrium.

Given tax collection and price of money sequences, a representative household chooses
sequences $(\vec c, \vec m)$ of consumption and real balances.

In competitive equilibrium, the price of money sequence $\vec q$ clears
markets, thereby reconciling  decisions of the government and the
representative household.

### The Household’s Problem

A representative household faces a nonnegative value of money sequence
$\vec q$ and sequences $\vec y, \vec x$ of income and total
tax collections, respectively.

The household chooses nonnegative
sequences $\vec c, \vec M$ of consumption and nominal balances,
respectively, to maximize

```{math}
:label: eqn_chang1

\sum_{t=0}^\infty \beta^t \left[ u(c_t) + v(q_t M_t ) \right]
```

subject to

```{math}
:label: eqn_chang2

q_t M_t  \leq y_t + q_t M_{t-1} - c_t - x_t
```

and

```{math}
:label: eqn_chang3

q_t M_t  \leq \bar m
```

Here $q_t$ is the reciprocal of the price level at $t$,
also known as the *value of money*.

Chang {cite}`chang1998credible` assumes that

* $u: \mathbb{R}_+ \rightarrow \mathbb{R}$ is twice continuously differentiable, strictly concave, and strictly increasing;
* $v: \mathbb{R}_+ \rightarrow \mathbb{R}$ is twice continuously differentiable and strictly concave;
* $u'(c)_{c \rightarrow 0}  = \lim_{m \rightarrow 0} v'(m) = +\infty$;
* there is a finite level $m= m^f$ such that $v'(m^f) =0$

Real balances carried out of a period equal $m_t = q_t M_t$.

Inequality {eq}`eqn_chang2` is the household’s time $t$ budget constraint.

It tells how real balances $q_t M_t$ carried out of period $t$ depend
on income, consumption, taxes, and real balances $q_t M_{t-1}$
carried into the period.

Equation {eq}`eqn_chang3` imposes an exogenous upper bound
$\bar m$ on the choice of real balances, where
$\bar m \geq m^f$.

### Government

The government chooses a sequence of inverse money growth rates with
time $t$ component
$h_t \equiv {M_{t-1}\over M_t} \in \Pi \equiv
[ \underline \pi, \overline \pi]$, where
$0 < \underline \pi < 1 < { 1 \over \beta } \leq \overline \pi$.

The government faces a sequence of budget constraints with time
$t$ component

$$
-x_t = q_t (M_t - M_{t-1})
$$

which, by using the definitions of $m_t$ and $h_t$, can also
be expressed as

```{math}
:label: eqn_chang2a

-x_t = m_t (1-h_t)
```

The  restrictions $m_t \in [0, \bar m]$ and $h_t \in \Pi$ evidently
imply that $x_t \in X \equiv [(\underline  \pi -1)\bar m, (\overline \pi -1) \bar m]$.

We define the set $E \equiv [0,\bar m] \times \Pi \times X$, so that we
require that $(m, h, x) \in E$.

To represent the idea that taxes are distorting, Chang makes the following
assumption about outcomes for per capita output:

```{math}
:label: eqn_chang3a

y_t = f(x_t)
```

where $f: \mathbb{R}\rightarrow \mathbb{R}$ satisfies $f(x)  > 0$,
is twice continuously differentiable, $f''(x) < 0$, and
$f(x) = f(-x)$ for all $x \in
\mathbb{R}$, so that subsidies and taxes are equally distorting.

The purpose is not to model the causes of tax distortions in any detail but simply to summarize
the *outcome* of those distortions via the function $f(x)$.

A key part of the specification is that tax distortions are increasing in the
absolute value of tax revenues.

The government chooses a competitive equilibrium that
maximizes {eq}`eqn_chang1`.

### Within-period Timing Protocol

For the results in this lecture, the *timing* of actions within a period is
important because of the incentives that it activates.

Chang assumed the following within-period timing of decisions:

* first, the government chooses $h_t$ and $x_t$;
* then given $\vec q$ and its expectations about future values of
  $x$ and $y$’s, the household chooses $M_t$ and therefore
  $m_t$ because $m_t = q_t M_t$;
* then output $y_t = f(x_t)$ is realized;
* finally $c_t = y_t$

This within-period timing confronts the government with
choices framed by how the private sector wants to respond when the
government takes time $t$ actions that differ from what the
private sector had expected.

This timing will shape the incentives confronting the government at each
history that are to be incorporated in the construction of the $\tilde D$
operator below.

### Household’s Problem

Given $M_{-1}$ and $\{q_t\}_{t=0}^\infty$, the household’s problem is

$$
\begin{aligned}
\mathcal{L} & = \max_{\vec c, \vec M}
\min_{\vec \lambda, \vec \mu} \sum_{t=0}^\infty \beta^t
\bigl\{ u(c_t) + v(M_t q_t) +
\lambda_t [ y_t - c_t - x_t + q_t M_{t-1} - q_t M_t ]  \\
& \quad \quad \quad  + \mu_t [\bar m - q_t  M_t] \bigr\}
\end{aligned}
$$

First-order conditions with respect to $c_t$ and $M_t$, respectively, are

$$
\begin{aligned}
u'(c_t) & = \lambda_t \\
q_t [ u'(c_t) - v'(M_t q_t) ] & \leq \beta u'(c_{t+1})
q_{t+1} , \quad = \ {\rm if} \ M_t q_t < \bar m
\end{aligned}
$$

Using $h_t = {M_{t-1}\over M_t}$ and $q_t = {m_t \over M_t}$ in
these first-order conditions and rearranging implies

```{math}
:label: eqn_chang4

m_t [u'(c_t) - v'(m_t) ] \leq \beta u'(f(x_{t+1})) m_{t+1} h_{t+1},
\quad = \text{ if } m_t < \bar m
```

Define the following key variable

```{math}
:label: eqn_chang5

\theta_{t+1} \equiv u'(f(x_{t+1})) m_{t+1} h_{t+1}
```

This is real money balances at time $t+1$ measured in units of marginal
utility, which Chang refers to as ‘the marginal utility of real balances’.

From the standpoint of the household at time $t$, equation {eq}`eqn_chang5`
shows that $\theta_{t+1}$ intermediates the influences of
$(\vec x_{t+1}, \vec m_{t+1})$ on the household’s choice of real
balances $m_t$.

By "intermediates" we mean that the future paths
$(\vec x_{t+1}, \vec m_{t+1})$ influence $m_t$ entirely through
their effects on the scalar $\theta_{t+1}$.

The observation that the one dimensional promised marginal utility of real
balances $\theta_{t+1}$ functions in this way is an important step
in constructing a class of competitive equilibria that have a recursive representation.

A closely related observation pervaded the analysis of Stackelberg plans in
{doc}`dynamic Stackelberg problems <dyn_stack>` and {doc}`the Calvo model <calvo>`.

### Competitive Equilibrium

**Definition:**

* A *government policy* is a pair of sequences $(\vec h,\vec x)$ where $h_t \in \Pi  \ \forall t \geq 0$.
* A *price system* is a non-negative value of money sequence $\vec q$.
* An *allocation* is a  triple of non-negative sequences $(\vec c, \vec m, \vec y)$.

It is required that time $t$ components $(m_t, x_t, h_t) \in E$.

**Definition:**

Given $M_{-1}$, a government policy $(\vec h, \vec x)$, price system $\vec q$, and allocation
$(\vec c, \vec m, \vec y)$ are said to be a *competitive equilibrium* if

* $m_t = q_t M_t$ and $y_t = f(x_t)$.
* The government budget constraint is satisfied.
* Given $\vec q, \vec x, \vec y$, $(\vec c, \vec m)$ solves the household’s problem.

### A Credible Government Policy

Chang works with

**A credible government policy with a recursive representation**

* Here there is no time $0$ Ramsey planner.
* Instead there is a sequence of governments, one for each $t$, that
  choose time $t$ government actions after forecasting what future governments will do.
* Let $w=\sum_{t=0}^\infty \beta^t \left[ u(c_t) + v(q_t M_t ) \right]$
  be a value associated with a particular competitive equilibrium.
* A recursive representation of a credible government policy is a pair of
  initial conditions $(w_0, \theta_0)$ and a five-tuple of functions

  $$
  h(w_t, \theta_t), m(h_t, w_t, \theta_t), x(h_t, w_t, \theta_t), \chi(h_t, w_t, \theta_t),\Psi(h_t, w_t, \theta_t)
  $$
  mapping $w_t,\theta_t$ and in some cases $h_t$ into
  $\hat h_t, m_t, x_t, w_{t+1}$, and $\theta_{t+1}$, respectively.
* Starting from an initial condition $(w_0, \theta_0)$, a credible
  government policy can be constructed by iterating on these functions in
  the following order that respects the within-period timing:

  ```{math}
  :label: chang501

  \begin{aligned}
  \hat h_t & = h(w_t,\theta_t) \\
  m_t & = m(h_t, w_t,\theta_t) \\
  x_t & = x(h_t, w_t,\theta_t) \\
  w_{t+1} & = \chi(h_t, w_t,\theta_t)  \\
  \theta_{t+1}  & = \Psi(h_t, w_t,\theta_t)
  \end{aligned}
  ```

* Here it is to be understood that $\hat h_t$ is the action that the
  government policy instructs the government to take, while $h_t$
  possibly not equal to $\hat h_t$ is some other action that the
  government is free to take at time $t$.

The plan is *credible* if it is in the time $t$ government’s interest to
execute it.

Credibility requires that the plan be such that for all possible choices of
$h_t$ that are consistent with competitive equilibria,

$$
\begin{split} & u(f(x(\hat h_t, w_t,\theta_t))) + v(m(\hat h_t, w_t,\theta_t))  + \beta \chi(\hat h_t, w_t,\theta_t) \\
&  \geq
u(f(x( h_t, w_t,\theta_t))) + v(m(h_t, w_t,\theta_t)) + \beta \chi(h_t, w_t,\theta_t) \end{split}
$$

so that at each instance and circumstance of choice, a government attains a
weakly higher lifetime utility with continuation value
$w_{t+1}=\Psi(h_t, w_t,\theta_t)$ by adhering to the plan and
confirming the associated time $t$ action $\hat h_t$ that
the public had expected earlier.

Please note the subtle change in arguments of the functions used to represent
a competitive equilibrium and a Ramsey plan, on the one hand, and a credible
government plan, on the other hand.

The extra arguments appearing in the functions used to represent a credible plan
come from allowing the government to contemplate disappointing the private sector’s
expectation about its time $t$ choice $\hat h_t$.

A credible plan induces the government to confirm the private sector’s expectation.

The recursive representation of the plan uses the evolution of continuation
values to deter the government from wanting to disappoint the private sector’s
expectations.

Technically, a Ramsey plan and a credible plan  both incorporate history dependence.

For a Ramsey plan, this is encoded in the dynamics of the state variable
$\theta_t$, a promised marginal utility that the Ramsey plan delivers to
the private sector.

For a credible government plan, we the two-dimensional state vector
$(w_t, \theta_t)$ encodes  history dependence.

### Sustainable Plans

A government strategy $\sigma$ and an allocation rule
$\alpha$ are said to constitute a *sustainable plan* (SP) if.

1. $\sigma$ is admissible.
1. Given $\sigma$, $\alpha$ is competitive.
1. After any history $\vec h^{t-1}$, the continuation of $\sigma$
   is optimal for the government; i.e., the sequence $\vec h_t$ induced
   by $\sigma$ after $\vec h^{t-1}$ maximizes over $CE_\pi$
   given $\alpha$.

Given any history $\vec h^{t-1}$, the continuation of a sustainable plan is a
sustainable plan.

Let $\Theta = \{ (\vec m, \vec x, \vec h) \in CE : \text{there is an SP whose outcome is} (\vec m, \vec x, \vec h) \}$.

Sustainable outcomes are elements of $\Theta$.

Now consider the space

$$
S = \Bigl\{ (w,\theta) : \text{there is a sustainable outcome }
    (\vec m, \vec x, \vec h) \in \Theta
$$

with value

$$
w = \sum_{t=0}^\infty \beta^t [u(f(x_t)) + v(m_t)]  \text{ and such that }
     u'(f(x_0)) (m_0 + x_0) = \theta \Bigr\}
$$

The space $S$ is a compact subset of $W \times \Omega$
where $W = [\underline w, \overline w]$ is the space of values
associated with sustainable plans. Here $\underline w$ and
$\overline w$ are finite bounds on the set of values.

Because there is at least one sustainable plan, $S$ is nonempty.

Now recall the within-period timing protocol, which we can depict
$(h,x) \rightarrow m=q M \rightarrow y = c$.

With this timing protocol in mind, the time $0$ component of an SP has the
following components:

1. A period $0$ action $\hat h \in \Pi$ that the public
   expects the government to take, together with subsequent within-period
   consequences $m(\hat h), x(\hat h)$ when the government acts as
   expected.
1. For any first-period action $h \neq \hat h$ with
   $h \in CE_\pi^0$, a pair of within-period consequences
   $m(h), x(h)$ when the government does not act as the public had
   expected.
1. For every $h \in \Pi$, a pair
   $(w'(h), \theta'(h))\in S$ to carry into next period.

These components must be such that it is optimal for the government to
choose $\hat h$ as expected; and for every possible
$h \in \Pi$, the government budget constraint and the household’s
Euler equation must hold with continuation $\theta$ being
$\theta'(h)$.

Given the timing protocol within the model, the representative
household’s response to a government deviation to $h \neq \hat h$
from a prescribed $\hat h$ consists of a first-period action
$m(h)$ and associated subsequent actions, together with future
equilibrium prices, captured by $(w'(h), \theta'(h))$.

At this point, Chang introduces an idea in the spirit of Abreu, Pearce, and Stacchetti {cite}`APS1990`.

Let $Z$ be a nonempty subset of $W \times \Omega$.

Think of using pairs $(w', \theta')$ drawn from $Z$ as candidate
continuation value, promised marginal utility pairs.

Define the following operator:

```{math}
:label: chang_operator

\begin{aligned}
\tilde D(Z) = \Bigl\{
(w,\theta): \text{there is } \hat h \in CE_\pi^0 \text{ and for each } h \in CE_\pi^0 \\
\text{ a four-tuple } (m(h), x(h), w'(h), \theta'(h)) \in [0,\bar m] \times X \times Z
\end{aligned}
```

such that

```{math}
:label: eqn_chang12

w = u(f(x(\hat h)))+ v(m(\hat h)) + \beta w'(\hat h)
```

```{math}
:label: eqn_chang13

\theta = u'(f(x(\hat h))) ( m(\hat h) + x(\hat h))
```

and for all $h \in CE_\pi^0$

```{math}
:label: eqn_chang14

w \geq u(f(x(h))) + v(m(h)) + \beta w'(h)
```

```{math}
:label: eqn_chang_15

x(h) = m(h) (h-1)
```

and

```{math}
:label: eqn_chang16

m(h) (u'(f(x(h))) - v'(m(h))) \leq \beta \theta'(h)
```

$$
\quad \quad \ \text{ with equality if } m(h) < \bar m  \Bigr\}
$$

This operator adds the key incentive constraint to the conditions that
had defined the earlier $D(Z)$ operator defined in  {doc}`competitive equilibria in the Chang model <chang_ramsey>`.

Condition {eq}`eqn_chang14` requires that the plan deter the government from wanting to
take one-shot deviations when candidate continuation values are drawn
from $Z$.

**Proposition:**

1. If $Z \subset \tilde D(Z)$, then $\tilde D(Z) \subset S$ (‘self-generation’).
1. $S = \tilde D(S)$ (‘factorization’).

**Proposition:**.

1. Monotonicity of $\tilde D$: $Z \subset Z'$ implies $\tilde D(Z) \subset \tilde D(Z')$.
1. $Z$ compact implies that $\tilde D(Z)$ is compact.

Chang establishes that $S$ is compact and that therefore there
exists a highest value SP and a lowest value SP.

Further, the preceding structure allows Chang to compute $S$ by iterating to convergence
on $\tilde D$ provided that one begins with a sufficiently large
initial set $Z_0$.

This structure delivers the following recursive representation of a
sustainable outcome:

1. choose an initial $(w_0, \theta_0) \in S$;
1. generate a sustainable outcome recursively by iterating on {eq}`chang501`, which we repeat here for convenience:

   $$
   \begin{aligned}
   \hat h_t & = h(w_t,\theta_t) \\
   m_t & = m(h_t, w_t,\theta_t) \\
   x_t & = x(h_t, w_t,\theta_t) \\
   w_{t+1} & = \chi(h_t, w_t,\theta_t)  \\
   \theta_{t+1}  & = \Psi(h_t, w_t,\theta_t)
   \end{aligned}
   $$

## Calculating the Set of Sustainable Promise-Value Pairs

Above we defined the $\tilde D(Z)$ operator as {eq}`chang_operator`.

Chang (1998) provides a method for dealing with the final three
constraints.

These incentive constraints ensure that the government wants to choose
$\hat h$ as the private sector had expected it to.

Chang's simplification starts from the idea that, when considering
whether or not to confirm the private sector's expectation, the
government only needs to consider the payoff of the *best* possible
deviation.

Equally, to provide incentives to the government, we only need to
consider the harshest possible punishment.

Let $h$ denote some possible deviation. Chang defines:

$$
P(h;Z) = \min u(f(x)) + v(m) + \beta w'
$$

where the minimization is subject to

$$
x = m(h-1)
$$

$$
m(h)(u'(f(x(h))) + v'(m(h))) \leq \beta \theta'(h) \text{ (with equality if } m(h) < \bar m) \}
$$

$$
(m,x,w',\theta') \in [0,\bar m] \times X \times Z
$$

For a given deviation $h$, this problem finds the worst possible
sustainable value.

We then define:

$$
BR(Z) = \max P(h;Z) \text{ subject to } h \in CE^0_\pi
$$

$BR(Z)$ is the value of the government's most tempting deviation.

With this in hand, we can define a new operator $E(Z)$ that is
equivalent to the $\tilde D(Z)$ operator but simpler to
implement:

$$
E(Z) = \Bigl\{ (w,\theta): \exists  h \in CE^0_\pi \text{ and } (m(h),x(h),w'(h),\theta'(h)) \in [0,\bar m] \times X \times Z
$$

such that

$$
w = u(f(x(h))) + v(m(h)) + \beta w'(h)
$$

$$
\theta = u'(f(x(h)))(m(h) + x(h))
$$

$$
x(h) = m(h)(h-1)
$$

$$
m(h)(u'(f(x(h))) - v'(m(h))) \leq \beta \theta'(h) \text{ (with equality if } m(h) < \bar m)
$$

and

$$
w \geq BR(Z) \Bigr\}
$$

Aside from the final incentive constraint, this is the same as the
operator in  {doc}`competitive equilibria in the Chang model <chang_ramsey>`.

Consequently, to implement this operator we just need to add one step to
our *outer hyperplane approximation algorithm* :

1. Initialize subgradients, $H$, and hyperplane levels,
   $C_0$.
1. Given a set of subgradients, $H$, and hyperplane levels,
   $C_t$, calculate $BR(S_t)$.
1. Given $H$, $C_t$, and $BR(S_t)$, for each
   subgradient $h_i \in H$:
    - Solve a linear program (described below) for each action in the
      action space.
    - Find the maximum and update the corresponding hyperplane level,
      $C_{i,t+1}$.
1. If $|C_{t+1}-C_t| > \epsilon$, return to 2.

**Step 1** simply creates a large initial set $S_0$.

Given some set $S_t$, **Step 2** then constructs the value
$BR(S_t)$.

To do this, we solve the following problem for each point in the action
space $(m_j,h_j)$:

$$
\min_{[w',\theta']} u(f(x_j)) + v(m_j) + \beta w'
$$

subject to

$$
H \cdot (w',\theta') \leq C_t
$$

$$
x_j = m_j(h_j-1)
$$

$$
m_j(u'(f(x_j)) - v'(m_j)) \leq \beta \theta'\hspace{2mm} (= \text{if } m_j < \bar m)
$$

This gives us a matrix of possible values, corresponding to each point
in the action space.

To find $BR(Z)$, we minimize over the $m$ dimension and
maximize over the $h$ dimension.

**Step 3** then constructs the set $S_{t+1} = E(S_t)$. The linear
program in Step 3 is designed to construct a set $S_{t+1}$ that is
as large as possible while satisfying the constraints of the
$E(S)$ operator.

To do this, for each subgradient $h_i$, and for each point in the
action space $(m_j,h_j)$, we solve the following problem:

$$
\max_{[w',\theta']} h_i \cdot (w,\theta)
$$

subject to

$$
H \cdot (w',\theta') \leq C_t
$$

$$
w = u(f(x_j)) + v(m_j) + \beta w'
$$

$$
\theta = u'(f(x_j))(m_j + x_j)
$$

$$
x_j = m_j(h_j-1)
$$

$$
m_j(u'(f(x_j)) - v'(m_j)) \leq \beta \theta'\hspace{2mm} (= \text{if } m_j < \bar m)
$$

$$
w \geq BR(Z)
$$

This problem maximizes the hyperplane level for a given set of actions.

The second part of Step 3 then finds the maximum possible hyperplane
level across the action space.

The algorithm constructs a sequence of progressively smaller sets $S_{t+1} \subset S_t \subset S_{t-1} \cdots
\subset S_0$.

**Step 4** ends the algorithm when the difference between these sets is
small enough.

We have created a Python class that solves the model assuming the
following functional forms:

$$
u(c) = log(c)
$$

$$
v(m) = \frac{1}{500}(m \bar m - 0.5m^2)^{0.5}
$$

$$
f(x) = 180 - (0.4x)^2
$$

The remaining parameters $\{\beta, \bar m, \underline h, \bar h\}$
are then variables to be specified for an instance of the Chang class.

Below we use the class to solve the model and plot the resulting
equilibrium set, once with $\beta = 0.3$ and once with
$\beta = 0.8$. We also plot the (larger) competitive equilibrium
sets, which we described in  {doc}`competitive equilibria in the Chang model <chang_ramsey>`.

(We have set the number of subgradients to 10 in order to speed up the
code for now. We can increase accuracy by increasing the number of subgradients)

The following code computes sustainable plans

```{code-cell} python3
---
load: _static/lecture_specific/chang_credible/changecon.py
tags: [collapse-20]
---
```

### Comparison of Sets

The set of $(w, \theta)$ associated with  sustainable plans is  smaller than the set of $(w, \theta)$
pairs associated with competitive equilibria, since the additional
constraints associated with sustainability must also be satisfied.

Let's compute two examples, one with a low $\beta$, another with a higher $\beta$

```{code-cell} python3
ch1 = ChangModel(β=0.3, mbar=30, h_min=0.9, h_max=2, n_h=8, n_m=35, N_g=10)
```

```{code-cell} python3
ch1.solve_sustainable()
```

The following plot shows both the set of $w,\theta$ pairs associated with competitive equilibria (in red)
and the smaller set of $w,\theta$ pairs associated with  sustainable plans (in blue).

```{code-cell} python3
def plot_equilibria(ChangModel):
    """
    Method to plot both equilibrium sets
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.set_xlabel('w', fontsize=16)
    ax.set_ylabel(r"$\theta$", fontsize=18)

    poly_S = polytope.Polytope(ChangModel.H, ChangModel.c1_s)
    poly_C = polytope.Polytope(ChangModel.H, ChangModel.c1_c)
    ext_C = polytope.extreme(poly_C)
    ext_S = polytope.extreme(poly_S)

    ax.fill(ext_C[:, 0], ext_C[:, 1], 'r', zorder=-1)
    ax.fill(ext_S[:, 0], ext_S[:, 1], 'b', zorder=0)

    # Add point showing Ramsey Plan
    idx_Ramsey = np.where(ext_C[:, 0] == max(ext_C[:, 0]))[0][0]
    R = ext_C[idx_Ramsey, :]
    ax.scatter(R[0], R[1], 150, 'black', 'o', zorder=1)
    w_min = min(ext_C[:, 0])

    # Label Ramsey Plan slightly to the right of the point
    ax.annotate("R", xy=(R[0], R[1]),
                xytext=(R[0] + 0.03 * (R[0] - w_min),
                R[1]), fontsize=18)

    plt.tight_layout()
    plt.show()

plot_equilibria(ch1)
```

Evidently, the Ramsey plan, denoted by the $R$, is not sustainable.

Let's raise the discount factor and recompute the sets

```{code-cell} python3
ch2 = ChangModel(β=0.8, mbar=30, h_min=0.9, h_max=1/0.8,
    n_h=8, n_m=35, N_g=10)
```

```{code-cell} python3
ch2.solve_sustainable()
```

Let's plot both sets

```{code-cell} python3
plot_equilibria(ch2)
```

Evidently, the Ramsey plan is now sustainable.
