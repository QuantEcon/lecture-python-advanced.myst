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

(opt_tax_amss)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Optimal Taxation without State-Contingent Debt

```{contents} Contents
:depth: 2
```

**Software Requirement:**

This lecture requires the use of some older software versions to run. If
you would like to execute this lecture please download the following
<a href=_static/downloads/amss_environment.yml download>amss_environment.yml</a>
file. This specifies the software required and an environment can be
created using [conda](https://docs.conda.io/en/latest/):

Open a terminal:

```{code-block} bash
conda env create --file amss_environment.yml
conda activate amss
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install --upgrade quantecon
```

## Overview

Let's start with following imports:

```{code-cell} ipython
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.optimize import root, fmin_slsqp
from scipy.interpolate import UnivariateSpline
from quantecon import MarkovChain
```

In {doc}`an earlier lecture <opt_tax_recur>`, we described a model of
optimal taxation with state-contingent debt due to
Robert E. Lucas, Jr.,  and Nancy Stokey  {cite}`LucasStokey1983`.

Aiyagari, Marcet, Sargent, and Seppälä {cite}`aiyagari2002optimal`  (hereafter, AMSS)
studied optimal taxation in a model without state-contingent debt.

In this lecture, we

* describe assumptions and equilibrium concepts
* solve the model
* implement the model numerically
* conduct some policy experiments
* compare outcomes with those in a corresponding complete-markets model

We begin with an introduction to the model.

## Competitive Equilibrium with Distorting Taxes

Many but not all features of the economy are identical to those of {doc}`the Lucas-Stokey economy <opt_tax_recur>`.

Let's start with things that are identical.

For $t \geq 0$, a history of the state is represented by $s^t = [s_t, s_{t-1}, \ldots, s_0]$.

Government purchases $g(s)$ are an exact time-invariant function of $s$.

Let $c_t(s^t)$,  $\ell_t(s^t)$, and $n_t(s^t)$ denote consumption,
leisure, and labor supply, respectively, at history $s^t$ at time $t$.

Each period a representative  household is endowed with one unit of time that can be divided between  leisure
$\ell_t$ and labor $n_t$:

```{math}
:label: feas1_amss

n_t(s^t) + \ell_t(s^t) = 1
```

Output equals $n_t(s^t)$ and can be divided between consumption $c_t(s^t)$ and $g(s_t)$

```{math}
:label: TSs_techr_amss

c_t(s^t) + g(s_t) = n_t(s^t)
```

Output is not storable.

The technology pins down a pre-tax wage rate to unity for all $t, s^t$.

A representative  household’s preferences over $\{c_t(s^t), \ell_t(s^t)\}_{t=0}^\infty$ are ordered by

```{math}
:label: TS_prefr_amss

\sum_{t=0}^\infty \sum_{s^t} \beta^t \pi_t(s^t) u[c_t(s^t), \ell_t(s^t)]
```

where

* $\pi_t(s^t)$ is a joint probability distribution over the sequence $s^t$, and
* the utility function $u$ is  increasing, strictly concave, and three times  continuously differentiable in both arguments.

The government imposes a flat rate tax $\tau_t(s^t)$ on labor income at time $t$, history $s^t$.

Lucas and Stokey assumed that there are complete markets in one-period Arrow securities; also see {doc}`smoothing models <smoothing>`.

It is at this point that AMSS {cite}`aiyagari2002optimal` modify the Lucas and Stokey economy.

AMSS allow the government to issue only one-period risk-free debt each period.

Ruling out complete markets in this way is a step in the direction of making total tax collections behave more like that prescribed in {cite}`Barro1979` than they do in {cite}`LucasStokey1983`.

### Risk-free One-Period Debt Only

In period $t$ and history $s^t$, let

* $b_{t+1}(s^t)$ be the amount of the time $t+1$ consumption good that at time $t$ the government promised to pay
* $R_t(s^t)$ be the gross interest rate on  risk-free one-period debt between periods $t$ and $t+1$
* $T_t(s^t)$ be a non-negative lump-sum transfer to the representative household [^fn_a]

That $b_{t+1}(s^t)$ is the same for all realizations of $s_{t+1}$ captures its *risk-free* character.

The market value at time $t$ of government debt maturing at time $t+1$ equals $b_{t+1}(s^t)$ divided by $R_t(s^t)$.

The government’s budget constraint in period $t$ at history $s^t$ is

```{math}
:label: TS_gov_wo

\begin{aligned}
b_t(s^{t-1})
    & =    \tau^n_t(s^t) n_t(s^t) - g_t(s_t) - T_t(s^t) +
                   {b_{t+1}(s^t) \over R_t(s^t )}
    \\
    & \equiv z(s^t) + {b_{t+1}(s^t) \over R_t(s^t )},
\end{aligned}
```

where $z(s^t)$ is the net-of-interest government surplus.

To rule out Ponzi schemes, we assume that the government is subject to a **natural debt limit** (to be discussed in a forthcoming lecture).

The consumption Euler equation for a representative household able to trade only one-period risk-free debt
with one-period gross interest rate $R_t(s^t)$ is

$$
{1 \over R_t(s^t)}
= \sum_{s^{t+1}\vert s^t} \beta  \pi_{t+1}(s^{t+1} | s^t)
                        { u_c(s^{t+1}) \over u_c(s^{t}) }
$$

Substituting this expression into the government’s budget constraint {eq}`TS_gov_wo`
yields:

```{math}
:label: TS_gov_wo2

b_t(s^{t-1}) =  z(s^t) + \beta  \sum_{s^{t+1}\vert s^t}  \pi_{t+1}(s^{t+1} | s^t)
                       { u_c(s^{t+1}) \over u_c(s^{t}) } \; b_{t+1}(s^t)
```

Components of $z(s^t)$ on the right side depend on $s^t$, but the left side is required to depend on $s^{t-1}$ only.

**This is what it means for one-period government debt to be risk-free**.

Therefore, the sum on the right side of equation {eq}`TS_gov_wo2` also has to depend only on $s^{t-1}$.

This requirement will give rise to **measurability constraints** on the Ramsey allocation to be discussed soon.

If we replace $b_{t+1}(s^t)$ on the right side of equation {eq}`TS_gov_wo2` by the right
side of next period’s budget constraint (associated with a
particular realization $s_{t}$) we get

$$
b_t(s^{t-1}) =  z(s^t) + \sum_{s^{t+1}\vert s^t} \beta  \pi_{t+1}(s^{t+1} | s^t)
                       { u_c(s^{t+1}) \over u_c(s^{t}) }
\, \left[z(s^{t+1}) + {b_{t+2}(s^{t+1}) \over R_{t+1}(s^{t+1})}\right]
$$

After making similar repeated substitutions for all future occurrences of
government indebtedness, and by invoking the natural debt limit, we
arrive at:

```{math}
:label: TS_gov_wo3

\begin{aligned}
b_t(s^{t-1})
    &=  \sum_{j=0}^\infty \sum_{s^{t+j} | s^t} \beta^j  \pi_{t+j}(s^{t+j} | s^t)
              { u_c(s^{t+j}) \over u_c(s^{t}) } \;z(s^{t+j})
        \end{aligned}
```

Now let's

* substitute the resource constraint into the net-of-interest government surplus, and
* use the household’s first-order condition $1-\tau^n_t(s^t)= u_{\ell}(s^t) /u_c(s^t)$ to eliminate the labor tax rate

so that we can express the net-of-interest government surplus $z(s^t)$ as

```{math}
:label: AMSS_44_2

z(s^t)
    = \left[1 - {u_{\ell}(s^t) \over u_c(s^t)}\right] \left[c_t(s^t)+g_t(s_t)\right]
        -g_t(s_t) - T_t(s^t)\,.
```

If we substitute the  appropriate versions of the right side of {eq}`AMSS_44_2` for $z(s^{t+j})$ into equation {eq}`TS_gov_wo3`,
we obtain a sequence of *implementability constraints* on a Ramsey allocation in an AMSS economy.

Expression {eq}`TS_gov_wo3` at time $t=0$ and initial state $s^0$
was also  an *implementability constraint* on a Ramsey allocation in a Lucas-Stokey economy:

```{math}
:label: TS_gov_wo4

b_0(s^{-1}) = \mathbb E_0 \sum_{j=0}^\infty \beta^j
               { u_c(s^{j}) \over u_c(s^{0}) } \;z(s^{j})
```

Indeed, it was the *only* implementability constraint there.

But now we also have a large number of additional implementability constraints

```{math}
:label: TS_gov_wo4a

b_t(s^{t-1}) =  \mathbb E_t \sum_{j=0}^\infty \beta^j
              { u_c(s^{t+j}) \over u_c(s^{t}) } \;z(s^{t+j})
```

Equation {eq}`TS_gov_wo4a` must hold for each $s^t$ for each $t \geq 1$.

### Comparison with Lucas-Stokey Economy

The expression on the right side of {eq}`TS_gov_wo4a` in the Lucas-Stokey (1983) economy would  equal the present value of a continuation stream of government surpluses evaluated at what would be competitive equilibrium Arrow-Debreu prices at date $t$.

In the Lucas-Stokey economy, that present value is measurable with respect to $s^t$.

In the AMSS economy, the restriction that government debt be risk-free imposes that that same present value must be measurable with respect to $s^{t-1}$.

In a language used in the literature on incomplete markets models, it can be said that the AMSS model requires that at each $(t, s^t)$ what would be the present value of continuation government surpluses in the Lucas-Stokey model must belong to  the **marketable subspace** of the AMSS model.

### Ramsey Problem Without State-contingent Debt

After we have substituted the resource constraint into the utility function, we can express the Ramsey problem as being to choose an allocation that solves

$$
\max_{\{c_t(s^t),b_{t+1}(s^t)\}}
\mathbb E_0 \sum_{t=0}^\infty \beta^t
                        u\left(c_t(s^t),1-c_t(s^t)-g_t(s_t)\right)
$$

where the maximization is subject to

```{math}
:label: AMSS_44

\mathbb E_{0} \sum_{j=0}^\infty \beta^j
      { u_c(s^{j}) \over u_c(s^{0}) } \;z(s^{j}) \geq b_0(s^{-1})
```

and

```{math}
:label: AMSS_46

\mathbb E_{t} \sum_{j=0}^\infty \beta^j
    { u_c(s^{t+j}) \over u_c(s^{t}) } \;
    z(s^{t+j}) = b_t(s^{t-1})
      \quad \forall \,  s^t
```

given $b_0(s^{-1})$.

#### Lagrangian Formulation

Let $\gamma_0(s^0)$ be a non-negative Lagrange multiplier on constraint {eq}`AMSS_44`.

As in the Lucas-Stokey economy, this multiplier is strictly positive when the government must resort to
distortionary taxation; otherwise it equals zero.

A consequence of the assumption that there are no markets in state-contingent securities  and that a market exists only in a risk-free security is that we have to attach stochastic processes $\{\gamma_t(s^t)\}_{t=1}^\infty$ of
Lagrange multipliers to the implementability constraints {eq}`AMSS_46`.

Depending on how the constraints  bind, these multipliers can be positive or negative:

$$
\begin{aligned}
   \gamma_t(s^t)
   &\;\geq\; (\leq)\;\, 0 \quad \text{if the constraint binds in this direction }
   \\
   & \mathbb E_{t} \sum_{j=0}^\infty \beta^j
    { u_c(s^{t+j}) \over u_c(s^{t}) } \;z(s^{t+j}) \;\geq \;(\leq)\;\, b_t(s^{t-1})
\end{aligned}
$$

A negative multiplier $\gamma_t(s^t)<0$ means that if we could
relax constraint {eq}`AMSS_46`, we would like to *increase* the beginning-of-period
indebtedness for that particular realization of history $s^t$.

That would let us reduce the beginning-of-period indebtedness for some other history [^fn_b].

These features flow from  the fact that the government cannot use state-contingent debt and therefore cannot allocate its indebtedness  efficiently across future states.

### Some Calculations

It is helpful to apply two transformations to the Lagrangian.

Multiply constraint {eq}`AMSS_44` by $u_c(s^0)$ and the constraints {eq}`AMSS_46` by $\beta^t u_c(s^{t})$.

Then a Lagrangian for the Ramsey problem can  be represented as

```{math}
:label: AMSS_lagr;a

\begin{aligned}
   J &= \mathbb E_{0} \sum_{t=0}^\infty \beta^t
                        \biggl\{ u\left(c_t(s^t), 1-c_t(s^t)-g_t(s_t)\right)\\
   &  \qquad + \gamma_t(s^t) \Bigl[ \mathbb E_{t} \sum_{j=0}^\infty \beta^j
         u_c(s^{t+j}) \,z(s^{t+j}) - u_c(s^{t}) \,b_t(s^{t-1}) \biggr\}
         \\
   &= \mathbb E_{0} \sum_{t=0}^\infty \beta^t
                         \biggl\{ u\left(c_t(s^t), 1-c_t(s^t)-g_t(s_t)\right)
        \\
   &  \qquad + \Psi_t(s^t)\, u_c(s^{t}) \,z(s^{t}) -
                   \gamma_t(s^t)\, u_c(s^{t}) \, b_t(s^{t-1})  \biggr\}
\end{aligned}
```

where

```{math}
:label: AMSS_lagr;

\Psi_t(s^t)=\Psi_{t-1}(s^{t-1})+\gamma_t(s^t)
 \quad \text{and} \quad
\Psi_{-1}(s^{-1})=0
```

In {eq}`AMSS_lagr;a`,  the second equality uses  the law of iterated expectations
and Abel’s summation formula (also called *summation by parts*, see
[this page](https://en.wikipedia.org/wiki/Abel%27s_summation_formula)).

First-order conditions with respect
to $c_t(s^t)$ can be expressed as

```{math}
:label: AMSS_foc;a

\begin{aligned}
  u_c(s^t)-u_{\ell}(s^t) &+ \Psi_t(s^t)\left\{ \left[
    u_{cc}(s^t) - u_{c\ell}(s^{t})\right]z(s^{t}) +
    u_{c}(s^{t})\,z_c(s^{t}) \right\}
    \\
    & \hspace{35mm} - \gamma_t(s^t)\left[
    u_{cc}(s^{t}) - u_{c\ell}(s^{t})\right]b_t(s^{t-1}) =0
\end{aligned}
```

and with respect to $b_t(s^t)$ as

```{math}
:label: AMSS_foc;b

\mathbb E_{t} \left[\gamma_{t+1}(s^{t+1})\,u_c(s^{t+1})\right] = 0
```

If we substitute $z(s^t)$ from {eq}`AMSS_44_2` and its derivative
$z_c(s^t)$ into the first-order condition {eq}`AMSS_foc;a`, we  find  two
differences from the corresponding condition for the optimal allocation
in a Lucas-Stokey economy with state-contingent government debt.

> 

1. The term involving $b_t(s^{t-1})$ in the first-order condition
   {eq}`AMSS_foc;a` does not appear in the corresponding expression
   for the Lucas-Stokey economy.
    * This term reflects the constraint that
      beginning-of-period government indebtedness must be the same across all
      realizations of next period’s state, a constraint that would  not be present if
      government debt could be state contingent.
1. The Lagrange multiplier $\Psi_t(s^t)$ in the first-order condition
   {eq}`AMSS_foc;a` may change over time in response to realizations of the state,
   while the multiplier $\Phi$ in the Lucas-Stokey economy is time-invariant.

We need some code from our {doc}`an earlier lecture <opt_tax_recur>`
on optimal taxation with state-contingent debt  sequential allocation implementation:

```
---
lineno-start: 1
---
import numpy as np
from scipy.optimize import root
from quantecon import MarkovChain


class SequentialAllocation:

    '''
    Class that takes CESutility or BGPutility object as input returns
    planner's allocation as a function of the multiplier on the
    implementability constraint μ.
    '''

    def __init__(self, model):

        # Initialize from model object attributes
        self.β, self.π, self.G = model.β, model.π, model.G
        self.mc, self.Θ = MarkovChain(self.π), model.Θ
        self.S = len(model.π)  # Number of states
        self.model = model

        # Find the first best allocation
        self.find_first_best()

    def find_first_best(self):
        '''
        Find the first best allocation
        '''
        model = self.model
        S, Θ, G = self.S, self.Θ, self.G
        Uc, Un = model.Uc, model.Un

        def res(z):
            c = z[:S]
            n = z[S:]
            return np.hstack([Θ * Uc(c, n) + Un(c, n), Θ * n - c - G])

        res = root(res, 0.5 * np.ones(2 * S))

        if not res.success:
            raise Exception('Could not find first best')

        self.cFB = res.x[:S]
        self.nFB = res.x[S:]

        # Multiplier on the resource constraint
        self.ΞFB = Uc(self.cFB, self.nFB)
        self.zFB = np.hstack([self.cFB, self.nFB, self.ΞFB])

    def time1_allocation(self, μ):
        '''
        Computes optimal allocation for time t >= 1 for a given μ
        '''
        model = self.model
        S, Θ, G = self.S, self.Θ, self.G
        Uc, Ucc, Un, Unn = model.Uc, model.Ucc, model.Un, model.Unn

        def FOC(z):
            c = z[:S]
            n = z[S:2 * S]
            Ξ = z[2 * S:]
            # FOC of c
            return np.hstack([Uc(c, n) - μ * (Ucc(c, n) * c + Uc(c, n)) - Ξ,
                              Un(c, n) - μ * (Unn(c, n) * n + Un(c, n)) \
                              + Θ * Ξ,  # FOC of n
                              Θ * n - c - G])

        # Find the root of the first-order condition
        res = root(FOC, self.zFB)
        if not res.success:
            raise Exception('Could not find LS allocation.')
        z = res.x
        c, n, Ξ = z[:S], z[S:2 * S], z[2 * S:]

        # Compute x
        I = Uc(c, n) * c + Un(c, n) * n
        x = np.linalg.solve(np.eye(S) - self.β * self.π, I)

        return c, n, x, Ξ

    def time0_allocation(self, B_, s_0):
        '''
        Finds the optimal allocation given initial government debt B_ and
        state s_0
        '''
        model, π, Θ, G, β = self.model, self.π, self.Θ, self.G, self.β
        Uc, Ucc, Un, Unn = model.Uc, model.Ucc, model.Un, model.Unn

        # First order conditions of planner's problem
        def FOC(z):
            μ, c, n, Ξ = z
            xprime = self.time1_allocation(μ)[2]
            return np.hstack([Uc(c, n) * (c - B_) + Un(c, n) * n + β * π[s_0]
                                            @ xprime,
                              Uc(c, n) - μ * (Ucc(c, n)
                                            * (c - B_) + Uc(c, n)) - Ξ,
                              Un(c, n) - μ * (Unn(c, n) * n
                                            + Un(c, n)) + Θ[s_0] * Ξ,
                              (Θ * n - c - G)[s_0]])

        # Find root
        res = root(FOC, np.array(
            [0, self.cFB[s_0], self.nFB[s_0], self.ΞFB[s_0]]))
        if not res.success:
            raise Exception('Could not find time 0 LS allocation.')

        return res.x

    def time1_value(self, μ):
        '''
        Find the value associated with multiplier μ
        '''
        c, n, x, Ξ = self.time1_allocation(μ)
        U = self.model.U(c, n)
        V = np.linalg.solve(np.eye(self.S) - self.β * self.π, U)
        return c, n, x, V

    def Τ(self, c, n):
        '''
        Computes Τ given c, n
        '''
        model = self.model
        Uc, Un = model.Uc(c, n), model.Un(c,  n)

        return 1 + Un / (self.Θ * Uc)

    def simulate(self, B_, s_0, T, sHist=None):
        '''
        Simulates planners policies for T periods
        '''
        model, π, β = self.model, self.π, self.β
        Uc = model.Uc

        if sHist is None:
            sHist = self.mc.simulate(T, s_0)

        cHist, nHist, Bhist, ΤHist, μHist = np.zeros((5, T))
        RHist = np.zeros(T - 1)

        # Time 0
        μ, cHist[0], nHist[0], _ = self.time0_allocation(B_, s_0)
        ΤHist[0] = self.Τ(cHist[0], nHist[0])[s_0]
        Bhist[0] = B_
        μHist[0] = μ

        # Time 1 onward
        for t in range(1, T):
            c, n, x, Ξ = self.time1_allocation(μ)
            Τ = self.Τ(c, n)
            u_c = Uc(c, n)
            s = sHist[t]
            Eu_c = π[sHist[t - 1]] @ u_c
            cHist[t], nHist[t], Bhist[t], ΤHist[t] = c[s], n[s], x[s] / u_c[s], \
                                                     Τ[s]
            RHist[t - 1] = Uc(cHist[t - 1], nHist[t - 1]) / (β * Eu_c)
            μHist[t] = μ

        return np.array([cHist, nHist, Bhist, ΤHist, sHist, μHist, RHist])

```

To analyze the AMSS model, we find it useful to adopt a recursive formulation
using techniques like those in our lectures on {doc}`dynamic Stackelberg models <dyn_stack>` and {doc}`optimal taxation with state-contingent debt <opt_tax_recur>`.

## Recursive Version of AMSS Model

We now describe a recursive formulation of the AMSS economy.

We have noted that from the point of view of the Ramsey planner, the restriction
to one-period risk-free securities

* leaves intact the single implementability constraint on allocations
  {eq}`TS_gov_wo4` from the Lucas-Stokey economy, but
* adds measurability constraints {eq}`TS_gov_wo3` on functions of tails of
  allocations at each time and history

We now explore how these constraints alter  Bellman equations for a time
$0$ Ramsey planner and for time $t \geq 1$, history $s^t$
continuation Ramsey planners.

### Recasting State Variables

In the AMSS setting, the government faces a sequence of budget constraints

$$
\tau_t(s^t) n_t(s^t) + T_t(s^t) +  b_{t+1}(s^t)/ R_t (s^t) =  g_t + b_t(s^{t-1})
$$

where $R_t(s^t)$ is the gross risk-free rate of interest between $t$
and $t+1$ at history $s^t$ and $T_t(s^t)$ are non-negative transfers.

Throughout this lecture, we shall set transfers to zero (for some issues about the limiting behavior of debt, this makes a possibly
important  difference from AMSS {cite}`aiyagari2002optimal`, who restricted transfers
to be non-negative).

In this case, the household faces a sequence of budget constraints

```{math}
:label: eqn:AMSSapp1

b_t(s^{t-1}) + (1-\tau_t(s^t)) n_t(s^t) = c_t(s^t) + b_{t+1}(s^t)/R_t(s^t)
```

The household’s first-order conditions are $u_{c,t} = \beta R_t \mathbb E_t u_{c,t+1}$
and $(1-\tau_t) u_{c,t} = u_{l,t}$.

Using these to eliminate $R_t$ and $\tau_t$ from  budget constraint
{eq}`eqn:AMSSapp1` gives

```{math}
:label: eqn:AMSSapp2a

b_t(s^{t-1}) + \frac{u_{l,t}(s^t)}{u_{c,t}(s^t)} n_t(s^t)
= c_t(s^t) + {\frac{\beta (\mathbb E_t u_{c,t+1}) b_{t+1}(s^t)}{u_{c,t}(s^t)}}
```

or

```{math}
:label: eqn:AMSSapp2

u_{c,t}(s^t) b_t(s^{t-1}) + u_{l,t}(s^t) n_t(s^t)
= u_{c,t}(s^t) c_t(s^t) + \beta (\mathbb E_t u_{c,t+1}) b_{t+1}(s^t)
```

Now define

```{math}
:label: eqn:AMSSapp3

x_t \equiv \beta b_{t+1}(s^t) \mathbb E_t u_{c,t+1} = u_{c,t} (s^t) {\frac{b_{t+1}(s^t)}{R_t(s^t)}}
```

and represent the household’s budget constraint at time $t$,
history $s^t$ as

```{math}
:label: eqn:AMSSapp4

{\frac{u_{c,t} x_{t-1}}{\beta \mathbb E_{t-1} u_{c,t}}} = u_{c,t} c_t - u_{l,t} n_t + x_t
```

for $t \geq 1$.

### Measurability Constraints

Write equation {eq}`eqn:AMSSapp2` as

```{math}
:label: eqn:AMSSapp2b

b_t(s^{t-1})  = c_t(s^t) -  { \frac{u_{l,t}(s^t)}{u_{c,t}(s^t)}} n_t(s^t) +
{\frac{\beta (\mathbb E_t u_{c,t+1}) b_{t+1}(s^t)}{u_{c,t}}}
```

The right side of equation {eq}`eqn:AMSSapp2b` expresses the time $t$ value of government debt
in terms of a linear combination of terms whose individual components
are measurable with respect to $s^t$.

The sum  of terms on the right side  of equation {eq}`eqn:AMSSapp2b` must equal
$b_t(s^{t-1})$.

That implies that it has to be *measurable* with respect to $s^{t-1}$.

Equations {eq}`eqn:AMSSapp2b` are the *measurability constraints* that the AMSS model adds to the single time $0$ implementation
constraint imposed in the Lucas and Stokey model.

### Two Bellman Equations

Let $\Pi(s|s_-)$ be a Markov transition matrix whose entries tell probabilities of moving from state $s_-$ to state $s$ in one period.

Let

* $V(x_-, s_-)$ be the continuation value of a continuation
  Ramsey plan at $x_{t-1} = x_-, s_{t-1} =s_-$ for $t \geq 1$
* $W(b, s)$ be the value of the Ramsey plan at time $0$ at
  $b_0=b$ and $s_0 = s$

We distinguish between two types of planners:

For $t \geq 1$, the value function for a **continuation Ramsey planner**
satisfies the Bellman equation

```{math}
:label: eqn:AMSSapp5

V(x_-,s_-) = \max_{\{n(s), x(s)\}} \sum_s \Pi(s|s_-) \left[ u(n(s) -
g(s), 1-n(s)) + \beta V(x(s),s) \right]
```

subject to the following collection of implementability constraints, one
for each $s \in {\cal S}$:

```{math}
:label: eqn:AMSSapp6

{\frac{u_c(s) x_- }{\beta \sum_{\tilde s} \Pi(\tilde s|s_-) u_c(\tilde s) }}
= u_c(s) (n(s) - g(s)) - u_l(s) n(s) + x(s)
```

A continuation Ramsey planner at $t \geq 1$ takes
$(x_{t-1}, s_{t-1}) = (x_-, s_-)$ as given and before
$s$ is realized chooses
$(n_t(s_t), x_t(s_t)) = (n(s), x(s))$ for $s \in  {\cal S}$.

The **Ramsey planner** takes $(b_0, s_0)$ as given and chooses $(n_0, x_0)$.

The value function $W(b_0, s_0)$   for the time $t=0$ Ramsey planner
satisfies the Bellman equation

```{math}
:label: eqn:AMSSapp100

W(b_0, s_0) = \max_{n_0, x_0} u(n_0 - g_0, 1-n_0) + \beta V(x_0,s_0)
```

where maximization is subject to

```{math}
:label: eqn:AMMSSapp101

u_{c,0} b_0 = u_{c,0} (n_0-g_0) - u_{l,0} n_0 + x_0
```

### Martingale Supercedes State-Variable Degeneracy

Let $\mu(s|s_-) \Pi(s|s_-)$ be a Lagrange multiplier on the constraint {eq}`eqn:AMSSapp6`
for state $s$.

After forming an appropriate Lagrangian, we find that the continuation Ramsey planner’s first-order
condition with respect to $x(s)$ is

```{math}
:label: eqn:AMSSapp7

\beta V_x(x(s),s) = \mu(s|s_-)
```

Applying the envelope theorem to Bellman equation {eq}`eqn:AMSSapp5` gives

```{math}
:label: eqn:AMSSapp8

V_x(x_-,s_-) = \sum_s \Pi(s|s_-) \mu(s|s_-) {\frac{u_c(s)}{\beta \sum_{\tilde s}
\Pi(\tilde s|s_-) u_c(\tilde s) }}
```

Equations {eq}`eqn:AMSSapp7` and {eq}`eqn:AMSSapp8` imply that

```{math}
:label: eqn:AMSSapp9

V_x(x_-, s_-) = \sum_{s} \left( \Pi(s|s_-) {\frac{u_c(s)}{\sum_{\tilde s}
\Pi(\tilde s| s_-) u_c(\tilde s)}} \right) V_x(x(s), s)
```

Equation {eq}`eqn:AMSSapp9` states that $V_x(x, s)$ is a *risk-adjusted martingale*.

Saying that $V_x(x, s)$ is a risk-adjusted martingale  means  that
$V_x(x, s)$  is a martingale with respect to the probability distribution
over $s^t$ sequences that are generated by the *twisted* transition probability matrix:

$$
\check \Pi(s|s_-) \equiv \Pi(s|s_-) {\frac{u_c(s)}{\sum_{\tilde s}
\Pi(\tilde s| s_-) u_c(\tilde s)}}
$$

**Exercise**: Please verify that $\check \Pi(s|s_-)$ is a valid Markov
transition density, i.e., that its elements are all non-negative and
that for each $s_-$, the sum over $s$ equals unity.

### Absence of State Variable Degeneracy

Along a Ramsey plan, the state variable $x_t = x_t(s^t, b_0)$
becomes a function of the history $s^t$ and initial
government debt $b_0$.

In {doc}`Lucas-Stokey model <opt_tax_recur>`, we
found that

* a counterpart to $V_x(x,s)$ is time-invariant and equal to
  the Lagrange multiplier on the Lucas-Stokey implementability constraint
* time invariance of $V_x(x,s)$  is the source of a key
  feature of the Lucas-Stokey model, namely, state variable degeneracy
  (i.e., $x_t$ is an exact function of $s_t$)

That $V_x(x,s)$ varies over time according to a twisted martingale
means that there is no state-variable degeneracy in the AMSS model.

In the AMSS model, both $x$ and $s$ are needed to describe the state.

This property of the AMSS model  transmits a twisted martingale
component to consumption, employment, and the tax rate.

### Digression on Non-negative Transfers

Throughout this lecture, we have imposed that transfers $T_t = 0$.

AMSS {cite}`aiyagari2002optimal` instead imposed a nonnegativity
constraint $T_t\geq 0$ on transfers.

They also considered a special case of quasi-linear preferences,
$u(c,l)= c + H(l)$.

In this case, $V_x(x,s)\leq 0$ is a non-positive martingale.

By the *martingale convergence theorem*  $V_x(x,s)$ converges almost surely.

Furthermore, when the Markov chain $\Pi(s| s_-)$ and the government
expenditure function $g(s)$ are such that $g_t$ is perpetually
random, $V_x(x, s)$ almost surely converges to zero.

For quasi-linear preferences, the first-order condition with respect to $n(s)$ becomes

$$
(1-\mu(s|s_-) ) (1 - u_l(s)) + \mu(s|s_-) n(s) u_{ll}(s) =0
$$

When $\mu(s|s_-) = \beta V_x(x(s),x)$ converges to zero, in the limit
$u_l(s)= 1 =u_c(s)$, so that $\tau(x(s),s) =0$.

Thus, in the limit, if $g_t$ is perpetually random,  the government
accumulates sufficient assets to finance all expenditures from earnings on those
assets, returning any excess revenues to the household as non-negative lump-sum transfers.

### Code

The recursive formulation is implemented as follows

```
---
lineno-start: 1
---
import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.optimize import root
from quantecon import MarkovChain


class RecursiveAllocationAMSS:

    def __init__(self, model, μgrid, tol_diff=1e-4, tol=1e-4):

        self.β, self.π, self.G = model.β, model.π, model.G
        self.mc, self.S = MarkovChain(self.π), len(model.π)  # Number of states
        self.Θ, self.model, self.μgrid = model.Θ, model, μgrid
        self.tol_diff, self.tol = tol_diff, tol

        # Find the first best allocation
        self.solve_time1_bellman()
        self.T.time_0 = True  # Bellman equation now solves time 0 problem

    def solve_time1_bellman(self):
        '''
        Solve the time  1 Bellman equation for calibration model and
        initial grid μgrid0
        '''
        model, μgrid0 = self.model, self.μgrid
        π = model.π
        S = len(model.π)

        # First get initial fit from Lucas Stokey solution.
        # Need to change things to be ex ante
        pp = SequentialAllocation(model)
        interp = interpolator_factory(2, None)

        def incomplete_allocation(μ_, s_):
            c, n, x, V = pp.time1_value(μ_)
            return c, n, π[s_] @ x, π[s_] @ V
        cf, nf, xgrid, Vf, xprimef = [], [], [], [], []
        for s_ in range(S):
            c, n, x, V = zip(*map(lambda μ: incomplete_allocation(μ, s_), μgrid0))
            c, n = np.vstack(c).T, np.vstack(n).T
            x, V = np.hstack(x), np.hstack(V)
            xprimes = np.vstack([x] * S)
            cf.append(interp(x, c))
            nf.append(interp(x, n))
            Vf.append(interp(x, V))
            xgrid.append(x)
            xprimef.append(interp(x, xprimes))
        cf, nf, xprimef = fun_vstack(cf), fun_vstack(nf), fun_vstack(xprimef)
        Vf = fun_hstack(Vf)
        policies = [cf, nf, xprimef]

        # Create xgrid
        x = np.vstack(xgrid).T
        xbar = [x.min(0).max(), x.max(0).min()]
        xgrid = np.linspace(xbar[0], xbar[1], len(μgrid0))
        self.xgrid = xgrid

        # Now iterate on Bellman equation
        T = BellmanEquation(model, xgrid, policies, tol=self.tol)
        diff = 1
        while diff > self.tol_diff:
            PF = T(Vf)

            Vfnew, policies = self.fit_policy_function(PF)
            diff = np.abs((Vf(xgrid) - Vfnew(xgrid)) / Vf(xgrid)).max()

            print(diff)
            Vf = Vfnew

        # Store value function policies and Bellman Equations
        self.Vf = Vf
        self.policies = policies
        self.T = T

    def fit_policy_function(self, PF):
        '''
        Fits the policy functions
        '''
        S, xgrid = len(self.π), self.xgrid
        interp = interpolator_factory(3, 0)
        cf, nf, xprimef, Tf, Vf = [], [], [], [], []
        for s_ in range(S):
            PFvec = np.vstack([PF(x, s_) for x in self.xgrid]).T
            Vf.append(interp(xgrid, PFvec[0, :]))
            cf.append(interp(xgrid, PFvec[1:1 + S]))
            nf.append(interp(xgrid, PFvec[1 + S:1 + 2 * S]))
            xprimef.append(interp(xgrid, PFvec[1 + 2 * S:1 + 3 * S]))
            Tf.append(interp(xgrid, PFvec[1 + 3 * S:]))
        policies = fun_vstack(cf), fun_vstack(
            nf), fun_vstack(xprimef), fun_vstack(Tf)
        Vf = fun_hstack(Vf)
        return Vf, policies

    def Τ(self, c, n):
        '''
        Computes Τ given c and n
        '''
        model = self.model
        Uc, Un = model.Uc(c, n), model.Un(c, n)

        return 1 + Un / (self.Θ * Uc)

    def time0_allocation(self, B_, s0):
        '''
        Finds the optimal allocation given initial government debt B_ and
        state s_0
        '''
        PF = self.T(self.Vf)
        z0 = PF(B_, s0)
        c0, n0, xprime0, T0 = z0[1:]
        return c0, n0, xprime0, T0

    def simulate(self, B_, s_0, T, sHist=None):
        '''
        Simulates planners policies for T periods
        '''
        model, π = self.model, self.π
        Uc = model.Uc
        cf, nf, xprimef, Tf = self.policies

        if sHist is None:
            sHist = simulate_markov(π, s_0, T)

        cHist, nHist, Bhist, xHist, ΤHist, THist, μHist = np.zeros((7, T))
        # Time 0
        cHist[0], nHist[0], xHist[0], THist[0] = self.time0_allocation(B_, s_0)
        ΤHist[0] = self.Τ(cHist[0], nHist[0])[s_0]
        Bhist[0] = B_
        μHist[0] = self.Vf[s_0](xHist[0])

        # Time 1 onward
        for t in range(1, T):
            s_, x, s = sHist[t - 1], xHist[t - 1], sHist[t]
            c, n, xprime, T = cf[s_, :](x), nf[s_, :](
                x), xprimef[s_, :](x), Tf[s_, :](x)

            Τ = self.Τ(c, n)[s]
            u_c = Uc(c, n)
            Eu_c = π[s_, :] @ u_c

            μHist[t] = self.Vf[s](xprime[s])

            cHist[t], nHist[t], Bhist[t], ΤHist[t] = c[s], n[s], x / Eu_c, Τ
            xHist[t], THist[t] = xprime[s], T[s]
        return np.array([cHist, nHist, Bhist, ΤHist, THist, μHist, sHist, xHist])


class BellmanEquation:
    '''
    Bellman equation for the continuation of the Lucas-Stokey Problem
    '''

    def __init__(self, model, xgrid, policies0, tol, maxiter=1000):

        self.β, self.π, self.G = model.β, model.π, model.G
        self.S = len(model.π)  # Number of states
        self.Θ, self.model, self.tol = model.Θ, model, tol
        self.maxiter = maxiter

        self.xbar = [min(xgrid), max(xgrid)]
        self.time_0 = False

        self.z0 = {}
        cf, nf, xprimef = policies0

        for s_ in range(self.S):
            for x in xgrid:
                self.z0[x, s_] = np.hstack([cf[s_, :](x),
                                            nf[s_, :](x),
                                            xprimef[s_, :](x),
                                            np.zeros(self.S)])

        self.find_first_best()

    def find_first_best(self):
        '''
        Find the first best allocation
        '''
        model = self.model
        S, Θ, Uc, Un, G = self.S, self.Θ, model.Uc, model.Un, self.G

        def res(z):
            c = z[:S]
            n = z[S:]
            return np.hstack([Θ * Uc(c, n) + Un(c, n), Θ * n - c - G])

        res = root(res, 0.5 * np.ones(2 * S))
        if not res.success:
            raise Exception('Could not find first best')

        self.cFB = res.x[:S]
        self.nFB = res.x[S:]
        IFB = Uc(self.cFB, self.nFB) * self.cFB + \
            Un(self.cFB, self.nFB) * self.nFB

        self.xFB = np.linalg.solve(np.eye(S) - self.β * self.π, IFB)

        self.zFB = {}
        for s in range(S):
            self.zFB[s] = np.hstack(
                [self.cFB[s], self.nFB[s], self.π[s] @ self.xFB, 0.])

    def __call__(self, Vf):
        '''
        Given continuation value function next period return value function this
        period return T(V) and optimal policies
        '''
        if not self.time_0:
            def PF(x, s): return self.get_policies_time1(x, s, Vf)
        else:
            def PF(B_, s0): return self.get_policies_time0(B_, s0, Vf)
        return PF

    def get_policies_time1(self, x, s_, Vf):
        '''
        Finds the optimal policies 
        '''
        model, β, Θ, G, S, π = self.model, self.β, self.Θ, self.G, self.S, self.π
        U, Uc, Un = model.U, model.Uc, model.Un

        def objf(z):
            c, n, xprime = z[:S], z[S:2 * S], z[2 * S:3 * S]

            Vprime = np.empty(S)
            for s in range(S):
                Vprime[s] = Vf[s](xprime[s])

            return -π[s_] @ (U(c, n) + β * Vprime)

        def cons(z):
            c, n, xprime, T = z[:S], z[S:2 * S], z[2 * S:3 * S], z[3 * S:]
            u_c = Uc(c, n)
            Eu_c = π[s_] @ u_c
            return np.hstack([
                x * u_c / Eu_c - u_c * (c - T) - Un(c, n) * n - β * xprime,
                Θ * n - c - G])

        if model.transfers:
            bounds = [(0., 100)] * S + [(0., 100)] * S + \
                [self.xbar] * S + [(0., 100.)] * S
        else:
            bounds = [(0., 100)] * S + [(0., 100)] * S + \
                [self.xbar] * S + [(0., 0.)] * S
        out, fx, _, imode, smode = fmin_slsqp(objf, self.z0[x, s_],
                                              f_eqcons=cons, bounds=bounds,
                                              full_output=True, iprint=0,
                                              acc=self.tol, iter=self.maxiter)

        if imode > 0:
            raise Exception(smode)

        self.z0[x, s_] = out
        return np.hstack([-fx, out])

    def get_policies_time0(self, B_, s0, Vf):
        '''
        Finds the optimal policies 
        '''
        model, β, Θ, G = self.model, self.β, self.Θ, self.G
        U, Uc, Un = model.U, model.Uc, model.Un

        def objf(z):
            c, n, xprime = z[:-1]

            return -(U(c, n) + β * Vf[s0](xprime))

        def cons(z):
            c, n, xprime, T = z
            return np.hstack([
                -Uc(c, n) * (c - B_ - T) - Un(c, n) * n - β * xprime,
                (Θ * n - c - G)[s0]])

        if model.transfers:
            bounds = [(0., 100), (0., 100), self.xbar, (0., 100.)]
        else:
            bounds = [(0., 100), (0., 100), self.xbar, (0., 0.)]
        out, fx, _, imode, smode = fmin_slsqp(objf, self.zFB[s0], f_eqcons=cons,
                                              bounds=bounds, full_output=True,
                                              iprint=0)

        if imode > 0:
            raise Exception(smode)

        return np.hstack([-fx, out])

```

## Examples

We now turn to some examples.

We will first build some useful functions for solving the model

```
---
lineno-start: 1
---
import numpy as np
from scipy.interpolate import UnivariateSpline


class interpolate_wrapper:

    def __init__(self, F):
        self.F = F

    def __getitem__(self, index):
        return interpolate_wrapper(np.asarray(self.F[index]))

    def reshape(self, *args):
        self.F = self.F.reshape(*args)
        return self

    def transpose(self):
        self.F = self.F.transpose()

    def __len__(self):
        return len(self.F)

    def __call__(self, xvec):
        x = np.atleast_1d(xvec)
        shape = self.F.shape
        if len(x) == 1:
            fhat = np.hstack([f(x) for f in self.F.flatten()])
            return fhat.reshape(shape)
        else:
            fhat = np.vstack([f(x) for f in self.F.flatten()])
            return fhat.reshape(np.hstack((shape, len(x))))


class interpolator_factory:

    def __init__(self, k, s):
        self.k, self.s = k, s

    def __call__(self, xgrid, Fs):
        shape, m = Fs.shape[:-1], Fs.shape[-1]
        Fs = Fs.reshape((-1, m))
        F = []
        xgrid = np.sort(xgrid)  # Sort xgrid
        for Fhat in Fs:
            F.append(UnivariateSpline(xgrid, Fhat, k=self.k, s=self.s))
        return interpolate_wrapper(np.array(F).reshape(shape))


def fun_vstack(fun_list):

    Fs = [IW.F for IW in fun_list]
    return interpolate_wrapper(np.vstack(Fs))


def fun_hstack(fun_list):

    Fs = [IW.F for IW in fun_list]
    return interpolate_wrapper(np.hstack(Fs))


def simulate_markov(π, s_0, T):

    sHist = np.empty(T, dtype=int)
    sHist[0] = s_0
    S = len(π)
    for t in range(1, T):
        sHist[t] = np.random.choice(np.arange(S), p=π[sHist[t - 1]])

    return sHist

```

### Anticipated One-Period War

In our lecture on {doc}`optimal taxation with state contingent debt <opt_tax_recur>`
we studied how the government manages uncertainty in a simple setting.

As in that lecture, we assume the one-period utility function

$$
u(c,n) = {\frac{c^{1-\sigma}}{1-\sigma}} - {\frac{n^{1+\gamma}}{1+\gamma}}
$$

```{note}
For convenience in  matching our computer code, we have expressed
utility as a function of $n$ rather than leisure $l$.
```

We consider the same government expenditure process studied in the lecture on
{doc}`optimal taxation with state contingent debt <opt_tax_recur>`.

Government expenditures are known for sure in all periods except one.

* For $t<3$ or $t > 3$ we assume that $g_t = g_l = 0.1$.
* At $t = 3$ a war occurs with probability 0.5.
    * If there is war, $g_3 = g_h = 0.2$.
    * If there is no war $g_3 = g_l = 0.1$.

A useful trick is to define  components of the state vector as the following six
$(t,g)$ pairs:

$$
(0,g_l), (1,g_l), (2,g_l), (3,g_l), (3,g_h), (t\geq 4,g_l)
$$

We think of these 6 states as corresponding to $s=1,2,3,4,5,6$.

The transition matrix is

$$
P = \begin{pmatrix}
  0 & 1 & 0 & 0   & 0   & 0\\
  0 & 0 & 1 & 0   & 0   & 0\\
  0 & 0 & 0 & 0.5 & 0.5 & 0\\
  0 & 0 & 0 & 0   & 0   & 1\\
  0 & 0 & 0 & 0   & 0   & 1\\
  0 & 0 & 0 & 0   & 0   & 1
\end{pmatrix}
$$

The government expenditure at  each state is

$$
g = \left(\begin{matrix} 0.1\\0.1\\0.1\\0.1\\0.2\\0.1 \end{matrix}\right)
$$

We assume the same utility parameters as in the {doc}`Lucas-Stokey economy <opt_tax_recur>`.

This utility function is implemented in the following class.

```
---
lineno-start: 1
---
import numpy as np


class CRRAutility:

    def __init__(self,
                 β=0.9,
                 σ=2,
                 γ=2,
                 π=0.5*np.ones((2, 2)),
                 G=np.array([0.1, 0.2]),
                 Θ=np.ones(2),
                 transfers=False):

        self.β, self.σ, self.γ = β, σ, γ
        self.π, self.G, self.Θ, self.transfers = π, G, Θ, transfers

    # Utility function
    def U(self, c, n):
        σ = self.σ
        if σ == 1.:
            U = np.log(c)
        else:
            U = (c**(1 - σ) - 1) / (1 - σ)
        return U - n**(1 + self.γ) / (1 + self.γ)

    # Derivatives of utility function
    def Uc(self, c, n):
        return c**(-self.σ)

    def Ucc(self, c, n):
        return -self.σ * c**(-self.σ - 1)

    def Un(self, c, n):
        return -n**self.γ

    def Unn(self, c, n):
        return -self.γ * n**(self.γ - 1)
```

The following figure plots the Ramsey plan under both complete and incomplete
markets for both possible realizations of the state at time $t=3$.

Optimal policies when  the government has  access to state contingent debt are
represented by black lines, while the optimal policies when there is only a risk-free bond are in red.

Paths with circles are histories in which there is peace, while those with
triangle denote war.

```{code-cell} python3
# Initialize μgrid for value function iteration
μ_grid = np.linspace(-0.7, 0.01, 200)

time_example = CRRAutility()

time_example.π = np.array([[0, 1, 0,   0,   0,  0],
                           [0, 0, 1,   0,   0,  0],
                           [0, 0, 0, 0.5, 0.5,  0],
                           [0, 0, 0,   0,   0,  1],
                           [0, 0, 0,   0,   0,  1],
                           [0, 0, 0,   0,   0,  1]])

time_example.G = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1])
time_example.Θ = np.ones(6)  # Θ can in principle be random

time_example.transfers = True             # Government can use transfers
# Solve sequential problem
time_sequential = SequentialAllocation(time_example)
# Solve recursive problem
time_bellman = RecursiveAllocationAMSS(time_example, μ_grid)

sHist_h = np.array([0, 1, 2, 3, 5, 5, 5])
sHist_l = np.array([0, 1, 2, 4, 5, 5, 5])

sim_seq_h = time_sequential.simulate(1, 0, 7, sHist_h)
sim_bel_h = time_bellman.simulate(1, 0, 7, sHist_h)
sim_seq_l = time_sequential.simulate(1, 0, 7, sHist_l)
sim_bel_l = time_bellman.simulate(1, 0, 7, sHist_l)

# Government spending paths
sim_seq_l[4] = time_example.G[sHist_l]
sim_seq_h[4] = time_example.G[sHist_h]
sim_bel_l[4] = time_example.G[sHist_l]
sim_bel_h[4] = time_example.G[sHist_h]

# Output paths
sim_seq_l[5] = time_example.Θ[sHist_l] * sim_seq_l[1]
sim_seq_h[5] = time_example.Θ[sHist_h] * sim_seq_h[1]
sim_bel_l[5] = time_example.Θ[sHist_l] * sim_bel_l[1]
sim_bel_h[5] = time_example.Θ[sHist_h] * sim_bel_h[1]


fig, axes = plt.subplots(3, 2, figsize=(14, 10))
titles = ['Consumption', 'Labor Supply', 'Government Debt',
          'Tax Rate', 'Government Spending', 'Output']

for ax, title, sim_l, sim_h, bel_l, bel_h in zip(axes.flatten(), titles,
                                                 sim_seq_l, sim_seq_h,
                                                 sim_bel_l, sim_bel_h):
    ax.plot(sim_l, '-ok', sim_h, '-^k', bel_l, '-or', bel_h, '-^r', alpha=0.7)
    ax.set(title=title)
    ax.grid()

plt.tight_layout()
plt.show()
```

How a Ramsey planner responds to  war depends on the structure of the asset market.

If it is able to trade state-contingent debt, then at time $t=2$

* the government purchases an Arrow security that pays off when $g_3 = g_h$
* the government sells an Arrow security that  pays off when $g_3 = g_l$
* These purchases are designed in such a way that regardless of whether or not there is a war at $t=3$, the government will begin  period $t=4$ with the *same* government debt

This pattern facilities smoothing tax rates across  states.

The government without state contingent debt cannot do this.

Instead, it must enter   time $t=3$ with the same level of debt falling due whether there is peace or war at $t=3$.

It responds to this constraint by smoothing tax rates across time.

To finance a war it raises taxes and issues more debt.

To service the additional debt burden, it raises taxes in all future periods.

The absence of state contingent debt leads to an important difference in the
optimal tax policy.

When the Ramsey planner has access to state contingent debt, the optimal tax
policy is history independent

* the tax rate is a function  of the current level of government spending only,
  given the Lagrange multiplier on the implementability constraint

Without state contingent debt, the optimal tax rate is history dependent.

* A war at time $t=3$ causes a permanent increase in the tax rate.

#### Perpetual War Alert

History dependence occurs more dramatically in a case in which the government
perpetually faces the prospect  of war.

This case was studied in the final example of the lecture on
{doc}`optimal taxation with state-contingent debt <opt_tax_recur>`.

There, each period the government faces a constant probability, $0.5$, of war.

In addition, this example features the following preferences

$$
u(c,n) = \log(c) + 0.69 \log(1-n)
$$

In accordance, we will re-define our utility function.

```
---
lineno-start: 1
---
import numpy as np

class LogUtility:

    def __init__(self,
                 β=0.9,
                 ψ=0.69,
                 π=0.5*np.ones((2, 2)),
                 G=np.array([0.1, 0.2]),
                 Θ=np.ones(2),
                 transfers=False):

        self.β, self.ψ, self.π = β, ψ, π
        self.G, self.Θ, self.transfers = G, Θ, transfers

    # Utility function
    def U(self, c, n):
        return np.log(c) + self.ψ * np.log(1 - n)

    # Derivatives of utility function
    def Uc(self, c, n):
        return 1 / c

    def Ucc(self, c, n):
        return -c**(-2)

    def Un(self, c, n):
        return -self.ψ / (1 - n)

    def Unn(self, c, n):
        return -self.ψ / (1 - n)**2
```

With these preferences, Ramsey tax rates will vary even in the Lucas-Stokey
model with state-contingent debt.

The figure below plots optimal tax policies for both the economy with
state contingent debt (circles) and the economy with only a risk-free bond
(triangles).

```{code-cell} python3
log_example = LogUtility()
log_example.transfers = True                    # Government can use transfers
log_sequential = SequentialAllocation(log_example)  # Solve sequential problem
log_bellman = RecursiveAllocationAMSS(log_example, μ_grid)

T = 20
sHist = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  0, 0, 0, 1, 1, 1, 1, 1, 1, 0])

# Simulate
sim_seq = log_sequential.simulate(0.5, 0, T, sHist)
sim_bel = log_bellman.simulate(0.5, 0, T, sHist)

titles = ['Consumption', 'Labor Supply', 'Government Debt',
          'Tax Rate', 'Government Spending', 'Output']

# Government spending paths
sim_seq[4] = log_example.G[sHist]
sim_bel[4] = log_example.G[sHist]

# Output paths
sim_seq[5] = log_example.Θ[sHist] * sim_seq[1]
sim_bel[5] = log_example.Θ[sHist] * sim_bel[1]

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for ax, title, seq, bel in zip(axes.flatten(), titles, sim_seq, sim_bel):
    ax.plot(seq, '-ok', bel, '-^b')
    ax.set(title=title)
    ax.grid()

axes[0, 0].legend(('Complete Markets', 'Incomplete Markets'))
plt.tight_layout()
plt.show()
```

When the government experiences a prolonged period of peace, it is able to reduce
government debt and set permanently lower tax rates.

However, the government  finances a long war by borrowing and raising taxes.

This results in a drift away from  policies with state contingent debt that
depends on the history of shocks.

This is even more evident in the following figure that plots the evolution of
the two policies over 200 periods.

```{code-cell} python3
T = 200  # Set T to 200 periods
sim_seq_long = log_sequential.simulate(0.5, 0, T)
sHist_long = sim_seq_long[-3]
sim_bel_long = log_bellman.simulate(0.5, 0, T, sHist_long)

titles = ['Consumption', 'Labor Supply', 'Government Debt',
          'Tax Rate', 'Government Spending', 'Output']

# Government spending paths
sim_seq_long[4] = log_example.G[sHist_long]
sim_bel_long[4] = log_example.G[sHist_long]

# Output paths
sim_seq_long[5] = log_example.Θ[sHist_long] * sim_seq_long[1]
sim_bel_long[5] = log_example.Θ[sHist_long] * sim_bel_long[1]

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

for ax, title, seq, bel in zip(axes.flatten(), titles, sim_seq_long, \
        sim_bel_long):
    ax.plot(seq, '-k', bel, '-.b', alpha=0.5)
    ax.set(title=title)
    ax.grid()

axes[0, 0].legend(('Complete Markets','Incomplete Markets'))
plt.tight_layout()
plt.show()
```

[^fn_a]: In an allocation that solves the Ramsey problem and that levies distorting
taxes on labor, why would the government ever want to hand revenues back
to the private sector? It would not in an economy with state-contingent debt, since
any such allocation could be improved by lowering distortionary taxes
rather than handing out lump-sum transfers. But, without state-contingent
debt there can be circumstances when a government would like to make
lump-sum transfers to the private sector.

[^fn_b]: From the first-order conditions for the Ramsey
problem, there exists another realization $\tilde s^t$ with
the same history up until the previous period, i.e., $\tilde s^{t-1}=
s^{t-1}$, but where the multiplier on constraint {eq}`AMSS_46` takes  a positive value, so
$\gamma_t(\tilde s^t)>0$.

