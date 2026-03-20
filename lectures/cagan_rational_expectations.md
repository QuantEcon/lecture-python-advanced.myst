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

(cagan_rational_expectations)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Demand for Money during Hyperinflations under Rational Expectations

```{index} single: Hyperinflation; Cagan model; Rational expectations
```

```{contents} Contents
:depth: 2
```

## Overview

This lecture presents the analysis in {cite}`Sargent77hyper`, which proposes
methods for estimating the demand schedule for money that {cite:t}`Cagan` used
in his famous study of hyperinflation.

{cite:t}`SargentWallace73` pointed out that under assumptions making Cagan's
adaptive expectations scheme equivalent to rational expectations, Cagan's
estimator of $\alpha$ — the slope of log real balances with respect to expected
inflation — is not statistically consistent.

This inconsistency matters because of a **paradox** that emerged when Cagan used
his estimates of $\alpha$ to calculate the sustained rates of inflation that would
maximize the flow of real resources that money creators could command by printing
money.  

That "optimal" rate is $-1/\alpha$.  

For each of the seven hyperinflations
in his sample, the reciprocal of Cagan's estimate of $-\alpha$ turned out to be
less — and often very much less — than the actual average rate of inflation,
suggesting that the creators of money expanded the money supply at rates far
exceeding the revenue-maximizing rate.

A natural explanation is that this paradox is a **statistical artifact** — a
consequence of biased estimates of $\alpha$.

Table 1 reproduces the relevant data from Cagan.

| Country    | (1) $-1/\alpha$ | (2) $(e^{1/\alpha}-1)\times 100$ | (3) Avg. actual inflation |
|------------|:-----------:|:-------------------:|:------------------:|
| Austria    | .117        | 12                  | 47                 |
| Germany    | .183        | 20                  | 322                |
| Greece     | .244        | 28                  | 365                |
| Hungary I  | .115        | 12                  | 46                 |
| Hungary II | .236        | 32                  | 19,800             |
| Poland     | .435        | 54                  | 81                 |
| Russia     | .327        | 39                  | 57                 |

Column (1): $-1/\alpha$ (continuously compounded), the rate per month that maximizes
the revenue of the money creator.  Column (2): $(e^{1/\alpha}-1)\times 100$
(neglects compounding).  Column (3): average actual rate of inflation per month.

The paper pursues three goals:

1. **Characterize the asymptotic bias** in Cagan's ordinary-least-squares estimator
   under the rational expectations version of his model.
2. **Derive a consistent estimator** — a full-information maximum likelihood
   estimator — for the bivariate rational-expectations model.
3. **Test the model** by overfitting a more general vector autoregressive,
   moving-average representation and computing likelihood-ratio statistics.

Our  key  tools are **bivariate Wold representations**, **Granger
causality**, and **vector time series methods** following
{cite}`granger1969causality`, {cite}`sims1972money`, {cite}`wilson1973estimation` and  {cite}`anderson2011statistical`.


```{note}
This lecture can be viewed as a bivariate version of the ''reverse engineering'' exercise of 
{cite:t}`Muth1960` that we described in {doc}`this lecture <muth_kalman>`.
From a technical point of view this lecture is an exercise in applying **vector
time series models**.  The model is interesting because it illustrates the
difference between Granger causality and simple notions of one series *leading*
another. It also illustrates a difference between Granger causality and the separate notion of **invariance
with respect to an intervention**.
```

We begin with imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
```

## Cagan's Model under Rational Expectations

For background on the Cagan model see {doc}`intro:cagan_ree` and
{doc}`intro:cagan_adaptive`.

### Portfolio balance and adaptive expectations

Cagan's {cite}`Cagan` model builds on a demand schedule for real balances:

$$
m_t - p_t = \alpha \pi_t + u_t, \qquad \alpha < 0
$$ (eq:portfolio_balance_re)

where $m_t$ is the log money supply, $p_t$ is the log price level, $\pi_t$ is
the expected rate of inflation (the public's subjective expectation of
$p_{t+1} - p_t$), and $u_t$ is a mean-zero random disturbance.

Cagan assumed $\pi_t$ obeys the adaptive expectations scheme

$$
\pi_t = \frac{1-\lambda}{1-\lambda L}(p_t - p_{t-1})
$$ (eq:adaptive_re)

where $L$ is the lag operator, $L^k x_t = x_{t-k}$, and $0 < \lambda < 1$.

Let $x_t \equiv p_t - p_{t-1}$ be the inflation rate and $\mu_t \equiv m_t - m_{t-1}$
be the percentage rate of money creation.

### Rational expectations  


Cagan's model of hyperinflation builds on a demand schedule for real balances
of the form

```{math}
:label: eq1
m_t - p_t = \alpha \pi_t + u_t, \qquad \alpha < 0,
```

where $m$ is the log of the money supply (which is always equal to the log of
money demand); $p$ is the log of the price level; $\pi_t$ is the expected rate
of inflation, i.e., the public's psychological expectation of $p_{t+1} - p_t$;
and $u_t$ is a random variable with mean zero. 


```{note}
A constant term has been omitted
from {eq}`eq1`, though one would be included in empirical work.
```

Cagan assumed that $\pi_t$ was formed via the adaptive expectations scheme

```{math}
\pi_t = \frac{1-\lambda}{1-\lambda L}(p_t - p_{t-1}),
```

or

```{math}
:label: eq2
\pi_t = \frac{1-\lambda}{1-\lambda L}\, x_t,
```

where $x_t = p_t - p_{t-1}$ is the rate of inflation, and $L$ is the lag
operator defined by $L^n x_t = x_{t-n}$.

Under rational expectations we require that

```{math}
:label: eq3
\pi_t = E_t x_{t+1},
```

where $E_t x_{t+1}$ is the mathematical expectation of $x_{t+1}$ conditional on
information available as of time $t$.[^info]  Using {eq}`eq3` and recursions on
{eq}`eq1`, it is straightforward to show that under rational expectations we must
have[^deriv]

```{math}
:label: eq4
\pi_t = E_t x_{t+1}
= \frac{1}{1-\alpha}\sum_{j=1}^{\infty}
    \left(\frac{-\alpha}{1-\alpha}\right)^{\!j-1} E_t\mu_{t+j}
  - \frac{1}{1-\alpha}\sum_{j=1}^{\infty}
    \left(\frac{-\alpha}{1-\alpha}\right)^{\!j-1}
    \bigl(E_t u_{t+j} - E_t u_{t+j-1}\bigr),
```

where $\mu_t = m_t - m_{t-1}$ is the percentage rate of increase of the money
supply.  

Equation {eq}`eq4` characterises the (systematic part of the)
stochastic process for inflation as a function of the (systematic part of the)
stochastic process for money creation. 

The model asserts that {eq}`eq4` is
invariant with respect to interventions in the form of changes in the stochastic
process governing money creation.  

In this sense, since changes in the stochastic
process for money creation are supposed to produce predictable changes in the
stochastic process for inflation, money "causes" inflation.

For Cagan's adaptive expectations scheme {eq}`eq2` to be equivalent to rational
expectations we require:

```{math}
:label: eq5
\frac{1-\lambda}{1-\lambda L}\, x_t
= \frac{1}{1-\alpha}\sum_{j=1}^{\infty}
    \left(\frac{-\alpha}{1-\alpha}\right)^{\!j-1} E_t\mu_{t+j}
  - \frac{1}{1-\alpha}\sum_{j=1}^{\infty}
    \left(\frac{-\alpha}{1-\alpha}\right)^{\!j-1}
    \bigl(E_t u_{t+j} - E_t u_{t+j-1}\bigr).
```

The necessary and sufficient condition for {eq}`eq5` to hold for all $\alpha$ and
all $t$ is

```{math}
E_t\mu_{t+j} - E_t(u_{t+j} - u_{t+j-1}) = \frac{1-\lambda}{1-\lambda L}\, x_t.
```

For an arbitrary $\mu$ process there exists a disturbance process $u_t$
satisfying the above restriction, one in which $E_t(u_{t+j} - u_{t+j-1})$ is a
complicated function of lagged $x$'s and lagged $\mu$'s.  

The most fruitful conditions to impose, however, are the following two that are *sufficient*
(though clearly not necessary) to satisfy {eq}`eq5`.  The first condition is

```{math}
:label: eq6
u_t = u_{t-1} + \eta_t,
```

where $\eta_t$ is a serially uncorrelated random term with mean zero and variance
$\sigma_\eta^2$; we assume that
$E[\eta_t \mid u_{t-1}, \mu_{t-2}, \ldots, x_{t-1}, x_{t-2}, \ldots] = 0$.
According to {eq}`eq6`, $u$ follows a random walk.  

Equation {eq}`eq6` implies
that

```{math}
E_t u_{t+j} = u_t, \qquad j \geq 0,
```

which implies that

```{math}
E_t u_{t+j} - E_t u_{t+j-1} = 0 \quad \text{for all } j \geq 1.
```

The second of the pair of sufficient conditions for {eq}`eq5` is

```{math}
:label: eq7
E_t\mu_{t+j} = E_t\mu_{t+1} \quad \text{for } j > 1,
```

so that a constant rate of money creation is expected to occur over the entire
future. 

Assuming {eq}`eq6` and {eq}`eq7` then implies that the appropriate
version of {eq}`eq5` is

```{math}
\left(\frac{1-\lambda}{1-\lambda L}\right) x_t
= E_t\mu_{t+1}\cdot\frac{1}{1-\alpha}
  \sum_{j=1}^{\infty}\left(\frac{-\alpha}{1-\alpha}\right)^{\!j-1},
```

or

```{math}
:label: eq8
\frac{1-\lambda}{1-\lambda L}\, x_t = E_t\mu_{t+1}.
```

A process that satisfies {eq}`eq8` is[^footprocess]

```{math}
:label: eq9
\mu_t = \left(\frac{1-\lambda}{1-\lambda L}\right) x_t + \varepsilon_t
       \;(= E_t x_{t+1} + \varepsilon_t),
```

where $\varepsilon_t$ is a serially uncorrelated random term with mean zero and
variance $\sigma_\varepsilon^2$, and that satisfies

```{math}
E(\varepsilon_t \mid x_{t-1}, x_{t-2}, \ldots, \mu_{t-1}, \mu_{t-2}, \ldots) = 0.
```

According to {eq}`eq9`, the rate of money creation equals the expected rate of
inflation plus a random term. 

Equation {eq}`eq9`, which has been arrived at in a
purely mechanical fashion by pursuing the implications of the assumption that
Cagan's adaptive expectations scheme is rational, is nevertheless of interest as
an hypothesis about the government's behaviour. 

For example, if the government
is creating money to finance a large part of a roughly fixed rate of real
government purchases, then there is a presumption that inflation and expected
inflation will feed back into money creation, an implication with which {eq}`eq9`
is consistent.  

Thus, when $\pi_t$ increases, causing $m_t - p_t$ to fall and
thereby causing $p_t$ to rise with a fixed $m_t$, money depreciates in value,
prompting the creators of money to increase the rate of printing money in order
to maintain their command over the flow of real resources (see {cite:t}`SargentWallace73`).  

Alternatively, equation {eq}`eq9` is compatible with a "real
bills" regime in which the monetary authority sets out to supply whatever money
the public demands at some fixed nominal interest rate or some fixed real money
supply.  

Equation {eq}`eq9` looks like a rule in which the monetary authority is
attempting to peg the (rate of growth of the) real money supply.  

During the German hyperinflation, German monetary officials in effect repeatedly acknowledged
that they were operating under a real-bills regime, acknowledgements made in
efforts to argue that their actions were not causing the inflation but were merely
responses to it.

The foregoing establishes that if equations {eq}`eq6` and {eq}`eq9` obtain,
Cagan's adaptive expectations scheme is compatible with rational expectations and
with the portfolio balance condition that he assumed.  Under these assumptions,
inflation and money creation form a bivariate stochastic process given by

```{math}
:label: eq10
\mu_t - x_t = \alpha(1-L)\!\left(\frac{1-\lambda}{1-\lambda L}\right) x_t + \eta_t,
```

```{math}
:label: eq9b
\mu_t = \left(\frac{1-\lambda}{1-\lambda L}\right) x_t + \varepsilon_t.
```

Equation {eq}`eq10` was obtained by first differencing {eq}`eq1` and then
substituting for $\pi_t$ from {eq}`eq2` and for $u_t - u_{t-1}$ from {eq}`eq6`.


The process {eq}`eq10`–{eq}`eq9b` can be rewritten as

```{math}
:label: eq11
(1-L)\,x_t
= \bigl(\lambda + \alpha(1-\lambda)\bigr)^{-1}(1-\lambda L)(\varepsilon_t - \eta_t),
```

```{math}
:label: eq12
(1-L)\,\mu_t
= \bigl[\bigl(\lambda + \alpha(1-\lambda)\bigr)^{-1}(1-\lambda)(\varepsilon_t - \eta_t)
   - \varepsilon_{t-1} + \varepsilon_t\bigr].
```

Equations {eq}`eq11` and {eq}`eq12` can be derived directly from {eq}`eq10` and
{eq}`eq9b`; alternatively, see {cite:t}`SargentWallace73` for a somewhat different
but equivalent way of deriving {eq}`eq11` and {eq}`eq12`.

We assume that the information available consists (at least) of
observations of current and past $p$'s and current and past $x$'s.  

Thus, 
$E_t x_{t+1} \equiv E[x_{t+1} \mid \mu_t, \mu_{t-1}, \ldots, x_t, x_{t-1}, \ldots]$.

Similarly, for $z_t$ any arbitrary random variable,
$E_t z_{t+1}$ denotes $E[z_{t+1} \mid \mu_t, \mu_{t-1}, \ldots, x_t, x_{t-1}, \ldots]$.

Substituting {eq}`eq3` into {eq}`eq1`, first differencing, and
shifting the time subscripts forward one period gives

$$ \mu_{t+1} - x_{t+1} = \alpha E_{t+1} x_{t+2} - \alpha x_{t+1} + (u_{t+1} - u_t). $$

Taking expectations conditional on information available at $t$ gives

$$
E_t x_{t+1} = \dfrac{1}{1-\alpha}\, E_t\mu_{t+1}
  - \dfrac{\alpha}{1-\alpha}\, E_t x_{t+2}
  - (E_t u_{t+1} - E_t u_t). 
$$

Recursion on this difference equation shows that equation {eq}`eq4` is indeed a
solution.

The sum 

$$
\dfrac{1}{1-\alpha}\displaystyle\sum_{j=1}^{\infty}
\!\left(\dfrac{-\alpha}{1-\alpha}\right)^{j-1}
$$

equals $1$ (a consequence of a geometric series with
ratio $-\alpha/(1-\alpha)$ that  satisfies $|-\alpha/(1-\alpha)| < 1$ when
$\alpha \in (-1, 0)$), yielding $E_t\mu_{t+1}$ on the right-hand side of {eq}`eq8`.

To see that process {eq}`eq9` satisfies {eq}`eq8`, write {eq}`eq9`
as

$$ \mu_{t+1} = (1-\lambda)\,x_{t+1}
  + \dfrac{(1-\lambda)\lambda}{1-\lambda L}\,x_t + \varepsilon_{t+1}
$$

Taking expectations conditional on information available at $t$:

$$ 
E_t\mu_{t+1} = (1-\lambda)\,E_t x_{t+1}
  + \dfrac{(1-\lambda)\lambda}{1-\lambda L}\,x_t 
$$
  
But $E_t x_{t+1} = \dfrac{1-\lambda}{1-\lambda L}\,x_t$, so

$$ 
E_t\mu_{t+1}
  = \bigl((1-\lambda) + \lambda\bigr)\!\left(\dfrac{1-\lambda}{1-\lambda L}\right) x_t
  = \dfrac{1-\lambda}{1-\lambda L}\,x_t, 
$$ 
  
as required.


In summary, under rational expectations we require

$$
\pi_t = E_t x_{t+1}
$$ (eq:rational_expectations)

where $E_t x_{t+1}$ is the mathematical expectation of $x_{t+1}$ conditional on
information available at time $t$.

Using {eq}`eq:rational_expectations` and recursions on {eq}`eq:portfolio_balance_re`,
it is straightforward to show that under rational expectations

$$
\pi_t = E_t x_{t+1} = -\frac{1}{\alpha}
       \sum_{j=0}^{\infty} \left(\frac{-1}{\alpha}\right)^{j-1}
       \bigl(E_t \mu_{t+j} - E_t \mu_{t+j-1}\bigr) -
       \sum_{j \geq 1} \left(\frac{-1}{\alpha}\right)^{j-1}
       (E_t u_{t+j} - E_t u_{t+j-1})
$$ (eq:rational_pi_general)

Equation {eq}`eq:rational_pi_general` characterizes the stochastic process for
inflation as a function of the stochastic process for money creation. 

The model asserts that {eq}`eq:rational_pi_general` is **invariant** with respect to
interventions in the form of changes in the money supply process.

For Cagan's adaptive scheme {eq}`eq:adaptive_re` to be equivalent to rational
expectations {eq}`eq:rational_expectations` requires

$$
\frac{1-\lambda}{1-\lambda L} x_t =
  E_t x_{t+1} - \sum_{j \geq 1}
  \left(\frac{-1}{\alpha}\right)^{j-1}
  (E_t u_{t+j} - E_t u_{t+j-1}).
$$ (eq:equivalence_condition)

### Two sufficient conditions

Necessary and sufficient conditions for {eq}`eq:equivalence_condition` to hold
for all $\alpha$ and all $t$ are subtle. 

A tractable pair of **sufficient**
conditions is:

**Condition 1.** The portfolio disturbance $u_t$ follows a random walk:

$$
u_t = u_{t-1} + \varepsilon_t
$$ (eq:random_walk_u)

where $\varepsilon_t$ is serially uncorrelated with
$E[\varepsilon_t | u_{t-1}, u_{t-2}, \ldots, x_t, x_{t-1}, \ldots] = 0$.
This implies $E_t u_{t+j} = u_t$ for $j \geq 0$, so that
$E_t u_{t+j} - E_t u_{t+j-1} = 0$ for all $j \geq 1$.

**Condition 2.** A constant rate of money creation is expected one period
ahead after the current one:

$$
E_t \mu_{t+j} = E_t \mu_{t+1}, \qquad j \geq 1.
$$ (eq:constant_mu_condition)

Under these two conditions, the appropriate rational-expectations formula reduces to:

$$
\pi_t = E_t x_{t+1} = \frac{1-\lambda}{1-\lambda L} x_t
$$ (eq:rational_equals_adaptive)

A process that satisfies {eq}`eq:rational_equals_adaptive` is

$$
\mu_t = \frac{1-\lambda}{1-\lambda L} x_t + \delta_t
$$ (eq:money_supply_rule)

where $\delta_t$ is a serially uncorrelated random variable satisfying
$E[\delta_t | x_{t-1}, x_{t-2}, \ldots, \mu_{t-1}, \mu_{t-2}, \ldots] = 0$.

Equation {eq}`eq:money_supply_rule` is the **money supply rule** implied by the
model. 

It says the rate of money creation equals expected inflation plus a random
term. 

This is consistent with a "real bills" regime in which the monetary
authority accommodates inflation, or with a government creating money to finance a
roughly fixed level of real purchases.

### The bivariate process for inflation and money creation

If {eq}`eq:random_walk_u` and {eq}`eq:money_supply_rule` hold, equations (10)
and (9) of the paper give the following bivariate system:

$$
\mu_t - x_t = \alpha(1-\lambda L) x_t + u_t - u_{t-1}
$$ (eq:system_eq10)

$$
\mu_t = \frac{1-\lambda}{1-\lambda L} x_t + \delta_t.
$$ (eq:system_eq9)

These can be rewritten as a **bivariate first-difference MA(1)** process:

$$
(1-L)x_t = (\lambda + \alpha(1-\lambda))^{-1}(1-\lambda L)(\delta_t - \varepsilon_t)
$$ (eq:bivariate_x)

$$
(1-L)\mu_t = (\lambda + \alpha(1-\lambda))^{-1}(1-\lambda)(\delta_t - \varepsilon_t)
             - \delta_{t-1} + \delta_t.
$$ (eq:bivariate_mu)

Equations {eq}`eq:bivariate_x`–{eq}`eq:bivariate_mu` show that the first
difference of inflation $(1-L)x_t$ and the first difference of percentage money
creation $(1-L)\mu_t$ are **correlated MA(1) processes**.

```{note}
The statistical model {eq}`eq:bivariate_x`–{eq}`eq:bivariate_mu` was constructed
to guarantee $E_{t-1} x_t = \frac{1-\lambda}{1-\lambda L} x_{t-1}$, which is the
condition that $\mu$ does **not** Granger-cause $x$.  

Thus, once lagged $x$'s
are taken into account, lagged $\mu$'s do not help predict current $x$.
```

```{code-cell} ipython3
def bivariate_ma1_moments(α, λ, σ_δ2=1.0, σ_ε2=0.5, σ_δε=0.0):
    """
    Compute the autocovariances of (1-L)x and (1-L)μ under the
    bivariate rational-expectations model of Sargent (1977).

    Parameters
    ----------
    α  : float  (< 0)  demand semi-elasticity
    λ  : float  (0 < λ < 1)  adaptive expectations parameter
    σ_δ2 : variance of money-supply shock δ_t
    σ_ε2 : variance of portfolio shock ε_t
    σ_δε : covariance of δ_t and ε_t

    Returns
    -------
    cxx : dict with keys 0, 1  — autocovariances of Δx
    cμμ : dict with keys 0, 1  — autocovariances of Δμ
    cxμ : dict with keys -1, 0, 1  — cross-covariances E[Δx_t Δμ_{t-τ}]
    """
    denom = λ + α * (1.0 - λ)
    if np.isclose(denom, 0.0):
        raise ValueError("λ + α(1-λ) must be nonzero.")
    φ = 1.0 / denom

    # Coefficients for Δx_t = φ(1 - λL)(δ_t - ε_t)
    # Δμ_t = φ(1-λ)(δ_t - ε_t) + (1-L)δ_t  =>  Δμ_t = [φ(1-λ)+1]δ_t - φ(1-λ)ε_t - δ_{t-1}

    A = φ * (1.0 - λ) + 1.0   # coefficient on δ_t in Δμ
    B = φ * (1.0 - λ)         # coefficient on ε_t (with minus sign) in Δμ

    # Δx: moving-average coefficients in (δ_t - ε_t): φ at lag 0, -φλ at lag 1
    # Autocovariances of Δx
    cx0 = φ**2 * (1 + λ**2) * (σ_δ2 - 2*σ_δε + σ_ε2)
    cx1 = -φ**2 * λ * (σ_δ2 - 2*σ_δε + σ_ε2)

    # Autocovariances of Δμ
    cμ0 = (A**2 + 1.0)*σ_δ2 + B**2*σ_ε2 - 2.0*A*B*σ_δε
    cμ1 = -A * σ_δ2

    # Cross-covariances r(τ) = E[Δx_t  Δμ_{t-τ}]
    rxμ_0   = φ*(A + λ)*σ_δ2 + φ*B*σ_ε2 - φ*(A + B + λ)*σ_δε
    rxμ_1   = -φ*λ*A*σ_δ2 - φ*λ*B*σ_ε2 + φ*λ*(A + B)*σ_δε
    rxμ_m1  = -φ*(σ_δ2 - σ_δε)

    cxx = {0: cx0, 1: cx1}
    cμμ = {0: cμ0, 1: cμ1}
    cxμ = {-1: rxμ_m1, 0: rxμ_0, 1: rxμ_1}
    return cxx, cμμ, cxμ
```

## Bias in Cagan's Estimator

### Bivariate Wold representation

A convenient way to evaluate the asymptotic bias in Cagan's estimator is to obtain
a bivariate Wold representation for $(\Delta x_t, \Delta \mu_t)$.

Decompose $\delta_t$ as

$$
\delta_t = \rho(\delta_t - \varepsilon_t) + v_t
$$ (eq:decompose_delta)

where $\rho$ is the regression coefficient of $\delta_t$ on $(\delta_t - \varepsilon_t)$:

$$
\rho = \frac{E[\delta_t(\delta_t - \varepsilon_t)]}{E[(\delta_t - \varepsilon_t)^2]}
     = \frac{\sigma_\delta^2 - \sigma_{\delta\varepsilon}}
            {\sigma_\delta^2 - 2\sigma_{\delta\varepsilon} + \sigma_\varepsilon^2}
$$ (eq:rho)

and $v_t$ is orthogonal to $(\delta_t - \varepsilon_t)$ by construction.

Substituting {eq}`eq:decompose_delta` into {eq}`eq:bivariate_mu` and using
{eq}`eq:bivariate_x` gives the **triangular bivariate Wold representation**:

$$
(1-L)x_t = \phi(1-\lambda L)(\delta_t - \varepsilon_t)
$$ (eq:wold_x)

$$
(1-L)\mu_t = [\phi(1-\lambda) + \rho(1-L)](\delta_t - \varepsilon_t) + (1-L)v_t
$$ (eq:wold_mu)

with fundamental noises $(\delta_t - \varepsilon_t)$ and $v_t$.

The triangular structure confirms that $\Delta x$ is **econometrically exogenous**
with respect to $\Delta\mu$, and that $\Delta x$ Granger-causes $\Delta\mu$ but not
vice versa.

### The population regression (Cagan's estimator)

Because $x$ is exogenous with respect to $\mu - x$, one can obtain the one-sided
projection of $\mu_t$ on current and past $x_t$'s.

Substituting {eq}`eq:wold_x` into {eq}`eq:wold_mu` and integrating (operating
with $(1-L)^{-1}$), the projection of $\mu_t - x_t$ against $\{x_t, x_{t-1}, \ldots\}$
is

$$
\mu_t - x_t = \left[\rho\alpha + \frac{\rho(1-\lambda)}{1-\lambda L}\right] x_t + \tilde{u}_t
$$ (eq:cagan_pop_regression)

where $\tilde{u}_t = \tilde{u}_{t-1} + v_t$ follows a random walk orthogonal to
the $x$ process.

Now Cagan regarded this population projection as giving estimates of the
structural equation

$$
m_t - p_t = \alpha \pi_t + u_t = \frac{\alpha(1-\lambda)}{1-\lambda L} x_t + u_t.
$$ (eq:structural_cagan)

Comparing {eq}`eq:cagan_pop_regression` with the corresponding structural form
shows that:

- Cagan's estimator of **$\lambda$ is consistent**.
- Cagan's estimator of **$\alpha$ is not consistent** in general, and obeys

$$
\operatorname{plim} \hat\alpha = \rho\alpha + (1-\rho)\frac{\rho}{\phi}
$$ (eq:plim_alpha)

where $\phi = (\lambda + \alpha(1-\lambda))^{-1}$.

If $\rho = 0$ (which holds when $\sigma_\varepsilon = 0$, i.e., no portfolio
shocks), then $\operatorname{plim} \hat\alpha = 0$.

If $\varepsilon_t = 0$ for all $t$ (no noise in portfolio balance), then $\rho = 1$
and $\operatorname{plim} \hat\alpha = \alpha$, so Cagan's estimator is consistent
in this special case.

```{note}
It is noteworthy that the residuals in {eq}`eq:cagan_pop_regression` follow a
random walk.  

{cite:t}`Cagan` and  {cite:t}`barro1970inflation` both reported highly serially correlated residuals
and very low Durbin–Watson statistics, which is consistent with this prediction.
```

```{code-cell} ipython3
def rho_from_moments(σ_δ2, σ_ε2, σ_δε):
    """
    Compute ρ = Cov(δ, δ-ε) / Var(δ-ε), the coefficient in the
    decomposition δ_t = ρ(δ_t - ε_t) + v_t.
    """
    var_diff = σ_δ2 - 2.0 * σ_δε + σ_ε2
    if np.isclose(var_diff, 0.0):
        return 1.0
    return (σ_δ2 - σ_δε) / var_diff


def plim_alpha_cagan(α, λ, σ_δ2=1.0, σ_ε2=0.5, σ_δε=0.0):
    """
    Asymptotic limit (population value) of Cagan's OLS estimator of α.
    """
    denom = λ + α * (1.0 - λ)
    φ = 1.0 / denom
    ρ = rho_from_moments(σ_δ2, σ_ε2, σ_δε)
    return ρ * α + (1.0 - ρ) * ρ / φ
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Bias of Cagan's OLS estimator of $\alpha$
    name: fig-cagan-bias
---
α_grid = np.linspace(-0.9, -0.05, 300)
λ = 0.5
σ_δ2, σ_ε2, σ_δε = 1.0, 0.5, 0.0

ρ = rho_from_moments(σ_δ2, σ_ε2, σ_δε)
plims = [plim_alpha_cagan(a, λ, σ_δ2, σ_ε2, σ_δε) for a in α_grid]

fig, ax = plt.subplots()
ax.plot(α_grid, α_grid, 'k--', lw=1.5, label='No bias (45° line)')
ax.plot(α_grid, plims, lw=2, label=rf'$\operatorname{{plim}}\hat\alpha$, $\lambda={λ}$')
ax.set_xlabel(r'True $\alpha$')
ax.set_ylabel(r'$\operatorname{plim}\hat\alpha$')
ax.legend()
ax.set_title("Asymptotic bias of Cagan's OLS estimator")
plt.tight_layout()
plt.show()
```

The figure confirms that Cagan's estimator is biased toward zero whenever
$\rho < 1$, i.e., whenever there are both monetary shocks and portfolio shocks.

The bias disappears only in the special case $\sigma_\varepsilon^2 = 0$.


## Consistent Estimator


Equations (11) and (12) form a bivariate first-order moving average process in
$(1-L)\mu_t$ and $(1-L)x_t$.  

Assuming that the white noises $\varepsilon_t$ and
$\eta_t$ are jointly normally distributed, the likelihood function of a sample of
length $T$ observations, $t = 1, \ldots, T$, generated by (11)–(12) can be written
down. 

To apply the method of maximum likelihood, it is most convenient to write
the model in its vector autoregressive form.

First note that from (9) we can write

```{math}
:label: eq23
\varepsilon_t = \mu_t - \frac{1-\lambda}{1-\lambda L}\, x_t .
```

Next from (11) we have

```{math}
:label: eq24
\varepsilon_t - \eta_t = \frac{(\lambda + \alpha(1-\lambda))(1-L)}{1-\lambda L}\, x_t .
```

Substituting {eq}`eq24` into {eq}`eq23` and rearranging gives

```{math}
:label: eq25
\eta_t = \mu_t
  - \left(\frac{1-\lambda + (\lambda + \alpha(1-\lambda))(1-L)}{1-\lambda L}\right) x_t .
```

In vector notation equations {eq}`eq23` and {eq}`eq25` can be written

```{math}
\begin{bmatrix} \varepsilon_t \\ \eta_t \end{bmatrix}
=
\begin{bmatrix}
  -\dfrac{(1-\lambda)}{1-\lambda L} & 1 \\[8pt]
  -\dfrac{1-\lambda + (\lambda+\alpha(1-\lambda))(1-L)}{1-\lambda L} & 1
\end{bmatrix}
\begin{bmatrix} x_t \\ \mu_t \end{bmatrix} .
```

Multiplying both sides of the equation by $(1-\lambda L)\cdot I$, where $I$ is the
$2\times 2$ identity matrix, gives

```{math}
\begin{bmatrix} (1-\lambda L)\varepsilon_t \\ (1-\lambda L)\eta_t \end{bmatrix}
=
\begin{bmatrix}
  -(1-\lambda) & 1-\lambda L \\
  -[1-\lambda+(\lambda+\alpha(1-\lambda))(1-L)] & 1-\lambda L
\end{bmatrix}
\begin{bmatrix} x_t \\ \mu_t \end{bmatrix} ,
```

or equivalently

```{math}
\begin{bmatrix} \varepsilon_t \\ \eta_t \end{bmatrix}
- \lambda I
\begin{bmatrix} \varepsilon_{t-1} \\ \eta_{t-1} \end{bmatrix}
=
\begin{bmatrix}
  -(1-\lambda) & 1 \\
  -(1+\alpha(1-\lambda)) & 1
\end{bmatrix}
\begin{bmatrix} x_t \\ \mu_t \end{bmatrix}
+
\begin{bmatrix}
  0 & -\lambda \\
  \lambda+\alpha(1-\lambda) & -\lambda
\end{bmatrix}
\begin{bmatrix} x_{t-1} \\ \mu_{t-1} \end{bmatrix} .
```

Let

```{math}
G_0 =
\begin{bmatrix}
  -(1-\lambda) & 1 \\
  -(1+\alpha(1-\lambda)) & 1
\end{bmatrix} .
```

Premultiplying the preceding equation by

```{math}
G_0^{-1} =
\frac{1}{\lambda+\alpha(1-\lambda)}
\begin{bmatrix}
  1 & -1 \\
  1+\alpha(1-\lambda) & -(1-\lambda)
\end{bmatrix}
```

gives

```{math}
G_0^{-1}
\begin{bmatrix} \varepsilon_t \\ \eta_t \end{bmatrix}
- \lambda I\, G_0^{-1}
\begin{bmatrix} \varepsilon_{t-1} \\ \eta_{t-1} \end{bmatrix}
=
\begin{bmatrix} x_t \\ \mu_t \end{bmatrix}
+ G_0^{-1}
\begin{bmatrix} 0 & -\lambda \\ \lambda+\alpha(1-\lambda) & -\lambda \end{bmatrix}
\begin{bmatrix} x_{t-1} \\ \mu_{t-1} \end{bmatrix} ,
```

or

```{math}
:label: eq26
\begin{bmatrix} a_{1t} \\ a_{2t} \end{bmatrix}
- \lambda I
\begin{bmatrix} a_{1,t-1} \\ a_{2,t-1} \end{bmatrix}
=
\begin{bmatrix} x_t \\ \mu_t \end{bmatrix}
+ G_0^{-1}
\begin{bmatrix} 0 & -\lambda \\ \lambda+\alpha(1-\lambda) & -\lambda \end{bmatrix}
\begin{bmatrix} x_{t-1} \\ \mu_{t-1} \end{bmatrix} ,
```

where

```{math}
\begin{bmatrix} a_{1t} \\ a_{2t} \end{bmatrix}
\equiv G_0^{-1}
\begin{bmatrix} \varepsilon_t \\ \eta_t \end{bmatrix} .
```

Computing $G_0^{-1}\begin{bmatrix}0&-\lambda\\\lambda+\alpha(1-\lambda)&-\lambda\end{bmatrix}$
explicitly and rearranging the above equation gives

```{math}
:label: eq27
\begin{bmatrix} x_t \\ \mu_t \end{bmatrix}
=
\begin{pmatrix} 1 & 0 \\ 1-\lambda & \lambda \end{pmatrix}
\begin{bmatrix} x_{t-1} \\ \mu_{t-1} \end{bmatrix}
+
\begin{bmatrix} a_{1t} \\ a_{2t} \end{bmatrix}
- \lambda I
\begin{bmatrix} a_{1,t-1} \\ a_{2,t-1} \end{bmatrix} .
```

Equation {eq}`eq27` is a vector first-order autoregression, first-order moving
average process. 

The random variables $a_{1t}$, $a_{2t}$ are the innovations in
the $x$ and $\mu$ processes, respectively — the one-period-ahead forecasting errors
for $x_t$ and $\mu_t$.  

The $a$'s are related to the $\varepsilon$'s and $\eta$'s
appearing in the structural equations of the model by

```{math}
:label: eq28
\begin{bmatrix} a_{1t} \\ a_{2t} \end{bmatrix}
=
\begin{bmatrix}
  \dfrac{1}{\lambda+\alpha(1-\lambda)}\,(\varepsilon_t - \eta_t) \\[10pt]
  \dfrac{1-\lambda}{\lambda+\alpha(1-\lambda)}\,(\varepsilon_t - \eta_t) + \varepsilon_t
\end{bmatrix} .
```

Notice that the first equation of {eq}`eq27` can be written as

```{math}
(1-L)\,x_t = (1-\lambda L)\,a_{1t} .
```

It is straightforward to write this in the autoregressive form

```{math}
:label: eq29
x_t = \left(\frac{1-\lambda}{1-\lambda L}\right) x_{t-1} + a_{1t} .
```

Since $E_{t-1}a_{1t} = 0$, we have

```{math}
E_{t-1}x_t = \left(\frac{1-\lambda}{1-\lambda L}\right) x_{t-1} .
```

The second equation of {eq}`eq27` can be written as

```{math}
(1-\lambda L)\,\mu_t = (1-\lambda)\,x_{t-1} + (1-\lambda L)\,a_{2t} .
```

But from {eq}`eq29` we have
$(1-\lambda)x_{t-1} = (1-\lambda L)x_t - (1-\lambda L)a_{1t}$,
which when substituted into the above equation gives

```{math}
(1-\lambda L)\,\mu_t
= (1-\lambda L)\,x_t - (1-\lambda L)\,a_{1t} + (1-\lambda L)\,a_{2t},
```

or

```{math}
:label: eq30
\mu_t = x_t + a_{2t} - a_{1t} .
```

From {eq}`eq30` it follows that

```{math}
:label: eq31
E_{t-1}\mu_t = E_{t-1}x_t .
```

The triangular character of representation {eq}`eq27` demonstrates that $\mu$ does
not "cause" in Granger's sense (i.e., help predict, once lagged own values are taken
into account) the variable $x$. 

Thus, $x$ is econometrically exogenous with
respect to $\mu$.[^sims] 

On the other hand, $x_t$ does cause the variable $\mu_t$.
Even stronger, the model implies that
$E_{t-1}\mu_t = E_{t-1}x_t = \left(\tfrac{1-\lambda}{1-\lambda L}\right)x_{t-1}$,
so that lagged $\mu$'s don't help predict $\mu$ once lagged $x$'s are taken into
account.  

That $x$ causes $\mu$ in Granger's sense is not to be confused with $x$'s
"leading" $\mu$ in any National Bureau sense. 

On the contrary, according to
{eq}`eq30`, $x_t$ and $\mu_t$ are "in phase" with one another, neither one leading
the other.  
 
  * The phase of their cross-spectrum equals zero at all frequencies.

Evidence that $x$ leads $\mu$ would not be consistent with the model being studied
here.

[^sims]: {cite}`sims1972money` proved the equivalence of Granger causality (see {cite}`granger1969causality`) with econometric exogeneity.

The vector autoregressive, moving average process {eq}`eq27` is in a form that can
be estimated by the maximum likelihood estimator described by {cite:t}`wilson1973estimation`

 It is
essential that the matrices multiplying current
$\begin{bmatrix}a_{1t}\\a_{2t}\end{bmatrix}$ and current
$\begin{bmatrix}x_t\\\mu_t\end{bmatrix}$ both be identity matrices in order to
apply the method, so that each $a_i$ process can be interpreted as the residual from
a vector autoregression either for $\mu_t$ or $x_t$.

Let

```{math}
a_t = \begin{bmatrix} a_{1t} \\ a_{2t} \end{bmatrix} ,
```

and let $D_a$ be the covariance matrix of $a_t$,

```{math}
D_a =
\begin{bmatrix} \sigma_{11} & \sigma_{12} \\ \sigma_{12} & \sigma_{22} \end{bmatrix}
= E\, a_t a_t' .
```

The likelihood function of the sample $t = 1, \ldots, T$ can now be written as

```{math}
:label: eq32
L(\lambda,\,\sigma_{11},\,\sigma_{12},\,\sigma_{22}\mid\mu_t,\,x_t)
= (2\pi)^{-T}\,|D_a|^{-T/2}
  \exp\!\left(-\tfrac{1}{2}\sum_{t=1}^{T} a_t' D_a^{-1} a_t\right).
```

Given initial values for $(a_{10}, a_{20})$ — equivalently for $(\varepsilon_0,
\eta_0)$ — and given a value of $\lambda$, equation {eq}`eq26` or {eq}`eq27` can be
used to solve for $a_t$, $t = 1, \ldots, T$.  (We take $a_{10} = a_{20} = 0$.)

{cite}`wilson1973estimation` notes that maximizing {eq}`eq32` is equivalent to minimizing with respect to
$\lambda$ the determinant of the estimated covariance matrix of the $a_t$'s,

```{math}
:label: eq33
|\hat{D}_a| \equiv \left|T^{-1}\sum_{t=1}^{T} \hat{a}_t \hat{a}_t'\right| ,
```

where the $\hat{a}_t$'s are determined by solving {eq}`eq27` recursively and so
depend on $\lambda$. 

The covariance matrix of the $a$'s is estimated as

```{math}
\hat{D}_a = T^{-1}\sum_{t=1}^{T} \hat{a}_t \hat{a}_t'
```

evaluated at the value of $\lambda$ that minimizes {eq}`eq33`. 

The resulting
estimates are known to be statistically consistent (see {cite}`wilson1973estimation`).

Notice that $\alpha$ does not appear explicitly in the likelihood function, but only
indirectly by way of the elements of $D_a$, namely $\sigma_{11}$, $\sigma_{12}$, and
$\sigma_{22}$.  

That this must be so can be seen by inspecting representation
{eq}`eq27`, in which $\lambda$ appears explicitly but $\alpha$ does not.  

On the
basis of the *four* parameters $\lambda$, $\sigma_{11}$, $\sigma_{12}$, and
$\sigma_{22}$ that are identified by {eq}`eq27` — i.e., that characterize the
likelihood function {eq}`eq32` — we can think of attempting to estimate the *five*
parameters of the model: $\alpha$, $\lambda$, $\sigma_\varepsilon^2$,
$\sigma_\eta^2$, and $\sigma_{\varepsilon\eta}$. 

Not surprisingly, some of the
parameters are underidentified.  

In particular, while $\lambda$ and
$\sigma_\varepsilon^2$ are identified, $\alpha$, $\sigma_\eta^2$, and
$\sigma_{\varepsilon\eta}$ are not separately identified.

To see that $\alpha$ and $\sigma_{\varepsilon\eta}$ are not identified, note that
from equation {eq}`eq28` the identifiable parameters $\sigma_{11}$, $\sigma_{12}$,
and $\sigma_{22}$ are related to the structural parameters $\sigma_\varepsilon^2$,
$\sigma_\eta^2$, $\sigma_{\varepsilon\eta}$, $\alpha$, and $\lambda$ by

```{math}
:label: eq34
\sigma_{11}
= \left(\frac{1}{\lambda+\alpha(1-\lambda)}\right)^{\!2}
  \bigl(\sigma_\varepsilon^2 + \sigma_\eta^2 - 2\sigma_{\varepsilon\eta}\bigr),
```

```{math}
:label: eq35
\sigma_{12}
= \frac{1-\lambda}{(\lambda+\alpha(1-\lambda))^2}
  \bigl(\sigma_\varepsilon^2 + \sigma_\eta^2 - 2\sigma_{\varepsilon\eta}\bigr)
  + \frac{1}{\lambda+\alpha(1-\lambda)}
  \bigl(\sigma_\varepsilon^2 - \sigma_{\varepsilon\eta}\bigr),
```

```{math}
:label: eq36
\sigma_{22}
= \left(\frac{1-\lambda}{\lambda+\alpha(1-\lambda)}\right)^{\!2}
  \bigl(\sigma_\varepsilon^2 + \sigma_\eta^2 - 2\sigma_{\varepsilon\eta}\bigr)
  + \sigma_\varepsilon^2
  + 2\!\left(\frac{1-\lambda}{\lambda+\alpha(1-\lambda)}\right)
  \bigl(\sigma_\varepsilon^2 - \sigma_{\varepsilon\eta}\bigr).
```

These equations imply

```{math}
:label: eq37
\sigma_{12}
= (1-\lambda)\,\sigma_{11}
  + \frac{1}{\lambda+\alpha(1-\lambda)}
  \bigl(\sigma_\varepsilon^2 - \sigma_{\varepsilon\eta}\bigr),
```

```{math}
:label: eq38
\sigma_{22}
= (1-\lambda)^2\,\sigma_{11}
  + \sigma_\varepsilon^2
  + \frac{2(1-\lambda)}{\lambda+\alpha(1-\lambda)}
  \bigl(\sigma_\varepsilon^2 - \sigma_{\varepsilon\eta}\bigr).
```

Do there exist offsetting changes in $\alpha$ and $\sigma_{\varepsilon\eta}$ that
leave both {eq}`eq37` and {eq}`eq38` satisfied with $\sigma_{11}$, $\sigma_{22}$,
and $\sigma_{12}$ unchanged?  

That is, holding $\lambda$ and $\sigma_\varepsilon^2$
constant, can we change $\alpha$ and $\sigma_{\varepsilon\eta}$ in offsetting ways
that leave $\sigma_{11}$, $\sigma_{12}$, and $\sigma_{22}$ constant? 

The answer is
yes, as can be seen by differentiating {eq}`eq37` and {eq}`eq38` and setting
$d\sigma_{12} = d\sigma_{11} = d\sigma_{22} = d\lambda = d\sigma_\varepsilon^2 = 0$:

```{math}
:label: eq39
0
= (1-\lambda)\bigl(\lambda+\alpha(1-\lambda)\bigr)^{-2}
  \bigl(\sigma_\varepsilon^2 - \sigma_{\varepsilon\eta}\bigr)\,d\alpha
  + \bigl(\lambda+\alpha(1-\lambda)\bigr)^{-1} d\sigma_{\varepsilon\eta}
  = 0,
```

```{math}
:label: eq40
0
= 2(1-\lambda)^2\bigl(\lambda+\alpha(1-\lambda)\bigr)^{-2}
  \bigl(\sigma_\varepsilon^2 - \sigma_{\varepsilon\eta}\bigr)\,d\alpha
  + 2(1-\lambda)\bigl(\lambda+\alpha(1-\lambda)\bigr)^{-1} d\sigma_{\varepsilon\eta}
  = 0.
```

Dividing {eq}`eq40` by $2(1-\lambda)$ gives {eq}`eq39`, which proves that if
$d\alpha$ and $d\sigma_{\varepsilon\eta}$ obey {eq}`eq39`, both {eq}`eq37` and
{eq}`eq38` will remain satisfied.  

Thus there exist offsetting changes in $\alpha$
and $\sigma_{\varepsilon\eta}$ that leave the identifiable parameters $\sigma_{11}$,
$\sigma_{12}$, and $\sigma_{22}$ unaltered.  

It follows that $\sigma_{\varepsilon\eta}$
and $\alpha$ are not separately identifiable.  

It is evident from {eq}`eq27` or
{eq}`eq32` that $\lambda$ is identified.  

To see that $\sigma_\varepsilon^2$ is
identifiable, simply recall that $\varepsilon_t$ obeys the feedback rule

```{math}
\mu_t = \frac{1-\lambda}{1-\lambda L}\,x_t + \varepsilon_t ,
```

so that, given $\lambda$ and samples of $\mu_t$ and $x_t$,
$\sigma_\varepsilon^2$ is identifiable as the variance of the residual in the above
equation.

To proceed to extract estimates of $\alpha$ it is necessary to impose a value of
$\sigma_{\varepsilon\eta}$.  We impose the condition $\sigma_{\varepsilon\eta} = 0$,
so that shocks to the money supply rule and shocks to portfolio balance are
uncorrelated. 

It is straightforward to calculate

```{math}
\begin{bmatrix}
  \sigma_\varepsilon^2 & \sigma_{\varepsilon\eta} \\
  \sigma_{\varepsilon\eta} & \sigma_\eta^2
\end{bmatrix}
= E
\begin{bmatrix} \varepsilon_t \\ \eta_t \end{bmatrix}
\begin{bmatrix} \varepsilon_t & \eta_t \end{bmatrix}
= G_0\, D_a\, G_0' ,
```

which expands to

```{math}
\begin{bmatrix}
  \sigma_\varepsilon^2 & \sigma_{\varepsilon\eta} \\
  \sigma_{\varepsilon\eta} & \sigma_\eta^2
\end{bmatrix}
=
\begin{bmatrix}
  (1-\lambda)^2\sigma_{11} + 2(1-\lambda)\sigma_{12} + \sigma_{22},
  & (1-\lambda)(1+\alpha(1-\lambda))\sigma_{11} - (2-\lambda+\alpha(1-\lambda))\sigma_{12} + \sigma_{22}
  \\[4pt]
  (1-\lambda)(1+\alpha(1-\lambda))\sigma_{11} - (2-\lambda+\alpha(1-\lambda))\sigma_{12} + \sigma_{22},
  & (1+\alpha(1-\lambda))^2\sigma_{11} - 2(1+\alpha(1-\lambda))\sigma_{12} + \sigma_{22}
\end{bmatrix} .
```

Imposing $\sigma_{\varepsilon\eta} = 0$, we have

```{math}
0 = \sigma_{\varepsilon\eta}
  = (1-\lambda)(1+\alpha(1-\lambda))\sigma_{11}
    - (2-\lambda+\alpha(1-\lambda))\sigma_{12}
    + \sigma_{22} ,
```

which implies that $\alpha$ is to be estimated by

```{math}
:label: eq41
\hat{\alpha}
= \frac{-\sigma_{11}}{(1-\lambda)\sigma_{11} - \sigma_{12}}
  + \frac{(2-\lambda)\,\sigma_{12}}{(1-\lambda)\bigl[(1-\lambda)\sigma_{11} - \sigma_{12}\bigr]}
  - \frac{\sigma_{22}}{(1-\lambda)\bigl[(1-\lambda)\sigma_{11} - \sigma_{12}\bigr]} .
```

Let this estimator of $\alpha$ be

```{math}
\hat{\alpha} = g(\lambda,\,\sigma_{11},\,\sigma_{12},\,\sigma_{22}) = g(\theta) .
```

Let $\Sigma_\theta$ be the estimated asymptotic covariance matrix of $\theta$.  

Then the asymptotic variance of $\hat{\alpha}$ will be estimated as

```{math}
\operatorname{var}\hat{\alpha}
= \left(\frac{\partial g}{\partial\theta}\right)_{\!\theta}
  \Sigma_\theta
  \left(\frac{\partial g}{\partial\theta}\right)_{\!\theta}' ,
```

where $(\partial g/\partial\theta)_\theta$ is the $(1\times 4)$ vector of partial
derivatives of $g$ with respect to $\theta$ evaluated at the maximum likelihood
estimates $\hat{\theta}$.  

The asymptotic covariance matrix of
$(\lambda,\sigma_{11},\sigma_{12},\sigma_{22})$ is given by

```{math}
\Sigma_\theta = \frac{1}{T}
\begin{bmatrix}
  T\sigma_\lambda^2 & 0 & 0 & 0 \\[4pt]
  0 & 2\sigma_{11}^2 & 2\sigma_{11}\sigma_{12} & 2\sigma_{12}^2 \\[4pt]
  0 & 2\sigma_{11}\sigma_{12} & \sigma_{11}\sigma_{22}+\sigma_{12}^2
    & 2\sigma_{12}\sigma_{22} \\[4pt]
  0 & 2\sigma_{12}^2 & 2\sigma_{12}\sigma_{22} & 2\sigma_{22}^2
\end{bmatrix} ,
```

where $T\sigma_\lambda^2$ is estimated by

```{math}
T\sigma_\lambda^2
= \left[-\frac{\partial^2\log L}{\partial\lambda^2}\right]_\theta^{-1}
```

and where $\log L$ is the natural logarithm of the likelihood function {eq}`eq32`.

Notice that the maximum likelihood estimate of $\lambda$ is asymptotically
orthogonal to the estimates $\sigma_{11}$, $\sigma_{12}$, $\sigma_{22}$.  The
preceding formula for $\Sigma_\theta$ can be derived by applying results of {cite}`wilson1973estimation` and {cite:t}`anderson2011statistical` [pp. 159–161].

In the computations summarized below, the
components $\sigma_{11}$, $\sigma_{12}$, and $\sigma_{22}$ were estimated by

```{math}
\hat{D}_a
= \begin{pmatrix} \hat{\sigma}_{11} & \hat{\sigma}_{12} \\
                   \hat{\sigma}_{12} & \hat{\sigma}_{22} \end{pmatrix}
= T^{-1}\sum_{t=1}^{T} \hat{a}_t \hat{a}_t' ,
```

the maximum likelihood estimator. 

The term
$\left(-\partial^2\log L/\partial\lambda^2\right)_\theta$
was estimated numerically in the course of minimizing {eq}`eq33` to obtain the
maximum likelihood estimates.

It bears emphasizing that $\alpha$ is identifiable at all only on the basis of a
restriction on $\sigma_{\varepsilon\eta}$, and that the estimator of $\alpha$
obtained by imposing $\sigma_{\varepsilon\eta} = 0$ depends sensitively on the
covariance matrix of the errors in forecasting $x_t$ and $\mu_t$ from the past.

The estimates of $\alpha$ thereby obtained ought to be regarded as very delicate.


### Implementing the MLE

The innovation recursions following directly from equations {eq}`eq29`–{eq}`eq30`
can be written compactly as

$$
a_{1t} = (x_t - x_{t-1}) + \lambda a_{1,t-1}
$$ (eq:a1_recursion)

$$
a_{2t} = \mu_t - x_t + a_{1t}
$$ (eq:a2_recursion)

A crucial feature of this representation is that **$\alpha$ does not appear in
{eq}`eq:a1_recursion`–{eq}`eq:a2_recursion`**: the only structural parameter
needed to extract innovations is $\lambda$. 

Maximizing the likelihood {eq}`eq32`
is therefore equivalent to choosing $\lambda$ to minimize

$$
\min_\lambda \; \det\!\Bigl(T^{-1} \textstyle\sum_{t=1}^T a_t a_t^\top\Bigr)
$$ (eq:mle_criterion)

where the innovations depend on $\lambda$ through {eq}`eq:a1_recursion`.

```{code-cell} ipython3
def simulate_bivariate(α, λ, T=200, σ_δ2=1.0, σ_ε2=0.5, σ_δε=0.0, seed=42):
    """
    Simulate the bivariate rational-expectations model and return
    arrays x (inflation) and μ (money growth).
    """
    rng = np.random.default_rng(seed)

    # Structural shocks
    cov = np.array([[σ_δ2, σ_δε], [σ_δε, σ_ε2]])
    shocks = rng.multivariate_normal([0.0, 0.0], cov, size=T)
    δ, ε = shocks[:, 0], shocks[:, 1]

    φ = 1.0 / (λ + α * (1.0 - λ))

    # Build Δx and Δμ from MA(1) representation
    Δx  = np.zeros(T)
    Δμ  = np.zeros(T)
    for t in range(T):
        d_prev = δ[t-1] if t > 0 else 0.0
        e_prev = ε[t-1] if t > 0 else 0.0
        Δx[t] = φ * (δ[t] - ε[t]) - φ * λ * (d_prev - e_prev)
        Δμ[t] = (φ*(1-λ) + 1)*δ[t] - φ*(1-λ)*ε[t] - d_prev

    x = np.cumsum(Δx)
    μ = np.cumsum(Δμ)
    return x, μ
```

```{code-cell} ipython3
def compute_innovations(x, μ, λ):
    """
    Recover the innovations (a_{1t}, a_{2t}) from observed x and μ
    using the recursion from eqs. (29)-(30) of the paper:

        a_{1t} = Δx_t + λ a_{1,t-1}
        a_{2t} = μ_t - x_t + a_{1t}

    Only λ is required — α does not enter the innovation extraction.

    Returns arrays a1 and a2 of length T.
    """
    T = len(x)
    a1 = np.zeros(T)
    a2 = np.zeros(T)

    a1_prev = 0.0
    x_prev  = 0.0

    for t in range(T):
        a1[t] = (x[t] - x_prev) + λ * a1_prev
        a2[t] = μ[t] - x[t] + a1[t]
        x_prev  = x[t]
        a1_prev = a1[t]

    return a1, a2
```

```{code-cell} ipython3
def mle_criterion(λ_val, x, μ):
    """
    Evaluate the MLE criterion det(D_a(λ)) for a given λ.

    Because α does not enter the innovation extraction, the determinant
    of the innovation covariance matrix depends only on λ.  Minimising
    this over λ gives the Wilson (1973) maximum likelihood estimate.
    """
    a1, a2 = compute_innovations(x, μ, λ_val)
    Da = np.cov(np.vstack([a1, a2]))
    return np.linalg.det(Da)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: MLE criterion as a function of $\lambda$
    name: fig-mle-criterion
---
# Simulate data from the true model
α_true, λ_true = -2.0, 0.6
x_sim, μ_sim = simulate_bivariate(α_true, λ_true, T=300)

λ_grid = np.linspace(0.1, 0.95, 80)
crit = [mle_criterion(lv, x_sim, μ_sim) for lv in λ_grid]  # α not needed here

fig, ax = plt.subplots()
ax.plot(λ_grid, crit, lw=2)
ax.axvline(λ_true, color='r', ls='--', label=rf'True $\lambda = {λ_true}$')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('det$(D_a(\lambda))$')
ax.set_title('MLE criterion')
ax.legend()
plt.tight_layout()
plt.show()
```

The MLE criterion attains its minimum near the true value $\lambda = 0.6$,
confirming that the Wilson–Sargent full-information maximum likelihood estimator
successfully recovers the adaptive expectations parameter.



## An Alternative Instrumental Variable Estimator

When $\sigma_{\delta\varepsilon} = 0$ (shocks to money demand and money supply
are uncorrelated), an **instrumental variable (IV) estimator** is available.

From the vector autoregressive representation {eq}`eq27`, the
innovations satisfy

$$
a_{2t} = \mu_t - E_{t-1}\mu_t = \mu_t - E_{t-1}x_t
$$ (eq:a2_as_forecast_error)

$$
a_{1t} = x_t - E_{t-1}x_t.
$$ (eq:a1_as_forecast_error)

Moreover,

$$
(a_{2t} - a_{1t})(a_{1t} - \lambda a_{1,t-1}) = 0
$$ (eq:iv_orthogonality)

This suggests the following two-step procedure:

**Step 1.** Estimate the univariate MA(1) for $(1-L)x_t$:

$$
(1-L)x_t = (1-\lambda L)a_{1t}
$$

by maximum likelihood.  This yields a consistent estimate of $\lambda$ and the
residuals $\hat a_{1t}$.

**Step 2.** Form the instrument $\hat\xi_t = \hat a_{1t} - \lambda \hat a_{1,t-1}$,
which is correlated with the regressors in Cagan's equation but orthogonal to
the disturbance $u_t$ when $\sigma_{\delta\varepsilon} = 0$.

Then estimate $\alpha$ from

$$
\hat\alpha =
\frac{\sum_t (\mu_t - E_{t-1}\mu_t - (\mu_{t-1} - E_{t-2}\mu_{t-1}))
              \cdot \hat\xi_t}
     {\sum_t \hat\xi_t^2 / \hat\alpha_{\text{first stage}}}
$$ (eq:iv_estimator)

by applying nonlinear least squares to the second-stage regression

$$
m_t - p_t = \frac{\alpha(1-\hat\lambda)}{1-\hat\lambda L}
             \sum_{i=0}^{\infty} \hat\lambda^i \hat\xi_{t-i}
             + \text{residual}.
$$ (eq:second_stage_iv)

This procedure yields consistent estimates of $\alpha$ and $\lambda$ on the
assumption that $\sigma_{\delta\varepsilon} = 0$.

```{code-cell} ipython3
def univariate_ma1_mle(Δx):
    """
    Estimate λ and the innovations a_{1t} from the univariate MA(1)
    (1-L)x_t = (1 - λL) a_{1t}  by minimizing the innovation variance.
    """
    T = len(Δx)

    def criterion(lam):
        a = np.zeros(T)
        a[0] = Δx[0]
        for t in range(1, T):
            a[t] = Δx[t] + lam * a[t-1]
        return np.var(a)

    result = minimize_scalar(criterion, bounds=(0.01, 0.99), method='bounded')
    λ_hat = result.x

    a_hat = np.zeros(T)
    a_hat[0] = Δx[0]
    for t in range(1, T):
        a_hat[t] = Δx[t] + λ_hat * a_hat[t-1]

    return λ_hat, a_hat
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: IV estimator — sampling distribution of $\hat\lambda$ and $\hat\alpha$
    name: fig-iv-distribution
---
α_true, λ_true = -2.0, 0.6
n_sims = 300
λ_hats = []
σ_δ2, σ_ε2, σ_δε = 1.0, 0.5, 0.0

for seed in range(n_sims):
    x_s, μ_s = simulate_bivariate(α_true, λ_true, T=150,
                                   σ_δ2=σ_δ2, σ_ε2=σ_ε2,
                                   σ_δε=σ_δε, seed=seed)
    Δx_s = np.diff(x_s)
    λ_h, _ = univariate_ma1_mle(Δx_s)
    λ_hats.append(λ_h)

fig, ax = plt.subplots()
ax.hist(λ_hats, bins=30, edgecolor='k', alpha=0.7)
ax.axvline(λ_true, color='r', lw=2, ls='--',
           label=rf'True $\lambda={λ_true}$')
ax.axvline(np.mean(λ_hats), color='b', lw=2, ls=':',
           label=rf'Mean $\hat\lambda={np.mean(λ_hats):.3f}$')
ax.set_xlabel(r'$\hat\lambda$')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.show()
```

The sampling distribution of $\hat\lambda$ is centered near the true value,
confirming consistency of the first-stage estimator.

## Testing the Rational Expectations Version of Cagan's Model

### The overparameterized system

Representation {eq}`eq27` is a special
case of the general vector ARMA(1,1):

$$
\begin{bmatrix} x_t \\ \mu_t \end{bmatrix}
=
C \begin{bmatrix} x_{t-1} \\ \mu_{t-1} \end{bmatrix}
+
\begin{bmatrix} a_{1t} \\ a_{2t} \end{bmatrix}
+
B \begin{bmatrix} a_{1,t-1} \\ a_{2,t-1} \end{bmatrix}
$$ (eq:general_varma)

where $C$ and $B$ are general $2\times 2$ matrices.

In the restricted model {eq}`eq27`, **seven linear restrictions** have been
imposed on the eight parameters $(c_{11}, c_{12}, c_{21}, c_{22}, b_{11}, b_{12},
b_{21}, b_{22})$ of the general system {eq}`eq:general_varma` so that the
systematic part involves only the single parameter $\lambda$.

The six overparameterizations used to test the model relax various subsets of these
restrictions.  The parameterizations are:

| # | $C$ | $B$ | Free parameters |
|---|-----|-----|-----------------|
| 1 | Full $2\times 2$ | Restricted $B(\lambda)$ | 4 |
| 2 | Restricted $C(\lambda)$ | Full $2\times 2$ | 1 |
| 3 | $\begin{bmatrix}c_1 & 0 \\ c_1 & c_1\end{bmatrix}$ | Restricted | 1 |
| 4 | $\begin{bmatrix}c_1 & c_{12} \\ c_1 & c_1\end{bmatrix}$ | Restricted | 2 |
| 5 | Full off-diagonal | Restricted | 2 |
| 6 | Restricted | Full off-diagonal | 2 |

The restricted model is tested against each overparameterization using
the likelihood-ratio statistic

$$
-2\log\!\frac{L(x_t, \mu_t; \hat\theta_0)}{L(x_t, \mu_t; \hat\theta_0, \hat{q})}
\;\overset{d}{\longrightarrow}\; \chi^2(q)
$$ (eq:lr_stat)

where $q$ is the number of restrictions relaxed.  High values lead to rejection
of the restricted model {eq}`eq27`.

### Empirical results

Table 2 reports the maximum likelihood estimates for Cagan's data under
$\sigma_{\delta\varepsilon} = 0$:

| Country | $\hat\lambda$ | $\hat\alpha$ | $\hat\sigma_{11}$ | $\hat\sigma_{12}$ | $\hat\sigma_{22}$ |
|---------|:---:|:---:|:---:|:---:|:---:|
| Germany (Oct '20–Jul '23) | .677 (.053) | −5.97 (4.62) | .0625 | .0158 | .0091 |
| Austria (Feb '21–Aug '22) | .754 (.059) | −0.31 (1.57) | .0385 | .0148 | .0085 |
| Greece (Feb '43–Aug '44) | .459 (.088) | −4.09 (2.97) | .0675 | .0245 | .0279 |
| Hungary I (Aug '22–Feb '24) | .418 (.067) | −1.84 (0.40) | .0362 | .0089 | .0060 |
| Russia (Feb '22–Jan '24) | .626 (.073) | −9.75 (10.74)| .0524 | .0138 | .0205 |
| Poland (May '22–Nov '23) | .536 (.072) | −2.53 (0.86) | .0566 | .0149 | .0089 |

Standard errors in parentheses.

Table 3 reports the chi-square statistics for Cagan's data:

| Country | Para. 1 $\chi^2(4)$ | 2 $\chi^2(1)$ | 3 $\chi^2(1)$ | 4 $\chi^2(2)$ | 5 $\chi^2(2)$ | 6 $\chi^2(2)$ |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| Germany | 0.52 | 1.12 | 2.06 | 0.95 | 3.37 | 2.14 |
| Russia | 0.21 | 3.05 | 2.84 | 3.90 | 7.79* | 0.97 |
| Greece | 1.04 | 1.53 | 0.25 | 4.14 | 1.87 | 0.40 |
| Hungary I | 4.13 | 7.57** | 3.13 | 7.57** | 7.62** | 0.24 |
| Poland | 0.19 | 0.04 | 0.22 | 0.31 | 0.56 | 0.53 |
| Austria | 2.77 | 4.97* | 0.63 | 4.97* | 10.05** | 7.13** |

Critical values: $\chi^2(1)_{.05} = 3.84$, $\chi^2(2)_{.05} = 5.99$, $\chi^2(4)_{.05} = 9.49$.  
\* Significant at .05.  \*\* Significant at .01.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: ML estimates of $\hat\lambda$ versus $|\hat\alpha|$
    name: fig-mle-estimates
---
countries = ['Germany', 'Austria', 'Greece', 'Hungary I', 'Russia', 'Poland']
λ_ml   = [.677, .754, .459, .418, .626, .536]
α_ml   = [-5.97, -0.31, -4.09, -1.84, -9.75, -2.53]
α_se   = [4.62,  1.57,  2.97,  0.40, 10.74,  0.86]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(countries, λ_ml, edgecolor='k', color='steelblue', alpha=0.8)
axes[0].set_ylabel(r'$\hat\lambda$')
axes[0].set_title(r'MLE estimates of $\lambda$')
axes[0].tick_params(axis='x', rotation=30)

axes[1].errorbar(range(len(countries)), α_ml, yerr=[2*s for s in α_se],
                 fmt='o', color='tomato', capsize=5, lw=2)
axes[1].axhline(0, color='k', lw=0.7, ls='--')
axes[1].set_xticks(range(len(countries)))
axes[1].set_xticklabels(countries, rotation=30)
axes[1].set_ylabel(r'$\hat\alpha$ (±2 s.e.)')
axes[1].set_title(r'MLE estimates of $\alpha$')

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Overfitting chi-square statistics
    name: fig-chisq-cagan
---
chi2 = np.array([
    [0.52, 1.12, 2.06, 0.95,  3.37,  2.14],   # Germany
    [0.21, 3.05, 2.84, 3.90,  7.79,  0.97],   # Russia
    [1.04, 1.53, 0.25, 4.14,  1.87,  0.40],   # Greece
    [4.13, 7.57, 3.13, 7.57,  7.62,  0.24],   # Hungary I
    [0.19, 0.04, 0.22, 0.31,  0.56,  0.53],   # Poland
    [2.77, 4.97, 0.63, 4.97, 10.05,  7.13],   # Austria
])
param_labels = ['1\n(df=4)', '2\n(df=1)', '3\n(df=1)',
                '4\n(df=2)', '5\n(df=2)', '6\n(df=2)']
crit_05 = [9.49, 3.84, 3.84, 5.99, 5.99, 5.99]

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=False)
for idx, (ax, country, row) in enumerate(
        zip(axes.flat, countries, chi2)):
    ax.bar(param_labels, row, edgecolor='k', color='steelblue', alpha=0.75)
    ax.step(np.arange(-0.5, 6.5), crit_05 + [crit_05[-1]],
            where='post', color='r', lw=2, ls='--', label='.05 critical')
    ax.set_title(country)
    ax.set_ylabel(r'$\chi^2$ statistic')
    if idx == 0:
        ax.legend(fontsize=8)

plt.suptitle("Overfitting chi-square statistics (Cagan's data)", y=1.01)
plt.tight_layout()
plt.show()
```

### Main findings

- For **Germany**, **Greece**, and **Poland** the model is not rejected at the
  .95 level by any of the six parameterizations.
- For **Hungary I** and **Austria** the model is rejected by several
  parameterizations.
- For **Russia** there is rejection under parameterization 5.

It is remarkable that a representation with only a **single free parameter**
($\lambda$) in its systematic part survives overfitting tests for three of the
six hyperinflations.

## Summary

The main results of this paper are:

1. Under the conditions that make Cagan's adaptive expectations scheme
   equivalent to rational expectations, **Cagan's OLS estimator of $\alpha$ is
   inconsistent** because inflation and money creation are determined
   simultaneously.

2. A **bivariate Wold representation** with a triangular structure shows that
   inflation Granger-causes money creation, but not vice versa — consistent with
   empirical findings that feedback runs from inflation to money creation.

3. The **structural parameter $\alpha$ is not identifiable** from the likelihood
   function alone.  Identification requires an additional restriction, namely
   $\sigma_{\delta\varepsilon} = 0$ (uncorrelated money-demand and money-supply
   shocks).  The resulting estimates of $\alpha$ carry very large standard errors.

4. The large standard errors mean that confidence intervals of two standard errors
   on each side of the point estimates include values of $\alpha$ that would imply
   money creators were maximizing seignorage revenue — potentially explaining the
   paradox noted by Cagan.

5. **Likelihood-ratio overfitting tests** do not decisively reject the one-parameter
   rational-expectations model for Germany, Greece, and Poland.

6. The results suggest that **the demand for money in hyperinflation may not have
   been as well isolated as previously thought**, and that the slope of the
   portfolio balance schedule is difficult or impossible to estimate precisely
   under the money supply regimes that prevailed during the hyperinflations.

## Exercises

```{exercise-start}
:label: ier77_ex1
```

Using `bivariate_ma1_moments`, compute all nonzero autocovariances of
$(1-L)x_t$ and $(1-L)\mu_t$ for $\alpha = -2.0$, $\lambda = 0.6$,
$\sigma_\delta^2 = 1.0$, $\sigma_\varepsilon^2 = 0.5$, and
$\sigma_{\delta\varepsilon} = 0$.

Verify numerically that the spectral density matrix is positive semidefinite
at several frequencies $\omega \in [0, \pi]$.

```{exercise-end}
```

```{solution-start} ier77_ex1
:class: dropdown
```

```{code-cell} ipython3
α, λ = -2.0, 0.6
cxx, cμμ, cxμ = bivariate_ma1_moments(α, λ)

print("Autocovariances of Δx:")
for τ, v in cxx.items():
    print(f"  c_xx({τ}) = {v:.6f}")

print("\nAutocovariances of Δμ:")
for τ, v in cμμ.items():
    print(f"  c_μμ({τ}) = {v:.6f}")

print("\nCross-covariances E[Δx_t  Δμ_{t-τ}]:")
for τ, v in cxμ.items():
    print(f"  c_xμ({τ}) = {v:.6f}")

# Check positive semidefiniteness of the spectral density matrix
ω_grid = np.linspace(0, np.pi, 50)
min_eig = np.inf
for ω in ω_grid:
    z = np.exp(-1j * ω)
    Sxx = cxx[1]*np.conj(z) + cxx[0] + cxx[1]*z
    Sμμ = cμμ[1]*np.conj(z) + cμμ[0] + cμμ[1]*z
    Sxμ = cxμ[-1]*z + cxμ[0] + cxμ[1]*np.conj(z)
    S = np.array([[np.real(Sxx), np.real(Sxμ)],
                  [np.real(Sxμ), np.real(Sμμ)]])
    eigs = np.linalg.eigvalsh(S)
    min_eig = min(min_eig, eigs.min())

print(f"\nMin eigenvalue of S(ω) over grid: {min_eig:.6f}")
print("Spectral density matrix positive semidefinite:", min_eig >= -1e-10)
```

```{solution-end}
```

````{exercise-start}
:label: ier77_ex2
````

Using `plim_alpha_cagan`, plot the asymptotic bias $\operatorname{plim}\hat\alpha - \alpha$
as a function of $\alpha$ for three values of $\lambda \in \{0.4, 0.6, 0.8\}$,
setting $\sigma_\delta^2 = 1$, $\sigma_\varepsilon^2 = 0.5$, and
$\sigma_{\delta\varepsilon} = 0$.

How does the bias depend on $\lambda$?

````{exercise-end}
````

```{solution-start} ier77_ex2
:class: dropdown
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Asymptotic bias of Cagan's estimator for different $\lambda$
    name: fig-cagan-bias-lambda
---
α_grid = np.linspace(-5.0, -0.1, 300)
λ_vals = [0.4, 0.6, 0.8]
σ_δ2, σ_ε2, σ_δε = 1.0, 0.5, 0.0

fig, ax = plt.subplots()
for λ in λ_vals:
    bias = [plim_alpha_cagan(a, λ, σ_δ2, σ_ε2, σ_δε) - a for a in α_grid]
    ax.plot(α_grid, bias, lw=2, label=rf'$\lambda={λ}$')

ax.axhline(0, color='k', lw=0.7, ls='--')
ax.set_xlabel(r'True $\alpha$')
ax.set_ylabel(r'$\operatorname{plim}\hat\alpha - \alpha$')
ax.set_title("Asymptotic bias of Cagan's estimator")
ax.legend()
plt.tight_layout()
plt.show()
```

The bias is always positive (Cagan's estimator overstates $\alpha$ toward zero)
and grows larger in magnitude as $|\alpha|$ increases.  A higher $\lambda$ reduces
$\phi$ and thereby reduces the bias slightly, but the qualitative pattern is the
same across all values of $\lambda$.

```{solution-end}
```

````{exercise-start}
:label: ier77_ex3
````

Use the univariate first-stage estimator `univariate_ma1_mle` to estimate
$\lambda$ from 500 simulated samples of length $T = 100$ from the true model
with $\alpha = -2.0$ and $\lambda = 0.6$.

Compute the mean and standard deviation of $\hat\lambda$ across simulations.
Compare with a sample of length $T = 500$.  What do you conclude about the
rate of convergence?

````{exercise-end}
````

```{solution-start} ier77_ex3
:class: dropdown
```

```{code-cell} ipython3
α_true, λ_true = -2.0, 0.6
n_sims = 500

for T in [100, 500]:
    λ_hats = []
    for seed in range(n_sims):
        x_s, _ = simulate_bivariate(α_true, λ_true, T=T, seed=seed)
        Δx_s = np.diff(x_s)
        λ_h, _ = univariate_ma1_mle(Δx_s)
        λ_hats.append(λ_h)
    print(f"T={T:4d}: mean λ̂ = {np.mean(λ_hats):.4f}, "
          f"std(λ̂) = {np.std(λ_hats):.4f}")
```

The standard deviation shrinks roughly as $1/\sqrt{T}$, consistent with
$\sqrt{T}$-consistent estimation of $\lambda$.

```{solution-end}
```
