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

{cite:t}`sargent1973rational` pointed out that under assumptions making Cagan's
adaptive expectations equivalent to rational expectations, Cagan's
estimator of $\alpha$ — the slope of log real balances with respect to expected
inflation — is not statistically consistent.

This inconsistency matters because of a paradox that emerged when Cagan used
his estimates of $\alpha$ to calculate the sustained rates of inflation that would
maximize the flow of real resources that money creators could command by printing
money.

That "optimal" rate is $-1/\alpha$.

For each of the seven hyperinflations
in his sample, the reciprocal of Cagan's estimate of $-\alpha$ turned out to be
less — and often very much less — than the actual average rate of inflation,
suggesting that the creators of money expanded the money supply at rates far
exceeding the revenue-maximizing rate.

A natural explanation is that this paradox is a statistical artifact — a
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

- Column (1): $-1/\alpha$ (continuously compounded), the rate per month that maximizes
the revenue of the money creator. 

- Column (2): $(e^{1/\alpha}-1)\times 100$
(neglects compounding). 

- Column (3): average actual rate of inflation per month.

The paper pursues three goals:

1. *Characterize the asymptotic bias* in Cagan's ordinary-least-squares estimator
   under the rational expectations version of his model.
2. *Derive a consistent estimator*, a full-information maximum likelihood
   estimator,  for the bivariate rational-expectations model.
3. *Test the model* by overfitting a more general vector autoregressive,
   moving-average representation and computing likelihood-ratio statistics.

Our key tools are bivariate Wold representations, Granger causality, and vector
time series methods following
{cite:t}`granger1969causality`, {cite:t}`sims1972money`, {cite:t}`wilson1973estimation` and {cite:t}`anderson2011statistical`.


```{note}
This lecture can be viewed as a bivariate version of the "reverse engineering" exercise of
{cite:t}`Muth1960` that we described in {doc}`muth_kalman`.

From a technical point of view this lecture is an exercise in applying vector
time series models.

The model is interesting because it illustrates the
difference between Granger causality and simple notions of one series *leading*
another.

It also illustrates a difference between Granger causality and the separate notion of
*invariance with respect to an intervention*.
```

We begin with imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
```

## Cagan's model under rational expectations

For background on the Cagan model see {doc}`intro:cagan_ree` and
{doc}`intro:cagan_adaptive`.

Cagan's model of hyperinflation builds on a demand schedule for real balances
of the form

```{math}
:label: eq1
m_t - p_t = \alpha \pi_t + u_t, \qquad \alpha < 0,
```

where $m$ is the log of the money supply (which is always equal to the log of
money demand); $p$ is the log of the price level; $\pi_t$ is the expected rate
of inflation, i.e., the public's subjective expectation of $p_{t+1} - p_t$;
and $u_t$ is a random variable with mean zero.

Let $x_t \equiv p_t - p_{t-1}$ be the inflation rate and
$\mu_t \equiv m_t - m_{t-1}$ be the percentage rate of money creation.


```{note}
A constant term has been omitted from {eq}`eq1`, though one would be included
in empirical work.
```

Cagan assumed that $\pi_t$ was formed via the adaptive expectations scheme

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
information available as of time $t$.[^info] 

Using {eq}`eq3` and recursions on
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

Equation {eq}`eq4` characterizes the stochastic process for inflation as a
function of the stochastic process for money creation.

The model asserts that {eq}`eq4` is invariant with respect to interventions in
the form of changes in the stochastic process governing money creation.

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

The paper therefore studies two sufficient conditions.

### Two sufficient conditions

The first sufficient condition is

```{math}
:label: eq6
u_t = u_{t-1} + \eta_t,
```

where $\eta_t$ is a serially uncorrelated random term with mean zero and variance
$\sigma_\eta^2$; we assume that
$E[\eta_t \mid u_{t-1}, \mu_{t-2}, \ldots, x_{t-1}, x_{t-2}, \ldots] = 0$.

Under {eq}`eq6`, $u_t$ follows a random walk, so

```{math}
E_t u_{t+j} = u_t, \qquad j \geq 0,
```

and hence

```{math}
E_t u_{t+j} - E_t u_{t+j-1} = 0 \quad \text{for all } j \geq 1.
```

The second sufficient condition is

```{math}
:label: eq7
E_t\mu_{t+j} = E_t\mu_{t+1} \quad \text{for } j > 1,
```

so a constant rate of money creation is expected over the future.

Under {eq}`eq6` and {eq}`eq7`, the geometric sum in {eq}`eq5` equals $1$
because $\lvert -\alpha / (1-\alpha) \rvert < 1$ whenever $\alpha < 0$.

Hence {eq}`eq5` reduces to

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

Equation {eq}`eq9` is compatible with a government that finances a substantial
share of roughly fixed real expenditures by printing money.

Equation {eq}`eq9` is also compatible with a real-bills regime in which the
monetary authority supplies whatever money the public demands at a fixed nominal
interest rate or a fixed real money supply.

During the German hyperinflation, officials repeatedly described policy in just
such real-bills terms, insisting that money creation was responding to inflation
rather than causing it.

### The bivariate process for inflation and money creation

The foregoing establishes that if equations {eq}`eq6` and {eq}`eq9` obtain,
Cagan's adaptive expectations scheme is compatible with rational expectations and
with the portfolio balance condition that he assumed.

Under these assumptions, inflation and money creation form the bivariate system

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
{eq}`eq9b`; alternatively, see {cite:t}`sargent1973rational` for a somewhat different
but equivalent way of deriving {eq}`eq11` and {eq}`eq12`.

```{note}
By construction, the model implies
$E_{t-1}x_t = \frac{1-\lambda}{1-\lambda L} x_{t-1}$, so lagged money growth
does not help predict current inflation once lagged inflation is known.

This Granger-causal pattern comes from the particular money-supply rule in
{eq}`eq9`, not from an invariant feature of the economy across monetary regimes.
```


## Bias in Cagan's estimator

### Bivariate Wold representation

A convenient way to evaluate the asymptotic bias in Cagan's estimator is to obtain
a bivariate Wold representation for $(\Delta x_t, \Delta \mu_t)$.

Let $\phi \equiv (\lambda + \alpha(1-\lambda))^{-1}$.

Decompose the money-supply shock as

$$
\varepsilon_t = \rho(\varepsilon_t - \eta_t) + v_t
$$ (eq:decompose_epsilon)

where $\rho$ is the regression coefficient of $\varepsilon_t$ on
$(\varepsilon_t - \eta_t)$:

$$
\rho = \frac{E[\varepsilon_t(\varepsilon_t - \eta_t)]}
            {E[(\varepsilon_t - \eta_t)^2]}
     = \frac{\sigma_\varepsilon^2 - \sigma_{\varepsilon\eta}}
            {\sigma_\varepsilon^2 - 2\sigma_{\varepsilon\eta} + \sigma_\eta^2}
$$ (eq:rho)

and $v_t$ is orthogonal to $(\varepsilon_t - \eta_t)$ by construction.

Substituting {eq}`eq:decompose_epsilon` into {eq}`eq12` and using {eq}`eq11`
gives the triangular bivariate Wold representation

$$
(1-L)x_t = \phi(1-\lambda L)(\varepsilon_t - \eta_t)
$$ (eq:wold_x)

$$
(1-L)\mu_t = [\phi(1-\lambda) + \rho(1-L)](\varepsilon_t - \eta_t) + (1-L)v_t
$$ (eq:wold_mu)

with fundamental noises $(\varepsilon_t - \eta_t)$ and $v_t$.

The triangular structure confirms that $\Delta x$ is econometrically exogenous
with respect to $\Delta\mu$, and that $\Delta x$ Granger-causes $\Delta\mu$ but
not vice versa.

### The population regression (Cagan's estimator)

Substituting {eq}`eq:wold_x` into {eq}`eq:wold_mu` and applying the summation
operator $(1-L)^{-1}$ gives the population regression that Cagan estimated:

$$
(m_t - p_t)
= \left[\rho\alpha - (1-\rho)\frac{\lambda}{1-\lambda}\right]
  \frac{1-\lambda}{1-\lambda L}\, x_t + \bar{u}_t
$$ (eq:cagan_pop_regression)

where $\bar{u}_t = \bar{u}_{t-1} + v_t$ follows a random walk orthogonal to the
$x$ process.

Now Cagan regarded this population projection as giving estimates of the
structural equation

$$
m_t - p_t = \alpha \pi_t + u_t = \frac{\alpha(1-\lambda)}{1-\lambda L} x_t + u_t.
$$ (eq:structural_cagan)

Comparing {eq}`eq:cagan_pop_regression` with the corresponding structural form
shows that:

- Cagan's estimator of $\lambda$ is *consistent*.
- Cagan's estimator of $\alpha$ is *not consistent* in general, and obeys

$$
\operatorname{plim}\hat\alpha
= \rho\alpha - (1-\rho)\frac{\lambda}{1-\lambda}
$$ (eq:plim_alpha)

If $\rho = 0$ and hence there are no money-supply shocks, then
$\operatorname{plim}\hat\alpha = -\lambda / (1-\lambda)$, which is the value
derived by {cite:t}`sargent1973rational`.

If $\eta_t = 0$ for all $t$, so there is no noise in the portfolio-balance
equation, then $\rho = 1$ and $\operatorname{plim}\hat\alpha = \alpha$.

```{note}
It is noteworthy that the residuals in {eq}`eq:cagan_pop_regression` follow a
random walk.

{cite:t}`Cagan` and {cite:t}`barro1970inflation` both reported highly serially correlated residuals
and very low Durbin–Watson statistics, which is consistent with this prediction.
```

The following functions implement $\rho$ from {eq}`eq:rho` and the probability limit of Cagan's estimator from {eq}`eq:plim_alpha`.

```{code-cell} ipython3
def rho_from_moments(σ_ε2, σ_η2, σ_εη):
    """
    Compute ρ = Cov(ε, ε-η) / Var(ε-η).
    """
    var_diff = σ_ε2 - 2.0 * σ_εη + σ_η2
    if np.isclose(var_diff, 0.0):
        raise ValueError("Var(ε_t - η_t) must be positive.")
    return (σ_ε2 - σ_εη) / var_diff


def plim_alpha_cagan(α, λ, σ_ε2=1.0, σ_η2=0.5, σ_εη=0.0):
    """
    Asymptotic limit (population value) of Cagan's OLS estimator of α.
    """
    ρ = rho_from_moments(σ_ε2, σ_η2, σ_εη)
    return ρ * α - (1.0 - ρ) * λ / (1.0 - λ)
```

We plot the probability limit of Cagan's estimator against the true $\alpha$ to visualize the bias.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Bias of Cagan's OLS estimator of $\alpha$
    name: fig-cagan-bias
---
α_grid = np.linspace(-8.0, -0.1, 400)
λ = 0.5
σ_ε2, σ_η2, σ_εη = 1.0, 0.5, 0.0

valid = ~np.isclose(λ + α_grid * (1.0 - λ), 0.0)
α_plot = α_grid[valid]
plims = [plim_alpha_cagan(a, λ, σ_ε2, σ_η2, σ_εη) for a in α_plot]
ws_limit = -λ / (1.0 - λ)

fig, ax = plt.subplots()
ax.plot(α_plot, α_plot, 'k--', lw=1.5, label='No bias (45° line)')
label = rf'$\operatorname{{plim}}\hat\alpha$, $\lambda={λ}$'
ax.plot(α_plot, plims, lw=2, label=label)
ax.axhline(ws_limit, color='r', ls=':', lw=1.5,
           label=rf'$-\lambda/(1-\lambda) = {ws_limit:.1f}$')
ax.set_xlabel(r'True $\alpha$')
ax.set_ylabel(r'$\operatorname{plim}\hat\alpha$')
ax.legend()
plt.tight_layout()
plt.show()
```

The probability limit is a weighted average of the true $\alpha$ and the
Wallace-Sargent value $-\lambda / (1-\lambda)$.

When the true $\alpha$ is more negative than $-\lambda / (1-\lambda)$, the bias
pulls Cagan's estimator toward zero.


## Consistent estimator


Equations {eq}`eq11` and {eq}`eq12` form a bivariate first-order moving average process in
$(1-L)\mu_t$ and $(1-L)x_t$.

Assuming that the white noises $\varepsilon_t$ and
$\eta_t$ are jointly normally distributed, the likelihood function of a sample of
length $T$ observations, $t = 1, \ldots, T$, generated by {eq}`eq11`–{eq}`eq12` can be written
down.

To apply the method of maximum likelihood, it is most convenient to write
the model in its vector autoregressive form.

First note that from {eq}`eq9b` we can write

```{math}
:label: eq23
\varepsilon_t = \mu_t - \frac{1-\lambda}{1-\lambda L}\, x_t .
```

Next from {eq}`eq11` we have

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

Thus, $x$ is econometrically exogenous with respect to $\mu$.[^sims]

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

Their cross-spectrum has zero phase at all frequencies.

Evidence that $x$ leads $\mu$ would not be consistent with the model being studied
here.

[^info]: The information set includes current and past values of $p_t$ and $x_t$, as in Sargent's footnote 3.

[^deriv]: Substitute {eq}`eq3` into {eq}`eq1`, first-difference, take conditional expectations, and solve the forward difference equation using $\lvert \alpha / (1-\alpha) \rvert < 1$.

[^footprocess]: This is not the only process satisfying {eq}`eq8`, but it is a convenient and economically interpretable one.

[^sims]: {cite}`sims1972money` proved the equivalence of Granger causality (see {cite}`granger1969causality`) with econometric exogeneity.

The vector autoregressive, moving average process {eq}`eq27` is in a form that can
be estimated by the maximum likelihood estimator described by {cite:t}`wilson1973estimation`.

It is essential that the matrices multiplying current
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
used to solve for $a_t$, $t = 1, \ldots, T$.

(We take $a_{10} = a_{20} = 0$.)

{cite:t}`wilson1973estimation` notes that maximizing {eq}`eq32` is equivalent to minimizing with respect to
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
$\sigma_{\varepsilon\eta}$.

We impose the condition $\sigma_{\varepsilon\eta} = 0$,
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
orthogonal to the estimates $\sigma_{11}$, $\sigma_{12}$, $\sigma_{22}$.

The preceding formula for $\Sigma_\theta$ can be derived by applying results of {cite}`wilson1973estimation` and {cite:t}`anderson2011statistical` [pp. 159–161].

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

A crucial feature of this representation is that $\alpha$ does not appear in
either {eq}`eq:a1_recursion` or {eq}`eq:a2_recursion`, so the only structural
parameter needed to extract innovations is $\lambda$.

Maximizing the likelihood {eq}`eq32`
is therefore equivalent to choosing $\lambda$ to minimize

$$
\min_\lambda \; \det\!\Bigl(T^{-1} \textstyle\sum_{t=1}^T a_t a_t^\top\Bigr)
$$ (eq:mle_criterion)

where the innovations depend on $\lambda$ through {eq}`eq:a1_recursion`.

The following function simulates the bivariate process {eq}`eq11`–{eq}`eq12` by drawing shocks $(\varepsilon_t, \eta_t)$ and constructing $\Delta x_t$ and $\Delta\mu_t$ from their MA(1) representations.

```{code-cell} ipython3
def simulate_bivariate(α, λ, T=200, σ_ε2=1.0,
                       σ_η2=0.5, σ_εη=0.0, seed=42):
    """
    Simulate the bivariate rational-expectations model and return
    arrays x (inflation) and μ (money growth).
    """
    rng = np.random.default_rng(seed)

    cov = np.array([[σ_ε2, σ_εη], [σ_εη, σ_η2]])
    shocks = rng.multivariate_normal([0.0, 0.0], cov, size=T)
    ε, η = shocks[:, 0], shocks[:, 1]

    denom = λ + α * (1.0 - λ)
    if np.isclose(denom, 0.0):
        raise ValueError("λ + α(1-λ) must be nonzero.")
    φ = 1.0 / denom

    Δx = np.zeros(T)
    Δμ = np.zeros(T)
    for t in range(T):
        e_prev = ε[t-1] if t > 0 else 0.0
        η_prev = η[t-1] if t > 0 else 0.0
        Δx[t] = φ * (ε[t] - η[t]) - φ * λ * (e_prev - η_prev)
        Δμ[t] = (φ * (1.0 - λ) + 1.0) * ε[t] - φ * (1.0 - λ) * η[t] - e_prev

    x = np.cumsum(Δx)
    μ = np.cumsum(Δμ)
    return x, μ
```

Next we implement the innovation recursions {eq}`eq:a1_recursion`–{eq}`eq:a2_recursion` to recover $(a_{1t}, a_{2t})$ from observed data.

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

With the innovations in hand, we can evaluate the MLE criterion {eq}`eq:mle_criterion` and minimize it over $\lambda$.

```{code-cell} ipython3
def mle_criterion(λ_val, x, μ):
    """
    Evaluate the MLE criterion det(D_a(λ)) for a given λ.

    Because α does not enter the innovation extraction, the determinant
    of the innovation second-moment matrix depends only on λ. Minimizing
    this over λ gives the Wilson (1973) maximum likelihood estimate.
    """
    a1, a2 = compute_innovations(x, μ, λ_val)
    A = np.vstack([a1, a2])
    Da = A @ A.T / len(x)
    return np.linalg.det(Da)


def estimate_lambda_mle(x, μ, bounds=(0.01, 0.99)):
    """
    Compute the Wilson-Sargent MLE of λ by minimizing eq. (33).
    """
    result = minimize_scalar(
        lambda λ_val: mle_criterion(λ_val, x, μ),
        bounds=bounds,
        method='bounded'
    )
    return result.x
```

We simulate a sample from the model and plot the MLE criterion as a function of $\lambda$ to verify that it is minimized near the true value.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: MLE criterion as a function of $\lambda$
    name: fig-mle-criterion
---
# Simulated sample
α_true, λ_true = -2.0, 0.6
x_sim, μ_sim = simulate_bivariate(α_true, λ_true, T=300)
λ_hat = estimate_lambda_mle(x_sim, μ_sim)

λ_grid = np.linspace(0.1, 0.95, 80)
# α unused
crit = [mle_criterion(lv, x_sim, μ_sim)
        for lv in λ_grid]

fig, ax = plt.subplots()
ax.plot(λ_grid, crit, lw=2)
ax.axvline(λ_true, color='r', ls='--', label=rf'True $\lambda = {λ_true}$')
ax.set_xlabel(r'$\lambda$')
ax.set_ylabel('det$(D_a(\lambda))$')
ax.legend()
plt.tight_layout()
plt.show()

print(f"Estimated λ from this sample: {λ_hat:.3f}")
```

The MLE criterion attains its minimum near the true value $\lambda = 0.6$,
confirming that the Wilson–Sargent full-information maximum likelihood estimator
successfully recovers the adaptive expectations parameter.



## An alternative instrumental variable estimator

When $\sigma_{\varepsilon\eta} = 0$ (money-supply shocks and portfolio-balance
shocks are uncorrelated), Sargent shows that an instrumental variable estimator
is available.

From the vector autoregressive representation {eq}`eq27`, the
innovations satisfy

$$
a_{2t} = \mu_t - E_{t-1}\mu_t = \mu_t - E_{t-1}x_t
$$ (eq:a2_as_forecast_error)

$$
a_{1t} = x_t - E_{t-1}x_t.
$$ (eq:a1_as_forecast_error)

On the restriction $\sigma_{\varepsilon\eta} = 0$, these one-step-ahead forecast
errors can be used to construct an instrument from the estimated innovations in
inflation.

The paper's two-step procedure is:

1. Estimate the univariate MA(1) for $(1-L)x_t$:

$$
(1-L)x_t = (1-\lambda L)a_{1t}
$$

by maximum likelihood.

This yields a consistent estimate of $\lambda$ and the
residuals $\hat a_{1t}$.

2. Use those fitted innovations to form an instrument for expected inflation,
   and then estimate Cagan's money-demand equation by nonlinear least squares.

This procedure yields consistent estimates of $\alpha$ and $\lambda$ when
$\sigma_{\varepsilon\eta} = 0$.

For this lecture, we illustrate the first stage.

It is the step that identifies $\lambda$ and constructs the estimated
inflation innovations that the paper then turns into an instrument for the
second-stage nonlinear regression.

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
        return np.mean(a**2)

    result = minimize_scalar(criterion, bounds=(0.01, 0.99), method='bounded')
    λ_hat = result.x

    a_hat = np.zeros(T)
    a_hat[0] = Δx[0]
    for t in range(1, T):
        a_hat[t] = Δx[t] + λ_hat * a_hat[t-1]

    return λ_hat, a_hat
```

We run a Monte Carlo experiment to examine the sampling distribution of the first-stage estimator $\hat\lambda$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: First-stage sampling distribution of $\hat\lambda$
    name: fig-iv-distribution
---
α_true, λ_true = -2.0, 0.6
n_sims = 300
λ_hats = []
σ_ε2, σ_η2, σ_εη = 1.0, 0.5, 0.0

for seed in range(n_sims):
    x_s, μ_s = simulate_bivariate(α_true, λ_true, T=150,
                                   σ_ε2=σ_ε2, σ_η2=σ_η2,
                                   σ_εη=σ_εη, seed=seed)
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
ax.set_ylabel('frequency')
ax.legend()
plt.tight_layout()
plt.show()
```

The sampling distribution of $\hat\lambda$ is centered near the true value,
confirming consistency of the first-stage estimator.

## Testing the rational expectations version of Cagan's model

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

In the restricted model {eq}`eq27`, *seven linear restrictions* have been
imposed on the eight parameters $(c_{11}, c_{12}, c_{21}, c_{22}, b_{11}, b_{12},
b_{21}, b_{22})$ of the general system {eq}`eq:general_varma` so that the
systematic part involves only the single parameter $\lambda$.

The six overparameterizations used to test the model relax various subsets of these
restrictions.

The parameterizations are:

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

where $q$ is the number of restrictions relaxed.

High values lead to rejection
of the restricted model {eq}`eq27`.

### Empirical results

Table 2 reports the maximum likelihood estimates for Cagan's data under
$\sigma_{\varepsilon\eta} = 0$:

| Country | $\hat\lambda$ | $\hat\alpha$ | $\hat\sigma_{11}$ | $\hat\sigma_{12}$ | $\hat\sigma_{22}$ |
|---------|:---:|:---:|:---:|:---:|:---:|
| Germany (Oct '20–Jul '23) | .677 (.053) | −5.97 (4.62) | .0625 | .0158 | .0091 |
| Austria (Feb '21–Aug '22) | .754 (.059) | −0.31 (1.57) | .0385 | .0148 | .0085 |
| Greece (Feb '43–Aug '44) | .459 (.088) | −4.09 (2.97) | .0675 | .0245 | .0279 |
| Hungary I (Aug '22–Feb '24) | .418 (.067) | −1.84 (0.40) | .0362 | .0089 | .0060 |
| Russia (Feb '22–Jan '24) | .626 (.073) | −9.75 (10.74)| .0524 | .0138 | .0205 |
| Poland (May '22–Nov '23) | .536 (.072) | −2.53 (0.86) | .0566 | .0149 | .0089 |

Standard errors in parentheses.

The paper also reports a parallel table for Barro's high-inflation data.

We keep the focus here on Cagan's sample because it is the source of the
original paradox in Table 1.

The striking feature is how loose the estimates of $\alpha$ become.

For Germany, Austria, and Russia the standard error is of the same order as the
point estimate, while Hungary I is the only case in which $\alpha$ is estimated
with much precision.

Hungary I is the standout case.

With $\hat\alpha = -1.84$, the implied revenue-maximizing inflation rate
$-1/\hat\alpha$ is about 54 percent per month, close to the observed 46 percent.

In Sargent's reading, this is the one hyperinflation in which the maximum
likelihood estimate substantially weakens Cagan's paradox.

For the other countries, the point estimates do not eliminate the paradox, but
two-standard-error bands still include values of $\alpha$ that would.

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
\* Significant at .05. \*\* Significant at .01.

The following figures visualize the estimates from Table 2 and the chi-square statistics from Table 3.

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
axes[0].tick_params(axis='x', rotation=30)

axes[1].errorbar(range(len(countries)), α_ml, yerr=[2*s for s in α_se],
                 fmt='o', color='tomato', capsize=5, lw=2)
axes[1].axhline(0, color='k', lw=0.7, ls='--')
axes[1].set_xticks(range(len(countries)))
axes[1].set_xticklabels(countries, rotation=30)
axes[1].set_ylabel(r'$\hat\alpha$ (±2 s.e.)')

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

plt.tight_layout()
plt.show()
```

### Main findings

- Germany, Greece, and Poland are not rejected at the .95 level by any of the
  six parameterizations.
- Hungary I and Austria are rejected by several parameterizations.
- Russia is rejected under parameterization 5.

It is remarkable that a representation with only a *single free parameter*
($\lambda$) in its systematic part survives overfitting tests for three of the
six hyperinflations.

## Summary

The main results of this paper are:

1. Under the conditions that make Cagan's adaptive expectations scheme
   equivalent to rational expectations, Cagan's OLS estimator of $\alpha$ is
   *inconsistent* because inflation and money creation are determined
   simultaneously.

2. A bivariate Wold representation with a triangular structure shows that
   inflation Granger-causes money creation, but not vice versa — consistent with
   empirical findings that feedback runs from inflation to money creation.

3. The structural parameter $\alpha$ is *not identifiable* from the likelihood
   function alone.

   Identification requires an additional restriction, namely
   $\sigma_{\varepsilon\eta} = 0$ (uncorrelated portfolio-balance and money-supply
   shocks).

   The resulting estimates of $\alpha$ carry very large standard errors.

4. The large standard errors mean that confidence intervals of two standard errors
   on each side of the point estimates include values of $\alpha$ that would imply
   money creators were maximizing seignorage revenue — potentially explaining the
   paradox noted by Cagan.

5. Likelihood-ratio overfitting tests do not decisively reject the one-parameter
   rational-expectations model for Germany, Greece, and Poland.

6. The results suggest that the demand for money in hyperinflation may *not* have
   been as well isolated as previously thought, and that the slope of the
   portfolio balance schedule is difficult or impossible to estimate precisely
   under the money supply regimes that prevailed during the hyperinflations.

## Exercises

The function below computes the autocovariances of $(1-L)x$ and $(1-L)\mu$ and their cross-covariances.

We will use these moments to evaluate the bias in Cagan's estimator and to construct a consistent estimator.

```{code-cell} ipython3
def bivariate_ma1_moments(α, λ, σ_ε2=1.0, σ_η2=0.5, σ_εη=0.0):
    """
    Compute the autocovariances of (1-L)x and (1-L)μ under the
    bivariate rational-expectations model of Sargent (1977).

    Parameters:

    α  : float  (< 0)  demand semi-elasticity
    λ  : float  (0 < λ < 1)  adaptive expectations parameter
    σ_ε2 : variance of money-supply shock ε_t
    σ_η2 : variance of portfolio shock η_t
    σ_εη : covariance of ε_t and η_t

    Returns:

    cxx : dict with keys 0, 1  — autocovariances of Δx
    cμμ : dict with keys 0, 1  — autocovariances of Δμ
    cxμ : dict with keys -1, 0, 1  — cross-covariances E[Δx_t Δμ_{t-τ}]
    """
    denom = λ + α * (1.0 - λ)
    if np.isclose(denom, 0.0):
        raise ValueError("λ + α(1-λ) must be nonzero.")
    φ = 1.0 / denom

    # MA(1) forms for Δx_t and Δμ_t
    # Δμ_t = [φ(1-λ)+1]ε_t - φ(1-λ)η_t - ε_{t-1}

    A = φ * (1.0 - λ) + 1.0
    B = φ * (1.0 - λ)
    var_diff = σ_ε2 - 2.0 * σ_εη + σ_η2

    cxx0 = φ**2 * (1 + λ**2) * var_diff
    cxx1 = -φ**2 * λ * var_diff

    cμμ0 = (A**2 + 1.0) * σ_ε2 + B**2 * σ_η2 - 2.0 * A * B * σ_εη
    cμμ1 = -A * σ_ε2 + B * σ_εη

    cxμ0 = φ * ((A + λ) * σ_ε2 + B * σ_η2 - (A + B + λ) * σ_εη)
    cxμ1 = -φ * λ * (A * σ_ε2 + B * σ_η2 - (A + B) * σ_εη)
    cxμm1 = -φ * (σ_ε2 - σ_εη)

    cxx = {0: cxx0, 1: cxx1}
    cμμ = {0: cμμ0, 1: cμμ1}
    cxμ = {-1: cxμm1, 0: cxμ0, 1: cxμ1}
    return cxx, cμμ, cxμ
```

```{exercise-start}
:label: ier77_ex1
```

Using `bivariate_ma1_moments`, compute all nonzero autocovariances of
$(1-L)x_t$ and $(1-L)\mu_t$ for $\alpha = -2.0$, $\lambda = 0.6$,
$\sigma_\varepsilon^2 = 1.0$, $\sigma_\eta^2 = 0.5$, and
$\sigma_{\varepsilon\eta} = 0$.

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

# Check PSD
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
setting $\sigma_\varepsilon^2 = 1$, $\sigma_\eta^2 = 0.5$, and
$\sigma_{\varepsilon\eta} = 0$.

How does the bias depend on $\lambda$ and on the location of the
Wallace-Sargent value $-\lambda / (1-\lambda)$?

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
σ_ε2, σ_η2, σ_εη = 1.0, 0.5, 0.0

fig, ax = plt.subplots()
for λ in λ_vals:
    valid = ~np.isclose(λ + α_grid * (1.0 - λ), 0.0)
    α_plot = α_grid[valid]
    bias = [plim_alpha_cagan(a, λ, σ_ε2, σ_η2, σ_εη) - a
            for a in α_plot]
    ax.plot(α_plot, bias, lw=2, label=rf'$\lambda={λ}$')

ax.axhline(0, color='k', lw=0.7, ls='--')
ax.set_xlabel(r'True $\alpha$')
ax.set_ylabel(r'$\operatorname{plim}\hat\alpha - \alpha$')
ax.legend()
plt.tight_layout()
plt.show()
```

The bias changes sign at the Wallace-Sargent value $-\lambda / (1-\lambda)$.

For $\alpha$ values more negative than that benchmark, the estimator is biased
toward zero, while for less negative values it is biased away from zero.

```{solution-end}
```

````{exercise-start}
:label: ier77_ex3
````

Use the univariate first-stage estimator `univariate_ma1_mle` to estimate
$\lambda$ from 500 simulated samples of length $T = 100$ from the true model
with $\alpha = -2.0$ and $\lambda = 0.6$.

Compute the mean and standard deviation of $\hat\lambda$ across simulations.

Compare with a sample of length $T = 500$.

What do you conclude about the
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
