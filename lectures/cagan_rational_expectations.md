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
money.  That "optimal" rate is $-1/\alpha$.  For each of the seven hyperinflations
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

The key methodological tools are **bivariate Wold representations**, **Granger
causality**, and **vector time series methods** following
{cite}`granger1969causal`, {cite}`sims1972money`, {cite}`wilson1973estimation`,
and {cite}`zellner_palm1974`.

```{note}
From a technical point of view this paper is an exercise in applying **vector
time series models**.  The model is interesting because it illustrates the
difference between Granger causality and simple notions of one series *leading*
another, and between Granger causality and the separate notion of **invariance
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
the expected rate of inflation (the public's psychological expectation of
$p_{t+1} - p_t$), and $u_t$ is a mean-zero random disturbance.

Cagan assumed $\pi_t$ obeys the adaptive expectations scheme

$$
\pi_t = \frac{1-\lambda}{1-\lambda L}(p_t - p_{t-1})
$$ (eq:adaptive_re)

where $L$ is the lag operator, $L^k x_t = x_{t-k}$, and $0 < \lambda < 1$.

Let $x_t \equiv p_t - p_{t-1}$ be the inflation rate and $\mu_t \equiv m_t - m_{t-1}$
be the percentage rate of money creation.

### Rational expectations

Under rational expectations we require

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
inflation as a function of the stochastic process for money creation.  The model
asserts that {eq}`eq:rational_pi_general` is **invariant** with respect to
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

The necessary and sufficient conditions for {eq}`eq:equivalence_condition` to hold
for all $\alpha$ and all $t$ are complex.  A tractable pair of **sufficient**
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

Under these two conditions the appropriate rational-expectations formula reduces to:

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
model.  It says the rate of money creation equals expected inflation plus a random
term.  This is consistent with a "real bills" regime in which the monetary
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
condition that $\mu$ does **not** Granger-cause $x$.  That is, once lagged $x$'s
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

## The Bias in Cagan's Estimator

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
random walk.  Cagan and Barro both reported highly serially correlated residuals
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

## A Consistent Estimator

### The vector autoregressive, moving average representation

Equations {eq}`eq:bivariate_x`–{eq}`eq:bivariate_mu` form a **bivariate
first-order moving average** process in $(1-L)x_t$ and $(1-L)\mu_t$.

The first-difference MA(1) representation {eq}`eq:bivariate_x` can be written in
a more useful **levels ARIMA** form.  Since $(1-L)x_t = (1-\lambda L)a_{1t}$,
the levels process obeys

$$
x_t = x_{t-1} + a_{1t} - \lambda a_{1,t-1}
$$ (eq:arima_x)

and the link between inflation and money creation (from equation (30) of the
paper) is

$$
\mu_t = x_t + a_{2t} - a_{1t}
$$ (eq:mu_eq_x_plus_innovations)

where $a_{1t}$ and $a_{2t}$ are the **innovations** — the one-period-ahead
forecast errors for $x_t$ and $\mu_t$ respectively — with $E_{t-1}a_{1t}=0$
and $E_{t-1}a_{2t}=0$.

From {eq}`eq:arima_x` and {eq}`eq:mu_eq_x_plus_innovations`, the innovations
can be extracted recursively as

$$
a_{1t} = (x_t - x_{t-1}) + \lambda a_{1,t-1}
$$ (eq:a1_recursion)

$$
a_{2t} = \mu_t - x_t + a_{1t}
$$ (eq:a2_recursion)

A crucial feature of this representation is that **$\alpha$ does not appear in
{eq}`eq:a1_recursion`–{eq}`eq:a2_recursion`**.  The only structural parameter
needed to extract innovations is $\lambda$.  The paper (p. 72) states: "Notice
that $\alpha$ does not appear explicitly in the likelihood function, but only
indirectly by way of the elements of $D_a$, namely $\sigma_{11}$, $\sigma_{12}$,
and $\sigma_{22}$."

The link to structural shocks is $a_{1t} = \phi(\delta_t - \varepsilon_t)$,
where $\phi = (\lambda + \alpha(1-\lambda))^{-1}$.  The second innovation
$a_{2t}$ is a linear combination of $\delta_t$ and its one-period change.

From {eq}`eq:arima_x`: $E_{t-1}x_t = x_{t-1} - \lambda a_{1,t-1}$.
From {eq}`eq:mu_eq_x_plus_innovations`: $E_{t-1}\mu_t = E_{t-1}x_t$,
confirming that inflation Granger-causes money creation with no reverse causality.

### Identification

Let $D_a$ denote the covariance matrix of the innovations:

$$
D_a = \begin{bmatrix} \sigma_{11} & \sigma_{12} \\ \sigma_{12} & \sigma_{22} \end{bmatrix}
= E\begin{bmatrix} a_{1t} \\ a_{2t} \end{bmatrix}
  \begin{bmatrix} a_{1t} & a_{2t} \end{bmatrix}.
$$ (eq:Da)

The identifiable parameters from the likelihood function — i.e., the parameters
that characterize the distribution of $(x_t, \mu_t)$ via the innovation
recursion {eq}`eq:a1_recursion`–{eq}`eq:a2_recursion` — are
$(\lambda, \sigma_{11}, \sigma_{12}, \sigma_{22})$ — four parameters in all.
From these four, we would like to recover the five structural parameters
$(\alpha, \lambda, \sigma_\delta^2, \sigma_\varepsilon^2, \sigma_{\delta\varepsilon})$.

The mapping from structural to identifiable parameters is:

$$
\sigma_{11} = \phi^2(1+\lambda^2)(\sigma_\delta^2 - 2\sigma_{\delta\varepsilon}
              + \sigma_\varepsilon^2)
$$ (eq:sigma11)

$$
\sigma_{12} = (1-\lambda)\phi \sigma_{11}^{1/2}
              \bigl[\phi(1-\lambda)(\sigma_\delta^2 - \sigma_{\delta\varepsilon})
              + \sigma_{\delta\varepsilon}\bigr]^{1/2}
\quad (\text{schematically})
$$ (eq:sigma12)

$$
\sigma_{22} = \bigl(\phi(1-\lambda)+1\bigr)^2\sigma_\delta^2
              + \bigl(\phi(1-\lambda)\bigr)^2 \sigma_\varepsilon^2
              - 2\phi(1-\lambda)\bigl(\phi(1-\lambda)+1\bigr)\sigma_{\delta\varepsilon}
              + \sigma_\varepsilon^2   \quad (\text{schematically})
$$ (eq:sigma22)

One can show that **there exist offsetting changes** in $\sigma_\delta^2$ and
$\sigma_{\delta\varepsilon}$ that leave $\sigma_{11}$, $\sigma_{12}$, and
$\sigma_{22}$ all unchanged.  Consequently, $\sigma_\varepsilon^2$ and $\alpha$
are **not separately identifiable** from the likelihood alone.

To proceed it is necessary to impose a restriction on $\sigma_{\delta\varepsilon}$.
{cite:t}`Sargent77hyper` imposes $\sigma_{\delta\varepsilon} = 0$ — i.e., shocks to
the money supply rule and shocks to portfolio balance are uncorrelated — and then
estimates $\alpha$ from the formula

$$
\hat\alpha =
\frac{(1-\lambda)^{-1}\sigma_{11}^{1/2}
      \bigl[(1-\lambda)^{-1}\sigma_{11}^{1/2} - \sigma_{12}\bigr]}
     {\sigma_{22} - \sigma_{12}^2/\sigma_{11}}.
$$ (eq:alpha_estimator)

```{note}
The estimates of $\alpha$ obtained by imposing $\sigma_{\delta\varepsilon} = 0$
are sensitive to the covariance matrix of the forecast errors and should be
regarded as **very delicate**.
```

### Maximum likelihood estimator

The likelihood function for a sample $t = 1, \ldots, T$ is

$$
L(\lambda, \sigma_{11}, \sigma_{12}, \sigma_{22} \mid x_t, \mu_t)
= (2\pi)^{-T} |D_a|^{-T/2}
  \exp\!\left(-\tfrac{1}{2} \sum_{t=1}^T a_t^\top D_a^{-1} a_t\right)
$$ (eq:likelihood)

where the innovations $a_t = (a_{1t}, a_{2t})^\top$ are recovered by iterating
{eq}`eq:a1_recursion`–{eq}`eq:a2_recursion` given initial conditions $a_{10} = 0$.

Maximizing {eq}`eq:likelihood` is equivalent to minimizing with respect to
$\lambda$ the determinant

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

The MLE criterion attains its minimum near the true value $\lambda = 0.6$.

## An Alternative Instrumental Variable Estimator

When $\sigma_{\delta\varepsilon} = 0$ (shocks to money demand and money supply
are uncorrelated), an **instrumental variable (IV) estimator** is available.

From the vector autoregressive representation {eq}`eq:varma_levels`, the
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

## Testing the Model

### The overparameterized system

Representation {eq}`eq:arima_x`–{eq}`eq:mu_eq_x_plus_innovations` is a special
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

In the restricted model {eq}`eq:arima_x`, **seven linear restrictions** have been
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
of the restricted model {eq}`eq:arima_x`.

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
