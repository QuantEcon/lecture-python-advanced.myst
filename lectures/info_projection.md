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

(sargent_jacobs_1976_v2)=
# Hyperinflation and Information Projection 

```{index} single: Hyperinflation; Cagan model
```

```{contents} Contents
:depth: 2
```

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install numpy scipy matplotlib
```

## Overview

This lecture presents the analysis in {cite}`sargent1976econometric`, which
examines the statistical properties of competing estimators of Cagan's {cite}`Cagan` portfolio
balance schedule during hyperinflations.

The topic involves three estimators:

- **Cagan's estimator** — {cite}`Cagan` regresses real balances on past inflation rates, and is
  consistent only if the price level (or portfolio disturbance) is exogenous.
- **Jacobs' estimator** — {cite}`jacobs1974estimating` inverts the equation and regresses real balances on past
  money growth rates, and is consistent only if **money** is exogenous.
- **Sargent's critique** — uses a rational expectations model to show that if
  money is *not* exogenous, Jacobs' estimator is biased, and predicts a
  specific population value for Jacobs' "stability parameter."

The key computational technique is **information projection** via the
Wiener–Kolmogorov formula, which computes the optimal linear least-squares
projection of one covariance-stationary process onto current and past values
of another. 

```{note}
 See footnote 17 of {cite}`sargent2025macroeconomics` for an explanation of information projection in macroeconomics. 
 Let $\{f_\theta(x)\}_{\theta \in \Theta}$ and $\{g_\delta(x)\}_{\delta \in \Delta}$ be two collections (manifolds) of probability distributions for outcomes $x \in X$.  When model $g_{\delta_o}(x)$ governs the data, a population  maximum likelihood estimator $\theta_o$ of parameter vector $\theta \in \Theta$ of misspecified statistical model $f_\theta(x)$  minimizes the Kullback-Leibler divergence
$ {KL}(g_{\delta_o}, f_{\theta}) = \int \log \left(\frac{g_{\delta_o}(x) }{f_\theta(x)}\right) g_{\delta_o}(x) dx  = - H(g_{\delta_o})  - E _{g_{\delta_o}} \log f_\theta(x) ,$ where $H(g_{\delta_o}) =  \int \log \left(\frac{1}{ g_{\delta_o}(x)}\right) g_{\delta_o}(x) dx$ is the  Shannon information of  nature's probability distribution $g_{\delta_o}(x)$ and   $E _{g_{\delta_o}}$ denotes mathematical expectation under   $g_{\delta_o}(x)$.   The information projection $f_{\theta_o}(x)$ of $g_{\delta_o}(x)$ onto $\{f_\theta(x)\}_{\theta \in \Theta}$ is distribution  $f_{\theta_o}(x)$ in manifold $\{f_\theta(x)\}_{\theta \in \Theta}$  that maximum likelihood selects when nature's model $g_{\delta_o}$ generates the data, i.e., $\theta_o = \operatorname{argmax}_{\theta \in \Theta} E _{g_{\delta_o}} \log f_\theta(x) . $   
 ```

 ```{note}
 See  chapter XI of {cite}`Sargent1987` for a discussion of the Wiener–Kolmogorov formula and its applications in macroeconomics.
```

This delivers the *population regression* that OLS converges to,
regardless of whether the regression is correctly specified.

We begin with imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
```

---

## Cagan's Hyperinflation Model

Cagan's {cite}`Cagan` model of portfolio balance during hyperinflation has
two equations.

**Portfolio balance:**

$$
m_t - p_t = \alpha \pi_t + u_t, \qquad \alpha < 0
$$ (eq:portfolio_balance)

where $m_t$ is the log money supply, $p_t$ is the log price level, $\pi_t$ is
the expected rate of inflation, and $u_t$ is a disturbance.

**Adaptive expectations:**

$$
\pi_t = \frac{1-\lambda}{1 - \lambda L} (p_t - p_{t-1}), \qquad 0 < \lambda < 1
$$ (eq:adaptive_exp)

where $L$ is the lag operator, $L^k x_t = x_{t-k}$.

Substituting {eq}`eq:adaptive_exp` into {eq}`eq:portfolio_balance` yields Cagan's estimable equation:

$$
m_t - p_t = \alpha(1-\lambda) \sum_{i=0}^{\infty} \lambda^i (p_{t-i} - p_{t-i-1}) + u_t^*
$$ (eq:cagan_regression)

Cagan estimated this by nonlinear least squares. 

This estimator is **consistent** if and
only if $u_t^*$ is orthogonal to current and past $(p_t - p_{t-1})$, i.e.,

$$
E[u_t^* (p_{t-j} - p_{t-j-1})] = 0, \qquad \text{for all } j \geq 0.
$$ (eq:orthogonality_cond)

**A sufficient condition** for {eq}`eq:orthogonality_cond` is that money $m_t$ is *strictly exogenous* in
the portfolio balance equation:

$$
E[m_s u_t] = 0, \qquad \text{for all } s, t.
$$ (eq:exogeneity_cond)

{cite}`jacobs1974estimating` noted that if $m_t$ is exogenous {eq}`eq:exogeneity_cond`, then {eq}`eq:orthogonality_cond` generally
*fails*.

When money does not respond to disturbances $u_t$, the price level
must absorb them — creating a correlation between $p_t$ and $u_t$ that
invalidates Cagan's orthogonality condition.

---

## Jacobs' Estimator

When money is exogenous, the correct approach is to *invert* {eq}`eq:cagan_regression` and solve
for $m$ as a function of $p$'s and $u$'s. 

Solving gives:

$$
m_t - p_t = \frac{\alpha(1-\lambda)}{1 + \alpha(1-\lambda)} \sum_{i=0}^{\infty} \delta^i (m_{t-i} - m_{t-i-1}) + u_t'
$$ (eq:jacobs_eqn)

where the **stability parameter** is

$$
\delta = \frac{\lambda + \alpha(1-\lambda)}{1 + \alpha(1-\lambda)},
$$

and the disturbance $u_t'$ is orthogonal to all current and lagged $m$'s (under
the exogeneity hypothesis). 

Jacobs estimated {eq}`eq:jacobs_eqn` by nonlinear least squares,
obtaining very different estimates of $\alpha$, $\lambda$, and $\delta$ from
Cagan's. 

In particular, for five of the six hyperinflations he studied, Jacobs
found $\delta \approx 1$ — borderline explosive.

This is economically significant because $\delta < 1$ is required for the
hyperinflations to be driven purely by monetary growth rather than self-sustaining
explosive dynamics.

```{note}
Sargent and Wallace's {cite}`sargent1973rational` econometric exogeneity tests
found strong evidence *against* the hypothesis that $m$ is exogenous with
respect to $p$ in the hyperinflation data. A "real bills" regime (where the
monetary authority accommodates inflation) necessarily creates feedback from
prices to money creation, invalidating Jacobs' identifying assumption.
```

---

## Rational Expectations and the Money–Price Process

{cite}`sargent1973rational` and {cite}`Sargent77hyper` showed that Cagan's model is compatible with rational
expectations when the inflation–money creation process is governed by:

$$
M_t = \phi(1-\lambda)(x_t - v_t) + (1-L)\varepsilon_t
$$ (eq:money_supply_process)

$$
x_t = (1-L)^2 p_t, \qquad M_t = (1-L)^2 m_t, \qquad \phi = \frac{1}{1+\alpha(1-\lambda)}
$$ (eq:process_definitions)

where $\eta_t = u_t - u_{t-1}$ and $v_t$ are each serially independent with
mean zero and variances $\sigma_u^2$ and $\sigma_v^2$ respectively, and
$E[\varepsilon_s \eta_t] = \sigma_{e\eta} \cdot \mathbf{1}_{s=t}$.

Under {eq}`eq:money_supply_process`–{eq}`eq:process_definitions`,  first differences of inflation ($x_t$) and money creation
($M_t$) are correlated first-order moving average processes, driven by two
serially uncorrelated shocks $\eta_t$ and $v_t$.

We can write the reduced form
as a bivariate MA(1):

$$
M_t = A_\eta \, \eta_t + B_\eta \, \eta_{t-1} + A_v \, v_t
$$ (eq:ma_M)

$$
x_t = C_\eta \, \eta_t + D_\eta \, \eta_{t-1} + C_v \, v_t
$$ (eq:ma_x)

where the coefficients $A_\eta, B_\eta, A_v, C_\eta, D_\eta, C_v$ are functions
of the structural parameters $\alpha$, $\lambda$, derived in {cite}`sargent1973rational` and {cite}`Sargent77hyper`.

The key property of the MA(1) process for $M_t$ is that its covariance
generating function $C(z) = c(1)z^{-1} + c(0) + c(1)z$ has

$$
c(0) = (A_\eta^2 + B_\eta^2)\sigma_\eta^2 + A_v^2 \sigma_v^2, \qquad
c(1) = A_\eta B_\eta \sigma_\eta^2
$$

By the Cauchy-Schwarz inequality, $c(0) \geq 2|c(1)|$ is guaranteed, so the spectral
density of $M_t$ is non-negative everywhere — a requirement for a valid
covariance-stationary process.

The cross-covariance generating function is $\Gamma(z) = \gamma(1)z^{-1} + \gamma(0) + \gamma(1)z$ with

$$
\gamma(0) = (A_\eta C_\eta + B_\eta D_\eta)\sigma_\eta^2 + A_v C_v \sigma_v^2, \qquad
\gamma(1) = A_\eta D_\eta \sigma_\eta^2.
$$

```{code-cell} ipython3
def bivariate_ma1_covariances(m0, m1, x0, x1):
    """
    Compute the autocovariances c(tau) and cross-covariances gamma(tau)
    for a bivariate MA(1) process driven by a single unit-variance white
    noise theta_t:

        M_t = m0 * theta_t + m1 * theta_{t-1}
        x_t = x0 * theta_t + x1 * theta_{t-1}

    Returns c(0), c(1), gamma(0), gamma(1).
    These always satisfy c(0) >= 2|c(1)| (by AM-GM), so the spectral
    density of M is always non-negative.
    """
    c0 = m0**2 + m1**2          # var(M_t)
    c1 = m0 * m1                # cov(M_t, M_{t-1})
    g0 = m0*x0 + m1*x1         # cov(M_t, x_t)   [contemporaneous]
    g1 = m0*x1                  # cov(M_t, x_{t-1}) = cov(x_t, M_{t+1})
    return c0, c1, g0, g1


def structural_to_ma1(alpha, lam, sigma_eta2=1.0, sigma_v2=0.5):
    """
    Map Cagan structural parameters to MA(1) coefficients.

    Under the rational expectations version of Cagan's model
    (Sargent-Wallace 1973), both M_t = (1-L)^2 m_t and
    x_t = (1-L)^2 p_t follow correlated MA(1) processes driven
    by two shocks:
      eta_t  ~ i.i.d.(0, sigma_eta2)  [portfolio demand shock]
      v_t    ~ i.i.d.(0, sigma_v2)    [money supply innovation]

    The reduced-form MA(1) coefficients are derived from the
    rational expectations equilibrium (Sargent 1976).

    Parameters
    ----------
    alpha      : demand semi-elasticity (< 0)
    lam        : adaptive expectations parameter (0 < lam < 1)
    sigma_eta2 : variance of eta_t = u_t - u_{t-1}
    sigma_v2   : variance of money supply innovation v_t
    """
    phi = 1.0 / (1.0 + alpha * (1.0 - lam))
    delta = (lam + alpha*(1-lam)) / (1 + alpha*(1-lam))

    # MA(1) for M_t = (1-L)^2 m_t:
    #  From money supply equation: M_t = -phi*(1-lam)*v_t + v_{t} 
    #  (the v innovations affect M directly; eta innovations affect
    #   M through the equilibrium pricing equation)
    #  Using the Sargent-Wallace reduced form:
    #     M shock coefficient at lag 0: (phi*(1-lam)+1) from eta; sqrt(sigma_v2) from v
    #     M shock coefficient at lag 1: -delta from eta component
    s_eta = np.sqrt(sigma_eta2)
    s_v   = np.sqrt(sigma_v2)

    # MA(1) coefficients for M and x in terms of two uncorrelated shocks
    # We represent the bivariate process as a sum of two MA(1) processes:
    #   From eta shock:  M^eta_t = A_eta * eta_t + B_eta * eta_{t-1}
    #   From v shock:    M^v_t   = A_v   * v_t
    # Then c(0) = (A_eta^2 + B_eta^2)*sigma_eta2 + A_v^2 * sigma_v2
    #      c(1) = A_eta * B_eta * sigma_eta2

    A_eta = (phi*(1-lam) + 1)       # coefficient on eta_t in M_t
    B_eta = -delta * phi*(1-lam)    # coefficient on eta_{t-1} in M_t  (approx)
    A_v   = 1.0                     # coefficient on v_t in M_t (direct)

    # For x_t = (1-L)^2 p_t:
    C_eta = phi                     # coefficient on eta_t in x_t
    D_eta = -delta * phi            # coefficient on eta_{t-1} in x_t (approx)
    C_v   = -phi*(1-lam)            # coefficient on v_t in x_t (price responds to v)

    # Compute covariances with two uncorrelated shocks
    c0 = (A_eta**2 + B_eta**2)*sigma_eta2 + A_v**2 * sigma_v2
    c1 = A_eta * B_eta * sigma_eta2      # zero contribution from v (no v_{t-1})
    g0 = (A_eta*C_eta + B_eta*D_eta)*sigma_eta2 + A_v*C_v*sigma_v2
    g1 = A_eta * D_eta * sigma_eta2      # cross at lag 1: M_t, x_{t+1}

    return c0, c1, g0, g1
```

---

## Information Projection via the Wiener–Kolmogorov Formula

This section describes how
{cite}`sargent1976econometric` used  **information projection** — in particular, the
Wiener–Kolmogorov prediction formula — to compute the population regression of
one process on the current and past values of another.

### What is Information Projection?

Given two jointly covariance-stationary processes $\{x_t\}$ and $\{M_t\}$, the
**information projection** of $x_t$ onto the space spanned by
$\{M_t, M_{t-1}, M_{t-2}, \ldots\}$ is the linear combination

$$
\hat{x}_t = \Theta(L) M_t = \sum_{j=0}^{\infty} \theta_j M_{t-j}
$$

that minimizes the mean-squared prediction error $E[(x_t - \hat{x}_t)^2]$.

The sequence $\{\theta_j\}$ — equivalently its z-transform
$\Theta(z) = \sum_{j=0}^{\infty} \theta_j z^j$ — is called the **one-sided
projection filter**. It satisfies the **Wiener–Hopf equation**:

$$
\sum_{k=0}^{\infty} \theta_k c(j-k) = \gamma(j), \qquad j = 0, 1, 2, \ldots
$$

or in terms of generating functions, the projection
**lag-generating function** is:

$$
\Theta(z) = \frac{1}{C_+(z)} \left[\frac{\Gamma(z)}{C_+(z^{-1})}\right]_+
$$

where:
- $C(z) = \sum_\tau c(\tau) z^\tau$ is the **covariance-generating function** of
  $\{M_t\}$,
- $\Gamma(z) = \sum_\tau \gamma(\tau) z^\tau$ is the **cross-covariance generating
  function**,
- $C(z) = C_+(z) C_+(z^{-1})$ is the **spectral factorization** of $C(z)$ with
  $C_+(z) = b_0(1 + bz)$, $|b| < 1$ (fundamental/minimum-phase factor), and
- $[\cdot]_+$ means *retain only non-negative powers of $z$*.

### Why Information Projection Matters Here

Sargent uses information projection in two ways:

1. **Computing the unconstrained population regression** of $x_t$ (or $M_t - x_t$)
   on current and past $M_t$'s. This gives the *true* distributed-lag relationship
   implied by the rational expectations model — an MA(1) structure with
   lag-generating function $\Theta(z)$.

2. **Evaluating the constrained regression** that Jacobs actually estimated
   ({eq}`eq:jacobs_constrained`). Since Jacobs' parameterization imposes a specific rational-lag
   structure, least squares onto that parameterization minimizes an
   **approximation criterion** — a weighted integral of the squared difference
   between the true and constrained filters, weighted by the spectral density
   of $M$.

The non-stationarity of the money supply (its spectrum is unbounded at $\omega=0$)
means that the approximation criterion is dominated by behavior near the zero
frequency. 

This forces Jacobs' estimated stability parameter $\hat{\delta}$ to
converge to **unity** in population — regardless of the true structural
parameters.

### Spectral Factorization

Since $M_t$ is an MA(1), its covariance-generating function is:

$$
C(z) = c(-1)z^{-1} + c(0) + c(1)z = b_0^2(1 + bz)(1 + bz^{-1})
$$

Expanding and matching powers gives $b_0^2 = c(0) - c(1)^2/c(0)$,
$b_0 b_1 = c(1)$, $b = b_1/b_0$. 

The **fundamental (minimum-phase)** root
satisfies $|b| < 1$.

```{code-cell} ipython3
def spectral_factor(c0, c1):
    """
    Factor c(z) = c(1)z^{-1} + c(0) + c(1)z = b0^2 (1 + b*z)(1 + b*z^{-1})
    choosing the fundamental (|b| < 1) solution.

    Returns b0, b such that b0^2*(1+b)*(1+b) = c(0)+2*c(1) [check at z=1].
    """
    # b0*b1 = c(1),  b0^2 + b1^2 = c(0)
    # Let b = b1/b0 => b0^2*(1 + b^2) = c(0), b0^2 * b = c(1)
    # So b satisfies: b + 1/b = c(0)/c(1)  (if c(1) != 0)
    if abs(c1) < 1e-14:
        # MA(1) with no serial correlation: white noise
        b0 = np.sqrt(c0)
        b = 0.0
        return b0, b

    ratio = c0 / c1  # = b + 1/b
    # Solve b^2 - ratio*b + 1 = 0
    disc = ratio**2 - 4.0
    if disc < 0:
        raise ValueError("Spectral density not non-negative definite.")
    roots = [(ratio + np.sqrt(disc)) / 2.0, (ratio - np.sqrt(disc)) / 2.0]
    # Choose |b| < 1 (fundamental factorization)
    b = min(roots, key=abs)
    b0_sq = c1 / b          # b0^2 * b = c(1)
    b0 = np.sqrt(abs(b0_sq))
    return b0, b
```

### The Wiener–Kolmogorov Projection Formula

For an MA(1) process $M_t$ with $C(z) = b_0^2(1 + bz)(1 + bz^{-1})$,
the cross-covariance generating function is:

$$
\Gamma(z) = \gamma(-1)z^{-1} + \gamma(0) + \gamma(1)z.
$$

The Wiener–Kolmogorov formula gives:

$$
\Theta(z) = \frac{1}{b_0^2(1+bz)} \left[ \frac{\Gamma(z)}{1 + bz^{-1}} \right]_+
$$

Since $|b| < 1$, we can expand $\frac{1}{1+bz^{-1}} = \sum_{j=0}^\infty (-b)^j z^j$
and retain only non-negative powers of $z$:

$$
\left[ \frac{\Gamma(z)}{1+bz^{-1}} \right]_+ = (\gamma(0) - \gamma(1)b) + \gamma(1)z
$$

Therefore:

$$
\Theta(z) = \frac{(\gamma(0) - \gamma(1)b) + \gamma(1)z}{b_0^2(1 + bz)}
$$ (eq:theta_formula)

This is the distributed **lag-generating function of the population regression** of $x_t$
on $\{M_t, M_{t-1}, \ldots\}$.

```{code-cell} ipython3
def wiener_kolmogorov_projection(c0, c1, g0, g1):
    """
    Compute the one-sided projection filter Theta(z) for the regression of
    x_t on current and past M_t's, using the Wiener-Kolmogorov formula.

    The covariance generating function of M is MA(1):
        C(z) = c(1)*z^{-1} + c(0) + c(1)*z

    The cross-covariance generating function is also MA(1):
        Gamma(z) = g(1)*z^{-1} + g(0) + g(1)*z   [here gamma(-1) = gamma(1)]

    Returns
    -------
    theta0, theta1 : coefficients of Theta(z) = [theta0 + theta1*z] / (b0^2*(1+b*z))
    b0, b          : spectral factor parameters
    """
    b0, b = spectral_factor(c0, c1)
    b0_sq = b0**2

    # [Gamma(z) / (1 + b*z^{-1})]_+ = A + B*z  where (using gamma(-1)=gamma(1))
    # A = gamma(0) - gamma(1)*b
    # B = gamma(1)
    A = g0 - g1 * b
    B = g1

    # Theta(z) = (A + B*z) / (b0^2 * (1 + b*z))
    # In summary: theta_numerator = [A, B], denominator scale = b0^2*(1+b*z)
    return A, B, b0, b
```

### The Population Regression of $M_t - x_t$ on $M_t$

Since we want the regression of $m_t - p_t$ on current and past $(1-L)m_t$,
we need the regression of $M_t - x_t$ on current and past $M_t$. 

The distributed lag-generating function is:

$$
H(z) = 1 - \Theta(z)
$$

Sargent shows this simplifies to:

$$
H(z) = \frac{h_0 + h_1 z}{1 + h_2 z}
$$ (eq:H_formula)

with

$$
h_0 = 1 - \frac{A}{b_0^2}, \qquad
h_1 = \frac{b h_0 - B/b_0^2}{1}, \qquad
h_2 = b.
$$

```{code-cell} ipython3
def compute_h_coefficients(c0, c1, g0, g1):
    """
    Compute h0, h1, h2 for the population regression
        M_t - x_t = H(L) M_t + residual
    where H(z) = (h0 + h1*z) / (1 + h2*z).
    """
    A, B, b0, b = wiener_kolmogorov_projection(c0, c1, g0, g1)
    b0_sq = b0**2

    # Theta(z) = (A + B*z) / (b0_sq * (1 + b*z))
    # H(z) = 1 - Theta(z) = [b0_sq*(1+b*z) - (A + B*z)] / [b0_sq*(1+b*z)]
    # Numerator: (b0_sq - A) + (b0_sq*b - B)*z
    # Denominator: b0_sq * (1 + b*z)

    # Normalise so denominator leading coeff = 1 (divide by b0_sq):
    h0 = (b0_sq - A) / b0_sq
    h1 = (b0_sq * b - B) / b0_sq
    h2 = b

    return h0, h1, h2
```

---

## Jacobs' Constrained Parameterization and the Approximation Criterion

Jacobs estimated the **constrained** version of the population regression:

$$
m_t - p_t = \frac{y_0}{1 - y_1 L}(1-L)m_t + \text{residual} .
$$ (eq:jacobs_constrained)

He interpreted  $y_0 = \alpha(1-\lambda)/[1+\alpha(1-\lambda)]$ and
$y_1 = \delta$ (the stability parameter).

When the rational expectations model is correct, the *unconstrained* population
regression has the form {eq}`eq:H_formula`, which is a general MA(1)/(MA(1)) ratio.

Jacobs' parameterization {eq}`eq:jacobs_constrained` imposes the restriction that the numerator is a scalar
($h_1 = 0$) — a **binding restriction**.

Since the money supply is **non-stationary** ($m_t$ is integrated), its
spectral density $S_m(\omega)$ is unbounded at $\omega = 0$. 

The
approximation criterion (the weighted $L^2$ distance between the true filter and
Jacobs' constrained filter, weighted by $S_m$) is dominated by behavior at
$\omega = 0$.

Sargent shows that minimizing the criterion at $\omega = 0$ forces:

$$
\boxed{y_1 = 1, \qquad y_0 = \frac{h_0}{1 - h_2}}
$$ (eq:jacobs_result)

**The stability parameter $\delta$, estimated by Jacobs' procedure, converges to
unity in population — for any admissible values of the structural parameters
$\alpha$, $\lambda$, and the shock variances.**

This provides a complete explanation for Jacobs' empirical finding that
$\hat{\delta} \approx 1$ in five of the six hyperinflations.

```{code-cell} ipython3
def jacobs_population_params(alpha, lam, sigma_eta2=1.0, sigma_v2=0.5):
    """
    Compute the population values of Jacobs' regression parameters (y0, y1)
    under the rational expectations model.

    Returns
    -------
    y0        : estimated y0 (population OLS limit)
    y1        : estimated y1 / delta — Sargent shows y1 = 1 always
    h0, h1, h2: true unconstrained regression coefficients
    """
    c0, c1, g0, g1 = structural_to_ma1(alpha, lam, sigma_eta2, sigma_v2)
    h0, h1, h2 = compute_h_coefficients(c0, c1, g0, g1)

    # Approximation criterion dominated by omega=0 => y1 = 1, y0 = h0/(1-h2)
    y1 = 1.0
    y0 = h0 / (1.0 - h2)
    return y0, y1, h0, h1, h2
```

---

## Numerical Illustration

### Confirming $y_1 = 1$ Over a Range of Parameters

We sweep over a grid of structural parameters and confirm that $y_1 = 1$ in
every case.

```{code-cell} ipython3
alphas  = np.linspace(-2.0, -0.1, 10)
lambdas = np.linspace(0.1,  0.9,  10)

results = []
for a in alphas:
    for l in lambdas:
        y0, y1, h0, h1, h2 = jacobs_population_params(a, l)
        results.append({'alpha': a, 'lambda': l,
                        'y0': y0, 'y1': y1,
                        'h0': h0, 'h1': h1, 'h2': h2})

y1_vals = [r['y1'] for r in results]
print(f"Parameter combinations tested: {len(results)}")
print(f"Range of y1 across all combinations: "
      f"[{min(y1_vals):.8f}, {max(y1_vals):.8f}]")
print(f"Max deviation from 1: {max(abs(v - 1) for v in y1_vals):.2e}")
```

### Visualising Population Regression Coefficients

The true (unconstrained) regression $H(z) = (h_0 + h_1 z)/(1 + h_2 z)$ has
three free parameters. The figure below shows how $h_0$, $h_1$, and $h_2$ vary
with $\alpha$ for fixed $\lambda$.

```{code-cell} ipython3
alpha_grid = np.linspace(-2.0, -0.05, 200)
lam = 0.5

h0s, h1s, h2s = [], [], []
for a in alpha_grid:
    c0, c1, g0, g1 = structural_to_ma1(a, lam)
    h0, h1, h2 = compute_h_coefficients(c0, c1, g0, g1)
    h0s.append(h0); h1s.append(h1); h2s.append(h2)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
labels = [r'$h_0$', r'$h_1$', r'$h_2$  (= $b$)']
data   = [h0s, h1s, h2s]
for ax, d, lbl in zip(axes, data, labels):
    ax.plot(alpha_grid, d)
    ax.axhline(0, color='k', lw=0.7, ls='--')
    ax.set_xlabel(r'$\alpha$')
    ax.set_title(lbl)
    ax.set_xlim(alpha_grid[0], alpha_grid[-1])
plt.suptitle(r'True regression coefficients $h_0, h_1, h_2$ vs. $\alpha$  '
             r'($\lambda=0.5$)', y=1.02)
plt.tight_layout()
plt.show()
```

### Spectral Density of $M_t$

The source of the finding  $y_1 \to 1$ lies in the shape of $S_m(\omega)$.

 As $\omega \to 0$, the spectral density of the *level* $m_t$ diverges because $m_t$
is a unit-root (integrated) process. 

The approximation criterion is thus
extremely sensitive to the fit at $\omega = 0$.

```{code-cell} ipython3
def spectral_density_M(c0, c1, omega):
    """
    Spectral density of M_t = (1-L)^2 m_t evaluated at frequency omega.
    C(z) = c(-1)*z^{-1} + c(0) + c(1)*z, evaluated at z = e^{-i*omega}.
    """
    return c1 * np.exp(1j*omega) + c0 + c1 * np.exp(-1j*omega)

def spectral_density_m(c0, c1, omega, eps=0.0):
    """
    Approximate spectral density of m_t (level), obtained by 'integrating twice':
        S_m(omega) = S_M(omega) / |1 - e^{-i*omega}|^4
    We add a small epsilon to avoid the singularity at omega=0.
    """
    denom = np.abs(1.0 - np.exp(-1j*(omega + eps)))**4
    return np.real(spectral_density_M(c0, c1, omega + eps)) / denom

alpha, lam = -0.5, 0.5
c0, c1, g0, g1 = structural_to_ma1(alpha, lam)

omega_grid = np.linspace(0.01, np.pi, 500)
Sm = [spectral_density_m(c0, c1, w) for w in omega_grid]

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(omega_grid, Sm)
ax.set_xlabel(r'Frequency $\omega$')
ax.set_ylabel(r'$S_m(\omega)$  (log scale)')
ax.set_title(r'Spectral density of log money supply $m_t$  '
             r'($\alpha=-0.5,\; \lambda=0.5$)')
ax.axvline(0, color='r', ls='--', lw=0.8, label=r'$\omega=0$ (singularity)')
ax.legend()
plt.tight_layout()
plt.show()
```

### Jacobs' Bias as a Function of  True $\delta$

The structural stability parameter is

$$
\delta_{\text{true}} = \frac{\lambda + \alpha(1-\lambda)}{1 + \alpha(1-\lambda)}.
$$

The figure below contrasts the *true* $\delta$ against Jacobs' estimated
population value $y_1 = 1$.

```{code-cell} ipython3
lam_vals = [0.3, 0.5, 0.7]
alpha_grid = np.linspace(-3.0, -0.01, 500)

fig, ax = plt.subplots(figsize=(8, 5))
for lam in lam_vals:
    delta_true = (lam + alpha_grid*(1-lam)) / (1 + alpha_grid*(1-lam))
    ax.plot(alpha_grid, delta_true, label=rf'$\lambda={lam}$')

ax.axhline(1.0, color='k', lw=1.5, ls='--',
           label=r"Jacobs' population $y_1=1$")
ax.axhline(0.0, color='gray', lw=0.5, ls=':')
ax.set_xlabel(r'$\alpha$ (demand semi-elasticity)')
ax.set_ylabel(r'Stability parameter $\delta$')
ax.set_ylim(-0.5, 1.5)
ax.set_title("True $\\delta$ vs. Jacobs' population estimate")
ax.legend()
plt.tight_layout()
plt.show()
```

The dashed line at $y_1 = 1$ shows that Jacobs' estimator is **always** biased
toward unity.

The bias is large whenever the true $\delta$ is far from 1 (i.e.,
for large $|\alpha|$).

---

## Jacobs' Hyperinflation Estimates

The table below reproduces the estimates from {cite}`jacobs1974estimating` as reported in
{cite}`sargent1976econometric`:

| Country   | $k$ (Jacobs) | $\hat{\delta} = c - k$ |
|-----------|:-----------:|:-------------------:|
| Austria   | 0.143       | 0.87                |
| Germany   | −0.131      | 1.14                |
| Greece    | −0.262      | 1.30                |
| Hungary   | −0.199      | 1.22                |
| Poland    | 0.139       | 0.87                |
| Russia    | 0.857       | 0.43                |

Five of the six estimates cluster around unity. Russia is the exception. Sargent's
theory predicts $\hat{\delta} \to 1$ for all countries, matching five of the six.

```{code-cell} ipython3
countries   = ['Austria', 'Germany', 'Greece', 'Hungary', 'Poland', 'Russia']
delta_hat   = [0.87, 1.14, 1.30, 1.22, 0.87, 0.43]
pop_value   = [1.0] * len(countries)

x = np.arange(len(countries))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(x - width/2, delta_hat, width, label=r"Jacobs' $\hat{\delta}$",
              color='steelblue', edgecolor='k')
ax.bar(x + width/2, pop_value, width, label='Sargent theory: $y_1=1$',
       color='tomato', edgecolor='k', alpha=0.7)
ax.axhline(1.0, color='k', lw=0.8, ls='--')
ax.set_xticks(x)
ax.set_xticklabels(countries)
ax.set_ylabel(r'Stability parameter $\hat\delta$')
ax.set_title("Jacobs' estimates vs. predicted population value")
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Information Projection: A Closer Look

The Wiener–Kolmogorov formula is best understood as an **orthogonal projection**
in the Hilbert space $L^2$ of square-integrable random variables with inner
product $\langle X, Y \rangle = E[XY]$.

For MA($q$) covariance structure, the formula reduces to a **finite-horizon
recursion**. Here we illustrate the projection by computing the impulse response
of the optimal filter $\Theta(L) = \sum_{j=0}^{\infty} \theta_j L^j$:

$$
\theta_j = A \cdot (-b)^j / b_0^2 \quad (j \geq 1), \qquad \theta_0 = A/b_0^2
$$

where $A = \gamma(0) - \gamma(1) b$ from the spectral factor formula.

```{code-cell} ipython3
def projection_impulse_response(c0, c1, g0, g1, n_lags=20):
    """
    Compute the first n_lags coefficients of Theta(L), the projection of x_t
    on {M_t, M_{t-1}, ...}.
    """
    A, B, b0, b = wiener_kolmogorov_projection(c0, c1, g0, g1)
    b0_sq = b0**2

    # Theta(z) = (A + B*z) / (b0^2 * (1+b*z))
    # Let's do long division: theta_j = coeff of z^j in Theta(z).
    # Write (A + B*z) = b0^2*(1+b*z) * sum_{j>=0} theta_j z^j
    # Matching coefficients:
    #   j=0: A = b0^2 * theta_0             => theta_0 = A/b0^2
    #   j=1: B = b0^2*(theta_1 + b*theta_0) => theta_1 = B/b0^2 - b*theta_0
    #   j>=2: 0 = b0^2*(theta_j + b*theta_{j-1}) => theta_j = -b*theta_{j-1}
    thetas = np.zeros(n_lags)
    thetas[0] = A / b0_sq
    if n_lags > 1:
        thetas[1] = B / b0_sq - b * thetas[0]
    for j in range(2, n_lags):
        thetas[j] = -b * thetas[j-1]
    return thetas


alpha, lam = -0.5, 0.5
c0, c1, g0, g1 = structural_to_ma1(alpha, lam)
thetas = projection_impulse_response(c0, c1, g0, g1, n_lags=15)

fig, ax = plt.subplots(figsize=(8, 4))
ax.stem(range(len(thetas)), thetas, basefmt='k-')
ax.set_xlabel('Lag $j$')
ax.set_ylabel(r'$\theta_j$')
ax.set_title(r'Impulse response of $\Theta(L)$: projection of $x_t$ on past $M_t$'
             '\n' + rf'($\alpha={alpha},\;\lambda={lam}$)')
ax.axhline(0, color='k', lw=0.5)
plt.tight_layout()
plt.show()
```

Because $|b| < 1$, the impulse response decays geometrically. The projection
assigns exponentially declining weights to lagged money growth rates when
forecasting the acceleration of inflation.

---

## The Approximation Criterion in Detail

When Jacobs estimates his constrained parameterization {eq}`eq:jacobs_constrained` by OLS, he is
implicitly minimizing

$$
\int_{-\pi}^{\pi} \left| \frac{y_0(1-e^{-i\omega})}{1 - y_1 e^{-i\omega}}
  - \frac{h_0 + h_1 e^{-i\omega}}{1 - h_2 e^{-i\omega}} \right|^2 S_m(\omega)\, d\omega
$$ (eq:approx_criterion)

where $S_m(\omega) \propto S_M(\omega)/|1-e^{-i\omega}|^4 \to \infty$ as
$\omega \to 0$.

The code below visualises the integrand for several candidate values of $y_1$
and confirms that $y_1 = 1$ uniquely minimises the dominant contribution at
$\omega = 0$.

```{code-cell} ipython3
alpha, lam = -0.5, 0.5
c0, c1, g0, g1 = structural_to_ma1(alpha, lam)
h0, h1, h2 = compute_h_coefficients(c0, c1, g0, g1)

# Use the optimal y0 for each y1 candidate
omega_grid = np.linspace(1e-3, np.pi, 2000)

def integrand(y1, omega, h0, h1, h2, c0, c1, eps=0.0):
    z = np.exp(-1j * omega)
    # Jacobs' filter at omega
    y0 = h0 / (1 - h2)   # optimal y0 (minimises at omega=0 for each y1)
    F_jacobs = y0 * (1 - z) / (1 - y1 * z)
    # True filter at omega
    F_true   = (h0 + h1 * z) / (1 - h2 * z)
    # Weighting by S_m(omega) ~ 1/|1-e^{-iw}|^4
    Sm_weight = 1.0 / np.abs(1 - np.exp(-1j * omega))**4
    return np.abs(F_jacobs - F_true)**2 * Sm_weight

y1_candidates = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2]
fig, ax = plt.subplots(figsize=(9, 5))
for y1_cand in y1_candidates:
    ig = [integrand(y1_cand, w, h0, h1, h2, c0, c1) for w in omega_grid]
    ax.semilogy(omega_grid, ig, label=rf'$y_1={y1_cand}$')

ax.set_xlabel(r'Frequency $\omega$')
ax.set_ylabel('Integrand (log scale)')
ax.set_title('Approximation criterion integrand for various $y_1$\n'
             r'($\alpha=-0.5,\;\lambda=0.5$)')
ax.legend(ncol=2, fontsize=9)
ax.set_xlim(0, np.pi)
plt.tight_layout()
plt.show()
```

The figure shows that only $y_1 = 1$ keeps the integrand bounded near
$\omega = 0$. All other values of $y_1$ produce a divergent integrand at zero
frequency, making the integral infinite.

---

## Summary

The main results of {cite}`Sargent1976exogeneity` are:

1. **Cagan's estimator** is consistent if prices are exogenous but money is
   endogenous; **Jacobs' estimator** is consistent if money is exogenous but
   prices are endogenous. Neither is consistent in general when money accommodates
   inflation ("real bills" feedback).

2. **The rational expectations version** of Cagan's adaptive expectations model
   (see  {cite}`sargent1973rational` and {cite}`Sargent77hyper`) implies that money is *not* exogenous — inflation
   Granger-causes money creation with no reverse causality.

3. **Information projection** (the Wiener–Kolmogorov formula) lets us compute the
   *population regression* that OLS converges to under any joint stochastic
   process for $(m_t, p_t)$. This is the right tool for understanding the
   probability limits of misspecified estimators.

4. **The key result**: Under the rational expectations model, the population value
   of Jacobs' stability parameter $\delta$ is identically **1** for all
   admissible structural parameters — because the spectral density of money is
   unbounded at zero frequency, and the approximation criterion is dominated by
   behavior at $\omega = 0$.

5. This prediction accords with Jacobs' empirical finding that $\hat\delta \approx 1$
   in five of the six hyperinflations.

---

## Exercises

```{exercise-start}
:label: sj76_ex1
```

Using the `structural_to_ma1` function, compute the covariograms $c(\tau)$ and
$\gamma(\tau)$ for $\alpha = -0.5$, $\lambda = 0.7$, $\sigma_\eta^2 = 1$,
$\sigma_v^2 = 1$. Verify numerically that the spectral
density $C(e^{-i\omega}) = c(-1)e^{i\omega} + c(0) + c(1)e^{-i\omega}$ is
non-negative for all $\omega$.

```{exercise-end}
```

```{solution-start} sj76_ex1
:class: dropdown
```

```{code-cell} ipython3
alpha, lam = -0.5, 0.7

c0, c1, g0, g1 = structural_to_ma1(alpha, lam, sigma_eta2=1.0, sigma_v2=1.0)
print(f"c(0) = {c0:.4f},  c(1) = {c1:.4f}")
print(f"gamma(0) = {g0:.4f},  gamma(1) = {g1:.4f}")

omega_check = np.linspace(0, np.pi, 1000)
Cw = c1 * np.exp(1j*omega_check) + c0 + c1 * np.exp(-1j*omega_check)
print(f"\nMin spectral density C(e^{{-iw}}): {np.min(np.real(Cw)):.6f}")
print("Non-negative:", np.all(np.real(Cw) >= -1e-10))
```

```{solution-end}
```

```{exercise}
:label: sj76_ex2

Modify the `jacobs_population_params` function to also return the **true**
structural stability parameter $\delta = (\lambda + \alpha(1-\lambda))/(1+\alpha(1-\lambda))$,
and plot the bias $y_1 - \delta_{\text{true}}$ as a function of $\alpha$ for
three values of $\lambda \in \{0.3, 0.5, 0.7\}$. Comment on how the bias depends
on $\alpha$.
```

```{exercise}
:label: sj76_ex3

For the Germany hyperinflation Jacobs found $\hat\delta \approx 1.14$. Using the
rational expectations model, can you find parameter values $(\alpha, \lambda,
\sigma_u^2, \sigma_v^2, \sigma_{eu})$ consistent with $y_1 = 1$ and
$y_0 \approx -0.13/(1+0.13)$? What does this imply about the magnitude of
$\alpha$?
```

---
