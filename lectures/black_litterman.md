---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(black_litterman)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

```{index} single: python
```

# Two Modifications of Mean-Variance Portfolio Theory

## Overview

This lecture describes extensions to the classical mean-variance portfolio theory summarized in our lecture {doc}`Elementary Asset Pricing Theory <asset_pricing_lph>`.

The classic theory described there assumes that a decision maker completely trusts the statistical model that he posits to govern the joint distribution of returns on a list of available assets.

Both extensions described here put distrust of that statistical model into the mind of the decision maker.

One is a model of Black and Litterman {cite}`black1992global` that imputes to the decision maker distrust of historically estimated mean returns but still complete trust of estimated covariances of returns.

The second model also imputes to the decision maker doubts about his statistical model, but now by saying that, because of that distrust, the decision maker uses a version of robust control theory described in the lecture {doc}`Robustness <robustness>`.


The famous **Black-Litterman** (1992) {cite}`black1992global` portfolio choice model was motivated by the finding that with high frequency or moderately high frequency data, means are more difficult to estimate than variances.

A model of **robust portfolio choice** that we'll describe below also begins from the same starting point.

To begin, we'll take for granted that means are more difficult to estimate than covariances and will focus on how Black and Litterman, on the one hand, and robust control theorists, on the other, would recommend modifying the **mean-variance portfolio choice model** to take that into account.

At the end of this lecture, we shall use some rates of convergence results and some simulations to verify how means are more difficult to estimate than variances.

Among the ideas in play in this lecture will be

- Mean-variance portfolio theory
- Bayesian approaches to estimating linear regressions
- A risk-sensitivity operator and its connection to robust control
  theory


In summary, we'll describe two ways to modify the classic mean-variance portfolio choice model in ways designed to make its recommendations more plausible.


Both of the adjustments that we describe are designed to confront a widely recognized embarrassment to mean-variance portfolio theory, namely, that it usually implies taking very extreme long-short portfolio positions.

They tame those positions in different ways: the Black-Litterman adjustment can reverse which assets are held short, while the robust adjustment scales all positions toward zero without changing their signs.

The two approaches build on a common and widespread hunch -- that because it is much easier statistically to estimate covariances of excess returns than it is to estimate their means, it makes sense to adjust investors' subjective beliefs about mean returns in order to render more plausible decisions.


Let's start with some imports:

```{code-cell} ipython3
import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt
from numba import jit
```

## Mean-variance portfolio choice

A risk-free security earns one-period net return $r_f$.

An $n \times 1$ vector of risky securities earns an $n \times 1$ vector $\vec r - r_f {\bf 1}$ of *excess returns*, where ${\bf 1}$ is an $n \times 1$ vector of ones.

The excess return vector is multivariate normal with mean $\mu$ and covariance matrix $\Sigma$, which we express either as

$$
\vec r - r_f {\bf 1} \sim {\mathcal N}(\mu, \Sigma)
$$

or

$$
\vec r - r_f {\bf 1} = \mu + C \epsilon
$$

where $\epsilon \sim {\mathcal N}(0, I)$ is an $n \times 1$ random vector.

Let $w$ be an $n \times 1$ vector of portfolio weights.

A portfolio consisting of $w$ earns returns

$$
w' (\vec r - r_f {\bf 1}) \sim {\mathcal N}(w' \mu, w' \Sigma w )
$$

The **mean-variance portfolio choice problem** is to choose $w$ to maximize

```{math}
:label: choice-problem

U(\mu,\Sigma;w) = w'\mu - \frac{\delta}{2} w' \Sigma w
```

where $\delta > 0$ is a risk-aversion parameter.

The first-order condition for maximizing {eq}`choice-problem` with respect to the vector $w$ is

$$
\mu = \delta \Sigma w
$$

which implies the following design of a risky portfolio:

```{math}
:label: risky-portfolio

w = (\delta \Sigma)^{-1} \mu
```

## Estimating mean and variance

The key inputs into the portfolio choice model {eq}`risky-portfolio` are

- estimates of the parameters $\mu, \Sigma$ of the random excess
  return vector $\vec r - r_f {\bf 1}$
- the risk-aversion parameter $\delta$

A standard way of estimating $\mu$ is maximum-likelihood or least squares; that amounts to estimating $\mu$ by a sample mean of excess returns and estimating $\Sigma$ by a sample covariance matrix.

## Black-Litterman starting point

When estimates of $\mu$ and $\Sigma$ from historical sample means and covariances have been combined with **plausible** values of the risk-aversion parameter $\delta$ to compute an optimal portfolio from formula {eq}`risky-portfolio`, a typical outcome has been $w$'s with **extreme long and short positions**.

A common reaction to these outcomes is that they are so implausible that a portfolio manager cannot recommend them to a customer.

```{code-cell} ipython3
np.random.seed(12)

N = 10                                           # Number of assets
T = 200                                          # Sample size

# random market portfolio (sum is normalized to 1)
w_m = np.random.rand(N)
w_m = w_m / (w_m.sum())

# True risk premia and variance of excess return (constructed
# so that the Sharpe ratio is 1)
μ = (np.random.randn(N) + 5)  /100      # Mean excess return (risk premium)
S = np.random.randn(N, N)        # Random matrix for the covariance matrix
V = S @ S.T           # Turn the random matrix into symmetric psd
# Make sure that the Sharpe ratio is one
Σ = V * (w_m @ μ)**2 / (w_m @ V @ w_m)

# Risk aversion of market portfolio holder
δ = 1 / np.sqrt(w_m @ Σ @ w_m)

# Generate a sample of excess returns
excess_return = stat.multivariate_normal(μ, Σ)
sample = excess_return.rvs(T)

# Estimate μ and Σ
μ_est = sample.mean(0).reshape(N, 1)
Σ_est = np.cov(sample.T)

w = np.linalg.solve(δ * Σ_est, μ_est)

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title('Mean-variance portfolio weights recommendation and the market portfolio')
ax.plot(np.arange(N)+1, w, 'o', c='k', label='$w$ (mean-variance)')
ax.plot(np.arange(N)+1, w_m, 'o', c='r', label='$w_m$ (market portfolio)')
ax.vlines(np.arange(N)+1, 0, w, lw=1)
ax.vlines(np.arange(N)+1, 0, w_m, lw=1)
ax.axhline(0, c='k')
ax.axhline(-1, c='k', ls='--')
ax.axhline(1, c='k', ls='--')
ax.set_xlabel('Assets')
ax.xaxis.set_ticks(np.arange(1, N+1, 1))
plt.legend(numpoints=1, fontsize=11)
plt.show()
```

Black and Litterman responded to this situation in the following way:

- They continue to accept {eq}`risky-portfolio` as a good model for choosing an optimal
  portfolio $w$.
- They want to continue to allow the customer to express his or her
  risk tolerance by setting $\delta$.
- Leaving $\Sigma$ at its maximum-likelihood value, they push
  $\mu$ away from its maximum-likelihood value in a way designed to make
  portfolio choices that are more plausible in terms of conforming to
  what most people actually do.

In particular, given $\Sigma$ and a plausible value of $\delta$, Black and Litterman reverse engineered a vector $\mu_{BL}$ of mean excess returns that makes the $w$ implied by formula {eq}`risky-portfolio` equal the **actual** market portfolio $w_m$, so that

$$
w_m = (\delta \Sigma)^{-1} \mu_{BL}
$$

## Details

Let's define

$$
w_m' \mu \equiv ( r_m - r_f)
$$

as the (scalar) excess return on the market portfolio $w_m$.

Define

$$
\sigma^2 = w_m' \Sigma w_m
$$

as the variance of the excess return on the market portfolio $w_m$.

Define

$$
{\bf SR}_m = \frac{ r_m - r_f}{\sigma}
$$

as the **Sharpe ratio** of the market portfolio $w_m$.

Let $\delta_m$ be the value of the risk aversion parameter that induces an investor to hold the market portfolio in light of the optimal portfolio choice rule {eq}`risky-portfolio`.

Evidently, portfolio rule {eq}`risky-portfolio` then implies that $r_m - r_f = \delta_m \sigma^2$ or

$$
\delta_m = \frac{r_m - r_f}{\sigma^2}
$$

or

$$
\delta_m = \frac{{\bf SR}_m}{\sigma}
$$

Following the Black-Litterman philosophy, our first step will be to back out a value of $\delta_m$ from

- an estimate of the Sharpe ratio, and
- our maximum likelihood estimate of $\sigma$ drawn from our
  estimates of $w_m$ and $\Sigma$

The second key Black-Litterman step is then to use this value of $\delta_m$ together with the maximum likelihood estimate of $\Sigma$ to deduce a $\mu_{BL}$ that verifies portfolio rule {eq}`risky-portfolio` at the market portfolio $w = w_m$

$$
\mu_{BL} = \delta_m \Sigma w_m
$$

The starting point of the Black-Litterman portfolio choice model is thus a pair $(\delta_m, \mu_{BL})$ that tells the customer to hold the market portfolio.

```{code-cell} ipython3
# Observed mean excess market return
r_m = w_m @ μ_est

# Estimated variance of the market portfolio
var_m = w_m @ Σ_est @ w_m

# Sharpe ratio
sr_m = r_m / np.sqrt(var_m)

# Risk aversion of market portfolio holder
d_m = r_m / var_m

# Derive "view" which would induce the market portfolio
μ_m = (d_m * Σ_est @ w_m).reshape(N, 1)

x = np.arange(N) + 1
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title(r'Difference between $\hat{\mu}$ (estimate) and $\mu_{BL}$ (market implied)')
ax.plot(x, μ_est, 'o', c='k', label=r'$\hat{\mu}$')
ax.plot(x, μ_m, 'o', c='r', label=r'$\mu_{BL}$')
ax.vlines(x, μ_m, μ_est, lw=1)
ax.axhline(0, c='k', ls='--')
ax.set_xlabel('Assets')
ax.xaxis.set_ticks(np.arange(1, N+1, 1))
plt.legend(numpoints=1)
plt.show()
```

## Adding views

Black and Litterman start with a baseline customer who asserts that he or she shares the **market's views**, which means that he or she believes that excess returns are governed by

```{math}
:label: excess-returns

\vec r - r_f {\bf 1} \sim {\mathcal N}( \mu_{BL}, \Sigma)
```

Black and Litterman would advise that customer to hold the market portfolio of risky securities.

Black and Litterman then imagine a customer who would like to express a view that differs from the market's.

The customer wants appropriately to mix his view with the market's before using {eq}`risky-portfolio` to choose a portfolio.

Suppose that the customer's view is expressed by a hunch that rather than {eq}`excess-returns`, excess returns are governed by

$$
\vec r - r_f {\bf 1} \sim {\mathcal N}( \hat \mu, \tau \Sigma)
$$

where $\tau > 0$ is a scalar parameter that determines how the decision maker wants to mix his view $\hat \mu$ with the market's view $\mu_{\bf BL}$.

Black and Litterman would then use a formula like the following one to mix the views $\hat \mu$ and $\mu_{\bf BL}$

```{math}
:label: mix-views

\tilde \mu = (\Sigma^{-1} + (\tau \Sigma)^{-1})^{-1} (\Sigma^{-1} \mu_{BL}  + (\tau \Sigma)^{-1} \hat \mu)
```

Black and Litterman would then advise the customer to hold the portfolio associated with these views implied by rule {eq}`risky-portfolio`:

$$
\tilde w = (\delta \Sigma)^{-1} \tilde \mu
$$

This portfolio $\tilde w$ will deviate from the market portfolio $w_m$ in amounts that depend on the mixing parameter $\tau$.

If $\hat \mu$ is the maximum likelihood estimator and $\tau$ is chosen heavily to weight this view, then the customer's portfolio will involve big short-long positions.

```{code-cell} ipython3
def black_litterman(λ, μ1, μ2, Σ1, Σ2):
    """
    Return the precision-weighted mixture of the means μ1 and μ2,

        (Σ1^{-1} + λ Σ2^{-1})^{-1} (Σ1^{-1} μ1 + λ Σ2^{-1} μ2),

    where λ scales the precision Σ2^{-1} of the second view.
    """
    Σ1_inv = np.linalg.inv(Σ1)
    Σ2_inv = np.linalg.inv(Σ2)

    μ_tilde = np.linalg.solve(Σ1_inv + λ * Σ2_inv,
                              Σ1_inv @ μ1 + λ * Σ2_inv @ μ2)
    return μ_tilde

τ = 1
μ_tilde = black_litterman(1, μ_m, μ_est, Σ_est, τ * Σ_est)

# The Black-Litterman recommendation for the portfolio weights
# (use the market-implied risk aversion d_m, so that μ_m reproduces w_m)
w_tilde = np.linalg.solve(d_m * Σ_est, μ_tilde)
```

```{code-cell} ipython3
def BL_plot(τ):
    μ_tilde = black_litterman(1, μ_m, μ_est, Σ_est, τ * Σ_est)
    w_tilde = np.linalg.solve(d_m * Σ_est, μ_tilde)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(np.arange(N)+1, μ_est, 'o', c='k',
               label=r'$\hat{\mu}$ (subj view)')
    ax[0].plot(np.arange(N)+1, μ_m, 'o', c='r',
               label=r'$\mu_{BL}$ (market)')
    ax[0].plot(np.arange(N)+1, μ_tilde, 'o', c='y',
               label=r'$\tilde{\mu}$ (mixture)')
    ax[0].vlines(np.arange(N)+1, μ_m, μ_est, lw=1)
    ax[0].axhline(0, c='k', ls='--')
    ax[0].set(xlim=(0, N+1), xlabel='Assets',
              title=r'Relationship between $\hat{\mu}$, $\mu_{BL}$, and $\tilde{\mu}$')
    ax[0].xaxis.set_ticks(np.arange(1, N+1, 1))
    ax[0].legend(numpoints=1)

    ax[1].set_title('Black-Litterman portfolio weight recommendation')
    ax[1].plot(np.arange(N)+1, w, 'o', c='k', label=r'$w$ (mean-variance)')
    ax[1].plot(np.arange(N)+1, w_m, 'o', c='r', label=r'$w_{m}$ (market, BL)')
    ax[1].plot(np.arange(N)+1, w_tilde, 'o', c='y',
               label=r'$\tilde{w}$ (mixture)')
    ax[1].vlines(np.arange(N)+1, 0, w, lw=1)
    ax[1].vlines(np.arange(N)+1, 0, w_m, lw=1)
    ax[1].axhline(0, c='k')
    ax[1].axhline(-1, c='k', ls='--')
    ax[1].axhline(1, c='k', ls='--')
    ax[1].set(xlim=(0, N+1), xlabel='Assets',
              title='Black-Litterman portfolio weight recommendation')
    ax[1].xaxis.set_ticks(np.arange(1, N+1, 1))
    ax[1].legend(numpoints=1)
    plt.show()

BL_plot(τ)
```

## Bayesian interpretation

Consider the following Bayesian interpretation of the Black-Litterman recommendation.

The prior belief over the mean excess returns is consistent with the market portfolio and is given by

$$
\mu \sim \mathcal{N}(\mu_{BL}, \Sigma)
$$

Given a particular realization of the mean excess returns $\mu$ one observes the average excess returns $\hat \mu$ on the market according to the distribution

$$
\hat \mu \mid \mu, \Sigma \sim \mathcal{N}(\mu, \tau\Sigma)
$$

where $\tau$ scales the uncertainty of the investor's own estimate $\hat \mu$ relative to the prior.

If $\hat \mu$ is the sample mean of $T$ i.i.d. observations, the natural value is $\tau = 1/T$, which puts almost all weight on $\hat \mu$ and so leads back to extreme long-short positions.

That is why practitioners following He and Litterman instead attach a small scalar to the prior, $\mu \sim \mathcal{N}(\mu_{BL}, \tau \Sigma)$, thereby expressing confidence in the market's view.

Exercise {ref}`bl_ex1` compares the two conventions.

Given the realized excess returns one should then update the prior over the mean excess returns according to Bayes' rule.

The corresponding posterior over mean excess returns is normally distributed with mean

$$
(\Sigma^{-1} + (\tau \Sigma)^{-1})^{-1} (\Sigma^{-1}\mu_{BL}   + (\tau \Sigma)^{-1} \hat \mu)
$$

The covariance matrix is

$$
(\Sigma^{-1} + (\tau \Sigma)^{-1})^{-1}
$$

Hence, the Black-Litterman recommendation is consistent with the Bayes update of the prior over the mean excess returns in light of the realized average excess returns on the market.

## Curve decolletage

Consider two independent "competing" views on the excess market returns

$$
\vec r_e  \sim {\mathcal N}( \mu_{BL}, \Sigma)
$$

and

$$
\vec r_e \sim {\mathcal N}( \hat{\mu}, \tau\Sigma)
$$

A special feature of the multivariate normal random variable $Z$ is that, after standardization, its density function depends only on the (Euclidean) length of the standardized realization.

Formally, let the $k$-dimensional random vector be

$$
Z\sim \mathcal{N}(\mu, \Sigma)
$$

then

$$
\bar{Z} \equiv \Sigma^{-1/2}(Z-\mu)\sim \mathcal{N}(\mathbf{0}, I)
$$

and so the points where the density takes the same value can be described by the ellipse (an ellipsoid when $k > 2$)

```{math}
:label: ellipse

\bar z \cdot \bar z =  (z - \mu)'\Sigma^{-1}(z - \mu) = \bar d
```

where $\bar d\in\mathbb{R}_+$ denotes the (transformation) of a particular density value.

The curves defined by equation {eq}`ellipse` can be labeled as iso-likelihood ellipses

> **Remark:** More generally there is a class of density functions
> that possesses this feature, i.e.
>
> $$
  \exists g: \mathbb{R}_+ \mapsto \mathbb{R}_+ \ \ \text{ and } \ \ c \geq 0,
  \ \ \text{s.t.  the density } \ \ f \ \ \text{of} \ \ Z  \ \
  \text{ has the form } \quad f(z) = c g(z\cdot z)
  $$
>
> This property is called **spherical symmetry** (see p. 81 of Leamer
> (1978) {cite}`leamer1978specification`).

In our specific example, we can use the pair $(\bar d_1, \bar d_2)$ as being two "likelihood" values for which the corresponding iso-likelihood ellipses in the excess return space are given by

$$
\begin{aligned}
(\vec r_e - \mu_{BL})'\Sigma^{-1}(\vec r_e - \mu_{BL}) &= \bar d_1 \\
(\vec r_e - \hat \mu)'\left(\tau \Sigma\right)^{-1}(\vec r_e - \hat \mu) &= \bar d_2
\end{aligned}
$$

Notice that for particular $\bar d_1$ and $\bar d_2$ values the two ellipses have a tangency point.

These tangency points, indexed by the pairs $(\bar d_1, \bar d_2)$, characterize points $\vec r_e$ from which there exists no deviation where one can increase the likelihood of one view without decreasing the likelihood of the other view.

The pairs $(\bar d_1, \bar d_2)$ for which there is such a point outline a curve in the excess return space.

This curve is reminiscent of the Pareto curve in an Edgeworth-box setting.

Dickey (1975) {cite}`Dickey1975` calls it a *curve decolletage*.

Leamer (1978) {cite}`leamer1978specification` calls it an *information contract curve* and describes it by the following program: maximize the likelihood of one view, say the Black-Litterman recommendation, while keeping the likelihood of the other view at least at a prespecified level indexed by $\bar d_2$.

Because each quadratic form below is, up to constants, minus twice the log likelihood of the corresponding view, this amounts to minimizing the first quadratic form subject to the second quadratic form being at most $\bar d_2$

$$
\begin{aligned}
 \bar d_1(\bar d_2) &\equiv \min_{\vec r_e} \ \ (\vec r_e - \mu_{BL})'\Sigma^{-1}(\vec r_e - \mu_{BL}) \\
\text{subject to }  \quad  &(\vec r_e - \hat\mu)'(\tau\Sigma)^{-1}(\vec r_e - \hat \mu) \leq \bar d_2
\end{aligned}
$$

Denoting the multiplier on the constraint by $\lambda \geq 0$, the first-order condition is

$$
2(\vec r_e - \mu_{BL} )'\Sigma^{-1} + \lambda 2(\vec r_e - \hat\mu)'(\tau\Sigma)^{-1} = \mathbf{0}
$$

which defines the *information contract curve* between $\mu_{BL}$ and $\hat \mu$

```{math}
:label: info-curve

\vec r_e = (\Sigma^{-1} + \lambda (\tau \Sigma)^{-1})^{-1} (\Sigma^{-1} \mu_{BL} + \lambda (\tau \Sigma)^{-1}\hat \mu )
```

Note that if $\lambda = 1$, {eq}`info-curve` is equivalent to {eq}`mix-views` and it identifies one point on the information contract curve.

Furthermore, because $\lambda$ is a function of the minimum likelihood $\bar d_2$ on the RHS of the constraint, by varying $\bar d_2$ (or $\lambda$ ), we can trace out the whole curve as the figure below illustrates.

```{code-cell} ipython3
np.random.seed(1987102)

N = 2                                           # Number of assets
T = 200                                         # Sample size
τ = 0.8

# Random market portfolio (sum is normalized to 1)
w_m = np.random.rand(N)
w_m = w_m / (w_m.sum())

μ = (np.random.randn(N) + 5) / 100
S = np.random.randn(N, N)
V = S @ S.T
Σ = V * (w_m @ μ)**2 / (w_m @ V @ w_m)

excess_return = stat.multivariate_normal(μ, Σ)
sample = excess_return.rvs(T)

μ_est = sample.mean(0).reshape(N, 1)
Σ_est = np.cov(sample.T)

var_m = w_m @ Σ_est @ w_m
d_m = (w_m @ μ_est) / var_m
μ_m = (d_m * Σ_est @ w_m).reshape(N, 1)

N_r1, N_r2 = 100, 100
r1 = np.linspace(-0.04, .1, N_r1)
r2 = np.linspace(-0.02, .15, N_r2)

λ_grid = np.linspace(.001, 20, 100)
curve = np.asarray([black_litterman(λ, μ_m, μ_est, Σ_est,
                                    τ * Σ_est).flatten() for λ in λ_grid])

λ = 1
```

```{code-cell} ipython3
def decolletage(λ):
    dist_r_BL = stat.multivariate_normal(μ_m.squeeze(), Σ_est)
    dist_r_hat = stat.multivariate_normal(μ_est.squeeze(), τ * Σ_est)

    X, Y = np.meshgrid(r1, r2)
    XY = np.stack((X, Y), axis=-1)
    Z_BL = dist_r_BL.pdf(XY)
    Z_hat = dist_r_hat.pdf(XY)

    μ_tilde = black_litterman(λ, μ_m, μ_est, Σ_est, τ * Σ_est).flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(X, Y, Z_hat, cmap='viridis', alpha =.4)
    ax.contourf(X, Y, Z_BL, cmap='viridis', alpha =.4)
    ax.contour(X, Y, Z_BL, [dist_r_BL.pdf(μ_tilde)], cmap='viridis', alpha=.9)
    ax.contour(X, Y, Z_hat, [dist_r_hat.pdf(μ_tilde)], cmap='viridis', alpha=.9)
    ax.scatter(μ_est[0], μ_est[1])
    ax.scatter(μ_m[0], μ_m[1])
    ax.scatter(μ_tilde[0], μ_tilde[1], c='k', s=20*3)

    ax.plot(curve[:, 0], curve[:, 1], c='k')
    ax.axhline(0, c='k', alpha=.8)
    ax.axvline(0, c='k', alpha=.8)
    ax.set_xlabel(r'Excess return on the first asset, $r_{e, 1}$')
    ax.set_ylabel(r'Excess return on the second asset, $r_{e, 2}$')
    ax.text(μ_est[0] + 0.003, μ_est[1], r'$\hat{\mu}$')
    ax.text(μ_m[0] + 0.003, μ_m[1] + 0.005, r'$\mu_{BL}$')
    plt.show()

decolletage(λ)
```

Note that the line that connects the two points $\hat \mu$ and $\mu_{BL}$ is linear, which comes from the fact that the covariance matrices of the two competing distributions (views) are proportional to each other.

To illustrate the fact that this is not necessarily the case, consider another example using the same parameter values, except that the "second view" constituting the constraint has covariance matrix $\tau I$ instead of $\tau \Sigma$.

This leads to the following figure, on which the curve connecting $\hat \mu$ and $\mu_{BL}$ bends

```{code-cell} ipython3
λ_grid = np.linspace(.001, 20000, 1000)
curve = np.asarray([black_litterman(λ, μ_m, μ_est, Σ_est,
                                    τ * np.eye(N)).flatten() for λ in λ_grid])
λ = 200
```

```{code-cell} ipython3
def decolletage(λ):
    dist_r_BL = stat.multivariate_normal(μ_m.squeeze(), Σ_est)
    dist_r_hat = stat.multivariate_normal(μ_est.squeeze(), τ * np.eye(N))

    X, Y = np.meshgrid(r1, r2)
    XY = np.stack((X, Y), axis=-1)
    Z_BL = dist_r_BL.pdf(XY)
    Z_hat = dist_r_hat.pdf(XY)

    μ_tilde = black_litterman(λ, μ_m, μ_est, Σ_est, τ * np.eye(N)).flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contourf(X, Y, Z_hat, cmap='viridis', alpha=.4)
    ax.contourf(X, Y, Z_BL, cmap='viridis', alpha=.4)
    ax.contour(X, Y, Z_BL, [dist_r_BL.pdf(μ_tilde)], cmap='viridis', alpha=.9)
    ax.contour(X, Y, Z_hat, [dist_r_hat.pdf(μ_tilde)], cmap='viridis', alpha=.9)
    ax.scatter(μ_est[0], μ_est[1])
    ax.scatter(μ_m[0], μ_m[1])

    ax.scatter(μ_tilde[0], μ_tilde[1], c='k', s=20*3)

    ax.plot(curve[:, 0], curve[:, 1], c='k')
    ax.axhline(0, c='k', alpha=.8)
    ax.axvline(0, c='k', alpha=.8)
    ax.set_xlabel(r'Excess return on the first asset, $r_{e, 1}$')
    ax.set_ylabel(r'Excess return on the second asset, $r_{e, 2}$')
    ax.text(μ_est[0] + 0.003, μ_est[1], r'$\hat{\mu}$')
    ax.text(μ_m[0] + 0.003, μ_m[1] + 0.005, r'$\mu_{BL}$')
    plt.show()

decolletage(λ)
```

## Black-Litterman recommendation as regularization

First, consider the OLS regression

$$
\min_{\beta} \Vert X\beta - y \Vert^2
$$

which yields the solution

$$
\hat{\beta}_{OLS} = (X'X)^{-1}X'y
$$

A common performance measure of estimators is the *mean squared error (MSE)*.

An estimator is "good" if its MSE is relatively small.

Suppose that $\beta_0$ is the "true" value of the coefficient, then the MSE of the OLS estimator is

$$
\text{mse}(\hat \beta_{OLS}, \beta_0) := \mathbb E \Vert \hat \beta_{OLS} - \beta_0\Vert^2 =
\underbrace{\mathbb E \Vert \hat \beta_{OLS} - \mathbb E
\hat \beta_{OLS}\Vert^2}_{\text{variance}} +
\underbrace{\Vert \mathbb E \hat\beta_{OLS} - \beta_0\Vert^2}_{\text{squared bias}}
$$

From this decomposition, one can see that in order for the MSE to be small, both the bias and the variance terms must be small.

For example, consider the case when $X$ is a $T$-vector of ones (where $T$ is the sample size), so $\hat\beta_{OLS}$ is simply the sample average, while $\beta_0\in \mathbb{R}$ is defined by the true mean of $y$.

In this example the MSE is

$$
\text{mse}(\hat \beta_{OLS}, \beta_0) = \underbrace{\frac{1}{T^2}
\mathbb E \left(\sum_{t=1}^{T} (y_{t}- \beta_0)\right)^2 }_{\text{variance}} +
\underbrace{0}_{\text{squared bias}}
$$

However, because there is a trade-off between the estimator's bias and variance, there are cases when by permitting a small bias we can substantially reduce the variance so overall the MSE gets smaller.

A typical scenario when this proves to be useful is when the number of coefficients to be estimated is large relative to the sample size.

In these cases, one approach to handle the bias-variance trade-off is the so called *Tikhonov regularization*.

A general form with regularization matrix $\Gamma$ can be written as

$$
\min_{\beta} \Big\{ \Vert X\beta - y \Vert^2 + \Vert \Gamma (\beta - \tilde \beta) \Vert^2 \Big\}
$$

which yields the solution

$$
\hat{\beta}_{Reg} = (X'X + \Gamma'\Gamma)^{-1}(X'y + \Gamma'\Gamma\tilde \beta)
$$

Substituting the value of $\hat{\beta}_{OLS}$ yields

$$
\hat{\beta}_{Reg} = (X'X + \Gamma'\Gamma)^{-1}(X'X\hat{\beta}_{OLS} + \Gamma'\Gamma\tilde \beta)
$$

Often, the regularization matrix takes the form $\Gamma = \sqrt{\lambda} I$ with $\lambda>0$ and $\tilde \beta = \mathbf{0}$.

Then the Tikhonov regularization is equivalent to what is called *ridge regression* in statistics.

To illustrate how this estimator addresses the bias-variance trade-off, we compute the MSE of the ridge estimator

$$
\text{mse}(\hat \beta_{\text{ridge}}, \beta_0) = \underbrace{\frac{1}{(T+\lambda)^2}
\mathbb E \left(\sum_{t=1}^{T} (y_{t}- \beta_0)\right)^2 }_{\text{variance}} +
\underbrace{\left(\frac{\lambda}{T+\lambda}\right)^2 \beta_0^2}_{\text{squared bias}}
$$

The ridge regression shrinks the coefficients of the estimated vector towards zero relative to the OLS estimates thus reducing the variance term at the cost of introducing a "small" bias.

However, there is nothing special about the zero vector.

When $\tilde \beta \neq \mathbf{0}$ shrinkage occurs in the direction of $\tilde \beta$.

Now, we can give a regularization interpretation of the Black-Litterman portfolio recommendation.

To this end, first simplify the equation {eq}`mix-views` that characterizes the Black-Litterman recommendation

$$
\begin{aligned}
\tilde \mu &= (\Sigma^{-1} + (\tau \Sigma)^{-1})^{-1} (\Sigma^{-1}\mu_{BL}  + (\tau \Sigma)^{-1}\hat \mu) \\
&= (1 + \tau^{-1})^{-1}\Sigma \Sigma^{-1} (\mu_{BL}  + \tau ^{-1}\hat \mu) \\
&= (1 + \tau^{-1})^{-1} ( \mu_{BL}  + \tau ^{-1}\hat \mu)
\end{aligned}
$$

In our case, $\hat \mu$ is the vector of estimated mean excess returns of securities.

It can be computed from a stacked linear regression of returns on constants (a system of seemingly unrelated regressions with the same regressors in every equation) where

- $y$ is the stacked vector of observed excess returns of size
  $(N T\times 1)$ -- $N$ securities and $T$
  observations.
- $X = I_{N} \otimes \iota_T$ where $I_N$
  is the identity matrix and $\iota_T$ is a column vector of
  ones.

Correspondingly, the OLS regression of $y$ on $X$ yields $\hat \beta_{OLS} = (X'X)^{-1} X'y = \hat \mu$, the vector of mean excess returns.

With $\Gamma = \sqrt{\tau}(I_{N} \otimes \iota_T)$, so that $\Gamma'\Gamma = \tau X'X = \tau T I_N$, we can write the regularized version of the mean excess return estimation

$$
\begin{aligned}
\hat{\beta}_{Reg} &= (X'X + \Gamma'\Gamma)^{-1}(X'X\hat{\beta}_{OLS} + \Gamma'\Gamma\tilde \beta) \\
&= \left((1 + \tau) T I_N\right)^{-1} T (\hat \beta_{OLS}  + \tau \tilde \beta) \\
&= (1 + \tau)^{-1} (\hat \beta_{OLS}  + \tau \tilde \beta) \\
&= (1 + \tau^{-1})^{-1} ( \tau^{-1}\hat \beta_{OLS}  +  \tilde \beta)
\end{aligned}
$$

Given that $\hat \beta_{OLS} = \hat \mu$ and $\tilde \beta = \mu_{BL}$ in the Black-Litterman model, we have the following interpretation of the model's recommendation.

The estimated (personal) view of the mean excess returns, $\hat{\mu}$ that would lead to extreme short-long positions are "shrunk" towards the conservative market view, $\mu_{BL}$, that leads to the more conservative market portfolio.

So the Black-Litterman procedure results in a recommendation that is a compromise between the conservative market portfolio and the more extreme portfolio that is implied by estimated "personal" views.

## A robust control operator

The Black-Litterman approach is partly inspired by the econometric insight that it is easier to estimate covariances of excess returns than the means.

That is what gave Black and Litterman license to adjust investors' perception of mean excess returns while not tampering with the covariance matrix of excess returns.

The robust control theory is another approach that also hinges on adjusting mean excess returns but not covariances.

Associated with a robust control problem is what Hansen and Sargent {cite}`HansenSargent2001`, {cite}`HansenSargent2008` call a ${\sf T}$ operator.

Let's define the ${\sf T}$ operator as it applies to the problem at hand.

Let $x$ be an $n \times 1$ Gaussian random vector with mean vector $\mu$ and covariance matrix $\Sigma = C C'$.

This means that $x$ can be represented as

$$
x = \mu + C \epsilon
$$

where $\epsilon \sim {\mathcal N}(0,I)$.

Let $\phi(\epsilon)$ denote the associated standardized Gaussian density.

Let $m(\epsilon,\mu)$ be a **likelihood ratio**, meaning that it satisfies

- $m(\epsilon, \mu) > 0$
- $\int m(\epsilon,\mu) \phi(\epsilon) d \epsilon =1$

That is, $m(\epsilon, \mu)$ is a non-negative random variable with mean 1.

Multiplying $\phi(\epsilon)$ by the likelihood ratio $m(\epsilon, \mu)$ produces a distorted distribution for $\epsilon$, namely

$$
\tilde \phi(\epsilon) = m(\epsilon,\mu) \phi(\epsilon)
$$

The next concept that we need is the **relative entropy** of the distorted distribution $\tilde \phi$ with respect to $\phi$.

**Relative entropy** is defined as

$$
{\rm ent} = \int \log m(\epsilon,\mu) m(\epsilon,\mu) \phi(\epsilon) d \epsilon
$$

or

$$
{\rm ent} = \int \log m(\epsilon,\mu) \tilde \phi(\epsilon) d \epsilon
$$

That is, relative entropy is the expected value of the log likelihood ratio $\log m$, where the expectation is taken with respect to the twisted density $\tilde \phi$.

Relative entropy is non-negative.

It is a measure of the discrepancy between two probability distributions.

As such, it plays an important role in governing the behavior of statistical tests designed to discriminate one probability distribution from another.

We are ready to define the ${\sf T}$ operator.

Let $V(x)$ be a value function.

Define

$$
\begin{aligned} {\sf T}\left(V(x)\right) & = \min_{m(\epsilon,\mu)} \int m(\epsilon,\mu)[V(\mu + C \epsilon) + \theta \log m(\epsilon,\mu) ] \phi(\epsilon) d \epsilon \cr
                        & = - \theta \log \int \exp \left( \frac{- V(\mu + C \epsilon)}{\theta} \right) \phi(\epsilon) d \epsilon \end{aligned}
$$

This asserts that ${\sf T}$ is an indirect utility function for a minimization problem in which an **adversary** chooses a distorted probability distribution $\tilde \phi$ to lower expected utility, subject to a penalty term that gets bigger the larger is relative entropy.

Here the penalty parameter

$$
\theta \in [\underline \theta, +\infty]
$$

is a robustness parameter.

When $\theta = +\infty$, there is no scope for the minimizing agent to distort the distribution, so no robustness to alternative distributions is acquired.

As $\theta$ is lowered, more robustness is achieved.

```{note}
The ${\sf T}$ operator is sometimes called a
*risk-sensitivity* operator.
```

We shall apply ${\sf T}$ to the special case of a linear value function $w'(\vec r - r_f {\bf 1})$ where $\vec r - r_f {\bf 1} \sim {\mathcal N}(\mu,\Sigma)$ or $\vec r - r_f {\bf 1} = \mu + C \epsilon$ and $\epsilon \sim {\mathcal N}(0,I)$.

The associated worst-case distribution of $\epsilon$ is Gaussian with mean $v =-\theta^{-1} C' w$ and covariance matrix $I$.

(When the value function is affine, the worst-case distribution distorts
the mean vector of $\epsilon$ but not the covariance matrix
of $\epsilon$.)

For utility function argument $w'(\vec r - r_f {\bf 1})$

$$
{\sf T} \left( w'(\vec r - r_f {\bf 1}) \right) = w' \mu - \frac{1}{2 \theta} w' \Sigma w
$$

and relative entropy is

$$
\frac{v'v}{2} = \frac{1}{2\theta^2}  w' C C' w
$$

## A robust mean-variance portfolio model

According to criterion {eq}`choice-problem`, the mean-variance portfolio choice problem chooses $w$ to maximize

$$
E [w' ( \vec r - r_f {\bf 1})] - \frac{\delta}{2} {\rm var} [ w' ( \vec r - r_f {\bf 1}) ]
$$

which equals

$$
w'\mu - \frac{\delta}{2} w' \Sigma w
$$

A robust decision maker can be modeled as replacing the mean return $E [w' ( \vec r - r_f {\bf 1})]$ with the risk-sensitive criterion

$$
{\sf T} [w' ( \vec r - r_f {\bf 1})] = w' \mu - \frac{1}{2 \theta} w' \Sigma w
$$

that comes from replacing the mean $\mu$ of $\vec r - r_f {\bf 1}$ with the worst-case mean

$$
\mu - \theta^{-1} \Sigma w
$$

and adding back the entropy penalty $\theta \frac{v'v}{2} = \frac{1}{2\theta} w' \Sigma w$, so that

$$
{\sf T} [w' ( \vec r - r_f {\bf 1})] = w' (\mu - \theta^{-1} \Sigma w ) + \frac{1}{2\theta} w' \Sigma w
$$

Notice how the worst-case mean vector depends on the portfolio $w$.

The operator ${\sf T}$ is the indirect utility function that emerges from solving a problem in which an agent who chooses probabilities does so in order to minimize the expected utility of a maximizing agent (in our case, the maximizing agent chooses portfolio weights $w$).

The robust version of the mean-variance portfolio choice problem is then to choose a portfolio $w$ that maximizes

$$
{\sf T} [w' ( \vec r - r_f {\bf 1})] - \frac{\delta}{2} w' \Sigma w
$$

or

```{math}
:label: robust-mean-variance

w' \mu - \frac{\gamma}{2} w' \Sigma w - \frac{\delta}{2} w' \Sigma w
```

The maximizer of {eq}`robust-mean-variance` is

$$
w_{\rm rob} = \frac{1}{\delta + \gamma } \Sigma^{-1} \mu
$$

where $\gamma \equiv \theta^{-1}$ is sometimes called the risk-sensitivity parameter.

An increase in the risk-sensitivity parameter $\gamma$ shrinks the portfolio weights toward zero in the same way that an increase in risk aversion does.

Indeed, $w_{\rm rob} = \frac{\delta}{\delta + \gamma} w$, a scalar multiple of the mean-variance portfolio $w = (\delta \Sigma)^{-1} \mu$ in {eq}`risky-portfolio`.

Robustness therefore shrinks the sizes of long and short positions but leaves their signs unchanged: the same assets are shorted.

In contrast, shrinking $\hat \mu$ toward $\mu_{BL}$, as Black and Litterman do, can reverse the sign of a position.

## Appendix

We want to illustrate the "folk theorem" that with high or moderate frequency data, it is more difficult to estimate means than variances.

In order to operationalize this statement, we take two analog estimators:

- sample average: $\bar X_N = \frac{1}{N}\sum_{i=1}^{N} X_i$
- sample variance:
  $S_N = \frac{1}{N-1}\sum_{i=1}^{N} (X_i - \bar X_N)^2$

to estimate the unconditional mean and unconditional variance of the random variable $X$, respectively.

To measure the "difficulty of estimation", we use *mean squared error* (MSE), that is the average squared difference between the estimator and the true value.

Assuming that the process $\{X_i\}$ is ergodic, both analog estimators are known to converge to their true values as the sample size $N$ goes to infinity.

More precisely for all $\varepsilon > 0$

$$
\lim_{N\to \infty} \ \ P\left\{ \left |\bar X_N - \mathbb E X \right| > \varepsilon \right\} = 0 \quad \quad
$$

and

$$
\lim_{N\to \infty} \ \ P \left\{ \left| S_N - \mathbb V X \right| > \varepsilon \right\} = 0
$$

A sufficient condition for these convergence results is that the associated MSEs vanish as $N$ goes to infinity, or in other words,

$$
\text{MSE}(\bar X_N, \mathbb E X) = o(1) \quad \quad  \text{and} \quad \quad \text{MSE}(S_N, \mathbb V X) = o(1)
$$

Even if the MSEs converge to zero, the associated rates might be different.

Looking at the limit of the *relative MSE* (as the sample size grows to infinity)

$$
\frac{\text{MSE}(S_N, \mathbb V X)}{\text{MSE}(\bar X_N, \mathbb E X)} = \frac{o(1)}{o(1)} \underset{N \to \infty}{\to} B
$$

can inform us about the relative (asymptotic) rates.

We will show that in general, with dependent data, the limit $B$ depends on the sampling frequency.

In particular, we find that the rate of convergence of the variance estimator is less sensitive to increased sampling frequency than the rate of convergence of the mean estimator.

Hence, we can expect the relative asymptotic rate, $B$, to get smaller with higher frequency data, illustrating that "it is more difficult to estimate means than variances".

That is, we need significantly more data to obtain a given precision of the mean estimate than for our variance estimate.

## Special case -- IID sample

We start our analysis with the benchmark case of IID data.

Consider a sample of size $N$ generated by the following IID process,

$$
X_i \sim \mathcal{N}(\mu, \sigma^2)
$$

Taking $\bar X_N$ to estimate the mean, the MSE is

$$
\text{MSE}(\bar X_N, \mu) = \frac{\sigma^2}{N}
$$

Taking $S_N$ to estimate the variance, the MSE is

$$
\text{MSE}(S_N, \sigma^2) = \frac{2\sigma^4}{N-1}
$$

Both estimators are unbiased and hence the MSEs reflect the corresponding variances of the estimators.

Furthermore, both MSEs are $o(1)$ with a (multiplicative) factor of difference in their rates of convergence:

$$
\frac{\text{MSE}(S_N, \sigma^2)}{\text{MSE}(\bar X_N, \mu)} = \frac{2\sigma^2 N}{N-1} \quad \underset{N \to \infty}{\to} \quad 2\sigma^2
$$

We are interested in how this (asymptotic) relative rate of convergence changes as increasing sampling frequency puts dependence into the data.

## Dependence and sampling frequency

To investigate how sampling frequency affects relative rates of convergence, we assume that the data are generated by a mean-reverting continuous time process of the form

$$
dX_t = -\kappa (X_t -\mu)dt + \sigma dW_t\quad\quad
$$

where $\mu$ is the unconditional mean, $\kappa > 0$ is a persistence parameter, and $\{W_t\}$ is a standardized Brownian motion.

Observations arising from this system in particular discrete periods $\mathcal T(h) \equiv \{nh : n \in \mathbb Z \}$ with $h>0$ can be described by the following process

$$
X_{t+1} = (1 - \exp(-\kappa h))\mu + \exp(-\kappa h)X_t + \epsilon_{t, h}
$$

where

$$
\epsilon_{t, h} \sim \mathcal{N}(0, \Sigma_h) \quad \text{with}\quad \Sigma_h = \frac{\sigma^2(1-\exp(-2\kappa h))}{2\kappa}
$$

We call $h$ the *frequency* parameter, whereas $n$ represents the number of *lags* between observations.

Strictly speaking, $h$ is the sampling interval, so a higher sampling frequency corresponds to a smaller $h$.

Hence, the effective distance between two observations $X_t$ and $X_{t+n}$ in the discrete time notation is equal to $h\cdot n$ in terms of the underlying continuous time process.

Straightforward calculations show that the autocorrelation function for the stochastic process $\{X_{t}\}_{t\in \mathcal T(h)}$ is

$$
\Gamma_h(n) \equiv \text{corr}(X_{t + h n}, X_t) = \exp(-\kappa h n)
$$

and the auto-covariance function is

$$
\gamma_h(n) \equiv \text{cov}(X_{t + h n}, X_t) = \frac{\exp(-\kappa h n)\sigma^2}{2\kappa}
$$

It follows that if $n=0$, the unconditional variance is given by $\gamma_h(0) = \frac{\sigma^2}{2\kappa}$ irrespective of the sampling frequency.

The following figure illustrates how the dependence between the observations is related to the sampling frequency

- For any given $h$, the autocorrelation converges to zero as we increase the distance $n$ between the observations. This represents the "weak dependence" of the $X$ process.

- Moreover, for a fixed lag length, $n$, the dependence vanishes as the sampling frequency goes to zero. In fact, letting $h$ go to $\infty$ gives back the case of IID data.

```{code-cell} ipython3
μ = .0
κ = .1
σ = .5
var_uncond = σ**2 / (2 * κ)

n_grid = np.linspace(0, 40, 100)
autocorr_h1 = np.exp(-κ * n_grid * 1)
autocorr_h2 = np.exp(-κ * n_grid * 2)
autocorr_h5 = np.exp(-κ * n_grid * 5)
autocorr_h1000 = np.exp(-κ * n_grid * 1e8)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(n_grid, autocorr_h1, label=r'$h=1$', c='darkblue', lw=2)
ax.plot(n_grid, autocorr_h2, label=r'$h=2$', c='darkred', lw=2)
ax.plot(n_grid, autocorr_h5, label=r'$h=5$', c='orange', lw=2)
ax.plot(n_grid, autocorr_h1000, label=r'"$h=\infty$"', c='darkgreen', lw=2)
ax.legend()
ax.grid()
ax.set(title=r'Autocorrelation functions, $\Gamma_h(n)$',
       xlabel=r'Lags between observations, $n$')
plt.show()
```

## Frequency and the mean estimator

Consider again the AR(1) process generated by discrete sampling with frequency $h$.

Assume that we have a sample of size $N$ and we would like to estimate the unconditional mean -- in our case the true mean is $\mu$.

Again, the sample average is an unbiased estimator of the unconditional mean

$$
\mathbb{E}[\bar X_N] = \frac{1}{N}\sum_{i = 1}^N \mathbb{E}[X_i] = \mathbb{E}[X_0] = \mu
$$

The variance of the sample mean is given by

$$
\begin{aligned}
\mathbb{V}\left(\bar X_N\right) &= \mathbb{V}\left(\frac{1}{N}\sum_{i = 1}^N X_i\right) \\
&= \frac{1}{N^2} \left(\sum_{i = 1}^N \mathbb{V}(X_i) + 2 \sum_{i = 1}^{N-1} \sum_{s = i+1}^N \text{cov}(X_i, X_s) \right) \\
&= \frac{1}{N^2} \left( N \gamma(0) + 2 \sum_{i=1}^{N-1} i \cdot \gamma\left(h\cdot (N - i)\right) \right) \\
&= \frac{1}{N^2} \left( N \frac{\sigma^2}{2\kappa} + 2 \sum_{i=1}^{N-1} i \cdot \exp(-\kappa h (N - i)) \frac{\sigma^2}{2\kappa} \right)
\end{aligned}
$$

It is explicit in the above equation that time dependence in the data inflates the variance of the mean estimator through the covariance terms.

Moreover, as we can see, a higher sampling frequency---smaller $h$---makes all the covariance terms larger, everything else being fixed.

This implies a relatively slower rate of convergence of the sample average for high-frequency data.

Intuitively, stronger dependence across observations for high-frequency data reduces the "information content" of each observation relative to the IID case.

We can upper bound the variance term in the following way, where $\rho \equiv \exp(-\kappa h)$ and $\gamma(0) = \frac{\sigma^2}{2\kappa}$ is the unconditional variance

$$
\begin{aligned}
\mathbb{V}(\bar X_N) &= \frac{\gamma(0)}{N^2} \left( N + 2 \sum_{i=1}^{N-1} i \cdot \rho^{N - i} \right) \\
&\leq \frac{\gamma(0)}{N} \left(1 + 2 \sum_{i=1}^{N-1} \rho^{i} \right) \\
&= \underbrace{\frac{\sigma^2}{2\kappa N}}_{\text{IID case}} \left(1 + 2 \rho \, \frac{1 - \rho^{N-1}}{1 - \rho} \right)
\end{aligned}
$$

Asymptotically, the term $\rho^{N-1}$ vanishes and the dependence in the data inflates the benchmark IID variance by a factor of

$$
\left(1 + \frac{2 \rho}{1 - \rho} \right) = \frac{1 + \exp(-\kappa h)}{1 - \exp(-\kappa h)}
$$

In fact, $N \, \mathbb{V}(\bar X_N)$ converges to $\gamma(0) \frac{1 + \rho}{1 - \rho}$, so the bound is attained in the limit.

This long run factor is larger the higher is the frequency (the smaller is $h$).

Therefore, we expect the asymptotic relative MSEs, $B$, to change with time-dependent data.

We just saw that the mean estimator's rate is roughly changing by a factor of

$$
\left(1 + \frac{2 \rho}{1 - \rho} \right) = \frac{1 + \exp(-\kappa h)}{1 - \exp(-\kappa h)}
$$

The variance estimator's asymptotic MSE can also be computed in closed form.

Because the process is Gaussian, Bartlett's formula implies that $N \, \mathbb{V}(S_N)$ converges to $2 \sum_{k=-\infty}^{\infty} \gamma_h(k)^2 = 2 \gamma(0)^2 \frac{1 + \rho^2}{1 - \rho^2}$, while the squared bias of $S_N$ is of order $N^{-2}$.

Dividing by the limit of $N \, \mathbb{V}(\bar X_N)$ gives

```{math}
:label: B-closed-form

B(h) = 2 \gamma(0) \, \frac{1 + \rho^2}{(1 + \rho)^2}, \qquad \rho = \exp(-\kappa h)
```

As $h \to \infty$, $\rho \to 0$ and $B(h)$ approaches the IID benchmark $2 \gamma(0)$, while as $h \to 0$, $\rho \to 1$ and $B(h)$ falls to $\gamma(0)$.

We can confirm this with (large sample) simulations, which show how the asymptotic relative MSE changes with the sampling frequency $h$ relative to the IID case that we compute in closed form.

```{code-cell} ipython3
@jit
def sample_generator(h, N, M):
    ϕ = (1 - np.exp(-κ * h)) * μ
    ρ = np.exp(-κ * h)
    s = σ**2 * (1 - np.exp(-2 * κ * h)) / (2 * κ)

    mean_uncond = μ
    std_uncond = np.sqrt(σ**2 / (2 * κ))

    ε_path = np.random.normal(0, np.sqrt(s), (M, N))

    y_path = np.zeros((M, N + 1))
    y_path[:, 0] = np.random.normal(mean_uncond, std_uncond, M)

    for i in range(N):
        y_path[:, i + 1] = ϕ + ρ * y_path[:, i] + ε_path[:, i]

    return y_path
```

```{code-cell} ipython3
# sample_generator is compiled by Numba, whose random number generator
# must be seeded from inside a jitted function
@jit
def set_seed(seed):
    np.random.seed(seed)

# Generate large sample for different frequencies
set_seed(1234)
N_app, M_app = 1000, 30000        # Sample size, number of simulations
h_grid = np.linspace(.1, 80, 30)

var_est_store = []
mean_est_store = []
labels = []

for h in h_grid:
    labels.append(h)
    sample = sample_generator(h, N_app, M_app)
    mean_est_store.append(np.mean(sample, 1))
    var_est_store.append(np.var(sample, 1))

var_est_store = np.array(var_est_store)
mean_est_store = np.array(mean_est_store)

# Save mse of estimators
mse_mean = np.var(mean_est_store, 1) + (np.mean(mean_est_store, 1) - μ)**2
mse_var = np.var(var_est_store, 1) \
          + (np.mean(var_est_store, 1) - var_uncond)**2

benchmark_rate = 2 * var_uncond       # IID case

# Relative MSE for large samples
rate_h = mse_var / mse_mean

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(h_grid, rate_h, c='darkblue', lw=2,
        label=r'large sample relative MSE, $B(h)$')
ax.axhline(benchmark_rate, c='k', ls='--', label=r'IID benchmark')
ax.set_title('Relative MSE for large samples as a function of sampling frequency\n'
             'MSE($S_N$) relative to MSE($\\bar X_N$)')
ax.set_xlabel('Sampling frequency, $h$')
ax.legend()
plt.show()
```

The above figure illustrates the relationship between the asymptotic relative MSEs and the sampling frequency

- We can see that with low-frequency data -- large values of $h$
  -- the ratio of asymptotic rates approaches the IID case.
- As $h$ gets smaller -- the higher the frequency -- the relative
  performance of the variance estimator is better in the sense that the
  ratio of asymptotic rates gets smaller. That is, as the time
  dependence gets more pronounced, the rate of convergence of the mean
  estimator's MSE deteriorates more than that of the variance
  estimator.

This experiment holds the number of observations $N$ fixed, so a higher sampling frequency also shortens the calendar span of the sample.

If the span is held fixed instead, sampling more often does not improve the estimate of the drift at all, while it does sharpen the estimate of the variance {cite}`merton1980estimating`; see Exercise {ref}`bl_ex3`.

## Exercises

```{exercise}
:label: bl_ex1

**How much does $\tau$ matter, and what value of $\tau$ would a Bayesian choose?**

Return to the 10-asset example at the start of the lecture (re-create its data with `np.random.seed(12)`, since later cells overwrite the variable names).

Use the market-implied risk aversion $\delta_m$ both to construct $\mu_{BL}$ and to compute portfolios.

1. Show analytically that $\tilde \mu = (\tau \mu_{BL} + \hat \mu)/(1+\tau)$, and hence that $\tilde w = (\tau w_m + w)/(1+\tau)$, where $w = (\delta_m \hat\Sigma)^{-1}\hat\mu$ is the mean-variance portfolio. Confirm this numerically on a grid of $\tau \in [10^{-3}, 10^3]$.
1. Plot $\|\tilde w - w_m\|$ and $\|\tilde w - w\|$ against $\tau$ on a log scale.
1. In the Bayesian interpretation of the lecture, $\hat\mu \mid \mu \sim \mathcal N(\mu, \tau\Sigma)$. If $\hat \mu$ is the sample mean of $T$ i.i.d. draws, what value of $\tau$ does that imply? Compute the implied portfolio and describe it.
1. Now put the scalar on the prior instead, as in the Black-Litterman calibration of He and Litterman: $\mu \sim \mathcal N(\mu_{BL}, \tau_p \Sigma)$ and $\hat\mu \mid \mu \sim \mathcal N(\mu, \Sigma/T)$. Compute the weight on the market portfolio and the extreme portfolio weights for $\tau_p \in \{0.01, 0.05, 0.25\}$.
```

```{solution-start} bl_ex1
:class: dropdown
```

Because both covariance matrices are proportional to $\Sigma$,

$$
\tilde \mu = \left( (1 + \tau^{-1}) \Sigma^{-1} \right)^{-1} \Sigma^{-1} \left( \mu_{BL} + \tau^{-1} \hat \mu \right)
= \frac{\tau \mu_{BL} + \hat \mu}{1 + \tau}
$$

Premultiplying by $(\delta_m \Sigma)^{-1}$ and using $(\delta_m \Sigma)^{-1} \mu_{BL} = w_m$ gives

$$
\tilde w = \frac{\tau w_m + w}{1 + \tau}
$$

So in this version of the model the Black-Litterman portfolio is simply a convex combination of the market portfolio and the mean-variance portfolio, with weight $\tau/(1+\tau)$ on the market.

A **large** $\tau$ expresses **little** confidence in the investor's estimate $\hat \mu$.

```{code-cell} ipython3
# Re-create the 10-asset example of the lecture (later cells overwrite these names)
np.random.seed(12)
N, T = 10, 200
w_m = np.random.rand(N)
w_m = w_m / w_m.sum()
μ = (np.random.randn(N) + 5) / 100
S = np.random.randn(N, N)
V = S @ S.T
Σ = V * (w_m @ μ)**2 / (w_m @ V @ w_m)
δ = 1 / np.sqrt(w_m @ Σ @ w_m)
sample = stat.multivariate_normal(μ, Σ).rvs(T)
μ_est = sample.mean(0).reshape(N, 1)
Σ_est = np.cov(sample.T)
d_m = (w_m @ μ_est) / (w_m @ Σ_est @ w_m)
μ_m = (d_m * Σ_est @ w_m).reshape(N, 1)
w = np.linalg.solve(d_m * Σ_est, μ_est)          # mean-variance weights

τ_grid = np.logspace(-3, 3, 200)
dist_m, dist_mv, gap = [], [], []
for τ in τ_grid:
    μ_tilde = black_litterman(1, μ_m, μ_est, Σ_est, τ * Σ_est)
    w_tilde = np.linalg.solve(d_m * Σ_est, μ_tilde)
    w_check = (τ * w_m.reshape(N, 1) + w) / (1 + τ)
    gap.append(np.max(np.abs(w_tilde - w_check)))
    dist_m.append(np.linalg.norm(w_tilde.flatten() - w_m))
    dist_mv.append(np.linalg.norm(w_tilde - w))

print(f"max |w_tilde - (τ w_m + w)/(1+τ)| over grid: {max(gap):.2e}")
print(f"||w - w_m|| = {np.linalg.norm(w.flatten() - w_m):.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(τ_grid, dist_m, lw=2, label=r'$\|\tilde w - w_m\|$')
ax.semilogx(τ_grid, dist_mv, lw=2, label=r'$\|\tilde w - w\|$')
ax.axvline(1 / T, c='k', ls='--', label=r'$\tau = 1/T$')
ax.set_xlabel(r'$\tau$')
ax.legend()
plt.show()

# Bayesian calibration: sampling covariance of μ̂ is Σ/T, so τ = 1/T
τ_T = 1 / T
w_T = np.linalg.solve(d_m * Σ_est,
                      black_litterman(1, μ_m, μ_est, Σ_est, τ_T * Σ_est))
print(f"τ = 1/T: weight on w_m = {τ_T / (1 + τ_T):.4f}, "
      f"min weight = {w_T.min():.3f}, max weight = {w_T.max():.3f}")

# He-Litterman placement: prior μ ~ N(μ_m, τ_p Σ), data μ̂ ~ N(μ, Σ/T)
for τ_p in [0.01, 0.05, 0.25]:
    μ_HL = black_litterman(1, μ_m, μ_est, τ_p * Σ_est, Σ_est / T)
    w_HL = np.linalg.solve(d_m * Σ_est, μ_HL)
    s = 1 / (1 + τ_p * T)
    print(f"τ_p = {τ_p:5.2f}: weight on w_m = {s:.3f}, "
          f"min weight = {w_HL.min():.3f}, max weight = {w_HL.max():.3f}")
```

The numerical check confirms the convex-combination formula to machine precision (about $10^{-15}$).

The two distance curves cross at $\tau = 1$, where $\tilde w$ lies halfway between $w$ and $w_m$ (which are $4.44$ apart).

The Bayesian calibration in part 3 is sobering.

If $\hat\mu$ is a sample mean of $T = 200$ observations, its sampling covariance is $\Sigma/T$, so $\tau = 1/T = 0.005$.

The posterior then puts weight $0.005$ on the market view, and the recommended portfolio has weights running from about $-1.08$ to $2.97$, essentially the extreme mean-variance portfolio.

Taken literally, the Bayesian interpretation with a prior covariance equal to the covariance of returns lets the data swamp the prior.

That prior is extremely diffuse: it says that uncertainty about the *mean* excess return is as large as the volatility of a single period's return.

Part 4 shows that what matters for the posterior is the ratio of prior precision to data precision, $\tau_p T$.

The weight on the market portfolio is $1/(1+\tau_p T)$: $0.333$, $0.091$ and $0.020$ for $\tau_p = 0.01, 0.05, 0.25$.

The most negative weight is correspondingly $-0.72$, $-0.99$ and $-1.07$.

So even a tight prior around market-implied returns gets overwhelmed by $200$ observations *if* the investor regards the sample mean as a view with sampling covariance $\Sigma/T$.

In practice Black-Litterman users make their views much less precise than $\Sigma/T$; that choice, and not the Bayesian arithmetic, is what keeps the recommended portfolio near the market.

```{solution-end}
```

```{exercise}
:label: bl_ex2

**The robust portfolio, its worst-case mean, and short positions.**

Use the 10-asset data from the start of the lecture and set $\theta = 0.05$, so that $\gamma = \theta^{-1} = 20$.

1. For the mean-variance portfolio $w$, minimize $w'(\hat\mu + C v) + \frac{\theta}{2} v'v$ numerically over the mean distortion $v$, where $CC' = \hat\Sigma$. Check that the minimizer is $v = -\theta^{-1} C' w$ and that the minimized value equals $w'\hat\mu - \frac{1}{2\theta} w'\hat\Sigma w$.
1. Maximize ${\sf T}[w'(\vec r - r_f {\bf 1})] - \frac{\delta}{2} w'\hat\Sigma w$ numerically and confirm that the maximizer is $w_{\rm rob} = (\delta + \gamma)^{-1} \hat\Sigma^{-1} \hat\mu$.
1. Let $\mu_{wc} = \hat\mu - \gamma \hat\Sigma w_{\rm rob}$ be the worst-case mean *at the robust optimum*. Show that a non-robust investor with risk aversion $\delta$ who believes $\mu_{wc}$ chooses $w_{\rm rob}$. In what sense is $\mu_{wc}$ like the Black-Litterman $\mu_{BL}$?
1. Compare the sign patterns of the mean-variance, robust, and Black-Litterman ($\tau = 1$) portfolios. Does robustness cure extreme long-short positions?
```

```{solution-start} bl_ex2
:class: dropdown
```

For part 3, note that

$$
(\delta \Sigma)^{-1} (\mu - \gamma \Sigma w_{\rm rob})
= \frac{\delta + \gamma}{\delta} w_{\rm rob} - \frac{\gamma}{\delta} w_{\rm rob}
= w_{\rm rob}
$$

The worst-case mean is therefore an "implied" mean in the same sense as $\mu_{BL}$: it is the belief about mean excess returns that rationalizes the recommended portfolio for an ordinary mean-variance investor.

The difference is where the belief comes from.

Black and Litterman choose the belief so that it supports the *market* portfolio, while the robust investor's belief comes from a malevolent agent's response to the investor's own portfolio.

```{code-cell} ipython3
from scipy.optimize import minimize

# Re-create the 10-asset example (same seed as the lecture)
np.random.seed(12)
N, T = 10, 200
w_m = np.random.rand(N)
w_m = w_m / w_m.sum()
μ = (np.random.randn(N) + 5) / 100
S = np.random.randn(N, N)
V = S @ S.T
Σ = V * (w_m @ μ)**2 / (w_m @ V @ w_m)
δ = 1 / np.sqrt(w_m @ Σ @ w_m)
sample = stat.multivariate_normal(μ, Σ).rvs(T)
μ_est = sample.mean(0)
Σ_est = np.cov(sample.T)
C = np.linalg.cholesky(Σ_est)

θ = 0.05
γ = 1 / θ

# (1) Worst-case mean shift for a given portfolio
w0 = np.linalg.solve(δ * Σ_est, μ_est)
obj = lambda v: w0 @ (μ_est + C @ v) + θ * v @ v / 2
v_star = minimize(obj, np.zeros(N), method='BFGS', options={'gtol': 1e-12}).x
print("max |v* - (-C'w/θ)|:", np.max(np.abs(v_star + C.T @ w0 / θ)))
print("T numeric:", obj(v_star),
      " closed form:", w0 @ μ_est - w0 @ Σ_est @ w0 / (2 * θ))

# (2) Robust portfolio and the worst-case mean as an "implied" mean
w_mv = np.linalg.solve(δ * Σ_est, μ_est)
w_rob = np.linalg.solve((δ + γ) * Σ_est, μ_est)
neg = lambda w: -(w @ μ_est - (δ + γ) / 2 * w @ Σ_est @ w)
w_num = minimize(neg, np.zeros(N), method='BFGS', options={'gtol': 1e-12}).x
print("max |w_num - w_rob|:", np.max(np.abs(w_num - w_rob)))

μ_wc = μ_est - γ * Σ_est @ w_rob
w_implied = np.linalg.solve(δ * Σ_est, μ_wc)
print("max |w(μ_wc) - w_rob|:", np.max(np.abs(w_implied - w_rob)))

# (3) Compare with mean-variance and Black-Litterman portfolios
print("w_rob / w_mv (elementwise):", np.round(w_rob / w_mv, 4))
print("δ/(δ+γ) =", round(δ / (δ + γ), 4))
d_m = (w_m @ μ_est) / (w_m @ Σ_est @ w_m)
μ_m = d_m * Σ_est @ w_m
w_bl = np.linalg.solve(δ * Σ_est,
                       black_litterman(1, μ_m, μ_est, Σ_est, 1.0 * Σ_est))
for name, x in [('mean-variance', w_mv), ('robust', w_rob),
                ('Black-Litterman', w_bl)]:
    print(f"{name:16s} short positions: {np.sum(x < 0)}, "
          f"sum |w| = {np.abs(x).sum():.3f}, sum w = {x.sum():.3f}")
print("sign pattern mv :", np.sign(w_mv).astype(int))
print("sign pattern rob:", np.sign(w_rob).astype(int))
print("sign pattern BL :", np.sign(w_bl).astype(int))

# Black-Litterman short positions as confidence in μ̂ falls (τ rises)
for τ in [1, 5, 10, 50]:
    w_bl_τ = np.linalg.solve(δ * Σ_est,
                             black_litterman(1, μ_m, μ_est, Σ_est, τ * Σ_est))
    print(f"τ = {τ:3d}: Black-Litterman short positions = {np.sum(w_bl_τ < 0)}")
```

The numerical minimization reproduces the worst-case distortion $v = -\theta^{-1}C'w$ (to about $4\times 10^{-8}$) and the closed form for ${\sf T}$ (to machine precision).

The numerically maximized robust portfolio matches $(\delta+\gamma)^{-1}\hat\Sigma^{-1}\hat\mu$.

A mean-variance investor who holds the worst-case mean $\mu_{wc}$ chooses exactly $w_{\rm rob}$ (error about $10^{-14}$).

Every element of $w_{\rm rob}/w$ equals $0.4803 = \delta/(\delta+\gamma)$.

Robustness of this kind is observationally equivalent to raising risk aversion from $\delta$ to $\delta + \gamma$.

It scales down gross exposure (here $\sum_i |w_i|$ falls from $12.08$ to $5.80$) but leaves the pattern of long and short positions untouched: both portfolios short the same 5 assets.

With $\tau = 1$ the Black-Litterman portfolio also still shorts those 5 assets in this example, but it lies on the way to $w_m$, which has no short positions.

Raising $\tau$ removes short positions one by one (4 remain at $\tau = 5$, 2 at $\tau = 10$, none at $\tau = 50$), something no value of $\theta$ can do in the robust model.

The Black-Litterman adjustment changes the *direction* of the portfolio, while this robust adjustment changes only its *scale*.

```{solution-end}
```

```{exercise}
:label: bl_ex3

**Means versus variances: a closed form for $B(h)$ and a fixed-span experiment.**

1. For the discretely sampled Ornstein-Uhlenbeck process in the appendix, let $\rho = \exp(-\kappa h)$ and $\gamma(0) = \sigma^2/(2\kappa)$. Starting from $N\,\mathbb V(\bar X_N) \to \gamma(0)\frac{1+\rho}{1-\rho}$ and Bartlett's formula $N\,\mathbb V(S_N) \to 2 \sum_{k=-\infty}^{\infty} \gamma_h(k)^2 = 2\gamma(0)^2 \frac{1+\rho^2}{1-\rho^2}$, verify the closed form {eq}`B-closed-form` for the asymptotic relative MSE $B(h)$. Plot it against the simulated `rate_h` from the lecture. Where does the simulation depart from the formula, and why?
1. The appendix holds the number of observations $N$ fixed as $h$ changes, so a higher frequency also means a shorter calendar span. Instead, hold the span fixed at $20$ years. Let log prices follow a Brownian motion with drift $m = 0.06$ and volatility $s = 0.2$, so that returns over an interval $h$ are i.i.d. $\mathcal N(m h, s^2 h)$. For annual, monthly, weekly and daily sampling, compute by simulation the relative RMSEs of the annualized drift estimator $\hat m = \sum_i r_i / 20$ and of the annualized variance estimator $\hat\sigma^2 = \sum_i (r_i - \bar r)^2/((n-1)h)$.
```

```{solution-start} bl_ex3
:class: dropdown
```

Dividing the two asymptotic variances (the squared biases are of smaller order) gives {eq}`B-closed-form`

$$
B(h) = 2 \gamma(0) \, \frac{1 + \rho^2}{(1 + \rho)^2}
$$

As $h \to \infty$, $\rho \to 0$ and $B \to 2\gamma(0)$, the IID benchmark.

As $h \to 0$, $\rho \to 1$ and $B \to \gamma(0)$.

The relative MSE therefore falls by at most a factor of two as the sampling frequency rises with $N$ fixed.

```{code-cell} ipython3
# Part 1: closed-form asymptotic relative MSE
ρ_grid = np.exp(-κ * h_grid)
B_asym = 2 * var_uncond * (1 + ρ_grid**2) / (1 + ρ_grid)**2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(h_grid, rate_h, lw=2, label='simulated $B(h)$, $N=1000$')
ax.plot(h_grid, B_asym, 'k--', lw=2, label='asymptotic formula')
ax.axhline(var_uncond, c='r', ls=':', label=r'limit as $h \to 0$')
ax.set_xlabel('sampling interval $h$')
ax.legend()
plt.show()

for k in [0, 1, 5, 29]:
    print(f"h = {h_grid[k]:6.2f}: simulated B = {rate_h[k]:.3f}, "
          f"asymptotic B = {B_asym[k]:.3f}, "
          f"N·MSE(mean)/γ(0) = {N_app * mse_mean[k] / var_uncond:7.2f}")

# Part 2: Merton's fixed-span experiment
np.random.seed(1234)
m, s = 0.06, 0.20          # annual drift and volatility of log price
span = 20                  # years of data
M = 200_000                # replications
for label, h in [('annual', 1), ('monthly', 1/12),
                 ('weekly', 1/52), ('daily', 1/252)]:
    n = int(round(span / h))
    # returns r_i ~ N(m h, s^2 h), i = 1, ..., n; use exact sampling distributions
    sum_r = np.random.normal(m * h * n, s * np.sqrt(h * n), M)
    ss = s**2 * h * np.random.chisquare(n - 1, M)     # sum of squared deviations
    m_hat = sum_r / span
    s2_hat = ss / ((n - 1) * h)
    rmse_m = np.sqrt(np.mean((m_hat - m)**2))
    rmse_s2 = np.sqrt(np.mean((s2_hat - s**2)**2))
    print(f"{label:8s} n = {n:5d}: RMSE(m̂)/m = {rmse_m / m:.3f}, "
          f"RMSE(σ̂²)/σ² = {rmse_s2 / s**2:.4f}")
```

The closed form tracks the simulation closely except at the very highest frequencies.

Take $h = 0.1$, where $1/(1-\rho) \approx 100$: a sample of $N = 1000$ spans only about ten "correlation times", so the asymptotic formula ($1.25$) overstates the finite-sample ratio (about $1.07$ in the simulation).

The printed values of $N \cdot \text{MSE}(\bar X_N)/\gamma(0)$ show what the ratio hides: going from $h = 80$ to $h = 0.1$ inflates the MSE of the mean by a factor of roughly $180$.

The MSE of the variance estimator also deteriorates, by a factor of roughly $80$, so their ratio falls only from about $2.5$ to about $1.1$.

Part 2 isolates the economically relevant comparison.

With the calendar span fixed at $20$ years, the relative RMSE of the drift estimator is $0.75$ at every sampling frequency.

That matches $s/(m\sqrt{20}) = 0.745$, because $\hat m = (\log P_{20} - \log P_0)/20$ depends only on the endpoints.

The relative RMSE of the variance estimator falls from $0.33$ (annual) to $0.09$ (monthly), $0.04$ (weekly) and $0.02$ (daily), in line with $\sqrt{2/(n-1)}$.

This is the point made by {cite:t}`merton1980estimating`.

Sampling more finely within a fixed span is useless for learning about mean returns but very informative about variances and covariances.

That asymmetry is what licenses Black and Litterman, and robust decision makers, to distrust estimated means while trusting estimated covariances.

```{solution-end}
```
