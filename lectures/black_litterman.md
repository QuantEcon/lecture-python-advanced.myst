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

The classic theory described there assumes that a decision maker completely trusts the statistical model that governs the joint distribution of returns on a list of available assets.

Both extensions described here put distrust of that statistical model into the mind of the decision maker.

One is a model of {cite:t}`black1992global` that imputes to the decision maker distrust of historically estimated mean returns but complete trust of estimated covariances of returns.

The second model also imputes to the decision maker doubts about the statistical model, but now by saying that, because of that distrust, the decision maker uses a version of robust control theory described in the lecture {doc}`Robustness <robustness>`.

The portfolio choice model of {cite:t}`black1992global` was motivated by the finding that with high frequency or moderately high frequency data, means are more difficult to estimate than variances.

A model of *robust portfolio choice* that we'll describe below also begins from the same starting point.

To begin, we'll take for granted that means are more difficult to estimate than covariances and will focus on how Black and Litterman, on the one hand, and robust control theorists, on the other, would recommend modifying the mean-variance portfolio choice model to take that into account.

In an appendix, we use some results on asymptotic mean squared errors and some simulations to verify that means are more difficult to estimate than variances.

Among the ideas in play in this lecture will be

- mean-variance portfolio theory
- Bayesian approaches to estimating linear regressions
- a risk-sensitivity operator and its connection to robust control theory

Both adjustments that we describe are designed to confront a widely recognized embarrassment to mean-variance portfolio theory, namely, that with estimated inputs it usually implies taking very extreme long-short portfolio positions.

They tame those positions in different ways.

The Black-Litterman adjustment pulls the portfolio toward the market portfolio, and so can reverse which assets are held short.

The simplest robust adjustment scales all positions toward zero without changing their signs, but a robust decision maker who distrusts only estimated means can also change the portfolio's direction, depending on how the distrust is measured.

The two approaches build on a common and widespread hunch -- that because it is much easier statistically to estimate covariances of excess returns than it is to estimate their means, it makes sense to adjust investors' subjective beliefs about mean returns in order to render more plausible decisions.

Let's start with some imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.optimize import minimize
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

where $C C' = \Sigma$ and $\epsilon \sim {\mathcal N}(0, I)$ is an $n \times 1$ random vector.

Let $w$ be an $n \times 1$ vector of portfolio weights on the risky securities, with the remaining fraction $1 - {\bf 1}' w$ of wealth held in the risk-free security.

The weights $w$ need not sum to one, and any of them can be negative.

A portfolio with weights $w$ earns the excess return

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

A standard way of estimating $\mu$ is maximum likelihood or least squares; that amounts to estimating $\mu$ by a sample mean of excess returns and estimating $\Sigma$ by a sample covariance matrix.

## Black-Litterman starting point

When estimates of $\mu$ and $\Sigma$ from historical sample means and covariances have been combined with *plausible* values of the risk-aversion parameter $\delta$ to compute an optimal portfolio from formula {eq}`risky-portfolio`, a typical outcome has been $w$'s with *extreme long and short positions*.

A common reaction to these outcomes is that they are so implausible that a portfolio manager cannot recommend them to a customer.

We illustrate with a simulated market of ten assets whose excess returns share a common market factor.

The market portfolio $w_m$ has a monthly Sharpe ratio of about $0.12$ and a monthly volatility of $4.5$ percent, values typical of a broad stock index.

As Black and Litterman assume, the true mean excess returns are close to the ones that make the market portfolio optimal: they equal those means plus small departures, with a standard deviation of $0.1$ percent a month.

An investor observes ten years of monthly excess returns, $T = 120$ observations, and estimates $\mu$ and $\Sigma$ by their sample counterparts $\hat \mu$ and $\hat \Sigma$.

The next function collects the true parameters and the estimates in a `Market` tuple.

```{code-cell} ipython3
Market = namedtuple('Market', ('w_m', 'μ', 'Σ', 'δ', 'μ_hat', 'Σ_hat', 'T'))

SR_m = 0.12      # monthly Sharpe ratio of the market portfolio

def simulate_market(N=10, T=120, σ_m=0.045, s_α=0.001, seed=0):
    """
    True parameters of a one-factor market of N assets, and estimates from a
    simulated sample of T monthly excess returns.  The market portfolio has
    volatility σ_m.  True mean excess returns equal the means that make the
    market portfolio optimal for risk aversion δ = SR_m / σ_m, plus N(0, s_α²)
    departures.
    """
    rng = np.random.default_rng(seed)
    w_m = rng.random(N)
    w_m = w_m / w_m.sum()                         # market portfolio
    β = rng.uniform(0.5, 1.5, N)                  # loadings on the market factor
    σ_ε = rng.uniform(0.03, 0.08, N)              # idiosyncratic volatilities
    Σ = np.outer(β, β) + np.diag(σ_ε**2) / σ_m**2
    Σ = Σ * σ_m**2 / (w_m @ Σ @ w_m)              # market volatility is σ_m
    δ = SR_m / σ_m                                # market risk aversion
    μ = δ * Σ @ w_m + s_α * rng.standard_normal(N)
    sample = rng.multivariate_normal(μ, Σ, size=T)
    return Market(w_m=w_m, μ=μ, Σ=Σ, δ=δ, μ_hat=sample.mean(axis=0),
                  Σ_hat=np.cov(sample.T), T=T)

mkt = simulate_market()
N = len(mkt.w_m)
```

To form a portfolio, the investor needs a value of $\delta$.

Following Black and Litterman, we use the risk aversion $\delta_m$ that would lead an investor who shares the market's views to hold the market portfolio, which we construct below.

To compute the portfolio $w^*$ that is optimal under the true parameters, we use the market's true risk aversion, which `simulate_market` stores as `δ`.

```{code-cell} ipython3
def market_risk_aversion(mkt):
    "Risk aversion implied by the market Sharpe ratio and the estimated market volatility."
    return SR_m / np.sqrt(mkt.w_m @ mkt.Σ_hat @ mkt.w_m)

δ_m = market_risk_aversion(mkt)
w_hat = np.linalg.solve(δ_m * mkt.Σ_hat, mkt.μ_hat)    # estimated inputs
w_star = np.linalg.solve(mkt.δ * mkt.Σ, mkt.μ)          # true inputs
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Estimated and true optimal portfolios
    name: fig-bl-mv-weights
---
x = np.arange(N) + 1
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, w_hat, 'o', c='k', label=r'$w$ (estimated $\hat\mu, \hat\Sigma$)')
ax.plot(x, w_star, 's', c='C2', label=r'$w^*$ (true $\mu, \Sigma$)')
ax.plot(x, mkt.w_m, 'o', c='r', label=r'$w_m$ (market portfolio)')
ax.vlines(x, 0, w_hat, lw=1)
ax.axhline(0, c='k')
ax.axhline(-1, c='k', ls='--')
ax.axhline(1, c='k', ls='--')
ax.set_xlabel('assets')
ax.set_ylabel('portfolio weight')
ax.xaxis.set_ticks(x)
ax.legend(numpoints=1)
plt.show()

se = np.sqrt(np.diag(mkt.Σ_hat) / mkt.T)          # standard errors of μ_hat
print(f"t-statistics of the estimated means: {np.round(mkt.μ_hat / se, 1)}")
print(f"standard errors: {100 * se.min():.2f} to {100 * se.max():.2f} percent a month")
print(f"sum of |w|: estimated {np.abs(w_hat).sum():.2f}, true {np.abs(w_star).sum():.2f}, "
      f"market {np.abs(mkt.w_m).sum():.2f}")
```

The portfolio $w^*$ that the true parameters recommend stays much closer to the market portfolio.

The portfolio computed from the estimates takes large long and short positions, with absolute weights summing to almost ten times those of the market portfolio.

The estimated means are imprecise, with $t$-statistics mostly between one and three.

Their standard errors are comparable to the means themselves and several times the true departures from the market-implied means, and the portfolio formula {eq}`risky-portfolio`, which inverts $\hat \Sigma$, magnifies those errors.

Black and Litterman responded to this situation in the following way:

- They continue to accept {eq}`risky-portfolio` as a good model for choosing an optimal
  portfolio $w$.
- They want to continue to allow the customer to express a
  risk tolerance by setting $\delta$.
- Leaving $\Sigma$ at its maximum-likelihood value, they push
  $\mu$ away from its maximum-likelihood value in a way designed to make
  portfolio choices that are more plausible in terms of conforming to
  what most people actually do.

In particular, given $\Sigma$ and a plausible value of $\delta$, Black and Litterman reverse engineered a vector $\mu_{BL}$ of mean excess returns that makes the $w$ implied by formula {eq}`risky-portfolio` equal the *actual* market portfolio $w_m$, so that

$$
w_m = (\delta \Sigma)^{-1} \mu_{BL}
$$

## Details

Let's define

$$
w_m' \mu \equiv ( r_m - r_f)
$$

as the (scalar) expected excess return on the market portfolio $w_m$.

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

Following the Black-Litterman philosophy, our first step is to back out a value of $\delta_m$ from

- an estimate of the Sharpe ratio, and
- our maximum likelihood estimate of $\sigma$ drawn from our
  estimates of $w_m$ and $\Sigma$

That is what the function `market_risk_aversion` does.

It takes the Sharpe ratio from long historical experience rather than from our ten-year sample, whose estimate of the market's mean excess return is itself imprecise.

The second key Black-Litterman step is then to use this value of $\delta_m$ together with the maximum likelihood estimate of $\Sigma$ to deduce a $\mu_{BL}$ that verifies portfolio rule {eq}`risky-portfolio` at the market portfolio $w = w_m$

$$
\mu_{BL} = \delta_m \Sigma w_m
$$

The starting point of the Black-Litterman portfolio choice model is thus a pair $(\delta_m, \mu_{BL})$ that tells the customer to hold the market portfolio.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Estimated and market-implied mean excess returns
    name: fig-bl-means
---
μ_BL = δ_m * mkt.Σ_hat @ mkt.w_m       # market-implied mean excess returns

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, mkt.μ_hat, 'o', c='k', label=r'$\hat{\mu}$ (estimate)')
ax.plot(x, μ_BL, 'o', c='r', label=r'$\mu_{BL}$ (market implied)')
ax.plot(x, mkt.μ, 's', c='C2', label=r'$\mu$ (true)')
ax.vlines(x, μ_BL, mkt.μ_hat, lw=1)
ax.axhline(0, c='k', ls='--')
ax.set_xlabel('assets')
ax.set_ylabel('mean excess return (per month)')
ax.xaxis.set_ticks(x)
ax.legend(numpoints=1)
plt.show()
```

In our simulated market, the market-implied means $\mu_{BL}$ are much closer to the true means than the estimates $\hat \mu$ are.

## Adding views

Black and Litterman start with a baseline customer who asserts that he or she shares the *market's views*, which means that he or she believes that excess returns are governed by

```{math}
:label: excess-returns

\vec r - r_f {\bf 1} \sim {\mathcal N}( \mu_{BL}, \Sigma)
```

Black and Litterman would advise that customer to hold the market portfolio of risky securities.

Black and Litterman then imagine a customer who would like to express a view that differs from the market's.

The customer wants appropriately to mix that view with the market's before using {eq}`risky-portfolio` to choose a portfolio.

Suppose that the customer's view is expressed by a hunch that rather than {eq}`excess-returns`, excess returns are governed by

$$
\vec r - r_f {\bf 1} \sim {\mathcal N}( \hat \mu, \tau \Sigma)
$$

where $\tau > 0$ is a scalar parameter that determines how the decision maker wants to mix the view $\hat \mu$ with the market's view $\mu_{BL}$.

Black and Litterman would then use a formula like the following one to mix the views $\hat \mu$ and $\mu_{BL}$

```{math}
:label: mix-views

\tilde \mu = (\Sigma^{-1} + (\tau \Sigma)^{-1})^{-1} (\Sigma^{-1} \mu_{BL}  + (\tau \Sigma)^{-1} \hat \mu)
```

Black and Litterman would then advise the customer to hold the portfolio associated with these views implied by rule {eq}`risky-portfolio`:

$$
\tilde w = (\delta \Sigma)^{-1} \tilde \mu
$$

This portfolio $\tilde w$ will deviate from the market portfolio $w_m$ in amounts that depend on the mixing parameter $\tau$.

If $\hat \mu$ is the maximum likelihood estimator and $\tau$ is small, so that the customer's view gets a heavy weight, then the customer's portfolio will involve big short-long positions.

The next function computes the mixture {eq}`mix-views`, with a weight $\lambda$ on the precision of the second view that we use later in the lecture.

```{code-cell} ipython3
def black_litterman(λ, μ1, μ2, Σ1, Σ2):
    """
    Return the precision-weighted mixture of the means μ1 and μ2,

        (Σ1^{-1} + λ Σ2^{-1})^{-1} (Σ1^{-1} μ1 + λ Σ2^{-1} μ2),

    where λ scales the precision Σ2^{-1} of the second view.
    """
    Σ1_inv = np.linalg.inv(Σ1)
    Σ2_inv = np.linalg.inv(Σ2)
    return np.linalg.solve(Σ1_inv + λ * Σ2_inv,
                           Σ1_inv @ μ1 + λ * Σ2_inv @ μ2)
```

The next figure shows the mixture $\tilde \mu$ and the recommended portfolio $\tilde w$ for three values of $\tau$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Black-Litterman mixtures and portfolios
    name: fig-bl-mixtures
---
τ_values = [0.1, 1, 10]
colors = ['C1', 'C4', 'C0']

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].plot(x, mkt.μ_hat, 'o', c='k', label=r'$\hat{\mu}$ (customer view)')
axes[0].plot(x, μ_BL, 'o', c='r', label=r'$\mu_{BL}$ (market)')
axes[0].vlines(x, μ_BL, mkt.μ_hat, lw=1, color='gray')
axes[1].plot(x, w_hat, 'o', c='k', label=r'$w$ (mean-variance)')
axes[1].plot(x, mkt.w_m, 'o', c='r', label=r'$w_m$ (market)')
axes[1].vlines(x, mkt.w_m, w_hat, lw=1, color='gray')
for τ, c in zip(τ_values, colors):
    μ_tilde = black_litterman(1, μ_BL, mkt.μ_hat, mkt.Σ_hat, τ * mkt.Σ_hat)
    w_tilde = np.linalg.solve(δ_m * mkt.Σ_hat, μ_tilde)
    axes[0].plot(x, μ_tilde, 'x', c=c, ms=8, mew=2, label=rf'$\tilde\mu$, $\tau = {τ}$')
    axes[1].plot(x, w_tilde, 'x', c=c, ms=8, mew=2, label=rf'$\tilde w$, $\tau = {τ}$')
axes[0].set(ylabel='mean excess return (per month)',
            title=r'$\hat{\mu}$, $\mu_{BL}$, and their mixtures $\tilde{\mu}$')
axes[1].set(ylabel='portfolio weight', title='Black-Litterman portfolio recommendations')
axes[1].axhline(0, c='k')
for ax in axes:
    ax.set_xlabel('assets')
    ax.xaxis.set_ticks(x)
    ax.legend(numpoints=1, fontsize=9)
plt.tight_layout()
plt.show()
```

Each mixture $\tilde \mu$ lies between $\hat \mu$ and $\mu_{BL}$, and each portfolio $\tilde w$ lies between the mean-variance portfolio $w$ and the market portfolio $w_m$.

As $\tau$ rises, the customer's view gets less weight and the recommended portfolio moves toward the market portfolio, shedding short positions one at a time; two remain at $\tau = 10$.

## Bayesian interpretation

Consider the following Bayesian interpretation of the Black-Litterman recommendation.

The prior belief over the mean excess returns is consistent with the market portfolio and is given by

$$
\mu \sim \mathcal{N}(\mu_{BL}, \Sigma)
$$

Given a particular realization of the mean excess returns $\mu$, one observes the average excess returns $\hat \mu$ on the assets according to the distribution

$$
\hat \mu \mid \mu, \Sigma \sim \mathcal{N}(\mu, \tau\Sigma)
$$

where $\tau$ scales the uncertainty of the investor's own estimate $\hat \mu$ relative to the prior.

Given the realized excess returns, one should then update the prior over the mean excess returns according to Bayes' rule.

The corresponding posterior over mean excess returns is normally distributed with mean

$$
(\Sigma^{-1} + (\tau \Sigma)^{-1})^{-1} (\Sigma^{-1}\mu_{BL}   + (\tau \Sigma)^{-1} \hat \mu)
$$

The covariance matrix is

$$
(\Sigma^{-1} + (\tau \Sigma)^{-1})^{-1}
$$

Hence, the Black-Litterman recommendation is consistent with the Bayes update of the prior over the mean excess returns in light of the realized average excess returns on the market.

If $\hat \mu$ is the sample mean of $T$ IID observations, the natural value is $\tau = 1/T$, which puts almost all weight on $\hat \mu$ and so leads back to extreme long-short positions.

Practitioners following {cite:t}`HeLitterman1999` instead attach a small scalar $\tau_p$ to the prior, $\mu \sim \mathcal{N}(\mu_{BL}, \tau_p \Sigma)$, and state their views with a covariance matrix that they choose, usually also proportional to $\tau_p$, rather than with the sampling covariance matrix $\Sigma / T$.

Only the ratio of the prior's covariance to the view's covariance matters for the posterior mean, so when the view's covariance is also proportional to $\tau_p$, $\tau_p$ cancels from it.

It is therefore the imprecision that practitioners attach to their views, and not the Bayesian arithmetic, that keeps their portfolios far from the extreme positions that $\tau = 1/T$ produces.

{ref}`bl_ex1` compares the two conventions.

## Views on a few assets

In practice, Black-Litterman users seldom hold a view about the mean excess return of every asset.

Instead they express a few views, each about a single asset or a portfolio of assets, such as "asset $i$ will outperform asset $j$ by $q$ percent a month."

{cite:t}`HeLitterman1999` write $k$ such views as

$$
P \mu = Q + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \Omega)
$$

where each row of the $k \times n$ matrix $P$ defines a portfolio, $Q$ is the $k \times 1$ vector of mean excess returns that the customer expects on those portfolios, and the covariance matrix $\Omega$ expresses how confident the customer is about the views.

Combining the views with the prior $\mu \sim \mathcal{N}(\mu_{BL}, \tau_p \Sigma)$ by Bayes' rule gives the posterior mean

```{math}
:label: bl-views

\tilde \mu = \left[ (\tau_p \Sigma)^{-1} + P' \Omega^{-1} P \right]^{-1}
\left[ (\tau_p \Sigma)^{-1} \mu_{BL} + P' \Omega^{-1} Q \right]
= \mu_{BL} + \tau_p \Sigma P' (\tau_p P \Sigma P' + \Omega)^{-1} (Q - P \mu_{BL})
```

Formula {eq}`mix-views` is the special case in which the customer holds a view about every asset, $P = I$ and $Q = \hat \mu$, and in which $\Omega = \tau \Sigma$ and the prior covariance is $\Sigma$.

Applying rule {eq}`risky-portfolio` to {eq}`bl-views` and using $(\delta \Sigma)^{-1} \mu_{BL} = w_m$ gives

$$
\tilde w = w_m + \frac{\tau_p}{\delta} P' (\tau_p P \Sigma P' + \Omega)^{-1} (Q - P \mu_{BL})
$$

So the recommended portfolio departs from the market portfolio only by holding more or less of the $k$ portfolios that the views are about.

A view about two assets changes the weights of those two assets and no others, even though, through the correlations in $\Sigma$, it changes the mean excess returns of all assets.

{cite:t}`HeLitterman1999` form portfolios with the covariance matrix $\Sigma + M$ of the predictive distribution, where $M$ is the posterior covariance matrix of $\mu$.

That choice also scales the market portfolio by $1/(1 + \tau_p)$, but either way the views change the portfolio only through the portfolios in the rows of $P$.

We illustrate with a single view: asset 2 will outperform asset 4 by $0.3$ percent a month, about $3.6$ percent a year.

Following a common practice, we set $\tau_p = 0.05$ and compare two levels of confidence, $\Omega = \tau_p P \Sigma P'$ and a ten times smaller $\Omega$.

```{code-cell} ipython3
def bl_views(μ_BL, Σ, τ_p, P, Q, Ω):
    "Posterior mean of μ given the prior N(μ_BL, τ_p Σ) and views P μ = Q + N(0, Ω)."
    A = τ_p * Σ @ P.T
    return μ_BL + A @ np.linalg.solve(P @ A + Ω, Q - P @ μ_BL)

P = np.zeros((1, N))
P[0, 1], P[0, 3] = 1, -1                 # asset 2 minus asset 4
Q = np.array([0.003])
τ_p = 0.05
print(f"market-implied spread P μ_BL = {100 * (P @ μ_BL)[0]:.2f} percent per month")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Effects of a relative view
    name: fig-bl-views
---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for k, (scale, label) in enumerate([(1.0, r'$\Omega = \tau_p P \Sigma P^\prime$'),
                                    (0.1, r'$\Omega = 0.1 \, \tau_p P \Sigma P^\prime$')]):
    Ω = scale * τ_p * P @ mkt.Σ_hat @ P.T
    μ_view = bl_views(μ_BL, mkt.Σ_hat, τ_p, P, Q, Ω)
    w_view = np.linalg.solve(δ_m * mkt.Σ_hat, μ_view)
    axes[0].bar(x + (k - 0.5) * 0.35, 100 * (μ_view - μ_BL), width=0.35, label=label)
    axes[1].bar(x + (k - 0.5) * 0.35, w_view - mkt.w_m, width=0.35, label=label)
axes[0].set(ylabel=r'$\tilde\mu - \mu_{BL}$ (percent per month)',
            title='change in mean excess returns')
axes[1].set(ylabel=r'$\tilde w - w_m$', title='change in portfolio weights')
for ax in axes:
    ax.axhline(0, c='k', lw=0.8)
    ax.set_xlabel('assets')
    ax.xaxis.set_ticks(x)
    ax.legend()
plt.tight_layout()
plt.show()
```

The market implies that asset 2 and asset 4 have about the same mean excess return, so the view departs from the market by about $0.3$ percent a month.

With $\Omega = \tau_p P \Sigma P'$, the view and the prior get equal weight, and the posterior mean of the spread lies halfway between the market's spread and the view's.

The view raises the mean excess return of asset 2 and lowers that of asset 4, and because the assets are correlated it moves the other means too.

But the recommended portfolio buys asset 2 and sells asset 4 in equal amounts and leaves every other weight at its market value.

The more confident view, with the smaller $\Omega$, moves the portfolio further.

## Curve decolletage

Write $\vec r_e = \vec r - r_f {\bf 1}$ for the vector of excess returns, and consider two independent "competing" views on it

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

where a larger value of $\bar d\in\mathbb{R}_+$ corresponds to a lower value of the density.

The curves defined by equation {eq}`ellipse` can be labeled as iso-likelihood ellipses.

> *Remark:* More generally there is a class of density functions
> that possesses this feature, i.e.
>
> $$
  \exists g: \mathbb{R}_+ \mapsto \mathbb{R}_+ \ \ \text{ and } \ \ c \geq 0,
  \ \ \text{s.t.  the density } \ \ f \ \ \text{of} \ \ Z  \ \
  \text{ has the form } \quad f(z) = c g(z\cdot z)
  $$
>
> This property is called **spherical symmetry** (see p. 81 of
> {cite:t}`leamer1978specification`).

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

{cite:t}`Dickey1975` calls it a *curve decolletage*.

{cite:t}`leamer1978specification` calls it an *information contract curve* and describes it by the following program: maximize the likelihood of one view, say the Black-Litterman recommendation, while keeping the likelihood of the other view at least at a prespecified level indexed by $\bar d_2$.

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

Furthermore, because $\lambda$ is a function of the minimum likelihood $\bar d_2$ on the RHS of the constraint, by varying $\bar d_2$ (or $\lambda$), we can trace out the whole curve as the figure below illustrates.

We use a two-asset version of our simulated market.

The next function draws iso-likelihood ellipses of the two views, the information contract curve, and the point on it that corresponds to a given $\lambda$, at which two of the ellipses are tangent.

```{code-cell} ipython3
def quad_form(Z, m, V):
    "(z - m)' V^{-1} (z - m) at every point z of the grid Z."
    D = Z - m
    return np.einsum('...i,ij,...j->...', D, np.linalg.inv(V), D)

def decolletage(μ_BL, μ_hat, Σ, Σ_view, λ):
    curve = np.array([black_litterman(l, μ_BL, μ_hat, Σ, Σ_view)
                      for l in np.logspace(-4, 4, 400)])
    μ_tilde = black_litterman(λ, μ_BL, μ_hat, Σ, Σ_view)

    centre, r = (μ_BL + μ_hat) / 2, 1.3 * np.abs(μ_hat - μ_BL).max()
    X, Y = np.meshgrid(np.linspace(centre[0] - r, centre[0] + r, 300),
                       np.linspace(centre[1] - r, centre[1] + r, 300))
    Z = np.stack((X, Y), axis=-1)

    fig, ax = plt.subplots(figsize=(8, 7))
    for m, V, color in [(μ_BL, Σ, 'C0'), (μ_hat, Σ_view, 'C1')]:
        d_star = quad_form(μ_tilde, m, V)          # the ellipse through μ_tilde
        ax.contour(X, Y, quad_form(Z, m, V), levels=d_star * np.array([0.25, 2.25]),
                   colors=color, linewidths=0.8, linestyles='--')
        ax.contour(X, Y, quad_form(Z, m, V), levels=[d_star], colors=color,
                   linewidths=2)
    ax.plot(curve[:, 0], curve[:, 1], c='k', lw=2)
    for point, label, color in [(μ_BL, r'$\mu_{BL}$', 'C0'), (μ_hat, r'$\hat{\mu}$', 'C1')]:
        ax.plot(*point, 'o', c=color, ms=8)
        ax.annotate(label, point, textcoords='offset points', xytext=(8, 8), fontsize=14)
    ax.plot(*μ_tilde, 'o', c='k', ms=8)

    handles = [plt.Line2D([], [], c='C0', lw=2), plt.Line2D([], [], c='C1', lw=2),
               plt.Line2D([], [], c='k', lw=2),
               plt.Line2D([], [], c='k', marker='o', ls='')]
    labels = ['iso-likelihood ellipses of the market view',
              "iso-likelihood ellipses of the customer's view",
              'information contract curve', rf'tangency point, $\lambda = {λ}$']
    ax.legend(handles, labels, loc='lower left', fontsize=9)
    ax.set_xlabel(r'excess return on the first asset, $r_{e, 1}$')
    ax.set_ylabel(r'excess return on the second asset, $r_{e, 2}$')
    ax.set_aspect('equal')
    plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Contract curve with proportional covariances
    name: fig-bl-decolletage
---
mkt2 = simulate_market(N=2, seed=4)
δ_m2 = market_risk_aversion(mkt2)
μ_BL2 = δ_m2 * mkt2.Σ_hat @ mkt2.w_m
τ = 0.8

decolletage(μ_BL2, mkt2.μ_hat, mkt2.Σ_hat, τ * mkt2.Σ_hat, λ=1)
```

The solid ellipses are the iso-likelihood ellipses of the two views that pass through the tangency point, and the dashed ones are other ellipses of the same two families.

Note that the curve that connects the two points $\hat \mu$ and $\mu_{BL}$ is a straight line, which comes from the fact that the covariance matrices of the two competing distributions (views) are proportional to each other.

To illustrate the fact that this is not necessarily the case, consider another example using the same parameter values, except that the "second view" constituting the constraint has covariance matrix $\tau \, \text{diag}(\Sigma)$, which keeps the variances of $\Sigma$ but ignores the correlation between the two assets.

This leads to the following figure, on which the curve connecting $\hat \mu$ and $\mu_{BL}$ bends.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Contract curve with a diagonal covariance
    name: fig-bl-decolletage-diag
---
decolletage(μ_BL2, mkt2.μ_hat, mkt2.Σ_hat, τ * np.diag(np.diag(mkt2.Σ_hat)), λ=1)
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

To illustrate how this estimator addresses the bias-variance trade-off, we compute the MSE of the ridge estimator in our example

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

The estimated (personal) view of the mean excess returns, $\hat{\mu}$, that would lead to extreme short-long positions is "shrunk" towards the conservative market view, $\mu_{BL}$, that leads to the more conservative market portfolio.

Applying rule {eq}`risky-portfolio` with $\delta = \delta_m$ shows that the portfolio is shrunk in the same way,

```{math}
:label: bl-convex

\tilde w = \frac{\tau w_m + w}{1 + \tau}
```

where $w = (\delta_m \Sigma)^{-1} \hat \mu$ is the mean-variance portfolio: the Black-Litterman portfolio is a convex combination of the market portfolio and the mean-variance portfolio, with weight $\tau / (1 + \tau)$ on the market.

So the Black-Litterman procedure results in a recommendation that is a compromise between the conservative market portfolio and the more extreme portfolio that is implied by estimated "personal" views.

## Does shrinkage pay?

The bias-variance trade-off suggests a question that our simulated market lets us answer: how much should the investor shrink?

We draw many samples of $T = 120$ months from the true model, form the Black-Litterman portfolio {eq}`bl-convex` from each sample for a range of values of $\tau$, and evaluate each portfolio with the true parameters.

We measure the performance of a portfolio $w$ by its certainty-equivalent loss,

$$
U(\mu, \Sigma; w^*) - U(\mu, \Sigma; w)
$$

the amount of expected excess return, net of the risk penalty, that the investor gives up by holding $w$ instead of the portfolio $w^*$ that is optimal under the true parameters, where $U$ is the objective {eq}`choice-problem` with the market's true risk aversion.

We report it in percent a year.

```{code-cell} ipython3
def simulate_estimates(mkt, M=1000, seed=1):
    "Estimates μ_hat and Σ_hat from M samples of size T drawn from the true model."
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mkt.μ, mkt.Σ, size=(M, mkt.T))
    μ_hats = samples.mean(axis=1)
    D = samples - μ_hats[:, None, :]
    Σ_hats = np.einsum('mti,mtj->mij', D, D) / (mkt.T - 1)
    return μ_hats, Σ_hats

def average_loss(W, mkt):
    "Average certainty-equivalent loss, in percent a year, of the portfolios in the rows of W."
    U = lambda W: W @ mkt.μ - mkt.δ / 2 * np.einsum('mi,ij,mj->m', W, mkt.Σ, W)
    w_star = np.linalg.solve(mkt.δ * mkt.Σ, mkt.μ)
    return 1200 * (U(w_star[None, :]) - U(W)).mean()

μ_hats, Σ_hats = simulate_estimates(mkt)
δ_ms = SR_m / np.sqrt(np.einsum('i,mij,j->m', mkt.w_m, Σ_hats, mkt.w_m))
W_hat = np.linalg.solve(δ_ms[:, None, None] * Σ_hats,          # mean-variance portfolios
                        μ_hats[:, :, None])[:, :, 0]

τ_grid = np.logspace(-2, 3, 51)
bl_loss = np.array([average_loss((τ * mkt.w_m + W_hat) / (1 + τ), mkt) for τ in τ_grid])
mv_loss = average_loss(W_hat, mkt)
market_loss = average_loss(mkt.w_m[None, :], mkt)
τ_best = τ_grid[np.argmin(bl_loss)]

U_star = 1200 * (mkt.μ @ w_star - mkt.δ / 2 * w_star @ mkt.Σ @ w_star)
print(f"certainty equivalent of the optimal portfolio: {U_star:.2f} percent a year")
print(f"loss: mean-variance {mv_loss:.2f}, market {market_loss:.2f}, "
      f"Black-Litterman with τ = 1 {bl_loss[np.argmin(np.abs(τ_grid - 1))]:.2f}, "
      f"best τ = {τ_best:.0f} {bl_loss.min():.2f}")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Losses from Black-Litterman portfolios
    name: fig-bl-shrinkage-loss
---
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(τ_grid, bl_loss, lw=2, label=r'Black-Litterman portfolio $\tilde w$')
ax.axhline(mv_loss, c='k', ls='--', label=r'mean-variance portfolio $w$ ($\tau = 0$)')
ax.axhline(market_loss, c='r', ls=':', label=r'market portfolio $w_m$ ($\tau = \infty$)')
ax.axvline(τ_best, c='gray', lw=0.8)
ax.set_xlabel(r'$\tau$')
ax.set_ylabel('certainty-equivalent loss (percent a year)')
ax.legend()
plt.show()
```

The mean-variance portfolio computed from the estimates is a disaster: on average it costs the investor about $26$ percent a year, more than six times what the optimal portfolio is worth.

Shrinking toward the market portfolio cuts the loss steeply, and the loss is smallest around $\tau = 50$, which puts about $98$ percent of the weight on the market portfolio, though the curve is nearly flat beyond about $\tau = 25$.

Shrinking all the way to the market portfolio gives up a little, because the true means depart from the market-implied ones.

Here the bias-variance trade-off is lopsided because, with ten years of monthly data, the variance of $\hat \mu$ is large relative to the bias of $\mu_{BL}$.

Estimation error is also severe in data: {cite:t}`DeMiguelGarlappiUppal2009` find that, across seven empirical data sets, none of fourteen models of optimal portfolio choice consistently beats simply dividing wealth equally among the assets.

{ref}`bl_ex1` relates the best $\tau$ to the prior of the Bayesian interpretation.

## A robust control operator

The Black-Litterman approach is partly inspired by the econometric insight that it is easier to estimate covariances of excess returns than the means.

That is what gave Black and Litterman license to adjust investors' perception of mean excess returns while not tampering with the covariance matrix of excess returns.

The robust control theory is another approach that also hinges on adjusting mean excess returns but not covariances.

Associated with a robust control problem is what {cite:t}`HansenSargent2001,HansenSargent2008` call a ${\sf T}$ operator.

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

The next concept that we need is the **relative entropy** of the distorted distribution $\tilde \phi$ with respect to $\phi$, which is defined as

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

This asserts that ${\sf T}$ is an indirect utility function for a minimization problem in which an *adversary* chooses a distorted probability distribution $\tilde \phi$ to lower expected utility, subject to a penalty term that gets bigger the larger is relative entropy.

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

Let $\gamma \equiv \theta^{-1}$, which is sometimes called the risk-sensitivity parameter.

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

An increase in the risk-sensitivity parameter $\gamma$ shrinks the portfolio weights toward zero in the same way that an increase in risk aversion does.

Indeed, $w_{\rm rob} = \frac{\delta}{\delta + \gamma} w$, a scalar multiple of the mean-variance portfolio $w = (\delta \Sigma)^{-1} \mu$ in {eq}`risky-portfolio`.

Robustness therefore shrinks the sizes of long and short positions but leaves their signs unchanged: the same assets are shorted.

In contrast, shrinking $\hat \mu$ toward $\mu_{BL}$, as Black and Litterman do, can reverse the sign of a position.

## Robustness about the mean

For the linear value function above, the adversary behind the ${\sf T}$ operator distorts only the mean of the excess returns, from $\mu$ to $\mu + C v$.

Its entropy penalty $\theta v'v/2$ can be written as $\frac{\theta}{2} (\mu_v - \mu)' \Sigma^{-1} (\mu_v - \mu)$, where $\mu_v = \mu + C v$ is the distorted mean.

So the penalty measures how far the distorted mean is from $\mu$ in the metric $\Sigma^{-1}$.

The starting point of this lecture suggests letting the decision maker choose that metric to reflect how much it distrusts each estimated mean.

The decision maker trusts the estimated covariance matrix, which we continue to write as $\Sigma$, distrusts the estimate $\hat \mu$, and lets an adversary choose a mean $\mu$ to lower its objective, subject to a penalty that grows with the distance of $\mu$ from $\hat \mu$:

$$
\max_w \min_\mu \left\{ w' \mu - \frac{\delta}{2} w' \Sigma w
+ \frac{\theta}{2} (\mu - \hat \mu)' \Psi^{-1} (\mu - \hat \mu) \right\}
$$

With $\Psi = \Sigma$, this is the ${\sf T}$-operator problem of the previous section, with $\hat \mu$ in the role of $\mu$; what is new is the freedom to choose $\Psi$.

A natural choice is the sampling covariance matrix of $\hat \mu$, $\Psi = \Sigma / T$.

Then the penalty is $\theta$ times the relative entropy of a sample of $T$ IID excess returns drawn with mean $\mu$ with respect to one drawn with mean $\hat \mu$, so that the decision maker worries most about means that would be hard to distinguish from $\hat \mu$ with a sample of the size that it has.

The worst-case mean is $\hat \mu - \theta^{-1} \Psi w$, and substituting it gives

$$
\max_w \left\{ w' \hat \mu - \frac{\delta}{2} w' \Sigma w - \frac{\gamma}{2} w' \Psi w \right\},
\qquad \gamma = \theta^{-1}
$$

so the robust portfolio is

```{math}
:label: robust-mean

w_{\rm rob} = (\delta \Sigma + \gamma \Psi)^{-1} \hat \mu
```

If $\Psi$ is proportional to $\Sigma$, as it is when $\Psi = \Sigma / T$, then {eq}`robust-mean` is again a scalar multiple of the mean-variance portfolio, and robustness only scales positions down.

If $\Psi$ is not proportional to $\Sigma$, robustness also changes the direction of the portfolio.

An example is $\Psi = \text{diag}(\Sigma) / T$, which describes a decision maker who doubts each estimated mean by as much as its own sampling variance warrants but ignores the correlations among the estimation errors.

As $\gamma$ grows, the robust portfolio then becomes proportional to $\text{diag}(\Sigma)^{-1} \hat \mu$, which holds every asset whose estimated mean is positive.

Equation {eq}`robust-mean` has a Bayesian reading that connects it to the Black-Litterman formula.

A short calculation shows that $w_{\rm rob} = (\delta \Sigma)^{-1} \bar \mu$, where

$$
\bar \mu = \left[ \frac{\gamma}{\delta} \Sigma^{-1} + \Psi^{-1} \right]^{-1} \Psi^{-1} \hat \mu
$$

is the posterior mean of $\mu$ given the prior $\mu \sim \mathcal{N}(0, (\delta / \gamma) \Sigma)$ and the view $\hat \mu \sim \mathcal{N}(\mu, \Psi)$.

So robustness about the mean shrinks $\hat \mu$ toward zero, along an information contract curve between $0$ and $\hat \mu$, while Black-Litterman shrinks $\hat \mu$ toward $\mu_{BL}$.

As in the decolletage figures, that curve is a straight line when $\Psi$ is proportional to $\Sigma$ and bends when $\Psi = \text{diag}(\Sigma)/T$ and the assets are correlated.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Robust portfolio weights
    name: fig-bl-robust-mean
---
Ψ_options = {r'$\Psi = \hat\Sigma / T$': mkt.Σ_hat / mkt.T,
             r'$\Psi = \mathrm{diag}(\hat\Sigma) / T$': np.diag(np.diag(mkt.Σ_hat)) / mkt.T}
γ_grid = np.logspace(0, 4, 81)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (label, Ψ) in zip(axes, Ψ_options.items()):
    W_rob = np.array([np.linalg.solve(δ_m * mkt.Σ_hat + γ * Ψ, mkt.μ_hat) for γ in γ_grid])
    ax.semilogx(γ_grid, W_rob, lw=2)
    ax.axhline(0, c='k', lw=0.8)
    ax.set_title(label)
    ax.set_xlabel(r'$\gamma$')
    shorts = (W_rob < 0).sum(axis=1)
    print(f"{label}: short positions at γ = 1, 100, 1000, 10000: "
          f"{shorts[[0, 40, 60, 80]]}")
axes[0].set_ylabel('portfolio weights')
plt.tight_layout()
plt.show()

# check the Bayesian reading of the robust portfolio
γ, Ψ = 100, np.diag(np.diag(mkt.Σ_hat)) / mkt.T
μ_bar = np.linalg.solve(γ / δ_m * np.linalg.inv(mkt.Σ_hat) + np.linalg.inv(Ψ),
                        np.linalg.solve(Ψ, mkt.μ_hat))
print(f"max |w_rob - (δΣ)^(-1) μ_bar| = "
      f"{np.abs(np.linalg.solve(δ_m * mkt.Σ_hat + γ * Ψ, mkt.μ_hat) - np.linalg.solve(δ_m * mkt.Σ_hat, μ_bar)).max():.1e}")
```

Each line traces the weight of one asset as $\gamma$ rises.

With $\Psi = \hat \Sigma / T$, all weights shrink in proportion and the four short positions remain short.

With $\Psi = \text{diag}(\hat \Sigma) / T$, the short positions shrink faster than the long ones and turn into long positions one by one, while one small long position grows before it shrinks.

{cite:t}`GarlappiUppalWang2007` study a related model in which an investor who is averse to ambiguity considers every mean in a confidence region around $\hat \mu$.

They compare a confidence region for all the means jointly with regions for each mean separately, a contrast like the one between our two choices of $\Psi$.

The next cell compares the average certainty-equivalent losses of the two robust portfolios with those of Black-Litterman portfolios, using the simulated samples of the section on whether shrinkage pays.

```{code-cell} ipython3
γ_values = np.logspace(0, 4, 41)
for label, make_Ψ in [('Ψ = Σ/T', lambda Σ: Σ / mkt.T),
                      ('Ψ = diag(Σ)/T', lambda Σ: np.diag(np.diag(Σ)) / mkt.T)]:
    losses = []
    for γ in γ_values:
        W_rob = np.array([np.linalg.solve(d * Σ + γ * make_Ψ(Σ), μ)
                          for d, Σ, μ in zip(δ_ms, Σ_hats, μ_hats)])
        losses.append(average_loss(W_rob, mkt))
    k = np.argmin(losses)
    print(f"robust, {label:13s}: smallest loss {losses[k]:.2f} percent a year at γ = {γ_values[k]:.0f}")
print(f"Black-Litterman      : smallest loss {bl_loss.min():.2f} percent a year at τ = {τ_best:.0f}")
```

Both kinds of robustness cut the loss of the mean-variance portfolio drastically.

Robustness with $\Psi = \text{diag}(\hat \Sigma)/T$ does better than robustness that only scales positions down, because it also corrects the portfolio's direction.

Black-Litterman shrinkage does best of all in our simulated market, because it shrinks toward the market-implied means, which are close to the true means, while robustness shrinks toward zero, that is, toward holding only the risk-free security.

## Appendix: means are harder to estimate than variances

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
\lim_{N\to \infty} \ \ \mathbb{P}\left\{ \left |\bar X_N - \mathbb E X \right| > \varepsilon \right\} = 0 \quad \quad
$$

and

$$
\lim_{N\to \infty} \ \ \mathbb{P} \left\{ \left| S_N - \mathbb V X \right| > \varepsilon \right\} = 0
$$

A sufficient condition for these convergence results is that the associated MSEs vanish as $N$ goes to infinity, or in other words,

$$
\text{MSE}(\bar X_N, \mathbb E X) = o(1) \quad \quad  \text{and} \quad \quad \text{MSE}(S_N, \mathbb V X) = o(1)
$$

Even if the MSEs converge to zero, they might do so at different speeds.

In the cases that we study, both MSEs shrink at the rate $1/N$, so we compare their constants through the limit

$$
B = \frac{\lim_{N \to \infty} N \, \text{MSE}(S_N, \mathbb V X)}{\lim_{N \to \infty} N \, \text{MSE}(\bar X_N, \mathbb E X)}
$$

which we call the asymptotic relative MSE.

Because the numerator is measured in units of the variance squared and the denominator in units of the variance, the level of $B$ depends on the units of $X$.

Its changes are what carry information, so we compare $B$ across sampling schemes relative to its value for independent data.

We will show that with dependent data, $B$ depends on the sampling frequency.

In particular, we find that the MSE of the variance estimator is less sensitive to increased sampling frequency than the MSE of the mean estimator.

Hence, we can expect $B$ to get smaller with higher frequency data, illustrating that "it is more difficult to estimate means than variances".

A sharper version of the point, which holds the calendar span of the data fixed, comes at the end of the appendix.

### Independent observations

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

Furthermore, both MSEs are $o(1)$, and their ratio converges:

$$
\frac{\text{MSE}(S_N, \sigma^2)}{\text{MSE}(\bar X_N, \mu)} = \frac{2\sigma^2 N}{N-1} \quad \underset{N \to \infty}{\to} \quad 2\sigma^2
$$

We are interested in how this ratio changes as increasing sampling frequency puts dependence into the data.

### Dependence and the sampling interval

To investigate how sampling frequency affects the relative MSE, we assume that the data are generated by a mean-reverting continuous time process of the form

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

We call $h$ the *sampling interval*, so that a higher sampling frequency corresponds to a smaller $h$, whereas $n$ represents the number of *lags* between observations.

Hence, the effective distance between two observations $X_t$ and $X_{t+n}$ in the discrete time notation is equal to $h\cdot n$ in terms of the underlying continuous time process.

Straightforward calculations show that the autocorrelation function for the stochastic process $\{X_{t}\}_{t\in \mathcal T(h)}$ is

$$
\Gamma_h(n) \equiv \text{corr}(X_{t + h n}, X_t) = \exp(-\kappa h n)
$$

and the auto-covariance function is

$$
\gamma_h(n) \equiv \text{cov}(X_{t + h n}, X_t) = \frac{\exp(-\kappa h n)\sigma^2}{2\kappa}
$$

It follows that if $n=0$, the unconditional variance is given by $\gamma_h(0) = \frac{\sigma^2}{2\kappa}$ irrespective of the sampling interval.

The following figure illustrates how the dependence between the observations is related to the sampling interval

- For any given $h$, the autocorrelation converges to zero as we increase the distance $n$ between the observations, which is the "weak dependence" of the $X$ process.
- For a fixed lag length, $n$, the dependence vanishes as the sampling interval grows, and letting $h$ go to $\infty$ gives back the case of IID data.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: Autocorrelation functions for different sampling intervals
    name: fig-bl-autocorrelation
---
μ = .0
κ = .1
σ = .5
var_uncond = σ**2 / (2 * κ)

n_grid = np.linspace(0, 40, 100)

fig, ax = plt.subplots(figsize=(8, 4))
for h, c in [(1, 'darkblue'), (2, 'darkred'), (5, 'orange')]:
    ax.plot(n_grid, np.exp(-κ * n_grid * h), label=rf'$h={h}$', c=c, lw=2)
ax.plot(n_grid, np.where(n_grid == 0, 1.0, 0.0), label=r'"$h=\infty$"',
        c='darkgreen', lw=2)
ax.legend()
ax.grid()
ax.set(xlabel=r'lags between observations, $n$', ylabel=r'$\Gamma_h(n)$')
plt.show()
```

### The sampling interval and the mean estimator

Consider again the AR(1) process generated by discrete sampling with interval $h$.

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
&= \frac{1}{N^2} \left( N \gamma_h(0) + 2 \sum_{i=1}^{N-1} i \cdot \gamma_h(N - i) \right) \\
&= \frac{1}{N^2} \left( N \frac{\sigma^2}{2\kappa} + 2 \sum_{i=1}^{N-1} i \cdot \exp(-\kappa h (N - i)) \frac{\sigma^2}{2\kappa} \right)
\end{aligned}
$$

It is explicit in the above equation that time dependence in the data inflates the variance of the mean estimator through the covariance terms.

Moreover, as we can see, a higher sampling frequency---smaller $h$---makes all the covariance terms larger, everything else being fixed.

This implies that, for a given sample size, the sample average is less precise with high-frequency data.

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

This long-run factor is larger the higher is the frequency (the smaller is $h$).

The variance estimator's asymptotic MSE can also be computed in closed form.

Because the process is Gaussian, Bartlett's formula implies that $N \, \mathbb{V}(S_N)$ converges to $2 \sum_{k=-\infty}^{\infty} \gamma_h(k)^2 = 2 \gamma(0)^2 \frac{1 + \rho^2}{1 - \rho^2}$, while the squared bias of $S_N$ is of order $N^{-2}$.

Dividing by the limit of $N \, \mathbb{V}(\bar X_N)$ gives

```{math}
:label: B-closed-form

B(h) = 2 \gamma(0) \, \frac{1 + \rho^2}{(1 + \rho)^2}, \qquad \rho = \exp(-\kappa h)
```

As $h \to \infty$, $\rho \to 0$ and $B(h)$ approaches the IID benchmark $2 \gamma(0)$, while as $h \to 0$, $\rho \to 1$ and $B(h)$ falls to $\gamma(0)$.

We can confirm this with (large sample) simulations, which show how the relative MSE changes with the sampling interval $h$, measured relative to its IID benchmark $2 \gamma(0)$.

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
---
mystnb:
  figure:
    caption: Relative MSE and the sampling interval
    name: fig-bl-relative-mse
---
# sample_generator is compiled by Numba, whose random number generator
# must be seeded from inside a jitted function
@jit
def set_seed(seed):
    np.random.seed(seed)

# Generate large samples for different sampling intervals
set_seed(1234)
N_app, M_app = 1000, 30000        # Sample size, number of simulations
h_grid = np.linspace(.1, 80, 30)

var_est_store = []
mean_est_store = []

for h in h_grid:
    sample = sample_generator(h, N_app - 1, M_app)    # N_app observations
    mean_est_store.append(np.mean(sample, 1))
    var_est_store.append(np.var(sample, 1, ddof=1))

var_est_store = np.array(var_est_store)
mean_est_store = np.array(mean_est_store)

# MSEs of the estimators
mse_mean = np.var(mean_est_store, 1) + (np.mean(mean_est_store, 1) - μ)**2
mse_var = np.var(var_est_store, 1) \
          + (np.mean(var_est_store, 1) - var_uncond)**2

benchmark_rate = 2 * var_uncond       # IID case

# Relative MSE for large samples
rate_h = mse_var / mse_mean

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(h_grid, rate_h / benchmark_rate, c='darkblue', lw=2,
        label=r'simulated relative MSE ($N = 1000$) divided by $2\gamma(0)$')
ax.axhline(1, c='k', ls='--', label=r'IID benchmark')
ax.set_xlabel('sampling interval, $h$')
ax.legend()
plt.show()
```

The above figure illustrates the relationship between the simulated relative MSE and the sampling interval

- With low-frequency data -- large values of $h$
  -- the relative MSE approaches its IID value.
- As $h$ gets smaller -- the higher the frequency -- the relative MSE
  gets smaller: as the time dependence gets more pronounced, the MSE of
  the mean estimator deteriorates more than that of the variance
  estimator.

At the smallest sampling interval, $h = 0.1$, the simulated ratio falls below its asymptotic limit of one half, because a sample of $N = 1000$ observations then spans only about ten times the process's correlation time $1/\kappa$; {ref}`bl_ex3` compares the simulation with formula {eq}`B-closed-form`.

### Holding the calendar span fixed

The experiment above holds the number of observations $N$ fixed, so a higher sampling frequency also shortens the calendar span of the sample.

For an investor, the economically relevant comparison holds the span fixed and asks whether sampling more often helps.

{cite:t}`merton1980estimating` showed that it does not help with the mean but does help with the variance.

Suppose that log prices follow a Brownian motion with annual drift $m = 0.06$ and annual volatility $s = 0.2$, observed over a span of $20$ years at intervals of length $h$, so that the $n = 20/h$ returns are IID $\mathcal{N}(m h, s^2 h)$.

We estimate the annual drift by $\hat m = \sum_i r_i / 20$ and the annual variance by $\hat s^2 = \sum_i (r_i - \bar r)^2 / ((n-1) h)$, and compute their relative root mean squared errors by simulation.

```{code-cell} ipython3
rng = np.random.default_rng(1234)
m, s = 0.06, 0.20          # annual drift and volatility of the log price
span = 20                  # years of data
M = 200_000                # replications
for label, h in [('annual', 1), ('monthly', 1/12),
                 ('weekly', 1/52), ('daily', 1/252)]:
    n = int(round(span / h))
    # draw the sufficient statistics from their exact sampling distributions
    sum_r = rng.normal(m * h * n, s * np.sqrt(h * n), M)
    ss = s**2 * h * rng.chisquare(n - 1, M)     # sum of squared deviations
    m_hat = sum_r / span
    s2_hat = ss / ((n - 1) * h)
    rmse_m = np.sqrt(np.mean((m_hat - m)**2))
    rmse_s2 = np.sqrt(np.mean((s2_hat - s**2)**2))
    print(f"{label:8s} n = {n:5d}: RMSE(m̂)/m = {rmse_m / m:.3f}, "
          f"RMSE(ŝ²)/s² = {rmse_s2 / s**2:.4f}")
```

With the span fixed at $20$ years, the relative RMSE of the drift estimator is about $0.75$ at every sampling frequency.

That matches $s/(m\sqrt{20}) = 0.745$, because $\hat m = (\log P_{20} - \log P_0)/20$ depends only on the first and last prices.

The relative RMSE of the variance estimator falls from $0.32$ (annual) to $0.09$ (monthly), $0.04$ (weekly) and $0.02$ (daily), in line with $\sqrt{2/(n-1)}$.

Sampling more finely within a fixed span is useless for learning about mean returns but very informative about variances and covariances.

That asymmetry is what licenses Black and Litterman, and robust decision makers, to distrust estimated means while trusting estimated covariances.

## Exercises

```{exercise}
:label: bl_ex1

*How much does $\tau$ matter, and what value of $\tau$ would a Bayesian choose?*

Use the simulated market `mkt` of the lecture, and use the market-implied risk aversion $\delta_m$ both to construct $\mu_{BL}$ and to compute portfolios.

1. Confirm the convex-combination formula {eq}`bl-convex` numerically on a grid of $\tau \in [10^{-3}, 10^3]$, and plot $\|\tilde w - w_m\|$ and $\|\tilde w - w\|$ against $\tau$ on a log scale.
1. In the Bayesian interpretation of the lecture, $\hat\mu \mid \mu \sim \mathcal N(\mu, \tau\Sigma)$. If $\hat \mu$ is the sample mean of $T$ IID draws, what value of $\tau$ does that imply? Compute the implied portfolio and describe it.
1. Now put the scalar on the prior instead, as in the calibration of He and Litterman: $\mu \sim \mathcal N(\mu_{BL}, \tau_p \Sigma)$ and $\hat\mu \mid \mu \sim \mathcal N(\mu, \Sigma/T)$. Show that the posterior mean puts weight $1/(1 + \tau_p T)$ on $\mu_{BL}$ and $\tau_p T / (1 + \tau_p T)$ on $\hat \mu$, and find the value of $\tau_p$ that reproduces the loss-minimizing $\tau$ of the section on whether shrinkage pays.
1. In our simulated market, the true means depart from $\delta \Sigma w_m$ by independent draws with standard deviation $0.001$. Compare the variance of these departures with $\tau_p$ times the average variance of the assets' excess returns. What do you conclude about the prior that the best $\tau$ implies?
```

```{solution-start} bl_ex1
:class: dropdown
```

```{code-cell} ipython3
T = mkt.T
w = np.linalg.solve(δ_m * mkt.Σ_hat, mkt.μ_hat)          # mean-variance weights

τ_grid_ex = np.logspace(-3, 3, 200)
dist_m, dist_mv, gap = [], [], []
for τ in τ_grid_ex:
    μ_tilde = black_litterman(1, μ_BL, mkt.μ_hat, mkt.Σ_hat, τ * mkt.Σ_hat)
    w_tilde = np.linalg.solve(δ_m * mkt.Σ_hat, μ_tilde)
    gap.append(np.max(np.abs(w_tilde - (τ * mkt.w_m + w) / (1 + τ))))
    dist_m.append(np.linalg.norm(w_tilde - mkt.w_m))
    dist_mv.append(np.linalg.norm(w_tilde - w))

print(f"max |w_tilde - (τ w_m + w)/(1+τ)| over the grid: {max(gap):.1e}")
print(f"||w - w_m|| = {np.linalg.norm(w - mkt.w_m):.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(τ_grid_ex, dist_m, lw=2, label=r'$\|\tilde w - w_m\|$')
ax.semilogx(τ_grid_ex, dist_mv, lw=2, label=r'$\|\tilde w - w\|$')
ax.axvline(1 / T, c='k', ls='--', label=r'$\tau = 1/T$')
ax.axvline(τ_best, c='gray', ls=':', label=r'loss-minimizing $\tau$')
ax.set_xlabel(r'$\tau$')
ax.legend()
plt.show()

# Bayesian calibration: sampling covariance of μ̂ is Σ/T, so τ = 1/T
τ_T = 1 / T
w_T = (τ_T * mkt.w_m + w) / (1 + τ_T)
print(f"τ = 1/T: weight on w_m = {τ_T / (1 + τ_T):.4f}, "
      f"min weight = {w_T.min():.2f}, max weight = {w_T.max():.2f}")

# He-Litterman placement: weight on μ̂ is τ_p T / (1 + τ_p T) = 1 / (1 + τ)
τ_p_best = 1 / (τ_best * T)
avg_var = np.mean(np.diag(mkt.Σ))
print(f"best τ = {τ_best:.0f} corresponds to τ_p = {τ_p_best:.5f}")
print(f"variance of departures 0.001² = {0.001**2:.1e}; "
      f"τ_p × average variance = {τ_p_best * avg_var:.1e}")
```

The numerical check confirms {eq}`bl-convex` to machine precision.

The two distance curves cross at $\tau = 1$, where $\tilde w$ lies halfway between $w$ and $w_m$.

If $\hat\mu$ is the sample mean of $T = 120$ observations, its sampling covariance is $\Sigma/T$, so the Bayesian interpretation of the lecture implies $\tau = 1/T \approx 0.008$.

The posterior then puts weight $0.008$ on the market view, and the recommended portfolio is essentially the extreme mean-variance portfolio.

Taken literally, a prior covariance equal to the covariance of returns lets the data swamp the prior.

That prior is extremely diffuse: it says that uncertainty about the *mean* excess return is as large as the volatility of a single month's return.

With the prior $\mathcal N(\mu_{BL}, \tau_p \Sigma)$ and the data $\mathcal N(\mu, \Sigma/T)$, the posterior mean is

$$
\frac{\tau_p^{-1} \mu_{BL} + T \hat \mu}{\tau_p^{-1} + T}
$$

which puts weight $1/(1 + \tau_p T)$ on $\mu_{BL}$ and $\tau_p T / (1 + \tau_p T)$ on $\hat \mu$.

The weight on $\hat \mu$ equals the weight $1/(1 + \tau)$ on the mean-variance portfolio in {eq}`bl-convex` when $\tau = 1/(\tau_p T)$.

The loss-minimizing $\tau$ therefore corresponds to a very small $\tau_p$, a prior that is very confident in the market-implied means.

It is about $300$ times smaller than the value $\tau_p = 0.05$ used by {cite:t}`HeLitterman1999`, which with the sampling covariance $\Sigma / T$ would put weight $6/7$ on $\hat \mu$ and leave the portfolio close to the extreme mean-variance portfolio.

That confidence is justified here: the variance of the true departures from the market-implied means is of the same order as $\tau_p$ times the average variance of returns.

So the best amount of shrinkage is roughly what a Bayesian with a correctly specified prior would choose.

```{solution-end}
```

```{exercise}
:label: bl_ex2

*The robust portfolio, its worst-case mean, and short positions.*

Use the simulated market `mkt` of the lecture, set $\delta = \delta_m$, and set $\theta = 1/\delta_m$, so that $\gamma = \theta^{-1} = \delta_m$.

1. For the mean-variance portfolio $w$, minimize $w'(\hat\mu + C v) + \frac{\theta}{2} v'v$ numerically over the mean distortion $v$, where $CC' = \hat\Sigma$. Check that the minimizer is $v = -\theta^{-1} C' w$ and that the minimized value equals $w'\hat\mu - \frac{1}{2\theta} w'\hat\Sigma w$.
1. Maximize ${\sf T}[w'(\vec r - r_f {\bf 1})] - \frac{\delta}{2} w'\hat\Sigma w$ numerically and confirm that the maximizer is $w_{\rm rob} = (\delta + \gamma)^{-1} \hat\Sigma^{-1} \hat\mu$.
1. Let $\mu_{wc} = \hat\mu - \gamma \hat\Sigma w_{\rm rob}$ be the worst-case mean *at the robust optimum*. Show that a non-robust investor with risk aversion $\delta$ who believes $\mu_{wc}$ chooses $w_{\rm rob}$. In what sense is $\mu_{wc}$ like the Black-Litterman $\mu_{BL}$?
1. Compare the sign patterns of the mean-variance, robust, and Black-Litterman ($\tau = 1$) portfolios. Does robustness of this kind cure extreme long-short positions?
```

```{solution-start} bl_ex2
:class: dropdown
```

For part 3, note that

$$
(\delta \hat \Sigma)^{-1} (\hat \mu - \gamma \hat \Sigma w_{\rm rob})
= \frac{\delta + \gamma}{\delta} w_{\rm rob} - \frac{\gamma}{\delta} w_{\rm rob}
= w_{\rm rob}
$$

The worst-case mean is therefore an "implied" mean in the same sense as $\mu_{BL}$: it is the belief about mean excess returns that rationalizes the recommended portfolio for an ordinary mean-variance investor.

The difference is where the belief comes from.

Black and Litterman choose the belief so that it supports the *market* portfolio, while the robust investor's belief comes from a malevolent agent's response to the investor's own portfolio.

```{code-cell} ipython3
μ_hat, Σ_hat = mkt.μ_hat, mkt.Σ_hat
C = np.linalg.cholesky(Σ_hat)
δ = δ_m
θ = 1 / δ_m
γ = 1 / θ

# (1) Worst-case mean shift for a given portfolio
w_mv = np.linalg.solve(δ * Σ_hat, μ_hat)
obj = lambda v: w_mv @ (μ_hat + C @ v) + θ * v @ v / 2
v_star = minimize(obj, np.zeros(N), method='BFGS', options={'gtol': 1e-12}).x
print(f"max |v* - (-C'w/θ)|: {np.max(np.abs(v_star + C.T @ w_mv / θ)):.1e}")
print(f"T numeric: {obj(v_star):.10f}, closed form: "
      f"{w_mv @ μ_hat - w_mv @ Σ_hat @ w_mv / (2 * θ):.10f}")

# (2) Robust portfolio and the worst-case mean as an "implied" mean
w_rob = np.linalg.solve((δ + γ) * Σ_hat, μ_hat)
neg = lambda w: -(w @ μ_hat - (δ + γ) / 2 * w @ Σ_hat @ w)
w_num = minimize(neg, np.zeros(N), method='BFGS', options={'gtol': 1e-12}).x
print(f"max |w_num - w_rob|: {np.max(np.abs(w_num - w_rob)):.1e}")

μ_wc = μ_hat - γ * Σ_hat @ w_rob
w_implied = np.linalg.solve(δ * Σ_hat, μ_wc)
print(f"max |w(μ_wc) - w_rob|: {np.max(np.abs(w_implied - w_rob)):.1e}")

# (3) Compare with mean-variance and Black-Litterman portfolios
print("w_rob / w_mv (elementwise):", np.round(w_rob / w_mv, 4))
w_bl = (1.0 * mkt.w_m + w_mv) / 2                    # τ = 1
for name, x_ in [('mean-variance', w_mv), ('robust', w_rob), ('Black-Litterman', w_bl)]:
    print(f"{name:16s} short positions: {np.sum(x_ < 0)}, "
          f"sum |w| = {np.abs(x_).sum():.2f}, sum w = {x_.sum():.2f}")
print("sign pattern mv :", np.sign(w_mv).astype(int))
print("sign pattern rob:", np.sign(w_rob).astype(int))
print("sign pattern BL :", np.sign(w_bl).astype(int))

# Black-Litterman short positions as confidence in μ̂ falls (τ rises)
for τ in [1, 3, 10, 30]:
    w_bl_τ = (τ * mkt.w_m + w_mv) / (1 + τ)
    print(f"τ = {τ:3d}: Black-Litterman short positions = {np.sum(w_bl_τ < 0)}")
```

The numerical minimization reproduces the worst-case distortion $v = -\theta^{-1}C'w$ and the closed form for ${\sf T}$, and the numerically maximized robust portfolio matches $(\delta+\gamma)^{-1}\hat\Sigma^{-1}\hat\mu$.

A mean-variance investor who holds the worst-case mean $\mu_{wc}$ chooses exactly $w_{\rm rob}$.

Every element of $w_{\rm rob}/w$ equals $\delta/(\delta+\gamma) = 1/2$.

Robustness of this kind is observationally equivalent to raising risk aversion from $\delta$ to $\delta + \gamma$.

It halves gross exposure but leaves the pattern of long and short positions untouched: both portfolios short the same assets.

With $\tau = 1$ the Black-Litterman portfolio lies halfway to $w_m$, which has no short positions, but it still shorts the same four assets.

Raising $\tau$ removes short positions one by one, leaving three at $\tau = 3$, two at $\tau = 10$, and one at $\tau = 30$, something no value of $\theta$ can do in this robust model.

The Black-Litterman adjustment changes the *direction* of the portfolio, while this robust adjustment changes only its *scale*.

As the section on robustness about the mean shows, a robust decision maker who distrusts only the mean, and measures that distrust in a metric that is not proportional to $\Sigma$, can also change the direction.

```{solution-end}
```

```{exercise}
:label: bl_ex3

*Means versus variances: a closed form for $B(h)$, and how much data is enough.*

1. For the discretely sampled Ornstein-Uhlenbeck process in the appendix, let $\rho = \exp(-\kappa h)$ and $\gamma(0) = \sigma^2/(2\kappa)$. Starting from $N\,\mathbb V(\bar X_N) \to \gamma(0)\frac{1+\rho}{1-\rho}$ and Bartlett's formula $N\,\mathbb V(S_N) \to 2 \sum_{k=-\infty}^{\infty} \gamma_h(k)^2 = 2\gamma(0)^2 \frac{1+\rho^2}{1-\rho^2}$, verify the closed form {eq}`B-closed-form` for the asymptotic relative MSE $B(h)$. Plot it against the simulated `rate_h` from the appendix. Where does the simulation depart from the formula, and why?
1. In the fixed-span example of the appendix, how many years of data does an investor need for the relative RMSE of the drift estimator to fall to $0.25$, and to $0.10$? How many daily observations are needed for the relative RMSE of the variance estimator to fall to the same values?
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

The asymptotic relative MSE therefore falls by at most a factor of two as the sampling frequency rises with $N$ fixed.

```{code-cell} ipython3
# Part 1: closed-form asymptotic relative MSE
ρ_grid = np.exp(-κ * h_grid)
B_asym = 2 * var_uncond * (1 + ρ_grid**2) / (1 + ρ_grid)**2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(h_grid, rate_h, lw=2, label='simulated relative MSE, $N=1000$')
ax.plot(h_grid, B_asym, 'k--', lw=2, label='asymptotic formula')
ax.axhline(var_uncond, c='r', ls=':', label=r'limit as $h \to 0$')
ax.set_xlabel('sampling interval $h$')
ax.legend()
plt.show()

for k in [0, 1, 5, 29]:
    print(f"h = {h_grid[k]:6.2f}: simulated ratio = {rate_h[k]:.3f}, "
          f"asymptotic B = {B_asym[k]:.3f}, "
          f"N·MSE(mean)/γ(0) = {N_app * mse_mean[k] / var_uncond:7.2f}")

# Part 2: data needed for a given relative RMSE
for target in [0.25, 0.10]:
    years = (s / (target * m))**2                  # s / (m √years) = target
    days = 2 / target**2 + 1                       # √(2 / (n - 1)) = target
    print(f"relative RMSE {target:.2f}: drift needs {years:7.0f} years, "
          f"variance needs {days:5.0f} daily observations")
```

The closed form tracks the simulation closely except at the very highest frequencies.

Take $h = 0.1$, where $1/(1-\rho) \approx 100$: a sample of $N = 1000$ spans only about ten "correlation times", so the asymptotic formula ($1.25$) overstates the finite-sample ratio (about $1.06$ in the simulation).

The printed values of $N \cdot \text{MSE}(\bar X_N)/\gamma(0)$ show what the ratio hides: going from $h = 80$ to $h = 0.1$ inflates the MSE of the mean by a factor of roughly $180$.

The MSE of the variance estimator also deteriorates, by a factor of roughly $75$, so their ratio falls only from about $2.5$ to about $1.1$.

For part 2, the relative RMSE of $\hat m$ is $s/(m \sqrt{\text{years}})$ whatever the sampling frequency, so a relative RMSE of $0.25$ requires about $180$ years of data and one of $0.10$ requires about $1{,}100$ years.

The relative RMSE of $\hat s^2$ is about $\sqrt{2/(n-1)}$, so the same precision requires about $33$ and $200$ daily observations, roughly six and a half weeks and ten months of trading days.

```{solution-end}
```
