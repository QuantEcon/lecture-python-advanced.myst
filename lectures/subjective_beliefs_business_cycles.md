---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(subjective_beliefs_bc)=
# Survey Data and Subjective Beliefs in Business Cycle Models

```{index} single: Subjective Beliefs; Business Cycles
```

## Overview

This lecture studies whether household survey data on macroeconomic
expectations can discipline business cycle models, following
{cite:t}`bhandari2025survey`.

A central finding is that household forecasts of unemployment and inflation
exhibit **systematic upward biases** relative to rational forecasts.

These biases, called *belief wedges*, are:

* **Persistent and countercyclical**: they are larger during recessions.
* **Positively correlated**: optimism/pessimism about unemployment and inflation
  move together.
* **One-factor in structure**: a single latent state accounts for most
  variation across wedges.

We follow {cite:t}`bhandari2025survey` in interpreting this evidence using
**robust preferences** ({cite:t}`HansenSargent2001`; {cite:t}`HansenSargent2008`).

Robust preferences provide a natural way to model pessimism and optimism: 
a pessimistic agent acts as if states that deliver low continuation values are
more likely than they really are, while an optimistic agent overweights states
that deliver high continuation values.

This is done through the distortion studied in {doc}`five_preferences`.

When calibrated to the Michigan Survey of
Consumers (1982Q1-2019Q4), this mechanism yields a time-varying *belief shock*
that substantially reduces the **unemployment volatility puzzle** ---
the fact that standard New Keynesian models with only technology and monetary
policy shocks generate far too little unemployment volatility.

In this lecture, we will cover:

* How to define and measure belief wedges from household survey data.
* Why optimal pessimism is a mean shift of the shock distribution,
  proportional to the shock's exposure to continuation values.
* How belief distortions propagate through a linearized DSGE model.
* Why a calibrated belief shock helps resolve the unemployment volatility
  puzzle.

We start with the following imports

```{code-cell} ipython3
import datetime
from typing import NamedTuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import norm
```

## Measuring belief wedges

### Definition

Let $E_t[\cdot]$ denote expectations under the **data-generating** (objective)
probability measure and $\tilde{E}_t[\cdot]$ denote **subjective** (survey)
expectations.

For any scalar variable $z_{t+1}$, the **one-period belief wedge** is

$$

\Delta_t^{(1)}(z) \;=\; \tilde{E}_t[z_{t+1}] - E_t[z_{t+1}].

$$

A positive wedge means households expect $z_{t+1}$ to be higher than the
data-generating forecast.

For unemployment and inflation, this sign convention corresponds to an upward
forecast bias.

The Michigan Survey asks about outcomes one year ahead, so the empirical
objects below are four-quarter wedges

$$

\Delta_t^{(4)}(z)
= \tilde E_t[z_{t+4}] - E_t[z_{t+4}].

$$

We work mostly with the one-period wedge because it is the cleanest way to
explain the theory; the appendix derives the multi-period version.

In the data, $\tilde{E}_t[\cdot]$ is measured from the Michigan Survey of
Consumers.

The benchmark $E_t[\cdot]$ is computed from a quarterly VAR, with Survey of
Professional Forecasters (SPF) forecasts used as a robustness check.

In the structural model, the same object is interpreted as the difference
between subjective and data-generating expectations.

Thus the lecture uses one object in two related ways: empirically, a belief
wedge is a survey forecast minus a statistical benchmark forecast; in the
model, it is a subjective expectation minus an objective expectation.

The raw Michigan unemployment question is categorical, so {cite:t}`bhandari2025survey`
convert it into a quantitative forecast using the Carlson--Parkin procedure as
adapted by {cite:t}`MankiwReisWolfers2003`.

### Replicating the wedges

{cite:t}`bhandari2025survey` document the properties of the belief wedges on
the sample 1982Q1--2019Q4, and we now replicate that evidence.

The raw inputs are extracted from the paper's publicly
available replication package
([doi:10.5281/zenodo.10194324](https://doi.org/10.5281/zenodo.10194324),
licensed CC-BY-4.0).

The first file contains the quarterly macroeconomic series for the
forecasting VAR.

(Monthly series are averaged to quarterly frequency.)

The second file contains monthly Michigan Survey aggregates: the mean
one-year-ahead inflation forecast and the shares of households answering
"more unemployment," "about the same," and "less unemployment," together with
the monthly unemployment rate.

```{code-cell} ipython3
data_path = '_static/lecture_specific/subjective_beliefs_business_cycles/'
macro_q = pd.read_csv(data_path + 'bbh_macro_quarterly.csv',
                      index_col='YYYYQ')
mich_m = pd.read_csv(data_path + 'bbh_michigan_monthly.csv',
                     index_col='yyyymm')
```

Quarters are indexed as `YYYYQ`, so `19821` means 1982Q1, and months as
`yyyymm`.

#### The VAR benchmark forecast

The data-generating forecast $E_t[\cdot]$ comes from a quarterly VAR with two
lags in nine variables: CPI inflation over the past year, annualized real GDP
growth, the unemployment rate, the log change in the relative price of
investment goods, capacity utilization, log hours worked per capita, the
consumption rate, the investment rate, and the federal funds rate.

```{code-cell} ipython3
q = macro_q
var_data = pd.DataFrame({
    'infl_yoy': 100 * (q.CPIAUCSL / q.CPIAUCSL.shift(4) - 1),
    'gdp_gr':   400 * np.log(q.GDPC1 / q.GDPC1.shift(1)),
    'unrate':   q.UNRATE,
    'dpiric':   100 * np.log(q.PIRIC / q.PIRIC.shift(1)),
    'cumfns':   q.CUMFNS,
    'hours_pc': 100 * np.log(q.PRS85006023 * q.CE16OV / q.CNP16OV),
    'cons_r':   100 * (q.PCEND + q.PCESV) / q.GDP,
    'inv_r':    100 * q.GPDI / q.GDP,
    'ffr':      q.FEDFUNDS,
})
output_gap = 100 * np.log(q.GDPC1 / q.GDPPOT)
```

The VAR is estimated by least squares on 1960Q1--2019Q4, and forecasts are
iterated four quarters ahead from every quarter.

```{code-cell} ipython3
def var_forecasts(data, first=19601, last=20194, horizon=4):
    """OLS VAR(2) and iterated `horizon`-step-ahead forecasts."""
    X, idx = data.values, data.index.values
    est = (idx >= first) & (idx <= last)
    Y_est = np.array([X[t] for t in range(2, len(idx)) if est[t]])
    X_est = np.array([np.concatenate([X[t-1], X[t-2], [1.0]])
                      for t in range(2, len(idx)) if est[t]])
    B = np.linalg.lstsq(X_est, Y_est, rcond=None)[0]

    forecast = np.full_like(X, np.nan)
    for t in range(1, len(idx)):
        z1, z2 = X[t], X[t-1]
        if np.isnan(z1).any() or np.isnan(z2).any():
            continue
        for h in range(horizon):
            z1, z2 = np.concatenate([z1, z2, [1.0]]) @ B, z1
        forecast[t] = z1
    return pd.DataFrame(forecast, index=idx, columns=data.columns)

E_t4 = var_forecasts(var_data)    # E_t[x_{t+4}], info through quarter t
```

#### From categorical answers to a forecast

The Michigan unemployment question is categorical, so we convert the answer
shares into a mean forecast with the Carlson--Parkin procedure.

Assume household forecasts of the change in unemployment over the next year
are normally distributed across households, $N(\mu_t, \sigma_t^2)$, and that a
household answers "about the same" when its forecast lies in $[-a, a]$.

The shares answering "more" ($q_t^u$) and "less" ($q_t^d$) then satisfy

$$

q_t^u = 1 - \Phi\!\left(\frac{a - \mu_t}{\sigma_t}\right),
\qquad
q_t^d = \Phi\!\left(\frac{-a - \mu_t}{\sigma_t}\right),

$$

where $\Phi$ is the standard normal cdf, and inverting the two equations
gives

$$

\sigma_t = \frac{2a}{\Phi^{-1}(1 - q_t^u) - \Phi^{-1}(q_t^d)},
\qquad
\mu_t = a - \sigma_t\, \Phi^{-1}(1 - q_t^u).

$$

The threshold $a$ scales the whole series; {cite:t}`bhandari2025survey` pin it
down by comparing the implied cross-sectional dispersion with that of SPF
forecasts.

We set $a = 1.045$, which reproduces their fitted forecast series.

```{code-cell} ipython3
def carlson_parkin(share_more, share_less, a=1.045):
    """Mean forecast implied by categorical shares (normal cross-section)."""
    z_up, z_down = norm.ppf(1 - share_more), norm.ppf(share_less)
    σ = 2 * a / (z_up - z_down)
    return a - σ * z_up
```

#### Constructing the wedges

Timing follows the paper: responses from the first month of quarter $t+1$ are
treated as forecasts made with information through quarter $t$.

The unemployment *level* forecast adds the expected change to the
unemployment rate in the month the forecast is made.

Each wedge then subtracts the corresponding VAR forecast: year-over-year
inflation, and the unemployment rate four quarters ahead.

Both wedges are measured in percentage points (pp), the unit we use for them
throughout the lecture.

```{code-cell} ipython3
def build_wedges(mich_m, E_t4, first=19821, last=20194):
    """Survey minus VAR forecasts for unemployment and inflation."""
    rows = []
    for yq in E_t4.index:
        if not first <= yq <= last:
            continue
        y, qq = yq // 10, yq % 10
        mm = (y + (qq == 4)) * 100 + (qq % 4) * 3 + 1  # 1st month of qtr t+1
        s = mich_m.loc[mm]
        total = s.share_more + s.share_same + s.share_less
        du = carlson_parkin(s.share_more / total, s.share_less / total)
        rows.append((yq,
                     s.unrate + du - E_t4.loc[yq, 'unrate'],
                     s.px1_mean - E_t4.loc[yq, 'infl_yoy']))
    return pd.DataFrame(rows, columns=['YYYYQ', 'unemp', 'infl']
                        ).set_index('YYYYQ')

wedges = build_wedges(mich_m, E_t4)
wedge_u, wedge_π = wedges.unemp, wedges.infl
W = np.column_stack([wedge_u, wedge_π])
eigvals = np.linalg.eigvalsh(np.cov(W, rowvar=False))
pc1_share = eigvals[-1] / eigvals.sum()

# Quarterly dates and NBER recessions for plotting
quarters = [datetime.date(yq // 10, 3 * (yq % 10) - 2, 1)
            for yq in wedges.index]


def fred_recession_spans(start, end):
    """NBER recession spans from FRED's monthly USREC indicator."""
    fetch_start = pd.Timestamp(start) - pd.DateOffset(years=1)
    fetch_end = pd.Timestamp(end) + pd.DateOffset(months=1)
    rec = pd.read_csv(
        'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC',
        parse_dates=['observation_date'],
        index_col='observation_date'
    )['USREC'].loc[fetch_start:fetch_end]
    rec = pd.to_numeric(rec, errors='coerce').fillna(0).astype(bool)

    starts = rec.index[rec & ~rec.shift(fill_value=False)]
    ends = rec.index[~rec & rec.shift(fill_value=False)]
    if rec.iloc[-1]:
        ends = ends.append(pd.DatetimeIndex([rec.index[-1]
                                             + pd.offsets.MonthBegin()]))
    return [(s.date(), e.date()) for s, e in zip(starts, ends)]


recessions = fred_recession_spans(quarters[0], quarters[-1])
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: replicated belief wedges, 1982Q1-2019Q4
    name: fig-sbbc-belief-wedges
---
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

axes[0].plot(quarters, wedge_u, color='steelblue', linewidth=2,
             label='Unemployment belief wedge')
axes[0].axhline(np.mean(wedge_u), color='steelblue', linestyle='--',
                linewidth=0.8, alpha=0.7)
axes[0].set_ylabel('percentage points')
axes[0].legend(loc='upper left')

axes[1].plot(quarters, wedge_π, color='darkorange', linewidth=2,
             label='Inflation belief wedge')
axes[1].axhline(np.mean(wedge_π), color='darkorange', linestyle='--',
                linewidth=0.8, alpha=0.7)
axes[1].set_ylabel('percentage points')
axes[1].legend(loc='upper left')

for ax in axes:
    for start, end in recessions:
        ax.axvspan(start, end, color='grey', alpha=0.25, linewidth=0)
    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(quarters[0], quarters[-1])

plt.tight_layout()
plt.show()
```

Both wedges are positive most of the time --- households persistently
overpredict unemployment and inflation --- and both rise during the shaded
NBER recessions.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: one-factor structure of belief wedges
    name: fig-sbbc-wedge-scatter
---
fig, ax = plt.subplots()
sc = ax.scatter(wedge_u, wedge_π, c=range(len(wedges)), cmap='RdYlGn_r',
                alpha=0.7, s=20)
plt.colorbar(sc, ax=ax, label='quarter (dark = recent)')
ax.set_xlabel('unemployment wedge (pp)')
ax.set_ylabel('inflation wedge (pp)')
corr = np.corrcoef(wedge_u, wedge_π)[0, 1]
ax.text(0.05, 0.90,
        f'correlation = {corr:.2f}\nPC1 share = {pc1_share:.3f}',
        transform=ax.transAxes, fontsize=11)
plt.tight_layout()
plt.show()
```

The scatter plot shows the one-factor structure.

Each point is one quarter, with the horizontal coordinate equal to the
unemployment wedge and the vertical coordinate equal to the inflation wedge.

The points form an upward-sloping cloud rather than a line: a common
pessimism factor drives both wedges, while survey noise and other
idiosyncratic variation keep them from being collinear.

The reported PC1 share of 0.809 is computed from the raw covariance matrix of
the two replicated wedges, so it reflects both their correlation and the
larger variance of the inflation wedge.

{cite:t}`bhandari2025survey` report 78.6% under their preferred
normalization; depending on the normalization, the first component explains
roughly 79--81% of the joint variation.

### Empirical facts

The figures above show three key empirical facts about the belief wedges:

- Both wedges are **positive on average**: households expect higher unemployment
and higher inflation than the rational forecast.

- Both wedges are **countercyclical**: they rise during every NBER recession in
the sample.

- The wedges are **positively correlated and share one dominant factor**: the
first principal component explains about four-fifths of their joint
variation.

A positive unemployment wedge is naturally read as pessimism, since
unemployment is high in bad times.

The positive inflation wedge carries the same interpretation because
households regard high inflation as a feature of bad times.

The same one-factor pattern appears in the cross section.

This evidence supports the interpretation that the wedges reflect a common
pessimism/optimism component rather than two unrelated forecast mistakes.

These moments are the calibration targets for the belief shock $\theta_t$,
the pessimism parameter formalized in the next section.

The code below also defines two *wedge loadings*, $c_u$ and $c_\pi$, that the
model illustrations later in the lecture use to map $\theta_t$ into wedges.

In the full model these loadings are endogenous equilibrium objects,
covariances between shocks and continuation values, but here we set them
directly so that the implied wedges equal the empirical means of 0.52 and
1.22 pp at $\theta_t = \mu_\theta$.

```{code-cell} ipython3
# Belief-shock calibration from the paper
μ_θ = 5.64   # mean of belief-shock parameter θ
ρ_θ = 0.714  # AR(1) persistence: autocorrelation of the wedges' first PC
σ_θ = 4.3    # innovation volatility

# Wedge loadings used later in the model illustrations
# In the full model these are equilibrium objects
c_u = 0.52 / μ_θ
c_π = 1.22 / μ_θ
```

## A model of pessimism

Before turning to the theory, the table below collects the notation used in
the rest of the lecture.

| Symbol | Meaning |
|---|---|
| $x_t$ | state (a vector in general; log consumption in the scalar example) |
| $\bar x$ | steady state of the state; $x_{1t}$ is the first-order deviation from $\bar x$ |
| $w_{t+1}$ | standard normal innovation under the data-generating measure |
| $m_{t+1}$ | likelihood ratio that distorts the data-generating measure |
| $\theta_t$ | belief factor: $\theta_t > 0$ is pessimism, $\theta_t < 0$ is optimism |
| $\bar\theta$ | loading of the belief factor on the state, $\theta_t = \bar\theta x_t$ |
| $\mu_\theta$, $\rho_\theta$, $\sigma_\theta$ | mean ($\mu_\theta = \bar\theta \bar x$), persistence, and innovation volatility of $\theta_t$ |
| $v_t$, $v_x$, $v_q$ | continuation value, its slope in the state, and its constant term |
| $\nu_t$ | subjective mean shift of the innovation $w_{t+1}$ |
| $\Delta_t^{(\tau)}(z)$ | $\tau$-period belief wedge for variable $z$ |

### Robust preferences

Why would households have systematically biased beliefs?

One answer comes from **robust control** or **multiplier preferences**
({cite:t}`HansenSargent2001`, {cite:t}`HansenSargent2008`).

Recall that an agent represented by multiplier preferences solves

$$
v_t \;=\; \min_{\substack{m_{t+1} > 0 \\ E_t[m_{t+1}] = 1}}
\Bigl\{
  u(x_t)
  + \beta E_t\!\left[m_{t+1} v_{t+1}\right]
  + \frac{\beta}{\theta_t}\, E_t\!\left[m_{t+1} \log m_{t+1}\right]
\Bigr\}.
$$

Here $m_{t+1}$ is a **likelihood ratio** (Radon–Nikodym derivative) that
distorts the reference measure, and the last term is an entropy penalty that
keeps the distortion from being too extreme.

Assume state vector $x_t \in \mathbb{R}^n$ follows a Markov law of motion

$$
x_{t+1} = \psi(x_t, w_{t+1}),
$$

where $w_{t+1} \sim N(0, I_k)$ is i.i.d. under the data-generating measure,
and the penalty parameter is linear in the state:

$$
\theta_t = \bar\theta x_t,
$$

for a $1 \times n$ vector of parameters $\bar\theta$.

The scalar $\theta_t$ controls the direction and strength of the belief tilt.

The minimization problem above corresponds to $\theta_t > 0$: larger
$\theta_t$ means more pessimism.

Because $\theta_t$ is linear in the state, it can turn negative, in which case
the inner problem becomes a maximization that tilts probability toward
high-continuation-value states, which corresponds to optimism.

The inner minimization has a closed-form solution.

Minimizing $E_t[m_{t+1} v_{t+1}] + \frac{1}{\theta_t} E_t[m_{t+1} \log m_{t+1}]$
state by state subject to $E_t[m_{t+1}] = 1$ gives the first-order condition

$$
v_{t+1} + \frac{1}{\theta_t}\left(\log m_{t+1} + 1\right) + \lambda_t = 0,
$$

where $\lambda_t$ is the multiplier on the constraint, so
$m_{t+1} \propto \exp(-\theta_t v_{t+1})$ and the constraint pins down the
normalization:

$$
m_{t+1}^* \;=\;
\frac{\exp(-\theta_t v_{t+1})}{E_t[\exp(-\theta_t v_{t+1})]}.
$$

Since $m_{t+1}^*$ assigns higher weight to states where $v_{t+1}$ is low (bad
outcomes), pessimistic agents effectively over-weight recessions in their
probability assessments.

Substituting $m_{t+1}^*$ back into the recursion gives the equivalent
**risk-sensitive** form

$$
v_t = u(x_t) - \frac{\beta}{\theta_t}
\log E_t\!\left[\exp(-\theta_t v_{t+1})\right],
$$

which replaces the expected continuation value with a soft minimum: as
$\theta_t \to 0$ it reduces to $u(x_t) + \beta E_t[v_{t+1}]$, and as
$\theta_t \to \infty$ it approaches the worst case over states in a bounded
or finite-state setting (with unbounded Gaussian shocks the soft minimum
instead falls without bound).

In the robust-control literature, this distortion expresses fear of model
misspecification.

{cite:t}`bhandari2025survey` instead take the recursion as a model of
pessimism and optimism, and let survey data discipline the process for
$\theta_t$.

Survey data also resolve an identification problem.

With log period utility, these preferences are mathematically equivalent to
Epstein--Zin preferences with time-varying risk aversion
$\gamma_t = \theta_t + 1$, so asset prices and macroeconomic aggregates alone
cannot distinguish time-varying pessimism from time-varying risk premia.

Survey forecasts can: fluctuations in rational risk premia leave forecasts
unbiased, whereas subjective beliefs show up directly as belief wedges.

### Connection to the belief wedge

The belief wedge is the expected deviation between subjective and objective
forecasts.

Using $\tilde{E}_t[\cdot] = E_t[m_{t+1}^* \cdot]$:

$$
\Delta_t^{(1)}(z)
= \tilde{E}_t[z_{t+1}] - E_t[z_{t+1}]
= E_t\!\left[m_{t+1}^* z_{t+1}\right] - E_t[z_{t+1}]
= \operatorname{Cov}_t(m_{t+1}^*, z_{t+1}).
$$

The last equality holds because $E_t[m_{t+1}^*] = 1$, so
$E_t[m^*_{t+1} z_{t+1}] - E_t[z_{t+1}]
= E_t[m^*_{t+1} z_{t+1}] - E_t[m^*_{t+1}]\,E_t[z_{t+1}]$.

So the belief wedge equals the covariance between the distorted likelihood
ratio and the variable of interest.

When $v_{t+1}$ is high in states where
$z_{t+1}$ is also high, $m_{t+1}^*$ will be low in those states, making the
covariance negative (i.e. the agent *underestimates* good-state variables).

For unemployment (which varies inversely with good economic outcomes), the
wedge is positive: pessimists expect higher unemployment than the model predicts.

### Illustration: optimal belief distortion

We now compute, in the smallest model that can carry them, the
risk-sensitive recursion, the optimal distortion $m_{t+1}^*$, and the belief
wedge in closed form.

The payoff is the central formula of the lecture: optimal pessimism is a
**mean shift** of the shock distribution, equal to $-\theta_t$ times the
exposure of the continuation value to the shock.

Consider an endowment economy in which log consumption $x_t$ follows the
linear process below.

$$
x_{t+1} = \rho_x x_t + \sigma_x w_{t+1}, \qquad w_{t+1} \sim N(0,1).
$$

Here $\rho_x$ is the persistence of the state, $\sigma_x$ is the standard
deviation of its innovation, and period utility is $u(x_t) = (1 - \beta)x_t$
for log utility in consumption.

The calculation has four steps:

1. guess an affine continuation value;
2. evaluate the risk-sensitive recursion in closed form;
3. derive the optimal belief distortion and the wedge it implies;
4. solve for the value-function slope, first with a fixed $\theta$ and then
   with a state-dependent $\theta_t$.

Steps 1--3 treat the current value of $\theta_t$ as given; they go through
unchanged whatever that value is.

With linear dynamics, Gaussian shocks, and linear utility, the continuation
value is affine in the state, so we guess $v_t = v_x x_t + v_q$ and verify
the guess by substitution.

(The affine guess with constant coefficients is exact when $\theta$ is
constant; when $\theta_t$ varies over time, it is the first-order
approximation of {cite:t}`bhandari2025survey`, whose perturbation method the
appendix describes.)

The slope $v_x = \partial v_t / \partial x_t$ measures how much the agent
values an extra unit of the state.

We solve it from the recursion in Step 4.

Since $-\theta_t v_{t+1} = -\theta_t(v_x \rho_x x_t + v_q) - \theta_t v_x
\sigma_x w_{t+1}$ is linear in the standard normal shock, the moment
generating function $E[\exp(a w)] = \exp(a^2/2)$ evaluates the expectation in
closed form:

$$
-\frac{1}{\theta_t} \log E_t\!\left[\exp(-\theta_t v_{t+1})\right]
= v_x \rho_x x_t + v_q - \frac{\theta_t}{2}\, v_x^2 \sigma_x^2.
$$

Pessimism subtracts half of $\theta_t$ times the conditional variance of
continuation values from the ordinary expectation, so it acts like an
endogenous discount on risky continuation utilities.


The same linearity pins down the optimal distortion.

Writing $\alpha_t = -\theta_t v_x \sigma_x$, the likelihood ratio becomes

$$
m_{t+1}^* = \frac{\exp(\alpha_t w_{t+1})}
                 {E_t[\exp(\alpha_t w_{t+1})]}
= \exp\!\left(\alpha_t w_{t+1} - \tfrac{1}{2} \alpha_t^2\right),
$$

which is exactly the density of $N(\alpha_t, 1)$ divided by the density of
$N(0, 1)$.

Pessimism is therefore a **mean shift**: under the subjective measure,

$$
w_{t+1} \;\sim\; N(\nu_t, 1),
\qquad
\nu_t = -\theta_t v_x \sigma_x.
$$

The drift $\nu_t$ is the negative of the pessimism parameter times the
exposure of the continuation value to the shock.

The agent tilts beliefs exactly in the
direction that hurts most, by an amount the entropy penalty limits.

The belief wedge for the state follows immediately:

$$
\Delta_t^{(1)}(x) = \tilde E_t[x_{t+1}] - E_t[x_{t+1}]
= \sigma_x \nu_t = -\theta_t v_x \sigma_x^2.
$$

Here $\tilde E_t$ denotes expectation under the subjective measure implied by
$m_{t+1}^*$, while $E_t$ denotes expectation under the data-generating measure.

When $v_x > 0$ (high consumption states are good) and $\theta_t > 0$
(pessimism), the wedge is negative, so the agent *underestimates* future
consumption.

For a variable that enters the value function with a negative sign, such as
unemployment, the same pessimism generates a *positive* wedge.

It remains to pin down the slope $v_x$ (Step 4), and here the distinction
between a fixed and a state-dependent pessimism parameter matters.

*Case 1: fixed $\theta$.*

Suppose first that $\theta_t = \theta$ is a constant.

Substituting the closed-form recursion back into the Bellman equation gives

$$
v_x x_t + v_q
= (1 - \beta)\, x_t
+ \beta\left(v_x \rho_x x_t + v_q - \frac{\theta}{2}\, v_x^2 \sigma_x^2\right).
$$

The variance penalty $-\frac{\theta}{2} v_x^2 \sigma_x^2$ does not involve
$x_t$, so matching coefficients on $x_t$ gives the rational-expectations
slope $v_x = u_x / (1 - \beta\rho_x)$ with $u_x = 1 - \beta$, while the
penalty only lowers the constant $v_q$.

With constant pessimism, the agent tilts beliefs, but the marginal value of
the state is unchanged.

*Case 2: state-dependent $\theta_t$.*

Now suppose, as in {cite:t}`bhandari2025survey`, that pessimism moves with
the state,

$$
\theta_t = \bar\theta(\bar x + x_t),
$$

where $\bar x$ is the steady state and $x_t$ the deviation from it; we
normalize $\bar x = 1$, so that the steady-state pessimism level is
$\mu_\theta = \bar\theta \bar x = \bar\theta$.

The variance penalty is now proportional to the state,

$$
-\frac{\beta}{2}\,\bar\theta(\bar x + x_t)\, v_x^2 \sigma_x^2,
$$

so it contributes to the coefficient on $x_t$, and matching coefficients
yields the **Riccati equation**

$$
v_x = u_x + \beta \rho_x v_x - \frac{\beta}{2}\,\mu_\theta\, \sigma_x^2 v_x^2,
\qquad u_x = 1 - \beta.
$$

The quadratic term is the price of state-dependent pessimism: it lowers the
marginal value of the state relative to the rational-expectations value
$v_x^{RE} = u_x / (1 - \beta\rho_x)$.

If $\theta_t$ were fixed, the same variance penalty would affect only the
constant term, as in Case 1; the Riccati term in the slope exists precisely
because pessimism varies with the state.

We now turn this illustration into code, building it up from small pieces.

Because the quantitative model uses the state-dependent specification, the
code implements Case 2.

The first ingredient is the slope $v_x$ of the continuation value.

It solves the scalar Riccati equation, which we write as a quadratic
$a v_x^2 + b v_x + c = 0$ and solve with the quadratic formula.

We keep the root that collapses to the rational-expectations value
$v_x^{RE} = u_x / (1 - \beta\rho_x)$ as the pessimism parameter $\mu_\theta \to 0$.

```{code-cell} ipython3
def solve_vx(β, ρ_x, σ_x, μ_θ):
    """
    Solve the scalar Riccati equation for the value-function slope vx:

        vx = u_x - (β/2) μ_θ σ_x**2 vx**2 + β ρ_x vx,   with u_x = 1 - β.
    """
    u_x = 1.0 - β                       # marginal utility of log consumption
    vx_re = u_x / (1.0 - β * ρ_x)       # rational-expectations (θ = 0) value

    # Coefficients of a vx**2 + b vx + c = 0
    a = 0.5 * β * σ_x**2 * μ_θ
    b = 1.0 - β * ρ_x
    c = -u_x

    if abs(a) < 1e-14:                  # no pessimism: equation is linear
        return vx_re

    disc = b**2 - 4.0 * a * c
    if disc < 0:                        # no real root: fall back to RE
        return vx_re

    # Keep the root closest to the rational-expectations value
    r1 = (-b + np.sqrt(disc)) / (2.0 * a)
    r2 = (-b - np.sqrt(disc)) / (2.0 * a)
    return r1 if abs(r1 - vx_re) < abs(r2 - vx_re) else r2
```

We store the primitives in a `NamedTuple`, together with the solved slope
$v_x$, and use `create_belief_model` to build an instance.

```{code-cell} ipython3
class BeliefModel(NamedTuple):
    β: float      # discount factor
    ρ_x: float    # persistence of log consumption
    σ_x: float    # volatility of the consumption innovation
    μ_θ: float    # mean of the belief-shock parameter θ
    ρ_θ: float    # AR(1) persistence of θ
    σ_θ: float    # volatility of the θ innovation
    vx: float     # slope of the linearized continuation value


def create_belief_model(β=0.994, ρ_x=0.85, σ_x=0.005,
                        μ_θ=5.64, ρ_θ=0.714, σ_θ=4.3):
    """Build a belief model, solving the Riccati equation for vx."""
    vx = solve_vx(β, ρ_x, σ_x, μ_θ)
    return BeliefModel(β=β, ρ_x=ρ_x, σ_x=σ_x,
                       μ_θ=μ_θ, ρ_θ=ρ_θ, σ_θ=σ_θ, vx=vx)
```

Two functions map a value of $\theta_t$ into the implied distortion.

The drift $\nu_t = -\theta_t v_x \sigma_x$ is the mean shift of the shock under
the subjective measure; the wedge $\Delta_t^{(1)}(x) = \sigma_x \nu_t$ is the
resulting forecast bias for the state.

```{code-cell} ipython3
def belief_drift(model, θ):
    """Mean shift of the shock under subjective beliefs: ν = -θ vx σ_x."""
    return -θ * model.vx * model.σ_x


def belief_wedge(model, θ):
    """One-period belief wedge for the state: Δ = σ_x ν = -θ vx σ_x**2."""
    return model.σ_x * belief_drift(model, θ)
```

A last helper simulates the AR(1) belief shock $\theta_t$.

This is a third specification of pessimism, distinct from the fixed $\theta$
of Case 1 and the state-dependent $\theta_t$ of Case 2: the quantitative
model treats $\theta_t$ as an exogenous AR(1) process calibrated to the
survey wedges, and the appendix shows how it fits into the perturbation
solution.

```{code-cell} ipython3
def simulate_θ(model, T=200, seed=42):
    """Simulate the AR(1) belief shock θ_t."""
    rng = np.random.default_rng(seed)
    θ = np.zeros(T)
    θ[0] = model.μ_θ
    for t in range(1, T):
        θ[t] = ((1 - model.ρ_θ) * model.μ_θ
                + model.ρ_θ * θ[t-1]
                + model.σ_θ * rng.standard_normal())
    return θ
```

Building the model at the baseline calibration, we compare the robust slope
$v_x$ with its rational-expectations counterpart, and report the mean belief
drift and wedge.

```{code-cell} ipython3
model = create_belief_model()

vx_re = (1 - model.β) / (1 - model.β * model.ρ_x)
print(f"RE value of v_x:      {vx_re:.8f}")
print(f"Robust value of v_x:  {model.vx:.8f}")
print(f"Belief drift at θ_bar:  ν = {belief_drift(model, model.μ_θ):.5f} "
      "(standard deviations of w)")
print(f"Belief wedge at θ_bar:  Δ = {belief_wedge(model, model.μ_θ) * 100:.5f} "
      "(% of consumption)")
```

Both the drift and the wedge are tiny at this calibration: log consumption
has a small innovation standard deviation, so the exposure $v_x \sigma_x$ of
the continuation value to the shock is small, and the entropy penalty allows
only a small tilt.

In the full model, the corresponding exposures of continuation values to
shocks are much larger, and they generate the percentage-point wedges seen in
the surveys.

The next figure illustrates the tilt.

Because the true drift $\nu_t = -\theta_t v_x \sigma_x \approx -0.001$ is
invisible on a density plot, the figure instead shifts each curve by the
**scaled drift** $\nu_t / \sigma_x = -\theta_t v_x$, magnifying the true
shift by a factor of $1/\sigma_x = 200$.

The plotted curves are therefore *not* the actual subjective densities of
$w_{t+1}$ at this calibration; they show the direction of the tilt and how it
grows with $\theta_t$, while the printed output reports the true drifts.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: objective and subjective shock distributions (drift scaled for
      visibility)
    name: fig-sbbc-shock-distributions
---
θ_vals = [0, model.μ_θ, 2 * model.μ_θ]
labels = ['θ = 0',
          f'θ = θ_bar = {model.μ_θ:.1f}  (mean)',
          f'θ = 2θ_bar  (pessimistic)']
colors = ['black', 'steelblue', 'firebrick']

# True drift ν = -θ vx σ_x, and the version scaled by 1/σ_x
# that the figure plots so that the shift is visible
ν_true = [belief_drift(model, θ) for θ in θ_vals]
ν_scaled = [ν / model.σ_x for ν in ν_true]

x_grid = np.linspace(-4, 4, 500)

fig, ax = plt.subplots()
for μ, label, color in zip(ν_scaled, labels, colors):
    ax.plot(x_grid, norm.pdf(x_grid, loc=μ),
            label=label, color=color, linewidth=2)

ax.axvline(0, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel(
    'scaled innovation shift: '
    '$\\nu_t / \\sigma_x = -\\theta_t v_x$'
)
ax.set_ylabel('density')
ax.legend()
plt.tight_layout()
plt.show()

print("True and scaled subjective drifts:")
for ν, ν_s, label in zip(ν_true, ν_scaled, labels):
    print(f"  {label:35s}  ν = {ν:9.5f}   ν/σ_x = {ν_s:.4f}")
```

The figure shows how pessimism (higher $\theta_t$) shifts the perceived
distribution of future shocks to the left.

The black curve is the objective distribution, centered at zero.

The blue and red curves are subjective distributions for progressively larger
values of $\theta_t$, with the mean shift drawn at the scaled drift
$\nu_t / \sigma_x$ rather than at the (tiny) true drift $\nu_t$.

An agent with $\theta_t > 0$
believes bad shocks are more likely than they actually are.

### Subjective dynamics

The mean shift changes the law of motion that agents perceive.

Substituting $w_{t+1} = \nu_t + \tilde w_{t+1}$, where
$\tilde w_{t+1} \sim N(0, 1)$ under the subjective measure, into the dynamics
of $x_t$ gives

$$
x_{t+1} = -\theta_t v_x \sigma_x^2 + \rho_x x_t + \sigma_x \tilde w_{t+1}.
$$

With the state-dependent specification of Case 2,
$\theta_t = \bar\theta(\bar x + x_t)$, collecting terms shows that subjective
beliefs change both the intercept and the slope of the perceived dynamics:

$$
\tilde\rho_x = \rho_x - \bar\theta\, v_x \sigma_x^2,
\qquad
\tilde\psi_q = -\bar\theta \bar{x}\, v_x \sigma_x^2.
$$

For a good state ($v_x > 0$), pessimism adds a negative drift.

For a bad state such as unemployment ($v_x < 0$), the same formula raises the
subjective persistence, so pessimists believe bad times last longer.

The code below compares objective and subjective forecast paths of the
consumption state, starting from the steady state, holding the pessimism
level fixed along the forecast path.

```{code-cell} ipython3
def forecast_paths(model, θ, x0=0.0, τ_max=20):
    """Objective and subjective forecast paths of the state x."""
    ν = belief_drift(model, θ)         # subjective mean of the shock
    obj = np.empty(τ_max + 1)
    subj = np.empty(τ_max + 1)
    obj[0] = subj[0] = x0
    for τ in range(τ_max):
        obj[τ+1] = model.ρ_x * obj[τ]
        subj[τ+1] = model.ρ_x * subj[τ] + model.σ_x * ν
    return obj, subj
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: objective and subjective forecasts of consumption
    name: fig-sbbc-subjective-forecasts
---
τ_max = 20
horizons = np.arange(τ_max + 1)

θ_levels = [model.μ_θ, 2 * model.μ_θ]
labels_f = [f'subjective, θ = θ_bar', f'subjective, θ = 2θ_bar']
colors_f = ['steelblue', 'firebrick']

fig, ax = plt.subplots()
obj, _ = forecast_paths(model, 0.0, τ_max=τ_max)
ax.plot(horizons, obj * 100, color='black', linewidth=2,
        label='objective forecast')
for θ, lab, c in zip(θ_levels, labels_f, colors_f):
    _, subj = forecast_paths(model, θ, τ_max=τ_max)
    ax.plot(horizons, subj * 100, color=c, linewidth=2,
            linestyle='--', label=lab)

ax.axhline(0, color='grey', linewidth=0.7, linestyle=':')
ax.set_xlabel('forecast horizon $\\tau$ (quarters)')
ax.set_ylabel('forecast of $x_{t+\\tau}$ (%)')
ax.legend()
plt.tight_layout()
plt.show()
```

Starting at the steady state, the objective forecast of consumption is flat
at zero.

The subjective forecasts drift persistently downward, and twice as fast when
pessimism is twice as high, because the constant drift
$\sigma_x \nu_t$ accumulates at the persistence $\rho_x$.

On average the feared bad times never arrive, so subjective forecasts are
systematically wrong, and that systematic error is exactly the belief wedge
measured in the surveys.

This figure is the scalar version of a key picture in
{cite:t}`bhandari2025survey`: after a pessimism shock in the structural
model, households expect consumption to fall further and recover far more
slowly than it actually does.

## Linear approximation with belief distortions

### The perturbation method

For quantitative analysis, {cite:t}`bhandari2025survey` extend the standard
first-order perturbation method to accommodate time-varying belief distortions.

Let the state vector be $x_t \in \mathbb{R}^n$ with **objective** law of
motion

$$
x_{t+1} = \psi_q + \psi_x x_t + \psi_w w_{t+1}, \qquad
w_{t+1} \sim N(0, I_k).
$$

To first order, the belief factor $\theta_t = \bar\theta x_t$ equals
$\bar\theta(\bar{x} + x_{1t})$.

Under the optimal belief distortion the shocks are re-centered:

$$
w_{t+1} \;\sim\; N\!\left(- \theta_t (v_x \psi_w)',\; I_k\right),
$$

where $v_x$ is the row vector of first derivatives of the continuation value
and $\bar{x}$ is the non-stochastic steady state.

In a standard first-order perturbation, belief distortions would vanish: as
shock volatility shrinks, the entropy penalty makes the distortion second
order.

{cite:t}`bhandari2025survey` avoid this by scaling $\theta_t$ jointly with the
shock volatility, which keeps the subjective model distinct from the
data-generating process in the linear solution; the appendix gives details.

The wedge formula comes directly from comparing the one-step-ahead objective
and subjective conditional means.

Under the objective measure, $E_t[w_{t+1}] = 0$, so

$$
E_t[x_{t+1}] = \psi_q + \psi_x x_t.
$$

Under the subjective measure, the shock mean is
$\tilde E_t[w_{t+1}] = -\theta_t (v_x \psi_w)'$, so

$$
\tilde E_t[x_{t+1}]
= \psi_q + \psi_x x_t
  - \theta_t \psi_w (v_x \psi_w)'.
$$

For any linear variable $z_t = \bar{z}' x_t$, the one-period belief wedge is
therefore

$$
\Delta_t^{(1)}(z)
\;=\; \tilde E_t[z_{t+1}] - E_t[z_{t+1}]
\;=\; -\theta_t\, \bar{z}' (\psi_w \psi_w') v_x'.
$$

Because the drift moves with $\theta_t$, subjective beliefs change both the
conditional mean and the persistence of the state: adverse states are more
persistent under the subjective measure than under the data-generating
measure.

### Riccati equation for $v_x$

The key object is $v_x$, which solves

$$
v_x
\;=\; u_x
  - \frac{\beta}{2}\, v_x \psi_w \psi_w' v_x' \bar\theta
  + \beta\, v_x \psi_x.
$$

This is a modified Riccati equation: the middle term arises from the entropy
penalty on beliefs and vanishes under rational expectations ($\bar\theta = 0$).

To see why the extra term has this form, focus on the continuation value.
Locally, write it as linear in next period's state,

$$

v_{t+1} \approx v_x x_{t+1}.

$$

Since $x_{t+1} = \psi_q + \psi_x x_t + \psi_w w_{t+1}$, the risky part of
continuation value is $v_x \psi_w w_{t+1}$, with conditional variance

$$

v_x \psi_w \psi_w' v_x'.

$$

Robust preferences replace the ordinary expected continuation value with an
entropy-adjusted one. For a Gaussian linear payoff, the minimization over
distorted beliefs gives

$$

E_t[v_{t+1}]
- \frac{\theta_t}{2}\,\operatorname{Var}_t(v_{t+1}).

$$

Thus the agent subtracts a variance penalty from continuation value. Because
$\theta_t = \bar\theta x_t$, matching the coefficient on $x_t$ in the Bellman
equation adds the term

$$

- \frac{\beta}{2}\,
  \left(v_x \psi_w \psi_w' v_x'\right)\bar\theta.

$$

The term is quadratic in $v_x$ because the variance of the continuation value
depends on the square of its exposure to shocks, $v_x \psi_w$.

### One-factor structure

An important consequence of the formula for $\Delta_t^{(1)}(z)$ is that the
*time variation* in all belief wedges is driven by the **single scalar** belief
factor $\theta_t \approx \bar\theta(\bar{x} + x_{1t})$.

The cross-sectional loadings $-\bar{z}'(\psi_w\psi_w')v_x'$ are
fixed by the model's structural parameters.

The loadings are not free parameters: they equal covariances of shocks with
the continuation value, objects that the equilibrium of the model determines.

The only free parameters describing beliefs are the three governing the
$\theta_t$ process, so every additional surveyed variable adds an
overidentifying restriction on the model.

This theoretical prediction matches the empirical finding that one principal
component explains about four-fifths of the joint variation in the
unemployment and inflation wedges.

The code below illustrates the one-factor structure, but with a shortcut: in
place of the model-implied loadings $-\bar{z}'(\psi_w\psi_w')v_x'$, it uses
the empirical loadings $c_u$ and $c_\pi$ defined earlier, so the lines pass
through the empirical mean wedges at $\theta = \mu_\theta$.

In the full model the loadings are endogenous, and the paper's structural
benchmark implies mean wedges of 0.55 and 0.90 rather than the data values
0.52 and 1.22.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: wedge loadings implied by one factor
    name: fig-sbbc-one-factor-loadings
---
θ_grid = np.linspace(0, 20, 200)

# Empirical loadings
loading_u = c_u   # 0.52 / 5.64 pp per unit of θ (unemployment)
loading_π = c_π  # 1.22 / 5.64 pp per unit of θ (inflation)

wedge_u_grid = loading_u * θ_grid
wedge_π_grid = loading_π * θ_grid

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(θ_grid, wedge_u_grid, color='steelblue', linewidth=2,
             label='$\\Delta^{(1)}(u)$')
axes[0].plot(θ_grid, wedge_π_grid, color='darkorange', linewidth=2,
             label='$\\Delta^{(1)}(\\pi)$')
axes[0].axvline(μ_θ, color='grey', linestyle='--', linewidth=0.9,
                label=f'$\\bar{{\\theta}} = {μ_θ}$')
axes[0].set_xlabel('belief-shock level $\\theta$')
axes[0].set_ylabel('belief wedge (pp)')
axes[0].legend()

θ_sim = simulate_θ(model, T=400, seed=7)
wu_sim = loading_u * θ_sim
w_π_sim = loading_π * θ_sim
axes[1].scatter(wu_sim, w_π_sim, c=θ_sim, cmap='Blues', alpha=0.6, s=12)
axes[1].set_xlabel('unemployment wedge (pp)')
axes[1].set_ylabel('inflation wedge (pp)')

plt.tight_layout()
plt.show()
```

The left panel plots the two one-period wedge formulas as functions of the
belief shock.

Both lines slope upward, but the inflation line is steeper because the
calibration assigns inflation a larger loading on $\theta_t$.

The vertical dashed line marks the average value $\bar{\theta}$, where the
lines match the empirical mean wedges of 0.52 and 1.22 percentage points.

The right panel simulates $\theta_t$ and plots the resulting unemployment and
inflation wedges against each other.

Since both are driven by the same scalar state, the simulated points trace out
an almost exact positive relation.

In the data the relation is looser because survey responses contain
measurement error; a hidden-factor model that allows for such error recovers a
belief factor whose path is close to the first principal component.

## A reduced-form emulator of the New Keynesian model

We now use the empirical belief factor to generate impulse responses.

The full model in {cite:t}`bhandari2025survey` is a New Keynesian model with
households, Calvo price setting, search-and-matching labor frictions, Nash
wage bargaining, TFP shocks, monetary-policy shocks, and the belief shock,
all calibrated inside the full equilibrium system.

Beliefs matter there because consumption decisions, vacancy posting, wage
bargaining, and price setting are forward-looking.

Rather than solve that system here, we use a small linear emulator that keeps
only the pieces needed for transparent impulse responses:

$$

s_{t+1} = A\, s_t + B\, \epsilon_{t+1},

$$

where $s_t = (u_t, \pi_t, y_t, \theta_t, a_t)'$ collects unemployment,
inflation, output, the belief shock, and TFP, and
$\epsilon_{t+1} \sim N(0, I_3)$ contains the three structural shocks.

The matrices below are chosen to reproduce selected signs and moments from
the paper; they are not obtained by solving the structural equilibrium
conditions.

The belief shock follows the persistence estimated from the survey wedges,
$\rho_\theta = 0.714$.

The two wedge loadings are chosen so that $c_u \mu_\theta = 0.52$ and
$c_\pi \mu_\theta = 1.22$ at $\mu_\theta = 5.64$.

The entries that connect $\theta_t$ to $u_t$, $\pi_t$, and $y_t$ should be
read as reduced-form summaries of the full subjective-expectations channel,
calibrated to give the right signs and a reasonable scale for the
belief-shock IRF: higher pessimism raises unemployment, temporarily raises
inflation, and lowers output.

One difference from the structural model deserves emphasis.

Unlike the full model, this reduced-form system makes $\theta_t$ move the
wedges and the macroeconomic variables directly.

In the structural model, $\theta_t$ matters through distorted probabilities
over payoff-relevant shocks, so the presence and propagation of fundamental
shocks are part of the mechanism.

We index the five state variables with named constants, so that later code can
refer to, say, the belief shock as `I_THETA` rather than a bare number.

```{code-cell} ipython3
# Position of each variable in the state vector s_t
I_U, I_PI, I_Y, I_THETA, I_A = 0, 1, 2, 3, 4
```

The object below stores the transition matrix, shock loadings, and the two
wedge loadings. That is enough to compute the impulse responses.

```{code-cell} ipython3
class NKModel(NamedTuple):
    A: np.ndarray    # state transition matrix
    B: np.ndarray    # shock loadings (columns: w_θ, w_a, w_r)
    c_u: float       # loading of the unemployment wedge on θ
    c_π: float       # loading of the inflation wedge on θ


def create_nk_model():
    """Build the reduced-form NK emulator (state and shock matrices)."""
    # Exogenous-process parameters from bhandari2025survey.
    ρ_θ, σ_θ = 0.714, 4.3
    ρ_a, σ_a = 0.840, 0.00568

    # Belief-wedge loadings on θ (match the mean empirical wedges)
    c_u = 0.52 / 5.64
    c_π = 1.22 / 5.64

    # Impact of the belief shock θ on the endogenous variables (per unit of θ)
    φ_u_θ = 0.00648 / σ_θ
    φ_π_θ = 0.00063 / σ_θ
    φ_y_θ = -0.00807 / σ_θ

    # Impact of TFP on the endogenous variables
    φ_u_a, φ_π_a, φ_y_a = -0.362, -0.1306, 1.0236

    # Endogenous persistence (quarterly)
    ρ_u, ρ_π, ρ_y = 0.35, 0.50, 0.35

    A = np.array([
        [ρ_u, 0,   0,   φ_u_θ, φ_u_a],   # unemployment
        [0,   ρ_π, 0,   φ_π_θ, φ_π_a],   # inflation
        [0,   0,   ρ_y, φ_y_θ, φ_y_a],   # output
        [0,   0,   0,   ρ_θ,   0    ],   # belief shock
        [0,   0,   0,   0,     ρ_a  ],   # TFP
    ])

    # Columns: [w_θ, w_a, w_r]
    B = np.array([
        [0,   0,    0.5e-3],   # MP -> unemployment
        [0,   0,   -0.1e-3],   # MP -> inflation
        [0,   0,   -0.5e-3],   # MP -> output
        [σ_θ, 0,    0     ],   # θ innovation
        [0,   σ_a,  0     ],   # TFP innovation
    ])
    return NKModel(A=A, B=B, c_u=c_u, c_π=c_π)
```

Impulse responses are computed by iterating $s_{t+1} = A s_t$ from the impact
column of $B$, and the two belief wedges are read off as $c_u \theta_t$ and
$c_\pi \theta_t$.

```{code-cell} ipython3
def irf(model, shock_idx, T=25):
    """
    Impulse responses to a one-standard-deviation shock.

    shock_idx : 0 = belief shock, 1 = TFP shock, 2 = monetary policy shock.

    Returns the state responses together with the unemployment and inflation
    wedge responses.
    """
    A, B = model.A, model.B
    resp = np.zeros((A.shape[0], T))
    s = B[:, shock_idx].copy()          # impact response
    for t in range(T):
        resp[:, t] = s
        s = A @ s

    wu = model.c_u * resp[I_THETA, :]
    w_π = model.c_π * resp[I_THETA, :]
    return resp, wu, w_π
```

```{code-cell} ipython3
nk = create_nk_model()
```

## Quantitative results

### Impulse responses to the belief shock

A positive innovation to $\theta_t$ makes households more pessimistic.

In the full structural model, higher pessimism makes households and firms act
as if bad future states are more likely. Vacancy posting weakens, output
falls, unemployment rises, and the two survey wedges jump together.

The reduced-form system below is calibrated to reproduce those signs and to
make the belief wedges decay with $\rho_\theta = 0.714$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: impulse responses to a belief shock
    name: fig-sbbc-belief-shock-irfs
---
T_irf = 25
periods = np.arange(T_irf)

resp_θ, wu_θ, w_π_θ = irf(nk, shock_idx=0, T=T_irf)

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

ylabels = ['unemployment (pp)', 'inflation (pp, ann.)', 'output (%)',
           'belief shock θ', 'unemployment wedge Δ(u) (pp)',
           'inflation wedge Δ(π) (pp)']
series = [resp_θ[0] * 100,   # unemployment in pp  (fraction * 100)
          resp_θ[1] * 400,   # inflation ann. pp  (quarterly frac * 400)
          resp_θ[2] * 100,   # output in %        (fraction * 100)
          resp_θ[3],         # belief shock θ
          wu_θ,              # unemp. wedge (pp): c_u * θ, already in pp
          w_π_θ]             # infl. wedge  (pp): c_π * θ, already in pp
colors = ['steelblue'] * 3 + ['purple', 'steelblue', 'darkorange']

for ax, ylabel, y, color in zip(axes, ylabels, series, colors):
    ax.plot(periods, y, color=color, linewidth=2)
    ax.axhline(0, color='grey', linewidth=0.7, linestyle='--')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('quarters')

plt.tight_layout()
plt.show()
```

The first row shows the macroeconomic responses, and the second row shows the
belief shock and the two implied survey wedges.

The shock raises unemployment, lowers output, and generates comoving
unemployment and inflation wedges. Inflation rises temporarily in this
calibration, then decays back toward zero.

### The unemployment volatility puzzle

A long-standing challenge for New Keynesian models is that standard TFP and
monetary policy shocks generate far too little unemployment volatility
({cite}`Shimer2005`).

In the paper's no-belief-shock economy, TFP and monetary policy shocks
produce unemployment volatility of only 0.55, compared to 1.70 in the data.

Adding the belief shock substantially closes the gap.

The emulator is calibrated to reproduce this experiment: we compute its
unconditional standard deviations from the discrete Lyapunov equation, with
and without the belief shock.

```{code-cell} ipython3
def simulate_nk(model, T=200, seed=42):
    """Simulate the model for T periods under the data-generating measure."""
    rng = np.random.default_rng(seed)
    A, B = model.A, model.B
    k = B.shape[1]
    s = np.zeros((A.shape[0], T))
    for t in range(1, T):
        s[:, t] = A @ s[:, t-1] + B @ rng.standard_normal(k)
    return s


def unconditional_stds(model, include_θ_shock=True):
    """Unconditional standard deviations from the discrete Lyapunov equation."""
    B_use = model.B.copy()
    if not include_θ_shock:
        B_use[:, 0] = 0.0               # shut down the belief shock
    Σ = solve_discrete_lyapunov(model.A, B_use @ B_use.T)
    return np.sqrt(np.diag(Σ))
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: model and data volatility comparison
    name: fig-sbbc-volatility-comparison
---
std_full = unconditional_stds(nk, include_θ_shock=True)
std_no_θ = unconditional_stds(nk, include_θ_shock=False)

labels_vol = ['Unemployment', 'Inflation', 'Output']
idx = [I_U, I_PI, I_Y]
scale = [100, 400, 100]    # convert to pp (unemployment, annualized inflation, %)

std_full_scaled = [std_full[i] * scale[j] for j, i in enumerate(idx)]
std_no_θ_scaled = [std_no_θ[i] * scale[j] for j, i in enumerate(idx)]

# Data standard deviations reported by bhandari2025survey.
data_std = [1.70, 1.37, 2.00]    # unemployment, inflation, output

x = np.arange(len(labels_vol))
width = 0.25

fig, ax = plt.subplots()
ax.bar(x - width, std_no_θ_scaled, width, label='No $\\theta_t$',
       color='steelblue', alpha=0.7)
ax.bar(x, std_full_scaled, width, label='Benchmark',
       color='firebrick', alpha=0.7)
ax.bar(x + width, data_std, width, label='Data',
       color='grey', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(labels_vol)
ax.set_ylabel('standard deviation (% or pp, ann.)')
ax.legend()
plt.tight_layout()
plt.show()
```

The bar chart compares three standard deviations: the emulator without the
belief shock, the emulator with it (labeled "Benchmark" after the paper's
benchmark economy), and the data.

The main message is visible in the unemployment bars.

Without the belief shock, unemployment volatility is far below its empirical
counterpart.

Adding the calibrated belief shock raises unemployment volatility from about
0.55 to about 1.39, moving the model much closer to the data value 1.70.

The belief shock also improves the model's fit to the historical record.

{cite:t}`bhandari2025survey` feed measured TFP innovations and the
belief-shock innovations extracted from the wedge data into the model and
compare the implied paths with the data.

The correlations between model-implied and actual paths are 0.51 for
unemployment, 0.83 for the unemployment wedge, and 0.79 for the inflation
wedge.

Shutting down fluctuations in $\theta_t$ makes the model overstate
unemployment during the late-1990s boom and understate it around the Great
Recession.

Through the lens of the model, the late 1990s were a period of relative
optimism, and much of the 2008--09 rise in unemployment reflected an increase
in pessimism.

One limitation of the benchmark model is its inflation cyclicality.

Inflation is nearly acyclical in the data (correlation with output of 0.10)
but countercyclical in the model (−0.82).

The missing ingredients are wage and price markup shocks, which account for
much of unconditional inflation variation in richer DSGE models but have
little explanatory power for output and unemployment, so adding them would
leave the belief-wedge mechanism intact.

### Impulse responses to TFP and monetary policy shocks

For completeness, we also show responses to the other two shocks.

The figure below has TFP responses in the top row and monetary-policy responses
in the bottom row.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: impulse responses to TFP and monetary policy shocks
    name: fig-sbbc-tfp-mp-irfs
---
resp_a,  _, _ = irf(nk, shock_idx=1, T=T_irf)   # TFP shock
resp_r,  _, _ = irf(nk, shock_idx=2, T=T_irf)   # Monetary policy shock

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

series_a = [resp_a[0]*100, resp_a[1]*400, resp_a[2]*100]
series_r = [resp_r[0]*100, resp_r[1]*400, resp_r[2]*100]
var_ylabels = ['unemployment (pp)', 'inflation (pp, ann.)', 'output (%)']

for j, (ylabel, ya, yr) in enumerate(zip(var_ylabels, series_a, series_r)):
    axes[j].plot(periods, ya, color='steelblue', linewidth=2,
                 label='TFP shock')
    axes[j].axhline(0, color='grey', linewidth=0.7, linestyle='--')
    axes[j].set_ylabel(ylabel)
    axes[j].set_xlabel('quarters')
    axes[j].legend(loc='best')

    axes[j+3].plot(periods, yr, color='darkorange', linewidth=2,
                   label='MP shock')
    axes[j+3].axhline(0, color='grey', linewidth=0.7, linestyle='--')
    axes[j+3].set_ylabel(ylabel)
    axes[j+3].set_xlabel('quarters')
    axes[j+3].legend(loc='best')

plt.tight_layout()
plt.show()
```

The TFP shock raises output and lowers unemployment.

Inflation also falls in this calibration, reflecting the disinflationary effect
of higher productivity.

All three responses are persistent because TFP itself is persistent.

The monetary-policy shock is much less persistent.

In the plotted reduced-form calibration, a contractionary monetary-policy
innovation raises unemployment, lowers inflation, and lowers output on impact,
but the responses fade quickly because the monetary-policy shock is i.i.d.

Unlike the belief-shock figure, these panels do not include belief wedges:
TFP and monetary-policy shocks do not move $\theta_t$ in this reduced-form
illustration, just as in the paper's benchmark, where $\theta_t$ is an
exogenous AR(1) process orthogonal to the other shocks.

### Role of firms' beliefs

In the benchmark model, **firms** as well as
households hold subjective beliefs.

What changes when firms instead have rational beliefs?

The key channel is through the price-setting equation.

Price-setting firms that share the household's pessimism put extra probability
weight on states with lower productivity and higher marginal costs.

The rational-firms experiment turns off belief distortions in firms'
forward-looking equations while keeping household beliefs subjective and
recalibrating $\theta_t$ so that the mean and volatility of the unemployment
wedge remain comparable.

If firms have rational beliefs, they see the household pessimism shock mainly
as a contraction in demand.

Inflation falls on impact, and the inflation wedge is too small.

Wages also fall by less.

Under Nash bargaining, the wage splits the gap between the firm's subjective
valuation of the match and the worker's subjective value of unemployment; a
rational firm does not mark down its valuation when $\theta_t$ rises, so the
perceived surplus stays larger and the bargained wage declines less.

Firm beliefs therefore strengthen the comovement between the unemployment
wedge and the inflation wedge, which is needed to match the data.

The sign and size of the inflation response to a belief shock are therefore
diagnostic: the survey evidence is hard to match unless firms, not only
households, put extra subjective probability on high-marginal-cost states.

### Two diagnostic variants

Two diagnostic variants show how survey wedges restrict the
model.

**No TFP shocks** --- Setting $\sigma_a = 0$ leaves only demand-type
disturbances: monetary policy and belief shocks, which lower economic activity
and inflation together.

The states households fear then combine high unemployment with *low*
inflation, so pessimism produces a negative average inflation wedge and
negative comovement between the two wedges --- both counterfactual.

This failure persists even after the belief-shock process is recalibrated (to
$\mu_\theta = 150$ and $\sigma_\theta = 117.7$) to keep the unemployment wedge
close to the benchmark: the mean inflation wedge is $-0.32$, its volatility is
only $0.26$, and it becomes procyclical.

**Rational firms** --- If households are pessimistic but firms have rational
beliefs, unemployment still responds strongly to a belief shock.

Inflation, however, falls on impact and the inflation wedge is much too small.

The rational-firms variant keeps the unemployment wedge close to the benchmark,
with mean $0.55$ and volatility $0.45$, but the inflation wedge falls to mean
$0.34$ and volatility $0.29$, compared with $0.90$ and $0.73$ in the
benchmark.

The table below summarizes the two restrictions.

| Moment | Benchmark | No TFP shocks | Rational firms |
|---|---:|---:|---:|
| Mean inflation wedge | 0.90 | −0.32 | 0.34 |
| Mean unemployment wedge | 0.55 | 0.54 | 0.55 |
| Volatility of inflation wedge | 0.73 | 0.26 | 0.29 |
| Volatility of unemployment wedge | 0.45 | 0.43 | 0.45 |
| Volatility of unemployment | 1.39 | 0.87 | 1.24 |
| Corr. inflation wedge, output | −0.67 | 0.50 | −0.53 |

These variants show why the benchmark needs both supply-side uncertainty and
firms' subjective beliefs to match the joint behavior of inflation and
unemployment forecasts.

### Countercyclicality of wedges

A final important prediction is that belief wedges are countercyclical.

Recessions are periods of high $\theta_t$, which raises both the unemployment
wedge and the inflation wedge simultaneously.

The code below simulates a
long run of the model and shows this property:

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: simulated countercyclicality of belief wedges
    name: fig-sbbc-countercyclical-wedges
---
sim = simulate_nk(nk, T=400, seed=99)
θ_sim = sim[I_THETA]
y_sim = sim[I_Y] * 100

wu_sim_series = nk.c_u * θ_sim
w_π_sim_series = nk.c_π * θ_sim

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

axes[0].plot(y_sim, color='steelblue', linewidth=2, label='Output gap (%)')
axes[0].axhline(0, color='grey', linestyle='--', linewidth=0.7)
axes[0].set_ylabel('%')
axes[0].legend(loc='upper right')

axes[1].plot(wu_sim_series, color='firebrick', linewidth=2,
             label='Unemployment belief wedge (pp)')
axes[1].set_ylabel('pp')
axes[1].legend(loc='upper right')

axes[2].plot(w_π_sim_series, color='darkorange', linewidth=2,
             label='Inflation belief wedge (pp)')
axes[2].set_ylabel('pp')
axes[2].legend(loc='upper right')
axes[2].set_xlabel('quarter')

plt.tight_layout()
plt.show()

corr_u = np.corrcoef(y_sim, wu_sim_series)[0, 1]
corr_π = np.corrcoef(y_sim, w_π_sim_series)[0, 1]
print(f"Corr(output gap, unemployment wedge) = {corr_u:.3f}  "
      f"(data: -0.49)")
print(f"Corr(output gap, inflation wedge)    = {corr_π:.3f}  "
      f"(data: -0.30)")
```

The top panel plots the simulated output gap, while the middle and bottom
panels plot the unemployment and inflation wedges generated by the same
simulation.

Periods with weak output tend to coincide with elevated wedges.

The simulated correlations are negative, confirming the countercyclicality
predicted by the model and documented in the survey data.

### Further empirical checks

Reduced-form evidence provides additional support for the mechanism.

In local projections of macroeconomic variables and survey forecasts on
innovations to the first principal component of the belief wedges, a positive
innovation predicts higher unemployment, higher belief wedges, and an
inflation response that is briefly positive before turning mildly negative.

A second check comes from forecast-error regressions of the form

$$

z_{t+j} - \tilde E_t[z_{t+j}]
= b_0 + b_z z_t + b_f \tilde E_{t-1}[z_{t+j-1}] + \varepsilon_{t+j},

$$

where $z_t$ is inflation or unemployment.

Under full-information rational expectations these errors should be mean zero
and unforecastable: $b_0 = b_z = b_f = 0$.

In the data, $b_0 < 0$ for both variables, restating the average upward biases.

The slopes reveal two different updating patterns: inflation forecast errors
are predicted by the lagged forecast ($b_f = -0.50$), the signature of
sluggish updating, while unemployment forecast errors are predicted by the
current unemployment rate ($b_z = -0.40$), the signature of overreaction.

The calibrated model reproduces the signs, magnitudes, and $R^2$ of all three
patterns.

Models of information frictions and models of overreaction imply $b_0 = 0$,
and each matches only one of the two slope patterns; the belief-wedge model
matches everything because the household pessimistically distorts the *joint*
distribution of inflation and unemployment.

A final check compares the belief factor with familiar sentiment measures.

The first principal component of the wedges has correlations of $-0.65$ with
the Michigan Consumer Sentiment Index and $-0.72$ with the Conference Board
Consumer Confidence Index, so the belief wedges capture what sentiment indices
capture --- with the advantage that wedges quantify the forecast bias in
percentage points, which is what makes the model calibration possible.

In contrast, the wedges are largely uncorrelated with dispersion in SPF
forecasts, so time-varying pessimism is distinct from forecaster disagreement.

## Extensions

Several extensions of the benchmark model are worth noting:

**Heterogeneous beliefs** --- The solution method allows the belief distortion
to be switched on or off equation by equation, so different agents can hold
different subjective beliefs.

The rational-firms variant above is one example, and the relative sizes of the
unemployment and inflation wedges identify whose beliefs are distorted.

With incomplete markets, heterogeneous exposures of continuation values to
shocks would generate belief heterogeneity across households endogenously,
with implications for saving, portfolio choice, and the design of social
insurance.

**Pessimism induced by TFP** --- The benchmark treats $\theta_t$ as an
exogenous AR(1) process.

Another specification makes negative TFP shocks raise pessimism.

This variant matches many unconditional moments: the inflation wedge mean is
$0.85$, the unemployment wedge mean is $0.56$, and unemployment volatility is
$1.49$.

Its weakness is dynamic: responses to TFP shocks become counterfactually large
relative to the VAR evidence, and the correlations of model-implied paths with
the data fall to 0.22 (unemployment), 0.20 (unemployment wedge), and 0.35
(inflation wedge), compared with 0.51, 0.83, and 0.79 in the benchmark.

{cite:t}`bhandari2025survey` read this as evidence that quantitatively
important movements in pessimism are orthogonal to productivity.

**Wage rigidity** --- Wage rigidity is important for amplification.

With flexible wages ($\chi_w = 0$), bargained wages absorb shocks, firm values
move less, and unemployment volatility falls from $1.39$ to $0.77$ --- the
Shimer-style amplification problem in another form.

Lower macroeconomic volatility feeds back into beliefs: with less to fear, the
covariance between forecasted variables and continuation values shrinks, and
unemployment-wedge volatility falls from $0.45$ to $0.13$.

**Beyond the first-order homoskedastic case** --- The approximation is designed
to keep subjective-belief effects alive in a linear solution.

In richer nonlinear or stochastic-volatility settings, belief wedges could also
move because the dispersion of continuation values changes.

We do not pursue those extensions here.

**Idiosyncratic risk** --- The benchmark model takes fluctuations in $\theta_t$ as
exogenous, but they can also be endogenized.

In a variant where households face uninsurable idiosyncratic risk, a rise in
that risk makes adverse states more likely from each household's viewpoint, so
pessimism and the belief wedges increase without any exogenous shock to
$\theta_t$.

The supporting empirical idea is that belief wedges comove with the
{cite}`Schmidt2016` index of idiosyncratic labor-income skewness, which
proxies for the risk of large losses such as job loss.

## Appendix: the series expansion method

This appendix gives the computational and theoretical details underlying the
linearization presented in the main lecture.

The formulas follow {cite:t}`bhandari2025survey`, but the notation needed for the
calculations below is introduced here.

### Multi-period belief wedges

The main text focused on the one-period belief wedge
$\Delta_t^{(1)}(z)$.

Longer-horizon survey forecasts require $\tau$-period-ahead wedges
$\Delta_t^{(\tau)}(z) = \tilde E_t[z_{t+\tau}] - E_t[z_{t+\tau}]$,
so we now derive their linear representation.

Under linear dynamics

$$

x_{t+1} = \psi_q + \psi_x x_t + \psi_w w_{t+1},
\qquad w_{t+1} \sim N(0, I_k),

$$

the $\tau$-period-ahead expectation of the state deviation under the
data-generating measure is

$$

E_t[x_{1,t+\tau}] = G_x^{(\tau)} x_{1t} + G_0^{(\tau)},
\qquad
G_x^{(\tau)} = \psi_x G_x^{(\tau-1)},
\quad
G_0^{(\tau)} = \psi_x G_0^{(\tau-1)} + \psi_q,

$$

with initial conditions $G_x^{(0)} = I$ and $G_0^{(0)} = 0$.

Under the **subjective** measure, the mean of $w_{t+1}$ is shifted to
$\nu_t = \bar H + HF x_{1t}$.

For the stationary model the relevant identifications are

$$

F = \bar\theta,
\qquad
H = -(v_x \psi_w)',
\qquad
\bar H = -\bar\theta\,\bar x\,(v_x \psi_w)'.

$$

The shift is equivalent to replacing the transition matrices by their
subjective counterparts

$$

\tilde\psi_x = \psi_x + \psi_w H F,
\qquad
\tilde\psi_q = \psi_q + \psi_w \bar H,

$$

so the subjective loadings $\tilde G_x^{(\tau)}$ and $\tilde G_0^{(\tau)}$
satisfy the same recursions with $\tilde\psi_x$ and $\tilde\psi_q$ in place
of $\psi_x$ and $\psi_q$.

The $\tau$-period belief wedge is then

$$

\Delta_t^{(\tau)} = \bigl(\tilde G_x^{(\tau)} - G_x^{(\tau)}\bigr) x_{1t}
  + \tilde G_0^{(\tau)} - G_0^{(\tau)},

$$

which reduces to the one-period wedge formula at $\tau = 1$.

The code below implements these recursions and shows how belief wedges grow
with the forecast horizon.

```{code-cell} ipython3
def compute_tau_wedge_loadings(ψ_x, ψ_w, H, H_bar, F, τ_max=20):
    """
    Compute tau-period belief wedge loadings.

    For simplicity we work with the scalar stationary case (all quantities
    are scalars or 1-d arrays).

    Returns
    -------
    wedge_const : array (tau_max,)   constant term of wedge  (G0_tilde - G0)
    wedge_slope : array (tau_max,)   x1t loading of wedge    (Gx_tilde - Gx)
    """
    n = ψ_x.shape[0]
    ψ_x_tild = ψ_x + ψ_w @ (H @ F)        # subjective transition matrix
    ψ_q_tild = (ψ_w @ H_bar).ravel()      # subjective intercept

    Gx = np.eye(n)
    Gx_tild = np.eye(n)
    G0 = np.zeros(n)
    G0_tild = np.zeros(n)

    wedge_const = np.zeros(τ_max)
    wedge_slope = np.zeros((τ_max, n))

    for τ in range(1, τ_max + 1):
        Gx = ψ_x @ Gx
        Gx_tild = ψ_x_tild @ Gx_tild
        G0 = ψ_x @ G0                     # ψ_q = 0 under the objective measure
        G0_tild = ψ_x_tild @ G0_tild + ψ_q_tild

        wedge_slope[τ - 1] = (Gx_tild - Gx)[0]
        wedge_const[τ - 1] = float((G0_tild - G0)[0])

    return wedge_const, wedge_slope
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: multi-period belief wedge profile
    name: fig-sbbc-horizon-wedge
---
# Scalar model objects
ψ_x_sc = np.array([[model.ρ_x]])
ψ_w_sc = np.array([[model.σ_x]])
F_sc = np.array([[model.μ_θ]])           # θ-bar
H_sc = np.array([[-model.vx * model.σ_x]])  # -(vx ψ_w)'
x_bar_sc = 1.0
H_bar_sc = -model.μ_θ * x_bar_sc * np.array([[model.vx * model.σ_x]])

τ_max = 20
wc, ws = compute_tau_wedge_loadings(ψ_x_sc, ψ_w_sc, H_sc, H_bar_sc, F_sc, τ_max)

# State deviation that raises the belief factor by one
# unconditional standard deviation of the belief shock.
θ_std = model.σ_θ / np.sqrt(1 - model.ρ_θ**2)
x_dev = θ_std / model.μ_θ

fig, ax = plt.subplots()
τ_grid = np.arange(1, τ_max + 1)
ax.plot(τ_grid, wc * 100,
        color='steelblue', linewidth=2, label='Wedge at mean ($x_{1t}=0$)')
ax.plot(τ_grid, (wc + ws[:, 0] * x_dev) * 100,
        color='firebrick', linewidth=2, linestyle='--',
        label='Wedge at $+1\\,\\sigma_\\theta$ deviation')
ax.axhline(0, color='grey', linewidth=0.7, linestyle=':')
ax.set_xlabel('forecast horizon $\\tau$ (quarters)')
ax.set_ylabel('belief wedge (pp)')
ax.legend()
plt.tight_layout()
plt.show()
```

The horizontal axis is the forecast horizon, from one quarter ahead to twenty
quarters ahead.

The blue line is the multi-period wedge evaluated at the mean state, and the
red dashed line evaluates the same wedge after a one-standard-deviation
increase in the belief shock.

The distance from zero grows with the horizon because pessimistic beliefs
affect not only the next shock but also the expected path of future states.

### The series expansion

{cite:t}`bhandari2025survey` solve the full general-equilibrium model
using a **series expansion** (perturbation) method
({cite}`BorovickaHansen2014`).

The key innovation is a **joint
perturbation** of the shock volatility $q$ and the penalty parameter
$\theta_t$.

#### Law of motion

Index the model by a scalar perturbation parameter $\mathsf{q}$ that
scales shock volatility:

$$

x_{t+1}(\mathsf{q}) = \psi\!\left(x_t(\mathsf{q}),\,
  \mathsf{q}\, w_{t+1},\, \mathsf{q}\right).

$$

Expanding around $\mathsf{q} = 0$ gives

$$

x_t(\mathsf{q}) \approx \bar x + \mathsf{q}\, x_{1t}
  + \tfrac{\mathsf{q}^2}{2}\, x_{2t} + \cdots

$$

The first-order dynamics are

$$

x_{1,t+1} = \psi_q + \psi_x x_{1t} + \psi_w w_{t+1}.

$$

#### Continuation value and the Riccati equation

To preserve a nontrivial role for beliefs at first order, the penalty
parameter is **jointly scaled** with $\mathsf{q}$: the effective
penalization in the perturbed recursion is
$\mathsf{q}/[\bar\theta(\bar x + x_{1t})]$,
which shrinks together with shock volatility.

This ensures that the
deterministic steady state does not collapse to the rational-expectations
solution.

Guessing $v_{1t} = v_x x_{1t} + v_q$ and matching coefficients yields
the **Riccati equation for $v_x$**:

$$

v_x = u_x - \frac{\beta}{2}\, v_x \psi_w \psi_w' v_x' \bar\theta
  + \beta\, v_x \psi_x,

$$

and the constant

$$

v_q = u_q - \frac{\beta}{2}\,\bar\theta\, \bar x\,
  v_x \psi_w \psi_w' v_x' + \beta\, v_x \psi_q + \beta v_q.

$$

The Riccati equation is quadratic in $v_x$.

For the stationary scalar case it reduces to

$$

a\, v_x^2 + b\, v_x + c = 0,
\qquad
a = \frac{\beta}{2}\sigma_x^2 \bar\theta,\quad
b = 1 - \beta\rho_x,\quad
c = -u_x.

$$

#### Shock distribution under subjective beliefs

Substituting the first-order expansion into the distortion formula shows that
the leading term $m_{0,t+1}$ is a lognormal change of measure.

With Gaussian shocks, this is equivalent to shifting the innovation mean
as follows:

$$

w_{t+1} \;\sim\;
N\!\left(-\bar\theta(\bar x + x_{1t})(v_x \psi_w)',\; I_k\right).

$$

Belief wedges for the state vector follow immediately:

$$

\Delta_t^{(1)} = \tilde E_t[x_{t+1}] - E_t[x_{t+1}]
= \psi_w\, \tilde E_t[w_{t+1}]
= -\bar\theta(\bar x + x_{1t})(\psi_w \psi_w') v_x'.

$$

#### Equilibrium conditions with subjective beliefs

The full model's equilibrium conditions take the form

$$

0 = E_t\!\left[\mathbb{M}_{t+1}\, g(x_{t+1}, x_t, x_{t-1}, w_{t+1}, w_t)\right],

$$

where $\mathbb{M}_{t+1} = \mathrm{diag}(m_{t+1}^{\sigma_1}, \ldots,
m_{t+1}^{\sigma_n})$ selects which equations involve subjective
expectations ($\sigma_i = 1$) versus objective ones ($\sigma_i = 0$).

First-order expansion of these conditions gives a system in the unknown
policy matrices $\psi_x, \psi_w, \psi_q$:

$$

0 = (g_{x^+}\psi_x + g_x - \mathbb{E})\,\psi_x + g_{x^-}

$$

$$

0 = (g_{x^+}\psi_x + g_x - \mathbb{E})\,\psi_w + g_w

$$

$$

0 = (g_{x^+}\psi_x + g_{x^+} + g_x)\,\psi_q + g_q
  - \mathbb{E}(\bar x + \psi_q),

$$

where the **belief distortion matrix** $\mathbb{E}$ collects the impact
of subjective expectations on each equation:

$$

\mathbb{E} = \operatorname{stack}\Bigl\{
  \sigma_i\, [g_{x^+}\psi_w + g_{w^+}]^i\,
  (v_x \psi_w)'\, \bar\theta
\Bigr\}.

$$

These equations are solved jointly with the Riccati equation for $v_x$.

Compared with the standard Blanchard–Kahn solution,
the only modification is the additive term $-\mathbb{E}$ that shifts the
characteristic matrix; when $\bar\theta = 0$ we recover the standard
rational-expectations solution.

#### The AR(1) belief shock as a special case

Now suppose $\theta_t$ is itself an exogenous AR(1) process:

$$

f_{t+1} = (1 - \rho_f)\bar f + \rho_f f_t + \sigma_f w_{t+1}^f.

$$

Appending $f_t$ to the state vector, the first-order dynamics become

$$

\begin{pmatrix} x_{1,t+1} \\ f_{1,t+1} \end{pmatrix}
= \begin{pmatrix} \psi_q \\ 0 \end{pmatrix}
+ \begin{pmatrix} \psi_x & \rho_f \psi_{xf} \\ 0 & \rho_f \end{pmatrix}
\begin{pmatrix} x_{1t} \\ f_{1t} \end{pmatrix}
+ \begin{pmatrix} \psi_w & \sigma_f \psi_{xf} \\ 0 & \sigma_f \end{pmatrix}
\begin{pmatrix} w_{t+1} \\ w_{t+1}^f \end{pmatrix}.

$$

The new coefficient $\psi_{xf}$ measures how a unit change in the belief
shock $f_{1t}$ feeds into the endogenous state variables.

It is determined by a backward-induction algorithm that iterates from a
distant terminal date $T$ (where belief distortions vanish) back to the
present.

The continuation value in the $f$-direction satisfies a separate recursion
for $v_f$, and the belief distortion matrix becomes

$$

\mathbb{E} = \operatorname{stack}\Bigl\{
  \sigma_i\bigl[
    g_{x^+}\psi_{xf}\sigma_f^2(v_f + v_x\psi_{xf})
    + (g_{x^+}\psi_w + g_{w^+})\psi_w' v_x'
  \bigr]^i
\Bigr\}\bar\theta_f.

$$

The algorithm therefore decomposes cleanly into two stages:

1. **Stage 1 (rational-expectations block)**: solve for
   $\psi_x$, $\psi_w$ using the standard Blanchard–Kahn method; these are
   *unaffected* by the belief shock.

2. **Stage 2 (belief distortion block)**: given $\psi_x, \psi_w, v_x$,
   iterate backward to convergence to find $\psi_{xf}$, $v_f$, and
   $\mathbb{E}$.

This separation is a major practical advantage: existing rational-expectations
solvers can be used for Stage 1 with only a wrapper for Stage 2.

In the scalar endowment model, Stage 2 is simple because the value function
is the only forward-looking object: the state is exogenous, so there is no
feedback coefficient $\psi_{xf}$ to solve for.

The code below is therefore a scalar toy illustration of the value block of
Stage 2, not an implementation of the full two-stage algorithm.

It solves the Riccati equation by iteration starting from the
rational-expectations value, as the paper does, and then reads off the
subjective transition coefficients from the re-centred shocks.

```{code-cell} ipython3
# Stage 1: rational-expectations objects of the scalar endowment model
ψ_x_re = model.ρ_x
ψ_w_re = model.σ_x
u_x = 1 - model.β

# Stage 2 value block: iterate on the Riccati equation
vx_iter = u_x / (1 - model.β * ψ_x_re)      # start at the RE value
for it in range(1, 200):
    vx_new = (u_x + model.β * ψ_x_re * vx_iter
              - 0.5 * model.β * model.μ_θ * ψ_w_re**2 * vx_iter**2)
    if abs(vx_new - vx_iter) < 1e-15:
        break
    vx_iter = vx_new

print(f"Riccati by iteration ({it} steps):  v_x = {vx_iter:.10f}")
print(f"Riccati by quadratic formula:     v_x = {model.vx:.10f}")

# Subjective transition coefficients implied by the re-centred shocks
ψ_x_subj = ψ_x_re - model.μ_θ * vx_iter * ψ_w_re**2
ψ_q_subj = -model.μ_θ * 1.0 * vx_iter * ψ_w_re**2   # steady state x_bar = 1

print(f"\nObjective persistence:   ψ_x      = {ψ_x_re:.8f}")
print(f"Subjective persistence:  ψ_x_subj = {ψ_x_subj:.8f}")
print(f"Subjective drift:        ψ_q_subj = {ψ_q_subj:.2e}")
```

The two solution methods agree.

For this consumption-type state ($v_x > 0$), the subjective persistence is
slightly *below* the objective one and the subjective drift is negative: the
pessimist expects good states to fade and bad outcomes to arrive.

For an unemployment-type state ($v_x < 0$), both signs flip, so subjective
persistence exceeds objective persistence --- the formal statement of
"pessimists believe bad times last longer".

### Sequence problem and dynamic consistency

The recursive formulation used throughout the lecture emerges from the
following sequence problem.

Define the discounted entropy functional

$$

\mathcal{E}_t \;=\; E_t \sum_{j=0}^{\infty} \beta^j
  \left[ M_{t,t+j} \frac{\beta}{\theta_{t+j}}
    E_{t+j}[m_{t+j+1} \log m_{t+j+1}]
  \right],

$$

where $M_{t,t+j} = \prod_{k=1}^j m_{t+k}$.

The agent's problem is

$$

v_t^* = \max_{\{y_{t+j}\}} \min_{\substack{m_{t+j}>0 \\ E_{t+j-1}[m_{t+j}]=1}}
  \sum_{j=0}^{\infty} \beta^j E_t[M_{t,t+j} u_{t+j}]
  + \mathcal{E}_t.

$$

The penalty functional $\mathcal{E}_t$ **discounts future entropies
weighted by future penalty parameters $\theta_{t+j}$**, which makes the
agent's choices dynamically consistent: she anticipates how her
pessimism will evolve.

This differs from the infinite-horizon discounted entropy used in
{cite}`HansenSargent2001`, which is not generally dynamically consistent
when $\theta_t$ is time-varying.

The recursive form is:

$$

\mathcal{E}_t = \frac{\beta}{\theta_t} E_t[m_{t+1} \log m_{t+1}]
  + \beta E_t[m_{t+1} \mathcal{E}_{t+1}].

$$

Under this penalty, the minimax inequality is an equality, and the value
function satisfies the recursive form stated in the main lecture:

$$

v_t^* = \max_{y_t} \min_{\substack{m_{t+1}>0 \\ E_t[m_{t+1}]=1}}
  u_t + \frac{\beta}{\theta_t} E_t[m_{t+1} \log m_{t+1}]
  + \beta E_t[m_{t+1} v_{t+1}^*].

$$

```{code-cell} ipython3
θ_path = np.array([3.0, 5.64, 8.0, 12.0])   # rising pessimism scenario

def one_period_entropy(θ, vx, σ_x):
    """
    Entropy E_t[m_{t+1} log m_{t+1}] under the optimal distortion
    for Gaussian shocks: = (1/2) * (θ * vx * σ_x)^2.
    """
    return 0.5 * (θ * vx * σ_x) ** 2

print("Effect of time-varying θ on entropy and belief wedge:")
print(f"{'θ_t':>8}  {'H_t (entropy)':>16}  {'Δ(x) = σ_x ν_t (pp)':>22}")
print('-' * 52)
for th in θ_path:
    H = one_period_entropy(th, model.vx, model.σ_x)
    bw = belief_wedge(model, th) * 100
    print(f"{th:>8.2f}  {H:>16.6f}  {bw:>22.4f}")

print()
print("The entropy penalty grows quadratically in θ,")
print("constraining the agent from distorting beliefs too heavily.")
```

## Summary

The lecture has built the mechanism from the survey object to the model object.

A belief wedge is the difference between a subjective forecast and an objective
forecast.

In the data, the unemployment and inflation wedges are positive on average,
countercyclical, and well described by one common factor.

Multiplier preferences generate exactly this kind of common factor: a higher
$\theta_t$ makes agents overweight states with low continuation value.

With Gaussian shocks, the optimal change of measure is especially simple: it
shifts the mean of the innovation by
$-\theta_t (v_x \psi_w)'$.

This mean shift implies belief wedges that are proportional to $\theta_t$ and
to the covariance between shocks and continuation values.

In the New Keynesian application, the same belief shock raises unemployment,
creates comoving unemployment and inflation forecast wedges, and helps close
the unemployment volatility gap left by TFP and monetary-policy shocks alone.

The survey wedges do double duty: they calibrate the belief-shock process, and
their joint behavior across variables --- means, comovement, cyclicality, and
forecast-error predictability --- over-identifies and thereby tests the model.

## Exercises

```{exercise-start}
:label: sbbc_ex1
```

**Belief wedge sign**

In the simple endowment economy built by `create_belief_model`, suppose the
state variable is log consumption $x_t$ with $\rho_x = 0.90$, $\sigma_x = 0.01$,
$\beta = 0.99$.

1. Compute $v_x$ under rational expectations and under pessimism
    $\mu_\theta = 4$.
2. What is the sign of the belief wedge for consumption growth?
3. If instead the agent forecasts unemployment (which enters the value
    function with a negative sign, so $u_x < 0$), what is the sign of the
    unemployment belief wedge?
```{exercise-end}
```

```{solution-start} sbbc_ex1
:label: sbbc_ex1_sol
:class: dropdown
```

*Part 1.* Under rational expectations ($\theta = 0$):

$$

v_x^{RE} = \frac{u_x}{1 - \beta \rho_x}
         = \frac{1 - \beta}{1 - \beta \rho_x}.

$$

```{code-cell} ipython3
β_ex = 0.99
ρ_x_ex = 0.90
σ_x_ex = 0.01
μ_θ_ex = 4.0

vx_re_ex = (1 - β_ex) / (1 - β_ex * ρ_x_ex)
print(f"v_x (rational expectations): {vx_re_ex:.4f}")

m_ex = create_belief_model(β=β_ex, ρ_x=ρ_x_ex,
                           σ_x=σ_x_ex, μ_θ=μ_θ_ex)
print(f"v_x (with pessimism θ_bar={μ_θ_ex}):   {m_ex.vx:.4f}")
```

*Part 2.* Under pessimism ($\theta_t > 0$), the consumption wedge is

$$
\Delta_t^{(1)}(x)
= -\theta_t v_x \sigma_x^2.
$$

Since $v_x > 0$ and $\theta_t > 0$, the wedge is **negative**: pessimistic
agents underestimate consumption growth relative to the model.

*Part 3.* For unemployment, $u_x < 0$, so $v_x^u < 0$.

The belief wedge becomes

$$
\Delta_t^{(1)}(u)
= -\theta_t v_x^u \sigma_x^2 > 0
$$

(positive, because pessimism makes agents over-estimate unemployment).
This matches the empirical finding of a positive mean unemployment wedge.

```{solution-end}
```

```{exercise-start}
:label: sbbc_ex2
```

**Persistence and wedge volatility**

Using `create_belief_model`, vary $\rho_\theta$ from 0.3 to
0.95 (holding $\sigma_\theta = 4.3$ fixed) and plot how the standard
deviation of the belief wedge changes.

Explain the economic intuition.
```{exercise-end}
```

```{solution-start} sbbc_ex2
:label: sbbc_ex2_sol
:class: dropdown
```

```{code-cell} ipython3
ρ_vals = np.linspace(0.3, 0.95, 30)
wedge_stds = []

for ρ in ρ_vals:
    m_temp = create_belief_model(ρ_θ=ρ)
    θ_sim_temp = simulate_θ(m_temp, T=5000, seed=0)
    wedge_sim_temp = belief_wedge(m_temp, θ_sim_temp)
    wedge_stds.append(np.std(wedge_sim_temp))

fig, ax = plt.subplots()
ax.plot(ρ_vals, np.array(wedge_stds) * 100, color='steelblue', linewidth=2)
ax.set_title('Persistence and belief-wedge volatility')
ax.set_xlabel('persistence $\\rho_\\theta$')
ax.set_ylabel('standard deviation of belief wedge (pp)')
plt.tight_layout()
plt.show()
```

The figure plots the persistence parameter $\rho_\theta$ on the horizontal
axis and the simulated standard deviation of the belief wedge on the vertical
axis.

The curve slopes upward.

Higher persistence $\rho_\theta$ means that a given innovation to $\theta_t$
has more prolonged effects: the unconditional variance of an AR(1) with
volatility $\sigma$ is $\sigma^2 / (1 - \rho^2)$, which increases in $\rho$.

Since the wedge is proportional to $\theta_t$, its standard deviation
inherits this relationship and rises with $\rho_\theta$.

```{solution-end}
```

```{exercise-start}
:label: sbbc_ex3
```

**Unemployment volatility decomposition**

Using the reduced-form NK model built by `create_nk_model`:

1. Compute the fraction of unemployment variance explained by each of the
    three shocks.

2. Show that the belief shock is the dominant driver of unemployment
    fluctuations, while TFP shocks matter much more for inflation and
    output than they do for unemployment.
```{exercise-end}
```

```{solution-start} sbbc_ex3
:label: sbbc_ex3_sol
:class: dropdown
```

```{code-cell} ipython3
shock_names = ['Belief shock (θ)', 'TFP shock', 'MP shock']
var_labels = ['Unemployment', 'Inflation', 'Output']

nk2 = create_nk_model()

n_states = nk2.A.shape[0]
var_by_shock = np.zeros((n_states, 3))

for j in range(3):
    B_j = np.outer(nk2.B[:, j], nk2.B[:, j])
    Σ_j = solve_discrete_lyapunov(nk2.A, B_j)
    var_by_shock[:, j] = np.diag(Σ_j)

var_total = var_by_shock.sum(axis=1)

print(f"{'Variable':<16}", *[f"{s:>20}" for s in shock_names])
print('-' * 77)
for i, label in zip([I_U, I_PI, I_Y], var_labels):
    shares = var_by_shock[i] / var_total[i] * 100
    print(f"{label:<16}", *[f"{s:>19.1f}%" for s in shares])
```

The belief shock accounts for the large majority of unemployment variance in
this calibrated emulator.

Technology shocks drive most of the inflation variance, and output variance
is split roughly evenly between the belief and TFP shocks.

Monetary policy shocks play a negligible role for all three variables.

This pattern matches the variance decomposition of the structural model, in
which the belief shock dominates unemployment while technology shocks account
for most of the variation in inflation.

```{solution-end}
```

```{exercise-start}
:label: sbbc_ex4
```

**Changing the degree of pessimism**

Solve the Riccati equation (`solve_vx`) for a grid of
$\mu_\theta$ values from 0 (rational expectations) to 15.

For each value,
compute the scaled subjective drift $\nu / \sigma_x = -\mu_\theta v_x$
and the steady-state belief wedge.

Discuss how the robust value function differs from
the rational-expectations value function.
```{exercise-end}
```

```{solution-start} sbbc_ex4
:label: sbbc_ex4_sol
:class: dropdown
```

```{code-cell} ipython3
μ_grid = np.linspace(0, 15, 100)
drift_norm = []
wedge_ss = []

for μ in μ_grid:
    m_temp = create_belief_model(μ_θ=μ)
    drift_norm.append(-μ * m_temp.vx)
    wedge_ss.append(belief_wedge(m_temp, μ) * 100)   # in pp

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle('Subjective drift and steady-state wedge')

axes[0].plot(μ_grid, drift_norm, color='steelblue', linewidth=2)
axes[0].axhline(0, color='grey', linestyle='--', linewidth=0.8)
axes[0].set_xlabel('mean pessimism $\\mu_\\theta$')
axes[0].set_ylabel('scaled drift $\\nu / \\sigma_x$')

axes[1].plot(μ_grid, np.array(wedge_ss), color='firebrick', linewidth=2)
axes[1].set_xlabel('mean pessimism $\\mu_\\theta$')
axes[1].set_ylabel('steady-state wedge (pp)')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
```

The left panel plots the scaled subjective drift
$\nu / \sigma_x = -\mu_\theta v_x$.

The right panel plots the corresponding steady-state belief wedge.

The scaled drift is the horizontal shift used in the shock-distribution
figure above, so it is easier to see than the tiny movement in $v_x$ itself.

The steady-state consumption wedge becomes more negative, approximately
linearly in magnitude, since
$\Delta^{(1)} \propto -\mu_\theta v_x \sigma_x^2$ and $v_x$ is approximately
constant for small $\mu_\theta$.

Finally, consider how the robust value function itself changes with
$\mu_\theta$.

```{code-cell} ipython3
vx_0 = create_belief_model(μ_θ=0).vx
vx_15 = create_belief_model(μ_θ=15).vx
print(f"v_x at μ_θ = 0:   {vx_0:.8f}")
print(f"v_x at μ_θ = 15:  {vx_15:.8f}")
print(f"relative change:  {(vx_15 - vx_0) / vx_0:.2e}")
```

The slope $v_x$ falls as $\mu_\theta$ rises --- the quadratic term in the
Riccati equation lowers the marginal value of the state --- but the change is
on the order of $10^{-5}$ in relative terms, because the quadratic term is
scaled by $\sigma_x^2$.

The robust value function therefore differs from its rational-expectations
counterpart mainly through the constant $v_q$, which falls as the variance
penalty grows.

Because $v_x$ is nearly constant, the drift and the wedge are approximately
linear in $\mu_\theta$, which is what both panels show.

```{solution-end}
```
