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

We interpret this evidence using
**robust preferences** ({cite:t}`HansenSargent2001`; {cite:t}`HansenSargent2008`).

Robust preferences provide model-consistent notions of pessimism and optimism:
a pessimistic agent acts as if states that deliver low continuation values are
more likely than they really are, while an optimistic agent overweights states
that deliver high continuation values.

When calibrated to the Michigan Survey of
Consumers (1982Q1-2019Q4), this mechanism yields a time-varying *belief shock*
that substantially reduces the **unemployment volatility puzzle** ---
the fact that standard New Keynesian models with only technology and monetary
policy shocks generate far too little unemployment volatility.

In this lecture, we will cover:

* How to define and measure belief wedges from household survey data.
* Why optimal pessimism is a mean shift of the shock distribution, with
  size equal to $\theta_t$ times the shock's exposure to continuation values.
* How belief distortions propagate through a linearized DSGE model.
* Why a calibrated belief shock helps resolve the unemployment volatility
  puzzle.

The lecture is self-contained in the following sense.

All empirical moments, calibration values, and model objects used in the code
are reported below, so no external data files are needed.

Benchmark numbers from {cite:t}`bhandari2025survey` are quoted for comparison,
but every computation below is generated from the equations and parameter
values stated here.

Some notation and units will be used throughout:

* `pp` means percentage points.
* Inflation is reported at an annualized rate when explicitly marked
  "ann."; otherwise it is a quarterly rate.
* $u_t$, $\pi_t$, and $y_t$ denote unemployment, inflation, and the output gap.
* $a_t$ is total factor productivity (TFP), and $w_t^r$ is a monetary-policy
  shock.
* $\theta_t$ is the belief-shock or pessimism state: larger $\theta_t$ means
  agents put more subjective probability on low-continuation-value states.

We start with the following imports

```{code-cell} ipython3
import datetime
from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.linalg import solve_discrete_lyapunov
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

We do not need those raw survey responses below.

Instead, the empirical section works with the already-constructed wedge
moments listed next.

### Empirical facts

Using data from 1982Q1 to 2019Q4, {cite:t}`bhandari2025survey` document:

| Statistic | Unemployment wedge | Inflation wedge |
|---|---|---|
| Mean | 0.52 pp | 1.22 pp |
| Standard deviation | 0.57 pp | 0.97 pp |
| Correlation with output gap | −0.49 | −0.30 |

Both wedges are **positive on average**: households expect higher unemployment
and higher inflation than the rational forecast.

Both wedges are **countercyclical**: they rise during every NBER recession in
the sample.

These facts are robust: they survive replacing the VAR forecast with the SPF
forecast, alternative VAR specifications, and extending the sample back to
1960.

A positive unemployment wedge is naturally read as pessimism, since
unemployment is high in bad times.

The positive inflation wedge carries the same interpretation because
households regard high inflation as a feature of bad times: in the Bank of
England's Inflation Attitudes Survey, 50% to 80% of households consistently
answer that faster inflation would weaken the economy.

The two wedges are positively correlated, with a correlation of 0.34, and the
first principal component of the wedge series explains **78.6%** of their
joint variation.

(The principal component is computed from the raw wedges, so it reflects both
their correlation and the larger variance of the inflation wedge.)

The same one-factor pattern appears in the cross section.

Michigan Survey households with high inflation forecasts are more likely to
expect unemployment to rise, to expect worse aggregate and personal financial
conditions, and to assign higher probability to losing their own jobs.

Average wedges also line up positively across demographic groups, and although
more educated and higher-income households make smaller forecast errors, every
group overpredicts both variables.

The FRBNY Survey of Consumer Expectations (SCE) shows the same patterns.

This evidence supports the interpretation that the wedges reflect a common
pessimism/optimism component rather than two unrelated forecast mistakes.

The following code simulates a stylized wedge process that matches the sample
length, the mean and standard deviation of each wedge, and the 0.34
correlation between the wedges.

The point is not to recreate the raw Michigan series.

Instead, the simulation separates the common pessimism factor from residual
survey noise, so the wedges are strongly related but not perfectly collinear.

```{code-cell} ipython3
# Baseline belief-shock parameters used throughout the lecture.
μ_θ = 5.64   # mean of belief-shock parameter θ
ρ_θ = 0.714  # AR(1) persistence of θ
σ_θ = 4.3    # innovation volatility

# Wedge loadings used later in the model illustrations.
c_u = 0.52 / μ_θ
c_π = 1.22 / μ_θ

# Empirical wedge moments used to discipline the simulation.
mean_u, mean_π = 0.52, 1.22
std_u, std_π = 0.57, 0.97
corr_wedges = 0.34

T = 152   # 38 years * 4 quarters

# Simulate the belief shock.
rng = np.random.default_rng(42)
θ = np.zeros(T)
θ[0] = μ_θ
for t in range(1, T):
    θ[t] = ((1 - ρ_θ) * μ_θ
            + ρ_θ * θ[t-1]
            + σ_θ * rng.standard_normal())

def standardize(x):
    """Return a de-meaned series with unit standard deviation."""
    return (x - np.mean(x)) / np.std(x)


def orthogonal_noise(rng, *basis, T=T):
    """Generate standardized noise orthogonal to the supplied basis series."""
    e = standardize(rng.standard_normal(T))
    for b in basis:
        e = e - (e @ b) / (b @ b) * b
    return standardize(e)


# Loading each wedge on the common factor with weight √corr makes the
# sample correlation between the two wedges equal corr_wedges exactly.
common_weight = np.sqrt(corr_wedges)
noise_weight = np.sqrt(1 - corr_wedges)

common_factor = standardize(θ)
noise_u = orthogonal_noise(rng, common_factor)
noise_π = orthogonal_noise(rng, common_factor, noise_u)

wedge_u = mean_u + std_u * (common_weight * common_factor
                            + noise_weight * noise_u)
wedge_π = mean_π + std_π * (common_weight * common_factor
                            + noise_weight * noise_π)

# Principal components of the raw (non-standardized) wedges, as in the paper.
wedges = np.column_stack([wedge_u, wedge_π])
eigvals = np.linalg.eigvalsh(np.cov(wedges, rowvar=False))
pc1_share = eigvals[-1] / eigvals.sum()

# Quarterly dates, 1982Q1-2019Q4
quarters = [datetime.date(1982 + (q // 4), 3 * (q % 4) + 1, 1)
            for q in range(T)]
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: simulated belief wedges
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
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: one-factor structure of belief wedges
    name: fig-sbbc-wedge-scatter
---
fig, ax = plt.subplots()
sc = ax.scatter(wedge_u, wedge_π, c=range(T), cmap='RdYlGn_r',
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

The first figure plots the simulated unemployment wedge in the top panel and
the simulated inflation wedge in the bottom panel.

The dashed horizontal lines show the sample means, which match the empirical
values reported above.

Both wedges have a common cyclical component driven by $\theta_t$, but they also
contain residual components that stand in for survey noise and other
idiosyncratic variation.

The scatter plot makes the one-factor structure visible.

Each point is one quarter, with the horizontal coordinate equal to the
unemployment wedge and the vertical coordinate equal to the inflation wedge.

The points form an upward-sloping cloud rather than a line: the common factor
drives both wedges, while survey noise keeps them from being collinear.

With the means, standard deviations, and correlation matched, the first
principal component of the simulated wedges accounts for about 78.5% of their
joint variation, essentially the 78.6% share in the data.

This is the one-factor structure that motivates the theoretical framework.

## A model of pessimism

### Robust preferences

Why would households have systematically biased beliefs?

One disciplined answer comes from **robust control** or **multiplier preferences**
({cite}`HansenSargent2001`, {cite}`HansenSargent2008`).

An agent represented by multiplier preferences solves

$$

V_t \;=\; \min_{\substack{m_{t+1} > 0 \\ E_t[m_{t+1}] = 1}}
\Bigl\{
  u(x_t)
  + \beta E_t\!\left[m_{t+1} V_{t+1}\right]
  + \frac{\beta}{\theta_t}\, E_t\!\left[m_{t+1} \log m_{t+1}\right]
\Bigr\}.

$$

Here $m_{t+1}$ is a **likelihood ratio** (Radon–Nikodym derivative) that
distorts the reference measure, and the last term is an entropy penalty that
keeps the distortion from being too extreme.

The state vector $x_t \in \mathbb{R}^n$ follows a Markov law of motion

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
high-continuation-value states --- optimism.

The inner minimization has a closed-form solution.

Minimizing $E_t[m_{t+1} V_{t+1}] + \frac{1}{\theta_t} E_t[m_{t+1} \log m_{t+1}]$
state by state subject to $E_t[m_{t+1}] = 1$ gives the first-order condition

$$

V_{t+1} + \frac{1}{\theta_t}\left(\log m_{t+1} + 1\right) + \lambda_t = 0,

$$

where $\lambda_t$ is the multiplier on the constraint, so
$m_{t+1} \propto \exp(-\theta_t V_{t+1})$ and the constraint pins down the
normalization:

$$

m_{t+1}^* \;=\;
\frac{\exp(-\theta_t V_{t+1})}{E_t[\exp(-\theta_t V_{t+1})]}.

$$

Since $m_{t+1}^*$ assigns higher weight to states where $V_{t+1}$ is low (bad
outcomes), pessimistic agents effectively over-weight recessions in their
probability assessments.

Substituting $m_{t+1}^*$ back into the recursion gives the equivalent
**risk-sensitive** form

$$

V_t = u(x_t) - \frac{\beta}{\theta_t}
\log E_t\!\left[\exp(-\theta_t V_{t+1})\right],

$$

which replaces the expected continuation value with a soft minimum: as
$\theta_t \to 0$ it reduces to $u(x_t) + \beta E_t[V_{t+1}]$, and as
$\theta_t \to \infty$ it approaches the worst case over states.

In the robust-control literature, this distortion expresses fear of model
misspecification.

{cite:t}`bhandari2025survey` instead take the recursion directly as a model of
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

When $V_{t+1}$ is high in states where
$z_{t+1}$ is also high, $m_{t+1}^*$ will be low in those states, making the
covariance negative --- i.e.\ the agent *underestimates* good-state variables.

For unemployment (which varies inversely with good economic outcomes), the
wedge is positive: pessimists expect higher unemployment than the model predicts.

### Illustration: optimal belief distortion

To see the mechanism concretely, consider an **endowment economy** with a
scalar log-consumption state $x_t$ and dynamics

$$

x_{t+1} = \rho_x x_t + \sigma_x w_{t+1}, \qquad w_{t+1} \sim N(0,1).

$$

With log utility, $u(x_t) = (1 - \beta)x_t$, and we guess that the
continuation value is linear: $V_t = V_x x_t + V_q$.

The Gaussian structure makes everything explicit.

Since $-\theta_t V_{t+1} = -\theta_t(V_x \rho_x x_t + V_q) - \theta_t V_x
\sigma_x w_{t+1}$ is linear in the standard normal shock, the moment
generating function $E[\exp(a w)] = \exp(a^2/2)$ evaluates the
risk-sensitive recursion in closed form:

$$

-\frac{1}{\theta_t} \log E_t\!\left[\exp(-\theta_t V_{t+1})\right]
= V_x \rho_x x_t + V_q - \frac{\theta_t}{2}\, V_x^2 \sigma_x^2.

$$

The expected continuation value is penalized by half the pessimism parameter
times the conditional variance of continuation values: pessimism acts like an
endogenous discount on risky continuation utilities.

The same linearity pins down the optimal distortion.

Writing $a_t = -\theta_t V_x \sigma_x$, the likelihood ratio becomes

$$

m_{t+1}^* = \frac{\exp(a_t w_{t+1})}{E_t[\exp(a_t w_{t+1})]}
= \exp\!\left(a_t w_{t+1} - \tfrac{1}{2} a_t^2\right),

$$

which is exactly the density of $N(a_t, 1)$ divided by the density of
$N(0, 1)$.

Pessimism is therefore a **mean shift**: under the subjective measure,

$$

w_{t+1} \;\sim\; N(\nu_t, 1),
\qquad
\nu_t = -\theta_t V_x \sigma_x.

$$

The drift $\nu_t$ is the pessimism parameter times the exposure of the
continuation value to the shock --- the agent slants beliefs exactly in the
direction that hurts most, by an amount the entropy penalty limits.

The belief wedge for the state follows immediately:

$$

\Delta_t^{(1)}(x) = \tilde E_t[x_{t+1}] - E_t[x_{t+1}]
= \sigma_x \nu_t = -\theta_t V_x \sigma_x^2.

$$

When $V_x > 0$ (high consumption states are good) and $\theta_t > 0$
(pessimism), the wedge is negative --- the agent *underestimates*
future consumption.

For a variable that enters the value function with a negative sign, such as
unemployment, the same pessimism generates a **positive** wedge.

Finally, substituting the closed-form recursion back into
$V_t = u(x_t) - \frac{\beta}{\theta_t}\log E_t[\exp(-\theta_t V_{t+1})]$
and matching coefficients on $x_t$ --- using the scaling of $\theta_t$
derived in the appendix, under which the penalty term contributes to the
slope with coefficient $\mu_\theta$ --- yields the **Riccati equation**

$$

V_x = u_x + \beta \rho_x V_x - \frac{\beta}{2}\,\mu_\theta\, \sigma_x^2 V_x^2,
\qquad u_x = 1 - \beta.

$$

The quadratic term is the price of pessimism: it lowers the marginal value of
the state relative to the rational-expectations value
$V_x^{RE} = u_x / (1 - \beta\rho_x)$.

We now turn this illustration into code, building it up from small pieces.

The first ingredient is the slope $V_x$ of the continuation value.

It solves the scalar Riccati equation, which we write as a quadratic
$a V_x^2 + b V_x + c = 0$ and solve with the quadratic formula.

We keep the root that collapses to the rational-expectations value
$V_x^{RE} = u_x / (1 - \beta\rho_x)$ as the pessimism parameter $\mu_\theta \to 0$.

```{code-cell} ipython3
def solve_Vx(β, ρ_x, σ_x, μ_θ):
    """
    Solve the scalar Riccati equation for the value-function slope Vx:

        Vx = u_x - (β/2) μ_θ σ_x**2 Vx**2 + β ρ_x Vx,   with u_x = 1 - β.
    """
    u_x = 1.0 - β                       # marginal utility of log consumption
    Vx_re = u_x / (1.0 - β * ρ_x)       # rational-expectations (θ = 0) value

    # Coefficients of a Vx**2 + b Vx + c = 0
    a = 0.5 * β * σ_x**2 * μ_θ
    b = 1.0 - β * ρ_x
    c = -u_x

    if abs(a) < 1e-14:                  # no pessimism: equation is linear
        return Vx_re

    disc = b**2 - 4.0 * a * c
    if disc < 0:                        # no real root: fall back to RE
        return Vx_re

    # Keep the root closest to the rational-expectations value
    r1 = (-b + np.sqrt(disc)) / (2.0 * a)
    r2 = (-b - np.sqrt(disc)) / (2.0 * a)
    return r1 if abs(r1 - Vx_re) < abs(r2 - Vx_re) else r2
```

We store the primitives in a `NamedTuple`, together with the solved slope
$V_x$, and use `create_belief_model` to build an instance.

```{code-cell} ipython3
class BeliefModel(NamedTuple):
    β: float      # discount factor
    ρ_x: float    # persistence of log consumption
    σ_x: float    # volatility of the consumption innovation
    μ_θ: float    # mean of the belief-shock parameter θ
    ρ_θ: float    # AR(1) persistence of θ
    σ_θ: float    # volatility of the θ innovation
    Vx: float     # slope of the linearized continuation value


def create_belief_model(β=0.994, ρ_x=0.85, σ_x=0.005,
                        μ_θ=5.64, ρ_θ=0.714, σ_θ=4.3):
    """Build a belief model, solving the Riccati equation for Vx."""
    Vx = solve_Vx(β, ρ_x, σ_x, μ_θ)
    return BeliefModel(β=β, ρ_x=ρ_x, σ_x=σ_x,
                       μ_θ=μ_θ, ρ_θ=ρ_θ, σ_θ=σ_θ, Vx=Vx)
```

Two functions map a value of $\theta_t$ into the implied distortion.

The drift $\nu_t = -\theta_t V_x \sigma_x$ is the mean shift of the shock under
the subjective measure; the wedge $\Delta_t^{(1)}(x) = \sigma_x \nu_t$ is the
resulting forecast bias for the state.

```{code-cell} ipython3
def belief_drift(model, θ):
    """Mean shift of the shock under subjective beliefs: ν = -θ Vx σ_x."""
    return -θ * model.Vx * model.σ_x


def belief_wedge(model, θ):
    """One-period belief wedge for the state: Δ = σ_x ν = -θ Vx σ_x**2."""
    return model.σ_x * belief_drift(model, θ)
```

A last helper simulates the AR(1) belief shock $\theta_t$.

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
$V_x$ with its rational-expectations counterpart, and report the mean belief
drift and wedge.

```{code-cell} ipython3
model = create_belief_model()

Vx_re = (1 - model.β) / (1 - model.β * model.ρ_x)
print(f"RE value of Vx:       {Vx_re:.4f}")
print(f"Robust value of Vx:   {model.Vx:.4f}")
print(f"Belief drift at θ_bar:   {belief_drift(model, model.μ_θ) * 100:.4f} pp")
print(f"Belief wedge at θ_bar:   {belief_wedge(model, model.μ_θ) * 100:.4f} pp")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: objective and subjective shock distributions
    name: fig-sbbc-shock-distributions
---
# Scale the tiny drift by σ_x for visibility.

θ_vals = [0, model.μ_θ, 2 * model.μ_θ]
labels = ['θ = 0  (rational)',
          f'θ = θ_bar = {model.μ_θ:.1f}  (mean)',
          f'θ = 2θ_bar  (pessimistic)']
colors = ['black', 'steelblue', 'firebrick']

# Drift in units of σ_x
ν_tilde = [-θ * model.Vx for θ in θ_vals]

x_grid = np.linspace(-4, 4, 500)

fig, ax = plt.subplots()
for μ, label, color in zip(ν_tilde, labels, colors):
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x_grid - μ)**2)
    ax.plot(x_grid, pdf, label=label, color=color, linewidth=2)

ax.axvline(0, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel(
    'innovation shift in units of $\\sigma_x$: '
    '$\\nu_t / \\sigma_x = -\\theta_t V_x$'
)
ax.set_ylabel('density')
ax.legend()
plt.tight_layout()
plt.show()

print("Mean shifts (in units of σ_x):")
for μ, label in zip(ν_tilde, labels):
    print(f"  {label:35s}  ν_tilde = {μ:.4f}")
```

The figure shows how pessimism (higher $\theta_t$) shifts the perceived
distribution of future shocks to the left.

The black curve is the objective distribution, centered at zero.

The blue and red curves are subjective distributions for progressively larger
values of $\theta_t$.

The horizontal axis measures the shift in units of $\sigma_x$, so the leftward
movement is the normalized subjective drift $\nu_t / \sigma_x$.

An agent with $\theta_t > 0$
believes bad shocks are more likely than they actually are.

### Subjective dynamics

The mean shift changes the law of motion that agents perceive.

Substituting $w_{t+1} = \nu_t + \tilde w_{t+1}$, where
$\tilde w_{t+1} \sim N(0, 1)$ under the subjective measure, into the dynamics
of $x_t$ gives

$$

x_{t+1} = -\theta_t V_x \sigma_x^2 + \rho_x x_t + \sigma_x \tilde w_{t+1}.

$$

With $\theta_t = \bar\theta(\bar x + x_t)$, collecting terms shows that
subjective beliefs change both the intercept and the slope of the perceived
dynamics:

$$

\tilde\rho_x = \rho_x - \bar\theta\, V_x \sigma_x^2,
\qquad
\tilde\psi_q = -\bar\theta \bar{x}\, V_x \sigma_x^2.

$$

For a good state ($V_x > 0$), pessimism adds a negative drift; for a bad
state such as unemployment ($V_x < 0$), the same formula raises the
subjective persistence --- pessimists believe bad times last longer.

The code below compares objective and subjective forecast paths of the
consumption state, starting from the steady state, holding the pessimism
level fixed along the forecast path.

```{code-cell} ipython3
def forecast_paths(model, θ, x0=0.0, τ_max=20):
    """Objective and subjective forecast paths E_t[x_{t+τ}], Ẽ_t[x_{t+τ}]."""
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
systematically wrong --- and that systematic error is exactly the belief
wedge measured in the surveys.

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

w_{t+1} \;\sim\; N\!\left(- \theta_t (V_x \psi_w)',\; I_k\right),

$$

where $V_x$ is the row vector of first derivatives of the continuation value
and $\bar{x}$ is the non-stochastic steady state.

In a standard first-order perturbation, belief distortions would vanish: as
shock volatility shrinks, the entropy penalty makes the distortion second
order.

{cite:t}`bhandari2025survey` avoid this by scaling $\theta_t$ jointly with the
shock volatility, which keeps the subjective model distinct from the
data-generating process in the linear solution; the appendix gives details.

The resulting **belief wedge** for any variable $z_t = \bar{z}' x_t$ is

$$

\Delta_t^{(1)}(z)
\;=\; -\theta_t\, \bar{z}' (\psi_w \psi_w') V_x'.

$$

Because the drift moves with $\theta_t$, subjective beliefs change both the
conditional mean and the persistence of the state: adverse states are more
persistent under the subjective measure than under the data-generating
measure.

### Riccati equation for $V_x$

The key object is $V_x$, which solves

$$

V_x
\;=\; u_x
  - \frac{\beta}{2}\, V_x \psi_w \psi_w' V_x' \bar\theta
  + \beta\, V_x \psi_x.

$$

This is a modified Riccati equation: the middle term arises from the entropy
penalty on beliefs and vanishes under rational expectations ($\bar\theta = 0$).

### One-factor structure

An important consequence of the formula for $\Delta_t^{(1)}(z)$ is that the
*time variation* in all belief wedges is driven by the **single scalar** belief
factor $\theta_t \approx \bar\theta(\bar{x} + x_{1t})$.

The cross-sectional loadings $-\bar{z}'(\psi_w\psi_w')V_x'$ are
fixed by the model's structural parameters.

The loadings are not free parameters: they equal covariances of shocks with
the continuation value, objects that the equilibrium of the model determines.

The only free parameters describing beliefs are the three governing the
$\theta_t$ process, so every additional surveyed variable adds an
overidentifying restriction on the model.

This theoretical prediction
matches the empirical finding that one principal component explains 78.6%
of the joint variation in the unemployment and inflation wedges.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: wedge loadings implied by one factor
    name: fig-sbbc-one-factor-loadings
---
θ_grid = np.linspace(0, 20, 200)

# Loadings match the mean empirical wedges.
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

## A New Keynesian model with belief distortions

### Model description

Following {cite:t}`bhandari2025survey`, we embed the belief-distortion
mechanism in a New Keynesian model with a **search-and-matching** labor market
({cite}`Shimer2005`; {cite}`ChristianoEichenbaumTrabandt2016`).

The economy is populated by a representative household with subjective
beliefs, competitive producers of a final good, monopolistic producers of
intermediate goods, and labor-market firms that hire workers through a
matching friction.

In the benchmark, all agents share the household's subjective beliefs;
the rational-firms variant below relaxes this.

#### Representative household

The household chooses consumption $C_t$ and bond holdings $B_{t+1}$ under
robust preferences:

$$

V_t = \max_{C_t,\, B_{t+1}}\;
\min_{\substack{m_{t+1} > 0 \\ E_t[m_{t+1}] = 1}}
\log C_t + \beta E_t\!\left[m_{t+1} V_{t+1}\right]
+ \frac{\beta}{\theta_t} E_t\!\left[m_{t+1} \log m_{t+1}\right],

$$

where the penalty parameter follows the exogenous AR(1) **belief shock**

$$

\theta_t = (1 - \rho_\theta)\mu_\theta + \rho_\theta \theta_{t-1}
+ \sigma_\theta w_t^\theta.

$$

This fits the framework above by appending $\theta_t$ to the state vector
$x_t$ and letting $\bar\theta$ select that component.

The household consists of a unit mass of workers who perfectly share
consumption risk.

A fraction $L_t$ is employed and earns the real wage $\xi_t$; the remaining
$1 - L_t$ are unemployed and receive benefits $D$ financed by lump-sum taxes.

The nominal budget constraint is

$$

P_t C_t + B_{t+1} \;\le\; (1 - L_t)\, P_t D + L_t P_t \xi_t
+ R_{t-1} B_t - T_t,

$$

where $P_t$ is the price level, $B_{t+1}$ are one-period bonds purchased at
$t$ with gross return $R_t$, and $T_t$ collects lump-sum taxes net of profits.

The process for $\theta_t$ is exogenous, but which states are *adverse* is
endogenous: equilibrium dynamics determine which states deliver low
continuation values $V_{t+1}$, and those are the states the household
overweights.

#### Labour market

At the end of each period, an employed worker keeps the job with probability
$\rho$, so $1 - \rho L_{t-1}$ workers search at the beginning of period $t$.

Employment evolves as

$$

L_t = \rho L_{t-1} + (1 - \rho L_{t-1})\, f_t,

$$

where $f_t$ is the job-finding probability, and measured unemployment is
$u_t = 1 - L_t$.

Labor-market firms post $v_t L_{t-1}$ vacancies at flow cost $\kappa_v$ each,
so labor market tightness is

$$

\zeta_t = \frac{v_t L_{t-1}}{1 - \rho L_{t-1}}.

$$

A Cobb--Douglas matching function
$M_t = \mu\, (v_t L_{t-1})^{\nu} (1 - \rho L_{t-1})^{1-\nu}$
combines vacancies and searchers, so

$$

f_t = \mu \zeta_t^{\nu},
\qquad
q_t = \frac{f_t}{\zeta_t},

$$

where $q_t$ is the probability of filling a vacancy.

Workers value jobs using the household's subjective beliefs.

Let $s_{t+1} = \beta C_t / C_{t+1}$ denote the household's stochastic discount
factor and $\tilde E_t[\cdot]$ the subjective expectation.

The values of unemployment and employment satisfy

$$

U_t = D + \tilde E_t\!\left[
s_{t+1}\bigl(f_{t+1} J^w_{t+1} + (1 - f_{t+1}) U_{t+1}\bigr)\right]

$$

and

$$

J^w_t = \xi_t
+ \tilde E_t\!\left[s_{t+1}\bigl(\rho + (1-\rho) f_{t+1}\bigr) J^w_{t+1}\right]
+ \tilde E_t\!\left[s_{t+1} (1-\rho)(1 - f_{t+1})\, U_{t+1}\right],

$$

where $\rho + (1-\rho)f_{t+1}$ combines keeping the current job with losing it
but immediately finding a new one.

A firm that employs a worker sells the labor services at the competitive price
$\vartheta_t$ and values the match at

$$

J_t = \vartheta_t - \xi_t + \rho\, \tilde E_t\!\left[s_{t+1} J_{t+1}\right],

$$

and free entry drives the expected profit from posting a vacancy to zero:

$$

J_t = \frac{\kappa_v}{q_t}.

$$

Wages are set by Nash bargaining with rigid wages, following
{cite}`Shimer2010`.

The firm and the worker bargain over a *target* wage $\xi_t^*$ that splits the
match surplus according to the bargaining weight $\eta$:

$$

\eta \left(J_t + \xi_t - \xi_t^*\right)
= (1 - \eta)\left(J^w_t - U_t + \xi_t^* - \xi_t\right).

$$

The actual wage adjusts only partially toward the target,

$$

\xi_t = \chi_w\, \xi_{t-1} + (1 - \chi_w)\, \xi_t^*,

$$

with $\chi_w = 0$ corresponding to flexible wages.

Vacancy posting and bargaining are forward-looking, so fluctuations in
$\theta_t$ directly move firms' incentives to create matches.

This is a sharp contrast with a Walrasian spot labor market, where one-period
contracts leave no role for beliefs about future economic conditions.

#### Production and price setting

A competitive final-good sector aggregates a continuum of intermediate
varieties,

$$

Y_t = \left[\int_0^1 (Y_{j,t})^{(\varepsilon-1)/\varepsilon}\, dj
\right]^{\varepsilon/(\varepsilon-1)},
\qquad
Y_{j,t} = \left(\frac{P_t}{P_{j,t}}\right)^{\varepsilon} Y_t,

$$

where $\varepsilon$ is the elasticity of substitution.

Monopolistic intermediate producers buy labor services at price $\vartheta_t$
and produce with the linear technology

$$

Y_{j,t} = \exp(a_t)\, l_{j,t} - \phi,

$$

where $a_t$ is log TFP and $\phi$ is a fixed cost.

Each producer reoptimizes its price with probability $1 - \chi_p$ (Calvo), so
price setting is a dynamic problem and is distorted by the firm's subjective
beliefs.

Goods and labor markets clear:

$$

C_t + \frac{\kappa_v}{q_t}\, h_t L_{t-1} = Y_t,
\qquad
\int_0^1 l_{j,t}\, dj = L_t,

$$

where $h_t = f_t (1 - \rho L_{t-1}) / L_{t-1}$ is the hiring rate and the
second term in the resource constraint is total vacancy-posting costs.

#### Monetary policy and shocks

The monetary authority follows a Taylor rule with smoothing,

$$

\log \frac{R_t}{\bar R}
= \rho_r \log \frac{R_{t-1}}{\bar R}
+ (1 - \rho_r)\left[
r_\pi \log \frac{\pi_t}{\bar\pi} + r_y \log \frac{Y_t}{Y^*}
\right]
+ \sigma_r w_t^r,

$$

where $\pi_t = P_t / P_{t-1}$ is gross inflation, $\bar\pi$ is the intercept,
and $Y^*$ is steady-state output.

TFP follows

$$

a_{t+1} = \rho_a a_t + \sigma_a w_{t+1}^a.

$$

The three innovations are independent under the data-generating measure $P$:

$$

(w_t^r,\, w_t^a,\, w_t^\theta)' \overset{iid}{\sim} N(0, I).

$$

Under the subjective measure they are not independent: the optimal distortion
ties their joint distribution to the current level of $\theta_t$, and this
subjective correlation structure is the source of the belief wedges.

#### Where beliefs enter

Every forward-looking equilibrium condition uses the subjective expectation
$\tilde E_t$: the household's consumption Euler equation, the worker and firm
match values, free entry, the bargaining problem, and the Calvo price setters'
first-order conditions.

In the linearized solution, each of these conditions is one row of the system
$0 = \tilde E_t[g(x_{t+1}, x_t, x_{t-1}, w_{t+1}, w_t)]$ described in the
appendix, where the belief distortion can be switched on or off row by row.

### Calibration

The model is calibrated to quarterly U.S. data, 1982Q1–2019Q4.

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Discount factor | $\beta$ | 0.994 | Quarterly |
| Elast. of substitution | $\varepsilon$ | 6 | Across intermediate goods |
| Price stickiness | $\chi_p$ | 0.75 | Calvo parameter |
| Wage rigidity | $\chi_w$ | 0.925 | Partial adjustment |
| Steady-state markup | $\lambda$ | 1.2 | |
| Policy-rule intercept | $\bar\pi$ | 0.01 | Targets 2% annual inflation |
| Policy-rule smoothing | $\rho_r$ | 0.84 | |
| Taylor-rule inflation loading | $r_\pi$ | 1.60 | |
| Taylor-rule output loading | $r_y$ | 0.028 | |
| Mean pessimism | $\mu_\theta$ | 5.64 | |
| Persistence of $\theta$ | $\rho_\theta$ | 0.714 | |
| Volatility of $\theta$ shock | $\sigma_\theta$ | 4.3 | |
| TFP persistence | $\rho_a$ | 0.840 | |
| TFP volatility | $100\sigma_a$ | 0.568 | |
| MP volatility | $100\sigma_r$ | 0.078 | |
| Job survival probability | $\rho$ | 0.89 | Separation rate $1-\rho=0.11$ |
| Matching efficiency | $\mu$ | 0.67 | |
| Matching-function curvature | $\nu$ | 0.72 | From {cite}`Shimer2005` |
| Worker bargaining weight | $\eta$ | 0.72 | From {cite}`Shimer2005` |
| Vacancy posting cost | $\kappa_v$ | 0.09 | |
| Unemployment benefit flow | $D$ | 0.57 | |

The calibration has three blocks.

Conventional steady-state targets pin down most parameters: a 1% annual real
return, a 20% markup with price spells of about three quarters, an 11%
quarterly job separation rate, a job-finding rate of 0.67, a flow value of
unemployment equal to 70% of wages, and steady-state labor market tightness
equal to one.

The TFP process is estimated from measured TFP, and the wage rigidity and
monetary policy parameters are chosen so that responses of inflation and
unemployment to TFP and monetary policy shocks match VAR evidence.

Matching these responses matters because the wedge formula above makes belief
distortions proportional to covariances of shocks with continuation values:
the model has to propagate fundamental shocks correctly before its belief
wedges can be meaningfully compared with the survey data.

The belief-shock parameters are read off the survey wedges:
$\rho_\theta = 0.714$ is the autocorrelation of the first principal component
of the wedges, and $\mu_\theta$ and $\sigma_\theta$ fit the means and
volatilities of the two wedges.

### A self-contained linear surrogate

Solving the full structural model requires the series expansion method
described in the appendix.

For computation in the main text, we use a small reduced-form vector
autoregression that is calibrated to reproduce the main benchmark moments in
the moment table reported below and the qualitative shape of the
belief-shock impulse responses:

$$

s_{t+1} = A\, s_t + B\, \epsilon_{t+1},

$$

where $s_t = (u_t, \pi_t, y_t, \theta_t, a_t)'$ collects unemployment,
inflation, output, the belief shock, and TFP, and
$\epsilon_{t+1} \sim N(0, I_3)$ contains the three structural shocks.

The coefficient matrices $A$ and $B$ are not the full structural solution.

A full structural solution would obtain them from equilibrium conditions:
households and firms distort probabilities according to continuation values,
and those distorted beliefs feed back into consumption, price setting, vacancy
posting, and wages.

Here we compress those equilibrium channels into a few linear loadings.

The loadings are chosen so that the volatilities of unemployment, inflation,
and output, computed from the Lyapunov equation, match the benchmark and
no-belief-shock columns of the moment table in the next section.

The volatility match is essentially exact; conditional objects, such as
impact magnitudes and correlations with output, are only qualitatively right.

This surrogate is useful for transparent computations, but it should not be
used to analyse the diagnostic variants such as "only $\theta_t$", no TFP
shocks, or rational firms.

For those variants, the relevant structural moments are reported directly
below.

We index the five state variables with named constants, so that later code can
refer to, say, the belief shock as `I_THETA` rather than a bare number.

```{code-cell} ipython3
# Position of each variable in the state vector s_t
I_U, I_PI, I_Y, I_THETA, I_A = 0, 1, 2, 3, 4
```

The factory `create_nk_model` builds the transition matrix $A$ and the shock
loadings $B$ from the calibration values stated above.

The belief shock $\theta_t$ and TFP $a_t$ follow AR(1) processes; the
endogenous variables inherit their own persistence and load on $\theta_t$,
$a_t$, and the monetary policy shock.

In the structural model, the effects of $\theta_t$ arise because pessimism
changes the subjective distribution of the fundamental shocks.

In the surrogate, these effects are summarized by the coefficients
$\phi_{u\theta}$, $\phi_{\pi\theta}$, and $\phi_{y\theta}$.

```{code-cell} ipython3
class NKModel(NamedTuple):
    A: np.ndarray    # state transition matrix
    B: np.ndarray    # shock loadings (columns: w_θ, w_a, w_r)
    c_u: float       # loading of the unemployment wedge on θ
    c_π: float       # loading of the inflation wedge on θ


def create_nk_model():
    """Build the pedagogical reduced-form NK model (state and shock matrices)."""
    # Exogenous-process parameters from the calibration table above.
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

For the unconditional moments we simulate the model and, separately, solve the
discrete Lyapunov equation $\Sigma = A\Sigma A' + BB'$ for the stationary
covariance.

Passing `include_θ_shock=False` zeros out the belief shock, which isolates the
contribution of the TFP and monetary policy shocks.

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
nk = create_nk_model()
```

## Quantitative results

### Benchmark moments and model targets

The table below collects the empirical moments and model moments most relevant
for this lecture.

All values are percentages or percentage points; inflation is annualized and
output is detrended.

| Moment | Data | Benchmark | No $\theta_t$ | Only $\theta_t$ |
|---|---:|---:|---:|---:|
| Mean inflation wedge | 1.22 | 0.90 | 0.00 | 0.00 |
| Mean unemployment wedge | 0.52 | 0.55 | 0.00 | 0.00 |
| Volatility of inflation wedge | 0.97 | 0.73 | 0.00 | 0.00 |
| Volatility of unemployment wedge | 0.57 | 0.45 | 0.00 | 0.00 |
| Volatility of inflation | 1.37 | 1.16 | 0.99 | 0.00 |
| Volatility of output | 2.00 | 2.22 | 1.55 | 0.00 |
| Volatility of unemployment | 1.70 | 1.39 | 0.55 | 0.00 |
| Corr. inflation wedge, output | −0.30 | −0.67 | 0.00 | 0.00 |
| Corr. unemployment wedge, output | −0.49 | −0.67 | 0.00 | 0.00 |
| Corr. inflation, output | 0.10 | −0.82 | −0.85 | 0.00 |
| Corr. unemployment, output | −0.87 | −0.74 | −0.33 | 0.00 |

Two lessons are important.

First, shutting down the belief shock returns the familiar unemployment
volatility puzzle: unemployment volatility falls from 1.39 in the benchmark to
0.55, far below the data value of 1.70.

Second, the "only $\theta_t$" column is zero in the structural model.

If TFP and monetary policy uncertainty are absent, there is no payoff-relevant
uncertainty for pessimistic agents to distort, so time variation in
$\theta_t$ alone does not move the economy.

This is why the benchmark emphasizes the interaction between belief shocks and
fundamental shocks, especially TFP shocks.

This point is also why the reduced-form surrogate should be read carefully.

In the structural model, $\theta_t$ changes how agents weight fundamental
shocks.

In the surrogate, that interaction is compressed into direct loadings from
$\theta_t$ to unemployment, inflation, and output.

Thus the surrogate is useful for the benchmark and no-belief-shock comparisons,
but the diagnostic columns in the table are structural results rather than
additional simulations of the surrogate.

### Impulse responses to the belief shock

A positive innovation to $\theta_t$ makes households more pessimistic.

The
mechanism works this way:

1. Households put more subjective probability on futures with low
   productivity, tight monetary policy, and continued pessimism, so they cut
   current consumption to smooth the bad times they anticipate.
2. Firms, who share these beliefs, mark down their valuation of new matches;
   vacancy posting and the job-finding rate fall, output falls, and
   unemployment rises.
3. Price setters expect lower future productivity and hence higher marginal
   costs, so they resist cutting prices despite the fall in demand.
4. The belief wedges jump on impact, then decay with the persistence
   $\rho_\theta = 0.714$.

In the structural benchmark, a one-standard-deviation belief shock raises
unemployment by about one percentage point and lowers output by about one
percent.

Inflation rises briefly and then falls, leaving a roughly zero cumulative
10-quarter response.

The contraction is sizable: the cumulative 10-quarter output loss is about
two-thirds of the loss following a one-standard-deviation drop in TFP.

The reduced-form impulse responses below are calibrated to reproduce these
signs and the volatility scale in the moment table above.

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

This figure has six panels.

The first row plots the macroeconomic responses of unemployment, inflation,
and output to a one-standard-deviation innovation in the belief shock.

The second row plots the belief shock itself and the two implied survey wedges.

The impulse responses show that a belief shock:

* Raises unemployment persistently.
* Raises inflation temporarily, with the response gradually decaying back to
  zero in this reduced-form representation.
* Lowers output, so the shock looks like a pessimistic recessionary force.
* Generates belief wedges for both unemployment and inflation that closely
  mirror the dynamics of $\theta_t$ itself --- consistent with the one-factor
  structure.

The structural model contains one additional object that the surrogate does not
capture: impulse responses under the **subjective** measure.

After a positive $\theta_t$ innovation, pessimistic agents behave as if future
TFP shocks are worse, monetary policy shocks are tighter, and future
pessimism is more persistent than under the data-generating measure.

Agents distort the TFP shock more than the monetary policy shock, because TFP
matters more for their continuation values.

Under the subjective measure, households expect consumption to fall further
and recover far more slowly than it actually does, and they expect inflation
to remain persistently high.

That subjective correlation structure is what makes both unemployment and
inflation forecasts biased upward.

### The unemployment volatility puzzle

A long-standing challenge for New Keynesian models is that standard TFP and
monetary policy shocks generate far too little unemployment volatility
({cite}`Shimer2005`).

In the no-belief-shock economy, TFP and monetary policy shocks produce
unemployment volatility of only 0.55, compared to 1.70 in the data.

Adding the belief shock substantially closes the gap:

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

# Data values from the benchmark moment table above.
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

The bar chart compares three standard deviations: the no-belief-shock economy,
the benchmark economy, and the data.

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
TFP and monetary-policy shocks do not move $\theta_t$ directly in this
pedagogical representation.

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
H = -(V_x \psi_w)',
\qquad
\bar H = -\bar\theta\,\bar x\,(V_x \psi_w)'.

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
H_sc = np.array([[-model.Vx * model.σ_x]])  # -(Vx ψ_w)'
x_bar_sc = 1.0
H_bar_sc = -model.μ_θ * x_bar_sc * np.array([[model.Vx * model.σ_x]])

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

Guessing $V_{1t} = V_x x_{1t} + V_q$ and matching coefficients yields
the **Riccati equation for $V_x$**:

$$

V_x = u_x - \frac{\beta}{2}\, V_x \psi_w \psi_w' V_x' \bar\theta
  + \beta\, V_x \psi_x,

$$

and the constant

$$

V_q = u_q - \frac{\beta}{2}\,\bar\theta\, \bar x\,
  V_x \psi_w \psi_w' V_x' + \beta\, V_x \psi_q + \beta V_q.

$$

The Riccati equation is quadratic in $V_x$.

For the stationary scalar case it reduces to

$$

a\, V_x^2 + b\, V_x + c = 0,
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
N\!\left(-\bar\theta(\bar x + x_{1t})(V_x \psi_w)',\; I_k\right).

$$

Belief wedges for the state vector follow immediately:

$$

\Delta_t^{(1)} = \tilde E_t[x_{t+1}] - E_t[x_{t+1}]
= \psi_w\, \tilde E_t[w_{t+1}]
= -\bar\theta(\bar x + x_{1t})(\psi_w \psi_w') V_x'.

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
  (V_x \psi_w)'\, \bar\theta
\Bigr\}.

$$

These equations are solved jointly with the Riccati equation for $V_x$.

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
for $V_f$, and the belief distortion matrix becomes

$$

\mathbb{E} = \operatorname{stack}\Bigl\{
  \sigma_i\bigl[
    g_{x^+}\psi_{xf}\sigma_f^2(V_f + V_x\psi_{xf})
    + (g_{x^+}\psi_w + g_{w^+})\psi_w' V_x'
  \bigr]^i
\Bigr\}\bar\theta_f.

$$

The algorithm therefore decomposes cleanly into two stages:

1. **Stage 1 (rational-expectations block)**: solve for
   $\psi_x$, $\psi_w$ using the standard Blanchard–Kahn method; these are
   *unaffected* by the belief shock.

2. **Stage 2 (belief distortion block)**: given $\psi_x, \psi_w, V_x$,
   iterate backward to convergence to find $\psi_{xf}$, $V_f$, and
   $\mathbb{E}$.

This separation is a major practical advantage: existing rational-expectations
solvers can be used for Stage 1 with only a wrapper for Stage 2.

In the scalar endowment model, Stage 2 is simple because the value function is
the only forward-looking object.

The code below carries out the two computable ingredients: it solves the
Riccati equation by iteration starting from the rational-expectations value,
as the paper does, and then reads off the subjective transition coefficients
from the re-centred shocks.

```{code-cell} ipython3
# Stage 1: rational-expectations objects of the scalar endowment model
ψ_x_re = model.ρ_x
ψ_w_re = model.σ_x
u_x = 1 - model.β

# Stage 2 value block: iterate on the Riccati equation
Vx_iter = u_x / (1 - model.β * ψ_x_re)      # start at the RE value
for it in range(1, 200):
    Vx_new = (u_x + model.β * ψ_x_re * Vx_iter
              - 0.5 * model.β * model.μ_θ * ψ_w_re**2 * Vx_iter**2)
    if abs(Vx_new - Vx_iter) < 1e-15:
        break
    Vx_iter = Vx_new

print(f"Riccati by iteration ({it} steps):  Vx = {Vx_iter:.10f}")
print(f"Riccati by quadratic formula:     Vx = {model.Vx:.10f}")

# Subjective transition coefficients implied by the re-centred shocks
ψ_x_subj = ψ_x_re - model.μ_θ * Vx_iter * ψ_w_re**2
ψ_q_subj = -model.μ_θ * 1.0 * Vx_iter * ψ_w_re**2   # x̄ normalized to 1

print(f"\nObjective persistence:   ψ_x  = {ψ_x_re:.8f}")
print(f"Subjective persistence:  ψ̃_x  = {ψ_x_subj:.8f}")
print(f"Subjective drift:        ψ̃_q  = {ψ_q_subj:.2e}")
```

The two solution methods agree.

For this consumption-type state ($V_x > 0$), the subjective persistence is
slightly *below* the objective one and the subjective drift is negative: the
pessimist expects good states to fade and bad outcomes to arrive.

For an unemployment-type state ($V_x < 0$), both signs flip, so subjective
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

V_t^* = \max_{\{y_{t+j}\}} \min_{\substack{m_{t+j}>0 \\ E_{t+j-1}[m_{t+j}]=1}}
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

V_t^* = \max_{y_t} \min_{\substack{m_{t+1}>0 \\ E_t[m_{t+1}]=1}}
  u_t + \frac{\beta}{\theta_t} E_t[m_{t+1} \log m_{t+1}]
  + \beta E_t[m_{t+1} V_{t+1}^*].

$$

```{code-cell} ipython3
θ_path = np.array([3.0, 5.64, 8.0, 12.0])   # rising pessimism scenario

def one_period_entropy(θ, Vx, σ_x):
    """
    Entropy E_t[m_{t+1} log m_{t+1}] under the optimal distortion
    for Gaussian shocks: = (1/2) * (θ * Vx * σ_x)^2.
    """
    return 0.5 * (θ * Vx * σ_x) ** 2

print("Effect of time-varying θ on entropy and belief wedge:")
print(f"{'θ_t':>8}  {'H_t (entropy)':>16}  {'Δ(x) = σ_x ν_t (pp)':>22}")
print('-' * 52)
for th in θ_path:
    H = one_period_entropy(th, model.Vx, model.σ_x)
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
$-\theta_t (V_x \psi_w)'$.

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

1. Compute $V_x$ under rational expectations and under pessimism
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

**Part (a)** --- Under rational expectations ($\theta = 0$):

$$

V_x^{RE} = \frac{u_x}{1 - \beta \rho_x}
         = \frac{1 - \beta}{1 - \beta \rho_x}.

$$

```{code-cell} ipython3
β_ex = 0.99
ρ_x_ex = 0.90
σ_x_ex = 0.01
μ_θ_ex = 4.0

Vx_re_ex = (1 - β_ex) / (1 - β_ex * ρ_x_ex)
print(f"V_x (rational expectations): {Vx_re_ex:.4f}")

m_ex = create_belief_model(β=β_ex, ρ_x=ρ_x_ex,
                           σ_x=σ_x_ex, μ_θ=μ_θ_ex)
print(f"V_x (with pessimism θ_bar={μ_θ_ex}):   {m_ex.Vx:.4f}")
```

**Part (b)** --- The belief wedge for consumption growth is

$$

\Delta_t^{(1)}(x)
= -\theta_t V_x \sigma_x^2.

$$

Since $V_x > 0$ and $\theta_t > 0$, the wedge is **negative**: pessimistic
agents underestimate consumption growth relative to the model.

**Part (c)** --- For unemployment, $u_x < 0$, so $V_x^u < 0$.

The belief wedge becomes

$$

\Delta_t^{(1)}(u)
= -\theta_t V_x^u \sigma_x^2 > 0

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

(a) Compute the fraction of unemployment variance explained by each of the
    three shocks.
(b) Show that the belief shock is the dominant driver of unemployment
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
this calibrated surrogate.

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

Solve the Riccati equation (`solve_Vx`) for a grid of
$\mu_\theta$ values from 0 (rational expectations) to 15.

For each value,
compute the steady-state (unconditional mean) belief wedge and the ratio of
robust to rational $V_x$.

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
Vx_vals = []
wedge_ss = []

Vx_re = (1 - 0.994) / (1 - 0.994 * 0.85)

for μ in μ_grid:
    m_temp = create_belief_model(μ_θ=μ)
    Vx_vals.append(m_temp.Vx)
    wedge_ss.append(belief_wedge(m_temp, μ) * 100)   # in pp

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle('Value sensitivity and steady-state wedge')

axes[0].plot(μ_grid, Vx_vals, color='steelblue', linewidth=2)
axes[0].axhline(Vx_re, color='grey', linestyle='--',
                label=f'RE value $V_x^{{RE}}={Vx_re:.3f}$')
axes[0].set_xlabel('mean pessimism $\\mu_\\theta$')
axes[0].set_ylabel('$V_x$')
axes[0].legend()

axes[1].plot(μ_grid, np.array(wedge_ss), color='firebrick', linewidth=2)
axes[1].set_xlabel('mean pessimism $\\mu_\\theta$')
axes[1].set_ylabel('steady-state wedge (pp)')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
```

The left panel plots the solved value-function slope $V_x$ against the mean
pessimism parameter $\mu_\theta$.

The dashed horizontal line is the rational-expectations benchmark.

The right panel plots the corresponding steady-state belief wedge.

As $\mu_\theta$ rises, the Riccati equation introduces an additional
curvature term that lowers $V_x$ (less marginal value to the current state)
because the agent effectively prices in the possibility of bad future
outcomes.

The steady-state consumption wedge becomes more negative, approximately
linearly in magnitude, since
$\Delta^{(1)} \propto -\mu_\theta V_x \sigma_x^2$ and $V_x$ is approximately
constant for small $\mu_\theta$.

```{solution-end}
```
