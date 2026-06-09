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
# Survey Data and Subjective Beliefs in Business Cycles

```{index} single: Subjective Beliefs; Business Cycles
```

## Overview

This lecture presents key ideas from {cite}`bhandari2025survey`, who study
whether household survey data on macroeconomic expectations can shed light on
business cycle dynamics.

Their central finding is that household forecasts of unemployment and inflation
exhibit **systematic upward biases** relative to professional forecasters and
model-consistent rational expectations.  These biases — which the authors call
*belief wedges* — are:

* **Persistent and countercyclical**: they are larger during recessions.
* **Positively correlated**: optimism/pessimism about unemployment and inflation
  move together.
* **One-factor in structure**: a single latent state accounts for most
  variation across wedges.

The paper interprets this evidence through the lens of
**robust preferences** ({cite}`HansenSargent2001`; {cite}`HansenSargent2008`).

A household that fears model misspecification behaves as if it tilts
probabilities toward bad outcomes.

When calibrated to the Michigan Survey of
Consumers (1982Q1–2019Q4), this mechanism yields a time-varying *belief shock*
that substantially reduces the well-known **unemployment volatility puzzle** —
the fact that standard New Keynesian models with only technology and monetary
policy shocks generate far too little unemployment volatility.

By the end of this lecture you will understand:

* How to define and measure belief wedges from household survey data.
* How robust preferences generate time-varying subjective beliefs.
* How belief distortions propagate through a linearised DSGE model.
* Why a calibrated belief shock helps resolve the unemployment volatility
  puzzle.

## Setup

```{code-cell} ipython3
import datetime

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

For any scalar variable $z_{t+1}$, the **one-period belief
wedge** is

$$

\Delta_t^{(1)}(z) \;=\; \tilde{E}_t[z_{t+1}] - E_t[z_{t+1}].

$$

A positive wedge means households are more pessimistic than the model predicts:
they expect $z_{t+1}$ to be higher than the model-consistent forecast.

For
unemployment and inflation this sign convention implies an upward bias.

In practice, {cite}`BhandariBorovickaHo2024` measure
$\tilde{E}_t[\cdot]$ from the Michigan Survey of Consumers, and
$E_t[\cdot]$ from a benchmark DSGE model estimated on the same data.

The
discrepancy is the wedge.

### Empirical facts

Using data from 1982Q1 to 2019Q4, the authors document:

| Statistic | Unemployment wedge | Inflation wedge |
|---|---|---|
| Mean | 0.52 pp | 1.22 pp |
| Standard deviation | 0.67 pp | 1.03 pp |
| Correlation with output gap | −0.49 | −0.30 |

Both wedges are **positive on average** (households are pessimistic) and
**countercyclical** (pessimism rises in recessions).

Moreover, the first
principal component of the joint wedge series explains **78.6%** of its
variation — a striking one-factor structure.

The following code simulates artificial wedge series that match these
moments, so we can visualise the key stylised facts before turning to theory.

```{code-cell} ipython3
# ---------------------------------------------------------------------------
# Simulate stylised belief-wedge time series calibrated to match the
# empirical moments in Bhandari, Borovicka, Ho (2025).
# ---------------------------------------------------------------------------

# Calibrated parameters (Table 1 of the paper)
μ_θ = 5.64   # mean of belief-shock parameter θ
ρ_θ = 0.714  # AR(1) persistence of θ
σ_θ = 4.3    # innovation volatility of θ (units of θ)

# Wedge loadings: Δᵤ = cᵤ θ,  Δπ = cπ θ  (c chosen to match the means)
c_u = 0.52 / μ_θ   # ≈ 0.0922 pp per unit of θ
c_π = 1.22 / μ_θ   # ≈ 0.2163 pp per unit of θ

T = 152   # 38 years × 4 quarters

# Simulate the belief-shock AR(1)
rng = np.random.default_rng(42)
θ = np.zeros(T)
θ[0] = μ_θ
for t in range(1, T):
    θ[t] = ((1 - ρ_θ) * μ_θ
            + ρ_θ * θ[t-1]
            + σ_θ * rng.standard_normal())

# Belief wedges (in percentage points)
wedge_u = c_u * θ
wedge_π = c_π * θ

# Generate quarters 1982Q1 – 2019Q4
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
# Show the one-factor structure: scatter of unemployment vs inflation wedge
fig, ax = plt.subplots()
sc = ax.scatter(wedge_u, wedge_π, c=range(T), cmap='RdYlGn_r',
                alpha=0.7, s=20)
plt.colorbar(sc, ax=ax, label='quarter (dark = recent)')
ax.set_xlabel('unemployment wedge (pp)')
ax.set_ylabel('inflation wedge (pp)')
corr = np.corrcoef(wedge_u, wedge_π)[0, 1]
ax.text(0.05, 0.93, f'correlation = {corr:.2f}',
        transform=ax.transAxes, fontsize=11)
plt.tight_layout()
plt.show()
```

The scatter plot reveals the strong positive correlation between the two
wedges.

Both series are high when the belief shock $\theta_t$ is high, and low otherwise.

This is the one-factor structure that motivates the
theoretical framework.

## A model of pessimism

### Robust preferences

Why would households have systematically biased beliefs?

One disciplined answer comes from **robust control** or **multiplier preferences**
({cite}`HansenSargent2001`, {cite}`HansenSargent2008`).

An agent who fears that her reference model may be misspecified solves

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

The scalar $\theta_t > 0$
controls the *degree* of concern for misspecification: larger $\theta_t$ means
more pessimism.

The inner minimisation has a closed-form solution:

$$

m_{t+1}^* \;=\;
\frac{\exp(-\theta_t V_{t+1})}{E_t[\exp(-\theta_t V_{t+1})]}.

$$

Since $m_{t+1}^*$ assigns higher weight to states where $V_{t+1}$ is low (bad
outcomes), pessimistic agents effectively over-weight recessions in their
probability assessments.

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

So the belief wedge equals the covariance between the distorted likelihood
ratio and the variable of interest.

When $V_{t+1}$ is high in states where
$z_{t+1}$ is also high, $m_{t+1}^*$ will be low in those states, making the
covariance negative — i.e.\ the agent *underestimates* good-state variables.

For unemployment (which varies inversely with good economic outcomes), the
wedge is positive: pessimists expect higher unemployment than the model predicts.

### Illustration: optimal belief distortion

To see the mechanism concretely, consider an **endowment economy** with a
scalar log-consumption state $x_t$ and dynamics

$$

x_{t+1} = \rho_x x_t + \sigma_x w_{t+1}, \qquad w_{t+1} \sim N(0,1).

$$

With log utility, the continuation value is linear: $V_t = V_x x_t + V_q$.


Under the objective measure, $x_{t+1}$ is normal with mean $\rho_x x_t$ and
standard deviation $\sigma_x$.

The distorted measure $m_{t+1}^*$ shifts the mean of $w_{t+1}$ to

$$

\nu_t \;=\; -\theta_t (V_x \sigma_x).

$$

Hence, under the subjective measure, the innovation distribution becomes

$$

w_{t+1} \;\sim\; N\!\left(\nu_t,\; 1\right).

$$

The belief wedge for the state variable $x$ is

$$

\Delta_t^{(1)}(x) \;=\; \sigma_x \nu_t \;=\; -\theta_t V_x \sigma_x^2.

$$

When $V_x > 0$ (good consumption state is good) and $\theta_t > 0$
(pessimism), the wedge is negative — the agent *underestimates*
consumption growth.

For unemployment (enter with a negative sign in the
value function), the same pessimism generates a **positive** wedge.

```{code-cell} ipython3
# ---------------------------------------------------------------------------
# Illustrate the optimal belief distortion in the simple endowment economy.
# ---------------------------------------------------------------------------

class BeliefDistortionModel:
    """
    Simple scalar AR(1) endowment economy illustrating the robust-preference
    mechanism from Bhandari, Borovicka, Ho (2024).

    State dynamics:   x_{t+1} = ρ_x * x_t + σ_x * w_{t+1}
    Period utility:   u(x_t)  = (1 - β) * x_t  [log utility]
    Continuation value (linearised):  V_t = Vx * x_t + Vq

    Under the distorted measure the shock innovation has mean
        ν_t = -θ_t * Vx * σ_x
    which produces the belief wedge
        Δ_t^(1)(x) = σ_x * ν_t = -θ_t * Vx * σ_x^2.
    """

    def __init__(self, β=0.994, ρ_x=0.85, σ_x=0.005,
                 μ_θ=5.64, ρ_θ=0.714, σ_θ=4.3):
        self.β = β
        self.ρ_x = ρ_x
        self.σ_x = σ_x
        self.μ_θ = μ_θ
        self.ρ_θ = ρ_θ
        self.σ_θ = σ_θ
        self.Vx = self._solve_Vx()

    def _solve_Vx(self):
        """Solve the scalar Riccati equation for Vx."""
        u_x = 1.0 - self.β        # marginal utility of log consumption

        a = (self.β / 2.0) * self.σ_x**2 * self.μ_θ
        b = -(1.0 - self.β * self.ρ_x)
        c = u_x

        # Rational-expectations (θ=0) solution
        Vx_re = u_x / (1.0 - self.β * self.ρ_x)

        if abs(a) < 1e-14:          # essentially no pessimism
            return Vx_re

        disc = b**2 - 4.0 * a * c
        if disc < 0:
            return Vx_re            # fall back to RE if no real root

        r1 = (-b + np.sqrt(disc)) / (2.0 * a)
        r2 = (-b - np.sqrt(disc)) / (2.0 * a)
        return r1 if abs(r1 - Vx_re) < abs(r2 - Vx_re) else r2

    def belief_drift(self, θ):
        """Mean shift under subjective beliefs."""
        return -θ * self.Vx * self.σ_x

    def belief_wedge(self, θ):
        """One-period belief wedge for the state."""
        return self.σ_x * self.belief_drift(θ)

    def simulate_θ(self, T=200, seed=42):
        """Simulate the AR(1) belief-shock process."""
        rng = np.random.default_rng(seed)
        θ = np.zeros(T)
        θ[0] = self.μ_θ
        for t in range(1, T):
            θ[t] = ((1 - self.ρ_θ) * self.μ_θ
                    + self.ρ_θ * θ[t - 1]
                    + self.σ_θ * rng.standard_normal())
        return θ

    def simulate(self, T=200, seed=42):
        """Simulate belief wedge time series."""
        θ = self.simulate_θ(T, seed)
        return θ, self.belief_wedge(θ)


model = BeliefDistortionModel()
print(f"RE value of Vx:       {(1-model.β)/(1-model.β*model.ρ_x):.4f}")
print(f"Robust value of Vx:   {model.Vx:.4f}")
print(f"Belief drift at θ̄:   {model.belief_drift(model.μ_θ)*100:.4f} pp")
print(f"Belief wedge at θ̄:   {model.belief_wedge(model.μ_θ)*100:.4f} pp")
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: objective and subjective shock distributions
    name: fig-sbbc-shock-distributions
---
# Compare the objective (N(0,1)) and subjective shock distributions.
# The actual drift ν = -θ * Vx * σ_x is tiny on a unit-shock axis.
# We plot the standardised drift ν / σ_x instead.

θ_vals = [0, model.μ_θ, 2 * model.μ_θ]
labels = ['θ = 0  (rational)',
          f'θ = θ̄ = {model.μ_θ:.1f}  (mean)',
          f'θ = 2θ̄  (pessimistic)']
colors = ['black', 'steelblue', 'firebrick']

# ν_tilde = ν / σ_x = -θ * Vx.
ν_tilde = [-θ * model.Vx for θ in θ_vals]

x_grid = np.linspace(-4, 4, 500)

fig, ax = plt.subplots()
for μ, label, color in zip(ν_tilde, labels, colors):
    pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x_grid - μ)**2)
    ax.plot(x_grid, pdf, label=label, color=color, linewidth=2)

ax.axvline(0, color='grey', linestyle=':', linewidth=0.8)
ax.set_xlabel(
    'standardised innovation $(w_{t+1} - \\nu_t)$ '
    'with $\\nu_t = -\\theta_t V_x \\sigma_x$'
)
ax.set_ylabel('density')
ax.legend()
plt.tight_layout()
plt.show()

print("Mean shifts (in units of σ_x):")
for μ, label in zip(ν_tilde, labels):
    print(f"  {label:35s}  ν̃ = {μ:.4f}")
```

The figure shows how pessimism (higher $\theta_t$) shifts the perceived
distribution of future shocks to the left.

An agent with $\theta_t > 0$
believes bad shocks are more likely than they actually are.

## Linear approximation with belief distortions

### The perturbation method

For quantitative analysis, {cite}`bhandari2025survey` extend the standard
first-order perturbation method to accommodate time-varying belief distortions.

Let the state vector be $x_t \in \mathbb{R}^n$ with **objective** law of
motion

$$

x_{t+1} = \psi_q + \psi_x x_t + \psi_w w_{t+1}, \qquad
w_{t+1} \sim N(0, I_k).

$$

Under the optimal belief distortion the shocks are re-centred:

$$

w_{t+1} \;\sim\; N\!\left(- \theta_t (\bar{x} + x_{1t})
      (V_x \psi_w)',\; I_k\right),

$$

where $V_x$ is the row vector of first derivatives of the continuation value
and $\bar{x}$ is the non-stochastic steady state.

The perturbation is exact
to first order.

The resulting **belief wedge** for any variable $z$ with model-consistent
expected value $\bar{z}' x$ is

$$

\Delta_t^{(1)}(z)
\;=\; -\theta_t (\bar{x} + x_{1t})\, \bar{z}' (\psi_w \psi_w') V_x'.

$$

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
*time variation* in all belief wedges is driven by the **single scalar**
$\theta_t$.

The cross-sectional loadings $\bar{z}'(\psi_w\psi_w')V_x'$ are
fixed by the model's structural parameters.

This theoretical prediction
matches the empirical finding that one principal component explains 78.6%
of the joint variation in household forecast errors.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: wedge loadings implied by one factor
    name: fig-sbbc-one-factor-loadings
---
# ---------------------------------------------------------------------------
# Demonstrate the one-factor structure by computing wedges for two
# different variables as θ varies, holding structural parameters fixed.
# ---------------------------------------------------------------------------

θ_grid = np.linspace(0, 20, 200)

# Loading vector, proportional to bar_z' * ψ_w * ψ_w' * Vx'.
# Calibrated so that at mean θ the steady-state wedges match the data.
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

# Scatter of (wedge_u, wedge_π) with θ as the driver.
θ_sim = model.simulate_θ(T=400, seed=7)
wu_sim = loading_u * θ_sim
w_π_sim = loading_π * θ_sim
axes[1].scatter(wu_sim, w_π_sim, c=θ_sim, cmap='Blues', alpha=0.6, s=12)
axes[1].set_xlabel('unemployment wedge (pp)')
axes[1].set_ylabel('inflation wedge (pp)')

plt.tight_layout()
plt.show()
```

## A New Keynesian model with belief distortions

### Model description

{cite}`bhandari2025survey` embed the belief-distortion mechanism in a
New Keynesian model with a **search-and-matching** labour market
({cite}`Shimer2005`; {cite}`ChristianoEichenbaumTrabandt2016`).

The key
components are:

**Households** — Have log utility in consumption and disutility of hours.
They apply robust preferences (indexed by $\theta_t$) when forming
subjective forecasts.

**Firms** — Post vacancies and match with workers.  Calvo-style price
stickiness (parameter $\chi_p$) and wage stickiness ($\chi_w$) generate
standard New Keynesian Phillips curves.

**Monetary policy** — A Taylor rule that reacts to inflation and the output gap.

**Exogenous shocks** — Three shocks drive the model:

1. **Belief shock** $\theta_t$: an AR(1) capturing time-varying pessimism.
2. **TFP shock** $a_t$: standard technology shock.
3. **Monetary policy shock** $r_t$: i.i.d.\ deviation from the Taylor rule.

### Calibration

The model is calibrated to quarterly U.S. data, 1982Q1–2019Q4.

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Discount factor | $\beta$ | 0.994 | Quarterly |
| Elast. of substitution | $\varepsilon$ | 6 | Price markup |
| Price stickiness | $\chi_p$ | 0.75 | Calvo parameter |
| Wage stickiness | $\chi_w$ | 0.925 | Calvo parameter |
| Mean pessimism | $\mu_\theta$ | 5.64 | |
| Persistence of $\theta$ | $\rho_\theta$ | 0.714 | |
| Volatility of $\theta$ shock | $\sigma_\theta$ | 4.3 | |
| TFP persistence | $\rho_a$ | 0.840 | |
| TFP volatility | $100\sigma_a$ | 0.568% | |
| MP volatility | $100\sigma_r$ | 0.078% | |
| Matching elasticity | $\eta$ | 0.72 | Hosios condition |
| Worker bargaining | $\mu$ | 0.67 | |
| Job-separation rate | $\rho$ | 0.89 | Quarterly |

### Simplified reduced-form representation

We capture the model's linearised solution through a reduced-form
vector autoregression

$$

s_{t+1} = A\, s_t + B\, \epsilon_{t+1},

$$

where $s_t = (u_t, \pi_t, y_t, \theta_t, a_t)'$ collects unemployment,
inflation, output, the belief shock, and TFP, and
$\epsilon_{t+1} \sim N(0, I_3)$ contains the three structural shocks.

The coefficient matrices $A$ and $B$ are calibrated so that the
impulse-response functions reproduce the key moments reported in Table 2 and
Figure 7 of {cite}`bhandari2025survey`.

```{code-cell} ipython3
class ReducedFormNKModel:
    """
    Reduced-form linear model calibrated to Bhandari, Borovicka, Ho (2024).

    State vector s_t = [u_t, π_t, y_t, θ_t, a_t]
    Shocks: ε = [w_θ, w_a, w_r]

    Belief wedges:
        Δ_u = c_u * θ_t
        Δ_π = c_π * θ_t
    """

    # Index map for the state vector
    I_U, I_PI, I_Y, I_THETA, I_A = 0, 1, 2, 3, 4

    def __init__(self):
        # ---- exogenous-process parameters (Table 1) ----
        self.ρ_θ = 0.714
        self.σ_θ = 4.3
        self.ρ_a = 0.840
        self.σ_a = 0.00568
        self.σ_r = 0.00078

        # ---- wedge loadings on θ ----
        self.c_u = 0.52 / 5.64
        self.c_π = 1.22 / 5.64

        # ---- calibrated impact effects ----
        # State variables are stored in FRACTIONS (e.g. u=0.06 for 6%).
        # Display code converts: *100 for u,y and *400 for π.
        #
        # Belief shock targets from Figure 7.
        φ_u_θ = 0.009 / self.σ_θ
        φ_π_θ =  0.000875 / self.σ_θ
        φ_y_θ = -0.009 / self.σ_θ

        # TFP shock targets from Figure 7.
        φ_u_a = -0.40
        φ_π_a = -0.10
        φ_y_a = 1.20

        # Persistence of endogenous variables (quarterly, reduced-form)
        ρ_u = 0.35
        ρ_π = 0.50
        ρ_y = 0.35

        # ---- state transition matrix ----
        self.A = np.array([
            [ρ_u,  0,      0,     φ_u_θ,  φ_u_a ],   # unemployment
            [0,      ρ_π, 0,     φ_π_θ, φ_π_a],   # inflation
            [0,      0,      ρ_y, φ_y_θ,  φ_y_a ],   # output
            [0,      0,      0,     self.ρ_θ, 0   ],   # belief shock
            [0,      0,      0,     0,         self.ρ_a],  # TFP
        ])

        # ---- shock loading matrix ----
        # Columns: [w_θ, w_a, w_r].  All entries in fraction units.
        self.B = np.array([
            [0,                0,             0.5e-3 ],   # MP → u fraction
            [0,                0,            -0.1e-3 ],   # MP → pi fraction
            [0,                0,            -0.5e-3 ],   # MP → y fraction
            [self.σ_θ, 0,             0      ],   # θ innovation
            [0,                self.σ_a,  0      ],   # TFP innovation
        ])

    def irf(self, shock_idx, T=25):
        """
        Impulse-response function for a one-std-dev shock.

        Parameters
        ----------
        shock_idx : int
            0 = belief shock, 1 = TFP shock, 2 = monetary policy shock
        T : int
            Number of periods

        Returns
        -------
        resp : ndarray (5, T)   responses of state vector
        wu   : ndarray (T,)     unemployment wedge response
        wpi  : ndarray (T,)     inflation wedge response
        """
        n = self.A.shape[0]
        resp = np.zeros((n, T))
        s = self.B[:, shock_idx].copy()   # impact response

        for t in range(T):
            resp[:, t] = s
            s = self.A @ s

        wu = self.c_u * resp[self.I_THETA, :]
        w_π = self.c_π * resp[self.I_THETA, :]
        return resp, wu, w_π

    def simulate(self, T=200, seed=42):
        """Simulate the model for T periods."""
        rng = np.random.default_rng(seed)
        k = self.B.shape[1]
        s = np.zeros((self.A.shape[0], T))
        for t in range(1, T):
            s[:, t] = self.A @ s[:, t-1] + self.B @ rng.standard_normal(k)
        return s

    def unconditional_stds(self, include_θ_shock=True):
        """
        Unconditional standard deviations computed from the Lyapunov equation.
        """
        B_use = self.B.copy()
        if not include_θ_shock:
            B_use[:, 0] = 0.0          # zero out the belief shock
        Σ = solve_discrete_lyapunov(self.A, B_use @ B_use.T)
        return np.sqrt(np.diag(Σ))


nk = ReducedFormNKModel()
```

## Quantitative results

### Impulse responses to the belief shock

A positive innovation to $\theta_t$ makes households more pessimistic.

The
mechanism works this way:

1. Pessimistic households expect worse future outcomes and reduce consumption
   demand.
2. Lower demand raises unemployment and reduces output.
3. Upward wage pressure from labour-market tightness feeds into inflation.
4. The belief wedges jump on impact, then decay with the persistence
   $\rho_\theta = 0.714$.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: impulse responses to a belief shock
    name: fig-sbbc-belief-shock-irfs
---
T_irf = 25
periods = np.arange(T_irf)

resp_θ, wu_θ, w_π_θ = nk.irf(shock_idx=0, T=T_irf)

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

ylabels = ['unemployment (pp)', 'inflation (pp, ann.)', 'output (%)',
           'belief shock θ', 'unemployment wedge Δ(u) (pp)',
           'inflation wedge Δ(π) (pp)']
series = [resp_θ[0] * 100,   # unemployment in pp  (fraction × 100)
          resp_θ[1] * 400,   # inflation ann. pp  (quarterly frac × 400)
          resp_θ[2] * 100,   # output in %        (fraction × 100)
          resp_θ[3],         # belief shock θ
          wu_θ,              # unemp. wedge (pp): c_u × θ, already in pp
          w_π_θ]             # infl. wedge  (pp): c_π × θ, already in pp
colors = ['steelblue'] * 3 + ['purple', 'steelblue', 'darkorange']

for ax, ylabel, y, color in zip(axes, ylabels, series, colors):
    ax.plot(periods, y, color=color, linewidth=2)
    ax.axhline(0, color='grey', linewidth=0.7, linestyle='--')
    ax.set_ylabel(ylabel)
    ax.set_xlabel('quarters')

plt.tight_layout()
plt.show()
```

The impulse responses show that a belief shock:

* Raises unemployment persistently (peak effect around 1 pp).
* Raises inflation on impact, as higher pessimism tightens labour markets
  in the model.
* Generates belief wedges for both unemployment and inflation that closely
  mirror the dynamics of $\theta_t$ itself — consistent with the one-factor
  structure.

### The unemployment volatility puzzle

A long-standing challenge for New Keynesian models is that standard TFP and
monetary policy shocks generate far too little unemployment volatility
({cite}`Shimer2005`).

With only TFP and monetary policy shocks, the model
produces unemployment volatility of roughly 0.55%, compared to about 1.70%
in the data.

Adding the belief shock substantially closes the gap:

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: model and data volatility comparison
    name: fig-sbbc-volatility-comparison
---
std_full = nk.unconditional_stds(include_θ_shock=True)
std_no_θ = nk.unconditional_stds(include_θ_shock=False)

labels_vol = ['Unemployment', 'Inflation', 'Output']
idx = [nk.I_U, nk.I_PI, nk.I_Y]
scale = [100, 400, 100]    # convert to pp (unemployment, annualised inflation, %)

std_full_scaled = [std_full[i] * scale[j] for j, i in enumerate(idx)]
std_no_θ_scaled = [std_no_θ[i] * scale[j] for j, i in enumerate(idx)]

# Reference values from Table 2 of the paper
data_std = [1.70, 1.07, 2.23]    # data standard deviations

x = np.arange(len(labels_vol))
width = 0.25

fig, ax = plt.subplots()
ax.bar(x - width, std_no_θ_scaled, width, label='Model (no belief shock)',
       color='steelblue', alpha=0.7)
ax.bar(x, std_full_scaled, width, label='Model (with belief shock)',
       color='firebrick', alpha=0.7)
ax.bar(x + width, data_std, width, label='Data',
       color='grey', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(labels_vol)
ax.set_ylabel('standard deviation (% or pp, ann.)')
ax.legend()
plt.tight_layout()
plt.show()

print("Unconditional standard deviations:")
print(f"{'Variable':<18} {'No belief shock':>16} "
      f"{'With belief shock':>18} {'Data':>10}")
print('-' * 65)
for label, std_n, std_f, std_d in zip(labels_vol, std_no_θ_scaled,
                                       std_full_scaled, data_std):
    print(f"{label:<18} {std_n:>16.2f} {std_f:>18.2f} {std_d:>10.2f}")
```

The table confirms the key quantitative message: without the belief shock,
unemployment volatility is far below its empirical counterpart, but adding
the calibrated belief shock nearly doubles it, bringing the model much closer
to the data.

### Impulse responses to TFP and monetary policy shocks

For completeness, we also show responses to the other two shocks.

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: impulse responses to TFP and monetary policy shocks
    name: fig-sbbc-tfp-mp-irfs
---
resp_a,  _, _ = nk.irf(shock_idx=1, T=T_irf)   # TFP shock
resp_r,  _, _ = nk.irf(shock_idx=2, T=T_irf)   # Monetary policy shock

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
axes = axes.flatten()

series_a = [resp_a[0]*100, resp_a[1]*400, resp_a[2]*100]
series_r = [resp_r[0]*100, resp_r[1]*400, resp_r[2]*100]
var_ylabels = ['unemployment (pp)', 'inflation (pp, ann.)', 'output (%)']

for j, (ylabel, ya, yr) in enumerate(zip(var_ylabels, series_a, series_r)):
    # TFP
    axes[j].plot(periods, ya, color='steelblue', linewidth=2,
                 label='TFP shock')
    axes[j].axhline(0, color='grey', linewidth=0.7, linestyle='--')
    axes[j].set_ylabel(ylabel)
    axes[j].set_xlabel('quarters')
    axes[j].legend(loc='best')

    # Monetary policy
    axes[j+3].plot(periods, yr, color='darkorange', linewidth=2,
                   label='MP shock')
    axes[j+3].axhline(0, color='grey', linewidth=0.7, linestyle='--')
    axes[j+3].set_ylabel(ylabel)
    axes[j+3].set_xlabel('quarters')
    axes[j+3].legend(loc='best')

plt.tight_layout()
plt.show()
```

### Role of firms' beliefs

{cite}`bhandari2025survey` also study a variant in which **firms** hold
subjective beliefs.

The key channel is through the price-setting equation:
when firms fear that future demand will be weaker than the model predicts,
they raise prices today to protect margins, generating **higher inflation** on
impact.

This mechanism strengthens the comovement between the unemployment
wedge and the inflation wedge, which is needed to match the data.

The sign of the inflation response to a belief shock is therefore a
diagnostic: positive responses to pessimistic shocks require firms (not just
households) to hold subjective beliefs.

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
sim = nk.simulate(T=400, seed=99)
θ_sim = sim[nk.I_THETA]
y_sim = sim[nk.I_Y] * 100

# c_u and c_pi are in pp per unit θ, so the wedge is already in pp
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

# Confirm countercyclicality numerically
corr_u = np.corrcoef(y_sim, wu_sim_series)[0, 1]
corr_π = np.corrcoef(y_sim, w_π_sim_series)[0, 1]
print(f"Corr(output gap, unemployment wedge) = {corr_u:.3f}  "
      f"(data: −0.49)")
print(f"Corr(output gap, inflation wedge)    = {corr_π:.3f}  "
      f"(data: −0.30)")
```

The simulated correlations are negative, confirming the countercyclicality
predicted by the model and documented in the survey data.

## Extensions

The paper explores several important extensions:

**Heterogeneous beliefs** — A natural question is whether households and
firms should hold the same subjective beliefs.

The paper shows that
allowing firms to be *rational* while households are pessimistic changes
the inflation dynamics substantially.

This separation is identified from
the relative sizes of the unemployment and inflation wedges.

**Higher-order perturbation** — The first-order approximation provides
clean analytical formulas for belief wedges, but second-order effects
(precautionary savings, volatility feedback) matter for welfare analysis.

The paper develops second-order expansions and shows they affect the wedge
levels but not the one-factor structure.

**Idiosyncratic risk** — In the full model households face idiosyncratic
labour-market risk.

The interaction between aggregate pessimism and
uninsurable idiosyncratic shocks amplifies the effect of belief distortions
on precautionary savings, strengthening the unemployment channel.

## Appendix: the series expansion method

This appendix follows the Online Appendix of {cite}`BhandariBorovickaHo2024`
and fills in the computational and theoretical details underlying the
linearisation presented in the main lecture.

### Multi-period belief wedges

The main text focused on the one-period belief wedge
$\Delta_t^{(1)}(z)$.

The paper also uses $\tau$-period-ahead wedges
$\Delta_t^{(\tau)}(z) = \tilde E_t[z_{t+\tau}] - E_t[z_{t+\tau}]$,
which are needed to match survey respondents' longer-horizon forecasts.

Under linear dynamics

$$

x_{t+1} = \psi_q + \psi_x x_t + \psi_w w_{t+1},
\qquad w_{t+1} \sim N(0, I_k),

$$

the $\tau$-period-ahead expectation under the data-generating measure
satisfies the recursion

$$

G_x^{(\tau)} = \psi_x G_x^{(\tau-1)} + \psi_x,
\qquad
G_x^{(0)} = 0,

$$

so that $E_t[x_{t+\tau} - x_t] = G_x^{(\tau)} x_{1t} + G_0^{(\tau)}$.

Under the **subjective** measure, the mean of $w_{t+1}$ is shifted to
$\nu_t = H + HF x_{1t}$ (equation OA.1 of the appendix).  For the
stationary model the relevant identifications are

$$

F = \bar\theta,
\qquad
H = -(V_x \psi_w)',
\qquad
\bar H = -\bar\theta\,\bar x\,(V_x \psi_w)'.

$$

The subjective expectation then obeys a modified recursion

$$

\tilde G_x^{(\tau)} = \psi_x \tilde G_x^{(\tau-1)} + \psi_x
  + \bigl(\psi_w + \tilde G_x^{(\tau-1)}\psi_w\bigr) HF,

$$

and the $\tau$-period belief wedge is

$$

\Delta_t^{(\tau)} = \bigl(\tilde G_x^{(\tau)} - G_x^{(\tau)}\bigr) x_{1t}
  + \tilde G_0^{(\tau)} - G_0^{(\tau)}.

$$

The code below implements these recursions and shows how belief wedges grow
with the forecast horizon.

```{code-cell} ipython3
def compute_tau_wedge_loadings(ψ_x, ψ_w, H, H_bar, F, τ_max=20):
    """
    Compute tau-period belief wedge loadings using the recursions from
    Online Appendix OA.1 of Bhandari, Borovicka, Ho (2024).

    For simplicity we work with the scalar stationary case (all quantities
    are scalars or 1-d arrays).

    Returns
    -------
    wedge_const : array (tau_max,)   constant term of wedge  (G0_tilde - G0)
    wedge_slope : array (tau_max,)   x1t loading of wedge    (Gx_tilde - Gx)
    """
    n = ψ_x.shape[0]
    Gx = np.zeros((n, n))
    Gx_tild = np.zeros((n, n))
    G0 = np.zeros(n)
    G0_tild = np.zeros(n)

    wedge_const = np.zeros(τ_max)
    wedge_slope = np.zeros((τ_max, n))

    for τ in range(1, τ_max + 1):
        # data-generating measure
        new_Gx = (Gx + np.eye(n)) @ ψ_x
        new_G0 = G0 + (Gx + np.eye(n)) @ ψ_w @ np.zeros(ψ_w.shape[1])
        # (constant-shock term is zero under objective measure)

        # subjective measure
        new_Gx_tild = ((Gx_tild + np.eye(n)) @ ψ_x
                       + (Gx_tild + np.eye(n)) @ ψ_w @ (H @ F))
        new_G0_tild = (G0_tild
                       + ((Gx_tild + np.eye(n)) @ ψ_w @ H_bar).ravel())

        Gx, G0 = new_Gx, new_G0
        Gx_tild, G0_tild = new_Gx_tild, new_G0_tild

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
# -----------------------------------------------------------------
# Illustrate the wedge horizon profile in the simple endowment economy
# using the solved BeliefDistortionModel.
# -----------------------------------------------------------------

# Scalar model: ψ_x = [[ρ_x]], ψ_w = [[σ_x]]
ψ_x_sc = np.array([[model.ρ_x]])
ψ_w_sc = np.array([[model.σ_x]])
F_sc = np.array([[model.μ_θ]])           # θ-bar
H_sc = np.array([[-model.Vx * model.σ_x]])  # -(Vx ψ_w)'
H_bar_sc = model.μ_θ * model.ρ_x * np.array([[-model.Vx * model.σ_x]])

τ_max = 20
wc, ws = compute_tau_wedge_loadings(ψ_x_sc, ψ_w_sc, H_sc, H_bar_sc, F_sc, τ_max)

# For illustration, evaluate at x1t = +1 std dev of θ.
θ_std = model.σ_θ / np.sqrt(1 - model.ρ_θ**2)

fig, ax = plt.subplots()
τ_grid = np.arange(1, τ_max + 1)
ax.plot(τ_grid, wc * 100,
        color='steelblue', linewidth=2, label='Wedge at mean ($x_{1t}=0$)')
ax.plot(τ_grid, (wc + ws[:, 0] * θ_std) * 100,
        color='firebrick', linewidth=2, linestyle='--',
        label='Wedge at $+1\\,\\sigma_\\theta$ deviation')
ax.axhline(0, color='grey', linewidth=0.7, linestyle=':')
ax.set_xlabel('forecast horizon $\\tau$ (quarters)')
ax.set_ylabel('belief wedge (pp)')
ax.legend()
plt.tight_layout()
plt.show()
```

### The series expansion

{cite}`BhandariBorovickaHo2024` solve the full general-equilibrium model
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
penalisation in the perturbed recursion (OA.8) is
$\mathsf{q}/[\bar\theta(\bar x + x_{1t})]$,
which shrinks together with shock volatility.

This ensures that the
deterministic steady state does not collapse to the rational-expectations
solution.

Guessing $V_{1t} = V_x x_{1t} + V_q$ and matching coefficients yields
the **Riccati equation for $V_x$** (equation OA.20 of the appendix):

$$

V_x = u_x - \frac{\beta}{2}\, V_x \psi_w \psi_w' V_x' \bar\theta
  + \beta\, V_x \psi_x,

$$

and the constant

$$

V_q = u_q - \frac{\beta}{2}\,\bar\theta\, \bar x\,
  V_x \psi_w \psi_w' V_x' + \beta\, V_x \psi_q + \beta V_q.

$$

The Riccati equation is quadratic in $V_x$.  For the stationary scalar case it
reduces to

$$

a\, V_x^2 + b\, V_x + c = 0,
\qquad
a = \frac{\beta}{2}\sigma_x^2 \bar\theta,\quad
b = -(1 - \beta\rho_x),\quad
c = u_x.

$$

#### Shock distribution under subjective beliefs

Substituting the first-order expansion into the distortion formula
(OA.10) shows that the leading term $m_{0,t+1}$ is a lognormal change of
measure.  With Gaussian shocks, this is equivalent to shifting the
innovation mean (equation OA.12):

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

These equations (OA.17–OA.21) are solved jointly with the Riccati
equation for $V_x$.

Compared with the standard Blanchard–Kahn solution,
the only modification is the additive term $-\mathbb{E}$ that shifts the
characteristic matrix; when $\bar\theta = 0$ we recover the standard
rational-expectations solution.

#### The AR(1) belief shock as a special case

In the paper's application $\theta_t$ is itself an exogenous AR(1)
process (equation OA.22):

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

It is determined
by the backward-induction algorithm (equations OA.31–OA.34), which iterates
from a distant terminal date $T$ (where belief distortions vanish) back to
the present.

The continuation value in the $f$-direction satisfies a separate recursion
for $V_f$ (equation OA.29), and the belief distortion matrix becomes

$$

\mathbb{E} = \operatorname{stack}\Bigl\{
  \sigma_i\bigl[
    g_{x^+}\psi_{xf}\sigma_f^2(V_f + V_x\psi_{xf})
    + (g_{x^+}\psi_w + g_{w^+})\psi_w' V_x'
  \bigr]^i
\Bigr\}\bar\theta_f.

$$

The algorithm therefore decomposes cleanly into two stages:

1. **Stage 1 (rational-expectations block)**: solve (OA.24) and (OA.26) for
   $\psi_x$, $\psi_w$ using the standard Blanchard–Kahn method — these are
   *unaffected* by the belief shock.

2. **Stage 2 (belief distortion block)**: given $\psi_x, \psi_w, V_x$,
   iterate (OA.31–OA.34) backward to convergence to find $\psi_{xf}$,
   $V_f$, and $\mathbb{E}$.

This separation is a major practical advantage: existing rational-expectations
solvers can be used for Stage 1 with only a wrapper for Stage 2.

```{code-cell} ipython3
# -----------------------------------------------------------------
# Demonstrate the limiting Stage 2 fixed point in a stylised scalar economy.
#
# Setup:
#   - Endogenous state x, belief shock f (= θ_t)
#   - ψ_x (1x1), ψ_w (1x1) known from Stage 1
#   - Vx known from the Riccati equation
#   - Solve the first-order fixed point for Vf and ψ_xf
# -----------------------------------------------------------------

β = model.β
ρ_x = model.ρ_x
σ_x = model.σ_x
ρ_f = model.ρ_θ
σ_f = model.σ_θ
Vx = model.Vx

# Stage 1 objects (RE solution)
ψ_x_s1 = ρ_x
ψ_w_s1 = σ_x

# gx+ = β * (1 - β) in the simple log-utility endowment economy
# (partial derivative of marginal utility w.r.t. x_{t+1})
gx_plus = β * (1 - β)

θ_f = 1.0   # f is θ in the partitioned state.

# First-order scalar fixed point.
#
# The full nonlinear backward recursion is stable in the paper's full model,
# but the stripped-down scalar example can diverge because it lacks the
# stabilising equilibrium blocks.  We therefore solve the first-order limiting
# system directly.
denom = gx_plus * ψ_x_s1 - (1 - β)
E_const = (gx_plus * ψ_w_s1) * ψ_w_s1 * Vx * θ_f
penalty_const = (β * θ_f / 2.0) * (Vx * ψ_w_s1)**2

A_fp = np.array([
    [1 - β * ρ_f, -β * ρ_f * Vx],
    [0.0,              1 + gx_plus * ρ_f / denom],
])
b_fp = np.array([
    -penalty_const,
    E_const / denom,
])

Vf, ψ_xf = np.linalg.solve(A_fp, b_fp)

print("Stage 2 fixed point:")
print(f"  Vf      = {Vf:.6f}")
print(f"  ψ_xf    = {ψ_xf:.6f}  (impact of belief shock on endogenous state)")
print()
print("Interpretation: a one-unit rise in f_t changes x by ψ_xf =",
      f"{ψ_xf:.4f} per period.")
print("The steady-state wedge for x: Δ = ψ_w * ν_bar =",
      f"{σ_x * (-model.μ_θ * (Vx * σ_x)):.4f}")
```

### Sequence problem and dynamic consistency

The recursive formulation used throughout the lecture emerges from the
following sequence problem (Online Appendix OA.3).  Define the discounted
entropy functional

$$

\mathcal{E}_t \;=\; E_t \sum_{j=0}^{\infty} \beta^j
  \left[ M_{t,t+j} \frac{\beta}{\theta_{t+j}}
    E_{t+j}[m_{t+j+1} \log m_{t+j+1}]
  \right],

$$

where $M_{t,t+j} = \prod_{k=1}^j m_{t+k}$.  The agent's problem is

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
when $\theta_t$ is time-varying.  The recursive form is:

$$

\mathcal{E}_t = \frac{\beta}{\theta_t} E_t[m_{t+1} \log m_{t+1}]
  + \beta E_t[m_{t+1} \mathcal{E}_{t+1}].

$$

Under this penalty, the minimax inequality is an equality, and the value
function satisfies the recursive form stated in the main lecture:

$$

V_t^* = \max_{y_t} \min_{\substack{m_{t+1}>0 \\ E_t[m_{t+1}]=1}}
  u_t + \frac{\beta}{\theta_t} E_t[m_{t+1} \log m_{t+1}]
  + E_t[m_{t+1} V_{t+1}^*].

$$

```{code-cell} ipython3
# -----------------------------------------------------------------
# Illustrate the role of dynamic consistency by comparing two penalty
# specifications:
#   (a) Paper specification: E_t = (β/θ_t) * H_t + β * E[m*E_{t+1}]
#       where H_t = E_t[m_{t+1} log m_{t+1}]
#   (b) A myopic version that uses only the one-period entropy:
#       E_t^{myopic} = (β/θ_t) * H_t
#
# We compare the implied belief wedge as θ varies.
# -----------------------------------------------------------------

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
    bw = model.belief_wedge(th) * 100
    print(f"{th:>8.2f}  {H:>16.6f}  {bw:>22.4f}")

print()
print("The entropy penalty grows quadratically in θ,")
print("constraining the agent from distorting beliefs too heavily.")
```

## Exercises

```{exercise-start}
:label: sbbc_ex1
```

**Belief wedge sign**

In the simple endowment economy of the `BeliefDistortionModel`, suppose the state
variable is log consumption $x_t$ with $\rho_x = 0.90$, $\sigma_x = 0.01$,
$\beta = 0.99$.

(a) Compute $V_x$ under rational expectations and under pessimism
    $\mu_\theta = 4$.
(b) What is the sign of the belief wedge for consumption growth?
(c) If instead the agent forecasts unemployment (which enters the value
    function with a negative sign, so $u_x < 0$), what is the sign of the
    unemployment belief wedge?
```{exercise-end}
```

```{solution-start} sbbc_ex1
:label: sbbc_ex1_sol
:class: dropdown
```

**Part (a)** — Under rational expectations ($\theta = 0$):

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

m_ex = BeliefDistortionModel(β=β_ex, ρ_x=ρ_x_ex,
                              σ_x=σ_x_ex, μ_θ=μ_θ_ex)
print(f"V_x (with pessimism θ̄={μ_θ_ex}):   {m_ex.Vx:.4f}")
```

**Part (b)** — The belief wedge for consumption growth is

$$

\Delta_t^{(1)}(x)
= -\theta_t V_x \sigma_x^2.

$$

Since $V_x > 0$ and $\theta_t > 0$, the wedge is **negative**: pessimistic
agents underestimate consumption growth relative to the model.

**Part (c)** — For unemployment, $u_x < 0$, so $V_x^u < 0$.  The belief
wedge becomes

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

Using the `BeliefDistortionModel` class, vary $\rho_\theta$ from 0.3 to
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
---
mystnb:
  figure:
    caption: persistence and belief-wedge volatility
    name: fig-sbbc-persistence-volatility
---
ρ_vals = np.linspace(0.3, 0.95, 30)
wedge_stds = []

for ρ in ρ_vals:
    m_temp = BeliefDistortionModel(ρ_θ=ρ)
    θ_sim_temp = m_temp.simulate_θ(T=5000, seed=0)
    wedge_sim_temp = m_temp.belief_wedge(θ_sim_temp)
    wedge_stds.append(np.std(wedge_sim_temp))

fig, ax = plt.subplots()
ax.plot(ρ_vals, np.array(wedge_stds) * 100, color='steelblue', linewidth=2)
ax.set_xlabel('persistence $\\rho_\\theta$')
ax.set_ylabel('standard deviation of belief wedge (pp)')
plt.tight_layout()
plt.show()
```

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

Using the `ReducedFormNKModel` class:

(a) Compute the fraction of unemployment variance explained by each of the
    three shocks.
(b) Show that the belief shock is the dominant driver of unemployment
    fluctuations, while TFP is the dominant driver of output fluctuations.
```{exercise-end}
```

```{solution-start} sbbc_ex3
:label: sbbc_ex3_sol
:class: dropdown
```

```{code-cell} ipython3
# Variance decomposition via the Lyapunov equation, shock-by-shock

shock_names = ['Belief shock (θ)', 'TFP shock', 'MP shock']
var_labels = ['Unemployment', 'Inflation', 'Output']

nk2 = ReducedFormNKModel()

# Compute the variance of each variable attributable to each shock
n_states = nk2.A.shape[0]
var_by_shock = np.zeros((n_states, 3))

for j in range(3):
    B_j = np.outer(nk2.B[:, j], nk2.B[:, j])
    Σ_j = solve_discrete_lyapunov(nk2.A, B_j)
    var_by_shock[:, j] = np.diag(Σ_j)

# Total variance
var_total = var_by_shock.sum(axis=1)

# Print share of variance for key variables
print(f"{'Variable':<16}", *[f"{s:>20}" for s in shock_names])
print('-' * 77)
for i, label in zip([nk2.I_U, nk2.I_PI, nk2.I_Y], var_labels):
    shares = var_by_shock[i] / var_total[i] * 100
    print(f"{label:<16}", *[f"{s:>19.1f}%" for s in shares])
```

The belief shock accounts for the majority of unemployment variance, as
reported in the paper.

Technology shocks drive most of the output variance
(through their high persistence and direct effect on productivity).

Monetary
policy shocks play a smaller role for both variables.

```{solution-end}
```

```{exercise-start}
:label: sbbc_ex4
```

**Changing the degree of pessimism**

Solve the Riccati equation in the `BeliefDistortionModel` for a grid of
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
---
mystnb:
  figure:
    caption: value sensitivity and steady-state wedge
    name: fig-sbbc-pessimism-riccati
---
μ_grid = np.linspace(0, 15, 100)
Vx_vals = []
wedge_ss = []

Vx_re = (1 - 0.994) / (1 - 0.994 * 0.85)

for μ in μ_grid:
    m_temp = BeliefDistortionModel(μ_θ=μ)
    Vx_vals.append(m_temp.Vx)
    wedge_ss.append(m_temp.belief_wedge(μ) * 100)   # in pp

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(μ_grid, Vx_vals, color='steelblue', linewidth=2)
axes[0].axhline(Vx_re, color='grey', linestyle='--',
                label=f'RE value $V_x^{{RE}}={Vx_re:.3f}$')
axes[0].set_xlabel('mean pessimism $\\mu_\\theta$')
axes[0].set_ylabel('$V_x$')
axes[0].legend()

axes[1].plot(μ_grid, np.array(wedge_ss), color='firebrick', linewidth=2)
axes[1].set_xlabel('mean pessimism $\\mu_\\theta$')
axes[1].set_ylabel('steady-state wedge (pp)')

plt.tight_layout()
plt.show()
```

As $\mu_\theta$ rises, the Riccati equation introduces an additional
curvature term that lowers $V_x$ (less marginal value to the current state)
because the agent effectively prices in the possibility of bad future
outcomes.

The steady-state wedge grows approximately linearly in
$\mu_\theta$, since $\Delta^{(1)} \propto \mu_\theta V_x \sigma_x^2$ and
$V_x$ is approximately constant for small $\mu_\theta$.

```{solution-end}
```
