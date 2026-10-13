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

(policy_counterfactuals)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

```{index} single: python
```

# Policy Counterfactuals from Impulse Responses

## Overview

How would an economy respond to its shocks if policy followed a different rule from the one that generated the historical data used to estimate a model?

Lucas {cite}`Lucas1976` taught us that a reliable answer requires a model whose private-sector
decision rules change appropriately when the policy rule changes.

McKay and Wolf {cite}`McKayWolf2023` claim that, in a particular class of linear models, an econometrician can
construct such Lucas-critique-robust counterfactuals *without* writing down the model.

They claim that what the econometrician needs instead are what they call "dynamic causal effects" of **policy shocks** --- surprise
deviations from a prevailing policy rule, both contemporaneous and anticipated ("news") --- together
with the effects of the non-policy shock of interest under the prevailing policy rule.

This lecture puts their result to work in the context of a generalized version of the  market for a single good studied in
{doc}`supply_demand_var` and {doc}`induced_priors`.

The market in those lectures  has no policy of the type that concerns McKay and Wolf, so we add one here: a per-unit tax on the good that the government adjusts
according to a rule that leans against movements in the price.

In this lecture, we shall

* show that the market with a tax fits McKay and Wolf's framework, and that their key assumption is
  precisely the invariance of the *open-loop* dynamic supply and demand curves of
  {doc}`supply_demand_var`,
* verify their identification result: impulse responses to tax shocks under one rule deliver the
  economy's exact behavior under another rule,
* compare it with the older approach of Sims and Zha {cite}`SimsZha2006`, which imposes the new rule
  with a sequence of policy *surprises* and so falls foul of the Lucas critique, and examine a version of
  Leeper and Zha's {cite}`LeeperZha2003` defense of that approach for "modest" policy interventions,
* study how well the counterfactual can be approximated with evidence on only a few policy shocks,
* contrast the uncertainty about a counterfactual that a structural modeler inherits from a prior over
  parameters, as in {doc}`induced_priors`, with the precision that McKay and Wolf's sufficient statistics
  deliver,
* use the market to sort procedures into those that require **invertibility** and those that do not, and
* relate McKay and Wolf's approach to the "dynamic programming squared" response to the Lucas critique,
  and state what their counterfactuals take for granted.

The invertibility item connects this lecture to the theme of {doc}`supply_demand_var` and {doc}`induced_priors`,
in which the supply shock is *not* fundamental for prices and quantities.

We shall see that

* external instruments recover the supply-shock impulse response that no rotation of VAR innovations
  can,
* whether the supply shock is fundamental depends on the *policy rule* --- a version of the Lucas critique
  that applies to VAR identification itself --- because a price-stabilizing tax rule is observationally
  equivalent to making suppliers more flexible, and
* McKay and Wolf's route to counterfactual *second moments* through VAR innovations is exact only under
  invertibility, although in this market the error it makes otherwise turns out to be tiny.

### Two responses to the Lucas critique

It helps to place McKay and Wolf's contribution beside another modern response to the Lucas critique,
the one pursued in the "dynamic programming squared" lectures of this series, such as {doc}`calvo`,
{doc}`calvo_abreu`, {doc}`dyn_stack`, {doc}`opt_tax_recur`, and {doc}`dovis_accounting_mf`.

Both responses start from the same object: the private sector's forward-looking Euler equations, which
make today's decisions depend on the entire expected future path of policy.

In the dynamic programming squared lectures, those equations are **implementability constraints** that a
government takes into account when it *chooses* policy, and whether the government can commit to its
choices is part of the model.

McKay and Wolf instead *measure* the mapping from policy paths to outcomes that those constraints imply,
and they treat the policy rule as something that the analyst picks.

This difference tells us what McKay and Wolf's counterfactuals take for granted.

* **Credibility.** The counterfactual rule is announced at date 0, understood, believed, and kept forever,
  as if chosen by a Ramsey planner who can commit. The method describes the effects of a *credible* rule,
  not of an *announced* one, and it is silent about whether a government would want to keep the rule.
* **Belief-neutral policy shocks.** The policy shocks used to measure the effects of policy must not change
  private agents' beliefs about future policy beyond the path that they announce. In theories of
  sustainable plans, by contrast, a deviation from expected policy *does* change those beliefs.
* **Linearity.** Private behavior is linear and depends on policy only through the expected path of the
  instrument. That excludes environments in which the policy regime switches endogenously with the state
  of the economy, as in the model of Bocola, Chaumont, Dovis, and Kirpalani {cite}`DovisAccountingMFrevised`.
* **Rules, not institutions.** The counterfactuals concern alternative *rules*, not alternative
  *institutions* that determine how a government chooses its policy.

We return to these points, from a different angle and with examples, in {ref}`pc_dp2` below.

In addition to what's in Anaconda, this lecture uses the quantecon library.

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install --upgrade quantecon
```

We'll make these imports:

```{code-cell} python3
import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
from quantecon import LQ
```

## A market with a tax

We use the market of {doc}`supply_demand_var` with the same parameter values.

Its supply side --- competitive firms facing quadratic costs of adjusting output, with each firm taking
the market price as given --- is a close cousin of the model in
{doc}`Rational Expectations Equilibrium <intermediate:rational_expectations>`, where the "big $Y$,
little $y$" device plays the role that "big $X$, little $x$" plays in {doc}`supply_demand_var`.

```{code-cell} python3
h_s = h_d = 1.0
g_s, g_d = 10.0, 0.1
β = 1 / 1.05
a = np.array([1, .8, .6, .4, .2])       # a(L): coefficients a0..a4

# MA polynomials for the shocks (coefficients on L^0..L^3)
B_d = np.convolve(np.convolve([1, .6], [1, .4]), [1, .2])
B_s = np.convolve(np.convolve([1, -.8], [1, .4]), [1, .2])

σ_s, σ_d = np.sqrt(0.5), 2.0            # shock standard deviations
```

As in {doc}`induced_priors`, we work with a *favorable* supply shock $\tilde w_{st} = -w_{st}$ and a
*positive* demand shock $\tilde w_{dt} = -w_{dt}$, each one standard deviation in size, so that both raise
quantity.

The government levies a tax $\tau_t$ per unit of the good, paid by suppliers, so that suppliers receive
$p_t - \tau_t$ when demanders pay $p_t$; a negative $\tau_t$ is a subsidy.

The supplier now maximizes

$$
E_0\sum_{t=0}^{\infty}\beta^t\Big\{ (p_t - \tau_t) q_t - \tfrac{h_s}{2}q_t^2
- \tfrac{g_s}{2}(q_t - q_{t-1})^2 - s_t q_t \Big\} ,
$$

while the demander's problem is unchanged.

The two Euler equations become

```{math}
:label: pc-supply
-E_t\big\{\big[h_s + g_s(1-\beta L^{-1})(1-L)\big]q_t\big\} + p_t - \tau_t = s_t ,
```

```{math}
:label: pc-demand
-E_t\big\{\big[h_d + g_d\, a(\beta L^{-1})a(L)\big]q_t\big\} - p_t = d_t ,
```

and eliminating $p_t$ gives the market Euler equation $E_t\, \psi(L) q_t = -(s_t + d_t + \tau_t)$, with
$\psi(L) = h_s + h_d + g_s(1-\beta L^{-1})(1-L) + g_d\, a(\beta L^{-1})a(L)$, in which the tax enters
exactly like a shock.

The government sets the tax by the rule

```{math}
:label: pc-rule
\tau_t = -\phi\, p_t + \nu_t ,
\qquad
\nu_t = \sum_{\ell \ge 0} \nu_{\ell, t-\ell} .
```

With $\phi > 0$ the government subsidizes the good when its price is high and taxes it when its price is
low, which stabilizes the price.

The policy shock $\nu_{\ell, t-\ell}$ is a deviation from the rule at date $t$ that was announced at date
$t - \ell$: $\nu_{0,t}$ is a contemporaneous surprise, while $\nu_{\ell, t-\ell}$ with $\ell > 0$ is
**news** about future policy.

### The McKay–Wolf framework

McKay and Wolf write a linear economy as two blocks,

$$
H_w \mathbf w + H_x \mathbf x + H_z \mathbf z + H_\varepsilon \boldsymbol\varepsilon = 0 ,
\qquad
A_x \mathbf x + A_z \mathbf z = \boldsymbol\nu ,
$$

where boldface letters denote entire time paths, $\mathbf x$ collects observable outcomes, $\mathbf z$ the
policy instrument, $\boldsymbol\varepsilon$ the non-policy shocks, and $\boldsymbol\nu$ the full menu of
contemporaneous and news policy shocks.

Their key assumption is that the private-sector block --- the matrices $H$ --- does not depend on the
coefficients $(A_x, A_z)$ of the policy rule: policy affects private decisions only through the expected
path of the instrument.

Our market satisfies this assumption, with observables $(\mathbf q, \mathbf p)$, instrument
$\mathbf z = \boldsymbol\tau$, and no unobserved endogenous variables $\mathbf w$.

To keep notation light, from now on we stack the observables and the instrument into one vector of
paths, $\mathbf x = (\mathbf q, \mathbf p, \boldsymbol\tau)$, and write the policy rule as
$R\, \mathbf x = \boldsymbol\nu$.

The private-sector block consists of the Euler equations {eq}`pc-supply`–{eq}`pc-demand`, which do not
involve $\phi$.

These are the **open-loop** dynamic supply and demand curves of {doc}`supply_demand_var`, valid for an
*arbitrary* path of prices --- and now taxes --- that the agents take as given.

The **closed-loop** decision rules that obtain inside a rational expectations equilibrium do depend on
$\phi$, because the equilibrium price process does: that dependence is the Lucas critique.

McKay and Wolf's assumption would fail if, for example, agents had less information than the government
and had to infer something about the economy from the tax rate.

Then the coefficients of the policy rule would affect what agents learn from observing the tax, as in
the environments of {doc}`knowing_forecasts_of_others`, where decision makers extract signals about
hidden states from equilibrium prices and quantities.

In our market, agents see the shocks $w_{st}, w_{dt}$ and know the policy rule.

### Solving in sequence space

Because the economy is linear, the impulse response to a shock that occurs at date 0 is the
perfect-foresight path that follows it, with the shock's entire future consequences --- including any
announced future deviations from the rule --- known at date 0.

Following McKay and Wolf, we compute such paths by stacking the equations for $t = 0, 1, \ldots, T-1$
into one linear system.

We set $q_t = 0$ for $t < 0$ (the economy starts at its steady state) and for $t \ge T$ (impulse responses
have died out by date $T$).

This is the finite-horizon approach of {doc}`lu_tricks`: the stacked first-order conditions form one
banded linear system, and as $T$ grows its solution converges to the infinite-horizon solution that
{doc}`supply_demand_var` obtains by factoring the characteristic polynomial into stable and unstable
parts.

The terminal condition $q_t = 0$ for $t \ge T$ selects the *bounded* solution.

McKay and Wolf assume that both the baseline and the counterfactual rules deliver a unique bounded
equilibrium, and we check below that our answers do not depend on $T$.

The term $g_d a(\beta L^{-1}) a(L) q_t$ in {eq}`pc-demand` equals
$g_d \sum_{i=0}^4 \sum_{k=0}^4 a_i a_k \beta^i q_{t+i-k}$.

```{code-cell} python3
T = 200     # length of the stacked paths

def private_block(g_s, g_d, T=T, h_s=h_s):
    """
    Stack the Euler equations for t = 0, ..., T-1.

    Unknowns are ordered x = (q, p, τ), each a path of length T, and the
    equations read  H @ x = (s, -d).
    """
    Φ_s, Φ_d = np.zeros((T, T)), np.zeros((T, T))
    for t in range(T):
        Φ_s[t, t] += h_s + g_s * (1 + β)
        if t >= 1:
            Φ_s[t, t - 1] -= g_s
        if t + 1 < T:
            Φ_s[t, t + 1] -= g_s * β
        Φ_d[t, t] += h_d
        for i in range(5):
            for k in range(5):
                if 0 <= t + i - k < T:
                    Φ_d[t, t + i - k] += g_d * a[i] * a[k] * β**i
    I, O = np.eye(T), np.zeros((T, T))
    return np.block([[-Φ_s, I, -I],      # supply:  p - τ - Φ_s q = s
                     [Φ_d, I, O]])       # demand:  p + Φ_d q = -d

def rule_block(φ, T=T):
    "The rule τ = -φ p + ν written as  R @ x = ν."
    return np.hstack([np.zeros((T, T)), φ * np.eye(T), np.eye(T)])

def system_inverse(g_s, g_d, φ, h_s=h_s):
    "Inverse of the stacked system: x = inverse @ (s, -d, ν)."
    return np.linalg.inv(np.vstack([private_block(g_s, g_d, h_s=h_s), rule_block(φ)]))

def forcing(shock, T=T, σ_s=σ_s):
    "Right-hand side for a favorable supply shock (0) or a positive demand shock (1)."
    s, d = np.zeros(T), np.zeros(T)
    if shock == 0:
        s[:4] = -σ_s * B_s
    else:
        d[:4] = -σ_d * B_d
    return np.concatenate([s, -d, np.zeros(T)])

def split(x):
    "Split a stacked path into (q, p, τ)."
    return x[:T], x[T:2 * T], x[2 * T:]
```

To check that the truncation does not matter, we solve for the response to a demand shock with $T = 150$
and $T = 300$ under three very different rules and compare the first 40 periods.

```{code-cell} python3
for φ in [-0.5, 0.5, 10.0]:
    paths = []
    for T_check in [150, 300]:
        M_check = np.vstack([private_block(g_s, g_d, T_check), rule_block(φ, T_check)])
        paths.append(np.linalg.solve(M_check, forcing(1, T_check))[:40])
    print(f"φ = {φ:5}: max |T=150 - T=300| = {np.abs(paths[0] - paths[1]).max():.1e}")
```

Before going on, we check that with no tax ($\phi = 0$, $\boldsymbol\nu = 0$) the sequence-space solution
reproduces the optimal-linear-regulator solution of {doc}`supply_demand_var`, whose code we repeat in the
hidden cell below.

```{code-cell} python3
---
tags: [hide-input]
---
def companion_shift(n):
    "Companion matrix that shifts an n-vector of lagged shocks."
    A = np.zeros((n, n))
    A[1:, :-1] = np.eye(n - 1)
    return A

def solve_equilibrium(h_s, h_d, g_s, g_d, β, a, B_s, B_d):
    "The planner's optimal linear regulator from supply_demand_var."
    n_q = 4
    A_w = np.zeros((8, 8))
    A_w[:4, :4] = companion_shift(4)
    A_w[4:, 4:] = companion_shift(4)
    C_w = np.zeros((8, 2))
    C_w[0, 0], C_w[4, 1] = 1.0, 1.0
    n = n_q + 8
    A = np.zeros((n, n))
    A[1:n_q, 0:n_q - 1] = np.eye(n_q - 1)
    A[n_q:, n_q:] = A_w
    B = np.zeros((n, 1)); B[0, 0] = 1.0
    C = np.zeros((n, 2)); C[n_q:, :] = C_w

    def q_lag(k):
        e = np.zeros(n); e[k - 1] = 1.0; return e

    s_sel = np.zeros(n); s_sel[n_q:n_q + 4] = B_s
    d_sel = np.zeros(n); d_sel[n_q + 4:] = B_d
    q1 = q_lag(1)
    a_vec = sum(a[k] * q_lag(k) for k in range(1, 5))
    Q = np.array([[(h_s + h_d) / 2 + g_s / 2 + g_d / 2 * a[0]**2]])
    R = g_d / 2 * np.outer(a_vec, a_vec) + g_s / 2 * np.outer(q1, q1)
    R += 1e-5 * np.eye(n)
    N = (g_d * a[0] * a_vec / 2 - g_s * q1 / 2
         + (s_sel + d_sel) / 2).reshape(1, n)
    lq = LQ(Q, R, A, B, C, N=N, beta=β)
    P, F, _ = lq.stationary_values()
    F = np.asarray(F).reshape(1, n)
    A_F = A - B @ F
    F_q = -F
    E_q1 = -F @ A_F
    G_p = (s_sel + h_s * F_q[0] + g_s * (1 + β) * F_q[0]
           - g_s * q1 - g_s * β * E_q1[0]).reshape(1, n)
    return A_F, C, np.vstack([F_q, G_p])

A_F, C, G_z = solve_equilibrium(h_s, h_d, g_s, g_d, β, a, B_s, B_d)
C_tilde = -C @ np.diag([σ_s, σ_d])
M_inv0 = system_inverse(g_s, g_d, 0.0)
for k in range(2):
    q, p, _ = split(M_inv0 @ forcing(k))
    A_F_power, gap = C_tilde.copy(), 0.0          # A_F^j C_tilde
    for j in range(30):
        gap = max(gap, np.abs(G_z @ A_F_power[:, k] - [q[j], p[j]]).max())
        A_F_power = A_F @ A_F_power
    print(f"shock {k}: max |sequence space - regulator| over 30 periods = {gap:.1e}")
```

The two methods agree up to the tiny ridge that regularizes the regulator's Riccati equation.

## How the policy rule shapes the response to shocks

We take $\phi = 0.5$ as the **baseline rule** that has prevailed historically, and compare it with no
policy ($\phi = 0$) and with aggressive price stabilization ($\phi = 2$).

```{code-cell} python3
φ_A = 0.5                      # baseline rule
H_plot = 20                    # horizons to plot
names = ["quantity $q$", "price $p$", "tax $\\tau$"]
shock_names = ["supply shock $\\tilde w_s$", "demand shock $\\tilde w_d$"]

fig, axes = plt.subplots(3, 2, figsize=(12, 9))
for φ, ls in [(0.0, ':'), (φ_A, '-'), (2.0, '--')]:
    M_inv = system_inverse(g_s, g_d, φ)
    for k in range(2):
        paths = split(M_inv @ forcing(k))
        for i in range(3):
            axes[i, k].plot(paths[i][:H_plot], ls, lw=2, label=f"$\\phi = {φ}$")
for i in range(3):
    for k in range(2):
        axes[i, k].axhline(0, color='k', lw=0.5)
        axes[i, k].set_title(f"response of {names[i]} to {shock_names[k]}")
axes[0, 0].legend()
for ax in axes[-1]:
    ax.set_xlabel("periods")
plt.tight_layout()
plt.show()
```

A more aggressive rule damps the price response to a demand shock by subsidizing the good when its price
rises, and quantity rises by more.

Every one of these responses reflects how forward-looking suppliers and demanders adjust their plans to
the rule: the closed-loop decision rules change with $\phi$.

### Price stabilization as more flexible supply

There is a simple way to see what the rule does.

When $\tau_t = -\phi p_t$, suppliers receive $p_t - \tau_t = (1 + \phi) p_t$, so the supply Euler equation
{eq}`pc-supply` becomes

$$
-E_t\big\{\big[h_s + g_s(1-\beta L^{-1})(1-L)\big]q_t\big\} + (1+\phi)\, p_t = s_t .
$$

Dividing by $1 + \phi$ shows that, as far as $(q_t, p_t)$ are concerned, the market with the tax rule
behaves *exactly* like a market without a tax whose supplier has

$$
\tilde h_s = \frac{h_s}{1+\phi}, \qquad \tilde g_s = \frac{g_s}{1+\phi}, \qquad
\tilde s_t = \frac{s_t}{1+\phi} .
$$

To someone who observes only prices and quantities, a price-stabilizing rule is indistinguishable from
suppliers who are *more flexible* --- with lower costs of producing and of adjusting output --- and who
face smaller cost shocks.

(Someone who also observes the tax can of course tell the two markets apart.)

We confirm the equivalence numerically.

```{code-cell} python3
for φ in [0.5, 2.0]:
    gaps = []
    for shock in range(2):
        q1, p1, _ = split(system_inverse(g_s, g_d, φ) @ forcing(shock))
        q2, p2, _ = split(system_inverse(g_s / (1 + φ), g_d, 0.0, h_s=h_s / (1 + φ))
                          @ forcing(shock, σ_s=σ_s / (1 + φ)))
        gaps.append(max(np.abs(q1 - q2).max(), np.abs(p1 - p2).max()))
    print(f"φ = {φ}: max |rule economy - flexible-supply economy| = {max(gaps):.1e}")
```

The equivalence explains a feature of the previous figure: under a more aggressive rule, quantity
responds more strongly to a demand shock, just as it would if suppliers could adjust output more easily.

It is, moreover, a compact statement of the Lucas critique in our market.

An econometrician who fit the dynamic supply curve to data generated under the rule, ignoring the tax,
would estimate the cost parameters $h_s/(1+\phi)$ and $g_s/(1+\phi)$ rather than $h_s$ and $g_s$, and
would mispredict what happens if the rule changes.

Later on, the same equivalence will explain why the policy rule affects whether a VAR can recover the
supply shock.

## McKay and Wolf's identification result

Their result combines two ingredients measured under the baseline rule: the responses to policy shocks,
which we compute first, and the response to the non-policy shock of interest.

### The causal effects of policy shocks

A policy shock $\nu_{\ell,0}$ announces at date 0 that the tax will deviate from the rule at date $\ell$.

Column $\ell$ of the matrix $\Theta_{\nu}$ below collects the responses of $(q, p, \tau)$ to such a shock
of size one under the baseline rule.

```{code-cell} python3
M_inv_A = system_inverse(g_s, g_d, φ_A)
Θ_ν = M_inv_A[:, 2 * T:]          # 3T x T: responses to news at every horizon

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ℓ, color in zip([0, 4, 8], ['C0', 'C1', 'C2']):
    paths = split(Θ_ν[:, ℓ])
    label = "surprise ($\\ell = 0$)" if ℓ == 0 else f"news about date {ℓ}"
    for i in range(3):
        axes[i].plot(paths[i][:H_plot], color=color, lw=2, label=label)
for i in range(3):
    axes[i].axhline(0, color='k', lw=0.5)
    axes[i].set_title(f"response of {names[i]}")
    axes[i].set_xlabel("periods")
axes[0].legend()
plt.tight_layout()
plt.show()
```

News of a future tax increase lowers quantity well *before* the tax rises: suppliers, who find it costly
to adjust output, begin cutting it as soon as they learn of the tax.

These anticipation effects are what make the policy *path*, and not just its current value, matter for
private decisions.

### Proposition 1

Let $\mathbf x_A(\varepsilon)$ be the response of $\mathbf x = (\mathbf q, \mathbf p, \boldsymbol\tau)$ to a
non-policy shock $\varepsilon$ under the baseline rule, and consider a counterfactual rule
$\tau_t = -\tilde\phi\, p_t$, written as $\tilde R\, \mathbf x = 0$.

McKay and Wolf's Proposition 1 says that the counterfactual response is given by the formula below,
where, following their notation, the subscript $A$ marks the baseline rule and $\tilde A$ the
counterfactual rule $\tilde R$:

```{math}
:label: pc-mw
\mathbf x_{\tilde A}(\varepsilon) = \mathbf x_A(\varepsilon) + \Theta_\nu \tilde{\boldsymbol\nu},
\qquad\text{where } \tilde{\boldsymbol\nu} \text{ solves }
\tilde R\,\big[\mathbf x_A(\varepsilon) + \Theta_\nu \tilde{\boldsymbol\nu}\big] = 0 .
```

The logic is that suppliers and demanders care only about the expected path of the tax.

A date-0 combination of contemporaneous and news shocks $\tilde{\boldsymbol\nu}$ can produce *any* expected
tax path, in particular the one that the counterfactual rule would produce.

Because the shocks are all announced at date 0, the new rule holds not just *ex post* along the path but
also *ex ante*, in agents' expectations from date 0 on.

The econometrician needs $\mathbf x_A(\varepsilon)$ and $\Theta_\nu$ --- objects that can be estimated
from data generated under the baseline rule --- but not the model's equations, nor even the baseline rule.

```{code-cell} python3
def mw_counterfactual(M_inv_base, x_A, φ_new):
    "McKay-Wolf Proposition 1: enforce τ = -φ_new p with date-0 policy shocks."
    Θ = M_inv_base[:, 2 * T:]
    R_new = rule_block(φ_new)
    ν = np.linalg.solve(R_new @ Θ, -R_new @ x_A)
    return x_A + Θ @ ν, ν
```

### The Sims–Zha alternative

Sims and Zha {cite}`SimsZha2006` proposed to enforce a counterfactual rule with a single kind of policy
shock, the contemporaneous surprise $\nu_{0,t}$, choosing a new value at each date so that the rule holds
period by period.

This uses less information --- only the first column of $\Theta_\nu$ --- but the rule then holds only *ex
post*: at each date, agents expect the *baseline* rule to apply in the future and are surprised again and
again.

A surprise at date $t$ has the same effects as one at date 0, shifted $t$ periods, and no effect before
date $t$.

```{code-cell} python3
def sims_zha(M_inv_A, x_A, φ_new):
    "Enforce τ = -φ_new p ex post with a sequence of contemporaneous surprises."
    θ0 = M_inv_A[:, 2 * T]                # response to a date-0 surprise
    Θ_sz = np.zeros((3 * T, T))
    for t in range(T):
        for b in range(3):                # shift each block (q, p, τ) by t periods
            Θ_sz[b * T + t:(b + 1) * T, t] = θ0[b * T:(b + 1) * T - t]
    R_new = rule_block(φ_new)
    ν = np.linalg.solve(R_new @ Θ_sz, -R_new @ x_A)
    return x_A + Θ_sz @ ν, ν
```

We compute the response to a demand shock under aggressive price stabilization, $\tilde\phi = 2$, in
three ways: by solving the model under the new rule (the truth), by McKay and Wolf's method, and by Sims
and Zha's.

```{code-cell} python3
φ_new = 2.0
x_A = M_inv_A @ forcing(1)
x_true = system_inverse(g_s, g_d, φ_new) @ forcing(1)
x_mw, ν_mw = mw_counterfactual(M_inv_A, x_A, φ_new)
x_sz, ν_sz = sims_zha(M_inv_A, x_A, φ_new)

print(f"max |McKay-Wolf - truth| = {np.abs(x_mw - x_true).max():.1e}")
print(f"max |Sims-Zha   - truth| = {np.abs(x_sz - x_true).max():.1e}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(3):
    ax = axes[i]
    ax.plot(split(x_A)[i][:H_plot], 'k:', lw=2, label=f"baseline rule $\\phi = {φ_A}$")
    ax.plot(split(x_true)[i][:H_plot], 'C0', lw=2, label=f"true, $\\tilde\\phi = {φ_new}$")
    ax.plot(split(x_mw)[i][:H_plot], 'C0o', ms=4, label="McKay-Wolf")
    ax.plot(split(x_sz)[i][:H_plot], 'C3--', lw=2, label="Sims-Zha")
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f"{names[i]}, demand shock")
    ax.set_xlabel("periods")
axes[0].legend(fontsize=8)
axes[3].bar(np.arange(H_plot) - 0.2, ν_mw[:H_plot], width=0.4, color='C0',
            label="McKay-Wolf: news at date 0 about date $t$")
axes[3].bar(np.arange(H_plot) + 0.2, ν_sz[:H_plot], width=0.4, color='C3',
            label="Sims-Zha: surprise at date $t$")
axes[3].axhline(0, color='k', lw=0.5)
axes[3].set_title("policy shocks used")
axes[3].set_xlabel("$t$")
axes[3].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

McKay and Wolf's construction reproduces the true counterfactual to machine precision.

The Sims–Zha construction does not: because suppliers and demanders keep expecting the old rule, they
plan for less price stabilization than they actually get.

How large is the Sims–Zha error?

Because the error comes from expectations, we expect it to grow with the weight that agents place on the
future, and the supplier's adjustment cost $g_s$ governs how forward-looking supply is.

```{code-cell} python3
g_grid = np.geomspace(0.1, 30, 12)
sz_errors = []
for g in g_grid:
    M_inv = system_inverse(g, g_d, φ_A)
    x_A_g = M_inv @ forcing(1)
    x_true_g = system_inverse(g, g_d, φ_new) @ forcing(1)
    x_sz_g, _ = sims_zha(M_inv, x_A_g, φ_new)
    sz_errors.append([np.abs(split(x_sz_g)[i] - split(x_true_g)[i]).max()
                      / np.abs(split(x_true_g)[i]).max() for i in range(2)])
sz_errors = np.array(sz_errors)

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(g_grid, sz_errors[:, 0], 'o-', label="quantity")
ax.semilogx(g_grid, sz_errors[:, 1], 's-', label="price")
ax.set_xlabel("supplier's adjustment cost $g_s$ (log scale)")
ax.set_ylabel("max relative error")
ax.set_title("Sims-Zha error, demand shock, $\\phi$ from 0.5 to 2")
ax.legend()
plt.tight_layout()
plt.show()
```

The error in quantity rises steadily with $g_s$: the more slowly suppliers can adjust, the more their
current decisions depend on expected future policy, and the more it matters that the Sims–Zha
counterfactual gets those expectations wrong.

### Modest policy interventions

Leeper and Zha {cite}`LeeperZha2003` offered a defense of the Sims–Zha approach: if a policy
intervention is **modest**, then ignoring its effect on expectations should do little harm.

Their notion of modesty is statistical: an intervention is modest if the policy shocks that implement it
are small relative to the historical variability of policy shocks, so that agents would not detect a
change in the policy regime.

Our market lets us examine a closely related question: how does the Sims–Zha error behave as the
counterfactual rule $\tilde\phi$ approaches the baseline $\phi_A = 0.5$, so that the shocks needed to
implement it shrink toward zero?

We compare the error with the size of the effect being estimated, the difference between the
counterfactual and the baseline responses.

```{code-cell} python3
φ_tildes = np.array([0.4, 0.45, 0.48, 0.52, 0.55, 0.6, 0.75, 1.0, 2.0, 4.0])
ratios, sizes = [], []
for φ_t in φ_tildes:
    x_true_t = system_inverse(g_s, g_d, φ_t) @ forcing(1)
    x_sz_t, _ = sims_zha(M_inv_A, x_A, φ_t)
    effect = np.abs(split(x_true_t)[0] - split(x_A)[0]).max()
    error = np.abs(split(x_sz_t)[0] - split(x_true_t)[0]).max()
    sizes.append(effect)
    ratios.append(error / effect)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(φ_tildes, sizes, 'o-', label="policy effect on $q$")
axes[0].plot(φ_tildes, np.array(ratios) * np.array(sizes), 's-', label="Sims-Zha error in $q$")
axes[0].set_xlabel("counterfactual rule $\\tilde\\phi$")
axes[0].set_title("absolute size")
axes[0].legend()
axes[1].plot(φ_tildes, ratios, 'o-', color='C3')
axes[1].set_ylim(0, 0.4)
axes[1].set_xlabel("counterfactual rule $\\tilde\\phi$")
axes[1].set_title("Sims-Zha error / policy effect")
for ax in axes:
    ax.axvline(φ_A, color='k', ls=':', lw=1)
plt.tight_layout()
plt.show()
```

As the intervention becomes more modest, the Sims–Zha error shrinks --- but only in proportion to the
effect of the intervention itself.

For interventions near the baseline rule, the Sims–Zha counterfactual misses roughly 30 percent of the
effect on quantity that it is trying to measure.

A modest intervention makes the *absolute* error small, but not the error relative to the policy effect.

The reason is that the expectational channel that Sims–Zha omits is a first-order part of how the
counterfactual rule works: it does not become negligible relative to the other channels as the rule
change becomes small.

The Sims–Zha path does correctly answer a different question: what happens if the government repeatedly
surprises agents who go on believing that the old rule is in force?

Leeper and Zha's modesty condition is designed to make that the relevant question, by ensuring that
agents would not detect the intervention; the answer differs from the effect of a *known* change in the
rule, which is what McKay and Wolf's construction delivers.

### Limited evidence on policy shocks

McKay and Wolf's exact result requires the effects of news shocks at *every* horizon.

In practice an econometrician has evidence on only a few distinct policy shocks, and McKay and Wolf
propose choosing the date-0 combination of the available shocks that enforces the counterfactual rule as
well as possible in a least-squares sense.

Suppose the econometrician knows the effects of a contemporaneous shock and of news shocks about the
next $n - 1$ periods.

This clean menu is an idealization: identified policy shocks in practice, such as the monetary policy
shocks of Romer and Romer or Gertler and Karadi that McKay and Wolf use, are *combinations* of surprises
and news about several horizons, each tracing out its own path for the policy instrument.

```{code-cell} python3
def mw_limited(M_inv_A, x_A, φ_new, n):
    "Least-squares McKay-Wolf counterfactual using news shocks for horizons 0..n-1."
    Ω = M_inv_A[:, 2 * T:2 * T + n]
    R_new = rule_block(φ_new)
    weights, *_ = np.linalg.lstsq(R_new @ Ω, -R_new @ x_A, rcond=None)
    return x_A + Ω @ weights

ns = np.arange(1, 16)
errors = np.array([[np.abs(split(mw_limited(M_inv_A, x_A, φ_new, n))[i]
                           - split(x_true)[i]).max()
                    / np.abs(split(x_true)[i]).max() for i in range(2)]
                   for n in ns])

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(ns, errors[:, 0], 'o-', label="quantity")
ax.semilogy(ns, errors[:, 1], 's-', label="price")
ax.set_xlabel("number of distinct policy shocks $n$ (news horizons $0, \\ldots, n-1$)")
ax.set_ylabel("max relative error")
ax.set_title("approximating the counterfactual with limited evidence")
ax.legend()
plt.tight_layout()
plt.show()
```

The approximation improves steadily as evidence on more distinct policy shocks becomes available.

## Model uncertainty and sufficient statistics

In {doc}`induced_priors`, a structural modeler who believes the model but is unsure of the adjustment
costs expresses that uncertainty with the prior $g_s \sim U[8, 12]$, $g_d \sim U[0.05, 0.15]$.

That prior induces a prior over the counterfactual response to a demand shock under the rule
$\tilde\phi = 2$.

McKay and Wolf's econometrician needs no prior over $(g_s, g_d)$: given estimates of $\mathbf x_A(\varepsilon)$
and $\Theta_\nu$ from data generated by the true economy, Proposition 1 delivers the true counterfactual
exactly.

In the language of McKay and Wolf, impulse responses to policy shocks are **sufficient statistics** for
policy counterfactuals: every model in their class that matches those impulse responses implies the same
counterfactual.

```{code-cell} python3
def fan(ax, paths, color, label, alpha=0.25):
    "Plot the 5-95 percentile band and the median of paths (draws x horizons)."
    lo, med, hi = np.percentile(paths, [5, 50, 95], axis=0)
    h = np.arange(paths.shape[1])
    ax.fill_between(h, lo, hi, color=color, alpha=alpha, lw=0)
    ax.plot(h, med, color=color, lw=2, label=label)

rng = np.random.default_rng(1234)
n_draws = 200
base_draws, cf_draws, sz_ratio_draws = [], [], []
for _ in range(n_draws):
    g_s_i, g_d_i = rng.uniform(8, 12), rng.uniform(0.05, 0.15)
    M_inv_i = system_inverse(g_s_i, g_d_i, φ_A)
    x_A_i = M_inv_i @ forcing(1)
    x_cf_i = np.linalg.solve(np.vstack([private_block(g_s_i, g_d_i), rule_block(φ_new)]),
                             forcing(1))
    x_sz_i, _ = sims_zha(M_inv_i, x_A_i, φ_new)
    base_draws.append(np.concatenate(split(x_A_i)[:2]))
    cf_draws.append(np.concatenate(split(x_cf_i)[:2]))
    sz_ratio_draws.append(np.abs(split(x_sz_i)[0] - split(x_cf_i)[0]).max()
                          / np.abs(split(x_cf_i)[0] - split(x_A_i)[0]).max())
base_draws, cf_draws = np.array(base_draws), np.array(cf_draws)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, ax in enumerate(axes):
    fan(ax, base_draws[:, i * T:i * T + H_plot], 'C7',
        f"induced prior, baseline rule $\\phi = {φ_A}$")
    fan(ax, cf_draws[:, i * T:i * T + H_plot], 'C1',
        f"induced prior, counterfactual rule $\\tilde\\phi = {φ_new}$")
    ax.plot(split(x_mw)[i][:H_plot], 'C0o', ms=4,
            label="McKay-Wolf, from the true economy's impulse responses")
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f"{names[i]}, demand shock")
    ax.set_xlabel("periods")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

print(f"Sims-Zha error / policy effect on q across the prior: "
      f"[{min(sz_ratio_draws):.2f}, {max(sz_ratio_draws):.2f}]")
```

The structural modeler's counterfactual band reflects uncertainty about $(g_s, g_d)$ that, in this
exercise, never gets resolved.

The gray band shows that much of this uncertainty is already present in the modeler's beliefs about how
the economy responds under the *baseline* rule.

The McKay–Wolf econometrician replaces that uncertainty with measurement: the impulse responses under
the baseline rule pin down the counterfactual without pinning down $(g_s, g_d)$.

The Sims–Zha error, meanwhile, is a robust feature of the model: across the whole prior, it misses about
a quarter of the policy effect on quantity.

Of course, a structural modeler who saw the same impulse responses could learn $(g_s, g_d)$ from them, as
in {ref}`ip_ex3` of {doc}`induced_priors`.

McKay and Wolf's point is that the counterfactual does not require that step, and therefore does not
depend on the details of a particular parametric model of the private sector.

## Invertibility

So far we have assumed that the econometrician knows the impulse responses to the non-policy shock and to
the policy shocks.

Where would they come from?

In {doc}`supply_demand_var` and {doc}`induced_priors` we saw that, because quantity adjusts sluggishly,
the supply shock is **not fundamental** for $(q_t, p_t)$: it is not a linear combination of current and
past prices and quantities.

Consequently, no rotation of the innovations of a vector autoregression for $(q_t, p_t)$ recovers the
supply shock --- not a Cholesky factorization, and not one selected by sign restrictions.

We now sort procedures into those that need invertibility and those that don't.

In McKay and Wolf's applications, the responses to policy shocks come from instruments --- narrative
measures or high-frequency financial market surprises --- rather than from the innovations of a VAR.

Policy shocks may then be a small part of the variation in the data, and we abstract from them in what
follows: we work with the economy under the baseline rule, without policy shocks, so that the
econometrician observes $(q_t, p_t)$ and $\tau_t = -\phi_A p_t$.

To compute the population VAR we write the moving-average representation, truncated after `H_ma` lags,
in state-space form and apply the Kalman filter, as in {doc}`supply_demand_var`.

{doc}`A First Look at the Kalman Filter <intermediate:kalman>` introduces the filter, and
{doc}`classical_filtering` derives the same Wold representation by factoring the spectral density.

```{code-cell} python3
H_ma = 80    # truncation lag for the moving-average representation

def structural_ma(g_s, g_d, φ, h_s=h_s, σ_s=σ_s):
    "Stacked responses of (q, p, τ) to the two structural shocks: a 3T x 2 array."
    M_inv = system_inverse(g_s, g_d, φ, h_s=h_s)
    return np.stack([M_inv @ forcing(k, σ_s=σ_s) for k in range(2)], axis=1)

def ma_qp(X):
    "MA coefficients D_j (2 x 2) of (q, p) from stacked paths, j = 0..H_ma-1."
    return np.stack([X[:H_ma], X[T:T + H_ma]], axis=1)

def wold(D):
    "Innovation covariance V and Wold MA coefficients C_j of z_t = sum_j D_j ε_(t-j)."
    k = D.shape[2]
    n = H_ma * k
    A = np.zeros((n, n)); A[k:, :-k] = np.eye(n - k)
    C = np.zeros((n, k)); C[:k] = np.eye(k)
    G = np.hstack(list(D))
    kalman = qe.Kalman(qe.LinearStateSpace(A, C, G, 1e-8 * np.eye(D.shape[1])))
    kalman.stationary_values()
    V = kalman.stationary_innovation_covar()
    C_wold = np.array(kalman.stationary_coefficients(H_ma - 1, 'ma'))
    return V, C_wold

def impulse_norms(D, V):
    "||L^{-1} d|| for each structural impact vector d (columns of D_0)."
    return np.linalg.norm(np.linalg.solve(np.linalg.cholesky(V), D[0]), axis=0)

X_A = structural_ma(g_s, g_d, φ_A)
D_A = ma_qp(X_A)
V_A, C_wold_A = wold(D_A)
print("||L^-1 d|| for (supply, demand) under the baseline rule:",
      np.round(impulse_norms(D_A, V_A), 4))
```

As in {doc}`induced_priors`, a norm below one says that the supply impact vector is not a rotation of the
VAR innovations.

### External instruments do not need invertibility

Suppose that the econometrician observes a noisy measure of the supply shock --- for example, a series on
input costs ---

$$
m_t = \tilde w_{st} + v_t ,
$$

where $v_t$ is measurement error, independent of everything else.

Write $z_t = (q_t, p_t)^\top = \sum_{j \ge 0} D_j \epsilon_{t-j}$ for the structural moving average, where
$\epsilon_t = (\tilde w_{st}, \tilde w_{dt})$ has unit variances and $D_h^{(s)}$ is the first column of
$D_h$.

Because $\tilde w_{st}$ is serially uncorrelated and independent of the demand shock,

$$
\mathrm{cov}(z_{t+h}, m_t) = \mathrm{cov}(z_{t+h}, \tilde w_{st}) = D_h^{(s)},
$$

the true response of $z$ at horizon $h$ to the supply shock.

So a **local projection** --- a regression of $z_{t+h}$ on $m_t$ --- recovers the supply-shock impulse
response up to scale, whether or not the supply shock is fundamental.

Stock and Watson {cite}`StockWatson2018` and Plagborg-Møller and Wolf {cite}`PlagborgMollerWolf2021`
develop this external-instrument logic; the latter show that ordering the instrument first in a VAR
estimates the same impulse responses.

This is how McKay and Wolf themselves estimate the responses to their non-policy shock: they order a
measured series of investment-specific technology news shocks first in a VAR.

We simulate a long sample under the baseline rule and compare three estimates of the response to a supply
shock, each normalized to raise quantity by one unit on impact.

```{code-cell} python3
rng = np.random.default_rng(0)
n_sim = 2_000_000
ε = rng.standard_normal((n_sim + H_ma, 2))
m = ε[H_ma:, 0] + rng.standard_normal(n_sim)        # noisy proxy for the supply shock

# simulate (q, p) from the moving-average representation
x_sim = np.zeros((n_sim, 2))
for i in range(2):
    for k in range(2):
        x_sim[:, i] += np.convolve(ε[:, k], D_A[:, i, k])[H_ma:H_ma + n_sim]

# local projections on the proxy
lp = np.array([[np.dot(x_sim[h:, i], m[:n_sim - h]) / np.dot(m, m)
                for i in range(2)] for h in range(H_plot)])
lp /= lp[0, 0]

truth = D_A[:H_plot, :, 0] / D_A[0, 0, 0]

# the population VAR's best Cholesky candidate: p first, second shock
P_swap = np.array([[0, 1], [1, 0]])
L_p = P_swap @ np.linalg.cholesky(P_swap @ V_A @ P_swap)
chol = np.array([C_wold_A[h] @ L_p[:, 1] for h in range(H_plot)])
chol /= chol[0, 0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, ax in enumerate(axes):
    ax.plot(truth[:, i], 'k-', lw=2, label="true supply-shock response")
    ax.plot(lp[:, i], 'C0o', ms=5, label="local projection on the proxy $m_t$")
    ax.plot(chol[:, i], 'C3--', lw=2, label="VAR, p-first Cholesky")
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f"response of {names[i]} (unit impact on $q$)")
    ax.set_xlabel("periods")
    ax.set_xlim(0, 12)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

The local projection recovers the true response, including the sign reversal after two periods, while the
Cholesky-identified VAR shock does not.

We used a very long sample because supply shocks account for only a small part of the variation in
prices, so the local-projection estimates are noisy in samples of realistic length; in population they
coincide exactly with the true response.

This is the constructive counterpart of the negative results in {doc}`induced_priors`: information from
outside the VAR --- here, a measured cost shock --- can recover a shock that prices and quantities alone
reveal only with a delay.

McKay and Wolf's conditional counterfactuals in Proposition 1 have the same property: they need impulse
responses to the non-policy shock and to the policy shocks, and if those are estimated with instruments,
invertibility plays no role.

Anticipated policy shocks are themselves a standard source of non-invertibility --- see Leeper, Walker,
and Yang {cite}`Leeper_Walker_Yang` on fiscal foresight, and {doc}`hs_invertibility_example` for the same
"news" problem in a permanent income model --- which is another reason why McKay and Wolf rely on
instrument-identified policy shocks rather than on VAR innovations.

### Invertibility depends on the policy rule

Whether the supply shock is fundamental for $(q_t, p_t)$ is a property of the *equilibrium*
moving-average representation, and that representation depends on the policy rule.

So the policy rule can change whether a VAR can recover the supply shock.

```{code-cell} python3
φ_grid = np.linspace(-0.5, 10, 22)
fig, ax = plt.subplots(figsize=(9, 4.5))
for g, color in [(3.0, 'C0'), (10.0, 'C1'), (30.0, 'C3')]:
    norms = []
    for φ in φ_grid:
        D = ma_qp(structural_ma(g, g_d, φ))
        V, _ = wold(D)
        norms.append(impulse_norms(D, V)[0])
    ax.plot(φ_grid, norms, 'o-', color=color, ms=4, label=f"$g_s = {g:g}$")
ax.axhline(1, color='k', lw=0.5)
ax.axvline(φ_A, color='k', ls=':', lw=1)
ax.set_xlabel("policy rule coefficient $\\phi$")
ax.set_ylabel("$\\|L^{-1} d_s\\|$")
ax.set_title("is the supply shock fundamental for $(q_t, p_t)$?")
ax.legend()
plt.tight_layout()
plt.show()
```

The more aggressively the government stabilizes the price, the closer the supply shock comes to being
fundamental.

For $g_s = 3$ the baseline rule already makes it fundamental, and for $g_s = 10$ a rule with
$\phi \gtrsim 5$ does; for $g_s = 30$, a still more aggressive rule would be needed.

A rule that destabilizes the price ($\phi < 0$) pushes the economy further from invertibility.

The equivalence that we derived earlier explains the pattern.

Under the rule $\phi$, prices and quantities behave as in a market without a tax whose supplier has
adjustment cost $g_s / (1 + \phi)$, and whose cost shocks are scaled down by the same factor.

Scaling a shock does not change whether it is fundamental, so the rule matters only through the effective
adjustment cost.

In {ref}`sdvar_ex2` of {doc}`supply_demand_var` we saw that the supply shock becomes fundamental when
suppliers can adjust quickly: price stabilization makes them behave as if they could.

```{code-cell} python3
for φ in [0.5, 2.0]:
    D_rule = ma_qp(structural_ma(g_s, g_d, φ))
    nrm_rule = impulse_norms(D_rule, wold(D_rule)[0])[0]
    D_flex = ma_qp(structural_ma(g_s / (1 + φ), g_d, 0.0, h_s=h_s / (1 + φ)))
    nrm_flex = impulse_norms(D_flex, wold(D_flex)[0])[0]
    print(f"φ = {φ}: ||L^-1 d_s|| under the rule {nrm_rule:.4f}, "
          f"in the flexible-supply market {nrm_flex:.4f}")
```

This is a version of the Lucas critique that applies to *identification*: a VAR identification scheme
that is valid under one policy regime can fail under another, because the zeros of the equilibrium
moving-average representation move with the policy rule.

### Counterfactual second moments through VAR innovations

McKay and Wolf also show how to predict the *unconditional* second moments of the data under a new rule.

If the econometrician could identify every structural shock, she could apply Proposition 1 to each one
and add up the resulting variances and autocovariances.

When the structural shocks cannot all be identified, McKay and Wolf propose applying Proposition 1 to the
impulse responses of the **Wold innovations** instead, and they note that this is valid only if the
structural moving-average representation is invertible.

Our market lets us check this.

```{code-cell} python3
def autocov(X, n_lags=12):
    "Autocovariances Γ_j (3 x 3), j = 0..n_lags, from stacked impulse responses X (3T x k)."
    P = np.stack([X[b * T:(b + 1) * T] for b in range(3)])     # 3 x T x k
    return np.array([np.einsum('itk,jtk->ij', P[:, j:], P[:, :T - j])
                     for j in range(n_lags + 1)])

def wold_route(g_s_, φ_base, φ_cf, σ_s_=σ_s):
    "Counterfactual autocovariances: truth versus the Wold-innovation route."
    M_inv_base = system_inverse(g_s_, g_d, φ_base)
    X_base = np.stack([M_inv_base @ forcing(k, σ_s=σ_s_) for k in range(2)], axis=1)
    X_true = structural_ma(g_s_, g_d, φ_cf, σ_s=σ_s_)
    D = ma_qp(X_base)
    V, C_wold = wold(D)
    W = C_wold[:H_ma] @ np.linalg.cholesky(V)       # responses to orthonormal Wold innovations
    X_w = []
    for k in range(2):
        q, p = np.zeros(T), np.zeros(T)
        q[:H_ma], p[:H_ma] = W[:, 0, k], W[:, 1, k]
        x_k, _ = mw_counterfactual(M_inv_base, np.concatenate([q, p, -φ_base * p]), φ_cf)
        X_w.append(x_k)
    X_w = np.stack(X_w, axis=1)
    Γ_true, Γ_w = autocov(X_true), autocov(X_w)
    return Γ_true, Γ_w, impulse_norms(D, V)[0]

print(" g_s   ||L^-1 d_s||   var(q): true   Wold route   max relative error in Γ_0..Γ_12")
for g in [0.5, 10.0]:
    Γ_true, Γ_w, nrm = wold_route(g, φ_A, φ_new)
    rel = np.abs(Γ_w - Γ_true).max() / np.abs(Γ_true).max()
    print(f"{g:4.1f}  {nrm:12.4f}  {Γ_true[0, 0, 0]:13.5f}  {Γ_w[0, 0, 0]:11.5f}  {rel:22.1e}")
```

When the supply shock is fundamental ($g_s = 0.5$), the Wold-innovation route reproduces the true
counterfactual autocovariances to machine precision.

When it is not ($g_s = 10$), the route makes an error, but the error is tiny.

Why is the route exact under invertibility and not otherwise?

Under invertibility, each Wold innovation is a linear combination of the *current* structural shocks, so
its impulse response is a linear combination of date-0 structural impulse responses, to which
{eq}`pc-mw` applies.

Under non-invertibility, a Wold innovation also contains *past* structural shocks, so its impulse response
includes the continuations of paths that started before date 0.

Proposition 1 treats such a continuation as a fresh date-0 path from the steady state, ignoring that
agents had already been responding to it, so it maps the continuation incorrectly.

```{code-cell} python3
X_A_struct = np.stack([M_inv_A @ forcing(k) for k in range(2)], axis=1)
X_new_struct = structural_ma(g_s, g_d, φ_new)

def continuation(x, k):
    "The part of each path (q, p, τ) from date k on, restarted at date 0."
    out = np.zeros_like(x)
    for b in range(3):
        out[b * T:(b + 1) * T - k] = x[b * T + k:(b + 1) * T]
    return out

for k in [1, 2]:
    mapped, _ = mw_counterfactual(M_inv_A, continuation(X_A_struct[:, 1], k), φ_new)
    target = continuation(X_new_struct[:, 1], k)
    print(f"demand shock, continuation from date {k}: "
          f"max |mapped - true| = {np.abs(mapped - target)[:T - 10].max():.3f}")
```

These errors are not small: Proposition 1 maps each such continuation badly.

The overall error in counterfactual second moments is nevertheless tiny because, in our market, the
weight that the Wold innovations put on past structural shocks is small --- the non-fundamental part of
the supply shock contributes little to the variance of prices and quantities.

(pc_dp2)=
## McKay and Wolf and dynamic programming squared

Several lectures in this series --- the "dynamic programming squared" lectures --- pursue a different
response to the Lucas critique: they model the government as an optimizing agent and ask which policies
it would choose and which it could credibly promise.

This section connects McKay and Wolf's framework to those lectures, and uses them to state precisely
what McKay and Wolf take for granted.

### The same constraint, measured or modeled

In {doc}`calvo`, a perfect-foresight Cagan demand for money implies that the rate of inflation,

$$
\theta_t = \frac{1}{1+\alpha} \sum_{j=0}^{\infty} \left(\frac{\alpha}{1+\alpha}\right)^j \mu_{t+j},
$$

depends on the entire future path of money growth $\{\mu_{t+j}\}$, so that inflation summarizes how
future policy affects today's real balances.

In {doc}`dyn_stack`, a Stackelberg follower's Euler equations are part of the law of motion that the
Stackelberg leader faces when it chooses its plan.

In {doc}`opt_tax_recur`, the household's first-order conditions collapse into an implementability
constraint that restricts the allocations a Ramsey planner can implement with distorting taxes.

In each case, private decisions depend on the whole expected path of a policy instrument, and a
policymaker who chooses that path takes the dependence into account.

Our Euler equations {eq}`pc-supply`–{eq}`pc-demand` play the same role, and McKay and Wolf's key
assumption is that they are the *only* channel through which policy matters.

The matrix $\Theta_\nu$ of responses to contemporaneous and news shocks is then an empirical counterpart
of the constraint that the planners of those lectures take from a model: it tells us which paths of
$(q_t, p_t)$ a policymaker can implement by choosing a path for the tax.

### Optimal policy and time inconsistency

McKay and Wolf's Proposition 2, which {ref}`pc_ex1` puts to work, is a Ramsey or Stackelberg problem
written in sequence space: choose a date-0 path for the instrument to minimize a loss, subject to the
private sector's Euler equations.

Like the Ramsey and Stackelberg plans of {doc}`calvo`, {doc}`dyn_stack`, and {doc}`opt_tax_recur`, the
resulting plan is time inconsistent, and {ref}`pc_ex_tic` reproduces, with McKay–Wolf objects, the
"reborn Stackelberg leader" calculation of {doc}`dyn_stack`.

{doc}`dyn_stack` and {doc}`opt_tax_recur` show how to implement such a plan recursively by handing each
continuation planner, as a state variable, the forward-looking variable that its predecessors promised
--- the follower's decision in {doc}`dyn_stack`, marginal-utility-weighted government debt in
{doc}`opt_tax_recur`.

McKay and Wolf's suggestion that a policymaker apply its targeting rule to *revisions* of its forecasts
at each date --- a "timeless perspective" --- is a device of the same kind.

### Credible rules and sustainable plans

McKay and Wolf's counterfactual uses a Ramsey timing protocol: the new rule is chosen once and for all at
date 0, believed by everyone, and followed forever.

{doc}`calvo_abreu` studies a different protocol, in which a sequence of policymakers each choose policy
for one period, and asks which plans are **sustainable** --- which plans each policymaker would want to
confirm rather than disappoint private expectations.

In the theory of sustainable plans developed by Abreu {cite}`Abreu`, Chari and Kehoe
{cite}`chari1990sustainable`, and Stokey {cite}`stokey1989reputation`, a plan is sustained by the private
sector's expectation that a deviation would trigger a switch to a worse continuation plan.

Two lessons for McKay and Wolf follow.

First, McKay and Wolf's counterfactuals tell us what would happen under a rule that is credible; they do
not tell us whether the rule *is* credible, and a rule that private agents did not fully believe would
have different effects.

Second, the theory of sustainable plans interprets a deviation from expected policy as an event that
*changes* private beliefs about future policy.

If measured policy shocks are deviations of that kind, then their estimated effects combine the effect of
the instrument path with a revision of beliefs about the rule, and the premise of Proposition 1 --- that a
deviation leaves agents' view of the rule unchanged except through the announced path --- fails.

Leeper and Zha's modesty condition, discussed above, is in effect a requirement that deviations not
trigger such revisions.

### Endogenous policy regimes and institutional counterfactuals

{doc}`dovis_accounting_mf` studies a model of Bocola, Chaumont, Dovis, and Kirpalani
{cite}`DovisAccountingMFrevised` in which a government that cannot fully commit delegates monetary policy
to a central bank with an inflation mandate, but may override the mandate when it needs revenue.

Whether it does depends on the fiscal state and on a random cost of overriding the mandate, so the
economy switches *endogenously* between monetary-dominant and fiscal-dominant regimes.

That model differs from McKay and Wolf's environment in three instructive ways.

* Policy is chosen, not given by a rule: the "rule" visible in the data is an equilibrium outcome that
  depends on the state of the economy.
* Because the regime switches with the state, private demand for money depends on the *probability* that
  the mandate will be honored, not only on the expected path of the policy instrument; the economy is
  nonlinear, and the certainty equivalence that lets McKay and Wolf work with expected paths does not
  apply.
* Its central counterfactual --- setting the cost of overriding the mandate to zero throughout --- changes
  the institutions that determine how policy is chosen, not the rule that policy follows; questions of
  this kind are beyond the reach of McKay and Wolf's method and call for a structural model of the
  policymaker.

That lecture also offers a concrete reminder about credibility: its particle filter dates Colombia's gain
in central-bank credibility to 1997 rather than to the 1991 constitutional reform, suggesting that the
public took years to believe in the new regime, whereas McKay and Wolf's counterfactuals assume that a new
rule is believed at once.

### What the method takes for granted

Collecting these points, McKay and Wolf's counterfactuals are exact under assumptions that the dynamic
programming squared lectures relax.

1. The private sector's behavior is linear and depends on policy only through the expected path of the
   instrument.
2. The counterfactual rule is credible: announced at date 0, understood, believed, and kept.
3. Measured policy shocks change the instrument path without changing beliefs about the rule.
4. The policy rule is a primitive chosen by the analyst, not the outcome of choices by a government that
   may be unable to commit.
5. The question concerns rules, not the institutions that shape how rules are chosen.
6. The counterfactual rule delivers a unique equilibrium --- a point on which McKay and Wolf are silent,
   and about which the theory of sustainable plans, with its many credible plans, offers a warning.

Our market satisfies the first assumption, and our computations impose the next three by construction.

Within these assumptions, McKay and Wolf replace a model of the private sector with measurements; the
dynamic programming squared lectures, in turn, replace an assumed policy rule with a model of the
policymaker.

## Summary

The market with a tax fits McKay and Wolf's framework, and their key assumption is the invariance of the
open-loop dynamic supply and demand curves of {doc}`supply_demand_var`.

In this market:

* Impulse responses to contemporaneous and news tax shocks under one rule deliver the *exact* response to
  supply and demand shocks under another rule.
* The Sims–Zha method, which uses only contemporaneous surprises, makes errors that grow with how
  forward-looking suppliers are.
* Making the intervention more modest shrinks the Sims–Zha error only in proportion to the policy effect
  itself: near the baseline rule it still misses about 30 percent of the effect on quantity.
* With evidence on only a few distinct policy shocks, a least-squares approximation improves steadily as
  evidence on more shocks becomes available.
* Impulse responses to policy shocks are sufficient statistics for the counterfactual: they deliver it
  exactly, while a structural modeler with a prior over parameters inherits a band of uncertainty.
* A price-stabilizing tax rule is observationally equivalent to more flexible suppliers, a compact form of
  the Lucas critique in this market.

The market also lets us sort procedures by whether they need invertibility.

| procedure | needs invertibility? | in this market, $g_s = 10$ |
|---|---|---|
| Cholesky or sign-restricted rotations of VAR innovations | yes | miss the supply-shock response |
| local projection on an external instrument | no | recovers the supply-shock response |
| McKay–Wolf counterfactual for a given shock (Proposition 1) | no, given the impulse responses | exact |
| Sims–Zha counterfactual for a given shock | no, given the impulse responses | not robust to the Lucas critique: errors of roughly 5 to 10 percent |
| McKay–Wolf counterfactual second moments via Wold innovations | yes | small error |

Every entry in the table presumes McKay and Wolf's environment: a linear private sector, a credible
counterfactual rule, and policy shocks that do not change beliefs about the rule ({ref}`pc_dp2`).

Two further lessons emerge.

First, invertibility is not a fixed property of an economy: it depends on the policy rule, so a VAR
identification scheme that works under one regime can fail under another --- in our market because
price stabilization acts like a reduction in suppliers' adjustment costs.

Second, in this market, non-invertibility matters a great deal for some objects --- the impulse response
to the supply shock --- and very little for others --- counterfactual second moments.

Finally, McKay and Wolf and the dynamic programming squared lectures offer complementary responses to the
Lucas critique.

McKay and Wolf measure the private sector's response to policy paths and take the policy rule, and its
credibility, as given; the dynamic programming squared lectures model the policymaker, and so can ask
which rules a government would choose, which it could credibly sustain, and how institutions shape both.

## Exercises

```{exercise}
:label: pc_ex1

McKay and Wolf's Proposition 2 shows that impulse responses to policy shocks also suffice to compute
**optimal policy** for a given loss function.

Consider a policymaker who, after a demand shock, wants to minimize

$$
\frac12 \sum_{t=0}^{T-1} \beta^t \left( p_t^2 + \lambda q_t^2 \right)
$$

by choosing the date-0 policy shocks $\boldsymbol\nu$.

1. Using only $\mathbf x_A(\varepsilon)$ and $\Theta_\nu$ under the baseline rule, compute the optimal
   responses of $(q_t, p_t, \tau_t)$ for $\lambda = 0.1$ and $\lambda = 1$.
2. Verify that the optimal allocation satisfies McKay and Wolf's *targeting rule*
   $\Theta_{p,\nu}^\top W \mathbf p + \lambda\, \Theta_{q,\nu}^\top W \mathbf q = 0$, where
   $W = \mathrm{diag}(1, \beta, \beta^2, \ldots)$ and $\Theta_{p,\nu}, \Theta_{q,\nu}$ are the price and
   quantity rows of $\Theta_\nu$.
```

```{solution-start} pc_ex1
:class: dropdown
```

The loss is a weighted sum of squares of $\mathbf x_A + \Theta_\nu \boldsymbol\nu$, so the optimal
$\boldsymbol\nu$ solves a least-squares problem.

```{code-cell} python3
disc = β ** np.arange(T)                  # discount weights, the diagonal of W
Θ_q, Θ_p = Θ_ν[:T], Θ_ν[T:2 * T]

def optimal_plan(x_base, Θ, λ, start=0):
    """
    Choose the policy shocks in the columns of Θ to minimize
    sum_{t >= start} β^(t-start) (p_t^2 + λ q_t^2), starting from the path x_base.
    """
    w = np.sqrt(disc[:T - start])
    q_b, p_b, _ = split(x_base)
    A_ls = np.vstack([w[:, None] * Θ[T + start:2 * T],
                      np.sqrt(λ) * w[:, None] * Θ[start:T]])
    b_ls = -np.concatenate([w * p_b[start:], np.sqrt(λ) * w * q_b[start:]])
    ν_opt, *_ = np.linalg.lstsq(A_ls, b_ls, rcond=None)
    return x_base + Θ @ ν_opt

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i in range(3):
    axes[i].plot(split(x_A)[i][:H_plot], 'k:', lw=2, label="baseline rule")
for λ, color in [(0.1, 'C0'), (1.0, 'C1')]:
    x_opt = optimal_plan(x_A, Θ_ν, λ)
    q_o, p_o, τ_o = split(x_opt)
    foc = Θ_p.T @ (disc * p_o) + λ * Θ_q.T @ (disc * q_o)
    print(f"λ = {λ}: max |targeting rule| = {np.abs(foc).max():.1e}")
    for i in range(3):
        axes[i].plot(split(x_opt)[i][:H_plot], color=color, lw=2,
                     label=f"optimal, $\\lambda = {λ}$")
for i in range(3):
    axes[i].axhline(0, color='k', lw=0.5)
    axes[i].set_title(f"{names[i]}, demand shock")
    axes[i].set_xlabel("periods")
axes[0].legend()
plt.tight_layout()
plt.show()
```

The targeting rule holds to machine precision: it is the first-order condition of the least-squares
problem.

A policymaker who cares mostly about the price ($\lambda = 0.1$) subsidizes the good heavily when the
demand shock raises its price; a larger weight on quantity tempers the subsidy, because the subsidy raises
quantity.

Two remarks connect this exercise to other ideas.

First, the targeting rule says that the policymaker's forecasts of its targets must be optimal within
the set of forecasts that policy can implement, which is Svensson's {cite}`Svensson1997` idea of
**forecast targeting**.

Second, the optimal plan is chosen once and for all at date 0, and it works partly through promises about
*future* taxes that shape suppliers' and demanders' expectations *today*.

It is a **Ramsey plan** with commitment --- a Stackelberg leader's plan in the language of
{doc}`dyn_stack` --- and like the plans in {doc}`calvo`, {doc}`dyn_stack`, and {doc}`opt_tax_recur` it is
time inconsistent, as the next exercise shows.

```{solution-end}
```

```{exercise}
:label: pc_ex_tic

Continue the previous exercise with $\lambda = 0.1$.

Suppose that the economy follows the date-0 optimal plan through date $k - 1$, and that at date $k$ a new
policymaker with the same loss function re-optimizes, choosing the policy shocks for dates $t \ge k$ to
minimize $\frac12 \sum_{t \ge k} \beta^{t-k} (p_t^2 + \lambda q_t^2)$.

The re-optimizing policymaker takes as given everything that happened before date $k$, including the
quantities $q_{k-1}, \ldots, q_{k-4}$ that enter the Euler equations from date $k$ on.

Show that for $k = 1, 2, 4$ the re-optimized path differs from the continuation of the date-0 plan.

*Hint:* to impose the history, replace the equations for dates $t < k$ in the stacked system by equations
that set $(q_t, p_t, \tau_t)$ equal to their values under the date-0 plan.
```

```{solution-start} pc_ex_tic
:class: dropdown
```

This is the "reborn Stackelberg leader" calculation of {doc}`dyn_stack`, carried out with McKay–Wolf
objects.

The re-optimizing policymaker could choose to continue the date-0 plan, which satisfies all the equations
from date $k$ on; if it chooses something else, the date-0 plan is time inconsistent.

```{code-cell} python3
λ = 0.1
x_plan = optimal_plan(x_A, Θ_ν, λ)
M_A = np.vstack([private_block(g_s, g_d), rule_block(φ_A)])

def continuation_loss(x, k):
    q, p, _ = split(x)
    return 0.5 * np.sum(disc[:T - k] * (p[k:]**2 + λ * q[k:]**2))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i in range(3):
    axes[i].plot(split(x_plan)[i][:H_plot], 'k', lw=2, label="date-0 plan")
for k, color in [(1, 'C0'), (2, 'C1'), (4, 'C2')]:
    M_k, rhs_k = M_A.copy(), forcing(1)
    for b in range(3):                    # impose the history before date k
        for t in range(k):
            M_k[b * T + t] = 0.0
            M_k[b * T + t, b * T + t] = 1.0
            rhs_k[b * T + t] = x_plan[b * T + t]
    M_k_inv = np.linalg.inv(M_k)
    x_k_base = M_k_inv @ rhs_k            # history, then the baseline rule
    Θ_k = M_k_inv[:, 2 * T:][:, k:]       # policy shocks for dates t >= k
    x_reopt = optimal_plan(x_k_base, Θ_k, λ, start=k)
    gap = max(np.abs(split(x_reopt)[i][k:k + 15] - split(x_plan)[i][k:k + 15]).max()
              for i in range(3))
    print(f"k = {k}: max |re-optimized - plan| = {gap:.3f};  continuation loss "
          f"{continuation_loss(x_reopt, k):.4f} vs {continuation_loss(x_plan, k):.4f}")
    for i in range(3):
        axes[i].plot(np.arange(k, H_plot), split(x_reopt)[i][k:H_plot], '--',
                     color=color, lw=2, label=f"re-optimized at date {k}")
for i in range(3):
    axes[i].axhline(0, color='k', lw=0.5)
    axes[i].set_title(f"{names[i]}, demand shock, $\\lambda = {λ}$")
    axes[i].set_xlabel("periods")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

At each date $k$ the re-optimizing policymaker achieves a lower continuation loss by abandoning the
date-0 plan.

The gains are small --- at $k = 1$ the continuation loss barely changes --- and so are the departures
from the plan, which are easiest to see in the price; but they are departures that the date-0
policymaker would not want.

The date-0 plan promised taxes for dates $t \ge k$ whose purpose was partly to shape expectations at
dates before $k$; once those dates have passed, the promises no longer serve that purpose, and a
re-optimizing policymaker wants to break them.

This is the time inconsistency that {doc}`calvo` studies in a model of inflation.

{doc}`dyn_stack` and {doc}`opt_tax_recur` show how to implement such a plan recursively: each continuation
planner inherits, as a state variable, the forward-looking variable that its predecessors promised, and is
obligated to deliver it.

A re-optimizing policymaker who honored those promises would continue the date-0 plan; McKay and Wolf's
"timeless perspective", which applies their targeting rule to revisions of forecasts at every date, is a
device of the same kind.

```{solution-end}
```

```{exercise}
:label: pc_ex2

The error that the Wold-innovation route makes under non-invertibility is small in our market because the
non-fundamental supply shock contributes little to the variance of prices and quantities.

Recompute the table of counterfactual second moments with a larger supply-shock standard deviation,
$\sigma_s \in \{2, 4\}$, and $g_s = 10$.

How does the error change?
```

```{solution-start} pc_ex2
:class: dropdown
```

```{code-cell} python3
print(" σ_s   ||L^-1 d_s||   max relative error in Γ_0..Γ_12")
for σ in [np.sqrt(0.5), 2.0, 4.0]:
    Γ_true, Γ_w, nrm = wold_route(10.0, φ_A, φ_new, σ_s_=σ)
    rel = np.abs(Γ_w - Γ_true).max() / np.abs(Γ_true).max()
    print(f"{σ:4.2f}  {nrm:12.4f}  {rel:30.1e}")
```

The error grows in proportion to the variance $\sigma_s^2$ of the supply shock, but it stays small.

The reason is that a larger $\sigma_s$ scales up the supply shock's contribution to the data without
making supply surprises any easier to detect: the degree of non-fundamentalness, measured by
$\|L^{-1} d_s\|$, is unchanged, but the part of the Wold innovations that reflects past supply shocks gets
more weight.

```{solution-end}
```

```{exercise}
:label: pc_ex3

Use McKay and Wolf's Proposition 1 to construct the "zeroing-out" counterfactual of Bernanke, Gertler,
and Watson {cite}`BernankeGertlerWatson1997`: how would quantity and price respond to a supply shock if
the government did not respond at all ($\tilde\phi = 0$)?

Compare the McKay–Wolf answer with the truth and with the Sims–Zha answer.
```

```{solution-start} pc_ex3
:class: dropdown
```

```{code-cell} python3
x_A_s = M_inv_A @ forcing(0)
x_true_s = system_inverse(g_s, g_d, 0.0) @ forcing(0)
x_mw_s, _ = mw_counterfactual(M_inv_A, x_A_s, 0.0)
x_sz_s, _ = sims_zha(M_inv_A, x_A_s, 0.0)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for i in range(3):
    axes[i].plot(split(x_A_s)[i][:H_plot], 'k:', lw=2, label=f"baseline, $\\phi = {φ_A}$")
    axes[i].plot(split(x_true_s)[i][:H_plot], 'C0', lw=2, label="true, no policy")
    axes[i].plot(split(x_mw_s)[i][:H_plot], 'C0o', ms=4, label="McKay-Wolf")
    axes[i].plot(split(x_sz_s)[i][:H_plot], 'C3--', lw=2, label="Sims-Zha")
    axes[i].axhline(0, color='k', lw=0.5)
    axes[i].set_title(f"{names[i]}, supply shock")
    axes[i].set_xlabel("periods")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

print(f"max |McKay-Wolf - truth| = {np.abs(x_mw_s - x_true_s).max():.1e}")
print(f"max |Sims-Zha   - truth| = {np.abs(x_sz_s - x_true_s).max():.1e}")
```

Again McKay and Wolf's construction is exact, while the Sims–Zha construction, which keeps agents expecting
the baseline rule, is not.

```{solution-end}
```

```{exercise}
:label: pc_ex_signs

In {doc}`induced_priors`, sign restrictions that the theory implies with prior probability one
identified a set of supply-shock responses that *excluded* the true response, because the supply shock is
not fundamental.

We have seen that a sufficiently aggressive price-stabilizing rule makes the supply shock fundamental.

1. For the rules $\phi = 0.5$ and $\phi = 6$, compute the signs of the responses of $(q, p)$ to the two
   shocks at horizons $0, \ldots, 5$ that hold for every draw from the prior
   $g_s \sim U[8, 12]$, $g_d \sim U[0.05, 0.15]$.
2. For each rule, give the econometrician the population VAR at the true parameters, impose those sign
   restrictions on rotations of the VAR innovations, and check whether the identified set of supply-shock
   responses contains the truth.
```

```{solution-start} pc_ex_signs
:class: dropdown
```

We compute the sign table from 150 prior draws and then sweep over rotations of the VAR innovations,
normalizing each shock to raise quantity on impact, as in {doc}`induced_priors`.

```{code-cell} python3
K_signs = 5
angles = np.linspace(0, 2 * np.pi, 20_000, endpoint=False)

def theory_sign_table(φ, n=150, seed=0):
    "Signs of the responses of (q, p) that hold for every prior draw: +1, -1, or 0."
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        g_s_i, g_d_i = rng.uniform(8, 12), rng.uniform(0.05, 0.15)
        M = np.vstack([private_block(g_s_i, g_d_i), rule_block(φ)])
        X = np.linalg.solve(M, np.column_stack([forcing(0), forcing(1)]))
        draws.append(ma_qp(X)[:K_signs + 1])
    draws = np.array(draws)
    return np.where((draws > 0).all(0), 1, np.where((draws < 0).all(0), -1, 0))

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for col, φ in enumerate([0.5, 6.0]):
    table = theory_sign_table(φ)
    D = ma_qp(structural_ma(g_s, g_d, φ))
    V, C_w = wold(D)
    L = np.linalg.cholesky(V)
    r1 = L @ np.vstack([np.cos(angles), np.sin(angles)])
    r2 = L @ np.vstack([-np.sin(angles), np.cos(angles)])
    r1, r2 = r1 * np.sign(r1[0]), r2 * np.sign(r2[0])   # each shock raises q on impact
    supply = np.einsum('tij,jm->tim', C_w[:H_plot], r1)
    demand = np.einsum('tij,jm->tim', C_w[:H_plot], r2)
    ok = (np.all(supply[:K_signs + 1] * table[:, :, [0]] >= 0, axis=(0, 1))
          & np.all(demand[:K_signs + 1] * table[:, :, [1]] >= 0, axis=(0, 1)))
    lo, hi = supply[:, :, ok].min(axis=2), supply[:, :, ok].max(axis=2)
    inside = (lo - 1e-10 <= D[:H_plot, :, 0]) & (D[:H_plot, :, 0] <= hi + 1e-10)
    print(f"φ = {φ}: ||L^-1 d_s|| = {impulse_norms(D, V)[0]:.3f}, "
          f"share of rotations kept = {ok.mean():.4f}, "
          f"truth inside the set: q {inside[:, 0].all()}, p {inside[:, 1].all()}")
    for i in range(2):
        ax = axes[i, col]
        ax.fill_between(range(H_plot), lo[:, i], hi[:, i], color='C3', alpha=0.3,
                        label=f"sign-identified set, K = {K_signs}")
        ax.plot(D[:H_plot, i, 0], 'k--', lw=2, label="true supply-shock response")
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(f"{names[i]}, supply shock, $\\phi = {φ}$")
        ax.set_xlim(0, 12)
axes[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

Under the baseline rule the supply shock is not fundamental, and the theory-implied sign restrictions
identify a set that misses the true response, just as in {doc}`induced_priors`.

Under the aggressive rule the supply shock is fundamental, the true response is one of the admissible
rotations, and the identified set contains it.

Notice also that the sign table itself can change with the rule: the theory's sign implications are
properties of an equilibrium, so they too are subject to the Lucas critique.

```{solution-end}
```
