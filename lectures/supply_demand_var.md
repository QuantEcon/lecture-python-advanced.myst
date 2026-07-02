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

(supply_demand_var)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

```{index} single: python
```

# A Single Market and the Interpretation of Vector Autoregressions

## Overview

This lecture is another member of the suite of lectures that use the quantecon `DLE` class and its
underlying `LQ` machinery to study models within the {cite}`HS2013` class described in
{doc}`Recursive Models of Dynamic Linear Economies <hs_recursive_models>`.

It is a companion to {doc}`hs_invertibility_example`.

That lecture studied a **shock-invertibility** problem inside a permanent income model.

This lecture studies the same problem inside a **single competitive market** for a good whose price
and quantity are both observed.

The market comes from the discrete-time example of Lars Peter Hansen and Thomas J. Sargent's
"Two difficulties in interpreting vector autoregressions" {cite}`HanSar1991diff`.

Along the way we will

* set up a market in which a representative supplier and a representative demander each solve a
  linear-quadratic dynamic optimization problem,
* explain why a vector autoregression fit to the market's price and quantity data can give a
  distorted picture of the shocks that actually move the agents — the difficulty that Christopher
  Sims's {cite}`Sims1980` **innovation accounting** runs into,
* derive **dynamic supply and demand curves** by factoring each agent's Euler equation into a
  stable root and an unstable root, and
* recast each side of the market as a price-taking **optimal linear regulator**, an instance of the
  "Big $X$, little $x$" device used throughout modern macroeconomics.

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

## The vector autoregression and its innovations

Let $z_t$ be an $n \times 1$ covariance stationary process — for us it will be the pair of quantity
and price in a single market.

By Wold's theorem (see {doc}`Classical Control with Linear Algebra <lu_tricks>`), $z_t$ has a
**vector autoregression**

```{math}
:label: sdvar-var
z_t = \sum_{j=1}^{\infty} A_j\, z_{t-j} + a_t ,
```

where the residual $a_t$ is a vector white noise, orthogonal to $z_{t-j}$ for all $j \geq 1$, with
covariance $E a_t a_t^T = V$.

Because $a_t$ lies in the space spanned by current and lagged $z$'s, it is **fundamental** for
$z_t$: the history of $z$'s reveals it.

Eliminating the lagged $z$'s gives the **Wold moving-average representation**

```{math}
:label: sdvar-wold
z_t = \sum_{j=0}^{\infty} C_j\, a_{t-j}, \qquad C_0 = I ,
```

and this representation induces the decomposition of the $j$-step-ahead prediction-error covariance

```{math}
:label: sdvar-vd
E\big(z_t - \hat E_{t-j} z_t\big)\big(z_t - \hat E_{t-j} z_t\big)^T = \sum_{k=0}^{j-1} C_k\, V\, C_k^T
```

that underlies Sims's {cite}`Sims1980` **innovation accounting** — the variance decompositions and
impulse responses that a researcher reads off an estimated autoregression.

Now suppose the equilibrium of an economic model has its own moving-average representation in terms
of the shocks that hit the agents' information sets,

```{math}
:label: sdvar-struct
z_t = \sum_{j=0}^{\infty} D_j\, \epsilon_{t-j},
```

where $\epsilon_t$ is the white noise that is fundamental **for the agents**.

The interpretive question is whether the autoregression's innovations $a_t$ equal the agents'
shocks $\epsilon_t$, and whether the response coefficients $C_j$ equal the economic responses $D_j$.

If they do, innovation accounting reads off the economics directly.

This lecture exhibits a single market in which they do **not**: the white noise that a vector
autoregression recovers from data on price and quantity is *not* the white noise that is
fundamental for the supplier and the demander.

We will see exactly *why*, and we will see the quantitative signature of the discrepancy.

## A single market

The econometrician observes only the $2 \times 1$ vector $y_t = (q_t, p_t)$ of quantity and price.

A representative supplier and a representative demander each solve a linear-quadratic dynamic
problem, taking the market price $\{p_t\}$ as an exogenous stochastic process.

**The supplier** chooses $\{q_t\}$ to maximize

```{math}
:label: sdvar-supplyobj
E_0\sum_{t=0}^{\infty}\beta^t\Big\{\, p_t q_t - \tfrac{h_s}{2}q_t^2 - \tfrac{g_s}{2}\big(q_t - q_{t-1}\big)^2 - s_t q_t \,\Big\} .
```

It earns revenue $p_t q_t$, pays a quadratic cost of *adjusting* output through the term
$\tfrac{g_s}{2}(q_t - q_{t-1})^2$, and is buffeted by a serially correlated cost shock $s_t$.

**The demander** chooses $\{q_t\}$ to maximize

```{math}
:label: sdvar-demandobj
E_0\sum_{t=0}^{\infty}\beta^t\Big\{\, -p_t q_t - \tfrac{h_d}{2}q_t^2 - \tfrac{g_d}{2}\big(a(L)q_t\big)^2 - d_t q_t \,\Big\} .
```

It pays $p_t q_t$ for the good, values a service flow $a(L)q_t$ generated by current and past
purchases through a durable-services technology
$a(L) = a_0 + a_1 L + a_2 L^2 + a_3 L^3 + a_4 L^4$, and is shifted by a demand shock $d_t$.

Here $\beta$ is a discount factor and $L$ is the lag operator, $L q_t = q_{t-1}$.

Differentiating {eq}`sdvar-supplyobj` and {eq}`sdvar-demandobj` with respect to $q_t$ gives the
supplier's and demander's **Euler equations**

```{math}
:label: sdvar-eulersupply
-E_t\Big\{\big[h_s + g_s(1-\beta L^{-1})(1-L)\big]q_t\Big\} + p_t = s_t ,
```

```{math}
:label: sdvar-eulerdemand
-E_t\Big\{\big[h_d + g_d\, a(\beta L^{-1})\,a(L)\big]q_t\Big\} - p_t = d_t ,
```

where $L^{-1}$ is the forward operator, $L^{-1} q_t = q_{t+1}$.

The supply and demand shocks are serially correlated,

```{math}
:label: sdvar-shocks
s_t = B_s(L)\, w_{st} , \qquad d_t = B_d(L)\, w_{dt} ,
```

with $B_s(z), B_d(z)$ having zeros outside the unit circle and $w_{st}, w_{dt}$ mutually
uncorrelated white noises that the agents observe.

The white noise fundamental for the agents is $\epsilon_t = (w_{st}, w_{dt})$.

We study the market at the parameters

$$
h_s = h_d = 1, \quad g_s = 10, \quad g_d = 0.1, \quad \beta = 1/1.05,
$$
$$
a(L) = 1 + .8L + .6L^2 + .4L^3 + .2L^4,
$$
$$
B_d(L) = (1+.6L)(1+.4L)(1+.2L), \qquad B_s(L) = (1-.8L)(1+.4L)(1+.2L),
$$
$$
E w_{st}^2 = .5, \qquad E w_{dt}^2 = 4, \qquad E w_{st} w_{dt} = 0 .
$$

Notice that the supplier's adjustment cost $g_s = 10$ is a hundred times the demander's $g_d = 0.1$.

Quantity will therefore adjust **sluggishly** — a feature that turns out to be the source of the
difficulty in interpreting a vector autoregression fit to $(q_t, p_t)$.

```{code-cell} python3
h_s = h_d = 1.0
g_s, g_d = 10.0, 0.1
β = 1 / 1.05
a = np.array([1, .8, .6, .4, .2])                      # a(L): coefficients a0..a4

# MA polynomials for the shocks (coefficients on L^0..L^3)
B_d = np.convolve(np.convolve([1, .6], [1, .4]), [1, .2])
B_s = np.convolve(np.convolve([1, -.8], [1, .4]), [1, .2])

σ_s2, σ_d2 = 0.5, 4.0                                   # shock variances
```

## The dynamic supply and demand curves

Before imposing market clearing it is illuminating to solve each agent's Euler equation
*separately*.

Each is an expectational difference equation in $q_t$, and each is solved by the standard device of
{doc}`Classical Control with Linear Algebra <lu_tricks>`:

> factor the characteristic operator into a stable root and an unstable root, solve the stable root
> backwards into a feedback on lagged quantities, and solve the unstable root forwards into a
> geometric sum of expected future variables.

The two solutions are *dynamic supply and demand curves*.

### The dynamic supply curve

Write the supplier's Euler equation {eq}`sdvar-eulersupply` as $E_t\,\phi_s(L)\,q_t = p_t - s_t$,
with characteristic operator

$$
\phi_s(L) = h_s + g_s(1-\beta L^{-1})(1-L) .
$$

Because $\phi_s$ is symmetric under $L \mapsto \beta L^{-1}$, the roots of its symbol come in a
reciprocal pair $(\delta_s,\ \beta/\delta_s)$, the two solutions of the supplier's characteristic
equation

```{math}
:label: sdvar-supplyroots
z^2 - \Big[(1+\beta) + \tfrac{h_s}{g_s}\Big]\,z + \beta = 0 .
```

Let $\delta_s$ be the smaller root, $|\delta_s| < \sqrt{\beta} < 1$.

```{code-cell} python3
# Solve the supplier's characteristic equation z^2 - [(1+β) + h_s/g_s] z + β = 0
supply_roots = np.roots([1, -((1 + β) + h_s / g_s), β])
δ_s = min(supply_roots, key=abs)

print(f"supply roots         : {np.sort(supply_roots)}")
print(f"δ_s                  : {δ_s:.4f}   (√β = {np.sqrt(β):.4f})")
print(f"δ_s / β              : {δ_s / β:.4f}")
```

The operator factors as

```{math}
:label: sdvar-supplyfactor
\phi_s(L) = \frac{g_s\beta}{\delta_s}\,\big(1 - \delta_s L^{-1}\big)\,\big(1 - \tfrac{\delta_s}{\beta}L\big),
```

an unstable **forward** factor $(1-\delta_s L^{-1})$ and a stable **backward** factor
$(1-\tfrac{\delta_s}{\beta}L)$.

Solving the forward root forward — a geometric sum of expected future variables, exactly as in
{doc}`lu_tricks` — and reading the backward root as a feedback on the lag of $q$, the supplier's
Euler equation becomes the **dynamic supply curve**

```{math}
:label: sdvar-supplycurve
q_t = \frac{\delta_s}{\beta}\,q_{t-1}
 \;+\; \frac{\delta_s}{g_s\beta}\,E_t\sum_{j=0}^{\infty}\delta_s^{\,j}\,\big(p_{t+j} - s_{t+j}\big).
```

Current quantity supplied is a geometrically declining feedback on its own lag $q_{t-1}$ plus the
conditional expectation of a *discounted geometric sum of future prices* $p_{t+j}$ and *future
supply shocks* $s_{t+j}$.

Higher expected future prices raise current supply; a higher expected cost shock lowers it.

Only the single lag $q_{t-1}$ appears, because the supplier's adjustment cost penalizes
$(1-L)q_t$ one period at a time.

### The dynamic demand curve

The demander's Euler equation {eq}`sdvar-eulerdemand` is $E_t\,\phi_d(L)\,q_t = -(p_t + d_t)$, with
characteristic operator

$$
\phi_d(L) = h_d + g_d\,a(\beta L^{-1})\,a(L).
$$

Since $a(L)$ has degree four, $\phi_d$ is again symmetric under $L \mapsto \beta L^{-1}$, but now
its symbol has eight roots in four reciprocal pairs $(\delta_{d,i},\ \beta/\delta_{d,i})$,
$i = 1,\dots,4$.

Collecting the four stable roots $|\delta_{d,i}| < \sqrt{\beta}$, the factorization is

```{math}
:label: sdvar-demandfactor
\phi_d(L) = \nu_d\, c_d(L)\, c_d(\beta L^{-1}),
\qquad
c_d(L) = \prod_{i=1}^{4}\Big(1 - \tfrac{\delta_{d,i}}{\beta}\,L\Big) = 1 - \sum_{k=1}^{4}\gamma_{d,k}\,L^{k},
```

with $\nu_d > 0$ a normalizing constant.

We find the eight roots numerically.

The symbol $\phi_d(z) = h_d + g_d\, a(z)\, a(\beta/z)$ is a Laurent polynomial; multiplying by $z^4$
turns it into an ordinary degree-8 polynomial whose roots we can read off.

```{code-cell} python3
# a(z) as an ordinary polynomial (highest power first)
a_z = np.poly1d(a[::-1])
# z^4 * a(β/z): coefficients of z^4 .. z^0 are a0, a1 β, ..., a4 β^4
a_βz = np.poly1d([a[k] * β**k for k in range(5)])

# z^4 * φ_d(z) = h_d z^4 + g_d * a(z) * (z^4 a(β/z))
poly = (g_d * a_z * a_βz).c
poly[4] += h_d
demand_roots = np.roots(poly)

moduli = np.abs(demand_roots)
δ_d = demand_roots[moduli < np.sqrt(β)]                # the four stable roots

print("moduli of the eight roots :", np.round(np.sort(moduli), 4))
print("stable δ_d,i moduli       :", np.round(np.sort(np.abs(δ_d)), 4))
```

The four stable roots are two complex-conjugate pairs, of modulus roughly $0.30$ and $0.39$.

Reading the backward factor $c_d(L)$ as a feedback on lags gives the **dynamic demand curve**

```{math}
:label: sdvar-demandcurve
q_t = \sum_{k=1}^{4}\gamma_{d,k}\,q_{t-k}
 \;-\; \frac{1}{\nu_d}\sum_{i=1}^{4} A_{d,i}\,E_t\sum_{j=0}^{\infty}\delta_{d,i}^{\,j}\,\big(p_{t+j} + d_{t+j}\big),
```

where the $A_{d,i}$ are the partial-fraction weights of the inverse forward factor.

Now current quantity demanded depends on **four** lags $q_{t-1},\dots,q_{t-4}$ — the durable-services
technology $a(L)$ spreads adjustment over four periods — and on a sum of geometric feed-forward
terms, one per stable root.

The price terms enter with a *negative* sign: higher expected future prices lower current demand.

We recover the feedback coefficients $\gamma_{d,k}$ from the stable roots.

```{code-cell} python3
# c_d(L) = prod_i (1 - (δ_d,i / β) L) = 1 - Σ γ_d,k L^k
c_d = np.array([1.0])
for r in δ_d / β:
    c_d = np.convolve(c_d, [1, -r])
c_d = c_d.real

γ_d = -c_d[1:]
print("γ_d,k =", np.round(γ_d, 4))
```

The feedback weights $(\gamma_{d,1},\dots,\gamma_{d,4}) \approx (-0.117,\,-0.079,\,-0.045,\,-0.017)$
decline smoothly with the lag.

### Both shocks shift both curves

Taken at face value, the dynamic supply curve {eq}`sdvar-supplycurve` seems to involve only supply
shocks and the dynamic demand curve {eq}`sdvar-demandcurve` only demand shocks.

But each curve also contains the conditional expectations $E_t\,p_{t+j}$ of *future prices*, and in
equilibrium the price process is driven by **both** shocks.

A demand surprise that moves expected future prices therefore shifts the dynamic *supply* curve, and
a supply surprise that moves expected future prices shifts the dynamic *demand* curve.

It is exactly this dependence on forecasts of future prices — absent from static supply and demand
curves — that couples the two sides of the market and makes the equilibrium dynamics richer than a
sequence of momentary intersections.

## Equilibrium

The rational expectations equilibrium equates quantity demanded to quantity supplied period by
period and requires that the price forecasts $E_t\,p_{t+j}$ that appear in both curves be the ones
generated by the equilibrium price process itself.

Because the market objectives are quadratic and the shocks enter linearly, the competitive
equilibrium coincides with the allocation chosen by a fictitious planner who maximizes the sum of
the two objectives.

The revenue terms $p_t q_t$ cancel when quantity supplied equals quantity demanded, so the planner
maximizes

```{math}
:label: sdvar-planner
E_0\sum_{t=0}^{\infty}\beta^t\Big\{ -\tfrac{h_s}{2}q_t^2 - \tfrac{g_s}{2}(q_t-q_{t-1})^2 - s_t q_t
 -\tfrac{h_d}{2}q_t^2 - \tfrac{g_d}{2}(a(L)q_t)^2 - d_t q_t \Big\} .
```

The equilibrium price is then the common marginal valuation; reading it off the supplier's Euler
equation {eq}`sdvar-eulersupply`,

```{math}
:label: sdvar-price
p_t = s_t + \big[h_s + g_s(1-\beta L^{-1})(1-L)\big]q_t .
```

We solve the planner's problem as a discounted **optimal linear regulator** with QuantEcon's `LQ`
class.

The control is $q_t$ and the state stacks the four lagged quantities $q_{t-1},\dots,q_{t-4}$ (the
demander's services technology reaches back four periods) together with a companion form that
carries the current and lagged white noises so that $s_t$ and $d_t$ are linear functions of the
state.

```{code-cell} python3
def companion_shift(n):
    "Companion matrix that shifts an n-vector of lagged shocks."
    A = np.zeros((n, n))
    A[1:, :-1] = np.eye(n - 1)
    return A

def solve_equilibrium(h_s, h_d, g_s, g_d, β, a, B_s, B_d):
    """
    Solve the market equilibrium as a planner's optimal linear regulator.

    State x = [q_{t-1}, ..., q_{t-4}, w_s(t..t-3), w_d(t..t-3)]   (dim 12)
    Control u = q_t.

    Returns the closed-loop transition A_F, shock loading C, the selector
    G_z mapping the state into (q_t, p_t), and the selectors for s_t, d_t.
    """
    n_q = 4
    # exogenous companion for the two MA(3) shock processes
    A_w = np.zeros((8, 8))
    A_w[:4, :4] = companion_shift(4)
    A_w[4:, 4:] = companion_shift(4)
    C_w = np.zeros((8, 2))
    C_w[0, 0], C_w[4, 1] = 1.0, 1.0                    # inject (w_s, w_d)

    n = n_q + 8
    A = np.zeros((n, n))
    A[1:n_q, 0:n_q - 1] = np.eye(n_q - 1)              # q lags shift
    A[n_q:, n_q:] = A_w
    B = np.zeros((n, 1)); B[0, 0] = 1.0                # new q_{t-1} = q_t
    C = np.zeros((n, 2)); C[n_q:, :] = C_w

    def q_lag(k):                                      # selector for q_{t-k}
        e = np.zeros(n); e[k - 1] = 1.0; return e

    s_sel = np.zeros(n); s_sel[n_q:n_q + 4] = B_s      # s_t = s_sel · x
    d_sel = np.zeros(n); d_sel[n_q + 4:] = B_d
    q1 = q_lag(1)
    a_vec = sum(a[k] * q_lag(k) for k in range(1, 5))  # lagged part of a(L)q_t

    # one-period loss = -(surplus), written as u'Qu + x'Rx + 2 u'Nx
    Q = np.array([[(h_s + h_d) / 2 + g_s / 2 + g_d / 2 * a[0]**2]])
    R = g_d / 2 * np.outer(a_vec, a_vec) + g_s / 2 * np.outer(q1, q1)
    R += 1e-5 * np.eye(n)                              # tiny ridge for the Riccati solver
    N = (g_d * a[0] * a_vec / 2 - g_s * q1 / 2 + (s_sel + d_sel) / 2).reshape(1, n)

    lq = LQ(Q, R, A, B, C, N=N, beta=β)
    P, F, _ = lq.stationary_values()
    F = np.asarray(F).reshape(1, n)
    A_F = A - B @ F

    # price p_t = s_t + h_s q_t + g_s[(1+β) q_t - q_{t-1} - β E_t q_{t+1}]
    F_q = -F                                           # q_t   = F_q · x
    E_q1 = -F @ A_F                                    # E_t q_{t+1} = E_q1 · x
    G_p = (s_sel + h_s * F_q[0] + g_s * (1 + β) * F_q[0]
           - g_s * q1 - g_s * β * E_q1[0]).reshape(1, n)
    G_z = np.vstack([F_q, G_p])                        # (q_t, p_t) = G_z · x
    return A_F, C, G_z, F, s_sel, d_sel

A_F, C, G_z, F, s_sel, d_sel = solve_equilibrium(h_s, h_d, g_s, g_d, β, a, B_s, B_d)

print("equilibrium feedback on q_{t-1..t-4} :", np.round(-F[0, :4], 4))
print("largest closed-loop eigenvalue       :", np.round(np.max(np.abs(np.linalg.eigvals(A_F))), 4))
```

The equilibrium is stationary — the largest closed-loop eigenvalue is well inside the unit circle.

The structural moving-average representation {eq}`sdvar-struct` of $z_t = (q_t, p_t)$ in the agents'
shocks $\epsilon_t = (w_{st}, w_{dt})$ is $z_t = G_z (I - A_F L)^{-1} C\, \epsilon_t$.

Its impulse responses are the **response to the agents' structural shocks**.

```{code-cell} python3
def structural_irf(A_F, C, G_z, T=25, scale=None):
    "Impulse responses of (q, p) to the structural shocks (w_s, w_d)."
    n = A_F.shape[0]
    if scale is None:
        scale = np.eye(C.shape[1])
    out = np.zeros((T, 2, 2))
    M = np.eye(n)
    for j in range(T):
        out[j] = G_z @ M @ C @ scale
        M = M @ A_F
    return out

scale = np.diag([np.sqrt(σ_s2), np.sqrt(σ_d2)])        # scale shocks to their std devs
irf_struct = structural_irf(A_F, C, G_z, scale=scale)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for col, name in enumerate(["supply shock $w_{st}$", "demand shock $w_{dt}$"]):
    axes[col].plot(irf_struct[:, 0, col], label="quantity $q_t$")
    axes[col].plot(irf_struct[:, 1, col], label="price $p_t$")
    axes[col].axhline(0, color="k", lw=0.5)
    axes[col].set_title(f"response to {name}")
    axes[col].set_xlabel("periods")
    axes[col].legend()
plt.tight_layout()
plt.show()
```

A **demand** surprise moves the price sharply on impact with only a small quantity response, because
the large adjustment cost $g_s$ prevents the supplier from expanding output quickly.

A **supply** surprise sends quantity and price off in opposite directions.

In both cases quantity moves *slowly* — the hallmark of the sluggish adjustment that $g_s = 10$
builds in.

## Why a vector autoregression misreads this market

The shocks $\epsilon_t = (w_{st}, w_{dt})$ are fundamental for the agents' information set.

The question is whether they are also fundamental for the *econometrician's* data $z_t = (q_t, p_t)$.

They are fundamental for $z_t$ if and only if the moving average $z_t = G_z (I-A_F L)^{-1} C\,
\epsilon_t$ has no zeros inside the unit circle.

For this market they do lie inside, and $\epsilon_t$ is **not** fundamental for $z_t$.

A vector autoregression fit to $(q_t, p_t)$ therefore does not recover $\epsilon_t$.

Instead it recovers a *different* white noise $\epsilon_t^*$ — the Wold innovation — that is a
one-sided distributed lag of current and past $\epsilon_t$'s,

```{math}
:label: sdvar-blaschke
\epsilon_t^* = G(L)^T \epsilon_t ,
```

obtained by "flipping" the inside-the-unit-circle zeros to the outside (a *Blaschke* factorization).

The Wold innovation mixes the agents' current surprise with **old news**.

We recover the Wold representation by passing the equilibrium through the **Kalman filter**, exactly
as in {doc}`hs_invertibility_example`.

The innovations representation delivered by the filter is the Wold representation, and the filter
that maps $\epsilon_t \mapsto \epsilon_t^*$ is the *whitener*.

```{code-cell} python3
# innovations (Wold) representation via the Kalman filter
G_meas = 1e-8 * np.eye(2)                              # negligible measurement error
lss = qe.LinearStateSpace(A_F, C @ scale, G_z, G_meas)
kalman = qe.Kalman(lss)

Σ_struct = (G_z @ C @ scale) @ (G_z @ C @ scale).T     # contemporaneous cov of structural innovation
Σ_wold = kalman.stationary_innovation_covar()          # contemporaneous cov of Wold innovation

print("contemporaneous covariance of the structural innovation R0 ε_t:")
print(np.round(Σ_struct, 4))
print("\ncontemporaneous covariance of the Wold innovation R0* ε_t*:")
print(np.round(Σ_wold, 4))
print("\ndifference (Wold − structural), a positive semidefinite matrix:")
print(np.round(Σ_wold - Σ_struct, 5))
```

The contemporaneous covariance of the Wold innovation *exceeds* that of the agents' structural
innovation.

This is the covariance inequality

```{math}
:label: sdvar-covineq
E\big(R_0^*\epsilon_t^*\big)\big(R_0^*\epsilon_t^*\big)^T \;\succeq\; E\big(R_0\,\epsilon_t\big)\big(R_0\,\epsilon_t\big)^T .
```

The innovation the econometrician sees carries *more* contemporaneous variance than the agents'
surprise, because it has folded in shocks that the agents already knew.

We can compare the impulse responses directly.

The Wold representation's impulse responses come from the Kalman filter's stationary moving-average
coefficients; the whitener's impulse responses show how the recovered innovation $\epsilon_t^*$
responds to the agents' structural shock $\epsilon_t$.

```{code-cell} python3
T = 25
whitener = kalman.whitener_lss()
ma_coefs = kalman.stationary_coefficients(T, 'ma')     # Wold MA coefficients
σ_wold = np.sqrt(np.diag(Σ_wold))

# response of (q, p) to the Wold innovations ε*_t (scaled to their std devs)
wold_irf = np.array([mc @ np.diag(σ_wold) for mc in ma_coefs])

# response of the recovered innovations ε*_t to the structural shocks ε_t
whit_irf = whitener.impulse_response(T)[1]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(wold_irf[:, 0, 0], label="quantity innov.")
axes[0].plot(wold_irf[:, 1, 0], label="price innov.")
axes[0].set_title("response of $(q,p)$ to the first Wold innovation")
axes[0].axhline(0, color="k", lw=0.5); axes[0].set_xlabel("periods"); axes[0].legend()

axes[1].plot([c[0, 0] for c in whit_irf], label="via supply shock $w_{st}$")
axes[1].plot([c[0, 1] for c in whit_irf], label="via demand shock $w_{dt}$")
axes[1].set_title("response of the recovered innovation $\\epsilon^*_t$")
axes[1].axhline(0, color="k", lw=0.5); axes[1].set_xlabel("periods"); axes[1].legend()
plt.tight_layout()
plt.show()
```

The right panel is the heart of the matter.

The supply shock $w_{st}$ enters the recovered innovations only as a **distributed lag**: because
quantity adjusts sluggishly, a supply surprise is revealed to the econometrician gradually, through
the path of $(q_t, p_t)$, rather than all at once.

The demand shock $w_{dt}$, by contrast, hits almost contemporaneously.

An econometrician who interpreted the autoregression's innovations as the agents' shocks would
therefore misattribute both the timing and the sources of the market's response.

```{admonition} The moral for innovation accounting
:class: tip

A vector autoregression always recovers *some* fundamental white noise $\epsilon^*_t$ for the
observed $z_t$, and Sims's innovation accounting always produces a tidy variance decomposition.

But the fundamental noise for the data need not be the fundamental noise for the agents.

When it is not — as in this market, where quantity adjusts sluggishly and so reveals supply
surprises only with a lag — the impulse responses and variance decompositions describe the data's
own internal forecasting structure, not the economy's response to the surprises that actually move
agents.

Reading economic meaning into them requires the restrictions of a model, not just the
autoregression.
```

## Each side of the market as a price-taking regulator

The dynamic supply and demand curves {eq}`sdvar-supplycurve`–{eq}`sdvar-demandcurve` are the
first-order conditions of two *distinct* optimization problems, and in the rational expectations
equilibrium **both agents are price takers**.

We now cast each problem as a discounted optimal linear regulator in which the agent faces the
equilibrium price as an *exogenous* stochastic process.

This is an instance of the "Big $X$, little $x$" device: append the exogenous aggregate law of
motion to the agent's own state and solve an ordinary linear regulator, exactly as in
[Section 50.7.1 of the QuantEcon dynamic Stackelberg lecture](https://python-advanced.quantecon.org/dyn_stack.html).

Write the equilibrium in state-space form

```{math}
:label: sdvar-bigX
X_{t+1} = A\,X_t + C\,w_{t+1}, \qquad p_t = G_p\,X_t ,
```

taken directly from the equilibrium computed above ($A = A_F$, and $G_p$ is the price row of $G_z$).

A price-taking agent treats $X_t$ as an exogenous Markov state it cannot influence.

Because $X_t$ is Markov, every conditional expectation of a future price is a *linear function of
the current state*, $E_t\,p_{t+j} = G_p\,A^{\,j}\,X_t$, so the geometric feed-forward sums in the
dynamic supply and demand curves collapse into linear functions of $X_t$.

We append the agent's own lagged quantity (or quantities) to $X_t$ and solve.

```{code-cell} python3
def supplier_regulator(A_F, C, G_z, s_sel, h_s, g_s, β):
    "Supplier's price-taking regulator facing the equilibrium price process."
    nX = A_F.shape[0]
    n = nX + 1                                         # append own lag q_{t-1}
    A = np.zeros((n, n)); A[:nX, :nX] = A_F
    B = np.zeros((n, 1)); B[nX, 0] = 1.0
    Cc = np.zeros((n, 2)); Cc[:nX, :] = C
    own = np.zeros(n); own[nX] = 1.0                   # picks own q_{t-1}
    p_sel = np.zeros(n); p_sel[:nX] = G_z[1]           # p_t = p_sel · state
    s_selX = np.zeros(n); s_selX[:nX] = s_sel

    Q = np.array([[(h_s + g_s) / 2]])
    R = g_s / 2 * np.outer(own, own)
    N = (-g_s / 2 * own + (s_selX - p_sel) / 2).reshape(1, n)
    lq = LQ(Q, R, A, B, Cc, N=N, beta=β)
    _, Fs, _ = lq.stationary_values()
    return np.asarray(Fs).reshape(1, n), nX

F_s, nX = supplier_regulator(A_F, C, G_z, s_sel, h_s, g_s, β)
print(f"supplier's feedback on its own lag q_(t-1) : {-F_s[0, nX]:.4f}")
print(f"stable root of the supply curve  δ_s / β   : {δ_s / β:.4f}")
```

The supplier's regulator reproduces the coefficient $\delta_s/\beta$ on its own lag from the dynamic
supply curve {eq}`sdvar-supplycurve` — the feed-forward part $-F_X X_t$ is exactly the geometric sum
$\tfrac{\delta_s}{g_s\beta}E_t\sum_j \delta_s^{\,j}(p_{t+j}-s_{t+j})$ written as a linear function of
the state.

The demander's regulator has the same shape but with a richer own state $(q_{t-1},\dots,q_{t-4})$.

```{code-cell} python3
def demander_regulator(A_F, C, G_z, d_sel, h_d, g_d, a, β):
    "Demander's price-taking regulator facing the equilibrium price process."
    nX = A_F.shape[0]
    n_lag = 4
    n = nX + n_lag                                     # append q_{t-1..t-4}
    A = np.zeros((n, n)); A[:nX, :nX] = A_F
    A[nX + 1:nX + n_lag, nX:nX + n_lag - 1] = np.eye(n_lag - 1)
    B = np.zeros((n, 1)); B[nX, 0] = 1.0
    Cc = np.zeros((n, 2)); Cc[:nX, :] = C

    def q_lag(k):                                      # own q_{t-k}, k = 1..4
        e = np.zeros(n); e[nX + k - 1] = 1.0; return e

    p_sel = np.zeros(n); p_sel[:nX] = G_z[1]
    d_selX = np.zeros(n); d_selX[:nX] = d_sel
    a_vec = sum(a[k] * q_lag(k) for k in range(1, 5))

    Q = np.array([[h_d / 2 + g_d / 2 * a[0]**2]])
    R = g_d / 2 * np.outer(a_vec, a_vec)
    N = (g_d * a[0] * a_vec / 2 + (d_selX + p_sel) / 2).reshape(1, n)
    lq = LQ(Q, R, A, B, Cc, N=N, beta=β)
    _, Fd, _ = lq.stationary_values()
    return np.asarray(Fd).reshape(1, n), nX

F_d, nX = demander_regulator(A_F, C, G_z, d_sel, h_d, g_d, a, β)
print("demander's feedback on q_(t-1..t-4) :", np.round(-F_d[0, nX:nX + 4], 4))
print("γ_d,k from the demand curve         :", np.round(γ_d, 4))
```

The demander's regulator reproduces the feedback coefficients $\gamma_{d,k}$ of the dynamic demand
curve {eq}`sdvar-demandcurve`.

### Big $X$ equals little $x$

Each agent's rule is a best response to the price process $X_t$, and *only* a best response: the
agent takes $X_t$ as given, and its own little-$x$ choice does not move Big $X$.

What closes the model is the requirement that the aggregate the agents respond to be consistent with
the aggregate their choices produce: the market clears, and the price process that both agents
forecast is the very process their market-clearing quantities generate.

We can verify this fixed point.

Facing the equilibrium price process, the supplier's best-response quantity rule should coincide
with the equilibrium quantity rule the planner computed.

```{code-cell} python3
# supplier's implied quantity rule, evaluated on the equilibrium state
# (the supplier's own lag equals the market's q_{t-1}, i.e. state component 0)
own_is_q1 = np.zeros(nX); own_is_q1[0] = 1.0
q_rule_supplier = -F_s[0, :nX] - F_s[0, nX] * own_is_q1
q_rule_equilibrium = G_z[0]                            # equilibrium q_t = G_z[0] · X_t

gap = np.max(np.abs(q_rule_supplier - q_rule_equilibrium))
print(f"max |supplier best response − equilibrium quantity rule| : {gap:.2e}")
```

The two rules agree to machine precision.

Feeding the equilibrium price process back into the agent's regulator returns a quantity rule
consistent with it — the "Big $X$, little $x$" fixed point.

### Open-loop versus closed-loop decision rules

The construction has produced two distinct pairs of decision rules.

The dynamic supply and demand curves {eq}`sdvar-supplycurve`–{eq}`sdvar-demandcurve` give each
agent's optimal quantity as a function of its own past quantities and of its forecasts
$E_t\,p_{t+j}$ of an **arbitrary** price process that it takes as given.

They are best responses to *whatever* price process the agent happens to face, and they assume
nothing about how that price is generated.

In this sense they are an **open-loop** pair, valid *outside* any particular rational expectations
equilibrium.

The regulator feedback rules $q_t = -F\,\widehat X_t$ are a different pair.

They are the *same* optimizing behavior with the *equilibrium* price process {eq}`sdvar-bigX`
substituted in.

These are a **closed-loop** pair — the price each agent responds to is now the very one the
equilibrium system produces — and they are the decision rules that obtain *inside* the rational
expectations equilibrium.

The two coincide only after one fixes the price process to be the equilibrium one; passing from the
open-loop curve to the closed-loop rule is exactly the act of imposing rational expectations on the
agents' price forecasts.

Whichever representation one adopts, both deliver the same message: each decision maker's quantity
*today* depends not on today's price alone but on the entire prospective **continuation path** of
the price, $\{p_{t+j}\}_{j\ge 0}$, forecast from today out into the indefinite future.

Supply and demand are inescapably **forward-looking**.

And because the equilibrium price path is itself driven by *both* disturbances, both pairs of rules
make each side's quantity depend on both shock processes: today's suppliers and today's demanders
both react to supply *and* demand shocks, through their common effect on the prospective path of
prices.

## Exercises

```{exercise}
:label: sdvar_ex1

The dynamic supply curve {eq}`sdvar-supplycurve` says that the supplier's feedback on its own lag
$q_{t-1}$ is $\delta_s/\beta$, a number that depends only on the supplier's own cost parameters
$(h_s, g_s, \beta)$ — **not** on the price process it faces.

Confirm this invariance numerically.

Re-solve the supplier's regulator, but replace the equilibrium price process by a *different*
exogenous price process — for example one in which the price is pure white noise, or one in which it
is far more persistent than in equilibrium.

Show that the feedback on $q_{t-1}$ is unchanged at $\delta_s/\beta$, while the feed-forward
response to the price changes.

Explain why the own-lag coefficient must be invariant.
```

```{solution-start} sdvar_ex1
:class: dropdown
```

The own-lag coefficient is the stable (backward) root of the supplier's characteristic operator
$\phi_s(L)$, which is a function of $(h_s, g_s, \beta)$ alone.

The forcing process the agent faces affects only the *particular* (forward-looking) part of the
solution — the geometric sum of expected future prices — not the *homogeneous* part that governs the
agent's own internal dynamics.

This is the certainty-equivalent separation of the linear regulator: the feedback on the endogenous
state solves a Riccati equation that does not involve the exogenous forcing.

We build a small stand-in price process and feed it to the supplier's regulator.

```{code-cell} python3
def make_ar_price(ρ, nX_extra=0):
    """A scalar AR(1) price process p_t = ρ p_{t-1} + w_t as a state-space block."""
    A = np.array([[ρ]])
    C = np.array([[1.0, 0.0]])          # driven by the first white noise only
    G_p = np.array([[1.0]])             # p_t = state
    s_sel = np.array([0.0])             # no separate cost shock in this stand-in
    G_z = np.vstack([np.zeros(1), G_p])
    return A, C, G_z, s_sel

for ρ in [0.0, 0.5, 0.95]:
    A_p, C_p, G_zp, s_p = make_ar_price(ρ)
    F_test, nX_test = supplier_regulator(A_p, C_p, G_zp, s_p, h_s, g_s, β)
    print(f"ρ = {ρ:4.2f} :  own-lag feedback = {-F_test[0, nX_test]:.4f}"
          f"   feed-forward on price = {-F_test[0, 0]:.4f}")

print(f"\nδ_s / β (target) = {δ_s / β:.4f}")
```

The own-lag feedback is $\delta_s/\beta$ for every price process, while the feed-forward loading on
the price changes with its persistence $\rho$: a more persistent price raises the discounted sum of
expected future prices and so raises the response of current supply.

```{solution-end}
```

```{exercise}
:label: sdvar_ex2

The difficulty in interpreting a vector autoregression of $(q_t, p_t)$ comes from the supplier's
adjustment cost $g_s$, which makes quantity adjust sluggishly and hides supply surprises from the
econometrician until they show up gradually in the data.

Investigate the role of $g_s$.

Recompute the equilibrium for $g_s \in \{0.1, 1, 10, 50, 100\}$ and, for each, use the whitener to
measure what *fraction* of the recovered innovation's response to the supply shock $w_{st}$ arrives
contemporaneously — at lag zero — rather than as a distributed lag over later periods.

Predict the direction of the effect before running the code, and explain it.
```

```{solution-start} sdvar_ex2
:class: dropdown
```

When $g_s$ is small, quantity adjusts almost freely, the supplier reveals its surprise almost at
once, and the econometrician's innovation coincides with the agents' — the whole response is
concentrated at lag zero and the market is (nearly) invertible.

As $g_s$ grows, adjustment becomes sluggish, the supply surprise is revealed only gradually through
the path of $(q_t, p_t)$, and a larger share of the response to $w_{st}$ shows up as a distributed
lag — the recovered innovation folds in more old news.

So the fraction at lag zero should *fall* as $g_s$ rises.

```{code-cell} python3
scale = np.diag([np.sqrt(σ_s2), np.sqrt(σ_d2)])
print(f"{'g_s':>6}   fraction of the supply-shock response at lag 0")
print(f"{'':>6}   (quantity innov., price innov.)")
for g in [0.1, 1.0, 10.0, 50.0, 100.0]:
    A_Fg, Cg, G_zg, *_ = solve_equilibrium(h_s, h_d, g, g_d, β, a, B_s, B_d)
    lss_g = qe.LinearStateSpace(A_Fg, Cg @ scale, G_zg, 1e-8 * np.eye(2))
    whit_g = qe.Kalman(lss_g).whitener_lss()
    resp = np.array([[c[0, 0], c[1, 0]] for c in whit_g.impulse_response(30)[1]])
    frac0 = np.abs(resp[0]) / np.abs(resp).sum(axis=0)
    print(f"{g:>6}   ({frac0[0]:.3f}, {frac0[1]:.3f})")
```

For $g_s \le 1$ the entire response arrives at lag zero: the market is invertible and a vector
autoregression recovers the agents' supply shock exactly.

As $g_s$ rises the fraction at lag zero falls monotonically toward roughly a quarter — three quarters
of the supply surprise is now revealed only with a lag.

This confirms that sluggish quantity adjustment is what drives a wedge between the shocks a vector
autoregression recovers and the shocks that actually move the agents.

```{solution-end}
```

```{exercise}
:label: sdvar_ex3

We verified the "Big $X$, little $x$" fixed point for the supplier.

Verify it for the **demander**: show that, facing the equilibrium price process, the demander's
best-response quantity rule reproduces the equilibrium quantity rule to machine precision.

The demander carries four own lags $(q_{t-1},\dots,q_{t-4})$; in equilibrium these equal the
market's lagged quantities, which are the first four components of the equilibrium state.
```

```{solution-start} sdvar_ex3
:class: dropdown
```

We map the demander's four own lags onto the first four components of the equilibrium state and
compare its implied quantity rule with the equilibrium rule.

```{code-cell} python3
# demander's own lags equal the market's lags: components 0..3 of the equilibrium state
own_to_state = np.zeros((4, nX))
own_to_state[np.arange(4), np.arange(4)] = 1.0

q_rule_demander = -F_d[0, :nX] - F_d[0, nX:nX + 4] @ own_to_state
q_rule_equilibrium = G_z[0]

gap = np.max(np.abs(q_rule_demander - q_rule_equilibrium))
print(f"max |demander best response − equilibrium quantity rule| : {gap:.2e}")
```

The demander's best response, facing the equilibrium price process, reproduces the equilibrium
quantity rule up to a tiny residual — the same fixed point that held for the supplier.

(The residual reflects the small ridge added to the planner's Riccati solve; without it the two
rules would agree to machine precision.)

Each agent is a price taker, and the equilibrium price process is precisely the one for which the
suppliers' and demanders' price-taking best responses clear the market period by period.

```{solution-end}
```
