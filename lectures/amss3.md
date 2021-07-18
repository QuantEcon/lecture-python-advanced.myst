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

(amss3)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Fiscal Risk and Government Debt

```{contents} Contents
:depth: 2
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython
---
tags: [hide-output]
---
!pip install --upgrade quantecon
```

## Overview

This lecture studies government debt in an AMSS
economy {cite}`aiyagari2002optimal` of the type described in {doc}`Optimal Taxation without State-Contingent Debt <amss>`.

We study the behavior of government debt  as time $t \rightarrow + \infty$.

We use these techniques

* simulations
* a regression coefficient from the tail of a long simulation that allows us to verify that  the asymptotic mean of government debt solves
  a fiscal-risk minimization  problem
* an approximation to the **mean** of an ergodic distribution of government debt
* an approximation  to the **rate of convergence** to an ergodic distribution of government debt

We apply tools that are applicable to  more general incomplete markets economies that are presented on pages 648 - 650 in section III.D
of {cite}`BEGS1` (BEGS).

We study an AMSS economy {cite}`aiyagari2002optimal`  with  three Markov states driving government expenditures.

* In a {doc}`previous lecture <amss2>`, we showed that with only two Markov states, it is possible that  endogenous
  interest rate fluctuations eventually can support complete markets allocations and Ramsey outcomes.
* The presence of three states  prevents the full spanning that eventually prevails in the two-state example featured in
  {doc}`Fiscal Insurance via Fluctuating Interest Rates <amss2>`.

The lack of full spanning means that the ergodic distribution of the par value of government debt is nontrivial, in contrast to the situation
in {doc}`Fiscal Insurance via Fluctuating Interest Rates <amss2>`  in which the ergodic distribution of the par value of government debt is concentrated on one point.

Nevertheless,   {cite}`BEGS1` (BEGS) establish that, for general settings that include ours, the Ramsey
planner steers government assets to a level that comes
**as close as possible** to providing full spanning in a precise a sense defined by
BEGS that we describe below.

We use code constructed in {doc}`Fluctuating Interest Rates Deliver Fiscal Insurance <amss2>`.

**Warning:** Key equations in  {cite}`BEGS1` section III.D carry  typos  that we correct below.

Let's start with some imports:

```{code-cell} ipython
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.optimize import minimize
```

## The Economy

As in {doc}`Optimal Taxation without State-Contingent Debt <amss>` and {doc}`Optimal Taxation with State-Contingent Debt <opt_tax_recur>`,
we assume that the  representative agent has  utility function

$$
u(c,n) = {\frac{c^{1-\sigma}}{1-\sigma}} - {\frac{n^{1+\gamma}}{1+\gamma}}
$$

We work directly with labor supply instead of leisure.

We assume that

$$
c_t + g_t = n_t
$$

The Markov state $s_t$ takes **three** values, namely,  $0,1,2$.

The initial Markov state is $0$.

The Markov transition matrix is $(1/3) I$ where $I$ is a $3 \times 3$ identity matrix, so the $s_t$ process is IID.

Government expenditures $g(s)$ equal $.1$ in Markov state $0$, $.2$ in Markov state $1$, and $.3$
in Markov state $2$.

We set preference parameters

$$
\begin{aligned}
\beta & = .9 \cr
\sigma & = 2  \cr
\gamma & = 2
\end{aligned}
$$

The following Python code sets up the economy

```{code-cell} python3
:load: _static/lecture_specific/amss2/crra_utility.py
```

### First and Second Moments

We'll want  first and second moments of some key random variables below.

The following code computes these moments; the code is recycled from {doc}`Fluctuating Interest Rates Deliver Fiscal Insurance <amss2>`.

```{code-cell} python3
def mean(x, s):
    '''Returns mean for x given initial state'''
    x = np.array(x)
    return x @ u.π[s]

def variance(x, s):
    x = np.array(x)
    return x**2 @ u.π[s] - mean(x, s)**2

def covariance(x, y, s):
    x, y = np.array(x), np.array(y)
    return x * y @ u.π[s] - mean(x, s) * mean(y, s)
```

## Long Simulation

To generate a long simulation we use the following code.

We begin by showing the code that we used in earlier lectures on the AMSS model.

Here it is

```{code-cell} python3
---
load: _static/lecture_specific/amss2/sequential_allocation.py
tags: [collapse-20]
---
```

```{code-cell} python3
---
load: _static/lecture_specific/amss2/recursive_allocation.py
tags: [collapse-20]
---
```

```{code-cell} python3
---
load: _static/lecture_specific/amss2/utilities.py
tags: [collapse-20]
---
```

Next, we show the code that we use to generate a very long simulation starting from initial
government debt equal to $-.5$.

Here is a graph of a long simulation of 102000 periods.

```{code-cell} python3
:tags: ["scroll-output"]
μ_grid = np.linspace(-0.09, 0.1, 100)

log_example = CRRAutility(π=np.full((3, 3), 1 / 3),
                          G=np.array([0.1, 0.2, .3]),
                          Θ=np.ones(3))

log_example.transfers = True        # Government can use transfers
log_sequential = SequentialAllocation(log_example)  # Solve sequential problem
log_bellman = RecursiveAllocationAMSS(log_example, μ_grid,
                                       tol=1e-12, tol_diff=1e-10)



T = 102000  # Set T to 102000 periods

sim_seq_long = log_sequential.simulate(0.5, 0, T)
sHist_long = sim_seq_long[-3]
sim_bel_long = log_bellman.simulate(0.5, 0, T, sHist_long)

titles = ['Government Debt', 'Tax Rate']

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

for ax, title, id in zip(axes.flatten(), titles, [2, 3]):
    ax.plot(sim_seq_long[id], '-k', sim_bel_long[id], '-.b', alpha=0.5)
    ax.set(title=title)
    ax.grid()

axes[0].legend(('Complete Markets', 'Incomplete Markets'))
plt.tight_layout()
plt.show()
```

```{figure} /_static/lecture_specific/amss3/amss3_g1.png

```

The long simulation apparently  indicates eventual convergence to an ergodic distribution.

It takes about 1000 periods to reach the ergodic distribution -- an outcome that is forecast by
approximations to rates of convergence that appear in BEGS {cite}`BEGS1` and that we discuss in {doc}`Fluctuating Interest Rates Deliver Fiscal Insurance <amss2>`.

Let's discard the first 2000 observations of the simulation and construct the histogram of
the par value of government debt.

We obtain the following graph for the histogram of the last 100,000 observations on the par value of government debt.

```{figure} /_static/lecture_specific/amss3/amss3_g3.png
```

The  black vertical line denotes the sample mean for the last 100,000 observations included in the histogram; the  green vertical line denotes the
value of $\frac{ {\mathcal B}^*}{E u_c}$, associated with a sample from our approximation to
the ergodic distribution where ${\mathcal B}^*$ is a regression coefficient to be described below;  the red vertical line denotes an approximation by {cite}`BEGS1` to the mean of the ergodic
distribution that can be computed **before**  the ergodic distribution has been approximated, as described below.

Before moving on to discuss the histogram and the vertical lines approximating the ergodic  mean of government debt in more detail, the following graphs show
government debt and taxes early in the simulation, for periods 1-100 and 101 to 200
respectively.

```{code-cell} python3
titles = ['Government Debt', 'Tax Rate']

fig, axes = plt.subplots(4, 1, figsize=(10, 15))

for i, id in enumerate([2, 3]):
    axes[i].plot(sim_seq_long[id][:99], '-k', sim_bel_long[id][:99],
                 '-.b', alpha=0.5)
    axes[i+2].plot(range(100, 199), sim_seq_long[id][100:199], '-k',
                   range(100, 199), sim_bel_long[id][100:199], '-.b',
                   alpha=0.5)
    axes[i].set(title=titles[i])
    axes[i+2].set(title=titles[i])
    axes[i].grid()
    axes[i+2].grid()

axes[0].legend(('Complete Markets', 'Incomplete Markets'))
plt.tight_layout()
plt.show()
```

```{figure} /_static/lecture_specific/amss3/amss3_g2.png

```

For the short samples early in our simulated sample of  102,000 observations, fluctuations in government debt and the tax rate
conceal the weak but inexorable force that the Ramsey planner puts into both series driving them toward ergodic marginal distributions that are far from
these early observations

* early observations are more influenced by the initial value of the par value of government debt than by the ergodic mean of the par value of government debt
* much later observations are more influenced by the ergodic mean and are independent of the par value of initial government debt

## Asymptotic Mean and Rate of Convergence

We apply the results of BEGS {cite}`BEGS1` to interpret

* the mean of the ergodic distribution of government debt
* the rate of convergence  to the ergodic distribution from an arbitrary initial government debt

We begin by computing  objects required by the theory of section III.i
of BEGS {cite}`BEGS1`.

As in {doc}`Fiscal Insurance via Fluctuating Interest Rates <amss2>`, we recall  that  BEGS {cite}`BEGS1` used a particular
notation to represent what we can regard as their  generalization of an AMSS model.

We introduce some of the  {cite}`BEGS1` notation so that readers can quickly relate notation that appears in key BEGS formulas to the notation
that we have used in previous lectures {doc}`here <amss>` and {doc}`here <amss2>`.

BEGS work with objects $B_t, {\mathcal B}_t, {\mathcal R}_t, {\mathcal X}_t$ that are related to  notation that we used in
earlier lectures by

$$
\begin{aligned}
{\mathcal R}_t & = \frac{u_{c,t}}{u_{c,t-1}} R_{t-1}  = \frac{u_{c,t}}{ \beta E_{t-1} u_{c,t}} \\
B_t & = \frac{b_{t+1}(s^t)}{R_t(s^t)} \\
b_t(s^{t-1}) & = {\mathcal R}_{t-1} B_{t-1} \\
{\mathcal B}_t & = u_{c,t} B_t = (\beta E_t u_{c,t+1}) b_{t+1}(s^t) \\
{\mathcal X}_t & = u_{c,t} [g_t - \tau_t n_t]
\end{aligned}
$$

BEGS {cite}`BEGS1` call ${\mathcal X}_t$ the **effective** government deficit and ${\mathcal B}_t$ the **effective** government debt.

Equation (44) of {cite}`BEGS1` expresses the time $t$ state $s$ government budget constraint as

```{math}
:label: eq_fiscal_risk_1

{\mathcal B}(s) = {\mathcal R}_\tau(s, s_{-}) {\mathcal B}_{-} + {\mathcal X}_{\tau} (s)
```

where the dependence on $\tau$ is meant to remind us that these objects depend on the tax rate;  $s_{-}$ is last period's Markov state.

BEGS interpret random variations in the right side of {eq}`eq_fiscal_risk_1`  as **fiscal risks** generated by

- interest-rate-driven fluctuations in time $t$ effective payments due on the government portfolio, namely,
  ${\mathcal R}_\tau(s, s_{-}) {\mathcal B}_{-}$,  and
- fluctuations in the effective government deficit ${\mathcal X}_t$

### Asymptotic Mean

BEGS give conditions under which the ergodic mean of ${\mathcal B}_t$ is approximated by

```{math}
:label: prelim_formula_1

{\mathcal B}^* = - \frac{\rm cov^{\infty}({\mathcal R}_t, {\mathcal X_t})}{\rm var^{\infty}({\mathcal R}_t)}
```

where the superscript $\infty$ denotes a moment taken with respect to an ergodic distribution.

Formula {eq}`prelim_formula_1` represents ${\mathcal B}^*$ as a regression coefficient of ${\mathcal X}_t$ on ${\mathcal R}_t$ in the ergodic
distribution.

Regression coefficient ${\mathcal B}^*$ solves  a variance-minimization problem:

```{math}
:label: eq_criterion_fiscal_1

{\mathcal B}^* = {\rm argmin}_{\mathcal B}  {\rm var}^\infty ({\mathcal R} {\mathcal B} + {\mathcal X})
```

The minimand in criterion {eq}`eq_criterion_fiscal_1`  measures  **fiscal risk** associated with a given tax-debt policy that appears on the right side
of equation {eq}`eq_fiscal_risk_1`.

Expressing formula {eq}`prelim_formula_1` in terms of  our notation tells us that the ergodic mean of the par value $b$ of government debt in the
AMSS model should be approximately

```{math}
:label: key_formula_1

\hat b = \frac{\mathcal B^*}{\beta E( E_t u_{c,t+1})} = \frac{\mathcal B^*}{\beta E( u_{c,t+1} )}
```

where mathematical expectations are taken with respect to the ergodic distribution.

### Rate of Convergence

BEGS also derive the following  approximation to the rate of convergence to ${\mathcal B}^{*}$ from an arbitrary initial condition.

> ```{math}
> :label: rate_of_convergence_1
> 
> \frac{ E_t  ( {\mathcal B}_{t+1} - {\mathcal B}^{*} )} { ( {\mathcal B}_{t} - {\mathcal B}^{*} )} \approx \frac{1}{1 + \beta^2 {\rm var}^\infty ({\mathcal R} )}
> ```
> 
> 

(See the equation above equation (47) in BEGS {cite}`BEGS1`)

### More Advanced Topic

The remainder of this lecture is about  technical material based on  formulas from BEGS {cite}`BEGS1`.

The topic involves interpreting  and extending formula {eq}`eq_criterion_fiscal_1` for the ergodic mean ${\mathcal B}^*$.

### Chicken and Egg

Notice how attributes of the ergodic distribution for ${\mathcal B}_t$  appear
on the right side of  formula {eq}`eq_criterion_fiscal_1` for approximating  the ergodic mean via ${\mathcal B}^*$.

Therefor,  formula  {eq}`eq_criterion_fiscal_1` is not useful for estimating  the mean of the ergodic **in advance** of actually approximating the ergodic distribution.

* we need to know the  ergodic distribution to compute the right side of formula {eq}`eq_criterion_fiscal_1`

So the primary use of equation {eq}`eq_criterion_fiscal_1` is how  it  **confirms** that
the ergodic distribution solves a **fiscal-risk minimization problem**.

As an example, notice how we used the formula for the mean of ${\mathcal B}$ in the ergodic distribution of the special AMSS economy in
{doc}`Fiscal Insurance via Fluctuating Interest Rates <amss2>`

* **first** we computed the ergodic distribution using a reverse-engineering construction
* **then** we verified that ${\mathcal B}^*$  agrees with the mean of that distribution

### Approximating the Ergodic Mean

BEGS also {cite}`BEGS1` propose  an approximation to  ${\mathcal B}^*$ that can be computed **without** first approximating  the
ergodic distribution.

To  construct the BEGS  approximation to ${\mathcal B}^*$, we just follow steps set forth on pages 648 - 650 of section III.D of
{cite}`BEGS1`

- notation in BEGS might be confusing at first sight, so
  it is important to stare and digest before computing
- there are also some sign errors in the {cite}`BEGS1` text that we'll want
  to correct here

Here is a step-by-step description of the BEGS {cite}`BEGS1` approximation procedure.

### Step by Step

**Step 1:** For a given $\tau$ we  compute a vector of
values $c_\tau(s), s= 1, 2, \ldots, S$ that satisfy

$$
(1-\tau) c_\tau(s)^{-\sigma} - (c_{\tau}(s) + g(s))^{\gamma} = 0
$$

This is a nonlinear equation to be solved for
$c_{\tau}(s), s = 1, \ldots, S$.

$S=3$ in our case, but we'll write code for a general integer
$S$.

**Typo alert:** Please note that there is a sign error in equation (42)
of BEGS {cite}`BEGS1` -- it should be a **minus** rather than a **plus** in the middle.

* We have made the appropriate correction in the above equation.

**Step 2:** Knowing $c_\tau(s), s=1, \ldots, S$ for a given
$\tau$, we want to compute the random variables

$$
{\mathcal  R}_\tau(s) = \frac{c_\tau(s)^{-\sigma}}{\beta \sum_{s'=1}^S c_\tau(s')^{-\sigma} \pi(s')}
$$

and

$$
{\mathcal X}_\tau(s) = (c_\tau(s) + g(s))^{1+ \gamma} - c_\tau(s)^{1-\sigma}
$$

each for $s= 1, \ldots, S$.

BEGS call ${\mathcal  R}_\tau(s)$
the **effective return** on risk-free debt and they call
${\mathcal X}_\tau(s)$ the **effective government deficit**.

**Step 3:** With the preceding objects in hand, for a given
${\mathcal B}$, we seek a $\tau$ that satisfies

$$
{\mathcal B} = - \frac{\beta} {1-\beta} E {\mathcal X_\tau} \equiv - \frac{\beta} {1-\beta} \sum_{s} {\mathcal X}_\tau(s) \pi(s)
$$

This equation says that at a constant discount factor $\beta$,  equivalent government debt ${\mathcal B}$ equals the
present value of the mean effective government **surplus**.

**Another typo alert**: there is a sign error in equation (46) of BEGS {cite}`BEGS1` --the left
side should be multiplied by $-1$.

* We have made this correction in the above equation.

For a given ${\mathcal B}$, let a $\tau$ that solves the
above equation be called $\tau(\mathcal B)$.

We'll use a Python root solver to find a $\tau$ that solves this
equation for a given ${\mathcal B}$.

We'll use this function to induce a function $\tau({\mathcal B})$.

**Step 4:** With a Python program that computes
$\tau(\mathcal B)$ in hand, next we write a Python function to
compute the random variable.

$$
J({\mathcal B})(s) =  \mathcal R_{\tau({\mathcal B})}(s) {\mathcal B} + {\mathcal X}_{\tau({\mathcal B})}(s) ,  \quad s = 1, \ldots, S
$$

**Step 5:** Now that we have a way to compute the random variable
$J({\mathcal B})(s), s= 1, \ldots, S$, via  a composition of  Python
functions, we can use the population variance  function that we
defined in the code above to construct a function
${\rm var}(J({\mathcal B}))$.

We put ${\rm var}(J({\mathcal B}))$ into a Python function minimizer and
compute

$$
{\mathcal B}^* = {\rm argmin}_{\mathcal B} {\rm var } (J({\mathcal B}) )
$$

**Step 6:** Next we take the minimizer ${\mathcal B}^*$ and the
Python functions for computing means and variances and compute

$$
{\rm rate} = \frac{1}{1 + \beta^2 {\rm var}( {\mathcal R}_{\tau({\mathcal B}^*)} )}
$$

Ultimate outputs of this string of calculations are two scalars

$$
({\mathcal B}^*, {\rm rate} )
$$

**Step 7:** Compute the divisor

$$
div = {\beta E u_{c,t+1}}
$$

and then compute the mean of the par value of government debt in the AMSS model

$$
\hat b = \frac{ {\mathcal B}^*}{div}
$$

In the two-Markov-state AMSS economy in {doc}`Fiscal Insurance via Fluctuating Interest Rates <amss2>`,
$E_t u_{c,t+1} = E u_{c,t+1}$ in the ergodic distribution.

We  have confirmed that
this formula very accurately describes a **constant** par value of government debt that

* supports full fiscal insurance via fluctuating interest parameters, and
* is the limit of government debt as $t \rightarrow +\infty$

In the three-Markov-state economy of this lecture, the par value of government debt fluctuates in a history-dependent way even asymptotically.

In this economy, $\hat b$ given by the above formula approximates the mean of the ergodic distribution of  the par value of  government debt

so while the approximation circumvents the chicken and egg problem that  surrounds
: the much better approximation associated with the green vertical line, it does so by enlarging the approximation error

* $\hat b$ is represented by the red vertical line plotted in the histogram of the last 100,000 observations of our simulation of the  par value of government debt plotted above
* the approximation is fairly accurate but not perfect

### Execution

Now let's move on to compute things step by step.

#### Step 1

```{code-cell} python3
u = CRRAutility(π=np.full((3, 3), 1 / 3),
                G=np.array([0.1, 0.2, .3]),
                Θ=np.ones(3))

τ = 0.05           # Initial guess of τ (to displays calcs along the way)
S = len(u.G)       # Number of states

def solve_c(c, τ, u):
    return (1 - τ) * c**(-u.σ) - (c + u.G)**u.γ

# .x returns the result from root
c = root(solve_c, np.ones(S), args=(τ, u)).x
c
```

```{code-cell} python3
root(solve_c, np.ones(S), args=(τ, u))
```

#### Step 2

```{code-cell} python3
n = c + u.G   # Compute labor supply
```

### Note about Code

Remember that in our code $\pi$ is a $3 \times 3$ transition
matrix.

But because we are studying an IID case, $\pi$ has identical
rows and we need only  to compute objects for one row of $\pi$.

This explains why at some places below we set $s=0$ just to pick
off the first row of $\pi$.

### Running the code

Let's take the code out for a spin.

First, let's compute ${\mathcal R}$ and ${\mathcal X}$
according to our formulas

```{code-cell} python3
def compute_R_X(τ, u, s):
    c = root(solve_c, np.ones(S), args=(τ, u)).x  # Solve for vector of c's
    div = u.β * (u.Uc(c[0], n[0]) * u.π[s, 0]  \
                 +  u.Uc(c[1], n[1]) * u.π[s, 1] \
                 +  u.Uc(c[2], n[2]) * u.π[s, 2])
    R = c**(-u.σ) / (div)
    X = (c + u.G)**(1 + u.γ) - c**(1 - u.σ)
    return R, X
```

```{code-cell} python3
c**(-u.σ) @ u.π
```

```{code-cell} python3
u.π
```

We only want unconditional expectations because we are in an IID case.

So we'll set $s=0$ and just pick off expectations associated with
the first row of $\pi$

```{code-cell} python3
s = 0

R, X = compute_R_X(τ, u, s)
```

Let's look at the random variables ${\mathcal R}, {\mathcal X}$

```{code-cell} python3
R
```

```{code-cell} python3
mean(R, s)
```

```{code-cell} python3
X
```

```{code-cell} python3
mean(X, s)
```

```{code-cell} python3
X @ u.π
```

#### Step 3

```{code-cell} python3
def solve_τ(τ, B, u, s):
    R, X = compute_R_X(τ, u, s)
    return ((u.β - 1) / u.β) * B - X @ u.π[s]
```

Note that $B$ is a scalar.

Let's try out our method computing $\tau$

```{code-cell} python3
s = 0
B = 1.0

τ = root(solve_τ, .1, args=(B, u, s)).x[0]  # Very sensitive to initial value
τ
```

In the above cell, B is fixed at 1 and $\tau$ is to be computed as
a function of B.

Note that 0.2 is the initial value for $\tau$ in the root-finding
algorithm.

#### Step 4

```{code-cell} python3
def min_J(B, u, s):
    # Very sensitive to initial value of τ
    τ = root(solve_τ, .5, args=(B, u, s)).x[0]
    R, X = compute_R_X(τ, u, s)
    return variance(R * B + X, s)
```

```{code-cell} python3
min_J(B, u, s)
```

#### Step 6

```{code-cell} python3
B_star = minimize(min_J, .5, args=(u, s)).x[0]
B_star
```

```{code-cell} python3
n = c + u.G  # Compute labor supply
```

```{code-cell} python3
div = u.β * (u.Uc(c[0], n[0]) * u.π[s, 0]  \
             +  u.Uc(c[1], n[1]) * u.π[s, 1] \
             +  u.Uc(c[2], n[2]) * u.π[s, 2])
```

```{code-cell} python3
B_hat = B_star/div
B_hat
```

```{code-cell} python3
τ_star = root(solve_τ, 0.05, args=(B_star, u, s)).x[0]
τ_star
```

```{code-cell} python3
R_star, X_star = compute_R_X(τ_star, u, s)
R_star, X_star
```

```{code-cell} python3
rate = 1 / (1 + u.β**2 * variance(R_star, s))
rate
```

```{code-cell} python3
root(solve_c, np.ones(S), args=(τ_star, u)).x
```

