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

(arellano)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Default Risk and Income Fluctuations

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} python
:tags: ["hide-output"]
!pip install --upgrade quantecon
```

## Overview

This lecture computes versions of  Arellano's  {cite}`arellano2008default`
model of sovereign default.

The model describes interactions among default risk, output, and an
equilibrium interest rate that includes a premium for endogenous default risk.

The decision maker is a government of a small open economy that borrows from
risk-neutral foreign creditors.

The foreign lenders must be compensated for default risk.

The government borrows and lends abroad in order to smooth the consumption of
its citizens.

The government repays its debt only if it wants to, but declining to pay has
adverse consequences.

The interest rate on government debt adjusts in response to the
state-dependent default probability chosen by government.

The model yields outcomes that help interpret sovereign default experiences,
including

* countercyclical interest rates on sovereign debt
* countercyclical trade balances
* high volatility of consumption relative to output

Notably, long recessions caused by bad draws in the income process increase the government's
incentive to default.

This can lead to

* spikes in interest rates
* temporary losses of access to international credit markets
* large drops in output, consumption, and welfare
* large capital outflows during recessions

Such dynamics are consistent with experiences of many countries.

Let's start with some imports:

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
from numba import njit, prange
```

## Structure

In this section we describe the main features of the model.

### Output, consumption and debt

A small open economy is endowed with an exogenous stochastically fluctuating potential output
stream $\{y_t\}$.

Potential output is realized only in periods in which the government honors its sovereign debt.

The output good can be traded or consumed.

The sequence $\{y_t\}$ is described by a Markov process with stochastic density kernel
$p(y, y')$.

Households within the country are identical and rank stochastic consumption streams according to

```{math}
:label: utility

\mathbb E \sum_{t=0}^{\infty} \beta^t u(c_t)
```

Here

* $0 < \beta < 1$ is a time discount factor
* $u$ is an increasing and strictly concave utility function

Consumption sequences enjoyed by households are affected by the government's decision to borrow or
lend internationally.

The government is benevolent in the sense that its aim is to maximize {eq}`utility`.

The government is the only domestic actor with access to foreign credit.

Because household are averse to consumption fluctuations, the government will try to smooth
consumption by borrowing from (and lending to) foreign creditors.

### Asset markets

The only credit instrument available to the government is a one-period bond traded in international credit markets.

The bond market has the following features

* The bond matures in one period and is not state contingent.
* A purchase of a bond with face value $B'$ is a claim to $B'$ units of the
  consumption good next period.
* To purchase $B'$  next period costs $q B'$ now, or, what is equivalent.
* For selling $-B'$ units of next period goods the seller earns $- q B'$ of today's
  goods.
    * If $B' < 0$, then $-q B'$ units of the good are received in the current period,
      for a promise to repay $-B'$ units next period.
    * There is an equilibrium  price function $q(B', y)$ that makes $q$ depend on both
      $B'$ and $y$.

Earnings on the government portfolio are distributed (or, if negative, taxed) lump sum to
households.

When the government is not excluded from financial markets, the one-period national budget
constraint is

```{math}
:label: resource

c = y + B - q(B', y) B'
```

Here and below, a prime denotes a next period value or a claim maturing next period.

To rule out Ponzi schemes, we also require that $B \geq -Z$ in every period.

* $Z$ is chosen to be sufficiently large that the constraint never binds in equilibrium.

### Financial markets

Foreign creditors

* are risk neutral
* know the domestic output stochastic process $\{y_t\}$ and observe
  $y_t, y_{t-1}, \ldots,$ at time $t$
* can borrow or lend without limit in an international credit market at a constant international
  interest rate $r$
* receive full payment if the government chooses to pay
* receive zero if the government defaults on its one-period debt due

When a government is expected to default next period with probability $\delta$, the expected
value of a promise to pay one unit of consumption next period is $1 - \delta$.

Therefore, the discounted expected value of a promise to pay $B$ next period is

```{math}
:label: epc

q = \frac{1 - \delta}{1 + r}
```

Next we turn to how the government in effect chooses the default probability $\delta$.

### Government's decisions

At each point in time $t$, the government chooses between

1. defaulting
1. meeting its current obligations and purchasing or selling an optimal quantity of  one-period
   sovereign debt

Defaulting means declining to repay all of its current obligations.

If the government defaults in the current period, then consumption equals current output.

But a sovereign default has two consequences:

1. Output immediately falls from $y$ to $h(y)$, where $0 \leq h(y) \leq y$.
    * It returns to $y$ only after the country regains access to international credit
      markets.
1. The country loses access to foreign credit markets.

### Reentering international credit market

While in a state of default, the economy regains access to foreign credit in each subsequent
period with probability $\theta$.

## Equilibrium

Informally, an equilibrium is a sequence of interest rates on its sovereign debt, a stochastic
sequence of government default decisions and an implied flow of household consumption such that

1. Consumption and assets satisfy the national budget constraint.
1. The government maximizes household utility taking into account
    * the resource constraint
    * the effect of its choices on the price of bonds
    * consequences of defaulting now for future net output and future borrowing and lending
      opportunities
1. The interest rate on the government's debt includes a risk-premium sufficient to make foreign
   creditors expect on average to earn the constant risk-free international interest rate.

To express these ideas more precisely, consider first the choices of the government, which

1. enters a period with initial assets $B$, or what is the same thing, initial debt to be
   repaid now of $-B$
1. observes current output $y$, and
1. chooses either
    1. to default, or
    1. to pay $-B$ and set next period's debt due to $-B'$

In a recursive formulation,

* state variables for the government comprise the pair $(B, y)$
* $v(B, y)$ is the optimum value of the government's problem when at the beginning of a
  period it faces the choice of whether to honor or default
* $v_c(B, y)$ is the value of choosing to pay obligations falling due
* $v_d(y)$ is the value of choosing to default

$v_d(y)$ does not depend on $B$ because, when access to credit is eventually regained,
net foreign assets equal $0$.

Expressed recursively, the value of defaulting is

$$
v_d(y) = u(h(y)) +
            \beta \int \left\{
            \theta v(0, y') + (1 - \theta) v_d(y')
            \right\}
            p(y, y') dy'
$$

The value of paying is

$$
v_c(B, y) = \max_{B' \geq -Z}
       \left\{
            u(y - q(B', y) B' + B) +
            \beta \int v(B', y') p(y, y') dy'
      \right\}
$$

The three value functions are linked by

$$
v(B, y) = \max\{ v_c(B, y), v_d(y) \}
$$

The government chooses to default when

$$
v_c(B, y) < v_d(y)
$$

and hence given $B'$ the probability of default next period is

```{math}
:label: delta

\delta(B', y) := \int \mathbb 1\{v_c(B', y') < v_d(y') \} p(y, y') dy'
```

Given zero profits for foreign creditors in equilibrium, we can combine {eq}`epc` and {eq}`delta`
to pin down the bond price function:

```{math}
:label: bondprice

q(B', y) = \frac{1 - \delta(B', y)}{1 + r}
```

### Definition of equilibrium

An *equilibrium* is

* a pricing function $q(B',y)$,
* a triple of value functions $(v_c(B, y), v_d(y), v(B,y))$,
* a decision rule telling the government when to default and when to pay as a function of the state
  $(B, y)$, and
* an asset accumulation rule that, conditional on choosing not to default, maps $(B,y)$ into
  $B'$

such that

* The three Bellman equations for $(v_c(B, y), v_d(y), v(B,y))$ are satisfied
* Given the price function $q(B',y)$, the default decision rule and the asset accumulation
  decision rule attain the optimal value function  $v(B,y)$, and
* The price function $q(B',y)$ satisfies equation {eq}`bondprice`

## Computation

Let's now compute an equilibrium of Arellano's model.

The equilibrium objects are the value function $v(B, y)$, the associated
default decision rule, and the pricing function $q(B', y)$.

We'll use our code to replicate Arellano's results.

After that we'll perform some additional simulations.

We use a slightly modified version of the algorithm recommended by Arellano.

* The appendix to {cite}`arellano2008default` recommends value function iteration until
  convergence, updating the price, and then repeating.
* Instead, we update the bond price at every value function iteration step.

The second approach is faster and the two different procedures deliver very similar results.

Here is a more detailed description of our algorithm:

1. Guess a pair of non-default and default value functions $v_c$ and $v_d$.
2. Using these functions, calculate the value function $v$, the corresponding default probabilities and the price function $q$.
2. At each pair $(B, y)$,
    1. update the value of defaulting $v_d(y)$.
    2. update the value of remaining $v_c(B, y)$.
4. Check for convergence. If converged, stop -- if not, go to step 2.

We use simple discretization on a grid of asset holdings and income levels.

The output process is discretized using a [quadrature method due to Tauchen](https://github.com/QuantEcon/QuantEcon.py/blob/master/quantecon/markov/approximation.py).

As we have in other places, we accelerate our code using Numba.

We define a class that will store parameters, grids and transition
probabilities.

```{code-cell} python
class Arellano_Economy:
    " Stores data and creates primitives for the Arellano economy. "

    def __init__(self,
            B_grid_size= 251,   # Grid size for bonds
            B_grid_min=-0.45,   # Smallest B value
            B_grid_max=0.45,    # Largest B value
            y_grid_size=51,     # Grid size for income
            β=0.953,            # Time discount parameter
            γ=2.0,              # Utility parameter
            r=0.017,            # Lending rate
            ρ=0.945,            # Persistence in the income process
            η=0.025,            # Standard deviation of the income process
            θ=0.282,            # Prob of re-entering financial markets
            def_y_param=0.969): # Parameter governing income in default

        # Save parameters
        self.β, self.γ, self.r, = β, γ, r
        self.ρ, self.η, self.θ = ρ, η, θ

        self.y_grid_size = y_grid_size
        self.B_grid_size = B_grid_size
        self.B_grid = np.linspace(B_grid_min, B_grid_max, B_grid_size)
        mc = qe.markov.tauchen(y_grid_size, ρ, η, 0, 3)
        self.y_grid, self.P = np.exp(mc.state_values), mc.P

        # The index at which B_grid is (close to) zero
        self.B0_idx = np.searchsorted(self.B_grid, 1e-10)

        # Output recieved while in default, with same shape as y_grid
        self.def_y = np.minimum(def_y_param * np.mean(self.y_grid), self.y_grid)

    def params(self):
        return self.β, self.γ, self.r, self.ρ, self.η, self.θ

    def arrays(self):
        return self.P, self.y_grid, self.B_grid, self.def_y, self.B0_idx
```

Notice how the class returns the data it stores as simple numerical values and
arrays via the methods `params` and `arrays`.

We will use this data in the Numba-jitted functions defined below.

Jitted functions prefer simple arguments, since type inference is easier.

Here is the utility function.


```{code-cell} python
@njit
def u(c, γ):
    return c**(1-γ)/(1-γ)
```

Here is a function to compute the bond price at each state, given $v_c$ and
$v_d$.

```{code-cell} python
@njit
def compute_q(v_c, v_d, q, params, arrays):
    """
    Compute the bond price function q(b, y) at each (b, y) pair.

    This function writes to the array q that is passed in as an argument.
    """

    # Unpack
    β, γ, r, ρ, η, θ = params
    P, y_grid, B_grid, def_y, B0_idx = arrays

    for B_idx in range(len(B_grid)):
        for y_idx in range(len(y_grid)):
            # Compute default probability and corresponding bond price
            delta = P[y_idx, v_c[B_idx, :] < v_d].sum()
            q[B_idx, y_idx] = (1 - delta ) / (1 + r)
```

Next we introduce Bellman operators that updated $v_d$ and $v_c$.

```{code-cell} python
@njit
def T_d(y_idx, v_c, v_d, params, arrays):
    """
    The RHS of the Bellman equation when income is at index y_idx and
    the country has chosen to default.  Returns an update of v_d.
    """
    # Unpack
    β, γ, r, ρ, η, θ = params
    P, y_grid, B_grid, def_y, B0_idx = arrays

    current_utility = u(def_y[y_idx], γ)
    v = np.maximum(v_c[B0_idx, :], v_d)
    cont_value = (θ * v + (1 - θ) * v_d) @ P[y_idx, :]

    return current_utility + β * cont_value


@njit
def T_c(B_idx, y_idx, v_c, v_d, q, params, arrays):
    """
    The RHS of the Bellman equation when the country is not in a
    defaulted state on their debt.  Returns a value that corresponds to
    v_c[B_idx, y_idx], as well as the optimal level of bond sales B'.
    """
    # Unpack
    β, γ, r, ρ, η, θ = params
    P, y_grid, B_grid, def_y, B0_idx = arrays
    B = B_grid[B_idx]
    y = y_grid[y_idx]

    # Compute the RHS of Bellman equation
    current_max = -1e10
    # Step through choices of next period B'
    for Bp_idx, Bp in enumerate(B_grid):
        c = y + B - q[Bp_idx, y_idx] * Bp
        if c > 0:
            v = np.maximum(v_c[Bp_idx, :], v_d)
            val = u(c, γ) + β * (v @ P[y_idx, :])
            if val > current_max:
                current_max = val
                Bp_star_idx = Bp_idx
    return current_max, Bp_star_idx
```

Here is a fast function that calls these operators in the right sequence.

```{code-cell} python
@njit(parallel=True)
def update_values_and_prices(v_c, v_d,      # Current guess of value functions
                             B_star, q,     # Arrays to be written to
                             params, arrays):

    # Unpack
    β, γ, r, ρ, η, θ = params
    P, y_grid, B_grid, def_y, B0_idx = arrays
    y_grid_size = len(y_grid)
    B_grid_size = len(B_grid)

    # Compute bond prices and write them to q
    compute_q(v_c, v_d, q, params, arrays)

    # Allocate memory
    new_v_c = np.empty_like(v_c)
    new_v_d = np.empty_like(v_d)

    # Calculate and return new guesses for v_c and v_d
    for y_idx in prange(y_grid_size):
        new_v_d[y_idx] = T_d(y_idx, v_c, v_d, params, arrays)
        for B_idx in range(B_grid_size):
            new_v_c[B_idx, y_idx], Bp_idx = T_c(B_idx, y_idx,
                                            v_c, v_d, q, params, arrays)
            B_star[B_idx, y_idx] = Bp_idx

    return new_v_c, new_v_d
```

We can now write a function that will use the `Arellano_Economy` class and the
functions defined above to compute the solution to our model.

We do not need to JIT compile this function since it only consists of outer
loops (and JIT compiling makes almost zero difference).

In fact, one of the jobs of this function is to take an instance of
`Arellano_Economy`, which is hard for the JIT compiler to handle, and strip it
down to more basic objects, which are then passed out to jitted functions.

```{code-cell} python
def solve(model, tol=1e-8, max_iter=10_000):
    """
    Given an instance of Arellano_Economy, this function computes the optimal
    policy and value functions.
    """
    # Unpack
    params = model.params()
    arrays = model.arrays()
    y_grid_size, B_grid_size = model.y_grid_size, model.B_grid_size

    # Initial conditions for v_c and v_d
    v_c = np.zeros((B_grid_size, y_grid_size))
    v_d = np.zeros(y_grid_size)

    # Allocate memory
    q = np.empty_like(v_c)
    B_star = np.empty_like(v_c, dtype=int)

    current_iter = 0
    dist = np.inf
    while (current_iter < max_iter) and (dist > tol):

        if current_iter % 100 == 0:
            print(f"Entering iteration {current_iter}.")

        new_v_c, new_v_d = update_values_and_prices(v_c, v_d, B_star, q, params, arrays)
        # Check tolerance and update
        dist = np.max(np.abs(new_v_c - v_c)) + np.max(np.abs(new_v_d - v_d))
        v_c = new_v_c
        v_d = new_v_d
        current_iter += 1

    print(f"Terminating at iteration {current_iter}.")
    return v_c, v_d, q, B_star
```

Finally, we write a function that will allow us to simulate the economy once
we have the policy functions

```{code-cell} python
def simulate(model, T, v_c, v_d, q, B_star, y_idx=None, B_idx=None):
    """
    Simulates the Arellano 2008 model of sovereign debt

    Here `model` is an instance of `Arellano_Economy` and `T` is the length of
    the simulation.  Endogenous objects `v_c`, `v_d`, `q` and `B_star` are
    assumed to come from a solution to `model`.

    """
    # Unpack elements of the model
    B0_idx = model.B0_idx
    y_grid = model.y_grid
    B_grid, y_grid, P = model.B_grid, model.y_grid, model.P

    # Set initial conditions to middle of grids
    if y_idx == None:
        y_idx = np.searchsorted(y_grid, y_grid.mean())
    if B_idx == None:
        B_idx = B0_idx
    in_default = False

    # Create Markov chain and simulate income process
    mc = qe.MarkovChain(P, y_grid)
    y_sim_indices = mc.simulate_indices(T+1, init=y_idx)

    # Allocate memory for outputs
    y_sim = np.empty(T)
    y_a_sim = np.empty(T)
    B_sim = np.empty(T)
    q_sim = np.empty(T)
    d_sim = np.empty(T, dtype=int)

    # Perform simulation
    t = 0
    while t < T:

        # Store the value of y_t and B_t
        y_sim[t] = y_grid[y_idx]
        B_sim[t] = B_grid[B_idx]

        # if in default:
        if v_c[B_idx, y_idx] < v_d[y_idx] or in_default:
            y_a_sim[t] = model.def_y[y_idx]
            d_sim[t] = 1
            Bp_idx = B0_idx
            # Re-enter financial markets next period with prob θ
            in_default = False if np.random.rand() < model.θ else True
        else:
            y_a_sim[t] = y_sim[t]
            d_sim[t] = 0
            Bp_idx = B_star[B_idx, y_idx]

        q_sim[t] = q[Bp_idx, y_idx]

        # Update time and indices
        t += 1
        y_idx = y_sim_indices[t]
        B_idx = Bp_idx

    return y_sim, y_a_sim, B_sim, q_sim, d_sim
```

## Results

Let's start by trying to replicate the results obtained in
{cite}`arellano2008default`.

In what follows, all results are computed using Arellano's parameter values.

The values can be seen in the `__init__` method of the `Arellano_Economy`
shown above.

For example, `r=0.017` matches the average quarterly rate on a 5 year US treasury over the period 1983--2001.

Details on how to compute the figures are reported as solutions to the
exercises.

The first figure shows the bond price schedule and replicates Figure 3 of
Arellano, where $y_L$ and $Y_H$ are particular below average and above average
values of output $y$.

```{figure} /_static/lecture_specific/arellano/arellano_bond_prices.png

```

* $y_L$ is 5% below the mean of the $y$ grid values
* $y_H$ is 5% above  the mean of the $y$ grid values

The grid used to compute this figure was relatively fine (`y_grid_size,
B_grid_size = 51, 251`), which explains the minor differences between this and
Arrelano's figure.

The figure shows that

* Higher levels of debt (larger $-B'$) induce larger discounts on the face value, which
  correspond to higher interest rates.
* Lower income also causes more discounting, as foreign creditors anticipate greater likelihood
  of default.

The next figure plots value functions and replicates the right hand panel of Figure 4 of
{cite}`arellano2008default`.

```{figure} /_static/lecture_specific/arellano/arellano_value_funcs.png

```

We can use the results of the computation to study the default probability $\delta(B', y)$
defined in {eq}`delta`.

The next plot shows these default probabilities over $(B', y)$ as a heat map.

```{figure} /_static/lecture_specific/arellano/arellano_default_probs.png

```

As anticipated, the probability that the government chooses to default in the following period
increases with indebtedness and falls with income.

Next let's run a time series simulation of $\{y_t\}$, $\{B_t\}$ and $q(B_{t+1}, y_t)$.

The grey vertical bars correspond to periods when the economy is excluded from financial markets because of a past default.

```{figure} /_static/lecture_specific/arellano/arellano_time_series.png

```

One notable feature of the simulated data is the nonlinear response of interest rates.

Periods of relative stability are followed by sharp spikes in the discount rate on government debt.

## Exercises

(arellano_ex1)=
```{exercise-start}
:label: arella_ex1
```

To the extent that you can, replicate the figures shown above

* Use the parameter values listed as defaults in `Arellano_Economy`.
* The time series will of course vary depending on the shock draws.

```{exercise-end}
```

```{solution-start} arella_ex1
:class: dropdown
```

Compute the value function, policy and equilibrium prices

```{code-cell} python
ae = Arellano_Economy()
```

```{code-cell} python
v_c, v_d, q, B_star = solve(ae)
```

Compute the bond price schedule as seen in figure 3 of Arellano (2008)

```{code-cell} python
# Unpack some useful names
B_grid, y_grid, P = ae.B_grid, ae.y_grid, ae.P
B_grid_size, y_grid_size = len(B_grid), len(y_grid)
r = ae.r

# Create "Y High" and "Y Low" values as 5% devs from mean
high, low = np.mean(y_grid) * 1.05, np.mean(y_grid) * .95
iy_high, iy_low = (np.searchsorted(y_grid, x) for x in (high, low))

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_title("Bond price schedule $q(y, B')$")

# Extract a suitable plot grid
x = []
q_low = []
q_high = []
for i, B in enumerate(B_grid):
    if -0.35 <= B <= 0:  # To match fig 3 of Arellano
        x.append(B)
        q_low.append(q[i, iy_low])
        q_high.append(q[i, iy_high])
ax.plot(x, q_high, label="$y_H$", lw=2, alpha=0.7)
ax.plot(x, q_low, label="$y_L$", lw=2, alpha=0.7)
ax.set_xlabel("$B'$")
ax.legend(loc='upper left', frameon=False)
plt.show()
```

Draw a plot of the value functions

```{code-cell} python
v = np.maximum(v_c, np.reshape(v_d, (1, y_grid_size)))

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_title("Value Functions")
ax.plot(B_grid, v[:, iy_high], label="$y_H$", lw=2, alpha=0.7)
ax.plot(B_grid, v[:, iy_low], label="$y_L$", lw=2, alpha=0.7)
ax.legend(loc='upper left')
ax.set(xlabel="$B$", ylabel="$v(y, B)$")
ax.set_xlim(min(B_grid), max(B_grid))
plt.show()
```

Draw a heat map for default probability

```{code-cell} python
xx, yy = B_grid, y_grid
zz = np.empty_like(v_c)

for B_idx in range(B_grid_size):
    for y_idx in range(y_grid_size):
        zz[B_idx, y_idx] = P[y_idx, v_c[B_idx, :] < v_d].sum()

# Create figure
fig, ax = plt.subplots(figsize=(10, 6.5))
hm = ax.pcolormesh(xx, yy, zz.T)
cax = fig.add_axes([.92, .1, .02, .8])
fig.colorbar(hm, cax=cax)
ax.axis([xx.min(), 0.05, yy.min(), yy.max()])
ax.set(xlabel="$B'$", ylabel="$y$", title="Probability of Default")
plt.show()
```

Plot a time series of major variables simulated from the model

```{code-cell} python
T = 250
np.random.seed(42)
y_sim, y_a_sim, B_sim, q_sim, d_sim = simulate(ae, T, v_c, v_d, q, B_star)

# Pick up default start and end dates
start_end_pairs = []
i = 0
while i < len(d_sim):
    if d_sim[i] == 0:
        i += 1
    else:
        # If we get to here we're in default
        start_default = i
        while i < len(d_sim) and d_sim[i] == 1:
            i += 1
        end_default = i - 1
        start_end_pairs.append((start_default, end_default))

plot_series = (y_sim, B_sim, q_sim)
titles = 'output', 'foreign assets', 'bond price'

fig, axes = plt.subplots(len(plot_series), 1, figsize=(10, 12))
fig.subplots_adjust(hspace=0.3)

for ax, series, title in zip(axes, plot_series, titles):
    # Determine suitable y limits
    s_max, s_min = max(series), min(series)
    s_range = s_max - s_min
    y_max = s_max + s_range * 0.1
    y_min = s_min - s_range * 0.1
    ax.set_ylim(y_min, y_max)
    for pair in start_end_pairs:
        ax.fill_between(pair, (y_min, y_min), (y_max, y_max),
                        color='k', alpha=0.3)
    ax.grid()
    ax.plot(range(T), series, lw=2, alpha=0.7)
    ax.set(title=title, xlabel="time")

plt.show()
```
```{solution-end}
```
