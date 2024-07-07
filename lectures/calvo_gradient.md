---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# A Model of Calvo

This lecture describes a  linear-quadratic versions of a model that Guillermo Calvo {cite}`Calvo1978` used to illustrate the **time inconsistency** of optimal government
plans.


The model focuses attention on intertemporal tradeoffs between

- welfare benefits that anticipations of future  deflation generate  by decreasing  costs of holding real money balances and thereby increasing a representative agent's *liquidity*, as measured by his or her holdings of real money balances, and
- costs associated with the  distorting taxes that the government must levy in order to acquire the paper money that it will  destroy  in order to generate anticipated deflation

The model features

- rational expectations
- costly government actions at all dates $t \geq 1$ that increase household utilities at dates before $t$


We'll use ideas from  papers by Cagan {cite}`Cagan` and  Calvo {cite}`Calvo1978`



## Model components

There is no uncertainty.

Let:

- $p_t$ be the log of the price level
- $m_t$ be the log of nominal money balances
- $\theta_t = p_{t+1} - p_t$ be the net rate of inflation between $t$ and $t+1$
- $\mu_t = m_{t+1} - m_t$ be the net rate of growth of nominal balances

The demand for real balances is governed by a perfect foresight
version of a Cagan {cite}`Cagan` demand function for real balances:

```{math}
:label: eq_old1

m_t - p_t = -\alpha(p_{t+1} - p_t) \: , \: \alpha > 0
```

for $t \geq 0$.

Equation {eq}`eq_old1` asserts that the demand for real balances is inversely
related to the public's expected rate of inflation, which  equals
the actual rate of inflation because there is no uncertainty here.

(When there is no uncertainty, an assumption of **rational expectations** that becomes equivalent to  **perfect foresight**).


Subtracting the demand function {eq}`eq_old1` at time $t$ from the demand
function at $t+1$ gives:

$$
\mu_t - \theta_t = -\alpha \theta_{t+1} + \alpha \theta_t
$$

or

```{math}
:label: eq_old2

\theta_t = \frac{\alpha}{1+\alpha} \theta_{t+1} + \frac{1}{1+\alpha} \mu_t
```

Because $\alpha > 0$,  $0 < \frac{\alpha}{1+\alpha} < 1$.

**Definition:** For  scalar $b_t$, let $L^2$ be the space of sequences
$\{b_t\}_{t=0}^\infty$ satisfying

$$
\sum_{t=0}^\infty  b_t^2 < +\infty
$$

We say that a sequence that belongs to $L^2$   is **square summable**.

When we assume that the sequence $\vec \mu = \{\mu_t\}_{t=0}^\infty$ is square summable and we require that the sequence $\vec \theta = \{\theta_t\}_{t=0}^\infty$ is square summable,
the linear difference equation {eq}`eq_old2` can be solved forward to get:

```{math}
:label: eq_old3

\theta_t = \frac{1}{1+\alpha} \sum_{j=0}^\infty \left(\frac{\alpha}{1+\alpha}\right)^j \mu_{t+j}
```

```{note}
Equations {eq}`eq_old1` and {eq}`eq_old3` show that $\theta_t$ intermediates
how choices of $\mu_{t+j}, \ j=0, 1, \ldots$ impinge on time $t$
real balances $m_t - p_t = -\alpha \theta_t$.  Chang {cite}`chang1998credible` exploits this
fact extensively.
``` 

An equivalence class of continuation money growth sequences $\{\mu_{t+j}\}_{j=0}^\infty$ deliver the same $\theta_t$.



The  government  values  a representative household's utility of real balances at time $t$ according to the utility function

```{math}
:label: eq_old5

U(m_t - p_t) = u_0 + u_1 (m_t - p_t) - \frac{u_2}{2} (m_t - p_t)^2, \quad u_0 > 0, u_1 > 0, u_2 > 0
```

The money demand function {eq}`eq_old1` and the utility function {eq}`eq_old5` imply that 

$$
U(-\alpha \theta_t) = u_1 + u_2 (-\alpha \theta_t) -\frac{u_2}{2}(-\alpha \theta_t)^2 . 
$$ (eq_old5a)


```{note}
The "bliss level" of real balances is  $\frac{u_1}{u_2}$ and the inflation rate that attains
it is $-\frac{u_1}{u_2 \alpha}$.
```

Via equation {eq}`eq_old3`, a government plan
$\vec \mu = \{\mu_t \}_{t=0}^\infty$ leads to a
sequence of inflation outcomes
$\vec \theta = \{ \theta_t \}_{t=0}^\infty$.

We assume that the government incurs  social costs $\frac{c}{2} \mu_t^2$ at
$t$ when it  changes the stock of nominal money
balances at rate $\mu_t$.

Therefore, the one-period welfare function of a benevolent government
is:

$$
v_0 = \sum_{t=0}^\infty \beta^t s(\theta_t, \mu_t) 
$$

where $\beta \in (0,1)$ is a discount factor and  the goverment's  one-period welfare function is  

$$
s(\theta_t,\mu_t) = U(-\alpha \theta_t) - \frac{c}{2} \mu_t^2  .
$$



## A Machine Learning approach

This notebook describes non Pythonic, Tom style pseudo code for  solving Calvo's Ramsey problem using 
a "machine learning" approach.

We want to compute a vector of money growth rates $(\mu_0, \mu_1, \ldots, \mu_{T-1}, \bar \mu)$ 
to maximize the function $\tilde V$ below.

## Parameters and variables

We'll start by setting them at the default values from {doc}`calvo`.

Parameters are

Demand for money: $\alpha > 0$, default $\alpha = 1$

   * Induced demand function for money parameter $\lambda = \frac{\alpha}{1+\alpha}$

Utility function $u_0, u_1, u_2 $ and $\beta \in (0,1)$

Cost parameter of tax distortions associated with setting $\mu_t \neq 0$ is $c$

Variables are

 * $\theta_t = p_{t+1} - p_t$ where $p_t$ is log of price level
 
 * $\mu_t = m_{t+1} - m_t $ where $m_t$ is log of money supply


Truncation parameter $T >0$

  * Positive integer $T$  is a new parameter that we'll use in our approximation algorithm below

### Basic objects

$$
(\vec \theta, \vec \mu) = \{\theta_t, \mu_t\}_{t=0}^\infty
$$

$$
V = \sum_{t=0}^\infty \beta^t (h_0 + h_1 \theta_t + h_2 \theta_t^2 -
\frac{c}{2} \mu_t^2 )
$$

where $h_0, h_1, h_2$ are chosen to make

$$
u_0 + u_1(-\alpha \theta_t) - \frac{u_2}{2} (-\alpha \theta_t)^2
$$

match 

$$
h_0 + h_1 \theta_t + h_2 \theta_t^2 
$$

To make them match, set

$$
\begin{aligned}
h_0 & = u_0 \cr
h_1 & = -\alpha u_1 \cr
h_2 & = - \frac{u_2 \alpha^2}{2}
\end{aligned}
$$

The inflation rate $\theta_t$ is determined by

$$
\theta_t = (1-\lambda) \sum_{j=0}^\infty \lambda^j \mu_{t+j}
$$

where 

$$
\lambda = \frac{\alpha}{1+\alpha}
$$

## Approximations

We will approximate $\vec \mu$ by a truncated  vector
in which

$$
\mu_t = \bar \mu \quad \forall t \geq T
$$

and we'll approximate $\vec \theta$ with a truncated vector in which

$$
\theta_t = \bar \theta \quad \forall t \geq T
$$

### Comment

We anticipate that under a Ramsey plan $\{\theta_t\}$ and $\{\mu_t\}$ will each converge to stationary values. That guess underlies the form of our approximating sequences.


## Formula for $\theta_t$

We want to write a function that takes 

$$
\tilde \mu = \begin{bmatrix}\mu_0 & \mu_1 & \cdots & \mu_{T-1} & \bar \mu
\end{bmatrix}
$$

as an input and give as an output


$$
\tilde \theta = \begin{bmatrix}\theta_0 & \theta_1 & \cdots & \theta_{T-1} & \bar \theta
\end{bmatrix}
$$

where $\theta_t$ satisfies

$$
\theta_t = (1-\lambda) \sum_{j=0}^{T-1-t} \lambda^j \mu_{t+j} + \lambda^{T-t} \bar \mu 
$$ 

for $t=0, 1, \ldots, T-1$ and $\bar \theta = \bar \mu$.

## Approximation for $V$

After we have computed the vectors $\tilde \mu$ and $\tilde \theta$
as described above, we want to write a function that computes

$$
\tilde V = \sum_{t=0}^\infty \beta^t (
h_0 + h_1 \tilde\theta_t + h_2 \tilde\theta_t^2 -
\frac{c}{2} \mu_t^2 )
$$

or more precisely 

$$ 
\tilde V = \sum_{t=0}^{T-1} \beta^t (h_0 + h_1 \tilde\theta_t + h_2 \tilde\theta_t^2 -
\frac{c}{2} \mu_t^2 ) + \frac{\beta^T}{1-\beta} (h_0 + h_1 \bar \mu + h_2 \bar \mu^2 - \frac{c}{2} \bar \mu^2 )
$$

where $\tilde \theta_t, \ t = 0, 1, \ldots , T-1$ satisfies formula (1).
## Code to write

We want to write code to maximize the criterion function by choice of the truncated vector  $\tilde \mu$.

We hope that answers will agree with those found in {doc}`calvo`.

## Implementation

We will implement the above in Python using JAX and Optax libraries.

We use the following imports in this lecture

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade quantecon
!pip install --upgrade optax
!pip install --upgrade statsmodels
```

```{code-cell} ipython3
from quantecon import LQ
import numpy as np
import jax.numpy as jnp
from jax import jit, grad
import optax
import statsmodels.api as sm
import matplotlib.pyplot as plt
```

First we copy the class `ChangLQ` to solve the LQ Chang model in {doc}`calvo`.

We hide this cell because we would like readers to refer to the previous lecture for the details of the class.

```{code-cell} ipython3
:tags: [hide-output]

class ChangLQ:
    """
    Class to solve LQ Chang model
    """
    def __init__(self, β, c, α=1, u0=1, u1=0.5, u2=3, T=1000, θ_n=200):
        # Record parameters
        self.α, self.u0, self.u1, self.u2 = α, u0, u1, u2
        self.β, self.c, self.T, self.θ_n = β, c, T, θ_n

        self.setup_LQ_matrices()
        self.solve_LQ_problem()
        self.compute_policy_functions()
        self.simulate_ramsey_plan()
        self.compute_θ_range()
        self.compute_value_and_policy()

    def setup_LQ_matrices(self):
        # LQ Matrices
        self.R = -np.array([[self.u0, -self.u1 * self.α / 2],
                            [-self.u1 * self.α / 2, 
                             -self.u2 * self.α**2 / 2]])
        self.Q = -np.array([[-self.c / 2]])
        self.A = np.array([[1, 0], [0, (1 + self.α) / self.α]])
        self.B = np.array([[0], [-1 / self.α]])

    def solve_LQ_problem(self):
        # Solve LQ Problem (Subproblem 1)
        lq = LQ(self.Q, self.R, self.A, self.B, beta=self.β)
        self.P, self.F, self.d = lq.stationary_values()

        # Compute g0, g1, and g2 (41.16)
        self.g0, self.g1, self.g2 = [-self.P[0, 0], 
                                     -2 * self.P[1, 0], -self.P[1, 1]]
        
        # Compute b0 and b1 (41.17)
        [[self.b0, self.b1]] = self.F

        # Compute d0 and d1 (41.18)
        self.cl_mat = (self.A - self.B @ self.F)  # Closed loop matrix
        [[self.d0, self.d1]] = self.cl_mat[1:]

        # Solve Subproblem 2
        self.θ_R = -self.P[0, 1] / self.P[1, 1]
        
        # Find the bliss level of θ
        self.θ_B = -self.u1 / (self.u2 * self.α)

    def compute_policy_functions(self):
        # Solve the Markov Perfect Equilibrium
        self.μ_MPE = -self.u1 / ((1 + self.α) / self.α * self.c 
                                 + self.α / (1 + self.α)
                                 * self.u2 + self.α**2 
                                 / (1 + self.α) * self.u2)
        self.θ_MPE = self.μ_MPE
        self.μ_CR = -self.α * self.u1 / (self.u2 * self.α**2 + self.c)
        self.θ_CR = self.μ_CR

        # Calculate value under MPE and CR economy
        self.J_θ = lambda θ_array: - np.array([1, θ_array]) \
                                   @ self.P @ np.array([1, θ_array]).T
        self.V_θ = lambda θ: (self.u0 + self.u1 * (-self.α * θ)
                              - self.u2 / 2 * (-self.α * θ)**2 
                              - self.c / 2 * θ**2) / (1 - self.β)
        
        self.J_MPE = self.V_θ(self.μ_MPE)
        self.J_CR = self.V_θ(self.μ_CR)

    def simulate_ramsey_plan(self):
        # Simulate Ramsey plan for large number of periods
        θ_series = np.vstack((np.ones((1, self.T)), np.zeros((1, self.T))))
        μ_series = np.zeros(self.T)
        J_series = np.zeros(self.T)
        θ_series[1, 0] = self.θ_R
        [μ_series[0]] = -self.F.dot(θ_series[:, 0])
        J_series[0] = self.J_θ(θ_series[1, 0])

        for i in range(1, self.T):
            θ_series[:, i] = self.cl_mat @ θ_series[:, i-1]
            [μ_series[i]] = -self.F @ θ_series[:, i]
            J_series[i] = self.J_θ(θ_series[1, i])

        self.J_series = J_series
        self.μ_series = μ_series
        self.θ_series = θ_series

    def compute_θ_range(self):
        # Find the range of θ in Ramsey plan
        θ_LB = min(min(self.θ_series[1, :]), self.θ_B)
        θ_UB = max(max(self.θ_series[1, :]), self.θ_MPE)
        θ_range = θ_UB - θ_LB
        self.θ_LB = θ_LB - 0.05 * θ_range
        self.θ_UB = θ_UB + 0.05 * θ_range
        self.θ_range = θ_range

    def compute_value_and_policy(self):        
        # Create the θ_space
        self.θ_space = np.linspace(self.θ_LB, self.θ_UB, 200)
        
        # Find value function and policy functions over range of θ
        self.J_space = np.array([self.J_θ(θ) for θ in self.θ_space])
        self.μ_space = -self.F @ np.vstack((np.ones(200), self.θ_space))
        x_prime = self.cl_mat @ np.vstack((np.ones(200), self.θ_space))
        self.θ_prime = x_prime[1, :]
        self.CR_space = np.array([self.V_θ(θ) for θ in self.θ_space])
        
        self.μ_space = self.μ_space[0, :]
        
        # Calculate J_range, J_LB, and J_UB
        self.J_range = np.ptp(self.J_space)
        self.J_LB = np.min(self.J_space) - 0.05 * self.J_range
        self.J_UB = np.max(self.J_space) + 0.05 * self.J_range
```

Now we compute the value of $V$ under this setup, and compare it against those obtained in {ref}`compute_lq`.

```{code-cell} ipython3
# Assume β=0.85, c=2, T=40.
T = 40
clq = ChangLQ(β=0.85, c=2, T=T)
```

```{code-cell} ipython3
@jit
def compute_θ(μ, α=1):
    λ = α / (1 + α)
    T = len(μ) - 1
    μbar = μ[-1]
    
    # Create an array of powers for λ
    λ_powers = λ ** jnp.arange(T + 1)
    
    # Compute the weighted sums for all t
    weighted_sums = jnp.array(
        [jnp.sum(λ_powers[:T-t] * μ[t:T]) for t in range(T)])
    
    # Compute θ values except for the last element
    θ = (1 - λ) * weighted_sums + λ**(T - jnp.arange(T)) * μbar
    
    # Set the last element
    θ = jnp.append(θ, μbar)
    
    return θ
    
@jit
def compute_V(μ, β, c, α=1, u0=1, u1=0.5, u2=3):
    θ = compute_θ(μ, α)
    
    h0 = u0
    h1 = -u1 * α
    h2 = -0.5 * u2 * α**2
    
    T = len(μ) - 1
    t = np.arange(T)
    
    # Compute sum except for the last element
    V_sum = np.sum(β**t * (h0 + h1 * θ[:T] + h2 * θ[:T]**2 - 0.5 * c * μ[:T]**2))
    
    # Compute the final term
    V_final = (β**T / (1 - β)) * (h0 + h1 * μ[-1] + h2 * μ[-1]**2 - 0.5 * c * μ[-1]**2)
    
    V = V_sum + V_final
    
    return V
```

```{code-cell} ipython3
V_val = compute_V(clq.μ_series, β=0.85, c=2)

# Check the result with the ChangLQ class in previous lecture
print(f'deviation = {np.linalg.norm(V_val - clq.J_series[0])}') # good!
```

Now we want to maximize the function $V$ by choice of $\mu$.

We will use the [`optax.adam`](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adam) from the `optax` library.

```{code-cell} ipython3
def adam_optimizer(grad_func, init_params, 
                   lr=0.1, 
                   max_iter=10_000, 
                   error_tol=1e-7):

    # Set initial parameters and optimizer
    params = init_params
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    # Update parameters and gradients
    @jit
    def update(params, opt_state):
        grads = grad_func(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, grads

    # Gradient descent loop
    for i in range(max_iter):
        params, opt_state, grads = update(params, opt_state)
        
        if jnp.linalg.norm(grads) < error_tol:
            print(f"Converged after {i} iterations.")
            break

        if i % 100 == 0: 
            print(f"Iteration {i}, grad norm: {jnp.linalg.norm(grads)}")
    
    return params
```

Here we use automatic differentiation functionality in JAX with `grad`.

```{code-cell} ipython3
:tags: [scroll-output]

# Initial guess for μ
μ_init = jnp.zeros(T)

# Maximization instead of minimization
grad_V = jit(grad(
    lambda μ: -compute_V(μ, β=0.85, c=2)))
```

```{code-cell} ipython3
%%time

# Optimize μ
optimized_μ = adam_optimizer(grad_V, μ_init)

print(f"optimized μ = \n{optimized_μ}")
```

```{code-cell} ipython3
print(f"original μ = \n{clq.μ_series}")
```

```{code-cell} ipython3
print(f'deviation = {np.linalg.norm(optimized_μ - clq.μ_series)}')
```

```{code-cell} ipython3
compute_V(optimized_μ, β=0.85, c=2)
```

```{code-cell} ipython3
compute_V(clq.μ_series, β=0.85, c=2)
```

## Regressing $\vec \theta_t$ and $\vec \mu_t$

```{code-cell} ipython3
# Compute θ using optimized_μ
θs = np.array(compute_θ(optimized_μ))
μs = np.array(optimized_μ)

# First regression: μ_t on a constant and θ_t
X1_θ = sm.add_constant(θs)
model1 = sm.OLS(μs, X1_θ)
results1 = model1.fit()

# Print regression summary
print("Regression of μ_t on a constant and θ_t:")
print(results1.summary(slim=True))
```

```{code-cell} ipython3
plt.scatter(θs, μs)
plt.plot(θs, results1.predict(X1_θ), 'C1', label='$\hat \mu_t$', linestyle='--')
plt.xlabel(r'$\theta_t$')
plt.ylabel(r'$\mu_t$')
plt.legend()
plt.show()
```

```{code-cell} ipython3
# Second regression: θ_{t+1} on a constant and θ_t
θ_t = np.array(θs[:-1])  # θ_t
θ_t1 = np.array(θs[1:])  # θ_{t+1}
X2_θ = sm.add_constant(θ_t)  # Add a constant term for the intercept
model2 = sm.OLS(θ_t1, X2_θ)
results2 = model2.fit()

# Print regression summary
print("\nRegression of θ_{t+1} on a constant and θ_t:")
print(results2.summary(slim=True))
```

```{code-cell} ipython3
plt.scatter(θ_t, θ_t1)
plt.plot(θ_t, results2.predict(X2_θ), color='C1', label='$\hat θ_t$', linestyle='--')
plt.xlabel(r'$\theta_t$')
plt.ylabel(r'$\theta_{t+1}$')
plt.legend()

plt.tight_layout()
plt.show()
```

## Regress $V$ and $\vec\theta_t$

+++

First, we modified the function `compute_V_t` to return a sequence of $\vec v_t$.

```{code-cell} ipython3
def compute_V_t(μ, β, c, α=1, u0=1, u1=0.5, u2=3):
    θ = compute_θ(μ, α)
    
    h0 = u0
    h1 = -u1 * α
    h2 = -0.5 * u2 * α**2
    
    T = len(μ)
    V_t = jnp.zeros(T)
    
    for t in range(T - 1):
        V_t = V_t.at[t].set(β**t * (h0 + h1 * θ[t] + h2 * θ[t]**2 - 0.5 * c * μ[t]**2))
    
    # Terminal condition
    V_t = V_t.at[T-1].set((β**(T-1) / (1 - β)) * (h0 + h1 * μ[-1] + h2 * μ[-1]**2 - 0.5 * c * μ[-1]**2))
    
    return V_t
```

```{code-cell} ipython3
# Compute v_t
v_ts = np.array(compute_V_t(optimized_μ, β=0.85, c=2))

# Initialize arrays for discounted sum of θ_t, θ_t^2, μ_t^2
βθ_t = np.zeros(T)
βθ_t2 = np.zeros(T)
βμ_t2 = np.zeros(T)

# Compute discounted sum of θ_t, θ_t^2, μ_t^2
for ts in range(T):
    βθ_t[ts] = sum(clq.β**t * θs[t] 
                   for t in range(ts + 1))
    βθ_t2[ts] = sum(clq.β**t * θs[t]**2 
                    for t in range(ts + 1))
    βμ_t2[ts] = sum(clq.β**t * μs[t]**2 
                    for t in range(ts + 1))

X = np.column_stack((βθ_t, βθ_t2, βμ_t2))
X_vt = sm.add_constant(X)

# Fit the model
model3 = sm.OLS(v_ts, X_vt).fit()
```

```{code-cell} ipython3
plt.figure()
plt.scatter(θs, v_ts)
plt.plot(θs, model3.predict(X_vt), color='C1', label='$\hat v_t$', linestyle='--')
plt.xlabel('$θ_t$')
plt.ylabel('$v_t$')
plt.legend()
plt.show()
```

## Restriction to $\mu_t = \bar \mu$

+++

In this case, we restrict $\mu_t = \bar \mu \text{ for } \forall t$

```{code-cell} ipython3
# Initial guess for single μ
μ_init = jnp.zeros(1)

# Maximization instead of minimization
grad_V = jit(grad(
    lambda μ: -compute_V(μ, β=0.85, c=2)))

# Optimize μ
optimized_μ_CR = adam_optimizer(grad_V, μ_init)

print(f"optimized μ = \n{optimized_μ_CR}")
```

Compare it to $\mu^{CR}$ in {doc}`calvo`, we again obtained a close estimate.

```{code-cell} ipython3
np.linalg.norm(clq.μ_CR - optimized_μ_CR)
```

```{code-cell} ipython3
compute_V(optimized_μ_CR, β=0.85, c=2)
```

```{code-cell} ipython3
compute_V(jnp.array([clq.μ_CR]), β=0.85, c=2)
```
