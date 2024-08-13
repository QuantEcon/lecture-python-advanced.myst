---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Another Look at Machine Learning a Ramsey Plan

I'd like  you  to use your gradient code to compute a Ramsey plan for a versionof Calvo's original model.

That model was not LQ.

But your code can be tweaked to compute the Ramsey plan I suspect.

I sketch the main idea here.

## Calvo's setup

The Ramsey planner's one-period social utility function
is 

$$
u(c) + v(m) \equiv u(f(x)) + v(m)
$$

where  in our notation Calvo or Chang's objects  become

$$ 
\begin{align*}
x & = \mu \cr 
m & = -\alpha \theta \cr 
u(c) & = u(f(x)) = u(f(\mu)) \cr 
v(m) & = v (-\alpha \theta)
\end{align*}
$$


In the quantecon lecture about the Chang-Calvo model, we deployed the following functional forms:

$$
u(c) = \log(c)
$$

$$
v(m) = \frac{1}{500}(m \bar m - 0.5m^2)^{0.5}
$$

$$
f(x) = 180 - (0.4x)^2
$$

where $\bar m$ is a parameter set somewhere in the quantecon code.

So with this parameterization of Calvo and Chang's functions, components of  our one-period criterion  become

$$
u(c_t) = \log (180 - (0.4 \mu_t)^2) 
$$

and

$$
v(m_t - p_t) = \frac{1}{500}((-\alpha \theta_t)  \bar m - 0.5(-\alpha \theta_t)^2)^{0.5}
$$

As in our ``calvo_machine_learning`` lecture, the Ramsey planner maximizes the criterion

$$
\sum_{t=0}^\infty \beta^t [ u(c_t) + v(m_t)] \tag{1}
$$

subject to the constraint 


$$
\theta_t = \frac{1}{1+\alpha} \sum_{j=0}^\infty \left(\frac{\alpha}{1+\alpha}\right)^j \mu_{t+j}, \quad t \geq 0 \tag{2}
$$


## Proposal to Humphrey

I'd like to take big parts of the code that you used for the ``gradient`` method only -- the first method you created -- and adapt it to compute a Ramsey plan for this non LQ version of the Calvo's model.

  * it is quite close to the version in the main part of Calvo's 1978 classic paper
  
I'd like you to use exactly the same assumptions about the $\{\mu_t\}_{t=0}^\infty$ process that is in the code, which means that you'll only have to compute a truncated sequence $\{\mu_t\}_{t=0}^T$ parameterized by   $T$ and $\bar \mu$ as in the code.

And it would be great if you could also compute the Ramsey plan restricted to a constant $\vec \mu$ sequence so that we could get our hands on that plan to plot and compare with the ordinary Ramsey plan. 


After you've computed the Ramsey plan, I'd like to ask you to plot the pretty graphs of $\vec \theta, \vec \mu$ and also $\vec v$ where (recycling notation) here $\vec v$ is a sequence of continuation values.

Qualitatively, these graphs should look a lot like the ones plotted in the ``calvo_machine_learning`` lecture.  

Later, after those plots are done, I'd like to describe some **nonlinear** regressions to run that we'll use to try to discover a recursive represenation of a Ramsey plan.  

  * this will be fun -- we might want to use some neural nets to approximate these functions -- then we'll **really** be doing machine learning. 
  

## Excuses and Apologies

I have recycled some notation -- e.g., $v$ and $v_t$.  And Calvo uses $m$ for what we call $m_t - p_t$ and so on. 

Later we can do some notation policing and cleaning up.

But right now, let's just proceed as best we can with notation that makes it easiest to take
the ``calvo_machine_learning`` code and apply it with minimal changes. 

Thanks!

```{code-cell} ipython3
from quantecon import LQ
import numpy as np
import jax.numpy as jnp
from jax import jit, grad
import optax
import statsmodels.api as sm
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
from collections import namedtuple

def create_ChangML(T=40, β=0.3, c=2, α=1, mbar=30.0):
    ChangML = namedtuple('ChangML', ['T', 'β', 'c', 'α', 
                                     'mbar'])

    return ChangML(T=T, β=β, c=c, α=α, mbar=mbar)


model = create_ChangML()
```

```{code-cell} ipython3
@jit
def compute_θ(μ, α):
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

def compute_V(μ, model):
    β, c, α, mbar,= model.β, model.c, model.α, model.mbar
    θ = compute_θ(μ, α)
    
    T = len(μ) - 1
    t = jnp.arange(T)
    
    def u(μ, m, mbar): 
        # print('μ', μ)
        f_μ = 180 - (0.4 * μ)**2
        # print('f_μ', f_μ)
        # print('m, mbar:', m, mbar, (mbar * m - 0.5 * m**2))
        v_m = 1 / 500 * jnp.sqrt(jnp.maximum(1e-16, mbar * m - 0.5 * m**2))
        # print('v_μ', v_m)
        return jnp.log(f_μ) + v_m
    
    # Compute sum except for the last element
    
    # TO TOM: -α*θ is causing trouble in utility calculation (u) above
    # As it makes mbar * m - 0.5 * m**2 < 0 and sqrt of negative
    # values returns NA
    V_sum = jnp.sum(β**t * u(μ[:T], -α*θ[:T], mbar))
    # print('V_sum', V_sum)
    
    # Compute the final term
    V_final = (β**T / (1 - β)) * u(μ[-1], -α*θ[-1], mbar)
    # print('V_final', V_final)
    V = V_sum + V_final
    
    return V

# compute_V(jnp.array([1.0, 1.0, 1.0]), model)
```

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

```{code-cell} ipython3
T = 40 

# Initial guess for μ
μ_init = jnp.ones(T)

# Maximization instead of minimization
grad_V = jit(grad(
    lambda μ: -compute_V(μ, model)))
```

```{code-cell} ipython3
%%time

# Optimize μ
optimized_μ = adam_optimizer(grad_V, μ_init)

print(f"optimized μ = \n{optimized_μ}")
```

```{code-cell} ipython3
compute_V(optimized_μ, model)
```

```{code-cell} ipython3
model = create_ChangML(β=0.8)

grad_V = jit(grad(
    lambda μ: -compute_V(μ, model)))
```

```{code-cell} ipython3
%%time

# Optimize μ
optimized_μ = adam_optimizer(grad_V, μ_init)

print(f"optimized μ = \n{optimized_μ}")
```

```{code-cell} ipython3
compute_V(optimized_μ, model)
```
