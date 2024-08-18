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


I'd like  you  to use your gradient code to compute a Ramsey plan for a version of Calvo's original model.

Calvo's 78 Econometrica  model was not LQ.

But I suspect  your code can be adapted to compute the Ramsey plan in Calvo's model.

I sketch the main idea here.

We will use a variation on the  non-linear-quadratic model  presented in  intro series quantecon lecture **Inflation Rate Laffer Curves** 

   <https://intro.quantecon.org/money_inflation_nonlinear.html>

(Some of the timing here is different than in that lecture)

Let  

* $m_t$ be the log of the money supply at the beginning of time $t$
* $p_t$ be the log of the price level at time $t$
  
The demand function for money is 

$$
m_{t} - p_t = -\alpha (p_{t+1} - p_t) \tag{1}
$$ 

where $\alpha \geq 0$.  

The law of motion for the money supply is

$$ 
\exp(m_{t+1}) - \exp(m_t) = -x_t \exp(p_t) \tag{2}
$$ 

where 

* $x_t$ is tax revenues used to withdraw money 

Given a sequence $\{x_t\}_{t=0}^\infty$ and an initial log money supply $m_0$, we want a function that solves
(1) and (2) for $\{m_{t+1}, p_t\}_{t=0}^\infty$.

This function will be an input into solving for a Ramsey plan using a version of the gradient algorithm. 

I'll move on to describe that function implicitly as a system of constraints. 


## More details

We'll impose a truncated $\vec x$ series in which

$$
x_t = \bar x \quad \forall t \geq T
$$

where $T$ is a positive integer greater than $1$.

We'll set $T$ in our  code. 


Our code will be cast in terms of three vectors

$$
\begin{align*} 
\vec x & = \{x_t\}_{t=0}^T \cr
\vec \mu & =  \{\mu_t\}_{t=0}^T \cr
\vec \theta & =  \{\theta_t\}_{t=0}^T 
\end{align*}
$$

We'll assume that 

$$
\begin{align*}
\mu_t & = \bar \mu  \quad \forall t \geq T \cr 
\theta_t & =  \bar \mu \quad \forall t \geq T
\end{align*}
$$


It follows that 

$$ 
m_t = p_t - \alpha \bar \mu \quad \forall t \geq T
$$

After a few lines of algebra, we can deduce from equation (2) that

$$
 \exp(\bar \mu(1-\alpha)) - \exp(-\alpha \bar \mu) = - \bar x  \tag {3}
$$

From a formula that appears in the ``calvo_machine_learning`` lecture, we know that

$$
\theta_t = (1-\lambda) \sum_{j=0}^{T-1-j} \mu_{t+j} + \lambda ^{T-t} \bar \mu ,\quad t = 0, 1, \ldots, T-1 \tag{4}
$$ 

It follows from (1) and (4) that 

$$
m_t - p_t = -\alpha \Bigl[ (1-\lambda) \sum_{j=0}^{T-1-j} ( m_{t+j+1} - m_{t+j} ) + \lambda ^{T-t} \bar \mu \Bigr]
\tag{5} $$

A possible  algorithm is 

 * given $\bar x$, solve (3) for $\bar \mu$
 * given $\vec x, m_0$, solve the system of   equations (2) and  (5) for $\vec \mu, \vec \theta$ 




## Calvo's Objective Function

The Ramsey planner's one-period social utility function
in our notation is 

$$
u(c) + j(m-p) \equiv u(f(x)) + j(m - p)
$$

where  

$$ 
\begin{align*}
m - p & = -\alpha \theta \cr 
u(c) & = u(f(x)) \cr 
j(m-p) & = j_0 + j_1 (m-p) - \frac{j_2}{2} (m-p)^2
\end{align*}
$$


where $f: \mathbb{R} \rightarrow \mathbb{R}$ satisfies
$f(x) >0$, $f(x)$ is twice continuously differentiable, f(0) = 0, f''(x) < 0 , f(x) = f(-x) for all $x$ in  $\mathbb{R}$.

We can assume 

 * some smooth single peaked $f$ function, e.g., a quadratic one 
 * the $j$ function can be just like our $u$ quadratic function in our machine learning lecture, or else some other monotone smooth function
 * 

As in our ``calvo_machine_learning`` lecture, given $m_0$, the Ramsey planner maximizes chooses $\vec x$ to maximize the criterion

$$
\sum_{t=0}^\infty \beta^t [ u(c_t) + j(m_t-p_t)] \tag{6}
$$

subject to  constraints (1), (2), and (5).  

One way to do this would be to maximize (6) with penalties
on deviations of $\vec x, \vec \mu, \vec \theta$ from the constraints (1), (2), (5). 

This is the kind of things done in the ML literature that Zejin is relyng on.



## Proposal to Humphrey

I'd like to take big parts of the code that you used for the ``gradient`` method only -- the first method you created -- and adapt it to compute a Ramsey plan for this non LQ version of the Calvo's model.

  * it is quite close to the version in the main part of Calvo's 1978 classic paper
  
And it would be great if you could also compute the Ramsey plan restricted to a constant $\vec \mu$ sequence so that we could get our hands on that plan to plot and compare with the ordinary Ramsey plan. 


After you've computed the Ramsey plan, I'd like to ask you to plot the pretty graphs of $\vec \theta, \vec \mu$ and also $\vec v$ where (recycling notation) here $\vec v$ is a sequence of continuation values.

Qualitatively, these graphs should look a lot like the ones plotted in the ``calvo_machine_learning`` lecture.  

Later, after those plots are done, I'd like to describe some **nonlinear** regressions to run that we'll use to try to discover a recursive represenation of a Ramsey plan.  

  * this will be fun -- we might want to use some neural nets to approximate these functions -- then we'll **really** be doing machine learning. 
  

## Excuses and Apologies
For  now, let's just proceed as best we can with notation that makes it easiest to take
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
