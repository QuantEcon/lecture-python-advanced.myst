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

```{code-cell} ipython3
from quantecon import LQ
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, lax
import optax
from jaxopt import Broyden
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections import namedtuple
```

## More details

We'll impose a truncated $\vec x$ series in which

$$
x_t = \bar x \quad \forall t \geq T
$$

where $T$ is a positive integer greater than $1$.

We'll set $T$ in our  code.

```{code-cell} ipython3
def create_ChangML(T=40, β=0.95, c=2, α=0.5, Λ=np.ones(3)):
    ChangML = namedtuple('ChangML', ['T', 'β', 'c', 'α', 'Λ'])

    return ChangML(T=T, β=β, c=c, α=α, Λ=Λ)
```

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

(Humphrey's Note: Should this be

$$
m_t - p_t = -\alpha \Bigl[ (1-\lambda) \sum_{j=0}^{T-1-t} ( m_{t+j+1} - m_{t+j} ) + \lambda ^{T-t} \bar \mu \Bigr]
\tag{5} $$
)

A possible  algorithm is 

 * given $\bar x$, solve (3) for $\bar \mu$
 * given $\vec x, m_0$, solve the system of   equations (2) and  (5) for $\vec \mu, \vec \theta$

```{code-cell} ipython3
def newton(f, x_0, tol=1e-5, max_iter=100):
    f_prime = grad(f)

    def q(x):
        return x - f(x) / f_prime(x)

    def body_fn(i, carry):
        x, error, converged = carry
        x_next = q(x)
        error = jnp.abs(x - x_next)
        converged = jnp.where(error < tol, True, converged)
        return x_next, error, converged

    init_carry = (x_0, tol + 1, False)
    final_carry = lax.fori_loop(0, max_iter, body_fn, init_carry)
    
    x, _, converged = final_carry
    return x, converged

def solve_mp(x, m0, model):
    α, T = model.α, model.T
    λ = α / (1 + α)
    m = jnp.zeros(T + 1).at[0].set(m0)
    p = jnp.zeros(T + 1)
    
    def solve_μ(μ):
        return jnp.exp(μ * (1 - α)) - jnp.exp(-α * μ) + x[-1]
    
    bar_μ, _ = newton(solve_μ, 1.0)
    
    for t in range(T):
        m_next = jnp.log(jnp.exp(m[t]) - x[t] * jnp.exp(p[t]))
        m = m.at[t + 1].set(m_next)

        θ_t = (1 - λ) * jnp.sum(
            (m[t + 1:T + 1] - m[t:T]) * λ ** jnp.arange(T - t)
        ) + λ ** (T - t) * bar_μ
        
        p = p.at[t].set(m[t] + α * θ_t)
        
    p = p.at[-1].set(m[T] + α * bar_μ)
    
    return m, p

model = create_ChangML()

x = np.full(model.T, 1)  # Tax revenue series
m0 = np.log(100)  # Initial log money supply

# Compute m_t and p_t
ms, ps = solve_mp(x, m0, model)

print("m_t:\n", ms)
print("p_t:\n", ps)
```

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

(Humphrey's Note:

So the planner's problem is 

$$
\max_{\vec x} \sum_{t=0}^\infty \beta^t [ u(c_t) + j(m_t-p_t)]
$$

with constraints

$$
m_{t} - p_t = -\alpha (p_{t+1} - p_t)
$$ 

$$ 
\exp(m_{t+1}) - \exp(m_t) = -x_t \exp(p_t)
$$ 

$$
m_t - p_t = -\alpha \Bigl[ (1-\lambda) \sum_{j=0}^{T-1-j} ( m_{t+j+1} - m_{t+j} ) + \lambda ^{T-t} \bar \mu \Bigr]$$)




One way to do this would be to maximize (6) with penalties
on deviations of $\vec x, \vec \mu, \vec \theta$ from the constraints (1), (2), (5). 

This is the kind of things done in the ML literature that Zejin is relyng on.

```{code-cell} ipython3
T=40
ms[:T] - ps[:T] + model.α*(ps[1:] - ps[:T])
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
f = lambda x: 180 - (0.4 * x)**2

def u(x):
    return jnp.log(f(x))

def j(x, j0=2.0, j1=0.8, j2=0.5):
    return j0 + j1 * x - (j2 / 2) * x**2

def objective(x, m0, model):
    β, T = model.β, model.T
    m, p = solve_mp(x, m0, model)
    return -jnp.sum(β**jnp.arange(T+1) * (u(f(x)) + j(m - p)))

grad_objective = grad(lambda x: objective(x, m0=jnp.log(100), model=model))
x_init = jnp.ones(model.T+1)# Initial guess for x
optimized_x = adam_optimizer(grad_objective, x_init)

# Compute the optimized m_t and p_t
optimized_m, optimized_p = solve_mp(optimized_x, m0=jnp.log(100), model=model)

print("Optimized x_t:", optimized_x)
print("Optimized m_t:", optimized_m)
print("Optimized p_t:", optimized_p)
```

```{code-cell} ipython3
xs = np.linespace(0, 100, 1)
```

$$
P(\vec x, \vec m, \vec p) = \lambda_1 \sum_{t=0}^\infty \beta^t \left( m_{t} - p_t + \alpha (p_{t+1} - p_t) \right)^2 
+ \lambda_2 \sum_{t=0}^\infty \beta^t \left( \exp(m_{t+1}) - \exp(m_t) + x_t \exp(p_t) \right)^2 
+ \lambda_3 \sum_{t=0}^\infty \beta^t \left( m_t - p_t + \alpha \left[ (1-\lambda) \sum_{j=0}^{T-1-j} ( m_{t+j+1} - m_{t+j} ) + \lambda ^{T-t} \bar \mu \right] \right)^2
$$

```{code-cell} ipython3
def penalty(m, p, x, model):
    α, β = model.α, model.β
    T, Λ = model.T, model.Λ
    
    λ = α / (1 + α)
    λ1, λ2, λ3 = Λ
    
    # Calculate bar_μ using the final values of p and m
    bar_μ = (p[-1] - m[-1]) / α
    

    penalty_1 = λ1 * jnp.sum(β ** jnp.arange(T) * (m[:-1] - p[:-1] + α * (p[1:] - p[:-1])) ** 2)
    print(m[:-1] - p[:-1] + α * (p[1:] - p[:-1]))
    

    penalty_2 = λ2 * jnp.sum(β ** jnp.arange(T) * 
                             (jnp.exp(m[1:]) - jnp.exp(m[:-1]) 
                              + x[:-1] * jnp.exp(p[:-1])) ** 2)
    print(jnp.exp(m[1:]) - jnp.exp(m[:-1]) 
                              + x[:-1] * jnp.exp(p[:-1]))

    penalty_3 = 0
    for t in range(T):
        sum_terms = jnp.sum(m[t+1:T] - m[t:T-1])
        penalty_3 += (m[t] - p[t] + α * ((1 - λ) * sum_terms + λ ** (T - t) * bar_μ)) ** 2
    print((m[t] - p[t] + α * ((1 - λ) * sum_terms + λ ** (T - t) * bar_μ)))
    penalty_3 = λ3 * jnp.sum(β ** jnp.arange(T) * penalty_3)
    

    total_penalty = penalty_1 + penalty_2 + penalty_3
    
    return total_penalty

penalty(optimized_m, optimized_p, optimized_x, model)
```

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
