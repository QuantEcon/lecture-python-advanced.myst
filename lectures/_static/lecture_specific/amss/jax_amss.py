"""
JAX-based AMSS model implementation.
Converted from NumPy/Numba classes to JAX pure functions and NamedTuple structures.
"""

import jax.numpy as jnp
from jax import jit, grad, vmap
import jax
from scipy.optimize import minimize  # Use scipy for now
from typing import NamedTuple, Callable
try:
    from .jax_utilities import UtilityFunctions
    from .jax_interpolation import nodes_from_grid, eval_linear_jax
except ImportError:
    from jax_utilities import UtilityFunctions  
    from jax_interpolation import nodes_from_grid, eval_linear_jax


class AMSSState(NamedTuple):
    """State variables for AMSS model."""
    s: int          # Current Markov state
    x: float        # Continuation value state variable 


class AMSSParams(NamedTuple):
    """Parameters for AMSS model."""
    β: float                    # Discount factor
    Π: jnp.ndarray             # Markov transition matrix  
    g: jnp.ndarray             # Government spending by state
    x_grid: tuple              # Grid parameters (x_min, x_max, x_num)
    bounds_v: jnp.ndarray      # Bounds for optimization
    utility: UtilityFunctions  # Utility functions


class AMSSPolicies(NamedTuple):
    """Policy functions for AMSS model."""
    V: jnp.ndarray             # Value function
    σ_v_star: jnp.ndarray      # Policy function for time t >= 1
    W: jnp.ndarray             # Value function for time 0
    σ_w_star: jnp.ndarray      # Policy function for time 0


@jit
def compute_consumption_leisure(l, g):
    """Compute consumption given leisure and government spending."""
    return (1 - l) - g


@jit  
def objective_V(σ, state, V, params: AMSSParams):
    """
    Objective function for time t >= 1 value function iteration.
    
    Parameters
    ----------
    σ : array
        Policy variables [l_1, ..., l_S, T_1, ..., T_S]
    state : tuple
        Current state (s_, x_)
    V : array
        Current value function
    params : AMSSParams
        Model parameters
        
    Returns
    -------
    float
        Negative of expected value (for minimization)
    """
    s_, x_ = state
    S = len(params.Π)
    
    l = σ[:S]
    T = σ[S:]
    
    c = compute_consumption_leisure(l, params.g)
    u_c = vmap(params.utility.Uc)(c, l)
    Eu_c = params.Π[s_] @ u_c
    
    x = (u_c * x_ / (params.β * Eu_c) - 
         u_c * (c - T) + 
         vmap(params.utility.Ul)(c, l) * (1 - l))
    
    # Interpolate next period value function
    x_nodes = nodes_from_grid(params.x_grid)
    V_next = jnp.array([eval_linear_jax(params.x_grid, V[s], jnp.array([x[s]]))[0] 
                        for s in range(S)])
    
    expected_value = params.Π[s_] @ (vmap(params.utility.U)(c, l) + params.β * V_next)
    
    return -expected_value  # Negative for minimization


@jit
def objective_W(σ, state, V, params: AMSSParams):
    """
    Objective function for time 0 problem.
    
    Parameters
    ----------
    σ : array
        Policy variables [l, T]
    state : tuple  
        Current state (s_, b_0)
    V : array
        Value function
    params : AMSSParams
        Model parameters
        
    Returns
    -------
    float
        Negative of value (for minimization)
    """
    s_, b_0 = state
    l, T = σ
    
    c = compute_consumption_leisure(l, params.g[s_])
    x = (-params.utility.Uc(c, l) * (c - T - b_0) + 
         params.utility.Ul(c, l) * (1 - l))
    
    V_next = eval_linear_jax(params.x_grid, V[s_], jnp.array([x]))[0]
    value = params.utility.U(c, l) + params.β * V_next
    
    return -value  # Negative for minimization


def solve_bellman_iteration(V, σ_v_star, params: AMSSParams, 
                           tol=1e-7, max_iter=1000, print_freq=10):
    """
    Solve the Bellman equation using value function iteration.
    
    Parameters
    ----------
    V : array
        Initial value function guess
    σ_v_star : array
        Initial policy function guess
    params : AMSSParams
        Model parameters
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    print_freq : int
        Print frequency
        
    Returns
    -------
    tuple
        Updated (V, σ_v_star)
    """
    S = len(params.Π)
    x_nodes = nodes_from_grid(params.x_grid)
    n_x = len(x_nodes)
    
    V_new = jnp.zeros_like(V)
    
    for iteration in range(max_iter):
        V_updated = jnp.zeros_like(V)
        σ_updated = jnp.zeros_like(σ_v_star)
        
        # Loop over states and grid points
        for s_ in range(S):
            for x_i in range(n_x):
                state = (s_, x_nodes[x_i])
                x0 = σ_v_star[s_, x_i]
                
                # Optimize using JAX
                bounds = [(params.bounds_v[i, 0], params.bounds_v[i, 1]) 
                         for i in range(len(params.bounds_v))]
                
                # Simple optimization using scipy-like interface
                result = minimize(
                    lambda σ: objective_V(σ, state, V, params),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds
                )
                
                if result.success:
                    V_updated = V_updated.at[s_, x_i].set(-result.fun)
                    σ_updated = σ_updated.at[s_, x_i].set(result.x)
                else:
                    print(f"Optimization failed at state {s_}, grid point {x_i}")
                    V_updated = V_updated.at[s_, x_i].set(V[s_, x_i])
                    σ_updated = σ_updated.at[s_, x_i].set(σ_v_star[s_, x_i])
        
        # Check convergence
        error = jnp.max(jnp.abs(V_updated - V))
        
        if error < tol:
            print(f'Successfully completed VFI after {iteration + 1} iterations')
            return V_updated, σ_updated
            
        if (iteration + 1) % print_freq == 0:
            print(f'Error at iteration {iteration + 1}: {error}')
            
        V = V_updated
        σ_v_star = σ_updated
    
    print(f'VFI did not converge after {max_iter} iterations')
    return V, σ_v_star


def solve_time_zero_problem(b_0, V, params: AMSSParams):
    """
    Solve the time 0 problem.
    
    Parameters
    ----------
    b_0 : float
        Initial debt
    V : array
        Value function from time 1 problem
    params : AMSSParams
        Model parameters
        
    Returns
    -------
    tuple
        (W, σ_w_star) where W is time 0 values and σ_w_star is time 0 policies
    """
    S = len(params.Π)
    W = jnp.zeros(S)
    σ_w_star = jnp.zeros((S, 2))
    
    bounds_w = [(-9.0, 1.0), (0.0, 10.0)]
    
    for s_ in range(S):
        state = (s_, b_0)
        x0 = jnp.array([-0.05, 0.5])  # Initial guess
        
        result = minimize(
            lambda σ: objective_W(σ, state, V, params),
            x0,
            method='L-BFGS-B', 
            bounds=bounds_w
        )
        
        W = W.at[s_].set(-result.fun)
        σ_w_star = σ_w_star.at[s_].set(result.x)
    
    print('Successfully solved the time 0 problem.')
    return W, σ_w_star


@jit
def simulate_amss(s_hist, b_0, policies: AMSSPolicies, params: AMSSParams):
    """
    Simulate AMSS model given state history and initial debt.
    
    Parameters
    ----------
    s_hist : array
        History of Markov states
    b_0 : float
        Initial debt level
    policies : AMSSPolicies
        Solved policy functions
    params : AMSSParams
        Model parameters
        
    Returns
    -------
    dict
        Simulation results with arrays for c, n, b, τ, g
    """
    T = len(s_hist)
    S = len(params.Π)
    x_nodes = nodes_from_grid(params.x_grid)
    
    # Pre-allocate arrays
    n_hist = jnp.zeros(T)
    x_hist = jnp.zeros(T)
    c_hist = jnp.zeros(T)
    τ_hist = jnp.zeros(T)
    b_hist = jnp.zeros(T)
    g_hist = jnp.zeros(T)
    
    # Time 0
    s_0 = s_hist[0]
    l_0, T_0 = policies.σ_w_star[s_0]
    c_0 = compute_consumption_leisure(l_0, params.g[s_0])
    x_0 = (-params.utility.Uc(c_0, l_0) * (c_0 - T_0 - b_0) + 
           params.utility.Ul(c_0, l_0) * (1 - l_0))
    
    n_hist = n_hist.at[0].set(1 - l_0)
    x_hist = x_hist.at[0].set(x_0)
    c_hist = c_hist.at[0].set(c_0)
    τ_hist = τ_hist.at[0].set(1 - params.utility.Ul(c_0, l_0) / params.utility.Uc(c_0, l_0))
    b_hist = b_hist.at[0].set(b_0)
    g_hist = g_hist.at[0].set(params.g[s_0])
    
    # Time t > 0
    for t in range(T - 1):
        x_ = x_hist[t]
        s_ = s_hist[t]
        
        # Interpolate policies for all states
        l = jnp.array([eval_linear_jax(params.x_grid, policies.σ_v_star[s_, :, s], 
                                      jnp.array([x_]))[0] for s in range(S)])
        T_vals = jnp.array([eval_linear_jax(params.x_grid, policies.σ_v_star[s_, :, S+s], 
                                           jnp.array([x_]))[0] for s in range(S)])
        
        c = compute_consumption_leisure(l, params.g)
        u_c = vmap(params.utility.Uc)(c, l)
        Eu_c = params.Π[s_] @ u_c
        
        x = (u_c * x_ / (params.β * Eu_c) - 
             u_c * (c - T_vals) + 
             vmap(params.utility.Ul)(c, l) * (1 - l))
        
        s_next = s_hist[t+1]
        c_next = c[s_next]
        l_next = l[s_next]
        
        x_hist = x_hist.at[t+1].set(x[s_next])
        n_hist = n_hist.at[t+1].set(1 - l_next)
        c_hist = c_hist.at[t+1].set(c_next)
        τ_hist = τ_hist.at[t+1].set(1 - params.utility.Ul(c_next, l_next) / params.utility.Uc(c_next, l_next))
        b_hist = b_hist.at[t+1].set(x_ / (params.β * Eu_c))
        g_hist = g_hist.at[t+1].set(params.g[s_next])
    
    return {
        'c': c_hist,
        'n': n_hist, 
        'b': b_hist,
        'τ': τ_hist,
        'g': g_hist
    }


def solve_amss_model(params: AMSSParams, V_init, σ_v_init, b_0, 
                     W_init=None, σ_w_init=None, **kwargs):
    """
    Solve the complete AMSS model.
    
    Parameters
    ----------
    params : AMSSParams
        Model parameters
    V_init : array
        Initial value function guess
    σ_v_init : array  
        Initial policy function guess
    b_0 : float
        Initial debt level
    W_init : array, optional
        Initial time 0 value function
    σ_w_init : array, optional
        Initial time 0 policy function
    **kwargs
        Additional arguments for solver
        
    Returns
    -------
    AMSSPolicies
        Solved policy functions
    """
    print("===============")
    print("Solve time 1 problem")  
    print("===============")
    V, σ_v_star = solve_bellman_iteration(V_init, σ_v_init, params, **kwargs)
    
    print("===============")
    print("Solve time 0 problem")
    print("===============")
    W, σ_w_star = solve_time_zero_problem(b_0, V, params)
    
    return AMSSPolicies(V=V, σ_v_star=σ_v_star, W=W, σ_w_star=σ_w_star)