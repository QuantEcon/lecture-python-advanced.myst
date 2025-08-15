"""
JAX-based interpolation utilities for AMSS model.
Converted from NumPy/SciPy to JAX.
"""

import jax.numpy as jnp
from jax import jit, vmap
import jax
from typing import NamedTuple


class GridParams(NamedTuple):
    """Parameters for interpolation grid."""
    x_min: float
    x_max: float
    num_points: int
    

def create_uniform_grid(params: GridParams):
    """Create uniform grid for interpolation. Not JIT-compiled due to concrete value requirement."""
    return jnp.linspace(params.x_min, params.x_max, params.num_points)


@jit 
def linear_interpolation_1d(x_grid, y_values, x_new):
    """
    Perform linear interpolation on 1D data.
    
    Parameters
    ----------
    x_grid : array
        Grid points for interpolation
    y_values : array  
        Function values at grid points
    x_new : float or array
        Points to interpolate at
        
    Returns
    -------
    float or array
        Interpolated values
    """
    return jnp.interp(x_new, x_grid, y_values)


@jit
def simulate_markov_chain(π, s_0, T, key):
    """
    Simulate Markov chain using JAX random number generation.
    
    Parameters
    ----------
    π : array
        Transition probability matrix
    s_0 : int
        Initial state
    T : int
        Number of periods to simulate
    key : PRNGKey
        JAX random key
        
    Returns
    -------
    array
        Simulated state history
    """
    from jax import random
    
    def scan_fn(state_key, t):
        s, key = state_key
        key, subkey = random.split(key)
        s_next = random.choice(subkey, jnp.arange(π.shape[1]), p=π[s])
        return (s_next, key), s_next
    
    keys = random.split(key, T)
    _, s_hist = jax.lax.scan(scan_fn, (s_0, keys[0]), jnp.arange(1, T))
    
    # Prepend initial state
    return jnp.concatenate([jnp.array([s_0]), s_hist])


# Convert UCGrid functionality to JAX
def create_ucgrid(x_min, x_max, x_num):
    """Create uniform grid compatible with original UCGrid interface."""
    return (x_min, x_max, x_num)


def nodes_from_grid(grid_params):
    """Extract grid nodes from grid parameters. Not JIT-compiled due to concrete values."""
    x_min, x_max, x_num = grid_params
    return jnp.linspace(x_min, x_max, x_num)


@jit  
def eval_linear_jax(grid_params, coeffs, x):
    """
    JAX version of eval_linear function.
    
    Parameters
    ----------
    grid_params : tuple
        Grid parameters (x_min, x_max, x_num)
    coeffs : array
        Coefficients for interpolation 
    x : float or array
        Points to evaluate at
        
    Returns
    -------
    float or array
        Interpolated values
    """
    x_min, x_max, x_num = grid_params
    x_grid = jnp.linspace(x_min, x_max, x_num)
    return jnp.interp(x, x_grid, coeffs)


# Vectorized version for multiple interpolations
eval_linear_vectorized = jit(vmap(eval_linear_jax, in_axes=(None, 0, None)))