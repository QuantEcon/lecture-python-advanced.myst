"""
JAX-based utilities for AMSS model.
Converted from NumPy/Numba to JAX with NamedTuple structures.
"""

import jax.numpy as jnp
from jax import jit, grad
from typing import NamedTuple


class CRRAUtilityParams(NamedTuple):
    """Parameters for CRRA utility function."""
    β: float = 0.9
    σ: float = 2.0 
    γ: float = 2.0


class LogUtilityParams(NamedTuple):
    """Parameters for logarithmic utility function."""
    β: float = 0.9
    ψ: float = 0.69


@jit
def crra_utility(c, l, params: CRRAUtilityParams):
    """
    CRRA utility function.
    
    Parameters
    ----------
    c : float or array
        Consumption
    l : float or array
        Leisure (note: l should not be interpreted as labor)
    params : CRRAUtilityParams
        Utility parameters
        
    Returns
    -------
    float or array
        Utility value
    """
    σ = params.σ
    # Use jnp.where for conditional logic in JAX
    U_c = jnp.where(σ == 1.0, 
                    jnp.log(c),
                    (c**(1 - σ) - 1) / (1 - σ))
    
    U_l = -(1 - l) ** (1 + params.γ) / (1 + params.γ)
    
    return U_c + U_l


@jit 
def log_utility(c, l, params: LogUtilityParams):
    """
    Logarithmic utility function.
    
    Parameters
    ----------
    c : float or array
        Consumption
    l : float or array
        Leisure
    params : LogUtilityParams
        Utility parameters
        
    Returns
    -------
    float or array
        Utility value
    """
    return jnp.log(c) + params.ψ * jnp.log(l)


# Create derivative functions using JAX autodiff
crra_utility_c = jit(grad(crra_utility, argnums=0))
crra_utility_l = jit(grad(crra_utility, argnums=1)) 
crra_utility_cc = jit(grad(crra_utility_c, argnums=0))
crra_utility_ll = jit(grad(crra_utility_l, argnums=1))

log_utility_c = jit(grad(log_utility, argnums=0))
log_utility_l = jit(grad(log_utility, argnums=1))
log_utility_cc = jit(grad(log_utility_c, argnums=0))
log_utility_ll = jit(grad(log_utility_l, argnums=1))


class AMSSModelParams(NamedTuple):
    """Parameters for AMSS model."""
    β: float
    Π: jnp.ndarray  # Transition matrix
    g: jnp.ndarray  # Government spending in each state
    x_grid: tuple   # Grid parameters (min, max, num_points)
    bounds_v: jnp.ndarray  # Bounds for value function optimization


class AMSSParams(NamedTuple):
    """Parameters for AMSS model."""
    β: float                    # Discount factor
    Π: jnp.ndarray             # Markov transition matrix  
    g: jnp.ndarray             # Government spending by state
    x_grid: tuple              # Grid parameters (x_min, x_max, x_num)
    bounds_v: jnp.ndarray      # Bounds for optimization
    utility: 'UtilityFunctions'  # Utility functions


class UtilityFunctions(NamedTuple):
    """Collection of utility functions and their derivatives."""
    U: callable       # Utility function U(c, l, params)
    Uc: callable      # Marginal utility of consumption  
    Ul: callable      # Marginal utility of leisure
    Ucc: callable     # Second derivative wrt consumption
    Ull: callable     # Second derivative wrt leisure
    params: NamedTuple  # Utility parameters


def create_crra_utility_functions(params: CRRAUtilityParams) -> UtilityFunctions:
    """Create CRRA utility functions with parameters."""
    
    @jit
    def U(c, l):
        return crra_utility(c, l, params)
    
    @jit 
    def Uc(c, l):
        return crra_utility_c(c, l, params)
        
    @jit
    def Ul(c, l):
        return crra_utility_l(c, l, params)
        
    @jit
    def Ucc(c, l):
        return crra_utility_cc(c, l, params)
        
    @jit  
    def Ull(c, l):
        return crra_utility_ll(c, l, params)
    
    return UtilityFunctions(U=U, Uc=Uc, Ul=Ul, Ucc=Ucc, Ull=Ull, params=params)


def create_log_utility_functions(params: LogUtilityParams) -> UtilityFunctions:
    """Create logarithmic utility functions with parameters."""
    
    @jit
    def U(c, l):
        return log_utility(c, l, params)
    
    @jit
    def Uc(c, l):
        return log_utility_c(c, l, params)
        
    @jit
    def Ul(c, l):
        return log_utility_l(c, l, params)
        
    @jit
    def Ucc(c, l):
        return log_utility_cc(c, l, params)
        
    @jit
    def Ull(c, l):
        return log_utility_ll(c, l, params)
    
    return UtilityFunctions(U=U, Uc=Uc, Ul=Ul, Ucc=Ucc, Ull=Ull, params=params)