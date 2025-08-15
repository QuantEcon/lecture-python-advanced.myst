"""
Simplified JAX-based AMSS model implementation for demonstration.
Shows key JAX concepts: NamedTuple, JIT, grad, vmap.
"""

import jax.numpy as jnp
from jax import jit, grad, vmap
from typing import NamedTuple
try:
    from .jax_utilities import *
    from .jax_interpolation import *
except ImportError:
    from jax_utilities import *
    from jax_interpolation import *


class AMSSSimpleParams(NamedTuple):
    """Simplified AMSS model parameters."""
    β: float
    Π: jnp.ndarray
    g: jnp.ndarray
    utility: UtilityFunctions


class AMSSSimpleState(NamedTuple):
    """State for simplified AMSS model."""
    c: jnp.ndarray  # Consumption by state
    l: jnp.ndarray  # Leisure by state
    τ: jnp.ndarray  # Tax rates by state


@jit
def compute_state_variables(c, l, g):
    """Compute derived state variables."""
    n = 1 - l  # Labor
    y = n      # Output (assuming unit productivity)
    return {'n': n, 'y': y, 'budget_residual': y - g - c}


@jit  
def compute_tax_rates(c, l, crra_params: CRRAUtilityParams):
    """Compute tax rates using marginal utilities."""
    Uc_vals = vmap(lambda ci, li: crra_utility_c(ci, li, crra_params))(c, l)
    Ul_vals = vmap(lambda ci, li: crra_utility_l(ci, li, crra_params))(c, l)
    return 1 - Ul_vals / Uc_vals


@jit
def ramsey_objective(allocations, crra_params: CRRAUtilityParams, g):
    """
    Simplified Ramsey objective function.
    
    Parameters
    ----------
    allocations : array
        Concatenated [c_values, l_values] for all states
    crra_params : CRRAUtilityParams
        Utility parameters
    g : array
        Government spending by state
        
    Returns
    -------
    float
        Negative of social welfare (for minimization)
    """
    S = len(g)
    c = allocations[:S]
    l = allocations[S:]
    
    # Compute utilities for each state
    utilities = vmap(lambda ci, li: crra_utility(ci, li, crra_params))(c, l)
    
    # Expected utility using stationary distribution
    # For simplicity, assume uniform weights
    expected_utility = jnp.mean(utilities)
    
    return -expected_utility


@jit
def budget_constraint(allocations, crra_params: CRRAUtilityParams, g):
    """
    Government budget constraint.
    
    Parameters
    ----------  
    allocations : array
        Concatenated [c_values, l_values] for all states
    crra_params : CRRAUtilityParams
        Utility parameters  
    g : array
        Government spending by state
        
    Returns
    -------
    array
        Budget constraint violations
    """
    S = len(g)
    c = allocations[:S]
    l = allocations[S:]
    
    n = 1 - l  # Labor
    τ = compute_tax_rates(c, l, crra_params)
    
    # Budget constraint: τ * n >= g (simplified, no debt dynamics)
    return τ * n - g


@jit
def feasibility_constraint(allocations, g):
    """
    Resource feasibility constraint.
    
    Parameters
    ----------
    allocations : array
        Concatenated [c_values, l_values] for all states  
    g : array
        Government spending by state
        
    Returns
    -------
    array
        Feasibility constraint violations
    """
    S = len(g)
    c = allocations[:S]
    l = allocations[S:]
    
    n = 1 - l  # Labor  
    y = n      # Output
    
    # Resource constraint: c + g <= y
    return c + g - y


def solve_simple_ramsey_log(log_params: LogUtilityParams, g, initial_guess=None):
    """
    Solve simplified Ramsey problem with log utility.
    
    Parameters
    ----------
    log_params : LogUtilityParams
        Log utility parameters
    g : array
        Government spending by state
    initial_guess : array, optional
        Initial guess for allocations
        
    Returns
    -------
    dict
        Solution with optimal allocations and tax rates
    """
    S = len(g)
    
    if initial_guess is None:
        # Simple initial guess
        c_guess = 0.5 * jnp.ones(S)
        l_guess = 0.5 * jnp.ones(S) 
        initial_guess = jnp.concatenate([c_guess, l_guess])
    
    # Define objectives for log utility
    @jit
    def log_ramsey_objective(allocations):
        c = allocations[:S]
        l = allocations[S:]
        utilities = vmap(lambda ci, li: log_utility(ci, li, log_params))(c, l)
        return -jnp.mean(utilities)
    
    @jit
    def log_budget_constraint(allocations):
        c = allocations[:S]
        l = allocations[S:]
        n = 1 - l
        Uc_vals = vmap(lambda ci, li: log_utility_c(ci, li, log_params))(c, l)
        Ul_vals = vmap(lambda ci, li: log_utility_l(ci, li, log_params))(c, l)
        τ = 1 - Ul_vals / Uc_vals
        return τ * n - g
    
    @jit
    def log_feasibility_constraint(allocations):
        c = allocations[:S]
        l = allocations[S:]
        n = 1 - l
        return c + g - n
    
    @jit
    def penalized_objective_log(allocations, penalty=1000.0):
        obj = log_ramsey_objective(allocations)
        budget_viol = log_budget_constraint(allocations)
        feasibility_viol = log_feasibility_constraint(allocations)
        
        penalty_term = (penalty * jnp.sum(jnp.maximum(0, -budget_viol)**2) +
                       penalty * jnp.sum(jnp.maximum(0, feasibility_viol)**2))
        
        return obj + penalty_term
    
    # Gradient descent
    learning_rate = 0.01
    num_iterations = 1000
    allocations = initial_guess
    grad_fn = jit(grad(penalized_objective_log))
    
    for i in range(num_iterations):
        grads = grad_fn(allocations)
        allocations = allocations - learning_rate * grads
        allocations = jnp.clip(allocations, 0.01, 0.99)
        
        if i % 200 == 0:
            obj_val = penalized_objective_log(allocations)
            print(f"Log utility iteration {i}: Objective = {obj_val:.6f}")
    
    # Extract results
    c_opt = allocations[:S]
    l_opt = allocations[S:]
    Uc_vals = vmap(lambda ci, li: log_utility_c(ci, li, log_params))(c_opt, l_opt)
    Ul_vals = vmap(lambda ci, li: log_utility_l(ci, li, log_params))(c_opt, l_opt)
    τ_opt = 1 - Ul_vals / Uc_vals
    
    return {
        'c': c_opt,
        'l': l_opt,
        'n': 1 - l_opt,
        'τ': τ_opt,
        'objective': log_ramsey_objective(allocations),
        'budget_constraint': log_budget_constraint(allocations),
        'feasibility_constraint': log_feasibility_constraint(allocations)
    }
    """
    Solve simplified Ramsey problem using constrained optimization.
    
    Parameters
    ----------
    crra_params : CRRAUtilityParams
        Utility parameters
    g : array
        Government spending by state
    initial_guess : array, optional
        Initial guess for allocations
        
    Returns
    -------
    dict
        Solution with optimal allocations and tax rates
    """
    S = len(g)
    
    if initial_guess is None:
        # Simple initial guess
        c_guess = 0.5 * jnp.ones(S)
        l_guess = 0.5 * jnp.ones(S) 
        initial_guess = jnp.concatenate([c_guess, l_guess])
    
    # For simplicity, use a penalty method approach
    @jit
    def penalized_objective(allocations, penalty=1000.0):
        obj = ramsey_objective(allocations, crra_params, g)
        
        # Add penalties for constraint violations
        budget_viol = budget_constraint(allocations, crra_params, g)
        feasibility_viol = feasibility_constraint(allocations, g)
        
        penalty_term = (penalty * jnp.sum(jnp.maximum(0, -budget_viol)**2) +  # Budget surplus penalty
                       penalty * jnp.sum(jnp.maximum(0, feasibility_viol)**2))  # Feasibility penalty
        
        return obj + penalty_term
    
    # Simple gradient descent (demonstrative)
    learning_rate = 0.01
    num_iterations = 1000
    
    allocations = initial_guess
    
    grad_fn = jit(grad(penalized_objective))
    
    for i in range(num_iterations):
        grads = grad_fn(allocations)
        allocations = allocations - learning_rate * grads
        
        # Clip to reasonable bounds
        allocations = jnp.clip(allocations, 0.01, 0.99)
        
        if i % 200 == 0:
            obj_val = penalized_objective(allocations)
            print(f"Iteration {i}: Objective = {obj_val:.6f}")
    
    # Extract results
    c_opt = allocations[:S]
    l_opt = allocations[S:]
    τ_opt = compute_tax_rates(c_opt, l_opt, crra_params)
    
    return {
        'c': c_opt,
        'l': l_opt,
        'n': 1 - l_opt,
        'τ': τ_opt,
        'objective': ramsey_objective(allocations, crra_params, g),
        'budget_constraint': budget_constraint(allocations, crra_params, g),
        'feasibility_constraint': feasibility_constraint(allocations, g)
    }


def create_amss_simple_example():
    """Create a simple AMSS example."""
    
    # Parameters
    β = 0.9
    σ = 2.0
    γ = 2.0
    
    # Two-state Markov chain
    Π = jnp.array([[0.8, 0.2],
                   [0.3, 0.7]])
    
    # Government spending in each state  
    g = jnp.array([0.1, 0.2])  # Low and high spending
    
    # Create utility parameters
    crra_params = CRRAUtilityParams(β=β, σ=σ, γ=γ)
    
    return crra_params, Π, g


# Example usage
if __name__ == "__main__":
    # Create and solve simple AMSS model
    crra_params, Π, g = create_amss_simple_example()
    
    print("Solving simplified AMSS model...")
    solution = solve_simple_ramsey(crra_params, g)
    
    print("\nOptimal allocations:")
    print(f"Consumption: {solution['c']}")
    print(f"Leisure: {solution['l']}")  
    print(f"Labor: {solution['n']}")
    print(f"Tax rates: {solution['τ']}")
    print(f"\nObjective value: {solution['objective']:.6f}")