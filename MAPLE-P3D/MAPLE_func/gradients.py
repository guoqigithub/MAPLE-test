"""
gradients.py - Heavy computational work for MUSE optimization

This module handles all the expensive gradient and Hessian computations
required for the MUSE (Simulation-based inference) optimization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, hessian, jacfwd, jvp, value_and_grad, vmap
from jax.scipy.sparse.linalg import cg
from functools import partial
from datetime import datetime, timedelta
from copy import copy
from tqdm import tqdm


def compute_muse_gradient(prob, xs, zs, theta, nsims, method, tolerances, pool):
    """
    Compute MUSE gradient using multiple simulations.
    
    Args:
        prob: MUSE problem instance
        xs: List of x data (observations + simulations)
        zs: List of z latent variables
        theta: Current parameter values
        nsims: Number of simulations
        method: Optimization method
        tolerances: Dict with theta_tol and z_tol
        pool: Multiprocessing pool
        
    Returns:
        Dict containing gradient components and statistics
    """
    theta_tol = tolerances.get('theta_tol', 1e-5)
    z_tol = tolerances.get('z_tol', 1e-6)
    
    # Parallel MAP estimation
    MAPs = get_MAPs_parallel(prob, list(zip(xs, zs)), theta, method, theta_tol, z_tol, pool)
    
    # Extract MAP results
    zs_updated = [MAP.z for MAP in MAPs]
    s_MAP_dat, *s_MAP_sims = [MAP.s for MAP in MAPs]
    s_tilde_MAP_dat, *s_tilde_MAP_sims = [MAP.s̃ for MAP in MAPs]
    
    # Compute MUSE gradient
    s_tilde_MUSE = _compute_muse_score(s_tilde_MAP_dat, s_tilde_MAP_sims)
    
    # Add prior gradient
    s_tilde_prior, H_prior = gradθ_hessθ_logPrior(prob, theta)
    s_tilde_post = s_tilde_MUSE + s_tilde_prior
    
    # Compute likelihood Hessian approximation
    H_inv_like_sims = _compute_likelihood_hessian_approx(s_tilde_MAP_sims)
    
    # Posterior Hessian
    H_inv_post = _compute_posterior_hessian_inv(H_inv_like_sims, H_prior)
    
    return {
        'zs_updated': zs_updated,
        's_MAP_sims': s_MAP_sims,
        's_tilde_post': s_tilde_post,
        'H_inv_post': H_inv_post,
        'H_prior': H_prior,
        'H_inv_like_sims': H_inv_like_sims
    }


def get_MAPs_parallel(prob, x_z_pairs, theta, method, theta_tol, z_tol, pool):
    """
    Parallel MAP estimation for multiple datasets.
    
    Args:
        prob: MUSE problem instance
        x_z_pairs: List of (x, z) pairs
        theta: Current parameters
        method: Optimization method
        theta_tol, z_tol: Tolerances
        pool: Multiprocessing pool
        
    Returns:
        List of MAP results
    """
    def get_MAP(x_z):
        x, z_prev = x_z
        return prob.z_MAP_and_score(x, z_prev, theta, method=method, 
                                   z_tol=z_tol, θ_tol=theta_tol)
    
    return list(pool.map(get_MAP, x_z_pairs))


def compute_fisher_matrix(prob, theta, nsims, method, tolerances, rng, pool):
    """
    Compute Fisher information matrix J.
    
    Args:
        prob: MUSE problem instance
        theta: Current parameters
        nsims: Number of simulations
        method: Optimization method
        tolerances: Tolerance dict
        rng: JAX random key
        pool: Multiprocessing pool
        
    Returns:
        Fisher matrix J
    """
    theta_tol = tolerances.get('theta_tol', 1e-5)
    z_tol = tolerances.get('z_tol', 1e-6)
    
    def get_s_MAP(rng_key):
        """Generate simulation and compute score"""
        (x, z) = prob.sample_x_z(rng_key, theta)
        z_MAP_guess = prob.z_MAP_guess_from_truth(x, z, theta)
        result = prob.z_MAP_and_score(x, z_MAP_guess, theta, 
                                     method=method, θ_tol=theta_tol, z_tol=z_tol)
        return result.s
    
    # Generate random keys
    rngs = _split_rng(rng, nsims)
    
    # Parallel computation
    s_MAP_sims = [s for s in pool.map(get_s_MAP, rngs) if s is not None]
    
    # Compute covariance matrix
    if len(s_MAP_sims) == 0:
        raise ValueError("No valid simulations generated")
    
    # Stack and compute covariance
    stacked_s = np.stack([_ravel_theta(s) for s in s_MAP_sims])
    
    # Check for NaN values
    if np.isnan(stacked_s).any():
        raise ValueError("NaN detected in score simulations")
    
    J = np.atleast_2d(np.cov(stacked_s, rowvar=False))
    
    return J, s_MAP_sims


def compute_hessian_matrix(prob, theta, z_MAP_sims, method, tolerances, step_size, rng, nsims, pool):
    """
    Compute Hessian matrix H using finite differences or implicit differentiation.
    
    Args:
        prob: MUSE problem instance
        theta: Current parameters
        z_MAP_sims: List of MAP z values from simulations
        method: Optimization method
        tolerances: Tolerance dict
        step_size: Step size for finite differences
        rng: JAX random key
        nsims: Number of simulations
        pool: Multiprocessing pool
        
    Returns:
        Hessian matrix H
    """
    theta_tol = tolerances.get('theta_tol', 1e-5)
    z_tol = tolerances.get('z_tol', 1e-6)
    implicit_diff_cgtol = tolerances.get('implicit_diff_cgtol', 1e-3)
    
    # Generate random keys
    rngs = _split_rng(rng, nsims)
    
    # Ensure z_MAP_sims has right length
    z_MAP_sims = (z_MAP_sims + [None] * max(0, nsims - len(z_MAP_sims)))[:nsims]
    
    # Partial function for Hessian computation
    _get_H_i_partial = partial(_get_H_i_implicit, 
                              θ=theta, 
                              implicit_diff_cgtol=implicit_diff_cgtol,
                              method=method, 
                              θ_tol=theta_tol, 
                              z_tol=z_tol,
                              step=step_size)
    
    # Compute Hessians in parallel
    with tqdm(total=nsims, desc="Computing Hessians") as pbar:
        def compute_with_progress(args):
            result = _get_H_i_partial(*args)
            pbar.update(1)
            return result
        
        Hs = [H for H in pool.map(compute_with_progress, zip(rngs, z_MAP_sims)) 
              if H is not None]
    
    if len(Hs) == 0:
        raise ValueError("No valid Hessian computations")
    
    # Average Hessians
    H = np.mean(np.array(Hs), axis=0)
    
    return H


def pjacobian(f, x, step, pmap=map, pbar=None):
    """
    Parallel Jacobian computation using finite differences.
    
    Args:
        f: Function to differentiate
        x: Point to evaluate Jacobian
        step: Step size (scalar or array)
        pmap: Parallel map function
        pbar: Progress bar
        
    Returns:
        Jacobian matrix
    """
    step = step + np.array(0 * x)  # make array if scalar
    
    def column(i):
        def v(ε):
            ε_vec = np.array(0 * x)
            ε_vec[i] = ε
            v = f(x + ε_vec)
            if pbar: 
                pbar.update()
            return v
        
        return (v(step[i]) - v(-step[i])) / (2 * step[i])
    
    return np.array(list(pmap(column, range(len(x)))))


def gradθ_hessθ_logPrior(prob, theta):
    """
    Compute prior gradient and Hessian.
    
    Args:
        prob: MUSE problem instance
        theta: Current parameters
        
    Returns:
        Tuple of (gradient, Hessian)
    """
    g = grad(prob.logPrior)(theta)
    H = hessian(prob.logPrior)(theta)
    return g, H


def update_theta(theta, gradient, hessian_inv, alpha, beta):
    """
    Update parameters with clipping and learning rate.
    
    Args:
        theta: Current parameters
        gradient: Gradient vector
        hessian_inv: Inverse Hessian matrix
        alpha: Learning rate
        beta: Clipping parameter
        
    Returns:
        Updated parameters
    """
    # Compute update step
    theta_ravel = _ravel_theta(theta)
    gradient_ravel = _ravel_theta(gradient)
    
    update_step = np.inner(hessian_inv, gradient_ravel)
    
    # Clip update
    clip_bounds = jnp.abs(theta_ravel * beta)
    update_clipped = np.clip(update_step, -clip_bounds, clip_bounds)
    
    # Apply update
    theta_new_ravel = theta_ravel - alpha * update_clipped
    
    return _unravel_theta(theta_new_ravel)


# Helper functions
def _compute_muse_score(s_tilde_MAP_dat, s_tilde_MAP_sims):
    """Compute MUSE score from MAP results"""
    mean_sim_score = np.nanmean(np.stack([_ravel_theta(s) for s in s_tilde_MAP_sims]), axis=0)
    muse_score = _ravel_theta(s_tilde_MAP_dat) - mean_sim_score
    return _unravel_theta(muse_score)


def _compute_likelihood_hessian_approx(s_tilde_MAP_sims):
    """Approximate likelihood Hessian from simulation variance"""
    epsilon = 1e-8  # Avoid division by zero
    variance = np.nanvar(np.stack([_ravel_theta(s) for s in s_tilde_MAP_sims]), axis=0)
    variance += epsilon
    return np.diag(-1 / variance)


def _compute_posterior_hessian_inv(H_inv_like_sims, H_prior):
    """Compute inverse posterior Hessian"""
    Nθ = int(np.sqrt(len(_ravel_theta(H_prior))))
    H_prior_matrix = _ravel_theta(H_prior).reshape(Nθ, Nθ)
    
    try:
        H_inv_post = np.linalg.pinv(np.linalg.pinv(H_inv_like_sims) + H_prior_matrix)
    except np.linalg.LinAlgError:
        print("Warning: Hessian inversion failed, using likelihood approximation only")
        H_inv_post = H_inv_like_sims
    
    return H_inv_post


def _get_H_i_implicit(rng, z_MAP, *, θ, implicit_diff_cgtol=1e-3, method=None, 
                     θ_tol=None, z_tol=None, step=None, prob=None):
    """
    Compute Hessian for single simulation using implicit differentiation.
    """
    cg_kwargs = dict(tol=implicit_diff_cgtol)
    
    (x, z) = prob.sample_x_z(rng, θ)
    if z_MAP is None:
        z_MAP_guess = prob.z_MAP_guess_from_truth(x, z, θ)
        z_MAP = prob.z_MAP_and_score(x, z_MAP_guess, θ, 
                                    method=method, θ_tol=θ_tol, z_tol=z_tol).z
    
    θ_vec, z_MAP_vec = _ravel_theta(θ), _ravel_z(z_MAP)
    
    # Non-implicit-diff term
    H1 = jacfwd(
        lambda θ1: grad(
            lambda θ2: prob.logLike(prob.sample_x_z(rng, _unravel_theta(θ1))[0], 
                                   z_MAP, _unravel_theta(θ2))
        )(θ_vec)
    )(θ_vec)
    
    # Implicit differentiation term
    dFdθ = jacfwd(
        lambda θ: grad(
            lambda z: prob.logLike(x, _unravel_z(z), _unravel_theta(θ))
        )(z_MAP_vec)
    )(θ_vec)
    
    dFdθ1 = jacfwd(
        lambda θ1: grad(
            lambda z: prob.logLike(prob.sample_x_z(rng, _unravel_theta(θ1))[0], 
                                  _unravel_z(z), θ)
        )(z_MAP_vec)
    )(θ_vec)
    
    inv_dFdz_dFdθ1 = vmap(
        lambda vec: cg(
            lambda vec: jvp(
                lambda z: grad(lambda z: prob.logLike(x, _unravel_z(z), θ))(z), 
                (z_MAP_vec,), (vec,)
            )[1], 
            vec, 
            **cg_kwargs
        )[0], 
        in_axes=1, out_axes=1
    )(dFdθ1)
    
    H2 = -dFdθ.T @ inv_dFdz_dFdθ1
    
    return H1 + H2


def _split_rng(rng, N):
    """Split JAX random key into N keys"""
    keys = []
    for i in range(N):
        rng, subkey = jax.random.split(rng)
        keys.append(rng)
    return keys


# Placeholder functions - these should be defined based on your specific theta/z structure
def _ravel_theta(theta):
    """Flatten theta to 1D array"""
    if hasattr(theta, 'flatten'):
        return theta.flatten()
    return np.array(theta).flatten()


def _unravel_theta(theta_flat):
    """Reshape flattened theta back to original structure"""
    # This needs to be implemented based on your theta structure
    return theta_flat


def _ravel_z(z):
    """Flatten z to 1D array"""
    if hasattr(z, 'flatten'):
        return z.flatten()
    return np.array(z).flatten()


def _unravel_z(z_flat):
    """Reshape flattened z back to original structure"""
    # This needs to be implemented based on your z structure
    return z_flat