"""
optimization.py - MUSE Optimization Module for P3D Analysis

This module contains the optimization classes and functions for running
MUSE inference on P3D cosmological data.

Author: P3D Analysis Framework
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, hessian, jacfwd, jvp, value_and_grad, vmap
from jax.scipy.optimize import minimize
from jax.scipy.sparse.linalg import cg
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool as Pool
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import copy
import scipy as sp


class MuseOptimizer:
    """
    Main optimization class for MUSE inference
    Handles the iterative optimization process with gradient estimation
    """
    
    def __init__(self, problem, config):
        """
        Initialize the MUSE optimizer
        
        Args:
            problem: P3DMuseProblem instance
            config: Configuration dictionary
        """
        self.problem = problem
        self.config = config
        
        # Optimization parameters
        self.method = config.get('optimization_method', 'l-bfgs-experimental-do-not-rely-on-this')
        self.θ_rtol = config.get('theta_rtol', 1e-5)
        self.z_rtol = config.get('z_rtol', 1e-5)
        self.θ_tol = config.get('theta_tol', 1e-5)
        self.z_tol = config.get('z_tol', 1e-6)
        self.learning_rate = config.get('learning_rate', 0.7)
        self.beta = config.get('beta', 0.25)
        self.lr_decay = config.get('lr_decay', 0.95)
        self.lr_decay_interval = config.get('lr_decay_interval', 3)
        
        # Setup multiprocessing
        self.pool = Pool()
        self.pmap = self.pool.map
        
        # Utility functions
        self.ravel_θ = lambda θ: np.array(θ).flatten()
        self.unravel_θ = lambda θ_vec: θ_vec.reshape(-1)
        
    def _split_rng(self, rng, N):
        """Split random key into N subkeys"""
        keyz = []
        for i in range(N):
            rng, subkey = jax.random.split(rng)
            keyz.append(rng)
        return keyz
    
    def _gradθ_hessθ_logPrior(self, θ):
        """Compute gradient and Hessian of log prior"""
        g = grad(self.problem.logPrior)(θ)
        H = hessian(lambda θ_vec: self.problem.logPrior(self.unravel_θ(θ_vec)))(self.ravel_θ(θ))
        return (g, H)
    
    def _get_MAPs(self, x_z, θ, method, z_tol, θ_tol):
        """Get MAP estimates for given data"""
        x, ẑ_prev = x_z
        result = self.problem.z_MAP_and_score(x, ẑ_prev, θ, method=method, z_tol=z_tol, θ_tol=θ_tol)
        return result
    
    def run_optimization(self, x_data, start_point, rng, maxsteps=200, nsims=10):
        """
        Main optimization loop
        
        Args:
            x_data: Observed data
            start_point: Starting parameter values
            rng: JAX random key
            maxsteps: Maximum optimization steps
            nsims: Number of simulations per step
            
        Returns:
            dict: Optimization results including history and final parameters
        """
        print("Starting MUSE optimization...")
        print(f"Parameters: maxsteps={maxsteps}, nsims={nsims}")
        
        # Initialize
        θ̃ = start_point.copy()
        θ = start_point.copy()
        θL = start_point.copy()
        Nθ = len(self.ravel_θ(θ̃))
        
        history = []
        α = self.learning_rate
        time_total = timedelta(0)
        
        # Initial samples
        z = jax.random.normal(rng, (self.config['simulation']['n_cells']**3,))
        xz_sims = [self.problem.sample_x_z(_rng, θ) for _rng in self._split_rng(rng, nsims)]
        xs = [x_data] + [x for (x, _) in xz_sims]
        ẑs = [z * 0] + [z * 0 for (x, z) in xz_sims]
        
        for i in range(1, maxsteps + 1):
            print(f"Optimization step: {i}, learning rate: {α:.6f}")
            t0 = datetime.now()
            
            # Update samples after first iteration
            if i > 1:
                xs = [x_data] + [self.problem.sample_x_z(_rng, θ)[0] 
                                for _rng in self._split_rng(rng, nsims)]
                θ_tol = np.sqrt(-np.diag(H̃_inv_post)) * self.θ_rtol
            
            # Check convergence
            if i > 2:
                Δθ̃ = self.ravel_θ(history[-1]["θ̃"]) - self.ravel_θ(history[-2]["θ̃"])
                # Add convergence check here if needed

            get_MAPs_partial = partial(self._get_MAPs, θ=θ, method=self.method, z_tol=self.z_tol, θ_tol=self.θ_tol)
            MAPs = list(self.pmap(get_MAPs_partial, zip(xs, ẑs)))

        
            ẑs = [MAP.z for MAP in MAPs]
            s_MAP_dat, *s_MAP_sims = [MAP.s for MAP in MAPs]
            s̃_MAP_dat, *s̃_MAP_sims = [MAP.s̃ for MAP in MAPs]
            
            # Compute MUSE gradient
            s̃_MUSE = self.unravel_θ(
                self.ravel_θ(s̃_MAP_dat) - 
                np.nanmean(np.stack(list(map(self.ravel_θ, s̃_MAP_sims))), axis=0)
            )
            
            # Prior terms
            s̃_prior, H̃_prior = self._gradθ_hessθ_logPrior(θ̃)
            s̃_post = self.unravel_θ(self.ravel_θ(s̃_MUSE) + self.ravel_θ(s̃_prior))
            
            # Compute Hessian approximation
            epsilon = 1e-8
            variance = np.nanvar(np.stack(list(map(self.ravel_θ, s̃_MAP_sims))), axis=0)
            variance += epsilon
            H̃_inv_like_sims = np.diag(-1 / variance)
            
            # Posterior Hessian
            try:
                H̃_inv_post = np.linalg.pinv(
                    np.linalg.pinv(H̃_inv_like_sims) + 
                    self.ravel_θ(H̃_prior).reshape(Nθ, Nθ)
                )
            except:
                print("Warning: Hessian inversion failed, using likelihood approximation")
                H̃_inv_post = H̃_inv_like_sims
            
            # Record step timing
            t = datetime.now() - t0
            time_total += t
            
            # Store history
            history.append({
                "step": i,
                "time": t,
                "θ̃": θ̃.copy(),
                "θ": θ.copy(),
                "s_MAP_dat": s_MAP_dat,
                "s_MAP_sims": s_MAP_sims,
                "s̃_MAP_dat": s̃_MAP_dat,
                "s̃_MAP_sims": s̃_MAP_sims,
                "s̃_MUSE": s̃_MUSE,
                "s̃_prior": s̃_prior,
                "s̃_post": s̃_post,
                "H̃_inv_post": H̃_inv_post,
                "H̃_prior": H̃_prior,
                "H̃_inv_like_sims": H̃_inv_like_sims,
                "θ_tol": self.θ_tol,
                "learning_rate": α
            })
            
            # Parameter update
            θ̃update = np.clip(
                np.inner(H̃_inv_post, self.ravel_θ(s̃_post)),
                -np.abs(self.ravel_θ(θ̃) * self.beta),
                np.abs(self.ravel_θ(θ̃) * self.beta)
            )
            θ̃ = self.unravel_θ(self.ravel_θ(θ̃) - α * θ̃update)
            θ = θ̃.copy()
            
            # Progress plotting
            if i % 3 == 0:
                if i % 6 == 0 and self.config.get('plot_progress', True):
                    self._plot_progress(start_point, θ, θL, 
                                      self.problem.get_fiducial_parameters())
                θL = θ.copy()
                
                # Learning rate decay
                if i % self.config.get('lr_major_decay_interval', 150) == 0 and i != 0:
                    α = 0.5
                if i % self.lr_decay_interval == 0:
                    α *= self.lr_decay
        
        return {
            "θ_final": θ,
            "θ̃_final": θ̃,
            "history": history,
            "total_time": time_total,
            "s_MAP_sims": s_MAP_sims,
            "ẑs": ẑs
        }
    
    def _plot_progress(self, start_point, θ, θL, θ_fid):
        """Plot optimization progress"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(start_point, "r.-", label="Start")
        plt.plot(θ, "b-", label="Current")
        plt.plot(θ_fid, "k:", label="Fiducial")
        plt.yscale("symlog")
        plt.ylabel("Parameter Values")
        plt.legend()
        plt.title("Parameter Evolution")
        
        plt.subplot(2, 1, 2)
        plt.plot((start_point - θ_fid) / θ_fid, "r.-", label="Start Error")
        plt.plot((θ - θ_fid) / θ_fid, "b-", label="Current Error")
        plt.plot((θL - θ_fid) / θ_fid, "g:", label="Previous Error")
        plt.ylim([-0.3, 0.3])
        plt.ylabel("Relative Error")
        plt.xlabel("Parameter Index")
        plt.legend()
        
        plt.tight_layout()
        plt.show()


class CovarianceEstimator:
    """
    Estimates covariance matrix using Fisher information
    Computes score covariance and Fisher matrix for final uncertainty estimation
    """
    
    def __init__(self, problem, config):
        """
        Initialize covariance estimator
        
        Args:
            problem: P3DMuseProblem instance
            config: Configuration dictionary
        """
        self.problem = problem
        self.config = config
        
        # Setup multiprocessing
        self.pool = Pool()
        self.pmap = self.pool.map
        
        # Get problem dimensions
        self.nc = config.get('simulation', {}).get('n_cells', 150)
        
        # Utility functions - these should match your problem's ravel/unravel functions
        self.ravel_θ = lambda θ: np.array(θ).flatten()
        self.unravel_θ = lambda θ_vec: θ_vec.reshape(-1)
        self.ravel_z = lambda z: np.array(z).flatten() 
        self.unravel_z = lambda z_vec: z_vec.reshape(self.nc, self.nc, self.nc)
        
    def _split_rng(self, rng, N):
        """Split random key into N subkeys"""
        keyz = []
        for i in range(N):
            rng, subkey = jax.random.split(rng)
            keyz.append(rng)
        return keyz
    
    def get_score_covariance(self, θ, s_MAP_sims, rng, nsims=200):
        """
        Compute score covariance matrix J
        
        Args:
            θ: Current parameter values
            s_MAP_sims: Existing MAP score simulations (will be modified in place)
            rng: JAX random key
            nsims: Number of simulations to add
            
        Returns:
            np.ndarray: Score covariance matrix J
        """
        print(f"Computing score covariance with {nsims} additional simulations...")
        
        method = self.config.get('optimization_method', 'l-bfgs-experimental-do-not-rely-on-this')
        θ_tol = self.config.get('theta_tol', 1e-5)
        z_tol = self.config.get('z_tol', 1e-6)
        
        def get_s_MAP(rng_key):
            """Generate single MAP score estimate"""
            try:
                (x, z) = self.problem.sample_x_z(rng_key, θ)
                
                # Check for NaN in sampled data
                if np.isnan(x).any() or np.isnan(z).any():
                    print("Warning: NaN detected in sampled (x, z)")
                    return None
                    
                # Use problem's method to get initial guess
                if hasattr(self.problem, 'z_MAP_guess_from_truth'):
                    z_MAP_guess = self.problem.z_MAP_guess_from_truth(x, z, θ)
                else:
                    # Fallback: use zero initialization
                    z_MAP_guess = self.unravel_z(np.zeros(self.nc**3))
                
                if np.isnan(z_MAP_guess).any():
                    print("Warning: NaN detected in z_MAP_guess")
                    return None
                
                result = self.problem.z_MAP_and_score(x, z_MAP_guess, θ, 
                                                    method=method, θ_tol=θ_tol, z_tol=z_tol)
                
                if result.s is None or np.isnan(result.s).any():
                    print("Warning: NaN detected in MAP score")
                    return None
                    
                return result.s
                
            except Exception as e:
                print(f"Warning: Score computation failed: {e}")
                return None
        
        # Generate new simulations
        if nsims > 0:
            rngs = self._split_rng(rng, nsims)
            print("Generating MAP scores...")
            new_sims = []
            
            for i, rng_key in enumerate(rngs):
                if (i + 1) % 20 == 0:
                    print(f"Completed {i + 1}/{nsims} score computations")
                    
                s = get_s_MAP(rng_key)
                if s is not None:
                    new_sims.append(s)
            
            print(f"Successfully generated {len(new_sims)} out of {nsims} scores")
            s_MAP_sims.extend(new_sims)
        
        if len(s_MAP_sims) == 0:
            print("Error: No valid score simulations available!")
            return None
        
        # Compute covariance
        try:
            stacked_s = np.stack([self.ravel_θ(s) for s in s_MAP_sims])
            print(f"Score covariance computed from {len(s_MAP_sims)} simulations")
            print(f"Stacked scores shape: {stacked_s.shape}")
            
            if np.isnan(stacked_s).any():
                print("Warning: NaN detected in stacked score simulations!")
                # Remove NaN rows
                valid_mask = ~np.isnan(stacked_s).any(axis=1)
                stacked_s = stacked_s[valid_mask]
                print(f"Removed NaN entries, using {len(stacked_s)} valid simulations")
                
                if len(stacked_s) == 0:
                    print("Error: No valid simulations after NaN removal!")
                    return None
            
            # Compute covariance matrix
            if len(stacked_s) == 1:
                print("Warning: Only one simulation available, using identity covariance")
                J = np.eye(stacked_s.shape[1])
            else:
                J = np.cov(stacked_s, rowvar=False)
                J = np.atleast_2d(J)
            
            # Check condition number
            cond_number = np.linalg.cond(J)
            print(f"Score covariance condition number: {cond_number:.2e}")
            
            if cond_number > 1e12:
                print("Warning: Score covariance matrix is nearly singular!")
                # Add regularization
                reg_param = 1e-8 * np.trace(J) / J.shape[0]
                J += reg_param * np.eye(J.shape[0])
                print(f"Added regularization: {reg_param:.2e}")
            
            return J
            
        except Exception as e:
            print(f"Error computing score covariance: {e}")
            return None
    
    def pjacobian(self, f, x, step):
        """
        Parallel Jacobian computation using finite differences
        
        Args:
            f: Function to differentiate
            x: Point at which to compute Jacobian
            step: Step size for finite differences
            
        Returns:
            np.ndarray: Jacobian matrix
        """
        print(f"Computing Jacobian at point with {len(x)} parameters")
        step = np.array(step) + np.array(0 * x)  # Ensure step is array
        
        def column(i):
            """Compute i-th column of Jacobian"""
            try:
                def v(ε):
                    ε_vec = np.array(0 * x, dtype=float)
                    ε_vec[i] = ε
                    return f(x + ε_vec)
                
                col = (v(step[i]) - v(-step[i])) / (2 * step[i])
                return col
            except Exception as e:
                print(f"Warning: Jacobian column {i} computation failed: {e}")
                return np.zeros_like(f(x))  # Return zero column on failure
        
        try:
            jacobian = np.array(list(self.pmap(column, range(len(x)))))
            return jacobian.T  # Transpose to get correct shape
        except Exception as e:
            print(f"Error in Jacobian computation: {e}")
            return None
    
    def _get_H_i(self, rng, z_MAP, θ, implicit_diff_cgtol=1e-3, method=None, 
                θ_tol=None, z_tol=None, step=None):
        """
        Compute Hessian for single simulation using implicit differentiation
        
        Args:
            rng: JAX random key
            z_MAP: MAP estimate of latent variables
            θ: Current parameters
            implicit_diff_cgtol: Conjugate gradient tolerance
            method: Optimization method
            θ_tol: Parameter tolerance
            z_tol: Latent variable tolerance
            step: Step size for finite differences
            
        Returns:
            np.ndarray: Hessian contribution from this simulation
        """
        try:
            cg_kwargs = dict(tol=implicit_diff_cgtol)
            
            (x, z) = self.problem.sample_x_z(rng, θ)
            
            # Get MAP estimate if not provided
            if z_MAP is None:
                if hasattr(self.problem, 'z_MAP_guess_from_truth'):
                    z_MAP_guess = self.problem.z_MAP_guess_from_truth(x, z, θ)
                else:
                    z_MAP_guess = self.unravel_z(np.zeros(self.nc**3))
                    
                z_MAP = self.problem.z_MAP_and_score(x, z_MAP_guess, θ, 
                                                   method=method, θ_tol=θ_tol, z_tol=z_tol).z
            
            θ_vec, z_MAP_vec = self.ravel_θ(θ), self.ravel_z(z_MAP)
            
            # Non-implicit-diff term
            H1 = jacfwd(
                lambda θ1: grad(
                    lambda θ2: self.problem.logLike(
                        self.problem.sample_x_z(rng, self.unravel_θ(θ1))[0], 
                        z_MAP, self.unravel_θ(θ2)
                    )
                )(θ_vec)
            )(θ_vec)
            
            # Implicit differentiation term
            dFdθ = jacfwd(
                lambda θ: grad(
                    lambda z: self.problem.logLike(x, self.unravel_z(z), self.unravel_θ(θ))
                )(z_MAP_vec)
            )(θ_vec)
            
            dFdθ1 = jacfwd(
                lambda θ1: grad(
                    lambda z: self.problem.logLike(
                        self.problem.sample_x_z(rng, self.unravel_θ(θ1))[0], 
                        self.unravel_z(z), θ
                    )
                )(z_MAP_vec)
            )(θ_vec)
            
            # Conjugate gradient solve
            def cg_solve(vec):
                def matvec(v):
                    return jvp(
                        lambda z: grad(lambda z: self.problem.logLike(x, self.unravel_z(z), θ))(z), 
                        (z_MAP_vec,), (v,)
                    )[1]
                return cg(matvec, vec, **cg_kwargs)[0]
            
            inv_dFdz_dFdθ1 = vmap(cg_solve, in_axes=1, out_axes=1)(dFdθ1)
            
            H2 = -dFdθ.T @ inv_dFdz_dFdθ1
            H = H1 + H2
            
            # Check for NaN/inf
            if np.isnan(H).any() or np.isinf(H).any():
                print("Warning: NaN/inf detected in Hessian computation")
                return None
                
            return H
            
        except Exception as e:
            print(f"Warning: Hessian computation failed: {e}")
            return None
    
    def get_fisher_matrix(self, θ, z_MAP_sims, rng, nsims=20):
        """
        Compute Fisher information matrix H
        
        Args:
            θ: Current parameters
            z_MAP_sims: MAP estimates from previous optimization
            rng: JAX random key
            nsims: Number of simulations for Fisher matrix
            
        Returns:
            np.ndarray: Fisher information matrix H
        """
        print(f"Computing Fisher information matrix with {nsims} simulations...")
        
        method = self.config.get('optimization_method', 'l-bfgs-experimental-do-not-rely-on-this')
        θ_tol = self.config.get('theta_tol', 1e-5)
        z_tol = self.config.get('z_tol', 1e-6)
        
        # Default step size - estimate from parameter scale
        if hasattr(self.problem, 'get_fiducial_parameters'):
            θ_fid = self.problem.get_fiducial_parameters()
            step = 0.01 * np.abs(θ_fid) + 1e-6  # 1% of fiducial + small constant
        else:
            step = 0.01 * np.abs(θ) + 1e-6
        
        # Prepare simulation inputs
        rngs = self._split_rng(rng, nsims)
        z_MAP_sims_padded = (z_MAP_sims + [None] * max(0, nsims - len(z_MAP_sims)))[:nsims]
        
        # Compute Hessian contributions
        print("Computing Hessian contributions...")
        Hs = []
        
        for i, (rng_i, z_MAP_i) in enumerate(zip(rngs, z_MAP_sims_padded)):
            try:
                H_i = self._get_H_i(rng_i, z_MAP_i, θ=θ, method=method, 
                                   θ_tol=θ_tol, z_tol=z_tol, step=step)
                if H_i is not None:
                    Hs.append(H_i)
                    
                if (i + 1) % 5 == 0:
                    print(f"Completed {i + 1}/{nsims} Hessian computations")
                    
            except Exception as e:
                print(f"Warning: Hessian computation {i} failed: {e}")
                continue
        
        if len(Hs) == 0:
            print("Error: No valid Hessian computations!")
            return None
        
        # Average Hessian contributions
        H_array = np.array(Hs)
        print(f"Hessian array shape: {H_array.shape}")
        
        # Remove outliers based on norm
        norms = np.array([np.linalg.norm(H_i) for H_i in Hs])
        median_norm = np.median(norms)
        mad = np.median(np.abs(norms - median_norm))
        threshold = median_norm + 5 * mad  # 5-MAD threshold
        
        valid_mask = norms <= threshold
        if np.sum(valid_mask) < len(Hs):
            print(f"Filtered {len(Hs) - np.sum(valid_mask)} outlier Hessian matrices")
            Hs = [H for H, valid in zip(Hs, valid_mask) if valid]
        
        H = np.mean(np.array(Hs), axis=0)
        print(f"Fisher matrix computed from {len(Hs)} valid simulations")
        
        # Check condition number
        cond_number = np.linalg.cond(H)
        print(f"Fisher matrix condition number: {cond_number:.2e}")
        
        if cond_number > 1e12:
            print("Warning: Fisher matrix is nearly singular!")
            # Add regularization
            reg_param = 1e-8 * np.trace(np.abs(H)) / H.shape[0]
            H += reg_param * np.eye(H.shape[0])
            print(f"Added regularization: {reg_param:.2e}")
        
        return H
    
    def compute_covariance(self, θ, J, H, s_MAP_sims=None, z_MAP_sims=None, rng=None):
        """
        Compute final parameter covariance matrix
        
        Args:
            θ: Final parameter estimates
            J: Score covariance matrix
            H: Fisher information matrix
            s_MAP_sims: MAP score simulations (optional)
            z_MAP_sims: MAP latent variable simulations (optional)
            rng: JAX random key (optional)
            
        Returns:
            dict: Covariance results including matrix and distribution
        """
        print("Computing final parameter covariance...")
        
        if J is None or H is None:
            print("Error: Missing required matrices J or H")
            return {"success": False, "error": "Missing required matrices"}
        
        Nθ = len(self.ravel_θ(θ))
        print(f"Parameter dimension: {Nθ}")
        print(f"Score covariance shape: {J.shape}")
        print(f"Fisher matrix shape: {H.shape}")
        
        # Compute prior Hessian
        try:
            # Get gradient and Hessian of log prior
            g_prior = grad(self.problem.logPrior)(θ)
            H_prior_func = hessian(self.problem.logPrior)
            H_prior_matrix = H_prior_func(θ)
            
            # Ensure proper shape
            H_prior = -np.array(H_prior_matrix).reshape(Nθ, Nθ)
            
        except Exception as e:
            print(f"Warning: Prior Hessian computation failed: {e}")
            print("Using zero prior (flat prior)")
            H_prior = np.zeros((Nθ, Nθ))
        
        # Compute posterior precision and covariance
        try:
            # Ensure matrices have correct shapes
            if J.shape != (Nθ, Nθ):
                print(f"Warning: J shape {J.shape} doesn't match expected {(Nθ, Nθ)}")
                return {"success": False, "error": "Shape mismatch in J"}
                
            if H.shape != (Nθ, Nθ):
                print(f"Warning: H shape {H.shape} doesn't match expected {(Nθ, Nθ)}")
                return {"success": False, "error": "Shape mismatch in H"}
            
            # Compute precision matrix
            J_inv = np.linalg.pinv(J)
            Σ_inv = H.T @ J_inv @ H + H_prior
            
            # Compute covariance matrix
            Σ = np.linalg.pinv(Σ_inv)
            
            # Check if covariance is positive definite
            eigenvals = np.linalg.eigvals(Σ)
            if np.any(eigenvals <= 0):
                print(f"Warning: Covariance matrix is not positive definite")
                print(f"Minimum eigenvalue: {np.min(eigenvals)}")
                
                # Regularize if needed
                if np.min(eigenvals) < -1e-10:
                    reg_param = -2 * np.min(eigenvals) + 1e-10
                    Σ += reg_param * np.eye(Nθ)
                    print(f"Added regularization: {reg_param:.2e}")
            
            # Create distribution
            θ_flat = self.ravel_θ(θ)
            
            if Nθ == 1:
                std_dev = np.sqrt(Σ[0, 0])
                if std_dev <= 0:
                    print("Warning: Invalid standard deviation")
                    std_dev = 1.0
                dist = sp.stats.norm(θ_flat[0], std_dev)
            else:
                try:
                    dist = sp.stats.multivariate_normal(θ_flat, Σ)
                except Exception as e:
                    print(f"Warning: Could not create multivariate normal: {e}")
                    dist = None
            
            # Compute parameter uncertainties
            param_std = np.sqrt(np.diag(Σ))
            
            success = True
            error_msg = None
            
            print("Covariance computation successful!")
            print(f"Parameter standard deviations: {param_std}")
            
        except Exception as e:
            print(f"Error: Covariance computation failed: {e}")
            Σ_inv = None
            Σ = None
            dist = None
            param_std = None
            success = False
            error_msg = str(e)
        
        return {
            "covariance": Σ,
            "precision": Σ_inv,
            "distribution": dist,
            "parameter_std": param_std,
            "fisher_matrix": H,
            "score_covariance": J,
            "prior_hessian": H_prior,
            "success": success,
            "error": error_msg
        }

def run_full_analysis(problem, config, x_data, start_point, rng):
    """
    Run complete MUSE analysis pipeline
    
    Args:
        problem: P3DMuseProblem instance
        config: Configuration dictionary
        x_data: Observed data
        start_point: Starting parameter values
        rng: JAX random key
        
    Returns:
        dict: Complete analysis results
    """
    print("="*60)
    print("Starting Full P3D MUSE Analysis")
    print("="*60)
    
    total_start_time = datetime.now()
    
    # Phase 1: Optimization
    print("\nPhase 1: Parameter Optimization")
    print("-" * 40)
    
    optimizer = MuseOptimizer(problem, config)
    opt_results = optimizer.run_optimization(
        x_data=x_data,
        start_point=start_point,
        rng=rng,
        maxsteps=config.get('max_optimization_steps', 200),
        nsims=config.get('optimization_nsims', 10)
    )
    
    θ_final = opt_results['θ_final']
    s_MAP_sims = opt_results['s_MAP_sims']
    z_MAP_sims = [z for z in opt_results['ẑs'] if z is not None]
    
    print(f"Optimization completed in {opt_results['total_time']}")
    print(f"Final parameters: {θ_final}")
    
    # Phase 2: Covariance Estimation
    print("\nPhase 2: Covariance Estimation")
    print("-" * 40)
    
    estimator = CovarianceEstimator(problem, config)
    
    # Compute score covariance
    rng, subkey = jax.random.split(rng)
    J = estimator.get_score_covariance(
        θ=θ_final,
        s_MAP_sims=s_MAP_sims,
        rng=subkey,
        nsims=config.get('fisher', {}).get('n_sims_fisher', 20)
    )
    
    # Compute Fisher matrix
    rng, subkey = jax.random.split(rng)
    H = estimator.get_fisher_matrix(
        θ=θ_final,
        z_MAP_sims=z_MAP_sims,
        rng=subkey,
        nsims=config.get('fisher', {}).get('n_sims_hessian', 20)
    )
    
    # Final covariance
    cov_results = estimator.compute_covariance(θ_final, J, H)
    
    total_time = datetime.now() - total_start_time
    
    print("\n" + "="*60)
    print("P3D MUSE Analysis Complete!")
    print(f"Total runtime: {total_time}")
    print("="*60)
    
    # Compile final results
    results = {
        "start_point": start_point,
        "θ_fiducial": problem.get_fiducial_parameters(),
        "θ_final": θ_final,
        "covariance_results": cov_results,
        "optimization_results": opt_results,
        "total_runtime": total_time,
        "config": config,
        "success": cov_results['success']
    }
    
    return results