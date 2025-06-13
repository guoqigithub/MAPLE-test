"""
Statistical analysis and uncertainty quantification for P3D inference.

This module handles:
- Final covariance matrix computation
- Parameter constraint extraction
- Posterior distribution creation
- Convergence diagnostics
- Bias analysis
- Forecasting capabilities
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, hessian
import scipy as sp
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings


def compute_final_covariance(H_matrix: np.ndarray, 
                           J_matrix: np.ndarray, 
                           H_prior: np.ndarray,
                           regularization: float = 1e-12) -> np.ndarray:
    """
    Compute final parameter covariance matrix from MUSE estimation.
    
    Formula: Σ = (H^T J^{-1} H + H_prior)^{-1}
    
    Args:
        H_matrix: Hessian matrix (n_sims, n_params, n_params)
        J_matrix: Fisher information matrix (n_params, n_params)  
        H_prior: Prior Hessian matrix (n_params, n_params)
        regularization: Small value added to diagonal for numerical stability
        
    Returns:
        Covariance matrix (n_params, n_params)
    """
    # Average Hessian over simulations
    H_avg = np.mean(H_matrix, axis=0)
    
    # Add regularization to J for numerical stability
    J_reg = J_matrix + regularization * np.eye(J_matrix.shape[0])
    
    # Compute precision matrix
    try:
        J_inv = np.linalg.pinv(J_reg)
        precision = H_avg.T @ J_inv @ H_avg + H_prior
        covariance = np.linalg.pinv(precision)
    except np.linalg.LinAlgError:
        warnings.warn("Covariance computation failed, using diagonal approximation")
        # Fallback to diagonal approximation
        diag_precision = np.diag(np.diag(H_avg.T @ J_inv @ H_avg + H_prior))
        covariance = np.linalg.pinv(diag_precision)
    
    return covariance


def parameter_constraints(theta_best: np.ndarray,
                         covariance_matrix: np.ndarray,
                         confidence_levels: List[float] = [0.68, 0.95],
                         parameter_names: Optional[List[str]] = None) -> Dict:
    """
    Extract parameter constraints at different confidence levels.
    
    Args:
        theta_best: Best-fit parameter values
        covariance_matrix: Parameter covariance matrix
        confidence_levels: Confidence levels for constraints
        parameter_names: Optional parameter names
        
    Returns:
        Dictionary with constraint information
    """
    n_params = len(theta_best)
    if parameter_names is None:
        parameter_names = [f"θ_{i}" for i in range(n_params)]
    
    # Extract standard deviations
    std_devs = np.sqrt(np.diag(covariance_matrix))
    
    # Compute confidence intervals
    constraints = {}
    for cl in confidence_levels:
        # For Gaussian posterior, confidence interval is ±z*σ
        z_score = stats.norm.ppf(0.5 + cl/2)
        
        intervals = {}
        for i, name in enumerate(parameter_names):
            lower = theta_best[i] - z_score * std_devs[i]
            upper = theta_best[i] + z_score * std_devs[i]
            intervals[name] = {
                'best': theta_best[i],
                'lower': lower, 
                'upper': upper,
                'sigma': std_devs[i],
                'relative_error': std_devs[i] / np.abs(theta_best[i])
            }
        
        constraints[f'{cl:.0%}'] = intervals
    
    return constraints


def create_posterior_distribution(theta_best: np.ndarray, 
                                 covariance_matrix: np.ndarray) -> stats.multivariate_normal:
    """
    Create scipy multivariate normal distribution for posterior sampling.
    
    Args:
        theta_best: Best-fit parameter values
        covariance_matrix: Parameter covariance matrix
        
    Returns:
        Scipy multivariate normal distribution
    """
    try:
        if len(theta_best) == 1:
            return stats.norm(theta_best[0], np.sqrt(covariance_matrix[0,0]))
        else:
            return stats.multivariate_normal(theta_best, covariance_matrix)
    except:
        warnings.warn("Could not create posterior distribution, covariance may be singular")
        return None


def convergence_diagnostics(history: List[Dict]) -> Dict:
    """
    Analyze optimization convergence from history.
    
    Args:
        history: List of optimization step dictionaries
        
    Returns:
        Dictionary with convergence diagnostics
    """
    if len(history) < 2:
        return {"converged": False, "reason": "Insufficient history"}
    
    # Extract parameter evolution
    theta_evolution = np.array([step["θ̃"] for step in history])
    n_steps, n_params = theta_evolution.shape
    
    # Compute parameter changes
    param_changes = np.diff(theta_evolution, axis=0)
    relative_changes = np.abs(param_changes) / (np.abs(theta_evolution[:-1]) + 1e-12)
    
    # Convergence criteria
    final_change = np.max(relative_changes[-1])
    avg_change_last_10 = np.mean(np.max(relative_changes[-10:], axis=1)) if n_steps > 10 else final_change
    
    # Gradient norm evolution
    if "s̃_post" in history[-1]:
        gradient_norms = [np.linalg.norm(step["s̃_post"]) for step in history if "s̃_post" in step]
        final_gradient_norm = gradient_norms[-1] if gradient_norms else np.inf
    else:
        final_gradient_norm = np.inf
    
    # Convergence assessment
    converged = (final_change < 1e-4 and 
                avg_change_last_10 < 1e-3 and 
                final_gradient_norm < 1e-2)
    
    diagnostics = {
        "converged": converged,
        "n_steps": n_steps,
        "final_parameter_change": final_change,
        "avg_change_last_10": avg_change_last_10,
        "final_gradient_norm": final_gradient_norm,
        "parameter_evolution": theta_evolution,
        "relative_changes": relative_changes
    }
    
    if not converged:
        reasons = []
        if final_change >= 1e-4:
            reasons.append("Large final parameter change")
        if avg_change_last_10 >= 1e-3:
            reasons.append("Parameters still changing")
        if final_gradient_norm >= 1e-2:
            reasons.append("Large gradient norm")
        diagnostics["non_convergence_reasons"] = reasons
    
    return diagnostics


def bias_analysis(theta_true: np.ndarray, 
                 theta_est: np.ndarray, 
                 covariance: np.ndarray,
                 parameter_names: Optional[List[str]] = None) -> Dict:
    """
    Analyze parameter bias and statistical consistency.
    
    Args:
        theta_true: True parameter values
        theta_est: Estimated parameter values
        covariance: Parameter covariance matrix
        parameter_names: Optional parameter names
        
    Returns:
        Dictionary with bias analysis results
    """
    if parameter_names is None:
        parameter_names = [f"θ_{i}" for i in range(len(theta_true))]
    
    # Compute bias statistics
    bias = theta_est - theta_true
    std_devs = np.sqrt(np.diag(covariance))
    normalized_bias = bias / std_devs
    
    # Chi-squared test for overall consistency
    chi2 = bias.T @ np.linalg.pinv(covariance) @ bias
    dof = len(theta_true)
    p_value = 1 - stats.chi2.cdf(chi2, dof)
    
    # Individual parameter consistency
    individual_results = {}
    for i, name in enumerate(parameter_names):
        individual_results[name] = {
            'true_value': theta_true[i],
            'estimated_value': theta_est[i],
            'bias': bias[i],
            'sigma': std_devs[i],
            'bias_in_sigma': normalized_bias[i],
            'significant_bias': np.abs(normalized_bias[i]) > 2.0
        }
    
    analysis = {
        'chi2_statistic': chi2,
        'dof': dof,
        'p_value': p_value,
        'consistent_at_95': p_value > 0.05,
        'max_bias_sigma': np.max(np.abs(normalized_bias)),
        'rms_bias_sigma': np.sqrt(np.mean(normalized_bias**2)),
        'individual_parameters': individual_results
    }
    
    return analysis


def forecast_constraints(fiducial_params: np.ndarray,
                        survey_specs: Dict,
                        param_names: Optional[List[str]] = None) -> Dict:
    """
    Forecast parameter constraints for survey design.
    
    Args:
        fiducial_params: Fiducial parameter values
        survey_specs: Survey specifications (volume, number of spectra, etc.)
        param_names: Parameter names
        
    Returns:
        Dictionary with forecasted constraints
    """
    # This is a placeholder - actual implementation would depend on 
    # Fisher matrix calculation for given survey specifications
    
    n_params = len(fiducial_params)
    if param_names is None:
        param_names = [f"θ_{i}" for i in range(n_params)]
    
    # Placeholder Fisher matrix scaling
    # In reality, this would be computed from survey specifications
    volume_factor = survey_specs.get('volume', 1.0)
    n_spectra_factor = survey_specs.get('n_spectra', 1000)
    
    # Rough scaling: constraints improve as sqrt(volume * n_spectra)
    improvement_factor = np.sqrt(volume_factor * n_spectra_factor / 1000)
    
    # Placeholder constraint estimates (would be computed from actual Fisher matrix)
    relative_errors = np.array([0.1, 0.05, 0.15, 0.08, 0.12])[:n_params] / improvement_factor
    
    forecast = {}
    for i, name in enumerate(param_names):
        forecast[name] = {
            'fiducial_value': fiducial_params[i],
            'relative_error': relative_errors[i],
            'absolute_error': relative_errors[i] * np.abs(fiducial_params[i])
        }
    
    return forecast


def model_comparison(models: Dict[str, Dict], 
                    data_chi2: Dict[str, float],
                    n_data: int) -> Dict:
    """
    Compare different models using information criteria.
    
    Args:
        models: Dictionary of model specifications
        data_chi2: Chi-squared values for each model
        n_data: Number of data points
        
    Returns:
        Dictionary with model comparison results
    """
    results = {}
    
    for model_name, model_spec in models.items():
        n_params = model_spec.get('n_params', 0)
        chi2 = data_chi2[model_name]
        
        # Information criteria
        aic = chi2 + 2 * n_params
        bic = chi2 + n_params * np.log(n_data)
        
        results[model_name] = {
            'chi2': chi2,
            'n_params': n_params,
            'aic': aic,
            'bic': bic,
            'reduced_chi2': chi2 / (n_data - n_params)
        }
    
    # Find best models
    best_aic = min(results.keys(), key=lambda k: results[k]['aic'])
    best_bic = min(results.keys(), key=lambda k: results[k]['bic'])
    
    results['best_aic'] = best_aic
    results['best_bic'] = best_bic
    
    return results


def effective_sample_size(samples: np.ndarray, 
                         autocorr_method: str = 'integrated') -> np.ndarray:
    """
    Compute effective sample size for parameter chains.
    
    Args:
        samples: Parameter samples (n_samples, n_params)
        autocorr_method: Method for autocorrelation computation
        
    Returns:
        Effective sample sizes for each parameter
    """
    n_samples, n_params = samples.shape
    eff_sizes = np.zeros(n_params)
    
    for i in range(n_params):
        chain = samples[:, i]
        
        if autocorr_method == 'integrated':
            # Integrated autocorrelation time
            autocorr = np.correlate(chain - np.mean(chain), 
                                  chain - np.mean(chain), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]
            
            # Find where autocorrelation drops below 1/e
            try:
                cutoff = np.where(autocorr < 1/np.e)[0][0]
                tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
                eff_sizes[i] = n_samples / (2 * tau_int)
            except:
                eff_sizes[i] = n_samples  # Fallback
        else:
            # Simple method - assume uncorrelated
            eff_sizes[i] = n_samples
    
    return eff_sizes