"""
Utility functions for P3D cosmological analysis pipeline
Handles I/O, plotting, helpers, and general utilities
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
import yaml
import logging
from datetime import datetime
from pathlib import Path


def setup_jax_gpu(device_id=1):
    """
    Setup JAX with GPU configuration
    
    Args:
        device_id: CUDA device ID to use
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    jax.config.update('jax_platform_name', 'gpu')
    
    from jax.lib import xla_bridge
    print(f"JAX version: {jax.__version__}")
    print(f"Backend platform: {xla_bridge.get_backend().platform}")
    
    return jax.random.PRNGKey(100)


def setup_logging(level='INFO', log_file=None):
    """
    Setup logging configuration
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional file to write logs to
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, level),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=getattr(logging, level), format=log_format)
    
    return logging.getLogger(__name__)


def load_fiducial_model(filename):
    """
    Load fiducial power spectrum data
    
    Args:
        filename: Path to .npy file containing power spectrum data
        
    Returns:
        m_array: Array with columns [k, l, P(k,l)]
        kmax: Maximum k value
    """
    m_array = np.load(filename)
    # Filter for l <= 4 (only monopole, quadrupole, hexadecapole)
    m_array = jnp.array(m_array[np.where(m_array[:,1] <= 4)])
    kmax = m_array[-1, 0]
    
    print(f"Loaded fiducial model: shape={m_array.shape}, kmax={kmax:.3f}")
    return m_array, kmax


def load_simulation_data(tau_mesh_file):
    """
    Load simulation data (tau mesh)
    
    Args:
        tau_mesh_file: Path to tau mesh file
        
    Returns:
        flux_sim: Flux field from simulation
        delta_flux_sim: Delta flux field
        mean_flux: Mean flux value
    """
    lin_modes_sim = np.load(tau_mesh_file)
    flux_sim = np.exp(-lin_modes_sim)
    mean_flux = np.mean(flux_sim)
    delta_flux_sim = flux_sim / mean_flux - 1
    
    print(f"Loaded simulation data: shape={flux_sim.shape}, mean_flux={mean_flux:.6f}")
    return flux_sim, delta_flux_sim, mean_flux


def load_observation_config(config_dir, prefix):
    """
    Load observational configuration files
    
    Args:
        config_dir: Directory containing config files
        prefix: Prefix for config files (e.g., "V1_DENSE_")
        
    Returns:
        Dictionary containing loaded arrays
    """
    config = {}
    files = ['naa.npy', 'kernel.npy', 'skewers_skn.npy', 
             'skewers_dla.npy', 'skewers_fin.npy']
    
    for file in files:
        filepath = os.path.join(config_dir, prefix + file)
        if os.path.exists(filepath):
            config[file.replace('.npy', '')] = np.load(filepath)
            print(f"Loaded {file}: shape={config[file.replace('.npy', '')].shape}")
        else:
            print(f"Warning: {filepath} not found")
    
    return config


def save_results(results, filename, format='pickle'):
    """
    Save optimization results to file
    
    Args:
        results: Dictionary containing results
        filename: Output filename
        format: 'pickle' or 'npz'
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
    elif format == 'npz':
        # Convert JAX arrays to numpy for saving
        np_results = {}
        for key, value in results.items():
            if hasattr(value, 'shape'):  # Array-like
                np_results[key] = np.array(value)
            else:
                np_results[key] = value
        np.savez_compressed(filename, **np_results)
    
    print(f"Results saved to {filename}")


def load_results(filename):
    """
    Load previous results
    
    Args:
        filename: Path to results file
        
    Returns:
        results: Loaded results dictionary
    """
    if filename.endswith('.pkl'):
        with open(filename, 'rb') as f:
            results = pickle.load(f)
    elif filename.endswith('.npz'):
        data = np.load(filename, allow_pickle=True)
        results = {key: data[key] for key in data.files}
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    print(f"Results loaded from {filename}")
    return results


def validate_inputs(config):
    """
    Validate input parameters and configurations
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if valid, raises ValueError if not
    """
    required_keys = ['box_size', 'mesh_shape', 'redshift']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate mesh shape
    if len(config['mesh_shape']) != 3:
        raise ValueError("mesh_shape must be 3D")
    
    # Validate box size
    if len(config['box_size']) != 3:
        raise ValueError("box_size must be 3D")
    
    # Check if mesh is cubic (recommended)
    if len(set(config['mesh_shape'])) > 1:
        print("Warning: Non-cubic mesh detected")
    
    print("Input validation passed")
    return True


def create_output_directory(base_dir="results", run_name=None):
    """
    Create timestamped output directory
    
    Args:
        base_dir: Base directory for outputs
        run_name: Optional run name
        
    Returns:
        output_dir: Path to created directory
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    output_dir = Path(base_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    return output_dir


def plot_optimization_progress(history, output_dir=None, show=True):
    """
    Plot convergence history
    
    Args:
        history: List of optimization history dictionaries
        output_dir: Directory to save plots
        show: Whether to display plots
    """
    if not history:
        print("No history to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Parameter evolution
    steps = range(1, len(history) + 1)
    theta_history = [h['θ̃'] for h in history]
    theta_array = np.array([np.array(theta).flatten() for theta in theta_history])
    
    axes[0, 0].plot(steps, theta_array)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Parameter Values')
    axes[0, 0].set_title('Parameter Evolution')
    axes[0, 0].set_yscale('symlog')
    
    # Learning rate / step size
    if 'α' in history[0]:
        alphas = [h.get('α', 0.7) for h in history]
        axes[0, 1].plot(steps, alphas)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Learning Rate α')
        axes[0, 1].set_title('Learning Rate Evolution')
    
    # Gradient norms
    if 's̃_post' in history[0]:
        grad_norms = [np.linalg.norm(np.array(h['s̃_post']).flatten()) for h in history]
        axes[1, 0].semilogy(steps, grad_norms)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('||Gradient||')
        axes[1, 0].set_title('Gradient Norm')
    
    # Computation time
    if 't' in history[0]:
        times = [h['t'].total_seconds() for h in history]
        axes[1, 1].plot(steps, np.cumsum(times))
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Cumulative Time (s)')
        axes[1, 1].set_title('Computation Time')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'optimization_progress.png', dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_parameter_comparison(theta_true, theta_estimated, errors=None, 
                            param_names=None, output_dir=None, show=True):
    """
    Generate comparison plots between true and estimated parameters
    
    Args:
        theta_true: True parameter values
        theta_estimated: Estimated parameter values  
        errors: Parameter uncertainties
        param_names: Names for parameters
        output_dir: Directory to save plots
        show: Whether to display plots
    """
    theta_true = np.array(theta_true).flatten()
    theta_estimated = np.array(theta_estimated).flatten()
    
    if param_names is None:
        param_names = [f'θ_{i}' for i in range(len(theta_true))]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Direct comparison
    axes[0].plot(theta_true, 'r.-', label='True', markersize=8)
    axes[0].plot(theta_estimated, 'b-', label='Estimated', markersize=6)
    
    if errors is not None:
        errors = np.array(errors).flatten()
        axes[0].fill_between(range(len(theta_estimated)), 
                           theta_estimated - errors,
                           theta_estimated + errors, 
                           alpha=0.3, color='blue')
    
    axes[0].set_xlabel('Parameter Index')
    axes[0].set_ylabel('Parameter Value')
    axes[0].set_title('Parameter Comparison')
    axes[0].legend()
    axes[0].set_yscale('symlog')
    
    # Fractional difference
    frac_diff = (theta_estimated - theta_true) / theta_true
    axes[1].plot(frac_diff, 'k-', marker='o', markersize=6)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Parameter Index')
    axes[1].set_ylabel('(Est - True) / True')
    axes[1].set_title('Fractional Difference')
    axes[1].set_ylim([-0.3, 0.3])
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'parameter_comparison.png', dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_covariance_matrix(covariance, param_names=None, output_dir=None, show=True):
    """
    Plot parameter covariance matrix
    
    Args:
        covariance: Covariance matrix
        param_names: Parameter names for labels
        output_dir: Directory to save plots
        show: Whether to display plots
    """
    correlation = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Covariance matrix
    im1 = axes[0].imshow(covariance, cmap='RdBu', origin='lower')
    axes[0].set_title('Covariance Matrix')
    plt.colorbar(im1, ax=axes[0])
    
    # Correlation matrix
    im2 = axes[1].imshow(correlation, cmap='RdBu', origin='lower', vmin=-1, vmax=1)
    axes[1].set_title('Correlation Matrix')
    plt.colorbar(im2, ax=axes[1])
    
    if param_names:
        for ax in axes:
            ax.set_xticks(range(len(param_names)))
            ax.set_yticks(range(len(param_names)))
            ax.set_xticklabels(param_names, rotation=45)
            ax.set_yticklabels(param_names)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(output_dir / 'covariance_matrix.png', dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def print_summary_statistics(theta_final, covariance, theta_true=None):
    """
    Print summary of final results
    
    Args:
        theta_final: Final parameter estimates
        covariance: Parameter covariance matrix
        theta_true: True parameter values (optional)
    """
    theta_final = np.array(theta_final).flatten()
    errors = np.sqrt(np.diag(covariance))
    
    print("\n" + "="*60)
    print("FINAL PARAMETER ESTIMATES")
    print("="*60)
    
    for i, (val, err) in enumerate(zip(theta_final, errors)):
        line = f"θ_{i:2d} = {val:12.6e} ± {err:12.6e}"
        if theta_true is not None:
            true_val = np.array(theta_true).flatten()[i]
            frac_diff = (val - true_val) / true_val
            line += f" (true: {true_val:12.6e}, diff: {frac_diff:+6.1%})"
        print(line)
    
    print("\nCovariance matrix condition number:", np.linalg.cond(covariance))
    print("Parameter correlations (max):", np.max(np.abs(
        covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance))) - np.eye(len(theta_final))
    )))


def export_to_yaml(results, filename):
    """
    Export results to YAML format
    
    Args:
        results: Results dictionary
        filename: Output YAML filename
    """
    # Convert numpy arrays to lists for YAML serialization
    yaml_results = {}
    for key, value in results.items():
        if hasattr(value, 'tolist'):  # numpy array
            yaml_results[key] = value.tolist()
        elif hasattr(value, '__array__'):  # JAX array
            yaml_results[key] = np.array(value).tolist()
        else:
            yaml_results[key] = value
    
    with open(filename, 'w') as f:
        yaml.dump(yaml_results, f, default_flow_style=False, indent=2)
    
    print(f"Results exported to YAML: {filename}")


def memory_usage_monitor():
    """
    Monitor memory usage (requires psutil)
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
        return memory_info.rss / 1024 / 1024
    except ImportError:
        print("psutil not available for memory monitoring")
        return None