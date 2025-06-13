#!/usr/bin/env python3
"""
Main script for P3D MUSE analysis
"""

import os
import sys
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Set CUDA device if available
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Configure JAX
jax.config.update('jax_platform_name', 'gpu')
from jax.lib import xla_bridge
print(f"JAX version: {jax.__version__}")
print(f"Platform: {xla_bridge.get_backend().platform}")

# Local imports
from config_generator import generate_default_config, load_config, save_config
from power_spectrum import setup_power_spectrum
from data_loader import setup_data_loaders
from muse_problem import create_muse_problem
from optimization import MuseOptimizer, CovarianceEstimator, run_full_analysis


def setup_environment():
    """Setup the computational environment"""
    print("Setting up environment...")
    print(f"JAX version: {jax.__version__}")
    print(f"Platform: {xla_bridge.get_backend().platform}")
    print(f"Available devices: {jax.devices()}")
    
    # Enable 64-bit precision if needed
    # jax.config.update("jax_enable_x64", True)


def create_default_config():
    """Create default configuration for P3D analysis"""
    config = {
        # Grid parameters
        'grid': {
            'bs': 150,  # box size in Mpc/h
            'nc': 150,  # number of pixels per side
        },
        
        # Redshift
        'redshift': {
            'z': 2.0
        },
        
        # File paths
        'files': {
            'pkell_file': "pkell_red_CRO.npy",
            'tau_mesh_file': "./tau_mesh_red_CRO512.npy",
            'config_prefix': "V1_DENSE_",
            'config_location': "./configs/"
        },
        
        # Optimization parameters
        'optimization': {
            'k_ind_optim_max': 20,
            'ell_bins': 3,
            'k_bins': 22,
            'maxsteps': 200,
            'nsims': 10,
            'method': "l-bfgs-experimental-do-not-rely-on-this",
            'theta_rtol': 1e-5,
            'z_rtol': 1e-5,
            'theta_tol': 1e-5,
            'z_tol': 1e-6,
            'alpha': 0.7,
            'beta': 0.25,
            'lr_decay_steps': 150,
            'lr_decay_factor': 0.95
        },
        
        # Noise parameters
        'noise': {
            'level': 1.0
        },
        
        # Covariance estimation
        'covariance': {
            'nsims_score': 200,
            'nsims_fisher': 20,
            'implicit_diff_cgtol': 1e-3,
            'finite_diff_step': None  # Will be auto-computed
        },
        
        # Output settings
        'output': {
            'save_results': True,
            'plot_progress': True,
            'save_intermediate': True,
            'output_dir': "./results/",
            'plot_frequency': 6  # Plot every N steps
        },
        
        # Random seed
        'random_seed': 100
    }
    
    return config


def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description='P3D MUSE Analysis')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./results/',
                        help='Output directory for results')
    parser.add_argument('--maxsteps', type=int, default=None,
                        help='Maximum optimization steps')
    parser.add_argument('--nsims', type=int, default=None,
                        help='Number of simulations for MUSE')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode with reduced parameters')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load or create configuration
    if args.config:
        try:
            config = load_config(args.config)
            print(f"Loaded configuration from {args.config}")
        except FileNotFoundError:
            print(f"Configuration file {args.config} not found. Creating default config.")
            config = create_default_config()
            save_config(config, args.config)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    if args.maxsteps:
        config['optimization']['maxsteps'] = args.maxsteps
    if args.nsims:
        config['optimization']['nsims'] = args.nsims
    if args.no_plot:
        config['output']['plot_progress'] = False
    
    # Test mode adjustments
    if args.test_mode:
        print("Running in test mode with reduced parameters...")
        config['optimization']['maxsteps'] = 10
        config['optimization']['nsims'] = 3
        config['covariance']['nsims_score'] = 10
        config['covariance']['nsims_fisher'] = 3
        config['grid']['nc'] = 64  # Smaller grid for faster testing
    
    # Create output directory
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # Save configuration
    config_file = os.path.join(config['output']['output_dir'], 'config.yaml')
    save_config(config, config_file)
    print(f"Configuration saved to {config_file}")
    
    try:
        # Run the analysis
        results = run_p3d_analysis(config)
        
        # Save results
        if config['output']['save_results']:
            save_results(results, config)
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_p3d_analysis(config):
    """
    Run the complete P3D MUSE analysis
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing analysis results
    """
    print("Starting P3D MUSE analysis...")
    start_time = datetime.now()
    
    # Step 1: Setup power spectrum
    print("Setting up power spectrum...")
    power_data = setup_power_spectrum(config)
    
    # Step 2: Load data
    print("Loading data...")
    data_loaders = setup_data_loaders(config)
    
    # Step 3: Create MUSE problem
    print("Creating MUSE problem...")
    problem = create_muse_problem(power_data, data_loaders, config)
    
    # Step 4: Prepare observation data
    print("Preparing observation data...")
    x_data = data_loaders['map_lya_sim']
    
    # Step 5: Get starting point
    start_point = problem.get_starting_point()
    
    # Step 6: Setup random number generator
    rng = jax.random.PRNGKey(config['random_seed'])
    
    # Step 7: Run optimization
    print("Running MUSE optimization...")
    optimizer = MuseOptimizer(problem, config)
    opt_results = optimizer.run_optimization(x_data, start_point, rng)
    
    # Step 8: Estimate covariance
    print("Estimating parameter covariance...")
    covariance_estimator = CovarianceEstimator(problem, config)
    
    # Get final parameters
    final_theta = opt_results['theta_final']
    s_MAP_sims = opt_results['s_MAP_sims']
    z_MAP_sims = opt_results.get('z_MAP_sims', None)
    
    # Compute covariance matrix
    cov_results = covariance_estimator.compute_covariance(
        final_theta, 
        s_MAP_sims=s_MAP_sims, 
        z_MAP_sims=z_MAP_sims, 
        rng=rng
    )
    
    # Combine results
    results = {
        'config': config,
        'power_data': power_data,
        'optimization': opt_results,
        'covariance': cov_results,
        'problem': problem,
        'runtime': datetime.now() - start_time,
        'final_parameters': final_theta,
        'parameter_covariance': cov_results['Sigma'],
        'fiducial_parameters': problem.get_fiducial_parameters()
    }
    
    print(f"Analysis completed in {results['runtime']}")
    
    return results


def save_results(results, config):
    """Save analysis results"""
    output_dir = config['output']['output_dir']
    
    # Save main results
    np.save(os.path.join(output_dir, 'final_parameters.npy'), results['final_parameters'])
    np.save(os.path.join(output_dir, 'parameter_covariance.npy'), results['parameter_covariance'])
    np.save(os.path.join(output_dir, 'fiducial_parameters.npy'), results['fiducial_parameters'])
    
    # Save optimization history
    if 'history' in results['optimization']:
        np.save(os.path.join(output_dir, 'optimization_history.npy'), results['optimization']['history'])
    
    # Save covariance components
    if 'J' in results['covariance']:
        np.save(os.path.join(output_dir, 'score_covariance_J.npy'), results['covariance']['J'])
    if 'H' in results['covariance']:
        np.save(os.path.join(output_dir, 'fisher_matrix_H.npy'), results['covariance']['H'])
    
    # Create summary plot
    if config['output']['plot_progress']:
        create_summary_plot(results, output_dir)
    
    print(f"Results saved to {output_dir}")


def create_summary_plot(results, output_dir):
    """Create summary plots of the analysis results"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Parameter evolution
        if 'history' in results['optimization']:
            # This would plot the optimization history
            pass
        
        # Plot 2: Final vs fiducial parameters
        final_params = results['final_parameters']
        fiducial_params = results['fiducial_parameters']
        
        axes[0, 1].plot(fiducial_params, 'k:', label='Fiducial')
        axes[0, 1].plot(final_params, 'r.-', label='Final')
        axes[0, 1].set_yscale('symlog')
        axes[0, 1].set_xlabel('Parameter index')
        axes[0, 1].set_ylabel('Parameter value')
        axes[0, 1].legend()
        axes[0, 1].set_title('Final vs Fiducial Parameters')
        
        # Plot 3: Relative differences
        rel_diff = (final_params - fiducial_params) / fiducial_params
        axes[1, 0].plot(rel_diff, 'k.-')
        axes[1, 0].set_xlabel('Parameter index')
        axes[1, 0].set_ylabel('Relative difference')
        axes[1, 0].set_title('Relative Parameter Differences')
        axes[1, 0].set_ylim([-0.5, 0.5])
        
        # Plot 4: Parameter uncertainties
        if 'Sigma' in results['covariance']:
            uncertainties = np.sqrt(np.diag(results['covariance']['Sigma']))
            axes[1, 1].plot(uncertainties / np.abs(final_params), 'b.-')
            axes[1, 1].set_xlabel('Parameter index')
            axes[1, 1].set_ylabel('Relative uncertainty')
            axes[1, 1].set_title('Parameter Uncertainties')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'summary_plot.png'), dpi=150)
        plt.close()
        
        print(f"Summary plot saved to {output_dir}/summary_plot.png")
        
    except Exception as e:
        print(f"Warning: Could not create summary plot: {e}")


if __name__ == "__main__":
    main()