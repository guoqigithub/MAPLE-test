Core Files
config_generator.py
def generate_default_config():
    """Generate default configuration dictionary"""

def save_config(config, filename="p3d_config.yaml"):
    """Save configuration to YAML file"""

def load_config(filename="p3d_config.yaml"):
    """Load configuration from YAML file"""

def update_config(config, updates):
    """Update configuration with new values"""

def validate_config(config):
    """Basic validation of configuration parameters"""



power_spectrum.py
def setup_kvectors(ptcl_grid_shape, ptcl_spacing):
    """Setup k-vectors for Fourier space operations"""

def load_fiducial_model(pkell_file):
    """Load fiducial power spectrum model"""

def setup_power_interpolation(m_array, k, kmax, k_ind_optim_max=20):
    """Setup power spectrum interpolation functions"""

def create_power_function(k, k_in, tff, k_ind_optim_max=20):
    """Create power spectrum function that can be optimized"""

def setup_power_spectrum(config):
    """Main setup function for power spectrum operations"""



data_loader.py
def load_observation_data(config):
    """Load observation data files"""

def load_simulation_data(config):
    """Load simulation data"""

def prepare_lya_map(sim_data, obs_data, config):
    """Prepare Lyman-alpha map from simulation and observation data"""
    def cic_readout_jit_jnc(mesh, naa, kernel):
        """Highly optimized CIC readout"""

def setup_data_loaders(config):
    """Main setup function for data loading"""

def validate_data(data):
    """Validate loaded data"""



muse_problem.py

"""
MUSE problem definition for P3D analysis
Defines the inference problem using the MUSE framework
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from muse_inference.jax import JaxMuseProblem

class P3DMuseProblem(JaxMuseProblem):
    """P3D-specific MUSE problem implementation"""
    
    def __init__(self, power_data, obs_data, lya_data, config, **kwargs):
    
    def gen_map_lya(self, theta, z):
        """Generate Lyman-alpha map from parameters and latent field"""
    
    def sample_x_z(self, key, θ):
        """Sample data and latent variables"""
    
    def logLike(self, x, z, θ):
        """Log-likelihood function"""
    
    def logPrior(self, θ):
        """Log-prior function"""
    
    def get_fiducial_parameters(self):
        """Get fiducial parameters"""
    
    def get_starting_point(self, perturbation_scale=0.01):
        """Get starting point for optimization"""

def create_muse_problem(power_data, data_loaders, config):
    """Create MUSE problem instance"""



optimization.py

class MuseOptimizer:
    """
    Main optimization class for MUSE inference
    """
    
    def __init__(self, problem, config):
        
    def _split_rng(self, rng, N):
        """Split random key into N subkeys"""
    
    def _gradθ_hessθ_logPrior(self, θ):
        """Compute gradient and Hessian of log prior"""
    
    def _get_MAPs(self, x_z, θ, method, z_tol, θ_tol):
        """Get MAP estimates for given data"""
    
    def run_optimization(self, x_data, start_point, rng, maxsteps=200, nsims=10):
        """
        Main optimization loop
        """

    def _plot_progress(self, start_point, θ, θL):
        """Plot optimization progress"""

class CovarianceEstimator:
    """
    Estimates covariance matrix using Fisher information
    """
    
    def __init__(self, problem, config):
        
    def _split_rng(self, rng, N):
        """Split random key into N subkeys"""
    
    def get_score_covariance(self, θ, s_MAP_sims, rng, nsims=200):
        """
        Compute score covariance matrix J
        """
        def get_s_MAP(rng_key):
    
    def pjacobian(self, f, x, step):
        """Parallel Jacobian computation"""
    
    def _get_H_i(self, rng, z_MAP, θ, implicit_diff_cgtol=1e-3, method=None, θ_tol=None, z_tol=None, step=None):
        """
        Compute Hessian for single simulation using implicit differentiation
        """
    
    def get_fisher_matrix(self, θ, z_MAP_sims, rng, nsims=20):
        """
        Compute Fisher information matrix H
        """
    
    def compute_covariance(self, θ, J, H, s_MAP_sims=None, z_MAP_sims=None, rng=None):
        """
        Compute final parameter covariance matrix
        """

def run_full_analysis(problem, config, x_data, start_point, rng):
    """
    Run complete MUSE analysis pipeline
    """





Additional Required Files
field_utils.py

cic_readout_jit_jnc()
gen_map_lya()
generate_linear_field()
apply_power_spectrum_convolution()
flux_to_lya_conversion()

gradients.py

get_MAPs()
get_J()
_get_H_i()
_get_H_i_old()
pjacobian()
gradθ_hessθ_logPrior()
compute_muse_gradient()
compute_hessian_approximations()

statistics.py

compute_fisher_matrix()
compute_covariance_matrix()
build_posterior_distribution()
compute_confidence_intervals()
parameter_marginalization()

utils.py

_split_rng()
ravel_θ() / unravel_θ()
ravel_z() / unravel_z()
setup_jax_environment()
plot_convergence()
plot_parameter_evolution()
save_results()
load_results()
z_MAP_guess_from_truth()