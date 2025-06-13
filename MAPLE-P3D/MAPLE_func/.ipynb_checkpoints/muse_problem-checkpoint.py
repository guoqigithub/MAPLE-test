"""
MUSE problem definition for P3D analysis
Defines the inference problem using the MUSE framework
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from muse_inference.jax import JaxMuseProblem
from scipy.special import legendre
from jax.scipy.ndimage import map_coordinates


class P3DMuseProblem(JaxMuseProblem):
    """P3D-specific MUSE problem implementation"""
    
    def __init__(self, power_data, obs_data, lya_data, config, **kwargs):
        """
        Initialize P3D MUSE problem
        
        Args:
            power_data: Dictionary containing power spectrum data and functions
            obs_data: Dictionary containing observation data (naa, kernel, skewers)
            lya_data: Dictionary containing Lyman-alpha data
            config: Configuration dictionary
        """
        super().__init__(**kwargs)
        
        # Store data
        self.power_data = power_data
        self.obs_data = obs_data
        self.lya_data = lya_data
        self.config = config
        
        # Extract key parameters from config
        self.nc = config['simulation']['n_cells']
        self.bs = config['simulation']['box_size']
        self.noise_level = config['optimization']['noise_level']
        self.k_ind_optim_max = config['fiducial_model']['k_ind_optim_max']
        self.ell_bins = config['fiducial_model']['ell_bins']
        
        # Setup power spectrum function
        self._setup_power_function()
        
        # Setup fiducial parameters
        self._setup_fiducial_parameters()
        
        # Precompute CIC readout function
        self.cic_readout = self._create_cic_readout()
    
    def _setup_power_function(self):
        """Setup power spectrum interpolation function"""
        k = self.power_data['k']
        k_in = self.power_data['k_in']
        tff = self.power_data['tff']
        kmu = self.power_data['kmu']
        
        def power_b(theta):
            """Power spectrum function that can be optimized"""
            tff_updated = tff.at[:, :self.k_ind_optim_max].set(
                theta.reshape(self.ell_bins, self.k_ind_optim_max)
            )
            
            # Interpolate each multipole
            func1 = map_coordinates(tff_updated[0], np.array([k_in]), mode="nearest", order=1)
            func1 = func1.reshape(k.shape[0], k.shape[1], k.shape[2])
            
            func2 = map_coordinates(tff_updated[1], np.array([k_in]), mode="nearest", order=1)
            func2 = func2.reshape(k.shape[0], k.shape[1], k.shape[2])
            
            func4 = map_coordinates(tff_updated[2], np.array([k_in]), mode="nearest", order=1)
            func4 = func4.reshape(k.shape[0], k.shape[1], k.shape[2])
            
            # Combine using Legendre polynomials
            func = jax.nn.relu(
                func1 * legendre(0)(kmu) + 
                func2 * legendre(2)(kmu) + 
                func4 * legendre(4)(kmu)
            )
            return func
        
        self.power_b = power_b
    
    def _setup_fiducial_parameters(self):
        """Setup fiducial parameters for optimization"""
        m_array = self.power_data['m_array']
        k_bins = self.config['fiducial_model']['k_bins']
        
        # Reshape fiducial parameters
        theta_fid = m_array[:, 2].reshape(self.ell_bins, k_bins)[:, :self.k_ind_optim_max]
        self.tf_cut_flat = theta_fid.flatten()
    
    def _create_cic_readout(self):
        """Create optimized CIC readout function"""
        @jit
        def cic_readout_jit_jnc(mesh, naa, kernel):
            """Highly optimized CIC readout"""
            meshvals = mesh.flatten()[naa].reshape(-1, 8).T
            weightedvals = meshvals.T * kernel[0]
            values = jnp.sum(weightedvals, axis=-1)
            return values
        
        return cic_readout_jit_jnc
    
    def gen_map_lya(self, theta, z):
        """Generate Lyman-alpha map from parameters and latent field"""
        # Reshape latent field to 3D grid
        modes = z[:self.nc**3].reshape((self.nc, self.nc, self.nc))
        
        # Get power spectrum for current parameters
        Plin = self.power_b(theta)
        
        # Apply power spectrum in Fourier space
        conv_field = jnp.fft.rfftn(modes).conj() * Plin**(1/2)
        lin_modes_real = jnp.fft.irfftn(conv_field).T[:, :, :]
        
        # Convert to flux field
        flux_mean = self.lya_data['flux_mean']
        flux_real = (lin_modes_real + 1) * flux_mean
        
        # Apply CIC readout to get observed values
        naa = self.obs_data['naa']
        kernel = self.obs_data['kernel']
        lya_values = self.cic_readout(flux_real, naa, kernel)
        
        return lya_values
    
    def sample_x_z(self, key, θ):
        """Sample data and latent variables"""
        keys = jax.random.split(key, 2)
        
        # Sample latent field
        z = jax.random.normal(keys[0], (self.nc * self.nc * self.nc,))
        
        # Generate noiseless observation
        x = self.gen_map_lya(θ, z)
        
        # Add observation noise
        skewers_skn = self.obs_data['skewers_skn']
        noise = (self.noise_level * skewers_skn) * jax.random.normal(keys[1], (self.obs_data['kernel'].shape[1],))
        x_hat = x + noise
        
        return (x_hat, z)
    
    def logLike(self, x, z, θ):
        """Log-likelihood function"""
        # Generate predicted observation
        x_pred = self.gen_map_lya(θ, z)
        
        # Compute likelihood
        skewers_skn = self.obs_data['skewers_skn']
        data_term = jnp.sum((x - x_pred)**2 / ((self.noise_level * skewers_skn)**2))
        prior_term = jnp.sum(z**2.0)
        
        return -(data_term + prior_term)
    
    def logPrior(self, θ):
        """Log-prior function"""
        prior_mean = self.tf_cut_flat * 1.2
        prior_std = self.tf_cut_flat * 0.4
        
        return -jnp.sum(((θ - prior_mean)**2 / (2 * prior_std**2)))
    
    def get_fiducial_parameters(self):
        """Get fiducial parameters"""
        return self.tf_cut_flat.copy()
    
    def get_starting_point(self, perturbation_scale=0.01):
        """Get starting point for optimization"""
        start_point = self.tf_cut_flat * 1.2
        if perturbation_scale > 0:
            key = jax.random.PRNGKey(42)
            perturbation = jax.random.normal(key, start_point.shape) * self.tf_cut_flat * perturbation_scale
            start_point = start_point + perturbation
        return start_point
    
    def z_MAP_guess_from_truth(self, x, z, θ):
        """Generate initial guess for z MAP estimation"""
        return jnp.zeros_like(z)


def create_muse_problem(power_data, data_loaders, config):
    """
    Create MUSE problem instance
    
    Args:
        power_data: Dictionary containing power spectrum data
        data_loaders: Dictionary containing observation and simulation data
        config: Configuration dictionary
    
    Returns:
        P3DMuseProblem instance
    """
    # Extract data components
    obs_data = {
        'naa': data_loaders['observation']['naa'],
        'kernel': data_loaders['observation']['kernel'],
        'skewers_skn': data_loaders['observation']['skewers_skn'],
        'skewers_dla': data_loaders['observation']['skewers_dla'],
        'skewers_fin': data_loaders['observation']['skewers_fin']
    }
    
    lya_data = {
        'flux_sim': data_loaders['lya']['map_lya_sim'],
        'flux_mean': data_loaders['lya']['flux_mean']
    }
    
    # Create problem instance
    problem = P3DMuseProblem(
        power_data=power_data,
        obs_data=obs_data,
        lya_data=lya_data,
        config=config,
        implicit_diff=True,
        jit=True
    )
    
    return problem


def create_test_problem(config):
    """
    Create a test problem for debugging/validation
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Simple test problem instance
    """
    # This would create simplified synthetic data for testing
    # Implementation depends on your testing needs
    pass