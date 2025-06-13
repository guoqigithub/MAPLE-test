"""
Power spectrum utilities for P3D analysis
Handles k-space operations and power spectrum interpolation
"""

import numpy as np
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from scipy.special import legendre
import jax

def setup_kvectors(ptcl_grid_shape, ptcl_spacing):
    """Setup k-vectors for Fourier space operations"""
    from helper_functions import rfftnfreq_2d
    
    kvec = rfftnfreq_2d(ptcl_grid_shape, ptcl_spacing)
    k = jnp.sqrt(sum(k**2 for k in kvec))
    
    # LOS (z) is along the 0 axis
    kz = jnp.ones(k.shape) * kvec[0]**2
    kx = jnp.ones(k.shape) * (kvec[1]**2 + kvec[2]**2)
    
    # Add tiny deltas to avoid div by zero errors
    kk = (kx + kz) + 1e-8
    kmu = jnp.sqrt(kz / (k**2 + 1e-5))
    
    return kvec, k, kz, kx, kk, kmu

def load_fiducial_model(pkell_file):
    """Load fiducial power spectrum model"""
    m_array = np.load(pkell_file)
    m_array = jnp.array(m_array[np.where(m_array[:, 1] <= 4)])
    kmax = m_array[-1, 0]
    
    print(f"Loaded model array shape: {m_array.shape}, kmax: {kmax}")
    return m_array, kmax

def setup_power_interpolation(m_array, k, kmax, k_ind_optim_max=20):
    """Setup power spectrum interpolation functions"""
    k_bins = 22
    ell_bins = 3
    
    # Scale k to match the Pk in simulation
    k_in = (k.flatten() / kmax * 22 - 0.05)
    
    # Extract different multipole components
    l0 = np.where(m_array[:, 1] == 0)[0]
    Pk_l0 = m_array[l0, 2]
    
    l2 = np.where(m_array[:, 1] == 2)[0]
    Pk_l2 = m_array[l2, 2]
    
    l4 = np.where(m_array[:, 1] == 4)[0]
    Pk_l4 = m_array[l4, 2]
    
    # Reshape for optimization
    tff = m_array[:, 2].reshape(ell_bins, k_bins)
    theta_fid = m_array[:, 2].reshape(ell_bins, k_bins)[:, :k_ind_optim_max]
    
    return k_in, tff, theta_fid, (Pk_l0, Pk_l2, Pk_l4)

def create_power_function(k, k_in, tff, k_ind_optim_max=20):
    """Create power spectrum function that can be optimized"""
    ell_bins, k_bins = tff.shape
    
    def power_b(theta, tff=tff):
        tff_updated = tff.at[:, :k_ind_optim_max].set(
            theta.reshape(ell_bins, k_ind_optim_max)
        )
        
        func1 = map_coordinates(tff_updated[0], np.array([k_in]), 
                              mode="nearest", order=1)
        func1 = func1.reshape(k.shape[0], k.shape[1], k.shape[2])
        
        func2 = map_coordinates(tff_updated[1], np.array([k_in]), 
                              mode="nearest", order=1)
        func2 = func2.reshape(k.shape[0], k.shape[1], k.shape[2])
        
        func4 = map_coordinates(tff_updated[2], np.array([k_in]), 
                              mode="nearest", order=1)
        func4 = func4.reshape(k.shape[0], k.shape[1], k.shape[2])
        
        # Combine multipoles using Legendre polynomials
        kmu = jnp.sqrt(k**2 / (k**2 + 1e-5))  # Recompute kmu here
        # func = jax.nn.relu(
        #     func1 * legendre(0)(kmu) + 
        #     func2 * legendre(2)(kmu) + 
        #     func4 * legendre(4)(kmu)
        # )
        P0 = jnp.ones_like(kmu)
        P2 = (3 * kmu**2 - 1) / 2
        P4 = (35 * kmu**4 - 30 * kmu**2 + 3) / 8
        func = jax.nn.relu(func1 * P0 + func2 * P2 + func4 * P4)
        
        return func
    
    return power_b

def setup_power_spectrum(config):
    """Main setup function for power spectrum operations"""
    sim_config = config['simulation']
    model_config = config['fiducial_model']
    
    # Setup basic parameters
    bs = sim_config['box_size']
    nc = sim_config['n_cells']
    
    ptcl_grid_shape = (nc,) * 3
    ptcl_spacing = bs / nc
    
    # Setup k-vectors
    kvec, k, kz, kx, kk, kmu = setup_kvectors(ptcl_grid_shape, ptcl_spacing)
    
    # Load fiducial model
    m_array, kmax = load_fiducial_model(model_config['pkell_file'])
    
    # Setup interpolation
    k_in, tff, theta_fid, multipoles = setup_power_interpolation(
        m_array, k, kmax, model_config['k_ind_optim_max']
    )
    
    # Create power function
    power_b = create_power_function(k, k_in, tff, model_config['k_ind_optim_max'])
    
    return {
        'kvec': kvec,
        'k': k,
        'kz': kz, 
        'kx': kx,
        'kk': kk,
        'kmu': kmu,
        'k_in': k_in,
        'tff': tff,
        'theta_fid': theta_fid,
        'power_b': power_b,
        'm_array': m_array,
        'kmax': kmax,
        'multipoles': multipoles
    }