"""
Field generation and Fourier space utilities for P3D analysis
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
from scipy.special import legendre
from jax.scipy.ndimage import map_coordinates


def setup_fourier_grid(mesh_shape, box_size):
    """
    Setup k-vectors and related arrays for Fourier space operations
    
    Args:
        mesh_shape: tuple of (nc, nc, nc) - grid dimensions
        box_size: tuple of (bs, bs, bs) - box size in Mpc/h
        
    Returns:
        dict containing k-vectors and derived arrays
    """
    nc = mesh_shape[0]
    bs = box_size[0]
    ptcl_spacing = bs / nc
    
    # Import helper functions - you'll need to implement this
    from helper_functions import rfftnfreq_2d
    
    kvec = rfftnfreq_2d(mesh_shape, ptcl_spacing)
    k = jnp.sqrt(sum(k**2 for k in kvec))
    
    # LOS (z) is along the 0 axis
    kz = jnp.ones(k.shape) * kvec[0]**2
    kx = jnp.ones(k.shape) * (kvec[1]**2 + kvec[2]**2)
    
    # Add tiny deltas to avoid div by zero errors
    kk = (kx + kz) + 1e-8
    kmu = jnp.sqrt(kz / (k**2 + 1e-5))
    
    return {
        'kvec': kvec,
        'k': k,
        'kz': kz,
        'kx': kx,
        'kk': kk,
        'kmu': kmu,
        'ptcl_spacing': ptcl_spacing
    }


def setup_power_interpolation(m_array, k_grid, kmax):
    """
    Setup power spectrum interpolation functions for l=0,2,4
    
    Args:
        m_array: power spectrum multipoles array
        k_grid: k-vector grid from setup_fourier_grid
        kmax: maximum k value for interpolation
        
    Returns:
        dict containing interpolated power spectrum functions
    """
    k = k_grid['k']
    
    # Scale k to match the Pk in simulation
    k_in = (k.flatten() / kmax * 22 - 0.05)
    
    # Extract l=0 multipole
    l0 = np.where(m_array[:, 1] == 0)[0]
    Pk_l0 = m_array[l0, 2]
    func1 = map_coordinates(Pk_l0, np.array([k_in]), mode="nearest", order=1)
    func1 = func1.reshape(k.shape)
    
    # Extract l=2 multipole
    l2 = np.where(m_array[:, 1] == 2)[0]
    Pk_l2 = m_array[l2, 2]
    func2 = map_coordinates(Pk_l2, np.array([k_in]), mode="nearest", order=1)
    func2 = func2.reshape(k.shape)
    
    # Extract l=4 multipole
    l4 = np.where(m_array[:, 1] == 4)[0]
    Pk_l4 = m_array[l4, 2]
    func4 = map_coordinates(Pk_l4, np.array([k_in]), mode="nearest", order=1)
    func4 = func4.reshape(k.shape)
    
    return {
        'k_in': k_in,
        'func1': func1,
        'func2': func2,
        'func4': func4,
        'k_shape': k.shape
    }


def power_multipole_expansion(theta, m_array, k_grid, k_ind_optim_max=20):
    """
    Compute full P(k,μ) from multipole expansion with updated parameters
    
    Args:
        theta: parameter vector to optimize
        m_array: base power spectrum array
        k_grid: k-vector grid
        k_ind_optim_max: number of k-bins to optimize
        
    Returns:
        P(k,μ) array
    """
    k = k_grid['k']
    kmu = k_grid['kmu']
    
    # Setup interpolation arrays
    ell_bins = 3
    k_bins = 22
    kmax = m_array[-1, 0]
    
    tff = m_array[:, 2].reshape(ell_bins, k_bins)
    tff = tff.at[:, :k_ind_optim_max].set(theta.reshape(ell_bins, k_ind_optim_max))
    
    k_in = (k.flatten() / kmax * 22 - 0.05)
    
    # Interpolate each multipole
    func1 = map_coordinates(tff[0], np.array([k_in]), mode="nearest", order=1)
    func1 = func1.reshape(k.shape)
    
    func2 = map_coordinates(tff[1], np.array([k_in]), mode="nearest", order=1)
    func2 = func2.reshape(k.shape)
    
    func4 = map_coordinates(tff[2], np.array([k_in]), mode="nearest", order=1)
    func4 = func4.reshape(k.shape)
    
    # Combine using Legendre polynomials
    power_func = jax.nn.relu(
        func1 * legendre(0)(kmu) + 
        func2 * legendre(2)(kmu) + 
        func4 * legendre(4)(kmu)
    )
    
    return power_func


def generate_lya_field(theta, z_modes, m_array, k_grid, mean_flux, mesh_shape):
    """
    Generate Lyman-alpha field from linear modes and power spectrum
    
    Args:
        theta: cosmological parameters
        z_modes: random Gaussian field in k-space
        m_array: power spectrum multipoles
        k_grid: Fourier grid setup
        mean_flux: mean flux level
        mesh_shape: grid dimensions
        
    Returns:
        3D Lyman-alpha flux field
    """
    nc = mesh_shape[0]
    
    # Reshape modes to 3D grid
    modes = z_modes[:nc**3].reshape(mesh_shape)
    
    # Get power spectrum for these parameters
    Plin = power_multipole_expansion(theta, m_array, k_grid)
    
    # Apply power spectrum in Fourier space
    conv_field = jnp.fft.rfftn(modes).conj() * Plin**(1/2)
    
    # Transform back to real space
    lin_modes_real = jnp.fft.irfftn(conv_field).T
    
    # Convert to flux field
    flux_real = (lin_modes_real + 1) * mean_flux
    
    return flux_real


@jit
def cic_readout(field_3d, naa, kernel):
    """
    Cloud-in-Cell readout for extracting skewer data from 3D field
    
    Args:
        field_3d: 3D field to sample from
        naa: precomputed neighbor indices
        kernel: precomputed CIC weights
        
    Returns:
        1D skewer values
    """
    # Optimized CIC interpolation
    meshvals = field_3d.flatten()[naa].reshape(-1, 8).T
    weightedvals = meshvals.T * kernel[0]
    values = jnp.sum(weightedvals, axis=-1)
    
    return values


def apply_noise_and_systematics(clean_data, skewers_skn, noise_level, rng_key):
    """
    Apply observational noise and systematic effects
    
    Args:
        clean_data: clean skewer data
        skewers_skn: noise model per skewer
        noise_level: overall noise scaling
        rng_key: JAX random key
        
    Returns:
        noisy observed data
    """
    import jax
    
    noise = (noise_level * skewers_skn) * jax.random.normal(rng_key, clean_data.shape)
    noisy_data = clean_data + noise
    
    return noisy_data


def load_precomputed_arrays(config):
    """
    Load precomputed arrays needed for field generation
    
    Args:
        config: configuration dictionary
        
    Returns:
        dict containing loaded arrays
    """
    # Load fiducial power spectrum
    m_array = np.load(config['power_spectrum_file'])
    m_array = m_array[np.where(m_array[:, 1] <= 4)]
    
    # Load CIC interpolation setup
    naa = np.load(config['naa_file'])
    kernel = np.load(config['kernel_file'])
    
    # Load noise models
    skewers_skn = np.load(config['skewers_skn_file'])
    skewers_dla = np.load(config['skewers_dla_file'])
    skewers_fin = np.load(config['skewers_fin_file'])
    
    # Load simulation data if available
    if 'simulation_file' in config:
        lin_modes_sim = np.load(config['simulation_file'])
        flux_sim = np.exp(-lin_modes_sim)
        mean_flux = np.mean(flux_sim)
    else:
        mean_flux = config.get('mean_flux', 1.0)
    
    return {
        'm_array': m_array,
        'naa': naa,
        'kernel': kernel,
        'skewers_skn': skewers_skn,
        'skewers_dla': skewers_dla,
        'skewers_fin': skewers_fin,
        'mean_flux': mean_flux
    }


def generate_mock_observation(theta, field_arrays, k_grid, mesh_shape, rng_key, noise_level=1.0):
    """
    Generate a complete mock observation including noise
    
    Args:
        theta: cosmological parameters
        field_arrays: precomputed arrays from load_precomputed_arrays
        k_grid: Fourier grid setup
        mesh_shape: grid dimensions
        rng_key: JAX random key
        noise_level: noise scaling factor
        
    Returns:
        mock observed skewer data
    """
    import jax
    
    nc = mesh_shape[0]
    keys = jax.random.split(rng_key, 2)
    
    # Generate random field
    z_modes = jax.random.normal(keys[0], (nc**3,))
    
    # Generate clean Lyman-alpha field
    flux_field = generate_lya_field(
        theta, z_modes, field_arrays['m_array'], 
        k_grid, field_arrays['mean_flux'], mesh_shape
    )
    
    # Extract skewer data
    clean_skewers = cic_readout(flux_field, field_arrays['naa'], field_arrays['kernel'])
    
    # Add noise
    noisy_skewers = apply_noise_and_systematics(
        clean_skewers, field_arrays['skewers_skn'], noise_level, keys[1]
    )
    
    return noisy_skewers, z_modes