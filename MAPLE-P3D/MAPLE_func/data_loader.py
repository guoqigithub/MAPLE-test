"""
Data loading utilities for P3D analysis
Handles loading of observation data and simulation data
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path

def load_observation_data(config):
    """Load observation data files"""
    data_config = config['data_files']
    prefix = data_config['prefix']
    location = data_config['location']
    
    # Construct file paths
    files = {
        'naa': f"{location}{prefix}{data_config['naa_file']}",
        'kernel': f"{location}{prefix}{data_config['kernel_file']}",
        'skewers_skn': f"{location}{prefix}{data_config['skewers_skn_file']}",
        'skewers_dla': f"{location}{prefix}{data_config['skewers_dla_file']}",
        'skewers_fin': f"{location}{prefix}{data_config['skewers_fin_file']}"
    }
    
    # Load data
    data = {}
    for key, filepath in files.items():
        if Path(filepath).exists():
            data[key] = np.load(filepath)
            print(f"Loaded {key}: {data[key].shape}")
        else:
            print(f"Warning: File not found: {filepath}")
            data[key] = None
    
    return data

def load_simulation_data(config):
    """Load simulation data"""
    model_config = config['fiducial_model']
    
    # Load tau mesh
    tau_mesh_file = model_config['tau_mesh_file']
    if Path(tau_mesh_file).exists():
        lin_modes_sim = np.load(tau_mesh_file)
        flux_sim = np.exp(-lin_modes_sim)
        delta_flux_sim = flux_sim / np.mean(flux_sim) - 1
        
        print(f"Loaded simulation data: {lin_modes_sim.shape}")
        
        return {
            'lin_modes_sim': lin_modes_sim,
            'flux_sim': flux_sim,
            'delta_flux_sim': delta_flux_sim
        }
    else:
        print(f"Warning: Simulation file not found: {tau_mesh_file}")
        return None

def prepare_lya_map(sim_data, obs_data, config):
    """Prepare Lyman-alpha map from simulation and observation data"""
    if sim_data is None or obs_data['naa'] is None or obs_data['kernel'] is None:
        print("Warning: Missing required data for Lya map preparation")
        return None
    
    from jax import jit
    
    @jit
    def cic_readout_jit_jnc(mesh, naa, kernel):
        """Highly optimized CIC readout"""
        meshvals = mesh.flatten()[naa].reshape(-1, 8).T
        weightedvals = meshvals.T * kernel[0]
        values = np.sum(weightedvals, axis=-1)
        return values
    
    # Prepare Lya map from simulation
    flux_sim = sim_data['flux_sim']
    naa = obs_data['naa']
    kernel = obs_data['kernel']
    skewers_skn = obs_data['skewers_skn']
    
    map_lya_sim = cic_readout_jit_jnc(flux_sim, naa, kernel)
    
    # Add noise
    import jax
    key = jax.random.PRNGKey(config['random_seed'])
    keys = jax.random.split(key, 2)
    noise_level = config['optimization']['noise_level']
    
    if skewers_skn is not None:
        noise = (noise_level * skewers_skn) * jax.random.normal(keys[1], (kernel.shape[1],))
        map_lya_sim += noise
    
    return {
        'map_lya_sim': map_lya_sim,
        'cic_readout_func': cic_readout_jit_jnc,
        'flux_mean': np.mean(flux_sim)
    }

def setup_data_loaders(config):
    """Main setup function for data loading"""
    # Load observation data
    obs_data = load_observation_data(config)
    
    # Load simulation data  
    sim_data = load_simulation_data(config)
    
    # Prepare Lya map
    lya_data = prepare_lya_map(sim_data, obs_data, config)
    
    return {
        'observation': obs_data,
        'simulation': sim_data,
        'lya': lya_data
    }

def validate_data(data):
    """Validate loaded data"""
    issues = []
    
    # Check observation data
    obs_data = data['observation']
    required_obs = ['naa', 'kernel', 'skewers_skn']
    
    for key in required_obs:
        if obs_data[key] is None:
            issues.append(f"Missing observation data: {key}")
    
    # Check simulation data
    if data['simulation'] is None:
        issues.append("Missing simulation data")
    
    # Check Lya data
    if data['lya'] is None:
        issues.append("Failed to prepare Lya map")
    
    if issues:
        print("Data validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("Data validation passed!")
    return True