"""
Configuration generator for P3D analysis
Generates YAML configuration files with default parameters
"""

import yaml
import numpy as np
from pathlib import Path

def generate_default_config():
    """Generate default configuration dictionary"""
    config = {
        'simulation': {
            'redshift': 2.0,
            'box_size': 150,  # Mpc/h
            'n_cells': 150,   # number of pixels per side
            'cuda_device': "1"
        },
        
        'fiducial_model': {
            'pkell_file': "../pkell_red_CRO.npy",
            'tau_mesh_file': "../tau_mesh_red_CRO512.npy",
            'k_bins': 22,
            'k_ind_optim_max': 20,
            'ell_bins': 3
        },
        
        'data_files': {
            'prefix': "V1_DENSE_",
            'location': "../configs/",
            'naa_file': "naa.npy",
            'kernel_file': "kernel.npy",
            'skewers_skn_file': "skewers_skn.npy",
            'skewers_dla_file': "skewers_dla.npy",
            'skewers_fin_file': "skewers_fin.npy"
        },
        
        'optimization': {
            'method': "l-bfgs-experimental-do-not-rely-on-this",
            'max_steps': 100, #200,
            'n_sims': 10,
            'theta_rtol': 1e-5,
            'z_rtol': 1e-5,
            'theta_tol': 1e-5,
            'z_tol': 1e-6,
            'alpha': 0.7,
            'beta': 0.25,
            'lr_decay_step': 150,
            'lr_decay_factor': 0.95,
            'noise_level': 1.0
        },
        
        'fisher': {
            'n_sims_fisher': 20,
            'n_sims_hessian': 20,
            'implicit_diff_cgtol': 1e-3,
            'finite_diff_step': None,  # Will be computed automatically
            'use_median': False
        },
        
        'prior': {
            'theta_scale': 1.2,
            'theta_width': 0.4
        },
        
        'output': {
            'save_history': True,
            'save_map_history': False,
            'plot_interval': 3,
            'detailed_plot_interval': 6,
            'output_dir': "../results/"
        },
        
        'random_seed': 100
    }
    
    return config

def save_config(config, filename="p3d_config.yaml"):
    """Save configuration to YAML file"""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to {output_path}")
    return output_path

def load_config(filename="p3d_config.yaml"):
    """Load configuration from YAML file"""
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

def update_config(config, updates):
    """Update configuration with new values"""
    def update_nested_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    return update_nested_dict(config, updates)

def validate_config(config):
    """Basic validation of configuration parameters"""
    required_sections = ['simulation', 'fiducial_model', 'data_files', 'optimization']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Check if files exist
    data_files = config['data_files']
    prefix = data_files['prefix']
    location = data_files['location']
    
    files_to_check = [
        f"{location}{prefix}{data_files['naa_file']}",
        f"{location}{prefix}{data_files['kernel_file']}",
        f"{location}{prefix}{data_files['skewers_skn_file']}",
        config['fiducial_model']['pkell_file'],
        config['fiducial_model']['tau_mesh_file']
    ]
    
    missing_files = []
    for file_path in files_to_check:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("Warning: The following files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    
    return len(missing_files) == 0

if __name__ == "__main__":
    # Generate and save default configuration
    config = generate_default_config()
    config_path = save_config(config)
    
    # Example of how to customize configuration
    custom_updates = {
        'simulation': {
            'redshift': 2.5,
            'box_size': 200
        },
        'optimization': {
            'max_steps': 300,
            'n_sims': 15
        }
    }
    
    custom_config = update_config(config, custom_updates)
    save_config(custom_config, "p3d_config_custom.yaml")
    
    print("Default and custom configuration files generated!")