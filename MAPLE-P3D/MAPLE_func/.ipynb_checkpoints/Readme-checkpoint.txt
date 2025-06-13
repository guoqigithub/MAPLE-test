Core Files
config_generator.py

generate_default_config()
save_config_to_yaml()
load_config_from_yaml()
validate_config()

power_spectrum.py

rfftnfreq_2d() (from helper_functions)
setup_kvectors()
load_fiducial_model()
power_b()
interpolate_power_spectrum()

data_loader.py

load_observational_data()
load_naa_kernel_skewers()
load_simulation_data()
preprocess_data()

muse_problem.py

Jax3DMuseProblem_flat (class)
sample_x_z()
logLike()
logPrior()

optimization.py

run_muse_optimization()
setup_optimization_parameters()
convergence_check()

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