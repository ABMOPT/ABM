from plot_functions import plot_sensitivity_analysis_bundle_size_and_delta, plot_opt_gap_vs_runtime, plot_opt_gap_vs_runtime_and_num_of_grads



opt_vals = {'MNIST8m': 0.3164476555988142, 'rcv1_test': 0.17663386332097009, 'epsilon': 0.35153871880845594}
all_bundle_sizes = [2, 5, 10]
all_deltas = [1e-05, 1e-07, 1e-09]
delta = 1e-07
bundle_size = 10
methods = ["ABM", "DAVERPG"]




# Suboptimality vs time.
fig1_benchmarking = "figures/benchmarking1.png"
plot_opt_gap_vs_runtime(opt_vals, fig1_benchmarking, methods)

fig2_benchmarking = "figures/benchmarking2.eps"
plot_opt_gap_vs_runtime_and_num_of_grads(opt_vals, fig2_benchmarking, methods)


# ABM's performance with different parameter values.
#fig_sensitivity_analysis = "figures/ABM_parameters.png"
#plot_sensitivity_analysis_bundle_size_and_delta(opt_vals, all_bundle_sizes, all_deltas, fig_sensitivity_analysis)





