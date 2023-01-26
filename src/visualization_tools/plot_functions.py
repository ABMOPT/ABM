from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np

# Plots the optimality gap vs. run time.
def plot_opt_gap_vs_runtime(names_opt_vals_dict, filepath_fig, methods):

    # Initialize figure with 2 rows and len(names_opt_vals_dict) columns.
    plt.figure(figsize = (20,4))
    gs1 = gridspec.GridSpec(1, len(names_opt_vals_dict))
    gs1.update(wspace=0.20, hspace=0.37) # set the spacing between axes. 

    DAVERPG_step_sizes = {"epsilon": 11.36, "MNIST8m": 0.106, "rcv1_test": 142.85}

    for method in methods:
        counter = 0
        for dataset_name, opt_val in names_opt_vals_dict.items():
            if method == "ABM":
                data_file_path = f"../../experiment_data/ABM/ABM_{dataset_name}_bundle_size=10_delta=1e-07.pickle"
            elif method == "DAVERPG":
                data_file_path = f"../../experiment_data/DAVERPG/theoretical_step_size/DAVERPG_{dataset_name}_step_size={DAVERPG_step_sizes[dataset_name]}.pickle"
            else:
                print("Not implemented method")
            with open(data_file_path, 'rb') as handle:
                data = pickle.load(handle) 
                objs = data['objs']
                total_run_time  = data['total_run_time']
                ax1 = plt.subplot(gs1[counter])
                plt.axis('on')
                label = method
                if label == "DAVERPG":
                    label = "DAve-RPG"
                ax1.semilogy(total_run_time, objs - opt_val + 1e-16, label = label )
                ax1.set_xlabel('Runtime (s)')
                ax1.legend()
                if counter == 0:
                    ax1.set_ylabel(r'$f(x_k) - f^*$', fontsize=12)
            counter += 1
   
    plt.savefig(filepath_fig, bbox_inches='tight') 
 



def plot_opt_gap_vs_runtime_and_num_of_grads(names_opt_vals_dict, filepath_fig, methods):

    # Initialize figure with 2 rows and len(names_opt_vals_dict) columns.
    plt.figure(figsize = (20,8))
    gs1 = gridspec.GridSpec(2, len(names_opt_vals_dict))
    gs1.update(wspace=0.15, hspace=0.19) # set the spacing between axes. 

    DAVERPG_step_sizes = {"epsilon": 11.36, "MNIST8m": 0.106, "rcv1_test": 142.85}

    for method in methods:
        counter = 0
        for dataset_name, opt_val in names_opt_vals_dict.items():
            if method == "ABM":
                data_file_path = f"../../experiment_data/ABM/ABM_{dataset_name}_bundle_size=10_delta=1e-07_run2.pickle"
            elif method == "DAVERPG":
                data_file_path = f"../../experiment_data/DAVERPG/DAVERPG_{dataset_name}_step_size={DAVERPG_step_sizes[dataset_name]}.pickle"
            with open(data_file_path, 'rb') as handle:
                data = pickle.load(handle)     
                objs = data['objs']
                total_run_time  = data['total_run_time']
                # Plot progress vs. communicated gradients.
                num_of_communicated_gradients = np.zeros(len(objs))
                if method == "DAVERPG":
                    num_of_communicated_gradients = np.arange(len(objs))
                else:
                    all_worker_sets = data['worker_sets']
                    for i in range(1, len(objs)):
                        num_of_communicated_gradients[i] = \
                            num_of_communicated_gradients[i-1] + len(all_worker_sets[i-1])
                    
                    ax1 = plt.subplot(gs1[counter])
                    plt.axis('on')
                    label = method
                    if label == "DAVERPG":
                     label = "DAve-RPG"
                    print(f"Method/num_of_communicated_gradients: {method} / {num_of_communicated_gradients[-1]}")
                    ax1.semilogy(num_of_communicated_gradients, objs - opt_val + 1e-16, label = label)
                    ax1.set_xlabel('Number of gradients')
                    ax1.legend()
                    if counter == 0:
                        ax1.set_ylabel(r'$f(x_k) - f^*$', fontsize=12)


                # Plot progress vs. runtime.
                ax1 = plt.subplot(gs1[counter + len(names_opt_vals_dict)])
                plt.axis('on')
                if label == "DAVERPG":
                     label = "DAve-RPG"
                ax1.semilogy(total_run_time, objs - opt_val + 1e-16, label = label )
                ax1.set_xlabel('Runtime (s)')
                ax1.legend()
                if counter == 0:
                    ax1.set_ylabel(r'$f(x_k) - f^*$', fontsize=12)
            counter += 1
   
    plt.savefig(filepath_fig, bbox_inches='tight') 

def plot_sensitivity_analysis_bundle_size_and_delta(names_opt_vals_dict, bundle_sizes, deltas, filepath_fig):
    
    colors = ['green', 'royalblue', 'darkorange']
    plt.figure(figsize = (20,8))
    gs1 = gridspec.GridSpec(2, len(names_opt_vals_dict))
    gs1.update(wspace=0.15, hspace=0.13) # set the spacing between axes. 
    counter = 0
    color_counter = 0
    for dataset_name, opt_val in names_opt_vals_dict.items(): 
        for bundle_size in bundle_sizes:
            with open(f"../../experiment_data/ABM/ABM_{dataset_name}_bundle_size={bundle_size}_delta=1e-07.pickle", 'rb') as handle:
                data = pickle.load(handle) 
                objs = data['objs']
                total_run_time  = data['total_run_time']            
                ax1 = plt.subplot(gs1[counter])
                plt.axis('on')
                ax1.semilogy(total_run_time, objs - opt_val + 1e-16, label = f"$m={bundle_size}$", color = colors[color_counter])
                ax1.legend()

            color_counter += 1
        color_counter = 0
        if counter == 0:
            ax1.set_ylabel(r'$f(x_k) - f^*$', fontsize=12)
        counter += 1


    colors = ['green', 'darkorange', 'royalblue']
    for dataset_name, opt_val in names_opt_vals_dict.items(): 
        for delta in deltas:
            file_name = f"../../experiment_data/ABM/effect_of_delta/ABM_{dataset_name}_bundle_size={10}_delta={delta}.pickle"
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle) 
                objs = data['objs']
                total_run_time  = data['total_run_time']
                ax1 = plt.subplot(gs1[counter] )
                plt.axis('on')
                ax1.semilogy(total_run_time, objs - opt_val + 1e-16, label = r'$\delta$' +  " = " + str(delta), color = colors[color_counter])
                ax1.set_xlabel('Runtime (s)')
                ax1.legend()
            color_counter += 1
        color_counter = 0
        if counter == 3:
            ax1.set_ylabel(r'$f(x_k) - f^*$', fontsize=12)
        counter += 1

    plt.savefig(filepath_fig, bbox_inches='tight') 

