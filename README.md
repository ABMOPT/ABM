# ABM
To run the experiments in the paper you must download the datasets and split them according to the number of workers you wish to use.
This can be done with the following steps.

1. Download the file `rcv1_test.binary.bz2` from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html. Place it in the folder `data`.
2. Run the command `python split_data.py` when standing in the root folder. This command first splits the `rcv1_test` dataset into 9 parts. Then it downloads the `epsilon` dataset and splits it into 9 parts. This step can take around 10 minutes. If you wish to use a different number of workers, change the variable `num_of_workers` in the file `split_data.py`.

Now the two folders `rcv1_test_num_of_workers=9` and `epsilon_num_of_workers=9` should exist inside the folder `data`.

You are now ready to run the code. Place yourself in the folder `runners` inside `src`. To run the asynchronous bundle method you can use a command of the form 

`mpiexec -n num_of_workers python runnerABM.py dataset max_iter max_time l2 l1 bundle_size scale_coeff_smoothness local_L0 delta track_delta`

Here the arguments are defined as follows.
* `num_of_workers` - number of workers.
* `dataset` - either epsilon or rcv1_test.
* `max_iter` - maximum number of iterations. If you only want to run for a particular time, set this parameter to -1.
* `max_time` - maximum run time.
* `bundle_size` - bundle size.
* `scale_coeff_smoothness` - this parameter is not mentioned in the paper and should be set to one.
* `local_L0` - initial smoothness estimate in the first iteration (before the adaptive estimation of the smoothness kicks in). Can be set to one. 
* `delta` - master problem tolerance
* `track_delta` - if the tolerance should be tracked (makes the algorithm slightly slower).

To reconstruct the benchmark results in the paper, run 

`mpiexec -n 10 python runnerABM.py rcv1_test -1 300 0.00002117029 3e-6 10 1 1 1e-7 False`                                    
`mpiexec -n 10 python runnerABM.py epsilon -1 700 0.000002 5e-5 10 1 1 1e-7 False`

The following commands run `DAve-RPG`. The last parameter is the step size.

`mpiexec -n 10 python runnerDAVERPG.py rcv1_test -1 300 0.00002117029 3e-6 142.85`                                    
`mpiexec -n 10 python runnerDAVERPG.py epsilon -1 700 0.000002 5e-5 11.36`

To plot the results, run the file  `run_plot_functions.py` inside the folder `visualization_tools`.

`
