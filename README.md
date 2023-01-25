# ABM
To run the experiments in the paper you must download the datasets and split them according to the number of workers you wish.
This can be done with the following steps.

1. Download the file `rcv1_test.binary.bz2` from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html. Place it in the folder `data`.
2. Run the command `python split_data.py` when standing in the folder root folder. This command first splits the `rcv1_test` dataset into 9 parts. Then it downloads the `epsilon` dataset and splits it into 9 parts. This step can take around 10 minutes. If you wish to use a different number of workers, change the variable 'num_of_workers' in the file split_data.py.

Now the two folders `rcv1_test_num_of_workers=9` and `epsilon_num_of_workers=9` should exist inside the folder `data`.

You are now ready to run the code. Place yourself in the folder `runners` inside `src`. To run the asynchronous bundle method you can use a command of the form 
`mpiexec -n num_of_workers python runnerABM.py dataset max_iter max_time l2 l1 bundle_size scale_coeff_smoothness local_L0 delta track_delta`

`mpiexec -n 10 python runnerABM.py rcv1_test -1 300 0.00002117029 3e-6 10 1 1 1e-7 False`                                    
`mpiexec -n 10 python runnerABM.py epsilon -1 700 0.000002 5e-5 10 1 1 1e-7 False`

The f


`
