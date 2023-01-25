# ABM
To run the experiments in the paper you must download the datasets and split them according to the number of workers you wish.
This can be done with the following steps.

1. Download the file `rcv1_test.binary.bz2` from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html. Place it in the folder `data`.
2. Run the command `python split_data.py` when standing in the folder root folder. This command first splits the rcv1_test dataset into 9 parts. Then it downloads the `epsilon` dataset and splits it into 9 parts. This step can take around 10 minutes. If you wish to use a different number of workers, change the variable 'num_of_workers' in the file split_data.py.

Now the two folders `rcv1_test_num_of_workers=9` and `epsilon_num_of_workers=9` should exist inside the folder `data`.

`
