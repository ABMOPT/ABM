import numpy as np
import h5sparse
import h5py
import pickle
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import time

# The following file is used to split the datasets into smaller pieces so each
# worker can read in a part of the data. The file supports the datasets
# rcv1_test, epsilon. You must download the file "rcv1_test.binary" from 
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html.
 
#  The first two datasets should be downloaded from
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html.
# I never got the dataset mnist8m from the above link to work, so instead
# I generated it myself following the instructions at https://leon.bottou.org/projects/infimnist.
# Generating the file was easy, so don't worry. You want to run the command 
# infimnist svm 10000 8109999 > mnist8m-libsvm.txt.

# The form of the objective function in logistic regression we use assumes 
# labels from {0, 1}, so note that the labels are converted to be consistent
# with this convention. 

# You may have to be careful with paths to folders when running this file.

# Many data sets for binary classification: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html


def unpickle(file):
    with open(file, 'rb') as fo:
         dict = pickle.load(fo, encoding='latin1') 
    return dict

""" This function splits a large data set into several smaller data sets. 
    If num_of_workers = 8 the data set is split among 8 files.
"""
def split_data(dataset_name, num_of_workers):
    # Dense and sparse data sets. Epsilon is in fact a dense dataset 
    # but we load it as a sparse and then convert to dense. 
    sparse_data = ["rcv1_test", 'epsilon']
    dense_data = []

    if dataset_name not in sparse_data + dense_data:
        raise Exception("Data set not available.")
                                                             
    # This data set is 3.6 GB, but it is indeed suitable for binary classification. 
    if dataset_name == "epsilon":
        from catboost.datasets import epsilon
        print("Loading data.")
        start = time.time()
        epsilon_train, epsilon_test = epsilon()
        data_epsilon_train = epsilon_train.to_numpy()
        data_epsilon_test = epsilon_test.to_numpy()
        data = np.concatenate((data_epsilon_train[:, 1:], data_epsilon_test[:, 1:]))       # Dense matrix of size 500000 x 2000.
        labels = np.concatenate((data_epsilon_train[:, 0], data_epsilon_test[:, 0]))
        labels[labels == -1] = 0
        print("data: ", data.shape)
    elif dataset_name == "rcv1_test":
        print("Loading rcv1_test")
        data_and_labels = load_svmlight_file("data/rcv1_test.binary")                      # Sparse matrix of size 677399 x 47236
        print("Finished loading rcv1_test")
        data = data_and_labels[0]                                                
        labels = data_and_labels[1] 
        labels[labels == -1] = 0                                                         
   
    # Folder name for storing data for this split.
    folder_name = "data/" + dataset_name + "_num_of_workers=" + str(num_of_workers)
    if not os.path.exists(folder_name):
        print("Creating folder")
        os.mkdir(folder_name)

    # We split the data randomly among the processes so that each has the same 
    # amount of samples.
    num_of_data_points = len(labels)
    shuffled_indices = np.arange(0, num_of_data_points)
    np.random.shuffle(shuffled_indices)
    num_of_data_points_each_worker = int(num_of_data_points/num_of_workers)

    current_index = 0
    for worker in range(1, num_of_workers):
        data_this_worker = \
            data[shuffled_indices[current_index:current_index + num_of_data_points_each_worker]]
        labels_this_worker = \
            labels[shuffled_indices[current_index:current_index + num_of_data_points_each_worker]]
        current_index += num_of_data_points_each_worker
        # Data sets with data matrices in sparse format. 
        if dataset_name in sparse_data:
            with h5sparse.File(folder_name + "/worker=" + str(worker), 'w') as h5f:
                h5f.create_dataset('data_matrix', data=data_this_worker)
                h5f.create_dataset('labels', data=labels_this_worker)
        # Data sets with dense data matrices.
        elif dataset_name in dense_data:
            with h5py.File(folder_name + "/worker=" + str(worker), 'w') as h5f:
                h5f.create_dataset('data_matrix', data=data_this_worker)
                h5f.create_dataset('labels', data=labels_this_worker)


    # Special case last worker
    data_this_worker = data[shuffled_indices[current_index:], :]
    labels_this_worker = labels[shuffled_indices[current_index :]]
    if dataset_name in sparse_data:
            with h5sparse.File(folder_name + "/worker=" + str(num_of_workers), 'w') as h5f:
                h5f.create_dataset('data_matrix', data=data_this_worker)
                h5f.create_dataset('labels', data=labels_this_worker)
    elif dataset_name in dense_data:
        with h5py.File(folder_name + "/worker=" + str(num_of_workers), 'w') as h5f:
            h5f.create_dataset('data_matrix', data=data_this_worker)
            h5f.create_dataset('labels', data=labels_this_worker)

    
if __name__ == "__main__":
    dataset_file_path1 = "rcv1_test"
    dataset_file_path2 = "epsilon" 
    num_of_workers = 9
    split_data(dataset_file_path1, num_of_workers)
    split_data(dataset_file_path2, num_of_workers)