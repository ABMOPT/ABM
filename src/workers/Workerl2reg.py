import h5sparse
import h5py
import numpy as np
from .loss_functions import log_reg_ell2_distributed

""" Represents the worker for logistic regression with ell2-regularization.
    The parameters required for initialization are self-explanatory.    
"""
class Workerl2reg(object):
    def __init__(self, rank_id, name_dataset, num_of_workers,
                 total_num_of_data_points_all_workers, l2):
        
        # Dense data matrices.
        if name_dataset in ["epsilon"]:
            h5_data = h5py.File(f"../../data/{name_dataset}_num_of_workers={num_of_workers}/worker={rank_id}", 'r')
            data = h5_data['data_matrix'][0:, :]
            #print("Denseness of data: ", np.count_nonzero(data)/data.shape[0]/data.shape[1])
        elif name_dataset in ["rcv1_test", "MNIST8m"]:
            h5_data = h5sparse.File(f"../../data/{name_dataset}_num_of_workers={num_of_workers}/worker={rank_id}", 'r')
            data = h5_data['data_matrix'][:]
            #print("Denseness of data: ", data.nnz/data.shape[0]/data.shape[1])
      
        labels = h5_data['labels'][:]
        
        # Initialize object representing the objective function
        self.obj_function = log_reg_ell2_distributed(data, labels, l2, total_num_of_data_points_all_workers)
        print(f"Worker {rank_id} initialized. Number of data points: {len(labels)}.")

    # Returns a numpy array.  
    def compute_gradient(self, weights):
        grad = self.obj_function.grad(weights)
        if(np.isnan(grad).any() or np.max(np.abs(grad)) > 10*10):
           raise Exception("NAN or INF in compute_gradient")
        return grad

    # Returns a scalar
    def compute_obj(self, weights):
        if(np.isnan(weights).any()):
            raise Exception("NAN in weights in compute_obj")
        obj = self.obj_function.loss(weights)
        if(np.isnan(obj).any()):
            raise Exception("NAN in compute_obj")
        return obj
    
    # Returns a tuple: (grad, obj)
    def compute_grad_and_obj(self, weights):
        grad, obj = self.obj_function.first_order_oracle(weights)
        if(np.isnan(grad).any()):
           raise Exception("NAN in gradient in compute_grad_and_obj")
        if(np.isnan(obj).any()):
            raise Exception("NAN in objective compute_grad_and_obj")
        return grad, obj

