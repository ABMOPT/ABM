import h5sparse
import h5py
import numpy as np
from .loss_functions import log_reg_ell2_distributed

def DAVE_RPG_prox_ell1(x, gamma):
    return np.maximum(np.abs(x) - gamma, 0)*np.sign(x)

""" Represents the worker for DAVERPG, logistic regression with elastic net regularization.    
    The following parameters are required for initialization:
    
    1. step_size - Used step size. Denoted by gamma in [KFJM18].
    2. ell1, ell2 - regularization parameters for the 1- resp. 2-norm.
                    Note that these values correspond to the "total"
                    regularization parameter.
    
    The other parameters are self-explanatory.
"""
class DAVERPG(object):
    def __init__(self, step_size, ell1, rank_id, name_dataset, num_of_workers,
                 total_num_of_data_points_all_workers, ell2, initial_weights):
        

        # Dense data matrices.
        if name_dataset in ["madelon", "mushrooms", "epsilon"]:
            h5_data = h5py.File(f"../../data/{name_dataset}_num_of_workers={num_of_workers}/worker={rank_id}", 'r')
            data = h5_data['data_matrix'][0:, :]
        elif name_dataset in ["covtype_scaled", "rcv1_test", "MNIST8m"]:
            h5_data = h5sparse.File(f"../../data/{name_dataset}_num_of_workers={num_of_workers}/worker={rank_id}", 'r')
            data = h5_data['data_matrix'][:]
        
        labels = h5_data['labels'][:]

        # Initialize object representing the objective function. This objective
        # function is 1/N*(empirical loss over local data set), where N is the number of TOTAL data points.
        # This is an important detail when implementing DAVERPG. 
        self.obj_function = log_reg_ell2_distributed(data, labels, ell2, total_num_of_data_points_all_workers)
        print(f"Worker {rank_id} initialized. Number of data points: {len(labels)}.")

        # Compute local smoothness parameter.
        #if name_dataset in ["epsilon"]:
        #    print(f"Computing L at worker {rank_id}")
        #    true_L = 0.25/data.shape[0]*np.max(np.linalg.eigvalsh(data.T @ data)) + ell2 
        #    print(f"True L worker {rank_id} : {true_L}")
        #elif name_dataset in ["rcv1_test", "MNIST8m"]:
        #    print(f"Computing L at worker {rank_id}")
        #    evals = scipy.sparse.linalg.eigsh(data.T @ data, k = 5)[0]
        #    true_L = 0.25/data.shape[0]*np.max(evals) + ell2 
        #    print(f"True L worker {rank_id} : {true_L}")

        self.step_size = step_size
        self.ell1 = ell1

        # The proportion of data points stored at this worker.
        self.proportion = data.shape[0]/total_num_of_data_points_all_workers
        
        # Last computed local iterate.
        self.last_computed_iterate = initial_weights      

    # We always use p = 1 inner prox steps.
    def compute_displacement(self, new_weights):
        z = DAVE_RPG_prox_ell1(new_weights, self.step_size*self.ell1)
        grad = self.obj_function.grad(z)

        # Note that on the next line we should scale the gradient by the proportion
        # of data stored by the worker. The reason is that our oracle that computes
        # the gradient takes a factor 1/total_num_of_data_points into account.
        x_plus = z - self.step_size/self.proportion*grad
        displacement = self.proportion * (x_plus - self.last_computed_iterate)
        self.last_computed_iterate = x_plus

        return displacement

    def compute_obj(self, weights):
        return self.obj_function.loss(weights)
    

    

