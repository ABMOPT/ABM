import numpy as np
import time

""" Represents the central server for the delay-tolerant proximal gradient method
    proposed in https://proceedings.mlr.press/v80/mishchenko18a.html.

    For initialization the following parameters are required:
    1. num_of_workers - The total number of workers is number of processes - 1.
"""
class DAVERPG_central_server(object):
    def __init__(self, num_of_workers):

        # Quantities required for the implementation.
        self.num_of_workers = num_of_workers                                   
        
        # Quantities required for tracking the progress upon termination.
        self.iter = 0
        self.all_objective_values = []
        self.all_iterates = []
        self.total_run_time = [0]

    def initialize_time(self):
        self.start_time = time.time()

    def return_information_upon_termination(self):
        info = {"objs": np.array(self.all_objective_values),
                "total_run_time": np.array(self.total_run_time)}
        return info
        
    def update_weights(self, weights, displacement):
        self.all_iterates.append(weights)
        weights = weights + displacement
        self.total_run_time.append(time.time() - self.start_time)
        self.iter += 1
        return weights
        
        
        
    
    