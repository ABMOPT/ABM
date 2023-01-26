import numpy as np
import numpy.linalg as LA
import time
from .subproblem_solver import proj_acc_grad_method, prox_ell1

""" Represents the central server for the asynchronous bundle method.

    For initialization the following parameters are required:
    1. bundle_size    - self-explanatory.
    2. dim            - dimension of the decision variable you are optimizing over.
    3. num_of_workers - The total number of workers is number of processes - 1.
    4. delta          - Tolerance in subproblem.
    5. scale_coeff_smoothness - scale coefficient for the estimate of the local
                                smoothness. 
    6. local_L0       - Initial smoothness estimate. Not important.
    7. settings_subproblem_solver - a dict with settings for AGD for solving
       the subproblem. Must specify max_iter, backtracking parameter eta,
       start value for L backtracking, and if the accuracy should be tracked.
       Tracking the accuracy takes more  because an additional gradient of the 
       objective function in the subproblem must be evaluated.                             
"""
class ParameterServerABM(object):
    def __init__(self, bundle_size, dim, num_of_workers, delta = 1e-5, ell1=-1,
                 scale_coeff_smoothness = 1, local_L0 = 1,
                 track_delta = True, 
                 settings_subproblem_solver = {"subprob_ell1_max_iter": 500, "eta": 1.5,
                 "L0_backtracking": 1e-5}):

        self.dim, self.num_of_workers = dim, num_of_workers
        self.bundle_size, self.iter = bundle_size, 0 
        self.scale_coeff_smoothness, self.ell1 = scale_coeff_smoothness, ell1
        self.track_delta = track_delta
        self.delta = delta
        self.settings_subproblem_solver = settings_subproblem_solver

        # Parameter for visualizing the effect of delta. Should probably be 
        # removed later.
        self.save_all_deltas_interval = 10

        # Array that stores the estimate of each worker's smoothness constant. 
        self.local_L = local_L0*np.ones(num_of_workers) if isinstance(local_L0, int) else local_L0
        
        # The central server stores the most recent gradient for every worker.
        # Say that there are 9 workers. G will then be a list containing
        # 9 arrays. The inner array for each worker stores the most recent gradients
        # for the worker.
        self.G = []
        for i in range(num_of_workers): self.G.append(np.zeros((dim, bundle_size)))
        
        # For each worker the central server stores the most recent iterate 
        # in which the worker has sent information to the central server.
        self.most_recent_iterates = np.zeros((dim, num_of_workers))
        
        # For each worker the central server stores the most recent query point.
        # Note that the central has not received information from the most recent
        # query point has. This quantity is used in "update_gradient_information",
        # and it should probably be updated after that function? Yes, should
        # probably be updated after update_weights since we then query the worker
        # in the newly computed weights.
        self.current_query_all_workers = np.zeros((dim, num_of_workers))

        # The number of cuts per worker is limited. When the number
        # of cuts for a worker reaches its limit, a cut must be removed. 
        # The following container is used to keep track of which cut 
        # should be removed next.
        self.index_to_replace = np.zeros((num_of_workers), dtype=int)

        # To simplify the implementation the central server keeps track of the
        # current bundle size for each oracle.
        self.current_bundle_size_each_worker = np.zeros((num_of_workers), dtype=int)

        # Container for storing the quantity grad f_i (z) @ z - f_i(z).
        # This quantity is used in the dual subproblem. This notation appears
        # in Lemma A.2 in the paper.
        self.v = np.zeros((bundle_size, num_of_workers))        
        
        # Quantities for tracking the performance.
        self.all_iterates = []
        self.all_objective_values = []
        self.all_solver_times = []
        self.all_setup_times = []
        self.all_subproblems_iter = []
        self.all_deltas = []
        self.all_worker_sets = []
        self.total_run_time = []
        self.all_deltas_several_subproblems = []
                
    def initialize_time(self):
        self.start_time = time.time()

    def return_information_upon_termination(self):
        info = {"objs": np.array(self.all_objective_values),
                "total_run_time": np.array(self.total_run_time),
                "subproblem_solver_time": np.array(self.all_solver_times),
                "subproblem_setup_time": np.array(self.all_setup_times),
                "all_subproblems_iter": np.array(self.all_subproblems_iter),
                "deltas": np.array(self.all_deltas),
                "worker_sets": self.all_worker_sets}
        return info

    def update_gradient_information(self, received_gradient, received_obj, worker_id):
        # received_gradient and received_obj have been queried in the point 
            
        # Store gradient in the bundle for the worker.
        self.G[worker_id-1][:, self.index_to_replace[worker_id-1]] = received_gradient
        self.current_bundle_size_each_worker[worker_id-1] = \
                   np.min((self.bundle_size, self.current_bundle_size_each_worker[worker_id-1] + 1 ))

        # Update quantity that appears in dual subproblem.
        self.v[self.index_to_replace[worker_id-1], worker_id-1] = \
         received_gradient @ self.current_query_all_workers[:, worker_id-1] - received_obj

        # Update local smoothness estimate.
        if self.current_bundle_size_each_worker[worker_id-1] >= 3:    
            second_latest_grad_index = (self.index_to_replace[worker_id-1] - 1) % self.bundle_size
            second_latest_grad = self.G[worker_id-1][:, second_latest_grad_index]
            self.local_L[worker_id-1] = \
             (LA.norm(received_gradient - second_latest_grad)/
              LA.norm(self.current_query_all_workers[:, worker_id-1] -
                      self.most_recent_iterates[:, worker_id-1]))
            
        # Update most recent iterate for worker.
        self.most_recent_iterates[:, worker_id-1] = self.current_query_all_workers[:, worker_id-1] 

        # Update index to replace.
        self.index_to_replace[worker_id-1] = (self.index_to_replace[worker_id-1] + 1) % self.bundle_size

    def solve_ell1_subproblem_dual_formulation(self, z_bar, L):
        start1 = time.time()
        current_total_bundle_size = np.sum(self.current_bundle_size_each_worker)
        all_G_flattened = np.empty((self.dim, current_total_bundle_size))
        all_v_flattened = np.empty((current_total_bundle_size))

        # Need to be extra careful (in the beginning) because the bundle may not be full.
        # NOTE Can be refactored to handle the case when the bundle is full more efficiently. 
        counter = 0
        lambda0 = np.zeros(current_total_bundle_size)
        for i in range(self.num_of_workers):
            all_G_flattened[:, counter:counter+self.current_bundle_size_each_worker[i]] = \
                self.G[i][:, 0:self.current_bundle_size_each_worker[i]]
            all_v_flattened[counter:counter+self.current_bundle_size_each_worker[i]] = \
                 self.v[0:self.current_bundle_size_each_worker[i], i]
            # Even initialization.
            #lambda0[counter:counter + self.current_bundle_size_each_worker[i]] = \
            # 1/self.current_bundle_size_each_worker[i]
            #counter += self.current_bundle_size_each_worker[i]
            
            # Initialize at the most recent gradient for each worker.
            index_most_recent = (self.index_to_replace[i] - 1) % self.bundle_size
            lambda0[counter + index_most_recent] = 1
            counter += self.current_bundle_size_each_worker[i]
           
        end1 = time.time()
        setup_time = end1 - start1
        lambda_opt, solve_time, grad, empirical_deltas, iter = \
            proj_acc_grad_method(all_v_flattened, all_G_flattened, z_bar, L, 
            self.current_bundle_size_each_worker, self.ell1, lambda0, self.delta,
            self.track_delta, self.settings_subproblem_solver)

        if self.iter % self.save_all_deltas_interval == 0 and self.track_delta:
            self.all_deltas_several_subproblems.append(empirical_deltas)
            
        # Compute achieved accuracy.
        empirical_delta = -1
        if self.track_delta:
            counter = 0
            for i in range(self.num_of_workers):
                empirical_delta += np.max(-grad[counter:counter+self.current_bundle_size_each_worker[i]])
                counter += self.current_bundle_size_each_worker[i]        
            empirical_delta += grad @ lambda_opt
            self.all_deltas.append(empirical_delta)
      
            #if LA.norm(lambda0 - lambda_opt) > 1e-5:
            #     print("LA.norm(lambda0 - lambda_opt): ", LA.norm(lambda0 - lambda_opt))
        
        x_plus = prox_ell1(z_bar - 1/L*all_G_flattened @ lambda_opt, self.ell1/L)
    
        return x_plus, setup_time, solve_time, iter, empirical_delta
 
    def update_weights(self, weights, worker_set):
        self.all_iterates.append(weights)
        self.all_worker_sets.append(worker_set) 
        
        # The following two lines are not important.
        if self.iter >= 5: factor = self.scale_coeff_smoothness
        else: factor = 2

        L = factor*np.sum(self.local_L)
        z_bar = 1/L*np.sum(factor*self.local_L*self.most_recent_iterates, axis = 1)
        new_weights, setup_time, solver_time, num_iter, emp_delta = \
        self.solve_ell1_subproblem_dual_formulation(z_bar, L)  
     
        for worker in worker_set:
            self.current_query_all_workers[:, worker-1] = new_weights

        self.all_setup_times.append(setup_time)
        self.all_solver_times.append(solver_time)
        self.all_subproblems_iter.append(num_iter)
        self.all_deltas.append(emp_delta)
        self.total_run_time.append(time.time() - self.start_time)
        self.iter += 1
        return new_weights

