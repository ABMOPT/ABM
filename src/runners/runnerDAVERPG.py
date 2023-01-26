from mpi4py import MPI
import numpy as np
from libsvm.svmutil import *
import time
import sys
import pickle
sys.path.append("..")
from central_servers.DAVERPG_central_server import DAVERPG_central_server
from workers.DAVERPG_worker import DAVERPG

""" mpiexec -n 10 python runnerDAVERPG.py rcv1_test -1 100 0.00002117029 3e-6 0.1 

    mpiexec -n 10 python runnerDAVERPG.py epsilon -1 100 0.000002 5e-5 0.1

    mpiexec -n 10 python runnerDAVERPG.py MNIST8m -1 100 6.06468594e-7 3e-3 0.1

    To run the script you must provide the following parameters:
    name_dataset - one of {rcv1_test, epsilon, MNIST8m}
    max_iter     - Number of iterations before terminating. One iteration counts
                   as the interaction between the central server and ONE worker.
                   If the termination should only be based on total run time, set
                   max_iter = -1.
    l2, l1       - Regularization parameters
    step_size    - Used step size.
"""
# Read some parameters from the buffer
name_dataset, max_iter, max_time = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
l2, l1, step_size = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])

# MPI quantities
comm = MPI.COMM_WORLD
rank = comm.Get_rank()            
size = comm.Get_size()       # Total number of processes. The total number
                             # of workers is size - 1.

if name_dataset == "rcv1_test":
    total_num_of_data_points_all_workers, dim = 677399, 47236
elif name_dataset == "epsilon":
    total_num_of_data_points_all_workers, dim = 500000, 2000
elif name_dataset == "MNIST8m":
    total_num_of_data_points_all_workers, dim = 1648890, 784
        
# Quantities all workers and the central server need.
iter_num = 0
num_of_workers = size - 1
weights = np.zeros(dim)       # Initial guess
terminate = False             # If a worker should terminate or not.

# Initialize central server and workers.
if rank == 0: 
    central_server = DAVERPG_central_server(num_of_workers)
    central_server.initialize_time()
else:
    worker = DAVERPG(step_size, l1, rank, name_dataset, num_of_workers, 
                     total_num_of_data_points_all_workers, l2, weights)
    
while True:
    # Each worker computes a displacement. She then sends the gradient to the
    # central server, and waits for a message.
    if rank != 0:
        displacement = worker.compute_displacement(weights) 
        message = [displacement, rank]
        comm.send(message, dest=0)
        while True:
            probe = comm.iprobe()
            if probe:
                message = comm.recv()
                weights, terminate = message[0], message[1]
                break

        if terminate:
            print(f"Worker {rank} finished.")
            sys.stdout.flush()
            break
    
    # The central server receives a displacement from a worker. She updates the
    # weights and returns the new weights to the worker. 
    else:
        probe = comm.iprobe()
        if probe:
            # Receive a displacement from a worker.
            message = comm.recv()
            displacement = message[0]
            worker_id = message[1]  # received from which worker
            weights = central_server.update_weights(weights, displacement)
            iter_num += 1

            # Check termination criteria. max_iter == -1 means that the 
            # termination criteria should only be based on run time.
            current_run_time = central_server.total_run_time[iter_num]
            if ((iter_num >= max_iter - 1 and max_iter != -1) or 
                current_run_time > max_time):
                terminate = True
            # Send the new weights back to the worker.
            msg = [weights, terminate]
            comm.send(msg, dest=worker_id)                

        # Tell each worker to terminate. 
        if terminate:
            for worker_identification in range(1, num_of_workers + 1):
                # Important if-clause to avoid deadlock. worked_id is the
                # worker that most recently communicated with the central server.
                if worker_identification != worker_id:
                    comm.recv(source=worker_identification)
                    comm.send([weights, terminate], dest=worker_identification)

            print(f"Central server terminates.")
            sys.stdout.flush()
            break

# Here the optimization is done. Now we track the progress. For every iteration 
# we do as follows: 1. Broadcast weight to workers. 2. Each worker evaluates
# its objective. 3. Sum the workers' objectives at the central server.
if rank == 0:
    central_server.all_iterates.append(weights)  # The last weights have not yet been added for DAVERPG.
    total_obj = np.zeros((1))
    obj = np.zeros((1))
else:
    total_obj = None 
    current_weights = np.empty(dim)

# Broadcast iteration number. iter_num is equal to the number of executed iterations.
iter_num = comm.bcast(iter_num, root=0) 
if rank == 0:
    print("iter_num: ", iter_num)

for iter in range(0, iter_num + 1):
    if rank == 0:
        current_weights = central_server.all_iterates[iter]
        # The converging master variable requires an additional prox-operation.
        current_weights = np.maximum(np.abs(current_weights) - l1*step_size, 0)*np.sign(current_weights)

    current_weights = comm.bcast(current_weights, root=0) 

    if rank != 0:
        obj = np.array([worker.compute_obj(current_weights)])

    comm.Reduce([obj, MPI.DOUBLE], [total_obj, MPI.DOUBLE], op=MPI.SUM, root=0)

    if rank == 0: 
        central_server.all_objective_values.append(total_obj[0] + l1*np.linalg.norm(current_weights, 1))
        if iter % 100 == 0:
            print(central_server.all_objective_values[iter])
   
# Let the central server save data
if rank == 0:
    filename = f"DAVERPG/DAVERPG_{name_dataset}_step_size={step_size}.pickle"
    information = central_server.return_information_upon_termination()
    with open("../../experiment_data/" + filename, 'wb') as handle:
        pickle.dump(information, handle)
   
                    
                






