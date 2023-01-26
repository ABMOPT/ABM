from mpi4py import MPI
import numpy as np
import sys
import pickle
sys.path.append("..")
from workers.Workerl2reg import Workerl2reg
from central_servers.ABM_central_server import ParameterServerABM

""" The following commands are used to run ABM.

    mpiexec -n 10 python runnerABM.py rcv1_test -1 500 0.00002117029 3e-6 10 1 1 1e-7 False

    mpiexec -n 10 python runnerABM.py epsilon -1 500 0.000002 5e-5 10 1 1 1e-7 False

    mpiexec -n 10 python runnerABM.py MNIST8m -1 500 6.06468594e-7 3e-3 10 1 1 1e-7 False

    The initial smoothness estimate is used for the first iteration(s) since
    no smoothness estimate is available for a worker that has only been queried 
    once. 
"""
# Read some parameters from the buffer
name_dataset, max_iter, max_time = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
l2, l1, bundle_size = float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
scale_coeff_smoothness, local_L0 = float(sys.argv[7]), float(sys.argv[8])
delta, track_delta = float(sys.argv[9]), sys.argv[9]

if track_delta == 'True': track_delta = True 
else: track_delta = False

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
weights = np.zeros(dim)         # Initial guess
terminate = False               # If a worker should terminate or not.

# Initialize central server. In the first iteration the central server
# receives the gradient and objective value from each worker. Every time 
# the central server receives a gradient she updates the gradient information.
if rank == 0: 
    central_server = ParameterServerABM(bundle_size, dim, num_of_workers, delta, l1, 
                                        track_delta=track_delta)   
    central_server.initialize_time()
    message = [np.zeros(dim), 0] 
    for worker_id in range(1, num_of_workers + 1):
        comm.Recv(message, source = worker_id, tag = worker_id)
        received_gradient = message[0]
        received_obj = message[1]
        central_server.update_gradient_information(received_gradient, received_obj, worker_id)     

    worker_set = np.arange(0, num_of_workers)
    weights = central_server.update_weights(weights, worker_set)
# Initialize workers. In the first iteration all workers send the 
# gradient and objective value to the central server.
else:
    worker = Workerl2reg(rank, name_dataset, num_of_workers,
                    total_num_of_data_points_all_workers, l2)
    grad, loss = worker.compute_grad_and_obj(weights)
    comm.Send([grad, loss], dest=0, tag=rank)

# After the initialization, broadcast the weight.
weights = comm.bcast(weights, root=0)

# Main loop.
while True:
    # Each worker computes the gradient + obj for her current weights. She 
    # then sends the gradient + obj to the central server, and waits for a message.
    if rank != 0:
        grad, loss = worker.compute_grad_and_obj(weights)
        message = [grad, loss, rank]
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
    
    # The central server receives gradients + objs from the workers. She tests
    # 'num_of_workers' times if she has received information. She keeps
    # track of the workers she has received information from. 
    else:
        worker_set = []
        has_received_new_gradient = False 
        for test_time in range(1, num_of_workers+1):  
            probe = comm.iprobe()
            if probe:
                message = comm.recv()
                has_received_new_gradient = True
                received_gradient, received_obj = message[0], message[1]
                worker_id = message[2]  # received from which worker
                worker_set.append(worker_id)
                central_server.update_gradient_information(received_gradient, received_obj, worker_id)
        
        # If the central server has received new gradients she updates the weights
        # and sends the weights back to the workers with whom she has communicated
        # with previously in this iteration. If the maximum number of iterations
        # has been reached we make sure the central server receives all messages
        # sent by workers before the central server terminates. 
        if has_received_new_gradient:
            weights = central_server.update_weights(weights, worker_set)
            iter_num += 1
            current_run_time = central_server.total_run_time[iter_num]
            if ((iter_num >= max_iter - 1 and max_iter != -1) or 
                current_run_time > max_time):
                terminate = True

            if not terminate:
                for worker_id in worker_set:
                    comm.send([weights, terminate], dest=worker_id)
            # If we should terminate we make sure that we receive the message
            # from each worker before we tell them to terminate.
            else:
                for worker_identification in range(1, num_of_workers + 1):
                    if worker_identification not in worker_set:
                        comm.recv(source=worker_identification)
            
                for worker_identification in range(1, num_of_workers + 1):
                    comm.send([weights, terminate], dest=worker_identification)

                print(f"Central server terminates.")
                sys.stdout.flush()
                break

# Here the optimization is done. Now we track the progress. First we 
# add the last iterate to the record of iterates and also initialize some
# quantities needed for the reduce operation. For every iteration we do as 
# follows: 1. Broadcast weight to workers. 2. Each worker evaluates
# its objective. 3. Sum the workers' objectives at the central server.
if rank == 0:
    central_server.all_iterates.append(weights)  # The last weights have not yet been added for ABM.
    total_obj = np.zeros((1))
    obj = np.zeros((1))
else:
    total_obj = None 
    current_weights = np.empty(dim)

iter_num = comm.bcast(iter_num, root=0) 
if rank == 0:
    print("Number of iterations: ", iter_num)

for iter in range(0, iter_num+1):
    if rank == 0:
        current_weights = central_server.all_iterates[iter]

    current_weights = comm.bcast(current_weights, root=0) 

    if rank != 0:
        obj = np.array([worker.compute_obj(current_weights)])

    comm.Reduce([obj, MPI.DOUBLE], [total_obj, MPI.DOUBLE], op=MPI.SUM, root=0)

    if rank == 0:
        if l1 > 0:
            central_server.all_objective_values.append(total_obj[0] + l1*np.linalg.norm(current_weights, 1))
        else:
            central_server.all_objective_values.append(total_obj[0])
        
# Let the central server save data
if rank == 0:
        filename = f"ABM_{name_dataset}_bundle_size={bundle_size}_delta={delta}.pickle"
        information = central_server.return_information_upon_termination()
        with open("../../experiment_data/ABM/" + filename, 'wb') as handle:
            pickle.dump(information, handle)    

    
        
                    
                






