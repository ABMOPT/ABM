import numpy as np
import time
from numpy import linalg as LA
from numba import jit

""" Projects the point v onto the probability simplex. """
@jit(nopython=True) 
def proj_simplex(v):
    u = np.sort(v)[::-1]
    sv = u.cumsum()
    ind = np.nonzero(u > (sv- 1) / np.arange(1, u.size+1))[0][-1]
    tau = (sv[ind] - 1) / (ind + 1) 
    x = np.maximum( v - tau, np.zeros_like(v) )
    return x

""" Projects the point x on a cartesian product of simplices. """
def proj_simplices(x, current_bundle_sizes):
    counter = 0
    for bundle_size in current_bundle_sizes:
        x[counter:counter+bundle_size] = proj_simplex(x[counter:counter+bundle_size])
        counter += bundle_size
    return x


""" Computes prox_{gamma*||.||_1}(u) """
def prox_ell1(x, gamma):
    return np.maximum(np.abs(x) - gamma, 0)*np.sign(x)
 
def proj_acc_grad_method(v, all_G, z_bar, L_subproblem, current_bundle_sizes, ell1, x0,
                         delta, track_delta, settings):
     """ The accelerated projected gradient method for solving the subproblem
         that arises in GMM when there is an ell-regularizer in the objective function.
         The implementation uses the backtracking procedure described on page 291 in
         Beck's book. 

        OBS: L_subproblem is the L in the subproblem. L is parameter in line search.

        v should be a vector, all_G should be all_G_flattened

        NOTE Would be nice to use a variant of FISTA that can decrease L in the
             backtracking. I believe that would make a difference. /DC
     """

     max_iter = settings['subprob_ell1_max_iter']
     L0, eta = settings['L0_backtracking'], settings['eta']
     begin = time.time()
     x, y, t, L = x0.copy(), x0.copy(), 1.0, L0
     tracked_deltas = []
    
     for iter in range(0, max_iter):
         # Check the empirical delta. Note that this requires an extra 
         # gradient evaluation. 
         if track_delta:
            computed_delta = compute_empirical_delta(current_bundle_sizes, z_bar, 
                                                L_subproblem, all_G, ell1, x, v)
            tracked_deltas.append(computed_delta)
        
         # Evaluate termination criteria every tenth iteration.
         if iter % 10 == 0 and iter >= 9:
            computed_delta = compute_empirical_delta(current_bundle_sizes, z_bar, 
                                                 L_subproblem, all_G, ell1, x, v)
            if computed_delta <= delta:
                break

         # Evaluate gradient and objective in y.
         gradient_step = z_bar - 1/L_subproblem * (all_G @ y) 
         prox_u = prox_ell1(gradient_step, 1/L_subproblem*ell1)                                            
         grad_y = v - all_G.T @ prox_u                                                          
         f_y = 0.5*L_subproblem*LA.norm(gradient_step)**2 + y @ v - \
               (ell1*LA.norm(prox_u, 1) + 0.5*L_subproblem*LA.norm(prox_u - gradient_step)**2)    
    
         while True:
             x1 = proj_simplices(y - 1/L*grad_y, current_bundle_sizes)

             # Evaluate objective in x1.
             gradient_step = z_bar - 1/L_subproblem * (all_G @ x1) 
             prox_u = prox_ell1(gradient_step, 1/L_subproblem*ell1)                                                                             
             f_x1 = 0.5*L_subproblem*LA.norm(gradient_step)**2 + x1 @ v - \
                (ell1*LA.norm(prox_u, 1) + 0.5*L_subproblem*LA.norm(prox_u - gradient_step)**2)    


             if f_x1 > (f_y + np.dot(grad_y, x1 - y) + 0.5*L*LA.norm(x1 - y)**2):
                 L = eta*L 
                 if L > 10000:
                    solve_time = time.time() - begin

                     # Evaluate gradient in iterate that is returned
                    gradient_step = z_bar - 1/L_subproblem * (all_G @ x) 
                    prox_u = prox_ell1(gradient_step, 1/L_subproblem*ell1)                                            
                    grad_x = v - all_G.T @ prox_u
                    #print("Subproblem solver terminates, stuck in line search.")
                    return x, solve_time, grad_x, tracked_deltas, iter
             else:
                 break
     
         t1 = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
         y = x1 + (t - 1) / t1 * (x1 - x)
         x, t = x1, t1 
        
     solve_time = time.time() - begin

     # Evaluate gradient in iterate that is returned
     gradient_step = z_bar - 1/L_subproblem * (all_G @ x) 
     prox_u = prox_ell1(gradient_step, 1/L_subproblem*ell1)                                            
     grad_x = v - all_G.T @ prox_u

     return x, solve_time, grad_x, tracked_deltas, iter


def compute_empirical_delta(current_bundle_sizes, z_bar, L_subproblem, all_G, ell1, x, v):
    gradient_step = z_bar - 1/L_subproblem * (all_G @ x) 
    prox_u = prox_ell1(gradient_step, 1/L_subproblem*ell1)                                            
    grad_x = v - all_G.T @ prox_u
    empirical_delta = 0
    counter = 0
    for i in range(len(current_bundle_sizes)):
        empirical_delta += np.max(-grad_x[counter:counter+current_bundle_sizes[i]])
        counter += current_bundle_sizes[i]
            
    empirical_delta += grad_x @ x
    return empirical_delta

# Not tested.
#def activate_jit_subproblem_solver():
#    dim, num_of_workers, bundle_size, L_subproblem = 10, 3, 3, 1
#    current_bundle_sizes = np.ones(num_of_workers)*bundle_size
#    current_total_bundle_size = np.sum(current_bundle_sizes)
#    ell1, delta, x0, track_delta = 1, 0, np.ones(dim)*0.3, True
#    z_bar = np.ones(dim)
#    v = np.ones(current_total_bundle_size)
#    G_flattened = np.ones((dim, current_total_bundle_size))
#    settings = {"subprob_ell1_max_iter": 5, "L0_backtracking": 0.01,
#                "eta": 1.5}
#    out1, out2, out3, out4 = proj_acc_grad_method(v, G_flattened, z_bar,
#       L_subproblem, current_bundle_sizes, ell1, x0, delta, track_delta, 
#       settings)