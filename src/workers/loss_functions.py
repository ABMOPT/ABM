import numpy as np
import numpy.linalg as LA
from .utils import logsig, expit_b

""" This class implements oracles for the following logistic loss function
    when the data is distributed among several workers. Let N denote the total 
    number of data points (among all workers). The implemented function is
    f(x) = 1/N \sum_{i=1}^N [ (1-y_i)a_i^T x + log(1 + exp(-a_i^T x)) + lambda*||x||_2^2].
    For this form on the loss function, the labels must be {0, 1}.

    A represents the data matrix, y the labels, l2 the regularization parameter.
"""
class log_reg_ell2_distributed:
    def __init__(self, A, y, l2, total_num_of_data_points_all_workers):
        self.A = A 
        self.l2 = l2
        self.y = y 
        self.num_of_data_points_this_worker = len(y)
        self.total_num_of_data_points_all_workers = total_num_of_data_points_all_workers
        
        # Throw error if the labels are not binary.
        if len(np.where(self.y == 0)[0]) + len(np.where(self.y == 1)[0]) != len(self.y):
            raise("Labels must be binary!")

    def loss(self, w):
        Aw = self.A.dot(w)
        loss = (np.sum((1-self.y)*Aw - logsig(Aw)) + \
               self.num_of_data_points_this_worker*
               (0.5*self.l2)*LA.norm(w)**2)/self.total_num_of_data_points_all_workers
        return loss

    def grad(self, w):
        Aw = self.A.dot(w)
        s = expit_b(Aw, self.y)
        grad = (self.num_of_data_points_this_worker*self.l2*w + 
                self.A.T.dot(s))/self.total_num_of_data_points_all_workers
        
        return grad
        
    def first_order_oracle(self, w):
        Aw = self.A.dot(w)
        s = expit_b(Aw, self.y)
        loss = (np.sum((1-self.y)*Aw - logsig(Aw)) + \
                self.num_of_data_points_this_worker*
                (0.5*self.l2)*LA.norm(w)**2)/self.total_num_of_data_points_all_workers
        
        grad = (self.num_of_data_points_this_worker*self.l2*w + 
                self.A.T.dot(s))/self.total_num_of_data_points_all_workers

        return grad, loss 