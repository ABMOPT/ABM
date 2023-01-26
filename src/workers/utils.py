import numpy as np
import numpy.linalg as LA
import scipy

def logsig(x):
        """ Compute the log-sigmoid function component-wise.
        See http://fa.bianp.net/blog/2019/evaluate_logistic/ for more details.

        logsig(x) = log(1/[1 + exp(-t)])


        """
        out = np.zeros_like(x)
        idx0 = x < -33
        out[idx0] = x[idx0]
        idx1 = (x >= -33) & (x < -18)
        out[idx1] = x[idx1] - np.exp(x[idx1])
        idx2 = (x >= -18) & (x < 37)
        out[idx2] = -np.log1p(np.exp(-x[idx2]))
        idx3 = x >= 37
        out[idx3] = -np.exp(-x[idx3])
        return out

def expit_b(x, b):
    """Compute sigmoid(x) - b component-wise."""
    idx = x < 0
    out = np.zeros_like(x)
    exp_x = np.exp(x[idx])
    b_idx = b[idx]
    out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
    exp_nx = np.exp(-x[~idx])
    b_nidx = b[~idx]
    out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
    return out

def safe_sparse_add(a, b):
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        # both are sparse, keep the result sparse
        return a + b
    else:
        # if of them is non-sparse, convert
        # everything to dense.
        if scipy.sparse.issparse(a):
            a = a.toarray()
            if a.ndim == 2 and b.ndim == 1:
                b.ravel()
        elif scipy.sparse.issparse(b):
            b = b.toarray()
            if b.ndim == 2 and a.ndim == 1:
                b = b.ravel()
        return a + b
        
def logistic_loss(A, y, w):
    """Logistic loss, numerically stable implementation.
    
    Parameters
    ----------
    x: array-like, shape (n_features,)
        Coefficients
    A: array-like, shape (n_samples, n_features)
        Data matrix
    b: array-like, shape (n_samples,)
        Labels
    Returns
    -------
    loss: float
    """
    z = np.dot(A, w)
    y = np.asarray(y)
    return np.mean((1-y)*z - logsig(z))

def logistic_grad(x, A, b):
    """
    Computes the gradient of the logistic loss.
        
        Parameters
        ----------
        x: array-like, shape (n_features,)
            Coefficients
        A: array-like, shape (n_samples, n_features)
            Data matrix
        b: array-like, shape (n_samples,)
            Labels
        Returns
        -------
        grad: array-like, shape (n_features,)    
    """
    z = A.dot(x)
    s = expit_b(z, b)
    return A.T.dot(s) / A.shape[0] 

