import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    X=np.array(X)
    mu=np.mean(X,axis=0)
    X_cen=X-mu
    if X.shape[0]==1 or X.ndim!=2:
        return None
    
    compute=1/(X.shape[0]-1)*(X_cen.T@X_cen)
    return compute
    pass