import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x=np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    m=len(x)
    if x.ndim==4:
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
        mu=np.mean(x,axis=(0,2,3),keepdims=True)
        var=np.mean((x-mu)**2,axis=(0,2,3),keepdims=True)
        xi_hat=(x-mu)/np.sqrt(var+eps)
        yi=gamma*xi_hat+beta
        return yi
    mu=np.mean(x,axis=0,keepdims=True)
    var=np.mean((x-mu)**2,axis=0,keepdims=True)
    xi_hat=(x-mu)/np.sqrt(var+eps)
    yi=gamma*xi_hat+beta
    return yi
    
    pass