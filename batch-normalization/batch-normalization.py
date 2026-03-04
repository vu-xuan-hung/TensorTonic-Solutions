import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x=np.array(x)
    gamma=np.array(gamma)
    beta=np.array(beta)  
    if x.ndim==4:
        gamma = gamma.reshape(1, -1, 1, 1)
        beta  = beta.reshape(1, -1, 1, 1)
        u=np.mean(x,axis=(0,2,3),keepdims=True)
        o=np.mean(np.square(x-u),axis=(0,2,3),keepdims=True)
        xi=(x-u)/np.sqrt(o+eps)
        yi=gamma*xi+beta
    else:
        u=np.mean(x,axis=0,keepdims=True)
        o=np.mean(np.square(x-u),axis=0,keepdims=True)
        xi=(x-u)/np.sqrt(o+eps)
        yi=gamma*xi+beta
    return yi