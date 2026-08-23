import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """Return the training-time BatchNorm output."""
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    if x.ndim == 2:
        u = x.mean(axis=0,keepdims=True)
        o = x.var(axis=0, keepdims = True)
        xi = (x-u)/np.sqrt(o+eps)
        return gamma*xi+beta
    else:
        gamma = gamma.reshape(1,-1,1,1)
        beta = beta.reshape(1,-1,1,1)
        u = x.mean(axis=(0,2,3),keepdims=True)
        o = x.var(axis=(0,2,3), keepdims = True)
        xi = (x-u)/np.sqrt(o+eps)
        return gamma*xi+beta

    