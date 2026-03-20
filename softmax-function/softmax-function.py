import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x=np.array(x)
    soft=[]
    if x.ndim == 1:
        m = np.max(x)
        return np.exp(x - m) / np.sum(np.exp(x - m))
    for i in x:
        max=np.max(i)
        soft.append(np.exp(i-max)/np.sum(np.exp(i-max)))
    return soft