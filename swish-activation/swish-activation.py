import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x=np.array(x)
    sigmoid=1/(1+np.exp(-x))
    swishs=x*sigmoid
    return swishs
    pass