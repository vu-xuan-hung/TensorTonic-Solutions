import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x=np.array(x)
    return np.maximum(x,0)
    pass