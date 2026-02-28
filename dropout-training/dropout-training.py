import numpy as np
import random
def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x=np.array(x)
    random_value=rng.random(x.shape) 
    mask=(random_value<(1-p)).astype(float)
    mask=mask*(1.0/(1.0-p))
    x=x*mask
    return x,mask