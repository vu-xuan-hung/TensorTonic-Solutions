import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a=np.array(a)
    b=np.array(b)
    y=np.array(y)
    if a.ndim==1:
        d=np.sqrt(np.sum(np.square(a-b)))
        l=y*d**2+(1-y)*np.maximum(0,margin-d)**2
    else:
        d=np.sqrt(np.sum(np.square(a-b),axis=1))
        l=y*d**2+(1-y)*np.maximum(0,margin-d)**2
    if reduction=="mean":
        return float(np.mean(l))
    else:
        return float(np.sum(l))
    pass