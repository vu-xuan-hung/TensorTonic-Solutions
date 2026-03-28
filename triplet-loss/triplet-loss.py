import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    anchor=np.array(anchor)
    positive=np.array(positive)
    negative=np.array(negative)
    if anchor.ndim==1:
        p=np.sum(np.square(anchor-positive))
        n=np.sum(np.square(anchor-negative))
        l=np.maximum(0,p-n+margin)
        return float(l)
    else:
        p=np.sum(np.square(anchor-positive),axis=1)
        n=np.sum(np.square(anchor-negative),axis=1)
        l=np.maximum(0,p-n+margin)
        return float(np.mean(l))
    pass