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
      pass
        
    d_ap=np.sum(np.square(anchor-positive),axis=-1)
    d_an=np.sum(np.square(anchor-negative),axis=-1)
    l=np.maximum(0,d_ap-d_an+margin)
    return float(np.mean(l))
    pass