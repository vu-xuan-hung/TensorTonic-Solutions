import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y=np.array(y)
    num,frequencies=np.unique(y,return_counts=True)
    p = np.zeros(len(frequencies))
    for i in range(len(frequencies)):
        p[i]=frequencies[i]/len(y)
    p=p[p>0]
    H=-np.sum(p*np.log2(p))
    return H