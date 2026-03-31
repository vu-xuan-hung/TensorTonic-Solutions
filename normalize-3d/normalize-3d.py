import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v=np.array(v,dtype=float)
    norm=np.zeros_like(v)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / norm
    for i in range(len(v)):
        if np.all(v[i]==0):
            norm[i]=np.zeros_like(v[i])
        else:
            norm[i]=v[i]/np.linalg.norm(v[i])
    return norm
    pass