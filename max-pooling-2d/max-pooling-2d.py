import numpy as np
def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    # Write code here
    X=np.array(X)
    H=len(X)
    W=len(X[1])
    out_h=H//pool_size
    out_w=W//pool_size
    out=np.zeros((out_h,out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = X[
                i*pool_size:(i+1)*pool_size,
                j*pool_size:(j+1)*pool_size
                ]
            out[i, j] = np.max(window)
    return out.tolist()