import numpy as np
def max_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping maximum-pooled windows.
    """
    # Write code here
    X = np.array(X)
    H,W = X.shape
    H_out = H//pool_size
    W_out = W//pool_size
    map = np.zeros((H_out,W_out))
    for i in range(H_out):
        for j in range(W_out):
            data = X[
                i*pool_size:(i+1)*pool_size,
                j*pool_size:(j+1)*pool_size
            ]
            map[i,j] = np.max(data)
    return map.tolist()
            