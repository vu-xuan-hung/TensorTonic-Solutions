import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    x=np.array(x)
    if x.ndim==3:
        gapc=np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            gapc[i]=np.mean(x[i,:,:],axis=(0,1))
    elif x.ndim==4:
        gapc=np.zeros((x.shape[0],x.shape[1]))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                gapc[i][j]=np.mean(x[i,j,:,:],axis=(0,1))
    else: 
        raise ValueError
    return gapc
    pass