import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here
    # x=np.array(x)
    # if x.ndim==3:
    #     gapc=np.zeros(x.shape[0])
    #     for i in range(x.shape[0]):
    #         gapc[i]=np.mean(x[i,:,:],axis=(0,1))# lấy i thì chỉ còn 2 axis
    # elif x.ndim==4:
    #     gapc=np.zeros((x.shape[0],x.shape[1]))
    #     for i in range(x.shape[0]):
    #         for j in range(x.shape[1]):
    #             gapc[i][j]=np.mean(x[i,j,:,:],axis=(0,1))
    # else: 
    #     raise ValueError
    # return gapc
    x=np.array(x)
    if x.ndim==3:
       return np.mean(x,axis=(1,2))#shape (C,)
    elif x.ndim==4:
        return np.mean(x,axis=(2,3))#giữ lại axis 0 và 1 (N, C)
    else: 
        raise ValueError
   