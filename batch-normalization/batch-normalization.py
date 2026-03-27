import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x=np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    m=len(x)
    if x.ndim==4:
        gamma = gamma.reshape(1, -1, 1, 1)#-1 trong reshape-> tự động suy kích thước sao cho sl không đổi
        beta = beta.reshape(1, -1, 1, 1)#64->(1,64,1,1)
        mu=np.mean(x,axis=(0,2,3),keepdims=True)#normalize per-channel, không phải toàn tensor
        var=np.mean((x-mu)**2,axis=(0,2,3),keepdims=True)#  (0,2,3) norm trên các W,H,N, không norm C vì C các channel 
        """
        Mục tiêu:axis=(0,2,3)
        Chuẩn hóa từng channel riêng biệt
        Tức là:
        mỗi channel có 1 mean riêng
        mỗi channel có 1 variance riêng
        """
        xi_hat=(x-mu)/np.sqrt(var+eps)
        yi=gamma*xi_hat+beta
        return yi
    mu=np.mean(x,axis=0,keepdims=True)#lấy dọc xuống, lấy giá trị của 1 feature
    var=np.mean((x-mu)**2,axis=0,keepdims=True)
    xi_hat=(x-mu)/np.sqrt(var+eps)
    yi=gamma*xi_hat+beta
    return yi
    
    pass