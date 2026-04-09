import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
def attention(Q,K,V,d_head):
    K=K.transpose(0,2,1)
    score=softmax(np.matmul(Q,K)/np.sqrt(d_head))
    return np.matmul(score,V)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    B,N,dimk=K.shape
    d_head=dimk//num_heads
    head=[]
    Q=(np.matmul(Q,W_q)).reshape(B,N,num_heads,d_head).transpose(0, 2, 1, 3)
    K=(np.matmul(K,W_k)).reshape(B,N,num_heads,d_head).transpose(0, 2, 1, 3)
    V=(np.matmul(V,W_v)).reshape(B,N,num_heads,d_head).transpose(0, 2, 1, 3)
    for i in range(num_heads):   
        Qi=Q[:,i,:,:]
        Ki=K[:,i,:,:]
        Vi=V[:,i,:,:]
        headi=attention(Qi,Ki,Vi,d_head)
        head.append(headi)
    concat=np.concatenate(head,axis=-1)#nối các headi vì dmodel đã chia
    return np.matmul(concat,W_o)
    
    pass