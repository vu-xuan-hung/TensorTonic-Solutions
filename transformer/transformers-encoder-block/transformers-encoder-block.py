import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    u=np.mean(x,axis=-1,keepdims=True)
    o= np.var(x, axis=-1, keepdims=True)
    layer=gamma*(x-u)/np.sqrt(o+eps)+beta
    return layer
    pass
def attention(Q,K,V,d_head):
    K=K.transpose(0,2,1)
    score=softmax(np.matmul(Q,K)/np.sqrt(d_head))
    return np.matmul(score,V)
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    B,N,dim=K.shape
    dim_head=dim//num_heads
    heads=[]
    Q=np.matmul(Q,W_q).reshape(B,N,num_heads,dim_head).transpose(0, 2, 1, 3)
    K=np.matmul(K,W_k).reshape(B,N,num_heads,dim_head).transpose(0, 2, 1, 3)
    V=np.matmul(V,W_v).reshape(B,N,num_heads,dim_head).transpose(0, 2, 1, 3)
    for i in range(num_heads):
        Qi=Q[:,i,:,:]
        Ki=K[:,i,:,:]
        Vi=V[:,i,:,:]
        headi=attention(Qi,Ki,Vi,dim_head)
        heads.append(headi)
    concat=np.concatenate(heads,axis=-1)
    return np.matmul(concat,W_o)
    pass

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    z1=np.matmul(x,W1)+b1
    a1=np.maximum(0,z1)
    z2=np.matmul(a1,W2)+b2
    return z2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    multi=multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)
    layer1=layer_norm(multi+x,gamma1,beta1)
    z2=feed_forward(layer1,W1,b1,W2,b2)
    layer2=layer_norm(z2+layer1,gamma2,beta2)
    return layer2
    pass