import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code 
    pe = np.zeros((seq_len, d_model))
    pos=np.arange(seq_len).reshape(-1,1)# (seq_len, 1)
    i=np.arange(d_model).reshape(1,-1)# (1, d_model)
    compute=pos/np.power(base,2*(i//2)/d_model)
    """
    (//2)->(2i, 2i+1) → cùng tần 
    [0, 0, 1, 1, 2, 2, ...]
    dim 0 & 1 → cùng tần số
    dim 2 & 3 → cùng tần số

    2 * (i // 2
    [0, 0, 2, 2, 4, 4, ...]
    0,0 → cho sin/cos pair
    2,2 → next pair
    """
    pe[:,0::2]=np.sin(compute[:,0::2])
    pe[:,1::2]=np.cos(compute[:,1::2])

    return pe