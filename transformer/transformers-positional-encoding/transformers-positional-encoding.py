import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pe = np.zeros((seq_length, d_model))
    pos= np.arange(seq_length).reshape(-1,1)
    i=np.arange(d_model).reshape(1,-1)
    compute=pos/(np.power(100000,2*(i//2)/d_model))
    pe[:,0::2]=np.sin(compute[:,0::2])
    pe[:,1::2]=np.cos(compute[:,1::2])
    return pe
    pass