import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    # YOUR CODE HERE
    h_list = []
    h_prev = h_0
    T = X.shape[1]
    for t in range(T):
        x = X[:,t,:]
        h = np.tanh(x@W_xh.T+h_prev@W_hh.T+b_h)
        h_list.append(h)
        h_prev = h
    H = np.stack(h_list, axis=1)
    return (H,h_prev)
    pass