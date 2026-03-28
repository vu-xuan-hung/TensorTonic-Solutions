import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code 
    x_t=np.array(x_t)
    h_prev=np.array(h_prev)
    Wx=np.array(Wx)
    Wh=np.array(Wh)
    b=np.array(b)
    ht=np.tanh(Wx.T@x_t+h_prev@Wh+b)
    return ht
    pass
