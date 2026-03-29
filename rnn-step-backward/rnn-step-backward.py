import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H
        np.outer(dz, x_t)
        
    """
    # Write code here
    x_t, h_prev, h_t, W, U, b = [np.array(x) for x in cache]
    dh = np.array(dh)
    dz=dh*(1-h_t**2)
    dW=np.outer(dz, x_t)
    dU=np.outer(dz, h_prev)
    db=dz
    dx_t=W.T@dz
    dh_prev=U.T@dz
    return dx_t,dh_prev,dW,dU,db
    
    pass
