import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    py=y_pred[np.arange(y_pred.shape[0]),y_true]
    """
    np.arange(N) = [0, 1, 2]
    labels       = [1, 2, 0]
    probs[[0,1,2], [1,2,0]]
    [
    probs[0][1],  # 0.9
    probs[1][2],  # 0.5
    probs[2][0]   # 0.6
    ]
    """
    loss=np.log(py)
    cross_entropyloss=-np.mean(loss)
    return cross_entropyloss