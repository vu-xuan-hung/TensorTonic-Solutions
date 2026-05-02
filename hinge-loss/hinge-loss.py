import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    y=np.array(y_true)
    s=np.array(y_score)
    losses=np.maximum(0, margin - y*s)
    if reduction == "mean":
        return np.mean(losses)
    elif reduction == "sum":
        return np.sum(losses)