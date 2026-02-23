import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _loss(p,y):
    return -np.mean(np.sum(y*np.log(p)+(1-y)*np.log(1-p)))
def _initialize(X):
    w=np.zeros(X.shape[1])
    b=0.0
    return w,b
def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code 
    w,b=_initialize(X)
    m=X.shape[0]
    for i in range(steps):
        p=_sigmoid(np.dot(X,w)+b)
        loss=_loss(p,y)
        dw=(1/m)*np.dot(X.T,(p-y))
        db=np.mean(p-y)
        w=w-lr*dw
        b=b-lr*db
    return w,b