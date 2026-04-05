import numpy as np
def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X=np.array(X)
    y=np.array(y)
    I=np.eye(X.shape[1])
    w = np.linalg.inv(X.T @ X + lam * I) @ (X.T @ y)
    return w