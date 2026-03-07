import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    
    
    d = gradients_F[0].shape[0]
    I = np.eye(d)
    skip=gradients_F[0]+I
    for i in range(1,len(gradients_F)):
        skip=skip@(I+gradients_F[i])
    return x@skip
   

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    no_skip=gradients_F[0]
    for i in range(1,len(gradients_F)):
        no_skip=gradients_F[i]@no_skip
    return x@no_skip
    pass
