import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    w=np.array(w)
    g=np.array(g)
    G=np.array(G)
    Gt=G+np.power(g,2)
    wt=w-g*lr/np.sqrt(Gt+eps)
    return (wt,Gt)
    pass