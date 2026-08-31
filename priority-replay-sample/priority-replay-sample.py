import numpy as np
def priority_replay_sample(priorities: list, alpha: float, beta: float) -> list:
    """
    Returns sampling probabilities and normalized importance weights.
    """
    # Write code here
    
    priorities = np.array(priorities)
    alpha = np.array(alpha)
    beta = np.array(beta)

    N=len(priorities)
    pi = np.pow(priorities,alpha)
    probility = pi/np.sum(pi)
    wi = np.pow((N*probility),-beta)
    wi_hat = wi/max(wi)
    return [probility.tolist(),wi_hat.tolist()]