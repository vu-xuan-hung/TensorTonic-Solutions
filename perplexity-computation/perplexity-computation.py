import numpy as np
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    prob_distributions=np.array(prob_distributions)
    actual_tokens=np.array(actual_tokens)
    Hi=0.0
    for i in range(len(prob_distributions)):
        pi=prob_distributions[i][actual_tokens[i]]
        Hi+=np.log(pi)
    H=-1/len(actual_tokens)*Hi
    PP=np.exp(H)
    return PP