import numpy as np
def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    recommended=np.array(recommended)
    relevant=np.array(relevant)
    top_k=recommended[:k]
    pre_k=top_k[np.isin(top_k,relevant)]
    recall_k=len(pre_k)/len(relevant)
    pre_k=len(pre_k)/k
    return [pre_k,recall_k]