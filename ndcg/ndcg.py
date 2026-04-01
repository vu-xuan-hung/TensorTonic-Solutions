import math
import numpy as np
def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    rel=np.array(relevance_scores)
    rel_k=rel[:k]
    DCG=0.0
    IDCG=0.0
    for i in range(len(rel_k)):
        DCG+=(2**rel[i]-1)/np.log2(i+2)
    idea=np.sort(rel)[::-1][:k]
    for i in range(len(idea)):
        IDCG+=(2**idea[i]-1)/np.log2(i+2)
    if IDCG==0:
        return 0.0
    NDCG=DCG/IDCG
    return NDCG
    pass