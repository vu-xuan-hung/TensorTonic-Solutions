import numpy as np
def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    set_a=np.array(set_a)
    set_a=np.unique(set_a)
    set_b=np.array(set_b)
    set_b=np.unique(set_b)
    intersection=0
    for i in set_a:
        if i in set_b:
            intersection+=1
    union=len(set_a)+len(set_b)-intersection
    if union==0:
        return 0
    return intersection/union
    
        
        
        