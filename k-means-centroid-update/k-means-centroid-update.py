import numpy as np
def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here
    points=np.array(points)
    assignments=np.array(assignments)
    centroids=np.zeros((k,points.shape[1]))
    for i in range(k):
        point=points[assignments==i]#Boolean indexing-> no ve True tai cac vi tri =i
        if len(point)>0:
            centroids[i]=np.mean(point,axis=0)
    return centroids.tolist()
        