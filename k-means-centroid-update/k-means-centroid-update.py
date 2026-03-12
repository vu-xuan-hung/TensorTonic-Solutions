import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """

    points = np.array(points)
    assignments = np.array(assignments)

    centroid = np.zeros((k, points.shape[1]))

    for i in range(k):
        point = points[assignments == i]

        if len(point) > 0:
            centroid[i] = np.mean(point, axis=0)

    return centroid.tolist()