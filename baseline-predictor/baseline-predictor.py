import numpy as np
def baseline_predict(ratings_matrix, target_pairs):
    """
    Compute baseline predictions using global mean and user/item biases.
    """
    
    ratings_matrix=np.array(ratings_matrix)
    n_users = ratings_matrix.shape[0]
    n_items = ratings_matrix.shape[1]
    
    bu = np.zeros(n_users)
    bi = np.zeros(n_items)
    mu=np.mean(ratings_matrix[ratings_matrix !=0])
  
    for i in range(n_users):
        user_ratings=ratings_matrix[i]
        non_zero = user_ratings[user_ratings != 0]
        bu[i]=np.mean(non_zero-mu)
    for i in range(n_items):
        item_ratings=ratings_matrix[:,i]
        non_zero = item_ratings[item_ratings != 0]
        bi[i]=np.mean(non_zero-mu)
    # Write code here
    Prediction =[]
    for (u,i) in target_pairs:
        pre=mu+bu[u]+bi[i]
        Prediction.append(pre)
    return Prediction