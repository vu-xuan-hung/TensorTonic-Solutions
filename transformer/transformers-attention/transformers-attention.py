import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    dimk=K.shape[-1]
    score=torch.matmul(Q,K.transpose(-2, -1))/math.sqrt(dimk)# transpose 2 chiều cuối, không liên quan đến batch
    weights=F.softmax(score,dim=-1)
    return torch.matmul(weights,V)
    pass