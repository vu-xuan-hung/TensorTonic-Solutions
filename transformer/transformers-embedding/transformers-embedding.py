import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    return nn.Embedding(vocab_size,d_model)
    pass

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
# Your code here
    result=torch.zeros(len(tokens),d_model)
    for idx,token in enumerate(tokens):
        result[idx]=embedding(token)*math.sqrt(d_model)
    return result
    pass