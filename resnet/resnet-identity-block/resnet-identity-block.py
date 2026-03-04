import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    
    def __init__(self, channels: int):
        self.channels = channels
        # Simplified: using dense layers instead of conv for demo
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:                     # (N, C)
            connect = x
            z1 = x @ self.W1.T               # (N, C)
            a1 = relu(z1)
            z2 = a1 @ self.W2.T               # (N, C)
            transformed = relu(z2)            
            return transformed + connect      
    
        # elif x.ndim == 4:                    # (N, C, H, W)
        #     N, C, H, W = x.shape
        #     x_flat = x.transpose(0, 2, 3, 1).reshape(N, H*W, C)   # (N, H*W, C)
    
        #     z1 = x_flat @ self.W1.T           # (N, H*W, C)
        #     a1 = relu(z1)
        #     z2 = a1 @ self.W2.T                # (N, H*W, C)
        #     transformed_flat = relu(z2)
    
        #     # reshape back to (N, C, H, W)
        #     transformed = transformed_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    
        #     return transformed + x             # skip connection, no final ReLU
    
        # else:
        #     raise ValueError(f"Unsupported input dimension: {x.ndim}")
     