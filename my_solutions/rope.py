import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def apply_rope(q, k):
    # 1. Compute position angles
    # 2. Split into even/odd pairs
    # 3. Apply rotation
    B,S,D = q.shape

    positions = torch.arange(S).unsqueeze(1)
    dims = torch.arange(0,D,2) 
    freqs = 1.0 / (10000.0 ** (dims / D))
    angles = positions * freqs
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    
    def rotate(x):
        # even dimensions get 
        x1 = x[...,0::2]
        x2 = x[...,1::2]
        
        # For each dimension, there are 2 such rotated values.
        # When you stack them together, they become 

        rotated = [x1 * cos_a - x2 * sin_a,
                  x1 * sin_a + x2 * cos_a]

        return torch.stack([x1 * cos_a - x2 * sin_a,
                    x1 * sin_a + x2 * cos_a], dim=-1).flatten(-2)

   
    return rotate(q), rotate(k)



    
    