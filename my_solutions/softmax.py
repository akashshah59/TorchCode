import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

# [4,3]
# [[0,0,0,0]
#  [0,0,0,0]]
# (4, 8) 

def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - torch.max(x)
    exps = torch.exp(x) 
    sums = torch.sum(exps, axis = -1).unsqueeze(-1) # Operate along that axis one by one.
    print(exps / sums.expand(exps.shape))
    return exps / sums.expand(exps.shape)
    