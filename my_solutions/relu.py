import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE
# max(0,x)

def relu(x: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(x.shape)
    return torch.where(x > 0, x, zeros)