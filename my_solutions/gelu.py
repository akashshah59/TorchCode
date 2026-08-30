import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def my_gelu(x):
    two = torch.tensor(2.0)
    return x * 0.5 * (1 + torch.erf(x / torch.sqrt(two)))