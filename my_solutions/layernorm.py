import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def my_layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim = -1 ,keepdim=True)
    var = x.var(dim = -1,keepdim=True, unbiased = False)

    return gamma * (x - mean) / torch.sqrt(var + eps) + beta
