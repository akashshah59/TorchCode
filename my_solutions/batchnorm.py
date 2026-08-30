import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def my_batch_norm(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps=1e-5,
    momentum=0.1,
    training=True,
):
    if training:
        # B * D 
        mean = torch.mean(x, dim = 0)
        var = torch.var(x, dim = 0, unbiased = False)
        
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(momentum * mean)
            running_var.mul_(1 - momentum).add_(momentum * var)
        
        return gamma * (x - mean) / torch.sqrt(var  + eps) + beta

    return gamma * (x - running_mean) / torch.sqrt(running_var  + eps) + beta 

