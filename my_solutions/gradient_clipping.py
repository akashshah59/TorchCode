import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def clip_grad_norm(parameters, max_norm):
    # Is gradient cliping applied at every level? 
    print(parameters)
    total_norm = torch.sqrt(
        sum(torch.sum(p.grad**2) for p in parameters 
            if p.grad is not None))
    
    if total_norm > max_norm:
        for p in parameters:
            p.grad = p.grad * max_norm / total_norm

    return total_norm

