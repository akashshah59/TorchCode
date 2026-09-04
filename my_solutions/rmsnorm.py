import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def rms_norm(x, weight, eps=1e-6):
    # Similar to layer norm 
    with torch.no_grad():
        rms_x = torch.sqrt(1 / x.shape[-1] * torch.sum(torch.square(x), dim = -1,keepdim = True) + eps)

    return torch.divide(x, rms_x)  * weight