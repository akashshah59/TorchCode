import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def kaiming_init(weight):
    fan_in = weight.shape[1]
    std = math.sqrt(2 / fan_in)
    weight.normal_(mean=0, std=std )

    return weight