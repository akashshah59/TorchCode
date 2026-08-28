import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            prob_matrix = torch.rand(x.shape)
            zeros = torch.zeros(x.shape)
            return torch.where(prob_matrix > self.p, x * 1 / (1-self.p), zeros)
        else:
            return x
