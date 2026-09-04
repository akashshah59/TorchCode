import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Go into each parameter and set requires_grad = False
        self.linear.weight.requires_grad_(False)
        self.linear.bias.requires_grad_(False)

        self.alpha = alpha 
        self.rank = rank
        self.lora_A =  nn.Parameter(torch.randn(rank, in_features)) # r * in 
        self.lora_B =  nn.Parameter(torch.zeros(out_features, rank)) # o * r  


        pass  # frozen linear + lora_A + lora_B

    def forward(self, x):

        with torch.no_grad():
            w = self.linear(x)

        # x = in_features .
        scaling = self.alpha / self.rank

        return w + scaling * (x  @ self.lora_A.T @ self.lora_B.T)