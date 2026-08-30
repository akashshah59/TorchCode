import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = nn.Parameter(torch.randn((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros((out_features,)))
        std = 1 / math.sqrt(in_features)
        with torch.no_grad():
            self.weight.normal_(mean=0, std=std )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x in , in * out , B * out 
        return  x @ self.weight.T + self.bias