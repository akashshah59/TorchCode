import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
        pass  # Initialize gate_proj, up_proj, down_proj

    def forward(self, x): 

        gate_proj = self.gate_proj(x) # dff 
        up_proj = self.up_proj(x) # dff

        return self.down_proj(F.silu(gate_proj) * up_proj)
        
