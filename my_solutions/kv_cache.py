import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scaling_d = d_model // num_heads

        # B, S ,d 
        self.W_q = nn.Linear(d_model, d_model, bias = False)
        self.W_k = nn.Linear(d_model, d_model, bias = False)
        self.W_v = nn.Linear(d_model, d_model, bias = False)
        self.W_o = nn.Linear(d_model, d_model, bias = False)

    def forward(self, x, cache=None):
        print(f"New call to attention with {x.shape[-2]}")
        # 1. Project Q, K, V from x
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        head_dim= self.scaling_d

        Q_proj=  self.W_q(x)
        K_proj = self.W_k(x)
        V_proj = self.W_v(x)


        # Each head must get full sequence. 
        q = Q_proj.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        # Enforce that each head contains all sequence * head dim together in one layout. 
        k = K_proj.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = V_proj.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)

        if cache is not None:
            # Decoding step. 
            print("We have a past cache")
            # Add new incremental sequence.
            print("Past cache shape:", cache[0].shape)
            k = torch.cat([cache[0],k], dim = 2)
            v = torch.cat([cache[1],v], dim = 2)
            print("New cache shape:", k.shape)
        
        # Only new scores computed. based on new q and concatenated k.
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        new_cache = (k,v)
        S_total = new_cache[0].shape[-2] # Seq_length.

        if seq_len > 1: # Prefill phase only. 
            S_past = S_total - seq_len
            print("Ones:", torch.ones(seq_len, S_total, dtype=torch.bool).shape)
            print("S_past:", S_past)
            mask = torch.triu(
                torch.ones(seq_len, S_total, dtype=torch.bool),
                diagonal=S_past + 1,
            )

            scores = scores.masked_fill(mask, float('-inf'))
            print("Mask:", mask.shape)
            print("Scores:", scores.shape)

        # Decode. 
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = self.W_o(attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))  
        return out, new_cache      
