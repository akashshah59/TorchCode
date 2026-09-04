import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def causal_attention(Q, K, V):
    S = Q.shape[-2]

    attention = torch.bmm(Q, K.transpose(-2,-1)) / math.sqrt(Q.shape[-1])
    
    # Use triu to create a mask. 

    mask = torch.triu(torch.ones(S, S, dtype =torch.bool), diagonal=1) 
    # Don't mask the diagonal. 

    # Based on the mask, create attention mask.
    scores = F.softmax(attention.masked_fill(mask.unsqueeze(0), float('-inf')), dim = -1)
    print(scores.shape)
    print(V.shape)
    return torch.bmm(scores, V)