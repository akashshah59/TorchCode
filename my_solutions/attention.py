import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def scaled_dot_product_attention(Q, K, V):
    # Q = B, seq_q, d_k
    # K = B, seq_q , d_k

    attention = torch.bmm(Q, K.transpose(-2,-1)) 
    softmax_attention = torch.softmax(attention / math.sqrt(Q.shape[-1]), dim = -1) 

    return torch.bmm(softmax_attention, V)