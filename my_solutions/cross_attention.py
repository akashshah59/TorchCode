import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        self.head_d = d_model // num_heads 
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv):
        
        B, SQ_q, d_model = x_q.shape
        B, SQ_kv,e_model =x_kv.shape # Encoder (B, S_kv,D)
        print(SQ_q, SQ_kv)

        q_projected = self.W_q(x_q) # B , SQ_q, D * D , D
        k_projected = self.W_k(x_kv)
        v_projected = self.W_v(x_kv)
        
        # k_projected = torch.matmul(x_kv @ self.W_k(x_kv) # B , SQ_q, D * D, D
        # v_projected = x_kv @ self.W_v(x_kv) 

        q_projected_reshaped = q_projected.view(B, SQ_q, self.num_heads, self.head_d).transpose(1,2)
        # Switch up num_heads with seq_length, because we want attention along sequence lengths. 
        k_projected_reshaped = k_projected.view(B, SQ_kv, self.num_heads, self.head_d).transpose(1,2)
        v_projected_reshaped = v_projected.view(B, SQ_kv, self.num_heads, self.head_d).transpose(1,2)

        # k_projected_reshaped.transpose(-2, -1) B, 


        # Multiply on dim, and not number of heads. We want to multiply each sequence_len * sequence_len for all heads , for all batches. 
        scores = torch.matmul(q_projected_reshaped,
                    k_projected_reshaped.transpose(-2, -1)) / math.sqrt(self.head_d)

        weights = torch.softmax(scores, dim=-1)
        # out: B , num_heads, 6, 10 
        print(weights.shape)
        print(v_projected_reshaped.shape)
        # Weighted attention on the final encoder sequence. 

        attn = torch.matmul(weights, v_projected_reshaped)
        # weights: [2, 4, 6, 10] v: torch.Size([2, 4, 6, 16])
        # out: [B, num_heads,]

        out = attn.transpose(1, 2).contiguous().view(B, SQ_q, -1)

        print(out.shape)

        return out
