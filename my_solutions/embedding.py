import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight_matrix = torch.randn(num_embeddings, embedding_dim)
        # self.weight = nn.Parameter(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(self.weight_matrix)

    def forward(self, indices):
        return self.weight[indices]