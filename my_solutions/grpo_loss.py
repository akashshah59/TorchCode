import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

from torch import Tensor

def grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor,
              eps: float = 1e-5) -> Tensor:

    # Assume this is the sum of log probabilities. 
    # print(logps.shape, logps)
    # print(group_ids.shape, group_ids)
    # print(rewards.shape, rewards)

    gids = group_ids.unique()
    advantages = torch.empty_like(rewards)
    
    for gid in gids:
        mask_index = gid == group_ids
        mu = rewards[mask_index].mean()
        sigma = rewards[mask_index].std(unbiased = False)

        A = (rewards[mask_index] - mu) / (sigma + eps)
        advantages[mask_index] = A
    
    return -(logps * advantages.detach()).mean()