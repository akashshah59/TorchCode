import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

# ✏️ YOUR IMPLEMENTATION HERE

def ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor,
             clip_ratio: float = 0.2) -> Tensor:

    old_logps_detached = old_logps.detach()
    advantages = advantages.detach()
    r = torch.exp(new_logps - old_logps_detached)

    unclipped = r * advantages
    clipped = torch.clamp(r, 1 - clip_ratio, 1 + clip_ratio) * advantages 

    return -torch.mean(torch.min(unclipped, clipped))
    # pass  # -mean(min(r * adv, clamp(r, 1-clip, 1+clip) * adv)) with gradients only through new_logps
