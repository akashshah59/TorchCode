import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def cross_entropy_loss(logits, targets):
    # Do not assume
    # B x C (number of classes)
    # B * 1 (Class values)
    
    stable_logits = logits - torch.max(logits) 
    exps = torch.exp(stable_logits)
    sums = torch.sum(exps,axis = -1)
    
    print(stable_logits.shape)
    target_logits = stable_logits[torch.arange(logits.size(0)), targets]

    print(target_logits.shape)
    # print(sums)

    return torch.mean(torch.log(sums) - target_logits)