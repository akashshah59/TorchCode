import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    
    return -torch.log(torch.sigmoid(beta * ((policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps)))).mean()
