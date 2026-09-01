import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        # Scale max_lr based on where we are in the warmup stage.
        lr = max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        # Where are we in the non-warmup stage? 
        lr = min_lr + 0.5 * (max_lr-min_lr) * (1 + math.cos(math.pi * progress))
    return lr