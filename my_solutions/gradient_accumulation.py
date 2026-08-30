import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def accumulated_step(model, optimizer, loss_fn, micro_batches):
    optimizer.zero_grad()
    for (x,y) in micro_batches:
        target = y
        preds = model(x)
        loss = loss_fn( preds, target) / len(micro_batches)
        loss.backward()

    optimizer.step()

    return loss.item()