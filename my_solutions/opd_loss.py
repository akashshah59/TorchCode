import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor

# ✏️ YOUR IMPLEMENTATION HERE
def opd_loss(student_logits: Tensor,
             teacher_logits: Tensor,
             teacher_weights: Tensor | None = None,
             mask: Tensor | None = None,
             temperature: float = 1.0) -> Tensor:

    teacher_logits = teacher_logits.detach()

    # Loss scalar is always some aggregation over a batch.
    num_teachers = teacher_logits.shape[0]
        
    t   = temperature
    
    # 1. FIX: Evaluate p_students at temperature t
    p_students = F.softmax(student_logits / t, dim=-1)
    logp_students = F.log_softmax(student_logits / t, dim=-1) # (B, S, V)
    logp_teacher = F.log_softmax(teacher_logits / t, dim=-1)  # (T, B, S, V) or (B, S, V)

    if len(logp_students.shape) != len(logp_teacher.shape):
        kl_per_token = (p_students.unsqueeze(0) * (logp_students.unsqueeze(0) - logp_teacher)).sum(dim=-1) # (T, B, S)
    else:
        kl_per_token = (p_students * (logp_students - logp_teacher)).sum(dim=-1) # (B, S)

    if torch.is_tensor(teacher_weights):
        teacher_weights = teacher_weights.view(-1, 1, 1)
    else:
        teacher_weights = torch.ones(num_teachers, device=student_logits.device).view(-1, 1, 1)

    # 2. FIX: Sum across the teacher dimension (dim=0) if 3D
    if kl_per_token.dim() == 3:
        kl_per_token = (teacher_weights * kl_per_token).sum(dim=0) / (teacher_weights.sum() + 1e-8)

    kl_per_token = kl_per_token * (t ** 2)

    # 3. FIX: Return the masked loss directly
    if torch.is_tensor(mask):
        mask = mask.to(kl_per_token.dtype)
        return (kl_per_token * mask).sum() / (mask.sum() + 1e-8)

    return kl_per_token.mean()

    
    
