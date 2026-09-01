import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ✏️ YOUR IMPLEMENTATION HERE

def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token):
    beams = [(0.0, [start_token])]

    for _ in range(max_len - 1):
        # Best beam already finished — no need to keep expanding.
        if beams[0][1][-1] == eos_token:
            break

        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_token:
                # Finished beams are carried forward unchanged, not re-expanded.
                candidates.append((score, seq))
                continue

            log_probs = log_prob_fn(torch.tensor(seq))
            top_k, top_k_idx = torch.topk(log_probs, beam_width)
            candidates += [
                (score + top_k[j].item(), seq + [top_k_idx[j].item()])
                for j in range(beam_width)
            ]

        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

    return beams[0][1]