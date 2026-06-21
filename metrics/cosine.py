
import torch

def cosine_similarity(a, b):
    sim = torch.nn.functional.cosine_similarity(a, b , dim=-1)
    return sim.mean().item()
