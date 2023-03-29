import torch

def fix_bounds(tensor: torch.Tensor):
    tensor[tensor > 1.0] = 1.0
    tensor[tensor < 0.0] = 0.0

    return tensor