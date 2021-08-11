import torch


def get_synthetic_mask(mask):
    return torch.any(mask, dim=-1, keepdim=True).bool()
