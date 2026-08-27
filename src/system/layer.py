import torch
import torch.nn as nn
from torch import Tensor


class GaussianNoiseInjection(nn.Module):
    def __init__(self, std: float = 0.01, p: float = 0.1):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, x: Tensor):
        if self.training and self.std > 0:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


def replace_dropout_with_noise(model: nn.Module, noise_std: float = 0.01):
    for name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            orig_p = child.p
            setattr(model, name, GaussianNoiseInjection(std=noise_std, p=orig_p))
        else:
            replace_dropout_with_noise(child, noise_std)
