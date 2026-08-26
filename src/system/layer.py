import torch
import torch.nn as nn
from torch import Tensor


class GaussianNoiseInjection(nn.Module):
    def __init__(self, std: float = 0.01):
        super().__init__()
        self.std = std

    def forward(self, x: Tensor):
        if self.training and self.std > 0:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


def replace_dropout_with_noise(model: nn.Module, noise_std: float = 0.01):
    for name, child in model.named_children():
        if isinstance(child, nn.Dropout):
            setattr(model, name, GaussianNoiseInjection(std=noise_std))
        else:
            replace_dropout_with_noise(child, noise_std)
