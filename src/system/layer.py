import torch
import torch.nn as nn
from torch import Tensor

from src.system.utils import trunc_normal_


# Copy & Paste from https://github.com/facebookresearch/dino/blob/main/utils.py
class DINOHead(nn.Module):
    def __init__(
        self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


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


def change_noise_std(model: nn.Module, new_std: float):
    for child in model.children():
        if child.__class__.__name__ == "GaussianNoiseInjection":
            child.std = new_std
        else:
            change_noise_std(child, new_std)
