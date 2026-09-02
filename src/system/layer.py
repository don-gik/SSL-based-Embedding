import torch
import torch.nn as nn
from torch import Tensor
from transformers.models.bert.modeling_bert import BertAttention


class BertAttentionHead(nn.Module):
    def __init__(self, config, projection_dim: None | int = None, orth: bool = False):
        super().__init__()
        self.attention = BertAttention(config)

        out_dim = projection_dim if projection_dim is not None else config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size * 2, out_dim),
        )

        if orth:
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(
                        layer.weight, gain=nn.init.calculate_gain("relu")
                    )

    def forward(self, hidden_states, attention_mask=None):
        # Bert Uses 4D Mask
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(
                dtype=hidden_states.dtype
            )
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        attention_outputs = self.attention(
            hidden_states, attention_mask=extended_attention_mask
        )

        if isinstance(attention_outputs, tuple):
            attention_output = attention_outputs[0]
        else:
            attention_output = attention_outputs

        mlp_outputs = self.mlp(attention_output)
        return mlp_outputs


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
