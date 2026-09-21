import torch
import torch.nn as nn


class GrokfastEMA:
    def __init__(self, alpha: float = 0.98, lamb: float = 2.0):
        self.alpha = alpha
        self.lamb = lamb
        self.grads = {}

    @torch.no_grad()
    def apply(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g = param.grad.data
                if name not in self.grads:
                    self.grads[name] = torch.zeros_like(g)

                self.grads[name] = self.alpha * self.grads[name] + (1 - self.alpha) * g
                param.grad.data = g + self.lamb * self.grads[name]
