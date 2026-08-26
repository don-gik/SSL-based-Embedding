import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def simcse_loss(z1: Tensor, z2: Tensor, temperature: float = 0.05):
    """SimCSE Loss.

    Args:
        z1 (torch.Tensor): [Batch_Size, Hidden_Dim]
        z2 (torch.Tensor): [Batch_Size, Hidden_Dim]
        temperature (float): default 0.05

    Returns:
        torch.Tensor: Cross Entropy Loss
    """
    # 1. L2 Norm
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)

    # 2. Cross Entrophy
    cos_sim = torch.mm(z1, z2.t()) / temperature

    # 3. Ans Label
    labels = torch.arange(cos_sim.size(0), device=cos_sim.device)

    # 4. CrossEntropyLoss
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(cos_sim, labels)
