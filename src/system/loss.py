import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoLoss(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        s_temp: float = 0.15,
        t_temp: float = 0.07,
        center_momentum: float = 0.9,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.s_temp = s_temp
        self.t_temp = t_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, vocab_size))

    def forward(self, s_logit: torch.Tensor, t_logit: torch.Tensor) -> torch.Tensor:
        s_log_probs = F.log_softmax(s_logit / self.s_temp, dim=-1)
        t_probs = F.softmax((t_logit - self.center) / self.t_temp, dim=-1)

        loss = torch.sum(-t_probs * s_log_probs, dim=-1).mean()
        return loss

    @torch.no_grad
    def update_center(self, t_logit: torch.Tensor):
        batch_center = torch.sum(t_logit, dim=0, keepdim=True)
        batch_center = batch_center / len(t_logit)

        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class CovarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        num_features = z.size(1)

        # Feature-wise Centering
        z_centered = (z - z.mean(dim=0, keepdim=True)) / (
            z.std(dim=0, keepdim=True) + 1e-5
        )

        # 768 x 768 Covariance
        cov_matrix = (z_centered.T @ z_centered) / (z.size(0) - 1)

        off_diag_cov = cov_matrix.pow(2)
        off_diag_cov.fill_diagonal_(0)

        cov_loss = off_diag_cov.sum() / num_features
        return cov_loss


class VarianceLoss(nn.Module):
    def __init__(self, target_std=0.25):
        super().__init__()
        self.target_std = target_std

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        std_z = torch.sqrt(z.var(dim=0) + 1e-04)
        var_loss = torch.mean((std_z - self.target_std) ** 2)
        return var_loss
