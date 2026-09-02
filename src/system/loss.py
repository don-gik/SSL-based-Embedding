import torch
import torch.nn as nn


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
