import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def simcse_loss(z1: Tensor, z2: Tensor, temperature: float = 0.1):
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


class DinoLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        center_momentum: float = 0.9,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        # 1. Sharpening
        student_out = student_output / self.student_temp
        teacher_out = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1
        )
        teacher_out = teacher_out.detach()

        # 2. Cross Entropy
        log_probs = F.log_softmax(student_out, dim=-1)
        loss = -torch.sum(teacher_out * log_probs, dim=-1).mean()

        # 3. Update Center
        self.update_center(teacher_output)

        return loss

    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )


class SigmoidDinoLoss(nn.Module):
    def __init__(
        self,
        out_dim: int,
        center_momentum: float = 0.9,
        student_temp: float = 0.2,
        teacher_temp: float = 0.05,
        warmup_steps: int = 500,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.warmup_steps = warmup_steps

        self.register_buffer("center", torch.zeros((1, out_dim)))

    def forward(self, z_student, z_teacher, global_step):
        # Center Update
        with torch.no_grad():
            self.update_center(z_teacher, global_step)

        # Teacher Centering & Sharpening
        z_teacher_centered = z_teacher - self.center
        p_teacher = F.sigmoid(z_teacher_centered / self.teacher_temp)

        # Student Sharpening & Cross-Entropy
        p_student = F.sigmoid(z_student / self.student_temp)

        loss = F.binary_cross_entropy(p_student, p_teacher.detach())
        return loss

    @torch.no_grad()
    def update_center(self, z_teacher, global_step):
        batch_center = z_teacher.mean(dim=0, keepdim=True)
        if global_step < self.warmup_steps:
            self.center = batch_center
        else:
            self.center = self.center * 0.9 + batch_center * 0.1


class CovarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        num_features = z.size(1)

        # Feature-wise Centering
        z_centered = z - z.mean(dim=0, keepdim=True)

        # 768 x 768 Covariance
        cov_matrix = (z_centered.T @ z_centered) / (z.size(0) - 1)

        off_diag_cov = cov_matrix.pow(2)
        off_diag_cov.fill_diagonal_(0)

        cov_loss = off_diag_cov.sum() / num_features
        return cov_loss
