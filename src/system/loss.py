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


class GramDiLoss(nn.Module):
    def __init__(
        self,
        center_momentum: float = 0.9,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

        self.register_buffer("center", None)

    def forward(self, student_emb, teacher_emb):
        s_norm = F.normalize(student_emb, p=2, dim=-1)
        t_norm = F.normalize(teacher_emb, p=2, dim=-1)

        # Batch Sim
        z_student = torch.matmul(s_norm, s_norm.T)
        z_teacher = torch.matmul(t_norm, t_norm.T)

        # Masking Trace
        B = student_emb.shape[0]
        mask = torch.eye(B, device=student_emb.device).bool()

        # Center Update
        with torch.no_grad():
            non_diag_teacher = z_teacher[~mask]
            batch_mean = non_diag_teacher.mean()
            self.update_center(batch_mean)

        # Teacher Centering & Sharpening
        z_teacher_centered = z_teacher - self.center
        z_teacher_masked = z_teacher_centered.masked_fill(mask, -1e9)
        p_teacher = F.softmax(z_teacher_masked / self.teacher_temp, dim=-1)

        # Student Sharpening & Cross-Entropy
        z_student_masked = z_student.masked_fill(mask, -1e9)
        p_student = F.log_softmax(z_student_masked / self.student_temp, dim=-1)

        loss = -torch.sum(p_teacher * p_student, dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, z_teacher):
        batch_center = z_teacher.mean(dim=0, keepdim=True)
        if self.center is None or self.center.shape != batch_center.shape:
            self.center = batch_center
        else:
            self.center = self.center * self.center_momentum + batch_center * (
                1 - self.center_momentum
            )
