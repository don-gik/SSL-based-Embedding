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


class LogVarianceLoss(nn.Module):
    def __init__(self, target_std=0.25):
        super().__init__()
        self.log_target_std = torch.log(torch.tensor(target_std))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        std_z = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-04)
        log_std_z = torch.log(std_z)

        diff = torch.relu(self.log_target_std - log_std_z)
        var_loss = torch.mean(diff**2)
        return var_loss


class CostDeflatedOTLoss(nn.Module):
    """Cost-Deflated Optimal Transport (Sinkhorn) Loss.

    Suppresses the top-k principal directions in feature space to force the network
    to utilize tail dimensions, maximizing effective rank without SVD gradient instability.
    """

    def __init__(
        self,
        k: int = 1,
        lambda_penalty: float = 1.0,
        tau: float = 0.1,
        sinkhorn_eps: float = 0.1,
        sinkhorn_iters: int = 10,
        power_iters: int = 7,
    ):
        super().__init__()
        self.k = k
        self.lambda_penalty = lambda_penalty
        self.tau = tau
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.power_iters = power_iters

    @torch.no_grad()
    def _get_top_k_vectors(self, M: torch.Tensor) -> torch.Tensor:
        """Extracts top-k principal feature directions V_k (D, k) from Teacher embeddings via Power Iteration."""
        _, d = M.shape
        V = torch.randn(d, self.k, device=M.device, dtype=M.dtype)
        V, _ = torch.linalg.qr(V)

        for _ in range(self.power_iters):
            V = M.T @ (M @ V)
            V, _ = torch.linalg.qr(V)
        return V

    @torch.no_grad()
    def _sinkhorn_knopp(self, C: torch.Tensor) -> torch.Tensor:
        """Computes doubly stochastic target matrix Q from deflated cost matrix C."""
        K = torch.exp(-C / self.sinkhorn_eps)
        u = torch.ones(C.size(0), 1, device=C.device, dtype=C.dtype)
        v = torch.ones(C.size(1), 1, device=C.device, dtype=C.dtype)

        for _ in range(self.sinkhorn_iters):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)

        Q = u * K * v.T
        return Q

    def forward(self, z_student: torch.Tensor, z_teacher: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_student (torch.Tensor): Predictor/Student embeddings [B, D]
            z_teacher (torch.Tensor): Teacher/Target embeddings [B, D]
        """
        # L2 Norm
        z_s = F.normalize(z_student, dim=-1)
        z_t = F.normalize(z_teacher, dim=-1)

        # Similarity Matrix
        S = z_s @ z_t.T  # [B, B]

        # Top-k Deflation via Teacher V_k
        V_k = self._get_top_k_vectors(z_t)  # [D, k]
        P_k = (z_s @ V_k) @ (z_t @ V_k).T  # [B, B]

        # Deflated Cost Matrix & Doubly Stochastic Target Q
        C_deflated = -S + self.lambda_penalty * P_k
        Q = self._sinkhorn_knopp(C_deflated)  # [B, B]

        # Cross-Entropy Loss against Soft Target Q
        P_logits = S / self.tau
        log_probs = F.log_softmax(P_logits, dim=-1)

        loss = -torch.sum(Q * log_probs, dim=-1).mean()
        return loss
