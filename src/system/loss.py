import torch
import torch.nn as nn
import torch.nn.functional as F


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
        z_teacher = z_teacher.detach()

        # L2 Norm
        z_s = F.normalize(z_student, dim=-1)
        z_t = F.normalize(z_teacher, dim=-1)

        z_s_cent = z_s - z_s.mean(dim=0, keepdim=True)
        z_t_cent = z_t - z_t.mean(dim=0, keepdim=True)

        # Similarity Matrix
        S = z_s @ z_t.T  # [B, B]

        # Top-k Deflation via Teacher V_k
        V_k = self._get_top_k_vectors(z_t_cent)  # [D, k]
        P_k = (z_s_cent @ V_k) @ (z_t_cent @ V_k).T  # [B, B]

        # Deflated Cost Matrix & Doubly Stochastic Target Q
        C_deflated = -S + self.lambda_penalty * P_k
        Q = self._sinkhorn_knopp(C_deflated)  # [B, B]
        Q = Q.detach()

        S_original = z_s @ z_t.T

        # Cross-Entropy Loss against Soft Target Q
        P_logits = S_original / self.tau
        log_probs = F.log_softmax(P_logits, dim=-1)

        loss = -torch.sum(Q * log_probs, dim=-1).mean()
        return loss
