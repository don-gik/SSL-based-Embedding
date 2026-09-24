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
        hidden_dim: int,
        k: int = 3,
        alpha: float = 0.15,
        tau: float = 0.1,
        gamma: float = 0.9,
        sinkhorn_eps: float = 0.15,
        sinkhorn_iters: int = 10,
        power_iters: int = 7,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters
        self.power_iters = power_iters

        self.register_buffer("t_center", torch.zeros(1, hidden_dim))
        self.center_momentum = center_momentum

        self.last_Q = None

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
    def _unbalanced_sinkhorn(self, C: torch.Tensor) -> torch.Tensor:
        """Unbalanced Sinkhorn-Knopp Algorithm (Chizat et al., 2018)"""
        K = torch.exp(-C / self.sinkhorn_eps)
        u = torch.ones(C.size(0), 1, device=C.device, dtype=C.dtype)
        v = torch.ones(C.size(1), 1, device=C.device, dtype=C.dtype)

        for _ in range(self.sinkhorn_iters):
            u = (1.0 / (K @ v + 1e-8)) ** self.gamma
            v = (1.0 / (K.T @ u + 1e-8)) ** self.gamma

        Q = u * K * v.T

        Q = Q / (Q.sum() + 1e-8) * C.size(0)
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

        # Similarity Matrix
        S_cent = z_s_cent @ z_t.T  # [B, B]
        S_original = z_s @ z_t.T

        with torch.no_grad():
            # 4. Top-k Projection
            V_k = self._get_top_k_vectors(z_t)
            P_k = (z_s_cent @ V_k) @ (z_t @ V_k).T

            # 5. Mahalanobis Cost
            S_mahalanobis = S_cent - self.alpha * P_k
            C_mahalanobis = 1.0 - S_mahalanobis

            # 6. Unbalanced Sinkhorn Target Q
            Q = self._unbalanced_sinkhorn(C_mahalanobis)

            self.last_Q = Q.detach()

        # 7. Cross-Entropy Loss against Soft Target Q
        P_logits = S_original / self.tau
        log_probs = F.log_softmax(P_logits, dim=-1)

        loss = -torch.sum(Q * log_probs, dim=-1).mean()
        return loss
