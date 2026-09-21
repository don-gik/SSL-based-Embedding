import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr


class Evaluator:
    def __init__(self, high_score_threshold: float = 0.8):
        self.stsb_data = load_dataset("sentence-transformers/stsb", split="validation")
        self.sentences1 = self.stsb_data["sentence1"]
        self.sentences2 = self.stsb_data["sentence2"]
        self.gold_scores = np.array(self.stsb_data["score"])
        self.high_score_threshold = high_score_threshold

        self.test_data = load_dataset("sentence-transformers/stsb", split="test")
        self.test_sentences1 = self.test_data["sentence1"]
        self.test_sentences2 = self.test_data["sentence2"]
        self.test_gold_scores = np.array(self.test_data["score"])

    @torch.no_grad()
    def eval(self, system, prefix: str | None = None, **kwargs) -> dict[str, float]:
        prefix_str = f"{prefix}/" if prefix else ""
        device = system.device

        emb1 = system.encode(self.sentences1, **kwargs)
        emb2 = system.encode(self.sentences2, **kwargs)

        if isinstance(emb1, np.ndarray):
            emb1 = torch.from_numpy(emb1)
        if isinstance(emb2, np.ndarray):
            emb2 = torch.from_numpy(emb2)

        emb1 = emb1.to(device)
        emb2 = emb2.to(device)

        # -------------------------------------------------------------
        # 1. Normalization & Cosine Similarity
        # -------------------------------------------------------------
        emb1_norm = F.normalize(emb1, p=2, dim=-1)
        emb2_norm = F.normalize(emb2, p=2, dim=-1)
        cosine_similarities = (emb1_norm * emb2_norm).sum(dim=-1).cpu().numpy()

        # Spearman & Pearson
        spearman_score, _ = spearmanr(self.gold_scores, cosine_similarities)
        pearson_score, _ = pearsonr(self.gold_scores, cosine_similarities)

        # -------------------------------------------------------------
        # Test Set Evaluation
        # -------------------------------------------------------------
        test_emb1 = system.encode(self.test_sentences1, **kwargs)
        test_emb2 = system.encode(self.test_sentences2, **kwargs)

        if isinstance(test_emb1, np.ndarray):
            test_emb1 = torch.from_numpy(test_emb1)
        if isinstance(test_emb2, np.ndarray):
            test_emb2 = torch.from_numpy(test_emb2)

        test_emb1 = test_emb1.to(device)
        test_emb2 = test_emb2.to(device)

        test_emb1_norm = F.normalize(test_emb1, p=2, dim=-1)
        test_emb2_norm = F.normalize(test_emb2, p=2, dim=-1)
        test_cosine_similarities = (
            (test_emb1_norm * test_emb2_norm).sum(dim=-1).cpu().numpy()
        )

        test_spearman_score, _ = spearmanr(
            self.test_gold_scores, test_cosine_similarities
        )
        test_pearson_score, _ = pearsonr(
            self.test_gold_scores, test_cosine_similarities
        )

        # -------------------------------------------------------------
        # 2. Alignment (GPU)
        # -------------------------------------------------------------
        pos_mask = torch.tensor(
            self.gold_scores >= self.high_score_threshold, device=device
        )
        if pos_mask.sum() > 0:
            diff = emb1_norm[pos_mask] - emb2_norm[pos_mask]
            alignment = float((diff**2).sum(dim=-1).mean().item())
        else:
            alignment = 0.0

        # -------------------------------------------------------------
        # 3. Uniformity
        # -------------------------------------------------------------
        all_emb = torch.cat([emb1, emb2], dim=0)
        all_norm = torch.cat([emb1_norm, emb2_norm], dim=0)
        N, D = all_norm.shape

        sim_matrix = all_norm @ all_norm.T
        sq_distances = torch.clamp(2.0 - 2.0 * sim_matrix, min=0.0, max=4.0)
        sq_distances.fill_diagonal_(float("inf"))

        uniformity = float(
            torch.log(
                torch.exp(-2.0 * sq_distances).sum() / (N * (N - 1) + 1e-8)
            ).item()
        )

        # -------------------------------------------------------------
        # 4. Dimensional Collapse & SVD (GPU)
        # -------------------------------------------------------------
        centered = all_emb - all_emb.mean(dim=0, keepdim=True)

        _, S, _ = torch.linalg.svd(centered, full_matrices=False)
        S_sum = S.sum() + 1e-8

        top1_sv_ratio = float((S[0] / S_sum).item())

        p = S / S_sum
        entropy = -torch.sum(p * torch.log(p + 1e-12))
        effective_rank = float(torch.exp(entropy).item())

        cum_var_ratio = torch.cumsum(S**2, dim=0) / (torch.sum(S**2) + 1e-8)
        dim_90_pct = float(
            torch.searchsorted(cum_var_ratio, torch.tensor(0.90, device=device)).item()
            + 1
        )

        # -------------------------------------------------------------
        # 5. Anisotropy & Space Statistics (GPU)
        # -------------------------------------------------------------
        off_diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)
        avg_random_cos_sim = float(sim_matrix[off_diag_mask].mean().item())

        std_per_dim = torch.std(all_emb, dim=0)
        dead_dims_count = float((std_per_dim < 1e-4).sum().item())

        cov = (centered.T @ centered) / (N - 1)
        cov.fill_diagonal_(0.0)
        cov_offdiag_abs_mean = float(cov.abs().mean().item())

        return {
            # Performance
            f"eval/{prefix_str}spearman": float(spearman_score),
            f"eval/{prefix_str}pearson": float(pearson_score),
            f"eval/{prefix_str}test_spearman": float(test_spearman_score),
            f"eval/{prefix_str}test_pearson": float(test_pearson_score),
            # Representation Space
            f"repr/{prefix_str}alignment": alignment,
            f"repr/{prefix_str}uniformity": uniformity,
            # Dimensional Collapse
            f"collapse/{prefix_str}top1_sv_ratio": top1_sv_ratio,
            f"collapse/{prefix_str}effective_rank": effective_rank,
            f"collapse/{prefix_str}dim_90pct_count": dim_90_pct,
            f"collapse/{prefix_str}dead_dims_count": dead_dims_count,
            # Anisotropy
            f"anisotropy/{prefix_str}avg_random_cos_sim": avg_random_cos_sim,
            f"anisotropy/{prefix_str}cov_offdiag_abs_mean": cov_offdiag_abs_mean,
            f"anisotropy/{prefix_str}feature_std_mean": float(
                std_per_dim.mean().item()
            ),
            f"anisotropy/{prefix_str}norm_mean": float(
                torch.linalg.norm(all_emb, dim=1).mean().item()
            ),
        }
