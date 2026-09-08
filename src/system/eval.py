import numpy as np
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr


class Evaluator:
    def __init__(self, high_score_threshold: float = 4.0):
        self.stsb_data = load_dataset("sentence-transformers/stsb", split="validation")
        self.sentences1 = self.stsb_data["sentence1"]
        self.sentences2 = self.stsb_data["sentence2"]
        self.gold_scores = np.array(self.stsb_data["score"])
        self.high_score_threshold = high_score_threshold

    def eval(self, system, prefix: str | None = None, **kwargs) -> dict[str, float]:
        prefix_str = f"{prefix}/" if prefix else ""

        embeddings1 = system.encode(self.sentences1, **kwargs)
        embeddings2 = system.encode(self.sentences2, **kwargs)

        # Normalization
        emb1_norm = embeddings1 / np.maximum(
            np.linalg.norm(embeddings1, axis=1, keepdims=True), 1e-8
        )
        emb2_norm = embeddings2 / np.maximum(
            np.linalg.norm(embeddings2, axis=1, keepdims=True), 1e-8
        )
        cosine_similarities = np.sum(emb1_norm * emb2_norm, axis=1)

        # Spearman & Pearson
        spearman_score, _ = spearmanr(self.gold_scores, cosine_similarities)
        pearson_score, _ = pearsonr(self.gold_scores, cosine_similarities)

        # Alignment
        pos_mask = self.gold_scores >= self.high_score_threshold
        if np.sum(pos_mask) > 0:
            diff = emb1_norm[pos_mask] - emb2_norm[pos_mask]
            alignment = float(np.mean(np.sum(diff**2, axis=1)))
        else:
            alignment = 0.0

        # Uniformity
        all_emb = np.concatenate([embeddings1, embeddings2], axis=0)
        N, D = all_emb.shape

        all_norm = all_emb / np.maximum(
            np.linalg.norm(all_emb, axis=1, keepdims=True), 1e-8
        )

        # clip + uniformity
        sq_distances = np.clip(2.0 - 2.0 * (all_norm @ all_norm.T), 0.0, 4.0)
        np.fill_diagonal(sq_distances, float("inf"))
        uniformity = float(
            np.log(np.sum(np.exp(-2.0 * sq_distances)) / (N * (N - 1) + 1e-8))
        )

        centered = all_emb - np.mean(all_emb, axis=0, keepdims=True)

        # SVD
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        S_sum = np.sum(S) + 1e-8

        # Top-1 sv ratio
        top1_sv_ratio = float(S[0] / S_sum)

        # Effective Rank
        p = S / S_sum
        entropy = -np.sum(p * np.log(p + 1e-12))
        effective_rank = float(np.exp(entropy))

        # 90% dim pct
        cum_var_ratio = np.cumsum(S**2) / (np.sum(S**2) + 1e-8)
        dim_90_pct = float(np.searchsorted(cum_var_ratio, 0.90) + 1)

        # Random cos sim
        off_diag_mask = ~np.eye(N, dtype=bool)
        avg_random_cos_sim = float(np.mean((all_norm @ all_norm.T)[off_diag_mask]))

        # std per dim
        std_per_dim = np.std(all_emb, axis=0)
        dead_dims_count = float(np.sum(std_per_dim < 1e-4))

        # Off diag cov
        cov = (centered.T @ centered) / (N - 1)
        np.fill_diagonal(cov, 0)
        cov_offdiag_abs_mean = float(np.mean(np.abs(cov)))

        return {
            # Performance
            f"eval/{prefix_str}spearman": float(spearman_score),
            f"eval/{prefix_str}pearson": float(pearson_score),
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
            f"anisotropy/{prefix_str}feature_std_mean": float(np.mean(std_per_dim)),
            f"anisotropy/{prefix_str}norm_mean": float(
                np.mean(np.linalg.norm(all_emb, axis=1))
            ),
        }
