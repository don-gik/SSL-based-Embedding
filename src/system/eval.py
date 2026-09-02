import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr


class Evaluator:
    def __init__(self):
        self.stsb_data = load_dataset("sentence-transformers/stsb", split="test")
        self.sentences1 = self.stsb_data["sentence1"]
        self.sentences2 = self.stsb_data["sentence2"]
        self.gold_scores = np.array(self.stsb_data["score"])

    def eval(self, system, prefix: str | None = None, **kwargs):
        embeddings1 = system.encode(self.sentences1, **kwargs)
        embeddings2 = system.encode(self.sentences2, **kwargs)

        # Cos sim + Spearman
        emb1_norm = embeddings1 / np.maximum(
            np.linalg.norm(embeddings1, axis=1, keepdims=True), 1e-12
        )
        emb2_norm = embeddings2 / np.maximum(
            np.linalg.norm(embeddings2, axis=1, keepdims=True), 1e-12
        )
        cosine_similarities = np.sum(emb1_norm * emb2_norm, axis=1)
        spearman_score, _ = spearmanr(self.gold_scores, cosine_similarities)

        # Space analysis
        all_emb = np.concatenate([embeddings1, embeddings2], axis=0)
        N, D = all_emb.shape

        all_norm = all_emb / np.maximum(
            np.linalg.norm(all_emb, axis=1, keepdims=True), 1e-12
        )
        centered = all_emb - np.mean(all_emb, axis=0, keepdims=True)
        cov = (centered.T @ centered) / (N - 1)
        np.fill_diagonal(cov, 0)

        return {
            f"eval/{prefix}_spearman": float(spearman_score),
            f"repr/{prefix}_anisotropy": float(np.mean(all_norm @ all_norm.T)),
            f"repr/{prefix}_feature_std_mean": float(np.mean(np.std(all_emb, axis=0))),
            f"repr/{prefix}_cov_offdiag_abs_mean": float(np.mean(np.abs(cov))),
            f"repr/{prefix}_norm_mean": float(np.mean(np.linalg.norm(all_emb, axis=1))),
        }
