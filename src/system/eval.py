import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr


class Evaluator:
    def __init__(self):
        self.stsb_data = load_dataset("sentence-transformers/stsb", split="test")
        self.sentences1 = self.stsb_data["sentence1"]
        self.sentences2 = self.stsb_data["sentence2"]
        self.gold_scores = np.array(self.stsb_data["score"])

    def eval(self, system):
        embeddings1 = system.encode(self.sentences1)
        embeddings2 = system.encode(self.sentences2)

        emb1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        emb2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        cosine_similarities = np.sum(emb1_norm * emb2_norm, axis=1)

        spearman_score, _ = spearmanr(self.gold_scores, cosine_similarities)
        return {"cosine_spearman": spearman_score}
