from datasets import load_dataset
from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
)


class Evaluator:
    def __init__(self):
        self.stsb_data = load_dataset("sentence-transformers/stsb", split="test")

        self.stsb_evaluator = EmbeddingSimilarityEvaluator(
            sentences1=self.stsb_data["sentence1"],
            sentences2=self.stsb_data["sentence2"],
            scores=[score / 5.0 for score in self.stsb_data["score"]],
            name="stsb-test",
            main_similarity="cosine",
        )

    def eval(self, system):
        return self.stsb_evaluator(system)
