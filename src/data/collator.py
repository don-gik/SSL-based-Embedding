import random


class RepetitionCollator:
    def __init__(self, tokenizer, rep_prob=0.1):
        self.tokenizer = tokenizer
        self.rep_prob = rep_prob

    def __call__(self, features):
        s_features = []
        t_features = []

        for f in features:
            input_ids = f["input_ids"]

            t_features.append({"input_ids": input_ids})

            # Word repetition augmentation
            aug_ids = []
            for token_id in input_ids:
                aug_ids.append(token_id)
                if token_id not in self.tokenizer.all_special_ids:
                    if random.random() < self.rep_prob:
                        aug_ids.append(token_id)

            s_features.append({"input_ids": aug_ids})

        # padding
        s_batch = self.tokenizer.pad(s_features, return_tensors="pt", padding=True)
        t_batch = self.tokenizer.pad(t_features, return_tensors="pt", padding=True)

        return {
            "s_input_ids": s_batch["input_ids"],
            "s_attention_mask": s_batch["attention_mask"],
            "t_input_ids": t_batch["input_ids"],
            "t_attention_mask": t_batch["attention_mask"],
        }
