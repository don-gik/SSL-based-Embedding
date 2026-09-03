import random


class RepetitionShuffleCollator:
    def __init__(self, tokenizer, rep_prob=0.1, shuffle_prob=0.1):
        self.tokenizer = tokenizer
        self.rep_prob = rep_prob
        self.shuffle_prob = shuffle_prob

    def __call__(self, features):
        s_features = []
        t_features = []

        for f in features:
            input_ids = f["input_ids"]

            t_features.append({"input_ids": input_ids})

            # Word repetition augmentation
            rep_ids = []
            for token_id in input_ids:
                rep_ids.append(token_id)
                if token_id not in self.tokenizer.all_special_ids:
                    if random.random() < self.rep_prob:
                        rep_ids.append(token_id)

            # Local token shuffle
            aug_ids = rep_ids.copy()
            i = 0
            while i < len(aug_ids):
                if (
                    aug_ids[i] not in self.tokenizer.all_special_ids
                    and random.random() < self.shuffle_prob
                ):
                    window_size = random.choice([2, 3])
                    end = i

                    for j in range(i, min(i + window_size, len(aug_ids))):
                        if aug_ids[j] in self.tokenizer.all_special_ids:
                            break
                        end = j + 1

                    if end - i > 1:
                        window = aug_ids[i:end]
                        random.shuffle(window)
                        aug_ids[i:end] = window

                    i = end
                else:
                    i += 1

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
