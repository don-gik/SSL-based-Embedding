import random


class RepetitionShuffleCollator:
    def __init__(self, tokenizer, rep_prob=0.05, shuffle_prob=0.05, ncrops=3, n_teacher_views=2):
        self.tokenizer = tokenizer
        self.rep_prob = rep_prob
        self.shuffle_prob = shuffle_prob
        self.ncrops = ncrops
        self.n_teacher_views = n_teacher_views

    def _augment(self, input_ids):
        rep_ids = []
        for token_id in input_ids:
            rep_ids.append(token_id)
            if token_id not in self.tokenizer.all_special_ids:
                if random.random() < self.rep_prob:
                    rep_ids.append(token_id)

        aug_ids = rep_ids.copy()
        i = 0
        while i < len(aug_ids):
            if aug_ids[i] not in self.tokenizer.all_special_ids and random.random() < self.shuffle_prob:
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

        return aug_ids

    def __call__(self, features):
        # Crop-major layout (required by DINOLoss.chunk): all crop k of every sample, then crop k+1.
        s_features = [{"input_ids": self._augment(f["input_ids"])} for _ in range(self.ncrops) for f in features]
        t_features = [
            {"input_ids": self._augment(f["input_ids"])} for _ in range(self.n_teacher_views) for f in features
        ]

        s_batch = self.tokenizer.pad(s_features, return_tensors="pt", padding=True)
        t_batch = self.tokenizer.pad(t_features, return_tensors="pt", padding=True)

        return {
            "s_input_ids": s_batch["input_ids"],
            "s_attention_mask": s_batch["attention_mask"],
            "t_input_ids": t_batch["input_ids"],
            "t_attention_mask": t_batch["attention_mask"],
        }
