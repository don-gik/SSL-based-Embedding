import logging
import os

import nltk
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)


def get_wikitext_sentence_dataloader(
    model_name="bert-base-uncased",
    batch_size=128,
    cache_dir="./.dataset_cache",
    num_workers=16,
    repetition=False,
):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    raw_datasets = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir
    )

    def chunk_to_sentences(batch):
        all_sentences = []
        for text in batch["text"]:
            text = text.strip()
            if not text or text.startswith(" = ") or text.endswith(" = "):
                continue

            sentences = nltk.sent_tokenize(text)
            for s in sentences:
                s = s.strip()
                if 5 <= len(s.split()) <= 80:
                    all_sentences.append(s)

        tokenized = tokenizer(all_sentences, truncation=False, add_special_tokens=True)

        filtered_input_ids = []
        filtered_attention_mask = []

        for input_ids, attn_mask in zip(
            tokenized["input_ids"], tokenized["attention_mask"]
        ):
            token_len = len(input_ids)
            if 10 <= token_len <= 60:
                filtered_input_ids.append(input_ids)
                filtered_attention_mask.append(attn_mask)

        return {
            "input_ids": filtered_input_ids,
            "attention_mask": filtered_attention_mask,
        }

    logger.info("Splitting into sentences and caching")
    processed_datasets = raw_datasets.map(
        chunk_to_sentences,
        batched=True,
        remove_columns=["text"],
        cache_file_names={
            "train": os.path.join(cache_dir, "train_103_sent.cache"),
            "validation": os.path.join(cache_dir, "val_103_sent.cache"),
            "test": os.path.join(cache_dir, "test_103_sent.cache"),
        },
    )

    processed_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

    if not repetition:
        from transformers import DataCollatorWithPadding

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, return_tensors="pt"
        )
    else:
        from src.data.collator import RepetitionShuffleCollator

        data_collator = RepetitionShuffleCollator(tokenizer=tokenizer, rep_prob=0.15)

    train_loader = DataLoader(
        processed_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,  # GPU Server Opt
        num_workers=num_workers,
    )

    return train_loader
