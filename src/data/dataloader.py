import logging
import os

import nltk
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

logger = logging.getLogger(__name__)


def get_wikitext_sentence_dataloader(
    model_name="bert-base-uncased", batch_size=32, cache_dir="./.dataset_cache"
):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=cache_dir)

    def chunk_to_sentences(batch):
        all_sentences = []
        for text in batch["text"]:
            text = text.strip()
            if not text or text.startswith(" = ") or text.endswith(" = "):
                continue

            sentences = nltk.sent_tokenize(text)
            all_sentences.extend(sentences)

        tokenized = tokenizer(
            all_sentences, truncation=True, max_length=512, add_special_tokens=True
        )
        return tokenized

    logger.info("Splitting into sentences and caching")
    processed_datasets = raw_datasets.map(
        chunk_to_sentences,
        batched=True,
        remove_columns=["text"],
        cache_file_names={
            "train": os.path.join(cache_dir, "train_sent.cache"),
            "validation": os.path.join(cache_dir, "val_sent.cache"),
            "test": os.path.join(cache_dir, "test_sent.cache"),
        },
    )

    processed_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        processed_datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,  # GPU Server Opt
    )

    return train_loader
