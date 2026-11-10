import logging
import os

import nltk
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from src.data.collator import RepetitionShuffleCollator

logger = logging.getLogger(__name__)


def get_wikitext_sentence_dataloader(
    model_name="bert-base-uncased",
    batch_size=128,
    cache_dir="./.dataset_cache",
    num_workers=16,
    num_proc=8,
    repetition=False,
    shuffle=False,
    ncrops=3,
    rep_prob=0.05,
    shuffle_prob=0.05,
):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    raw_datasets = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir, split="train")

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

        for input_ids, attn_mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
            token_len = len(input_ids)
            if 10 <= token_len <= 60:
                filtered_input_ids.append(input_ids)
                filtered_attention_mask.append(attn_mask)

        return {
            "input_ids": filtered_input_ids,
            "attention_mask": filtered_attention_mask,
        }

    logger.info("Splitting into sentences and caching (train split only; validation is the STSB Evaluator)")
    # This map (nltk sent_tokenize + full tokenization) is the ONLY thing keeping the GPU idle
    # at startup. It is cached in cache_file_name; but any change to chunk_to_sentences /
    # tokenizer invalidates the fingerprint -> full re-run. num_proc parallelizes the re-run.
    train_dataset = raw_datasets.map(
        chunk_to_sentences,
        batched=True,
        remove_columns=["text"],
        num_proc=num_proc,
        cache_file_name=os.path.join(cache_dir, "train_103_sent.cache"),
    )

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Augmentation strength is what carries the DINO signal: with near-identity crops
    # (prob 0.05) every crop matches its own teacher trivially, the loss carries no
    # per-sentence information, and the backbone drifts toward a narrow cone (erank 142->40
    # observed at step 2k). repetition/shuffle are on/off gates; the *_prob values set strength.
    if not repetition:
        rep_prob = 0.0
    if not shuffle:
        shuffle_prob = 0.0
    data_collator = RepetitionShuffleCollator(
        tokenizer=tokenizer, rep_prob=rep_prob, shuffle_prob=shuffle_prob, ncrops=ncrops
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        pin_memory=True,  # GPU Server Opt
        num_workers=num_workers,
    )

    # No validation dataloader: validation is the Evaluator (STSB + backbone embeddings),
    # run directly from TheSystem.on_train_batch_end at eval_every_steps intervals.
    return train_loader
