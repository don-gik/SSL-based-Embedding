import logging

import lightning as L
import torch
from omegaconf import DictConfig
from transformers import BertModel

from src.system.layer import replace_dropout_with_noise
from src.system.loss import simcse_loss

logger = logging.getLogger(__name__)


class SimCSENoiseSystem(L.LightningModule):
    def __init__(self, cfg: DictConfig, device_info: tuple[str, int | str, str, bool]):
        super().__init__()
        self.cfg = cfg

        accelerator, _, _, _ = device_info
        attn_mode = "sdpa" if accelerator == "gpu" else "eager"

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", attn_implementation=attn_mode
        )

        noise_std = cfg.get("noise_std", 0.01)
        replace_dropout_with_noise(self.bert, noise_std=noise_std)

        logger.info("SimCSE Noise System initialized.")

    def training_step(self, batch, batch_idx):
        if batch_idx % 1 == 0:
            logger.debug(f"Batch Index: {batch_idx}")

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        combined_input_ids = torch.cat([input_ids, input_ids], dim=0)
        combined_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

        outputs = self.bert(
            input_ids=combined_input_ids, attention_mask=combined_attention_mask
        )

        combined_embeddings = self.get_sentence_embedding(
            outputs, {"attention_mask": combined_attention_mask}
        )

        batch_size = input_ids.size(0)
        z1 = combined_embeddings[:batch_size]
        z2 = combined_embeddings[batch_size:]

        temp = self.cfg.get("temperature", 0.05)
        loss = simcse_loss(z1, z2, temperature=temp)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def get_sentence_embedding(self, outputs, batch):
        embeddings = outputs.last_hidden_state
        attention_mask = batch["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(embeddings).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.get("lr", 3e-5))
