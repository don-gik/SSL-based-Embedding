import copy
import logging

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoTokenizer, BertModel

from src.system.eval import Evaluator
from src.system.layer import change_noise_std, replace_dropout_with_noise
from src.system.loss import GramDiLoss

logger = logging.getLogger(__name__)


class GramDiNoiseSystem(L.LightningModule):
    def __init__(self, cfg: DictConfig, device_info: tuple[str, int | str, str, bool]):
        super().__init__()
        self.cfg = cfg
        self.ema_decay = cfg.get("ema_decay", 0.99)

        accelerator, _, _, _ = device_info
        attn_mode = "sdpa" if accelerator == "gpu" else "eager"

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", attn_implementation=attn_mode
        ).train()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        noise_std = cfg.get("noise_std", 0.05)
        replace_dropout_with_noise(self.bert, noise_std=noise_std)

        self.teacher_bert = copy.deepcopy(self.bert)
        self.teacher_bert.requires_grad_(False)

        teacher_noise_std = cfg.get("teacher_noise_std", 0.01)
        change_noise_std(self.teacher_bert, new_std=teacher_noise_std)
        self.teacher_bert.eval()

        self.evaluator = Evaluator()
        self.model_card_data = None

        self.gramdi_loss = GramDiLoss()

        logger.info("SimCSE Noise System initialized.")

    def on_fit_start(self):
        self.teacher_bert = self.teacher_bert.to(self.device)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        s_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            t_output = self.teacher_bert(
                input_ids=input_ids, attention_mask=attention_mask
            )

        s_embed = self.get_sentence_embedding(
            s_output, {"attention_mask": attention_mask}
        )
        with torch.no_grad():
            t_embed = self.get_sentence_embedding(
                t_output, {"attention_mask": attention_mask}
            )

        s_norm = F.normalize(s_embed, p=2, dim=-1)
        t_norm = F.normalize(t_embed, p=2, dim=-1)

        z_student = torch.matmul(s_norm, s_norm.T)
        z_teacher = torch.matmul(t_norm, t_norm.T)

        loss = self.gramdi_loss(z_student, z_teacher)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            # backbone
            for s, t in zip(self.bert.parameters(), self.teacher_bert.parameters()):
                t.data.mul_(self.ema_decay).add_(s.data, alpha=1.0 - self.ema_decay)

    def get_sentence_embedding(self, outputs, batch):
        embeddings = outputs.last_hidden_state
        attention_mask = batch["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(embeddings).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.get("lr", 3e-5))

    def validation_step(self, batch, batch_idx):
        metrics = self.evaluator.eval(self)
        spearman_score = metrics["cosine_spearman"]

        self.log("val_stsb_spearman", spearman_score, prog_bar=True, on_epoch=True)
        logger.info(f"Step {self.global_step} STSb Spearman: {spearman_score:.4f}")

    def on_validation_start(self):
        self.bert.eval()

    def on_validation_end(self):
        self.bert.train()

    @torch.no_grad()
    def encode(
        self,
        sentences: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        self.eval()
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch_text = sentences[i : i + batch_size]

            inputs = self.tokenizer(
                batch_text, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.bert(**inputs)
            embeddings = self.get_sentence_embedding(outputs, inputs)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
