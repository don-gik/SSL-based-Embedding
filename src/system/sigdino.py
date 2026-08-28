import copy
import logging

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer, BertModel

from src.system.eval import Evaluator
from src.system.layer import change_noise_std, replace_dropout_with_noise
from src.system.loss import CovarianceLoss, SigmoidDinoLoss, VarianceLoss

logger = logging.getLogger(__name__)


class SigDinoNoiseSystem(L.LightningModule):
    def __init__(self, cfg: DictConfig, device_info: tuple[str, int | str, str, bool]):
        super().__init__()
        self.cfg = cfg
        self.ema_decay = cfg.get("ema_decay", 0.995)

        accelerator, _, _, _ = device_info
        attn_mode = "sdpa" if accelerator == "gpu" else "eager"

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", attn_implementation=attn_mode
        ).train()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        out_dim = self.bert.config.hidden_size

        noise_std = cfg.get("noise_std", 0.05)
        replace_dropout_with_noise(self.bert, noise_std=noise_std)

        self.teacher_bert = copy.deepcopy(self.bert)
        self.teacher_bert.requires_grad_(False)

        teacher_noise_std = cfg.get("teacher_noise_std", 0.01)
        change_noise_std(self.teacher_bert, new_std=teacher_noise_std)
        self.teacher_bert.train()

        self.evaluator = Evaluator()
        self.model_card_data = None

        self.sigdinoloss = SigmoidDinoLoss(out_dim=out_dim)
        self.covarianceloss = CovarianceLoss()
        self.varianceloss = VarianceLoss()

        self.cov_weight = 0.5
        self.var_weight = 3.0

        logger.info("Sigmoid Dino Noise System initialized.")

    def on_fit_start(self):
        self.teacher_bert = self.teacher_bert.to(self.device)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        output1 = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        with torch.no_grad():
            output2 = self.teacher_bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        embedding1 = self.get_sentence_embedding(
            output1, {"attention_mask": attention_mask}
        )
        with torch.no_grad():
            embedding2 = self.get_sentence_embedding(
                output2, {"attention_mask": attention_mask}
            )

        dino_loss = self.sigdinoloss(embedding1, embedding2, self.global_step)
        cov_loss = self.covarianceloss(embedding1) * self.cov_weight
        var_loss = self.varianceloss(embedding1) * self.var_weight
        loss = dino_loss + cov_loss + var_loss

        self.log("dino_loss", dino_loss)
        self.log("cov_loss", cov_loss)
        self.log("var_loss", var_loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
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
        pass

    def on_validation_start(self):
        self.bert.eval()

    def on_validation_end(self):
        self.bert.train()

    def on_validation_epoch_end(self):
        metrics = self.evaluator.eval(self)
        self.log_dict(metrics, prog_bar=True, on_epoch=True)

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
