import logging

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, BertModel

from src.system.eval import Evaluator

logger = logging.getLogger(__name__)


class AnotherSystem(L.LightningModule):
    def __init__(self, cfg: DictConfig, device_info: tuple[str, int | str, str, bool]):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "tokenizer", "evaluator"])

        self.cfg = cfg
        self.ema_decay = cfg.get("ema_decay", 0.995)

        self.s_bert, self.t_bert, self.tokenizer = self.setup_bert(device_info)

        hidden_dim = self.s_bert.config.hidden_size

        def build_mlp(hidden_dim):
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(self.s_bert.config.hidden_dropout_prob),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.s_head = build_mlp(hidden_dim).train()
        self.t_head = build_mlp(hidden_dim).eval()
        self.predictor = build_mlp(hidden_dim).train()

        self.register_buffer("t_center", torch.zeros(1, hidden_dim))
        self.center_momentum = cfg.get("center_momentum", 0.9)

        self.evaluator = Evaluator()

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        comb_input_ids = torch.cat([input_ids, input_ids], dim=0)
        comb_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

        # Student
        s_bert_outs = self.s_bert(
            input_ids=comb_input_ids, attention_mask=comb_attention_mask
        )
        s_pooled = self.get_sentence_embedding(
            s_bert_outs, {"attention_mask": comb_attention_mask}
        )  # [B, D]

        s_embed = self.s_head(F.dropout(s_pooled, p=0.1, training=True))  # [B, D]
        p_embed = self.predictor(s_embed)  # [B, D]

        # Teacher
        with torch.no_grad():
            t_bert_outs = self.t_bert(
                input_ids=comb_input_ids, attention_mask=comb_attention_mask
            )
            t_pooled = self.get_sentence_embedding(
                t_bert_outs, {"attention_mask": comb_attention_mask}
            )  # [B, D]

            t_embed = self.t_head(t_pooled)  # [B, D]

            batch_center = t_embed.mean(dim=0, keepdim=True)
            self.t_center = self.t_center * self.center_momentum + batch_center * (
                1.0 - self.center_momentum
            )
            t_embed = t_embed - self.t_center

        p_z1, p_z2 = torch.chunk(p_embed, 2, dim=0)
        with torch.no_grad():
            t_z1, t_z2 = torch.chunk(t_embed, 2, dim=0)

        def cosine_loss(p, z):
            p = F.normalize(p, p=2, dim=-1)
            z = F.normalize(z, p=2, dim=-1)
            return -(p * z).sum(dim=-1).mean()

        loss = (cosine_loss(p_z1, t_z2) + cosine_loss(p_z2, t_z1)) * 0.5

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def setup_bert(self, device_info) -> tuple[BertModel, AutoTokenizer]:
        accelerator, _, _, _ = device_info
        attn_mode = "sdpa" if accelerator == "gpu" else "eager"
        model_name = self.cfg.get("model_name", "bert-large-uncased")

        def build_model():
            model = BertModel.from_pretrained(
                model_name,
                attn_implementation=attn_mode,
                hidden_dropout_prob=0.15,
                attention_probs_dropout_prob=0.15,
                output_hidden_states=True,
            )

            for p in model.parameters():
                p.requires_grad = False

            for name, p in model.named_parameters():
                if "LayerNorm" in name or "bias" in name:
                    p.requires_grad = True

            peft_config = LoraConfig(
                r=self.cfg.get("lora_r", 8),
                lora_alpha=self.cfg.get("lora_alpha", 16),
                target_modules=["query", "value"],
                layers_to_transform=list(
                    range(
                        model.config.num_hidden_layers - 4,
                        model.config.num_hidden_layers,
                    )
                ),
                lora_dropout=0.05,
                bias="none",
            )
            return get_peft_model(model, peft_config)

        s_bert = build_model().train()
        t_bert = build_model().eval()

        for (s_name, s_p), (t_name, t_p) in zip(
            s_bert.named_parameters(), t_bert.named_parameters()
        ):
            if not s_p.requires_grad:
                t_p.data = s_p.data  # Shares same weight when requiring grad
            else:
                t_p.requires_grad = False

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        return s_bert, t_bert, tokenizer

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            # head
            for s, t in zip(self.s_head.parameters(), self.t_head.parameters()):
                t.data.mul_(self.ema_decay).add_(s.data, alpha=1.0 - self.ema_decay)

    def get_sentence_embedding(self, outputs, batch):
        if hasattr(outputs, "last_hidden_state"):
            embeddings = outputs.last_hidden_state
        elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
            embeddings = outputs["last_hidden_state"]
        else:
            embeddings = outputs

        attention_mask = batch["attention_mask"]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(embeddings).float()

        sum_embeddings = torch.sum(embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.cfg.get("lr", 5e-5), weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_steps, eta_min=1e-6
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

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

            s_bert_outs = self.t_bert(**inputs)
            pooled = self.get_sentence_embedding(s_bert_outs, inputs)
            embeddings = self.s_head(pooled)

            embeddings = F.normalize(embeddings, p=2, dim=-1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
