import logging

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, BertModel, get_cosine_schedule_with_warmup

from src.system.eval import Evaluator

logger = logging.getLogger(__name__)


class AnotherSystem(L.LightningModule):
    def __init__(self, cfg: DictConfig, device_info: tuple[str, int | str, str, bool]):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "tokenizer", "evaluator"])

        self.cfg = cfg
        self.ema_decay = cfg.get("ema_decay", 0.996)

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
        self.center_momentum = cfg.get("center_momentum", 0.95)

        self.evaluator = Evaluator()

    def training_step(self, batch, batch_idx):
        # Students
        s_outs = self.s_bert(
            input_ids=batch["s_input_ids"], attention_mask=batch["s_attention_mask"]
        )
        s_pooled = self.get_sentence_embedding(s_outs, batch)
        s_embed = self.s_head(s_pooled)  # [B, D]

        # Teacher
        with torch.no_grad():
            t_outs = self.t_bert(
                input_ids=batch["t_input_ids"], attention_mask=batch["t_attention_mask"]
            )
            t_pooled = self.get_sentence_embedding(t_outs, batch)
            t_embed = self.t_head(t_pooled)  # [B, D]

            batch_center = t_embed.mean(dim=0, keepdim=True)
            self.t_center = self.t_center * self.center_momentum + batch_center * (
                1.0 - self.center_momentum
            )
            t_embed = t_embed - self.t_center

        batch_size = s_embed.size(0)
        perm = torch.randperm(batch_size)

        lam = torch.rand(batch_size, 1, device=self.device) * 0.8 + 0.1

        s_mixed = lam * s_embed + (1.0 - lam) * s_embed[perm]

        # Predictor
        p_mixed = self.predictor(s_mixed)
        p_norm = F.normalize(p_mixed, p=2, dim=-1)

        t_norm_1 = F.normalize(t_embed, p=2, dim=-1)
        t_norm_2 = F.normalize(t_embed[perm], p=2, dim=-1)

        cos_1 = (p_norm * t_norm_1).sum(dim=-1, keepdim=True)
        cos_2 = (p_norm * t_norm_2).sum(dim=-1, keepdim=True)

        loss_calib_1 = F.mse_loss(cos_1, lam)
        loss_calib_2 = F.mse_loss(cos_2, 1.0 - lam)

        p_orig = F.normalize(self.predictor(s_embed), p=2, dim=-1)
        loss_cos_sim = -(p_orig * t_norm_1).sum(dim=-1).mean()

        loss = loss_cos_sim + 1.0 * (loss_calib_1 + loss_calib_2)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def setup_bert(self, device_info) -> tuple[BertModel, AutoTokenizer]:
        accelerator, _, _, _ = device_info
        attn_mode = "sdpa" if accelerator == "gpu" else "eager"
        model_name = self.cfg.get("model_name", "bert-base-uncased")

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
            model = get_peft_model(model, peft_config)

            for name, p in model.named_parameters():
                if "LayerNorm" in name or "bias" in name:
                    p.requires_grad = True

            return model

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
            # Backbone LoRA
            for s_p, t_p in zip(self.s_bert.parameters(), self.t_bert.parameters()):
                if s_p.requires_grad:
                    t_p.data.mul_(self.ema_decay).add_(
                        s_p.data, alpha=1.0 - self.ema_decay
                    )

            # Head
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
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.05),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_start(self):
        self.s_bert.eval()

    def on_validation_end(self):
        self.s_bert.train()

    def on_validation_epoch_end(self):
        metrics = {}
        metrics.update(self.evaluator.eval(self, prefix="head", use_head=True))
        metrics.update(self.evaluator.eval(self, prefix="backbone", use_head=False))
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
        use_head = kwargs.get("use_head", False)
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch_text = sentences[i : i + batch_size]

            inputs = self.tokenizer(
                batch_text, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            s_bert_outs = self.s_bert(**inputs)
            pooled = self.get_sentence_embedding(s_bert_outs, inputs)

            embeddings = self.s_head(pooled) if use_head else pooled
            embeddings = F.normalize(embeddings, p=2, dim=-1)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
