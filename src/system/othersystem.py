import logging

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoTokenizer, BertModel

from src.system.eval import Evaluator
from src.system.layer import BertAttentionHead
from src.system.loss import DinoLoss

logger = logging.getLogger(__name__)


class OtherSystem(L.LightningModule):
    def __init__(self, cfg: DictConfig, device_info: tuple[str, int | str, str, bool]):
        super().__init__()
        self.save_hyperparameters(
            ignore=["cfg", "bert", "tokenizer", "dino_loss", "var_loss", "cov_loss"]
        )

        self.cfg = cfg
        self.ema_decay = cfg.get("ema_decay", 0.995)

        self.bert, self.tokenizer = self.setup_bert(device_info)

        hidden_dim = self.bert.config.hidden_size
        vocab_size = self.bert.config.vocab_size

        self.s_head = BertAttentionHead(
            self.bert.config, projection_dim=hidden_dim
        ).train()
        self.t_head = BertAttentionHead(
            self.bert.config, projection_dim=hidden_dim, orth=True
        ).eval()

        self.cov_weight = 0.05
        self.var_weight = 1.0

        self.dino_loss = self.setup_loss(vocab_size)

        self.evaluator = Evaluator()

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        comb_input_ids = torch.cat([input_ids, input_ids], dim=0)
        comb_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

        with torch.no_grad():
            output = self.bert(
                input_ids=comb_input_ids, attention_mask=comb_attention_mask
            ).last_hidden_state

        s_output = self.s_head(output, comb_attention_mask)
        with torch.no_grad():
            t_output = self.t_head(output, comb_attention_mask)

        s_embed = self.get_sentence_embedding(
            s_output, {"attention_mask": comb_attention_mask}
        )
        with torch.no_grad():
            t_embed = self.get_sentence_embedding(
                t_output, {"attention_mask": comb_attention_mask}
            )

        word_embeddings_weight = self.bert.embeddings.word_embeddings.weight

        s_logit = F.linear(s_embed, word_embeddings_weight)
        with torch.no_grad():
            t_logit = F.linear(t_embed, word_embeddings_weight)

        s_z1, s_z2 = torch.chunk(s_logit, 2, dim=0)
        with torch.no_grad():
            t_z1, t_z2 = torch.chunk(t_logit, 2, dim=0)

        loss = (self.dino_loss(s_z1, t_z2) + self.dino_loss(s_z2, t_z1)) * 0.5
        self.dino_loss.update_center(t_logit)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def setup_bert(self, device_info) -> tuple[BertModel, AutoTokenizer]:
        accelerator, _, _, _ = device_info
        attn_mode = "sdpa" if accelerator == "gpu" else "eager"

        bert = BertModel.from_pretrained(
            "bert-base-uncased",
            attn_implementation=attn_mode,
            hidden_dropout_prob=0.15,
            attention_probs_dropout_prob=0.15,
        ).train()

        for param in bert.parameters():
            param.requires_grad = False

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        return (bert, tokenizer)

    def setup_loss(self, vocab_size: int) -> DinoLoss:
        return DinoLoss(vocab_size)

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
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(trainable_params, lr=self.cfg.get("lr", 3e-5))

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

            bert_output = self.bert(**inputs).last_hidden_state
            head_output = self.s_head(bert_output, inputs["attention_mask"])
            embeddings = self.get_sentence_embedding(head_output, inputs)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)
