import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel, get_cosine_schedule_with_warmup

from src.system.eval import Evaluator
from src.system.loss import CostDeflatedOTLoss
from src.system.update import GrokfastEMA


class TheSystem(L.LightningModule):
    def __init__(self, cfg: DictConfig, device_info: tuple[str, int | str, str, bool]):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "tokenizer", "evaluator"])

        self.cfg = cfg
        self.ema_decay = cfg.get("ema_decay", 0.99)

        self.s_bert, self.t_bert, self.tokenizer = self.setup_bert(
            device_info, use_lora=False
        )

        hidden_dim = self.s_bert.config.hidden_size

        # def build_mlp(hidden_dim):
        #     return nn.Sequential(
        #         nn.Linear(hidden_dim, hidden_dim),
        #         nn.LayerNorm(hidden_dim),
        #         nn.GELU(),
        #         nn.Dropout(self.s_bert.config.hidden_dropout_prob),
        #         nn.Linear(hidden_dim, hidden_dim),
        #     )

        # self.s_head = build_mlp(hidden_dim).train()
        # self.t_head = build_mlp(hidden_dim).eval()
        # self.predictor = build_mlp(hidden_dim).train()

        self.ot_loss_fn = CostDeflatedOTLoss(
            hidden_dim=hidden_dim,
            k=cfg.get("ot_k", 3),
            alpha=cfg.get("alpha", 0.15),
            tau=cfg.get("tau", 0.1),
            gamma=cfg.get("gamma", 0.9),
            sinkhorn_eps=cfg.get("sinkhorn_eps", 0.15),
            sinkhorn_iters=cfg.get("sinkhorn_iters", 10),
            center_momentum=cfg.get("center_momentum", 0.9),
        )

        self.grokfast = GrokfastEMA()

        self.evaluator = Evaluator()

    def on_before_optimizer_step(self, optimizer):
        self.grokfast.apply(self)

    def training_step(self, batch, batch_idx):
        # Students
        s_outs = self.s_bert(
            input_ids=batch["s_input_ids"], attention_mask=batch["s_attention_mask"]
        )
        s_embed = self.get_sentence_embedding(
            s_outs, {"attention_mask": batch["s_attention_mask"]}
        )

        # Teacher
        with torch.no_grad():
            t_outs = self.t_bert(
                input_ids=batch["t_input_ids"], attention_mask=batch["t_attention_mask"]
            )
            t_embed = self.get_sentence_embedding(
                t_outs, {"attention_mask": batch["t_attention_mask"]}
            )

            # batch_center = t_embed.mean(dim=0, keepdim=True)
            # self.t_center = self.t_center * self.center_momentum + batch_center * (
            #     1.0 - self.center_momentum
            # )
            # t_embed = t_embed - self.t_center

        # p_embed = self.predictor(s_embed)

        # target = torch.ones(s_embed.size(0), device=p_embed.device)
        loss = self.ot_loss_fn(s_embed, t_embed)

        if self.global_step % 200 == 0 and self.ot_loss_fn.last_Q is not None:
            self._log_q_heatmap(self.ot_loss_fn.last_Q)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def _log_q_heatmap(self, Q: torch.Tensor):
        if not hasattr(self.logger, "experiment") or not hasattr(
            self.logger.experiment, "add_figure"
        ):
            return

        Q_np = Q.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(5, 4.5), dpi=100)
        im = ax.imshow(Q_np, cmap="magma", aspect="equal", interpolation="nearest")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        ax.set_title(
            f"Target Q Matrix (Step {self.global_step})", fontsize=10, fontweight="bold"
        )
        ax.set_xlabel("Teacher Samples", fontsize=8)
        ax.set_ylabel("Student Samples", fontsize=8)
        ax.tick_params(labelsize=8)
        plt.tight_layout()

        self.logger.experiment.add_figure(
            "OT/Q_Assignment", fig, global_step=self.global_step
        )
        plt.close(fig)

    def setup_bert(
        self, device_info, use_lora: bool = False
    ) -> tuple[BertModel, AutoTokenizer]:
        accelerator, _, _, _ = device_info
        attn_mode = "sdpa" if accelerator == "gpu" else "eager"
        model_name = self.cfg.get("model_name", "bert-base-uncased")

        def build_model():
            model = BertModel.from_pretrained(
                model_name,
                attn_implementation=attn_mode,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                output_hidden_states=True,
            )

            if use_lora:
                # --- LoRA ---
                for p in model.parameters():
                    p.requires_grad = False

                for name, p in model.named_parameters():
                    if "LayerNorm" in name or "bias" in name:
                        p.requires_grad = True

                peft_config = LoraConfig(
                    r=self.cfg.get("lora_r", 16),
                    lora_alpha=self.cfg.get("lora_alpha", 16),
                    target_modules=["query", "value"],
                    layers_to_transform=list(
                        range(
                            model.config.num_hidden_layers - 8,
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
            else:
                # --- FFT (Full Fine-Tuning) ---
                for p in model.parameters():
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
                t_p.data = s_p.data.clone()

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

            # # Head
            # for s, t in zip(self.s_head.parameters(), self.t_head.parameters()):
            #     t.data.mul_(self.ema_decay).add_(s.data, alpha=1.0 - self.ema_decay)

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
            trainable_params, lr=self.cfg.get("lr", 5e-5), weight_decay=0.05
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
        # metrics.update(self.evaluator.eval(self, prefix="head", use_head=True))
        metrics.update(self.evaluator.eval(self, prefix="backbone"))
        self.log_dict(metrics, prog_bar=True, on_epoch=True)

    @torch.no_grad()
    def encode(
        self,
        sentences: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = False,
        **kwargs,
    ) -> torch.Tensor | np.ndarray:
        self.eval()
        # use_head = kwargs.get("use_head", False)
        all_embeddings = []

        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding sentences")

        for i in iterator:
            batch_text = sentences[i : i + batch_size]

            inputs = self.tokenizer(
                batch_text, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            s_bert_outs = self.s_bert(**inputs)
            embeddings = self.get_sentence_embedding(s_bert_outs, inputs)

            embeddings = F.normalize(embeddings, p=2, dim=-1)

            all_embeddings.append(embeddings)

        full_embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_numpy:
            return full_embeddings.cpu().numpy()

        return full_embeddings
