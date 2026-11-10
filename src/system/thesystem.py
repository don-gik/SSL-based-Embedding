import logging
import time

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel, get_cosine_schedule_with_warmup

from src.system.eval import Evaluator
from src.system.layer import DINOHead
from src.system.loss import DINOLoss

logger = logging.getLogger(__name__)


class EvalCallback(L.Callback):
    """Runs the STSB evaluation at a fixed step interval from a callback registered BEFORE
    ModelCheckpoint. Lightning calls callback hooks before the LightningModule hook, so doing
    this in TheSystem.on_train_batch_end meant ModelCheckpoint ran with an empty monitor and
    never saved. Registered order in trainer callbacks must be [EvalCallback, ModelCheckpoint]."""

    def __init__(self, eval_every_steps: int, baseline: bool = True):
        self.eval_every_steps = eval_every_steps
        self.baseline = baseline

    def on_train_start(self, trainer, pl_module):
        # Reference point at step 0: untrained-student score through the SAME pipeline.
        # Raw-cosine STSB of bert-base-uncased is ~0.46 (test) / ~0.50 (val) - SentEval's
        # 80+ numbers use the BLEVP projection probe and often STS-finetuned checkpoints,
        # so compare training drift against this baseline, not against SentEval tables.
        if self.baseline:
            logger.info("[eval] step-0 baseline eval starting (before first batch)")
            pl_module._run_evaluation(self_log=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.eval_every_steps > 0 and trainer.global_step > 0 and trainer.global_step % self.eval_every_steps == 0:
            pl_module._run_evaluation()


# from src.system.update import GrokfastEMA


class TheSystem(L.LightningModule):
    def __init__(self, cfg: DictConfig, device_info: tuple[str, int | str, str, bool]):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "tokenizer", "evaluator"])

        self.cfg = cfg
        self.ema_decay = cfg.get("ema_decay", 0.996)

        self.s_bert, self.t_bert, self.tokenizer = self.setup_bert(
            device_info, use_lora=bool(self.cfg.get("use_lora", False))
        )

        bert_hidden_dim = self.s_bert.config.hidden_size
        bottleneck_dim = cfg.get("bottleneck_dim", 256)
        hidden_dim = cfg.get("hidden_dim", 1024)
        out_dim = cfg.get("out_dim", 4096)

        ncrops = cfg.get("ncrops", 3)
        warmup_teacher_temp = cfg.get("warmup_teacher_temp", 0.1)
        teacher_temp = cfg.get("teacher_temp", 0.1)
        warmup_teacher_temp_epochs = cfg.get("warmup_teacher_temp_epochs", 0)
        student_temp = cfg.get("student_temp", 0.2)

        # DINO's BatchNorm variant (main_dino.py: DINOHead(use_bn=True)). The head had NO
        # normalization except the L2 normalize before the cosine last layer, which deletes
        # magnitude and leaves only a narrow angular cap -> every sentence gets the same
        # prototype logits -> teacher target constant -> backbone drifted in the loss's
        # unconstrained directions (erank 141->87 at step 2750). BN equalizes per-dim batch
        # scale across the MLP and is one of DINO's ablated anti-collapse mechanisms.
        head_use_bn = bool(cfg.get("head_use_bn", True))
        logger.info(f"DINOHead use_bn={head_use_bn} (student train-mode BN, teacher eval-mode BN)")

        self.s_head = DINOHead(
            in_dim=bert_hidden_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            use_bn=head_use_bn,
        ).train()
        self.t_head = DINOHead(
            in_dim=bert_hidden_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            use_bn=head_use_bn,
        ).eval()

        self.loss = DINOLoss(
            out_dim=out_dim,
            ncrops=ncrops,
            warmup_teacher_temp=warmup_teacher_temp,
            teacher_temp=teacher_temp,
            warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
            nepochs=cfg.get("epochs", 30),
            student_temp=student_temp,
        )

        # self.grokfast = GrokfastEMA(
        #     alpha=cfg.get("grokfast_alpha", 0.98),
        #     lamb=cfg.get("grokfast_lamb", 2.0),
        # )

        self.evaluator = Evaluator(cache_dir=cfg.get("cache_dir", "./.dataset_cache"))

    # def on_before_optimizer_step(self, optimizer):
    #     self.grokfast.apply(self)

    def training_step(self, batch, batch_idx):
        # Students: [ncrops*B, out_dim] (crop-major, matches collator order)
        s_outs = self.s_bert(input_ids=batch["s_input_ids"], attention_mask=batch["s_attention_mask"])
        s_embed = self.get_sentence_embedding(s_outs, {"attention_mask": batch["s_attention_mask"]})

        # Teacher: [2*B, out_dim] (2 augmented views, crop-major)
        with torch.no_grad():
            t_outs = self.t_bert(input_ids=batch["t_input_ids"], attention_mask=batch["t_attention_mask"])
            t_embed = self.get_sentence_embedding(t_outs, {"attention_mask": batch["t_attention_mask"]})

        # Heads + loss must be fp32 end-to-end. With bf16-mixed autocast, logits are *stored* as
        # bf16 and the student/teacher logit gap delta is erased by quantization before any later
        # .float() can recover it: measured residual grad at perfect match is ~1.0 (bf16) vs ~2e-7
        # (fp32) -> Adam drifts forever (softmax cancellation). The heads are tiny; fp32 is free.
        with torch.autocast(device_type=self.device.type, enabled=False):
            s_logit = self.s_head(s_embed.float())  # [ncrops*B, out_dim]
            with torch.no_grad():
                t_logit = self.t_head(t_embed.float())  # [2*B, out_dim]
            loss = self.loss(s_logit, t_logit, self.current_epoch)

        # batch_size = number of unique sentences (teacher has 2 views per sentence)
        B = batch["t_input_ids"].size(0) // 2
        # Align TB x-axis with eval scalars: inside training_step global_step is not yet
        # incremented (0-based), while eval's add_scalar runs in on_train_batch_end (1-based).
        # Lightning's logger would flush train_loss at the 0-based index, so write it manually.
        if self.logger is not None:
            self.logger.experiment.add_scalar("train_loss", loss.detach().float(), self.global_step + 1)
        # explicit on_step/on_epoch: unspecified args default to on_step=True in the training loop,
        # which forked names like "train_loss_epoch" into "train_loss_epoch_epoch" in TensorBoard.
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False, logger=False, batch_size=B)
        self.log("train_loss_epoch", loss, prog_bar=False, on_step=False, on_epoch=True, logger=False, batch_size=B)
        return loss

    def on_train_epoch_end(self):
        # Write the epoch-mean with an x-axis aligned to global_step (Lightning's own epoch-end
        # flush lands one step early, breaking alignment with the eval scalars).
        mean = self.trainer.callback_metrics.get("train_loss_epoch")
        if mean is not None and self.logger is not None:
            self.logger.experiment.add_scalar("train_loss_epoch", float(mean), self.global_step)

    def on_train_end(self):
        # don't exit while the diagnostics worker thread still has pending figures to serialize
        q = getattr(self.evaluator, "_diag_queue", None)
        if q is not None:
            q.join()

    def setup_bert(self, device_info, use_lora: bool = False) -> tuple[BertModel, AutoTokenizer]:
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

        for (s_name, s_p), (t_name, t_p) in zip(s_bert.named_parameters(), t_bert.named_parameters()):
            if not s_p.requires_grad:
                t_p.data = s_p.data  # Shares same weight when requiring grad
            else:
                t_p.data = s_p.data.clone()

            t_p.requires_grad = False

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return s_bert, t_bert, tokenizer

    def _ema_momentum(self) -> float:
        # Original DINO momentum ramp: d = d0 + (1 - d0) * epoch / nepochs.
        # The lagging EMA teacher is DINO's PRIMARY anti-collapse mechanism; a fixed low
        # momentum lets the teacher chase the student and removes the stabilizing delay.
        nepochs = max(int(self.cfg.get("epochs", 30)), 1)
        d0 = self.ema_decay
        return min(1.0, d0 + (1.0 - d0) * (self.current_epoch / nepochs))

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            momentum = self._ema_momentum()
            # Backbone LoRA
            for s_p, t_p in zip(self.s_bert.parameters(), self.t_bert.parameters()):
                if s_p.requires_grad:
                    t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)

            # Head
            for s, t in zip(self.s_head.parameters(), self.t_head.parameters()):
                t.data.mul_(momentum).add_(s.data, alpha=1.0 - momentum)

            # BatchNorm running stats are BUFFERS, so the parameters() loop above never touches
            # them, and the teacher head is permanently .eval() -> its running stats would stay
            # at init (mean=0, var=1) forever while the student's converge. That would make the
            # EMA-identical heads numerically different functions (teacher BN = plain affine),
            # i.e. an arbitrary train/eval asymmetry in the middle of the distillation path.
            # The teacher consumes the same distribution as the student (EMA'd weights, same
            # data), so its BN should hold the current stats: copy, do not EMA them.
            # named_buffers() registration order is identical because both heads are built from
            # the same DINOHead signature; names are dotted ("mlp.1.running_mean") so zip is
            # safer than getattr(self.t_head, name).
            for (_, s_buf), (_, t_buf) in zip(self.s_head.named_buffers(), self.t_head.named_buffers()):
                t_buf.copy_(s_buf)

        # EMA update of the teacher happens every batch; validation is run by EvalCallback
        # (registered before ModelCheckpoint) so the monitor metric exists when checkpointing.

    def _run_evaluation(self, self_log: bool = True):
        t0 = time.perf_counter()
        diag_every = int(self.cfg.get("diag_every_evals", 1))
        self._eval_count = getattr(self, "_eval_count", 0) + 1
        # figures/histograms are the expensive part (matplotlib render + TB serialization ~3s) —
        # run them only every diag_every-th evaluation (eval #1, #1+N, ...).
        # run figures at eval #1, #1+N, #1+2N... (works for diag_every=1 too; `% N == 1` would
        # never fire when N==1)
        show_figs = (self._eval_count - 1) % diag_every == 0
        logger.info(f"[eval] starting step {self.global_step} (eval #{self._eval_count}, figures={show_figs})")
        exp = self.logger.experiment if (self.logger is not None and show_figs) else None
        metrics = self.evaluator.eval(self, prefix="backbone", experiment=exp, step=self.global_step)
        # metrics.update(self.evaluator.eval(self, prefix="head", use_head=True))
        metrics["eval/time_seconds"] = time.perf_counter() - t0
        if self.trainer is not None and self.trainer.global_rank == 0:
            logger.info(f"[eval] step {self.global_step}: evaluator took {metrics['eval/time_seconds']:.1f}s")

        for key, value in metrics.items():
            # Write scalars to TensorBoard directly: Lightning's log(logger=True) only flushes step
            # metrics every `log_every_n_steps` batches (default 50), which silently drops step-logged
            # metrics. logger=False keeps Lightning from double-writing.
            if self.logger is not None:
                self.logger.experiment.add_scalar(key, value, self.global_step)
            # callback_metrics (for ModelCheckpoint monitor) + one headline metric on the bar.
            # self.log(on_step=True) is only legal inside batch/epoch hooks -> the step-0
            # baseline call from on_train_start passes self_log=False (TB add_scalar above).
            if self_log:
                self.log(
                    key,
                    value,
                    on_step=True,
                    on_epoch=False,
                    logger=False,
                    sync_dist=False,
                    prog_bar=(key == "eval/backbone/spearman"),
                )

        # evaluator restores global train mode — put frozen teacher back to eval explicitly
        self.t_bert.eval()
        self.t_head.eval()

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
        optimizer = torch.optim.AdamW(trainable_params, lr=self.cfg.get("lr", 5e-5), weight_decay=0.05)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.05),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

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

            inputs = self.tokenizer(batch_text, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            s_bert_outs = self.s_bert(**inputs)
            embeddings = self.get_sentence_embedding(s_bert_outs, inputs)

            embeddings = F.normalize(embeddings, p=2, dim=-1)

            all_embeddings.append(embeddings)

        full_embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_numpy:
            return full_embeddings.cpu().numpy()

        return full_embeddings
