import os
import ssl

# HARD GUARD (must be set before torch/lightning import): this project never calls torch.compile().
# A stray compile (e.g. an edited server copy) spawns torch._inductor compile_worker processes and
# pegs the CPU for minutes before epoch 0. TORCH_COMPILE_DISABLE forces Dynamo to fall back to eager.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

import faulthandler
import logging
import os
import time

import hydra
import lightning as L
import nltk
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from src.data import get_wikitext_sentence_dataloader
from src.dev import setup_device
from src.system import TheSystem
from src.system.thesystem import EvalCallback

logger = logging.getLogger(__name__)

nltk_data_dir = os.path.expanduser("~/nltk_data")
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.warning(f"NLTK punkt dataset not found in {nltk_data_dir}")

torch.set_float32_matmul_precision("high")


class TimedModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that logs write time — slow NFS writes of the ~900MB state_dict are a
    prime suspect for the recurring multi-minute GPU-idle stalls at eval steps."""

    def save_checkpoint(self, trainer, pl_module, filepath, *args, **kwargs):
        t0 = time.perf_counter()
        out = super().save_checkpoint(trainer, pl_module, filepath, *args, **kwargs)
        logger.info(f"checkpoint written in {time.perf_counter() - t0:.1f}s -> {filepath}")
        return out


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Every N seconds print the Python stack of ALL threads to stderr. If a single thread is
    # pegged at 100% CPU, its stack in this dump is the culprit — works without ptrace perms
    # (unlike py-spy). Set stack_dump_secs: 0 to disable.
    dump_secs = int(cfg.get("stack_dump_secs", 120))
    if dump_secs > 0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(dump_secs, repeat=True)
        logger.info(f"[stack-dump] will print all-thread stacks every {dump_secs}s to stderr")

    # Once the HF caches (.dataset_cache, transformers cache) are populated, every run still
    # makes unauthenticated hub version-check requests (wikitext, bert x2, tokenizer) which take
    # minutes on a slow/rate-limited server network. offline=true skips the hub entirely.
    # First run on a fresh machine needs `offline=false` once.
    if cfg.get("offline", True):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        logger.info("HF offline mode enabled (cache-only). Use offline=false for first run.")

    (accelerate, devices, precision, use_compile) = setup_device()
    system_dict: dict[str, L.LightningModule] = {
        "the system": TheSystem,
    }

    logger.info("Running test with stuffs below : ")
    logger.info(f"{accelerate}, {devices}, {precision}, {use_compile}")

    try:
        checkpoint_callback = TimedModelCheckpoint(
            monitor="eval/backbone/spearman",
            mode="max",
            save_top_k=1,
            every_n_train_steps=cfg.get("eval_every_steps", 1000),
            filename="best-model-{epoch:02d}-{step}",
            save_weights_only=True,
        )

        tensorboardlogger = TensorBoardLogger("tb_logs", name=cfg.get("system_name", default_value="SimCSE_Noise"))
        t0 = time.perf_counter()
        train_dataloader = get_wikitext_sentence_dataloader(
            model_name=cfg.get("model_name", "bert-base-uncased"),
            repetition=cfg.get("repetition", False),
            shuffle=cfg.get("shuffle", False),
            batch_size=cfg.get("batch_size", 128),
            cache_dir=cfg.get("cache_dir", "./.dataset_cache"),
            # ncrops MUST match DINOLoss(ncrops=...) in TheSystem — the collator builds the
            # crop-major student batch that loss.chunk(ncrops) splits. Silent mismatch = crash
            # or wrong pairing.
            ncrops=cfg.get("ncrops", 3),
            rep_prob=cfg.get("rep_prob", 0.05),
            shuffle_prob=cfg.get("shuffle_prob", 0.05),
        )
        logger.info(f"dataloader ready in {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        system = system_dict[cfg.get("system_name", default_value="SimCSE_Noise")](
            cfg, (accelerate, devices, precision, use_compile)
        )
        logger.info(f"system ready in {time.perf_counter() - t0:.1f}s")

        logger.info(f"System: {cfg.get('system_name', default_value='SimCSE_Noise')}")

        lr_callback = LearningRateMonitor()

        # IMPORTANT: eval_callback must come before checkpoint_callback — ModelCheckpoint reads
        # the monitor metric that EvalCallback populates in the same on_train_batch_end.
        callbacks = [lr_callback]
        eval_cb = EvalCallback(int(cfg.get("eval_every_steps", 1000)), baseline=cfg.get("baseline_eval", True))
        callbacks.append(eval_cb)
        callbacks.append(checkpoint_callback)

        trainer = L.Trainer(
            logger=tensorboardlogger,
            max_epochs=cfg.epochs,
            callbacks=callbacks,
            accelerator="auto",
            precision=precision,  # was silently ignored -> Trainer ran default 32-true on GPU
            gradient_clip_val=1.0,
            log_every_n_steps=10,
        )
        trainer.fit(
            model=system,
            train_dataloaders=train_dataloader,
        )

        logger.info("Epoch Finished.")

    except Exception as e:
        logger.error(f"Error : {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
