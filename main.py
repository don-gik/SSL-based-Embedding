import ssl

try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

import logging
import os

import hydra
import lightning as L
import nltk
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from src.data import get_wikitext_sentence_dataloader
from src.dev import setup_device
from src.system import (
    DinoNoiseSystem,
    EMANoiseSystem,
    SigDinoNoiseSystem,
    SimCSENoiseSystem,
)

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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    (accelerate, devices, precision, use_compile) = setup_device()
    system_dict: dict[str, L.LightningModule] = {
        "SimCSE_Noise": SimCSENoiseSystem,
        "EMA_Noise": EMANoiseSystem,
        "Dino_Noise": DinoNoiseSystem,
        "SigDino_Noise": SigDinoNoiseSystem,
    }

    logger.info("Running test with stuffs below : ")
    logger.info(f"{accelerate}, {devices}, {precision}, {use_compile}")

    try:
        dummy_ds = TensorDataset(torch.zeros(1))
        dummy_loader = DataLoader(dummy_ds, batch_size=1)

        dataloader = get_wikitext_sentence_dataloader()
        system = system_dict[cfg.get("system_name", default_value="SimCSE_Noise")](
            cfg, (accelerate, devices, precision, use_compile)
        )

        logger.info(f"System: {cfg.get("system_name", default_value="SimCSE_Noise")}")

        trainer = L.Trainer(
            max_epochs=cfg.epochs,
            accelerator="auto",
            val_check_interval=100,
            check_val_every_n_epoch=None,
        )
        trainer.fit(
            model=system,
            train_dataloaders=dataloader,
            val_dataloaders=dummy_loader,
        )

        logger.info("Epoch Finished.")

    except Exception as e:
        logger.error(f"Error : {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
