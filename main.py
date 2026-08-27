import ssl

try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

import logging

import hydra
import lightning as L
import nltk
from omegaconf import DictConfig

from src.data import get_wikitext_sentence_dataloader
from src.dev import setup_device
from src.system import SimCSENoiseSystem

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    (accelerate, devices, precision, use_compile) = setup_device()

    logger.info("Running test with stuffs below : ")
    logger.info(f"{accelerate}, {devices}, {precision}, {use_compile}")

    try:
        dataloader = get_wikitext_sentence_dataloader()
        system = SimCSENoiseSystem(cfg, (accelerate, devices, precision, use_compile))
        trainer = L.Trainer(
            max_epochs=cfg.epochs,
            accelerator="auto",
            val_check_interval=10,
            check_val_every_n_epoch=None,
        )
        trainer.fit(model=system, train_dataloaders=dataloader)

        logger.info("Epoch Finished.")

    except Exception as e:
        logger.error(f"Error : {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
