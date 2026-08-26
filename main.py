import logging

import hydra
import nltk
from lightning import L
from omegaconf import DictConfig

from src.data import get_wikitext_sentence_dataloader
from src.dev import setup_devices
from src.system import SimCSENoiseSystem

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    (accelerate, devices, precision, use_compile) = setup_devices()

    logger.info("Running test with stuffs below : ")
    logger.info(f"{accelerate}, {devices}, {precision}, {use_compile}")

    try:
        dataloader = get_wikitext_sentence_dataloader()
        system = SimCSENoiseSystem(cfg)
        trainer = L.Trainer(max_epochs=cfg.epochs, accelerator="auto")
        trainer.fit(model=system, train_dataloaders=dataloader)

        logger.info("Epoch Finished.")

    except Exception as e:
        logger.error(f"Error : {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
