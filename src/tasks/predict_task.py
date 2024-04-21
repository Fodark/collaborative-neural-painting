from typing import Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from src import utils

log = utils.get_pylogger(__name__)


# @utils.task_wrapper
def predict(cfg: DictConfig) -> None:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.predict(model=model, ckpt_path=cfg.ckpt_path)

    log.info("Done instantiating model")
