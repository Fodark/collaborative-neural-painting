from typing import List, Tuple

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def find_lr(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, auto_lr_find=True
    )

    # trainer.tune(model, datamodule=datamodule)
    lr_finder = trainer.tuner.lr_find(model, min_lr=1e-8, num_training=100, datamodule=datamodule)
    # object_dict = {
    #     "cfg": cfg,
    #     "datamodule": datamodule,
    #     "model": model,
    #     "callbacks": callbacks,
    #     "logger": logger,
    #     "trainer": trainer,
    # }

    # if logger:
    #     log.info("Logging hyperparameters!")
    #     utils.log_hyperparameters(object_dict)

    # lr_finder = trainer.tune(model, train_dataloaders=datamodule)
    # log.info(lr_finder)

    # Plot with
    fig = lr_finder.plot(suggest=True)
    new_lr = lr_finder.suggestion()

    # save fig
    fig.savefig("lr_finder_plot.png")
    # log new lr
    log.info(f"New LR: {new_lr}")
