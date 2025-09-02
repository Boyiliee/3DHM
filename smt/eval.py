import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
import torch
import os
from smt import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    if cfg.trainer.strategy:
        if cfg.trainer.strategy["_target_"] == "pytorch_lightning.strategies.DeepSpeedStrategy":
            cfg.trainer.strategy["stage"]=0
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    checkpoint_path = cfg.ckpt_path
    if (cfg.configs.weights_path is not None):
        if os.path.exists(cfg.configs.weights_path):
            av = torch.load(cfg.configs.weights_path, map_location=torch.device('cpu'))
            if "module" in av.keys():
                av = av["module"]
                av = {key.replace("module.", ""): value for key, value in av.items()}
            elif "state_dict" in av.keys():
                av = av["state_dict"]
            
            model.load_state_dict(av, strict=cfg.configs.load_strict)
            log.info("Loading weights from weights " +
                        cfg.configs.weights_path)
            checkpoint_path = None
    model.strict_loading = False
    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
    # for predictions use trainer.predict(...)
    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
