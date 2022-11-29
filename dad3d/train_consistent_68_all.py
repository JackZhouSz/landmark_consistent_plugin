import os
from typing import Dict, Any
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from model_training.utils import load_hydra_config, create_logger
from model_training.train.trainer import DAD3DTrainer
from model_training.model import load_model_68
from model_training.train.flame_lightning_model_consistent_all import FlameLightningModel
from model_training.data import FlameConsistentDatasetAll

import pdb
import tensorboardX

logger = create_logger(__name__)

torch.autograd.set_detect_anomaly(True)


def train_consistent_68(config):    
    train_dataset = FlameConsistentDatasetAll.from_config(config=config["train"])
    val_dataset = FlameConsistentDatasetAll.from_config(config=config["val"])
    model = load_model_68(config["model"], config["constants"])
    dad3d_net = FlameLightningModel(model=model, config=config, train=train_dataset, val=val_dataset)
    # pdb.set_trace()
    # model_traced = dad3d_net.export_jit_model(config["model"]['model_config']['ckpt_path'])
    # torch.jit.save(model_traced, config["model"]['model_config']['ckpt_path'].replace('.ckpt', '.trcd'))
    dad3d_trainer = DAD3DTrainer(dad3d_net, config)
    dad3d_trainer.fit()


def prepare_experiment(hydra_config: DictConfig) -> Dict[str, Any]:
    experiment_dir = os.getcwd()
    save_path = os.path.join(experiment_dir, "experiment_consistent_config_68_all.yaml")
    OmegaConf.set_struct(hydra_config, False)
    hydra_config["yaml_path"] = save_path
    hydra_config["experiment"]["folder"] = experiment_dir
    logger.info(OmegaConf.to_yaml(hydra_config, resolve=True))
    config = load_hydra_config(hydra_config)
    with open(save_path, "w") as f:
        OmegaConf.save(config=config, f=f.name)
    return config

@hydra.main(config_name="train_consistent_68_all", config_path="model_training/config")
def run_experiment(hydra_config: DictConfig) -> None:
    config = prepare_experiment(hydra_config)
    logger.info("Experiment dir %s" % config["experiment"]["folder"])
    train_consistent_68(config)


if __name__ == "__main__":
    run_experiment()
