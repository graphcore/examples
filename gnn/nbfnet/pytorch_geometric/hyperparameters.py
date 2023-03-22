# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#
import dataclasses
import os
from typing import List, Dict, Any, Optional
import yaml


@dataclasses.dataclass
class DatasetHyperparameters:
    name: str
    version: Optional[str] = None
    add_inverse_train: Optional[bool] = True
    add_inverse_test: Optional[bool] = True


@dataclasses.dataclass
class ModelHyperparameters:
    input_dim: int
    hidden_dims: List
    message_fct: str
    aggregation_fct: str
    num_mlp_layers: int
    relation_learning: str
    adversarial_temperature: float


@dataclasses.dataclass
class ExecutionHyperparameters:
    batch_size_train: int
    batch_size_test: int
    num_negative: int
    check_negatives: bool
    lr: float
    num_epochs: int
    dtype: str
    loss_scale: int
    replicas: int
    device_iterations: int
    gradient_accumulation: int
    do_valid: bool
    do_test: bool
    edge_dropout: Optional[float] = 0.0
    pipeline: Optional[Dict] = None
    device: Optional[str] = "ipu"


@dataclasses.dataclass
class Config:
    dataset: DatasetHyperparameters
    model: ModelHyperparameters
    execution: ExecutionHyperparameters


def config_from_yaml(path: str):
    with open(path, "r", encoding="utf-8") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return Config(
        dataset=DatasetHyperparameters(**params["dataset"]),
        model=ModelHyperparameters(**params["model"]),
        execution=ExecutionHyperparameters(**params["execution"]),
    )


def config_to_yaml(config: Config, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        yaml.dump(dataclasses.asdict(config), file)


def recursive_replace(config: Config, new_config: Dict[str, Any]) -> Config:
    for key, val in new_config.items():
        if dataclasses.is_dataclass(getattr(config, key)):
            setattr(config, key, recursive_replace(getattr(config, key), val))
        else:
            setattr(config, key, val)
    return config
