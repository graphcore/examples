# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, root_validator, validator

ALLOWED_DATASET_TYPE = ["arxiv", "ppi", "reddit", "generated"]
ALLOWED_LOGGING_TYPE = ["DEBUG", "INFO", "ERROR", "CRITICAL", "WARNING"]
# Allowed precision types are float16 (fp16), float32 (fp32) and mixed precision with
# compute dtype float16 and variable dtype float 32 (mixed)
ALLOWED_PRECISION_TYPE = ["fp16", "fp32", "mixed"]
ALLOWED_PARTIALS_TYPE = ["half", "float"]
ALLOWED_DEVICE_OPTIONS = ["ipu", "cpu"]

"""
ALLOWED_ADJACENCY_MODE: Alternatives to transform the adjacency matrix for the convolution.
"normalised_regularised" implements Eq. (1) from paper: A_tilde = A',
    where A' is the normalised and regularised adjacency.
"self_connections_scaled_by_degree" implements Eq. (10): A_tilde = (D + I)^(-1) @ (A + I).
"normalised_regularised_self_connections_scaled_by_degree" implements Eqs. (9) and (10):
    A_tilde = (D + I)^(-1) @ (A' + I), where A' is the normalised and regularised adjacency.
"self_connections_scaled_by_degree_with_diagonal_enhancement" implements Eqs. (10) + (11):
    A_tilde = A_1 + lambda * diag(A_1), where A_1 = (D + I)^(-1) @ (A + I)
"""
ALLOWED_ADJACENCY_MODE = [
    "normalised",
    "normalised_regularised",
    "self_connections_scaled_by_degree",
    "normalised_regularised_self_connections_scaled_by_degree",
    "self_connections_scaled_by_degree_with_diagonal_enhancement"
]


class AdjacencyOptions(BaseModel):
    transform_mode: str
    diag_lambda: Optional[float]
    regularisation: Optional[float]

    @root_validator()
    def check_needed_parameters_per_option(cls, values):
        transform_mode = values.get("transform_mode")
        diag_lambda = values.get("diag_lambda")
        regularisation = values.get("regularisation")

        def warn_unused_parameters(warn_diag=False, warn_reg=False):
            if warn_diag and diag_lambda is not None:
                logging.warning("'diag_lambda' parameter from config will not be used.")
            if warn_reg and regularisation is not None:
                logging.warning("'regularisation' parameter from config will not be used.")

        if transform_mode in [
            "normalised_regularised",
            "normalised_regularised_self_connections_scaled_by_degree"
        ]:
            if regularisation is None:
                raise ValueError("'regularisation' parameter is needed in config file.")
            warn_unused_parameters(warn_diag=True)

        elif transform_mode in ["normalised", "self_connections_scaled_by_degree"]:
            warn_unused_parameters(warn_diag=True, warn_reg=True)

        elif transform_mode == "self_connections_scaled_by_degree_with_diagonal_enhancement":
            if diag_lambda is None:
                raise ValueError("'diag_lambda' parameter is needed in config file.")
            warn_unused_parameters(warn_reg=True)

        return values

    @validator("transform_mode", always=True)
    def adjacency_transform_mode_type_match(cls, value):
        if value not in ALLOWED_ADJACENCY_MODE:
            raise ValueError(f"Unrecognised adjacency transform mode: `{value}`."
                             f" Choose one of {ALLOWED_ADJACENCY_MODE}")
        return value


class ModelOptions(BaseModel):
    hidden_size: int
    num_layers: int
    dropout: float
    adjacency: AdjacencyOptions
    first_layer_precalculation: bool = False


class IPUConfigOptions(BaseModel):
    pipeline_stages: List[List[str]]
    pipeline_device_mapping: List[int]
    matmul_available_memory_proportion_per_pipeline_stage: List[float]
    # Flag indicating the partial type of matmul, can be set to "half" or "float".
    matmul_partials_type: str = "float"


    @validator("matmul_partials_type", always=True)
    def matmul_partials_type_match(cls, value):
        if value not in ALLOWED_PARTIALS_TYPE:
            raise ValueError(f"Unrecognised matmul partials type: `{value}`."
                             f" Choose one of {ALLOWED_PARTIALS_TYPE}")
        return value

    @root_validator()
    def check_pipeline_stages_compatibility(cls, values):
        num_pipeline_stages = len(values.get("pipeline_stages"))
        device_mapping = values.get("pipeline_device_mapping")
        num_device_mapping = len(device_mapping)
        num_ipus_per_replica = max(device_mapping) + 1
        length_matmul_amp_list = len(values.get(
            "matmul_available_memory_proportion_per_pipeline_stage"))
        if num_pipeline_stages != num_device_mapping:
            raise ValueError(
                f"Number of pipeline stages ({num_pipeline_stages}) does not"
                f" match the number of device mappings ({num_device_mapping})."
                " Ensure the `pipeline_stages` and `pipeline_device_mapping`"
                " lists are the same length.")
        if length_matmul_amp_list != num_ipus_per_replica:
            raise ValueError(
                "Available memory proportion must be set for each of the "
                f" {num_ipus_per_replica} IPUs in the pipeline.")
        return values


class SetOptions(BaseModel):
    # Clustering
    num_clusters: int
    clusters_per_batch: int

    gradient_accumulation_steps_per_replica: int

    # Steps per execution, which is per replica when using Keras model API.
    steps_per_execution: int

    ipu_config: IPUConfigOptions

    device: str = "ipu"

    @validator("device", always=True)
    def device_match(cls, value):
        if value not in ALLOWED_DEVICE_OPTIONS:
            raise ValueError(f"Unrecognised device type `{value}` in"
                             f" {cls.__name__} options."
                             f" Choose one of {ALLOWED_DEVICE_OPTIONS}")
        if value == "cpu":
            logging.warning("Requested to run on CPU device for"
                            f" {cls.__name__}. The IPU config options will be"
                            " ignored, and any IPU specific layers will be"
                            " replaced with their generic equivalent.")
        return value


class TrainingOptions(SetOptions):
    lr: float
    epochs: int
    loss_scaling: Optional[int] = None


class ValidationOptions(SetOptions):
    pass


class Options(BaseModel):
    # Model
    model: ModelOptions

    # Training
    do_training: bool = True
    training: TrainingOptions

    # Validation
    do_validation: bool = True
    validation: ValidationOptions

    # Dataset
    dataset_name: str
    data_path: Path = "data"

    # Optimization
    precision: str = "fp32"

    # Logging
    name: str = "Cluster-GCN"
    logging: str = "INFO"
    wandb: bool = False
    executions_per_log: int = 1

    # Checkpointing
    save_ckpt_path: Optional[Path] = Path(__file__).parent.parent.joinpath("checkpoints").absolute()
    load_ckpt_path: Optional[Path] = None
    executions_per_ckpt: int = 0

    # Misc.
    seed: int
    compile_only: bool = False

    @validator("dataset_name", always=True)
    def dataset_name_match(cls, value):
        if value not in ALLOWED_DATASET_TYPE:
            raise ValueError(f"Unrecognised dataset name: `{value}`."
                             f" Choose one of {ALLOWED_DATASET_TYPE}")
        return value

    @validator("logging", always=True)
    def logging_type_match(cls, value):
        if value not in ALLOWED_LOGGING_TYPE:
            raise ValueError(f"Unrecognised logging type: `{value}`."
                             f" Choose one of {ALLOWED_LOGGING_TYPE}")
        return value

    @validator("precision", always=True)
    def precision_type_match(cls, value):
        if value not in ALLOWED_PRECISION_TYPE:
            raise ValueError(f"Unrecognised precision type: `{value}`."
                             f" Choose one of {ALLOWED_PRECISION_TYPE}")
        return value
