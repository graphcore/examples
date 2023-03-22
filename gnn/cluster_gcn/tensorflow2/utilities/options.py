# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, root_validator, validator, PositiveInt

from utilities.constants import MethodMaxNodesEdges


ALLOWED_DATASET_TYPE = ["ogbn-arxiv", "ogbn-products", "ogbn-mag", "ogbn-lsc-mag240", "ppi", "reddit", "generated"]
ALLOWED_LOGGING_TYPE = ["DEBUG", "INFO", "ERROR", "CRITICAL", "WARNING"]
# Allowed precision types are float16 (fp16), float32 (fp32) and mixed precision with
# compute dtype float16 and variable dtype float 32 (mixed)
ALLOWED_PRECISION_TYPE = ["fp16", "fp32", "mixed"]
ALLOWED_PARTIALS_TYPE = ["half", "float"]
ALLOWED_DEVICE_OPTIONS = ["ipu", "cpu"]

"""
ALLOWED_MAX_EDGES_COMPUTE_METHOD: Method to compute the maximum number of edges per batch:
- average: Take the average number of edges per cluster times the number of clusters per batch.
- average_plus_std: Take the average number of edges per cluster plus one standard distribution,
    times the number of clusters per batch.
- upper_bound: Take the sum of the N highest number of edges among all clusters, where N is the
    number of clusters per batch.
"""
ALLOWED_MAX_EDGES_COMPUTE_METHOD = ["average", "average_plus_std", "upper_bound"]

"""
ALLOWED_ADJACENCY_TRANSFORM: Alternatives to transform the adjacency matrix for the convolution.
"normalised_regularised" implements Eq. (1) from paper: A_tilde = A',
    where A' is the normalised and regularised adjacency.
"self_connections_scaled_by_degree" implements Eq. (10): A_tilde = (D + I)^(-1) @ (A + I).
"normalised_regularised_self_connections_scaled_by_degree" implements Eqs. (9) and (10):
    A_tilde = (D + I)^(-1) @ (A' + I), where A' is the normalised and regularised adjacency.
"self_connections_scaled_by_degree_with_diagonal_enhancement" implements Eqs. (10) + (11):
    A_tilde = A_1 + lambda * diag(A_1), where A_1 = (D + I)^(-1) @ (A + I)
"""
ALLOWED_ADJACENCY_TRANSFORM = [
    "normalised",
    "normalised_regularised",
    "self_connections_scaled_by_degree",
    "normalised_regularised_self_connections_scaled_by_degree",
    "self_connections_scaled_by_degree_with_diagonal_enhancement",
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

        if transform_mode in ["normalised_regularised", "normalised_regularised_self_connections_scaled_by_degree"]:
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
        if value not in ALLOWED_ADJACENCY_TRANSFORM:
            raise ValueError(
                f"Unrecognised adjacency transform mode: '{value}'." f" Choose one of {ALLOWED_ADJACENCY_TRANSFORM}"
            )
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

    enable_recomputation: bool = False

    # Designate this amount of tiles on an IPU exclusively for IO.
    # This can overlap the data transfer time and the compute.
    # This should be set based on a trade off of how many tiles you
    # can sacrifice from computation to improve IO.
    num_io_tiles: int = 0

    @root_validator()
    def check_pipeline_stages_compatibility(cls, values):
        num_pipeline_stages = len(values.get("pipeline_stages"))
        device_mapping = values.get("pipeline_device_mapping")
        num_device_mapping = len(device_mapping)
        num_ipus_per_replica = max(device_mapping) + 1
        length_matmul_amp_list = len(values.get("matmul_available_memory_proportion_per_pipeline_stage"))
        if num_pipeline_stages != num_device_mapping:
            raise ValueError(
                f"Number of pipeline stages ({num_pipeline_stages}) does not"
                f" match the number of device mappings ({num_device_mapping})."
                " Ensure the `pipeline_stages` and `pipeline_device_mapping`"
                " lists are the same length."
            )
        if length_matmul_amp_list != num_ipus_per_replica:
            raise ValueError(
                "Available memory proportion must be set for each of the "
                f" {num_ipus_per_replica} IPUs in the pipeline."
            )
        return values

    @root_validator()
    def recomputation_validation(cls, values):
        recomputation = values.get("enable_recomputation")
        len_pipeline_stages = len(values.get("pipeline_stages"))
        if recomputation and len_pipeline_stages < 2:
            raise ValueError(
                "Recomputation requires a minimum of 2 pipeline stages. " f"Only {len_pipeline_stages} given."
            )
        return values


class SetOptions(BaseModel):
    # Batching
    # micro_batch_size denotes the number of super-clusters (i.e. combined from the clusters
    # generated by metis). For optimal results this is set by default to be 1.
    micro_batch_size: int = 1
    gradient_accumulation_steps_per_replica: int
    replicas: PositiveInt = 1
    dataset_prefetch_depth: int = 10

    # Clustering
    max_nodes_per_batch: Optional[int]
    num_clusters: Optional[int]
    clusters_per_batch: int

    # Optimization
    precision: str = "fp32"

    executions_per_epoch: Optional[PositiveInt] = 1

    epochs_per_execution: Optional[PositiveInt] = 1

    ipu_config: IPUConfigOptions

    device: str = "ipu"

    use_sparse_representation: bool = False

    @validator("micro_batch_size", always=True)
    def micro_batch_size_limit(cls, value):
        if value != 1:
            raise ValueError(f"micro_batch_size `{value}` is not allowed." f" Only the value of 1 is supported.")
        return value

    @validator("device", always=True)
    def device_match(cls, value):
        if value not in ALLOWED_DEVICE_OPTIONS:
            raise ValueError(
                f"Unrecognised device type `{value}` in"
                f" {cls.__name__} options."
                f" Choose one of {ALLOWED_DEVICE_OPTIONS}"
            )
        if value == "cpu":
            logging.warning(
                "Requested to run on CPU device for"
                f" {cls.__name__}. The IPU config options will be"
                " ignored, and any IPU specific layers will be"
                " replaced with their generic equivalent."
            )
        return value

    @root_validator(pre=False)
    def precision_type_match(cls, values):
        precision = values.get("precision")
        if precision not in ALLOWED_PRECISION_TYPE:
            raise ValueError(
                f"Unrecognised precision type: `{values.get('precision')}`." f" Choose one of {ALLOWED_PRECISION_TYPE}"
            )
        if values.get("device") != "ipu" and precision == "fp16":
            raise ValueError(f"Sparse tensors in precision {precision} on CPU is not" " supported.")
        return values

    @root_validator(pre=False)
    def set_executions_and_epochs(cls, values):
        if values.get("executions_per_epoch") > 1 and values.get("epochs_per_execution") > 1:
            raise ValueError(f"executions_per_epoch and epochs_per_execution can't" " be set at the same time.")
        return values

    @root_validator(pre=False)
    def replica_device_match(cls, values):
        replicas = values.get("replicas")
        device = values.get("device")
        if replicas > 1 and device != "ipu":
            raise ValueError(
                f"Replicas > 1 is only supported on IPU."
                f" Found `{values.get('replicas')}` and"
                f"`{values.get('device')}`."
            )
        return values


class TrainingOptions(SetOptions):
    lr: float
    epochs: int
    loss_scaling: Optional[int] = None

    do_live_validation: bool = False
    validation_frequency: int = 10


class ValidationOptions(SetOptions):
    pass


class TestOptions(SetOptions):
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

    # Test
    do_test: bool = True
    test: TestOptions

    # Clusters
    method_max_nodes: str = "upper_bound"
    method_max_edges: str = "upper_bound"
    inter_cluster_ratio: float = 0.0
    # cluster_node_edge_imbalance_ratio:
    # Tuple of floats greater than 1.0. The first value representing the
    # acceptable tolerance to balance the nodes per cluster, the second
    # to balance the edges per cluster. The lower the value, the stricter
    # this constraint. Results vary between datasets. An example of a
    # value would be [1.01, 1.05], where we are giving a large tolerance
    # to the balance of edges and a strict tolerance to the balance of nodes.
    cluster_node_edge_imbalance_ratio: Optional[tuple] = None
    calculate_cluster_statistics: bool = False
    regenerate_clustering_cache: bool = False
    save_clustering_cache: bool = True

    # Dataset
    dataset_name: str
    data_path: Path = "data"
    pca_features_path: Path = None
    regenerate_dataset_cache: bool = False
    save_dataset_cache: bool = True

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
    fp_exceptions: bool = False

    @root_validator(pre=True)
    def cluster_options_validation(cls, values):
        max_nodes_per_batch_training = values.get("training").get("max_nodes_per_batch")
        max_nodes_per_batch_validation = values.get("training").get("max_nodes_per_batch")
        num_clusters_training = values.get("training").get("num_clusters")
        num_clusters_validation = values.get("validation").get("num_clusters")

        if (max_nodes_per_batch_training or max_nodes_per_batch_validation) and (
            num_clusters_training or num_clusters_validation
        ):
            raise ValueError(
                "Set either max_nodes_per_batch for both training"
                "and validation or num_clusters for both training and"
                " validation. If max_nodes_per_batch is used, the"
                " number of clusters will be calculated automatically"
                " based on this, and vice versa."
            )
        if (num_clusters_training or num_clusters_validation) and None in (
            num_clusters_training,
            num_clusters_validation,
        ):
            raise ValueError(
                "If setting the number of clusters, you must set" " the num_clusters option for all dataset splits"
            )
        if (max_nodes_per_batch_training or max_nodes_per_batch_validation) and None in (
            max_nodes_per_batch_training,
            max_nodes_per_batch_validation,
        ):
            raise ValueError(
                "If setting the max nodes per batch, you must set"
                " the max_nodes_per_batch option for all dataset"
                " splits."
            )
        return values

    @validator("dataset_name", always=True)
    def dataset_name_match(cls, value):
        if value not in ALLOWED_DATASET_TYPE:
            raise ValueError(f"Unrecognised dataset name: `{value}`." f" Choose one of {ALLOWED_DATASET_TYPE}")
        return value

    @validator("method_max_edges", always=True)
    def method_max_edges_allowed(cls, value):
        if value not in ALLOWED_MAX_EDGES_COMPUTE_METHOD:
            raise ValueError(
                f"Unrecognised method to compute the maximum number of edges:"
                f"`{value}`. Choose one of {ALLOWED_MAX_EDGES_COMPUTE_METHOD}."
            )
        return value

    @validator("logging", always=True)
    def logging_type_match(cls, value):
        if value not in ALLOWED_LOGGING_TYPE:
            raise ValueError(f"Unrecognised logging type: `{value}`." f" Choose one of {ALLOWED_LOGGING_TYPE}")
        return value
