# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, root_validator, validator

from keras_extensions.learning_rate.scheduler_builder import AVAILABLE_SCHEDULERS
from keras_extensions.optimization import ALLOWED_OPTIMIZERS
from utilities.argparser import ALLOWED_GLUE_TASKS


class BertModel(BaseModel):
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    max_position_embeddings: int
    type_vocab_size: int
    initializer_range: float
    layer_norm_eps: float
    position_embedding_type: str


class LearningRateOptions(BaseModel):
    schedule_name: str
    max_learning_rate: float
    warmup_frac: float

    @validator("schedule_name", always=True)
    def name_match(cls, v):
        if v not in AVAILABLE_SCHEDULERS.keys():
            raise ValueError(f"Unrecognised learning rate schedule: `{v}`." f" Choose one of {AVAILABLE_SCHEDULERS}")
        return v


class OptimizerOptions(BaseModel):
    name: str
    weight_decay_rate: float
    loss_scaling: Optional[int] = None
    learning_rate: LearningRateOptions

    @validator("name", always=True)
    def name_match(cls, v):
        if v not in ALLOWED_OPTIMIZERS:
            raise ValueError(f"Unrecognised optimizer name: `{v}`." f" Choose one of {ALLOWED_OPTIMIZERS}")
        return v


class GlobalBatchOptions(BaseModel):
    micro_batch_size: int
    replicas: int
    grad_acc_steps_per_replica: int


class IPUConfigOptions(BaseModel):
    pipeline_stages: List[List[str]]
    pipeline_device_mapping: List[int]
    replicated_tensor_sharding: bool
    matmul_available_memory_proportion_per_pipeline_stage: List[float]


class SharedOptions(BaseModel):
    dataset_dir: Path
    generated_dataset: bool = False
    config: Path

    bert_config: BertModel
    max_seq_length: int
    max_predictions_per_seq: int

    global_batch: GlobalBatchOptions

    # Optimizations

    # Flag indicating if some layers should be replaced for optimization.
    replace_layers: bool = True

    # Flag indicating if some layers should be outlined for optimization.
    use_outlining: bool = True

    # Flag indicating if cls (transform) layer is included in the MLM Prediction head.
    use_cls_layer: bool = False

    # Flag indicating if MLM head prediction bias is used.
    use_prediction_bias: bool = True

    # Flag indicating if self attention qkv bias is used.
    use_qkv_bias: bool = False

    # Flag indicating if qkv weights are created separately as three individual weights.
    use_qkv_split: bool = False

    # Flag indicating if attention projection bias is used.
    use_projection_bias: bool = False

    # Flag indicating recomputation checkpoint is enabled.
    enable_recomputation: bool = True

    # Flag indicating if the optimizer state is offchip or onchip.
    optimizer_state_offchip: bool = True

    # Flag indicating the partial type of matmul, can be set to "half" or "float".
    matmul_partials_type: str = "half"

    # Flag indicating the serialization factor for embedding matmul, vocab_size must be divisible by it.
    embedding_serialization_factor: int = 2

    # The minimum remote tensor size (bytes) for partial variable offloading
    min_remote_tensor_size: int = 50000

    enable_stochastic_rounding: bool = True

    seed: int

    # Flag to turn on/off the compile_only mode.
    # Use TF_POPLAR_FLAGS=--executable_cache_path=/path/to/storage before using compile_only mode
    compile_only: bool = False

    ipu_config: IPUConfigOptions

    optimizer_opts: OptimizerOptions

    # Optional options
    name: Optional[str] = None
    fp_exceptions: bool = False
    total_num_train_samples: Optional[int] = None

    # Checkpointing
    save_ckpt_path: Optional[Path] = Path(__file__).parent.parent.joinpath("checkpoints").absolute()
    pretrained_ckpt_path: Optional[Path] = None
    ckpt_every_n_steps_per_execution: int = 2000

    # Logging
    logging: str = "INFO"
    global_batches_per_log: int
    enable_wandb: bool
    wandb_tags: List[str] = []
    wandb_entity_name: str = "sw-apps"
    wandb_project_name: str = "TF2-BERT"

    @root_validator()
    def RTS_replica_check(cls, values):
        if values.get("replicas") == 1 and values.get("replicated_tensor_sharding"):
            logging.warning("Replicated tensor sharding is not recommended with 1 replica.")
        return values

    @root_validator()
    def RTS_optimizer_state_check(cls, values):
        if values.get("optimizer_state_offchip") and values.get("replicated_tensor_sharding"):
            raise ValueError("Replicated tensor sharding cannot be used with optimizer state offchip.")
        return values

    @root_validator()
    def qkv_flag_match(cls, values):
        if values.get("use_qkv_bias") and not values.get("use_qkv_split"):
            raise ValueError(
                "Unrecognised combination of flag options,"
                "use_qkv_bias should be disabled when not using use_qkv_split."
            )
        return values

    @validator("matmul_partials_type", always=True)
    def matmul_partials_type_match(cls, v):
        allowed = ["half", "float"]
        if v not in allowed:
            raise ValueError(f"Unrecognised matmul partials type: `{v}`." f" Choose one of {allowed}")
        return v


class PretrainingOptions(SharedOptions):
    show_accuracy: bool = True


class SQuADOptions(SharedOptions):
    bert_config: Optional[BertModel]
    max_seq_length: Optional[int] = None
    max_predictions_per_seq: Optional[int]
    bert_model_name: Optional[str]

    num_epochs: int

    output_dir: str

    do_training: bool = True
    do_validation: bool = True


class GLUEOptions(SharedOptions):
    bert_config: Optional[BertModel]
    max_seq_length: Optional[int] = None
    max_predictions_per_seq: Optional[int]
    bert_model_name: Optional[str]

    num_epochs: int

    output_dir: str

    do_training: bool = True
    do_validation: bool = True
    # Test results are only to be scored on the private GLUE ranking
    do_test: bool = False

    glue_task: str = "mrpc"

    @validator("glue_task", always=True)
    def glue_task_match(cls, v):
        if v not in ALLOWED_GLUE_TASKS:
            raise ValueError(f"Unrecognised GLUE task: `{v}`." f" Choose one of {ALLOWED_GLUE_TASKS}")
        return v
