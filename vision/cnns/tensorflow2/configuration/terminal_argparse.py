# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import argparse
import logging
from pathlib import Path

import horovod.tensorflow as hvd
import popdist
import popdist.tensorflow
import yaml
from custom_exceptions import DimensionError, UnallowedConfigurationError
from examples_utils import parse_yaml_config
from precision import Precision
from datasets.dataset_factory import AVAILABLE_DATASETS
from optimizers.optimizer_factory import AVAILABLE_OPTIMIZERS
from schedules.scheduler_factory import AVAILABLE_SCHEDULERS
from tensorflow.python.ipu import distributed
from tensorflow.python.ipu.config import SchedulingAlgorithm, StochasticRoundingBehaviour
from tensorflow.python.ipu.ops.pipelining_ops import PipelineSchedule


def add_arguments(parser):
    # Configuration
    parser.add_argument("--config", type=str, help="Select from available configurations")
    parser.add_argument(
        "--config-path",
        type=str,
        default=str(Path(Path(__file__).parent, Path("../configs.yml"))),
        help="path to the configuration file",
    )
    parser.add_argument(
        "--on-demand",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, it will defer connection to when the IPU is needed",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="If true, the executable will be generated without attaching to the ipu.",
    )

    # Randomness, reproducibility and determinism
    parser.add_argument("--seed", type=int, default=None, help="Set a global seed for the random number generator.")
    parser.add_argument(
        "--shuffle",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, shuffling in the training dataset pipeline is enabled. "
        "Shuffling in the validation pipeline remains disabled.",
    )
    parser.add_argument(
        "--deterministic",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="If true, sets the dataset pipeline to be deterministic and seeds prngs, "
        "if it hasn't been already done. Enforcing determinism degrades "
        "the data processing performance. Currently, determinism cannot be "
        "guaranteed for the ImageNet dataset.",
    )

    # Checkpoints
    parser.add_argument("--ckpts-per-epoch", type=str_to_float, default=1, help="Checkpointing frequency, per epoch")
    parser.add_argument("--first-ckpt-epoch", type=float, default=0.0, help="First checkpoint, in epochs")
    parser.add_argument(
        "--checkpoint-input-dir",
        type=str,
        default=None,
        help="Path to load checkpoints from, if no argument is provided, directory /tmp/checkpoints_current_time/ will be used",
    )
    parser.add_argument(
        "--checkpoint-output-dir",
        type=str,
        default=None,
        help="Path to save checkpoints to, if no argument is provided, directory /tmp/checkpoints_current_time/ will be used",
    )
    parser.add_argument(
        "--ckpt-all-instances",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Allow all instances to create a checkpoint. By default only local instance 0 does checkpointing.",
    )
    parser.add_argument(
        "--clean-dir",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="If true, it will delete the checkpoint directory (and all the files inside)",
    )

    # Dataset and model choice arguments
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=AVAILABLE_DATASETS, help="Name of dataset to use"
    )
    parser.add_argument("--dataset-path", type=str, default=".", help="Path to dataset")
    parser.add_argument("--model-name", type=str, default="toy_model", help="Name of model to use")
    parser.add_argument(
        "--eight-bit-transfer",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable/disable input transfer in 8 bit",
    )
    parser.add_argument(
        "--synthetic-data",
        type=str,
        default=None,
        choices=["host", "ipu"],
        help="Enable usage of synthetic data on the host or ipu. Corresponding options are 'host' or 'ipu'",
    )

    # Training parameter arguments
    parser.add_argument(
        "--training", type=str_to_bool, nargs="?", const=True, default=True, help="Enable/disable training"
    )
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro batch size, in number of samples")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--logs-per-epoch", type=str_to_float, default=1, help="Logging frequency, per epoch")
    parser.add_argument(
        "--weight-updates-per-epoch",
        type=int,
        default=-1,
        help="number of weight updates per run on the device for one epoch",
    )
    parser.add_argument("--num-replicas", type=int, default=1, help="Number of training replicas")
    parser.add_argument(
        "--gradient-accumulation-count", type=int, default=None, help="Number of gradients accumulated by each replica"
    )
    parser.add_argument("--global-batch-size", type=int, default=None, help="Global batch size, in number of samples")
    parser.add_argument(
        "--precision",
        type=str,
        default="16.16",
        choices=Precision.supported_precisions,
        help="<compute precision>.<weight update precision> both 16 or 32",
    )
    parser.add_argument(
        "--pipeline-splits",
        type=str,
        nargs="*",
        default=[],
        help="Model layers that define the start of a new pipeline stage. E.g. conv2d_1 max_pooling2d",
    )
    parser.add_argument(
        "--device-mapping",
        type=int,
        nargs="*",
        default=None,
        help="List mapping pipeline stages to IPU numbers. E.g. 0 1 1 0",
    )
    parser.add_argument(
        "--pipeline-schedule",
        type=str,
        default="Grouped",
        choices=[str(p).split(".")[-1] for p in list(PipelineSchedule)],
        help="Pipelining schedule. Choose between 'Interleaved', 'Grouped' and 'Sequential'.",
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=AVAILABLE_OPTIMIZERS, help="The name of the optimizer to use"
    )
    parser.add_argument(
        "--optimizer-params",
        type=yaml.safe_load,
        default='{"momentum": 0}',
        help="Parameters to configure the optimizer with. To pass this argument from the terminal "
        'use --optimizer-params \'{"arg1": value1, "arg2": value2...}\' format.',
    )
    parser.add_argument(
        "--loss-scaling",
        type=float,
        default=0.0,
        help="The value of static loss scaling. When equal to 0, loss scaling is disabled. "
        "The loss scale is adjusted based on the replication factor, due to risk of loss underflow.",
    )
    parser.add_argument(
        "--auto-loss-scaling",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable automatic loss scaling for half precision training. "
        "Note that this is an experimental feature and it is not guaranteed to work on all configurations.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0, help="The value of weight decay used by the optimizer."
    )
    parser.add_argument(
        "--l2-regularization", type=float, default=0.0, help="The value of l2 regularization used by the optimizer."
    )
    parser.add_argument(
        "--recomputation",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable/disable recomputation of activations in the backward pass",
    )
    parser.add_argument(
        "--accelerator-side-preprocess",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="When enabled some preprocessing steps (depending on the chosen dataset), are run "
        "on the accelerator rather on the host.",
    )
    parser.add_argument(
        "--accelerator-side-reduction",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Requires distributed training. When enabled the reduction over replicas for logging "
        "is performed on the device rather than the host.",
    )
    parser.add_argument(
        "--stochastic-rounding",
        type=str,
        default="ON",
        choices=[str(p).split(".")[-1] for p in list(StochasticRoundingBehaviour)],
        help="Enable one of three different stochastic rounding modes: ON, OFF or RI (Replica Identical).",
    )
    parser.add_argument(
        "--optimizer-state-offloading",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable/disable the offloading of the optimizer state to the IPU remote memory.",
    )
    parser.add_argument(
        "--fp-exceptions",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable/disable floating point exceptions.",
    )
    parser.add_argument(
        "--nanoo",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="NaN on overflow. When False it saturates instead.",
    )
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="const",
        choices=list(AVAILABLE_SCHEDULERS.keys()),
        help="Type of learning rate schedule. By default, a constant learning rate of 1e-3 is used.",
    )
    parser.add_argument(
        "--lr-warmup-params",
        type=yaml.safe_load,
        default=None,
        help="Parameters to configure the warmup of learning rate. To pass this argument from the terminal "
        'type --lr-schedule-params \'{"warmup_mode": <mode>, "warmup_epochs": <epochs>}\' format.',
    )
    parser.add_argument(
        "--lr-schedule-params",
        type=yaml.safe_load,
        default='{"initial_learning_rate": 0.0001}',
        help="Parameters to configure learning rate scheduler. To pass this argument from the terminal "
        'type --lr-schedule-params \'{"arg1": value1, "arg2": value2...}\' format.',
    )
    parser.add_argument(
        "--lr-staircase",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Apply a staircase on the learning rate schedule, making learning rate values constant throughout "
        "one epoch.",
    )
    parser.add_argument(
        "--dbn-replica-group-size",
        type=int,
        default=1,
        help="Distributed Batch Norm (DBN) option specifies how many replicas to aggregate the batch statistics across. "
        "DBN is disabled when ==1. It can be enabled only if model fits on a single ipu (num ipus per replica ==1), "
        "model is replicated (num replicas > 1) and replication factor is divisible by dbn replica group size.",
    )
    parser.add_argument("--label-smoothing", type=float, default=None, help="Smoothing factor added to each zero label")
    parser.add_argument(
        "--pipeline-num-parallel",
        type=int,
        default=48,
        help="Number of images to process in parallel on the host side.",
    )
    parser.add_argument(
        "--norm-layer",
        type=yaml.safe_load,
        default='{"name": "custom_batch_norm", "momentum": 0.97}',
        help="Type of normalisation layer to use. When using group norm specify either num_groups or channels_per_group. "
        "When using batch norm specify momentum.",
    )
    parser.add_argument(
        "--fused-preprocessing",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Use fused operations for preprocessing images on device.",
    )

    # Poplar optimizations
    parser.add_argument(
        "--half-partials",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Accumulate matmul and convolution partial results in half precision",
    )
    parser.add_argument(
        "--internal-exchange-optimization-target",
        type=str,
        default=None,
        choices=["cycles", "memory", "balanced"],
        help="Set poplar internal exchange optimization target. Default is cycles.",
    )
    parser.add_argument(
        "--max-cross-replica-buffer-size",
        type=int,
        default=0,
        help="The maximum number of bytes that can be waiting before a cross replica sum op is scheduled. "
        "0 (default) means that they are scheduled immediately.",
    )
    parser.add_argument(
        "--max-reduce-many-buffer-size",
        type=int,
        default=0,
        help="The maximum size (in bytes) a cluster of reduce operations can reach before it is scheduled. "
        "These clusters are lowered to popops ReduceMany operations.",
    )
    parser.add_argument(
        "--conv-dithering",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable dithering of the convolution start tile to improve tile memory balance",
    )
    parser.add_argument(
        "--gather-conv-output",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Reduce sync cost of small sized all-reduces. Useful when paired with distributed batch norm",
    )
    parser.add_argument(
        "--min-remote-tensor-size",
        type=int,
        default=128,
        help="The minimum size (in bytes) a tensor must be in order to be considered for being stored in remote memory.",
    )
    parser.add_argument(
        "--stable-norm",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable/disable numerically more stable but less parallelizable normalization layers.",
    )
    parser.add_argument(
        "--available-memory-proportion",
        type=float,
        nargs="*",
        default=[],
        help="The percentage of IPU memory dedicated to convolutions and matmuls.",
    )
    parser.add_argument(
        "--scheduling-algorithm",
        type=str,
        default="CHOOSE_BEST",
        choices=[str(p).split(".")[-1] for p in list(SchedulingAlgorithm)],
        help="Controls the algorithm that the scheduler uses for planning the graph layout in the tile memory.",
    )
    parser.add_argument(
        "--prefetch-depth",
        type=int,
        default=1,
        help="Controls the infeeds prefetch depth, which represents how many samples are prefetched onto the IPU ahead of time.",
    )

    # Evaluation and logging choice arguments
    parser.add_argument(
        "--mlperf-logging", type=str_to_bool, nargs="?", const=True, default=False, help="TTT measurement as in MLPerf."
    )
    parser.add_argument(
        "--wandb",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable/disable logging to Weights & Biases",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=None, help="Weights & Biases tags")
    parser.add_argument(
        "--validation", type=str_to_bool, nargs="?", const=True, default=True, help="Enable/disable validation"
    )
    parser.add_argument(
        "--validation-micro-batch-size",
        type=int,
        default=None,
        help="Validation micro batch size, in number of samples",
    )
    parser.add_argument("--validation-num-replicas", type=int, default=None, help="Number of validation replicas")
    parser.add_argument(
        "--pipeline-validation-model",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Reuse the training pipeline splits for validation",
    )

    parser.add_argument(
        "--sweep",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Computes metrics needed for hyperparameter optimization sweeps",
    )
    parser.add_argument(
        "--target-accuracy", type=float, default=0.759, help="Target accuracy, useful for hyperparameter optimization."
    )

    return parser


def str_to_bool(value):
    # boolean args can be used as flags to  set value = const
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {"false", "f", "0", "no", "n", "off"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y", "on"}:
        return True
    raise argparse.ArgumentTypeError(f"{value} is not a valid boolean value")


def str_to_float(frac_str):

    list_string = frac_str.split("/")

    if len(list_string) == 1:
        return float(list_string[0])

    elif len(list_string) == 2:
        try:
            num, denom = float(list_string[0]), float(list_string[1])
        except:
            raise argparse.ArgumentTypeError(
                f"Could not parse {frac_str} as a fraction. The fraction numerator or denominator could be missing or one of those could not be parsed as a floating point number."
            )

        return num / denom

    else:
        raise argparse.ArgumentTypeError(f"Number should be provided as float or fraction like a/b")


def handle_cmdline_arguments(parser=None):
    # create an argument parser
    parser = parser or argparse.ArgumentParser(
        description="TF2 classification", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = add_arguments(parser)
    hparams = parser.parse_args()
    hparams = parse_yaml_config(hparams, parser)

    # prepare wandb params
    hparams.wandb_params = {}
    if hparams.wandb_run_name is not None:
        hparams.wandb_params["run_name"] = hparams.wandb_run_name
    if hparams.wandb_tags is not None:
        hparams.wandb_params["tags"] = hparams.wandb_tags

    hparams.validation_micro_batch_size = hparams.validation_micro_batch_size or hparams.micro_batch_size

    hparams.num_pipeline_stages = len(hparams.pipeline_splits) + 1

    hparams.num_ipus_per_replica = (
        hparams.num_pipeline_stages if not hparams.device_mapping else max(hparams.device_mapping) + 1
    )

    # check if the script has been called by poprun
    hparams.distributed_training = popdist.isPopdistEnvSet()

    if hparams.distributed_training:

        if hparams.num_replicas != popdist.getNumTotalReplicas():
            logging.warning(
                f"Replication factor given to poprun (=={popdist.getNumTotalReplicas()}) "
                f"does not match the config (=={hparams.num_replicas}). Poprun will override the config."
            )
            hparams.num_replicas = popdist.getNumTotalReplicas()

        hvd.init()
        popdist.init()
        hparams.num_hosts = hvd.size() // hvd.local_size()

        logging.info(f"Total number of instances {popdist.getNumInstances()}")
        logging.info(f"Local number of instances {hvd.local_size()}")

        if (
            not hparams.pipeline_validation_model
            and hparams.validation
            and hparams.training
            and hparams.num_ipus_per_replica > 1
        ):
            raise ValueError(
                "A pipelined model over more than 1 IPU trained with poprun, must also be pipelined during validation."
                " Set --pipeline-validation-model True."
            )

        if popdist.getInstanceIndex() != 0:
            hparams.wandb = False

        if hvd.local_rank() != 0 and not hparams.ckpt_all_instances:
            hparams.clean_dir = False

    else:
        if hparams.ckpt_all_instances:
            hparams.ckpt_all_instances = False
            logging.warning("There is only one instance, setting --ckpt-all-instances to False.")

        if hparams.accelerator_side_reduction:
            logging.warning(
                "--accelerator-side-reduction can only be enabled with distributed training strategy. "
                "Overwriting the setting to False."
            )
            hparams.accelerator_side_reduction = False

    hparams.num_instances = popdist.getNumInstances() if hparams.distributed_training else 1

    if hparams.pipeline_validation_model:
        hparams.validation_num_replicas = hparams.validation_num_replicas or hparams.num_replicas
        hparams.validation_gradient_accumulation_count = 2 * (len(hparams.pipeline_splits) + 1)
        hparams.validation_ipus_per_replica = hparams.num_ipus_per_replica
    else:
        hparams.validation_num_replicas = hparams.validation_num_replicas or (
            hparams.num_replicas * hparams.num_ipus_per_replica
        )
        hparams.validation_gradient_accumulation_count = 1
        hparams.validation_ipus_per_replica = 1

    if hparams.validation:
        if hparams.distributed_training:
            if hparams.validation_num_replicas != popdist.getNumTotalReplicas() * popdist.getNumIpusPerReplica():
                logging.warning(
                    f"Validation replication factor given to poprun "
                    f"(=={popdist.getNumTotalReplicas() * popdist.getNumIpusPerReplica()}) "
                    f"does not match the config (=={hparams.validation_num_replicas}). Poprun will override the config."
                )
                hparams.validation_num_replicas = popdist.getNumTotalReplicas() * popdist.getNumIpusPerReplica()

            if hparams.validation_ipus_per_replica != popdist.getNumIpusPerReplica():
                raise ValueError(
                    f"The number of ipus per replica in validation does not match the value provided to poprun"
                    f"({hparams.validation_ipus_per_replica} != {popdist.getNumIpusPerReplica()})"
                )

    # when neither option is specified, assume gradient accumulation count 1
    if hparams.gradient_accumulation_count is None and hparams.global_batch_size is None:
        hparams.gradient_accumulation_count = 1

    if hparams.recomputation and not len(hparams.pipeline_splits):
        raise ValueError("Recomputation requires a pipelined model. " 'Make sure "--pipeline-splits" is defined')

    if (not len(hparams.pipeline_splits)) and hparams.pipeline_validation_model:
        logging.warn("Pipeline splits have not been defined, turning off the pipeline-validation-model option")
        hparams.pipeline_validation_model = False

    if hparams.logs_per_epoch < 0:
        raise ValueError(f"--logs-per-epoch should be non-negative (>=0), it is {hparams.logs_per_epoch}")

    # check for partial logs, example --logs-per-epoch 0.5 and --epochs 5
    if (
        (hparams.logs_per_epoch > 0)
        and (hparams.logs_per_epoch < 1)
        and (hparams.num_epochs % (1 / hparams.logs_per_epoch) != 0)
    ):
        raise ValueError(
            f"It is not possible to log {1/hparams.logs_per_epoch} epochs a time for {hparams.num_epochs} epochs"
        )

    if hparams.ckpts_per_epoch < 0:
        raise ValueError(f"--ckpts-per-epoch should be non-negative (>=0), it is {hparams.ckpts_per_epoch}")

    if hparams.first_ckpt_epoch < 0:
        raise ValueError(f"--first-ckpt-epoch should be non-negative (>=0), it is {hparams.first_ckpt_epoch}")

    if hparams.device_mapping:
        if len(hparams.device_mapping) != hparams.num_pipeline_stages:
            raise DimensionError(
                f"The number of device assignments {len(hparams.device_mapping)} is not equal to the number of pipeline splits + 1: {hparams.num_pipeline_stages}."
            )

        if len(set(hparams.device_mapping)) != max(hparams.device_mapping) + 1:
            raise DimensionError(
                f"The model is pipelined over {len(set(hparams.device_mapping))} different IPUs, but one or more stages are being assigned to IPU {max(hparams.device_mapping) + 1}"
            )

    if hparams.eight_bit_transfer and not hparams.accelerator_side_preprocess:
        raise UnallowedConfigurationError(
            f"When eight bit transfer is enabled the normalisation must be done on the device. "
            f"If you want to keep 8bit io, set --accelerator-side-preprocess to True."
        )

    if len(hparams.available_memory_proportion) > 1 and hparams.num_pipeline_stages == 1:
        raise UnallowedConfigurationError(
            "Setting available memory proportion per pipeline stage, "
            "but no pipeline stages defined. Please use --pipeline-splits to define the pipeline stages"
        )

    if (
        len(hparams.available_memory_proportion) > 1
        and len(hparams.available_memory_proportion) != 2 * hparams.num_pipeline_stages
    ):
        raise DimensionError(
            "Define a single global value of available memory proportion or two values per pipeline stage. "
            f"There are {hparams.num_pipeline_stages} pipeline stages defined and {len(hparams.available_memory_proportion)} values of "
            "available memory proportion"
        )

    if hparams.ckpts_per_epoch == 0 and hparams.ckpt_all_instances:
        raise ValueError(
            "All instances cannot save weights when checkpointing is disabled. "
            "Specify a non zero --ckpts-per-epoch to save weights for all instances, or disable --ckpt-all-instances otherwise."
        )

    if hparams.dbn_replica_group_size > 1 and hparams.num_ipus_per_replica != 1:
        raise ValueError("Distributed Batch Norm can only be applied when model fits on a single ipu.")

    if hparams.dbn_replica_group_size > 1 and hparams.num_replicas % hparams.dbn_replica_group_size != 0:
        raise ValueError(
            "Distributed Batch Norm can only be applied when model is replicated, "
            "and replication factor is divisible by dbn-replica-group-size."
        )

    if hparams.fused_preprocessing is True and hparams.accelerator_side_preprocess is False:
        raise ValueError(
            "Fused preprocessing can only be done in the IPU. "
            "Set both --fused_preprocessing and --accelerator-side-preprocess to True"
        )

    if hparams.norm_layer["name"] not in {"batch_norm", "custom_batch_norm", "group_norm"}:
        raise ValueError(f'Normalization layer {hparams.norm_layer["name"]} not supported.')

    wandb_params_keys = set(hparams.wandb_params.keys())
    possible_keys = {"entity", "project_name", "run_name", "tags"}
    unexpected_keys = wandb_params_keys - possible_keys
    if len(unexpected_keys) > 0:
        raise ValueError(f"wandb params contains unexpected fields: {unexpected_keys}")

    if hparams.on_demand and hparams.compile_only:
        logging.warning(
            "Both --on-demand and --compile-only control the attachment to IPUs, "
            "only one can be enabled at the time. Overwriting --on-demand to False."
        )
        hparams.on_demand = False

    if hparams.deterministic and hparams.shuffle:
        logging.warning(
            "In order to enforce a deterministic run shuffling must be disabled. " "Overwriting --shuffle with False."
        )
        hparams.shuffle = False

    if hparams.auto_loss_scaling:
        if hparams.optimizer != "sgd":
            raise ValueError(
                "Only SGD optimizer is supported when enabling automatic loss scaling. "
                f"Please change your specified {hparams.optimizer} optimizer."
            )

        if hparams.optimizer_params["momentum"] != 0:
            raise ValueError(
                "No optimizer momentum is supported when enabling automatic loss scaling. "
                f"Please change your specified momentum to 0."
            )

        not_allowed_optimizer_params_with_als = ["eeta", "epsilon", "weight_decay"]
        if bool(set(hparams.optimizer_params.keys()) & set(not_allowed_optimizer_params_with_als)):
            raise ValueError(
                "No optimizer parameters are supported when enabling automatic loss scaling. "
                f"Please do not specify values for {list(set(hparams.optimizer_params.keys()) & set(not_allowed_optimizer_params_with_als))}"
            )

        if hparams.lr_schedule != "const":
            raise ValueError(
                "Only constant learning rate schedule is supported when enabling automatic loss scaling. "
                f"Please change your specified {hparams.lr_schedule} learning rate schedule."
            )

        not_allowed_lr_schedule_params_with_als = ["epochs_to_total_decay", "end_learning_rate_ratio", "power"]
        if bool(set(hparams.lr_schedule_params.keys()) & set(not_allowed_lr_schedule_params_with_als)):
            raise ValueError(
                "Only initial_learning_rate can be specified for the learning rate schedule when enabling automatic loss scaling. "
                f"Please do not specify values for {list(set(hparams.lr_schedule_params.keys()) & set(not_allowed_lr_schedule_params_with_als))}"
            )

        if hparams.lr_warmup_params is not None:
            raise ValueError(
                "No learning rate warmup is supported when enabling automatic loss scaling. "
                f"Please do not specify values for {hparams.lr_warmup_params}"
            )

    logging.info(f"hyperparams = {hparams}")
    return hparams
