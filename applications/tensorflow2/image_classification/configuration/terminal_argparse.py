# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import yaml
import argparse
from pathlib import Path
from precision import Precision
from tensorflow.python.ipu.ops import pipelining_ops
from schedules.scheduler_builder import AVAILABLE_SCHEDULERS
from optimizers.optimizer_factory import AVAILABLE_OPTIMIZERS
from ipu_config import AVAILABLE_SR_OPTIONS


def add_arguments(parser):
    # Configuration
    parser.add_argument('--config', type=str,
                        help="Select from available configurations")
    parser.add_argument('--config-path', type=str, default=str(Path(Path(__file__).parent, Path("../configs.yml"))),
                        help='path to the configuration file')
    parser.add_argument('--on-demand', type=str_to_bool, nargs='?', const=True, default=True,
                        help='If true, it will defer connection to when the IPU is needed')
    parser.add_argument('--seed', type=int, default=None,
                        help='Set a global seed for the random number generator.')

    # Checkpoints
    parser.add_argument('--checkpoints', type=str_to_bool, nargs='?', const=True, default=True,
                        help='If true, it will save weights to files at each callback')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Path to checkpoints, if no argument is provided, directory /tmp/checkpoints_current_time/ will be used')
    parser.add_argument('--ckpt-all-instances', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Allow all instances to create a checkpoint. By default only local instance 0 does checkpointing.')
    parser.add_argument('--clean-dir', type=str_to_bool, nargs='?', const=True, default=True,
                        help='If true, it will delete the checkpoint directory (and all the files inside)')

    # Dataset and model choice arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Name of dataset to use')
    parser.add_argument('--dataset-path', type=str, default='.',
                        help='Path to dataset')
    parser.add_argument('--model-name', type=str, default='toy_model',
                        help='Name of model to use')
    parser.add_argument('--eight-bit-transfer', type=str_to_bool, nargs="?", const=True, default=False,
                        help='Enable/disable input transfer in 8 bit')
    parser.add_argument('--synthetic-data', type=str, default=None,
                        help='Enable usage of synthetic data on the host or ipu. Corresponding options are \'host\' or \'ipu\'')

    # Training parameter arguments
    parser.add_argument('--training', type=str_to_bool, nargs="?", const=True, default=True,
                        help='Enable/disable training')
    parser.add_argument('--micro-batch-size', type=int, default=1,
                        help="Micro batch size, in number of samples")
    parser.add_argument('--num-epochs', type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument('--logs-per-epoch', type=str_to_float, default='1',
                        help="Logging frequency, per epoch")
    parser.add_argument('--weight-updates-per-epoch', type=int, default=-1,
                        help="number of weight updates per run on the device for one epoch")
    parser.add_argument('--num-replicas', type=int, default=1,
                        help="Number of training replicas")
    parser.add_argument('--gradient-accumulation-count', type=int, default=None,
                        help="Number of gradients accumulated by each replica")
    parser.add_argument('--global-batch-size', type=int, default=None,
                        help='Global batch size, in number of samples')
    parser.add_argument('--precision', type=str, default='16.16', choices=Precision.supported_precisions,
                        help="<compute precision>.<weight update precision> both 16 or 32")
    parser.add_argument('--pipeline-splits', type=str, nargs='*', default=[],
                        help="Model layers that define the start of a new pipeline stage. E.g. conv2d_1 max_pooling2d")
    parser.add_argument('--device-mapping', type=int, nargs='*', default=None,
                        help="List mapping pipeline stages to IPU numbers. E.g. 0 1 1 0")
    parser.add_argument('--pipeline-schedule', type=str, default='Grouped', choices=[str(p).split(".")[-1] for p in list(
        pipelining_ops.PipelineSchedule)], help="Pipelining schedule. Choose between 'Interleaved', 'Grouped' and 'Sequential'.")
    parser.add_argument('--optimizer', type=str, default='sgd', choices=AVAILABLE_OPTIMIZERS,
                        help='The name of the optimizer to use')
    parser.add_argument('--optimizer-params', type=yaml.safe_load, default='{"momentum": 0}',
                        help='Parameters to configure the optimizer with. To pass this argument from the terminal '
                             'use --optimizer-params \'{"arg1": value1, "arg2": value2...}\' format.')
    parser.add_argument('--loss-scaling', type=float, default=0.,
                        help='The value of static loss scaling. When equal to 0, loss scaling is disabled.')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='The value of weight decay used by the optimizer.')
    parser.add_argument('--l2-regularization', type=float, default=0.,
                        help='The value of l2 regularization used by the optimizer.')
    parser.add_argument('--recomputation', type=str_to_bool, nargs="?", const=True, default=False,
                        help='Enable/disable recomputation of activations in the backward pass')
    parser.add_argument('--accelerator-side-preprocess', type=str_to_bool, nargs="?", const=True, default=False,
                        help='When enabled some preprocessing steps (depending on the chosen dataset), are run '
                             'on the accelerator rather on the host.')
    parser.add_argument('--accelerator-side-reduction', type=str_to_bool, nargs="?", const=True, default=False,
                        help='Requires distributed training. When enabled the reduction over replicas for logging '
                             'is performed on the device rather than the host.')
    parser.add_argument('--stochastic-rounding', type=str, default='ON', choices=AVAILABLE_SR_OPTIONS.keys(),
                        help='Enable one of three different stochastic rounding modes: ON, OFF or RI (Replica Identical).')
    parser.add_argument('--optimizer-state-offloading', type=str_to_bool, nargs='?', const=True, default=True,
                        help='Enable/disable the offloading of the optimizer state to the IPU remote memory.')
    parser.add_argument('--fp-exceptions', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Enable/disable floating point exceptions.')
    parser.add_argument('--lr-schedule', type=str, default='const', choices=list(AVAILABLE_SCHEDULERS.keys()),
                        help='Type of learning rate schedule. By default, a constant learning rate of 1e-3 is used.')
    parser.add_argument('--lr-warmup-params', type=yaml.safe_load, default=None,
                        help='Parameters to configure the warmup of learning rate. To pass this argument from the terminal '
                             'type --lr-schedule-params \'{"warmup_mode": <mode>, "warmup_epochs": <epochs>}\' format.')
    parser.add_argument('--lr-schedule-params', type=yaml.safe_load, default='{"initial_learning_rate": 0.0001}',
                        help='Parameters to configure learning rate scheduler. To pass this argument from the terminal '
                             'type --lr-schedule-params \'{"arg1": value1, "arg2": value2...}\' format.')
    parser.add_argument('--lr-staircase', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Apply a staircase on the learning rate schedule, making learning rate values constant throughout '
                             'one epoch.')
    parser.add_argument('--dbn-replica-group-size', type=int, default=1,
                        help='Distributed Batch Norm (DBN) option specifies how many replicas to aggregate the batch statistics across. '
                             'DBN is disabled when ==1. It can be enabled only if model fits on a single ipu (num ipus per replica ==1), '
                             'model is replicated (num replicas > 1) and replication factor is divisible by dbn replica group size.')
    parser.add_argument('--label-smoothing', type=float, default=None,
                        help='Smoothing factor added to each zero label')
    parser.add_argument('--pipeline-num-parallel', type=int, default=48,
                        help='Number of images to process in parallel on the host side.')
    parser.add_argument('--norm-layer', type=yaml.safe_load, default='{"name": "custom_batch_norm", "momentum": 0.97}',
                        help='Type of normalisation layer to use. When using group norm specify either num_groups or channels_per_group. '
                             'When using batch norm specify momentum.')
    parser.add_argument('--fused-preprocessing', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Use fused operations for preprocessing images on device.')

    # Poplar optimizations
    parser.add_argument('--half-partials', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Accumulate matmul and convolution partial results in half precision')
    parser.add_argument('--internal-exchange-optimization-target', type=str, default=None, choices=['cycles', 'memory', 'balanced'],
                        help='Set poplar internal exchange optimization target. Default is cycles.')
    parser.add_argument('--max-cross-replica-buffer-size', type=int, default=0,
                        help='The maximum number of bytes that can be waiting before a cross replica sum op is scheduled. '
                        '0 (default) means that they are scheduled immediately.')
    parser.add_argument('--max-reduce-many-buffer-size', type=int, default=0,
                        help='The maximum size (in bytes) a cluster of reduce operations can reach before it is scheduled. '
                             'These clusters are lowered to popops ReduceMany operations.')
    parser.add_argument('--gather-conv-output', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Reduce sync cost of small sized all-reduces. Useful when paired with distributed batch norm')
    parser.add_argument('--stable-norm', type=str_to_bool, nargs='?', const=True, default=False,
                        help='Enable/disable numerically more stable but less parallelizable normalization layers.')
    parser.add_argument('--available-memory-proportion', type=float, nargs='*', default=[],
                        help='The percentage of IPU memory dedicated to convolutions and matmuls.')

    # Evaluation and logging choice arguments
    parser.add_argument('--wandb', type=str_to_bool, nargs="?", const=True, default=False,
                        help='Enable/disable logging to Weights & Biases')
    parser.add_argument('--wandb-params', type=yaml.safe_load, default='{}',
                        help='Parameters to configure Weights & Biases. Available options include "project_name" and "run_name", '
                        'passed a dictionary --wandb-params \'{"entity": <str>, "project_name": <str>, "run_name": <str>, "tags": [<str>, ...]}\'.')
    parser.add_argument('--validation', type=str_to_bool, nargs="?", const=True, default=True,
                        help='Enable/disable validation')
    parser.add_argument('--validation-micro-batch-size', type=int, default=None,
                        help="Validation micro batch size, in number of samples")
    parser.add_argument('--validation-num-replicas', type=int, default=None,
                        help="Number of validation replicas")
    parser.add_argument('--pipeline-validation-model', type=str_to_bool, nargs="?", const=True, default=False,
                        help='Reuse the training pipeline splits for validation')

    return parser


def str_to_bool(value):
    # boolean args can be used as flags to  set value = const
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n', 'off'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y', 'on'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def str_to_float(frac_str):

    list_string = frac_str.split('/')

    if len(list_string) == 1:
        return float(list_string[0])

    elif len(list_string) == 2:
        try:
            num, denom = float(list_string[0]), float(list_string[1])
        except:
            raise argparse.ArgumentTypeError(
                f'Could not parse {frac_str} as a fraction. The fraction numerator or denominator could be missing or one of those could not be parsed as a floating point number.')

        return num / denom

    else:
        raise argparse.ArgumentTypeError(f'Number should be provided as float or fraction like a/b')
