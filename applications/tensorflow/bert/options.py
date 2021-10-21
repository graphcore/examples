# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import sys

from tensorflow.python import ipu

import log
from modeling import BertConfig


def _str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def add_common_arguments(parser, required=True):
    group = parser.add_argument_group('Common options')

    # Training options
    # fmt:off
    group.add_argument('--batch-size', type=int,
                       help="Set batch-size for training graph")
    group.add_argument('--global-batch-size', type=int, default=None,
                       help="The total batch size at which we want the model to run")
    group.add_argument('--base-learning-rate', type=float, default=2e-5,
                       help="Base learning rate exponent (2**N). blr = lr /  bs")
    group.add_argument('--num-train-steps', type=int,
                       help="Number of training steps.")
    group.add_argument('--loss-scaling', type=float, default=1,
                       help="Loss scaling factor.")
    group.add_argument('--loss-scaling-by-step', type=str, default=None,
                       help="Specify changing loss scaling factors at given training steps, as a dictionary.")
    group.add_argument('--steps-per-ckpts', type=int, default=256,
                       help="Steps per checkpoints")
    group.add_argument('--optimizer', type=str, default="momentum", choices=['sgd', 'momentum', 'adamw', 'lamb'],
                       help="Optimizer")
    group.add_argument('--momentum', type=float, default=0.984375, help="Momentum coefficient.")
    group.add_argument('--beta1', type=float, default=0.9, help="lamb/adam beta1 coefficient.")
    group.add_argument('--beta2', type=float, default=0.999, help="lamb/adam beta2 coefficient.")
    group.add_argument('--weight-decay-rate', type=float, default=0.0,
                       help="Weight decay to use during optimisation.")
    group.add_argument('--epsilon', type=float, default=1e-4,
                       help="Optimiser epsilon value.")
    group.add_argument('--lr-schedule', default='exponential', choices=["custom", "natural_exponential", "polynomial"],
                       help="Learning rate schedule function.")
    group.add_argument('--lr-schedule-by-step', type=str,
                       help="Dictonary of changes in the learning rate at specified steps.")
    group.add_argument('--warmup', default=0.1,
                       help="Learning rate schedule warm-up period, in epochs (float) or number of steps (integer).")
    group.add_argument('--seed', default=None,
                       help="Seed for randomizing training")
    group.add_argument('--wandb', action='store_true',
                       help="Enable logging and experiment tracking with Weights & Biases.")
    group.add_argument('--wandb-name', type=str, default=None,
                       help="Override Weights&Biases run name.")
    group.add_argument('--save-path', type=str, default="checkpoints",
                       help='Save checkpoints to this directory.')
    group.add_argument('--init-checkpoint', type=str, default=None,
                       help='Initialise a new training session from this checkpoint.')
    group.add_argument('--restore-dir', type=str, default=None,
                       help='Path to directory containing the checkpoint to restore.')
    group.add_argument('--restore-optimiser-from-checkpoint', default=True, action="store_true")
    group.add_argument('--save-optimiser-to-checkpoint', default=True, action="store_true")
    group.add_argument('--epochs', default=None, type=float,
                       help='Number of epochs we want to let the training last.')

    # BERT options
    group.add_argument('--vocab-size', type=int,
                       help="""Vocabulary size of `inputs_ids` in `BertModel`.""")
    group.add_argument('--hidden-size', type=int,
                       help="""Size ofthe encoder layers and the pooler layer.""")
    group.add_argument('--num-hidden-layers', type=int,
                       help="""Number of hidden layers in the Transformer encoder.""")
    group.add_argument('--num-attention-heads', type=int,
                       help="""Number of attention heads for each attention layer in the Transformer encoder.""")
    group.add_argument('--intermediate-size', type=int,
                       help="""The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.""")
    group.add_argument('--hidden-act', type=int,
                       help="""The non-linear activation function (function or string) in the encoder and pooler.""")
    group.add_argument('--hidden-dropout-prob', type=int,
                       help="""The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.""")
    group.add_argument('--attention-probs-dropout-prob', type=int,
                       help="""The dropout ratio for the attention probabilities.""")
    group.add_argument('--max-position-embeddings', type=int,
                       help= """The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).""")
    group.add_argument('--type-vocab-size', type=int,
                       help= """The vocabulary size of the `token_type_ids` passed into `BertModel`.""")
    group.add_argument('--initializer-range', type=int,
                       help= """The stdev of the truncated-normal-initializer for initializing all weight matrices.""")
    group.add_argument('--disable-acc', default=False, action='store_true',
                       help='Makes the calculation of the accuracy optional.')
    group.add_argument('--use-qkv-bias', default=False, action='store_true',
                       help='Use biases for QKV layer.')
    group.add_argument('--use-qkv-split', default=False, action='store_true',
                       help='Split QKV layer.')

    # Model options
    group.add_argument('--use-attention-projection-bias', type=_str_to_bool, default=True,
                       help="Whether to use bias in linear projection behind attention layer.")
    group.add_argument('--use-cls-layer', type=_str_to_bool, default=True,
                       help="""Include the CLS layer in pretraining. This layer comes after the encoders but before the projection for the MLM loss.""")
    group.add_argument('--use-prediction-bias', type=_str_to_bool, default=True,
                       help="""Whether to use bias in mlm prediction.""")
    group.add_argument('--static-mask', action='store_true', default=False,
                       help="Use if the pretraining dataset was created with the masked tokens always at the beginning of the input tensor.")
    group.add_argument("--max-predictions-per-seq", type=int, default=20,
                       help="the number of masked words per sentence.")
    group.add_argument('--duplicate-factor', default=5, type=int,
                       help='The amount of duplication factor inside the dataset.')

    # GroupBert architecture options
    group.add_argument('--groupbert', action='store_true', default=False,
                       help="Use GroupBert architecture")
    group.add_argument('--groupbert-ffn-output-groups', type=int, default=4,
                       help="Set number of groups for FFN layer in GroupBert")
    group.add_argument('--groupbert-conv-kernel-size', type=int, default=7,
                       help="Set size of convolution kernel in GroupBert")
    group.add_argument('--groupbert-conv-group-size', type=int, default=16,
                       help="Set size of convolution groups in GroupBert")

    # IPU options
    pipeline_schedule_options = [_.name for _ in ipu.ops.pipelining_ops.PipelineSchedule]
    schedulers_available = [_.name for _ in ipu.config.SchedulingAlgorithm]
    recomputation_modes_available = [p.name for p in ipu.ops.pipelining_ops.RecomputationMode]

    group.add_argument('--gradient-accumulation-count', type=int, default=None,
                       help="Number of gradients to accumulate in the pipeline. Must also set --shards > 1.")
    group.add_argument('--pipeline-schedule', type=str, default="Interleaved", choices=pipeline_schedule_options,
                       help="Pipelining scheduler.")
    group.add_argument('--replicas', type=int, default=1,
                       help="Replicate graph over N workers to increase batch to batch-size*N")
    group.add_argument('--precision', type=str, default="16", choices=["16", "32"],
                       help="Precision of Ops(weights/activations/gradients) data types: 16, 32.")
    group.add_argument('--batches-per-step', type=int, default=1,
                       help="Maximum number of batches to perform on the device before returning to the host.")
    group.add_argument('--available-memory-proportion', type=str, default=0.23,
                       help="Proportion of IPU memory available to matmul operations. A list can be used to specify the value for each IPU.")
    group.add_argument('--variable-offloading', type=_str_to_bool, default=True,
                       help="Enable offloading of training variables into remote memory.")
    group.add_argument('--min-remote-tensor-size', type=int, default=128,
                       help="The minimum remote tensor size (bytes) for partial variable offloading")
    group.add_argument('--replicated-tensor-sharding', action="store_true", default=False,
                       help="Enable replicated tensor sharding of optimizer state.")
    group.add_argument('--stochastic-rounding', type=_str_to_bool, default=True,
                       help="Enable stochastic rounding. Set to False when run evaluation.")
    group.add_argument('--no-outlining', type=_str_to_bool, default=False,
                       help="Disable TF outlining optimisations. This will increase memory for a small throughput improvement.")
    group.add_argument("--enable_recomputation", default=True, action="store_true",
                       help="Recompute activations during backward pass")
    group.add_argument('--fp-exceptions', default=False, action="store_true",
                       help="Enable floating-point exeptions.")
    group.add_argument('--partials-type', type=str, default="half", choices=["half", "float"],
                       help="Floating-point precision of data in matmul and convolution operations..")
    group.add_argument('--max-cross-replica-sum-buffer-size', type=int, default=10 * 1024 * 1024,
                       help="The maximum number of bytes that can be waiting before a cross replica sum op is scheduled.")
    group.add_argument('--max-reduce-scatter-buffer-size', type=int, default=0,
                       help="The maximum number of bytes that can be waiting before reduce-scatter op is scheduled.")
    group.add_argument('--scheduler', type=str, default="CLUSTERING", choices=schedulers_available,
                       help="Forces the compiler to use a specific scheduler when ordering the instructions.")
    group.add_argument('--recomputation-mode', type=str, default="RecomputeAndBackpropagateInterleaved", choices=recomputation_modes_available)
    group.add_argument('--increase-optimiser-precision', action='store_true', default=False,
                       help="In the LAMB optimiser, it performs more operations in fp32. This operation increase precision in the weight update but consumes more memory and reduce the Tput.")
    group.add_argument('--use-nvlamb', action='store_true', default=False,
                       help="Flag to use the global normalisation for the gradients.")
    group.add_argument('--use-debiasing', action='store_true', default=False,
                       help="Flag to use the de biasing for the momenta of LAMB")
    group.add_argument('--reduction-type', type=str, choices=['sum', 'mean'], default='mean',
                       help='The reduction type applied to the pipeline, the choice is between summation and mean.')
    group.add_argument('--weight-norm-clip', type=float, default=0.,
                       help='The value from which we want to clip the w_norm value, value of 0 is no weight clipping.')
    group.add_argument('--compile-only', action="store_true", default=False,
                       help="Configure Poplar to only compile the graph. This will not acquire any IPUs and thus facilitate profiling without using hardware resources.")
    group.add_argument('--matmul-serialize-factor', type=int, default=6,
                       help='Serialization factor of the embeddings lookup and projection. Must be a divisor of vocab_size.')
    group.add_argument('--glue-dropout-prob', type=float, default=0.1,
                       help='GLUE tasks dropout probability.')
    group.add_argument('--pipeline-stages', type=str,
                       help="""Pipeline stages, a list of [emb, pos, hid, mlm, nsp] layers forming the pipeline.""")
    group.add_argument('--device-mapping', type=str,
                       help="""Mapping of pipeline stages to IPU""")
    group.add_argument('--sync-replicas-independently', action='store_true', default=False,
                       help='All the replicas will be in sync.')
    group.add_argument('--log-all-workers', action='store_true',
                       help='Allow all the workers to log into the terminal and the files.')

    # Dataset options
    group.add_argument('--train-file', type=str, required=False,
                       help="path to wiki/corpus training dataset tfrecord file.")
    group.add_argument("--seq-length", type=int, default=128,
                       help="the max sequence length.")
    group.add_argument('--parallel-io-threads', type=int, default=4,
                       help="Number of cpu threads used to do data prefetch.")
    group.add_argument('--generated-data', action="store_true", default=False,
                       help="Generates synthetic-data on the host and then use it for training.")
    group.add_argument('--synthetic-data', action='store_true',
                       help="Run the model completely detaching it from the host.")
    group.add_argument('--dataset-repeat', type=int, default=1, help="Number of times dataset to repeat.")

    # Env flag specific arguments
    group.add_argument('--execution-profile', action='store_true',
                       help='Sets the Poplar engine options to output an execution profile to the profile-dir.')
    group.add_argument('--memory-profile', action='store_true',
                       help='Sets the Poplar engine options to output a memory profile to the profile-dir.')
    group.add_argument('--profile-dir', type=str, default='./',
                       help='Defines the directory where the profile will be found.')
    group.add_argument('--progress-bar', type=str, choices=['auto', 'true', 'false'], default='auto',
                       help='The compilation progress bar for the compilation. Pass false to disable it.')

    # fmt:on
    # Add logging-specific arguments
    log.add_arguments(parser)

    return parser


def create_command_line_parser():
    parser = argparse.ArgumentParser(description="""BERT on Graphcore's IPU""",
                                     add_help=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_argument_group("""Main options""")
    group.add_argument('--help', action='store_true', default=False, help="Display help.")
    group.add_argument('--task', type=str, choices=['pretraining'], help="Type of NLP task.")
    group.add_argument('--config', type=str, help='BERT configuration file in JSON format.')
    return parser


def create_all_options_parser():
    parser = create_command_line_parser()
    parser = add_common_arguments(parser)
    return parser


def make_global_options(task_specific_parsers=[]):
    # Parse command-line arguments
    command_line_parser = create_command_line_parser()
    all_options_parser = create_all_options_parser()

    for task_parser in task_specific_parsers:
        all_options_parser = task_parser(all_options_parser)

    known_command_line_args, unknown_command_line_args = command_line_parser.parse_known_args()

    if known_command_line_args.help or known_command_line_args.config is None:
        all_options_parser.print_help()
        sys.exit(os.EX_OK)

    # Parse options specified in the configuration file into
    config_file_path = known_command_line_args.config
    opts_from_config_file = BertConfig.from_json_file(config_file_path)

    # Build the global options structure from the default options
    current_options = vars(all_options_parser.parse_args())

    unknown_options = [opt for opt in opts_from_config_file.keys() if opt not in current_options.keys()]

    if unknown_options:
        logging.error(f"Unonwn options: {unknown_options}")
        sys.exit(os.EX_USAGE)

    # Overwrite global options by those specified in the config file.
    current_options.update(opts_from_config_file)
    options_namespace = argparse.Namespace(**current_options)

    # Overwrite with command-line arguments
    all_options_namespace = all_options_parser.parse_args(unknown_command_line_args, options_namespace)
    logging.info(f"Overwrite configuration parameters: {', '.join(unknown_command_line_args)}")

    # argparse.Namespace -> dict()
    opts = vars(all_options_namespace)

    return opts
