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
from fastspeech2 import FastSpeech2Config


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
    group.add_argument('--train', action='store_true',
                       help="Whether to run training loop or not. Default is True.")
    group.add_argument('--eval', action='store_true',
                       help="""
                       Whether to run evaluation loop or not. Default is False.
                       If both `--train` and `--eval` are set, then the evaluation will be done after trainning.
                       """)
    group.add_argument('--batch-size', type=int,
                       help="Set batch-size for training graph")
    group.add_argument('--base-learning-rate', type=float, default=2e-5,
                       help="Base learning rate exponent (2**N). blr = lr /  bs")
    group.add_argument('--loss-scaling', type=float, default=1,
                       help="Loss scaling factor.")
    group.add_argument('--optimizer', type=str, default="adamw", choices=['sgd', 'adam', 'adamw'],
                       help="Optimizer")
    group.add_argument('--beta1', type=float, default=0.9,
                       help="Adam/AdamW beta1 coefficient.")
    group.add_argument('--beta2', type=float, default=0.999,
                       help="Adam/AdamW beta2 coefficient.")
    group.add_argument('--weight-decay-rate', type=float, default=0.0,
                       help="Weight decay to use during optimisation.")
    group.add_argument('--decay-steps', type=int, default=1000,
                       help="Learning rate schedule decay steps.")
    group.add_argument('--decay-date', type=float, default=0.9,
                       help="Learning rate schedule decay rate. Only for Exponential.")
    group.add_argument('--epsilon', type=float, default=1e-4,
                       help="Optimiser epsilon value.")
    group.add_argument('--lr-schedule', default='exponential', choices=["exponential", "cosine", "polynomial"],
                       help="Learning rate schedule function.")
    group.add_argument('--warmup', default=0.1, type=float,
                       help="Learning rate schedule warm-up period, in epochs (float) or number of steps (integer).")
    group.add_argument('--seed', default=None,
                       help="Seed for randomizing training")
    group.add_argument('--wandb', type=_str_to_bool, default=False,
                       help="Enable logging and experiment tracking with Weights & Biases.")
    group.add_argument('--wandb-name', type=str, default=None,
                       help="Override Weights&Biases run name.")
    group.add_argument('--init-checkpoint', type=str, default=None,
                       help='Initialise a new training session from this checkpoint.')
    group.add_argument('--epochs', default=None, type=int,
                       help='Number of epochs we want to let the training last.')
    group.add_argument('--epochs-per-save', default=5, type=int,
                       help='Number of epochs to save model.')
    group.add_argument('--steps-per-epoch', default=None, type=int,
                       help='Specifies the total number of steps to be performed per epoch.')

    # FastSpeech2 options
    group.add_argument('--vocab-size', type=int,
                       help="""Vocabulary size of `FastSpeech2`.""")
    # Encoder
    group.add_argument('--encoder-hidden-size', type=int,
                       help="""Size ofthe encoder layers and the pooler layer.""")
    group.add_argument('--encoder-num-hidden-layers', type=int,
                       help="""Number of hidden layers in the encoder.""")
    group.add_argument('--encoder-num-attention-heads', type=int,
                       help="""Number of attention heads for each attention layer in the encoder.""")
    group.add_argument('--encoder-intermediate-size', type=int,
                       help="""The size of the "intermediate" (i.e., feed-forward) layer in the encoder.""")
    group.add_argument('--encoder-intermediate-kernel-size', type=int,
                       help="""The convolution kernel size of the "intermediate" (i.e., feed-forward) layer in the encoder.""")
    group.add_argument('--encoder-hidden-act', type=int,
                       help="""The non-linear activation function (function or string) in the encoder and pooler.""")
    group.add_argument('--output-attentions', type=_str_to_bool, default=False,
                       help="Whether to output the attentions.")
    group.add_argument('--output-hidden-states', type=_str_to_bool, default=False,
                       help="Whether to output the hidden states.")
    group.add_argument('--layer-norm-eps', type=float, default=1e-4,
                       help="Layer Normalization epsilon value.")
    # Decoder
    group.add_argument('--decoder-hidden-size', type=int,
                       help="""Size ofthe decoder layers and the pooler layer.""")
    group.add_argument('--decoder-num-hidden-layers', type=int,
                       help="""Number of hidden layers in the decoder.""")
    group.add_argument('--decoder-num-attention-heads', type=int,
                       help="""Number of attention heads for each attention layer in the decoder.""")
    group.add_argument('--decoder-intermediate-size', type=int,
                       help="""The size of the "intermediate" (i.e., feed-forward) layer in the decoder.""")
    group.add_argument('--decoder-intermediate-kernel-size', type=int,
                       help="""The convolution kernel size of the "intermediate" (i.e., feed-forward) layer in the decoder.""")
    group.add_argument('--decoder-hidden-act', type=int,
                       help="""The non-linear activation function (function or string) in the decoder and pooler.""")

    # Duration predictor(Only for FastSpeech)
    group.add_argument('--duration-predictor-num-conv-layers', type=int, default=2,
                       help="""Number of convolution layers in the duration predictor.""")
    group.add_argument('--duration-predictor-kernel-size', type=int, default=3,
                       help="""The convolution kernel size of the duration predictor.""")
    group.add_argument('--duration-predictor-dropout-probs', type=int, default=0.5,
                       help="""The dropout probability for duration predictor.""")
    group.add_argument('--duration-predictor-filters', type=int, default=256,
                       help="""Number of filters in the duration predictor.""")
    # Variance predictor and Postnet
    group.add_argument('--variant-predictor-num-conv-layers', type=int, default=2,
                       help="""Number of convolution layers in the variant predictor.""")
    group.add_argument('--variant-predictor-kernel-size', type=int, default=3,
                       help="""The convolution kernel size of the variant predictor.""")
    group.add_argument('--variant-predictor-dropout-rate', type=int, default=0.5,
                       help="""The dropout probability for variant predictor.""")
    group.add_argument('--variant-predictor-filter', type=int, default=256,
                       help="""Number of filters in the variant predictor.""")
    group.add_argument('--use-postnet', type=_str_to_bool, default=True,
                       help="Whether to use postnet after decoder.")
    group.add_argument('--postnet-num-conv-layers', type=int,
                       help="""Number of convolution layers in the postnet.""")
    group.add_argument('--postnet-conv-kernel-size', type=int,
                       help="""The convolution kernel size of the postnet.""")
    group.add_argument('--postnet-dropout-rate', type=int,
                       help="""The dropout probability for postnet.""")
    group.add_argument('--postnet-conv-filters', type=int,
                       help="""Number of convolution filters in the postnet.""")

    group.add_argument('--hidden-dropout-prob', type=float, default=0.2,
                       help="""The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.""")
    group.add_argument('--attention-probs-dropout-prob', type=float, default=0.1,
                       help="""The dropout ratio for the attention probabilities.""")
    group.add_argument('--max-position-embeddings', type=int,
                       help="""The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).""")
    group.add_argument('--initializer-range', type=float, default=0.02,
                       help="""The stdev of the truncated-normal-initializer for initializing all weight matrices.""")
    group.add_argument('--max-seq-length', type=int, default=189,
                       help='The maximuim sequence length.')
    group.add_argument('--max-wave-length', type=int, default=870,
                       help='The maximuim wave length.')
    group.add_argument('--num-mels', type=int, default=80,
                       help='The number of mel-spectrograms dimensions.')

    # IPU options
    pipeline_schedule_options = [
        _.name for _ in ipu.pipelining_ops.PipelineSchedule]
    schedulers_available = ['Clustering',
                            'PostOrder', 'LookAhead', 'ShortestPath']
    recomputation_modes_available = [
        p.name for p in ipu.pipelining_ops.RecomputationMode]

    group.add_argument('--gradient-accumulation-count', type=int, default=None,
                       help="Number of gradients to accumulate in the pipeline. Must also set --shards > 1.")
    group.add_argument('--pipeline-schedule', type=str, default="Interleaved", choices=pipeline_schedule_options,
                       help="Pipelining scheduler.")
    group.add_argument('--replicas', type=int, default=1,
                       help="Replicate graph over N workers to increase batch to batch-size*N")
    group.add_argument('--precision', type=str, default="16", choices=["16", "32"],
                       help="Precision of Ops(weights/activations/gradients) data types: 16, 32.")
    group.add_argument('--available-memory-proportion', type=str, default=0.23,
                       help="Proportion of IPU memory available to matmul operations. A list can be used to specify the value for each IPU.")
    group.add_argument('--variable-offloading', type=_str_to_bool, default=True,
                       help="Enable offloading of training variables into remote memory.")
    group.add_argument('--stochastic-rounding', type=_str_to_bool, default=True,
                       help="Enable stochastic rounding. Set to False when run evaluation.")
    group.add_argument('--fp-exceptions', default=False, action="store_true",
                       help="Enable floating-point exeptions.")
    group.add_argument('--partials-type', type=str, default="half", choices=["half", "float"],
                       help="Floating-point precision of data in matmul and convolution operations..")
    group.add_argument('--scheduler', type=str, default='Clustering', choices=schedulers_available,
                       help="Forces the compiler to use a specific scheduler when ordering the instructions.")
    group.add_argument('--recomputation-mode', type=str,
                       default="RecomputeAndBackpropagateInterleaved", choices=recomputation_modes_available)
    group.add_argument('--reduction-type', type=str, choices=['sum', 'mean'], default='mean',
                       help='The reduction type applied to the pipeline, the choice is between summation and mean.')
    group.add_argument('--weight-norm-clip', type=float, default=0.,
                       help='The value from which we want to clip the w_norm value, value of 0 is no weight clipping.')
    group.add_argument('--num-io-tiles', type=int, default=0,
                       help='The number of IO tiles, default is 0 which means no IO tiles are used.')

    # Dataset options
    group.add_argument('--data-path', type=str, required=False,
                       help="path to training/validation dataset tfrecord file.")
    group.add_argument('--parallell-io-threads', type=int, default=4,
                       help="Number of cpu threads used to do data prefetch.")
    group.add_argument('--generated-data', action="store_true", default=False,
                       help="Generates synthetic-data on the host and then use it for training.")

    # Add logging-specific arguments
    group_log = parser.add_argument_group('Logging')
    group_log.add_argument('--log-dir', type=str,
                           help="Log and weights save directory")
    group_log.add_argument('--steps-per-logs', type=int, default=1,
                           help="Logs per epoch (if number of epochs specified)")
    return parser


def create_command_line_parser():
    parser = argparse.ArgumentParser(description="""FastSpeech2 on Graphcore's IPU""",
                                     add_help=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_argument_group("""Main options""")
    group.add_argument('--help', action='store_true',
                       default=False, help="Display help.")
    group.add_argument('--config', type=str,
                       help='FastSpeech2 configuration file in JSON format.')
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
    opts_from_config_file = FastSpeech2Config.from_json_file(config_file_path)

    # Build the global options structure from the default options
    current_options = vars(all_options_parser.parse_args())

    unknown_options = [opt for opt in opts_from_config_file.keys(
    ) if opt not in current_options.keys()]

    if unknown_options:
        logging.error(f"Unonwn options: {unknown_options}")
        sys.exit(os.EX_USAGE)

    # Overwrite global options by those specified in the config file.
    current_options.update(opts_from_config_file)
    options_namespace = argparse.Namespace(**current_options)

    # Overwrite with command-line arguments
    all_options_namespace = all_options_parser.parse_args(
        unknown_command_line_args, options_namespace)
    logging.info(
        f"Overwrite configuration parameters: {', '.join(unknown_command_line_args)}")

    # argparse.Namespace -> dict()
    opts = vars(all_options_namespace)

    return opts
