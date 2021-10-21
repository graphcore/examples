# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import argparse
import datetime
import logging
from typing import List, Optional

import numpy as np
import onnx

import popdist

from bert_model import BertConfig
from .weight_loading import load_initializers_from_onnx
from .distributed import setup_comm

logger = logging.getLogger(__name__)


def save_model_statistics(model_path, writer, i=0):
    initializers = load_initializers_from_onnx(model_path)
    for name, np_weight in initializers.items():
        name = name.replace(":", "_")
        writer.add_histogram(name, np_weight, i)
        writer.add_scalar(f"L2/{name}", np.linalg.norm(np_weight.astype(np.float32)), i)


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def parser_from_NamedTuple(parser, ntuple, args={}):
    for key in ntuple._fields:
        string = "--" + key.replace("_", "-")
        t = ntuple._field_types[key]
        default = ntuple._field_defaults.get(key, None)
        kwargs = dict(
            string=string,
            type=t,
            default=default,
            dest=key
        )
        if t is bool:
            # Make bool a flag
            kwargs["nargs"] = "?"
            kwargs["const"] = not kwargs["default"]
            kwargs["type"] = str_to_bool
        else:
            for _t in (str, int, float):
                if t == List[_t]:
                    kwargs["type"] = _t
                    kwargs["nargs"] = '+'
                    break
                if t == Optional[_t]:
                    kwargs["type"] = _t
                    break

        args_or_help = args.get(key, None)
        if isinstance(args_or_help, dict):
            kwargs.update(**args_or_help)
        else:
            kwargs["help"] = args_or_help
        string = kwargs["string"]
        del kwargs["string"]
        parser.add_argument(string, **kwargs)


class ScheduleArgumentParser(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(ScheduleArgumentParser, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs)
        self.default_input = kwargs['default']

    def __call__(self, parser, namespace, values, option_string=None):
        schedule = {}
        if len(values) == 0:
            schedule = self.default_input

        for kv in values:
            training_proportion, lr = kv.split(":")
            try:
                schedule[int(training_proportion)] = float(lr)
            except ValueError as ex:
                logger.warning("Invalid Learning Rate Schedule provided. "
                               "It should be a set of int:float pairs.")
                raise ex

        setattr(namespace, self.dest, schedule)


class ValidationConfig(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        value = None
        try:
            value = json.loads(values)
        except json.decoder.JSONDecodeError as e:
            pass
        if os.path.isfile(values):
            with open(values, 'r') as f:
                value = json.load(f)
        if value is not None:
            setattr(namespace, self.dest, value)


def parse_bert_args(args_string=None):
    pparser = argparse.ArgumentParser("Config Parser", add_help=False)
    pparser.add_argument("--config", type=str)
    pargs, remaining_argv = pparser.parse_known_args(args_string)

    parser = argparse.ArgumentParser(
        "PopART BERT", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # TODO: Organise Argument Groups.
    group = parser.add_argument_group("Model Config")
    parser_from_NamedTuple(group, BertConfig, args={
        "micro_batch_size": "Set the micro batch-size",
        "host_embedding": dict(
            choices=["NONE", "WORD", "ALL", "MERGE"],
            help="Enable embedding lookup on CPU. Values: "
            "NONE = use IPU; "
            "WORD = use CPU for word embedding and IPU for position; "
            "ALL = use CPU; "
            "MERGE = use CPU and add word and position embedding together"
        ),
        "sequence_length": "Set the max sequence length",
        "mask_tokens": "Set the max number of masked tokens in a sequence (PRETRAINING only)",
        "vocab_length": "Set the size of the vocabulary",
        "hidden_size": "Set the size of the hidden state of the transformer layers",
        "ff_size__": dict(
            string="--ff-size",
            help="Set the size of the intermediate state of the feed forward layers, by default 4x the hidden_size"
        ),
        "attention_heads": "Set the number of heads in self attention",
        "inference": "Create a model for inference. Otherwise a trainable model is created and trained.",
        "num_layers": "Set the number of transformer layers",
        "layers_per_ipu": "Set the number of layers on each IPU. "
                          "If specified as more than one value, the sum of the values must equal num_layers.",
        "available_memory_proportion": "This option specifies the proportion of total tile memory the Encoder MatMul's can use. "
                                       "Can be specified as either a single value for all IPUs or a value for each Encoder IPU.",
        "recompute_checkpoint_every_layer": "This controls how recomputation is handled in pipelining. "
                                            "If True the output of each layer will be stashed keeping the max liveness "
                                            "of activations to be at most one layer. "
                                            "However, the stash size scales with the number of pipeline stages so this may not always be beneficial. "
                                            "The added stash + code could be greater than the reduction in temporary memory.",
        "no_dropout": "Don't use dropout",
        "no_attn_dropout": "Don't use dropout on attention scores",
        "dropout_prob": "Set the dropout probability",
        "layer_norm_eps": "Set the layer norm epsilon value",
        "popart_dtype": dict(
            string="--dtype",
            choices=["FLOAT16", "FLOAT"],
            help="Set the data type used"
        ),
        "task": dict(
            choices=["PRETRAINING", "SQUAD", "MRPC"],
            help="Set the task. Pretraining (Masked-LM & Next Sentence Prediction), SQuAD, MRPC"
        ),
        "split_linear_layers": "Memory Optimisation to serialise MatMul Operations.",
        "no_mask": "Don't apply padding masks to the attention scores",
        "use_default_available_memory_proportion": "Use the poplibs default value for availableMemoryProportion option on the encoder matmuls.",
        "update_embedding_dict": "Include the sparse update to the word Embedding_Dict.",
        "no_cls_layer": "Don't include the CLS layer in pretraining. This layer comes after the encoders but before the projection for the MLM loss.",
        "projection_bias": "Include bias to the projection layer.",
        "embedding_serialization_vocab_steps": "Factor by which embedding layer is serialized along the vocab dimension.",
        "use_packed_sequence_format": "Set to true when using a packed-sequence dataset (pretraining only)",
        "max_sequences_per_pack": "When using the packed_sequence_format, this specifies the maximum number of sequences per pack to expect"
    })

    group = parser.add_argument_group("SQuAD Config")
    group.add_argument("--vocab-file", type=str,
                       help="Path to the vocab file")
    group.add_argument("--do-lower-case", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Use this when using a uncased vocab")
    group.add_argument("--squad-results-dir", type=str, default="squad_results",
                       help="Path to directory to write results (Note: will be created if path does not exist)")
    group.add_argument("--squad-evaluate-script", type=str,
                       help="Path to SQuAD evaulate-v1.1.py script")
    group.add_argument("--squad-lr-scale", type=float, default=1.0,
                       help="Scale the learning rate of the SQuAD layers.")
    group.add_argument("--disable-attention-dropout-bwd", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Enable DisableAttnDropoutBwdPattern pattern")

    group = parser.add_argument_group("Batch Config")
    group.add_argument("--global-batch-size", type=int,
                       help="Set the global batch size of the model. "
                            "If specified Gradient Accumulation will be set such that 'batch-size * replication-factor * gradient-accumulation-factor == global-batch-size'")
    group.add_argument("--gradient-accumulation-factor", type=int,
                       help="Set how many gradients to accumulate before updating the weights.")
    group.add_argument("--replication-factor", type=int, default=1,
                       help="Replicates the graph by this factor across IPUs to achieve data parallel execution.")

    group = parser.add_argument_group("Optimizer Config")
    group.add_argument("--optimizer", type=str, choices=['SGD', 'ADAM', 'ADAM_NO_BIAS', 'LAMB', 'LAMB_NO_BIAS'], default="SGD",
                       help="Set the optimizer type")
    group.add_argument("--learning-rate", type=float, default=0.0008,
                       help="Set a constant learning rate")
    group.add_argument("--loss-scaling", type=float, default=4.0,
                       help="Set the loss scaling. This helps prevent underflow during backpropagation.")
    group.add_argument("--weight-decay", type=float, default=0,
                       help="Set the weight decay, not used for bias and norms parameters")
    # SGD+M
    group.add_argument("--momentum", type=float, default=0.984375,
                       help="Set the optimizer momentum value")
    group.add_argument("--dampening", type=float,
                       help="Set the optimizer dampening value. (Note: this will be set to momentum value by default)")
    group.add_argument("--velocity-scaling", type=float, default=1.0,
                       help="Set the velocity scaling. This helps prevent overflow when accumulating gradients.")
    # Adam/Lamb
    group.add_argument("--beta1", type=float, default=0.9,
                       help="Set the Adam/Lamb beta1 value")
    group.add_argument("--beta2", type=float, default=0.999,
                       help="Set the Adam/Lamb beta2 value")
    group.add_argument("--max-weight-norm", type=float, default=None,
                       help="Set the max value for R1 (weight norm) in the Lamb optimizer. Default is no clipping")
    group.add_argument("--use-half-optimizer-state", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Set accl1 datatype in Adam based optimizers to FLOAT16")
    group.add_argument("--epochs-inference", type=int, default=1,
                       help="Number of epochs to run inference for")
    group.add_argument("--stochastic-rounding", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Turn on Stochastic Rounding")
    group.add_argument("--gradient-reduction-type", type=str, choices=["Sum", "Mean", "RunningMean"], default="Sum",
                       help="Set how gradients are reduced over the global batch size.")
    group.add_argument("--learning-rate-function", choices=['Scheduled', 'Linear'], default='Scheduled',
                       help="Specify the Learning Rate Scheduler. "
                            "Scheduled will follow the arguments specified by 'lr-schedule-by-*'. "
                            "Linear will follow the arguments specified by {'learning-rate','lr-warmup-steps','lr-steps-per-decay-update'}")

    group = group.add_mutually_exclusive_group()
    group.add_argument("--epochs", type=int,
                       help="Number of epochs to train for")
    group.add_argument("--training-steps", type=int,
                       help="Number of steps to train for")

    group = parser.add_argument_group("Continuous Pipelining Config")
    group.add_argument("--continuous-pipeline-optimizer-scaling", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Turn on hyperparameter scaling by pipeline stage. See 'bert_optimizer/continuous_pipeline.py' for more details")

    group = parser.add_argument_group("Linear Optimizer Config")
    group.add_argument("--lr-warmup-start", type=int, default=1e-7,
                       help="Specify the initial learning rate when using warmup.")
    group.add_argument("--lr-warmup-steps", type=int, default=0,
                       help="Specify the number of steps to Warmup over. Will update the optimizer every step and start from lr = 0.")
    group.add_argument("--lr-steps-per-warmup-update", type=int, default=1,
                       help="Specify the number of steps between each optimizer update during warmup.")
    group.add_argument("--lr-steps-per-decay-update", type=int, default=1,
                       help="Specify the number of steps between each optimizer update after warmup.")

    _group = parser.add_argument_group("Scheduled Optimizer Config")
    group = _group.add_mutually_exclusive_group()
    group.add_argument("--lr-schedule-by-epoch", action=ScheduleArgumentParser, nargs="*", default=None,
                       help="A schedule for learning rate warmup and decay, provided as space-separated "
                            "<int>:<float> pairs. The first item is the epoch at which to update and the second is "
                            "the learning rate at that epoch. \n"
                            "E.g.: --lr-schedule-by-epoch 0:0.00001 1:0.0001 3:0.0008 5:0.00004 10:0.00002")
    group.add_argument("--lr-schedule-by-step", action=ScheduleArgumentParser, nargs="*", default=None,
                       help="A schedule for learning rate warmup and decay, provided as space-separated "
                            "<int>:<float> pairs. The first item is the step at which to update and the second is "
                            "the learning rate at that step. \n"
                            "E.g.: --lr-schedule-by-step 0:0.00001 2500:0.0001 10000:0.0008 50000:0.00004 100000:0.00002")

    group = _group.add_mutually_exclusive_group()
    group.add_argument("--ls-schedule-by-epoch", action=ScheduleArgumentParser, nargs="*", default=None,
                       help="A schedule for loss scaling, provided as space-separated <int>:<float> pairs. "
                            "The first item is the spoch at which to update and the second is "
                            "the loss scaling at that epoch. \n"
                            "E.g.: --ls-schedule-by-step 0:0.00001 2500:0.0001 10000:0.0008 50000:0.00004 100000:0.00002")
    group.add_argument("--ls-schedule-by-step", action=ScheduleArgumentParser, nargs="*", default=None,
                       help="A schedule for loss scaling, provided as space-separated <int>:<float> pairs. "
                            "The first item is the step at which to update and the second is "
                            "the loss scaling at that step. \n"
                            "E.g.: --ls-schedule-by-step 0:0.00001 2500:0.0001 10000:0.0008 50000:0.00004 100000:0.00002")

    group = parser.add_argument_group("Initialisation Config", "Flags for initialising the weights")
    group = group.add_mutually_exclusive_group()
    group.add_argument("--tf-checkpoint", type=str,
                       help="Path to Tensorflow Checkpoint to initialise the model.")
    group.add_argument("--onnx-checkpoint", type=str,
                       help="Path to .onnx file created by this application to initialise the model.")
    group.add_argument("--wandb-checkpoint", type=str,
                       help="Weights and Biases model artifact path. Should contain a model.onnx file to load.")

    group = parser.add_argument_group("Data Config")
    group.add_argument("--input-files", type=str, nargs="*", default = [],
                       help="Files to load data from. "
                            "For Pretraining: Binary files created by bert_data/create_pretraining_data.py. "
                            "For SQuAD: Path to train-v1.1.json")
    group.add_argument("--shuffle", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Shuffle Dataset")
    group.add_argument("--overwrite-cache", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Regenerates the SQuAD dataset instead of loading the cache if available")
    group.add_argument("--no-drop-remainder", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Adjust the batches_per_step to perfectly divide the dataset so no data is missed. Only available for SQuAD.")
    group.add_argument("--generated-data", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Generate a random dataset on the host machine. Creates enough data for one step per epoch. "
                            "Increase --epochs for multiple performance measurements.")
    group.add_argument("--synthetic-data", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Generate a synthetic dataset on the IPU device. Creates enough data for one step per epoch. "
                            "Note that using this option will remove all Host I/O from the model. "
                            "Increase --epochs for multiple perfomance measurements.")
    group.add_argument("--synthetic-data-initializer", type=str, choices=['random_normal', 'zeros'], default="random_normal",
                       help="Specify to the synthetic data initializer with either 'random_normal' or 'zeros'. ")
    group.add_argument("--duplication-factor", type=int, default=1,
                       help="Set the number of times the dataset has been duplicated. This reduces the samples per epoch to"
                            " (# of samples in input-files)/duplication-factor")
    group.add_argument("--epochs-to-cache", type=int, default=0,
                       help="Number of epochs of data to load into memory during PRETRAINING. Default is to load input files as needed.")

    group = parser.add_argument_group("Execution Config")
    group.add_argument("--pipeline", type=str_to_bool, nargs="?", const=True, default=None,
                       help="Build and execute the graph with Pipeline annotations.")
    group.add_argument("--batches-per-step", type=int, default=250,
                       help="Set the number of batches (weight updates) to execute before returning to the Host")
    group.add_argument("--floating-point-exceptions", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Turn on floating point exceptions")
    group.add_argument("--seed", type=int, default=1984,
                       help="Set the host and device seed")
    group.add_argument("--profile", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Generate a profile directory to be analysed by popvision.")
    group.add_argument("--profile-dir", type=str,
                       help="Path to profile directory.")
    group.add_argument("--profile-instrument", type=str_to_bool, nargs="?", const=True, default=True,
                       help="Include cycle count instrumentation when profiling.")
    group.add_argument("--no-outlining", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Disable PopART outlining optimisations. This will increase memory for a small throughput improvement.")
    group.add_argument("--engine-cache", type=lambda arg: None if not arg else arg,
                       help="Path to store a cache of the engine compilation.")
    group.add_argument("--variable-weights-inference", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Force the weights to be variables during inference. Required for loading weights from a checkpoint when using a cached engine.")
    group.add_argument("--continue-training-from-epoch", type=int, default=0,
                       help="Training epoch at which to start hyperparameter schedules when loading from a checkpoint")
    group.add_argument("--no-training", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Don't run the training loop")
    group.add_argument("--no-validation", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Don't validate training. This includes validation at the end of training")
    group.add_argument("--validation-config", action=ValidationConfig,
                       help="Path to preset config for validation. If set by the `--config` file, it definied as a dict instead")
    group.add_argument("--low-latency-inference", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Use input/output callbacks to minimise inference latency for tasks that support this mode.")
    group.add_argument("--minimum-latency-inference", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Disable IO pre-fetching to minimize inference latency, at the cost of reduced throughput. Also sets low-latency-inference=true. ")
    group.add_argument("--inference-lm-perplexity", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Calculate the LM perplexity metric as part of inference (e^loss).")
    group.add_argument("--realtime-scheduler", action="store_true",
                       help="Set a realtime scheduler for this process. Only activated during inference. \
                             (IMPORTANT: Requires non-interactive sudo, otherwise has no effect)")
    group.add_argument("--max-copy-merge-size", type=int, default=-1,
                       help="Set the value for Poplar engine option 'opt.maxCopyMergeSize'. Set to -1 to use Poplar's default.")
    group.add_argument("--disable-fully-connected-pass", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Adding fully connected pass to some matmuls causes large transposes before operations during training. "
                       "Note: This will improve throughput at the cost of memory.")
    group.add_argument("--group-host-syncs", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Groups the host-device synchronisations more efficiently, higher throughput can be reached at the expense of sum liveness")
    group.add_argument("--optimizer-state-offchip", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Keep the optimizer state off chip. Only streaming it on when needed for the weight update.")
    group.add_argument("--replicated-tensor-sharding", type=str_to_bool, nargs="?", const=False, default=False,
                       help="Shard the optimizer state across replicas")
    group.add_argument("--enable-half-partials", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Enable half partials for matmuls and convolutions globally.")
    group.add_argument('--internal-exchange-optimisation-target', type=str, default=None,
                       choices=["balanced", "cycles", "memory"],
                       help="The optimisation approach for internal exchanges.")

    group = parser.add_argument_group("Logging Config")
    group.add_argument("--report-hw-cycle-count", action="store_true",
                       help='Report the number of cycles each "session.run" takes.')
    group.add_argument("--log-dir", type=str, default="logs",
                       help="Path to save Tensorboard logs")
    group.add_argument("--steps-per-log", type=int, default=1,
                       help="Number of session.run to execute before logging training metrics")
    group.add_argument("--aggregate-metrics-over-steps", type=int,
                       help="Number of steps to aggregate metrics over. Default is the number of steps per epoch")
    group.add_argument("--epochs-per-save", type=int, default=-1,
                       help="Number of epochs between each save of the model. Also saves at the end of training")
    group.add_argument("--steps-per-save", type=int, default=-1,
                       help="Number of steps between each save of the model. Also saves at the end of training")
    group.add_argument("--no-model-save", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Don't save the model. Useful for testing.")
    group.add_argument("--checkpoint-dir", type=str, default="ckpts",
                       help="Path to directory to save model checkpoints.")
    group.add_argument("--save-initializers-externally", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Save weight and optimizer state initializers external to the model.onnx file.")
    group.add_argument("--wandb", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Enable logging to Weights and Biases.")
    group.add_argument("--wandb-save-checkpoints", type=str,
                       help="Save the checkpoint directory as an artifact under this name in Weights and Biases.")
    group.add_argument("--log-level", type=str, default='INFO',
                       choices=['NOTSET', 'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
                       help="Set the logging level")

    group = parser.add_argument_group("Device Config")
    group.add_argument("--device-id", type=int, default=None,
                       help="Select a specific IPU device.")
    group.add_argument("--device-connection-type", type=str, default="always",
                       choices=["always", "ondemand", "offline"],
                       help="Set the popart.DeviceConnectionType.")
    group.add_argument("--device-version", type=str, default=None,
                       help="Set the IPU version (for offline compilation).")
    group.add_argument("--device-tiles", type=int, default=None,
                       help="Set the number of tiles (for offline compilation).")
    group.add_argument("--device-ondemand-timeout", type=int, default=int(1e4),
                       help="Set the seconds to wait for an ondemand device to become before available before exiting.")
    group.add_argument("--use-ipu-model", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Target the IPU Model.")
    group.add_argument("--compile-only", action="store_true",
                       help="Enable compilation only mode (with device_connection_type=offline), terminating after training graph is compiled.")

    # This is here only for the help message
    group.add_argument("--config", type=str,
                       help="Path to preset config")
    defaults = dict()
    if pargs.config is not None:
        with open(pargs.config, "r") as f:
            preset = json.load(f)
        clean_exclusive_presets(parser, preset, remaining_argv)
        defaults.update(**preset)

    # Check the default args are valid
    known_args = set(vars(parser.parse_args("")))
    unknown_defaults = set(defaults) - known_args

    if unknown_defaults:
        raise ValueError(f"Unknown arg(s) in config file: {unknown_defaults}")

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)

    set_popdist_args(args)

    set_batch_arguments(args)

    args.sdk_version = get_sdk_version()

    # Invalidate incompatible options
    if args.no_drop_remainder and args.task != "SQUAD":
        raise RuntimeError(f"--no-drop-remainder is only compatible with SQUAD and not with {args.task}, aborting")
    if args.host_embedding != "NONE" and args.task != "SQUAD":
        raise RuntimeError(f"--host-embedding is only compatible with SQUAD and not with {args.task}, aborting")
    if args.synthetic_data and args.generated_data:
        raise RuntimeError("choose either --synthetic-data or --generated-data, not both, aborting")
    check_training_duration(args)

    # set low-latency-inference if minimum-latency-inference set
    if args.minimum_latency_inference:
        args.low_latency_inference = True

    # Append datetime string to checkpoints path and create the subdirectory
    args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                       datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.compile_only:
        # Set device to offline (and mk2)
        args.device_connection_type = "offline"
        args.device_version = "ipu2"

        # Enforce that an engine cache path is set
        if not args.engine_cache:
            default_path = "./session_cache/executable.popart"
            logger.warning(
                "Warning: --engine-cache path must be set for compile " +
                "only mode. Defaulting to '{}'.".format(default_path))
            args.engine_cache = default_path

        # Enforce generated data and remove real data path
        if not args.generated_data:
            logger.warning(
                "Warning: --generated-data must be set for compile only " +
                "mode. Defaulting to using generated data. --input-files" +
                " will be ignored for compile only model.")
            args.generated_data = True
            # Setting to default: empty list
            args.input_files = []

    return args


def check_training_duration(args):
    if not args.inference and not args.no_training and \
       args.epochs is None and args.training_steps is None:
        raise RuntimeError("Either epochs or training-steps is required, aborting")


def global_batch_size(args):
    return args.micro_batch_size * args.replication_factor * args.gradient_accumulation_factor * args.popdist_size


def set_batch_arguments(args):
    if args.global_batch_size is None:
        if args.gradient_accumulation_factor is None:
            args.gradient_accumulation_factor = 1
        args.global_batch_size = global_batch_size(args)
    else:
        if args.gradient_accumulation_factor is not None and args.global_batch_size != global_batch_size(args):
            raise RuntimeError("User settings for global batch size and factors are inconsistent.\n\n"
                               f"  Specified global batch size: {args.global_batch_size}\n\n"
                               f"  Inferred global batch size: {global_batch_size(args)}\n"
                               f"    micro batch size: {args.micro_batch_size}\n"
                               f"    gradient accumulation factor: {args.gradient_accumulation_factor}\n"
                               f"    replication factor: {args.replication_factor}")

        denom = args.micro_batch_size * args.replication_factor * args.popdist_size
        if args.global_batch_size % denom != 0:
            raise RuntimeError("Unable to set gradient accumulation to match the global batch size")

        args.gradient_accumulation_factor = args.global_batch_size // denom
        logger.info(f"Set Gradient Accumulation Factor to {args.gradient_accumulation_factor}")


def clean_exclusive_presets(parser, preset, remaining_argv):
    """Modifies the presets dictionary in-place to remove any defaults that would violate
    a mutually exclusive constraint."""
    def mutually_exclusive_action_on_cmd(group):
        for action in group._group_actions:
            if any([opt in remaining_argv for opt in action.option_strings]):
                return action.dest
        return None

    def remove_mutually_exclusive_clashes(group, presets, cmd_arg):
        for action in group._group_actions:
            if action.dest != cmd_arg and action.dest in presets:
                del presets[action.dest]

    for group in parser._mutually_exclusive_groups:
        cmd_arg = mutually_exclusive_action_on_cmd(group)
        if cmd_arg is not None:
            remove_mutually_exclusive_clashes(group, preset, cmd_arg)


def get_validation_args(args):
    validation_kwargs = dict(
        inference=True,
        tf_checkpoint=None,
        gradient_accumulation_factor=1,
        use_popdist=False
    )
    if not args.no_training:
        validation_kwargs["onnx_checkpoint"] = os.path.join(args.checkpoint_dir, "model.onnx")
    if args.validation_config is not None:
        validation_kwargs.update(**args.validation_config)
    if args.engine_cache:
        validation_kwargs["variable_weights_inference"] = True

    args = vars(args)
    args.update(**validation_kwargs)
    return argparse.Namespace(**args)


def set_popdist_args(args):
    if not popdist.isPopdistEnvSet():
        args.use_popdist = False
        args.popdist_size = 1
        args.popdist_rank = 0
        return

    if args.inference:
        raise RuntimeError("Distributed execution is only supported for training")

    try:
        import horovod.popart as hvd
        hvd.init()
    except ImportError:
        raise ImportError("Could not find the PopART horovod extension. "
                          "Please install the horovod .whl provided in the Poplar SDK.")

    args.use_popdist = True
    popdist_local_factor = popdist.getNumLocalReplicas()
    if args.replication_factor > 1 and args.replication_factor != popdist_local_factor:
        logger.warning(f"Overwriting the local replication factor {args.replication_factor} to {popdist_local_factor}")
    args.replication_factor = popdist_local_factor

    args.popdist_size = popdist.getNumTotalReplicas() // popdist.getNumLocalReplicas()
    args.popdist_rank = popdist.getReplicaIndexOffset() // popdist.getNumLocalReplicas()
    args.checkpoint_dir = args.checkpoint_dir + "_rank_" + str(args.popdist_rank)

    from mpi4py import MPI
    setup_comm(MPI.COMM_WORLD)


def get_sdk_version():
    sdk_path = os.environ.get("POPLAR_SDK_ENABLED", None)
    if sdk_path:
        return os.path.split(os.path.split(sdk_path)[0])[1]
    return "Unknown"
