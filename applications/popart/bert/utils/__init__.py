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
from typing import List, Optional
import numpy as np
import onnx
from logging import getLogger
from bert_model import BertConfig

from .weight_loading import load_initializers_from_onnx

logger = getLogger(__name__)


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
        "batch_size": "Set the micro batch-size",
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
        "replicated_weight_sharding": "Shard weight tensors over the replicas, only supported in phased_execution mode.",
        "no_mask": "Don't apply padding masks to the attention scores",
        "use_default_available_memory_proportion": "Use the poplibs default value for availableMemoryProportion option on the encoder matmuls.",
        "update_embedding_dict": "Include the sparse update to the word Embedding_Dict.",
        "no_cls_layer": "Don't include the CLS layer in pretraining. This layer comes after the encoders but before the projection for the MLM loss.",
        "projection_bias": "Include bias to the projection layer.",
        "embedding_serialization_vocab_steps": "Factor by which embedding layer is serialized, only supported in phased_execution mode.",
        "num_attention_splits": "Factor by which attention layer is serialized, only supported in phased_execution mode.",
        "num_ffwd_splits": "Factor by which feedforward layer is serialized, only supported in phased_execution mode.",
        "split_transformer": "Place attention and feedforward layers in separate phased_execution scope.",
        "batch_serialize": "Factor by which a micro-batch is serialized, only supported in phased_execution mode.",
        "num_io_tiles": "The number of tiles on each IPU dedicated to doing collective communication ops with the replicated weights. Only supported in phased_execution mode."
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

    group = parser.add_argument_group("Training Config")
    group.add_argument("--gradient-accumulation-factor", type=int, default=1,
                       help="Set how many gradients to accumulate before updating the weights. \
                            Note: This affects the calculation of effective batch size")
    group.add_argument("--replication-factor", type=int, default=1,
                       help="Replicates the graph by this factor across IPUs to achieve data parallel execution. (Note: This changes the effective batch size)")
    group.add_argument("--optimizer", type=str, choices=['SGD', 'ADAM', 'ADAM_NO_BIAS', 'LAMB', 'LAMB_NO_BIAS'], default="SGD",
                       help="Set the optimizer type")
    group.add_argument("--learning-rate", type=float, default=0.0008,
                       help="Set the learning rate")
    # SGD+M
    group.add_argument("--momentum", type=float, default=0.984375,
                       help="Set the optimizer momentum value")
    group.add_argument("--dampening", type=float,
                       help="Set the optimizer dampening value. (Note: this will be set to momentum value by default)")
    group.add_argument("--velocity-scaling", type=float, default=1.0,
                       help="Set the velocity scaling. This helps prevent overflow when accumulating gradients.")
    # Adam
    group.add_argument("--beta1", type=float, default=0.9,
                       help="Set the Adam/Lamb beta1 value")
    group.add_argument("--beta2", type=float, default=0.999,
                       help="Set the Adam/Lamb beta2 value")
    group.add_argument("--max-weight-norm", type=float, default=None,
                       help="Set the max value for R1 (weight norm) in the Lamb optimizer. Default is no clipping")
    group.add_argument("--loss-scaling", type=float, default=4.0,
                       help="Set the loss scaling. This helps prevent underflow during backpropagation.")
    group.add_argument("--weight-decay", type=float, default=0, help="Set the weight decay, not used for bias and norms parameters")
    group.add_argument("--epochs", type=int, default=35,
                       help="Number of epochs to train for")
    group.add_argument("--epochs-inference", type=int, default=1,
                       help="Number of epochs to run inference for")
    group.add_argument("--stochastic-rounding", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Turn on Stochastic Rounding")
    group.add_argument("--gradient-reduction-type", type=str, choices=["Sum", "Mean"], default="Sum",
                       help="Set how gradients are reduced over the global batch size.")

    group = parser.add_argument_group("Continuous Pipelining Config")
    group.add_argument("--pipeline-lr-scaling", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Enable learning rate scaling per pipeline stage")
    group.add_argument("--pipeline-lr-scaling-offset", type=float, default=0.25,
                       help="Set the value for learning rate scaling on the first pipeline stage. Learning rates will be scaled "
                            "linearly from this offset (default: 0.25) to 1 as pipeline stage increases to account for increased errors "
                            "at lower-level stages when pipelining. (Note: for pipelines with few stages, this should be increased)")
    group.add_argument("--pipeline-momentum-scaling", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Enable momentum and dampening scaling per pipeline stage")
    group.add_argument("--pipeline-momentum-scaling-offset", type=float, default=0.1,
                       help="Set the value momentum scaling on the last pipeline stage. Momentums will be scaled "
                            "linearly from this offset (default: 0.1) to 1 as pipeline stage decrease to account for increased errors "
                            "at lower-level stages when pipelining. (Note: for pipelines with few stages, this should be increased)")
    group.add_argument("--pipeline-dampening-scaling-offset", type=float,
                       help="Set the value for dampening scaling on the last pipeline stage. Dampenings will be scaled "
                            "linearly from this offset (default: same as momentum) to 1 as pipeline stage decrease to account for increased errors "
                            "at lower-level stages when pipelining. (Note: this will be set to the momentum offset by default)")

    _group = parser.add_argument_group("Hyperparameter Schedule Config")
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

    group = parser.add_argument_group("Data Config")
    group.add_argument("--input-files", type=str, nargs="*",
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

    group = parser.add_argument_group("Execution Mode")
    emode = group.add_mutually_exclusive_group()
    emode.add_argument("--virtual-graph", type=str_to_bool, nargs="?", const=True, default=None,
                       help="Build and execute the graph with only VirtualGraph annotations.")
    emode.add_argument("--pipeline", type=str_to_bool, nargs="?", const=True, default=None,
                       help="Build and execute the graph with Pipeline annotations.")
    emode.add_argument("--phased-execution", type=str_to_bool, nargs="?", const=True, default=None,
                       help="Build and execute the graph with ExecutionPhase annotations.")
    phased_emode = group.add_mutually_exclusive_group()
    phased_emode.add_argument("--single-device", action='store_const', const='SINGLE', dest='phased_execution_type',
                              help="Execute on a single device per replica. Only used in phased_execution mode.")
    phased_emode.add_argument("--dual-device", action='store_const', const='DUAL', dest='phased_execution_type',
                              help="Execute using two devices per replica. Only used in phased_execution mode.")
    parser.set_defaults(phased_execution_type='SINGLE')


    group = parser.add_argument_group("Execution Config")
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
    group.add_argument("--enable-half-partials", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Enable half partials for matmuls and convolutions globally.")
    group.add_argument('--internal-exchange-optimisation-target', type=str, default=None,
                       choices=["balanced", "cycles", "memory"],
                       help="The optimisation approach for internal exchanges.")
    group.add_argument("--activations-on-chip", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Leave activations stashed on chip, i.e don't move the activations to streaming memory.")

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
    group.add_argument("--wandb", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Enable logging to Weights and Biases.")
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

    group = parser.add_argument_group("Distribution Config")
    group.add_argument("--mpi-distributed", type=str_to_bool, nargs="?", const=True, default=False,
                       help="Enable distributed training with MPI backend. Distributed training with MPI is currently in preview."
                       "Full support for distributed training will be coming in a future release.")
    group.add_argument("--mpi-rank", type=int, default=0, help="Input the MPI rank of this process. This value will be overwritten by the rank determined by the MPI controller")
    group.add_argument("--mpi-size", type=int, default=1, help="Input the MPI size. This value will be overwritten by the size determined by the MPI controller")

    # This is here only for the help message
    group.add_argument("--config", type=str,
                       help="Path to preset config")
    defaults = dict(execution_mode="DEFAULT")
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

    set_execution_mode(args)

    set_mpi_args(args)

    # Invalidate incompatible options
    if args.no_drop_remainder and args.task != "SQUAD":
        raise RuntimeError(f"--no-drop-remainder is only compatible with SQUAD and not with {args.task}, aborting")
    if args.batch_serialize > args.batch_size:
        raise RuntimeError("--batch-serialize cannot be > --batch_size, aborting")
    if args.split_linear_layers and args.batch_serialize > 1:
        raise RuntimeError("--split-linear-layers is incompatible with --batch-serialize > 1, aborting")
    if args.synthetic_data and args.generated_data:
        raise RuntimeError("choose either --synthetic-data or --generated-data, not both. Aborting")

    # set low-latency-inference if minimum-latency-inference set
    if args.minimum_latency_inference:
        args.low_latency_inference = True

    # Append datetime string to checkpoints path and create the subdirectory
    args.checkpoint_dir = os.path.join(args.checkpoint_dir,
                                       datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_args(args)
    return args


def set_execution_mode(args):
    if args.pipeline:
        args.execution_mode = "PIPELINE"
    elif args.phased_execution:
        args.execution_mode = "PHASED"
    elif args.virtual_graph or \
        args.execution_mode == "PIPELINE" and args.pipeline is False or \
            args.execution_mode == "PHASED" and args.phased_execution is False:
        args.execution_mode = "DEFAULT"
    return args


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


def save_args(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)


def get_validation_args(args):
    validation_kwargs = dict(
        inference=True,
        tf_checkpoint=None,
        gradient_accumulation_factor=1
    )
    if not args.no_training:
        validation_kwargs["onnx_checkpoint"] = os.path.join(args.checkpoint_dir, "model.onnx")
    if args.engine_cache:
        validation_kwargs["engine_cache"] = args.engine_cache + "val"
    if args.validation_config is not None:
        validation_kwargs.update(**args.validation_config)

    args = vars(args)
    args.update(**validation_kwargs)
    return argparse.Namespace(**args)


def set_mpi_args(args):
    if not args.mpi_distributed:
        return

    logger.warning("Distributed training with MPI is currently in preview."
                   "Full support for distributed training will be coming in a future release.")
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()
    if args.mpi_rank != mpi_rank and args.mpi_rank != 0:
        logger.warning(f"Overwriting the MPI rank provided {args.mpi_rank} to {mpi_rank}")
    args.mpi_rank = mpi_rank

    if args.mpi_size != mpi_size and args.mpi_size > 1:
        logger.warning(f"Overwriting the MPI size provided {args.mpi_size} to {mpi_size}")
    args.mpi_size = mpi_size

    is_distributed = mpi_size > 1
    if is_distributed:
        if args.inference:
            raise RuntimeError("Distributed execution only supported for training")
        if args.task != "SQUAD":
            raise RuntimeError("Distributed training only supported with SQUAD")

    if is_distributed:
        args.checkpoint_dir = args.checkpoint_dir + "_rank_" + str(args.mpi_rank)
