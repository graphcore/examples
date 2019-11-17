# Copyright 2019 Graphcore Ltd.
import os
import sys
import json
import argparse
import math
from typing import List, Any, Optional
import numpy as np
import onnx
from logging import getLogger
from onnx import TensorProto, numpy_helper

from bert_model import BertConfig

logger = getLogger(__name__)


def load_initializers_from_onnx(model_path):
    initializers = {}
    model = onnx.load(model_path)
    for weight in model.graph.initializer:
        if weight.data_type == TensorProto.FLOAT16:
            int_data = np.asarray(weight.int32_data, np.int32)
            np_weight = int_data.view(dtype=np.float16).reshape(weight.dims)
        else:
            np_weight = numpy_helper.to_array(weight)
        initializers[weight.name] = np_weight
    return initializers


def save_model_statistics(model_path, writer, i=0):
    initializers = load_initializers_from_onnx(model_path)
    for name, np_weight in initializers.items():
        name = name.replace(":", "_")
        writer.add_histogram(name, np_weight, i)
        writer.add_scalar(f"L2/{name}", np.linalg.norm(np_weight), i)


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
            del kwargs["type"]
            del kwargs["default"]
            kwargs["action"] = "store_false" if ntuple._field_defaults[key] else "store_true"
        else:
            for _t in (str, int):
                if t == List[_t]:
                    kwargs["type"] = _t
                    kwargs["nargs"] = '*'
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
        self._nargs = nargs
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
                logger.warn("Invalid Learning Rate Schedule provided. "
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
        "layers_per_ipu": "Set the number of layers on each IPU",
        "no_dropout": "Don't use dropout",
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
        "positional_embedding_init_fn": dict(
            choices=["DEFAULT", "TRANSFORMER", "SIMPLIFIED"],
            help="Set the function used to initialise the positional embeddings"
        ),
        "custom_ops": dict(
            choices=["gather", "attention", "feed_forward"],
            help="Use Custom Operators"
        ),
        "split_linear_layers": "Memory Optimisation to serialise MatMul Operations. Required for large sequence_length",
        "squeeze_model": "Try to use fewer IPUs by placing the input embedding and loss onto the \
                            same IPUs as the first and last tranformer layers respectively",
        "no_mask": "Don't apply padding masks to the attention scores",
        "projection_serialization_steps": "Split the final MLM projection into this many steps"
    })
    group.add_argument("--use-ipu-model", action="store_true",
                       help="Target the IpuModel (acquires a real IPU device by default). \
                             WARNING: The custom ops do not have validated cycle estimates \
                             so do not rely on the model's cycle report.")

    group = parser.add_argument_group("SQuAD Config")
    group.add_argument("--vocab-file", type=str,
                       help="Path to the vocab file")
    group.add_argument("--do-lower-case", action="store_true",
                       help="Use this when using a uncased vocab")
    group.add_argument("--squad-results-dir", type=str, default="squad_results",
                       help="Path to directory to write results (Note: will be created if path does not exist)")
    group.add_argument("--squad-evaluate-script", type=str,
                       help="Path to SQuAD evaulate-v1.1.py script")

    group = parser.add_argument_group("Training Config")
    group.add_argument("--gradient-accumulation-factor", type=int, default=1,
                       help="Set how many gradients to accumulate before updating the weights. (Note: This changes the effective batch size)")
    group.add_argument("--replication-factor", type=int, default=1,
                       help="Replicates the graph by this factor across IPUs to achieve data parallel execution. (Note: This changes the effective batch size)")
    group.add_argument("--learning-rate", type=float, default=0.0008,
                       help="Set the learning rate")
    group.add_argument("--momentum", type=float, default=0.984375,
                       help="Set the optimizer momentum value")
    group.add_argument("--dampening", type=float,
                       help="Set the optimizer dampening value. (Note: this will be set to momentum value by default)")
    group.add_argument("--velocity-scaling", type=float, default=1.0,
                       help="Set the velocity scaling. This helps prevent overflow when accumulating gradients.")
    group.add_argument("--loss-scaling", type=float, default=4.0,
                       help="Set the loss scaling. This helps prevent underflow during backpropagation.")
    group.add_argument("--epochs", type=int, default=35,
                       help="Number of epochs to train for")
    group.add_argument("--stochastic-rounding", action="store_true",
                       help="Turn on Stochastic Rounding")

    group = parser.add_argument_group("Continuous Pipelining Config")
    group.add_argument("--pipeline-lr-scaling", action="store_true",
                       help="Enable learning rate scaling per pipeline stage")
    group.add_argument("--pipeline-lr-scaling-offset", type=float, default=0.25,
                       help="Set the value for learning rate scaling on the first pipeline stage. Learning rates will be scaled "
                            "linearly from this offset (default: 0.25) to 1 as pipeline stage increases to account for increased errors "
                            "at lower-level stages when pipelining. (Note: for pipelines with few stages, this should be increased)")
    group.add_argument("--pipeline-momentum-scaling", action="store_true",
                       help="Enable momentum and dampening scaling per pipeline stage")
    group.add_argument("--pipeline-momentum-scaling-offset", type=float, default=0.1,
                       help="Set the value momentum scaling on the last pipeline stage. Momentums will be scaled "
                            "linearly from this offset (default: 0.1) to 1 as pipeline stage decrease to account for increased errors "
                            "at lower-level stages when pipelining. (Note: for pipelines with few stages, this should be increased)")
    group.add_argument("--pipeline-dampening-scaling-offset", type=float,
                       help="Set the value for dampening scaling on the last pipeline stage. Dampenings will be scaled "
                            "linearly from this offset (default: same as momentum) to 1 as pipeline stage decrease to account for increased errors "
                            "at lower-level stages when pipelining. (Note: this will be set to the momentum offset by default)")

    group = parser.add_mutually_exclusive_group()
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

    group = parser.add_mutually_exclusive_group()
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
    group.add_argument("--tf-checkpoint", type=str,
                       help="Path to Tensorflow Checkpoint to initialise the model.")
    group.add_argument("--onnx-checkpoint", type=str,
                       help="Path to .onnx file created by this application to initialise the model.")

    group = parser.add_argument_group("Data Config")
    group.add_argument("--input-files", type=str, nargs="*",
                       help="Files to load data from. "
                            "For Pretraining: Binary files created by bert_data/create_pretraining_data.py. "
                            "For SQuAD: Path to train-v1.1.json")
    group.add_argument("--shuffle", action="store_true",
                       help="Shuffle Dataset")
    group.add_argument("--overwrite-cache", action="store_true",
                       help="Regenerates the SQuAD dataset instead of loading the cache if available")
    group.add_argument("--no-drop-remainder", action="store_true",
                       help="Adjust the batches_per_step to perfectly divide the dataset so no data is missed. Only available for SQuAD.")
    group.add_argument("--synthetic-data", action="store_true",
                       help="Generate a synthetic dataset. Creates enough data for one step per epoch. "
                            "Increase --epochs for multiple perfomance measurements.")
    group.add_argument("--duplication-factor", type=int, default=1,
                       help="Set the number of times the dataset has been duplicated. This reduces the samples per epoch to"
                            " (# of samples in input-files)/duplication-factor")
    group.add_argument("--epochs-to-cache", type=int, default=0,
                       help="Number of epochs of data to load into memory during PRETRAINING. Default is to load input files as needed.")

    group = parser.add_argument_group("Execution Config")
    group.add_argument("--batches-per-step", type=int, default=250,
                       help="Set the number of batches (weight updates) to execute before returning to the Host")
    group.add_argument("--floating-point-exceptions", action="store_true",
                       help="Turn on floating point exceptions")
    group.add_argument("--seed", type=int, default=1984,
                       help="Set the host and device seed")
    group.add_argument("--graph-report", type=str,
                       help="Path to save a poplar Graph Report")
    group.add_argument("--execution-report", type=str,
                       help="Path to save a poplar Execution Report. NOTE: this will run the graph with instrumentation and for only one 'batch'")
    group.add_argument("--gc-profile", action="store_true",
                       help="Run the model and save the reports with gcprofile.save_popart_reports")
    group.add_argument("--no-outlining", action="store_true",
                       help="Disable PopART outlining optimisations. This will increase memory for a small throughput improvement.")
    group.add_argument("--engine-cache", type=str,
                       help="Path to store a cache of the engine compilation.")
    group.add_argument("--log-dir", type=str, default="logs",
                       help="Path to save Tensorboard logs")
    group.add_argument("--steps-per-log", type=int, default=1,
                       help="Number of session.run to execute before logging training metrics")
    group.add_argument("--aggregate-metrics-over-steps", type=int,
                       help="Number of steps to aggregate metrics over. Default is the number of steps per epoch")
    group.add_argument("--pipeline", action="store_true",
                       help="Pipeline execution.")
    group.add_argument("--epochs-per-save", type=int, default=-1,
                       help="Number of epochs between each save of the model. Also saves at the end of training")
    group.add_argument("--steps-per-save", type=int, default=-1,
                       help="Number of steps between each save of the model. Also saves at the end of training")
    group.add_argument("--checkpoint-dir", type=str, default="ckpts",
                       help="Path to directory to save model checkpoints.")
    group.add_argument("--no-training", action="store_true",
                       help="Don't run the training loop")
    group.add_argument("--no-validation", action="store_true",
                       help="Don't validate training. This includes validation at the end of training")
    group.add_argument("--no-model-save", action="store_true",
                       help="Don't save the model. Useful for testing.")
    group.add_argument("--validation-config", action=ValidationConfig,
                       help="Path to preset config for validation. If set by the `--config` file, it definied as a dict instead")
    group.add_argument("--low-latency-inference", action="store_true",
                       help="Use input/output callbacks to minimise inference latency for tasks that support this mode.")
    group.add_argument("--max-copy-merge-size", type=int, default=-1,
                       help="Set the value for Poplar engine option 'opt.maxCopyMergeSize'. Set to -1 to use Poplar's default.")
    group.add_argument("--disable-fully-connected-pass", action="store_true",
                       help="Adding fully connected pass to some matmuls causes large transposes before operations during training. "
                       "Note: This will improve throughput at the cost of memory.")
    group.add_argument("--log-level", type=str, default='INFO',
                       choices=['NOTSET', 'INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
                       help="Set the logging level")
    # This is here only for the help message
    group.add_argument("--config", type=str,
                       help="Path to preset config")

    if pargs.config is not None:
        with open(pargs.config, "r") as f:
            preset = json.load(f)
        parser.set_defaults(**preset)

    args = parser.parse_args(remaining_argv)
    validate_args(args)
    save_args(args)
    return args


def save_args(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)


def get_validation_args(args):
    validation_kwargs = dict(
        inference=True,
        tf_checkpoint=None,
        onnx_checkpoint=os.path.join(args.checkpoint_dir, "model.onnx"),
    )
    if args.engine_cache:
        validation_kwargs["engine_cache"] = args.engine_cache + "val"
    if args.validation_config is not None:
        validation_kwargs.update(**args.validation_config)

    args = vars(args)
    args.update(**validation_kwargs)
    return argparse.Namespace(**args)


def validate_args(args):
    if args.tf_checkpoint and args.onnx_checkpoint:
        raise RuntimeError("--tf-checkpoint and --onnx-checkpoint cannot both be set")
