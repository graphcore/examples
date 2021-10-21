# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
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

import argparse
import sys

import yaml

config_file = "./configs.yml"


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def parse_args(args=None):
    pparser = argparse.ArgumentParser("ViT Configuration name", add_help=False)
    pparser.add_argument("--config",
                         type=str,
                         help="Configuration Name",
                         default='vit_base_16')
    pargs, remaining_args = pparser.parse_known_args(args=args)
    config_name = pargs.config

    parser = argparse.ArgumentParser(
        "Poptorch ViT",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Execution
    parser.add_argument("--batch-size", type=int, help="Set the micro batch-size")
    parser.add_argument("--training-steps", type=int, help="Number of training steps")
    parser.add_argument("--batches-per-step", type=int, help="Number of batches per training step")
    parser.add_argument("--replication-factor", type=int, help="Number of replicas")
    parser.add_argument("--gradient-accumulation", type=int,
                        help="Number of gradients to accumulate before updating the weights")
    parser.add_argument("--stochastic-rounding", type=str_to_bool,
                        nargs="?", const=True, default=True,
                        help="enable stochastic rounding")
    parser.add_argument("--recompute-checkpoint-every-layer", type=str_to_bool,
                        nargs="?", const=True, default=False,
                        help="This controls how recomputation is handled in pipelining. "
                        "If True the output of each encoder layer will be stashed keeping "
                        "the max liveness of activations to be at most one layer. "
                        "However, the stash size scales with the number of pipeline stages "
                        "so this may not always be beneficial. "
                        "The added stash + code could be greater than "
                        "the reduction in temporary memory.")
    parser.add_argument("--ipus-per-replica", type=int,
                        help="Number of IPUs required by each replica")
    parser.add_argument("--matmul-proportion", type=float, nargs="+",
                        help="Relative IPU memory proportion size allocated for matmul")
    parser.add_argument("--random-seed", type=int, help="Seed for RNG")
    parser.add_argument('--precision', choices=['16.16', '16.32', '32.32'], default='16.16',
                        help="Precision of Ops(weights/activations/gradients) and "
                        "Master data types: 16.16, 16.32, 32.32")
    parser.add_argument('--normalization-location', choices=['host', 'ipu', 'none'], default='host',
                        help='Location of the data normalization')
    parser.add_argument("--layers-per-ipu", type=int, nargs="+",
                        help="number of layers placed on each IPU")
    parser.add_argument("--prefetch-depth", type=int, help="Prefetch buffering depth")
    parser.add_argument("--enable-rts", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling RTS")

    # Optimizer
    parser.add_argument("--optimizer", type=str, choices=['SGD', ],
                        help="optimizer to use for the training")
    parser.add_argument("--learning-rate", type=float,
                        help="Learning rate value for constant schedule, "
                        "maximum for linear schedule.")
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "linear"],
                        help="Type of learning rate schedule. "
                        "--learning-rate will be used as the max value")
    parser.add_argument("--loss-scaling", type=float,
                        help="Loss scaling factor (recommend using powers of 2)")
    parser.add_argument("--weight-decay", type=float, help="Set the weight decay")
    parser.add_argument("--momentum", type=float, help="The momentum factor of SGD optimizer")
    parser.add_argument("--warmup-steps", type=int, help="Number of warmup steps")


    # Model
    parser.add_argument("--vocab-size", type=int, help="Set the size of the vocabulary")
    parser.add_argument("--hidden-size", type=int,
                        help="The size of the hidden state of the transformer layers")
    parser.add_argument("--num-hidden-layers", type=int, help="The number of transformer layers")
    parser.add_argument("--num-attention-heads", type=int,
                        help="Set the number of heads in self attention")
    parser.add_argument("--mlp-dim", type=int, help="The size of mlp dimention")
    parser.add_argument("--hidden-dropout-prob", type=float, help="MLP dropout probability")
    parser.add_argument("--patches-size", type=float, nargs="+", help="The size of image tokens")
    parser.add_argument("--num-labels", type=int, help="The number of classes")
    parser.add_argument("--attention-probs-dropout-prob", type=float, help="Attention dropout probability")
    parser.add_argument("--layer-norm-eps", type=float, help="LayerNorm epsilon")

    # Dataset
    parser.add_argument('--dataset', choices=['cifar10', 'imagenet', 'synthetic', 'generated'],
                        default='cifar10', help="Choose data")
    parser.add_argument("--input-files", type=str, nargs="+", help="Input data files")
    parser.add_argument("--synthetic-data", type=str_to_bool, nargs="?", const=True, default=False,
                        help="No Host/IPU I/O, random data created on device")

    # Misc
    parser.add_argument("--dataloader-workers", type=int, help="The number of dataloader workers")
    parser.add_argument("--custom-ops", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Enable custom ops")
    parser.add_argument("--wandb", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling logging to Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="torch-vit",
                        help="wandb project name")
    parser.add_argument("--executable-cache-dir", type=str, default="",
                        help="Directory where Poplar executables are cached. If set, recompilation of identical graphs can be avoided. "
                        "Required for both saving and loading executables.")
    parser.add_argument("--profile-dir", type=str, help="Directory for profiling results")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="",
                        help="Directory where checkpoints will be saved and restored from."
                        "This can be either an absolute or relative path. If this is "
                        "not specified, only end of run checkpoint is saved in an automatically "
                        "generated directory at the root of this project. Specifying directory is"
                        "recommended to keep track of checkpoints.")
    parser.add_argument("--checkpoint-save-steps", type=int, default=100,
                        help="Option to checkpoint model after n steps.")
    parser.add_argument("--restore-epochs-and-optimizer", type=str_to_bool,
                        nargs="?", const=True, default=False,
                        help="Restore epoch and optimizer state to continue training. "
                        "This should normally be True when resuming a "
                        "previously stopped run, otherwise False.")
    parser.add_argument("--checkpoint-file", type=str, default="",
                        help="Checkpoint to be retrieved for further training. This can"
                        "be either an absolute or relative path to the checkpoint file.")
    parser.add_argument("--restore", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Restore a checkpoint model to continue training.")

    # This is here only for the help message
    parser.add_argument("--config", type=str, help="Configuration name")

    # Load the yaml
    yaml_args = dict()
    if config_name is not None:
        with open(config_file, "r") as f:
            try:
                yaml_args.update(**yaml.safe_load(f)[config_name])
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)

    # Check the yaml args are valid
    known_args = set(vars(parser.parse_args("")))
    unknown_args = set(yaml_args) - known_args

    if unknown_args:
        print(f" Warning: Unknown arg(s) in config file: {unknown_args}")

    parser.set_defaults(**yaml_args)
    args = parser.parse_args(remaining_args)

    # Expand matmul_proportion input into list representation
    if isinstance(args.matmul_proportion, float):
        args.matmul_proportion = [args.matmul_proportion] * args.ipus_per_replica

    if len(args.matmul_proportion) != args.ipus_per_replica:
        if len(args.matmul_proportion) == 1:
            args.matmul_proportion = args.matmul_proportion * args.ipus_per_replica
        else:
            raise ValueError(f"Length of matmul_proportion doesn't match ipus_per_replica: "
                             f"{args.matmul_proportion} vs {args.ipus_per_replica}")

    args.global_batch_size = args.replication_factor * args.gradient_accumulation * args.batch_size
    args.samples_per_step = args.global_batch_size * args.batches_per_step
    return args
