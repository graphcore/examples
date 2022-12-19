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
import popdist
import horovod.torch as hvd

config_file = "./configs.yml"


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise argparse.ArgumentTypeError(f'{value} is not a valid boolean value')


def init_popdist(args):
    hvd.init()
    popdist.init()
    if popdist.getNumTotalReplicas() != args.replication_factor:
        print(f"The number of replicas is overridden by PopRun. "
              f"The new value is {popdist.getNumTotalReplicas()}.")
    args.replication_factor = int(popdist.getNumLocalReplicas())

    args.popdist_rank = popdist.getInstanceIndex()
    args.popdist_size = popdist.getNumInstances()


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        "Poptorch ViT",
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration Name",
        required=True)
    pargs, remaining_args = parser.parse_known_args(args=args)

    # Execution
    parser.add_argument("--micro-batch-size", type=int,
                        help="Set the micro batch-size")
    parser.add_argument("--training-steps", type=int,
                        default=0, help="Number of training steps")
    parser.add_argument("--epochs", type=int, default=0,
                        help="Number of epochs")
    parser.add_argument("--device-iterations", type=int,
                        help="Number of batches per training step")
    parser.add_argument("--replication-factor", type=int,
                        help="Number of replicas")
    parser.add_argument("--gradient-accumulation", type=int,
                        help="Number of gradients to accumulate before updating the weights")
    parser.add_argument("--half-partials", type=str_to_bool,
                        nargs="?", const=True, default=True,
                        help="Set the data type of partial results for matrix multiplication "
                        "and convolution operators")
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
    parser.add_argument("--layers-per-ipu", type=int, nargs="+",
                        help="number of layers placed on each IPU")
    parser.add_argument("--prefetch-depth", type=int,
                        help="Prefetch buffering depth")
    parser.add_argument("--pretrain", type=str_to_bool, nargs="?", const=True, default=False,
                        help="A flag that marks if training from scracth or not")
    parser.add_argument("--reduction-type", type=str, choices=["sum", "mean"], default=None,
                        help="reduction type of accumulation and replication.")
    parser.add_argument("--layer-norm-eps", type=float,
                        help="LayerNorm epsilon")
    parser.add_argument("--enable-rts", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling RTS")
    parser.add_argument("--optimizer-state-offchip", type=str_to_bool, nargs="?", const=True, default=True,
                        help="Set the tensor storage location for optimizer state to be offchip.")
    # Optimizer
    parser.add_argument("--optimizer", type=str, choices=['SGD', 'Adam'],
                        help="optimizer to use for the training")
    parser.add_argument("--learning-rate", type=float,
                        help="Learning rate value for constant schedule, "
                        "maximum for linear schedule.")
    parser.add_argument("--lr-schedule", type=str, choices=["constant", "linear", "cosine"],
                        help="Type of learning rate schedule. "
                        "--learning-rate will be used as the max value")
    parser.add_argument("--auto-loss-scaling", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enable automatic loss scaling for half precision training.")
    parser.add_argument("--loss-scaling", type=float,
                        help="Loss scaling factor (recommend using powers of 2)")
    parser.add_argument("--weight-decay", type=float,
                        help="Set the weight decay")
    parser.add_argument("--momentum", type=float,
                        help="The momentum factor of optimizer")
    parser.add_argument("--warmup-steps", type=int,
                        help="Number of warmup steps")
    parser.add_argument("--adam-betas", nargs="+", type=float, default=None,
                        help="betas in ADAM or LAMB optimizer, [beta1, beta2]. "
                        "None will result in default setting in Adam / LAMB optimizer.")
    parser.add_argument("--adam-eps", type=float, default=None,
                        help="Optimizer term added to the denominator to ensure numerical stability/")
    parser.add_argument("--bias-correction", type=bool, default=None,
                        help="Compute Adam/LAMB with bias correction")
    parser.add_argument("--use-combined-accum", type=str_to_bool,
                        nargs="?", const=True, default=False,
                        help="PopTorch uses a single tensor (the combined tensor) for gradient accumulation"
                        " and velocity of the SGD optimizer.")
    parser.add_argument("--accum-type", type=str, choices=['fp16', 'fp32'], default='fp16',
                        help="Data type used for gradients.")
    parser.add_argument("--first-order-type", type=str, choices=['fp16', 'fp32'], default='fp32',
                        help="Data type used to store the first order momentum values for each parameter.")
    parser.add_argument("--second-order-type", type=str, choices=['fp16', 'fp32'], default='fp32',
                        help="Data type used to store the second order momentum values for each parameter.")
    parser.add_argument('--max-norm', type=float,
                        default=65535, help="the max weight norm for lamb")
    parser.add_argument('--max-norm-bias', type=float,
                        default=0.0, help="the max weight norm for bias param")

    # Model
    parser.add_argument("--hidden-size", type=int,
                        help="The size of the hidden state of the transformer layers")
    parser.add_argument("--num-hidden-layers", type=int,
                        help="The number of transformer layers")
    parser.add_argument("--num-attention-heads", type=int,
                        help="Set the number of heads in self attention")
    parser.add_argument("--mlp-dim", type=int,
                        help="The size of mlp dimention")
    parser.add_argument("--hidden-dropout-prob", type=float, nargs="?", const=True,
                        help="Dropout probability")
    parser.add_argument("--patches-size", type=float,
                        nargs="+", help="The size of image tokens")
    parser.add_argument("--num-labels", type=int, help="The number of classes")
    parser.add_argument("--attention-probs-dropout-prob", type=float, nargs="?", const=True,
                        help="Attention dropout probability")
    parser.add_argument("--representation-size", type=int, default=None,
                        help="Representation size of head when pretraining")
    parser.add_argument('--drop_path_rate', type=float,
                        default=0.1, help="stochastic depth rate")
    parser.add_argument("--loss", type=str, choices=['SigmoidCELoss', 'CELoss'],
                        help="Loss function for the training")
    parser.add_argument("--recompute-mid-layers", nargs="+", type=float, default=None,
                        help="Index of layers to add a recompute point in the middle "
                        "of an attention block")

    # Dataset
    parser.add_argument('--dataset', choices=['cifar10', 'imagenet1k', 'generated'],
                        default='cifar10', help="Choose data")
    parser.add_argument("--dataset-path", type=str, help="Input data files")
    parser.add_argument("--rebatched-worker-size", type=int, default=None,
                        help="Set the rebatched worker size")
    parser.add_argument("--synthetic-data", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Random data created on IPU")
    parser.add_argument("--mixup", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling mixup data augmentation")
    parser.add_argument("--alpha", type=float,
                        help="alpha parameter in beta distribution when applying mixup")
    parser.add_argument("--extra-aug", type=str, choices=["imagenet_policy", ],
                        help="extra data augmentation pipelines", default="cutout_basic_randaug")
    parser.add_argument("--byteio", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Use byte data format to transfer data from host to IPU")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of iterations when using generated data")

    # Misc
    parser.add_argument("--dataloader-workers", type=int,
                        help="The number of dataloader workers")
    parser.add_argument("--wandb", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Enabling logging to Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="torch-vit",
                        help="wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="wandb run name")
    parser.add_argument("--executable-cache-dir", type=str, default="",
                        help="Directory where Poplar executables are cached. If set, recompilation of identical graphs can be avoided. "
                        "Required for both saving and loading executables.")
    parser.add_argument("--profile-dir", type=str,
                        help="Directory for profiling results")

    # Checkpointing
    parser.add_argument("--checkpoint-output-dir", type=str, default="",
                        help="Directory where checkpoints will be saved and restored from."
                        "This can be either an absolute or relative path. If this is "
                        "not specified, only end of run checkpoint is saved in an automatically "
                        "generated directory at the root of this project. Specifying directory is"
                        "recommended to keep track of checkpoints.")
    parser.add_argument("--checkpoint-save-steps", type=int, default=100,
                        help="Option to checkpoint model after n steps.")
    parser.add_argument("--resume-training-from-checkpoint", type=str_to_bool, nargs="?", const=True, default=False,
                        help="Restore both the model checkpoint and training state in order to resume a training run.")
    parser.add_argument("--pretrained-checkpoint", type=str, default="",
                        help="Checkpoint to be retrieved for further training. This can"
                        "be either an absolute or relative path to the checkpoint file.")

    # Load the yaml
    yaml_args = dict()

    with open(config_file, "r", encoding="UTF-8") as f:
        try:
            yaml_args.update(**yaml.safe_load(f)[pargs.config])
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit(1)

    # Check the yaml args are valid
    known_args = set(vars(parser.parse_args()))
    unknown_args = set(yaml_args) - known_args

    if unknown_args:
        raise ValueError(f"Unknown arg(s) in config file: {unknown_args}")

    parser.set_defaults(**yaml_args)
    args = parser.parse_args(['--config', pargs.config] + remaining_args)

    # Expand matmul_proportion input into list representation
    if isinstance(args.matmul_proportion, float):
        args.matmul_proportion = [
            args.matmul_proportion] * args.ipus_per_replica
    else:
        if len(args.matmul_proportion) != args.ipus_per_replica:
            if len(args.matmul_proportion) == 1:
                args.matmul_proportion = args.matmul_proportion * args.ipus_per_replica
            else:
                raise ValueError(
                    f"Length of matmul_proportion doesn't match ipus_per_replica: "
                    f"{args.matmul_proportion} vs {args.ipus_per_replica}")

    args.use_popdist = False
    if popdist.isPopdistEnvSet():
        args.use_popdist = True
        init_popdist(args)

    args.global_batch_size = args.replication_factor * \
        args.gradient_accumulation * args.micro_batch_size
    args.samples_per_step = args.global_batch_size * args.device_iterations
    return args


if __name__ == "__main__":
    config = parse_args()
    print(config)
