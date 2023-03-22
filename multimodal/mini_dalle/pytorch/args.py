# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import os
import sys
import yaml
import argparse
import torch
import popdist
import popdist.poptorch
import horovod.torch as hvd

config_file = os.path.join(os.path.dirname(__file__), "configs.yml")


def str_to_bool(value):
    if isinstance(value, bool) or value is None:
        return value
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise argparse.ArgumentTypeError(f"{value} is not a valid boolean value")


def init_popdist(args):
    hvd.init()
    args.use_popdist = True
    if popdist.getNumTotalReplicas() != args.replication_factor:
        print(f"The number of replicas is overridden by PopRun. " f"The new value is {popdist.getNumTotalReplicas()}.")
    args.replication_factor = int(popdist.getNumLocalReplicas())
    args.popdist_rank = popdist.getInstanceIndex()
    args.popdist_size = popdist.getNumInstances()


def parse_args(args=None):
    pparser = argparse.ArgumentParser("DALL-E Configuration name", add_help=False)
    pparser.add_argument("--config", type=str, help="Configuration Name", default="unit_test")
    pargs, remaining_args = pparser.parse_known_args(args=args)
    config_name = pargs.config

    parser = argparse.ArgumentParser(
        "PopTorch mini DALL-E", add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Execution
    parser.add_argument("--batch-size", type=int, help="Set the micro batch-size")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--device-iterations", type=int, help="Number of batches per training step")
    parser.add_argument("--replication-factor", type=int, default=1, help="Number of replicas")
    parser.add_argument(
        "--gradient-accumulation", type=int, help="Number of gradients to accumulate before updating the weights."
    )
    parser.add_argument(
        "--stochastic-rounding",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable stochastic rounding",
    )
    parser.add_argument(
        "--embedding-serialization-factor", type=int, default=1, help="Matmul serialization factor the embedding layers"
    )
    parser.add_argument(
        "--enable-half-partials",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable half partials for matmuls and convolutions globally",
    )
    parser.add_argument("--enable-rts", type=str_to_bool, nargs="?", const=True, default=False, help="Enabling RTS")
    parser.add_argument(
        "--optimizer-state-offchip",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Set the tensor storage location for optimizer state to be offchip.",
    )
    parser.add_argument("--ipus-per-replica", type=int, default=1, help="Number of IPUs required by each replica")
    parser.add_argument(
        "--layers-per-ipu", type=int, nargs="+", default=[0, 0, 8, 8], help="number of layers placed on each IPU"
    )
    parser.add_argument("--cls-ipu-id", type=int, default=None, help="IPU id of classification layer")
    parser.add_argument("--embedding-ipu-id", type=int, default=0, help="IPU id of image embedding and text embedding")
    parser.add_argument(
        "--matmul-proportion",
        type=float,
        nargs="+",
        default=[0.6],
        help="Relative IPU memory proportion size allocated for matmul",
    )
    parser.add_argument(
        "--async-dataloader",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable asynchronous mode in the DataLoader",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Seed for RNG")
    parser.add_argument("--fp16", action="store_true", help="Use fp16, otherwise use fp32")

    # Optimizer
    parser.add_argument(
        "--optimizer", type=str, choices=["Adam", "AdamW"], default="Adam", help="optimizer to use for the training"
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Learning rate value for constant schedule, maximum for linear schedule."
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["ReduceLROnPlateau", "multi_step", "constant"],
        help="Learning rate scheduler",
    )
    parser.add_argument("--loss-scaling", type=float, help="Loss scaling factor (recommend using powers of 2)")
    parser.add_argument("--enable-als", action="store_true", help="Enable automatic loss scaling")
    parser.add_argument("--weight-decay", type=float, default=0, help="Set the weight decay")
    parser.add_argument(
        "--enable-half-first-order-momentum",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Use float16 for the first order momentum in the optimizer.",
    )

    # Model
    parser.add_argument("--hidden-size", type=int, help="The size of the hidden state of the transformer layers")
    parser.add_argument("--text-seq-len", type=int, help="The max text sequence length")
    parser.add_argument("--num-hidden-layers", type=int, help="The number of transformer layers")
    parser.add_argument("--num-attention-heads", type=int, help="Set the number of heads in self attention")
    parser.add_argument("--dim-head", type=int, help="The number of head dimension in self attention")
    parser.add_argument("--ff-dropout", type=float, nargs="?", const=True, help="Attention dropout probability")
    parser.add_argument("--attn-dropout", type=float, nargs="?", const=True, help="Feed forward dropout probability")
    parser.add_argument(
        "--sandwich-norm", type=str_to_bool, nargs="?", const=True, default=False, help="Use Sandwich LayerNorm"
    )
    parser.add_argument("--loss-img-weight", default=7, type=int, help="Image loss weight")
    parser.add_argument(
        "--attn-types",
        default="full",
        type=str,
        help="comma separated list of attention types(full, axial_row, axial_col, conv_like).",
    )
    parser.add_argument("--bpe-path", type=str, help="Path to BPE json file")
    parser.add_argument(
        "--truncate-captions",
        action="store_true",
        help="Captions passed in which exceed the max token length will be truncated.",
    )

    # Dataset
    parser.add_argument(
        "--byteio",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Use byte data format to transfer image data from host to IPU",
    )
    parser.add_argument("--input-folder", type=str, default=None, help="Path to folder of images and text for training")
    parser.add_argument(
        "--generated-data",
        action="store_true",
        help="Random data created on CPU ram, dataloader feeds the same data to IPU every step",
    )
    parser.add_argument(
        "--synthetic-data", action="store_true", help="Random data created on the IPU, no host to IPU IO"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-output-dir",
        type=str,
        default="",
        help="Directory where checkpoints will be saved to.\
                             This can be either an absolute or relative path.",
    )
    parser.add_argument(
        "--checkpoint-save-steps", default=1000, type=int, help="Option to checkpoint model after n steps."
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        help="Checkpoint to be retrieved for further training. This can"
        "be either an absolute or relative path to the checkpoint file.",
    )
    parser.add_argument("--vae_path", type=str, help="path to trained discrete VAE")
    parser.add_argument(
        "--vqgan_model_path", type=str, default=None, help="path to trained VQGAN weights. This should be a .ckpt file."
    )
    parser.add_argument(
        "--vqgan_config_path", type=str, default=None, help="path to trained VQGAN config. This should be a .yaml file."
    )

    # Misc
    parser.add_argument("--dataloader-workers", type=int, help="The number of dataloader workers")
    parser.add_argument(
        "--wandb", type=str_to_bool, nargs="?", const=True, default=False, help="Enabling logging to Weights and Biases"
    )
    parser.add_argument("--wandb-project-name", default="miniDALL-E", help="Wandb project name")
    parser.add_argument("--wandb-run-name", default=None, help="Weights & Biases run name.")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Create an offline IPU target that can only be used for offline compilation.",
    )
    parser.add_argument(
        "--executable-cache-dir",
        type=str,
        default="",
        help="Directory where Poplar executables are cached. If set, recompilation of identical graphs can be avoided. "
        "Required for both saving and loading executables.",
    )
    parser.add_argument("--profile-dir", type=str, help="Directory for profiling results")

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

    # Initialise PopDist
    if popdist.isPopdistEnvSet():
        init_popdist(args)
        hvd.broadcast(torch.Tensor([args.random_seed]), root_rank=0)
    else:
        args.use_popdist = False

    # Expand matmul_proportion input into list representation
    if isinstance(args.matmul_proportion, float):
        args.matmul_proportion = [args.matmul_proportion] * args.ipus_per_replica

    if len(args.matmul_proportion) != args.ipus_per_replica:
        if len(args.matmul_proportion) == 1:
            args.matmul_proportion = args.matmul_proportion * args.ipus_per_replica
        else:
            raise ValueError(
                f"Length of matmul_proportion doesn't match ipus_per_replica: "
                f"{args.matmul_proportion} vs {args.ipus_per_replica}"
            )

    if args.synthetic_data or args.generated_data:
        print(
            "Warning: One of --synthetic-data or --generated-data have been "
            "provided. This will override the value passed to --input-folder, "
            "and real data will not be used."
        )
        args.input_folder = None
    if args.synthetic_data and args.generated_data:
        print(
            "Warning: Both of --synthetic-data and --generated-data have been "
            "provided. Only one can be provided. Exiting."
        )
        sys.exit(1)

    return args


def sync_metrics(outputs, factor=1, average=True):
    if popdist.isPopdistEnvSet():
        if isinstance(outputs, float):
            return float(hvd.allreduce(torch.Tensor([outputs]), average=average).item())
        else:
            return [hvd.allreduce(output.div(factor), average=average).mean().item() for output in outputs]
    else:
        if isinstance(outputs, float):
            return outputs
        else:
            return [output.div(factor).mean().item() for output in outputs]
