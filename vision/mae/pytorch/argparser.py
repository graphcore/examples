# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import yaml
from util.log import logger

config_file = os.path.join(os.path.dirname(__file__), "configs.yml")


def get_args_parser():
    parser = argparse.ArgumentParser("MAE fine-tuning for image classification", add_help=False)
    parser.add_argument("--config", default="vit_base_finetune", type=str, help="Configuration name")
    pargs, remaining_args = parser.parse_known_args()
    config_name = pargs.config
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * gradient_accumulation_count * # ipus",
    )
    parser.add_argument("--epochs", default=50, type=int)

    # Model parameters
    parser.add_argument(
        "--model", default="vit_large_patch16", type=str, metavar="MODEL", help="Name of model to train"
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument("--drop_path", type=float, default=0.1, metavar="PCT", help="Drop path rate (default: 0.1)")

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad", type=float, default=None, metavar="NORM", help="Clip gradient norm (default: None, no clipping)"
    )
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument("--layer_decay", type=float, default=0.65, help="layer-wise lr decay from ELECTRA/BEiT")

    parser.add_argument(
        "--min_lr", type=float, default=1e-6, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )

    parser.add_argument("--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR")

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")

    # * Random Erase params
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT", help="Random erase prob (default: 0.25)")
    parser.add_argument("--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")')
    parser.add_argument("--recount", type=int, default=1, help="Random erase count (default: 1)")
    parser.add_argument(
        "--resplit", action="store_true", default=False, help="Do not random erase first (clean) augmentation split"
    )

    # * Mixup params
    parser.add_argument("--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0.")
    parser.add_argument("--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0.")
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    # Dataset parameters
    parser.add_argument("--data_path", default="/datasets01/imagenet_full_size/061417/", type=str, help="dataset path")
    parser.add_argument("--nb_classes", default=1000, type=int, help="number of the classification types")

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--print_freq", default=10)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--generated_data", action="store_true", help="Use host generated data instead of real imagenet data."
    )
    parser.add_argument("--saveckp_freq", default=10)
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument("--pipeline", type=int, nargs="+", help="set modules on multi ipus")
    parser.add_argument("--gradient_accumulation_count", default=256, type=int, help="gradient accumulate")
    parser.add_argument("--device_iterations", default=1, type=int, help="device iteration number")
    parser.add_argument("--replica", default=4, type=int, help="model replic count")
    parser.add_argument("--half", action="store_true", help="if use float16")
    parser.add_argument("--ipus", default=4, type=int, help="ipu count for one model")
    parser.add_argument(
        "--async_type", default="normal", type=str, choices=["async", "rebatch", "normal"], help="use async data loader"
    )
    parser.add_argument("--rebatched_worker_size", type=int, default=128, help="rebatched worker size")
    parser.add_argument("--loss_scale", type=float, default=128.0)
    parser.add_argument("--output", default="./ipu_out", help="path where to save, empty for no saving")
    parser.add_argument("--log", default="log_info.txt", help="path where to tensorboard log")

    # WandB related
    parser.add_argument("--wandb", action="store_true", help="Turn on Weights and Biases logging.")
    parser.add_argument("--wandb_project_name", default="torch-mae", type=str, help="Weights and Biases project name.")
    parser.add_argument("--wandb_run_name", default=None, type=str, help="Weights and Biases run name.")

    # compile only
    parser.add_argument("--compile_only", action="store_true", help="Exit after compiling model.")

    yaml_args = dict()
    if config_name is not None:
        with open(config_file, "r") as f:
            try:
                yaml_args.update(**yaml.safe_load(f)[config_name])
            except yaml.YAMLError as exc:
                logger.info(exc)
                sys.exit()
    # check the yaml args are valid
    known_args = set(vars(parser.parse_args("")))
    unknown_args = set(yaml_args) - known_args
    if unknown_args:
        logger.info(f" Warning: Unknown arg(s) in config file: {unknown_args}")
    parser.set_defaults(**yaml_args)
    args = parser.parse_args()
    # helper argsipu_per_replica=args.ipus
    args.pretrain = False

    return args
