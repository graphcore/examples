# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import random
from pathlib import Path
import torch
import horovod.torch as hvd
import logging
import weight_avg
import sys
import import_helper
import utils


def generate_random_seed(distributed=False):
    seed = random.randint(0, 2**32-1)
    if distributed:
        seed_tensor = torch.Tensor([seed])
        seed_tensor = hvd.broadcast(seed_tensor, root_rank=0)
        seed = int(seed_tensor.item())
    return seed


def parse_arguments():
    common_parser = utils.get_common_parser()
    parser = argparse.ArgumentParser(description='CNN training in PopTorch', parents=[common_parser])
    parser.add_argument('--data', choices=['cifar10', 'imagenet', 'synthetic', 'generated'], default='cifar10', help="Choose data")
    parser.add_argument('--precision', choices=['16.16', '16.32', '32.32'], default='16.16', help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 16.32, 32.32")
    parser.add_argument('--imagenet-data-path', type=str, default="/localdata/datasets/imagenet-raw-data", help="Path of the raw imagenet data")
    parser.add_argument('--gradient-accumulation', type=int, default=1, help="Number of batches to accumulate before a gradient update")
    parser.add_argument('--lr', type=float, default=0.01, help="Initial learning rate")
    parser.add_argument('--weight-decay', type=float, default=0.0001, help="L2 parameter penalty")
    parser.add_argument('--optimizer', choices=['sgd', 'sgd_combined', 'adamw', 'rmsprop', 'rmsprop_tf'], default='sgd', help="Define the optimizer")
    parser.add_argument('--optimizer-eps', type=float, default=1e-8, help="Small constant added to the updater term denominator for numerical stability.")
    parser.add_argument('--momentum', type=float, default=0.0, help="Momentum factor")
    parser.add_argument('--rmsprop-decay', type=float, default=0.99, help="RMSprop smoothing constant")
    parser.add_argument('--epoch', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--checkpoint-path', type=str, default="", help="Checkpoint path(if it is not defined, no checkpoint is created")
    parser.add_argument('--validation-mode', choices=['none', 'during', 'after'], default="after", help='The model validation mode. none=no validation; during=validate after every epoch; after=validate after the training')
    parser.add_argument('--disable-metrics', action='store_true', help='Do not calculate metrics during training, useful to measure peak throughput')
    parser.add_argument('--wandb', action='store_true', help="Add Weights & Biases logging")
    parser.add_argument('--wandb-weight-histogram', action='store_true', help="Log the weight histogram with Weights & Biases")
    parser.add_argument('--seed', type=int, help="Set the random seed")
    parser.add_argument('--recompute-mode', default="none", choices=['none', 'auto', 'manual'], help="Select single IPU recompute mode. If the model is multi "
                        "stage (pipelined) the recomputation is always enabled. Auto mode selects the recompute checkpoints automatically. Rest of the network will be recomputed. It is possible to extend the recompute checkpoints "
                        " with the --recompute-checkpoints option. In manual mode no recompute checkpoint is added, they need to be determined by the user.")
    parser.add_argument('--recompute-checkpoints', type=str, nargs='+', default=[], help='List of recomputation checkpoints. List of regex rules for the layer names must be provided. (Example: Select convolutional layers: .*conv.*)')
    parser.add_argument('--disable-stable-batchnorm', action='store_true', help="There are two implementations of the batch norm layer. "
                        "The default version is numerically more stable. The less stable is faster.")
    parser.add_argument('--offload-optimizer', action='store_true', help='Offload the optimizer from the IPU memory')
    parser.add_argument('--available-memory-proportion', type=float, default=[], nargs='+',
                        help='Proportion of memory which is available for convolutions. Use a value of less than 0.6')
    parser.add_argument('--logs-per-epoch', type=int, default=1, help="The number of times the resuls are logged and a checkpoint is saved in each epoch")
    parser.add_argument('--validation-frequency', type=int, default=4, help="How many training epochs to run between validation steps")
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing factor (Default=0 => no smoothing)')
    # LR schedule related params
    parser.add_argument('--lr-schedule', choices=["step", "cosine", "exponential"], default="step", help="Learning rate schedule")
    parser.add_argument('--lr-decay', type=float, default=0.5, help="Learning rate decay")
    parser.add_argument('--lr-epoch-decay', type=int, nargs='+', default=[], help="List of epoch, when lr drops")
    parser.add_argument('--warmup-epoch', type=int, default=0, help="Number of learning rate warmup epochs")
    parser.add_argument('--lr-scheduler-freq', type=float, default=0, help="Number of lr scheduler updates per epoch (0 to disable and update every iteration)")
    # half precision training params
    parser.add_argument('--loss-scaling', type=float, default=1.0, help="Loss scaling factor. This value is reached by the end of the training.")
    parser.add_argument('--loss-velocity-scaling-ratio', type=float, default=1.0, help="Only for sgd_combined optimizer: Loss Velocity / Velocity scaling ratio. In case of large number of replicas >1.0 can increase numerical stability")
    parser.add_argument('--initial-loss-scaling', type=float, help="Initial loss scaling factor. The loss scaling interpolates between this and loss-scaling value."
                        "Example: 100 epoch, initial loss scaling 16, loss scaling 128: Epoch 1-25 ls=16;Epoch 26-50 ls=32;Epoch 51-75 ls=64;Epoch 76-100 ls=128")
    parser.add_argument('--enable-stochastic-rounding', action="store_true", help="Enable Stochastic Rounding")
    parser.add_argument('--enable-fp-exceptions', action="store_true", help="Enable Floating Point Exceptions")
    parser.add_argument('--webdataset-percentage-to-use', type=int, default=100, choices=range(1, 101), help="Percentage of dataset to be used for traini")
    parser.add_argument('--use-bbox-info', action='store_true', help='Use bbox information for training: reject the augmenetation, which does not overlap with the object.')
    parser.add_argument('--mixup-alpha', type=float, default=0.0, help="The first shape parameter of the beta distribution used to sample mixup coefficients. The second shape parameter is the same as the first one. Value of 0.0 means mixup is disabled.")
    parser.add_argument('--cutmix-lambda-low', type=float, default=0.0, help="Lower bound for the cutmix lambda coefficient (lambda is sampled uniformly from [low, high)). If both bounds are set to 0.0 or 1.0, cutmix is disabled. If both bounds are equal, lambda always equals that value.")
    parser.add_argument('--cutmix-lambda-high', type=float, default=0.0, help="Higher bound for the cutmix lambda coefficient (lambda is sampled uniformly from [low, high)). If both bounds are set to 0.0 or 1.0, cutmix is disabled. If both bounds are equal, lambda always equals that value.")
    parser.add_argument('--cutmix-disable-prob', type=float, default=0.0, help="Probability that cutmix is disabled for a particular batch.")
    parser.add_argument("--compile-only", action="store_true", help="Create an offline IPU target that can only be used for offline compilation.")

    # weight averaging params
    weight_avg.add_parser_arguments(parser)

    opts = utils.parse_with_config(parser, Path(__file__).parent.absolute().joinpath("configs.yml"))
    if opts.initial_loss_scaling is None:
        opts.initial_loss_scaling = opts.loss_scaling

    utils.handle_distributed_settings(opts)

    if opts.seed is None:
        opts.seed = generate_random_seed(opts.use_popdist)

    # setup logging
    utils.Logger.setup_logging_folder(opts)

    num_stages = len(opts.pipeline_splits)+1
    num_amps = len(opts.available_memory_proportion)
    if num_stages > 1 and num_amps > 0 and num_amps != num_stages and num_amps != 1:
        logging.error(f'--available-memory-proportion number of elements should be either 1 or equal to the number of pipeline stages: {num_stages}')
        sys.exit()

    if opts.weight_avg_strategy != 'none' and opts.checkpoint_path == '':
        logging.error('Please provide a --checkpoint-path folder to apply weight averaging to.')
        sys.exit()

    if opts.batch_size == 1 and opts.norm_type == "batch":
        logging.warning("BatchNorm with batch size of 1 may cause instability during inference.")

    if num_stages > 1:
        logging.info("Recomputation is always enabled when using pipelining.")

    if opts.recompute_mode == "none" and len(opts.recompute_checkpoints) > 0 and num_stages == 1:
        logging.warning("Recomputation is not enabled, while recomputation checkpoints are provided.")

    if opts.eight_bit_io and opts.normalization_location == 'host':
        logging.warning("for eight-bit input, please use IPU-side normalisation, setting normalisation to IPU")
        opts.normalization_location = 'ipu'

    if opts.wandb_weight_histogram:
        assert opts.wandb, "Need to enable W&B with --wandb to log the histogram of the weights"

    assert opts.mixup_alpha >= 0.0, "Mixup alpha must be >= 0.0"
    opts.mixup_enabled = opts.mixup_alpha > 0.0

    assert opts.cutmix_lambda_low >= 0.0, "Lower bound for cutmix lambda must be >= 0.0"
    assert opts.cutmix_lambda_high <= 1.0, "Higher bound for cutmix lambda must be <= 1.0"
    assert opts.cutmix_lambda_low <= opts.cutmix_lambda_high, "Lower bound for cutmix lambda must be <= higher bound"
    assert 0.0 <= opts.cutmix_disable_prob <= 1.0, "Probability for disabling cutmix must be in [0, 1]"
    opts.cutmix_enabled = (opts.cutmix_lambda_low, opts.cutmix_lambda_high) not in ((0.0, 0.0), (1.0, 1.0))

    if opts.mixup_enabled and opts.cutmix_enabled and opts.batch_size < 4:
        logging.error('Using mixup and cutmix together requires at least batch size 4')
        sys.exit()
    elif opts.mixup_enabled and opts.batch_size < 2:
        logging.error('Using mixup requires at least batch size 2')
        sys.exit()
    elif opts.cutmix_enabled and opts.batch_size < 3:
        logging.error('Using cutmix requires at least batch size 3')
        sys.exit()

    if opts.compile_only and (opts.data != "generated"):
        logging.warning(
              "Warning: --generated-data must be set for compile only " +
              "mode. Defaulting to using generated data. --input-files" +
              " will be ignored for compile only mode.")
        opts.data = "generated"
        # Removing real data path
        opts.imagenet_data_path = None

    return opts
