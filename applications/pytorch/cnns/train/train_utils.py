# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import random
import torch
import logging
import popdist
import popdist.poptorch
import horovod.torch as hvd
import sys
import weight_avg
sys.path.append('..')
import models
import utils


def generate_random_seed(distributed=False):
    seed = random.randint(0, 2**32-1)
    if distributed:
        seed_tensor = torch.Tensor([seed])
        seed_tensor = hvd.broadcast(seed_tensor, root_rank=0)
        seed = int(seed_tensor.item())
    return seed


def init_popdist(args):
    hvd.init()
    args.use_popdist = True
    if popdist.getNumTotalReplicas() != args.replicas:
        logging.warn(f"The number of replicas is overridden by poprun. The new value is {popdist.getNumTotalReplicas()}.")
    args.replicas = int(popdist.getNumLocalReplicas())
    args.popdist_rank = popdist.getInstanceIndex()
    args.popdist_size = popdist.getNumInstances()


def parse_arguments():
    common_parser = utils.get_common_parser()
    parser = argparse.ArgumentParser(description='CNN training in PopTorch', parents=[common_parser])
    parser.add_argument('--data', choices=['cifar10', 'imagenet', 'synthetic', 'generated'], default='cifar10', help="Choose data")
    parser.add_argument('--precision', choices=['16.16', '16.32', '32.32'], default='16.16', help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 16.32, 32.32")
    parser.add_argument('--imagenet-data-path', type=str, default="/localdata/datasets/imagenet-raw-data", help="Path of the raw imagenet data")
    parser.add_argument('--gradient-accumulation', type=int, default=1, help="Number of batches to accumulate before a gradient update")
    parser.add_argument('--lr', type=float, default=0.01, help="Initial learning rate")
    parser.add_argument('--weight-decay', type=float, default= 0.0001, help="L2 parameter penalty")
    parser.add_argument('--momentum', type=float, default=0.0, help="Momentum factor")
    parser.add_argument('--rmsprop-decay', type=float, default=0.99, help="RMSprop smoothing constant")
    parser.add_argument('--epoch', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--checkpoint-path', type=str, default="", help="Checkpoint path(if it is not defined, no checkpoint is created")
    parser.add_argument('--validation-mode', choices=['none', 'during', 'after'], default="after", help='The model validation mode. none=no validation; during=validate after every epoch; after=validate after the training')
    parser.add_argument('--disable-metrics', action='store_true', help='Do not calculate metrics during training, useful to measure peak throughput')
    parser.add_argument('--wandb', action='store_true', help="Add Weights & Biases logging")
    parser.add_argument('--seed', type=int, help="Set the random seed")
    parser.add_argument('--enable-recompute', action='store_true', help='Enable the recomputation of network activations during backward pass instead '
                        'of caching them during forward pass. This option turns on the recomputation for single-stage models. If the model is multi '
                        'stage (pipelined) the recomputation is always enabled.')
    parser.add_argument('--recompute-checkpoints', type=str, nargs='+', default=[], help='List of recomputation checkpoint rules: [conv:store convolution activations|norm: store normlayer activations]')
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
    parser.add_argument('--optimizer', choices=['sgd', 'adamw', 'rmsprop'], default='sgd', help="Define the optimizer")
    # half precision training params
    parser.add_argument('--loss-scaling', type=float, default=1.0, help="Loss scaling factor. This value is reached by the end of the training.")
    parser.add_argument('--loss-velocity-scaling-ratio', type=float, default=1.0, help="Only for SGD optimizer: Loss Velocity / Velocity scaling ratio. In case of large number of replicas >1.0 can increase numerical stability")
    parser.add_argument('--initial-loss-scaling', type=float, help="Initial loss scaling factor. The loss scaling interpolates between this and loss-scaling value."
                        "Example: 100 epoch, initial loss scaling 16, loss scaling 128: Epoch 1-25 ls=16;Epoch 26-50 ls=32;Epoch 51-75 ls=64;Epoch 76-100 ls=128")
    parser.add_argument('--enable-stochastic-rounding', action="store_true", help="Enable Stochastic Rounding")
    parser.add_argument('--enable-fp-exceptions', action="store_true", help="Enable Floating Point Exceptions")
    # weight averaging params
    weight_avg.add_parser_arguments(parser)

    opts = utils.parse_with_config(parser, "configs.yml")
    if opts.initial_loss_scaling is None:
        opts.initial_loss_scaling = opts.loss_scaling

    # Initialise popdist
    if popdist.isPopdistEnvSet():
        init_popdist(opts)
    else:
        opts.use_popdist = False

    if opts.seed is None:
        opts.seed = generate_random_seed(opts.use_popdist)

    # setup logging
    utils.Logger.setup_logging_folder(opts)

    num_stages = len(opts.pipeline_splits)+1
    num_amps = len(opts.available_memory_proportion)

    if num_stages == 1 and num_amps > 0:
        logging.error('--available-memory-proportion should only be set when pipelining')
        sys.exit()
    elif num_stages > 1 and num_amps > 0 and num_amps != num_stages and num_amps != 1:
            logging.error(f'--available-memory-proportion number of elements should be either 1 or equal to the number of pipeline stages: {num_stages}')
            sys.exit()

    if opts.weight_avg_strategy != 'none' and opts.checkpoint_path == '':
        logging.error('Please provide a --checkpoint-path folder to apply weight averaging to.')
        sys.exit()

    if opts.batch_size == 1 and opts.norm_type == "batch":
        logging.warning("BatchNorm with batch size of 1 may cause instability during inference.")

    if num_stages > 1:
        logging.info("Recomputation is always enabled when using pipelining.")

    if not opts.enable_recompute and len(opts.recompute_checkpoints) > 0:
        logging.warning("Recomputation is not enabled, whlile recomputation checkpoints are provided.")

    return opts
