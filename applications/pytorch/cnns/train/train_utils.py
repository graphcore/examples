# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import argparse
import torch
import logging
import sys
sys.path.append('..')
import models
import utils


def accuracy(predictions, labels):
    _, ind = torch.max(predictions, 1)
    # provide labels only for samples, where prediction is available (during the training, not every samples prediction is returned for efficiency reasons)
    labels = labels[-predictions.size()[0]:]
    accuracy = torch.sum(torch.eq(ind, labels)).item() / labels.size()[0] * 100.0
    return accuracy


def parse_arguments():
    common_parser = utils.get_common_parser()
    parser = argparse.ArgumentParser(description='CNN training in PopTorch', parents=[common_parser])
    parser.add_argument('--data', choices=['cifar10', 'imagenet', 'synthetic'], default='cifar10', help="Choose data")
    parser.add_argument('--precision', choices=['16.16', '16.32', '32.32'], default='16.16', help="Precision of Ops(weights/activations/gradients) and Master data types: 16.16, 16.32, 32.32")
    parser.add_argument('--imagenet-data-path', type=str, default="/localdata/datasets/imagenet-raw-data", help="Path of the raw imagenet data")
    parser.add_argument('--gradient-accumulation', type=int, default=1, help="Number of batches to accumulate before a gradient update")
    parser.add_argument('--lr', type=float, default=0.01, help="Initial learning rate")
    parser.add_argument('--weight-decay', type=float, default= 0.0001, help="L2 parameter penalty")
    parser.add_argument('--momentum', type=float, default=0.0, help="Momentum factor")
    parser.add_argument('--rmsprop-decay', type=float, default=0.99, help="RMSprop smoothing constant")
    parser.add_argument('--epoch', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--checkpoint-path', type=str, default="", help="Checkpoint path(if it is not defined, no checkpoint is created")
    parser.add_argument('--no-validation', action='store_true', help='Do not validate at the end of the training')
    parser.add_argument('--disable-metrics', action='store_true', help='Do not calculate metrics during training, useful to measure peak throughput')
    parser.add_argument('--wandb', action='store_true', help="Add Weights & Biases logging")
    parser.add_argument('--enable-pipeline-recompute', action='store_true', help='Enable the recomputation of network activations during backward pass instead of caching them during forward pass')
    parser.add_argument('--available-memory-proportion', type=float, default=[], nargs='+',
                        help='Proportion of memory which is available for convolutions. Use a value of less than 0.6')
    parser.add_argument('--logs-per-epoch', type=int, default=1, help="The number of times the resuls are logged and a checkpoint is saved in each epoch")
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing factor (Default=0 => no smoothing)')
    # LR schedule related params
    parser.add_argument('--lr-schedule', choices=["step", "cosine", "exponential"], default="step", help="Learning rate schedule")
    parser.add_argument('--lr-decay', type=float, default=0.5, help="Learning rate decay")
    parser.add_argument('--lr-epoch-decay', type=int, nargs='+', default=[], help="List of epoch, when lr drops")
    parser.add_argument('--warmup-epoch', type=int, default=0, help="Number of learning rate warmup epochs")
    parser.add_argument('--lr-scheduler-freq', type=float, default=0, help="Number of lr scheduler updates per epoch (0 to disable and update every iteration)")
    parser.add_argument('--optimizer', choices=['sgd', 'adamw', 'rmsprop'], default='sgd', help="Define the optimizer")
    # half precision training params
    parser.add_argument('--loss-scaling', type=float, default=1.0, help="Loss scaling factor")
    parser.add_argument('--enable-stochastic-rounding', action="store_true", help="Enable Stochastic Rounding")
    parser.add_argument('--reduction', choices=['sum', 'mean'], default='sum', help="Applied reduction for loss and gradients")
    # weight averaging params
    parser.add_argument('--weight-avg-strategy', default='none', choices=['mean', 'exponential', 'none'], help="Weight average strategy")
    parser.add_argument('--weight-avg-exp-decay', type=float, default=0.99, help="The exponential decay constant, applied if exponential weight average strategy is chosen")

    opts = utils.parse_with_config(parser, "configs.yml")
    # setup logging
    utils.Logger.setup_logging_folder(opts)

    num_stages = len(opts.pipeline_splits)+1
    num_amps = len(opts.available_memory_proportion)

    if opts.enable_pipeline_recompute and len(opts.pipeline_splits) == 0:
        logging.warn("autorecompute for single phase (non pipelined) models are not supported")

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

    return opts
