# Copyright 2019 Graphcore Ltd.
import numpy as np


class LearningRate:
    """An exponential learning rate schedule with optional warmup"""
    def __init__(self, opts, total_iterations):
        self.decay = opts["lr_decay_rate"]
        self.lr = (2 ** opts["base_learning_rate"]) * opts["total_batch_size"]
        self.iterations_per_epoch = total_iterations / opts["epochs"]
        self.freq = opts['epochs']/opts["lr_drops"]
        self.drops = 0
        self.warmup = opts['warmup_epochs'] if opts["epochs"] else False
        if self.warmup:
            self.warmup_length = self.iterations_per_epoch * opts["warmup_epochs"]

    def feed_dict_lr(self, iteration):
        epoch = iteration / self.iterations_per_epoch
        if epoch/self.freq >= self.drops + 1:
            n_drops = int(np.floor((epoch/self.freq) - self.drops))
            assert n_drops > 0
            self.lr *= (self.decay ** n_drops)
            self.drops += n_drops

        if self.warmup and iteration < self.warmup_length:
            return (iteration * self.lr) / self.warmup_length
        else:
            return self.lr


def add_arguments(parser):
    lr_group = parser.add_argument_group('Exponential Learning Rate Decay')
    lr_group.add_argument('--lr-decay-rate', type=float,
                          help="Learning rate rate")
    lr_group.add_argument('--lr-drops', type=int,
                          help="Number of equally spaced learning rate drops")
    lr_group.add_argument('--warmup-epochs', type=int, default=5,
                          help="Warmup length in epochs (Default=5, set to 0 for no warmup)")
    return parser


def set_defaults(opts):
    opts['summary_str'] += "Exponential LR schedule\n"
    if opts["warmup_epochs"] > 0:
        opts['summary_str'] += " Warmup: {} epochs\n".format('{warmup_epochs}')
    else:
        opts['summary_str'] += " No warmup\n"

    opts['summary_str'] += (" Decay Rate: {lr_decay_rate}\n"
                            " Decayed {lr_drops} times)\n")
    return opts
