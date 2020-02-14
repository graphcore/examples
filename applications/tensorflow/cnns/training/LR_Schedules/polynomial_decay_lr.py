# Copyright 2019 Graphcore Ltd.
import tensorflow as tf


class LearningRate:
    def __init__(self, opts, total_iterations):
        self.initial_lr = opts["poly_lr_initial_lr"]
        self.decay_steps = total_iterations
        self.power = opts["poly_lr_decay_power"]
        self.end_learning_rate = opts["poly_lr_end_lr"]

    def feed_dict_lr(self, iteration):
        return (self.initial_lr - self.end_learning_rate) * (1.0 - iteration / self.decay_steps) ** self.power + self.end_learning_rate


def add_arguments(parser):
    lr_group = parser.add_argument_group(
        'Polynomial Decay Learning Rate. Use with --lr-schedule poly_decay_tf.')

    lr_group.add_argument('--poly-lr-decay-power', type=float,
                          help="Exponent of polynomial decribing the decay. Default 1.0 (linear decay).")
    lr_group.add_argument('--poly-lr-initial-lr', type=float,
                          help="Initial learning rate, before decay.")
    lr_group.add_argument('--poly-lr-end-lr', type=float,
                          help="Final learning rate, after poly-lr-decay-steps.")
    return parser


def set_defaults(opts):
        # We only need to set defaults for the following if the user has specified a polynomial learning rate
    if not opts["poly_lr_initial_lr"]:
        opts["poly_lr_initial_lr"] = 0.0012
    if not opts["poly_lr_end_lr"]:
        opts["poly_lr_end_lr"] = 0.0001
    if not opts["poly_lr_decay_power"]:
        opts["poly_lr_decay_power"] = 1

    opts['summary_str'] += "Polynomial decay applied to learning rate with exponent {poly_lr_decay_power}. Initial rate of {poly_lr_initial_lr}, decaying to {poly_lr_end_lr}.\n"

    return opts
