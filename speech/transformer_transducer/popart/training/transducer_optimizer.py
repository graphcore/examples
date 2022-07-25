# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import sys
import numpy as np

import logging_util

# set up logging
logger = logging_util.get_basic_logger('TRANSDUCER_OPTIMIZER')


class TransducerOptimizerFactory:
    def __init__(self, optimizer_type, base_lr, min_lr, exp_gamma, steps_per_epoch, warmup_epochs, hold_epochs,
                 beta1=None, beta2=None, weight_decay=None, opt_eps=None, loss_scaling=None, gradient_clipping_norm=None, max_weight_norm=None):
        """ Class for creating and updating popart optimizers
            :param str optimizer_type: optimizer type - 'SGD' or 'LAMB'
            :param float base_lr: base learning rate
            :param float min_lr: minimum learning rate
            :param float exp_gamma: gamma factor for exponential lr scheduler
            :param int steps_per_epoch: training steps per one epoch
            :param int warmup_epochs: initial number of epochs of increasing learning rate
            :param int hold_epochs: number of epochs of constant learning rate after warmup
        """
        self.optimizer_type = optimizer_type
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.exp_gamma = exp_gamma
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.hold_epochs = hold_epochs
        self.current_lr = None
        self.max_weight_norm = max_weight_norm if max_weight_norm is not None else np.finfo(np.float16).max

        if optimizer_type == 'SGD':
            self.optimizer_options = {"defaultLearningRate": (base_lr, False)}
            if weight_decay is not None:
                self.optimizer_options["defaultWeightDecay"] = (weight_decay, True)
            if loss_scaling is not None:
                self.optimizer_options["lossScaling"] = (loss_scaling, True)
        elif optimizer_type == 'LAMB':
            self.optimizer_options = {"defaultLearningRate": (base_lr, False),
                                      "defaultBeta1": (beta1, True),
                                      "defaultBeta2": (beta2, True),
                                      "defaultWeightDecay": (weight_decay, True),
                                      "defaultEps": (opt_eps, True),
                                      "lossScaling": (loss_scaling, True),
                                      "defaultMaxWeightNorm": (self.max_weight_norm, True)
                                      }
            self.gradient_clipping_norm = gradient_clipping_norm
        else:
            logger.error("Not a valid optimizer option {}".format(optimizer_type))
            sys.exit(-1)

    def update_and_create(self, step, epoch):
        """ updates the learning rate and returns a new popart optimizer object:
        the learning-rate schedule used here is same as for RNN-T reference model
        """

        new_lr = self.get_new_lr(step, epoch)

        logger.info("Setting learning-rate to {}".format(new_lr))
        self.optimizer_options["defaultLearningRate"] = (new_lr, False)

        if self.optimizer_type == 'SGD':
            optimizer = popart.SGD(self.optimizer_options)
        elif self.optimizer_type == 'LAMB':
            if self.gradient_clipping_norm is None:
                optimizer = popart.Adam(self.optimizer_options, mode=popart.AdamMode.Lamb)
            else:
                optimizer = popart.Adam(self.optimizer_options, mode=popart.AdamMode.Lamb,
                                        clip_norm_settings=[popart.ClipNormSettings.clipAllWeights(self.gradient_clipping_norm)])

        self.current_lr = new_lr

        return optimizer

    def get_new_lr(self, step, epoch):
        """ returns new learning rate based on current step and epoch numbers """

        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        hold_steps = self.hold_epochs * self.steps_per_epoch

        if step < warmup_steps:
            a = (step + 1) / (warmup_steps + 1)
        elif step < warmup_steps + hold_steps:
            a = 1.0
        else:
            a = self.exp_gamma ** (epoch - self.warmup_epochs - self.hold_epochs)

        # get new learning rate
        new_lr = max(a * self.base_lr, self.min_lr)
        return new_lr
