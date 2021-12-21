# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Optional
import tensorflow as tf
import tensorflow_addons as tfa
from optimizers.l2_regularizer import add_l2_regularization
from optimizers.lars_optimizer import LARSIpuOptimizer
from normalization import batch_norm
from batch_config import BatchConfig
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import popdist

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utilities import verify_all_params_present

AVAILABLE_OPTIMIZERS = ['sgd', 'lars']


class OptimizerFactory:

    @staticmethod
    def get_optimizer(optimizer_name: str,
                      optimizer_params: dict,
                      loss_scaling: Optional[float],
                      l2_regularization: float,
                      bn_momentum: float,
                      batch_config: BatchConfig,
                      lr_scheduler: LearningRateSchedule,
                      wd_scheduler: LearningRateSchedule,
                      distributed_training: bool):

        if optimizer_name not in AVAILABLE_OPTIMIZERS:
            raise NameError(f'Optimizer {optimizer_name} not supported. Supported optimizers: {AVAILABLE_OPTIMIZERS}')

        optimizer_params = optimizer_params.copy()

        if optimizer_name == 'sgd':
            expected_params = ['momentum']
            verify_all_params_present(list(optimizer_params.keys()), expected_params, optimizer_name, '--optimizer-params')
            optimizer_class = tfa.optimizers.SGDW
            optimizer_params['weight_decay'] = wd_scheduler

        elif optimizer_name == 'lars':
            expected_params = ['momentum', 'weight_decay', 'eeta', 'epsilon']
            verify_all_params_present(list(optimizer_params.keys()), expected_params, optimizer_name, '--optimizer-params')
            optimizer_class = LARSIpuOptimizer
            optimizer_params['exclude_from_layer_adaptation'] = ['beta', 'gamma', 'bias']

        if distributed_training:
            def gradient_normalizer(grads_and_vars): return \
                [(grad / batch_config.gradient_accumulation_count, var)
                    for grad, var in grads_and_vars]
        else:
            def gradient_normalizer(grads_and_vars): return \
                [(grad / popdist.getNumTotalReplicas() / batch_config.gradient_accumulation_count, var)
                    for grad, var in grads_and_vars]

        optimizer_params['learning_rate'] = lr_scheduler
        optimizer_params['gradient_transformers'] = [gradient_normalizer]
        if l2_regularization:
            optimizer_class = add_l2_regularization(optimizer_class, l2_regularization)
        optimizer_class = batch_norm.add_bn_moving_vars_updates_to_optimizer(optimizer_class,
                                                                             bn_momentum=bn_momentum,
                                                                             loss_scaling=loss_scaling)
        optimizer = optimizer_class(**optimizer_params)
        if loss_scaling:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer,
                                                                    dynamic=False,
                                                                    initial_scale=loss_scaling)

        return optimizer
