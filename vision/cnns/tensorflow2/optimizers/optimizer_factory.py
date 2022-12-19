# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
from batch_config import BatchConfig
from normalization import batch_norm
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import keras
from keras.ipu.optimizers import ALSOptimizer

from schedules.decorators import FP32StepLearningRateSchedule
from optimizers.l2_regularizer import add_l2_regularization
from optimizers.lars_optimizer import LARSIpuOptimizer
from optimizers.loss_scale_optimizer import add_loss_scaling_to_optimizer

sys.path.append(str(Path(__file__).absolute().parent.parent))
from utilities import verify_params_present
from keras.optimizer_v2 import gradient_descent
AVAILABLE_OPTIMIZERS = ['sgd', 'lars', 'rmsprop']


def fp32_step_retriever_to_optimizer(optimizer_class, weight_decay: bool):
    class FP32StepRetriever(optimizer_class):
        @property
        def weights(self):
            """
            Returns variables of this Optimizer based on the order created.
            Includes FP32 step variables from learning weight and weight decay schedulers.
            """
            weights = self._weights + [self.lr.step]
            if weight_decay:
                weights += [self.weight_decay.step]
            return weights

    return FP32StepRetriever


class OptimizerFactory:

    @staticmethod
    def get_optimizer(optimizer_name: str,
                      optimizer_params: dict,
                      loss_scaling: Optional[float],
                      auto_loss_scaling: Optional[bool],
                      l2_regularization: float,
                      batch_config: BatchConfig,
                      lr_scheduler: LearningRateSchedule,
                      wd_scheduler: LearningRateSchedule,
                      distributed_training: bool,
                      norm_layer_params: dict,
                      model: keras.Model):

        if optimizer_name not in AVAILABLE_OPTIMIZERS:
            raise NameError(f'Optimizer {optimizer_name} not supported. Supported optimizers: {AVAILABLE_OPTIMIZERS}')

        optimizer_params = optimizer_params.copy()

        if optimizer_name == 'sgd':
            expected_params = ['momentum']
            verify_params_present(list(optimizer_params.keys()), expected_params, optimizer_name, '--optimizer-params')
            optimizer_class = tfa.optimizers.SGDW
            optimizer_params['weight_decay'] = wd_scheduler

        elif optimizer_name == 'lars':
            expected_params = ['momentum', 'weight_decay', 'eeta', 'epsilon']
            verify_params_present(list(optimizer_params.keys()), expected_params, optimizer_name, '--optimizer-params')
            optimizer_class = LARSIpuOptimizer
            optimizer_params['exclude_from_layer_adaptation'] = ['beta', 'gamma', 'bias']

        elif optimizer_name == 'rmsprop':
            expected_params = ['momentum', 'base_decay_exponent', 'epsilon']
            verify_params_present(list(optimizer_params.keys()), expected_params, optimizer_name, '--optimizer-params')
            optimizer_class = RMSprop
            optimizer_params['rho'] = (
                1 - ((2 ** optimizer_params['base_decay_exponent']) * batch_config.global_batch_size))
            del optimizer_params['base_decay_exponent']

        if auto_loss_scaling:
            optimizer_class = gradient_descent.SGD
            optimizer_params = {}

        if not distributed_training and (not auto_loss_scaling or batch_config.gradient_accumulation_count > 1):
            def gradient_normalizer(grads_and_vars):
                return [(grad / batch_config.num_replicas, var)
                        for grad, var in grads_and_vars]
            optimizer_params['gradient_transformers'] = [gradient_normalizer]

        if distributed_training and auto_loss_scaling and not batch_config.gradient_accumulation_count > 1:
            def gradient_multiplier(grads_and_vars):
                return [(grad * batch_config.num_replicas, var)
                        for grad, var in grads_and_vars]
            optimizer_params['gradient_transformers'] = [gradient_multiplier]

        optimizer_params['learning_rate'] = lr_scheduler
        if l2_regularization:
            optimizer_class = add_l2_regularization(optimizer_class, l2_regularization)

        if loss_scaling and not auto_loss_scaling:
            optimizer_class = add_loss_scaling_to_optimizer(optimizer_class, loss_scaling * batch_config.num_replicas)

        if norm_layer_params['name'] == 'custom_batch_norm':
            optimizer_class = batch_norm.add_bn_moving_vars_updates_to_optimizer(optimizer_class,
                                                                                 model=model,
                                                                                 batch_config=batch_config,
                                                                                 bn_momentum=norm_layer_params['momentum'])

        if isinstance(lr_scheduler, FP32StepLearningRateSchedule):
            # When FP32StepLearningRateSchedule is used make sure that step variables are
            # included in the self.weights property of the optimiser.
            optimizer_class = fp32_step_retriever_to_optimizer(
                optimizer_class=optimizer_class,
                weight_decay='weight_decay' in optimizer_params and isinstance(optimizer_params['weight_decay'], FP32StepLearningRateSchedule)
            )

        optimizer = optimizer_class(**optimizer_params)
        
        if auto_loss_scaling:
            optimizer = ALSOptimizer(optimizer)

        return optimizer
