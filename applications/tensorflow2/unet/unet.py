# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu
from time import perf_counter

from model import model_fn
from model_utils import set_pipeline_options
from utils import configure_ipu, PerfCallback
from losses import dice_coef_accuracy_fn, dice_ce_loss, ce_loss


logger = logging.getLogger(__name__)


def get_optimizer(args):
    def gradient_normalizer(grads_and_vars):
        return [(grad / args.replicas / args.gradient_accumulation_count, var) for grad, var in grads_and_vars]

    if args.optimizer == "adam":
        optimizer_instance = keras.optimizers.Adam(
            learning_rate=args.learning_rate, epsilon=1e-4, gradient_transformers=[gradient_normalizer])
    else:
        # Create learning rate schedule
        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
            args.learning_rate,
            decay_steps=args.num_epochs,
            decay_rate=args.decay_rate,
            staircase=False)

        optimizer_instance = keras.optimizers.SGD(
            learning_rate=learning_rate_fn, momentum=args.momentum, gradient_transformers=[gradient_normalizer])

    # Use loss scaling for FP16
    if args.dtype == 'float16':
        optimizer_instance = tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer_instance, False, args.loss_scale)
    return optimizer_instance


def create_model(args):
    model = keras.Model(*model_fn(args))
    if args.nb_ipus_per_replica > 1:
        set_pipeline_options(model, args)
        model.print_pipeline_stage_assignment_summary()
    elif args.nb_ipus_per_replica == 1:
        model.set_gradient_accumulation_options(gradient_accumulation_steps_per_replica=args.gradient_accumulation_count,
                                                offload_weight_update_variables=False)
    model.compile(optimizer=get_optimizer(args), loss=dice_ce_loss,
                  # Number of micro batches to process sequentially in a single execution
                  steps_per_execution=args.steps_per_execution if args.nb_ipus_per_replica > 0 else None,
                  metrics=[dice_coef_accuracy_fn, ce_loss])

    return model


def train_model(args, model, ds_train, ds_eval):
    callbacks = []

    # Record throughput
    callbacks.append(PerfCallback(
        steps_per_execution=args.steps_per_execution, batch_size=args.micro_batch_size))
    eval_accuracy = None
    eval_loss = None
    if args.nb_ipus_per_replica <= 1:
        executions = args.num_epochs
    else:
        executions = int(args.gradient_accumulation_count *
                         args.num_epochs / args.steps_per_execution)
        assert executions > 0, \
            f"gradient accumulation count * nb of executions " \
            f"({args.gradient_accumulation_count * args.num_executions}) " \
            f"needs to be at least the nb of steps per execution ({args.steps_per_execution})"

    additional_args = {}
    if args.eval:
        callbacks.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.model_dir, 'checkpoints'),
                                                         monitor='val_dice_coef_accuracy_fn',
                                                         save_best_only=True,
                                                         save_weights_only=True))
        if args.eval_freq > executions:
            logger.warning(
                f"The number of executions in model.fit ({executions}) needs to be at least the validation frequency ({args.eval_freq}).")
            args.eval_freq = min(args.eval_freq, executions)

        additional_args = {"validation_data": ds_eval,
                           "validation_steps": args.gradient_accumulation_count,
                           "validation_freq": args.eval_freq}

    elif not args.benchmark:
        callbacks.append(keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.model_dir, 'checkpoints'),
                                                         monitor='dice_coef_accuracy_fn',
                                                         save_best_only=True,
                                                         save_weights_only=True))

    train_result = model.fit(ds_train,
                             steps_per_epoch=args.steps_per_execution,
                             epochs=executions,
                             callbacks=callbacks,
                             **additional_args)
    if args.eval:
        eval_accuracy = train_result.history['val_dice_coef_accuracy_fn']
        eval_loss = train_result.history['val_loss']
    return eval_accuracy, eval_loss


def infer_model(args, model, ds_infer):
    if args.benchmark:
        # Warmup
        model.predict(ds_infer, steps=args.steps_per_execution)

        t0 = perf_counter()
        model.predict(ds_infer, steps=args.steps_per_execution)
        t1 = perf_counter()
        duration = t1 - t0
        total_nb_samples = args.steps_per_execution * \
            args.micro_batch_size * args.replicas
        tput = f"{total_nb_samples / duration:0.15f}"
        logger.info(
            f'Inference\t Time: {duration} seconds\t Throughput {tput} images/sec.')
    else:
        if args.model_dir:
            model.load_weights(os.path.join(
                args.model_dir, 'checkpoints')).expect_partial()
        predictions = model.predict(ds_infer, steps=args.steps_per_execution)
        binary_masks = [
            np.argmax(p, axis=-1).astype(np.uint8) * 255 for p in predictions]
        prediction_tif = [Image.fromarray(mask).resize(size=(512, 512), resample=Image.BILINEAR)
                          for mask in binary_masks]
        output_dir = os.path.join(args.model_dir, 'predictions')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        prediction_tif[0].save(os.path.join(output_dir, 'test-masks.tif'),
                               compression="tiff_deflate",
                               save_all=True,
                               append_images=prediction_tif[1:])

        logger.info(f"Predictions saved at {output_dir}.")


def get_strategy(args):
    if args.nb_ipus_per_replica > 0:
        logger.info("On IPU...")
        # Create an IPU distribution strategy
        strategy = ipu.ipu_strategy.IPUStrategy()
    else:
        logger.info("On CPU...")
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    return strategy


def unet(args, ds_train, ds_eval, ds_infer):
    tf.keras.backend.clear_session()
    eval_accuracy = None
    eval_loss = None

    if args.nb_ipus_per_replica > 0:
        configure_ipu(args)

    strategy = get_strategy(args)
    with strategy.scope():
        model = create_model(args)
        model.summary()
        if args.train:
            logger.info("Training model...")
            eval_accuracy, eval_loss = train_model(
                args, model, ds_train, ds_eval)
            logger.info("Training complete")

        if args.infer:
            logger.info("Start inference...")
            infer_model(args, model, ds_infer)
            logger.info("Inference complete")
        return eval_accuracy, eval_loss
