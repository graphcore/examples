# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import time
from typing import Optional

import tensorflow as tf
import wandb
from absl import logging
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import metrics as metrics_mod
import random

import numpy as np
from tensorflow.python import ipu
import xpu


class ThroughputCallback(keras.callbacks.Callback):
    def __init__(self, samples_per_epoch, log_wandb=True):
        super().__init__()
        self.time = 0
        self.samples_per_epoch = samples_per_epoch
        self.log_wandb = log_wandb

    def on_epoch_begin(self, epoch, logs=None):
        self.time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        time_per_epoch = time.time() - self.time
        samples_per_sec = self.samples_per_epoch / time_per_epoch
        if self.log_wandb:
            wandb_logs = {"epoch": epoch, "throughput": samples_per_sec}
            wandb_logs.update(logs)
            wandb.log(wandb_logs)

        logging.info(f"\nThroughput: {samples_per_sec:.2f} graphs/sec")


def get_optimizer(
    name="adam",
    learning_rate=1e-5,
    l2_regularization=None,
    dtype="float32",
    m_dtype=None,
    v_dtype=None,
    clip_value=None,
    loss_scale=1,
    gradient_accumulation_factor=1,
    outline_apply_gradients=False,
    replicas=1,
):
    def clip_gradients(grads_and_vars):
        return [(tf.clip_by_norm(g, clip_value), v) for g, v in grads_and_vars]

    if clip_value:
        # i think the gradient accumulation factor is incorrect in rescale_gradients.
        # just changing it for the gradient clipping case to ensure i dont break everything else for now
        def rescale_gradients(grads_and_vars):
            return [(g / replicas, v) for g, v in grads_and_vars]

        gradient_transformer = lambda grads_and_vars: clip_gradients(rescale_gradients(grads_and_vars))
    else:

        def rescale_gradients(grads_and_vars):
            return [(g / (gradient_accumulation_factor * replicas), v) for g, v in grads_and_vars]

        gradient_transformer = rescale_gradients

    if name == "sgd":
        opt_class = add_l2_regularization(tf.keras.optimizers.SGD, l2_regularization)
        opt = opt_class(learning_rate=learning_rate, gradient_transformers=[gradient_transformer])
    elif name == "tf_adam":
        opt_class = add_l2_regularization(tf.keras.optimizers.Adam, l2_regularization)
        opt = opt_class(learning_rate=learning_rate, gradient_transformers=[gradient_transformer])
    elif name == "adam":
        opt_class = add_l2_regularization(xpu.AdamIpuOptimizer, l2_regularization)
        opt = opt_class(
            learning_rate=learning_rate,
            gradient_transformers=[gradient_transformer],
            m_dtype=m_dtype,
            v_dtype=v_dtype,
            outline_apply_gradients=outline_apply_gradients,
        )
    else:
        raise NotImplementedError(f"Optimizer {name} is not supported.")

    opt = keras.mixed_precision.LossScaleOptimizer(opt, dynamic=False, initial_scale=loss_scale)

    return opt


def add_l2_regularization(optimizer_class, l2_regularization):
    if not l2_regularization:
        return optimizer_class

    class L2Regularizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(L2Regularizer, self).__init__(*args, **kwargs)
            self.l2_regularization = l2_regularization

        def _resource_apply_dense(self, grad, var, apply_state):
            return super()._resource_apply_dense(grad + var * self.l2_regularization, var, apply_state)

    return L2Regularizer


def size_hr(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def print_trainable_variables(model, log_wandb=False):
    logging.info("Trainable Variables:")
    total_parameters = 0
    total_size = 0
    for variable in model.trainable_variables:
        variable_parameters = 1
        for DIM in variable.shape:
            variable_parameters *= DIM
        variable_size = variable_parameters * variable.dtype.size
        logging.info(f"{variable.name}, {variable.shape}, {variable.dtype} ({size_hr(variable_size)})")
        total_parameters += variable_parameters
        total_size += variable_size
    logging.info(f"Total Parameters: {total_parameters:,}  ({size_hr(total_size)})")
    if log_wandb:
        wandb.log({"parameters": total_parameters, "parameter size (MiB)": total_size / (1024**2)})


def str_dtype_to_tf_dtype(str_dtype):
    return {"float16": tf.float16, "mixed_float16": tf.float32, "float32": tf.float32}[str_dtype]


def convert_loss_and_metric_reductions_to_fp32(model):
    model.compiled_loss._loss_metric = metrics_mod.Mean(name="loss", dtype=tf.float32)
    model.compiled_loss._per_output_metrics = [
        metrics_mod.Mean(name=n + "_loss", dtype=tf.float32) for n in model.compiled_loss._output_names
    ]
    metrics = model.compiled_metrics._weighted_metrics
    for k in metrics.keys():
        if not isinstance(metrics[k], list):
            metric_obj = metrics_mod.get(metrics[k])
            if not isinstance(metrics[k], metrics_mod.MeanMetricWrapper):
                metrics[k] = metrics_mod.MeanMetricWrapper(metric_obj, dtype=tf.float32, name=f"{metric_obj.__name__}")
            continue
        new_metrics_list = []
        for m in metrics[k]:
            metric_obj = metrics_mod.get(m)
            if isinstance(metric_obj, metrics_mod.MeanMetricWrapper):
                assert metric_obj.dtype == "float32", "metrics must accumulate in float32"
                new_metrics_list += [metric_obj]
            else:
                new_metrics_list += [
                    metrics_mod.MeanMetricWrapper(metric_obj, dtype=tf.float32, name=f"{metric_obj.__name__}")
                ]
        metrics[k] = new_metrics_list
    model.compiled_metrics._weighted_metrics = metrics


def options_validator(cfg):
    """Validate the option combinations that depend on each other.
    NOTE: This is WIP and not exhaustive.

    Args:
        cfg : config object from jsonargparse
    """
    # Edges + Noisy Edges
    if cfg.model.use_noisy_edges is True and cfg.model.use_edges is False:
        raise ValueError(f"Cannot use noisy edges without use edges set to True.")

    if cfg.upload_final_ckpt is True and cfg.wandb is False:
        logging.warning(
            f"`cfg.upload_final_ckpt` is {cfg.upload_final_ckpt} but `cfg.wandb` is {cfg.wandb}."
            " Can't upload checkpoint without wandb activated."
        )


def set_random_seeds(seed=42):
    ipu.utils.reset_ipu_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *args, **kwargs):
        elapsed_time = time.time() - self.t
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        # Timedelta makes for human readable format - not safe for maths operations
        logging.info(f"\U0001F551 Elapsed time step for {self.name}: {elapsed_time} (HH:MM:SS)")
