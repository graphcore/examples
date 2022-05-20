# Copyright (c) 2021 Graphcore Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import random
import logging
import os
import json
import numpy as np
import tensorflow as tf

from functools import partial
from tensorflow.python import ipu
from tensorflow import keras
from wandb.keras import WandbCallback
from datetime import datetime
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from fastspeech2 import build_model, build_pipeline_model
from dataloader import LJSpeechCharLevelDataset
from utils import create_ipu_config, ThroughputCallback, ModelCheckpoint, LearningRateLogger, CompilationTimeCallback
from options import make_global_options
from optimizer import AdamWeightDecay, WarmUp


def setup_logger():
    logFormatter = logging.Formatter(
        '%(asctime)s.%(msecs)06d: %(levelname)-1.1s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger("FastSpeech2 Training")
    logger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger


def set_randome_seed(seed=1989):
    # set random seed
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    ipu.utils.reset_ipu_seed(seed)


def set_poplar_engine_options():
    # load the user defined json object
    if os.environ.get('POPLAR_ENGINE_OPTIONS'):
        poplar_engine_options = json.loads(
            os.environ.get('POPLAR_ENGINE_OPTIONS').encode())
    else:
        poplar_engine_options = {}
    # Specifies how the invocation of stream callbacks is parallelised.
    # This would speed up the throughput for multiple replicas/instances.
    streamcallbacks = {
        "streamCallbacks.multiThreadMode": "collaborative",
        "streamCallbacks.numWorkerThreads": "auto"}
    poplar_engine_options.update(streamcallbacks)
    os.environ['POPLAR_ENGINE_OPTIONS'] = json.dumps(poplar_engine_options)


def masked_loss(y_gt, y_pred, loss_fn, **kwargs):
    """Calculate 2d loss by removing mask, normally it's durrations/f0s/energys loss."""
    real_len = tf.reduce_sum(tf.cast(tf.math.not_equal(
        y_gt, 0), tf.float32), axis=1)   # shape [B,]
    max_len = tf.shape(y_gt)[1]
    max_len = tf.cast(max_len, real_len.dtype)

    if len(y_gt.shape) == 3:
        # Mel shape is [B, MelLength, num_mels], every element should be same in last dimension
        real_len = tf.reduce_mean(real_len, axis=-1)
    ratio = max_len / tf.reduce_max(real_len)
    loss = loss_fn(y_gt, y_pred)
    if len(loss.shape) == 2:
        loss = tf.reduce_mean(loss, axis=-1)
    loss = tf.math.multiply(loss, tf.cast(ratio, loss.dtype))
    return loss


def masked_mse_loss(y_gt, y_pred, **kwargs):
    loss_fn = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE)
    loss = masked_loss(y_gt, y_pred, loss_fn, **kwargs)
    return loss


def masked_mae_loss(y_gt, y_pred, **kwargs):
    loss_fn = tf.keras.losses.MeanAbsoluteError(
        reduction=tf.keras.losses.Reduction.NONE)
    loss = masked_loss(y_gt, y_pred, loss_fn, **kwargs)
    return loss


def masked_log_duration_loss(y_gt, y_pred, **kwargs):
    loss_fn = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE)
    log_y_gt = tf.math.log(tf.cast(tf.math.add(y_gt, 1), tf.float32))
    log_y_gt = tf.cast(log_y_gt, y_pred.dtype)
    loss = masked_loss(log_y_gt, y_pred, loss_fn, **kwargs)
    return loss


def get_lr_scheduler(schedule):
    # Using learning rate scheduler
    if schedule == "exponential":
        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=opts["base_learning_rate"],
            decay_steps=opts["decay_steps"],
            decay_rate=opts["decay_rate"],
            staircase=True
        )
    elif schedule == "polynomial":
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=opts["base_learning_rate"],
            decay_steps=opts["decay_steps"],
            end_learning_rate=1e-6
        )
    elif schedule == "cosine":
        learning_rate_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=opts["base_learning_rate"],
            decay_steps=opts["decay_steps"],
            alpha=0.0
        )
    else:
        raise ValueError(f"Schedule {schedule} has not supported.")
    return learning_rate_fn


def get_optimizer(opts):
    if opts["optimizer"].lower() == "sgd":
        optim = tf.keras.optimizers.SGD(
            learning_rate=opts["base_learning_rate"],
            momentum=0.0,
        )
    elif opts["optimizer"].lower() == "adam":
        optim = tf.keras.optimizers.Adam(
            learning_rate=opts["base_learning_rate"],
            beta_1=opts["beta1"],
            beta_2=opts["beta2"],
            epsilon=opts["epsilon"] if opts["precision"] == "16" else 1e-3)
    elif opts["optimizer"].lower() == "adamw":
        lr_schedule = get_lr_scheduler(opts["lr_schedule"])
        lr_schedule = WarmUp(
            initial_learning_rate=opts["base_learning_rate"],
            decay_schedule_fn=lr_schedule,
            warmup_steps=int(
                opts["warmup"]*opts["steps_per_epoch"]*opts["epochs"]),
            power=1.0)

        optim = AdamWeightDecay(
            learning_rate=lr_schedule,
            weight_decay_rate=opts["weight_decay_rate"],
            beta_1=opts["beta1"],
            beta_2=opts["beta2"],
            epsilon=opts["epsilon"] if opts["precision"] == "16" else 1e-3,
            exclude_from_weight_decay=[
                "GroupNorm", "Group_Norm", "LayerNorm", "layer_norm", "bias"],
        )

    else:
        raise NotImplementedError(
            f"Optimizer {opts['optimizer']} not support yet.")

    if opts["precision"] == "16":
        optim = LossScaleOptimizer(optim,
                                   dynamic=False,
                                   initial_scale=int(opts["loss_scaling"]))

    return optim


def setup_loss_dict():
    loss_dict = {
        "f0_predictor": masked_mse_loss,
        "energy_predictor": masked_mse_loss,
        "duration_predictor": masked_log_duration_loss,
        "mel_before": masked_mae_loss,
        "mel_after": masked_mae_loss,
    }
    return loss_dict


def train(model, train_datasets, opts, wandb=None):
    steps_per_epoch = opts["steps_per_epoch"]
    samples_per_epoch = opts["batch_size"] * steps_per_epoch
    callbacks = [
        ThroughputCallback(samples_per_epoch=samples_per_epoch),
        LearningRateLogger(steps_per_epoch=steps_per_epoch)
    ]
    if wandb is not None:
        callbacks.append(WandbCallback())

    if opts["log_dir"]:
        prefix = datetime.now().strftime('%Y%m%d%H%M')[2:]
        ckpt_path = opts["log_dir"] + f"/{prefix}"
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
        ckptfile = ckpt_path + "/epoch{epoch:02d}.h5"
        callbacks.append(ModelCheckpoint(
            epochs_per_save=opts["epochs_per_save"],
            filepath=ckptfile,
            save_best_only=False,
            save_weights_only=True,
            verbose=1))

    history = model.fit(
        train_datasets(),
        epochs=opts["epochs"],
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        workers=64,
        use_multiprocessing=True,
        verbose=2)

    if wandb is not None:
        for i in range(1, opts["epochs"]+1):
            wandb.log({
                "epochs": i,
                "loss_train": history.history["loss"][i-1],
            })
    return history


def evaluation(model, valid_datasets, opts, ckpt_path, wandb=None):
    samples_per_epoch = opts["batch_size"] * opts["steps_per_epoch"]
    callbacks = [
        ThroughputCallback(samples_per_epoch=samples_per_epoch)
    ]
    if wandb is not None:
        callbacks.append(WandbCallback())

    for i in range(opts["epochs"]):
        if i % opts["epochs_per_save"] == 0:
            print(f"Evaluation on epoch {i+1}/{opts['epochs']}...")
            cpath = ckpt_path + f"/epoch{i+1:02d}.h5"
            if i == opts["epochs"] - 1:
                cpath = ckpt_path + f"/model.h5"
            model.load_weights(cpath)
            print(f"{model.metrics_names}")
            eval_results = model.evaluate(
                valid_datasets(), steps=opts["steps_per_epoch"], callbacks=callbacks)
            for name, res in zip(model.metrics_names, eval_results):
                print(f"{name} = {res}")
            if wandb is not None:
                wandb.log({
                    "epochs": i,
                    "loss_eval": eval_results[0].mean()
                })
    return eval_results


def init_wandb(opts):
    # apply wandb or not
    if opts["wandb"]:
        import wandb
        wandb.init(project=opts["wandb_name"],
                   dir=opts["log_dir"], sync_tensorboard=True)
        wandb.config.update(opts)
        wandb.run.name = f"[FP{opts['precision']}]bs{opts['batch_size']}_ga{opts['gradient_accumulation_count']}_r{opts['replicas']}_spe{opts['steps_per_epoch']}"
        return wandb
    return None


def run_model(opts, use_pipeline_model=True):
    wandb = init_wandb(opts)
    set_randome_seed(int(opts["seed"]))
    set_poplar_engine_options()
    logger = setup_logger()
    data_type = tf.float16 if opts["precision"] == "16" else tf.float32
    if opts["precision"] == "16":
        policy = tf.keras.mixed_precision.Policy("float16")
        tf.keras.mixed_precision.set_global_policy(policy)
    num_ipus_per_replica = 2
    num_ipus = num_ipus_per_replica * int(opts["replicas"])
    assert num_ipus & (
        num_ipus-1) == 0, f"You‘re trying to apply {num_ipus} IPUs, but we only support to apply the power of 2 IPUs."
    logger.info(f"Options: {opts}")
    # Set up the IPU system.
    cfg = create_ipu_config(
        available_memory_proportion=opts["available_memory_proportion"],
        num_required_ipus=num_ipus,
        partials_type=opts["partials_type"],
        fp_exceptions=opts["fp_exceptions"],
        enable_stochastic_rounding=opts["stochastic_rounding"],
        num_io_tiles=opts["num_io_tiles"])

    train_datasets = LJSpeechCharLevelDataset(opts, is_train=True)
    val_datasets = LJSpeechCharLevelDataset(opts, is_train=False)

    pipeline_schedule = ipu.pipelining_ops.PipelineSchedule.Grouped
    if opts["pipeline_schedule"] == "Interleaved":
        pipeline_schedule = ipu.pipelining_ops.PipelineSchedule.Interleaved

    optim = get_optimizer(opts)
    loss_dict = setup_loss_dict()
    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
        if num_ipus == 1:
            fastspeech2 = tf.keras.Model(*build_model(opts, training=True))
        else:
            fastspeech2 = keras.Model(
                *build_pipeline_model(opts, training=True))
            fastspeech2.set_pipelining_options(
                gradient_accumulation_steps_per_replica=int(
                    opts["gradient_accumulation_count"]),
                recomputation_mode=ipu.ops.pipelining_ops.RecomputationMode.Auto,
                pipeline_schedule=pipeline_schedule,
                offload_weight_update_variables=opts["variable_offloading"],
                device_mapping=[0, 1],
            )
            # Set the infeed and outfeed options.
            fastspeech2.set_infeed_queue_options(prefetch_depth=2)
            fastspeech2.set_outfeed_queue_options(buffer_depth=2)
            fastspeech2.print_pipeline_stage_assignment_summary()

        fastspeech2.compile(optimizer=optim,
                            loss=loss_dict,
                            steps_per_execution=opts["steps_per_epoch"]
                            )
        fastspeech2.summary()

        if opts["train"]:
            train_start_time = time.time()
            history = train(fastspeech2, train_datasets=train_datasets,
                            opts=opts, wandb=wandb)
            training_time = time.time() - train_start_time
            logger.info(f"[Duration: {training_time:.2f}s]Training finish.")
        if opts["eval"]:
            logger.info("Start to evaluate...")
            eval_res = evaluation(fastspeech2, valid_datasets=val_datasets, opts=opts,
                                  ckpt_path=opts["init_checkpoint"], wandb=wandb)


if __name__ == "__main__":
    opts = make_global_options([])
    run_model(opts, use_pipeline_model=True)
