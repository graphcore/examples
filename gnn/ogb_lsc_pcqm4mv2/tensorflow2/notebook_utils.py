# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import logging
import math
import os
import random
import shutil
import string
import tempfile
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.python.ipu import optimizers
import wandb

import utils
import xpu
from custom_callbacks import CheckpointCallback
from data_utils.input_spec import create_inputs_from_features
from model.utils import create_model, get_loss_functions, get_metrics, get_tf_dataset
from pipeline.pipeline_stage_assignment import pipeline_model
from pipeline.pipeline_stage_names import PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES
from utils import (
    ThroughputCallback,
    convert_loss_and_metric_reductions_to_fp32,
    get_optimizer,
    str_dtype_to_tf_dtype,
    set_random_seeds,
)


def predict(preprocessed_dataset, checkpoint_path, fold, cfg):
    """
    Function to run inference and give predictions of molecule samples.
    Args:
        preprocessed_dataset: dataset object, preprocessed dataset with all the features needed for inference.
        checkpoint_path: string, path to the checkpoint.
        fold: string, name of the fold, choose from ["valid", "test-dev" and "test-challenge"].
        cfg: config object, configurations of the model and inference.
    """
    set_random_seeds()
    tf.keras.mixed_precision.set_global_policy(cfg.model.dtype)

    print("Creating input specification...")
    input_spec = create_inputs_from_features(dataset=preprocessed_dataset, cfg=cfg, fold="test-dev")

    losses, loss_weights = get_loss_functions(preprocessed_dataset, cfg)
    metrics = get_metrics(preprocessed_dataset.denormalize, cfg)
    optimizer_options = dict(
        name=cfg.model.opt.lower(),
        learning_rate=cfg.model.lr,
        l2_regularization=cfg.model.l2_regularization,
        dtype=str_dtype_to_tf_dtype(cfg.model.dtype),
        m_dtype=str_dtype_to_tf_dtype(cfg.model.adam_m_dtype),
        v_dtype=str_dtype_to_tf_dtype(cfg.model.adam_v_dtype),
        clip_value=cfg.model.grad_clip_value,
        loss_scale=cfg.model.loss_scaling,
        gradient_accumulation_factor=cfg.ipu_opts.gradient_accumulation_factor,
        replicas=cfg.ipu_opts.replicas,
        outline_apply_gradients=not cfg.ipu_opts.offload_optimizer_state,  # bug where outlining causes issues
    )

    print("Configuring the IPUs...")
    strategy = xpu.configure_and_get_strategy(
        num_replicas=1, num_ipus_per_replica=1, stochastic_rounding=False, cfg=cfg
    )

    with strategy.scope():
        print("Creating TensorFlow dataset from preprocessed dataset...")
        batch_generator, ground_truth_and_masks = get_tf_dataset(
            preprocessed_dataset=preprocessed_dataset,
            split_name=fold,
            shuffle=False,
            options=cfg,
            input_spec=input_spec,
        )
        ds = batch_generator.get_tf_dataset()
        ground_truth, include_mask = ground_truth_and_masks
        ground_truth = ground_truth[include_mask]

        print("Constructing the model...")
        model = create_model(batch_generator, preprocessed_dataset, cfg, input_spec=input_spec)
        model.compile(
            optimizer=get_optimizer(**optimizer_options),
            loss=losses,
            loss_weights=loss_weights,
            weighted_metrics=metrics,
            steps_per_execution=batch_generator.batches_per_epoch,
        )
        if cfg.model.dtype == "float16":
            # the loss reduction is set by backend.floatx by default
            # must be forced to reduce in float32 to avoid overflow
            convert_loss_and_metric_reductions_to_fp32(model)

        if checkpoint_path is not None:
            print(f"Loading the checkpoint from {checkpoint_path}...")
            model.load_weights(checkpoint_path).expect_partial()

        print(f"Running `model.predict` to generate predictions...")

        prediction = model.predict(ds, steps=batch_generator.batches_per_epoch)

        if isinstance(prediction, list) and len(prediction) > 1:
            prediction = prediction[0]
        prediction = prediction.squeeze()

        prediction = preprocessed_dataset.denormalize(prediction)

        if len(include_mask) > len(prediction):
            include_mask = include_mask[: len(prediction)]
            ground_truth = ground_truth[: len(prediction)]

        if len(include_mask) > 1:
            include_mask = include_mask.squeeze()

        prediction = prediction[: len(include_mask)][include_mask == 1]

    return prediction, ground_truth


def train(preprocessed_dataset, cfg):
    """
    Utility function to run training on the PCQM4Mv2 dataset from notebook example
    Args:
        preprocessed_dataset: dataset object, preprocessed dataset with all the features needed for inference.
        cfg: config object, configurations of the model and inference.
    """

    def cosine_lr(epoch, lr):
        completed_fraction = epoch / cfg.model.epochs
        cosine = tf.cos(tf.constant(math.pi, dtype=tf.float32) * completed_fraction)
        cosine_decayed = 0.5 * (1.0 + cosine)
        lr = max(cosine_decayed * cfg.model.lr, cfg.model.min_lr)
        if cfg.model.lr_warmup_epochs > 0 and epoch < cfg.model.lr_warmup_epochs:
            init_prop = cfg.model.lr_init_prop
            w_ep = cfg.model.lr_warmup_epochs
            lr_scale = (init_prop * (w_ep - epoch) / w_ep) + (epoch / w_ep)
            lr *= lr_scale
        return lr

    def linear_lr(epoch, lr):
        warmup = cfg.model.lr_warmup_epochs
        if warmup > 0 and epoch < warmup:
            lr = epoch * cfg.model.lr / warmup
        else:
            p = (epoch - warmup) / (cfg.model.epochs - warmup)
            lr = p * cfg.model.min_lr + (1 - p) * cfg.model.lr
        return lr

    set_random_seeds()
    tf.keras.mixed_precision.set_global_policy(cfg.model.dtype)
    # ------------ CHECKPOINTS ---------------
    # Create a directory for the checkpoints based on model name and time
    # Want to check directory exists before lengthy data processing
    if cfg.save_checkpoints_locally:
        if not Path(cfg.checkpoint_dir).is_dir():
            os.mkdir(Path(cfg.checkpoint_dir))
        assert Path(cfg.checkpoint_dir).is_dir()
        date_string = (
            "model-"
            + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            + "-"
            + "".join(random.choice(string.ascii_lowercase) for _ in range(10))
        )
        model_dir = os.path.join(cfg.checkpoint_dir, date_string)
    else:
        # If no checkpoint specified temp path is given
        model_dir = tempfile.mkdtemp()
    logging.info(f"Model weights will be saved to {model_dir}")

    graph_data = preprocessed_dataset
    input_spec = create_inputs_from_features(dataset=graph_data, cfg=cfg, fold="train")

    batch_generator, _ = get_tf_dataset(
        preprocessed_dataset=graph_data, split_name="train", shuffle=True, options=cfg, input_spec=input_spec
    )

    if cfg.model.target_total_batch_size:
        if cfg.model.micro_batch_size and (not cfg.ipu_opts.gradient_accumulation_factor):
            fixed_batch_size = (
                float(cfg.model.micro_batch_size)
                * float(cfg.ipu_opts.replicas)
                * batch_generator.stats["avg_pack"]["graphs"]
            )
            new_GA = max(int(round(cfg.model.target_total_batch_size / fixed_batch_size)), 1)
            logging.info(f"Setting gradient_accumulation_factor to {new_GA}")
            cfg.ipu_opts.gradient_accumulation_factor = new_GA
        else:
            fixed_batch_size = (
                float(cfg.ipu_opts.gradient_accumulation_factor)
                * float(cfg.ipu_opts.replicas)
                * batch_generator.stats["avg_pack"]["graphs"]
            )
            new_micro_batch_size = int(round(cfg.model.target_total_batch_size / fixed_batch_size))
            new_micro_batch_size = max(new_micro_batch_size, 1)
            logging.info(f"Changing micro_batch_size from {cfg.model.micro_batch_size} to {new_micro_batch_size}")
            cfg.model.micro_batch_size = new_micro_batch_size
            batch_generator.change_micro_batch_size(cfg.model.micro_batch_size)

        batch_generator.get_averaged_global_batch_size(
            cfg.model.micro_batch_size, cfg.ipu_opts.gradient_accumulation_factor, cfg.ipu_opts.replicas
        )
        logging.info(f"Dataset stats: {batch_generator.stats}")

    if cfg.ipu_opts.num_pipeline_stages > 1:
        round_to = float(2 * cfg.ipu_opts.num_pipeline_stages)
        new_GA = int(max(round(cfg.ipu_opts.gradient_accumulation_factor / round_to), 1) * round_to)
        if new_GA != cfg.ipu_opts.gradient_accumulation_factor:
            logging.info(f"Rounding gradient_accumulation_factor to a multiple of {round_to}: {new_GA}")
            cfg.ipu_opts.gradient_accumulation_factor = new_GA
            batch_generator.get_averaged_global_batch_size(
                cfg.model.micro_batch_size, cfg.ipu_opts.gradient_accumulation_factor, cfg.ipu_opts.replicas
            )
            logging.info(f"Dataset stats: {batch_generator.stats}")

    if cfg.wandb:
        try:
            wandb.log({"dataset_stats": batch_generator.stats})
        except AttributeError:
            logging.info(f"Batch generator {batch_generator} has no stats object.")
    tf_dataset = batch_generator.get_tf_dataset()
    steps_per_epoch = batch_generator.batches_per_epoch
    steps_per_execution_per_replica = steps_per_epoch // cfg.ipu_opts.replicas
    steps_per_execution_per_replica = cfg.ipu_opts.gradient_accumulation_factor * (
        steps_per_execution_per_replica // cfg.ipu_opts.gradient_accumulation_factor
    )
    new_steps_per_epoch = steps_per_execution_per_replica * cfg.ipu_opts.replicas
    if new_steps_per_epoch != steps_per_epoch:
        logging.warning(
            "Steps per epoch has been truncated from"
            f" {steps_per_epoch} to {new_steps_per_epoch}"
            " in order for it to be divisible by steps per execution."
        )
    steps_per_epoch = new_steps_per_epoch
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Steps per execution per replica: {steps_per_execution_per_replica}")

    optimizer_options = dict(
        name=cfg.model.opt.lower(),
        learning_rate=cfg.model.lr,
        l2_regularization=cfg.model.l2_regularization,
        dtype=str_dtype_to_tf_dtype(cfg.model.dtype),
        m_dtype=str_dtype_to_tf_dtype(cfg.model.adam_m_dtype),
        v_dtype=str_dtype_to_tf_dtype(cfg.model.adam_v_dtype),
        clip_value=cfg.model.grad_clip_value,
        loss_scale=cfg.model.loss_scaling,
        gradient_accumulation_factor=cfg.ipu_opts.gradient_accumulation_factor,
        replicas=cfg.ipu_opts.replicas,
        outline_apply_gradients=not cfg.ipu_opts.offload_optimizer_state,
    )

    # ------------ TRAINING LOOP ---------------
    if cfg.do_training:
        num_pipeline_stages_training = cfg.ipu_opts.num_pipeline_stages
        strategy_training = xpu.configure_and_get_strategy(
            num_replicas=cfg.ipu_opts.replicas, num_ipus_per_replica=num_pipeline_stages_training, cfg=cfg
        )
        with strategy_training.scope():
            model = create_model(batch_generator, graph_data, cfg, input_spec=input_spec)
            model.summary()
            utils.print_trainable_variables(model, log_wandb=cfg.wandb)
            if num_pipeline_stages_training > 1:
                pipeline_model(
                    model=model,
                    config=cfg,
                    pipeline_names=PIPELINE_NAMES,
                    pipeline_allocate_previous=PIPELINE_ALLOCATE_PREVIOUS,
                    num_pipeline_stages=num_pipeline_stages_training,
                    matmul_partials_type="half",
                )

            losses, loss_weights = get_loss_functions(graph_data, cfg)
            metrics = get_metrics(graph_data.denormalize, cfg)

            callbacks = [
                ThroughputCallback(
                    # the throughput depends on the COMPUTE batch size, not the TOTAL batch size
                    samples_per_epoch=batch_generator.n_graphs_per_epoch,
                    log_wandb=cfg.wandb,
                )
            ]
            if cfg.model.learning_rate_schedule == "cosine":
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(cosine_lr))
            elif cfg.model.learning_rate_schedule == "linear":
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(linear_lr))
            if cfg.execution_profile:
                callbacks.append(tf.keras.callbacks.TensorBoard(profile_batch=[2], log_dir="logs"))

            logging.info("Running training...")
            logging.info(f"Saving weights to {model_dir}")
            model_path = os.path.join(model_dir, "model-{epoch:05d}")
            callbacks.append(
                CheckpointCallback(
                    use_wandb=cfg.wandb,
                    upload_to_wandb=cfg.upload_final_ckpt,
                    save_checkpoints_locally=cfg.save_checkpoints_locally,
                    filepath=model_path,
                    verbose=1,
                    save_best_only=False,
                    save_weights_only=True,
                    period=cfg.checkpoint_every_n_epochs,
                    total_epochs=cfg.model.epochs,
                    model_name="model",
                )
            )

            model.compile(
                optimizer=get_optimizer(**optimizer_options),
                loss=losses,
                loss_weights=loss_weights,
                weighted_metrics=metrics,
                steps_per_execution=steps_per_execution_per_replica,
            )
            if cfg.model.dtype == "float16":
                # the loss reduction is set by backend.floatx by default
                # must be forced to reduce in float32 to avoid overflow
                convert_loss_and_metric_reductions_to_fp32(model)

            # if the total batch size exceeds the compute batch size
            if xpu.IS_IPU:
                model.set_gradient_accumulation_options(
                    gradient_accumulation_steps_per_replica=cfg.ipu_opts.gradient_accumulation_factor,
                    gradient_accumulation_reduction_method=optimizers.GradientAccumulationReductionMethod.RUNNING_MEAN,
                    dtype=str_dtype_to_tf_dtype(cfg.ipu_opts.gradient_accumulation_dtype or cfg.model.dtype),
                    offload_weight_update_variables=cfg.ipu_opts.offload_optimizer_state,
                    replicated_optimizer_state_sharding=cfg.ipu_opts.RTS,
                )
            model.fit(
                tf_dataset, steps_per_epoch=steps_per_epoch, epochs=cfg.model.epochs, callbacks=callbacks, verbose=1
            )

    if cfg.checkpoint_path:
        checkpoint_paths = {-1: cfg.checkpoint_path}
    else:
        checkpoint_paths = {
            epoch: os.path.join(model_dir, f"model-{epoch:05d}")
            for epoch in range(1, cfg.model.epochs + 1, cfg.validate_every_n_epochs)
        }
    # Add the final checkpoint - catches final epoch != % checkpoint frequency
    # This will be removed if it doesn't exist.
    checkpoint_paths["FINAL"] = os.path.join(model_dir, "model-FINAL")

    # Filter the checkpoints for only checkpoints that exist
    checkpoint_paths = {e: p for e, p in checkpoint_paths.items() if os.path.exists(p + ".index")}

    return checkpoint_paths
