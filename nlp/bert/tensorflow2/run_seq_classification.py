# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import argparse
from datetime import datetime
import logging
import math

import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu.gradient_accumulation import GradientAccumulationReductionMethod
from transformers import BertConfig, TFBertForSequenceClassification
import wandb

from data_utils.batch_config import BatchConfig, Task
from data_utils.glue.load_glue_data import get_glue_data
from keras_extensions.callbacks.callback_factory import CallbackFactory
from keras_extensions.learning_rate.scheduler_builder import get_lr_scheduler
from keras_extensions.optimization import get_optimizer
from model.convert_bert_model import convert_tf_bert_model, post_process_bert_input_layer
from model.accuracy import classification_accuracy_fn
from model.losses import ClassificationLossFunction, ClassificationLossFunctionRegression
from model.pipeline_stage_names import PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES
from utilities.argparser import add_shared_arguments, add_glue_arguments, combine_config_file_with_args
from utilities.assign_pipeline_stages import GluePipelineStagesAssigner
from utilities.checkpoint_utility import load_checkpoint_into_model
from utilities.ipu_utils import create_ipu_strategy, get_poplar_options_per_pipeline_stage, set_random_seeds
from utilities.metric_enqueuer import wrap_loss_in_enqueuer, wrap_stateless_metric_in_enqueuer
from utilities.options import GLUEOptions


def fine_tune_glue(config):
    """Main function to run fine tuning BERT for GLUE.
    :param config: A pydantic model object that contains the configuration to
        options for this application. See utilities/options.py for accepted
        options.
    """
    if config.bert_model_name is None and config.pretrained_ckpt_path is None:
        logging.warning(
            "GLUE requires a pretrained model, either"
            " `bert_model_name` (via config file) or"
            " `pretrained_ckpt_path` (via config file or"
            " `--pretrained-ckpt-path` command line argument)"
            " but none provided."
        )
    if config.bert_model_name is not None and config.pretrained_ckpt_path is not None:
        logging.warning(
            "Only one checkpoint is accepted, but two provided:"
            " `bert_model_name`={config.bert_model_name}, and"
            " `pretrained_ckpt_path`={config.pretrained_ckpt_path}."
        )

    universal_run_name = config.config.stem if config.name is None else config.name
    universal_run_name += f"-{config.glue_task}"
    universal_run_name += f"-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.info(f"Universal name for run: {universal_run_name}")

    if config.enable_wandb:
        wandb.init(entity="sw-apps", project="TF2-BERT", name=universal_run_name, config=config, tags=config.wandb_tags)

    num_pipeline_stages = len(config.ipu_config.pipeline_device_mapping)
    num_ipus_per_replicas = max(config.ipu_config.pipeline_device_mapping) + 1

    # Load training and validation data
    # =================================
    dataset, eval_dataset, test_dataset, num_train_samples, num_eval_samples, num_test_samples, _ = get_glue_data(
        config.glue_task,
        config.global_batch.micro_batch_size,
        config.dataset_dir,
        config.max_seq_length,
        generated_dataset=config.generated_dataset,
    )

    total_num_train_samples = (
        config.num_epochs * num_train_samples.numpy()
        if config.total_num_train_samples is None
        else config.total_num_train_samples
    )
    train_config = BatchConfig(
        micro_batch_size=config.global_batch.micro_batch_size,
        total_num_train_samples=total_num_train_samples,
        num_replicas=config.global_batch.replicas,
        gradient_accumulation_count=config.global_batch.grad_acc_steps_per_replica,
        num_pipeline_stages=num_pipeline_stages,
        dataset_size=num_train_samples.numpy(),
        global_batches_per_log=config.global_batches_per_log,
        task=Task.OTHER,
    )

    policy = tf.keras.mixed_precision.Policy("float16")
    tf.keras.mixed_precision.set_global_policy(policy)

    strategy = create_ipu_strategy(
        num_ipus_per_replica=num_ipus_per_replicas,
        num_replicas=config.global_batch.replicas,
        enable_recomputation=config.enable_recomputation,
        enable_stochastic_rounding=config.enable_stochastic_rounding,
        compile_only=config.compile_only,
    )
    set_random_seeds(config.seed)
    with strategy.scope():
        if config.bert_model_name is not None:
            # Instantiate the pretrained model given in the config.
            model = TFBertForSequenceClassification.from_pretrained(config.bert_model_name)
        else:
            bert_config = BertConfig(**config.bert_config.dict(), hidden_act=ipu.nn_ops.gelu)
            model = TFBertForSequenceClassification(config=bert_config)
        # Convert to functional model exposing the encoder stages.
        model = convert_tf_bert_model(
            model,
            dataset,
            post_process_bert_input_layer,
            replace_layers=config.replace_layers,
            use_outlining=config.use_outlining,
            enable_recomputation=config.enable_recomputation,
            embedding_serialization_factor=config.embedding_serialization_factor,
            use_prediction_bias=config.use_prediction_bias,
            use_projection_bias=config.use_projection_bias,
            use_cls_layer=config.use_cls_layer,
            use_qkv_bias=config.use_qkv_bias,
            use_qkv_split=config.use_qkv_split,
            rename_outputs={"classifier": "labels"},
        )

        # Load from pretrained checkpoint if requested.
        if config.pretrained_ckpt_path:
            logging.info("Attempting to load pretrained checkpoint from path " f"{config.pretrained_ckpt_path}")
            _ = load_checkpoint_into_model(
                model=model, pretrained_ckpt_path=config.pretrained_ckpt_path, expect_partial=not config.do_training
            )

        # Configure pipeline stages
        # =========================
        if num_pipeline_stages > 1:
            pipeline_assigner = GluePipelineStagesAssigner(PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES)
            assignments = model.get_pipeline_stage_assignment()
            assignments = pipeline_assigner.assign_glue_pipeline_stages(assignments, config.ipu_config.pipeline_stages)
            model.set_pipeline_stage_assignment(assignments)
            model.print_pipeline_stage_assignment_summary(print_fn=logging.info)
            poplar_options_per_pipeline_stage = get_poplar_options_per_pipeline_stage(
                num_ipus_per_replicas,
                config.ipu_config.pipeline_device_mapping,
                config.ipu_config.matmul_available_memory_proportion_per_pipeline_stage,
                config.matmul_partials_type,
            )
            model.set_pipelining_options(
                gradient_accumulation_steps_per_replica=config.global_batch.grad_acc_steps_per_replica,
                gradient_accumulation_reduction_method=GradientAccumulationReductionMethod.RUNNING_MEAN,
                pipeline_schedule=ipu.ops.pipelining_ops.PipelineSchedule.Grouped,
                device_mapping=config.ipu_config.pipeline_device_mapping,
                offload_weight_update_variables=config.optimizer_state_offchip,
                forward_propagation_stages_poplar_options=poplar_options_per_pipeline_stage,
                backward_propagation_stages_poplar_options=poplar_options_per_pipeline_stage,
                recomputation_mode=ipu.pipelining_ops.RecomputationMode.RecomputeAndBackpropagateInterleaved,
            )

        # Define optimiser with polynomial decay learning rate.
        queues_to_outfeed = []
        lr_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST)
        lr_scheduler = get_lr_scheduler(
            scheduler_name=config.optimizer_opts.learning_rate.schedule_name,
            max_learning_rate=config.optimizer_opts.learning_rate.max_learning_rate,
            warmup_frac=config.optimizer_opts.learning_rate.warmup_frac,
            num_train_steps=train_config.num_train_steps,
            queue=lr_outfeed_queue,
        )
        queues_to_outfeed.append(lr_outfeed_queue)

        loss_function = (
            ClassificationLossFunctionRegression if config.glue_task == "stsb" else ClassificationLossFunction
        )
        glue_loss = wrap_loss_in_enqueuer(loss_function, ["glue_loss"])()
        queues_to_outfeed.append(glue_loss.outfeed_queue)

        metrics = []
        if config.glue_task != "stsb":
            # Note that the stsb task is a regression and the accuracy
            # is not a valid metric
            accuracy = wrap_stateless_metric_in_enqueuer("glue_accuracy", classification_accuracy_fn, ["glue_accuracy"])
            metrics.append(accuracy)
            queues_to_outfeed.append(accuracy.outfeed_queue)

        if config.do_training:
            # Prepare optimizer.
            optimizer = get_optimizer(
                optimizer_name=config.optimizer_opts.name,
                num_replicas=config.global_batch.replicas,
                learning_rate_schedule=lr_scheduler,
                use_outlining=config.use_outlining,
                weight_decay_rate=config.optimizer_opts.weight_decay_rate,
            )
            # Compile and train the functional model
            model.compile(
                optimizer=optimizer,
                loss={"labels": glue_loss},
                metrics=metrics,
                steps_per_execution=train_config.steps_per_execution,
            )
            # Set up callbacks
            callbacks = CallbackFactory.get_callbacks(
                universal_run_name=universal_run_name,
                batch_config=train_config,
                model=model,
                checkpoint_path=config.save_ckpt_path,
                ckpt_every_n_steps_per_execution=config.ckpt_every_n_steps_per_execution,
                outfeed_queues=queues_to_outfeed,
                enable_wandb=config.enable_wandb,
            )
            # Print configs to be logged in wandb's terminal.
            logging.info(f"Application config:\n{config}")
            logging.info(f"Training batch config:\n{train_config}")

            model.fit(
                dataset,
                steps_per_epoch=(
                    train_config.total_num_micro_batches
                    if train_config.epochs < 1
                    else train_config.num_micro_batches_per_epoch
                ),
                batch_size=config.global_batch.micro_batch_size,
                epochs=math.ceil(train_config.epochs),
                callbacks=callbacks,
                verbose=0,
            )
        if config.do_validation:
            eval_batch_config = BatchConfig(
                micro_batch_size=config.global_batch.micro_batch_size,
                total_num_train_samples=num_eval_samples.numpy(),
                num_replicas=1,
                gradient_accumulation_count=10,
                dataset_size=num_eval_samples.numpy(),
                task=Task.OTHER,
            )
            # Re-compile the model for prediction if needed.
            eval_dataset = eval_dataset.take(eval_batch_config.num_micro_batches_per_epoch)

            if train_config.steps_per_execution != eval_batch_config.steps_per_execution or not config.do_training:
                model.compile(metrics=[accuracy], steps_per_execution=eval_batch_config.steps_per_execution)
            # Get predictions for the validation data.
            logging.info(f"Running inference:\nGenerating predictions on the validation data...")
            validation_results = model.evaluate(eval_dataset, verbose=0)
            logging.info(f"Validation performance for {config.glue_task}:\naccuracy = {validation_results[1]:.3f}")

        if config.do_test:
            test_batch_config = BatchConfig(
                micro_batch_size=config.global_batch.micro_batch_size,
                total_num_train_samples=num_test_samples.numpy(),
                num_replicas=1,
                gradient_accumulation_count=10,
                dataset_size=num_test_samples.numpy(),
                task=Task.OTHER,
            )
            # Re-compile the model for prediction if needed.
            test_dataset = test_dataset.take(test_batch_config.num_micro_batches_per_epoch)
            if train_config.steps_per_execution != test_batch_config.steps_per_execution or not config.do_training:
                model.compile(steps_per_execution=test_batch_config.steps_per_execution)
            # Get predictions for the test data.
            logging.info(f"Running inference:\nGenerating predictions on the test data...")
            model.predict(test_dataset, verbose=0)


if __name__ == "__main__":
    # Combine arguments and config file
    parser = argparse.ArgumentParser(
        description="TF2 BERT GLUE Fine Tuning", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = add_shared_arguments(parser)
    parser = add_glue_arguments(parser)
    args = parser.parse_args()
    config = combine_config_file_with_args(args, GLUEOptions)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=config.logging, datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Prevent doubling of TF logs.
    tf.get_logger().propagate = False

    # Run GLUE fine tuning
    fine_tune_glue(config)
