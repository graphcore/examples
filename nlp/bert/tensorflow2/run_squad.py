# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.

import argparse
import logging
import math
from datetime import datetime

from datasets import load_metric
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu.gradient_accumulation import GradientAccumulationReductionMethod
from transformers import BertConfig, TFBertForQuestionAnswering
import wandb

from data_utils.batch_config import BatchConfig, Task
from data_utils.squad_v1.load_squad_data import format_raw_data_for_metric, get_prediction_dataset, get_squad_data
from data_utils.squad_v1.postprocess_squad_predictions import postprocess_qa_predictions
from keras_extensions.callbacks.callback_factory import CallbackFactory
from keras_extensions.learning_rate.scheduler_builder import get_lr_scheduler
from keras_extensions.optimization import get_optimizer
from model.losses import QuestionAnsweringLossFunction
from model.convert_bert_model import convert_tf_bert_model, post_process_bert_input_layer
from model.pipeline_stage_names import PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES
from utilities.argparser import add_shared_arguments, add_squad_arguments, combine_config_file_with_args
from utilities.assign_pipeline_stages import PipelineStagesAssigner
from utilities.checkpoint_utility import load_checkpoint_into_model
from utilities.ipu_utils import create_ipu_strategy, get_poplar_options_per_pipeline_stage, set_random_seeds
from utilities.metric_enqueuer import wrap_loss_in_enqueuer, wrap_stateful_metric_in_enqueuer
from utilities.options import SQuADOptions


def fine_tune_squad(config):
    """Main function to run fine tuning BERT for SQuAD.
    :param config: A pydantic model object that contains the configuration to
        options for this application. See utilities/options.py for accepted
        options.
    """
    if config.bert_model_name is None and config.pretrained_ckpt_path is None:
        logging.warning(
            "SQuAD requires a pretrained model, either"
            " `bert_model_name` (via config file) or"
            " `pretrained_ckpt_path` (via config file or"
            " `--pretrained-ckpt-path` command line argument)"
            " but none provided."
        )
    if config.bert_model_name is not None and config.pretrained_ckpt_path is not None:
        logging.warning(
            "Only one checkpoint is accepted, but two provided:"
            f" `bert_model_name`={config.bert_model_name}, and"
            f" `pretrained_ckpt_path`={config.pretrained_ckpt_path}."
        )

    universal_run_name = config.config.stem if config.name is None else config.name
    universal_run_name += f"-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.info(f"Universal name for run: {universal_run_name}")

    if config.enable_wandb:
        wandb.init(entity="sw-apps", project="TF2-BERT", name=universal_run_name, config=config, tags=config.wandb_tags)

    num_pipeline_stages = len(config.ipu_config.pipeline_device_mapping)
    num_ipus_per_replica = max(config.ipu_config.pipeline_device_mapping) + 1

    # Load training and validation data
    # =================================
    train_dataset, eval_dataset, num_train_samples, num_eval_samples, raw_datasets = get_squad_data(
        config.global_batch.micro_batch_size, config.dataset_dir, config.generated_dataset, config.max_seq_length
    )
    total_num_train_samples = (
        config.num_epochs * num_train_samples.numpy()
        if config.total_num_train_samples is None
        else config.total_num_train_samples
    )
    train_batch_config = BatchConfig(
        micro_batch_size=config.global_batch.micro_batch_size,
        num_replicas=config.global_batch.replicas,
        total_num_train_samples=total_num_train_samples,
        gradient_accumulation_count=config.global_batch.grad_acc_steps_per_replica,
        num_pipeline_stages=num_pipeline_stages,
        dataset_size=num_train_samples.numpy(),
        global_batches_per_log=config.global_batches_per_log,
        task=Task.OTHER,
    )

    # Create model
    # ============
    policy = tf.keras.mixed_precision.Policy("float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    strategy = create_ipu_strategy(
        num_ipus_per_replica=num_ipus_per_replica,
        num_replicas=config.global_batch.replicas,
        enable_stochastic_rounding=config.enable_stochastic_rounding,
        enable_recomputation=config.enable_recomputation,
        compile_only=config.compile_only,
    )
    set_random_seeds(config.seed)
    with strategy.scope():
        # Instantiate the pretrained model given in the config.
        if config.bert_model_name is not None:
            model = TFBertForQuestionAnswering.from_pretrained(config.bert_model_name)
        else:
            bert_config = BertConfig(**config.bert_config.dict(), hidden_act=ipu.nn_ops.gelu)
            model = TFBertForQuestionAnswering(config=bert_config)

        # Convert subclass model to functional, expand main layers to enable pipelining, and replace some layers to
        # optimise performance.
        model = convert_tf_bert_model(
            model,
            train_dataset,
            post_process_bert_input_layer,
            replace_layers=config.replace_layers,
            use_outlining=config.use_outlining,
            enable_recomputation=config.enable_recomputation,
            embedding_serialization_factor=config.embedding_serialization_factor,
            rename_outputs={"tf.compat.v1.squeeze": "start_positions", "tf.compat.v1.squeeze_1": "end_positions"},
            use_prediction_bias=config.use_prediction_bias,
            use_projection_bias=config.use_projection_bias,
            use_cls_layer=config.use_cls_layer,
            use_qkv_bias=config.use_qkv_bias,
            use_qkv_split=config.use_qkv_split,
        )
        # Load from pretrained checkpoint if requested.
        if config.pretrained_ckpt_path:
            logging.info("Attempting to load pretrained checkpoint from path " f"{config.pretrained_ckpt_path}")
            _ = load_checkpoint_into_model(
                model=model, pretrained_ckpt_path=config.pretrained_ckpt_path, expect_partial=config.do_training
            )

        # Configure pipeline stages
        # =========================
        if num_pipeline_stages > 1:
            pipeline_assigner = PipelineStagesAssigner(PIPELINE_ALLOCATE_PREVIOUS, PIPELINE_NAMES)
            assignments = model.get_pipeline_stage_assignment()
            assignments = pipeline_assigner.assign_pipeline_stages(assignments, config.ipu_config.pipeline_stages)
            model.set_pipeline_stage_assignment(assignments)
            model.print_pipeline_stage_assignment_summary(print_fn=logging.info)
            poplar_options_per_pipeline_stage = get_poplar_options_per_pipeline_stage(
                num_ipus_per_replica,
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

        if config.do_training:
            # Compile the model for training
            # ==============================
            # Wrap loss in an out-feed queue.
            loss = wrap_loss_in_enqueuer(
                QuestionAnsweringLossFunction, ["end_positions_loss", "start_positions_loss"]
            )()
            # Wrap accuracy in an out-feed queue.
            accuracy = wrap_stateful_metric_in_enqueuer(
                tf.keras.metrics.SparseCategoricalAccuracy, ["end_positions_accuracy", "start_positions_accuracy"]
            )()
            # Define optimiser with polynomial decay learning rate.
            lr_outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
                outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.LAST
            )
            lr_scheduler = get_lr_scheduler(
                scheduler_name=config.optimizer_opts.learning_rate.schedule_name,
                max_learning_rate=config.optimizer_opts.learning_rate.max_learning_rate,
                warmup_frac=config.optimizer_opts.learning_rate.warmup_frac,
                num_train_steps=train_batch_config.num_train_steps,
                queue=lr_outfeed_queue,
            )
            # Prepare optimizer.
            optimizer = get_optimizer(
                optimizer_name=config.optimizer_opts.name,
                num_replicas=config.global_batch.replicas,
                learning_rate_schedule=lr_scheduler,
                use_outlining=config.use_outlining,
                weight_decay_rate=config.optimizer_opts.weight_decay_rate,
            )
            # Compile the model.
            model.compile(
                optimizer=optimizer,
                loss={"end_positions": loss, "start_positions": loss},
                metrics={"end_positions": accuracy, "start_positions": accuracy},
                steps_per_execution=train_batch_config.steps_per_execution,
            )

            # Train the model
            # ===============
            # Set up callbacks
            callbacks = CallbackFactory.get_callbacks(
                universal_run_name=universal_run_name,
                batch_config=train_batch_config,
                model=model,
                checkpoint_path=config.save_ckpt_path,
                ckpt_every_n_steps_per_execution=config.ckpt_every_n_steps_per_execution,
                outfeed_queues=[lr_outfeed_queue, loss.outfeed_queue, accuracy.outfeed_queue],
                enable_wandb=config.enable_wandb,
            )
            # Print configs to be logged in wandb's terminal.
            logging.info(f"Application config:\n{config}")
            logging.info(f"Training batch config:\n{train_batch_config}")
            # Train the model
            # Set verbose to 0 so the default progress bar, which is unreliable
            # with `steps_per_execution > 1`, is hidden in favour of using a
            # logging callback included in callbacks dir
            model.fit(
                train_dataset,
                steps_per_epoch=(
                    train_batch_config.total_num_micro_batches
                    if train_batch_config.epochs < 1
                    else train_batch_config.num_micro_batches_per_epoch
                ),
                epochs=math.ceil(train_batch_config.epochs),
                callbacks=callbacks,
                verbose=0,
            )

    if config.do_validation:
        # Evaluate the model on the validation set
        # ========================================
        # Prepare the dataset to be evaluated in the IPU.
        eval_batch_config = BatchConfig(
            micro_batch_size=config.global_batch.micro_batch_size,
            total_num_train_samples=num_eval_samples.numpy(),
            gradient_accumulation_count=config.global_batch.grad_acc_steps_per_replica,
            dataset_size=num_eval_samples.numpy(),
            task=Task.OTHER,
        )
        eval_pred_dataset = get_prediction_dataset(eval_dataset, eval_batch_config.num_micro_batches_per_epoch)
        with strategy.scope():
            # Re-compile the model for prediction if needed.
            if (
                train_batch_config.steps_per_execution != eval_batch_config.steps_per_execution
                or not config.do_training
            ):
                model.compile(steps_per_execution=eval_batch_config.steps_per_execution)
            # Get predictions for the validation data.
            logging.info(f"Running inference:\nGenerating predictions on the validation data...")
            # Set verbose to 0 so the default progress bar is hidden in
            # favour of using a logging callback included in callbacks dir
            predictions = model.predict(
                eval_pred_dataset,
                batch_size=eval_batch_config.micro_batch_size,
                verbose=0,
            )
        # The predictions for the end position is first in the model outputs tuple (note the output of model.summary()).
        end_predictions, start_predictions = predictions
        # Match the predictions to answers in the original context.
        # This will also write out the predictions to a json file in the directory given by `output_dir`.
        final_predictions = postprocess_qa_predictions(
            list(raw_datasets["validation"].as_numpy_iterator()),
            list(eval_dataset.take(eval_batch_config.num_micro_batches_per_epoch).unbatch().as_numpy_iterator()),
            (start_predictions, end_predictions),
            output_dir=config.output_dir,
        )
        # Format the predictions and the actual labels as expected by the metric.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
        formatted_labels = format_raw_data_for_metric(raw_datasets["validation"])
        metric = load_metric("squad")
        metrics = metric.compute(predictions=formatted_predictions, references=formatted_labels)
        logging.info("Evaluation metrics:")
        for key, value in metrics.items():
            logging.info(f"{key}: {value:.3f}")


if __name__ == "__main__":
    # Combine arguments and config file
    parser = argparse.ArgumentParser(
        description="TF2 BERT SQuAD Fine Tuning", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = add_shared_arguments(parser)
    parser = add_squad_arguments(parser)
    args = parser.parse_args()
    config = combine_config_file_with_args(args, SQuADOptions)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", level=config.logging, datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Prevent doubling of TF logs.
    tf.get_logger().propagate = False

    # Run SQuAD fine tuning
    fine_tune_squad(config)
